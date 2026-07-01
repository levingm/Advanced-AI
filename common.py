"""Gemeinsame Pipeline für Einlagenvolumen-Prognosen und Kovariaten-Vergleich.

Szenario 2A: Realistische Prognose ohne ex-post Zukunftswissen.
- Sequenzmodelle: nur historisches Fenster als Input
- TFT: Kovariaten per Naive-Hold (letzter Wert), Zielvariablen unbekannt
- Naive Benchmarks: Random Walk, Naive Hold (Level konstant)
- Rolling-Origin-Evaluation für robuste Modellvergleiche
"""

from __future__ import annotations


import importlib
import itertools
import warnings
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks import EarlyStopping
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
# FIX 6: RobustScaler statt MinMaxScaler (robust gegen Ausreißer in Finanzdaten)
from sklearn.preprocessing import RobustScaler



# Nur bekannte, harmlose Warnungen unterdrücken
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning, module=r"lightning(\..+)?")
warnings.filterwarnings("ignore", category=UserWarning, module=r"pytorch_forecasting(\..+)?")

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------
DATEN = "Data all variables.xlsx"
TARGET_COL = "Einlagevolumen"
TARGET_DIFF = "Diff_Volume"

ALL_COVARIATES = [
    "€STR",
    "Einlagezinssatz",
    "GPRC_DEU",
    "MoM Inflation",
    "DAX",
    "10Y Bond",
]

GEDAECHTNIS = 30
PROGNOSEHORIZONT = 100
SIMULATIONEN = 5
SEED = 42

TRAINING_ENDE = "2023-11-30"
HISTORIE_ANFANG = "2023-06-01"
HISTORIE_ENDE = "2023-11-30"

GROUP_ID = "series"

TFT_MAX_EPOCHS = 40
TFT_BATCH_SIZE = 64
TFT_LEARNING_RATE = 1e-3
# Konsistent mit FFN-internem validation_fraction; 15% reicht für EarlyStopping
TFT_VAL_FRACTION = 0.15

ALL_MODELS = ["naive_rw", "naive_hold", "linreg", "ffn", "xgboost", "tft"]

# PERF: Cache für getuntete Hyperparameter (Key: model_name)
# Verhindert wiederholtes Tuning in jeder Simulation
_TUNED_PARAMS_CACHE: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Ergebnis-Datenklasse
# ---------------------------------------------------------------------------
@dataclass
class ForecastResult:
    """Ergebnis einer einzelnen Modell-×-Kovariaten-Evaluation."""

    model: str
    covariates: list[str]
    covariate_label: str
    training_end: str       # explizit: welches Origin wurde bewertet?
    diff_pred: np.ndarray
    volume_pred: np.ndarray
    mae_diff: float
    rmse_diff: float
    r2_diff: float
    mae_volume: float
    rmse_volume: float
    r2_volume: float


# ---------------------------------------------------------------------------
# Kovariatenkombinationen
# ---------------------------------------------------------------------------
def variablen_for(covariates: list[str]) -> list[str]:
    """Kombiniert Target + Kovariaten für DataFrame-Selektion."""
    return [TARGET_COL] + list(covariates)


def generate_covariate_combinations(
    mode: str = "presets",
) -> list[tuple[list[str], str]]:
    """Liefert (Kovariatenliste, Label)-Paare für den Benchmark-Loop.

    Args:
        mode: "presets" | "pairs" | "singletons" | "all" | "full"
    """
    combos: list[tuple[list[str], str]] = []

    if mode in ("presets", "all"):
        presets = {
            "alle": ALL_COVARIATES,
            "nur_zinsen": ["€STR", "Einlagezinssatz"],
            "nur_maerkte": ["DAX", "10Y Bond"],
            "nur_makro": ["GPRC_DEU", "MoM Inflation"],
            "zinsen_makro": ["€STR", "Einlagezinssatz", "GPRC_DEU", "MoM Inflation"],
            "ohne_dax": [c for c in ALL_COVARIATES if c != "DAX"],
            "ohne_inflation": [c for c in ALL_COVARIATES if c != "MoM Inflation"],
            "nur_estr": ["€STR"],
            "nur_einlagezins": ["Einlagezinssatz"],
        }
        combos.extend((cols, name) for name, cols in presets.items())

    if mode in ("pairs", "all"):
        for a, b in itertools.combinations(ALL_COVARIATES, 2):
            combos.append(([a, b], f"pair_{a}_{b}".replace(" ", "")))

    if mode in ("singletons", "all"):
        for c in ALL_COVARIATES:
            combos.append(([c], f"single_{c}".replace(" ", "")))

    if mode == "full":
        for r in range(1, len(ALL_COVARIATES) + 1):
            for combo in itertools.combinations(ALL_COVARIATES, r):
                cols = list(combo)
                label = "full_" + "_".join(c.replace(" ", "") for c in cols)
                combos.append((cols, label))

    # Duplikate entfernen (Reihenfolge irrelevant)
    seen: set[tuple[str, ...]] = set()
    unique: list[tuple[list[str], str]] = []
    for cols, label in combos:
        key = tuple(sorted(cols))
        if key not in seen:
            seen.add(key)
            unique.append((cols, label))
    return unique


# FIX 9: Pre-built Dict – O(1) statt O(n²) pro Aufruf.
# Lazy initialisiert beim ersten Aufruf von covariates_for_label().
_LABEL_MAP: dict[str, list[str]] | None = None


def _build_label_map() -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    for mode in ("presets", "singletons", "pairs", "full"):
        for cols, lbl in generate_covariate_combinations(mode):
            mapping.setdefault(lbl, cols)
    return mapping


def covariates_for_label(label: str) -> list[str]:
    """Liefert Kovariatenliste zu einem Benchmark-Label (O(1))."""
    global _LABEL_MAP
    if _LABEL_MAP is None:
        _LABEL_MAP = _build_label_map()
    if label not in _LABEL_MAP:
        raise ValueError(f"Unbekanntes Kovariate-Label: {label!r}")
    return _LABEL_MAP[label]


# ---------------------------------------------------------------------------
# Daten laden + vorbereiten
# ---------------------------------------------------------------------------

# FIX 10: Explizit invalidierbar statt stiller lru_cache-Falle
@lru_cache(maxsize=1)
def _load_base_frame() -> pd.DataFrame:
    """Lädt Excel-Datei einmalig; alle weiteren Aufrufe nutzen den Cache."""
    raw = pd.read_excel(DATEN)
    raw["Datum"] = pd.to_datetime(raw["Datum"])
    raw = raw.sort_values("Datum").set_index("Datum")
    cols = [TARGET_COL] + ALL_COVARIATES
    return raw[cols].copy()


def clear_data_cache() -> None:
    """Invalidiert den Daten-Cache (z.B. nach Austausch von DATEN)."""
    _load_base_frame.cache_clear()


def _next_date_in_index(df: pd.DataFrame, date: str) -> str:
    """Erstes verfügbares Datum im Index nach `date`."""
    ts = pd.Timestamp(date)
    later = df.index[df.index > ts]
    if len(later) == 0:
        raise ValueError(f"Kein Datum nach {date} im Datenindex verfügbar.")
    return str(later[0].date())


def load_prepared_df(
    covariates: list[str],
    training_end: str = TRAINING_ENDE,
    prognosehorizont: int = PROGNOSEHORIZONT,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Lädt und bereitet Daten vor.

    Args:
        covariates: Zu nutzende Kovariaten.
        training_end: Letztes Trainingsdatum (für Rolling-Origin parametrierbar).
        prognosehorizont: Länge des Prognose-Horizonts.

    Returns:
        df:         Vollständiger Frame inkl. NaN (echte Test-Levels abrufbar)
        df_clean:   Frame ohne NaN (für Sequenz-Training)
        train_df:   Trainingsdaten bis training_end
        future_df:  Referenzdaten für den Test-Horizont (nicht als Modell-Input)
    """
    variablen = variablen_for(covariates)
    df = _load_base_frame()[variablen].copy()
    df[TARGET_DIFF] = df[TARGET_COL].diff()
    df_clean = df.dropna()

    train_df = df_clean.loc[:training_end].copy()
    input_start = _next_date_in_index(df_clean, training_end)
    future_df = df_clean.loc[input_start:].iloc[:prognosehorizont].copy()
    return df, df_clean, train_df, future_df


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def create_sequences(
    X_arr: np.ndarray, y_arr: np.ndarray, gd: int, hz: int
) -> tuple[np.ndarray, np.ndarray]:
    """Erstellt Multi-Step-Input/Output-Sequenzen aus skalierten Arrays.

    Args:
        X_arr: Features (n_samples, n_features) – bereits skaliert
        y_arr: Targets  (n_samples, 1)          – bereits skaliert
        gd:    Gedächtnis (Encoder-Länge)
        hz:    Prognose-Horizont

    Returns:
        X_seq: (n_seq, gd * n_features)
        y_seq: (n_seq, hz)
    """
    X_seq, y_seq = [], []
    for i in range(len(X_arr) - gd - hz + 1):
        X_seq.append(X_arr[i : i + gd].flatten())
        y_seq.append(y_arr[i + gd : i + gd + hz].flatten())
    return np.array(X_seq), np.array(y_seq)


def compute_metrics(
    true_diff: np.ndarray,
    pred_diff: np.ndarray,
    true_volume: np.ndarray,
    pred_volume: np.ndarray,
) -> dict[str, float]:
    """MAE, RMSE, R² auf Differenzen und Level."""
    return {
        "mae_diff":    float(mean_absolute_error(true_diff, pred_diff)),
        "rmse_diff":   float(np.sqrt(mean_squared_error(true_diff, pred_diff))),
        "r2_diff":     float(r2_score(true_diff, pred_diff)),
        "mae_volume":  float(mean_absolute_error(true_volume, pred_volume)),
        "rmse_volume": float(np.sqrt(mean_squared_error(true_volume, pred_volume))),
        "r2_volume":   float(r2_score(true_volume, pred_volume)),
    }


def _true_test_arrays(
    df: pd.DataFrame,
    df_clean: pd.DataFrame,
    input_start: str,
    prognosehorizont: int = PROGNOSEHORIZONT,
) -> tuple[np.ndarray, np.ndarray]:
    """Extrahiert echte Test-Arrays via Datum-basierter Indexierung.

    FIX 1.2: .loc[input_start:] statt .iloc[positional_offset:] verhindert
    den Off-by-1-Fehler wenn df und df_clean unterschiedliche Längen haben.
    FIX 8: Explizite Längenprüfung statt stiller Fehlausrichtung.
    """
    true_diff = df_clean.loc[input_start:].iloc[:prognosehorizont][TARGET_DIFF].to_numpy()
    true_volume = df.loc[input_start:].iloc[:prognosehorizont][TARGET_COL].to_numpy()

    if len(true_diff) < prognosehorizont:
        raise ValueError(
            f"Zu wenig Testdaten für Differenzen: {len(true_diff)} < {prognosehorizont}. "
            f"Liegt input_start={input_start!r} zu nah am Datenende?"
        )
    if len(true_volume) < prognosehorizont:
        raise ValueError(
            f"Zu wenig Testdaten für Volumen: {len(true_volume)} < {prognosehorizont}."
        )
    return true_diff, true_volume


def reconstruct_volume(diff_pred: np.ndarray, last_level: float) -> np.ndarray:
    """Rekonstruiert Volumen-Level aus Differenzen via Kumulation."""
    return last_level + np.cumsum(diff_pred)


def _snap_to_available_date(index: pd.DatetimeIndex, date: str) -> pd.Timestamp:
    """Liefert das letzte verfügbare Datum im Index <= `date`.

    Wirft ValueError, falls kein Datum <= `date` existiert (z.B. Origin liegt
    vor dem Datenbeginn).
    """
    ts = pd.Timestamp(date)
    avail = index[index <= ts]
    if len(avail) == 0:
        raise ValueError(
            f"Kein verfügbares Datum <= {date} im Datenindex (Datenbeginn: {index.min()})."
        )
    return avail[-1]


def _get_loc_snapped(df: pd.DataFrame, date: str) -> int:
    """Positionsindex des letzten verfügbaren Datums <= `date` (snapped)."""
    snapped = _snap_to_available_date(df.index, date)
    return df.index.get_loc(snapped)


def _loc_value_snapped(df: pd.DataFrame, date: str, col: str):
    """Wert von `col` am letzten verfügbaren Datum <= `date` (snapped)."""
    snapped = _snap_to_available_date(df.index, date)
    return df.loc[snapped, col]


# ---------------------------------------------------------------------------
# FIX 2: Naive Benchmarks
# ---------------------------------------------------------------------------

def predict_naive_rw(
    covariates: list[str],
    simulations: int = 1,
    seed: int = SEED,
    training_end: str = TRAINING_ENDE,
) -> tuple[np.ndarray, np.ndarray]:
    """Naiver Benchmark – Random Walk: letzte Differenz konstant fortgeschrieben.

    Formal: ŷ_{t+h} = y_t + h * Δy_t  für alle h = 1..PROGNOSEHORIZONT
    Kein Zukunftswissen. Schwieriger Benchmark für Finanzzeitreihen.

    Hinweis (BUGFIX 1): `covariates` wird hier bewusst ignoriert – Naive-Modelle
    benötigen keine Kovariaten und dürfen daher mit covariates=[] aufgerufen werden.

    Returns:
        (volume_pred, diff_pred) – diff_pred ist die rohe, konstante Modell-Differenz
        (BUGFIX 5: direkt statt über np.diff(volume_pred) rückgerechnet).
    """
    df, df_clean, _, _ = load_prepared_df(covariates, training_end=training_end)
    last_diff = float(df_clean.loc[:training_end, TARGET_DIFF].iloc[-1])
    last_level = float(_loc_value_snapped(df, training_end, TARGET_COL))
    diff_pred = np.full(PROGNOSEHORIZONT, last_diff)
    return reconstruct_volume(diff_pred, last_level), diff_pred


def predict_naive_hold(
    covariates: list[str],
    simulations: int = 1,
    seed: int = SEED,
    training_end: str = TRAINING_ENDE,
) -> tuple[np.ndarray, np.ndarray]:
    """Naiver Benchmark – Naive Hold: letztes Volumen konstant (Drift = 0).

    Formal: ŷ_{t+h} = y_t  für alle h.

    Hinweis (BUGFIX 1): `covariates` wird hier bewusst ignoriert – Naive-Modelle
    benötigen keine Kovariaten und dürfen daher mit covariates=[] aufgerufen werden.

    Returns:
        (volume_pred, diff_pred) – diff_pred ist hier konstant 0 (BUGFIX 5: direkt
        statt über np.diff(volume_pred) rückgerechnet).
    """
    df, _, _, _ = load_prepared_df(covariates, training_end=training_end)
    last_level = float(_loc_value_snapped(df, training_end, TARGET_COL))
    volume_pred = np.full(PROGNOSEHORIZONT, last_level)
    diff_pred = np.zeros(PROGNOSEHORIZONT)
    return volume_pred, diff_pred


# ---------------------------------------------------------------------------
# Sequenzmodelle: LinReg, FFN, XGBoost – OPTIMIERT
# ---------------------------------------------------------------------------

# PERF 3.2: Schrumpftes Tuning-Grid für schnelleres Benchmarking
# Vorher: FFN 2×3×2=12, XGBoost 3×2×2×2=24 Kombinationen
# Nachher: FFN 1×2×1=2, XGBoost 2×1×1×1=2 Kombinationen
def _get_tune_grid(model_name: str) -> dict:
    """Gibt das Tuning-Grid für FFN/XGBoost zurück (schrumpft für Speed)."""
    if model_name == "ffn":
        return {
            "hidden_layer_sizes": [(64, 32)],           # 1 Option statt 2
            "alpha": [1e-4, 1e-3],                      # 2 Optionen (reduziert)
            "learning_rate_init": [1e-3],               # 1 Option (fix)
        }
    elif model_name == "xgboost":
        return {
            "max_depth": [3, 4],                        # 2 Optionen (reduziert von 3)
            "learning_rate": [0.05],                    # 1 Option (fix)
            "subsample": [0.8],                         # 1 Option (fix)
            "colsample_bytree": [0.7],                  # 1 Option (fix)
        }
    else:
        raise ValueError(f"Tune-Grid nicht für {model_name} definiert.")


def tune_sequence_model(model_name: str, X: np.ndarray, y: np.ndarray, seed: int = SEED):
    """Zeitserien-konformes Tuning mit reduziertem Grid.
    
    PERF: Cacht Ergebnis, damit wiederholte Aufrufe (z.B. in Simulations-Loop)
    nicht nochmal tunen.
    
    Returns gefittetes Modell mit besten Parametern.
    """
    cache_key = model_name
    if cache_key in _TUNED_PARAMS_CACHE:
        print(f"[TUNING] {model_name} – nutze gecachtete Parameter.")
        params = _TUNED_PARAMS_CACHE[cache_key]
        if model_name == "ffn":
            return MLPRegressor(
                hidden_layer_sizes=params["hidden_layer_sizes"],
                activation="relu",
                alpha=params["alpha"],
                learning_rate_init=params["learning_rate_init"],
                max_iter=500,
                early_stopping=False,
                random_state=seed,
            )
        elif model_name == "xgboost":
            XGBRegressor = importlib.import_module("xgboost").XGBRegressor
            return XGBRegressor(
                max_depth=params["max_depth"],
                learning_rate=params["learning_rate"],
                subsample=params["subsample"],
                colsample_bytree=params["colsample_bytree"],
                n_estimators=100,
                random_state=seed,
            )

    # Erstes Mal: Grid-Search durchführen
    tscv = TimeSeriesSplit(n_splits=2)  # PERF 3.2: 2 statt 3 Splits
    param_grid = _get_tune_grid(model_name)

    if model_name == "ffn":
        base = MLPRegressor(
            random_state=seed,
            max_iter=500,
            early_stopping=False,
            activation="relu",
        )
        gs = GridSearchCV(base, param_grid, cv=tscv, scoring="neg_mean_absolute_error", n_jobs=1)
        gs.fit(X, y)
        best_params = gs.best_params_
        _TUNED_PARAMS_CACHE[cache_key] = best_params
        print(f"[TUNING] FFN best params: {best_params} (score={gs.best_score_:.4f}) – GECACHT")
        return gs.best_estimator_

    elif model_name == "xgboost":
        XGBRegressor = importlib.import_module("xgboost").XGBRegressor
        base = XGBRegressor(
            n_estimators=100,
            random_state=seed,
            n_jobs=1,
            verbosity=0,
            objective="reg:squarederror",
        )
        gs = GridSearchCV(base, param_grid, cv=tscv, scoring="neg_mean_absolute_error", n_jobs=1)
        try:
            gs.fit(X, y)
            best_params = gs.best_params_
            _TUNED_PARAMS_CACHE[cache_key] = best_params
            print(
                f"[TUNING] XGBoost best params: {best_params} (score={gs.best_score_:.4f}) – GECACHT"
            )
            return gs.best_estimator_
        except Exception:
            from sklearn.multioutput import MultiOutputRegressor
            wrapped = MultiOutputRegressor(base)
            wrapped_param_grid = {f"estimator__{k}": v for k, v in param_grid.items()}
            gs_wrapped = GridSearchCV(
                wrapped,
                wrapped_param_grid,
                cv=tscv,
                scoring="neg_mean_absolute_error",
                n_jobs=1,
            )
            try:
                gs_wrapped.fit(X, y)
                best_params = gs_wrapped.best_params_
                _TUNED_PARAMS_CACHE[cache_key] = best_params
                print(
                    f"[TUNING] XGBoost (MultiOutputRegressor) best params: "
                    f"{best_params} (score={gs_wrapped.best_score_:.4f}) – GECACHT"
                )
                return gs_wrapped.best_estimator_
            except Exception as exc:
                print(f"[TUNING] XGBoost-Tuning fehlgeschlagen ({exc}); ungetunte Defaults.")
                wrapped.fit(X, y)
                return wrapped


def fit_sequence_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int = SEED,
    bootstrap: bool = False,
    do_tune: bool = False,
) -> object:
    """Instanziiert und trainiert ein Sequenzmodell.

    PERF: do_tune=False reicht für den Benchmark (Tuning ist bereits gecacht).
    Falls benötigt: do_tune=True erzwingt Tuning (benutze nur einmal pro Model!).
    """
    if model_name == "linreg":
        model = MultiOutputRegressor(LinearRegression())
        if bootstrap:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(X_train), size=len(X_train), replace=True)
            model.fit(X_train[idx], y_train[idx])
        else:
            model.fit(X_train, y_train)

    elif model_name == "ffn":
        if do_tune:
            model = tune_sequence_model("ffn", X_train, y_train, seed=seed)
        else:
            n = len(X_train)
            val_fraction = 0.15
            n_val = max(1, int(n * val_fraction))
            if n - n_val < 1:
                model = MLPRegressor(
                    hidden_layer_sizes=(128, 64),
                    activation="relu",
                    max_iter=500,
                    early_stopping=False,
                    random_state=seed,
                    learning_rate_init=1e-3,
                )
                model.fit(X_train, y_train)
            else:
                X_fit, y_fit = X_train[: n - n_val], y_train[: n - n_val]
                X_val, y_val = X_train[n - n_val :], y_train[n - n_val :]

                model = MLPRegressor(
                    hidden_layer_sizes=(128, 64),
                    activation="relu",
                    max_iter=1,
                    warm_start=True,
                    early_stopping=False,
                    random_state=seed,
                    learning_rate_init=1e-3,
                )

                patience = 20
                max_total_iter = 500
                best_val_mae = np.inf
                best_params: list[np.ndarray] | None = None
                no_improve = 0

                for _ in range(max_total_iter):
                    model.fit(X_fit, y_fit)
                    val_pred = model.predict(X_val)
                    val_mae = float(mean_absolute_error(y_val, val_pred))
                    if val_mae < best_val_mae:
                        best_val_mae = val_mae
                        best_params = [c.copy() for c in model.coefs_], [i.copy() for i in model.intercepts_]
                        no_improve = 0
                    else:
                        no_improve += 1
                        if no_improve >= patience:
                            break

                if best_params is not None:
                    model.coefs_, model.intercepts_ = best_params
                model.n_iter_ = max_total_iter - no_improve

    elif model_name == "xgboost":
        xgb_kwargs = dict(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.7,
            colsample_bytree=0.7,
            random_state=seed,
            n_jobs=1,
        )
        try:
            XGBRegressor = importlib.import_module("xgboost").XGBRegressor
        except Exception as exc:
            raise ImportError(
                "xgboost konnte nicht importiert werden. Bitte installiere xgboost "
                "(z.B. pip install xgboost) oder prüfe die Installation."
            ) from exc

        if torch.cuda.is_available():
            xgb_kwargs.update(tree_method="gpu_hist", predictor="gpu_predictor", gpu_id=0)

        base_model = XGBRegressor(**xgb_kwargs)

        if do_tune:
            try:
                tuned = tune_sequence_model("xgboost", X_train, y_train, seed=seed)
                return tuned
            except Exception:
                pass

        try:
            base_model.fit(X_train, y_train)
            model = base_model
        except Exception:
            print("[XGBOOST] Native multi-output nicht verfügbar — Fallback auf MultiOutputRegressor.")
            model = MultiOutputRegressor(base_model)
            model.fit(X_train, y_train)

    else:
        raise ValueError(f"Unbekanntes Sequenzmodell: {model_name!r}")

    return model


def _prepare_sequence_data(
    covariates: list[str],
    training_end: str = TRAINING_ENDE,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, RobustScaler, float]:
    """Bereitet Sequenzdaten für Nicht-TFT-Modelle vor.

    FIX 1.3: Sequenz-Split ohne Leakage – kein zukünftiges Target im Training.
    FIX 1.4: Off-by-one bei x_input behoben; Fenster inkl. training_end.
    FIX 1.5: Scaler nur auf Trainingsdaten gefittet.
    FIX 6:   RobustScaler statt MinMaxScaler.
    """
    variablen = variablen_for(covariates)
    df, df_clean, _, _ = load_prepared_df(covariates, training_end=training_end)

    t = _get_loc_snapped(df_clean, training_end)
    snapped_training_end = df_clean.index[t]

    # Scaler ausschließlich auf Trainingsbereich fitten
    train_X = df_clean.iloc[: t + 1][variablen].values
    train_y = df_clean.iloc[: t + 1][[TARGET_DIFF]].values

    scaler_x = RobustScaler().fit(train_X)
    scaler_y = RobustScaler().fit(train_y)

    X_data = scaler_x.transform(df_clean[variablen].values)
    y_data = scaler_y.transform(df_clean[[TARGET_DIFF]].values)

    X_seq, y_seq = create_sequences(X_data, y_data, GEDAECHTNIS, PROGNOSEHORIZONT)

    # FIX 1.3: Nur Sequenzen zulassen, deren letztes Target <= t
    # Sequenz i: Targets bei [i+GD, i+GD+PH-1] → i <= t-GD-PH+1
    n_train = max(0, t - GEDAECHTNIS - PROGNOSEHORIZONT + 2)
    X_train, y_train = X_seq[:n_train], y_seq[:n_train]

    # FIX 1.4: Fenster [t-GD+1 : t+1] → inkl. training_end
    x_input = X_data[t - GEDAECHTNIS + 1 : t + 1].flatten().reshape(1, -1)
    last_level = float(df.loc[snapped_training_end, TARGET_COL])

    return df, df_clean, X_train, y_train, x_input, scaler_y, last_level


def predict_sequence_model(
    model_name: str,
    covariates: list[str],
    simulations: int = 1,
    seed: int = SEED,
    training_end: str = TRAINING_ENDE,
) -> tuple[np.ndarray, np.ndarray]:
    """Ensemble-Prognose mit Sequenzmodellen.

    PERF: Tuning wird nur EINMAL durchgeführt (cacht die Parameter).
    Alle Simulationen verwenden die gleichen besten Parameter, nur mit
    verschiedenen Zufalls-Seeds.

    Returns:
        (volume_pred, diff_pred) – beide als Ensemble-Mittelwert über die
        Simulationen. diff_pred ist der direkte Modell-Output.
    """
    _, _, X_train, y_train, x_input, scaler_y, last_level = _prepare_sequence_data(
        covariates, training_end=training_end
    )

    # PERF: Tuning einmalig VOR der Simulations-Loop
    if model_name in ("ffn", "xgboost") and simulations > 1:
        try:
            _ = tune_sequence_model(model_name, X_train, y_train, seed=seed)
        except Exception as exc:
            print(f"[PREDICT] Tuning für {model_name} fehlgeschlagen: {exc}")

    diff_ensemble: list[np.ndarray] = []
    volume_ensemble: list[np.ndarray] = []

    for i in range(simulations):
        use_boot = simulations > 1
        m = fit_sequence_model(
            model_name,
            X_train,
            y_train,
            seed=seed + i,
            bootstrap=use_boot,
            do_tune=False,  # PERF: Tuning-Parameter bereits cacht (s.o.)
        )
        raw = m.predict(x_input).reshape(PROGNOSEHORIZONT, 1)
        diff_pred = scaler_y.inverse_transform(raw).flatten()
        diff_ensemble.append(diff_pred)
        volume_ensemble.append(reconstruct_volume(diff_pred, last_level))

    return np.mean(volume_ensemble, axis=0), np.mean(diff_ensemble, axis=0)


# ---------------------------------------------------------------------------
# TFT
# ---------------------------------------------------------------------------

def _add_tft_columns(frame: pd.DataFrame, time_offset: int = 0) -> pd.DataFrame:
    """Ergänzt time_idx und GROUP_ID für pytorch-forecasting."""
    out = frame.reset_index()
    if "Datum" not in out.columns:
        out = out.rename(columns={out.columns[0]: "Datum"})
    out[GROUP_ID] = "deposits"
    out["time_idx"] = np.arange(time_offset, time_offset + len(out))
    return out


def _to_numpy(pred: object) -> np.ndarray:
    if hasattr(pred, "output"):
        pred = pred.output
    if isinstance(pred, torch.Tensor):
        return pred.detach().cpu().numpy()
    return np.asarray(pred)


def fit_tft(
    train_df: pd.DataFrame,
    covariates: list[str],
    seed: int,
) -> tuple[TemporalFusionTransformer, TimeSeriesDataSet]:
    """Trainiert TFT im Standard-Benchmark-Design (Kovariaten als unknown_reals)."""
    pl.seed_everything(seed, workers=True)
    torch.manual_seed(seed)

    train_tft_full = _add_tft_columns(train_df)
    n = len(train_tft_full)
    n_val = max(PROGNOSEHORIZONT + 1, int(n * TFT_VAL_FRACTION))
    split_idx = n - n_val          # letzter time_idx im Training

    train_only = train_tft_full[train_tft_full["time_idx"] <= split_idx]

    # STANDARD-BENCHMARK: Alle makroökonomischen Kovariaten sind zukünftig UNBEKANNT.
    # Sie werden nur im Encoder (Vergangenheit) zur Kontextbildung genutzt.
    unknown_reals = [TARGET_COL, TARGET_DIFF] + list(covariates)
    known_reals = []  # Leer im Standard-Benchmark

    training = TimeSeriesDataSet(
        train_only,
        time_idx="time_idx",
        target=TARGET_DIFF,
        group_ids=[GROUP_ID],
        max_encoder_length=GEDAECHTNIS,
        min_encoder_length=max(1, GEDAECHTNIS // 2),
        max_prediction_length=PROGNOSEHORIZONT,
        min_prediction_length=PROGNOSEHORIZONT,
        time_varying_known_reals=known_reals,
        time_varying_unknown_reals=unknown_reals,
        target_normalizer=GroupNormalizer(groups=[GROUP_ID]),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    # Voller Frame für Encoder-Historie; nur Val-Zeitraum wird vorhergesagt
    validation = TimeSeriesDataSet.from_dataset(
        training,
        train_tft_full,               # voller Frame → Encoder hat vollständige Historie
        predict=False,
        stop_randomization=True,
        min_prediction_idx=split_idx + 1,   # nur Zeitschritte nach split_idx vorhersagen
    )

    train_loader = training.to_dataloader(train=True, batch_size=TFT_BATCH_SIZE, num_workers=0)
    val_loader = validation.to_dataloader(train=False, batch_size=TFT_BATCH_SIZE, num_workers=0)

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=TFT_LEARNING_RATE,
        hidden_size=32,
        attention_head_size=4,
        dropout=0.1,
        hidden_continuous_size=16,
        loss=MAE(),
        optimizer="adam",
    )

    trainer = pl.Trainer(
        max_epochs=TFT_MAX_EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        gradient_clip_val=0.1,
        callbacks=[EarlyStopping(monitor="val_loss", patience=5, mode="min")],
        enable_progress_bar=False,
        logger=False,
    )
    trainer.fit(tft, train_loader, val_loader)
    return tft, training


def predict_tft_multi_step(
    model: TemporalFusionTransformer,
    training_dataset: TimeSeriesDataSet,
    history_df: pd.DataFrame,
) -> np.ndarray:
    """Prognostiziert mit TFT über mehrere Schritte (Multi-Step).

    SZENARIO 2A: Keine zukünftigen Kovariatenwerte (nur historisch).
    Das DataFrame history_df endet an TRAINING_ENDE.

    Args:
        model: Trainiertes TFT-Modell
        training_dataset: Training-Dataset (für Metadaten)
        history_df: Historische Daten für Encoding (bis TRAINING_ENDE)

    Returns:
        Prognose-Differenzen für PROGNOSEHORIZONT Schritte
    """
    history_prep = _add_tft_columns(history_df)
    max_time_idx = history_prep["time_idx"].max()

    # Future DataFrame: Struktur für TFT, aber ohne echte Werte
    future_rows = []
    for i in range(1, PROGNOSEHORIZONT + 1):
        row = {GROUP_ID: "deposits", "time_idx": max_time_idx + i}
        for var in training_dataset.time_varying_unknown_reals:
            row[var] = np.nan
        row["Datum"] = pd.Timestamp("1900-01-01")
        future_rows.append(row)

    future_tft = pd.DataFrame(future_rows)
    combined = pd.concat([history_prep, future_tft], ignore_index=True)

    predict_ds = TimeSeriesDataSet.from_dataset(
        training_dataset,
        combined,
        predict=True,
        stop_randomization=True,
    )
    loader = predict_ds.to_dataloader(train=False, batch_size=1, num_workers=0)
    pred = _to_numpy(
        model.predict(
            loader,
            trainer_kwargs=dict(
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                logger=False,
                enable_checkpointing=False,
            ),
        )
    )
    flat = pred.reshape(-1)
    return flat[-PROGNOSEHORIZONT:]


def predict_tft_forecast(
    covariates: list[str],
    simulations: int = 1,
    seed: int = SEED,
    training_end: str = TRAINING_ENDE,
) -> tuple[np.ndarray, np.ndarray]:
    """Prognostiziert mit TFT (Ensemble über Seeds).

    SZENARIO 2A: Nur historische Kovariaten, keine Zukunftswerte.

    Args:
        covariates: Liste der Kovariaten-Namen
        simulations: Anzahl der Ensemble-Durchläufe
        seed: Basis-Seed
        training_end: Letztes Trainingsdatum

    Returns:
        (volume_pred, diff_pred)
    """
    df, df_clean, train_df, _ = load_prepared_df(covariates, training_end=training_end)
    min_train = GEDAECHTNIS + PROGNOSEHORIZONT + 10
    if len(train_df) < min_train:
        raise ValueError(f"Zu wenig Trainingsdaten: {len(train_df)} < {min_train}")

    last_level = float(_loc_value_snapped(df, training_end, TARGET_COL))
    diff_ensemble: list[np.ndarray] = []

    for i in range(simulations):
        model, ds = fit_tft(train_df, covariates, seed=seed + i)
        diff_pred = predict_tft_multi_step(model, ds, train_df)
        diff_ensemble.append(diff_pred)

    mean_diff = np.mean(diff_ensemble, axis=0)
    return reconstruct_volume(mean_diff, last_level), mean_diff


# ---------------------------------------------------------------------------
# Evaluation + Results
# ---------------------------------------------------------------------------

def run_single_evaluation(
    model_name: str,
    covariates: list[str],
    covariate_label: str,
    simulations: int = 1,
    training_end: str = TRAINING_ENDE,
) -> ForecastResult:
    """Führt eine komplette Modell-Evaluation durch."""
    if not covariates and model_name not in ("naive_rw", "naive_hold"):
        raise ValueError("Mindestens eine Kovariate erforderlich (außer für Naive-Modelle).")

    df, df_clean, _, _ = load_prepared_df(covariates, training_end=training_end)

    # Naiv-Modelle
    if model_name == "naive_rw":
        volume_pred, diff_pred = predict_naive_rw(
            covariates, simulations=simulations, seed=SEED, training_end=training_end
        )
    elif model_name == "naive_hold":
        volume_pred, diff_pred = predict_naive_hold(
            covariates, simulations=simulations, seed=SEED, training_end=training_end
        )
    elif model_name == "tft":
        volume_pred, diff_pred = predict_tft_forecast(
            covariates, simulations=simulations, seed=SEED, training_end=training_end
        )
    else:
        volume_pred, diff_pred = predict_sequence_model(
            model_name,
            covariates,
            simulations=simulations,
            seed=SEED,
            training_end=training_end,
        )

    # Test-Wahrheiten
    input_start = _next_date_in_index(df_clean, training_end)
    true_diff, true_volume = _true_test_arrays(df, df_clean, input_start, PROGNOSEHORIZONT)

    metrics = compute_metrics(true_diff, diff_pred, true_volume, volume_pred)
    return ForecastResult(
        model=model_name,
        covariates=covariates,
        covariate_label=covariate_label,
        training_end=training_end,
        diff_pred=diff_pred,
        volume_pred=volume_pred,
        **metrics,
    )


def results_to_dataframe(results: Iterable[ForecastResult]) -> pd.DataFrame:
    """Konvertiert ForecastResults zu pandas DataFrame (sortiert nach MAE_Volumen)."""
    rows = []
    for r in results:
        rows.append(
            {
                "Modell": r.model,
                "Kovariaten": ", ".join(r.covariates),
                "Label": r.covariate_label,
                "n_Kovariaten": len(r.covariates),
                "MAE_Diff": r.mae_diff,
                "RMSE_Diff": r.rmse_diff,
                "R2_Diff": r.r2_diff,
                "MAE_Volumen": r.mae_volume,
                "RMSE_Volumen": r.rmse_volume,
                "R2_Volumen": r.r2_volume,
            }
        )
    return pd.DataFrame(rows).sort_values(["Modell", "MAE_Volumen"])


def generate_rolling_origins(freq: str = "MS", max_origins: int | None = None) -> list[str]:
    """Generiert Rolling-Origin-Daten."""
    df = _load_base_frame()
    ts_range = pd.date_range(start="2023-07-01", end=TRAINING_ENDE, freq=freq)
    origins = [str(d.date()) for d in ts_range]
    if max_origins:
        origins = origins[-max_origins:]
    return origins


def run_rolling_evaluation(
    model_name: str,
    covariates: list[str],
    covariate_label: str,
    simulations: int = 1,
    origins: list[str] | None = None,
) -> pd.DataFrame:
    """Rolling-Origin-Evaluation."""
    if origins is None:
        origins = generate_rolling_origins()

    results = []
    for origin in origins:
        try:
            result = run_single_evaluation(
                model_name=model_name,
                covariates=covariates,
                covariate_label=covariate_label,
                simulations=simulations,
                training_end=origin,
            )
            row = {
                "origin": origin,
                "model": result.model,
                "label": result.covariate_label,
                "mae_diff": result.mae_diff,
                "rmse_diff": result.rmse_diff,
                "r2_diff": result.r2_diff,
                "mae_volume": result.mae_volume,
                "rmse_volume": result.rmse_volume,
                "r2_volume": result.r2_volume,
            }
            results.append(row)
        except Exception as exc:
            print(f"  [Rolling] Origin {origin} / {model_name}: {exc}")
    return pd.DataFrame(results)


def aggregate_rolling_results(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregiert Rolling-Ergebnisse pro Modell."""
    return df.groupby("model").agg(
        {
            "mae_volume": ["mean", "std"],
            "rmse_volume": ["mean", "std"],
            "r2_volume": ["mean", "std"],
        }
    )
