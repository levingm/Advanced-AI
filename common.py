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
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
# FIX 6: RobustScaler statt MinMaxScaler (robust gegen Ausreißer in Finanzdaten)
from sklearn.preprocessing import RobustScaler
from sklearn.base import clone


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

# --- Punkt 1 (Lags): Kovariaten wirken in der Realität meist verzögert auf
# das Einlagenverhalten (z.B. Zinsänderungen brauchen Zeit, bis Kunden
# reagieren). Ohne Lags kann kein Sequenzmodell diesen Mechanismus abbilden,
# selbst wenn er in den Daten steckt. LAG_STEPS definiert die Standard-Lags
# in Handelstagen; mit use_lags=True werden für jede Kovariate zusätzliche
# Spalten f"{col}_lag{L}" erzeugt (kurzfristig: 1 Tag, mittelfristig: 1 Woche,
# längerfristig: ca. 1 Monat).
LAG_STEPS = [1, 5, 20]

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

# ---------------------------------------------------------------------------
# Tuning-Cache: Tuning läuft einmalig pro (model_name, cov_key, training_end).
# Schlüssel: Tupel (model_name, frozenset(covariates+lag_flags), training_end)
# Wert: fertig gefitteter Estimator (wird anschließend via clone() kopiert).
# Persistiert über Rolling-Origins hinweg innerhalb einer Python-Session.
# Cache löschen: tc.clear_tuning_cache() oder tc._TUNING_CACHE.clear()
# ---------------------------------------------------------------------------
_TUNING_CACHE: dict[tuple, object] = {}


def _tuning_cache_key(
    model_name: str,
    covariates: list[str],
    training_end: str,
    use_lags: bool,
) -> tuple:
    """Eindeutiger, hashbarer Cache-Schlüssel für ein Tuning-Ergebnis."""
    return (model_name, tuple(sorted(covariates)), training_end, use_lags)


def clear_tuning_cache() -> None:
    """Leert den modulglobalen Tuning-Cache (z.B. nach Datenwechsel)."""
    _TUNING_CACHE.clear()
    print("[TUNING-CACHE] Cache geleert.")
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
def variablen_for(covariates: list[str], use_lags: bool = False) -> list[str]:
    """Kombiniert Target + Kovariaten (+ optional Lag-Spalten) für DataFrame-Selektion.

    Args:
        covariates: Basis-Kovariaten (Spaltennamen zum Zeitpunkt t)
        use_lags:   Falls True, werden zusätzlich die Lag-Spalten
                    f"{col}_lag{L}" für L in LAG_STEPS aufgenommen (Punkt 1).
                    Die Originalspalte zu t bleibt zusätzlich erhalten, damit
                    sowohl sofortige als auch verzögerte Wirkung verglichen
                    werden können.
    """
    cols = [TARGET_COL] + list(covariates)
    if use_lags:
        cols += lag_column_names(covariates)
    return cols


def lag_column_names(covariates: list[str], lags: list[int] | None = None) -> list[str]:
    """Liefert die Namen aller Lag-Spalten für die gegebenen Kovariaten."""
    lags = lags if lags is not None else LAG_STEPS
    return [f"{col}_lag{L}" for col in covariates for L in lags]


def add_lags(
    df: pd.DataFrame,
    covariates: list[str],
    lags: list[int] | None = None,
) -> pd.DataFrame:
    """Ergänzt verzögerte Kovariaten-Spalten f"{col}_lag{L}" (Punkt 1: Lags).

    Begründung (siehe Kritik 2.3): Zinsänderungen, Marktbewegungen etc. wirken
    in der Realität meist verzögert auf das Einlagenverhalten. Ohne Lags
    unterstellt der Code implizit eine sofortige, zeitgleiche Wirkung – das
    ist ökonomisch unplausibel. add_lags() erzeugt zusätzliche Spalten, die
    den Wert der jeweiligen Kovariate L Handelstage zuvor enthalten, sodass
    Modelle (insbesondere LinReg/FFN/XGBoost, die keine eigene Gedächtnis-
    struktur für Kovariaten-Lags haben) verzögerte Effekte direkt als Feature
    sehen können.

    Wichtig: shift() arbeitet positions-/indexbasiert auf dem bestehenden
    DatetimeIndex (Handelstage), nicht auf Kalendertagen – ein "Lag von 5"
    bedeutet daher 5 Handelstage, nicht 5 Kalendertage.
    """
    lags = lags if lags is not None else LAG_STEPS
    out = df.copy()
    for col in covariates:
        if col not in out.columns:
            continue
        for L in lags:
            out[f"{col}_lag{L}"] = out[col].shift(L)
    return out


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
    use_lags: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Lädt und bereitet Daten vor.

    Args:
        covariates: Zu nutzende Kovariaten.
        training_end: Letztes Trainingsdatum (für Rolling-Origin parametrierbar).
        prognosehorizont: Länge des Prognose-Horizonts.
        use_lags: Falls True, werden zusätzlich gelagte Kovariaten-Spalten
            (Punkt 1) erzeugt und mitgeführt. Die ersten max(LAG_STEPS) Zeilen
            fallen dabei zwangsläufig durch dropna() weg (kein Lag verfügbar).

    Returns:
        df:         Vollständiger Frame inkl. NaN (echte Test-Levels abrufbar)
        df_clean:   Frame ohne NaN (für Sequenz-Training)
        train_df:   Trainingsdaten bis training_end
        future_df:  Referenzdaten für den Test-Horizont (nicht als Modell-Input)
    """
    base_cols = [TARGET_COL] + list(covariates)
    df = _load_base_frame()[base_cols].copy()
    if use_lags and covariates:
        df = add_lags(df, covariates)
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


# BUGFIX 2: Rolling-Origins können auf Wochenenden/Feiertage fallen, die im
# (nur Bankarbeitstage enthaltenden) Datenindex nicht existieren. Statt eines
# rohen .loc[]/.get_loc()-Zugriffs (KeyError) snappen wir konsequent auf den
# letzten verfügbaren Handelstag <= dem angefragten Datum.
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
# Sequenzmodelle: LinReg, FFN, XGBoost
# ---------------------------------------------------------------------------

# -------------------------
# Hilfsfunktion: Zeitserien-konformes Tuning (FFN + XGBoost)
# -------------------------
# Punkt 3: Einheitliches Tuning-Budget – jedes Modell testet eine
# vergleichbare Anzahl Hyperparameter-Kombinationen. Die genaue Grid-Größe
# ist je Modell dokumentiert in tune_sequence_model (LinReg: 8, FFN: 4,
# XGBoost: 12). Ein einziger globaler TUNING_BUDGET-Wert passte nicht zu
# den sehr unterschiedlichen Laufzeiten der Modelle.

def tune_sequence_model(model_name: str, X: np.ndarray, y: np.ndarray, seed: int = SEED):
    """Grid-Search-Tuning mit TimeSeriesSplit für linreg, FFN und XGBoost.

    Alle drei Modelle bekommen ein Parameter-Grid mit annähernd demselben
    Budget TUNING_BUDGET an Kombinationen, damit der spätere Modellvergleich
    nicht durch unterschiedlich tiefes Tuning verzerrt wird (Punkt 3).
    Gibt ein gefittetes Modell zurück (beste Param.-Kombination).
    Ziel: zeitserien-konformes CV (kein shuffle), vermeidet Leak im FFN.

    Performance-Entscheidungen:
    - n_splits=3 (TimeSeriesSplit): Minimum für stabile CV-Schätzung bei kurzen
      Zeitreihen. Mehr Splits verlängern die Laufzeit linear.
    - XGBoost: n_estimators=50 im Tuning-Grid (statt 100) – genug, um
      Hyperparameter zu unterscheiden; nach Bestimmung der besten Params
      wird mit vollen 100 Trees auf den Trainingsdaten refittet.
    - n_jobs=-1 im GridSearchCV: parallelisiert die CV-Folds über alle Kerne.
    """
    tscv = TimeSeriesSplit(n_splits=3)

    if model_name == "linreg":
        base = MultiOutputRegressor(Ridge(random_state=seed))
        # 8 Log-äquidistante Alpha-Werte: ausreichend, um das Optimum zu finden,
        # schneller als 12 (und identische Ergebnisse, da LinReg-CV trivial schnell ist).
        alphas = np.logspace(-3, 3, 8).tolist()
        param_grid = {"estimator__alpha": alphas}
        gs = GridSearchCV(
            base, param_grid, cv=tscv, scoring="neg_mean_absolute_error", n_jobs=-1
        )
        gs.fit(X, y)
        print(f"[TUNING] LinReg (Ridge) best: alpha={gs.best_params_['estimator__alpha']:.4g} "
              f"score={gs.best_score_:.4f}")
        return gs.best_estimator_

    elif model_name == "ffn":
        # FFN-Tuning: kleines Grid, weil MLPRegressor langsam ist.
        # 2 × 2 = 4 Kombinationen – bewusst klein gehalten; der Großteil der
        # Laufzeitoptimierung kommt vom Early-Stopping (patience=20) im Predict-Pfad.
        param_grid = {
            "hidden_layer_sizes": [(64, 32), (128, 64)],
            "alpha": [1e-4, 1e-3],
        }
        n_combos = len(param_grid["hidden_layer_sizes"]) * len(param_grid["alpha"])
        base = MLPRegressor(
            random_state=seed, max_iter=200, early_stopping=False,
            learning_rate_init=1e-3, n_iter_no_change=20,
        )
        gs = GridSearchCV(
            base, param_grid, cv=tscv, scoring="neg_mean_absolute_error", n_jobs=-1
        )
        gs.fit(X, y)
        print(f"[TUNING] FFN best ({n_combos} Kombinationen): {gs.best_params_} "
              f"score={gs.best_score_:.4f}")
        return gs.best_estimator_

    elif model_name == "xgboost":
        XGB = importlib.import_module("xgboost").XGBRegressor
        # n_estimators=50 im Tuning (schnell); nach Param-Selektion wird mit
        # vollen 100 Trees refittet (passiert automatisch via gs.best_estimator_.fit).
        # n_jobs=-1: parallelisiert XGBoost intern + GridSearchCV-Folds.
        base = XGB(
            n_estimators=50, random_state=seed, n_jobs=-1, verbosity=0,
            objective="reg:squarederror",
        )
        # 3 × 2 × 2 = 12 Kombinationen – maximale Parallelisierung durch n_jobs=-1
        param_grid = {
            "max_depth": [3, 4, 5],
            "learning_rate": [0.05, 0.1],
            "subsample": [0.7, 0.9],
        }
        n_combos = len(param_grid["max_depth"]) * len(param_grid["learning_rate"]) * len(param_grid["subsample"])
        gs = GridSearchCV(
            base, param_grid, cv=tscv, scoring="neg_mean_absolute_error", n_jobs=-1
        )
        try:
            gs.fit(X, y)
            print(f"[TUNING] XGBoost native best ({n_combos} Kombinationen): "
                  f"{gs.best_params_} score={gs.best_score_:.4f}")
            return gs.best_estimator_
        except Exception:
            wrapped = MultiOutputRegressor(base)
            wrapped_param_grid = {f"estimator__{k}": v for k, v in param_grid.items()}
            gs_wrapped = GridSearchCV(
                wrapped, wrapped_param_grid, cv=tscv,
                scoring="neg_mean_absolute_error", n_jobs=-1,
            )
            try:
                gs_wrapped.fit(X, y)
                print(
                    f"[TUNING] XGBoost (MultiOutputRegressor) best: "
                    f"{gs_wrapped.best_params_} score={gs_wrapped.best_score_:.4f}"
                )
                return gs_wrapped.best_estimator_
            except Exception as exc2:
                print(
                    f"[TUNING] XGBoost-Tuning vollständig fehlgeschlagen ({exc2}); "
                    "verwende ungetunte Default-Parameter."
                )
                wrapped.fit(X, y)
                return wrapped

    else:
        raise ValueError("Tuning only supported for 'linreg', 'ffn' and 'xgboost'.")


def fit_sequence_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int = SEED,
    bootstrap: bool = False,
    do_tune: bool = False,
    tuned_estimator: object | None = None,
) -> object:
    """Instanziiert und trainiert ein Sequenzmodell.

    tuned_estimator: Falls gesetzt, wird ein Klon dieses estimators verwendet
    (nützlich, wenn Tuning einmalig durchgeführt und das Ergebnis wiederverwendet wird).
    """
    # Helper: fit a cloned estimator on (maybe bootstrapped) training data
    def _fit_cloned(estimator):
        model_clone = clone(estimator)
        if bootstrap:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(X_train), size=len(X_train), replace=True)
            model_clone.fit(X_train[idx], y_train[idx])
        else:
            model_clone.fit(X_train, y_train)
        return model_clone

    if model_name == "linreg":
        if tuned_estimator is not None:
            return _fit_cloned(tuned_estimator)
        if do_tune:
            model = tune_sequence_model("linreg", X_train, y_train, seed=seed)
            return model
        model = MultiOutputRegressor(LinearRegression())
        if bootstrap:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(X_train), size=len(X_train), replace=True)
            model.fit(X_train[idx], y_train[idx])
        else:
            model.fit(X_train, y_train)
        return model

    elif model_name == "ffn":
        if tuned_estimator is not None:
            return _fit_cloned(tuned_estimator)
        if do_tune:
            model = tune_sequence_model("ffn", X_train, y_train, seed=seed)
            return model

        # existing FFN training logic (keine Änderung der Semantik)
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
        return model

    elif model_name == "xgboost":
        # prepare xgboost kwargs (GPU-override if available)
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

        if tuned_estimator is not None:
            # Clone and fit tuned estimator on (possibly bootstrapped) X_train
            return _fit_cloned(tuned_estimator)

        if do_tune:
            try:
                tuned = tune_sequence_model("xgboost", X_train, y_train, seed=seed)
                return tuned
            except Exception:
                pass

        base_model = XGBRegressor(**xgb_kwargs)

        try:
            base_model.fit(X_train, y_train)
            model = base_model
        except Exception:
            print("[XGBOOST] Native multi-output nicht verfügbar — Fallback auf MultiOutputRegressor.")
            model = MultiOutputRegressor(base_model)
            model.fit(X_train, y_train)

        return model

    else:
        raise ValueError(f"Unbekanntes Sequenzmodell: {model_name!r}")


def _prepare_sequence_data(
    covariates: list[str],
    training_end: str = TRAINING_ENDE,
    use_lags: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, RobustScaler, float]:
    """Bereitet Sequenzdaten für Nicht-TFT-Modelle vor.

    FIX 1.3: Sequenz-Split ohne Leakage – kein zukünftiges Target im Training.
    FIX 1.4: Off-by-one bei x_input behoben; Fenster inkl. training_end.
    FIX 1.5: Scaler nur auf Trainingsdaten gefittet.
    FIX 6:   RobustScaler statt MinMaxScaler.
    Punkt 1: use_lags=True nimmt zusätzlich verzögerte Kovariaten-Spalten auf.
    """
    variablen = variablen_for(covariates, use_lags=use_lags)
    df, df_clean, _, _ = load_prepared_df(covariates, training_end=training_end, use_lags=use_lags)

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
    use_lags: bool = False,
    do_tune: bool = False,
    tuned_estimator: object | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Ensemble-Prognose mit Sequenzmodellen.

    Tuning-Caching (Performance-Fix):
    Tuning läuft maximal einmal pro eindeutiger Kombination aus
    (model_name, covariates, training_end, use_lags) – auch über Rolling-
    Origins hinweg. Das Ergebnis wird in _TUNING_CACHE gespeichert und bei
    allen nachfolgenden Aufrufen mit identischem Schlüssel wiederverwendet.
    Dadurch entfällt das bisher 10×-redundante Tuning pro Simulation und das
    N_Origins-fache Tuning im Rolling-Benchmark.
    """
    _, _, X_train, y_train, x_input, scaler_y, last_level = _prepare_sequence_data(
        covariates, training_end=training_end, use_lags=use_lags
    )

    # Tuned estimator bestimmen: Priorität (1) explizit übergeben, (2) Cache,
    # (3) jetzt tunen und cachen.
    if do_tune and tuned_estimator is None:
        cache_key = _tuning_cache_key(model_name, covariates, training_end, use_lags)
        if cache_key in _TUNING_CACHE:
            tuned_estimator = _TUNING_CACHE[cache_key]
            print(f"[TUNING-CACHE] Hit für {model_name}/{cache_key[1]} "
                  f"@ {training_end} – Tuning übersprungen.")
        else:
            try:
                print(f"[TUNING] {model_name} | {sorted(covariates)} | {training_end} "
                      f"{'(+lags)' if use_lags else ''} ...")
                tuned_estimator = tune_sequence_model(model_name, X_train, y_train, seed=seed)
                _TUNING_CACHE[cache_key] = tuned_estimator
            except Exception as exc:
                print(f"[TUNING] Fehlgeschlagen ({exc}). Fahre ohne Tuning fort.")
                tuned_estimator = None

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
            do_tune=False,
            tuned_estimator=tuned_estimator,
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
    """Ergänzt time_idx, GROUP_ID und Kalenderfeatures für pytorch-forecasting."""
    out = frame.reset_index()
    if "Datum" not in out.columns:
        out = out.rename(columns={out.columns[0]: "Datum"})
    out[GROUP_ID] = "deposits"
    out["time_idx"] = np.arange(time_offset, time_offset + len(out))
    out = add_calendar_features(out, date_col="Datum")
    return out


# ---------------------------------------------------------------------------
# Punkt 6: Echte time_varying_known_reals für TFT (Kalenderfeatures)
# ---------------------------------------------------------------------------

CALENDAR_FEATURE_COLS = ["wochentag", "ist_monatsultimo", "tag_im_monat"]


def add_calendar_features(frame: pd.DataFrame, date_col: str = "Datum") -> pd.DataFrame:
    """Ergänzt Kalenderfeatures, die für JEDEN zukünftigen Tag exakt bekannt sind.

    Begründung (Kritik Punkt 3.1): TFT zieht seinen Mehrwert aus echten
    time_varying_known_reals – Größen, die für den Prognosehorizont bereits
    feststehen. Im bisherigen Code war known_reals=[] und alle Kovariaten
    liefen als unknown_reals mit naiver Fortschreibung in der Zukunft; TFT
    hatte dadurch keinen einzigen Informationsvorteil gegenüber den
    Sequenzmodellen und wurde faktisch zu einem überdimensionierten
    Sequenzmodell degradiert.

    Kalenderfeatures sind die einzigen Größen in diesem Datensatz, die per
    Definition für die Zukunft exakt bekannt sind (kein Naive-Hold-Trick
    nötig) – Wochentag, Monatsultimo-Indikator (potenziell relevant für
    Lohnzahlungstermine/Monatsabschluss-Effekte bei NMD, siehe Kritik 2.1)
    und Tag im Monat. Sie sind kein Ersatz für echte ökonomische
    Zukunftsinformation, geben TFT aber zumindest einen echten
    `known_reals`-Informationsvorteil, statt komplett ohne einen solchen
    zu laufen.
    """
    out = frame.copy()
    dates = pd.to_datetime(out[date_col])
    out["wochentag"] = dates.dt.dayofweek.astype(float)
    out["ist_monatsultimo"] = dates.dt.is_month_end.astype(float)
    out["tag_im_monat"] = dates.dt.day.astype(float)
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
    use_calendar_known_reals: bool = True,
) -> tuple[TemporalFusionTransformer, TimeSeriesDataSet]:
    """Trainiert TFT.

    Args:
        use_calendar_known_reals: Punkt 6 – falls True (Standard), werden
            Kalenderfeatures (siehe add_calendar_features) als echte
            time_varying_known_reals genutzt, sodass TFT einen tatsächlichen
            Informationsvorteil gegenüber den Sequenzmodellen hat. Falls
            False, läuft die ursprüngliche, explizit unterkonfigurierte
            Baseline-Variante (known_reals=[]) – nützlich, um den Effekt der
            known_reals im Paper explizit auszuweisen.
    """
    pl.seed_everything(seed, workers=True)
    torch.manual_seed(seed)

    train_tft_full = _add_tft_columns(train_df)
    n = len(train_tft_full)
    n_val = max(PROGNOSEHORIZONT + 1, int(n * TFT_VAL_FRACTION))
    split_idx = n - n_val          # letzter time_idx im Training

    train_only = train_tft_full[train_tft_full["time_idx"] <= split_idx]

    # Makroökonomische Kovariaten bleiben unknown_reals (sie sind in der
    # Realität für die Zukunft nicht bekannt). Kalenderfeatures sind dagegen
    # für JEDEN Tag (Vergangenheit wie Zukunft) exakt berechenbar und gehen
    # daher als known_reals ein (Punkt 6).
    #
    # BUGFIX TFT-NaN: TARGET_COL ("Einlagevolumen") darf NICHT in unknown_reals
    # stehen – es ist das Target, kein Feature. pytorch-forecasting validiert
    # alle als unknown_reals deklarierten Spalten auf NaN-Freiheit im gesamten
    # Frame (inkl. Validierungsfenster), was bei TARGET_COL im Decoder-Bereich
    # fehlschlägt. TARGET_DIFF ist ebenfalls das (differenzierte) Target und
    # wird als solches vom Dataset intern verwaltet; es braucht nicht extra als
    # Feature deklariert zu werden.
    unknown_reals = list(covariates)  # nur externe Kovariaten
    known_reals = list(CALENDAR_FEATURE_COLS) if use_calendar_known_reals else []

    # TARGET_DIFF im val-Frame (split_idx < time_idx <= n) könnte NaN enthalten,
    # falls dort noch keine echten Differenzen berechnet wurden. Wir füllen
    # fehlende Werte mit 0 – pytorch-forecasting normalisiert den Target-Wert
    # ohnehin intern; 0-Füllung im Val-Frame beeinflusst nur den Val-Loss, nicht
    # die Test-Prognose.
    train_tft_full[TARGET_DIFF] = train_tft_full[TARGET_DIFF].fillna(0.0)
    train_tft_full[TARGET_COL] = train_tft_full[TARGET_COL].fillna(
        train_tft_full[TARGET_COL].ffill()
    )

    # Kovariaten in train_tft_full forward-füllen (falls irgendwo NaN durch
    # Lag-Berechnung oder fehlende Marktdaten entstanden sind).
    for col in list(covariates) + list(CALENDAR_FEATURE_COLS):
        if col in train_tft_full.columns:
            train_tft_full[col] = train_tft_full[col].ffill().bfill()

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
    covariates: list[str],
) -> np.ndarray:
    """Multi-Step-TFT-Prognose für den Standard-Benchmark.

    BUGFIX 4: Kovariaten im Vorhersagefenster (Decoder) werden NICHT mehr auf
    NaN gesetzt. pytorch_forecasting normalisiert Inputs auch im Decoder
    (z.B. via GroupNormalizer), wodurch NaN-Werte zu NaN-Outputs führen, die
    sich still durch reconstruct_volume fortpflanzen (flache/NaN-Linie ohne
    Crash). Stattdessen wird der letzte bekannte Wert (Naive-Hold) für jede
    Kovariate fortgeschrieben – das Modell ignoriert sie ohnehin nicht aktiv
    im Decoder (unknown_reals), aber ein gültiger, konstanter Wert verhindert
    die NaN-Propagation, ohne neues Zukunftswissen einzuführen.

    Punkt 6: Die Kalenderfeatures (wochentag, ist_monatsultimo, tag_im_monat)
    werden dagegen NICHT per Naive-Hold fortgeschrieben, sondern direkt aus
    dem tatsächlichen Zukunftsdatum jeder Decoder-Zeile berechnet – das ist
    der ganze Punkt von time_varying_known_reals: diese Werte sind für die
    Zukunft exakt bekannt, kein Trick nötig.
    """
    history_prep = _add_tft_columns(history_df)
    max_time_idx = int(history_prep["time_idx"].max())
    last_date = history_prep["Datum"].max()

    future_rows = []
    for i in range(1, PROGNOSEHORIZONT + 1):
        row = {
            GROUP_ID: "deposits",
            "time_idx": max_time_idx + i,
            "Datum": last_date + pd.Timedelta(days=i),
            TARGET_COL: np.nan,
            TARGET_DIFF: np.nan,
        }
        # BUGFIX 4: Naive-Hold (letzter bekannter Wert) statt NaN, um
        # NaN-Propagation durch die Normalisierung zu verhindern.
        for cov in covariates:
            row[cov] = float(history_prep[cov].iloc[-1])
        future_rows.append(row)

    future_df = pd.DataFrame(future_rows)
    # Punkt 6: Kalenderfeatures aus dem echten Zukunftsdatum berechnen statt
    # fortzuschreiben – sie sind die einzigen hier wirklich "known reals".
    future_df = add_calendar_features(future_df, date_col="Datum")

    combined = pd.concat([history_prep, future_df], ignore_index=True)
    predict_ds = TimeSeriesDataSet.from_dataset(
        training_dataset, combined, predict=True, stop_randomization=True
    )
    loader = predict_ds.to_dataloader(train=False, batch_size=1, num_workers=0)
    trainer_device_kwargs = dict(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
    )
    pred = _to_numpy(
        model.predict(
            loader,
            trainer_kwargs=trainer_device_kwargs,
        )
    )
    flat = pred.reshape(-1)
    if len(flat) < PROGNOSEHORIZONT:
        raise ValueError(f"TFT lieferte {len(flat)} Werte, erwartet {PROGNOSEHORIZONT}.")
    return flat[-PROGNOSEHORIZONT:]


def predict_tft_forecast(
    covariates: list[str],
    simulations: int = 1,
    seed: int = SEED,
    training_end: str = TRAINING_ENDE,
    use_calendar_known_reals: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Ensemble-Prognose mit TFT (verschiedene Seeds).

    Args:
        use_calendar_known_reals: Punkt 6 – siehe fit_tft(). Standard True
            (TFT bekommt echte known_reals statt komplett ohne sie zu laufen).

    Returns:
        (volume_pred, diff_pred) – beide als Ensemble-Mittelwert. diff_pred ist
        der direkte TFT-Decoder-Output (BUGFIX 5: nicht über
        np.diff(volume_pred) rückgerechnet).
    """
    df, _, train_df, _ = load_prepared_df(covariates, training_end=training_end)
    min_train = GEDAECHTNIS + PROGNOSEHORIZONT + int(
        (GEDAECHTNIS + PROGNOSEHORIZONT) * TFT_VAL_FRACTION
    ) + 10
    if len(train_df) < min_train:
        raise ValueError(f"Zu wenig Trainingsdaten für TFT: {len(train_df)} < {min_train}.")

    last_level = float(_loc_value_snapped(df, training_end, TARGET_COL))
    diff_ensemble: list[np.ndarray] = []
    volume_ensemble: list[np.ndarray] = []
    for i in range(simulations):
        m, ds = fit_tft(train_df, covariates, seed=seed + i, use_calendar_known_reals=use_calendar_known_reals)
        diff_pred = predict_tft_multi_step(m, ds, train_df, covariates)
        diff_ensemble.append(diff_pred)
        volume_ensemble.append(reconstruct_volume(diff_pred, last_level))
    return np.mean(volume_ensemble, axis=0), np.mean(diff_ensemble, axis=0)


# ---------------------------------------------------------------------------
# Zentrale Evaluationsfunktionen
# ---------------------------------------------------------------------------

_SEQUENCE_MODELS = {"linreg", "ffn", "xgboost"}
_NAIVE_MODELS = {"naive_rw", "naive_hold"}


def run_single_evaluation(
    model_name: str,
    covariates: list[str],
    covariate_label: str,
    simulations: int = 1,
    training_end: str = TRAINING_ENDE,
    use_lags: bool = False,
    do_tune: bool = False,
    use_calendar_known_reals: bool = True,
    tuned_estimator: object | None = None,
) -> ForecastResult:
    """Führt eine vollständige Modell-Evaluation durch.

    Args:
        model_name:      Eines von ALL_MODELS
        covariates:      Kovariatenliste (für naive Modelle darf dies leer sein,
                          siehe BUGFIX 1)
        covariate_label: Bezeichner für Ergebniszeile
        simulations:     Anzahl Ensemble-Läufe
        training_end:    Letztes Trainingsdatum (für Rolling-Origin)
        use_lags:        Punkt 1 – verzögerte Kovariaten als Zusatzfeatures
                         (nur für Sequenzmodelle wirksam; naive/TFT ignorieren dies)
        do_tune:         Punkt 3 – GridSearchCV mit einheitlichem Budget für
                         alle Sequenzmodelle aktivieren (statt nur für ffn/xgboost)
        use_calendar_known_reals: Punkt 6 – nur für TFT relevant. True (Standard)
                         gibt TFT echte time_varying_known_reals (Kalenderfeatures)
                         statt komplett ohne known_reals zu laufen (siehe fit_tft).

    Returns:
        ForecastResult mit allen Metriken und Prognose-Arrays
    """
    # BUGFIX 1: Naive Modelle (naive_rw, naive_hold) benötigen keine Kovariaten
    # und werden im Benchmark bewusst mit covariates=[] aufgerufen. Der
    # Pflicht-Check gilt daher nur noch für Nicht-Naive-Modelle.
    if not covariates and model_name not in _NAIVE_MODELS:
        raise ValueError("Mindestens eine Kovariate erforderlich.")

    df, df_clean, _, _ = load_prepared_df(covariates, training_end=training_end, use_lags=use_lags)
    input_start = _next_date_in_index(df_clean, training_end)
    true_diff, true_volume = _true_test_arrays(df, df_clean, input_start)

    # BUGFIX 5: Alle predict_*-Funktionen liefern jetzt (volume_pred, diff_pred)
    # direkt aus dem jeweiligen Modell-Output, statt diff_pred nachträglich
    # per np.diff(volume_pred) numerisch zurückzurechnen.
    if model_name in _SEQUENCE_MODELS:
        volume_pred, diff_pred = predict_sequence_model(
            model_name,
            covariates=covariates,
            simulations=simulations,
            training_end=training_end,
            use_lags=use_lags,
            do_tune=do_tune,
            tuned_estimator=tuned_estimator,
        )
    elif model_name == "tft":
        volume_pred, diff_pred = predict_tft_forecast(
            covariates=covariates,
            simulations=simulations,
            training_end=training_end,
            use_calendar_known_reals=use_calendar_known_reals,
        )
    elif model_name == "naive_rw":
        volume_pred, diff_pred = predict_naive_rw(
            covariates=covariates, simulations=simulations, training_end=training_end
        )
    elif model_name == "naive_hold":
        volume_pred, diff_pred = predict_naive_hold(
            covariates=covariates, simulations=simulations, training_end=training_end
        )
    else:
        raise ValueError(
            f"Unbekanntes Modell: {model_name!r}. Gültig: {ALL_MODELS}"
        )

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


# ---------------------------------------------------------------------------
# FIX 1: Rolling-Origin-Evaluation
# ---------------------------------------------------------------------------

def generate_rolling_origins(
    freq: str = "MS",
    last_origin: str = TRAINING_ENDE,
    min_train_rows: int | None = None,
) -> list[str]:
    """Generiert Evaluations-Origins für Rolling-Origin-Evaluation.

    Args:
        freq:           Pandas-Frequenz der Origins (z.B. "MS" monatlich, "YS" jährlich)
                        oder Jahres-Schritt als "<n>Y" (z.B. "5Y" = alle 5 Jahre).
        last_origin:    Letztes/spätestes Origin-Datum
        min_train_rows: Mindest-Trainingszeilen (default: GD + PH + Puffer)

    Returns:
        Sortierte Liste von Datumsstrings (YYYY-MM-DD)
    """
    min_rows = min_train_rows or (GEDAECHTNIS + PROGNOSEHORIZONT + 20)

    # Basis-Frame für Datumsverfügbarkeit laden
    df_base = _load_base_frame()[[TARGET_COL]].copy()
    df_base[TARGET_DIFF] = df_base[TARGET_COL].diff()
    df_clean = df_base.dropna()

    if len(df_clean) < min_rows + PROGNOSEHORIZONT:
        raise ValueError("Zu wenig Daten für Rolling-Origin-Evaluation.")

    # Frühestes sinnvolles Origin
    earliest = df_clean.index[min_rows - 1]
    last_ts = pd.Timestamp(last_origin)

    # Unterstützung für Jahres-Intervalle im Format '5Y', '10Y' usw.
    import re
    m = re.match(r"^(\d+)Y$", str(freq))
    if m:
        step = int(m.group(1))
        candidates = pd.date_range(earliest, last_ts, freq=pd.DateOffset(years=step))
    else:
        # sonst direkt mit Pandas-Frequenz-String arbeiten (z.B. "MS", "YS", "A")
        candidates = pd.date_range(earliest, last_ts, freq=freq)

    origins: list[str] = []
    for cand in candidates:
        # Nächstliegendes verfügbares Datum <= Kandidat
        avail = df_clean.index[df_clean.index <= cand]
        if len(avail) >= min_rows:
            origins.append(str(avail[-1].date()))

    # Sicherstellen, dass last_origin immer enthalten ist
    if last_origin not in origins:
        origins.append(last_origin)

    return sorted(set(origins))


# ---------------------------------------------------------------------------
# Punkt 5: Zinsregime als explizites Split-Kriterium
# ---------------------------------------------------------------------------

# Feste Regimegrenzen (in Prozentpunkten) für €STR – grobe, aber für den
# Euroraum 2022-2024 plausible Einteilung: negative/Nullzins-Phase,
# Zinswende/Übergang, Hochzinsphase. Diese Schwellen sind eine bewusste,
# dokumentierte Vereinfachung (siehe Kritik Punkt 1) und sollten im Paper
# explizit benannt und ggf. an den tatsächlichen Datenzeitraum angepasst
# werden – sie ersetzen keine offizielle EZB-Zyklusdatierung.
REGIME_BINS = [-np.inf, 0.0, 2.0, np.inf]
REGIME_LABELS = ["negativ_null", "zinswende", "hochzins"]


def assign_interest_regime(
    origins: list[str],
    regime_col: str = "€STR",
    bins: list[float] | None = None,
    labels: list[str] | None = None,
) -> dict[str, str]:
    """Ordnet jedem Rolling-Origin ein Zinsregime zu (Punkt 5).

    Begründung (Kritik Punkt 1): Ohne explizite Regime-Kennzeichnung lässt
    sich nicht prüfen, ob ein Modell über verschiedene Zinsphasen robust ist
    oder nur in einer bestimmten Phase zufällig gut performt. Diese Funktion
    liest den Wert von `regime_col` (Standard: €STR) am jeweiligen Origin-
    Datum und ordnet ihn anhand fester Schwellen (REGIME_BINS) einem
    benannten Regime zu. Die Regime-Information kann anschließend zur
    stratifizierten Auswertung der Rolling-Ergebnisse genutzt werden
    (siehe aggregate_rolling_results, run_rolling_evaluation(regime_col=...)).

    Returns:
        dict {origin_datum_str: regime_label}
    """
    bins = bins if bins is not None else REGIME_BINS
    labels = labels if labels is not None else REGIME_LABELS
    base = _load_base_frame()

    result: dict[str, str] = {}
    for origin in origins:
        try:
            val = float(_loc_value_snapped(base, origin, regime_col))
        except Exception:
            result[origin] = "unbekannt"
            continue
        idx = np.digitize([val], bins)[0] - 1
        idx = min(max(idx, 0), len(labels) - 1)
        result[origin] = labels[idx]
    return result


def run_rolling_evaluation(
    model_name: str,
    covariates: list[str],
    covariate_label: str,
    simulations: int = 1,
    origins: list[str] | None = None,
    freq: str = "MS",
    use_lags: bool = False,
    do_tune: bool = False,
    regime_col: str | None = None,
) -> pd.DataFrame:
    """Rollierende Out-of-Sample-Evaluation über mehrere Origins.

    FIX 1: Ersetzt die Einzelpunkt-Evaluation durch mehrere Startpunkte,
    um modellspezifische Ergebnisse von perioden-spezifischen Zufällen zu trennen.

    Args:
        model_name:      Modellname
        covariates:      Kovariatenliste
        covariate_label: Bezeichner
        simulations:     Ensemble-Läufe pro Origin
        origins:         Explizite Liste von training_end-Daten (oder None für Auto)
        freq:            Frequenz der Auto-Origins
        use_lags:        Punkt 1 – verzögerte Kovariaten als Zusatzfeatures
        do_tune:         Punkt 3 – einheitliches Tuning-Budget für alle Sequenzmodelle
        regime_col:      Punkt 5 – falls gesetzt (z.B. "€STR"), wird pro Origin
                         das Zinsregime annotiert (siehe assign_interest_regime)

    Returns:
        DataFrame mit einer Zeile pro Origin und allen Metriken (inkl. rohen
        Fehler-Arrays in den Spalten 'diff_errors'/'volume_errors' für
        spätere Signifikanztests, siehe diebold_mariano_test).
    """
    if origins is None:
        min_rows = None
        if model_name == "tft":
            # gleiche Logik wie in predict_tft_forecast für minimales Trainingsvolumen
            min_rows = GEDAECHTNIS + PROGNOSEHORIZONT + int((GEDAECHTNIS + PROGNOSEHORIZONT) * TFT_VAL_FRACTION) + 10
        origins = generate_rolling_origins(freq=freq, min_train_rows=min_rows)

    # Falls tuning gewünscht ist: führe es EINMALIG vor der Origins-Schleife auf der
    # größten Trainingsmenge (TRAINING_ENDE) aus und reiche das Ergebnis weiter.
    precomputed_tuned: object | None = None
    if do_tune and model_name in _SEQUENCE_MODELS:
        try:
            # _prepare_sequence_data liefert X_train/y_train für TRAINING_ENDE
            _, _, X_full, y_full, _, _, _ = _prepare_sequence_data(covariates, training_end=TRAINING_ENDE, use_lags=use_lags)
            print(f"[TUNING] Führe einmaliges Tuning für {model_name} (Rolling) durch...")
            precomputed_tuned = tune_sequence_model(model_name, X_full, y_full, seed=SEED)
            print(f"[TUNING] Fertig: Verwende getunte Parameter für {model_name} in Rolling.")
        except Exception as exc:
            print(f"[TUNING] Einmaliges Tuning für {model_name} in Rolling fehlgeschlagen: {exc}. Fahre ohne Tuning fort.")

    rows = []
    for origin in origins:
        try:
            df_o, df_clean_o, _, _ = load_prepared_df(
                covariates, training_end=origin, use_lags=use_lags
            )
            input_start_o = _next_date_in_index(df_clean_o, origin)
            true_diff_o, true_volume_o = _true_test_arrays(df_o, df_clean_o, input_start_o)

            r = run_single_evaluation(
                model_name=model_name,
                covariates=covariates,
                covariate_label=covariate_label,
                simulations=simulations,
                training_end=origin,
                use_lags=use_lags,
                do_tune=False,  # tuning bereits vorab erledigt (falls gewünscht)
                tuned_estimator=precomputed_tuned,
            )
            row = {
                "Modell":        r.model,
                "Label":         r.covariate_label,
                "training_end":  r.training_end,
                "MAE_Diff":      r.mae_diff,
                "RMSE_Diff":     r.rmse_diff,
                "R2_Diff":       r.r2_diff,
                "MAE_Volumen":   r.mae_volume,
                "RMSE_Volumen":  r.rmse_volume,
                "R2_Volumen":    r.r2_volume,
                # Für Punkt 4 (Diebold-Mariano) und Punkt 2 (CI in Top-5):
                # rohe Vorhersage-, Fehler- und Wahrwerte-Arrays mitführen.
                "diff_pred":     r.diff_pred,
                "volume_pred":   r.volume_pred,
                "true_diff":     true_diff_o,
                "true_volume":   true_volume_o,
            }
            if regime_lookup is not None:
                row["Regime"] = regime_lookup.get(origin, "unbekannt")
            rows.append(row)
        except Exception as exc:
            print(f"[Fehler] {model_name} / {origin}: {exc}")

    if not rows:
        raise ValueError("Keine Rolling-Ergebnisse – Datenverfügbarkeit prüfen.")

    df = pd.DataFrame(rows)
    df["training_end"] = pd.to_datetime(df["training_end"])
    return df.sort_values(["Modell", "training_end"]).reset_index(drop=True)


def aggregate_rolling_results(rolling_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregiert Rolling-Ergebnisse zu Mittelwert ± Std pro Modell × Label
    (bzw. zusätzlich pro Zinsregime, falls eine 'Regime'-Spalte vorhanden ist,
    siehe Punkt 5 / assign_interest_regime).
    """
    metric_cols = ["MAE_Diff", "RMSE_Diff", "R2_Diff", "MAE_Volumen", "RMSE_Volumen", "R2_Volumen"]
    group_cols = ["Modell", "Label"]
    if "Regime" in rolling_df.columns:
        group_cols = group_cols + ["Regime"]
    return (
        rolling_df
        .groupby(group_cols)[metric_cols]
        .agg(["mean", "std"])
        .round(4)
    )


# ---------------------------------------------------------------------------
# Punkt 4: Diebold-Mariano-Test für Modellvergleiche
# ---------------------------------------------------------------------------

def diebold_mariano_test(
    errors_1: np.ndarray,
    errors_2: np.ndarray,
    h: int = 1,
    loss: str = "mse",
) -> dict[str, float]:
    """Diebold-Mariano-Test auf gleiche Prognosegüte zweier Modelle.

    Begründung (Kritik Punkt 5.2): Ranking-Unterschiede zwischen Modellen
    über Rolling-Origins werden bisher nur deskriptiv (Mean ± Std) berichtet.
    Ohne Signifikanztest lässt sich nicht sagen, ob ein Modell "wirklich"
    besser ist oder die Differenz im Rauschen verschwindet. Der DM-Test prüft
    H0: E[loss(e1) - loss(e2)] = 0, mit einer Newey-West-artigen
    Varianzkorrektur für die Autokorrelation, die bei h-Schritt-Prognosen
    durch überlappende Horizonte entsteht (Diebold & Mariano, 1995).

    Args:
        errors_1, errors_2: Rohe Prognosefehler (true - pred) der beiden
            Modelle, paarig zur selben Testperiode/denselben Origins.
            Erwartet als 1D-Arrays gleicher Länge (z.B. ein Fehlerwert pro
            Rolling-Origin, üblicherweise der mittlere Fehler über den
            Prognosehorizont je Origin – siehe `rolling_loss_series`).
        h: Prognosehorizont in Schritten (für die Newey-West-Bandbreite,
           Lag = h - 1). Bei aggregierten Pro-Origin-Fehlern i.d.R. h=1.
        loss: "mse" (quadratischer Fehler) oder "mae" (absoluter Fehler).

    Returns:
        dict mit 'dm_stat', 'p_value', 'mean_loss_diff' (loss1 - loss2;
        negativ = Modell 1 ist im Mittel besser).
    """
    e1 = np.asarray(errors_1, dtype=float)
    e2 = np.asarray(errors_2, dtype=float)
    if e1.shape != e2.shape:
        raise ValueError(f"errors_1 {e1.shape} und errors_2 {e2.shape} müssen gleiche Form haben.")
    n = len(e1)
    if n < 2:
        raise ValueError("Diebold-Mariano-Test benötigt mindestens 2 Beobachtungen.")

    if loss == "mse":
        l1, l2 = e1 ** 2, e2 ** 2
    elif loss == "mae":
        l1, l2 = np.abs(e1), np.abs(e2)
    else:
        raise ValueError("loss muss 'mse' oder 'mae' sein.")

    d = l1 - l2
    d_mean = float(np.mean(d))

    # Newey-West-Schätzer der Long-Run-Varianz mit Bandbreite (h - 1).
    max_lag = max(0, h - 1)
    gamma_0 = float(np.var(d, ddof=0))
    var_d = gamma_0
    for lag in range(1, max_lag + 1):
        if lag >= n:
            break
        cov = float(np.mean((d[lag:] - d_mean) * (d[:-lag] - d_mean)))
        var_d += 2 * (1 - lag / (max_lag + 1)) * cov

    if var_d <= 0:
        # Entartete (quasi-konstante) Differenzreihe: kein sinnvoller Test möglich.
        return {"dm_stat": float("nan"), "p_value": float("nan"), "mean_loss_diff": d_mean}

    dm_stat = d_mean / np.sqrt(var_d / n)

    # Kleine-Stichproben-Korrektur nach Harvey, Leybourne & Newbold (1997).
    correction = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_stat_corrected = dm_stat * correction

    # p-Wert über t-Verteilung mit n-1 Freiheitsgraden (HLN-Empfehlung).
    from scipy import stats as _stats
    p_value = float(2 * (1 - _stats.t.cdf(np.abs(dm_stat_corrected), df=n - 1)))

    return {
        "dm_stat": float(dm_stat_corrected),
        "p_value": p_value,
        "mean_loss_diff": d_mean,
    }


def rolling_loss_series(
    rolling_df: pd.DataFrame,
    target: str = "diff",
) -> pd.Series:
    """Extrahiert eine Pro-Origin-Fehlerreihe aus run_rolling_evaluation()-Output.

    Für jeden Origin wird der mittlere Prognosefehler (true - pred) über den
    gesamten Prognosehorizont gebildet. Diese verdichtete Reihe (eine Zahl
    pro Origin, zeitlich geordnet) ist die übliche Eingabe für den
    Diebold-Mariano-Test bei Rolling-Origin-Evaluationen.

    Args:
        rolling_df: Output von run_rolling_evaluation (benötigt Spalten
            'true_diff'/'diff_pred' bzw. 'true_volume'/'volume_pred' sowie
            'training_end').
        target: "diff" oder "volume".

    Returns:
        pd.Series, indiziert nach training_end (sortiert), mit einem
        Fehlerwert pro Origin.
    """
    true_col, pred_col = (
        ("true_diff", "diff_pred") if target == "diff" else ("true_volume", "volume_pred")
    )
    if true_col not in rolling_df.columns or pred_col not in rolling_df.columns:
        raise ValueError(
            f"rolling_df fehlen die Spalten {true_col!r}/{pred_col!r}. "
            "Wurde run_rolling_evaluation() mit den aktuellen Code-Stand erzeugt?"
        )
    sorted_df = rolling_df.sort_values("training_end")
    errors = [
        float(np.mean(np.asarray(t) - np.asarray(p)))
        for t, p in zip(sorted_df[true_col], sorted_df[pred_col])
    ]
    return pd.Series(errors, index=sorted_df["training_end"].values)


def compare_models_dm(
    rolling_df_1: pd.DataFrame,
    rolling_df_2: pd.DataFrame,
    target: str = "diff",
    loss: str = "mse",
) -> dict[str, float]:
    """Vergleicht zwei Modelle (je ein run_rolling_evaluation()-Output) via
    Diebold-Mariano-Test auf denselben Origins.

    Praktischer Wrapper um diebold_mariano_test + rolling_loss_series: richtet
    die beiden Fehlerreihen über gemeinsame training_end-Werte aus, bevor
    getestet wird (falls die Origin-Listen leicht voneinander abweichen).
    """
    s1 = rolling_loss_series(rolling_df_1, target=target)
    s2 = rolling_loss_series(rolling_df_2, target=target)
    common_idx = s1.index.intersection(s2.index)
    if len(common_idx) < 2:
        raise ValueError("Zu wenig gemeinsame Origins für einen DM-Test.")
    return diebold_mariano_test(s1.loc[common_idx].values, s2.loc[common_idx].values, h=1, loss=loss)


# ---------------------------------------------------------------------------
# Ergebnis-Formatierung
# ---------------------------------------------------------------------------

def results_to_dataframe(results: Iterable[ForecastResult]) -> pd.DataFrame:
    """Konvertiert ForecastResults in einen sortierten DataFrame."""
    rows = []
    for r in results:
        rows.append({
            "Modell":        r.model,
            "Kovariaten":    ", ".join(r.covariates),
            "Label":         r.covariate_label,
            "training_end":  r.training_end,
            "n_Kovariaten":  len(r.covariates),
            "MAE_Diff":      r.mae_diff,
            "RMSE_Diff":     r.rmse_diff,
            "R2_Diff":       r.r2_diff,
            "MAE_Volumen":   r.mae_volume,
            "RMSE_Volumen":  r.rmse_volume,
            "R2_Volumen":    r.r2_volume,
        })
    return pd.DataFrame(rows).sort_values(["Modell", "MAE_Volumen"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Plot-Unterstützung
# ---------------------------------------------------------------------------

def run_ensemble_volume_paths(
    model_name: str,
    covariates: list[str] | None = None,
    simulations: int = SIMULATIONEN,
    training_end: str = TRAINING_ENDE,
    do_tune: bool = False,
    use_lags: bool = False,
    use_calendar_known_reals: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Für Plot-Skripte: Mittelwert + KI-Bänder + Roh-DataFrame.

    Args:
        use_lags: Punkt 1 – verzögerte Kovariaten als Zusatzfeatures (nur
            für Sequenzmodelle wirksam; naive/TFT ignorieren dies).
        use_calendar_known_reals: Punkt 6 – nur für TFT relevant, siehe
            fit_tft().

    Returns:
        (mean_forecast, ki90_bands, ki98_bands, all_paths, df)

    Die KI-Bänder werden konsistent aus den Simulationen (Empirische Quantile) berechnet.
    Für naive Modelle erzeugen wir Pfade über Residual-Bootstrap, damit die CI vergleichbar sind.
    """
    covariates = covariates or ALL_COVARIATES
    df, _, train_df, _ = load_prepared_df(covariates, training_end=training_end, use_lags=use_lags)

    if model_name in _NAIVE_MODELS:
        # Residual-Resampling aus Trainings-Differenzen:
        diffs = train_df[TARGET_DIFF].dropna().values
        # BUGFIX 2: training_end snappen, falls es auf ein Wochenende/Feiertag fällt.
        anchor_level = float(_loc_value_snapped(df, training_end, TARGET_COL))
        # one-step naive RW: letzte train diff (same as predict_naive_rw basis) or zero-drift for hold
        if model_name == "naive_rw":
            base_diff = float(diffs[-1])
            # Erzeuge Simulationen durch Bootstrapping von Residualen (one-step residuals)
            paths = []
            rng = np.random.default_rng(SEED)
            for i in range(simulations):
                resampled = rng.choice(diffs - np.mean(diffs), size=PROGNOSEHORIZONT, replace=True)
                sim_diff = np.full(PROGNOSEHORIZONT, base_diff) + resampled
                paths.append(reconstruct_volume(sim_diff, anchor_level))
            paths = np.array(paths)
        else:  # naive_hold
            base_level = anchor_level
            paths = []
            rng = np.random.default_rng(SEED)
            for i in range(simulations):
                # small fluctuations from residuals around zero
                resampled = rng.choice(diffs - np.mean(diffs), size=PROGNOSEHORIZONT, replace=True)
                sim = base_level + np.cumsum(resampled)  # simulate small walk around hold
                paths.append(sim)
            paths = np.array(paths)

    elif model_name == "tft":
        # BUGFIX 2: training_end snappen, falls es auf ein Wochenende/Feiertag fällt.
        last_level = float(_loc_value_snapped(df, training_end, TARGET_COL))
        paths_list = []
        for i in range(simulations):
            m, ds = fit_tft(
                train_df, covariates, seed=SEED + i, use_calendar_known_reals=use_calendar_known_reals
            )
            dp = predict_tft_multi_step(m, ds, train_df, covariates)
            paths_list.append(reconstruct_volume(dp, last_level))
        paths = np.array(paths_list)

    else:
        # Sequenzmodelle mit optionalem Tuning
        _, _, X_train, y_train, x_input, scaler_y, last_level = _prepare_sequence_data(
            covariates, training_end=training_end, use_lags=use_lags
        )
        paths_list = []
        for i in range(simulations):
            # BUGFIX 3: use_boot gilt für alle Sequenzmodelle, nicht nur linreg.
            use_boot = simulations > 1
            m = fit_sequence_model(model_name, X_train, y_train, seed=SEED + i, bootstrap=use_boot, do_tune=do_tune)
            raw = m.predict(x_input).reshape(PROGNOSEHORIZONT, 1)
            dp = scaler_y.inverse_transform(raw).flatten()
            paths_list.append(reconstruct_volume(dp, last_level))
        paths = np.array(paths_list)

    mean = paths.mean(axis=0)
    ki90 = np.percentile(paths, [5, 95], axis=0)
    ki98 = np.percentile(paths, [1, 99], axis=0)
    return mean, ki90, ki98, paths, df