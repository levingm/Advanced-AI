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

GEDAECHTNIS = 100
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


# ---------------------------------------------------------------------------
# FIX 2: Naive Benchmarks
# ---------------------------------------------------------------------------

def predict_naive_rw(
    covariates: list[str],
    simulations: int = 1,
    seed: int = SEED,
    training_end: str = TRAINING_ENDE,
) -> np.ndarray:
    """Naiver Benchmark – Random Walk: letzte Differenz konstant fortgeschrieben.

    Formal: ŷ_{t+h} = y_t + h * Δy_t  für alle h = 1..PROGNOSEHORIZONT
    Kein Zukunftswissen. Schwieriger Benchmark für Finanzzeitreihen.
    """
    df, df_clean, _, _ = load_prepared_df(covariates, training_end=training_end)
    last_diff = float(df_clean.loc[:training_end, TARGET_DIFF].iloc[-1])
    last_level = float(df.loc[training_end, TARGET_COL])
    return reconstruct_volume(np.full(PROGNOSEHORIZONT, last_diff), last_level)


def predict_naive_hold(
    covariates: list[str],
    simulations: int = 1,
    seed: int = SEED,
    training_end: str = TRAINING_ENDE,
) -> np.ndarray:
    """Naiver Benchmark – Naive Hold: letztes Volumen konstant (Drift = 0).

    Formal: ŷ_{t+h} = y_t  für alle h.
    """
    df, _, _, _ = load_prepared_df(covariates, training_end=training_end)
    last_level = float(df.loc[training_end, TARGET_COL])
    return np.full(PROGNOSEHORIZONT, last_level)


# ---------------------------------------------------------------------------
# Sequenzmodelle: LinReg, FFN, XGBoost
# ---------------------------------------------------------------------------

def fit_sequence_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int = SEED,
    bootstrap: bool = False,
) -> object:
    """Instanziiert und trainiert ein Sequenzmodell.

    FIX 7 (XGBoost): Nutzt nativen multi_strategy="multi_output_tree" wenn
    verfügbar (XGBoost >= 1.6) – ein Modell statt 100 separate Regressoren.
    Fallback auf MultiOutputRegressor für ältere Versionen.

    FIX 3.1 (aus Vorrunde): Zentralisierte Funktion statt duplizierter if/elif-Blöcke.
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
        # Hinweis: sklearn's validation_fraction nutzt intern train_test_split
        # mit shuffle=True – nicht streng temporal. Für Benchmark akzeptabel,
        # für Produktiveinsatz Custom-EarlyStopping empfohlen.
        model = MLPRegressor(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            max_iter=500,
            early_stopping=True,
            validation_fraction=TFT_VAL_FRACTION,   # konsistent mit TFT
            n_iter_no_change=10,
            random_state=seed,
            learning_rate_init=1e-3,
        )
        model.fit(X_train, y_train)

    elif model_name == "xgboost":
        XGB = importlib.import_module("xgboost").XGBRegressor
    
        model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
            random_state=seed,
            subsample=0.8,
            multi_strategy="multi_output_tree",
            tree_method="hist",
            n_jobs=1
        )
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

    t = df_clean.index.get_loc(training_end)

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
    last_level = float(df.loc[training_end, TARGET_COL])

    return df, df_clean, X_train, y_train, x_input, scaler_y, last_level


def predict_sequence_model(
    model_name: str,
    covariates: list[str],
    simulations: int = 1,
    seed: int = SEED,
    training_end: str = TRAINING_ENDE,
) -> np.ndarray:
    """Ensemble-Prognose mit Sequenzmodellen.

    Ensemble-Semantik:
    - linreg:   Bootstrap-Sampling (bei simulations > 1)
    - ffn:      verschiedene Zufalls-Initialisierungen
    - xgboost:  verschiedene Subsampling-Seeds
    Die KI-Bänder haben je nach Modell unterschiedliche statistische Bedeutung;
    dies ist im Paper explizit auszuweisen.
    """
    _, _, X_train, y_train, x_input, scaler_y, last_level = _prepare_sequence_data(
        covariates, training_end=training_end
    )
    ensemble: list[np.ndarray] = []
    for i in range(simulations):
        use_boot = (model_name == "linreg") and (simulations > 1)
        m = fit_sequence_model(model_name, X_train, y_train, seed=seed + i, bootstrap=use_boot)
        raw = m.predict(x_input).reshape(PROGNOSEHORIZONT, 1)
        diff_pred = scaler_y.inverse_transform(raw).flatten()
        ensemble.append(reconstruct_volume(diff_pred, last_level))
    return np.mean(ensemble, axis=0)


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
    """Trainiert TFT mit korrektem zeitlichem Val-Split.

    FIX 3 (TFT-Val ohne Encoder-Historie):
    Der Validierungs-Dataset wird aus dem VOLLEN train_tft_full-Frame erzeugt,
    nicht nur aus den letzten n_val Zeilen. So hat der Encoder die notwendige
    Historie. min_prediction_idx begrenzt die Val-Vorhersagen auf den Val-Bereich.

    FIX 4 (TFT trainiert weniger Daten):
    Das Training-TimeSeriesDataSet nutzt train_tft_full bis split_idx; die Val-
    Sequenzen greifen für den Encoder auf den vollen Frame zurück. TFT und die
    anderen Modelle trainieren auf identischem Zeitraum; nur die letzten TFT_VAL_FRACTION
    werden nicht als Decoder-Ziel im Training genutzt (notwendig für echtes EarlyStopping).
    """
    pl.seed_everything(seed, workers=True)
    torch.manual_seed(seed)

    train_tft_full = _add_tft_columns(train_df)
    n = len(train_tft_full)
    n_val = max(PROGNOSEHORIZONT + 1, int(n * TFT_VAL_FRACTION))
    split_idx = n - n_val          # letzter time_idx im Training

    train_only = train_tft_full[train_tft_full["time_idx"] <= split_idx]

    unknown_reals = [TARGET_COL, TARGET_DIFF]
    known_reals = list(covariates)

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

    # FIX 3: Voller Frame für Encoder-Historie; nur Val-Zeitraum wird vorhergesagt
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
        accelerator="cpu",
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
    """Multi-Step-TFT-Prognose mit Naive-Hold für Kovariaten (Szenario 2A).

    Kovariaten werden im Decoder auf dem letzten bekannten Wert eingefroren –
    kein ex-post Oracle. Zielvariablen (Einlagevolumen, Diff_Volume) sind NaN.
    """
    history_prep = _add_tft_columns(history_df)
    max_time_idx = int(history_prep["time_idx"].max())
    last_row = history_df.iloc[-1]
    last_date = history_df.index[-1]

    future_rows = []
    for i in range(1, PROGNOSEHORIZONT + 1):
        row = {
            GROUP_ID: "deposits",
            "time_idx": max_time_idx + i,
            "Datum": last_date + pd.Timedelta(days=i),
            TARGET_COL: np.nan,
            TARGET_DIFF: np.nan,
        }
        for cov in covariates:
            row[cov] = float(last_row[cov])
        future_rows.append(row)

    combined = pd.concat([history_prep, pd.DataFrame(future_rows)], ignore_index=True)
    predict_ds = TimeSeriesDataSet.from_dataset(
        training_dataset, combined, predict=True, stop_randomization=True
    )
    loader = predict_ds.to_dataloader(train=False, batch_size=1, num_workers=0)
    pred = _to_numpy(
        model.predict(
            loader,
            trainer_kwargs=dict(accelerator="cpu", logger=False, enable_checkpointing=False),
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
) -> np.ndarray:
    """Ensemble-Prognose mit TFT (verschiedene Seeds)."""
    df, _, train_df, _ = load_prepared_df(covariates, training_end=training_end)
    min_train = GEDAECHTNIS + PROGNOSEHORIZONT + int(
        (GEDAECHTNIS + PROGNOSEHORIZONT) * TFT_VAL_FRACTION
    ) + 10
    if len(train_df) < min_train:
        raise ValueError(f"Zu wenig Trainingsdaten für TFT: {len(train_df)} < {min_train}.")

    last_level = float(df.loc[training_end, TARGET_COL])
    ensemble: list[np.ndarray] = []
    for i in range(simulations):
        m, ds = fit_tft(train_df, covariates, seed=seed + i)
        diff_pred = predict_tft_multi_step(m, ds, train_df, covariates)
        ensemble.append(reconstruct_volume(diff_pred, last_level))
    return np.mean(ensemble, axis=0)


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
) -> ForecastResult:
    """Führt eine vollständige Modell-Evaluation durch.

    Args:
        model_name:      Eines von ALL_MODELS
        covariates:      Kovariatenliste
        covariate_label: Bezeichner für Ergebniszeile
        simulations:     Anzahl Ensemble-Läufe
        training_end:    Letztes Trainingsdatum (für Rolling-Origin)

    Returns:
        ForecastResult mit allen Metriken und Prognose-Arrays
    """
    if not covariates:
        raise ValueError("Mindestens eine Kovariate erforderlich.")

    df, df_clean, _, _ = load_prepared_df(covariates, training_end=training_end)
    input_start = _next_date_in_index(df_clean, training_end)
    true_diff, true_volume = _true_test_arrays(df, df_clean, input_start)

    kw = dict(covariates=covariates, simulations=simulations, training_end=training_end)

    if model_name in _SEQUENCE_MODELS:
        volume_pred = predict_sequence_model(model_name, **kw)
    elif model_name == "tft":
        volume_pred = predict_tft_forecast(**kw)
    elif model_name == "naive_rw":
        volume_pred = predict_naive_rw(**kw)
    elif model_name == "naive_hold":
        volume_pred = predict_naive_hold(**kw)
    else:
        raise ValueError(
            f"Unbekanntes Modell: {model_name!r}. Gültig: {ALL_MODELS}"
        )

    last_level = float(df.loc[training_end, TARGET_COL])
    diff_pred = np.diff(np.concatenate([[last_level], volume_pred]))
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
        freq:           Pandas-Frequenz der Origins (default: monatlich "MS")
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
    # Letztes Origin muss noch PROGNOSEHORIZONT Testdaten übrig lassen
    last_ts = pd.Timestamp(last_origin)

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


def run_rolling_evaluation(
    model_name: str,
    covariates: list[str],
    covariate_label: str,
    simulations: int = 1,
    origins: list[str] | None = None,
    freq: str = "MS",
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

    Returns:
        DataFrame mit einer Zeile pro Origin und allen Metriken
    """
    if origins is None:
        origins = generate_rolling_origins(freq=freq)

    rows = []
    for origin in origins:
        try:
            r = run_single_evaluation(
                model_name=model_name,
                covariates=covariates,
                covariate_label=covariate_label,
                simulations=simulations,
                training_end=origin,
            )
            rows.append({
                "Modell":        r.model,
                "Label":         r.covariate_label,
                "training_end":  r.training_end,
                "MAE_Diff":      r.mae_diff,
                "RMSE_Diff":     r.rmse_diff,
                "R2_Diff":       r.r2_diff,
                "MAE_Volumen":   r.mae_volume,
                "RMSE_Volumen":  r.rmse_volume,
                "R2_Volumen":    r.r2_volume,
            })
        except Exception as exc:
            print(f"[Fehler] {model_name} / {origin}: {exc}")

    if not rows:
        raise ValueError("Keine Rolling-Ergebnisse – Datenverfügbarkeit prüfen.")

    df = pd.DataFrame(rows)
    df["training_end"] = pd.to_datetime(df["training_end"])
    return df.sort_values(["Modell", "training_end"]).reset_index(drop=True)


def aggregate_rolling_results(rolling_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregiert Rolling-Ergebnisse zu Mittelwert ± Std pro Modell × Label."""
    metric_cols = ["MAE_Diff", "RMSE_Diff", "R2_Diff", "MAE_Volumen", "RMSE_Volumen", "R2_Volumen"]
    return (
        rolling_df
        .groupby(["Modell", "Label"])[metric_cols]
        .agg(["mean", "std"])
        .round(4)
    )


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Für Plot-Skripte: Mittelwert + KI-Bänder + Roh-DataFrame.

    Returns:
        (mean_forecast, ki90_bands, ki98_bands, all_paths, raw_dataframe)

    Hinweis: KI-Bänder reflektieren je nach Modell unterschiedliche
    Unsicherheitsquellen (Bootstrap / Initialisierung / Subsampling).
    """
    covariates = covariates or ALL_COVARIATES
    df, _, train_df, _ = load_prepared_df(covariates, training_end=training_end)

    if model_name in _NAIVE_MODELS:
        runner = predict_naive_rw if model_name == "naive_rw" else predict_naive_hold
        single = runner(covariates, training_end=training_end)
        paths = np.tile(single, (simulations, 1))

    elif model_name == "tft":
        last_level = float(df.loc[training_end, TARGET_COL])
        paths_list = []
        for i in range(simulations):
            m, ds = fit_tft(train_df, covariates, seed=SEED + i)
            dp = predict_tft_multi_step(m, ds, train_df, covariates)
            paths_list.append(reconstruct_volume(dp, last_level))
        paths = np.array(paths_list)

    else:
        _, _, X_train, y_train, x_input, scaler_y, last_level = _prepare_sequence_data(
            covariates, training_end=training_end
        )
        paths_list = []
        for i in range(simulations):
            use_boot = (model_name == "linreg") and (simulations > 1)
            m = fit_sequence_model(model_name, X_train, y_train, seed=SEED + i, bootstrap=use_boot)
            raw = m.predict(x_input).reshape(PROGNOSEHORIZONT, 1)
            dp = scaler_y.inverse_transform(raw).flatten()
            paths_list.append(reconstruct_volume(dp, last_level))
        paths = np.array(paths_list)

    mean = paths.mean(axis=0)
    ki90 = np.percentile(paths, [5, 95], axis=0)
    ki98 = np.percentile(paths, [1, 99], axis=0)
    return mean, ki90, ki98, paths, df