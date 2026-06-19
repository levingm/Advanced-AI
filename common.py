"""Gemeinsame Pipeline für Treesearch-Prognosen und Kovariaten-Vergleich.

Szenario 2A: Realistische Prognose ohne Zukunftswissen der Kovariaten.
Alle Modelle erhalten nur historische Fenster; keine echten zukünftigen Kovariatenwerte.
"""

from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass
from typing import Callable, Iterable

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks import EarlyStopping
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore", category=UserWarning)

# --- Standard-Konfiguration ---
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
INPUT_ANFANG = "2023-12-01"
HISTORIE_ANFANG = "2023-06-01"
HISTORIE_ENDE = "2023-11-30"

GROUP_ID = "series"

TFT_MAX_EPOCHS = 40
TFT_BATCH_SIZE = 64
TFT_LEARNING_RATE = 1e-3
TFT_VAL_FRACTION = 0.2  # Letzten 20% der Trainingsperiode für Validierung


@dataclass
class ForecastResult:
    """Ergebnis einer einzelnen Modell × Kovariaten-Evaluation."""

    model: str
    covariates: list[str]
    covariate_label: str
    diff_pred: np.ndarray
    volume_pred: np.ndarray
    mae_diff: float
    rmse_diff: float
    r2_diff: float
    mae_volume: float
    rmse_volume: float
    r2_volume: float


def variablen_for(covariates: list[str]) -> list[str]:
    """Kombiniert Target + Kovariaten für Datenladung."""
    return [TARGET_COL] + list(covariates)


def generate_covariate_combinations(
    mode: str = "presets",
) -> list[tuple[list[str], str]]:
    """Liefert (Kovariatenliste, Label) für den Benchmark-Loop.

    Args:
        mode: "presets" (vordefinierte Sets), "pairs" (alle 2er-Kombinationen),
              "singletons" (einzelne Variablen), "all" (presets+pairs+singletons),
              "full" (alle möglichen Kombinationen).
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
            cols = [a, b]
            combos.append((cols, f"pair_{a}_{b}".replace(" ", "")))

    if mode in ("singletons", "all"):
        for c in ALL_COVARIATES:
            combos.append(([c], f"single_{c}".replace(" ", "")))

    if mode == "full":
        for r in range(1, len(ALL_COVARIATES) + 1):
            for combo in itertools.combinations(ALL_COVARIATES, r):
                cols = list(combo)
                label = "full_" + "_".join(c.replace(" ", "") for c in cols)
                combos.append((cols, label))

    # Duplikate entfernen (Reihenfolge der Kovariaten egal)
    seen: set[tuple[str, ...]] = set()
    unique: list[tuple[list[str], str]] = []
    for cols, label in combos:
        key = tuple(sorted(cols))
        if key not in seen:
            seen.add(key)
            unique.append((cols, label))
    return unique


def load_prepared_df(
    covariates: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Lädt und bereitet Daten vor.

    KORREKTUR 1.1: Typannotation korrigiert (4 statt 3 DataFrames).

    Returns:
        df: Vollständiger DF mit NaNs (für echte Test-Werte)
        df_clean: DF ohne NaNs (für Training/Sequenzen)
        train_df: Trainingsdaten bis TRAINING_ENDE
        future_df: Zukünftige Daten für Forecast (nur als Referenz, nicht für Input)
    """
    variablen = variablen_for(covariates)
    df = pd.read_excel(DATEN)
    df["Datum"] = pd.to_datetime(df["Datum"])
    df = df.sort_values("Datum").set_index("Datum")
    df = df[variablen].copy()
    df[TARGET_DIFF] = df[TARGET_COL].diff()
    df_clean = df.dropna()

    train_df = df_clean.loc[:TRAINING_ENDE].copy()
    future_df = df_clean.loc[INPUT_ANFANG:].iloc[:PROGNOSEHORIZONT].copy()
    return df, df_clean, train_df, future_df


def create_sequences(
    X_arr: np.ndarray, y_arr: np.ndarray, gd: int, hz: int
) -> tuple[np.ndarray, np.ndarray]:
    """Erstellt Multi-Step-Sequenzen aus Zeitreihendaten.

    Args:
        X_arr: Feature-Array (n_samples, n_features)
        y_arr: Target-Array (n_samples, n_targets)
        gd: Gedächtnis (Encoder-Länge)
        hz: Prognose-Horizont (Decoder-Länge)

    Returns:
        X_seq: Array von Sequenzen (n_sequences, gd * n_features)
        y_seq: Array von Targets (n_sequences, hz * n_targets)
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
    """Berechnet MAE, RMSE, R² auf Differenzen und Level."""
    return {
        "mae_diff": float(mean_absolute_error(true_diff, pred_diff)),
        "rmse_diff": float(np.sqrt(mean_squared_error(true_diff, pred_diff))),
        "r2_diff": float(r2_score(true_diff, pred_diff)),
        "mae_volume": float(mean_absolute_error(true_volume, pred_volume)),
        "rmse_volume": float(np.sqrt(mean_squared_error(true_volume, pred_volume))),
        "r2_volume": float(r2_score(true_volume, pred_volume)),
    }


def _true_test_arrays(
    df: pd.DataFrame, df_clean: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Extrahiert echte Test-Werte basierend auf Daten, nicht Positionen.

    KORREKTUR 1.2: Verwendet Datum-basierte Indexierung (loc + iloc),
    um sicherzustellen, dass df und df_clean korrekt ausgerichtet sind.
    """
    # Differenzen: aus df_clean (ohne erste NaN-Zeile), ab INPUT_ANFANG
    true_diff = (
        df_clean.loc[INPUT_ANFANG:]
        .iloc[:PROGNOSEHORIZONT][TARGET_DIFF]
        .to_numpy()
    )

    # Volumen (Level): aus df (mit eventuellen NaNs), ab INPUT_ANFANG
    true_volume = (
        df.loc[INPUT_ANFANG:].iloc[:PROGNOSEHORIZONT][TARGET_COL].to_numpy()
    )

    return true_diff, true_volume


def reconstruct_volume(diff_pred: np.ndarray, last_level: float) -> np.ndarray:
    """Rekonstruiert Volumen-Level aus Differenzen.

    Args:
        diff_pred: Prognostizierte Differenzen
        last_level: Letztes bekanntes Volumen (an TRAINING_ENDE)

    Returns:
        Rekonstruiertes Volumen für den Prognosehorizont
    """
    return last_level + np.cumsum(diff_pred)


def fit_sequence_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int = SEED,
) -> object:
    """Trainiert ein Sequence-Modell (Linreg, FFN, XGBoost).

    VERBESSERUNG 3.1: Extrahierter Code zur Vermeidung von Duplikation.

    Args:
        model_name: "linreg", "ffn" oder "xgboost"
        X_train: Trainingsdaten (n_samples, n_features)
        y_train: Trainings-Targets (n_samples, n_targets)
        seed: Random seed

    Returns:
        Trainiertes Modell-Objekt
    """
    if model_name == "linreg":
        # Bootstrapping für Unsicherheitsquantifizierung
        rng = np.random.default_rng(seed)
        boot_idx = rng.choice(len(X_train), size=len(X_train), replace=True)
        model = MultiOutputRegressor(LinearRegression())
        model.fit(X_train[boot_idx], y_train[boot_idx])
    elif model_name == "ffn":
        model = MLPRegressor(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=seed,
            learning_rate_init=1e-3,
        )
        model.fit(X_train, y_train)
    elif model_name == "xgboost":
        from xgboost import XGBRegressor

        model = MultiOutputRegressor(
            XGBRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=5,
                random_state=seed,
                subsample=0.8,
            )
        )
        model.fit(X_train, y_train)
    else:
        raise ValueError(f"Unbekanntes Sequenzmodell: {model_name}")

    return model


def _prepare_sequence_data(
    covariates: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, object, float]:
    """Bereitet Daten für Sequence-Modelle vor.

    KORREKTIONEN 1.3, 1.4, 1.5:
    - Skalierung nur auf Trainingsdaten (kein Leakage)
    - Korrekter Train/Test-Split ohne zukünftige Targets im Training
    - Off-by-one-Fehler bei x_input behoben
    - Szenario 2A: Nur historische Kovariaten, keine Zukunftswerte

    Returns:
        (df, df_clean, X_train, y_train, x_input, scaler_y, last_level)
    """
    variablen = variablen_for(covariates)
    df, df_clean, _, _ = load_prepared_df(covariates)

    # Split: Trainings- und Test-Indizes
    t_train = df_clean.index.get_loc(TRAINING_ENDE)  # Letzte Training-Position

    # Skalierung NUR auf Trainingsdaten (KORREKTUR 1.5)
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    train_data = df_clean.loc[:TRAINING_ENDE][variablen].values
    train_diff = df_clean.loc[:TRAINING_ENDE][[TARGET_DIFF]].values

    scaler_x.fit(train_data)
    scaler_y.fit(train_diff)

    # Gesamte Daten skalieren (mit Train-Parametern)
    X_data = scaler_x.transform(df_clean[variablen].values)
    y_data = scaler_y.transform(df_clean[[TARGET_DIFF]].values)

    # Sequenzen erstellen
    X_seq, y_seq = create_sequences(X_data, y_data, GEDAECHTNIS, PROGNOSEHORIZONT)

    # Data Leakage verhindern (KORREKTUR 1.3)
    # Nur Sequenzen nehmen, deren Targets vollständig im Trainingsbereich liegen
    max_train_start = t_train - GEDAECHTNIS - PROGNOSEHORIZONT + 1
    n_train_seqs = max(0, max_train_start + 1)
    X_train, y_train = X_seq[:n_train_seqs], y_seq[:n_train_seqs]

    # Input für Prognose: letztes vollständiges Fenster bis TRAINING_ENDE (KORREKTUR 1.4)
    x_input = X_data[t_train - GEDAECHTNIS + 1 : t_train + 1].flatten().reshape(1, -1)

    last_level = float(df.loc[TRAINING_ENDE, TARGET_COL])

    return df, df_clean, X_train, y_train, x_input, scaler_y, last_level


def predict_sequence_model(
    model_name: str,
    covariates: list[str],
    simulations: int = 1,
    seed: int = SEED,
) -> np.ndarray:
    """Prognostiziert mit Sequence-Modellen (Ensemble über Seeds).

    Args:
        model_name: "linreg", "ffn" oder "xgboost"
        covariates: Liste der zu nutzenden Kovariaten
        simulations: Anzahl der Ensemble-Durchläufe
        seed: Basis-Seed für Reproduzierbarkeit

    Returns:
        Array mit Prognose-Volumen (gemittelt über Simulations)
    """
    df, df_clean, X_train, y_train, x_input, scaler_y, last_level = _prepare_sequence_data(
        covariates
    )
    ensemble: list[np.ndarray] = []

    for i in range(simulations):
        model = fit_sequence_model(model_name, X_train, y_train, seed=seed + i)

        raw = model.predict(x_input).reshape(PROGNOSEHORIZONT, 1)
        diff_pred = scaler_y.inverse_transform(raw).flatten()
        ensemble.append(reconstruct_volume(diff_pred, last_level))

    return np.mean(ensemble, axis=0)


def _add_tft_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Präpariert DataFrame für TFT (fügt time_idx und group_id hinzu)."""
    out = frame.reset_index()
    if "Datum" not in out.columns:
        out = out.rename(columns={out.columns[0]: "Datum"})
    out[GROUP_ID] = "deposits"
    out["time_idx"] = np.arange(len(out))
    return out


def _to_numpy(pred) -> np.ndarray:
    """Konvertiert PyTorch/andere Outputs zu NumPy."""
    if hasattr(pred, "output"):
        pred = pred.output
    if isinstance(pred, torch.Tensor):
        return pred.detach().cpu().numpy()
    return np.asarray(pred)


def fit_tft(
    train_df: pd.DataFrame, known_reals: list[str], seed: int
) -> tuple[TemporalFusionTransformer, TimeSeriesDataSet]:
    """Trainiert TFT mit echtem Val-Split (KORREKTUR 1.6).

    KORREKTUR 1.6: Train/Val-Split auf Zeitachse (nicht per Loader-Flag).
    Letzte TFT_VAL_FRACTION (20%) der Trainingsdaten als Validierung.

    SZENARIO 2A: Kovariaten sind historisch; es gibt keine "zukünftigen" Kovariate
    für den Forecast-Horizont. TFT wird mit last_known_reals = nan trainiert.

    Args:
        train_df: Trainingsdaten (bis TRAINING_ENDE)
        known_reals: Liste von Kovariaten-Namen
        seed: Random seed

    Returns:
        (tft_model, training_dataset)
    """
    pl.seed_everything(seed, workers=True)
    torch.manual_seed(seed)

    # Train/Val-Split auf Zeitachse
    n_train = len(train_df)
    n_val = max(1, int(n_train * TFT_VAL_FRACTION))
    split_idx = n_train - n_val

    train_part = train_df.iloc[:split_idx]
    val_part = train_df.iloc[split_idx:]

    train_tft = _add_tft_columns(train_part)
    val_tft = _add_tft_columns(val_part)

    # Trainings-Dataset (mit allen Daten, aber Split wird im Loader definiert)
    training = TimeSeriesDataSet(
        train_tft,
        time_idx="time_idx",
        target=TARGET_DIFF,
        group_ids=[GROUP_ID],
        max_encoder_length=GEDAECHTNIS,
        min_encoder_length=max(1, GEDAECHTNIS // 2),
        max_prediction_length=PROGNOSEHORIZONT,
        min_prediction_length=PROGNOSEHORIZONT,
        time_varying_known_reals=known_reals,
        time_varying_unknown_reals=[TARGET_COL, TARGET_DIFF],
        target_normalizer=GroupNormalizer(groups=[GROUP_ID]),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    # Validierungs-Dataset (separater Zeitbereich)
    validation = TimeSeriesDataSet.from_dataset(
        training, val_tft, predict=False, stop_randomization=True
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
    # Für TFT brauchen wir ein DataFrame mit time_idx für die Zukunft
    # Aber ohne echte zukünftige Kovariate-Werte (Szenario 2A)
    # → Wir erstellen einen "dummy" future DataFrame
    history_prep = _add_tft_columns(history_df)
    max_time_idx = history_prep["time_idx"].max()

    # Future DataFrame: Struktur, aber nan Kovariaten
    future_rows = []
    for i in range(1, PROGNOSEHORIZONT + 1):
        row = {GROUP_ID: "deposits", "time_idx": max_time_idx + i}
        # Kovariaten setzen auf NaN (nicht verfügbar, Szenario 2A)
        for cov in training_dataset.time_varying_known_reals:
            row[cov] = np.nan
        # Target-Spalten setzen auf NaN (werden prognostiziert)
        row[TARGET_COL] = np.nan
        row[TARGET_DIFF] = np.nan
        row["Datum"] = pd.Timestamp("1900-01-01")  # Dummy
        future_rows.append(row)

    future_tft = pd.DataFrame(future_rows)
    combined = pd.concat([history_prep, future_tft], ignore_index=True)

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
    return flat[-PROGNOSEHORIZONT:]


def predict_tft_forecast(
    covariates: list[str],
    simulations: int = 1,
    seed: int = SEED,
) -> np.ndarray:
    """Prognostiziert mit TFT (Ensemble über Seeds).

    SZENARIO 2A: Nur historische Kovariaten, keine Zukunftswerte.

    Args:
        covariates: Liste der Kovariaten-Namen
        simulations: Anzahl der Ensemble-Durchläufe
        seed: Basis-Seed

    Returns:
        Gemittelte Prognose des Volumens
    """
    df, df_clean, train_df, _ = load_prepared_df(covariates)
    min_train = GEDAECHTNIS + PROGNOSEHORIZONT + 10
    if len(train_df) < min_train:
        raise ValueError(f"Zu wenig Trainingsdaten: {len(train_df)} < {min_train}")

    last_level = float(df.loc[TRAINING_ENDE, TARGET_COL])
    ensemble: list[np.ndarray] = []
    for i in range(simulations):
        model, ds = fit_tft(train_df, covariates, seed=seed + i)
        diff_pred = predict_tft_multi_step(model, ds, train_df)
        ensemble.append(reconstruct_volume(diff_pred, last_level))
    return np.mean(ensemble, axis=0)


MODEL_RUNNERS: dict[str, Callable[..., np.ndarray]] = {
    "xgboost": predict_sequence_model,
    "linreg": predict_sequence_model,
    "ffn": predict_sequence_model,
    "tft": predict_tft_forecast,
}


def run_single_evaluation(
    model_name: str,
    covariates: list[str],
    covariate_label: str,
    simulations: int = 1,
) -> ForecastResult:
    """Führt eine komplette Modell-Evaluation durch.

    Args:
        model_name: "linreg", "ffn", "xgboost" oder "tft"
        covariates: Liste der Kovariaten
        covariate_label: Label für Ergebnisse
        simulations: Anzahl Ensemble-Durchläufe

    Returns:
        ForecastResult mit Metriken und Prognosen
    """
    if not covariates:
        raise ValueError("Mindestens eine Kovariate erforderlich.")

    df, df_clean, _, _ = load_prepared_df(covariates)
    true_diff, true_volume = _true_test_arrays(df, df_clean)

    if model_name == "tft":
        volume_pred = predict_tft_forecast(covariates, simulations=simulations)
    else:
        volume_pred = predict_sequence_model(model_name, covariates, simulations=simulations)

    last_level = float(df.loc[TRAINING_ENDE, TARGET_COL])
    diff_pred = np.diff(np.concatenate([[last_level], volume_pred]))

    metrics = compute_metrics(true_diff, diff_pred, true_volume, volume_pred)
    return ForecastResult(
        model=model_name,
        covariates=covariates,
        covariate_label=covariate_label,
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


def run_ensemble_volume_paths(
    model_name: str,
    covariates: list[str] | None = None,
    simulations: int = SIMULATIONEN,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Für Plot-Skripte: Mittelwert + KI-Bänder + Roh-DataFrame.

    Returns:
        (mean_forecast, ki90_bands, ki98_bands, all_paths, dataframe)
    """
    covariates = covariates or ALL_COVARIATES
    df, df_clean, _, _ = load_prepared_df(covariates)

    if model_name == "tft":
        last_level = float(df.loc[TRAINING_ENDE, TARGET_COL])
        paths = []
        for i in range(simulations):
            _, _, train_df, _ = load_prepared_df(covariates)
            model, ds = fit_tft(train_df, covariates, seed=SEED + i)
            diff_pred = predict_tft_multi_step(model, ds, train_df)
            paths.append(reconstruct_volume(diff_pred, last_level))
    else:
        _, _, X_train, y_train, x_input, scaler_y, last_level = _prepare_sequence_data(
            covariates
        )
        paths = []
        for i in range(simulations):
            model = fit_sequence_model(model_name, X_train, y_train, seed=SEED + i)
            raw = model.predict(x_input).reshape(PROGNOSEHORIZONT, 1)
            diff_pred = scaler_y.inverse_transform(raw).flatten()
            paths.append(reconstruct_volume(diff_pred, last_level))

    paths = np.array(paths)
    mean = paths.mean(axis=0)
    ki90 = np.percentile(paths, [5, 95], axis=0)
    ki98 = np.percentile(paths, [1, 99], axis=0)
    return mean, ki90, ki98, paths, df
