"""Gemeinsame Pipeline für Treesearch-Prognosen und Kovariaten-Vergleich."""

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


@dataclass
class ForecastResult:
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
    return [TARGET_COL] + list(covariates)


def generate_covariate_combinations(
    mode: str = "presets",
) -> list[tuple[list[str], str]]:
    """Liefert (Kovariatenliste, Label) für den Benchmark-Loop."""
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


def load_prepared_df(covariates: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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


def create_sequences(X_arr: np.ndarray, y_arr: np.ndarray, gd: int, hz: int):
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
    return {
        "mae_diff": float(mean_absolute_error(true_diff, pred_diff)),
        "rmse_diff": float(np.sqrt(mean_squared_error(true_diff, pred_diff))),
        "r2_diff": float(r2_score(true_diff, pred_diff)),
        "mae_volume": float(mean_absolute_error(true_volume, pred_volume)),
        "rmse_volume": float(np.sqrt(mean_squared_error(true_volume, pred_volume))),
        "r2_volume": float(r2_score(true_volume, pred_volume)),
    }


def _true_test_arrays(df: pd.DataFrame, df_clean: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    start = df.index.get_loc(INPUT_ANFANG)
    true_diff = df_clean.iloc[start : start + PROGNOSEHORIZONT][TARGET_DIFF].values
    true_volume = df.iloc[start : start + PROGNOSEHORIZONT][TARGET_COL].values
    return true_diff, true_volume


def reconstruct_volume(diff_pred: np.ndarray, last_level: float) -> np.ndarray:
    return last_level + np.cumsum(diff_pred)


def _prepare_sequence_data(
    covariates: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    variablen = variablen_for(covariates)
    df, df_clean, _, _ = load_prepared_df(covariates)

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_data = scaler_x.fit_transform(df_clean[variablen].values)
    y_data = scaler_y.fit_transform(df_clean[[TARGET_DIFF]].values)

    X_seq, y_seq = create_sequences(X_data, y_data, GEDAECHTNIS, PROGNOSEHORIZONT)
    split_idx = df_clean.index.get_loc(TRAINING_ENDE) - GEDAECHTNIS
    X_train, y_train = X_seq[:split_idx], y_seq[:split_idx]

    input_idx = df_clean.index.get_loc(TRAINING_ENDE)
    x_input = X_data[input_idx - GEDAECHTNIS : input_idx].flatten().reshape(1, -1)
    last_level = float(df.loc[TRAINING_ENDE, TARGET_COL])

    return df, df_clean, X_train, y_train, x_input, scaler_y, last_level


def predict_sequence_model(
    model_name: str,
    covariates: list[str],
    simulations: int = 1,
    seed: int = SEED,
) -> np.ndarray:
    df, df_clean, X_train, y_train, x_input, scaler_y, last_level = _prepare_sequence_data(
        covariates
    )
    ensemble: list[np.ndarray] = []

    for i in range(simulations):
        if model_name == "linreg":
            rng = np.random.default_rng(seed + i)
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
                random_state=seed + i,
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
                    random_state=seed + i,
                    subsample=0.8,
                )
            )
            model.fit(X_train, y_train)
        else:
            raise ValueError(f"Unbekanntes Sequenzmodell: {model_name}")

        raw = model.predict(x_input).reshape(PROGNOSEHORIZONT, 1)
        diff_pred = scaler_y.inverse_transform(raw).flatten()
        ensemble.append(reconstruct_volume(diff_pred, last_level))

    return np.mean(ensemble, axis=0)


def _add_tft_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.reset_index()
    if "Datum" not in out.columns:
        out = out.rename(columns={out.columns[0]: "Datum"})
    out[GROUP_ID] = "deposits"
    out["time_idx"] = np.arange(len(out))
    return out


def _to_numpy(pred):
    if hasattr(pred, "output"):
        pred = pred.output
    if isinstance(pred, torch.Tensor):
        return pred.detach().cpu().numpy()
    return np.asarray(pred)


def fit_tft(train_df: pd.DataFrame, known_reals: list[str], seed: int):
    pl.seed_everything(seed, workers=True)
    torch.manual_seed(seed)

    train_tft = _add_tft_columns(train_df)
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

    train_loader = training.to_dataloader(train=True, batch_size=TFT_BATCH_SIZE, num_workers=0)
    val_loader = training.to_dataloader(train=False, batch_size=TFT_BATCH_SIZE, num_workers=0)

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
    model, training_dataset, history_df: pd.DataFrame, future_df: pd.DataFrame
) -> np.ndarray:
    combined = pd.concat([history_df, future_df], axis=0)
    combined = combined[~combined.index.duplicated(keep="last")]
    combined = _add_tft_columns(combined)

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
    df, df_clean, train_df, future_df = load_prepared_df(covariates)
    min_train = GEDAECHTNIS + PROGNOSEHORIZONT + 10
    if len(train_df) < min_train:
        raise ValueError(f"Zu wenig Trainingsdaten: {len(train_df)} < {min_train}")

    future_df = future_df.copy()
    future_df[TARGET_COL] = train_df[TARGET_COL].iloc[-1]
    future_df[TARGET_DIFF] = train_df[TARGET_DIFF].iloc[-1]

    last_level = float(df.loc[TRAINING_ENDE, TARGET_COL])
    ensemble: list[np.ndarray] = []
    for i in range(simulations):
        model, ds = fit_tft(train_df, covariates, seed=seed + i)
        diff_pred = predict_tft_multi_step(model, ds, train_df, future_df)
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
    """Für Plot-Skripte: Mittelwert + KI-Bänder + Roh-DataFrame."""
    covariates = covariates or ALL_COVARIATES
    variablen = variablen_for(covariates)
    df, df_clean, _, _ = load_prepared_df(covariates)

    if model_name == "tft":
        df_raw, train_df, future_df = df, df_clean.loc[:TRAINING_ENDE].copy(), None
        future_df = df_clean.loc[INPUT_ANFANG:].iloc[:PROGNOSEHORIZONT].copy()
        future_df[TARGET_COL] = train_df[TARGET_COL].iloc[-1]
        future_df[TARGET_DIFF] = train_df[TARGET_DIFF].iloc[-1]
        last_level = float(df.loc[TRAINING_ENDE, TARGET_COL])
        paths = []
        for i in range(simulations):
            model, ds = fit_tft(train_df, covariates, seed=SEED + i)
            diff_pred = predict_tft_multi_step(model, ds, train_df, future_df)
            paths.append(reconstruct_volume(diff_pred, last_level))
    else:
        _, _, X_train, y_train, x_input, scaler_y, last_level = _prepare_sequence_data(
            covariates
        )
        paths = []
        for i in range(simulations):
            if model_name == "linreg":
                rng = np.random.default_rng(SEED + i)
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
                    random_state=SEED + i,
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
                        random_state=SEED + i,
                        subsample=0.8,
                    )
                )
                model.fit(X_train, y_train)
            else:
                raise ValueError(model_name)

            raw = model.predict(x_input).reshape(PROGNOSEHORIZONT, 1)
            diff_pred = scaler_y.inverse_transform(raw).flatten()
            paths.append(reconstruct_volume(diff_pred, last_level))

    paths = np.array(paths)
    mean = paths.mean(axis=0)
    ki90 = np.percentile(paths, [5, 95], axis=0)
    ki98 = np.percentile(paths, [1, 99], axis=0)
    return mean, ki90, ki98, paths, df
