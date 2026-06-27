"""
==============================================================================
DIRECT MULTI-STEP XGBOOST FORECAST
Past LOOKBACK days -> Next HORIZON days
Target: Einlagevolumen
==============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =============================================================================
# 0. CONFIGURATION
# =============================================================================
FILE_PATH = "Data all variables.xlsx"

VARIABLES = [
    "Einlagevolumen",
    "€STR",
    "Einlagezinssatz",
    "GPRC_DEU",
    "MoM Inflation",
    "DAX",
    "10Y Bond",
]

TARGET = "Einlagevolumen"

LOOKBACK = 150
HORIZON = 100
TRAINING_END = "2023-11-30"

N_MODELS = 3
SEED = 42


# =============================================================================
# 1. LOAD DATA
# =============================================================================
df = pd.read_excel(FILE_PATH)
df["Datum"] = pd.to_datetime(df["Datum"])
df = df.sort_values("Datum").set_index("Datum")

df = df[VARIABLES].dropna()

df["Diff_Volume"] = df[TARGET].diff()
df = df.dropna()


# =============================================================================
# 2. SCALE DATA
# =============================================================================
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(df[VARIABLES])
y_scaled = scaler_y.fit_transform(df[["Diff_Volume"]])


# =============================================================================
# 3. CREATE SEQUENCES
# =============================================================================
def create_sequences(X, y, lookback, horizon):
    X_seq, y_seq = [], []

    for i in range(len(X) - lookback - horizon + 1):
        X_seq.append(X[i : i + lookback].flatten())
        y_seq.append(y[i + lookback : i + lookback + horizon].flatten())

    return np.array(X_seq), np.array(y_seq)


X_seq, y_seq = create_sequences(X_scaled, y_scaled, LOOKBACK, HORIZON)


# =============================================================================
# 4. TRAIN / TEST SPLIT
# =============================================================================
training_end_loc = df.index.get_indexer(
    [pd.to_datetime(TRAINING_END)],
    method="nearest"
)[0]

split_idx = training_end_loc - LOOKBACK

X_train = X_seq[:split_idx]
y_train = y_seq[:split_idx]

print(f"Training rows: {len(X_train)}")
print(f"Input shape:   {X_train.shape}")
print(f"Output shape:  {y_train.shape}")


# =============================================================================
# 5. TRAIN XGBOOST ENSEMBLE
# =============================================================================
ensemble_forecasts = []

for i in range(N_MODELS):
    print(f"Training model {i + 1}/{N_MODELS}...")

    base_model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.02,
        max_depth=3,
        min_child_weight=3,
        subsample=0.80,
        colsample_bytree=0.80,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=SEED + i,
        n_jobs=-1,
    )

    model = MultiOutputRegressor(base_model)
    model.fit(X_train, y_train)

    print(f"Finished model {i + 1}/{N_MODELS}")

    x_input = X_scaled[
        training_end_loc - LOOKBACK : training_end_loc
    ].flatten().reshape(1, -1)

    pred_scaled_diff = model.predict(x_input).reshape(HORIZON, 1)
    pred_diff = scaler_y.inverse_transform(pred_scaled_diff).flatten()

    last_actual_level = df[TARGET].iloc[training_end_loc - 1]
    pred_level = last_actual_level + np.cumsum(pred_diff)

    ensemble_forecasts.append(pred_level)


ensemble_forecasts = np.array(ensemble_forecasts)

forecast_mean = ensemble_forecasts.mean(axis=0)
forecast_p05 = np.percentile(ensemble_forecasts, 5, axis=0)
forecast_p95 = np.percentile(ensemble_forecasts, 95, axis=0)


# =============================================================================
# 6. ACTUAL VALUES AND METRICS
# =============================================================================
forecast_dates = df.index[training_end_loc : training_end_loc + HORIZON]
actual_values = df[TARGET].iloc[
    training_end_loc : training_end_loc + HORIZON
].values

valid_len = min(len(actual_values), len(forecast_mean))

actual_values = actual_values[:valid_len]
forecast_mean = forecast_mean[:valid_len]
forecast_p05 = forecast_p05[:valid_len]
forecast_p95 = forecast_p95[:valid_len]
forecast_dates = forecast_dates[:valid_len]

mae = mean_absolute_error(actual_values, forecast_mean)
rmse = np.sqrt(mean_squared_error(actual_values, forecast_mean))
r2 = r2_score(actual_values, forecast_mean)

print("\n" + "=" * 60)
print("DIRECT MULTI-STEP XGBOOST RESULTS")
print("=" * 60)
print(f"Forecast horizon: {valid_len} observations")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.4f}")


# =============================================================================
# 7. SAVE RESULTS
# =============================================================================
results = pd.DataFrame({
    "Date": forecast_dates,
    "Actual": actual_values,
    "XGBoost_Forecast": forecast_mean,
    "Lower_90": forecast_p05,
    "Upper_90": forecast_p95,
})

results.to_csv("xgboost_multistep_results.csv", index=False)
print("Saved results to xgboost_multistep_results.csv")


# =============================================================================
# 8. PLOT
# =============================================================================
history_start = max(0, training_end_loc - 200)
history = df[TARGET].iloc[history_start:training_end_loc]

plt.figure(figsize=(14, 7))

plt.plot(
    history.index,
    history.values,
    label="Historical Data",
    color="blue",
)

plt.plot(
    forecast_dates,
    actual_values,
    label="Actual Test Values",
    color="black",
    linestyle="--",
)

plt.plot(
    forecast_dates,
    forecast_mean,
    label="XGBoost Direct Multi-Step Forecast",
    color="red",
    linewidth=2,
)

plt.fill_between(
    forecast_dates,
    forecast_p05,
    forecast_p95,
    color="red",
    alpha=0.15,
    label="90% Ensemble Interval",
)

plt.title(
    f"Direct Multi-Step XGBoost Forecast: "
    f"Past {LOOKBACK} Days → Next {HORIZON} Days"
)
plt.xlabel("Date")
plt.ylabel("Einlagevolumen")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("xgboost_multistep_forecast.png", dpi=150)
plt.show()