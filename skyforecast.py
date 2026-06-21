import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

# --- Imports for v0.22.0 ---
from skforecast.recursive import ForecasterRecursive
from skforecast.model_selection import backtesting_forecaster, TimeSeriesFold

import warnings
warnings.filterwarnings('ignore')

# ==========================================
# --- 1. CONFIGURATION ---
# ==========================================
Daten = "Data all variables.xlsx"
Target = "Einlagevolumen"
Exog_Vars = ["€STR", "Einlagezinssatz", "GPRC_DEU", "MoM Inflation", "DAX", "10Y Bond"]

Gedächtnis = 30          
Seed = 42

TrainingEnde = "2023-11-30" 
HistorieAnfang = "2023-06-01"

# ==========================================
# --- 2. DATA PREPARATION ---
# ==========================================
print("Loading data and setting frequency...")
df = pd.read_excel(Daten)
df["Datum"] = pd.to_datetime(df["Datum"])
df = df.sort_values("Datum").set_index("Datum")

# --- FIX 1: Set Frequency ---
# This tells pandas and skforecast that the data is daily.
# .asfreq('D') will ensure the index has freq='D'. 
# .ffill() handles any tiny gaps (like leap years or missing data points).
df = df.asfreq('D')
df = df.ffill() 

# Real-world lag: Use yesterday's market to predict today's volume
df_exog = df[Exog_Vars].shift(1)
df_target = df[Target]
data = pd.concat([df_target, df_exog], axis=1).dropna()

# ==========================================
# --- 3. THE FORECASTER ---
# ==========================================
forecaster = ForecasterRecursive(
    estimator = XGBRegressor(
        random_state=Seed,
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8
    ),
    lags = Gedächtnis,
    differentiation = 1, # Mathematical stationarization
    transformer_y = MinMaxScaler(),
    transformer_exog = MinMaxScaler()
)

# ==========================================
# --- 4. THE CV STRATEGY (TimeSeriesFold) ---
# ==========================================
print("Configuring TimeSeriesFold with differentiation sync...")

# Calculate the count of samples for training
init_train_size = len(data.loc[:TrainingEnde])

cv = TimeSeriesFold(
    initial_train_size = init_train_size,
    steps              = 1,      
    refit              = True,   
    fixed_train_size   = False,
    # --- FIX 2: Synchronize Differentiation ---
    # This must match the differentiation in the forecaster (1)
    differentiation    = 1       
)

# ==========================================
# --- 5. BACKTESTING EXECUTION ---
# ==========================================
print("Running Dynamic Walk-Forward Backtesting...")

metrics, predictions = backtesting_forecaster(
    forecaster = forecaster,
    y          = data[Target],
    exog       = data[Exog_Vars],
    cv         = cv,
    metric     = 'mean_absolute_error',
    n_jobs     = 'auto',
    verbose    = False
)

# ==========================================
# --- 6. FINAL COMPARISON PLOT ---
# ==========================================
print("Generating Results...")

actuals = data.loc[predictions.index, Target]
history = data.loc[HistorieAnfang:TrainingEnde, Target]

plt.figure(figsize=(15, 8))

plt.plot(history.index, history.values, label="History (Static Training)", color="#1f77b4", alpha=0.8)
plt.plot(actuals.index, actuals.values, label="Actual Reality", color="black", linestyle="--", alpha=0.7)
plt.plot(predictions.index, predictions['pred'], label="XGBoost Walk-Forward (Dynamic)", color="red", linewidth=2)

plt.axvline(pd.to_datetime(TrainingEnde), color='grey', linestyle='--', label='Train/Test Split')
plt.title("Master's Project: Dynamic Recursive Walk-Forward Forecasting (v0.22.0)", fontsize=14)
plt.ylabel("Deposit Volume [Mrd. €]")
plt.xlabel("Date")
plt.legend(loc="upper left")
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.show()

print("\n--- Performance Metrics ---")
print(metrics)