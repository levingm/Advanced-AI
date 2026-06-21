import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
import warnings

# Suppress minor warnings for clean output
warnings.filterwarnings('ignore')

# ==========================================
# --- 1. CONFIGURATION (Development Setup) ---
# ==========================================
Daten = "Data all variables.xlsx"
Variablen = ["Einlagevolumen", "€STR", "Einlagezinssatz", "GPRC_DEU", "MoM Inflation", "DAX", "10Y Bond"]
target_col = "Einlagevolumen"

Gedächtnis = 30          # Lookback (History) - Reduced to 30 to prevent feature explosion (30 days * 7 vars = 210 features)
Prognosehorizont = 100   # Future (Forecast)
Simulationen = 20        # Number of models for uncertainty estimation
Seed = 42

# Strict Chronological Splitting (Crucial for Time Series)
TrainingEnde = "2023-11-30" 
InputAnfang = "2023-12-01" 
HistorieAnfang, HistorieEnde = "2023-06-01", "2023-11-30"

# ==========================================
# --- 2. DATA PREPARATION & LEAKAGE PREVENTION ---
# ==========================================
print("Loading and preprocessing data...")
df = pd.read_excel(Daten)
df["Datum"] = pd.to_datetime(df["Datum"])
df = df.sort_values("Datum").set_index("Datum")
df = df[Variablen]

# SCIENTIFIC STEP: Calculate Differences (Stationarization)
df['Diff_Volume'] = df[target_col].diff()
df = df.dropna()

# [MASTER LEVEL FIX]: Split Data FIRST, then scale. 
# This prevents future information from leaking into the training phase.
df_train = df.loc[:TrainingEnde].copy()
df_test = df.loc[InputAnfang:].copy()

# Initialize Scalers
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

# Fit scalers ONLY on training data
train_features_scaled = scaler_X.fit_transform(df_train[Variablen].values)
train_target_scaled = scaler_y.fit_transform(df_train[['Diff_Volume']].values)

# Transform test data using the knowledge from training ONLY
test_features_scaled = scaler_X.transform(df_test[Variablen].values)
test_target_scaled = scaler_y.transform(df_test[['Diff_Volume']].values)

# Recombine temporarily just for sequence generation (safely scaled)
X_data_safe = np.vstack((train_features_scaled, test_features_scaled))
y_data_safe = np.vstack((train_target_scaled, test_target_scaled))

# ==========================================
# --- 3. SEQUENCE GENERATION (Direct Multi-Step) ---
# ==========================================
def create_sequences(X_arr, y_arr, gd, hz):
    X_seq, y_seq = [], []
    # We stop creating sequences where we don't have enough future data
    for i in range(len(X_arr) - gd - hz + 1):
        X_seq.append(X_arr[i : i + gd].flatten()) 
        y_seq.append(y_arr[i + gd : i + gd + hz].flatten()) 
    return np.array(X_seq), np.array(y_seq)

X_seq, y_seq = create_sequences(X_data_safe, y_data_safe, Gedächtnis, Prognosehorizont)

# We must ensure our training sequences only contain data up to 'TrainingEnde'
# Calculate how many sequences belong to the training set
split_idx = len(df_train) - Gedächtnis - Prognosehorizont + 1

X_train, y_train = X_seq[:split_idx], y_seq[:split_idx]
# X_test, y_test could be used here for out-of-sample metrics later

# ==========================================
# --- 4. ENSEMBLE TRAINING (Gradient Boosting) ---
# ==========================================
ensemble_preds_diffs = []

print(f"Training XGBoost Direct-Multistep Ensemble ({Simulationen} runs)...")
for i in range(Simulationen):
    # [MASTER LEVEL]: Using subsample and colsample_bytree to induce variance for our Prediction Intervals
    base_model = XGBRegressor(
        n_estimators=100, 
        learning_rate=0.05, 
        max_depth=4,              # Lower depth prevents overfitting on time series
        subsample=0.7,            # Row sampling
        colsample_bytree=0.7,     # Feature sampling
        random_state=Seed + i,
        n_jobs=-1
    )
    
    # MultiOutputRegressor wraps the XGBoost to predict all 100 days directly
    model = MultiOutputRegressor(base_model)
    model.fit(X_train, y_train)
    
    # --- 5. INFERENCE (Forecasting the Unseen Future) ---
    # Get the exact snapshot of the last 'Gedächtnis' days right before the test period
    input_idx = len(df_train)
    x_input = X_data_safe[input_idx - Gedächtnis : input_idx].flatten().reshape(1, -1)
    
    # Predict the next 100 DIFFERENCES
    raw_diff_pred = model.predict(x_input).reshape(Prognosehorizont, 1)
    
    # Inverse Scale the differences back to real € amounts
    inv_diff_pred = scaler_y.inverse_transform(raw_diff_pred).flatten()
    ensemble_preds_diffs.append(inv_diff_pred)

# Convert to numpy array (Shape: 20 simulations x 100 days)
ensemble_preds_diffs = np.array(ensemble_preds_diffs)

# ==========================================
# --- 6. SCIENTIFIC RECONSTRUCTION (Diff -> Level) ---
# ==========================================
print("Reconstructing Volume Levels...")
# Anchor point: The literal last known volume before the future begins
last_actual_level = df_train.iloc[-1][target_col]

# Reconstruct each of the 20 simulations
ensemble_preds_levels = []
for diff_path in ensemble_preds_diffs:
    reconstructed_path = last_actual_level + np.cumsum(diff_path)
    ensemble_preds_levels.append(reconstructed_path)

ensemble_preds_levels = np.array(ensemble_preds_levels)

# Calculate Statistics over the simulations
Mittelwert = ensemble_preds_levels.mean(axis=0)
# Use the correct terminology: Prediction Interval (Prädiktionsintervall)
pi90 = np.percentile(ensemble_preds_levels, [5, 95], axis=0)

# Real actual values for comparison (Out-of-sample)
real_values = df_test[target_col].iloc[:Prognosehorizont].values

# ==========================================
# --- 7. FINAL SCIENTIFIC PLOT ---
# ==========================================
print("Plotting results...")
# Get history for the plot
Historie_plot = df_train.loc[HistorieAnfang:TrainingEnde, target_col]
x_Zukunft = df_test.index[:Prognosehorizont]

plt.figure(figsize=(14, 7))

# 1. Plot Historical Training Data
plt.plot(Historie_plot.index, Historie_plot.values, color="#1f77b4", label="Historical Data (Train)", linewidth=2)

# 2. Plot True Future Data (Test Set)
plt.plot(x_Zukunft, real_values, linestyle="--", color="#2ca02c", label="Actual Values (Test/Out-of-Sample)", linewidth=2)

# 3. Plot AI Forecast
plt.plot(x_Zukunft, Mittelwert, color="#d62728", label="XGBoost Direct Forecast (Mean)", linewidth=2)

# 4. Plot Uncertainty (Prediction Intervals, NOT Confidence Intervals)
plt.fill_between(x_Zukunft, pi90[0], pi90[1], color="#d62728", alpha=0.15, label="90% Prediction Interval")

# Scientific Formatting
plt.title("Out-of-Sample Multi-Step Forecast of Deposit Volumes using XGBoost", fontsize=14, fontweight='bold')
plt.ylabel("Deposit Volume [Mrd. €]", fontsize=12)
plt.xlabel("Date", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc="upper left", fontsize=11, framealpha=0.9)
plt.tight_layout()

# Draw a vertical line to explicitly show the Train/Test split
plt.axvline(pd.to_datetime(TrainingEnde), color='black', linestyle=':', linewidth=1.5, label='Train/Test Split')

plt.show()