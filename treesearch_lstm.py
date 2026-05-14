import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from tqdm import tqdm

# --- 1. CONFIGURATION ---
Daten = "Data all variables.xlsx"
# All 7 variables as inputs
Variablen = ["Einlagevolumen", "€STR", "Einlagezinssatz", "GPRC_DEU", "MoM Inflation", "DAX", "10Y Bond"]
target_col_idx = 0       # Index of "Einlagevolumen"

Gedächtnis = 100         # Lookback (History)
Prognosehorizont = 100   # Future (Forecast)
Simulationen = 5        # Ensemble size for Uncertainty
Seed = 42

# Define your Split Point
TrainingEnde = "2023-11-30" 
InputAnfang = "2023-12-01" 
HistorieAnfang, HistorieEnde = "2023-06-01", "2023-11-30"

# --- 2. DATA PREPARATION ---
df = pd.read_excel(Daten)
df["Datum"] = pd.to_datetime(df["Datum"])
df = df.sort_values("Datum").set_index("Datum")
df = df[Variablen]

# SCIENTIFIC STEP: Calculate Differences for the Target
# This solves the "Jump" problem because the model now learns the "Move" not the "Level"
df['Diff_Volume'] = df['Einlagevolumen'].diff()

# Scaling Features (0 to 1)
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler() # Separate scaler for the target "Diff"

df_clean = df.dropna()
X_data = scaler_X.fit_transform(df_clean[Variablen].values)
y_data = scaler_y.fit_transform(df_clean[['Diff_Volume']].values)

# --- 3. SEQUENCE GENERATION (Direct Multi-Step) ---
def create_sequences(X_arr, y_arr, gd, hz):
    X_seq, y_seq = [], []
    for i in range(len(X_arr) - gd - hz + 1):
        # Input: All 7 features for the last 100 days
        X_seq.append(X_arr[i : i + gd].flatten()) 
        # Output: Only the DIFF of Volume for the next 100 days
        y_seq.append(y_arr[i + gd : i + gd + hz].flatten()) 
    return np.array(X_seq), np.array(y_seq)

X_seq, y_seq = create_sequences(X_data, y_data, Gedächtnis, Prognosehorizont)

# Training Split
split_idx = df_clean.index.get_loc(TrainingEnde) - Gedächtnis
X_train, y_train = X_seq[:split_idx], y_seq[:split_idx]

# --- 4. ENSEMBLE TRAINING ---
ensemble_preds = []

print(f"Training Scientific Treesearch Ensemble...")
for i in tqdm(range(Simulationen)):
    model = MultiOutputRegressor(XGBRegressor(
        n_estimators=100, 
        learning_rate=0.05, 
        max_depth=5, 
        random_state=Seed + i,
        subsample=0.8
    ))
    
    model.fit(X_train, y_train)
    
    # 5. INFERENCE
    # Get the snapshot right before the forecast
    input_idx = df_clean.index.get_loc(TrainingEnde)
    x_input = X_data[input_idx - Gedächtnis : input_idx].flatten().reshape(1, -1)
    
    # Predict the next 100 DIFFERENCES
    raw_diff_pred = model.predict(x_input).reshape(Prognosehorizont, 1)
    
    # Inverse Scale the differences
    inv_diff_pred = scaler_y.inverse_transform(raw_diff_pred).flatten()
    
    # SCIENTIFIC STEP: Reconstruct Total Volume (Cumulative Sum)
    # New Level = Last Level + Sum of Predicted Changes
    last_actual_level = df.loc[TrainingEnde, "Einlagevolumen"]
    reconstructed_volume = last_actual_level + np.cumsum(inv_diff_pred)
    
    ensemble_preds.append(reconstructed_volume)

ensemble_preds = np.array(ensemble_preds)

# --- 6. STATISTICS ---
Mittelwert = ensemble_preds.mean(axis=0)
ki90 = np.percentile(ensemble_preds, [5, 95], axis=0)
ki98 = np.percentile(ensemble_preds, [1, 99], axis=0)

# Real values for comparison
real_start_idx = df.index.get_loc(InputAnfang)
real_values = df.iloc[real_start_idx : real_start_idx + Prognosehorizont, 0].values

# --- 7. FINAL PLOT (Scientific Style) ---
Historie = df.loc[HistorieAnfang:HistorieEnde, "Einlagevolumen"]
x_Zukunft = pd.date_range(InputAnfang, periods=Prognosehorizont, freq="D")

plt.figure(figsize=(15, 8))

# Reality
plt.plot(Historie.index, Historie.values, color="blue", label="Historische Daten (Training)")
plt.plot(x_Zukunft, real_values[:len(x_Zukunft)], "--", color="black", label="Tatsächliche Reale Werte (Test)")

# AI Forecast
plt.plot(x_Zukunft, Mittelwert, color="red", label="XGBoost Ensemble Mittelwert (Diff-anchored)", linewidth=2)

# Uncertainty Bands
plt.fill_between(x_Zukunft, ki90[0], ki90[1], color="red", alpha=0.2, label="90% Konfidenzintervall")
plt.fill_between(x_Zukunft, ki98[0], ki98[1], color="red", alpha=0.1, label="98% Konfidenzintervall")

plt.title(f"Rigorous Treesearch Forecast: Level Reconstruction via Predicted Differences", fontsize=14)
plt.ylabel("Einlagevolumen [Mrd. €]")
plt.xlabel("Datum")
plt.grid(True, alpha=0.3)
plt.legend(loc="upper left")
plt.tight_layout()
plt.show()