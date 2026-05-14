import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load and Sort Data
df = pd.read_excel('data_pchip.xlsx') 
df['Datum'] = pd.to_datetime(df['Datum'])
df = df.sort_values('Datum')

# 2. Feature Engineering
df['Spread'] = df['€STR'] - df['Einlagezinssatz'] 
df['Year'] = df['Datum'].dt.year
df['Month'] = df['Datum'].dt.month

# THE KEY CHANGES:
# A) Predict the CHANGE (Diff) instead of the total
df['Diff_Volume'] = df['Einlagevolumen'].diff() 

# B) Use the last volume as a feature (to know where we started)
df['Prev_Volume'] = df['Einlagevolumen'].shift(1)

# C) Rolling average of interest rates (helps model see the trend)
df['Mean_Zins_7W'] = df['Einlagezinssatz'].shift(1).rolling(window=7).mean()

# Drop rows with NaN (created by diff and shift)
df = df.dropna()

# 3. Define Features and Target
# Note: We don't include Datum, but we include the Year/Month we extracted
features = ['€STR', 'Einlagezinssatz', 'Spread', 'Year', 'Month', 'Prev_Volume', 'Mean_Zins_7W']
target = 'Diff_Volume' 

# 4. Time-Series Split (80/20)
train_size = int(len(df) * 0.8)
train, test = df.iloc[:train_size], df.iloc[train_size:]

X_train, y_train = train[features], train[target]
X_test, y_test = test[features], test[target]

# 5. Model: XGBoost
model = XGBRegressor(n_estimators=500, learning_rate=0.01, max_depth=3)
model.fit(X_train, y_train)

# 6. Evaluation (On the Difference)
preds_diff = model.predict(X_test)
print(f"MAE (on weekly change): {mean_absolute_error(y_test, preds_diff):.2f}")
print(f"R2 Score (on weekly change): {r2_score(y_test, preds_diff):.2f}")

# 7. Predict for Scenario
last_volume = df['Einlagevolumen'].iloc[-1]
last_zins_mean = df['Mean_Zins_7W'].iloc[-1]

scenario = pd.DataFrame({
    '€STR': [2.0],
    'Einlagezinssatz': [1.5],
    'Spread': [0.5],
    'Year': [2025],
    'Month': [10],
    'Prev_Volume': [last_volume],
    'Mean_Zins_7W': [last_zins_mean]
})

# The model predicts the CHANGE
predicted_change = model.predict(scenario)[0]
predicted_total = last_volume + predicted_change

print("-" * 70)
print(f"Predicted CHANGE for Oct 2025: {predicted_change:.2f}")
print(f"Predicted TOTAL VOLUME for Oct 2025: {predicted_total:.2f}")