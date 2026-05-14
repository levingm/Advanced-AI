import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. --- DEFINE YOUR START AND SPLIT DATES HERE ---
START_DATE = '2003-01-27'  # The model will IGNORE everything before this

split_dates = [
    '2021-01-23',
    '2022-01-23',
    '2023-01-23', 
    '2024-01-23', 
    '2025-01-23',
    '2025-02-23',
    '2025-03-23',
    '2025-04-23',
    '2025-05-23',
]
# -------------------------------------------------

# 2. Load and Prep Data
df = pd.read_excel('data_pchip.xlsx') 
df['Datum'] = pd.to_datetime(df['Datum'])
df = df.sort_values('Datum')

# Feature Engineering (Done on full set to keep lags consistent)
df['Diff_Volume'] = df['Einlagevolumen'].diff() 
df['Prev_Volume'] = df['Einlagevolumen'].shift(1)
df['Spread'] = df['€STR'] - df['Einlagezinssatz'] 
df['Year'] = df['Datum'].dt.year
df['Month'] = df['Datum'].dt.month
df['Mean_Zins_7W'] = df['Einlagezinssatz'].shift(1).rolling(window=7).mean()
df = df.dropna()

features = ['€STR', 'Einlagezinssatz', 'Spread', 'Year', 'Month', 'Prev_Volume', 'Mean_Zins_7W']
target = 'Diff_Volume'

# 3. BACKTESTING LOOP
print(f"Training Start Date: {START_DATE}")
print(f"{'Split Date':<15} | {'Train Rows':<10} | {'Test Rows':<10} | {'MAE':<10} | {'R2':<10}")
print("-" * 65)

for d in split_dates:
    # Filter: Only take data BETWEEN Start Date and the Split Date
    train = df[(df['Datum'] >= START_DATE) & (df['Datum'] < d)]
    test = df[df['Datum'] >= d]
    
    if len(train) < 50 or len(test) < 2:
        print(f"{d:<15} | Not enough data in this window.")
        continue

    # Train Model
    model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=5)
    model.fit(train[features], train[target])

    # Predict
    preds = model.predict(test[features])
    
    # Metrics
    mae = mean_absolute_error(test[target], preds)
    r2 = r2_score(test[target], preds)
    
    print(f"{d:<15} | {len(train):<10} | {len(test):<10} | {mae:<10.4f} | {r2:<10.4f}")

print("-" * 65)
