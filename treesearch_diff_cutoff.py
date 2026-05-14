import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. --- DEFINE YOUR START AND SPLIT DATES HERE ---
START_DATE = '2003-01-27' 

split_dates = [
    '2025-03-15',
    
    ]
# -------------------------------------------------

# 2. Load and Prep Data
df = pd.read_excel('Data all variables.xlsx') 
df['Datum'] = pd.to_datetime(df['Datum'])
df = df.sort_values('Datum')

# Feature Engineering
df['Diff_Volume'] = df['Einlagevolumen'].diff() 
df['Prev_Volume'] = df['Einlagevolumen'].shift(1)

# Yearly Lag (Date-based merge for leap-year rigor)
df['Date_Last_Year'] = df['Datum'] - pd.DateOffset(years=1)
df_helper = df[['Datum', 'Einlagevolumen']].copy()
df_helper.columns = ['Date_Last_Year', 'Prev_Volume_Yearly']

df = pd.merge(df, df_helper, on='Date_Last_Year', how='left')
df = df.drop(columns=['Date_Last_Year'])
df['Prev_Volume_Yearly'] = df['Prev_Volume_Yearly'].ffill() 

df['Spread'] = df['€STR'] - df['Einlagezinssatz'] 
df['Year'] = df['Datum'].dt.year
df['Month'] = df['Datum'].dt.month
df['Mean_Zins_7W'] = df['Einlagezinssatz'].shift(1).rolling(window=7).mean()

# Sine/Cosine Seasonality (Optional but recommended for rigor)
df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)

df = df.dropna()

# Update features list
features = [
    '€STR', 
    'Einlagezinssatz', 
    'Spread', 
    'Year', 
    'Prev_Volume', 
    'Mean_Zins_7W', 
    'Prev_Volume_Yearly', 
    'Month_Sin', 
    'Month_Cos', 
    'GPRC_DEU', 
    'MoM Inflation', 
    'DAX', 
    '10Y Bond'
]

target = 'Diff_Volume'

# 3. BACKTESTING LOOP
print(f"Training Start Date: {START_DATE}")
# Adjusted header for the new column
print(f"{'Split Date':<15} | {'Train':<6} | {'Test':<6} | {'MAE':<8} | {'R2':<8} | {'Adj. R2':<8}")
print("-" * 75)

for d in split_dates:
    train = df[(df['Datum'] >= START_DATE) & (df['Datum'] < d)]
    test = df[df['Datum'] >= d]
    
    if len(train) < 50 or len(test) < len(features) + 2:
        print(f"{d:<15} | Not enough rows for Adj. R2 calculation.")
        continue

    # Train Model
    model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=5)
    model.fit(train[features], train[target])

    # Predict
    preds = model.predict(test[features])
    
    # 4. Metric Calculations
    mae = mean_absolute_error(test[target], preds)
    r2 = r2_score(test[target], preds)
    
    # Adjusted R2 Formula: 1 - [(1-R2)*(n-1) / (n-k-1)]
    n = len(test)
    k = len(features)
    adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - k - 1))
    
    print(f"{d:<15} | {len(train):<6} | {len(test):<6} | {mae:<8.4f} | {r2:<8.4f} | {adj_r2:<8.4f}")

print("-" * 75)