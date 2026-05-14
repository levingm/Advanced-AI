import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# 1. --- DEFINE YOUR START AND SPLIT DATES HERE ---
START_DATE = '2025-03-15'
END_DATE = '2025-03-15' 
split_dates = [
    '2025-03-15',
    
    
    ]

# 2. Load and Prep Data
df = pd.read_excel('Data all variables.xlsx') 
df['Datum'] = pd.to_datetime(df['Datum'])
df = df.sort_values('Datum')

# Feature Engineering
df['Diff_Volume'] = df['Einlagevolumen'].diff() 
df['Prev_Volume'] = df['Einlagevolumen'].shift(1)
df['Date_Last_Year'] = df['Datum'] - pd.DateOffset(years=1)
df_helper = df[['Datum', 'Einlagevolumen']].copy(); df_helper.columns = ['Date_Last_Year', 'Prev_Volume_Yearly']
df = pd.merge(df, df_helper, on='Date_Last_Year', how='left').drop(columns=['Date_Last_Year'])
df['Prev_Volume_Yearly'] = df['Prev_Volume_Yearly'].ffill() 

df['Spread'] = df['€STR'] - df['Einlagezinssatz'] 
df['Year'] = df['Datum'].dt.year
df['Month'] = df['Datum'].dt.month
df['Mean_Zins_7W'] = df['Einlagezinssatz'].shift(1).rolling(window=7).median()
df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)
df = df.dropna()

features = ['€STR', 'Einlagezinssatz', 'Spread', 'Year', 'Prev_Volume', 'Mean_Zins_7W', 
            'Prev_Volume_Yearly', 'Month_Sin', 'Month_Cos', 'GPRC_DEU', 'MoM Inflation', 'DAX', '10Y Bond']
target = 'Diff_Volume'

# 3. BACKTESTING LOOP
print(f"{'Split Date':<15} | {'Train':<6} | {'Test':<6} | {'MAE':<8} | {'R2':<8} | {'Adj. R2':<8}")
print("-" * 75)

for d in split_dates:
    train = df[(df['Datum'] >= START_DATE) & (df['Datum'] < d) & (df['Datum']<= END_DATE)]
    test = df[df['Datum'] >= d].copy()
    
    if len(train) < 50 or len(test) < len(features) + 2:
        continue

    model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=5)
    model.fit(train[features], train[target])
    preds = model.predict(test[features])
    
    # Metrics
    mae = mean_absolute_error(test[target], preds)
    r2 = r2_score(test[target], preds)
    adj_r2 = 1 - ((1 - r2) * (len(test) - 1) / (len(test) - len(features) - 1))
    
    print(f"{d:<15} | {len(train):<6} | {len(test):<6} | {mae:<8.4f} | {r2:<8.4f} | {adj_r2:<8.4f}")

# 4. --- PLOTTING SECTION (Using the last split date) ---
# Reconstruct Total Volume for the plot
# --- RECURSIVE FORECASTING SECTION ---
# We start with the very first row of the test set
current_data = test.iloc[0:1].copy()
predictions = []

# We loop through the test set one by one
for i in range(len(test)):
    # 1. Predict the change for the current week
    pred_diff = model.predict(current_data[features])[0]
    
    # 2. Calculate the total volume based on our PREVIOUS PREDICTION
    # (Except for the first step where we use the last training point)
    if i == 0:
        prev_vol = test['Prev_Volume'].iloc[0]
    else:
        prev_vol = predictions[-1]
        
    current_total = prev_vol + pred_diff
    predictions.append(current_total)
    
    # 3. Prepare the data for the NEXT week
    if i + 1 < len(test):
        # We take the actual external data for next week (DAX, Bonds, etc.)
        current_data = test.iloc[i+1:i+2].copy()
        # BUT we overwrite the 'Prev_Volume' with our own AI prediction!
        current_data['Prev_Volume'] = current_total
        # Update other lags if necessary (like Mean_Zins_7W if it used Volume)

test['Recursive_Prediction'] = predictions

# --- UPDATED PLOT ---
plt.figure(figsize=(12, 6))
plt.plot(test['Datum'], test['Einlagevolumen'], color='blue', label='Actual Volume (Reality)', linewidth=2)
plt.plot(test['Datum'], test['Recursive_Prediction'], color='red', linestyle='--', label='Recursive AI Forecast', linewidth=2)

plt.title('Realistic Multi-Step Forecast (Accumulated Error)', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()