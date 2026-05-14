import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load Data
df = pd.read_excel('data_pchip.xlsx') 
df['Datum'] = pd.to_datetime(df['Datum'])
df = df.sort_values('Datum')

# 2. Feature Engineering (DO THIS BEFORE SPLITTING)
df['Spread'] = df['€STR'] - df['Einlagezinssatz'] 
df['Year'] = df['Datum'].dt.year
df['Month'] = df['Datum'].dt.month
df['Prev_Volume'] = df['Einlagevolumen'].shift(1) # Important Lag
df['Mean_Week'] = df['Einlagezinssatz'].shift(1).rolling(window=7).mean()
#df['Med_Week'] = df['Einlagezinssatz'].shift(1).rolling(window=7).median()
# Drop the very first row because it has no "previous" volume
df = df.dropna()

# 3. Define Features (Make sure names match your Excel headers exactly)
# I used '€STR' here because that's what is in your Excel
#features = ['€STR', 'Einlagezinssatz', 'Spread', 'Year', 'Month', 'Prev_Volume']
features = ['€STR', 'Einlagezinssatz', 'Spread', 'Year', 'Month', 'Prev_Volume', 'Mean_Week']

target = 'Einlagevolumen'

# 4. Time-Series Split (Now that the columns exist!)
train_size = int(len(df) * 0.8)
train, test = df.iloc[:train_size], df.iloc[train_size:]

X_train, y_train = train[features], train[target]
X_test, y_test = test[features], test[target]

# 5. Model: XGBoost
model = XGBRegressor(n_estimators=100, learning_rate=0.05)
model.fit(X_train, y_train)

# 6. Evaluation
preds = model.predict(X_test)
print(f"MAE: {mean_absolute_error(y_test, preds):.2f}")
print(f"R2 Score: {r2_score(y_test, preds):.2f}")

# 7. Predict for a specific scenario
# We need to get the "Last Known Volume" to use as 'Prev_Volume' for our prediction
last_volume = df['Einlagevolumen'].iloc[-1]
mean_week = df['Mean_Week'].iloc[-7]

scenario = pd.DataFrame({
    '€STR': [2.0],
    'Einlagezinssatz': [1.5],
    'Spread': [0.5],
    'Year': [2025],
    'Month': [10],
    'Prev_Volume': [last_volume], # We tell the model where we are starting from
    'Mean_Week': [mean_week]
})

future_vol = model.predict(scenario)
print(f"\nPredicted Volume for October 2025: {future_vol[0]:.2f}")