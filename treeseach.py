import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# 1. Load Data (Replace 'your_data.csv' with your actual file)
df = pd.read_excel('Datensätze/Masterarbeit Werte pchip.xlsx') 

# Creating a dummy version of your data for the example:
data = {
    'Datum': pd.to_datetime(['2003-01-27', '2003-02-03', '2025-09-22', '2025-09-29']),
    'Einlagevolumen': [367.12, 368.88, 1871.21, 1868.47],
    'STR': [1.67, 1.68, 1.93, 1.92],
    'Einlagezinssatz': [1.20, 1.25, 0.44, 0.44]
}
df = pd.DataFrame(data)

# 2. Feature Engineering (Critical for Banking tasks)
df['Spread'] = df['STR'] - df['Einlagezinssatz'] # Difference to market
df['Year'] = df['Datum'].dt.year
df['Month'] = df['Datum'].dt.month

# 3. Time-Series Split (Don't shuffle! Train on past, test on "future")
df = df.sort_values('Datum')
train_size = int(len(df) * 0.8)
train, test = df.iloc[:train_size], df.iloc[train_size:]

features = ['STR', 'Einlagezinssatz', 'Spread', 'Year', 'Month']
target = 'Einlagevolumen'

X_train, y_train = train[features], train[target]
X_test, y_test = test[features], test[target]

# 4. Model: XGBoost
model = XGBRegressor(n_estimators=100, learning_rate=0.05)
model.fit(X_train, y_train)

# 5. Evaluation
preds = model.predict(X_test)
print(f"MAE: {mean_absolute_error(y_test, preds):.2f}")
print(f"R2 Score: {r2_score(y_test, preds):.2f}")

# 6. Predict for a specific scenario
# Scenario: Market rate (STR) stays at 2.0%, but we raise our rate to 1.5%
scenario = pd.DataFrame({
    'STR': [2.0],
    'Einlagezinssatz': [1.5],
    'Spread': [0.5],
    'Year': [2025],
    'Month': [10]
})
future_vol = model.predict(scenario)
print(f"\nPredicted Volume for scenario: {future_vol[0]:.2f}")