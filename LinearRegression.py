import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION
# ==========================================
START_DATE = '2005-01-01'  
END_DATE   = '2018-12-31'  
TEST_DAYS  = 20            
# ==========================================

# 1. DATA LOADING & CLEANING
file_path = 'Data all variables.xlsx' 
df = pd.read_excel(file_path)

df['Datum'] = pd.to_datetime(df['Datum'])
df.set_index('Datum', inplace=True)
df.sort_index(inplace=True)

# 2. FEATURE ENGINEERING
# El Lag ahora toma el último registro disponible sin importar el salto de días
df['Lag_1_Day'] = df['Einlagevolumen'].shift(1)
df['Lag_1_Week'] = df['Einlagevolumen'].rolling(window=7).mean().shift(1)

# --- APPLY DATE FILTERING ---
df = df.loc[START_DATE : END_DATE]
df = df.dropna()

print(f"Analysis Period: {df.index.min().date()} to {df.index.max().date()}")
print(f"Total observations: {len(df)}")

# 3. TRAIN-TEST SPLIT
train = df.iloc[:-TEST_DAYS]
test = df.iloc[-TEST_DAYS:]

y_train = train['Einlagevolumen']
y_test = test['Einlagevolumen']

# Feature Lists
base_features = ['€STR', 'Einlagezinssatz', 'GPRC_DEU', 'MoM Inflation', 'DAX', '10Y Bond']
lag_features = base_features + ['Lag_1_Day']

# Helper functions
def get_adj_r2(y_true, y_pred, n_samples, n_features):
    r2 = r2_score(y_true, y_pred)
    return 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)

def get_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# --- MODEL 1: BASE LINEAR REGRESSION ---
X_train_base = train[base_features]
X_test_base = test[base_features]
lr_base = LinearRegression().fit(X_train_base, y_train)
pred_base = lr_base.predict(X_test_base)

# --- MODEL 2: LAGGED (1 DAY) REGRESSION ---
X_train_lag = train[lag_features]
X_test_lag = test[lag_features]
lr_lag = LinearRegression().fit(X_train_lag, y_train)
pred_lag = lr_lag.predict(X_test_lag)

# --- MODEL 4: SARIMA (1,1,1) ---
# Al no tener frecuencia fija, SARIMA tratará los datos como una secuencia simple
sarima = SARIMAX(train['Einlagevolumen'], 
                 exog=train[base_features], 
                 order=(1, 1, 1),
                 enforce_stationarity=False,
                 enforce_invertibility=False) 
sarima_fit = sarima.fit(disp=False)
pred_sarima = sarima_fit.get_forecast(steps=TEST_DAYS, 
                                      exog=test[base_features]).predicted_mean

# 4. COMPREHENSIVE EVALUATION
models_metrics = {
    'Base LR': [np.sqrt(mean_squared_error(y_test, pred_base)),
                get_mape(y_test, pred_base),
                get_adj_r2(y_test, pred_base, len(y_test), 6)],
    
    'Lagged LR (1D)': [np.sqrt(mean_squared_error(y_test, pred_lag)),
                       get_mape(y_test, pred_lag),
                       get_adj_r2(y_test, pred_lag, len(y_test), 7)],
    
    'SARIMA': [np.sqrt(mean_squared_error(y_test, pred_sarima)),
               get_mape(y_test, pred_sarima),
               r2_score(y_test, pred_sarima)]
}

results_df = pd.DataFrame(models_metrics, index=['RMSE', 'MAPE (%)', 'Adj. R-Squared']).T
print("\n--- Model Comparison Table ---")
print(results_df.to_string())

# 5. VISUALIZATION
plt.figure(figsize=(12, 6))
plt.plot(test.index, y_test, label='Actual Data', color='black', lw=2, marker='o', markersize=4)
plt.plot(test.index, pred_lag, label='Lagged LR (1D)', ls='--')
plt.plot(test.index, pred_sarima, label='SARIMA', ls=':')
plt.title(f'Prediction Comparison (Observed Days only: {START_DATE} to {END_DATE})')
plt.ylabel('Mio €')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


# as an autoregresive model SARIMA(1,1,1) only "behaves" nice during the first predictions, 
# after that errors take a toll on the predictions unlike lagged linear regression which
# has the advantage of having the day before variable import seaborn as sns



plt.figure(figsize=(15, 7))

# --- Plot 1: Actual vs Predicted (Identity Line) ---
plt.subplot(1, 2, 1)

# Adding the three models for comparison
plt.scatter(test['Einlagevolumen'], pred_base, color='green', alpha=0.4, label='Normal LR (No Lag)')
plt.scatter(test['Einlagevolumen'], pred_lag, color='blue', alpha=0.5, label='Lagged LR (1D)')
plt.scatter(test['Einlagevolumen'], pred_sarima, color='red', alpha=0.5, label='SARIMA')

# The "Perfect Prediction" Reference Line
min_val = test['Einlagevolumen'].min()
max_val = test['Einlagevolumen'].max()
plt.plot([min_val, max_val], [min_val, max_val], color='black', lw=1.5, ls='--', label='Perfect Prediction')

plt.title('Actual vs Predicted: Model Comparison', fontsize=14)
plt.xlabel('Actual Values (Mio €)', fontsize=12)
plt.ylabel('Predicted Values (Mio €)', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.2)

# --- Plot 2: Feature Importance (Focus on Lagged Model) ---
plt.subplot(1, 2, 2)
# Using coefficients from your Lagged LR as it's the most explanatory
importance = pd.Series(lr_lag.coef_, index=lag_features).sort_values()
importance.plot(kind='barh', color='teal')

plt.title('Feature Importance (Lagged LR Coefficients)', fontsize=14)
plt.xlabel('Coefficient Weight (Impact)', fontsize=12)
plt.axvline(0, color='black', lw=1) # Zero line reference

plt.tight_layout()
plt.show()