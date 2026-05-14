import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.stattools import durbin_watson
import warnings

warnings.filterwarnings('ignore')

# 1. DATA LOADING & CLEANING
# Use your specific file name here
file_path = 'data_pchip.xlsx' 
df = pd.read_excel(file_path)

if df.shape[1] > 4:
    df = df.iloc[:, :-2]

df['Datum'] = pd.to_datetime(df['Datum'])
df.set_index('Datum', inplace=True)
df.sort_index(inplace=True)

# 2. FEATURE ENGINEERING (Lag Variables)
df['Lag_1_Day'] = df['Einlagevolumen'].shift(1)
df['Lag_1_Week'] = df['Einlagevolumen'].rolling(window=7).mean().shift(1)
df = df.dropna()

# 3. TRAIN-TEST SPLIT
test_size = 20
train = df.iloc[:-test_size]
test = df.iloc[-test_size:]

y_train = train['Einlagevolumen']
y_test = test['Einlagevolumen']

# Helper function to calculate Adjusted R-squared
def get_adj_r2(y_true, y_pred, n_samples, n_features):
    r2 = r2_score(y_true, y_pred)
    return 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)

# Helper function for MAPE
def get_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# --- MODEL 1: BASE LINEAR REGRESSION ---
X_train_base = train[['€STR', 'Einlagezinssatz']]
X_test_base = test[['€STR', 'Einlagezinssatz']]
lr_base = LinearRegression().fit(X_train_base, y_train)
pred_base = lr_base.predict(X_test_base)

# --- MODEL 2: LAGGED (1 DAY) REGRESSION ---
X_train_lag = train[['€STR', 'Einlagezinssatz', 'Lag_1_Day']]
X_test_lag = test[['€STR', 'Einlagezinssatz', 'Lag_1_Day']]
lr_lag = LinearRegression().fit(X_train_lag, y_train)
pred_lag = lr_lag.predict(X_test_lag)

# --- MODEL 3: WEEKLY AVERAGE REGRESSION ---
X_train_avg = train[['€STR', 'Einlagezinssatz', 'Lag_1_Week']]
X_test_avg = test[['€STR', 'Einlagezinssatz', 'Lag_1_Week']]
lr_avg = LinearRegression().fit(X_train_avg, y_train)
pred_avg = lr_avg.predict(X_test_avg)

# --- MODEL 4: SARIMA (1,1,1) ---
sarima = SARIMAX(train['Einlagevolumen'], 
                 exog=train[['€STR', 'Einlagezinssatz']], 
                 order=(1, 1, 1))
sarima_fit = sarima.fit(disp=False)
pred_sarima = sarima_fit.get_forecast(steps=test_size, 
                                      exog=test[['€STR', 'Einlagezinssatz']]).predicted_mean

# 4. COMPREHENSIVE EVALUATION
models_metrics = {
    'Base LR': [rmse_base := np.sqrt(mean_squared_error(y_test, pred_base)),
                get_mape(y_test, pred_base),
                get_adj_r2(y_test, pred_base, len(y_test), 2)],
    
    'Lagged LR (1D)': [rmse_lag := np.sqrt(mean_squared_error(y_test, pred_lag)),
                       get_mape(y_test, pred_lag),
                       get_adj_r2(y_test, pred_lag, len(y_test), 3)],
    
    'Avg LR (1W)': [rmse_avg := np.sqrt(mean_squared_error(y_test, pred_avg)),
                    get_mape(y_test, pred_avg),
                    get_adj_r2(y_test, pred_avg, len(y_test), 3)],
    
    'SARIMA': [rmse_sarima := np.sqrt(mean_squared_error(y_test, pred_sarima)),
               get_mape(y_test, pred_sarima),
               r2_score(y_test, pred_sarima)] # SARIMA Adj R2 is complex, using standard R2
}

# Print Comparison Table
results_df = pd.DataFrame(models_metrics, index=['RMSE', 'MAPE (%)', 'Adj. R-Squared']).T
print("\n--- Model Comparison Table ---")
print(results_df)

print(f"\nSARIMA AIC: {sarima_fit.aic:.2f}")
print(f"SARIMA BIC: {sarima_fit.bic:.2f}")

# 5. RESIDUAL ANALYSIS (Durbin-Watson)
# Values near 2.0 mean no autocorrelation (good). Near 0 or 4 means patterns left (bad).
residuals_lag = y_test - pred_lag
dw_stat = durbin_watson(residuals_lag)
print(f"\nDurbin-Watson (Lagged Model): {dw_stat:.2f}")

# 6. VISUALIZATION
plt.figure(figsize=(12, 6))
plt.plot(test.index, y_test, label='Actual Data', color='black', lw=2)
plt.plot(test.index, pred_lag, label=f'Lagged LR (RMSE: {rmse_lag:.2f})', ls='--')
plt.plot(test.index, pred_sarima, label=f'SARIMA (RMSE: {rmse_sarima:.2f})', ls=':')
plt.title('Bank Deposit Prediction - Final Model Comparison')
plt.ylabel('Volume (Mio €)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()