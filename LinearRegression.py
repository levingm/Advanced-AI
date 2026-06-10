"""
==============================================================================
LINEAR REGRESSION + SARIMA
==============================================================================
"""

# ==============================================================================
# 0. CONFIGURATION
# ==============================================================================
FILE_PATH  = 'Data all variables.xlsx'
START_DATE = '2005-01-01'
END_DATE   = '2018-12-31'
TEST_DAYS  = 20

TARGET   = 'Einlagevolumen'
FEATURES = ['€STR', 'Einlagezinssatz', 'GPRC_DEU', 'MoM Inflation', 'DAX', '10Y Bond']

# ==============================================================================
# 1. IMPORT
# ==============================================================================
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from sklearn.linear_model    import LinearRegression
from sklearn.metrics         import (mean_squared_error, mean_absolute_error,
                                     r2_score, mean_absolute_percentage_error)

import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax   import SARIMAX
from statsmodels.tsa.stattools            import adfuller
from statsmodels.stats.stattools          import durbin_watson
from statsmodels.stats.diagnostic         import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor

plt.style.use('seaborn-v0_8-whitegrid')
PALETTE = {'actual': '#1f1f1f', 'base': '#2196F3', 'lag': '#4CAF50', 'sarima': '#9C27B0'}

# ==============================================================================
# 2. DATA
# ==============================================================================
df_raw = pd.read_excel(FILE_PATH)
df_raw['Datum'] = pd.to_datetime(df_raw['Datum'])
df_raw.set_index('Datum', inplace=True)
df_raw.sort_index(inplace=True)

# ==============================================================================
# 3. LAGGED 1-DAY FEATURE
# ==============================================================================
df = df_raw.copy()
df['Lag_1_Day'] = df[TARGET].shift(1)

df = df.loc[START_DATE:END_DATE].dropna()

print(f"\nAnalysis period : {df.index.min().date()} → {df.index.max().date()}")
print(f"Observations    : {len(df):,}  |  Test window: {TEST_DAYS} obs")

# ==============================================================================
# 4. TRAIN / TEST SPLIT
# ==============================================================================
train = df.iloc[:-TEST_DAYS].copy()
test  = df.iloc[-TEST_DAYS:].copy()
y_train, y_test = train[TARGET], test[TARGET]
LAG_FEATURES = FEATURES + ['Lag_1_Day']

# ==============================================================================
# 5. BASE LR, LAGGED LR, SARIMA
# ==============================================================================
models, preds = {}, {}

# Base LR
lr_base = LinearRegression().fit(train[FEATURES], y_train)
models['Base LR'] = lr_base
preds['Base LR']  = lr_base.predict(test[FEATURES])

# Lagged LR
lr_lag = LinearRegression().fit(train[LAG_FEATURES], y_train)
models['Lagged LR'] = lr_lag
preds['Lagged LR']  = lr_lag.predict(test[LAG_FEATURES])

# SARIMA(1,1,1)
sarima_fit = SARIMAX(
    y_train,
    exog=train[FEATURES],
    order=(1, 1, 1),
    enforce_stationarity=False,
    enforce_invertibility=False
).fit(disp=False)
sarima_forecast = sarima_fit.get_forecast(steps=TEST_DAYS, exog=test[FEATURES])
pred_sarima     = sarima_forecast.predicted_mean
conf_int        = sarima_forecast.conf_int(alpha=0.05)
preds['SARIMA'] = pred_sarima.values

# ==============================================================================
# 6. METRICS 
# ==============================================================================
def compute_metrics(y_true, y_pred, n_features):
    n      = len(y_true)
    r2     = r2_score(y_true, y_pred)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
    rmse   = np.sqrt(mean_squared_error(y_true, y_pred))
    mae    = mean_absolute_error(y_true, y_pred)
    mape   = mean_absolute_percentage_error(y_true, y_pred) * 100
    dir_acc = (np.mean(np.sign(np.diff(np.array(y_true))) ==
                       np.sign(np.diff(np.array(y_pred)))) * 100
               if n > 1 else np.nan)
    return {
        'R²':           round(r2,      4),
        'Adj. R²':      round(adj_r2,  4),
        'RMSE':         round(rmse,    4),
        'MAE':          round(mae,     4),
        'MAPE (%)':     round(mape,    4),
        'Dir. Acc (%)': round(dir_acc, 2),
    }

metric_rows = {
    'Base LR':   compute_metrics(y_test, preds['Base LR'],   len(FEATURES)),
    'Lagged LR': compute_metrics(y_test, preds['Lagged LR'], len(LAG_FEATURES)),
    'SARIMA':    compute_metrics(y_test, pred_sarima.values, len(FEATURES) + 2),
}

results_df = pd.DataFrame(metric_rows).T
print("\n" + "=" * 60)
print(f"OUT-OF-SAMPLE METRICS  (test set: last {TEST_DAYS} observations)")
print("=" * 60)
print(results_df.to_string())

# ==============================================================================
# 8. VISUALS
# ==============================================================================

# DAY FORECAST
fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(test.index, y_test,             label='Actual',    color=PALETTE['actual'], lw=2.5, marker='o', ms=4)
ax.plot(test.index, preds['Base LR'],   label='Base LR',   color=PALETTE['base'],   ls='--', lw=1.8)
ax.plot(test.index, preds['Lagged LR'], label='Lagged LR',     color=PALETTE['lag'],    ls='-.', lw=2)
ax.plot(test.index, pred_sarima,        label='SARIMA(1,1,1)', color=PALETTE['sarima'], ls=':',  lw=2)
ax.set_title(f'Out-of-Sample Forecast — Base LR vs Lagged LR vs SARIMA\n'
             f'Train: {START_DATE} → {train.index.max().date()}  |  Test: last {TEST_DAYS} obs',
             fontsize=13)
ax.set_ylabel('Deposit Volume (Mio €)')
ax.legend()
plt.tight_layout()
plt.savefig('02_forecast_all_models.png', dpi=150)
plt.show()

# LAGGED LR RESIDUALS
resid_train = y_train.values - lr_lag.predict(train[LAG_FEATURES])

fig = plt.figure(figsize=(14, 9))
gs  = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.3)

ax1 = fig.add_subplot(gs[0, :])
ax1.plot(train.index, resid_train, color='steelblue', alpha=0.7, lw=0.8)
ax1.axhline(0, color='red', ls='--', lw=1)
ax1.set_title('Residuals over Time (in-sample)')
ax1.set_ylabel('Residual (Mio €)')

ax2 = fig.add_subplot(gs[1, 0])
stats.probplot(resid_train, plot=ax2)
ax2.set_title('Q-Q Plot — Normality of Residuals')

ax3 = fig.add_subplot(gs[1, 1])
ax3.scatter(lr_lag.predict(train[LAG_FEATURES]), resid_train, alpha=0.3, s=10, color='steelblue')
ax3.axhline(0, color='red', ls='--')
ax3.set_xlabel('Fitted Values')
ax3.set_ylabel('Residuals')
ax3.set_title('Residuals vs Fitted (Homoskedasticity Check)')

plt.suptitle('Residual Diagnostics — Lagged LR', fontsize=14, fontweight='bold')
plt.savefig('03_residual_diagnostics.png', dpi=150)
plt.show()

# --- 8e. SARIMA forecast with 95% confidence interval ---
fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(test.index, y_test,             label='Actual',        color=PALETTE['actual'], lw=2.5, marker='o', ms=4)
ax.plot(test.index, preds['Lagged LR'], label='Lagged LR',     color=PALETTE['lag'],    ls='-.', lw=2)
ax.plot(test.index, pred_sarima,        label='SARIMA(1,1,1)', color=PALETTE['sarima'], ls='--', lw=2)
ax.fill_between(test.index,
                conf_int.iloc[:, 0], conf_int.iloc[:, 1],
                alpha=0.15, color=PALETTE['sarima'], label='SARIMA 95% CI')
ax.set_title('SARIMA vs Lagged LR — Forecast with 95% Confidence Interval', fontsize=13)
ax.set_ylabel('Deposit Volume (Mio €)')
ax.legend()
plt.tight_layout()
plt.savefig('06_sarima_forecast_ci.png', dpi=150)
plt.show()

print("""
NOTE — SARIMA error accumulation:
  SARIMA is autoregressive: each step uses its own prior prediction, so
  errors compound over the forecast horizon. Lagged LR avoids this because
  it uses the actual observed value (Lag_1_Day) at each test step.
  This is a key methodological limitation to address in your thesis.
""")