import yfinance as yf
import pandas as pd

# Download
dax = yf.download("^GDAXI", start="2000-01-01", interval="1d")

# Index -> Spalte
dax = dax.reset_index()

# 👉 FIX: MultiIndex entfernen
if isinstance(dax.columns, pd.MultiIndex):
    dax.columns = dax.columns.get_level_values(0)

# Auswahl
dax = dax[["Date", "Open", "High", "Low", "Close", "Volume"]]

# Export
dax.to_excel("dax_daily.xlsx", index=False)

print("✅ Fertig!")
