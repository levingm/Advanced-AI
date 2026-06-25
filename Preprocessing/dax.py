import yfinance as yf
import pandas as pd

# 1. Download
dax = yf.download("^GDAXI", start="2000-01-01", interval="1d")

# 2. MultiIndex sofort plattmachen (bevor reset_index)
if isinstance(dax.columns, pd.MultiIndex):
    dax.columns = dax.columns.get_level_values(0)

# 3. Index zu Spalte machen
dax = dax.reset_index()

# 4. Auswahl der Spalten
dax = dax[["Date", "Open", "High", "Low", "Close", "Volume"]]

# ⭐ DIE RADIKALE LÖSUNG: Datum in echten Text umwandeln (YYYY-MM-DD)
# Das verhindert, dass Excel die Zahl 36528 anzeigt.
dax["Date"] = dax["Date"].dt.strftime('%Y-%m-%d')

# 5. Export
dax.to_excel("dax_daily.xlsx", index=False)

print("✅ Fertig! Schau jetzt in die Excel-Datei.")