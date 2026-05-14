import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator

# -------------------------------
# 1. Excel-Datei einlesen
# -------------------------------
file_path = "geopolitical risk index.xls"

df = pd.read_excel(file_path)

# Falls Spalten anders heißen, automatisch erkennen
df.columns = [c.strip() for c in df.columns]

# Erwartet: 'month' und ein Wert (z. B. GPRC_DEU)
date_col = [c for c in df.columns if "month" in c.lower()][0]
value_col = [c for c in df.columns if c != date_col][0]

df = df[[date_col, value_col]].dropna()

# -------------------------------
# 2. Datum korrekt parsen
# -------------------------------
df[date_col] = pd.to_datetime(df[date_col], dayfirst=True)

df = df.sort_values(by=date_col)

# -------------------------------
# 3. Numerische Zeitachse
# -------------------------------
x = df[date_col].map(pd.Timestamp.toordinal).values
y = df[value_col].values

# -------------------------------
# 4. PCHIP-Modell
# -------------------------------
pchip = PchipInterpolator(x, y, extrapolate=False)

# -------------------------------
# 5. Tägliche Zeitreihe
# -------------------------------
daily_dates = pd.date_range(
    start=df[date_col].min(),
    end=df[date_col].max(),
    freq="D"
)

x_daily = daily_dates.map(pd.Timestamp.toordinal).values

# -------------------------------
# 6. Interpolation
# -------------------------------
y_daily = pchip(x_daily)

# -------------------------------
# 7. Ergebnis
# -------------------------------
df_daily = pd.DataFrame({
    "date": daily_dates,
    value_col: y_daily
})

# -------------------------------
# 8. Export
# -------------------------------
output_file = "daily_pchip_interpolated.xlsx"
df_daily.to_excel(output_file, index=False)

print("Fertig! Datei gespeichert als:", output_file)
print(df_daily.head())
