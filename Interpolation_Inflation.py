import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator

# -------------------------------
# 1. Datei roh einlesen
# -------------------------------
file_path = "Inflation monthly.xlsx"
df_raw = pd.read_excel(file_path, header=None)

# -------------------------------
# 2. Datenblock erkennen
# -------------------------------
# Finde erste Zeile mit Jahr (numeric)
start_idx = None
for i, row in df_raw.iterrows():
    if pd.to_numeric(row[0], errors="coerce") > 1900:
        start_idx = i
        break

if start_idx is None:
    raise ValueError("Konnte Start der Tabelle nicht finden")

# -------------------------------
# 3. Relevante Spalten extrahieren
# -------------------------------
# Struktur laut Datei:
# col0 = Jahr
# col1 = Monat (Text)
# col2 = YoY Inflation
df = df_raw.iloc[start_idx:, [0, 1, 2]].copy()

df.columns = ["year", "month", "inflation"]

# -------------------------------
# 4. Daten bereinigen
# -------------------------------
# "-" zu NaN
df["inflation"] = df["inflation"].replace("-", np.nan)

# Monat Mapping
month_map = {
    "Januar": 1, "Februar": 2, "März": 3, "April": 4,
    "Mai": 5, "Juni": 6, "Juli": 7, "August": 8,
    "September": 9, "Oktober": 10, "November": 11, "Dezember": 12
}

df["month_num"] = df["month"].map(month_map)

# Datum bauen
df["date"] = pd.to_datetime(
    dict(year=df["year"], month=df["month_num"], day=1),
    errors="coerce"
)

# numerische Werte
df["inflation"] = pd.to_numeric(df["inflation"], errors="coerce")

# Aufräumen
df = df.dropna(subset=["date", "inflation"])
df = df.sort_values("date")

# -------------------------------
# 5. PCHIP Interpolation
# -------------------------------
x = df["date"].map(pd.Timestamp.toordinal).values
y = df["inflation"].values

pchip = PchipInterpolator(x, y, extrapolate=False)

# -------------------------------
# 6. tägliche Werte erzeugen
# -------------------------------
daily_dates = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
x_daily = daily_dates.map(pd.Timestamp.toordinal).values

y_daily = pchip(x_daily)

# -------------------------------
# 7. Ergebnis
# -------------------------------
df_daily = pd.DataFrame({
    "date": daily_dates,
    "inflation_yoy": y_daily
})

# -------------------------------
# 8. Export
# -------------------------------
output_file = "inflation_daily_pchip.xlsx"
df_daily.to_excel(output_file, index=False)

print("✅ Fertig:", output_file)
print(df_daily.head())
