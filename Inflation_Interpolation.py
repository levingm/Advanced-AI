import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator

# Datei einlesen
file_path = "Inflation monthly.xlsx"
df = pd.read_excel(file_path, engine="openpyxl")

# Spalten umbenennen (falls nötig anpassen)
df.columns = ["date", "value"]

# Datumsformat konvertieren
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# "-" oder ungültige Werte -> NaN
df["value"] = pd.to_numeric(df["value"], errors="coerce")

# NaNs entfernen (Interpolation braucht saubere Stützstellen)
df = df.dropna(subset=["date", "value"]).sort_values("date")

# Zeit in numerische Form bringen (Tage seit Start)
x = (df["date"] - df["date"].min()).dt.days
y = df["value"]

# PCHIP Interpolator erstellen
pchip = PchipInterpolator(x, y)

# Täglichen Datumsbereich erzeugen
daily_dates = pd.date_range(start=df["date"].min(),
                            end=df["date"].max(),
                            freq="D")

# Tägliche x-Werte berechnen
x_daily = (daily_dates - df["date"].min()).days

# Interpolation durchführen
y_daily = pchip(x_daily)

# Ergebnis-DataFrame
df_daily = pd.DataFrame({
    "date": daily_dates,
    "interpolated_value": y_daily
})

# Neue Excel speichern
output_file = "Inflation_daily_pchip.xlsx"
df_daily.to_excel(output_file, index=False, engine="openpyxl")

print(f"Fertig! Datei gespeichert unter: {output_file}")
