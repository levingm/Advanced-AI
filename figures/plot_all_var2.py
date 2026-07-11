from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel("Data all variables.xlsx")
df["Datum"] = pd.to_datetime(df["Datum"])
df = df.sort_values("Datum").set_index("Datum")

# Normalize 0-1
df_norm = (df - df.min()) / (df.max() - df.min())

df_norm.plot(figsize=(14, 6), linewidth=0.8, alpha=0.8)
plt.title("Normalized variables (0-1 scaled, exploratory only)")
plt.tight_layout()
plt.savefig("overview_normalized.png", dpi=150)
plt.show()