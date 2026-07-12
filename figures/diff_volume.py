import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_excel("Data all variables.xlsx")
df["Datum"] = pd.to_datetime(df["Datum"])
df = df.sort_values("Datum").set_index("Datum")

# Compute first difference of Einlagevolumen
diff_volume = df["Einlagevolumen"].diff().dropna()

# Plot
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(diff_volume.index, diff_volume.values, 
        linewidth=0.8, color="steelblue")
ax.axhline(y=0, color="red", linestyle="--", 
           linewidth=0.6, label="Zero line")
ax.set_ylabel(r"$\Delta\,\mathrm{Einlagevolumen}_t$", fontsize=10)
ax.set_xlabel("Date")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("figures/diff_volume.png", dpi=300)
plt.show()