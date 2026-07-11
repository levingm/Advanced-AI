import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel("Data all variables.xlsx")
df["Datum"] = pd.to_datetime(df["Datum"])
df = df.sort_values("Datum").set_index("Datum")

preestr_start = pd.Timestamp("2017-03-15")
estr_start = pd.Timestamp("2019-10-01")

fig, axes = plt.subplots(len(df.columns), 1, figsize=(12, 14), sharex=True)

for ax, col in zip(axes, df.columns):
    if col == "€STR":
        # Before Pre-€STR: reconstructed
        ax.plot(
            df.index[df.index < preestr_start],
            df.loc[df.index < preestr_start, col],
            color="lightgrey",
            linewidth=0.8,
            label="Reconstructed"
        )
        # Pre-€STR period
        ax.plot(
            df.index[(df.index >= preestr_start) & (df.index < estr_start)],
            df.loc[(df.index >= preestr_start) & (df.index < estr_start), col],
            color="orange",
            linewidth=0.8,
            label="Pre-€STR"
        )
        # Official €STR
        ax.plot(
            df.index[df.index >= estr_start],
            df.loc[df.index >= estr_start, col],
            color="steelblue",
            linewidth=0.8,
            label="€STR"
        )
        ax.legend(fontsize=6, loc="upper left")
    else:
        ax.plot(df.index, df[col], linewidth=0.8, color="steelblue")

    ax.set_ylabel(col, fontsize=8)

axes[-1].set_xlabel("Date")
plt.tight_layout()
plt.savefig("figures/overview.png", dpi=300)
plt.show()