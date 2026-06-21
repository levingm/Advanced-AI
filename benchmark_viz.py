"""Visualisierungen für Kovariaten-Benchmark-Ergebnisse."""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns


def save_heatmap(df: pd.DataFrame, path: str = "benchmark_heatmap.png") -> None:
    """MAE_Volumen nach Modell × Kovariaten-Label (Heatmap)."""
    pivot = df.pivot_table(
        index="Label", columns="Modell", values="MAE_Volumen", aggfunc="mean"
    ).sort_index()

    fig, ax = plt.subplots(
        figsize=(max(8, len(pivot.columns) * 1.8), max(6, len(pivot) * 0.45))
    )
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd_r", ax=ax, linewidths=0.4)
    ax.set_title("MAE Volumen [Mrd. €]: Modell × Kovariaten-Set", fontsize=13)
    ax.set_ylabel("Kovariaten-Label")
    ax.set_xlabel("Modell")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Heatmap gespeichert: {path}")


def save_boxplot(df: pd.DataFrame, path: str = "benchmark_boxplot.png") -> None:
    """Fehlerverteilung pro Modell über alle Kovariaten-Sets (Boxplot)."""
    fig, ax = plt.subplots(figsize=(9, 5))
    order = df.groupby("Modell")["MAE_Volumen"].median().sort_values().index
    sns.boxplot(data=df, x="Modell", y="MAE_Volumen", order=order, ax=ax)
    ax.set_title("MAE Volumen – Verteilung über Kovariaten-Sets")
    ax.set_ylabel("MAE Volumen [Mrd. €]")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Boxplot gespeichert: {path}")


def save_scatter(
    df: pd.DataFrame, path: str = "benchmark_scatter_n_covariates.png"
) -> None:
    """MAE_Volumen vs. Anzahl Kovariaten (Scatter)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df, x="n_Kovariaten", y="MAE_Volumen", hue="Modell", ax=ax, s=80)
    ax.set_title("Prognosegüte vs. Anzahl Kovariaten")
    ax.set_ylabel("MAE Volumen [Mrd. €]")
    ax.set_xlabel("Anzahl Kovariaten")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Scatter gespeichert: {path}")


def save_top5_table(df: pd.DataFrame, path: str = "benchmark_top5.csv") -> pd.DataFrame:
    """Top-5 Kombinationen gesamt nach MAE_Volumen."""
    cols = ["Modell", "Label", "MAE_Volumen", "RMSE_Volumen", "R2_Volumen", "Kovariaten"]
    top5 = df.nsmallest(5, "MAE_Volumen")[cols]
    top5.to_csv(path, index=False, encoding="utf-8-sig")
    print("\n--- Top-5 Kombinationen (MAE_Volumen) ---")
    print(top5.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"Top-5 CSV: {path}")
    return top5


# FIX 1: Visualisierung für Rolling-Origin-Ergebnisse
def save_rolling_plot(
    rolling_df: pd.DataFrame,
    path: str = "benchmark_rolling_timeline.png",
    metric: str = "MAE_Volumen",
) -> None:
    """Zeitverlauf der Prognosegüte über alle Rolling-Origins pro Modell.

    Zeigt, ob ein Modell konsistent besser ist oder nur in bestimmten Perioden.
    Naive-Benchmarks werden als gestrichelte Referenzlinien eingezeichnet.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    naive_models = {"naive_rw", "naive_hold"}
    other_models = sorted(
        rolling_df["Modell"].unique(),
        key=lambda m: rolling_df.loc[rolling_df["Modell"] == m, metric].mean()
    )

    for model in other_models:
        sub = rolling_df[rolling_df["Modell"] == model].sort_values("training_end")
        if model in naive_models:
            ax.plot(
                sub["training_end"], sub[metric],
                linestyle="--", linewidth=1.2, alpha=0.7, label=model,
            )
        else:
            ax.plot(
                sub["training_end"], sub[metric],
                marker="o", markersize=3, linewidth=1.5, label=model,
            )

    ax.set_title(f"Rolling-Origin-Evaluation: {metric} über Zeit", fontsize=13)
    ax.set_ylabel(f"{metric} [Mrd. €]" if "Volumen" in metric else metric)
    ax.set_xlabel("Trainings-Ende (Rolling Origin)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Rolling-Plot gespeichert: {path}")


def save_rolling_summary(
    rolling_df: pd.DataFrame,
    path: str = "benchmark_rolling_summary.png",
) -> None:
    """Boxplot der Rolling-MAE pro Modell (zeigt Konsistenz über Perioden)."""
    fig, ax = plt.subplots(figsize=(9, 5))
    order = (
        rolling_df.groupby("Modell")["MAE_Volumen"]
        .median()
        .sort_values()
        .index
    )
    sns.boxplot(data=rolling_df, x="Modell", y="MAE_Volumen", order=order, ax=ax)
    ax.set_title("Rolling-Origin: MAE Volumen – Verteilung über Origins pro Modell")
    ax.set_ylabel("MAE Volumen [Mrd. €]")
    ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Rolling-Summary-Plot gespeichert: {path}")


def save_all_visualizations(df: pd.DataFrame, prefix: str = "benchmark") -> None:
    """Erzeugt alle Einzel-Benchmark-Visualisierungen (2a–2d)."""
    save_heatmap(df, f"{prefix}_heatmap.png")
    save_boxplot(df, f"{prefix}_boxplot.png")
    save_scatter(df, f"{prefix}_scatter_n_covariates.png")
    save_top5_table(df, f"{prefix}_top5.csv")