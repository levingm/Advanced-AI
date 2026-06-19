"""Visualisierungen für Kovariaten-Benchmark-Ergebnisse (2a–2d)."""

from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def save_heatmap(df: pd.DataFrame, path: str = "benchmark_heatmap.png"):
    """2a: MAE_Volumen nach Modell × Kovariaten-Label."""
    pivot = df.pivot_table(
        index="Label", columns="Modell", values="MAE_Volumen", aggfunc="mean"
    )
    pivot = pivot.sort_index()

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 1.5), max(6, len(pivot) * 0.4)))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd_r", ax=ax)
    ax.set_title("MAE Volumen: Modell × Kovariaten-Set")
    ax.set_ylabel("Kovariaten-Label")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Heatmap gespeichert: {path}")


def save_boxplot(df: pd.DataFrame, path: str = "benchmark_boxplot.png"):
    """2b: Fehlerverteilung pro Modell über alle Kovariaten-Sets."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x="Modell", y="MAE_Volumen", ax=ax)
    ax.set_title("MAE Volumen – Verteilung über Kovariaten-Sets")
    ax.set_ylabel("MAE Volumen [Mrd. €]")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Boxplot gespeichert: {path}")


def save_scatter(df: pd.DataFrame, path: str = "benchmark_scatter_n_covariates.png"):
    """2c: MAE_Volumen vs. Anzahl Kovariaten."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        data=df, x="n_Kovariaten", y="MAE_Volumen", hue="Modell", ax=ax, s=80
    )
    ax.set_title("Prognosegüte vs. Anzahl Kovariaten")
    ax.set_ylabel("MAE Volumen [Mrd. €]")
    ax.set_xlabel("Anzahl Kovariaten")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Scatter gespeichert: {path}")


def save_top5_table(df: pd.DataFrame, path: str = "benchmark_top5.csv") -> pd.DataFrame:
    """2d: Top-5 Kombinationen gesamt nach MAE_Volumen."""
    cols = ["Modell", "Label", "MAE_Volumen", "RMSE_Volumen", "R2_Volumen", "Kovariaten"]
    top5 = df.nsmallest(5, "MAE_Volumen")[cols]
    top5.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\n--- Top-5 Kombinationen (MAE_Volumen) ---")
    print(top5.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nTop-5 CSV: {path}")
    return top5


def save_all_visualizations(df: pd.DataFrame, prefix: str = "benchmark"):
    """Erzeugt alle Benchmark-Visualisierungen."""
    save_heatmap(df, f"{prefix}_heatmap.png")
    save_boxplot(df, f"{prefix}_boxplot.png")
    save_scatter(df, f"{prefix}_scatter_n_covariates.png")
    save_top5_table(df, f"{prefix}_top5.csv")
