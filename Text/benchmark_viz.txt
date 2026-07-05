"""Visualisierungen für Kovariaten-Benchmark-Ergebnisse."""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
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


def save_top5_table(
    df: pd.DataFrame,
    path: str = "benchmark_top5.csv",
    rolling_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Top-5 Kombinationen gesamt nach MAE_Volumen.

    Punkt 2 (Konfidenzintervalle): Die reine Top-5-Liste aus einem einzigen
    Testfenster suggeriert eine Präzision, die statistisch nicht gedeckt ist
    (Kritik 5.1/6.2: "explorativ, nicht konfirmatorisch"). Falls `rolling_df`
    übergeben wird (Output von run_rolling_evaluation für dieselben
    Modell×Label-Kombinationen über mehrere Origins), wird zusätzlich eine
    95%-Bootstrap-Konfidenzintervall-Spalte ergänzt (CI_lower/CI_upper für
    MAE_Volumen), die die tatsächliche Streuung über die Rolling-Origins
    widerspiegelt. Ohne rolling_df bleibt die Tabelle explizit als
    "Einzelfenster, nicht abgesichert" markiert, statt unkommentiert
    Präzision zu suggerieren.

    Args:
        df: Einzelpunkt-Benchmark-Ergebnisse (ein Testfenster).
        path: CSV-Ausgabepfad.
        rolling_df: Optionaler Rolling-Origin-Output für dieselben
            Modell×Label-Kombinationen, zur CI-Schätzung.
    """
    cols = ["Modell", "Label", "MAE_Volumen", "RMSE_Volumen", "R2_Volumen", "Kovariaten"]
    top5 = df.nsmallest(5, "MAE_Volumen")[cols].copy()

    if rolling_df is not None and len(rolling_df) > 0:
        ci_lower, ci_upper, n_origins = [], [], []
        rng = np.random.default_rng(42)
        for _, row in top5.iterrows():
            sub = rolling_df[
                (rolling_df["Modell"] == row["Modell"]) & (rolling_df["Label"] == row["Label"])
            ]
            if len(sub) >= 2:
                vals = sub["MAE_Volumen"].to_numpy()
                # Einfaches Bootstrap-CI (95%) über die Rolling-Origin-Werte,
                # statt eine Normalverteilung zu unterstellen.
                boot_means = [
                    rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(2000)
                ]
                ci_lower.append(round(float(np.percentile(boot_means, 2.5)), 4))
                ci_upper.append(round(float(np.percentile(boot_means, 97.5)), 4))
                n_origins.append(len(sub))
            else:
                ci_lower.append(float("nan"))
                ci_upper.append(float("nan"))
                n_origins.append(len(sub))
        top5["CI95_lower"] = ci_lower
        top5["CI95_upper"] = ci_upper
        top5["n_Origins"] = n_origins
        caveat = (
            "Hinweis: CI95 basiert auf Bootstrap über die verfügbaren Rolling-Origins "
            "(siehe n_Origins). Bei wenigen Origins ist das Intervall entsprechend breit/unsicher."
        )
    else:
        top5["CI95_lower"] = float("nan")
        top5["CI95_upper"] = float("nan")
        top5["n_Origins"] = 0
        caveat = (
            "ACHTUNG: Kein rolling_df übergeben – diese Top-5-Liste basiert auf EINEM "
            "einzigen Testfenster ohne Unsicherheitsschätzung. Bei vielen getesteten "
            "Kovariaten-Kombinationen (siehe Kritik Punkt 5.1, p-Hacking-Risiko) ist "
            "das Ranking rein explorativ und NICHT als robuste/konfirmatorische "
            "Aussage über 'die besten Kovariaten' zu verstehen. Für belastbare "
            "Aussagen --rolling verwenden und rolling_df hier übergeben."
        )

    top5.to_csv(path, index=False, encoding="utf-8-sig")
    print("\n--- Top-5 Kombinationen (MAE_Volumen) ---")
    print(top5.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n{caveat}")
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


def save_all_visualizations(
    df: pd.DataFrame, prefix: str = "benchmark", rolling_df: pd.DataFrame | None = None
) -> None:
    """Erzeugt alle Einzel-Benchmark-Visualisierungen (2a–2d).

    Args:
        rolling_df: Optional – falls vorhanden, fließt es als CI-Quelle in
            save_top5_table ein (Punkt 2).
    """
    save_heatmap(df, f"{prefix}_heatmap.png")
    save_boxplot(df, f"{prefix}_boxplot.png")
    save_scatter(df, f"{prefix}_scatter_n_covariates.png")
    save_top5_table(df, f"{prefix}_top5.csv", rolling_df=rolling_df)