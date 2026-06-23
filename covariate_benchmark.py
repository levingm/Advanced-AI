"""
Kovariaten-Benchmark: vergleicht Prognosegüte über Modell × Kovariaten-Kombination.

Ausgabe:
  - Tabelle in der Konsole (sortiert nach MAE_Volumen)
  - CSV: covariate_benchmark_results.csv
  - Optional: Visualisierungen (--viz) und Rolling-Origin-Evaluation (--rolling)
"""

from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

import common as tc
from benchmark_viz import (
    save_all_visualizations,
    save_boxplot,
    save_heatmap,
    save_rolling_plot,
    save_scatter,
    save_top5_table,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark: Modellgüte nach Kovariaten-Kombination"
    )
    parser.add_argument(
        "--modelle",
        nargs="+",
        # FIX 2: Naive Benchmarks als wählbare Modelle
        default=["naive_rw", "naive_hold", "linreg", "xgboost"],
        choices=tc.ALL_MODELS,
        help="Zu testende Modelle (tft ist langsam; naive_rw/naive_hold = Benchmarks)",
    )
    parser.add_argument(
        "--kovariaten-modus",
        default="presets",
        choices=["presets", "singletons", "pairs", "all", "full"],
        help="presets=sinnvolle Sets; all=presets+pairs+singletons; full=alle 2^6-1 Kombinationen",
    )
    parser.add_argument(
        "--simulationen",
        type=int,
        default=1,
        help="Ensemble-Läufe pro Konfiguration (1 für schnellen Benchmark)",
    )
    parser.add_argument(
        "--output",
        default="covariate_benchmark_results.csv",
        help="CSV-Ausgabedatei (Einzelpunkt-Evaluation)",
    )
    parser.add_argument("--plot", action="store_true", help="Balkendiagramm speichern")
    parser.add_argument(
        "--plot-file",
        default="covariate_benchmark_plot.png",
        help="PNG für Benchmark-Balkendiagramm",
    )
    parser.add_argument(
        "--viz",
        nargs="?",
        const="all",
        default=None,
        choices=["all", "heatmap", "box", "scatter", "top5"],
        help="Analyse-Plots: all = Heatmap + Box + Scatter + Top5",
    )
    # FIX 1: Rolling-Origin-Evaluation als CLI-Option
    parser.add_argument(
        "--rolling",
        action="store_true",
        help="Rollierende Out-of-Sample-Evaluation über mehrere Origins",
    )
    parser.add_argument(
        "--rolling-freq",
        default="MS",
        help="Frequenz der Rolling-Origins (Pandas-Alias, z.B. 'MS'=monatlich, 'QS'=quartalsweise)",
    )
    parser.add_argument(
        "--rolling-output",
        default="covariate_benchmark_rolling.csv",
        help="CSV-Ausgabe für Rolling-Ergebnisse",
    )
    return parser.parse_args()


def print_summary_table(df: pd.DataFrame) -> None:
    print("\n" + "=" * 110)
    print("KOVARIATEN-BENCHMARK (sortiert nach MAE_Volumen)")
    print("=" * 110)
    cols = [
        "Modell", "Label", "n_Kovariaten",
        "MAE_Diff", "RMSE_Diff", "R2_Diff",
        "MAE_Volumen", "RMSE_Volumen", "R2_Volumen",
        "Kovariaten",
    ]
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n--- Beste Kombination pro Modell (nach MAE_Volumen) ---")
    best = df.loc[df.groupby("Modell")["MAE_Volumen"].idxmin()]
    print(best[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def save_benchmark_plot(df: pd.DataFrame, path: str) -> None:
    best = df.loc[df.groupby("Modell")["MAE_Volumen"].idxmin()].copy()
    best = best.sort_values("MAE_Volumen")

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(
        best["Modell"] + "\n" + best["Label"],
        best["MAE_Volumen"],
        color="steelblue",
        alpha=0.85,
    )
    ax.set_ylabel("MAE Volumen [Mrd. €]")
    ax.set_title("Beste Kovariaten-Kombination pro Modell")
    ax.tick_params(axis="x", rotation=25)
    for bar, val in zip(bars, best["MAE_Volumen"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=8,
        )
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Plot gespeichert: {path}")


def run_single_benchmark(args: argparse.Namespace) -> pd.DataFrame:
    """Einzelpunkt-Evaluation (klassischer Benchmark)."""
    combos = tc.generate_covariate_combinations(args.kovariaten_modus)
    tasks = [(m, cols, label) for m in args.modelle for cols, label in combos]
    print(
        f"\nEinzelpunkt-Benchmark: {len(args.modelle)} Modell(e) × {len(combos)} "
        f"Kovariaten-Sets = {len(tasks)} Läufe"
    )
    print(f"Modus: {args.kovariaten_modus} | Simulationen: {args.simulationen}")
    print(
        "\nHinweis: Ergebnisse basieren auf einem einzelnen Testfenster. "
        "Für robuste Aussagen --rolling verwenden.\n"
    )

    results = []
    for model_name, covariates, label in tqdm(tasks, desc="Benchmark"):
        try:
            results.append(
                tc.run_single_evaluation(
                    model_name=model_name,
                    covariates=covariates,
                    covariate_label=label,
                    simulations=args.simulationen,
                )
            )
        except Exception as exc:
            print(f"\n[Fehler] {model_name} / {label}: {exc}")

    if not results:
        raise SystemExit("Keine Ergebnisse – prüfe Daten und Parameter.")

    df = tc.results_to_dataframe(results)
    df.to_csv(args.output, index=False, encoding="utf-8-sig")
    print_summary_table(df)
    print(f"\nCSV gespeichert: {args.output}")
    return df


def run_rolling_benchmark(args: argparse.Namespace) -> pd.DataFrame:
    """Rollierende Evaluation über alle Modelle × Kovariaten-Kombinationen."""
    combos = tc.generate_covariate_combinations(args.kovariaten_modus)
    origins = tc.generate_rolling_origins(freq=args.rolling_freq)
    tasks = [(m, cols, label) for m in args.modelle for cols, label in combos]
    print(
        f"\nRolling-Benchmark: {len(args.modelle)} Modell(e) × {len(combos)} Sets "
        f"× {len(origins)} Origins = {len(tasks) * len(origins)} Läufe"
    )
    print(f"Origins ({args.rolling_freq}): {origins[0]} … {origins[-1]}")

    all_rows: list[pd.DataFrame] = []
    for model_name, covariates, label in tqdm(tasks, desc="Rolling-Benchmark"):
        try:
            df_roll = tc.run_rolling_evaluation(
                model_name=model_name,
                covariates=covariates,
                covariate_label=label,
                simulations=args.simulationen,
                origins=origins,
            )
            all_rows.append(df_roll)
        except Exception as exc:
            print(f"\n[Fehler] {model_name} / {label}: {exc}")

    if not all_rows:
        raise SystemExit("Keine Rolling-Ergebnisse.")

    df_full = pd.concat(all_rows, ignore_index=True)
    df_full.to_csv(args.rolling_output, index=False, encoding="utf-8-sig")
    print(f"\nRolling-CSV gespeichert: {args.rolling_output}")

    print("\n--- Aggregierte Rolling-Metriken (Mittelwert ± Std) ---")
    agg = tc.aggregate_rolling_results(df_full)
    print(agg.to_string())
    return df_full


def main() -> None:
    args = parse_args()

    # --- Einzelpunkt-Benchmark ---
    df = run_single_benchmark(args)

    if args.plot:
        save_benchmark_plot(df, args.plot_file)

    if args.viz:
        prefix = args.output.replace(".csv", "")
        if args.viz == "all":
            save_all_visualizations(df, prefix=prefix)
        elif args.viz == "heatmap":
            save_heatmap(df, f"{prefix}_heatmap.png")
        elif args.viz == "box":
            save_boxplot(df, f"{prefix}_boxplot.png")
        elif args.viz == "scatter":
            save_scatter(df, f"{prefix}_scatter_n_covariates.png")
        elif args.viz == "top5":
            save_top5_table(df, f"{prefix}_top5.csv")

    # --- Rolling-Origin-Evaluation (optional) ---
    if args.rolling:
        df_roll = run_rolling_benchmark(args)
        prefix_roll = args.rolling_output.replace(".csv", "")
        save_rolling_plot(df_roll, f"{prefix_roll}_timeline.png")


if __name__ == "__main__":
    main()