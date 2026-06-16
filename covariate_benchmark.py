"""
Kovariaten-Benchmark: vergleicht Prognosegüte über Modell × Kovariaten-Kombination.

Ausgabe:
  - Tabelle in der Konsole (sortiert nach MAE_Volumen)
  - CSV: covariate_benchmark_results.csv
  - optional: Balkendiagramm der besten Kombinationen pro Modell
"""

import argparse

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

import treesearch_common as tc


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark: Modellgüte nach Kovariaten-Kombination"
    )
    parser.add_argument(
        "--modelle",
        nargs="+",
        default=["linreg", "xgboost"],
        choices=["xgboost", "linreg", "ffn", "tft"],
        help="Zu testende Modelle (tft ist langsam)",
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
        help="CSV-Ausgabedatei",
    )
    parser.add_argument("--plot", action="store_true", help="Balkendiagramm speichern")
    parser.add_argument(
        "--plot-file",
        default="covariate_benchmark_plot.png",
        help="PNG für Benchmark-Plot",
    )
    return parser.parse_args()


def print_summary_table(df: pd.DataFrame):
    print("\n" + "=" * 100)
    print("KOVARIATEN-BENCHMARK (sortiert nach MAE_Volumen)")
    print("=" * 100)
    cols = [
        "Modell",
        "Label",
        "n_Kovariaten",
        "MAE_Diff",
        "RMSE_Diff",
        "R2_Diff",
        "MAE_Volumen",
        "RMSE_Volumen",
        "R2_Volumen",
        "Kovariaten",
    ]
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n--- Beste Kombination pro Modell (nach MAE_Volumen) ---")
    best = df.loc[df.groupby("Modell")["MAE_Volumen"].idxmin()]
    print(best[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def save_benchmark_plot(df: pd.DataFrame, path: str):
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
  plt.tight_layout()
  for bar, val in zip(bars, best["MAE_Volumen"]):
      ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.3f}", ha="center", va="bottom", fontsize=8)
  plt.savefig(path, dpi=150)
  plt.close()
  print(f"\nPlot gespeichert: {path}")


def main():
    args = parse_args()
    combos = tc.generate_covariate_combinations(args.kovariaten_modus)
    total = len(combos) * len(args.modelle)

    print(f"Benchmark: {len(args.modelle)} Modell(e) × {len(combos)} Kovariaten-Sets = {total} Läufe")
    print(f"Modus: {args.kovariaten_modus} | Simulationen: {args.simulationen}")

    results = []
    tasks = [(m, cols, label) for m in args.modelle for cols, label in combos]

    for model_name, covariates, label in tqdm(tasks, desc="Benchmark"):
        try:
            result = tc.run_single_evaluation(
                model_name=model_name,
                covariates=covariates,
                covariate_label=label,
                simulations=args.simulationen,
            )
            results.append(result)
        except Exception as exc:
            print(f"\n[Fehler] {model_name} / {label}: {exc}")

    if not results:
        raise SystemExit("Keine Ergebnisse – prüfe Daten und Parameter.")

    df = tc.results_to_dataframe(results)
    df.to_csv(args.output, index=False, encoding="utf-8-sig")
    print_summary_table(df)
    print(f"\nCSV gespeichert: {args.output}")

    if args.plot:
        save_benchmark_plot(df, args.plot_file)


if __name__ == "__main__":
    main()
