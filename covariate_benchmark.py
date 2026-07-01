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
    save_rolling_summary,
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
    # Punkt 1: Lags für Kovariaten
    parser.add_argument(
        "--lags",
        action="store_true",
        help=(
            "Verzögerte Kovariaten-Features (t-1, t-5, t-20 Handelstage) zusätzlich "
            "zu den Originalwerten verwenden (siehe common.LAG_STEPS). Nur für "
            "Sequenzmodelle wirksam (linreg/ffn/xgboost)."
        ),
    )
    # Punkt 3: Einheitliches Tuning-Budget
    parser.add_argument(
        "--tune",
        action="store_true",
        help=(
            "GridSearchCV mit einheitlichem Parameter-Budget (TUNING_BUDGET Kombinationen) "
            "für ALLE Sequenzmodelle aktivieren (linreg/ffn/xgboost), statt ungetunter "
            "Defaults bzw. nur teilweisem Tuning."
        ),
    )
    # Punkt 5: Zinsregime als Split-Kriterium (nur sinnvoll mit --rolling)
    parser.add_argument(
        "--regime",
        nargs="?",
        const="€STR",
        default=None,
        help=(
            "Zinsregime pro Rolling-Origin annotieren (Spalte 'Regime'), basierend auf "
            "dem Wert der angegebenen Spalte (Standard: €STR) am jeweiligen Origin-Datum. "
            "Nur mit --rolling wirksam. Siehe common.assign_interest_regime/REGIME_BINS."
        ),
    )
    # Punkt 6: TFT-Baseline (alte, unterkonfigurierte Variante ohne known_reals)
    parser.add_argument(
        "--tft-baseline",
        action="store_true",
        help=(
            "Falls TFT in --modelle enthalten ist: die ursprüngliche Baseline-Variante "
            "ohne time_varying_known_reals verwenden (known_reals=[]), statt der "
            "Standard-Variante mit echten Kalender-known_reals. Nützlich, um den Effekt "
            "der known_reals im Paper explizit zu zeigen."
        ),
    )
    # Punkt 4: Diebold-Mariano-Test zwischen den besten Modellen (nur mit --rolling)
    parser.add_argument(
        "--dm-test",
        action="store_true",
        help=(
            "Nach --rolling: Diebold-Mariano-Test zwischen den paarweise besten "
            "Modell×Label-Kombinationen durchführen (Signifikanz der Rangfolge prüfen)."
        ),
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

    # Für naive Modelle reicht ein Lauf (sie verwenden keine Kovariaten im Benchmark-Design)
    naive_models = set(["naive_rw", "naive_hold"])
    tasks = []
    for m in args.modelle:
        if m in naive_models:
            # Ein Lauf ohne Kovariaten (Label 'naive' unterscheidet sie)
            tasks.append((m, [], "naive"))
        else:
            for cols, label in combos:
                tasks.append((m, cols, label))

    print(
        f"\nEinzelpunkt-Benchmark: {len(args.modelle)} Modell(e) × {len(combos)} "
        f"Kovariaten-Sets = {len(tasks)} Läufe"
    )
    print(
        f"Modus: {args.kovariaten_modus} | Simulationen: {args.simulationen} | "
        f"Lags: {args.lags} | Tuning: {args.tune}"
    )
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
                    use_lags=args.lags,
                    do_tune=args.tune,
                    use_calendar_known_reals=not args.tft_baseline,
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

    naive_models = set(["naive_rw", "naive_hold"])
    tasks = []
    for m in args.modelle:
        if m in naive_models:
            tasks.append((m, [], "naive"))
        else:
            for cols, label in combos:
                tasks.append((m, cols, label))

    print(
        f"\nRolling-Benchmark: {len(args.modelle)} Modell(e) × {len(combos)} Sets "
        f"× {len(origins)} Origins = {len(tasks) * len(origins)} Läufe"
    )
    print(f"Origins ({args.rolling_freq}): {origins[0]} … {origins[-1]}")
    if args.regime:
        print(f"Zinsregime-Annotation aktiv (Spalte: {args.regime!r})")

    all_rows: list[pd.DataFrame] = []
    for model_name, covariates, label in tqdm(tasks, desc="Rolling-Benchmark"):
        try:
            df_roll = tc.run_rolling_evaluation(
                model_name=model_name,
                covariates=covariates,
                covariate_label=label,
                simulations=args.simulationen,
                origins=origins,
                use_lags=args.lags,
                do_tune=args.tune,
                regime_col=args.regime,
            )
            all_rows.append(df_roll)
        except Exception as exc:
            print(f"\n[Fehler] {model_name} / {label}: {exc}")

    if not all_rows:
        raise SystemExit("Keine Rolling-Ergebnisse.")

    df_full = pd.concat(all_rows, ignore_index=True)

    # Die rohen Vorhersage-/Wahrwerte-Arrays (diff_pred, volume_pred, true_diff,
    # true_volume) sind für Diebold-Mariano (--dm-test) bzw. Top-5-CIs (Punkt 2)
    # gedacht, aber nicht CSV-tauglich (NumPy-Arrays je Zelle). Für den
    # CSV-Export entfernen wir sie; df_full (mit Arrays) bleibt im Speicher für
    # nachgelagerte Analysen (z.B. --dm-test) erhalten.
    array_cols = [c for c in ["diff_pred", "volume_pred", "true_diff", "true_volume"] if c in df_full.columns]
    df_csv = df_full.drop(columns=array_cols)
    df_csv.to_csv(args.rolling_output, index=False, encoding="utf-8-sig")
    print(f"\nRolling-CSV gespeichert: {args.rolling_output} (ohne Rohdaten-Arrays)")

    print("\n--- Aggregierte Rolling-Metriken (Mittelwert ± Std) ---")
    agg = tc.aggregate_rolling_results(df_csv)
    print(agg.to_string())
    return df_full


def run_dm_test_report(df_roll: pd.DataFrame) -> None:
    """Punkt 4: Diebold-Mariano-Test zwischen den paarweise besten
    Modell×Label-Kombinationen (nach mittlerem MAE_Volumen über die Origins).

    Reduziert auf die Top-3 Kombinationen, da paarweise Tests bei vielen
    Kombinationen schnell zu multiplem Testen führen (dasselbe Problem wie
    Kritik 5.1, nur auf der Signifikanztest-Ebene) – hier explizit auf
    wenige, vorab inhaltlich interessante Vergleiche begrenzt.
    """
    ranking = (
        df_roll.groupby(["Modell", "Label"])["MAE_Volumen"]
        .mean()
        .sort_values()
        .reset_index()
    )
    if len(ranking) < 2:
        print("\n[DM-Test] Zu wenige Modell×Label-Kombinationen für einen Vergleich.")
        return

    top_n = ranking.head(3)
    print("\n--- Diebold-Mariano-Test: paarweiser Vergleich der Top-Kombinationen ---")
    print(
        "Hinweis: H0 = beide Modelle haben gleiche Prognosegüte. p < 0.05 spricht "
        "gegen H0 (signifikanter Unterschied). Mehrfachvergleiche -> Bonferroni-"
        "Korrektur in Betracht ziehen, falls mehr als ein Paar getestet wird.\n"
    )
    rows = top_n.to_dict("records")
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            a, b = rows[i], rows[j]
            sub_a = df_roll[(df_roll["Modell"] == a["Modell"]) & (df_roll["Label"] == a["Label"])]
            sub_b = df_roll[(df_roll["Modell"] == b["Modell"]) & (df_roll["Label"] == b["Label"])]
            try:
                result = tc.compare_models_dm(sub_a, sub_b, target="volume", loss="mse")
                sig = "***" if result["p_value"] < 0.05 else "(n.s.)"
                print(
                    f"{a['Modell']}/{a['Label']} (MAE={a['MAE_Volumen']:.4f}) vs. "
                    f"{b['Modell']}/{b['Label']} (MAE={b['MAE_Volumen']:.4f}): "
                    f"DM={result['dm_stat']:.3f}, p={result['p_value']:.4f} {sig}"
                )
            except Exception as exc:
                print(f"[DM-Test] Fehler bei {a['Modell']}/{a['Label']} vs. {b['Modell']}/{b['Label']}: {exc}")


def main() -> None:
    args = parse_args()

    # --- Einzelpunkt-Benchmark ---
    df = run_single_benchmark(args)

    if args.plot:
        save_benchmark_plot(df, args.plot_file)

    # --- Rolling-Origin-Evaluation (optional) ---
    # Vor den Visualisierungen ausgeführt, damit save_top5_table (Punkt 2)
    # die Rolling-Ergebnisse für Konfidenzintervalle nutzen kann.
    df_roll_full: pd.DataFrame | None = None
    if args.rolling:
        df_roll_full = run_rolling_benchmark(args)
        prefix_roll = args.rolling_output.replace(".csv", "")
        save_rolling_plot(df_roll_full, f"{prefix_roll}_timeline.png")
        # FIX: Rolling-Summary (Boxplots der Rolling-Verteilung) speichern
        save_rolling_summary(df_roll_full, f"{prefix_roll}_summary.png")

        if args.dm_test:
            run_dm_test_report(df_roll_full)

    if args.viz:
        prefix = args.output.replace(".csv", "")
        if args.viz == "all":
            save_all_visualizations(df, prefix=prefix, rolling_df=df_roll_full)
        elif args.viz == "heatmap":
            save_heatmap(df, f"{prefix}_heatmap.png")
        elif args.viz == "box":
            save_boxplot(df, f"{prefix}_boxplot.png")
        elif args.viz == "scatter":
            save_scatter(df, f"{prefix}_scatter_n_covariates.png")
        elif args.viz == "top5":
            save_top5_table(df, f"{prefix}_top5.csv", rolling_df=df_roll_full)


if __name__ == "__main__":
    main()