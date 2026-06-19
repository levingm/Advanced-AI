"""Forecast-Plot für eine Modell × Kovariaten-Kombination (Option 3)."""

import argparse

import common as tc
from treesearch_plot import plot_forecast

MODEL_LABELS = {
    "xgboost": "XGBoost",
    "linreg": "Lineare Regression",
    "ffn": "FFN",
    "tft": "TFT",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Forecast-Plot für Modell + Kovariaten")
    parser.add_argument(
        "--modell",
        required=True,
        choices=["xgboost", "linreg", "ffn", "tft"],
    )
    parser.add_argument(
        "--label",
        default="alle",
        help="Benchmark-Label (z.B. alle, nur_zinsen, single_DAX)",
    )
    parser.add_argument(
        "--kovariaten",
        nargs="*",
        default=None,
        help="Explizite Kovariaten (überschreibt --label)",
    )
    parser.add_argument("--simulationen", type=int, default=tc.SIMULATIONEN)
    parser.add_argument(
        "--output",
        default=None,
        help="PNG-Pfad (default: forecast_{modell}_{label}.png)",
    )
    parser.add_argument("--show", action="store_true", help="Plot anzeigen statt nur speichern")
    return parser.parse_args()


def main():
    args = parse_args()
    covariates = args.kovariaten if args.kovariaten else tc.covariates_for_label(args.label)
    label_slug = args.label if not args.kovariaten else "custom"
    out = args.output or f"forecast_{args.modell}_{label_slug}.png"

    plot_forecast(
        model_label=MODEL_LABELS[args.modell],
        model_name=args.modell,
        title_suffix=f"{MODEL_LABELS[args.modell]} / {label_slug}",
        covariates=covariates,
        simulations=args.simulationen,
        save_path=None if args.show else out,
    )
    if not args.show:
        print(f"Plot gespeichert: {out}")


if __name__ == "__main__":
    main()
