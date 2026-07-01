"""Gemeinsamer Plot für alle Modell-Skripte."""

import matplotlib.pyplot as plt
import pandas as pd

import common as tc


def plot_forecast(
    model_label: str,
    model_name: str,
    title_suffix: str,
    covariates=None,
    simulations: int | None = None,
    save_path: str | None = None,
    use_lags: bool = False,
    do_tune: bool = False,
    use_calendar_known_reals: bool = True,
):
    covariates = covariates or tc.ALL_COVARIATES
    sims = simulations if simulations is not None else tc.SIMULATIONEN
    mittelwert, ki90, ki98, _, df = tc.run_ensemble_volume_paths(
        model_name,
        covariates,
        simulations=sims,
        use_lags=use_lags,
        do_tune=do_tune,
        use_calendar_known_reals=use_calendar_known_reals,
    )


    input_anfang = tc._next_date_in_index(df, tc.TRAINING_ENDE)

    real_values = df.loc[input_anfang:].iloc[: tc.PROGNOSEHORIZONT][tc.TARGET_COL].values
    historie = df.loc[tc.HISTORIE_ANFANG : tc.HISTORIE_ENDE, tc.TARGET_COL]
    x_zukunft = pd.date_range(input_anfang, periods=tc.PROGNOSEHORIZONT, freq="D")

    fig, axes = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={"height_ratios": [3, 1]})

    axes[0].plot(historie.index, historie.values, color="blue", label="Historische Daten (Training)")
    axes[0].plot(
        x_zukunft,
        real_values[: len(x_zukunft)],
        "--",
        color="black",
        label="Tatsächliche Reale Werte (Test)",
    )
    axes[0].plot(
        x_zukunft,
        mittelwert,
        color="red",
        label=f"{model_label} Ensemble Mittelwert (Diff-anchored)",
        linewidth=2,
    )
    axes[0].fill_between(
        x_zukunft, ki90[0], ki90[1], color="red", alpha=0.2, label="90% Konfidenzintervall"
    )
    axes[0].fill_between(
        x_zukunft, ki98[0], ki98[1], color="red", alpha=0.1, label="98% Konfidenzintervall"
    )
    axes[0].set_title(
        f"Treesearch Forecast: Level via Predicted Differences ({title_suffix})",
        fontsize=14,
    )
    axes[0].set_ylabel("Einlagevolumen [Mrd. €]")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper left")

    residuals = real_values[: len(mittelwert)] - mittelwert
    axes[1].bar(x_zukunft[: len(residuals)], residuals, color="gray", alpha=0.7, width=0.8)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Residuum (Ist − Prognose)")
    axes[1].set_xlabel("Datum")
    axes[1].set_title("Out-of-Sample Fehler")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def main():
    raise NotImplementedError("plot_forecast() aus einem Modell-Skript aufrufen.")


if __name__ == "__main__":
    main()