"""Gemeinsamer Plot für alle Treesearch-Modell-Skripte."""

import matplotlib.pyplot as plt
import pandas as pd

from treesearch_common import (
    ALL_COVARIATES,
    HISTORIE_ANFANG,
    HISTORIE_ENDE,
    INPUT_ANFANG,
    PROGNOSEHORIZONT,
    TARGET_COL,
    run_ensemble_volume_paths,
)


def plot_forecast(model_label: str, model_name: str, title_suffix: str, covariates=None):
    covariates = covariates or ALL_COVARIATES
    mittelwert, ki90, ki98, _, df = run_ensemble_volume_paths(model_name, covariates)

    real_start = df.index.get_loc(INPUT_ANFANG)
    real_values = df.iloc[real_start : real_start + PROGNOSEHORIZONT][TARGET_COL].values
    historie = df.loc[HISTORIE_ANFANG:HISTORIE_ENDE, TARGET_COL]
    x_zukunft = pd.date_range(INPUT_ANFANG, periods=PROGNOSEHORIZONT, freq="D")

    plt.figure(figsize=(15, 8))
    plt.plot(historie.index, historie.values, color="blue", label="Historische Daten (Training)")
    plt.plot(
        x_zukunft,
        real_values[: len(x_zukunft)],
        "--",
        color="black",
        label="Tatsächliche Reale Werte (Test)",
    )
    plt.plot(
        x_zukunft,
        mittelwert,
        color="red",
        label=f"{model_label} Ensemble Mittelwert (Diff-anchored)",
        linewidth=2,
    )
    plt.fill_between(
        x_zukunft, ki90[0], ki90[1], color="red", alpha=0.2, label="90% Konfidenzintervall"
    )
    plt.fill_between(
        x_zukunft, ki98[0], ki98[1], color="red", alpha=0.1, label="98% Konfidenzintervall"
    )
    plt.title(
        f"Rigorous Treesearch Forecast: Level Reconstruction via Predicted Differences "
        f"({title_suffix})",
        fontsize=14,
    )
    plt.ylabel("Einlagevolumen [Mrd. €]")
    plt.xlabel("Datum")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.show()


def main():
    raise NotImplementedError("plot_forecast() aus einem Modell-Skript aufrufen.")


if __name__ == "__main__":
    main()
