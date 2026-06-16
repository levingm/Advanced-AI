from treesearch_plot import plot_forecast


def main():
    plot_forecast(
        model_label="XGBoost",
        model_name="xgboost",
        title_suffix="XGBoost",
    )


if __name__ == "__main__":
    main()
