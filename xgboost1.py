from treesearch_plot import plot_forecast
import common as tc

def main():
    # Holt automatisch das vordefinierte Set (z.B. "nur_makro" oder "ohne_dax")
    mein_set = tc.covariates_for_label("nur_makro") 

    plot_forecast(
        model_label="XGBoost",
        model_name="xgboost",
        title_suffix="XGBoost (nur Makro)",
        covariates=mein_set,
    )

if __name__ == "__main__":
    main()
