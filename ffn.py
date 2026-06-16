from treesearch_plot import plot_forecast


def main():
    plot_forecast(
        model_label="FFN",
        model_name="ffn",
        title_suffix="Feedforward Neural Network",
    )


if __name__ == "__main__":
    main()
