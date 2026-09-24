from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


INPUT_PATH = Path(
    "results/tables/visual_robustness_summary.csv"
)

OUTPUT_PATH = Path(
    "results/figures/visual_robustness_comparison.png"
)


def main():
    df = pd.read_csv(INPUT_PATH)

    plot_df = df.pivot(
        index="model",
        columns="transformation",
        values="change_rate_percent"
    )

    plot_df.plot(
        kind="bar",
        figsize=(11, 6)
    )

    plt.title("Model Robustness Under Image Transformations")
    plt.xlabel("Model")
    plt.ylabel("Prediction Change Rate (%)")
    plt.xticks(rotation=15)
    plt.legend(title="Image transformation")
    plt.tight_layout()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.savefig(
        OUTPUT_PATH,
        dpi=300
    )

    plt.show()

    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()