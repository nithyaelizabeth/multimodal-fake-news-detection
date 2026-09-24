from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


INPUT_PATH = Path("results/tables/verite_type_analysis.csv")
OUTPUT_PATH = Path("results/figures/verite_type_analysis.png")


def main():
    df = pd.read_csv(INPUT_PATH)

    plot_df = df.pivot(
        index="Model",
        columns="VERITE Type",
        values="Accuracy"
    )

    ax = plot_df.plot(
        kind="bar",
        figsize=(11, 6)
    )

    plt.title("Model Accuracy by VERITE Misinformation Type")
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.xticks(rotation=15)
    plt.legend(title="VERITE Type")

    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=3)

    plt.tight_layout()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=300)
    plt.show()

    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()