from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


INPUT_PATH = Path(
    "results/tables/cross_dataset_comparison.csv"
)

TABLE_PATH = Path(
    "results/tables/generalization_gap.csv"
)

FIGURE_PATH = Path(
    "results/figures/generalization_gap.png"
)


def main():
    df = pd.read_csv(INPUT_PATH)

    f1_table = df.pivot(
        index="Model",
        columns="Dataset",
        values="F1-score"
    ).reset_index()

    f1_table["F1 Gap"] = (
        f1_table["Fakeddit"] - f1_table["VERITE"]
    )

    f1_table.to_csv(TABLE_PATH, index=False)

    print("\nGeneralization results:")
    print(f1_table.to_string(index=False))

    ax = f1_table.set_index("Model")[
        ["Fakeddit", "VERITE"]
    ].plot(
        kind="bar",
        figsize=(10, 5)
    )

    plt.title("Cross-Dataset F1-Score Comparison")
    plt.xlabel("Model")
    plt.ylabel("F1-score")
    plt.ylim(0, 1)
    plt.xticks(rotation=15)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=3)

    plt.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURE_PATH, dpi=300)
    plt.show()

    print("\nSaved:", TABLE_PATH)
    print("Saved:", FIGURE_PATH)


if __name__ == "__main__":
    main()