"""Generate dissertation-ready Chapter 5 result figures."""

from pathlib import Path

import matplotlib.pyplot as plt


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "results" / "figures" / "dissertation"


def horizontal_bar(filename, title, labels, values, colour):
    fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=180)
    positions = range(len(labels))
    bars = ax.barh(positions, values, color=colour, edgecolor="#333333", linewidth=0.5)
    ax.set_yticks(list(positions), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Macro F1 score")
    ax.set_title(title, pad=14, weight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values):
        ax.text(value + 0.015, bar.get_y() + bar.get_height() / 2, f"{value:.4f}", va="center")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    horizontal_bar(
        "figure_5_1_common_subset_model_comparison.png",
        "Common-subset model comparison",
        ["BERT", "ViLT", "CLIP", "BLIP+BERT"],
        [0.9119, 0.5963, 0.4903, 0.2902],
        "#2878B5",
    )
    horizontal_bar(
        "figure_5_2_verite_external_generalisation.png",
        "External generalisation on VERITE",
        ["BLIP+BERT", "BERT", "CLIP", "ViLT"],
        [0.5629, 0.5261, 0.3112, 0.1288],
        "#E07A3F",
    )
    horizontal_bar(
        "figure_5_3_vilt_modality_ablation.png",
        "ViLT modality ablation",
        ["Correct image", "Blank image", "Shuffled image"],
        [0.7300, 0.4418, 0.3761],
        "#3A9D5D",
    )
    horizontal_bar(
        "figure_5_4_controlled_multimodal_fusion.png",
        "Controlled multimodal fusion",
        ["Normal fusion", "Modality dropout", "Text only", "Image only"],
        [0.8541, 0.8482, 0.8296, 0.7888],
        "#7C5AB8",
    )


if __name__ == "__main__":
    main()
