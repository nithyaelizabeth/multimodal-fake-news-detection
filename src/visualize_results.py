from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


TABLES_DIR = Path("results/tables")
FIGURES_DIR = Path("results/figures")

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

results_df = pd.read_csv(TABLES_DIR / "model_comparison.csv")
predictions_df = pd.read_csv(TABLES_DIR / "model_predictions.csv")
metrics = ["Accuracy", "Precision", "Recall", "F1-score"]

results_df.set_index("Model")[metrics].plot(
    kind="bar",
    figsize=(10, 5)
)

plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "model_comparison.png", dpi=300)
plt.close()
models = {
    "BERT": "bert_prediction",
    "CLIP_MLP": "clip_prediction",
    "ViLT": "vilt_prediction"
}

for model_name, prediction_column in models.items():
    matrix = confusion_matrix(
        predictions_df["true_label"],
        predictions_df[prediction_column],
        labels=[0, 1]
    )

    display = ConfusionMatrixDisplay(
        matrix,
        display_labels=["Real", "Fake"]
    )

    display.plot(cmap="Blues")
    plt.title(f"{model_name} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / f"{model_name.lower()}_confusion_matrix.png",
        dpi=300
    )
    plt.close()

print("Figures saved in results/figures")