from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


INPUT_PATH = Path("results/tables/model_predictions.csv")
OUTPUT_PATH = Path("results/tables/class_error_analysis.csv")

MODELS = {
    "BERT": "bert_prediction",
    "CLIP + MLP": "clip_prediction",
    "ViLT": "vilt_prediction",
    "BLIP Caption + BERT": "blip_prediction"
}


def main():
    df = pd.read_csv(INPUT_PATH)
    results = []

    for model_name, prediction_column in MODELS.items():
        true_labels = df["true_label"]
        predictions = df[prediction_column]

        tn, fp, fn, tp = confusion_matrix(
            true_labels,
            predictions,
            labels=[0, 1]
        ).ravel()

        report = classification_report(
            true_labels,
            predictions,
            labels=[0, 1],
            output_dict=True,
            zero_division=0
        )

        results.append({
            "Model": model_name,
            "True Real": tn,
            "False Fake": fp,
            "False Real": fn,
            "True Fake": tp,
            "Real Recall": report["0"]["recall"],
            "Fake Recall": report["1"]["recall"],
            "Macro F1": report["macro avg"]["f1-score"]
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_PATH, index=False)

    print(results_df.to_string(index=False))
    print("\nSaved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()