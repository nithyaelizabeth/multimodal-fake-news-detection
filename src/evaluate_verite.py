from pathlib import Path
import json

import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

from predict_bert import predict as predict_bert
from predict_clip import predict as predict_clip
from predict_vilt import predict as predict_vilt
from predict_blip_caption_bert import predict as predict_blip

DATA_PATH = Path("data/processed/verite_test.csv")
OUTPUT_PATH = Path("results/tables/verite_model_predictions.csv")


def calculate_metrics(name, true_labels, predictions):
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels,
        predictions,
        average="binary",
        zero_division=0
    )

    return {
        "Model": name,
        "Samples": len(true_labels),
        "Accuracy": accuracy_score(true_labels, predictions),
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    }


def main():
    df = pd.read_csv(DATA_PATH)

    # Evaluate the complete VERITE dataset, preserving its natural distribution.
    test_subset = df.copy()

    records = []
    failures = []

    for index, row in tqdm(
        test_subset.iterrows(),
        total=len(test_subset)
    ):
        text = str(row["text"])
        image = str(row["image"])
        true_label = int(row["label"])

        try:
            bert_label, bert_confidence = predict_bert(text)
            clip_label, clip_confidence = predict_clip(text, image)
            vilt_label, vilt_confidence = predict_vilt(text, image)
            blip_label, blip_confidence, caption = predict_blip(text, image)
        except Exception as error:
            print(f"\nSkipping row {index}: {error}")
            failures.append({"row_id": index, "error": repr(error)})
            continue

        records.append({
            "row_id": index,
            "text": text,
            "image": image,
            "true_label": true_label,
            "bert_prediction": 1 if bert_label == "Fake" else 0,
            "bert_confidence": bert_confidence,
            "clip_prediction": 1 if clip_label == "Fake" else 0,
            "clip_confidence": clip_confidence,
            "vilt_prediction": 1 if vilt_label == "Fake" else 0,
            "vilt_confidence": vilt_confidence,
            "blip_prediction": 1 if blip_label == "Fake" else 0,
            "blip_confidence": blip_confidence,
            "caption": caption
        })

    predictions_df = pd.DataFrame(records)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(OUTPUT_PATH, index=False)

    failure_path = Path("results/tables/verite_evaluation_failures.csv")
    pd.DataFrame(failures, columns=["row_id", "error"]).to_csv(failure_path, index=False)
    coverage = {
        "requested_samples": len(test_subset),
        "successful_samples": len(predictions_df),
        "failed_samples": len(failures),
        "coverage_percent": 100 * len(predictions_df) / len(test_subset),
    }
    Path("results/tables/verite_evaluation_coverage.json").write_text(
        json.dumps(coverage, indent=2), encoding="utf-8"
    )

    if predictions_df.empty:
        print("No predictions were generated. Check for skipped rows or model errors.")
        return

    true_labels = predictions_df["true_label"]

    results = [
        calculate_metrics(
            "BERT",
            true_labels,
            predictions_df["bert_prediction"]
        ),
        calculate_metrics(
            "CLIP + MLP",
            true_labels,
            predictions_df["clip_prediction"]
        ),
        calculate_metrics(
            "ViLT",
            true_labels,
            predictions_df["vilt_prediction"]
        ),
        calculate_metrics(
            "BLIP + BERT",
            true_labels,
            predictions_df["blip_prediction"]
        ),
    ]

    results_df = pd.DataFrame(results)
    results_df.to_csv(
        "results/tables/verite_model_comparison.csv",
        index=False
    )

    print("\nModel comparison:")
    print(results_df.to_string(index=False))
    print("\nCoverage:", json.dumps(coverage, indent=2))


if __name__ == "__main__":
    main()
