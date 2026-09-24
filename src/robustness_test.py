import random
import re
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from predict_bert import predict as predict_bert
from predict_clip import predict as predict_clip
from predict_vilt import predict as predict_vilt
from predict_blip_caption_bert import predict as predict_blip


DATA_PATH = Path("data/processed/multimodal_test.csv")
OUTPUT_PATH = Path("results/tables/robustness_results.csv")

SAMPLE_SIZE = 500

random.seed(42)
def remove_punctuation(text):
    return re.sub(r"[^\w\s]", "", str(text))


def remove_one_word(text):
    words = str(text).split()

    if len(words) <= 3:
        return str(text)

    del words[len(words) // 2]

    return " ".join(words)


def add_noise(text):
    return str(text) + " breaking news"


def label_number(label):
    return 1 if label == "Fake" else 0
def main():
    df = pd.read_csv(DATA_PATH)

    real_samples = df[df["label"] == 0].sample(
        n=SAMPLE_SIZE // 2,
        random_state=42
    )

    fake_samples = df[df["label"] == 1].sample(
        n=SAMPLE_SIZE // 2,
        random_state=42
    )

    test_subset = pd.concat([
        real_samples,
        fake_samples
    ]).sample(
        frac=1,
        random_state=42
    )

    perturbations = {
        "remove_punctuation": remove_punctuation,
        "remove_one_word": remove_one_word,
        "add_noise": add_noise
    }

    records = []
    failures = []
    for row_id, row in tqdm(
        test_subset.iterrows(),
        total=len(test_subset),
        desc="Robustness testing"
    ):
        original_text = str(row["text"])
        image = str(row["image"])
        true_label = int(row["label"])

        try:
            bert_label, _ = predict_bert(original_text)

            clip_label, _ = predict_clip(
                original_text,
                image
            )

            vilt_label, _ = predict_vilt(
                original_text,
                image
            )

            blip_label, _, caption = predict_blip(
                original_text,
                image
            )

            original_predictions = {
                "BERT": label_number(bert_label),
                "CLIP + MLP": label_number(clip_label),
                "ViLT": label_number(vilt_label),
                "BLIP Caption + BERT": label_number(blip_label)
            }
            for perturbation_name, perturbation_function in perturbations.items():
                changed_text = perturbation_function(original_text)

                changed_bert_label, _ = predict_bert(
                    changed_text
                )

                changed_clip_label, _ = predict_clip(
                    changed_text,
                    image
                )

                changed_vilt_label, _ = predict_vilt(
                    changed_text,
                    image
                )

                changed_blip_label, _, _ = predict_blip(
                    changed_text,
                    image,
                    existing_caption=caption
                )

                changed_predictions = {
                    "BERT": label_number(changed_bert_label),
                    "CLIP + MLP": label_number(changed_clip_label),
                    "ViLT": label_number(changed_vilt_label),
                    "BLIP Caption + BERT": label_number(
                        changed_blip_label
                    )
                }

                for model_name, original_prediction in original_predictions.items():
                    changed_prediction = changed_predictions[model_name]

                    records.append({
                        "row_id": row_id,
                        "model": model_name,
                        "perturbation": perturbation_name,
                        "true_label": true_label,
                        "original_text": original_text,
                        "changed_text": changed_text,
                        "original_prediction": original_prediction,
                        "perturbed_prediction": changed_prediction,
                        "prediction_changed": (
                            original_prediction != changed_prediction
                        )
                    })
        except Exception as error:
            print(f"\nSkipping row {row_id}: {error}")
            failures.append({"row_id": row_id, "error": repr(error)})

    results_df = pd.DataFrame(records)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results_df.to_csv(
        OUTPUT_PATH,
        index=False
    )
    pd.DataFrame(failures, columns=["row_id", "error"]).to_csv(
        "results/tables/robustness_failures.csv", index=False
    )
    coverage = {
        "requested_samples": len(test_subset),
        "successful_samples": int(results_df["row_id"].nunique()) if not results_df.empty else 0,
        "failed_samples": len(failures),
    }
    coverage["coverage_percent"] = 100 * coverage["successful_samples"] / len(test_subset)
    Path("results/tables/robustness_coverage.json").write_text(
        json.dumps(coverage, indent=2), encoding="utf-8"
    )

    if results_df.empty:
        print("No results were produced.")
        return

    summary_df = (
        results_df
        .groupby(["model", "perturbation"])["prediction_changed"]
        .agg(["count", "mean"])
        .reset_index()
    )

    summary_df["change_rate_percent"] = (
        summary_df["mean"] * 100
    )

    summary_path = Path(
        "results/tables/robustness_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    print("\nRobustness summary:")
    print(summary_df.to_string(index=False))

    print("\nSaved:", OUTPUT_PATH)
    print("Saved:", summary_path)
    print("Coverage:", json.dumps(coverage, indent=2))
if __name__ == "__main__":
    main()
