"""Evaluate whether multimodal models genuinely use image information.

This script is intentionally separate from training. It compares predictions
under correct, shuffled, and blank images and adds caption/text ablations for
BLIP + BERT. It does not modify model checkpoints.

Run manually when ready:
    .venv/bin/python src/modality_ablation.py --sample-size 200
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score
from tqdm import tqdm

from predict_blip_caption_bert import predict as predict_blip
from predict_clip import predict as predict_clip
from predict_vilt import predict as predict_vilt


def label_number(label: str) -> int:
    return 1 if label == "Fake" else 0


def balanced_sample(frame: pd.DataFrame, size: int, seed: int) -> pd.DataFrame:
    if size < 2 or size % 2:
        raise ValueError("--sample-size must be an even number of at least 2")
    real = frame[frame["label"] == 0].sample(n=size // 2, random_state=seed)
    fake = frame[frame["label"] == 1].sample(n=size // 2, random_state=seed)
    return pd.concat([real, fake]).sample(frac=1, random_state=seed)


def shuffled_images(frame: pd.DataFrame, seed: int) -> list[str]:
    images = frame["image"].astype(str).tolist()
    rng = np.random.default_rng(seed)
    order = np.arange(len(images))
    # A cyclic shift after random permutation guarantees no row keeps its image.
    permutation = rng.permutation(order)
    mapping = np.empty_like(permutation)
    mapping[permutation] = np.roll(permutation, 1)
    return [images[index] for index in mapping]


def make_blank_image() -> Path:
    handle = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    path = Path(handle.name)
    handle.close()
    Image.new("RGB", (384, 384), color=(127, 127, 127)).save(path)
    return path


def metric_row(model: str, condition: str, group: pd.DataFrame) -> dict[str, object]:
    truth = group["true_label"]
    prediction = group["prediction"]
    return {
        "Model": model,
        "Condition": condition,
        "Samples": len(group),
        "Accuracy": accuracy_score(truth, prediction),
        "Balanced Accuracy": balanced_accuracy_score(truth, prediction),
        "Macro F1": f1_score(truth, prediction, average="macro", zero_division=0),
        "Real Recall": recall_score(truth, prediction, pos_label=0, zero_division=0),
        "Fake Recall": recall_score(truth, prediction, pos_label=1, zero_division=0),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=Path, default=Path("data/processed/multimodal_test.csv")
    )
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("results/tables/modality_ablation_predictions.csv"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/tables/modality_ablation_summary.csv"),
    )
    parser.add_argument(
        "--failures",
        type=Path,
        default=Path("results/tables/modality_ablation_failures.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subset = balanced_sample(pd.read_csv(args.data), args.sample_size, args.seed).copy()
    subset["shuffled_image"] = shuffled_images(subset, args.seed)
    blank_path = make_blank_image()
    records: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    try:
        for row_id, row in tqdm(subset.iterrows(), total=len(subset), desc="Ablation"):
            text = str(row["text"])
            correct_image = str(row["image"])
            shuffled_image = str(row["shuffled_image"])
            true_label = int(row["label"])

            direct_conditions = {
                "correct_image": correct_image,
                "shuffled_image": shuffled_image,
                "blank_image": str(blank_path),
            }
            for condition, image in direct_conditions.items():
                for model_name, predictor in (
                    ("CLIP + MLP", predict_clip),
                    ("ViLT", predict_vilt),
                ):
                    try:
                        label, confidence = predictor(text, image)
                        records.append({
                            "row_id": row_id,
                            "model": model_name,
                            "condition": condition,
                            "true_label": true_label,
                            "prediction": label_number(label),
                            "confidence": confidence,
                        })
                    except Exception as error:
                        failures.append({
                            "row_id": row_id,
                            "model": model_name,
                            "condition": condition,
                            "error": repr(error),
                        })

            try:
                _, _, correct_caption = predict_blip(text, correct_image)
                _, _, shuffled_caption = predict_blip(text, shuffled_image)
                _, _, blank_caption = predict_blip(text, str(blank_path))
                caption_conditions = {
                    "text_plus_correct_caption": (text, correct_caption),
                    "text_plus_shuffled_caption": (text, shuffled_caption),
                    "text_plus_blank_caption": (text, blank_caption),
                    "caption_only": ("", correct_caption),
                    "text_only": (text, ""),
                }
                for condition, (condition_text, caption) in caption_conditions.items():
                    label, confidence, _ = predict_blip(
                        condition_text, correct_image, existing_caption=caption
                    )
                    records.append({
                        "row_id": row_id,
                        "model": "BLIP Caption + BERT",
                        "condition": condition,
                        "true_label": true_label,
                        "prediction": label_number(label),
                        "confidence": confidence,
                    })
            except Exception as error:
                failures.append({
                    "row_id": row_id,
                    "model": "BLIP Caption + BERT",
                    "condition": "caption_generation_or_classification",
                    "error": repr(error),
                })
    finally:
        blank_path.unlink(missing_ok=True)

    predictions = pd.DataFrame(records)
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.predictions, index=False)
    pd.DataFrame(
        failures, columns=["row_id", "model", "condition", "error"]
    ).to_csv(args.failures, index=False)
    if predictions.empty:
        raise RuntimeError("No ablation predictions were generated; inspect the failure file")

    # Compare conditions only on rows for which every condition for that model
    # succeeded. This paired design prevents failed image URLs from changing the
    # sample composition between the correct and ablated conditions.
    expected_conditions = {
        "CLIP + MLP": {"correct_image", "shuffled_image", "blank_image"},
        "ViLT": {"correct_image", "shuffled_image", "blank_image"},
        "BLIP Caption + BERT": {
            "text_plus_correct_caption",
            "text_plus_shuffled_caption",
            "text_plus_blank_caption",
            "caption_only",
            "text_only",
        },
    }
    paired_parts = []
    paired_coverage = {}
    for model, conditions in expected_conditions.items():
        model_rows = predictions[predictions["model"] == model]
        condition_sets = [
            set(model_rows[model_rows["condition"] == condition]["row_id"])
            for condition in conditions
        ]
        common_rows = set.intersection(*condition_sets) if condition_sets else set()
        paired_parts.append(model_rows[model_rows["row_id"].isin(common_rows)])
        paired_coverage[model] = len(common_rows)
    paired_predictions = pd.concat(paired_parts, ignore_index=True)

    summary_rows = [
        metric_row(model, condition, group)
        for (model, condition), group in paired_predictions.groupby(["model", "condition"])
    ]
    summary = pd.DataFrame(summary_rows)

    reference_condition = {
        "CLIP + MLP": "correct_image",
        "ViLT": "correct_image",
        "BLIP Caption + BERT": "text_plus_correct_caption",
    }
    reference_f1 = {
        model: float(
            summary[(summary["Model"] == model) & (summary["Condition"] == condition)][
                "Macro F1"
            ].iloc[0]
        )
        for model, condition in reference_condition.items()
    }
    summary["Macro F1 Drop vs Correct"] = summary.apply(
        lambda row: reference_f1[row["Model"]] - row["Macro F1"], axis=1
    )
    summary.to_csv(args.summary, index=False)

    coverage = {
        "requested_samples": len(subset),
        "prediction_rows": len(predictions),
        "failure_rows": len(failures),
        "paired_samples_by_model": paired_coverage,
        "seed": args.seed,
    }
    Path("results/tables/modality_ablation_coverage.json").write_text(
        json.dumps(coverage, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(json.dumps(coverage, indent=2))


if __name__ == "__main__":
    main()
