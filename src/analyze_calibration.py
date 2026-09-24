"""Analyse probability calibration for all fake-news classifiers.

This script uses existing prediction CSV files. It does not train models or run
inference.

Run manually when ready:
    MPLBACKEND=Agg .venv/bin/python src/analyze_calibration.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss


WIDE_MODELS = {
    "BERT (four-model subset)": ("bert_prediction", "bert_confidence"),
    "CLIP + MLP": ("clip_prediction", "clip_confidence"),
    "ViLT": ("vilt_prediction", "vilt_confidence"),
    "BLIP Caption + BERT": ("blip_prediction", "blip_confidence"),
}


def predicted_confidence_to_fake_probability(
    predictions: pd.Series, confidence: pd.Series
) -> pd.Series:
    """Convert confidence in the predicted class to P(fake)."""
    return pd.Series(
        np.where(predictions.astype(int) == 1, confidence, 1.0 - confidence),
        index=predictions.index,
    ).clip(0.0, 1.0)


def load_generic(path: Path, model_name: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"true_label", "prediction", "probability"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return pd.DataFrame({
        "model": model_name,
        "true_label": frame["true_label"].astype(int),
        "prediction": frame["prediction"].astype(int),
        "fake_probability": frame["probability"].astype(float).clip(0.0, 1.0),
    })


def load_wide(path: Path) -> list[pd.DataFrame]:
    frame = pd.read_csv(path)
    outputs = []
    for model_name, (prediction_column, confidence_column) in WIDE_MODELS.items():
        if prediction_column not in frame or confidence_column not in frame:
            continue
        predictions = frame[prediction_column].astype(int)
        outputs.append(pd.DataFrame({
            "model": model_name,
            "true_label": frame["true_label"].astype(int),
            "prediction": predictions,
            "fake_probability": predicted_confidence_to_fake_probability(
                predictions, frame[confidence_column].astype(float)
            ),
        }))
    return outputs


def reliability_bins(frame: pd.DataFrame, number_of_bins: int) -> pd.DataFrame:
    edges = np.linspace(0.0, 1.0, number_of_bins + 1)
    bin_ids = np.minimum(
        np.digitize(frame["fake_probability"], edges[1:-1], right=False),
        number_of_bins - 1,
    )
    working = frame.copy()
    working["bin"] = bin_ids
    rows = []
    for bin_id in range(number_of_bins):
        group = working[working["bin"] == bin_id]
        rows.append({
            "bin": bin_id,
            "lower_bound": edges[bin_id],
            "upper_bound": edges[bin_id + 1],
            "samples": len(group),
            "mean_fake_probability": group["fake_probability"].mean(),
            "observed_fake_rate": group["true_label"].mean(),
        })
    return pd.DataFrame(rows)


def expected_calibration_error(bins: pd.DataFrame) -> float:
    populated = bins[bins["samples"] > 0]
    total = populated["samples"].sum()
    return float(
        (
            populated["samples"]
            / total
            * (populated["mean_fake_probability"] - populated["observed_fake_rate"]).abs()
        ).sum()
    )


def confidence_threshold_rows(frame: pd.DataFrame, model: str) -> list[dict[str, object]]:
    confidence = np.maximum(frame["fake_probability"], 1.0 - frame["fake_probability"])
    correct = frame["prediction"] == frame["true_label"]
    rows = []
    for threshold in (0.50, 0.60, 0.70, 0.80, 0.90, 0.95):
        selected = confidence >= threshold
        rows.append({
            "Model": model,
            "Confidence Threshold": threshold,
            "Selected Samples": int(selected.sum()),
            "Coverage": float(selected.mean()),
            "Accuracy": float(correct[selected].mean()) if selected.any() else np.nan,
        })
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bert", type=Path, default=Path("results/predictions/bert_test.csv")
    )
    parser.add_argument(
        "--tfidf", type=Path, default=Path("results/predictions/tfidf_test.csv")
    )
    parser.add_argument(
        "--wide", type=Path, default=Path("results/tables/model_predictions.csv")
    )
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/tables/calibration")
    )
    parser.add_argument(
        "--figure", type=Path, default=Path("results/figures/calibration_reliability.png")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames = []
    if args.bert.exists():
        frames.append(load_generic(args.bert, "BERT (reproducible 20k)"))
    if args.tfidf.exists():
        frames.append(load_generic(args.tfidf, "TF-IDF + Logistic Regression"))
    if args.wide.exists():
        frames.extend(load_wide(args.wide))
    if not frames:
        raise FileNotFoundError("No compatible prediction files were found")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    threshold_rows = []
    all_bins = []

    for frame in frames:
        model = str(frame["model"].iloc[0])
        bins = reliability_bins(frame, args.bins)
        bins.insert(0, "Model", model)
        all_bins.append(bins)
        confidence = np.maximum(frame["fake_probability"], 1.0 - frame["fake_probability"])
        correct = frame["prediction"] == frame["true_label"]
        summary_rows.append({
            "Model": model,
            "Samples": len(frame),
            "Brier Score": brier_score_loss(frame["true_label"], frame["fake_probability"]),
            "Expected Calibration Error": expected_calibration_error(bins),
            "Mean Confidence": float(confidence.mean()),
            "Mean Confidence Correct": float(confidence[correct].mean()),
            "Mean Confidence Incorrect": float(confidence[~correct].mean()),
        })
        threshold_rows.extend(confidence_threshold_rows(frame, model))

    summary = pd.DataFrame(summary_rows)
    thresholds = pd.DataFrame(threshold_rows)
    bins_frame = pd.concat(all_bins, ignore_index=True)
    summary.to_csv(args.output_dir / "calibration_summary.csv", index=False)
    thresholds.to_csv(args.output_dir / "confidence_thresholds.csv", index=False)
    bins_frame.to_csv(args.output_dir / "reliability_bins.csv", index=False)

    plt.figure(figsize=(8, 7))
    plt.plot([0, 1], [0, 1], "--", color="black", label="Perfect calibration")
    for model, group in bins_frame[bins_frame["samples"] > 0].groupby("Model"):
        plt.plot(
            group["mean_fake_probability"],
            group["observed_fake_rate"],
            marker="o",
            label=model,
        )
    plt.xlabel("Mean predicted probability of fake")
    plt.ylabel("Observed fake-news rate")
    plt.title("Reliability Diagram")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(fontsize=8)
    plt.tight_layout()
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.figure, dpi=300)
    plt.close()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
