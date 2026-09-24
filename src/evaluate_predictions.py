"""Evaluate any model's saved predictions with bootstrap confidence intervals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "fake_precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "fake_recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "fake_f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "real_recall": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
    }


def bootstrap_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    repetitions: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    values = {name: [] for name in metrics(y_true, y_pred)}
    for _ in range(repetitions):
        indexes = rng.integers(0, len(y_true), len(y_true))
        sample_metrics = metrics(y_true[indexes], y_pred[indexes])
        for name, value in sample_metrics.items():
            values[name].append(value)
    return {
        name: {
            "lower_95": float(np.percentile(samples, 2.5)),
            "upper_95": float(np.percentile(samples, 97.5)),
        }
        for name, samples in values.items()
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--true-column", default="true_label")
    parser.add_argument("--prediction-column", default="prediction")
    parser.add_argument("--model-name", default="model")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.predictions)
    required = {args.true_column, args.prediction_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")

    clean = frame[[args.true_column, args.prediction_column]].dropna().astype(int)
    if clean.empty or not set(clean.to_numpy().ravel()).issubset({0, 1}):
        raise ValueError("Labels and predictions must be non-empty binary values (0 or 1)")

    y_true = clean[args.true_column].to_numpy()
    y_pred = clean[args.prediction_column].to_numpy()
    scores = metrics(y_true, y_pred)
    intervals = bootstrap_intervals(y_true, y_pred, args.bootstrap, args.seed)
    result = {
        "model": args.model_name,
        "samples": len(clean),
        "bootstrap_repetitions": args.bootstrap,
        "seed": args.seed,
        "metrics": {
            name: {"value": float(value), **intervals[name]}
            for name, value in scores.items()
        },
    }

    output = args.output or Path("results/tables") / f"{args.model_name.lower().replace(' ', '_')}_evaluation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
