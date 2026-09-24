"""Train a reproducible TF-IDF + Logistic Regression fake-news baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


def stratified_sample(frame: pd.DataFrame, maximum: int, seed: int) -> pd.DataFrame:
    if maximum >= len(frame):
        return frame.reset_index(drop=True)
    proportions = frame["label"].value_counts(normalize=True)
    pieces = []
    remaining = maximum
    labels = sorted(frame["label"].unique())
    for position, label in enumerate(labels):
        candidates = frame[frame["label"] == label]
        count = remaining if position == len(labels) - 1 else round(maximum * proportions[label])
        count = min(count, len(candidates))
        pieces.append(candidates.sample(n=count, random_state=seed))
        remaining -= count
    return pd.concat(pieces).sample(frac=1, random_state=seed).reset_index(drop=True)


def score(labels, predictions) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(labels, predictions),
        "balanced_accuracy": balanced_accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("data/splits/fakeddit_train.csv"))
    parser.add_argument("--validation", type=Path, default=Path("data/splits/fakeddit_validation.csv"))
    parser.add_argument("--test", type=Path, default=Path("data/splits/fakeddit_test.csv"))
    parser.add_argument("--bert-predictions", type=Path, default=Path("results/predictions/bert_test.csv"))
    parser.add_argument("--train-samples", type=int, default=20000)
    parser.add_argument("--validation-samples", type=int, default=5000)
    parser.add_argument("--test-samples", type=int, default=5000)
    parser.add_argument("--max-features", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--predictions", type=Path, default=Path("results/predictions/tfidf_test.csv"))
    parser.add_argument("--summary", type=Path, default=Path("results/tables/tfidf_training_summary.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train = stratified_sample(pd.read_csv(args.train), args.train_samples, args.seed)
    validation = stratified_sample(
        pd.read_csv(args.validation), args.validation_samples, args.seed
    )
    full_test = pd.read_csv(args.test)

    # Use exactly the same test sample IDs as the completed BERT pilot when available.
    if args.bert_predictions.exists():
        bert_ids = pd.read_csv(args.bert_predictions, usecols=["sample_id"])["sample_id"]
        test = full_test.set_index("sample_id").loc[bert_ids].reset_index()
    else:
        test = stratified_sample(full_test, args.test_samples, args.seed)

    print(f"Rows: train={len(train)}, validation={len(validation)}, test={len(test)}")
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        max_features=args.max_features,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    train_features = vectorizer.fit_transform(train["text"].astype(str))
    validation_features = vectorizer.transform(validation["text"].astype(str))
    test_features = vectorizer.transform(test["text"].astype(str))

    candidates = []
    best_model = None
    best_f1 = -1.0
    for c_value in (0.1, 0.5, 1.0, 2.0, 5.0):
        model = LogisticRegression(
            C=c_value,
            class_weight="balanced",
            max_iter=1000,
            random_state=args.seed,
            solver="liblinear",
        )
        model.fit(train_features, train["label"])
        validation_scores = score(validation["label"], model.predict(validation_features))
        candidates.append({"C": c_value, **validation_scores})
        if validation_scores["macro_f1"] > best_f1:
            best_f1 = validation_scores["macro_f1"]
            best_model = model

    predictions = best_model.predict(test_features)
    probabilities = best_model.predict_proba(test_features)[:, 1]
    test_scores = score(test["label"], predictions)
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "sample_id": test["sample_id"],
        "true_label": test["label"].astype(int),
        "prediction": predictions.astype(int),
        "probability": probabilities,
    }).to_csv(args.predictions, index=False)

    summary = {
        "model": "TF-IDF + Logistic Regression",
        "seed": args.seed,
        "train_samples": len(train),
        "validation_samples": len(validation),
        "test_samples": len(test),
        "vocabulary_size": len(vectorizer.vocabulary_),
        "validation_candidates": candidates,
        "selected_C": best_model.C,
        "test_metrics": test_scores,
        "predictions": str(args.predictions),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
