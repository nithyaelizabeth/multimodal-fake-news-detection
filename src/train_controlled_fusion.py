"""Train controlled text, image and multimodal classifiers on cached features."""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class Classifier(nn.Module):
    def __init__(self, text_dim: int, image_dim: int, condition: str, hidden: int, dropout: float):
        super().__init__()
        self.condition = condition
        input_dim = text_dim if condition == "text_only" else image_dim
        if condition in {"normal_fusion", "modality_dropout"}:
            input_dim = text_dim + image_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 2)
        )

    def forward(self, text, image, modality_dropout_probability=0.0):
        if self.condition == "text_only":
            features = text
        elif self.condition == "image_only":
            features = image
        else:
            if self.training and self.condition == "modality_dropout":
                selected = torch.rand(len(text), device=text.device) < modality_dropout_probability
                drop_text = selected & (torch.rand(len(text), device=text.device) < 0.5)
                drop_image = selected & ~drop_text
                text = text.masked_fill(drop_text[:, None], 0)
                image = image.masked_fill(drop_image[:, None], 0)
            features = torch.cat([text, image], dim=1)
        return self.network(features)


def load_features(path: Path):
    # Older feature archives created by this project stored sample IDs as an
    # object array. allow_pickle is needed only to read those locally generated
    # string IDs; all arrays are immediately converted to fixed numeric/string
    # dtypes and validated below.
    data = np.load(path, allow_pickle=True)
    required = {"sample_ids", "labels", "text_features", "image_features"}
    missing = required.difference(data.files)
    if missing:
        raise ValueError(f"{path} is missing arrays: {sorted(missing)}")
    sample_ids = np.asarray(data["sample_ids"], dtype=str)
    labels = np.asarray(data["labels"], dtype=np.int64)
    text_features = np.asarray(data["text_features"], dtype=np.float32)
    image_features = np.asarray(data["image_features"], dtype=np.float32)
    sizes = {len(sample_ids), len(labels), len(text_features), len(image_features)}
    if len(sizes) != 1 or not len(labels):
        raise ValueError(f"{path} contains empty or differently sized arrays")
    return sample_ids, labels, text_features, image_features


def metrics(labels, predictions):
    return {
        "accuracy": accuracy_score(labels, predictions),
        "balanced_accuracy": balanced_accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
        "real_recall": recall_score(labels, predictions, pos_label=0, zero_division=0),
        "fake_recall": recall_score(labels, predictions, pos_label=1, zero_division=0),
    }


def predict(model, loader, device):
    model.eval(); probabilities = []
    with torch.inference_mode():
        for text, image, _ in loader:
            logits = model(text.to(device), image.to(device))
            probabilities.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    probability = np.concatenate(probabilities)
    return probability, (probability >= 0.5).astype(int)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/controlled_fusion"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--modality-dropout", type=float, default=0.3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    return parser.parse_args()


def main():
    args = parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    train_ids, train_y, train_t, train_i = load_features(args.train)
    val_ids, val_y, val_t, val_i = load_features(args.validation)
    test_ids, test_y, test_t, test_i = load_features(args.test)
    text_scaler, image_scaler = StandardScaler(), StandardScaler()
    train_t = text_scaler.fit_transform(train_t); val_t = text_scaler.transform(val_t); test_t = text_scaler.transform(test_t)
    train_i = image_scaler.fit_transform(train_i); val_i = image_scaler.transform(val_i); test_i = image_scaler.transform(test_i)

    def loader(text, image, labels, shuffle=False):
        dataset = TensorDataset(torch.tensor(text, dtype=torch.float32), torch.tensor(image, dtype=torch.float32), torch.tensor(labels, dtype=torch.long))
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = loader(train_t, train_i, train_y, True)
    val_loader = loader(val_t, val_i, val_y)
    test_loader = loader(test_t, test_i, test_y)
    class_counts = np.bincount(train_y, minlength=2)
    weights = torch.tensor(len(train_y) / (2 * class_counts), dtype=torch.float32, device=device)
    rows = []
    conditions = ["text_only", "image_only", "normal_fusion", "modality_dropout"]

    for seed in args.seeds:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        for condition in conditions:
            model = Classifier(train_t.shape[1], train_i.shape[1], condition, args.hidden_size, args.dropout).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
            criterion = nn.CrossEntropyLoss(weight=weights)
            best_state, best_f1, stale = None, -1.0, 0
            for epoch in range(1, args.epochs + 1):
                model.train()
                for text, image, labels in train_loader:
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(text.to(device), image.to(device), args.modality_dropout)
                    loss = criterion(logits, labels.to(device)); loss.backward(); optimizer.step()
                _, val_prediction = predict(model, val_loader, device)
                val_f1 = f1_score(val_y, val_prediction, average="macro", zero_division=0)
                if val_f1 > best_f1:
                    best_f1, best_state, stale = val_f1, copy.deepcopy(model.state_dict()), 0
                else:
                    stale += 1
                    if stale >= args.patience:
                        break
            model.load_state_dict(best_state)
            probability, prediction = predict(model, test_loader, device)
            result = {"condition": condition, "seed": seed, "validation_macro_f1": best_f1, **metrics(test_y, prediction)}
            rows.append(result); print(result)
            pd.DataFrame({"sample_id": test_ids, "true_label": test_y, "prediction": prediction, "probability": probability}).to_csv(args.output_dir / f"{condition}_seed_{seed}_predictions.csv", index=False)
            torch.save(model.state_dict(), args.output_dir / f"{condition}_seed_{seed}.pth")

    raw = pd.DataFrame(rows)
    raw.to_csv(args.output_dir / "all_runs.csv", index=False)
    summary = raw.groupby("condition").agg({metric: ["mean", "std"] for metric in ["accuracy", "balanced_accuracy", "macro_f1", "real_recall", "fake_recall"]})
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.reset_index().to_csv(args.output_dir / "summary.csv", index=False)
    metadata = {"train_samples": len(train_y), "validation_samples": len(val_y), "test_samples": len(test_y), "seeds": args.seeds, "modality_dropout_probability": args.modality_dropout}
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
