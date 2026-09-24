"""Train and evaluate a reproducible BERT baseline on the shared manifests.

Example pilot run:
    .venv/bin/python src/train_bert_reproducible.py \
        --max-train-samples 20000 \
        --max-validation-samples 5000 \
        --max-test-samples 5000 \
        --epochs 3
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.optimization import Adafactor


class TextDataset(Dataset):
    def __init__(self, frame: pd.DataFrame) -> None:
        self.sample_ids = frame["sample_id"].astype(str).tolist()
        self.texts = frame["text"].astype(str).tolist()
        self.labels = frame["label"].astype(int).tolist()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, object]:
        return {
            "sample_id": self.sample_ids[index],
            "text": self.texts[index],
            "label": self.labels[index],
        }


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def stratified_sample(frame: pd.DataFrame, maximum: int | None, seed: int) -> pd.DataFrame:
    if maximum is None or maximum >= len(frame):
        return frame.reset_index(drop=True)
    fractions = frame["label"].value_counts(normalize=True)
    pieces = []
    remaining = maximum
    labels = sorted(frame["label"].unique())
    for index, label in enumerate(labels):
        available = frame[frame["label"] == label]
        count = remaining if index == len(labels) - 1 else round(maximum * fractions[label])
        count = min(count, len(available))
        pieces.append(available.sample(n=count, random_state=seed))
        remaining -= count
    return pd.concat(pieces).sample(frac=1, random_state=seed).reset_index(drop=True)


def make_collator(tokenizer, max_length: int):
    def collate(rows: list[dict[str, object]]) -> dict[str, object]:
        encoded = tokenizer(
            [str(row["text"]) for row in rows],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor([int(row["label"]) for row in rows])
        encoded["sample_ids"] = [str(row["sample_id"]) for row in rows]
        return encoded

    return collate


def class_weights(frame: pd.DataFrame, device: torch.device) -> torch.Tensor:
    counts = frame["label"].value_counts().reindex([0, 1], fill_value=0).to_numpy()
    if (counts == 0).any():
        raise ValueError("Training data must contain both labels")
    weights = len(frame) / (2.0 * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def evaluate(model, loader: DataLoader, device: torch.device, loss_function) -> dict[str, object]:
    model.eval()
    labels: list[int] = []
    predictions: list[int] = []
    probabilities: list[float] = []
    sample_ids: list[str] = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            batch_labels = batch.pop("labels").to(device)
            batch_ids = batch.pop("sample_ids")
            inputs = {name: value.to(device) for name, value in batch.items()}
            logits = model(**inputs).logits
            total_loss += loss_function(logits, batch_labels).item()
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = logits.argmax(dim=1)
            labels.extend(batch_labels.cpu().tolist())
            predictions.extend(preds.cpu().tolist())
            probabilities.extend(probs.cpu().tolist())
            sample_ids.extend(batch_ids)

    return {
        "loss": total_loss / max(len(loader), 1),
        "accuracy": accuracy_score(labels, predictions),
        "balanced_accuracy": balanced_accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
        "sample_ids": sample_ids,
        "labels": labels,
        "predictions": predictions,
        "probabilities": probabilities,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("data/splits/fakeddit_train.csv"))
    parser.add_argument("--validation", type=Path, default=Path("data/splits/fakeddit_validation.csv"))
    parser.add_argument("--test", type=Path, default=Path("data/splits/fakeddit_test.csv"))
    parser.add_argument("--model-name", default="bert-base-uncased")
    parser.add_argument("--output-dir", type=Path, default=Path("models/bert_reproducible"))
    parser.add_argument("--predictions", type=Path, default=Path("results/predictions/bert_test.csv"))
    parser.add_argument("--run-summary", type=Path, default=Path("results/tables/bert_training_summary.json"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Accumulate this many mini-batches before an optimizer step",
    )
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument(
        "--optimizer",
        choices=("auto", "adamw", "adafactor"),
        default="auto",
        help="auto uses memory-efficient Adafactor on MPS and AdamW elsewhere",
    )
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--empty-cache-steps",
        type=int,
        default=100,
        help="Release cached accelerator memory after this many optimizer steps; 0 disables it",
    )
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-validation-samples", type=int)
    parser.add_argument("--max-test-samples", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = choose_device(args.device)
    print(f"Device: {device}")

    train_frame = stratified_sample(pd.read_csv(args.train), args.max_train_samples, args.seed)
    validation_frame = stratified_sample(
        pd.read_csv(args.validation), args.max_validation_samples, args.seed
    )
    test_frame = stratified_sample(pd.read_csv(args.test), args.max_test_samples, args.seed)
    print(f"Rows: train={len(train_frame)}, validation={len(validation_frame)}, test={len(test_frame)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
    model.to(device)
    collator = make_collator(tokenizer, args.max_length)
    generator = torch.Generator().manual_seed(args.seed)

    train_loader = DataLoader(
        TextDataset(train_frame), batch_size=args.batch_size, shuffle=True,
        collate_fn=collator, num_workers=args.num_workers, generator=generator,
    )
    validation_loader = DataLoader(
        TextDataset(validation_frame), batch_size=args.batch_size, shuffle=False,
        collate_fn=collator, num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        TextDataset(test_frame), batch_size=args.batch_size, shuffle=False,
        collate_fn=collator, num_workers=args.num_workers,
    )

    loss_function = CrossEntropyLoss(weight=class_weights(train_frame, device))
    if args.gradient_accumulation_steps < 1:
        raise ValueError("--gradient-accumulation-steps must be at least 1")
    optimizer_name = args.optimizer
    if optimizer_name == "auto":
        optimizer_name = "adafactor" if device.type == "mps" else "adamw"
    if optimizer_name == "adafactor":
        optimizer = Adafactor(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
        )
    else:
        # foreach=False avoids a temporary list of parameter-sized tensors.
        optimizer = AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            foreach=False,
        )
    print(f"Optimizer: {optimizer_name}")
    best_f1 = -1.0
    epochs_without_improvement = 0
    history = []
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        optimizer.zero_grad(set_to_none=True)
        optimizer_steps = 0
        for batch_index, batch in enumerate(progress, start=1):
            labels = batch.pop("labels").to(device)
            batch.pop("sample_ids")
            inputs = {name: value.to(device) for name, value in batch.items()}
            logits = model(**inputs).logits
            loss = loss_function(logits, labels)
            (loss / args.gradient_accumulation_steps).backward()

            should_step = (
                batch_index % args.gradient_accumulation_steps == 0
                or batch_index == len(train_loader)
            )
            if should_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                if (
                    device.type == "mps"
                    and args.empty_cache_steps > 0
                    and optimizer_steps % args.empty_cache_steps == 0
                ):
                    torch.mps.empty_cache()
            total_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        validation = evaluate(model, validation_loader, device, loss_function)
        epoch_result = {
            "epoch": epoch,
            "train_loss": total_loss / max(len(train_loader), 1),
            "validation_loss": validation["loss"],
            "validation_accuracy": validation["accuracy"],
            "validation_balanced_accuracy": validation["balanced_accuracy"],
            "validation_macro_f1": validation["macro_f1"],
        }
        history.append(epoch_result)
        print(json.dumps(epoch_result, indent=2))

        if validation["macro_f1"] > best_f1:
            best_f1 = float(validation["macro_f1"])
            epochs_without_improvement = 0
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print("Early stopping")
                break

    # Release the training model and optimizer before loading the best checkpoint.
    # Keeping both copies simultaneously can exhaust unified memory on Apple MPS.
    del optimizer
    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
    best_model = AutoModelForSequenceClassification.from_pretrained(args.output_dir).to(device)
    test = evaluate(best_model, test_loader, device, loss_function)
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "sample_id": test["sample_ids"],
        "true_label": test["labels"],
        "prediction": test["predictions"],
        "probability": test["probabilities"],
    }).to_csv(args.predictions, index=False)

    summary = {
        "model_name": args.model_name,
        "seed": args.seed,
        "device": str(device),
        "optimizer": optimizer_name,
        "class_weighted_loss": True,
        "train_samples": len(train_frame),
        "validation_samples": len(validation_frame),
        "test_samples": len(test_frame),
        "hyperparameters": {
            "epochs_requested": args.epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "max_length": args.max_length,
            "patience": args.patience,
        },
        "history": history,
        "test_metrics": {
            "accuracy": test["accuracy"],
            "balanced_accuracy": test["balanced_accuracy"],
            "macro_f1": test["macro_f1"],
        },
        "predictions": str(args.predictions),
    }
    args.run_summary.parent.mkdir(parents=True, exist_ok=True)
    args.run_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
