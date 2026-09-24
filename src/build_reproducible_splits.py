"""Build deterministic, leakage-aware Fakeddit data splits.

Rows sharing either normalized text or an image URL are assigned to the same
split. This prevents a model from seeing the same post (or image) during both
training and evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd


DEFAULT_INPUTS = (
    Path("data/processed/multimodal_train.csv"),
    Path("data/processed/multimodal_test.csv"),
)


class DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def normalize_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).lower()).strip()


def stable_fraction(value: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def assign_split(group_id: str, seed: int, train: float, validation: float) -> str:
    number = stable_fraction(group_id, seed)
    if number < train:
        return "train"
    if number < train + validation:
        return "validation"
    return "test"


def read_inputs(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        frame = pd.read_csv(path, usecols=["text", "image", "label"])
        frame["source_file"] = path.name
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    data = data.dropna(subset=["text", "image", "label"]).copy()
    data["text"] = data["text"].map(normalize_text)
    data["image"] = data["image"].astype(str).str.strip()
    data["label"] = data["label"].astype(int)
    data = data[data["label"].isin([0, 1])]
    data = data[(data["text"] != "") & (data["image"] != "")]
    return data.drop_duplicates(subset=["text", "image", "label"]).reset_index(drop=True)


def create_groups(data: pd.DataFrame) -> list[int]:
    sets = DisjointSet(len(data))
    first_text: dict[str, int] = {}
    first_image: dict[str, int] = {}

    for row_id, (text, image) in enumerate(zip(data["text"], data["image"])):
        if text in first_text:
            sets.union(row_id, first_text[text])
        else:
            first_text[text] = row_id
        if image in first_image:
            sets.union(row_id, first_image[image])
        else:
            first_image[image] = row_id

    return [sets.find(row_id) for row_id in range(len(data))]


def verify_no_leakage(data: pd.DataFrame) -> None:
    for column in ("text", "image"):
        split_counts = data.groupby(column, sort=False)["split"].nunique()
        leaked = int((split_counts > 1).sum())
        if leaked:
            raise RuntimeError(f"Leakage check failed: {leaked} {column} values cross splits")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, default=list(DEFAULT_INPUTS))
    parser.add_argument("--output-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    test_ratio = 1.0 - args.train_ratio - args.validation_ratio
    if min(args.train_ratio, args.validation_ratio, test_ratio) <= 0:
        raise ValueError("Train, validation, and test ratios must all be positive")

    data = read_inputs(args.inputs)
    roots = create_groups(data)
    root_keys = {
        root: hashlib.sha256(
            f"{data.at[root, 'text']}\n{data.at[root, 'image']}".encode("utf-8")
        ).hexdigest()
        for root in set(roots)
    }
    data["group_id"] = [root_keys[root] for root in roots]
    data["split"] = data["group_id"].map(
        lambda value: assign_split(value, args.seed, args.train_ratio, args.validation_ratio)
    )
    data.insert(0, "sample_id", [f"fakeddit-{i:07d}" for i in range(len(data))])
    verify_no_leakage(data)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ordered = ["sample_id", "text", "image", "label", "group_id", "source_file"]
    summary: dict[str, object] = {
        "seed": args.seed,
        "ratios": {
            "train": args.train_ratio,
            "validation": args.validation_ratio,
            "test": test_ratio,
        },
        "input_files": [str(path) for path in args.inputs],
        "total_unique_rows": len(data),
        "unique_groups": data["group_id"].nunique(),
        "splits": {},
    }

    for split in ("train", "validation", "test"):
        part = data[data["split"] == split][ordered].reset_index(drop=True)
        part.to_csv(args.output_dir / f"fakeddit_{split}.csv", index=False)
        summary["splits"][split] = {
            "rows": len(part),
            "label_counts": {str(k): int(v) for k, v in part["label"].value_counts().sort_index().items()},
        }

    (args.output_dir / "split_metadata.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
