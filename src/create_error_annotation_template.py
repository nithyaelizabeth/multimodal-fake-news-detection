"""Create a balanced manual error-analysis annotation template.

This script only reads existing prediction tables. It does not train models or
run inference.

Run manually when ready:
    .venv/bin/python src/create_error_annotation_template.py --total-errors 100
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


MODELS = {
    "BERT": ("bert_prediction", "bert_confidence"),
    "CLIP + MLP": ("clip_prediction", "clip_confidence"),
    "ViLT": ("vilt_prediction", "vilt_confidence"),
    "BLIP Caption + BERT": ("blip_prediction", "blip_confidence"),
}

ERROR_CATEGORIES = (
    "ambiguous_or_insufficient_text",
    "satire_or_humour",
    "misleading_caption",
    "out_of_context_image",
    "incorrect_generated_caption",
    "named_entity_or_event_confusion",
    "external_knowledge_required",
    "dataset_label_ambiguity",
    "image_quality_or_retrieval_problem",
    "real_class_prediction_bias",
    "other",
)


def predicted_class_confidence(prediction: pd.Series, confidence: pd.Series) -> pd.Series:
    # Existing four-model predictions already store confidence in the predicted class.
    return confidence.astype(float)


def allocate_counts(total: int, number_of_models: int) -> list[int]:
    base, remainder = divmod(total, number_of_models)
    return [base + (1 if index < remainder else 0) for index in range(number_of_models)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("results/tables/model_predictions.csv"),
    )
    parser.add_argument("--total-errors", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/tables/error_annotation_template.csv"),
    )
    parser.add_argument(
        "--codebook",
        type=Path,
        default=Path("results/tables/error_annotation_codebook.md"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.total_errors < len(MODELS):
        raise ValueError("--total-errors must be at least the number of models")
    frame = pd.read_csv(args.predictions)
    required = {"row_id", "text", "image", "true_label"}
    for prediction_column, confidence_column in MODELS.values():
        required.update((prediction_column, confidence_column))
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")

    requested_counts = allocate_counts(args.total_errors, len(MODELS))
    selected_frames = []
    for model_index, (model, (prediction_column, confidence_column)) in enumerate(MODELS.items()):
        errors = frame[frame[prediction_column] != frame["true_label"]].copy()
        requested = requested_counts[model_index]
        if len(errors) < requested:
            raise ValueError(f"{model} has only {len(errors)} errors; requested {requested}")

        # Preserve both classes when possible and choose deterministically.
        per_class = requested // 2
        real_errors = errors[errors["true_label"] == 0].sample(
            n=min(per_class, int((errors["true_label"] == 0).sum())), random_state=args.seed
        )
        fake_needed = requested - len(real_errors)
        fake_errors = errors[errors["true_label"] == 1].sample(
            n=min(fake_needed, int((errors["true_label"] == 1).sum())), random_state=args.seed
        )
        selected = pd.concat([real_errors, fake_errors])
        if len(selected) < requested:
            remaining = errors.drop(index=selected.index).sample(
                n=requested - len(selected), random_state=args.seed
            )
            selected = pd.concat([selected, remaining])

        selected_frames.append(pd.DataFrame({
            "annotation_id": [f"E{model_index + 1}-{i + 1:03d}" for i in range(len(selected))],
            "row_id": selected["row_id"].to_numpy(),
            "model": model,
            "text": selected["text"].to_numpy(),
            "image": selected["image"].to_numpy(),
            "generated_caption": selected.get("caption", pd.Series("", index=selected.index)).to_numpy(),
            "true_label": selected["true_label"].astype(int).to_numpy(),
            "prediction": selected[prediction_column].astype(int).to_numpy(),
            "confidence": predicted_class_confidence(
                selected[prediction_column], selected[confidence_column]
            ).to_numpy(),
            "error_category": "",
            "secondary_category": "",
            "explanation": "",
            "image_helpful": "",
            "caption_quality": "",
            "label_ambiguous": "",
            "reviewer_notes": "",
        }))

    template = pd.concat(selected_frames, ignore_index=True)
    template = template.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(args.output, index=False)

    codebook_lines = [
        "# Error Annotation Codebook",
        "",
        f"Annotate {len(template)} errors without looking at aggregate model results.",
        "Use one primary category and, when necessary, one secondary category.",
        "",
        "## Error categories",
        "",
    ]
    codebook_lines.extend(f"- `{category}`" for category in ERROR_CATEGORIES)
    codebook_lines.extend([
        "",
        "## Controlled fields",
        "",
        "- `image_helpful`: yes, no, unclear, or inaccessible",
        "- `caption_quality`: correct, partially_correct, incorrect, irrelevant, or not_applicable",
        "- `label_ambiguous`: yes, no, or unclear",
        "",
        "## Annotation guidance",
        "",
        "Base the category on the likely reason for the model error, not merely on the",
        "predicted class. Keep explanations concise and evidence-based. If an image",
        "cannot be opened, mark it inaccessible rather than guessing its content.",
    ])
    args.codebook.write_text("\n".join(codebook_lines), encoding="utf-8")
    print(f"Saved {len(template)} annotations to {args.output}")
    print(f"Saved codebook to {args.codebook}")


if __name__ == "__main__":
    main()
