from io import BytesIO
import json

import requests
from PIL import Image, ImageEnhance, ImageFilter
import tempfile
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from predict_clip import predict as predict_clip
from predict_vilt import predict as predict_vilt
from predict_blip_caption_bert import predict as predict_blip


def load_image(url):
    response = requests.get(
        url,
        timeout=15,
        headers={"User-Agent": "Mozilla/5.0"}
    )
    response.raise_for_status()

    return Image.open(
        BytesIO(response.content)
    ).convert("RGB")


def blur_image(image):
    return image.filter(
        ImageFilter.GaussianBlur(radius=3)
    )


def grayscale_image(image):
    return image.convert("L").convert("RGB")


def darken_image(image):
    return ImageEnhance.Brightness(image).enhance(0.5)


def crop_image(image):
    width, height = image.size

    return image.crop((
        width * 0.1,
        height * 0.1,
        width * 0.9,
        height * 0.9
    )).resize((width, height))
DATA_PATH = Path("data/processed/multimodal_test.csv")
OUTPUT_PATH = Path("results/tables/visual_robustness_results.csv")

SAMPLE_SIZE = 200
def label_number(label):
    return 1 if label == "Fake" else 0
def save_temporary_image(image):
    temporary_file = tempfile.NamedTemporaryFile(
        suffix=".jpg",
        delete=False
    )

    temporary_path = Path(temporary_file.name)
    temporary_file.close()

    image.convert("RGB").save(
        temporary_path,
        format="JPEG"
    )

    return temporary_path
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

    transformations = {
        "blur": blur_image,
        "grayscale": grayscale_image,
        "darken": darken_image,
        "crop": crop_image
    }

    records = []
    failures = []
    for row_id, row in tqdm(
        test_subset.iterrows(),
        total=len(test_subset),
        desc="Visual robustness testing"
    ):
        text = str(row["text"])
        image_url = str(row["image"])
        true_label = int(row["label"])

        try:
            original_image = load_image(image_url)

            clip_label, _ = predict_clip(text, image_url)
            vilt_label, _ = predict_vilt(text, image_url)
            blip_label, _, original_caption = predict_blip(
                text,
                image_url
            )

            original_predictions = {
                "CLIP + MLP": label_number(clip_label),
                "ViLT": label_number(vilt_label),
                "BLIP Caption + BERT": label_number(blip_label)
            }

            for transformation_name, transformation in transformations.items():
                transformed_image = transformation(
                    original_image.copy()
                )

                temporary_path = save_temporary_image(
                    transformed_image
                )

                try:
                    changed_clip_label, _ = predict_clip(
                        text,
                        str(temporary_path)
                    )

                    changed_vilt_label, _ = predict_vilt(
                        text,
                        str(temporary_path)
                    )

                    changed_blip_label, _, changed_caption = predict_blip(
                        text,
                        str(temporary_path)
                    )

                    changed_predictions = {
                        "CLIP + MLP": label_number(
                            changed_clip_label
                        ),
                        "ViLT": label_number(
                            changed_vilt_label
                        ),
                        "BLIP Caption + BERT": label_number(
                            changed_blip_label
                        )
                    }

                    for model_name, original_prediction in original_predictions.items():
                        changed_prediction = changed_predictions[
                            model_name
                        ]

                        records.append({
                            "row_id": row_id,
                            "model": model_name,
                            "transformation": transformation_name,
                            "true_label": true_label,
                            "original_prediction": original_prediction,
                            "transformed_prediction": changed_prediction,
                            "prediction_changed": (
                                original_prediction
                                != changed_prediction
                            ),
                            "original_caption": (
                                original_caption
                                if model_name == "BLIP Caption + BERT"
                                else ""
                            ),
                            "transformed_caption": (
                                changed_caption
                                if model_name == "BLIP Caption + BERT"
                                else ""
                            )
                        })

                finally:
                    temporary_path.unlink(
                        missing_ok=True
                    )

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
        "results/tables/visual_robustness_failures.csv", index=False
    )
    coverage = {
        "requested_samples": len(test_subset),
        "successful_samples": int(results_df["row_id"].nunique()) if not results_df.empty else 0,
        "failed_samples": len(failures),
    }
    coverage["coverage_percent"] = 100 * coverage["successful_samples"] / len(test_subset)
    Path("results/tables/visual_robustness_coverage.json").write_text(
        json.dumps(coverage, indent=2), encoding="utf-8"
    )

    if results_df.empty:
        print("No results were produced.")
        return

    summary_df = (
        results_df
        .groupby(["model", "transformation"])["prediction_changed"]
        .agg(["count", "mean"])
        .reset_index()
    )

    summary_df["change_rate_percent"] = (
        summary_df["mean"] * 100
    )

    summary_path = Path(
        "results/tables/visual_robustness_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    print("\nVisual robustness summary:")
    print(summary_df.to_string(index=False))

    print("\nSaved:", OUTPUT_PATH)
    print("Saved:", summary_path)
    print("Coverage:", json.dumps(coverage, indent=2))


if __name__ == "__main__":
    main()
