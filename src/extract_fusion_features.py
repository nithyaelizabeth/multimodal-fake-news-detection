"""Extract frozen BERT text and CLIP image features for controlled fusion.

The output contains only rows whose images were successfully loaded, ensuring
that text-only, image-only and fusion models use exactly the same samples.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer, CLIPImageProcessor, CLIPModel


def device_name(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def stratified_sample(frame: pd.DataFrame, maximum: int | None, seed: int) -> pd.DataFrame:
    if maximum is None or maximum >= len(frame):
        return frame.reset_index(drop=True)
    parts = []
    for label, group in frame.groupby("label"):
        count = round(maximum * len(group) / len(frame))
        parts.append(group.sample(n=min(count, len(group)), random_state=seed))
    sampled = pd.concat(parts).sample(frac=1, random_state=seed)
    return sampled.head(maximum).reset_index(drop=True)


def load_image(source: str, timeout: int) -> Image.Image | None:
    try:
        if source.startswith(("http://", "https://")):
            response = requests.get(
                source, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"}
            )
            response.raise_for_status()
            return Image.open(BytesIO(response.content)).convert("RGB")
        return Image.open(source).convert("RGB")
    except Exception:
        return None


def batches(length: int, size: int):
    for start in range(0, length, size):
        yield start, min(start + size, length)


def clip_feature_tensor(output, model: CLIPModel) -> torch.Tensor:
    """Support both old and new Transformers get_image_features APIs."""
    if isinstance(output, torch.Tensor):
        return output
    image_embeds = getattr(output, "image_embeds", None)
    if image_embeds is not None:
        return image_embeds
    pooled = getattr(output, "pooler_output", None)
    if pooled is not None:
        if pooled.shape[-1] != model.config.projection_dim:
            pooled = model.visual_projection(pooled)
        return pooled
    raise TypeError(f"Unsupported CLIP image feature output: {type(output).__name__}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--failures", type=Path, required=True)
    parser.add_argument("--bert-model", default="models/bert_reproducible")
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-workers", type=int, default=8)
    parser.add_argument("--image-timeout", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    frame = stratified_sample(pd.read_csv(args.input), args.max_samples, args.seed)
    required = {"sample_id", "text", "image", "label"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    device = torch.device(device_name(args.device))
    print(f"Rows requested: {len(frame)}; device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
    text_model = AutoModel.from_pretrained(args.bert_model).to(device).eval()
    text_chunks = []
    with torch.inference_mode():
        for start, end in batches(len(frame), args.batch_size):
            encoded = tokenizer(
                frame.iloc[start:end]["text"].fillna("").astype(str).tolist(),
                padding=True, truncation=True, max_length=128, return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            output = text_model(**encoded).last_hidden_state[:, 0, :]
            text_chunks.append(output.float().cpu().numpy())
    text_features = np.concatenate(text_chunks)
    del text_model
    if device.type == "mps":
        torch.mps.empty_cache()

    # Only image features are required. Loading CLIPProcessor also requests
    # text/processor metadata that is absent from some otherwise complete
    # Hugging Face caches, causing an unnecessary network failure.
    try:
        processor = CLIPImageProcessor.from_pretrained(
            args.clip_model, local_files_only=True
        )
        image_model = CLIPModel.from_pretrained(
            args.clip_model, local_files_only=True
        )
    except OSError:
        # Fall back to Hugging Face only when the model is genuinely not cached.
        processor = CLIPImageProcessor.from_pretrained(args.clip_model)
        image_model = CLIPModel.from_pretrained(args.clip_model)
    image_model = image_model.to(device).eval()
    image_features = np.zeros((len(frame), image_model.config.projection_dim), dtype=np.float32)
    image_ok = np.zeros(len(frame), dtype=bool)
    loader = partial(load_image, timeout=args.image_timeout)
    with ThreadPoolExecutor(max_workers=args.image_workers) as pool:
        for start, end in batches(len(frame), args.batch_size):
            images = list(pool.map(loader, frame.iloc[start:end]["image"].astype(str)))
            valid = [(offset, image) for offset, image in enumerate(images) if image is not None]
            if not valid:
                continue
            inputs = processor(images=[item[1] for item in valid], return_tensors="pt")
            pixels = inputs["pixel_values"].to(device)
            with torch.inference_mode():
                features = clip_feature_tensor(
                    image_model.get_image_features(pixel_values=pixels), image_model
                )
                features = torch.nn.functional.normalize(features.float(), dim=1)
            for row, feature in zip((start + item[0] for item in valid), features.cpu().numpy()):
                image_features[row] = feature
                image_ok[row] = True
            print(f"Processed {end}/{len(frame)}; usable images={image_ok[:end].sum()}")

    failed = frame.loc[~image_ok, ["sample_id", "image"]].copy()
    failed["reason"] = "image_load_failed"
    args.failures.parent.mkdir(parents=True, exist_ok=True)
    failed.to_csv(args.failures, index=False)

    keep = image_ok
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        # Explicit Unicode dtype avoids NumPy object arrays and allows the
        # archive to be loaded with allow_pickle=False.
        sample_ids=np.asarray(frame.loc[keep, "sample_id"].astype(str).tolist(), dtype=str),
        labels=frame.loc[keep, "label"].astype(np.int64).to_numpy(),
        text_features=text_features[keep].astype(np.float32),
        image_features=image_features[keep].astype(np.float32),
    )
    print(f"Saved {keep.sum()}/{len(frame)} paired rows to {args.output}")
    print(f"Failures: {args.failures}")


if __name__ == "__main__":
    main()
