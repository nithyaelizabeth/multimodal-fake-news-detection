from functools import lru_cache
from io import BytesIO
from pathlib import Path

import requests
import torch
from PIL import Image
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    BlipForConditionalGeneration,
    BlipProcessor
)


CAPTION_MODEL_NAME = "Salesforce/blip-image-captioning-base"
CLASSIFIER_DIR = Path("models/blip_caption_bert")


def load_image(image_source):
    if image_source.startswith(("http://", "https://")):
        response = requests.get(
            image_source,
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")

    return Image.open(image_source).convert("RGB")


@lru_cache(maxsize=1)
def load_resources():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    blip_processor = BlipProcessor.from_pretrained(
        CAPTION_MODEL_NAME
    )

    blip_model = BlipForConditionalGeneration.from_pretrained(
        CAPTION_MODEL_NAME
    ).to(device)

    classifier_tokenizer = BertTokenizer.from_pretrained(
        CLASSIFIER_DIR
    )

    classifier = BertForSequenceClassification.from_pretrained(
        CLASSIFIER_DIR
    ).to(device)

    blip_model.eval()
    classifier.eval()

    return (
        blip_processor,
        blip_model,
        classifier_tokenizer,
        classifier,
        device
    )


def generate_caption(image, processor, model, device):
    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=50
        )

    return processor.decode(
        output[0],
        skip_special_tokens=True
    )


def predict(text, image_source, existing_caption=None):
    (
        blip_processor,
        blip_model,
        tokenizer,
        classifier,
        device
    ) = load_resources()

    caption = existing_caption

    if caption is None:
        image = load_image(image_source)
        caption = generate_caption(
            image,
            blip_processor,
            blip_model,
            device
        )

    combined_text = (
        str(text)
        + " [IMAGE CAPTION] "
        + str(caption)
    )

    inputs = tokenizer(
        combined_text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = classifier(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)
        prediction = torch.argmax(
            probabilities,
            dim=1
        ).item()

    label = "Fake" if prediction == 1 else "Real"
    confidence = probabilities[0, prediction].item()

    return label, confidence, caption


if __name__ == "__main__":
    news_text = input("Enter news text: ")
    image_source = input(
        "Enter image URL or local image path: "
    )

    label, confidence, caption = predict(
        news_text,
        image_source
    )

    print("\nGenerated caption:", caption)
    print("Prediction:", label)
    print(f"Confidence: {confidence * 100:.2f}%")