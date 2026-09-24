from io import BytesIO
from functools import lru_cache
from pathlib import Path

import requests
import torch
import torch.nn as nn
from PIL import Image
from transformers import ViltModel, ViltProcessor


MODEL_NAME = "dandelin/vilt-b32-mlm"
WEIGHTS_PATH = Path("models/vilt_fakeddit/vilt_fake_news_classifier.pth")


class ViLTFakeNewsClassifier(nn.Module):
    def __init__(self, vilt_base):
        super().__init__()

        self.vilt = vilt_base

        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        pixel_values,
        pixel_mask
    ):
        outputs = self.vilt(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            pixel_values=pixel_values,
            pixel_mask=pixel_mask
        )

        return self.classifier(outputs.pooler_output)


def load_image(image_source):
    if image_source.startswith(("http://", "https://")):
        response = requests.get(
            image_source,
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_source).convert("RGB")

    return image.resize((384, 384))


def load_classifier(device):
    vilt_base = ViltModel.from_pretrained(MODEL_NAME)
    classifier = ViLTFakeNewsClassifier(vilt_base)

    state_dict = torch.load(
        WEIGHTS_PATH,
        map_location=device,
        weights_only=True
    )

    classifier.load_state_dict(state_dict)
    classifier.to(device)
    classifier.eval()

    return classifier


@lru_cache(maxsize=1)
def load_resources():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = ViltProcessor.from_pretrained(MODEL_NAME)
    classifier = load_classifier(device)
    return processor, classifier, device


def predict(text, image_source):
    processor, classifier, device = load_resources()
    image = load_image(image_source)

    max_length = getattr(processor.tokenizer, "model_max_length", 40)
    max_length = min(max_length, 40)
    inputs = processor(
        image,
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        logits = classifier(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs["token_type_ids"],
            pixel_values=inputs["pixel_values"],
            pixel_mask=inputs["pixel_mask"]
        )

        probabilities = torch.softmax(logits, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()

    label = "Fake" if prediction == 1 else "Real"
    confidence = probabilities[0, prediction].item()

    return label, confidence


if __name__ == "__main__":
    text = input("Enter news text: ")
    image_source = input("Enter image URL or local image path: ")

    label, confidence = predict(text, image_source)

    print("\nPrediction:", label)
    print(f"Confidence: {confidence * 100:.2f}%")
