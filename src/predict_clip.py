from io import BytesIO
from functools import lru_cache
from pathlib import Path

import requests
import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


MODEL_NAME = "openai/clip-vit-base-patch32"
WEIGHTS_PATH = Path("models/clip_fakeddit/clip_fake_news_classifier.pth")


class CLIPFakeNewsClassifier(nn.Module):
    def __init__(self, clip_model):
        super().__init__()

        self.clip = clip_model
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2)
        )

    def forward(self, input_ids, attention_mask, pixel_values):
        outputs = self.clip(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values
        )

        combined_features = torch.cat(
            (outputs.text_embeds, outputs.image_embeds),
            dim=1
        )

        return self.classifier(combined_features)


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


def load_classifier(device):
    clip_base = CLIPModel.from_pretrained(MODEL_NAME)

    classifier = CLIPFakeNewsClassifier(clip_base)

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
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    classifier = load_classifier(device)
    return processor, classifier, device


def predict(text, image_source):
    processor, classifier, device = load_resources()
    image = load_image(image_source)

    inputs = processor(
        text=[text],
        images=image,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        logits = classifier(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs["pixel_values"]
        )

        probabilities = torch.softmax(logits, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()

    label = "Fake" if prediction == 1 else "Real"
    confidence = probabilities[0, prediction].item()

    return label, confidence


if __name__ == "__main__":
    news_text = input("Enter news text: ")
    image_source = input("Enter image URL or local image path: ")

    label, confidence = predict(news_text, image_source)

    print("\nPrediction:", label)
    print(f"Confidence: {confidence * 100:.2f}%")
