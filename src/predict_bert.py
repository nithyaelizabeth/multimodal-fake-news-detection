from pathlib import Path
from functools import lru_cache

import torch
from transformers import BertTokenizer, BertForSequenceClassification


MODEL_DIR = Path("models/bert_fakeddit_model")


@lru_cache(maxsize=1)
def load_resources():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
    model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device).eval()
    return tokenizer, model, device


def predict(text):
    tokenizer, model, device = load_resources()

    inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()

    label = "Fake" if prediction == 1 else "Real"
    confidence = probabilities[0][prediction].item()

    return label, confidence


if __name__ == "__main__":
    text = "breaking news government announces new education policy"

    label, confidence = predict(text)

    print("Prediction:", label)
    print("Confidence:", confidence*100, "%")
    
