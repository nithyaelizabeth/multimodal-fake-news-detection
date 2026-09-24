import pandas as pd

from predict_bert import predict as predict_bert
from predict_clip import predict as predict_clip
from predict_vilt import predict as predict_vilt
from predict_blip_caption_bert import predict as predict_blip


DATA_PATH = "data/processed/multimodal_test.csv"


def main():
    df = pd.read_csv(DATA_PATH)

    row_number = int(input(f"Choose row number (0-{len(df) - 1}): "))
    sample = df.iloc[row_number]

    text = str(sample["text"])
    image = str(sample["image"])
    true_label_number = int(sample["label"])
    true_label = "Fake" if true_label_number == 1 else "Real"

    print("\nText:", text)
    print("Image:", image)
    print("True label:", true_label)

    bert_label, bert_confidence = predict_bert(text)
    clip_label, clip_confidence = predict_clip(text, image)
    vilt_label, vilt_confidence = predict_vilt(text, image)
    blip_label, blip_confidence, caption = predict_blip(text, image)

    print("\nModel comparison")
    print("-" * 50)
    print(f"BERT:       {bert_label:<5} | {bert_confidence * 100:.2f}%")
    print(f"CLIP + MLP: {clip_label:<5} | {clip_confidence * 100:.2f}%")
    print(f"ViLT:       {vilt_label:<5} | {vilt_confidence * 100:.2f}%")
    print(f"BLIP + BERT: {blip_label:<5} | {blip_confidence * 100:.2f}%")
    print(f"Generated caption: {caption}")
    print(f"True label: {true_label}")


if __name__ == "__main__":
    main()