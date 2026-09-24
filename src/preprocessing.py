import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


RAW_DATA_PATH = Path("data/raw/fakeddit.csv")
PROCESSED_DIR = Path("data/processed")

TEXT_COLUMN = "clean_title"
LABEL_COLUMN = "2_way_label"
IMAGE_COLUMN = "image_url"


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s.,!?']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading:", RAW_DATA_PATH)
    df = pd.read_csv(RAW_DATA_PATH)

    print("Raw rows:", len(df))
    print("Columns:", df.columns.tolist())

    required_columns = [
        TEXT_COLUMN,
        LABEL_COLUMN,
        IMAGE_COLUMN
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    # Select and rename the required columns.
    df = df[required_columns].copy()

    df = df.rename(columns={
        TEXT_COLUMN: "text",
        LABEL_COLUMN: "label",
        IMAGE_COLUMN: "image"
    })

    # Remove rows missing text, label, or image URL.
    df = df.dropna(
        subset=["text", "label", "image"]
    ).copy()

    df["text"] = df["text"].apply(clean_text)
    df["image"] = df["image"].astype(str).str.strip()
    df["label"] = df["label"].astype(int)

    # Remove empty text and image values.
    df = df[
        (df["text"] != "") &
        (df["image"] != "")
    ].copy()

    # Remove duplicate text-image pairs.
    df = df.drop_duplicates(
        subset=["text", "image"]
    ).reset_index(drop=True)

    df["text_length"] = df["text"].apply(
        lambda text: len(text.split())
    )

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["label"]
    )

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    df.to_csv(
        PROCESSED_DIR / "clean_fakeddit.csv",
        index=False
    )

    train_df.to_csv(
        PROCESSED_DIR / "train.csv",
        index=False
    )

    test_df.to_csv(
        PROCESSED_DIR / "test.csv",
        index=False
    )

    print("\nClean rows:", len(df))
    print("Training rows:", len(train_df))
    print("Testing rows:", len(test_df))

    print("\nLabel distribution:")
    print(df["label"].value_counts().sort_index())

    print("\nSaved columns:")
    print(df.columns.tolist())

    print("\nFiles saved in:", PROCESSED_DIR)


if __name__ == "__main__":
    main()