from pathlib import Path
import pandas as pd


INPUT_PATH = Path(
    "data/external/dataexternalverite-repository/VERITE/VERITE.csv"
)

OUTPUT_PATH = Path(
    "data/processed/external_test.csv"
)

IMAGE_ROOT = INPUT_PATH.parent


def main():
    df = pd.read_csv(INPUT_PATH)

    print("Original shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("Labels:", df["label"].value_counts())

    df = df[["caption", "image_path", "label"]].copy()

    df = df.rename(columns={
        "caption": "text",
        "image_path": "image"
    })

    df["label"] = df["label"].str.lower().map({
        "true": 0,
        "truthful": 0,
        "miscaptioned": 1,
        "out-of-context": 1
    })

    df["image"] = df["image"].apply(
        lambda path: str((IMAGE_ROOT / str(path)).resolve())
    )

    df = df.dropna(subset=["text", "image", "label"])
    df["label"] = df["label"].astype(int)

    df["image_exists"] = df["image"].apply(
        lambda path: Path(path).exists()
    )

    print("\nImage availability:")
    print(df["image_exists"].value_counts())

    df = df[df["image_exists"]].drop(
        columns=["image_exists"]
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print("\nFinal shape:", df.shape)
    print(df["label"].value_counts())
    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()