from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


DATA_PATH = Path("data/processed/clean_fakeddit.csv")
FIGURES_DIR = Path("results/figures")
TABLES_DIR = Path("results/tables")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    print("Dataset shape:", df.shape)
    print("Columns:", df.columns.tolist())

    label_counts = df["label"].value_counts().sort_index()
    label_percent = df["label"].value_counts(normalize=True).sort_index() * 100

    label_summary = pd.DataFrame({
        "count": label_counts,
        "percentage": label_percent
    })

    print("\nLabel distribution:")
    print(label_summary)

    label_summary.to_csv(TABLES_DIR / "label_distribution.csv")

    plt.figure(figsize=(6, 4))
    plt.bar(["Real", "Fake"], label_counts.values)
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Number of Samples")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "class_distribution.png", dpi=300)
    plt.close()

    df["text_length"] = df["text"].astype(str).apply(lambda text: len(text.split()))

    print("\nText length summary:")
    print(df["text_length"].describe())

    df["text_length"].describe().to_csv(TABLES_DIR / "text_length_summary.csv")

    plt.figure(figsize=(8, 4))
    plt.hist(df["text_length"], bins=50)
    plt.title("Text Length Distribution")
    plt.xlabel("Number of Words")
    plt.ylabel("Number of Samples")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "text_length_distribution.png", dpi=300)
    plt.close()

    if "image" in df.columns:
        image_available = df["image"].notna() & (df["image"].astype(str).str.strip() != "")
        image_counts = image_available.value_counts()

        image_summary = pd.DataFrame({
            "count": image_counts
        })

        image_summary.to_csv(TABLES_DIR / "image_availability.csv")

        print("\nImage availability:")
        print(image_counts)

        plt.figure(figsize=(6, 4))
        plt.bar(["Has image", "No image"], [
            int(image_available.sum()),
            int((~image_available).sum())
        ])
        plt.title("Image Availability")
        plt.xlabel("Image Status")
        plt.ylabel("Number of Samples")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "image_availability.png", dpi=300)
        plt.close()

    print("\nEDA completed.")
    print(f"Figures saved in: {FIGURES_DIR}")
    print(f"Tables saved in: {TABLES_DIR}")


if __name__ == "__main__":
    main()