import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

raw_path = Path("data/raw/multimodal_train.tsv")
output_dir = Path("data/processed")

# Read TSV with explicit tab separator to avoid parser errors
if raw_path.suffix == ".tsv":
    df = pd.read_csv(raw_path, sep="\t")
else:
    df = pd.read_csv(raw_path)

print("Raw columns:", df.columns.tolist())

df = df[["clean_title", "image_url", "2_way_label"]].copy()

df = df.rename(columns={
    "clean_title": "text",
    "image_url": "image",
    "2_way_label": "label"
})

df = df.dropna(subset=["text", "image", "label"])
df = df[df["image"].astype(str).str.startswith(("http://", "https://"))]

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

output_dir.mkdir(parents=True, exist_ok=True)

train_df.to_csv(output_dir / "multimodal_train.csv", index=False)
test_df.to_csv(output_dir / "multimodal_test.csv", index=False)

print("Saved columns:", test_df.columns.tolist())
print("Test rows:", len(test_df))