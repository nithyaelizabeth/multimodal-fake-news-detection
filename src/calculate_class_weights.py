import pandas as pd
import torch
from sklearn.utils.class_weight import compute_class_weight


df = pd.read_csv("data/processed/multimodal_train.csv")

import numpy as np

classes = np.array(sorted(df["label"].unique()))

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=df["label"]
)

class_weights = torch.tensor(
    weights,
    dtype=torch.float
)

print("Class counts:")
print(df["label"].value_counts().sort_index())

print("\nClass weights:")
for label, weight in zip(classes, weights):
    name = "Real" if label == 0 else "Fake"
    print(f"{name}: {weight:.4f}")