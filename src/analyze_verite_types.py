import pandas as pd
from sklearn.metrics import accuracy_score


data = pd.read_csv("data/processed/verite_test.csv")
predictions = pd.read_csv("results/tables/verite_model_predictions.csv")

# row_id contains the original dataframe index.
predictions = predictions.merge(
    data[["verite_type"]],
    left_on="row_id",
    right_index=True,
    how="left"
)

if "verite_type" not in predictions.columns:
    raise KeyError("verite_type column is missing after merge; check that verite_test.csv and verite_model_predictions.csv are aligned.")

models = {
    "BERT": "bert_prediction",
    "CLIP + MLP": "clip_prediction",
    "ViLT": "vilt_prediction",
    "BLIP + BERT": "blip_prediction"
}

results = []

for model, column in models.items():
    for content_type, group in predictions.groupby("verite_type"):
        results.append({
            "Model": model,
            "VERITE Type": content_type,
            "Samples": len(group),
            "Accuracy": accuracy_score(
                group["true_label"],
                group[column]
            )
        })

results_df = pd.DataFrame(results)

results_df.to_csv(
    "results/tables/verite_type_analysis.csv",
    index=False
)

print(results_df.to_string(index=False))