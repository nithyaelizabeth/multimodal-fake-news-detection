import pandas as pd


verite_data = pd.read_csv(
    "data/processed/verite_test.csv"
)

predictions = pd.read_csv(
    "results/tables/verite_model_predictions.csv"
)

predictions = predictions.merge(
    verite_data[["verite_type"]],
    left_on="row_id",
    right_index=True,
    how="left"
)

predictions.to_csv(
    "results/tables/verite_model_predictions_with_type.csv",
    index=False
)

print("Saved: results/tables/verite_model_predictions_with_type.csv")
print(predictions[["row_id", "true_label", "verite_type"]].head())