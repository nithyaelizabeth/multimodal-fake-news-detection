import pandas as pd
from sklearn.metrics import accuracy_score


def analyze(path, changed_column, group_column):
    df = pd.read_csv(path)
    rows = []

    for (model, change_type), group in df.groupby(
        ["model", group_column]
    ):
        rows.append({
            "Model": model,
            "Change": change_type,
            "Samples": len(group),
            "Original Accuracy": accuracy_score(
                group["true_label"],
                group["original_prediction"]
            ),
            "Changed Accuracy": accuracy_score(
                group["true_label"],
                group[changed_column]
            ),
            "Prediction Change Rate": (
                group["prediction_changed"].mean() * 100
            )
        })

    return pd.DataFrame(rows)
text_results = analyze(
    "results/tables/robustness_results.csv",
    "perturbed_prediction",
    "perturbation"
)

text_results.to_csv(
    "results/tables/text_robustness_accuracy.csv",
    index=False
)

print("\nText robustness:")
print(text_results.to_string(index=False))
visual_results = analyze(
    "results/tables/visual_robustness_results.csv",
    "transformed_prediction",
    "transformation"
)

visual_results.to_csv(
    "results/tables/visual_robustness_accuracy.csv",
    index=False
)

print("\nVisual robustness:")
print(visual_results.to_string(index=False))