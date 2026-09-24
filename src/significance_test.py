import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar


df = pd.read_csv("results/tables/model_predictions.csv")

models = {
    "BERT": "bert_prediction",
    "CLIP + MLP": "clip_prediction",
    "ViLT": "vilt_prediction",
    "BLIP Caption + BERT": "blip_prediction"
}


def compare_models(name_a, column_a, name_b, column_b):
    correct_a = df[column_a] == df["true_label"]
    correct_b = df[column_b] == df["true_label"]

    a_correct_b_wrong = int((correct_a & ~correct_b).sum())
    a_wrong_b_correct = int((~correct_a & correct_b).sum())

    table = [
        [0, a_correct_b_wrong],
        [a_wrong_b_correct, 0]
    ]

    result = mcnemar(table, exact=True)

    return {
        "Model A": name_a,
        "Model B": name_b,
        "A correct, B wrong": a_correct_b_wrong,
        "A wrong, B correct": a_wrong_b_correct,
        "p-value": result.pvalue,
        "Significant": result.pvalue < 0.05
    }


results = []

model_items = list(models.items())

for i in range(len(model_items)):
    for j in range(i + 1, len(model_items)):
        name_a, column_a = model_items[i]
        name_b, column_b = model_items[j]

        results.append(
            compare_models(name_a, column_a, name_b, column_b)
        )

results_df = pd.DataFrame(results)

results_df.to_csv(
    "results/tables/model_significance_tests.csv",
    index=False
)

print(results_df.to_string(index=False))