import pandas as pd


def analyze(path, prediction_column, model_name):
    df = pd.read_csv(path)

    counts = df[prediction_column].value_counts().sort_index()
    percentages = (
        df[prediction_column].value_counts(normalize=True).sort_index() * 100
    )

    print(f"\n{model_name}")
    print("Real predictions:", counts.get(0, 0))
    print("Fake predictions:", counts.get(1, 0))
    print("Real percentage:", percentages.get(0, 0))
    print("Fake percentage:", percentages.get(1, 0))


df_path = "results/tables/model_predictions.csv"

analyze(df_path, "bert_prediction", "BERT")
analyze(df_path, "clip_prediction", "CLIP + MLP")
analyze(df_path, "vilt_prediction", "ViLT")
analyze(df_path, "blip_prediction", "BLIP Caption + BERT")
import pandas as pd

df = pd.read_csv("results/tables/model_predictions.csv")

columns = [
    "true_label",
    "bert_prediction",
    "clip_prediction",
    "vilt_prediction",
    "blip_prediction"
]

for column in columns:
    counts = df[column].value_counts().sort_index()
    percentages = df[column].value_counts(normalize=True).sort_index() * 100

    print(f"\n{column}")
    print("Real:", counts.get(0, 0), f"({percentages.get(0, 0):.2f}%)")
    print("Fake:", counts.get(1, 0), f"({percentages.get(1, 0):.2f}%)")