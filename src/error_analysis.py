import pandas as pd
import pandas as pd


predictions_df = pd.read_csv(
    "results/tables/model_predictions.csv"
)

predictions_df["bert_correct"] = (
    predictions_df["bert_prediction"]
    == predictions_df["true_label"]
)

predictions_df["clip_correct"] = (
    predictions_df["clip_prediction"]
    == predictions_df["true_label"]
)

predictions_df["vilt_correct"] = (
    predictions_df["vilt_prediction"]
    == predictions_df["true_label"]
)
predictions_df["blip_correct"] = (
    predictions_df["blip_prediction"]
    == predictions_df["true_label"]
)


# BERT failed, but at least one multimodal model succeeded.
multimodal_helped = predictions_df[
    (~predictions_df["bert_correct"])
    & (
        predictions_df["clip_correct"]
        | predictions_df["vilt_correct"]
        | predictions_df["blip_correct"]
    )
]

# BERT succeeded, but at least one multimodal model failed.
bert_helped = predictions_df[
    predictions_df["bert_correct"]
    & (
        ~predictions_df["clip_correct"]
        | ~predictions_df["vilt_correct"]
        | ~predictions_df["blip_correct"]
    )
]


print("\nCases where a multimodal model helped:")
print(
    multimodal_helped[
        [
            "row_id",
            "text",
            "true_label",
            "bert_prediction",
            "clip_prediction",
            "vilt_prediction",
            "blip_prediction"
        ]
    ].to_string(index=False)
)


print("\nCases where BERT helped:")
print(
    bert_helped[
        [
            "row_id",
            "text",
            "true_label",
            "bert_prediction",
            "clip_prediction",
            "vilt_prediction",
            "blip_prediction"
        ]
    ].to_string(index=False)
)


multimodal_helped.to_csv(
    "results/tables/multimodal_helped_examples.csv",
    index=False
)

bert_helped.to_csv(
    "results/tables/bert_helped_examples.csv",
    index=False
)

print("\nSaved error-analysis tables.")