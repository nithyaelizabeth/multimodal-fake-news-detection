from pathlib import Path
import pandas as pd


TABLES = Path("results/tables")
OUTPUT = TABLES / "final_research_summary.csv"

performance = pd.read_csv(TABLES / "model_comparison.csv")
class_errors = pd.read_csv(TABLES / "class_error_analysis.csv")
name_map = {
    "BLIP + BERT": "BLIP Caption + BERT",
    "BLIP+BERT": "BLIP Caption + BERT"
}

performance["Model"] = performance["Model"].replace(name_map)
class_errors["Model"] = class_errors["Model"].replace(name_map)
summary = performance.merge(
    class_errors[
        [
            "Model",
            "Real Recall",
            "Fake Recall",
            "Macro F1",
            "False Fake",
            "False Real"
        ]
    ],
    on="Model",
    how="left"
)

summary["Main Finding"] = summary["Model"].map({
    "BERT": "Best and most balanced classifier",
    "CLIP + MLP": "High real recall but misses many fake posts",
    "ViLT": "Best multimodal classifier",
    "BLIP Caption + BERT": "Strong real-class bias"
})

summary.to_csv(OUTPUT, index=False)

print(summary.to_string(index=False))
print("\nSaved:", OUTPUT)