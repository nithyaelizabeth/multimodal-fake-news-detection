from pathlib import Path
import pandas as pd


TABLES_DIR = Path("results/tables")
OUTPUT_PATH = TABLES_DIR / "cross_dataset_comparison.csv"


fakeddit = pd.read_csv(
    TABLES_DIR / "model_comparison.csv"
)

verite = pd.read_csv(
    TABLES_DIR / "verite_model_comparison.csv"
)

fakeddit["Dataset"] = "Fakeddit"
verite["Dataset"] = "VERITE"

combined = pd.concat(
    [fakeddit, verite],
    ignore_index=True
)

combined["Dataset Order"] = combined["Dataset"].map({
    "Fakeddit": 0,
    "VERITE": 1
})

combined = combined.sort_values(
    ["Model", "Dataset Order"]
).drop(columns="Dataset Order")

combined.to_csv(OUTPUT_PATH, index=False)

print(combined.to_string(index=False))
print("\nSaved:", OUTPUT_PATH)