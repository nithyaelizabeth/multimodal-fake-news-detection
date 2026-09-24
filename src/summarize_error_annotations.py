"""Summarize the objective and completed fields in the error annotation sheet.

This script does not infer qualitative error causes. Those fields require a
human reviewer to inspect the text and image.
"""

from pathlib import Path

import pandas as pd


INPUT = Path("results/tables/error_annotation_template.csv")
OUTPUT = Path("results/tables/error_annotation_progress.csv")


def main() -> None:
    frame = pd.read_csv(INPUT)
    required = {"model", "true_label", "prediction", "confidence", "error_category"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    frame["error_direction"] = frame.apply(
        lambda row: "Real predicted as Fake"
        if int(row["true_label"]) == 0 and int(row["prediction"]) == 1
        else "Fake predicted as Real",
        axis=1,
    )

    rows = []
    for (model, direction), group in frame.groupby(["model", "error_direction"]):
        rows.append(
            {
                "Model": model,
                "Error direction": direction,
                "Selected errors": len(group),
                "Mean confidence": group["confidence"].mean(),
                "High-confidence errors (>=0.90)": int((group["confidence"] >= 0.90).sum()),
                "Manually annotated": int(group["error_category"].notna().sum()),
            }
        )

    summary = pd.DataFrame(rows).sort_values(["Model", "Error direction"])
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT, index=False)
    print(summary.to_string(index=False))
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
