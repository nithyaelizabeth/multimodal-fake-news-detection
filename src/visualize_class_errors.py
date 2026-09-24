import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/tables/class_error_analysis.csv")

ax = df.set_index("Model")[["Real Recall", "Fake Recall"]].plot(
    kind="bar",
    figsize=(10, 5)
)

plt.title("Class-Specific Recall by Model")
plt.ylabel("Recall")
plt.ylim(0, 1)
plt.xticks(rotation=15)

for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", padding=3)

plt.tight_layout()
plt.savefig(
    "results/figures/class_specific_recall.png",
    dpi=300
)
plt.show()