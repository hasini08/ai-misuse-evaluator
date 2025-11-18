import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/results_scored.csv")

plt.figure(figsize=(8, 4))
plt.bar(df["name"], df["responsibility_score"])
plt.title("AI Responsibility Score by Use Case")
plt.xlabel("Use Case")
plt.ylabel("Responsibility Score (0–10)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("data/score_plot.png")
plt.show()

