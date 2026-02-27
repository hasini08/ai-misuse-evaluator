import os
import pandas as pd
import matplotlib.pyplot as plt

RESULTS = "data/results_scored.csv"
SENS = "data/sensitivity_results.csv"

def main():
    os.makedirs("figures", exist_ok=True)

    df = pd.read_csv(RESULTS).sort_values("justifiability_score", ascending=False)

    # Bar chart
    plt.figure(figsize=(10, 5))
    plt.bar(df["use_case_name"], df["justifiability_score"])
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Justifiability Score (0–100)")
    plt.title("Justifiability Score by AI Use Case")
    plt.tight_layout()
    plt.savefig("figures/score_bar.png", dpi=200)
    plt.show()

    # Correlation matrix
    cols = ["emissions_kgco2e", "ethical_risk_score", "creative_displacement_score", "justifiability_score"]
    corr = df[cols].corr(numeric_only=True)

    plt.figure(figsize=(6, 5))
    plt.imshow(corr, interpolation="nearest")
    plt.xticks(range(len(cols)), cols, rotation=45, ha="right")
    plt.yticks(range(len(cols)), cols)
    plt.colorbar()
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig("figures/correlation_matrix.png", dpi=200)
    plt.show()

    # Sensitivity plot (if file exists)
    if os.path.exists(SENS):
        sens = pd.read_csv(SENS)
        plt.figure(figsize=(7, 4))
        plt.plot(sens["w_emissions"], sens["mean_score"], marker="o")
        plt.xlabel("Emissions Weight")
        plt.ylabel("Mean Justifiability Score")
        plt.title("Sensitivity Analysis: Weight vs Mean Score")
        plt.tight_layout()
        plt.savefig("figures/sensitivity_weights.png", dpi=200)
        plt.show()

if __name__ == "__main__":
    main()