import os
import pandas as pd
import matplotlib.pyplot as plt


RESULTS = "data/results_scored.csv"
SENS = "data/sensitivity_results.csv"
PARETO = "data/pareto_frontier.csv"
MC = "data/montecarlo_stability.csv"


def plot_scores_top_bottom(df: pd.DataFrame, n: int = 10):
    """Plot top N and bottom N scores for readability when many use cases exist."""
    df_sorted = df.sort_values("justifiability_score", ascending=False)

    top = df_sorted.head(n)
    bottom = df_sorted.tail(n)

    combined = pd.concat([top, bottom]).copy()
    combined["group"] = ["Top"] * len(top) + ["Bottom"] * len(bottom)

    plt.figure(figsize=(10, 8))
    plt.barh(combined["use_case_name"], combined["justifiability_score"])
    plt.xlabel("Justifiability Score (0–100)")
    plt.title(f"Top {n} and Bottom {n} AI Use Cases by Justifiability Score")
    plt.tight_layout()
    plt.savefig("figures/score_top_bottom.png", dpi=200)
    plt.show()


def plot_score_distribution(df: pd.DataFrame):
    plt.figure(figsize=(8, 4))
    plt.hist(df["justifiability_score"], bins=10)
    plt.xlabel("Justifiability Score (0–100)")
    plt.ylabel("Count")
    plt.title("Distribution of Justifiability Scores")
    plt.tight_layout()
    plt.savefig("figures/score_distribution.png", dpi=200)
    plt.show()


def plot_correlation_matrix(df: pd.DataFrame):
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


def plot_sensitivity():
    if not os.path.exists(SENS):
        return
    sens = pd.read_csv(SENS)
    plt.figure(figsize=(7, 4))
    plt.plot(sens["w_emissions"], sens["mean_score"], marker="o")
    plt.xlabel("Emissions Weight")
    plt.ylabel("Mean Justifiability Score")
    plt.title("Sensitivity Analysis: Emissions Weight vs Mean Score")
    plt.tight_layout()
    plt.savefig("figures/sensitivity_weights.png", dpi=200)
    plt.show()


def plot_pareto(scored: pd.DataFrame):
    if "pareto_optimal" not in scored.columns:
        return

    pareto = scored[scored["pareto_optimal"] == True]
    non = scored[scored["pareto_optimal"] == False]

    plt.figure(figsize=(7, 5))
    plt.scatter(non["component_emissions"], non["component_ethics"], label="Non-Pareto")
    plt.scatter(pareto["component_emissions"], pareto["component_ethics"], label="Pareto-optimal")
    plt.xlabel("Emissions Component (0–100)")
    plt.ylabel("Ethics Component (0–100)")
    plt.title("Pareto Frontier (Emissions vs Ethics Components)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/pareto_emissions_vs_ethics.png", dpi=200)
    plt.show()


def plot_montecarlo():
    if not os.path.exists(MC):
        return
    mc = pd.read_csv(MC).sort_values("win_rate", ascending=False).head(10)

    plt.figure(figsize=(8, 4))
    plt.barh(mc["use_case_name"], mc["win_rate"])
    plt.xlabel("Win Rate (Top-ranked frequency)")
    plt.title("Monte Carlo Robustness: Top 10 Most Frequent Winners")
    plt.tight_layout()
    plt.savefig("figures/montecarlo_top10.png", dpi=200)
    plt.show()


def main():
    os.makedirs("figures", exist_ok=True)

    df = pd.read_csv(RESULTS)
    plot_scores_top_bottom(df, n=10)
    plot_score_distribution(df)
    plot_correlation_matrix(df)

    # If pareto file exists, prefer it because it contains pareto_optimal flag
    if os.path.exists(PARETO):
        dfp = pd.read_csv(PARETO)
        plot_pareto(dfp)
    else:
        plot_pareto(df)

    plot_sensitivity()
    plot_montecarlo()


if __name__ == "__main__":
    main()