import os

import matplotlib.pyplot as plt
import pandas as pd


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
    plt.close()


def plot_score_distribution(df: pd.DataFrame):
    plt.figure(figsize=(8, 4))
    plt.hist(df["justifiability_score"], bins=10)
    plt.xlabel("Justifiability Score (0–100)")
    plt.ylabel("Count")
    plt.title("Distribution of Justifiability Scores")
    plt.tight_layout()
    plt.savefig("figures/score_distribution.png", dpi=200)
    plt.close()


def plot_correlation_matrix(df: pd.DataFrame):
    # Include transparency_score if present
    candidate_cols = [
        "emissions_kgco2e", "ethical_risk_score",
        "creative_displacement_score", "transparency_score",
        "justifiability_score",
    ]
    cols = [c for c in candidate_cols if c in df.columns]
    short_labels = {
        "emissions_kgco2e": "Emissions",
        "ethical_risk_score": "Ethics",
        "creative_displacement_score": "Creativity",
        "transparency_score": "Transparency",
        "justifiability_score": "Score",
    }
    labels = [short_labels.get(c, c) for c in cols]
    corr = df[cols].corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr, interpolation="nearest", vmin=-1, vmax=1, cmap="RdYlGn")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(labels, fontsize=9)
    # Annotate cells with r values
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Correlation Matrix (5 Dimensions + Score)")
    plt.tight_layout()
    plt.savefig("figures/correlation_matrix.png", dpi=200)
    plt.close()


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
    plt.close()


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
    plt.close()


def plot_component_breakdown(df: pd.DataFrame):
    """
    Grouped bar chart showing mean component score for each of the 5 dimensions,
    split by purpose category (essential / beneficial / low_benefit / harmful).
    Shows clearly how transparency varies across use case types.
    """
    comp_cols = [
        ("component_emissions",   "Emissions"),
        ("component_ethics",      "Ethics"),
        ("component_creativity",  "Creativity"),
        ("component_purpose",     "Purpose"),
        ("component_transparency","Transparency"),
    ]
    comp_cols = [(c, l) for c, l in comp_cols if c in df.columns]
    if not comp_cols or "purpose_category" not in df.columns:
        return

    categories = ["essential", "beneficial", "low_benefit", "harmful"]
    categories = [c for c in categories if c in df["purpose_category"].values]
    dim_labels  = [l for _, l in comp_cols]
    dim_cols    = [c for c, _ in comp_cols]

    x = range(len(dim_labels))
    width = 0.18
    colours = ["#4ade80", "#60a5fa", "#f4a261", "#e63946"]

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, cat in enumerate(categories):
        sub = df[df["purpose_category"] == cat]
        means = [sub[col].mean() for col in dim_cols]
        offset = (i - len(categories) / 2 + 0.5) * width
        bars = ax.bar([xi + offset for xi in x], means, width,
                      label=cat, color=colours[i], alpha=0.85)

    ax.set_xticks(list(x))
    ax.set_xticklabels(dim_labels, fontsize=10)
    ax.set_ylabel("Mean Component Score (0–100)")
    ax.set_title("Mean Component Score by Dimension and Purpose Category")
    ax.legend(title="Purpose", fontsize=8)
    ax.set_ylim(0, 110)
    ax.axhline(50, color="grey", linewidth=0.5, linestyle="--")
    plt.tight_layout()
    plt.savefig("figures/component_breakdown_by_category.png", dpi=200)
    plt.close()


def plot_transparency_distribution(df: pd.DataFrame):
    """
    Bar chart of transparency score counts (1–5) coloured by purpose category.
    Shows how transparency is distributed across the dataset.
    """
    if "transparency_score" not in df.columns:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Left: distribution of raw transparency_score values
    ax = axes[0]
    counts = df["transparency_score"].value_counts().sort_index()
    ax.bar(counts.index, counts.values, color="#a78bfa", edgecolor="white")
    ax.set_xlabel("Transparency Score (1–5)")
    ax.set_ylabel("Number of Use Cases")
    ax.set_title("Distribution of Transparency Scores")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xticklabels(["1\n(Opaque)", "2", "3", "4", "5\n(Explainable)"])

    # Right: mean transparency score per purpose category
    ax2 = axes[1]
    if "purpose_category" in df.columns:
        grp = df.groupby("purpose_category")["transparency_score"].mean().reindex(
            ["essential", "beneficial", "low_benefit", "harmful"]
        ).dropna()
        colours = ["#4ade80", "#60a5fa", "#f4a261", "#e63946"]
        ax2.bar(grp.index, grp.values, color=colours[:len(grp)], edgecolor="white")
        ax2.set_ylabel("Mean Transparency Score")
        ax2.set_title("Mean Transparency Score by Purpose Category")
        ax2.set_ylim(0, 5.5)
        ax2.axhline(3, color="grey", linewidth=0.5, linestyle="--", label="Neutral (3)")
        ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("figures/transparency_distribution.png", dpi=200)
    plt.close()


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
    plt.close()


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
    plot_component_breakdown(df)
    plot_transparency_distribution(df)


if __name__ == "__main__":
    main()