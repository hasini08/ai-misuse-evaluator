import argparse
import os

import pandas as pd
import numpy as np

from scoring_engine import DEFAULT_WEIGHTS, compute_scores

INPUT = "data/usecases.csv"
OUTPUT = "data/results_scored.csv"
ANALYSIS_TXT = "data/analysis_summary.txt"
SENS_CSV = "data/sensitivity_results.csv"
PARETO_CSV = "data/pareto_frontier.csv"
MC_CSV = "data/montecarlo_stability.csv"


def rank_stability(df_base: pd.DataFrame, df_alt: pd.DataFrame) -> float:
    """
    Spearman rank correlation between base and alternative rankings.
    Implemented without SciPy: Spearman = Pearson correlation of ranks.
    """
    base = df_base[["use_case_name", "justifiability_score"]].copy()
    alt = df_alt[["use_case_name", "justifiability_score"]].copy()
    merged = base.merge(alt, on="use_case_name", suffixes=("_base", "_alt"))

    base_rank = merged["justifiability_score_base"].rank(method="average")
    alt_rank = merged["justifiability_score_alt"].rank(method="average")

    return float(base_rank.corr(alt_rank))


def baseline_comparison(df_raw: pd.DataFrame, scored_default: pd.DataFrame) -> float:
    """
    Compare default scoring vs an equal-weight baseline using rank correlation.
    """
    baseline = compute_scores(
        df_raw,
        w_emissions=0.25,
        w_ethics=0.25,
        w_creativity=0.25,
        w_purpose=0.25
    )
    corr = baseline["justifiability_score"].rank().corr(
        scored_default["justifiability_score"].rank()
    )
    return float(corr)


def advanced_analysis(
    scored: pd.DataFrame,
    df_raw: pd.DataFrame | None = None,
    sensitivity: pd.DataFrame | None = None,
    montecarlo: pd.DataFrame | None = None
) -> str:
    """
    Detailed text report suitable for dissertation evidence.
    Includes: top/bottom cases, component stats, correlations, category breakdown,
    Pareto summary, sensitivity robustness, Monte Carlo stability, baseline comparison.
    """
    lines: list[str] = []
    lines.append("=== Advanced Analysis Summary ===")
    lines.append(f"Number of use cases: {len(scored)}")
    lines.append("")

    srt = scored.sort_values("justifiability_score", ascending=False)

    # Top/bottom
    lines.append("Top 3 use cases (overall score):")
    for _, r in srt.head(3).iterrows():
        lines.append(f"  - {r['use_case_name']}: {r['justifiability_score']} ({r.get('label','')})")
    lines.append("")

    lines.append("Bottom 3 use cases (overall score):")
    for _, r in srt.tail(3).iterrows():
        lines.append(f"  - {r['use_case_name']}: {r['justifiability_score']} ({r.get('label','')})")
    lines.append("")

    # Component statistics + importance
    component_cols = ["component_emissions", "component_ethics", "component_creativity", "component_purpose"]
    contribution_cols = [
        "contribution_emissions",
        "contribution_ethics",
        "contribution_creativity",
        "contribution_purpose",
    ]
    available_components = [column for column in component_cols if column in scored.columns]
    available_contributions = [column for column in contribution_cols if column in scored.columns]

    if available_components:
        lines.append("Component summary (mean ± std):")
        for column in available_components:
            lines.append(f"  - {column}: mean={scored[column].mean():.2f}, std={scored[column].std():.2f}")
        lines.append("")

    if available_contributions:
        lines.append("Weighted contribution summary (mean contribution to final score):")
        for column in available_contributions:
            lines.append(f"  - {column}: mean={scored[column].mean():.2f}")
        lines.append("")

        def driver_text(row):
            comps = {column: float(row[column]) for column in available_contributions}
            strongest = max(comps, key=comps.get)
            weakest = min(comps, key=comps.get)
            return strongest, weakest, comps

        top_row = srt.iloc[0]
        bottom_row = srt.iloc[-1]

        t_strong, t_weak, t_comps = driver_text(top_row)
        b_strong, b_weak, b_comps = driver_text(bottom_row)

        lines.append("Score driver breakdown (best and worst cases):")
        lines.append(f"  Best case: {top_row['use_case_name']}")
        lines.append(f"    strongest component: {t_strong}={t_comps[t_strong]:.2f}")
        lines.append(f"    weakest component:   {t_weak}={t_comps[t_weak]:.2f}")
        lines.append(f"  Worst case: {bottom_row['use_case_name']}")
        lines.append(f"    strongest component: {b_strong}={b_comps[b_strong]:.2f}")
        lines.append(f"    weakest component:   {b_weak}={b_comps[b_weak]:.2f}")
        lines.append("")

    if {"base_score", "guardrail_multiplier", "justifiability_score"}.issubset(scored.columns):
        guarded = scored[scored["guardrail_multiplier"] < 1.0].sort_values("justifiability_score")
        lines.append("Guardrail penalty summary:")
        lines.append(f"  - cases with penalties applied: {len(guarded)}")
        if len(guarded) > 0:
            for _, row in guarded.head(5).iterrows():
                lines.append(
                    f"  - {row['use_case_name']}: base={row['base_score']:.2f}, "
                    f"multiplier={row['guardrail_multiplier']:.2f}, final={row['justifiability_score']:.2f}"
                )
        lines.append("")

    # Correlations (Pearson)
    corr_cols = [
        "emissions_kgco2e",
        "ethical_risk_score",
        "creative_displacement_score",
        "justifiability_score",
    ]
    if all(c in scored.columns for c in corr_cols):
        corr = scored[corr_cols].corr(numeric_only=True)["justifiability_score"].drop("justifiability_score")
        lines.append("Correlation with Justifiability Score (Pearson):")
        for k, v in corr.items():
            lines.append(f"  - {k}: {v:.3f}")
        lines.append("")

    # Group analysis by purpose category
    if "purpose_category" in scored.columns:
        grp = scored.groupby("purpose_category")["justifiability_score"].agg(["count", "mean", "min", "max"]).reset_index()
        lines.append("Scores by purpose_category:")
        for _, r in grp.iterrows():
            lines.append(
                f"  - {r['purpose_category']}: n={int(r['count'])}, mean={r['mean']:.2f}, min={r['min']:.2f}, max={r['max']:.2f}"
            )
        lines.append("")

    # Pareto summary (if present)
    if "pareto_optimal" in scored.columns:
        pareto_n = int(scored["pareto_optimal"].sum())
        lines.append(f"Pareto-optimal solutions: {pareto_n}/{len(scored)}")
        pareto_cases = scored[scored["pareto_optimal"]].sort_values("justifiability_score", ascending=False)
        lines.append("Top Pareto cases:")
        for _, r in pareto_cases.head(5).iterrows():
            lines.append(f"  - {r['use_case_name']}: {r['justifiability_score']}")
        lines.append("")

    # Sensitivity summary
    if sensitivity is not None and len(sensitivity) > 0:
        lines.append("Sensitivity analysis (weight variation):")
        best_stability = float(sensitivity["rank_stability_spearman"].max())
        worst_stability = float(sensitivity["rank_stability_spearman"].min())
        lines.append(f"  - rank stability (Spearman): min={worst_stability:.3f}, max={best_stability:.3f}")

        disruptive = sensitivity.sort_values("rank_stability_spearman").iloc[0]
        lines.append(
            f"  - most disruptive setting: w_emissions={disruptive['w_emissions']} "
            f"(stability={disruptive['rank_stability_spearman']})"
        )
        lines.append("")

    # Monte Carlo summary
    if montecarlo is not None and len(montecarlo) > 0:
        lines.append("Monte Carlo weight robustness:")
        top_winner = montecarlo.sort_values("win_rate", ascending=False).iloc[0]
        lines.append(
            f"  - most frequent top-ranked case: {top_winner['use_case_name']} "
            f"(win_rate={top_winner['win_rate']})"
        )
        lines.append("  - Top 5 winners:")
        for _, r in montecarlo.sort_values("win_rate", ascending=False).head(5).iterrows():
            lines.append(f"    * {r['use_case_name']}: win_rate={r['win_rate']}")
        lines.append("")

    # Baseline comparison (equal weights)
    if df_raw is not None:
        base_corr = baseline_comparison(df_raw, scored)
        lines.append("Baseline comparison (equal weights vs default):")
        lines.append(f"  - rank correlation (Spearman via ranks): {base_corr:.3f}")
        lines.append("")

    return "\n".join(lines)


def run_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vary emissions weight and re-balance other weights.
    Report mean score + rank stability vs baseline.
    """
    base = compute_scores(df)
    results = []

    default_other = DEFAULT_WEIGHTS["ethics"] + DEFAULT_WEIGHTS["creativity"] + DEFAULT_WEIGHTS["purpose"]

    for w_em in [0.20, 0.30, 0.40, 0.50, 0.60]:
        remaining = 1.0 - w_em
        w_eth = remaining * (DEFAULT_WEIGHTS["ethics"] / default_other)
        w_cre = remaining * (DEFAULT_WEIGHTS["creativity"] / default_other)
        w_pur = remaining * (DEFAULT_WEIGHTS["purpose"] / default_other)

        alt = compute_scores(
            df,
            w_emissions=w_em,
            w_ethics=w_eth,
            w_creativity=w_cre,
            w_purpose=w_pur
        )

        stab = rank_stability(base, alt)
        results.append({
            "w_emissions": w_em,
            "w_ethics": round(w_eth, 4),
            "w_creativity": round(w_cre, 4),
            "w_purpose": round(w_pur, 4),
            "mean_score": round(float(alt["justifiability_score"].mean()), 3),
            "rank_stability_spearman": round(float(stab), 3),
        })

    return pd.DataFrame(results)


def pareto_frontier(scored: pd.DataFrame) -> pd.DataFrame:
    """
    Mark Pareto-optimal use cases (non-dominated).
    Maximises component columns: emissions/ethics/creativity/purpose.
    """
    required = ["component_emissions", "component_ethics", "component_creativity", "component_purpose"]
    for c in required:
        if c not in scored.columns:
            raise ValueError(f"Pareto analysis requires column: {c}")

    data = scored[required].to_numpy()
    is_pareto = np.ones(data.shape[0], dtype=bool)

    for i in range(data.shape[0]):
        if not is_pareto[i]:
            continue
        # dominated if another point >= in all and > in at least one
        dominated = np.any(
            (np.all(data >= data[i], axis=1) & np.any(data > data[i], axis=1))
            & (np.arange(data.shape[0]) != i)
        )
        is_pareto[i] = not dominated

    out = scored.copy()
    out["pareto_optimal"] = is_pareto
    return out


def monte_carlo_weight_stability(df: pd.DataFrame, iterations: int = 200) -> pd.DataFrame:
    """
    Randomly sample weights (Dirichlet) and record top-ranked use case frequency.
    """
    winners = []

    for _ in range(iterations):
        w = np.random.dirichlet(np.ones(4), size=1)[0]
        w_em, w_eth, w_cre, w_pur = map(float, w)

        alt = compute_scores(
            df,
            w_emissions=w_em,
            w_ethics=w_eth,
            w_creativity=w_cre,
            w_purpose=w_pur
        )

        top_case = alt.sort_values("justifiability_score", ascending=False).iloc[0]["use_case_name"]
        winners.append(top_case)

    counts = pd.Series(winners).value_counts()
    out = pd.DataFrame({
        "use_case_name": counts.index,
        "win_count": counts.values,
        "win_rate": (counts.values / iterations).round(3)
    })
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=INPUT)
    parser.add_argument("--output", default=OUTPUT)
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--sensitivity", action="store_true")
    parser.add_argument("--pareto", action="store_true")
    parser.add_argument("--montecarlo", action="store_true")
    parser.add_argument("--mc_iters", type=int, default=200)
    args = parser.parse_args()

    os.makedirs("data", exist_ok=True)

    # Load dataset
    df = pd.read_csv(args.input)

    # Core scoring
    scored = compute_scores(df)

    # Optional Pareto analysis
    if args.pareto:
        scored = pareto_frontier(scored)
        scored.to_csv(PARETO_CSV, index=False)
        print(f"✅ Pareto frontier written to {PARETO_CSV}")

    # Always write main scored file
    scored.to_csv(args.output, index=False)

    print("\n✅ Pipeline complete")
    print("Output:", args.output)
    print(scored[["use_case_name", "justifiability_score", "label"]].sort_values("justifiability_score", ascending=False))

    # Optional Sensitivity
    sens_df = None
    if args.sensitivity:
        sens_df = run_sensitivity(df)
        sens_df.to_csv(SENS_CSV, index=False)
        print(f"\n✅ Sensitivity results written to {SENS_CSV}")
        print(sens_df)

    # Optional Monte Carlo
    mc_df = None
    if args.montecarlo:
        mc_df = monte_carlo_weight_stability(df, iterations=args.mc_iters)
        mc_df.to_csv(MC_CSV, index=False)
        print(f"\n✅ Monte Carlo stability written to {MC_CSV}")
        print(mc_df.head(10))

    # Advanced Analysis Report
    if args.analyze:
        report = advanced_analysis(scored, df_raw=df, sensitivity=sens_df, montecarlo=mc_df)
        with open(ANALYSIS_TXT, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n✅ Advanced analysis written to {ANALYSIS_TXT}")

        # also print baseline correlation to console (nice for meetings)
        base_corr = baseline_comparison(df, scored)
        print(f"\nBaseline vs Default Rank Correlation: {base_corr:.3f}")


if __name__ == "__main__":
    main()
