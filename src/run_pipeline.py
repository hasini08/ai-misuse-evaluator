"""
run_pipeline.py  —  AI Misuse Evaluator pipeline
=================================================
Orchestrates scoring, sensitivity, Monte Carlo, Pareto, ablation,
threshold sensitivity, and statistical analysis.
 
Usage:
    python src/run_pipeline.py --analyze --sensitivity --pareto --montecarlo --ablation --threshold
"""
 
import argparse
import os
 
import math
 
import numpy as np
import pandas as pd
 
from scoring_engine import DEFAULT_WEIGHTS, compute_scores
 
# ── p-value implementation (scipy optional) ───────────────────────────────────
try:
    from scipy import stats as _scipy_stats
    def _pearson_pvalue(r: float, n: int) -> float:
        """Exact p-value via scipy."""
        _, p = _scipy_stats.pearsonr.__wrapped__ if hasattr(_scipy_stats.pearsonr, '__wrapped__') else (None, None)
        # simpler: recompute via t-distribution
        if abs(r) >= 1.0:
            return 0.0
        t = r * math.sqrt(n - 2) / math.sqrt(1 - r ** 2)
        return float(2 * _scipy_stats.t.sf(abs(t), df=n - 2))
except ImportError:
    def _pearson_pvalue(r: float, n: int) -> float:
        """
        Two-tailed p-value for Pearson r without scipy.
        Uses the t-distribution with df = n-2.
        For df >= 30, the normal approximation error < 0.002.
        """
        if abs(r) >= 1.0:
            return 0.0
        t = r * math.sqrt(n - 2) / math.sqrt(1 - r ** 2)
        # erfc(x/sqrt(2)) = 2*(1 - Phi(x)) — accurate normal approx for df >= 30
        return math.erfc(abs(t) / math.sqrt(2))
 
# ── Output paths ──────────────────────────────────────────────────────────────
INPUT        = "data/usecases.csv"
OUTPUT       = "data/results_scored.csv"
ANALYSIS_TXT = "data/analysis_summary.txt"
SENS_CSV     = "data/sensitivity_results.csv"
PARETO_CSV   = "data/pareto_frontier.csv"
MC_CSV       = "data/montecarlo_stability.csv"
ABLATION_CSV = "data/ablation_results.csv"
THRESH_CSV   = "data/threshold_sensitivity.csv"
 
 
# ════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════
 
def rank_stability(df_base: pd.DataFrame, df_alt: pd.DataFrame) -> float:
    """
    Spearman rank correlation between base and alternative rankings.
    Implemented as Pearson correlation of ranks (mathematically equivalent
    to Spearman's rho) — no SciPy dependency needed here.
    """
    merged = df_base[["use_case_name", "justifiability_score"]].merge(
        df_alt[["use_case_name", "justifiability_score"]],
        on="use_case_name", suffixes=("_base", "_alt")
    )
    return float(
        merged["justifiability_score_base"].rank(method="average").corr(
            merged["justifiability_score_alt"].rank(method="average")
        )
    )
 
 
def pearson_with_pvalue(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    """Return (r, p-value) for Pearson correlation."""
    r = float(np.corrcoef(x.astype(float), y.astype(float))[0, 1])
    p = _pearson_pvalue(r, n=len(x))
    return round(r, 3), round(p, 4)
 
 
# ════════════════════════════════════════════════════════════════════════════
# FIX 1 — SENSITIVITY (5 DIMENSIONS)
# ════════════════════════════════════════════════════════════════════════════
 
def run_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vary the emissions weight from 0.20 to 0.60 in steps of 0.10,
    re-balancing ALL FOUR remaining weights (ethics, creativity, purpose,
    transparency) proportionally.  Previously only rebalanced 3 weights —
    transparency was ignored.  Now correctly uses all five dimensions.
    """
    base = compute_scores(df)
    results = []
 
    # Sum of non-emissions default weights (used to split the remainder fairly)
    other_sum = (
        DEFAULT_WEIGHTS["ethics"]
        + DEFAULT_WEIGHTS["creativity"]
        + DEFAULT_WEIGHTS["purpose"]
        + DEFAULT_WEIGHTS["transparency"]
    )
 
    for w_em in [0.20, 0.30, 0.40, 0.50, 0.60]:
        remaining = 1.0 - w_em
        w_eth = remaining * (DEFAULT_WEIGHTS["ethics"]       / other_sum)
        w_cre = remaining * (DEFAULT_WEIGHTS["creativity"]   / other_sum)
        w_pur = remaining * (DEFAULT_WEIGHTS["purpose"]      / other_sum)
        w_tr  = remaining * (DEFAULT_WEIGHTS["transparency"] / other_sum)
 
        alt = compute_scores(
            df,
            w_emissions=w_em,
            w_ethics=w_eth,
            w_creativity=w_cre,
            w_purpose=w_pur,
            w_transparency=w_tr,
        )
 
        stab = rank_stability(base, alt)
        results.append({
            "w_emissions":            w_em,
            "w_ethics":               round(w_eth, 4),
            "w_creativity":           round(w_cre, 4),
            "w_purpose":              round(w_pur, 4),
            "w_transparency":         round(w_tr,  4),
            "mean_score":             round(float(alt["justifiability_score"].mean()), 3),
            "rank_stability_spearman": round(stab, 3),
        })
 
    return pd.DataFrame(results)
 
 
# ════════════════════════════════════════════════════════════════════════════
# FIX 1 — MONTE CARLO (5 DIMENSIONS)
# ════════════════════════════════════════════════════════════════════════════
 
def monte_carlo_weight_stability(df: pd.DataFrame, iterations: int = 200) -> pd.DataFrame:
    """
    Sample 200 weight vectors from a flat 5-dimensional Dirichlet distribution
    and record which use case ranks first under each.
 
    Previously used Dirichlet(ones(4)) — only 4 dimensions, ignoring transparency.
    Now correctly uses Dirichlet(ones(5)) for all five scoring dimensions.
    """
    winners = []
 
    for _ in range(iterations):
        # 5-dimensional flat Dirichlet: equal prior over all weight combinations
        w = np.random.dirichlet(np.ones(5))
        w_em, w_eth, w_cre, w_pur, w_tr = map(float, w)
 
        alt = compute_scores(
            df,
            w_emissions=w_em,
            w_ethics=w_eth,
            w_creativity=w_cre,
            w_purpose=w_pur,
            w_transparency=w_tr,
        )
 
        top_case = (
            alt.sort_values("justifiability_score", ascending=False)
            .iloc[0]["use_case_name"]
        )
        winners.append(top_case)
 
    counts = pd.Series(winners).value_counts()
    return pd.DataFrame({
        "use_case_name": counts.index,
        "win_count":     counts.values,
        "win_rate":      (counts.values / iterations).round(3),
    })
 
 
# ════════════════════════════════════════════════════════════════════════════
# FIX 2 — ABLATION STUDY
# ════════════════════════════════════════════════════════════════════════════
 
def run_ablation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Leave-one-dimension-out ablation study.
 
    For each of the five dimensions, set its weight to 0 (removing it from
    the score) and re-run scoring.  Record:
      - how many use cases change score band (Low / Medium / High)
      - the Spearman rank stability vs the full model
      - the mean absolute score change
 
    This directly answers: "does each dimension actually add information?"
    A dimension that causes many band changes when removed is genuinely load-
    bearing.  One that causes no changes would be a candidate for pruning.
    """
    base = compute_scores(df)
    base_labels = base.set_index("use_case_name")["label"].astype(str)
 
    # Five configurations: each drops one dimension
    ablation_configs = {
        "drop_emissions":    dict(w_emissions=0.0,  w_ethics=DEFAULT_WEIGHTS["ethics"],
                                   w_creativity=DEFAULT_WEIGHTS["creativity"],
                                   w_purpose=DEFAULT_WEIGHTS["purpose"],
                                   w_transparency=DEFAULT_WEIGHTS["transparency"]),
        "drop_ethics":       dict(w_emissions=DEFAULT_WEIGHTS["emissions"],  w_ethics=0.0,
                                   w_creativity=DEFAULT_WEIGHTS["creativity"],
                                   w_purpose=DEFAULT_WEIGHTS["purpose"],
                                   w_transparency=DEFAULT_WEIGHTS["transparency"]),
        "drop_creativity":   dict(w_emissions=DEFAULT_WEIGHTS["emissions"],  w_ethics=DEFAULT_WEIGHTS["ethics"],
                                   w_creativity=0.0,
                                   w_purpose=DEFAULT_WEIGHTS["purpose"],
                                   w_transparency=DEFAULT_WEIGHTS["transparency"]),
        "drop_purpose":      dict(w_emissions=DEFAULT_WEIGHTS["emissions"],  w_ethics=DEFAULT_WEIGHTS["ethics"],
                                   w_creativity=DEFAULT_WEIGHTS["creativity"],
                                   w_purpose=0.0,
                                   w_transparency=DEFAULT_WEIGHTS["transparency"]),
        "drop_transparency": dict(w_emissions=DEFAULT_WEIGHTS["emissions"],  w_ethics=DEFAULT_WEIGHTS["ethics"],
                                   w_creativity=DEFAULT_WEIGHTS["creativity"],
                                   w_purpose=DEFAULT_WEIGHTS["purpose"],
                                   w_transparency=0.0),
    }
 
    rows = []
    for config_name, weights in ablation_configs.items():
        ablated = compute_scores(df, **weights)
        ablated_labels = ablated.set_index("use_case_name")["label"].astype(str)
 
        # Cases whose band changed when this dimension was removed
        band_changes = int((base_labels != ablated_labels).sum())
 
        # Mean absolute score difference
        merged = base[["use_case_name", "justifiability_score"]].merge(
            ablated[["use_case_name", "justifiability_score"]],
            on="use_case_name", suffixes=("_base", "_ablated")
        )
        mean_abs_change = float(
            (merged["justifiability_score_base"] - merged["justifiability_score_ablated"])
            .abs().mean()
        )
 
        spearman = rank_stability(base, ablated)
 
        rows.append({
            "ablated_dimension":    config_name.replace("drop_", ""),
            "band_changes":         band_changes,
            "pct_band_changes":     round(band_changes / len(df) * 100, 1),
            "mean_abs_score_change": round(mean_abs_change, 2),
            "rank_stability":       round(spearman, 3),
        })
 
    result = pd.DataFrame(rows).sort_values("band_changes", ascending=False)
    return result
 
 
# ════════════════════════════════════════════════════════════════════════════
# FIX 3 — CORRELATIONS WITH p-VALUES
# ════════════════════════════════════════════════════════════════════════════
 
def compute_correlations_with_pvalues(scored: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Pearson r and two-tailed p-value for each input dimension
    against the Justifiability Score.  Also flags statistical significance
    at alpha = 0.05 and alpha = 0.01.
    """
    target = scored["justifiability_score"]
    dims = [
        ("emissions_kgco2e",            "Environmental emissions"),
        ("ethical_risk_score",           "Ethical risk"),
        ("creative_displacement_score",  "Creative displacement"),
        ("transparency_score",           "Transparency"),
    ]
 
    rows = []
    for col, label in dims:
        if col not in scored.columns:
            continue
        r, p = pearson_with_pvalue(scored[col], target)
        rows.append({
            "dimension":   label,
            "column":      col,
            "pearson_r":   r,
            "p_value":     round(p, 4),
            "sig_0.05":    "yes" if p < 0.05 else "no",
            "sig_0.01":    "yes" if p < 0.01 else "no",
            "n":           len(scored),
        })
 
    return pd.DataFrame(rows)
 
 
# ════════════════════════════════════════════════════════════════════════════
# FIX 4 — THRESHOLD SENSITIVITY ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
 
def run_threshold_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Test how many use cases change score band when the Low/Medium and
    Medium/High thresholds are varied around their default values (40, 70).
 
    Threshold configurations tested:
      Conservative  (35, 65) — wider Medium band
      Default       (40, 70) — project default
      Strict        (45, 75) — narrower High band
      Midpoint      (33, 67) — equal-tertile split
 
    For each configuration, counts:
      - cases in each band
      - cases that would change band vs default
    """
    scored = compute_scores(df)
    scores = scored["justifiability_score"]
 
    threshold_configs = {
        "conservative (35/65)": (35, 65),
        "default (40/70)":      (40, 70),
        "strict (45/75)":       (45, 75),
        "equal-tertile (33/67)":(33, 67),
    }
 
    # Label under the default thresholds (40, 70) for comparison
    def label_series(s, lo, hi):
        return pd.cut(s, bins=[-1, lo, hi, 101], labels=["Low", "Medium", "High"]).astype(str)
 
    default_labels = label_series(scores, 40, 70)
 
    rows = []
    for config_name, (lo, hi) in threshold_configs.items():
        labels = label_series(scores, lo, hi)
        band_counts = labels.value_counts().to_dict()
        changes_vs_default = int((labels != default_labels).sum()) if config_name != "default (40/70)" else 0
 
        rows.append({
            "threshold_config":     config_name,
            "low_threshold":        lo,
            "high_threshold":       hi,
            "n_low":                band_counts.get("Low", 0),
            "n_medium":             band_counts.get("Medium", 0),
            "n_high":               band_counts.get("High", 0),
            "changes_vs_default":   changes_vs_default,
            "pct_changed":          round(changes_vs_default / len(df) * 100, 1),
        })
 
    return pd.DataFrame(rows)
 
 
# ════════════════════════════════════════════════════════════════════════════
# PARETO FRONTIER (updated for 5 dimensions)
# ════════════════════════════════════════════════════════════════════════════
 
def pareto_frontier(scored: pd.DataFrame) -> pd.DataFrame:
    """
    Mark Pareto-optimal use cases across all five component dimensions.
    A use case is Pareto-optimal if no other use case is >= in all
    components and > in at least one.
    """
    required = [
        "component_emissions", "component_ethics", "component_creativity",
        "component_purpose", "component_transparency",
    ]
    # Fall back gracefully if transparency column not present
    available = [c for c in required if c in scored.columns]
 
    data = scored[available].to_numpy()
    is_pareto = np.ones(data.shape[0], dtype=bool)
    idx = np.arange(data.shape[0])
 
    for i in range(data.shape[0]):
        if not is_pareto[i]:
            continue
        dominated = np.any(
            (np.all(data >= data[i], axis=1) & np.any(data > data[i], axis=1))
            & (idx != i)
        )
        is_pareto[i] = not dominated
 
    out = scored.copy()
    out["pareto_optimal"] = is_pareto
    return out
 
 
# ════════════════════════════════════════════════════════════════════════════
# BASELINE COMPARISON
# ════════════════════════════════════════════════════════════════════════════
 
def baseline_comparison(df_raw: pd.DataFrame, scored_default: pd.DataFrame) -> dict:
    """
    Compare default scoring against three baselines:
      1. Equal weights across all 5 dimensions
      2. Ethics-only (essentially the EU AI Act risk level)
      3. Emissions-only (purely Green AI approach)
 
    Returns a dict of Spearman rank correlations.
    """
    equal = compute_scores(df_raw, w_emissions=0.2, w_ethics=0.2,
                           w_creativity=0.2, w_purpose=0.2, w_transparency=0.2)
    ethics_only = compute_scores(df_raw, w_emissions=0.001, w_ethics=1.0,
                                 w_creativity=0.001, w_purpose=0.001, w_transparency=0.001)
    emissions_only = compute_scores(df_raw, w_emissions=1.0, w_ethics=0.001,
                                    w_creativity=0.001, w_purpose=0.001, w_transparency=0.001)
 
    return {
        "equal_weights_5d":  round(rank_stability(scored_default, equal), 3),
        "ethics_only":       round(rank_stability(scored_default, ethics_only), 3),
        "emissions_only":    round(rank_stability(scored_default, emissions_only), 3),
    }
 
 
# ════════════════════════════════════════════════════════════════════════════
# ADVANCED ANALYSIS REPORT
# ════════════════════════════════════════════════════════════════════════════
 
def advanced_analysis(
    scored: pd.DataFrame,
    df_raw: pd.DataFrame | None = None,
    sensitivity: pd.DataFrame | None = None,
    montecarlo: pd.DataFrame | None = None,
    ablation: pd.DataFrame | None = None,
    threshold_sens: pd.DataFrame | None = None,
) -> str:
    lines = []
    lines.append("=" * 60)
    lines.append("  AI Misuse Evaluator — Advanced Analysis Summary")
    lines.append("=" * 60)
    lines.append(f"Number of use cases: {len(scored)}")
    lines.append("")
 
    srt = scored.sort_values("justifiability_score", ascending=False)
 
    # Top / bottom
    lines.append("Top 5 use cases:")
    for _, r in srt.head(5).iterrows():
        lines.append(f"  {r['use_case_name']:<45} {r['justifiability_score']:>6.2f}  ({r.get('label','')})")
    lines.append("")
    lines.append("Bottom 5 use cases:")
    for _, r in srt.tail(5).iterrows():
        lines.append(f"  {r['use_case_name']:<45} {r['justifiability_score']:>6.2f}  ({r.get('label','')})")
    lines.append("")
 
    # ── FIX 3: Correlations with p-values ────────────────────────────────
    lines.append("─" * 60)
    lines.append("CORRELATIONS WITH JUSTIFIABILITY SCORE (Pearson r, two-tailed)")
    lines.append("─" * 60)
    corr_df = compute_correlations_with_pvalues(scored)
    for _, row in corr_df.iterrows():
        sig = "***" if row["p_value"] < 0.001 else ("**" if row["p_value"] < 0.01 else ("*" if row["p_value"] < 0.05 else ""))
        lines.append(
            f"  {row['dimension']:<30}  r = {row['pearson_r']:>6.3f}  "
            f"p = {row['p_value']:.4f}  n={row['n']}  {sig}"
        )
    lines.append("  (* p<0.05  ** p<0.01  *** p<0.001)")
    lines.append("")
 
    # Category breakdown
    if "purpose_category" in scored.columns:
        lines.append("─" * 60)
        lines.append("SCORES BY PURPOSE CATEGORY")
        lines.append("─" * 60)
        grp = scored.groupby("purpose_category")["justifiability_score"].agg(
            ["count", "mean", "min", "max"]
        ).reset_index()
        for _, r in grp.iterrows():
            lines.append(
                f"  {r['purpose_category']:<15}  n={int(r['count'])}  "
                f"mean={r['mean']:.2f}  min={r['min']:.2f}  max={r['max']:.2f}"
            )
        lines.append("")
 
    # Guardrail
    if {"base_score", "guardrail_multiplier"}.issubset(scored.columns):
        guarded = scored[scored["guardrail_multiplier"] < 1.0].sort_values("justifiability_score")
        lines.append("─" * 60)
        lines.append("GUARDRAIL PENALTIES")
        lines.append("─" * 60)
        lines.append(f"  Cases penalised: {len(guarded)}/{len(scored)}")
        for _, row in guarded.iterrows():
            lines.append(
                f"  {row['use_case_name']:<45}  base={row['base_score']:.1f} "
                f"x{row['guardrail_multiplier']:.2f} → {row['justifiability_score']:.2f}"
            )
        lines.append("")
 
    # Pareto
    if "pareto_optimal" in scored.columns:
        pareto_cases = scored[scored["pareto_optimal"]].sort_values("justifiability_score", ascending=False)
        lines.append("─" * 60)
        lines.append("PARETO-OPTIMAL USE CASES (non-dominated across all 5 dimensions)")
        lines.append("─" * 60)
        lines.append(f"  Count: {len(pareto_cases)}/{len(scored)}")
        for _, r in pareto_cases.iterrows():
            lines.append(f"  {r['use_case_name']:<45}  score={r['justifiability_score']:.2f}")
        lines.append("")
 
    # Sensitivity
    if sensitivity is not None and len(sensitivity) > 0:
        lines.append("─" * 60)
        lines.append("SENSITIVITY ANALYSIS (emissions weight varied, 5-dimension model)")
        lines.append("─" * 60)
        best  = float(sensitivity["rank_stability_spearman"].max())
        worst = float(sensitivity["rank_stability_spearman"].min())
        lines.append(f"  Rank stability range:  min={worst:.3f}  max={best:.3f}")
        for _, row in sensitivity.iterrows():
            lines.append(
                f"  w_em={row['w_emissions']:.2f}  "
                f"w_eth={row['w_ethics']:.3f}  "
                f"w_tr={row['w_transparency']:.3f}  "
                f"stability={row['rank_stability_spearman']:.3f}"
            )
        lines.append("")
 
    # Monte Carlo
    if montecarlo is not None and len(montecarlo) > 0:
        lines.append("─" * 60)
        lines.append("MONTE CARLO ROBUSTNESS (5-dimensional Dirichlet, 200 iterations)")
        lines.append("─" * 60)
        for _, r in montecarlo.sort_values("win_rate", ascending=False).head(5).iterrows():
            bar = "█" * int(r["win_rate"] * 40)
            lines.append(f"  {r['use_case_name']:<45}  win_rate={r['win_rate']:.3f}  {bar}")
        lines.append("")
 
    # ── FIX 2: Ablation ──────────────────────────────────────────────────
    if ablation is not None and len(ablation) > 0:
        lines.append("─" * 60)
        lines.append("ABLATION STUDY (leave-one-dimension-out)")
        lines.append("─" * 60)
        lines.append("  Dimension removed    Band changes   %      Mean |Δscore|  Rank stability")
        lines.append("  " + "-" * 70)
        for _, row in ablation.iterrows():
            lines.append(
                f"  {row['ablated_dimension']:<20} "
                f"{row['band_changes']:>4} / {len(scored)}   "
                f"{row['pct_band_changes']:>5.1f}%   "
                f"{row['mean_abs_score_change']:>8.2f}        "
                f"{row['rank_stability']:>6.3f}"
            )
        lines.append("")
 
    # ── FIX 4: Threshold sensitivity ─────────────────────────────────────
    if threshold_sens is not None and len(threshold_sens) > 0:
        lines.append("─" * 60)
        lines.append("THRESHOLD SENSITIVITY (band boundary variation)")
        lines.append("─" * 60)
        lines.append(f"  {'Configuration':<26} {'Low':>5} {'Med':>5} {'High':>5} {'Changes vs default':>20}")
        lines.append("  " + "-" * 65)
        for _, row in threshold_sens.iterrows():
            lines.append(
                f"  {row['threshold_config']:<26} "
                f"{row['n_low']:>5} "
                f"{row['n_medium']:>5} "
                f"{row['n_high']:>5}   "
                f"{row['changes_vs_default']:>4} ({row['pct_changed']:.1f}%)"
            )
        lines.append("")
 
    # Baseline comparison
    if df_raw is not None:
        baselines = baseline_comparison(df_raw, scored)
        lines.append("─" * 60)
        lines.append("BASELINE COMPARISON (Spearman rank correlation with default 5D model)")
        lines.append("─" * 60)
        for name, corr in baselines.items():
            lines.append(f"  {name:<30}  r = {corr:.3f}")
        lines.append("  (lower = more different from default = multi-dimensional adds value)")
        lines.append("")
 
    return "\n".join(lines)
 
 
# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════
 
def main():
    parser = argparse.ArgumentParser(
        description="AI Misuse Evaluator — full analysis pipeline"
    )
    parser.add_argument("--input",      default=INPUT)
    parser.add_argument("--output",     default=OUTPUT)
    parser.add_argument("--analyze",    action="store_true", help="Write full analysis report")
    parser.add_argument("--sensitivity",action="store_true", help="Run 5D sensitivity analysis")
    parser.add_argument("--pareto",     action="store_true", help="Compute Pareto frontier (5D)")
    parser.add_argument("--montecarlo", action="store_true", help="Run 5D Monte Carlo (200 iters)")
    parser.add_argument("--ablation",   action="store_true", help="Run leave-one-dimension-out ablation")
    parser.add_argument("--threshold",  action="store_true", help="Run band threshold sensitivity")
    parser.add_argument("--mc_iters",   type=int, default=200)
    args = parser.parse_args()
 
    os.makedirs("data", exist_ok=True)
 
    df = pd.read_csv(args.input)
    scored = compute_scores(df)
 
    # Pareto (must run before saving scored, adds column)
    if args.pareto:
        scored = pareto_frontier(scored)
        scored.to_csv(PARETO_CSV, index=False)
        print(f"✅ Pareto frontier written to {PARETO_CSV}")
 
    scored.to_csv(args.output, index=False)
    print("\n✅ Scoring complete →", args.output)
    print(
        scored[["use_case_name", "justifiability_score", "label"]]
        .sort_values("justifiability_score", ascending=False)
        .to_string(index=False)
    )
 
    # Sensitivity
    sens_df = None
    if args.sensitivity:
        sens_df = run_sensitivity(df)
        sens_df.to_csv(SENS_CSV, index=False)
        print(f"\n✅ Sensitivity results → {SENS_CSV}")
        print(sens_df.to_string(index=False))
 
    # Monte Carlo
    mc_df = None
    if args.montecarlo:
        np.random.seed(42)          # reproducible
        mc_df = monte_carlo_weight_stability(df, iterations=args.mc_iters)
        mc_df.to_csv(MC_CSV, index=False)
        print(f"\n✅ Monte Carlo results → {MC_CSV}")
        print(mc_df.head(10).to_string(index=False))
 
    # Ablation
    ablation_df = None
    if args.ablation:
        ablation_df = run_ablation(df)
        ablation_df.to_csv(ABLATION_CSV, index=False)
        print(f"\n✅ Ablation study → {ABLATION_CSV}")
        print(ablation_df.to_string(index=False))
 
    # Threshold sensitivity
    threshold_df = None
    if args.threshold:
        threshold_df = run_threshold_sensitivity(df)
        threshold_df.to_csv(THRESH_CSV, index=False)
        print(f"\n✅ Threshold sensitivity → {THRESH_CSV}")
        print(threshold_df.to_string(index=False))
 
    # Full analysis report
    if args.analyze:
        report = advanced_analysis(
            scored,
            df_raw=df,
            sensitivity=sens_df,
            montecarlo=mc_df,
            ablation=ablation_df,
            threshold_sens=threshold_df,
        )
        with open(ANALYSIS_TXT, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n✅ Analysis report → {ANALYSIS_TXT}")
        print(report)
 
 
if __name__ == "__main__":
    main()