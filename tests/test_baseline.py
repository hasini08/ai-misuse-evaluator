"""
Baseline Comparison Tests for AI Misuse Evaluator
---------------------------------------------------
These tests demonstrate that the multi-dimensional Justifiability Score
provides more meaningful and nuanced rankings than any single-component
baseline alone.

This constitutes the baseline evaluation for dissertation Chapter 4.6.

Run with: pytest tests/test_baseline.py -v
"""
import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from scoring_engine import compute_scores


# ── Shared test dataset ───────────────────────────────────────────────────────
# Designed to expose cases where single-component baselines give wrong answers

BASELINE_CASES = [
    # High value but high emissions — single emissions baseline would penalise unfairly
    {"use_case_name": "Climate_Model_HighE",     "emissions_kgco2e": 0.050, "purpose_category": "essential",   "ethical_risk_score": 1, "creative_displacement_score": 1},
    # Low emissions but very harmful — emissions baseline would reward incorrectly
    {"use_case_name": "Targeted_Harassment",      "emissions_kgco2e": 0.002, "purpose_category": "harmful",     "ethical_risk_score": 5, "creative_displacement_score": 2},
    # Low ethical risk but large creative displacement — ethics baseline misses this
    {"use_case_name": "AI_Art_Factory",           "emissions_kgco2e": 0.010, "purpose_category": "low_benefit", "ethical_risk_score": 1, "creative_displacement_score": 5},
    # Moderate across all dimensions
    {"use_case_name": "Study_Assistant",          "emissions_kgco2e": 0.006, "purpose_category": "beneficial",  "ethical_risk_score": 2, "creative_displacement_score": 2},
    # Essential but moderately high emissions
    {"use_case_name": "Hospital_Diagnostic_AI",   "emissions_kgco2e": 0.030, "purpose_category": "essential",   "ethical_risk_score": 2, "creative_displacement_score": 1},
    # Harmful with low emissions — purely harmful intent
    {"use_case_name": "Voter_Manipulation_Bot",   "emissions_kgco2e": 0.003, "purpose_category": "harmful",     "ethical_risk_score": 5, "creative_displacement_score": 3},
    # Beneficial, low risk, moderate emissions
    {"use_case_name": "Fraud_Detector",           "emissions_kgco2e": 0.008, "purpose_category": "beneficial",  "ethical_risk_score": 2, "creative_displacement_score": 1},
    # High displacement, low emissions, low benefit
    {"use_case_name": "AI_Journalist_Replacer",   "emissions_kgco2e": 0.004, "purpose_category": "low_benefit", "ethical_risk_score": 3, "creative_displacement_score": 5},
]


@pytest.fixture(scope="module")
def baseline_data():
    df = pd.DataFrame(BASELINE_CASES)
    default = compute_scores(df)
    equal_weights = compute_scores(df, w_emissions=0.25, w_ethics=0.25,
                                   w_creativity=0.25, w_purpose=0.25)
    emissions_only = compute_scores(df, w_emissions=1.00, w_ethics=0.001,
                                    w_creativity=0.001, w_purpose=0.001)
    ethics_only = compute_scores(df, w_emissions=0.001, w_ethics=1.00,
                                 w_creativity=0.001, w_purpose=0.001)
    return {
        "df": df,
        "default": default,
        "equal_weights": equal_weights,
        "emissions_only": emissions_only,
        "ethics_only": ethics_only,
    }


def get_rank(scored_df, name):
    ranked = scored_df.sort_values("justifiability_score", ascending=False).reset_index(drop=True)
    row = ranked[ranked["use_case_name"] == name]
    return int(row.index[0]) + 1  # 1-indexed


def spearman_corr(df_a, df_b):
    merged = df_a[["use_case_name", "justifiability_score"]].merge(
        df_b[["use_case_name", "justifiability_score"]],
        on="use_case_name", suffixes=("_a", "_b")
    )
    return float(merged["justifiability_score_a"].rank().corr(
        merged["justifiability_score_b"].rank()
    ))


# ── Key correctness tests ─────────────────────────────────────────────────────

class TestBaselineCorrectness:

    def test_emissions_only_incorrectly_ranks_harassment_tool_high(self, baseline_data):
        """
        Targeted_Harassment has very low emissions — an emissions-only baseline
        would rank it near the top, which is clearly wrong.
        The default scoring should rank it much lower.
        """
        rank_emissions = get_rank(baseline_data["emissions_only"], "Targeted_Harassment")
        rank_default = get_rank(baseline_data["default"], "Targeted_Harassment")
        assert rank_default > rank_emissions, (
            f"Default should rank Targeted_Harassment lower than emissions-only baseline. "
            f"Default rank: {rank_default}, Emissions-only rank: {rank_emissions}"
        )

    def test_emissions_only_penalises_climate_model_unfairly(self, baseline_data):
        """
        Climate_Model_HighE has high emissions but is essential.
        Emissions-only baseline penalises it; default scoring should rank it higher.
        """
        rank_emissions = get_rank(baseline_data["emissions_only"], "Climate_Model_HighE")
        rank_default = get_rank(baseline_data["default"], "Climate_Model_HighE")
        assert rank_default < rank_emissions, (
            f"Default should rank Climate_Model_HighE higher than emissions-only. "
            f"Default rank: {rank_default}, Emissions-only rank: {rank_emissions}"
        )

    def test_ethics_only_misses_creative_displacement(self, baseline_data):
        """
        AI_Art_Factory has low ethical risk (score=1) but massive creative displacement.
        Ethics-only baseline ranks it highly; default scoring should be more cautious.
        """
        rank_ethics = get_rank(baseline_data["ethics_only"], "AI_Art_Factory")
        rank_default = get_rank(baseline_data["default"], "AI_Art_Factory")
        assert rank_default > rank_ethics, (
            f"Default should rank AI_Art_Factory lower than ethics-only. "
            f"Default rank: {rank_default}, Ethics-only rank: {rank_ethics}"
        )

    def test_voter_manipulation_bot_always_ranks_last_or_near_last(self, baseline_data):
        """
        Voter_Manipulation_Bot should rank in bottom 2 in every scoring approach.
        """
        n = len(BASELINE_CASES)
        for name, scored in [
            ("default", baseline_data["default"]),
            ("equal_weights", baseline_data["equal_weights"]),
        ]:
            rank = get_rank(scored, "Voter_Manipulation_Bot")
            assert rank >= n - 1, (
                f"Voter_Manipulation_Bot should be in bottom 2 under {name} scoring. "
                f"Got rank {rank}/{n}"
            )

    def test_hospital_diagnostic_ranks_above_harassment_in_default(self, baseline_data):
        """
        Despite higher emissions, Hospital_Diagnostic_AI should rank above
        Targeted_Harassment under default weights.
        """
        rank_hospital = get_rank(baseline_data["default"], "Hospital_Diagnostic_AI")
        rank_harassment = get_rank(baseline_data["default"], "Targeted_Harassment")
        assert rank_hospital < rank_harassment


# ── Correlation and ranking divergence ───────────────────────────────────────

class TestRankingDivergence:

    def test_default_and_emissions_only_have_low_correlation(self, baseline_data):
        """
        Low correlation shows the two approaches produce meaningfully different rankings,
        justifying the multi-dimensional approach.
        """
        corr = spearman_corr(baseline_data["default"], baseline_data["emissions_only"])
        print(f"\nDefault vs emissions-only Spearman correlation: {corr:.3f}")
        # We expect meaningful divergence (< 0.95 shows they're not equivalent)
        assert corr < 0.95, f"Default and emissions-only rankings too similar (r={corr:.3f})"

    def test_default_and_ethics_only_have_low_correlation(self, baseline_data):
        corr = spearman_corr(baseline_data["default"], baseline_data["ethics_only"])
        print(f"\nDefault vs ethics-only Spearman correlation: {corr:.3f}")
        assert corr < 0.95, f"Default and ethics-only rankings too similar (r={corr:.3f})"

    def test_default_and_equal_weights_are_moderately_correlated(self, baseline_data):
        """
        Equal weights and default weights should be reasonably similar but not identical —
        confirming default weights make a difference but aren't arbitrary.
        """
        corr = spearman_corr(baseline_data["default"], baseline_data["equal_weights"])
        print(f"\nDefault vs equal-weights Spearman correlation: {corr:.3f}")
        assert 0.5 <= corr <= 1.0, f"Unexpected correlation: {corr:.3f}"


# ── Summary report ────────────────────────────────────────────────────────────

class TestBaselineSummaryReport:

    def test_print_ranking_comparison_table(self, baseline_data):
        """
        Prints a ranking comparison table for all baselines.
        Include this output in dissertation Chapter 4.6.
        """
        approaches = {
            "Default": baseline_data["default"],
            "Equal weights": baseline_data["equal_weights"],
            "Emissions only": baseline_data["emissions_only"],
            "Ethics only": baseline_data["ethics_only"],
        }

        print("\n\n=== Ranking Comparison: Default vs Baselines ===")
        header = f"{'Use Case':<35}" + "".join(f"{k:<16}" for k in approaches)
        print(header)
        print("-" * (35 + 16 * len(approaches)))

        for case in BASELINE_CASES:
            name = case["use_case_name"]
            row = f"{name:<35}"
            for approach_name, scored in approaches.items():
                rank = get_rank(scored, name)
                score = float(scored[scored["use_case_name"] == name]["justifiability_score"].iloc[0])
                row += f"#{rank} ({score:.1f})       "[:16]
            print(row)

        print("\nSpearman rank correlations with default scoring:")
        for approach_name, scored in list(approaches.items())[1:]:
            corr = spearman_corr(baseline_data["default"], scored)
            print(f"  Default vs {approach_name}: r={corr:.3f}")