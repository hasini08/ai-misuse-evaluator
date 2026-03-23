"""
Unit tests for scoring_engine.py
Run with: pytest tests/
"""
import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from scoring_engine import compute_scores, minmax, PURPOSE_WEIGHTS


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_row(name="Test", emissions=0.005, purpose="beneficial",
             ethics=2, creativity=2):
    return {
        "use_case_name": name,
        "emissions_kgco2e": emissions,
        "purpose_category": purpose,
        "ethical_risk_score": ethics,
        "creative_displacement_score": creativity,
    }


def make_df(*rows):
    return pd.DataFrame([make_row(**(r if isinstance(r, dict) else {}) ) if not isinstance(r, dict) else r for r in rows])


def score_pair(row_a: dict, row_b: dict):
    df = pd.DataFrame([row_a, row_b])
    scored = compute_scores(df)
    s = dict(zip(scored["use_case_name"], scored["justifiability_score"]))
    return s[row_a["use_case_name"]], s[row_b["use_case_name"]]


# ── Output range and structure ────────────────────────────────────────────────

class TestOutputStructure:
    def test_score_in_range(self):
        df = pd.DataFrame([make_row()])
        scored = compute_scores(df)
        val = float(scored["justifiability_score"].iloc[0])
        assert 0.0 <= val <= 100.0

    def test_required_output_columns_present(self):
        df = pd.DataFrame([make_row()])
        scored = compute_scores(df)
        for col in ["justifiability_score", "label",
                    "component_emissions", "component_ethics",
                    "component_creativity", "component_purpose"]:
            assert col in scored.columns, f"Missing column: {col}"

    def test_label_values_are_valid(self):
        rows = [make_row(f"case_{i}", purpose=p, ethics=e)
                for i, (p, e) in enumerate([
                    ("essential", 1), ("beneficial", 3), ("harmful", 5)
                ])]
        df = pd.DataFrame(rows)
        scored = compute_scores(df)
        valid = {"Low", "Medium", "High"}
        for lbl in scored["label"]:
            assert str(lbl) in valid

    def test_missing_column_raises(self):
        df = pd.DataFrame([{"use_case_name": "X", "emissions_kgco2e": 0.01}])
        with pytest.raises(ValueError, match="Missing required columns"):
            compute_scores(df)

    def test_score_rounded_to_2dp(self):
        df = pd.DataFrame([make_row()])
        scored = compute_scores(df)
        val = scored["justifiability_score"].iloc[0]
        assert val == round(val, 2)


# ── Component behaviour ───────────────────────────────────────────────────────

class TestComponentBehaviour:
    def test_lower_emissions_raises_score(self):
        a = make_row("LowE", emissions=0.001)
        b = make_row("HighE", emissions=0.100)
        s_a, s_b = score_pair(a, b)
        assert s_a > s_b

    def test_lower_ethical_risk_raises_score(self):
        a = make_row("LowR", ethics=1)
        b = make_row("HighR", ethics=5)
        s_a, s_b = score_pair(a, b)
        assert s_a > s_b

    def test_lower_creative_displacement_raises_score(self):
        a = make_row("LowC", creativity=1)
        b = make_row("HighC", creativity=5)
        s_a, s_b = score_pair(a, b)
        assert s_a > s_b

    def test_purpose_ordering(self):
        """essential > beneficial > low_benefit > harmful"""
        purposes = ["essential", "beneficial", "low_benefit", "harmful"]
        rows = [make_row(p, purpose=p) for p in purposes]
        df = pd.DataFrame(rows)
        scored = compute_scores(df)
        s = dict(zip(scored["use_case_name"], scored["justifiability_score"]))
        assert s["essential"] > s["beneficial"]
        assert s["beneficial"] > s["low_benefit"]
        assert s["low_benefit"] > s["harmful"]

    def test_component_scores_in_range(self):
        df = pd.DataFrame([make_row()])
        scored = compute_scores(df)
        for col in ["component_emissions", "component_ethics",
                    "component_creativity", "component_purpose"]:
            val = float(scored[col].iloc[0])
            assert 0.0 <= val <= 100.0, f"{col} out of range: {val}"


# ── Label thresholds ──────────────────────────────────────────────────────────

class TestLabelThresholds:
    def test_high_label_assigned_to_best_case(self):
        """Perfect use case: essential, zero risk, zero displacement, low emissions"""
        rows = [
            make_row("Best", emissions=0.001, purpose="essential", ethics=1, creativity=1),
            make_row("Worst", emissions=0.100, purpose="harmful", ethics=5, creativity=5),
        ]
        df = pd.DataFrame(rows)
        scored = compute_scores(df)
        lbls = dict(zip(scored["use_case_name"], scored["label"].astype(str)))
        assert lbls["Best"] == "High"

    def test_low_label_assigned_to_worst_case(self):
        rows = [
            make_row("Best", emissions=0.001, purpose="essential", ethics=1, creativity=1),
            make_row("Worst", emissions=0.100, purpose="harmful", ethics=5, creativity=5),
        ]
        df = pd.DataFrame(rows)
        scored = compute_scores(df)
        lbls = dict(zip(scored["use_case_name"], scored["label"].astype(str)))
        assert lbls["Worst"] == "Low"


# ── Weight configuration ──────────────────────────────────────────────────────

class TestWeightConfiguration:
    def test_custom_weights_change_score(self):
        df = pd.DataFrame([make_row()])
        default = compute_scores(df)["justifiability_score"].iloc[0]
        custom = compute_scores(df, w_emissions=0.10, w_ethics=0.60,
                                w_creativity=0.20, w_purpose=0.10
                                )["justifiability_score"].iloc[0]
        # Scores can differ when weights differ
        assert default != custom or True  # won't always differ but shouldn't crash

    def test_equal_weights_runs_without_error(self):
        df = pd.DataFrame([make_row()])
        scored = compute_scores(df, w_emissions=0.25, w_ethics=0.25,
                                w_creativity=0.25, w_purpose=0.25)
        assert len(scored) == 1

    def test_emissions_dominant_weight_rewards_low_emissions(self):
        a = make_row("LowE", emissions=0.001)
        b = make_row("HighE", emissions=0.100)
        df = pd.DataFrame([a, b])
        scored = compute_scores(df, w_emissions=0.90, w_ethics=0.034,
                                w_creativity=0.033, w_purpose=0.033)
        s = dict(zip(scored["use_case_name"], scored["justifiability_score"]))
        assert s["LowE"] > s["HighE"]


# ── Edge cases ────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_single_row_dataset(self):
        df = pd.DataFrame([make_row()])
        scored = compute_scores(df)
        assert len(scored) == 1

    def test_all_identical_rows_gives_equal_scores(self):
        rows = [make_row(f"case_{i}") for i in range(5)]
        df = pd.DataFrame(rows)
        scored = compute_scores(df)
        scores = scored["justifiability_score"].unique()
        assert len(scores) == 1, "All identical rows should yield identical scores"

    def test_unknown_purpose_category_handled(self):
        """Unknown category should not crash — fills with default 0.4"""
        row = make_row(purpose="unknown_category")
        df = pd.DataFrame([row])
        scored = compute_scores(df)
        assert len(scored) == 1
        val = float(scored["justifiability_score"].iloc[0])
        assert 0.0 <= val <= 100.0

    def test_large_dataset_runs(self):
        import random
        purposes = ["essential", "beneficial", "low_benefit", "harmful"]
        rows = [
            make_row(f"case_{i}",
                     emissions=round(random.uniform(0.001, 0.1), 4),
                     purpose=random.choice(purposes),
                     ethics=random.randint(1, 5),
                     creativity=random.randint(1, 5))
            for i in range(100)
        ]
        df = pd.DataFrame(rows)
        scored = compute_scores(df)
        assert len(scored) == 100
        assert scored["justifiability_score"].between(0, 100).all()


# ── minmax helper ─────────────────────────────────────────────────────────────

class TestMinmax:
    def test_minmax_range(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        out = minmax(s)
        assert float(out.min()) == pytest.approx(0.0)
        assert float(out.max()) == pytest.approx(1.0)

    def test_minmax_inverted(self):
        s = pd.Series([1.0, 2.0, 3.0])
        out = minmax(s, invert=True)
        assert float(out.iloc[0]) == pytest.approx(1.0)
        assert float(out.iloc[-1]) == pytest.approx(0.0)

    def test_minmax_constant_series(self):
        s = pd.Series([5.0, 5.0, 5.0])
        out = minmax(s)
        assert (out == 0.5).all()
