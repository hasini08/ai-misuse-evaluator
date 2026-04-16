"""
Tests for the transparency/explainability dimension added in v2.
Run with: pytest tests/test_transparency.py -v
"""
import os, sys
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from scoring_engine import compute_scores, DEFAULT_WEIGHTS


def make_row(name="Test", emissions=0.005, purpose="beneficial",
             ethics=2, creativity=2, transparency=3):
    return {
        "use_case_name": name,
        "emissions_kgco2e": emissions,
        "purpose_category": purpose,
        "ethical_risk_score": ethics,
        "creative_displacement_score": creativity,
        "transparency_score": transparency,
    }


class TestTransparencyComponent:

    def test_higher_transparency_raises_score(self):
        """A more transparent system should score higher, all else equal."""
        a = pd.DataFrame([make_row("Transparent", transparency=5)])
        b = pd.DataFrame([make_row("Opaque", transparency=1)])
        sa = compute_scores(a)["justifiability_score"].iloc[0]
        sb = compute_scores(b)["justifiability_score"].iloc[0]
        assert sa > sb

    def test_transparency_norm_bounds(self):
        """component_transparency must be in [0, 100]."""
        for t in [1, 2, 3, 4, 5]:
            df = pd.DataFrame([make_row(transparency=t)])
            scored = compute_scores(df)
            val = float(scored["component_transparency"].iloc[0])
            assert 0.0 <= val <= 100.0, f"transparency={t} gave component={val}"

    def test_transparency_score_1_gives_minimum_component(self):
        df = pd.DataFrame([make_row(transparency=1)])
        scored = compute_scores(df)
        assert float(scored["component_transparency"].iloc[0]) == pytest.approx(0.0)

    def test_transparency_score_5_gives_maximum_component(self):
        df = pd.DataFrame([make_row(transparency=5)])
        scored = compute_scores(df)
        assert float(scored["component_transparency"].iloc[0]) == pytest.approx(100.0)

    def test_transparency_out_of_range_raises(self):
        df = pd.DataFrame([make_row(transparency=6)])
        with pytest.raises(ValueError, match="within 1-5"):
            compute_scores(df)

    def test_transparency_zero_raises(self):
        df = pd.DataFrame([make_row(transparency=0)])
        with pytest.raises(ValueError, match="within 1-5"):
            compute_scores(df)

    def test_missing_transparency_column_defaults_to_neutral(self):
        """Datasets without transparency_score should still score (defaults to 0.5)."""
        row = {
            "use_case_name": "NoTransparency",
            "emissions_kgco2e": 0.005,
            "purpose_category": "beneficial",
            "ethical_risk_score": 2,
            "creative_displacement_score": 2,
        }
        df = pd.DataFrame([row])
        scored = compute_scores(df)
        assert len(scored) == 1
        val = float(scored["justifiability_score"].iloc[0])
        assert 0.0 <= val <= 100.0
        # Neutral transparency means component = 50
        assert float(scored["component_transparency"].iloc[0]) == pytest.approx(50.0)

    def test_transparency_contribution_column_present(self):
        df = pd.DataFrame([make_row()])
        scored = compute_scores(df)
        assert "contribution_transparency" in scored.columns

    def test_transparency_weight_zero_removes_contribution(self):
        """Setting w_transparency=0 should make contribution_transparency = 0."""
        df = pd.DataFrame([make_row()])
        scored = compute_scores(df, w_transparency=0.0)
        assert float(scored["contribution_transparency"].iloc[0]) == pytest.approx(0.0)

    def test_transparency_dominant_weight_rewards_explainability(self):
        a = make_row("Trans5", transparency=5)
        b = make_row("Trans1", transparency=1)
        df = pd.DataFrame([a, b])
        scored = compute_scores(df, w_emissions=0.1, w_ethics=0.1,
                                w_creativity=0.1, w_purpose=0.1, w_transparency=0.60)
        s = dict(zip(scored["use_case_name"], scored["justifiability_score"]))
        assert s["Trans5"] > s["Trans1"]


class TestFiveDimensionIntegration:

    def test_five_component_columns_present(self):
        df = pd.DataFrame([make_row()])
        scored = compute_scores(df)
        for col in ["component_emissions", "component_ethics", "component_creativity",
                    "component_purpose", "component_transparency"]:
            assert col in scored.columns, f"Missing: {col}"

    def test_five_contribution_columns_present(self):
        df = pd.DataFrame([make_row()])
        scored = compute_scores(df)
        for col in ["contribution_emissions", "contribution_ethics", "contribution_creativity",
                    "contribution_purpose", "contribution_transparency"]:
            assert col in scored.columns, f"Missing: {col}"

    def test_default_weights_include_transparency(self):
        assert "transparency" in DEFAULT_WEIGHTS
        assert DEFAULT_WEIGHTS["transparency"] > 0

    def test_full_dataset_scores_with_transparency(self):
        """The complete usecases.csv with transparency column should score without errors."""
        data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'usecases.csv')
        if not os.path.exists(data_path):
            pytest.skip("usecases.csv not found")
        df = pd.read_csv(data_path)
        assert "transparency_score" in df.columns, "Dataset is missing transparency_score column"
        scored = compute_scores(df)
        assert len(scored) == len(df)
        assert scored["justifiability_score"].between(0, 100).all()

    def test_opaque_harmful_scores_lower_than_transparent_beneficial(self):
        """An opaque harmful tool should always score lower than a transparent beneficial one."""
        rows = [
            make_row("Good", purpose="beneficial", ethics=1, creativity=1, transparency=5, emissions=0.005),
            make_row("Bad",  purpose="harmful",   ethics=5, creativity=5, transparency=1, emissions=0.005),
        ]
        df = pd.DataFrame(rows)
        scored = compute_scores(df)
        s = dict(zip(scored["use_case_name"], scored["justifiability_score"]))
        assert s["Good"] > s["Bad"]