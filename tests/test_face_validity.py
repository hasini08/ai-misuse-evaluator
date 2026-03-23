"""
Face Validity Tests for AI Misuse Evaluator
--------------------------------------------
These tests verify that the Justifiability Score aligns with expert intuition
on hand-labelled use cases. Each case has a clearly expected outcome that any
reasonable observer would agree with.

This constitutes the face validity evaluation for dissertation Chapter 3.5.2.

Run with: pytest tests/test_face_validity.py -v
"""
import pytest
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from scoring_engine import compute_scores


# ── Ground truth dataset ──────────────────────────────────────────────────────
# Each case is hand-labelled with an expected score band.
# Rationale column is for dissertation documentation only.

GROUND_TRUTH = [
    # Clearly HIGH justifiability (score should be > 70)
    {
        "use_case_name": "Cancer_Detection_AI",
        "emissions_kgco2e": 0.003,
        "purpose_category": "essential",
        "ethical_risk_score": 1,
        "creative_displacement_score": 1,
        "expected_band": "High",
        "rationale": "Life-saving medical application, minimal risk, low emissions"
    },
    {
        "use_case_name": "Earthquake_Early_Warning",
        "emissions_kgco2e": 0.005,
        "purpose_category": "essential",
        "ethical_risk_score": 1,
        "creative_displacement_score": 1,
        "expected_band": "High",
        "rationale": "Critical safety infrastructure, very low risk"
    },
    {
        "use_case_name": "Accessible_Speech_To_Text",
        "emissions_kgco2e": 0.002,
        "purpose_category": "essential",
        "ethical_risk_score": 1,
        "creative_displacement_score": 1,
        "expected_band": "High",
        "rationale": "Accessibility tool, negligible risk, tiny footprint"
    },
    {
        "use_case_name": "Drug_Interaction_Checker",
        "emissions_kgco2e": 0.004,
        "purpose_category": "essential",
        "ethical_risk_score": 2,
        "creative_displacement_score": 1,
        "expected_band": "High",
        "rationale": "Patient safety tool, low risk when used alongside clinicians"
    },
    {
        "use_case_name": "Wildfire_Spread_Predictor",
        "emissions_kgco2e": 0.006,
        "purpose_category": "essential",
        "ethical_risk_score": 1,
        "creative_displacement_score": 1,
        "expected_band": "High",
        "rationale": "Environmental and public safety use, minimal displacement"
    },

    # Clearly LOW justifiability (score should be < 40)
    {
        "use_case_name": "Autonomous_Misinformation_Bot",
        "emissions_kgco2e": 0.015,
        "purpose_category": "harmful",
        "ethical_risk_score": 5,
        "creative_displacement_score": 3,
        "expected_band": "Low",
        "rationale": "Explicitly harmful purpose, high ethical risk"
    },
    {
        "use_case_name": "Deepfake_Nonconsensual_Generator",
        "emissions_kgco2e": 0.020,
        "purpose_category": "harmful",
        "ethical_risk_score": 5,
        "creative_displacement_score": 5,
        "expected_band": "Low",
        "rationale": "Harmful and unethical, displaces human creative consent"
    },
    {
        "use_case_name": "Targeted_Harassment_Tool",
        "emissions_kgco2e": 0.008,
        "purpose_category": "harmful",
        "ethical_risk_score": 5,
        "creative_displacement_score": 2,
        "expected_band": "Low",
        "rationale": "Primary use case is to harm individuals"
    },
    {
        "use_case_name": "Voter_Suppression_AI",
        "emissions_kgco2e": 0.010,
        "purpose_category": "harmful",
        "ethical_risk_score": 5,
        "creative_displacement_score": 2,
        "expected_band": "Low",
        "rationale": "Designed to undermine democratic processes"
    },
    {
        "use_case_name": "High_Emission_Novelty_Chatbot",
        "emissions_kgco2e": 0.500,
        "purpose_category": "low_benefit",
        "ethical_risk_score": 3,
        "creative_displacement_score": 4,
        "expected_band": "Low",
        "rationale": "Enormous environmental cost for trivial benefit"
    },

    # MEDIUM justifiability (score should be 40–70) — genuinely ambiguous cases
    {
        "use_case_name": "AI_Resume_Screener",
        "emissions_kgco2e": 0.005,
        "purpose_category": "beneficial",
        "ethical_risk_score": 4,
        "creative_displacement_score": 1,
        "expected_band": "Medium",
        "rationale": "Useful but carries real bias/fairness risk"
    },
    {
        "use_case_name": "Social_Media_Content_Ranker",
        "emissions_kgco2e": 0.012,
        "purpose_category": "low_benefit",
        "ethical_risk_score": 3,
        "creative_displacement_score": 3,
        "expected_band": "Medium",
        "rationale": "Moderate risk, moderate benefit, moderate footprint"
    },
    {
        "use_case_name": "AI_Music_Composer",
        "emissions_kgco2e": 0.010,
        "purpose_category": "low_benefit",
        "ethical_risk_score": 2,
        "creative_displacement_score": 5,
        "expected_band": "Medium",
        "rationale": "Low ethical risk but high creative displacement pulls it down"
    },
    {
        "use_case_name": "Predictive_Policing_Tool",
        "emissions_kgco2e": 0.008,
        "purpose_category": "beneficial",
        "ethical_risk_score": 5,
        "creative_displacement_score": 1,
        "expected_band": "Medium",
        "rationale": "Claimed benefit but very high ethical risk due to bias potential"
    },
    {
        "use_case_name": "AI_Legal_Document_Drafter",
        "emissions_kgco2e": 0.007,
        "purpose_category": "beneficial",
        "ethical_risk_score": 3,
        "creative_displacement_score": 3,
        "expected_band": "Medium",
        "rationale": "Useful tool but displaces legal professionals and carries moderate risk"
    },
]


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def face_validity_results():
    """Compute scores for all ground truth cases once."""
    df = pd.DataFrame(GROUND_TRUTH).drop(columns=["expected_band", "rationale"])
    scored = compute_scores(df)
    return scored


class TestFaceValidity:

    def _get_score(self, results, name):
        row = results[results["use_case_name"] == name]
        assert len(row) == 1, f"Use case not found: {name}"
        return float(row["justifiability_score"].iloc[0]), str(row["label"].iloc[0])

    # High justifiability cases
    def test_cancer_detection_is_high(self, face_validity_results):
        score, label = self._get_score(face_validity_results, "Cancer_Detection_AI")
        assert label == "High", f"Cancer_Detection_AI should be High, got {label} (score={score})"

    def test_earthquake_warning_is_high(self, face_validity_results):
        score, label = self._get_score(face_validity_results, "Earthquake_Early_Warning")
        assert label == "High", f"Earthquake_Early_Warning should be High, got {label} (score={score})"

    def test_accessibility_tool_is_high(self, face_validity_results):
        score, label = self._get_score(face_validity_results, "Accessible_Speech_To_Text")
        assert label == "High", f"Accessible_Speech_To_Text should be High, got {label} (score={score})"

    def test_drug_checker_is_high(self, face_validity_results):
        score, label = self._get_score(face_validity_results, "Drug_Interaction_Checker")
        assert label == "High", f"Drug_Interaction_Checker should be High, got {label} (score={score})"

    def test_wildfire_predictor_is_high(self, face_validity_results):
        score, label = self._get_score(face_validity_results, "Wildfire_Spread_Predictor")
        assert label == "High", f"Wildfire_Spread_Predictor should be High, got {label} (score={score})"

    # Low justifiability cases
    def test_misinformation_bot_is_low(self, face_validity_results):
        score, label = self._get_score(face_validity_results, "Autonomous_Misinformation_Bot")
        assert label == "Low", f"Autonomous_Misinformation_Bot should be Low, got {label} (score={score})"

    def test_deepfake_generator_is_low(self, face_validity_results):
        score, label = self._get_score(face_validity_results, "Deepfake_Nonconsensual_Generator")
        assert label == "Low", f"Deepfake_Nonconsensual_Generator should be Low, got {label} (score={score})"

    def test_harassment_tool_is_low(self, face_validity_results):
        score, label = self._get_score(face_validity_results, "Targeted_Harassment_Tool")
        assert label == "Low", f"Targeted_Harassment_Tool should be Low, got {label} (score={score})"

    def test_voter_suppression_is_low(self, face_validity_results):
        score, label = self._get_score(face_validity_results, "Voter_Suppression_AI")
        assert label == "Low", f"Voter_Suppression_AI should be Low, got {label} (score={score})"

    def test_high_emission_novelty_bot_is_low(self, face_validity_results):
        score, label = self._get_score(face_validity_results, "High_Emission_Novelty_Chatbot")
        assert label == "Low", f"High_Emission_Novelty_Chatbot should be Low, got {label} (score={score})"

    # Ordering checks (don't prescribe exact label for medium cases,
    # but assert clear orderings hold)
    def test_essential_low_risk_beats_harmful_high_risk(self, face_validity_results):
        s_cancer, _ = self._get_score(face_validity_results, "Cancer_Detection_AI")
        s_misinfo, _ = self._get_score(face_validity_results, "Autonomous_Misinformation_Bot")
        assert s_cancer > s_misinfo

    def test_resume_screener_below_cancer_detection(self, face_validity_results):
        s_cancer, _ = self._get_score(face_validity_results, "Cancer_Detection_AI")
        s_resume, _ = self._get_score(face_validity_results, "AI_Resume_Screener")
        assert s_cancer > s_resume

    def test_resume_screener_above_deepfake(self, face_validity_results):
        s_resume, _ = self._get_score(face_validity_results, "AI_Resume_Screener")
        s_deepfake, _ = self._get_score(face_validity_results, "Deepfake_Nonconsensual_Generator")
        assert s_resume > s_deepfake


class TestFaceValiditySummary:
    """
    Generates a summary table of scores vs expected bands.
    Useful for including directly in dissertation Chapter 3.5.2.
    """
    def test_print_face_validity_table(self, face_validity_results):
        """Not a pass/fail test — prints a summary table for dissertation evidence."""
        gt_df = pd.DataFrame(GROUND_TRUTH)[["use_case_name", "expected_band", "rationale"]]
        merged = face_validity_results.merge(gt_df, on="use_case_name")
        merged["actual_band"] = merged["label"].astype(str)
        merged["match"] = merged["expected_band"] == merged["actual_band"]

        print("\n\n=== Face Validity Results ===")
        print(f"{'Use Case':<40} {'Expected':<10} {'Actual':<10} {'Score':<8} {'Match'}")
        print("-" * 80)
        for _, r in merged.sort_values("justifiability_score", ascending=False).iterrows():
            match_str = "✓" if r["match"] else "✗ FAIL"
            print(f"{r['use_case_name']:<40} {r['expected_band']:<10} {r['actual_band']:<10} {r['justifiability_score']:<8} {match_str}")

        n_match = int(merged["match"].sum())
        total = len(merged)
        accuracy = n_match / total * 100
        print(f"\nFace validity accuracy: {n_match}/{total} ({accuracy:.0f}%)")
        assert accuracy >= 80.0, f"Face validity accuracy {accuracy:.0f}% is below acceptable threshold (80%)"