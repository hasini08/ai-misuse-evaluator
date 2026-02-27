import pandas as pd
from src.scoring_engine import compute_scores

def test_essential_scores_higher_than_harmful():
    df = pd.DataFrame([
        {"use_case_name":"Essential", "emissions_kgco2e":0.01, "purpose_category":"essential", "ethical_risk_score":1, "creative_displacement_score":1},
        {"use_case_name":"Harmful", "emissions_kgco2e":0.01, "purpose_category":"harmful", "ethical_risk_score":5, "creative_displacement_score":1},
    ])
    scored = compute_scores(df)
    s = dict(zip(scored["use_case_name"], scored["justifiability_score"]))
    assert s["Essential"] > s["Harmful"]

def test_higher_emissions_reduces_score():
    df = pd.DataFrame([
        {"use_case_name":"LowE", "emissions_kgco2e":0.001, "purpose_category":"beneficial", "ethical_risk_score":2, "creative_displacement_score":2},
        {"use_case_name":"HighE", "emissions_kgco2e":0.010, "purpose_category":"beneficial", "ethical_risk_score":2, "creative_displacement_score":2},
    ])
    scored = compute_scores(df)
    s = dict(zip(scored["use_case_name"], scored["justifiability_score"]))
    assert s["LowE"] > s["HighE"]

def test_output_in_range():
    df = pd.DataFrame([
        {"use_case_name":"X", "emissions_kgco2e":0.005, "purpose_category":"low_benefit", "ethical_risk_score":3, "creative_displacement_score":4},
    ])
    scored = compute_scores(df)
    val = float(scored["justifiability_score"].iloc[0])
    assert 0.0 <= val <= 100.0