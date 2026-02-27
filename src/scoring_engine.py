import pandas as pd
import numpy as np

PURPOSE_WEIGHTS = {
    "essential": 1.0,
    "beneficial": 0.8,
    "low_benefit": 0.4,
    "harmful": 0.1,
}

def minmax(series: pd.Series, invert: bool = False) -> pd.Series:
    s = series.astype(float)
    mn, mx = s.min(), s.max()
    if mx == mn:
        # if all values equal, make them neutral (0.5) or all-good (1.0)
        out = pd.Series([0.5] * len(s), index=s.index)
    else:
        out = (s - mn) / (mx - mn)
    return 1.0 - out if invert else out

def compute_scores(
    df: pd.DataFrame,
    w_emissions: float = 0.40,
    w_ethics: float = 0.25,
    w_creativity: float = 0.20,
    w_purpose: float = 0.15,
) -> pd.DataFrame:
    """
    Returns df with:
      - normalised component columns
      - justifiability_score in [0,100]
      - label in {Low, Medium, High}
    """
    required = ["use_case_name", "emissions_kgco2e", "purpose_category", "ethical_risk_score", "creative_displacement_score"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()

    # Emissions: lower is better => invert=True
    out["emissions_norm"] = minmax(out["emissions_kgco2e"], invert=True)

    # Ethics/Creativity: stored as 1–5 where 1 is best. Convert to 0–1 goodness directly:
    # risk_norm = (score-1)/4 gives 0 (best) .. 1 (worst); invert to make 1 best .. 0 worst
    out["ethics_norm"] = 1.0 - ((out["ethical_risk_score"].astype(float) - 1.0) / 4.0)
    out["creativity_norm"] = 1.0 - ((out["creative_displacement_score"].astype(float) - 1.0) / 4.0)

    # Purpose: map category to numeric goodness (0.1..1.0)
    out["purpose_norm"] = out["purpose_category"].str.lower().map(PURPOSE_WEIGHTS).fillna(0.4)

    # Weighted aggregation (normalise by sum of weights to keep 0–1)
    w_sum = w_emissions + w_ethics + w_creativity + w_purpose
    score_0_1 = (
        w_emissions * out["emissions_norm"]
        + w_ethics * out["ethics_norm"]
        + w_creativity * out["creativity_norm"]
        + w_purpose * out["purpose_norm"]
    ) / w_sum

    out["justifiability_score"] = (100.0 * score_0_1).round(2)

    out["label"] = pd.cut(
        out["justifiability_score"],
        bins=[-1, 40, 70, 101],
        labels=["Low", "Medium", "High"]
    )

    # Optional: component breakdown for explainability (looks advanced)
    out["component_emissions"] = (100.0 * out["emissions_norm"]).round(2)
    out["component_ethics"] = (100.0 * out["ethics_norm"]).round(2)
    out["component_creativity"] = (100.0 * out["creativity_norm"]).round(2)
    out["component_purpose"] = (100.0 * out["purpose_norm"]).round(2)

    return out