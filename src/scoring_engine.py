import numpy as np
import pandas as pd

PURPOSE_WEIGHTS = {
    "essential": 1.0,
    "beneficial": 0.8,
    "low_benefit": 0.4,
    "harmful": 0.0,
}

DEFAULT_WEIGHTS = {
    "emissions": 0.40,
    "ethics": 0.25,
    "creativity": 0.20,
    "purpose": 0.15,
}


def minmax(series: pd.Series, invert: bool = False) -> pd.Series:
    s = series.astype(float)
    mn, mx = s.min(), s.max()
    if mx == mn:
        out = pd.Series([0.5] * len(s), index=s.index)
    else:
        out = (s - mn) / (mx - mn)
    return 1.0 - out if invert else out


def robust_emissions_goodness(series: pd.Series, upper_quantile: float = 0.95) -> pd.Series:
    """
    Convert emissions to a 0-1 goodness score using clipped log scaling.

    This prevents a single extreme emitter from compressing the rest of the
    dataset into nearly identical scores.
    """
    s = series.astype(float).clip(lower=0.0)
    if s.nunique(dropna=False) <= 1:
        return pd.Series([0.5] * len(s), index=s.index)

    upper = float(s.quantile(upper_quantile))
    if upper <= 0:
        upper = float(s.max())

    clipped = s.clip(upper=upper)
    transformed = np.log1p(clipped)
    return minmax(pd.Series(transformed, index=s.index), invert=True)


def validate_input(df: pd.DataFrame) -> None:
    if (df["emissions_kgco2e"].astype(float) < 0).any():
        raise ValueError("emissions_kgco2e must be non-negative")

    for column in ["ethical_risk_score", "creative_displacement_score"]:
        values = df[column].astype(float)
        if ((values < 1) | (values > 5)).any():
            raise ValueError(f"{column} must be within 1-5")


def compute_scores(
    df: pd.DataFrame,
    w_emissions: float = DEFAULT_WEIGHTS["emissions"],
    w_ethics: float = DEFAULT_WEIGHTS["ethics"],
    w_creativity: float = DEFAULT_WEIGHTS["creativity"],
    w_purpose: float = DEFAULT_WEIGHTS["purpose"],
) -> pd.DataFrame:
    """
    Return the scored dataframe with component breakdowns and final labels.
    """
    required = [
        "use_case_name",
        "emissions_kgco2e",
        "purpose_category",
        "ethical_risk_score",
        "creative_displacement_score",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()
    out["purpose_category"] = out["purpose_category"].astype(str).str.lower()
    validate_input(out)

    out["emissions_norm"] = robust_emissions_goodness(out["emissions_kgco2e"])
    out["ethics_norm"] = 1.0 - ((out["ethical_risk_score"].astype(float) - 1.0) / 4.0)
    out["creativity_norm"] = 1.0 - ((out["creative_displacement_score"].astype(float) - 1.0) / 4.0)
    out["purpose_norm"] = out["purpose_category"].map(PURPOSE_WEIGHTS).fillna(0.4)

    weight_sum = w_emissions + w_ethics + w_creativity + w_purpose
    if weight_sum <= 0:
        raise ValueError("Sum of weights must be positive")

    score_0_1 = (
        w_emissions * out["emissions_norm"]
        + w_ethics * out["ethics_norm"]
        + w_creativity * out["creativity_norm"]
        + w_purpose * out["purpose_norm"]
    ) / weight_sum
    out["base_score"] = 100.0 * score_0_1

    # Non-compensatory guardrail: clearly harmful high-risk systems should not
    # end up with strong final scores just because they are efficient.
    harmful = out["purpose_category"].eq("harmful")
    severe_harm = harmful & (out["ethical_risk_score"].astype(float) >= 4.0)
    out["guardrail_multiplier"] = 1.0
    out.loc[harmful, "guardrail_multiplier"] *= 0.60
    out.loc[severe_harm, "guardrail_multiplier"] *= 0.70

    out["justifiability_score"] = (out["base_score"] * out["guardrail_multiplier"]).round(2)
    out["label"] = pd.cut(
        out["justifiability_score"],
        bins=[-1, 40, 70, 101],
        labels=["Low", "Medium", "High"],
    )

    out["component_emissions"] = (100.0 * out["emissions_norm"]).round(2)
    out["component_ethics"] = (100.0 * out["ethics_norm"]).round(2)
    out["component_creativity"] = (100.0 * out["creativity_norm"]).round(2)
    out["component_purpose"] = (100.0 * out["purpose_norm"]).round(2)

    out["contribution_emissions"] = (100.0 * (w_emissions * out["emissions_norm"] / weight_sum)).round(2)
    out["contribution_ethics"] = (100.0 * (w_ethics * out["ethics_norm"] / weight_sum)).round(2)
    out["contribution_creativity"] = (100.0 * (w_creativity * out["creativity_norm"] / weight_sum)).round(2)
    out["contribution_purpose"] = (100.0 * (w_purpose * out["purpose_norm"] / weight_sum)).round(2)

    return out
