"""Feature-based splicing-rescue proxy components."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .numeric import safe_float


def numeric_series(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    """Return a numeric series with missing values filled."""
    if column not in df:
        return pd.Series([default] * len(df), index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").fillna(default)


def clip01(values: pd.Series | np.ndarray | float) -> pd.Series | float:
    """Clip values to the 0-1 range."""
    if isinstance(values, pd.Series):
        return values.clip(lower=0.0, upper=1.0)
    return float(max(0.0, min(1.0, safe_float(values, 0.0))))


def minmax_score(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """Scale a numeric series to 0-1."""
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().sum() == 0:
        return pd.Series([0.0] * len(series), index=series.index)
    fill = values.median()
    values = values.fillna(fill)
    lo = float(values.min())
    hi = float(values.max())
    if hi == lo:
        return pd.Series([0.5] * len(values), index=values.index)
    scaled = (values - lo) / (hi - lo)
    return scaled if higher_is_better else 1.0 - scaled


def zscore(series: pd.Series) -> pd.Series:
    """Return a stable z-score series."""
    values = pd.to_numeric(series, errors="coerce").fillna(pd.to_numeric(series, errors="coerce").median())
    std = float(values.std(ddof=0))
    if not np.isfinite(std) or std == 0:
        return pd.Series([0.0] * len(values), index=series.index)
    return (values - float(values.mean())) / std


def target_binding_strength_score(target_delta_g: pd.Series, strong_threshold: float = -7.0) -> pd.Series:
    """Score target docking strength where more negative deltaG is better."""
    values = pd.to_numeric(target_delta_g, errors="coerce")
    score = (strong_threshold - values) / 4.0 + 0.5
    return clip01(score.fillna(0.0))


def selectivity_gap_score(weighted_gap: pd.Series, minimum_gap: float = 1.0) -> pd.Series:
    """Score target-vs-decoy selectivity gap."""
    values = pd.to_numeric(weighted_gap, errors="coerce").fillna(0.0)
    return clip01((values - (minimum_gap - 1.5)) / 3.0)


def off_target_safety_score(off_target_risk: pd.Series, maximum_risk: float = 0.40) -> pd.Series:
    """Convert off-target risk into a favorable score."""
    values = pd.to_numeric(off_target_risk, errors="coerce").fillna(0.5)
    return clip01(1.0 - values / max(maximum_risk * 2.0, 0.1))


def infer_feature_based_components(candidates: pd.DataFrame, landmarks: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Infer feature-based support components for each candidate."""
    thresholds = config.get("splicing_rescue_scoring", {}).get("thresholds", {})
    strong_delta = safe_float(thresholds.get("strong_target_binding_deltaG", -7.0), -7.0)
    min_gap = safe_float(thresholds.get("minimum_selectivity_gap", 1.0), 1.0)
    max_risk = safe_float(thresholds.get("maximum_off_target_risk_score", 0.40), 0.40)
    result = pd.DataFrame(index=candidates.index)
    result["molecule_id"] = candidates["molecule_id"].astype(str)
    result["target_binding_strength_score"] = target_binding_strength_score(numeric_series(candidates, "target_deltaG", default=np.nan), strong_delta)
    gap_component = selectivity_gap_score(numeric_series(candidates, "weighted_selectivity_gap", default=0.0), min_gap)
    risk_component = off_target_safety_score(numeric_series(candidates, "off_target_risk_score", default=0.5), max_risk)
    beneficial_landmarks = int(landmarks.get("is_beneficial_target_region", pd.Series(dtype=bool)).astype(str).str.lower().isin({"true", "1"}).sum())
    caution_landmarks = int(landmarks.get("is_caution_region", pd.Series(dtype=bool)).astype(str).str.lower().isin({"true", "1"}).sum())
    landmark_confidence = min(1.0, beneficial_landmarks / max(1.0, beneficial_landmarks + caution_landmarks))
    # Feature fallback cannot know a ligand's exact position. It therefore uses
    # binding/selectivity plus landmark availability as a conservative proxy.
    result["beneficial_landmark_proximity_score"] = (0.55 * result["target_binding_strength_score"] + 0.45 * landmark_confidence).clip(0, 1)
    result["u1_support_score"] = (0.60 * result["target_binding_strength_score"] + 0.40 * gap_component).clip(0, 1)
    result["exon7_boundary_support_score"] = (0.50 * result["target_binding_strength_score"] + 0.50 * risk_component).clip(0, 1)
    result["contact_density_score"] = 0.0
    result["hydrogen_bond_proxy_score"] = 0.0
    result["stacking_proxy_score"] = 0.0
    result["feature_based_support_score"] = (
        0.35 * result["target_binding_strength_score"]
        + 0.25 * gap_component
        + 0.20 * risk_component
        + 0.20 * result["beneficial_landmark_proximity_score"]
    ).clip(0, 1)
    return result
