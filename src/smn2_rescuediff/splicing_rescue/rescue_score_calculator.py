"""Predicted splicing-rescue proxy score calculation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .numeric import coerce_numeric_columns, safe_float
from .splicing_feature_scorer import numeric_series


RESCUE_COLUMNS = [
    "molecule_id",
    "canonical_smiles",
    "target_deltaG",
    "weighted_selectivity_gap",
    "off_target_risk_score",
    "target_binding_strength_score",
    "beneficial_landmark_proximity_score",
    "u1_support_score",
    "exon7_boundary_support_score",
    "contact_density_score",
    "hydrogen_bond_proxy_score",
    "stacking_proxy_score",
    "splice_site_occlusion_penalty",
    "pose_support_score",
    "feature_based_support_score",
    "predicted_splicing_rescue_score",
    "splicing_rescue_score_percentile",
    "splicing_rescue_tier",
    "pose_analysis_status",
    "run_mode",
    "is_dry_run_input",
    "notes",
]

NUMERIC_RESCUE_COLUMNS = {
    "target_deltaG",
    "weighted_selectivity_gap",
    "off_target_risk_score",
    "target_binding_strength_score",
    "beneficial_landmark_proximity_score",
    "u1_support_score",
    "exon7_boundary_support_score",
    "contact_density_score",
    "hydrogen_bond_proxy_score",
    "stacking_proxy_score",
    "splice_site_occlusion_penalty",
    "pose_support_score",
    "feature_based_support_score",
    "predicted_splicing_rescue_score",
    "splicing_rescue_score_percentile",
}


def _percentiles(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    return series.rank(method="average", pct=True) * 100.0


def _tier(score: float, percentile: float, threshold: float) -> str:
    if pd.isna(score) or score <= 0:
        return "insufficient_evidence"
    if score >= max(threshold, 0.75) or percentile >= 85:
        return "high_predicted_rescue"
    if score >= max(threshold * 0.75, 0.50) or percentile >= 60:
        return "moderate_predicted_rescue"
    if score >= max(threshold * 0.45, 0.25):
        return "low_predicted_rescue"
    return "insufficient_evidence"


def calculate_splicing_rescue_scores(
    candidates: pd.DataFrame,
    contacts: pd.DataFrame,
    feature_scores: pd.DataFrame,
    config: dict,
    used_dry_run_inputs: bool,
) -> pd.DataFrame:
    """Calculate normalized predicted splicing-rescue proxy scores."""
    scoring = config.get("splicing_rescue_scoring", {})
    weights = scoring.get("weights", {})
    thresholds = scoring.get("thresholds", {})
    score_mode = str(scoring.get("score_mode", "hybrid_pose_and_feature"))
    minimum_score = safe_float(thresholds.get("minimum_splicing_rescue_score_for_advancement", 0.60), 0.60)

    work = candidates.copy()
    work = work.merge(feature_scores, on="molecule_id", how="left", suffixes=("", "_feature"))
    work = work.merge(contacts, on="molecule_id", how="left", suffixes=("", "_contact"))
    for column in [
        "target_binding_strength_score",
        "beneficial_landmark_proximity_score",
        "u1_support_score",
        "exon7_boundary_support_score",
        "contact_density_score",
        "hydrogen_bond_proxy_score",
        "stacking_proxy_score",
        "feature_based_support_score",
        "pose_support_score",
        "splice_site_occlusion_penalty",
    ]:
        if column not in work:
            work[column] = 0.0
        work[column] = pd.to_numeric(work[column], errors="coerce").fillna(0.0).clip(0, 1)

    missing_pose = work.get("pose_analysis_status", pd.Series(["missing_pose_feature_based_fallback"] * len(work))).astype(str).str.contains(
        "missing|failed|fallback|no_receptor", case=False, regex=True
    )
    off_target = numeric_series(work, "off_target_risk_score", default=0.5).clip(0, 1)

    positive = (
        safe_float(weights.get("beneficial_landmark_proximity", 2.0), 2.0) * work["beneficial_landmark_proximity_score"]
        + safe_float(weights.get("u1_support_region_proximity", 2.0), 2.0) * work["u1_support_score"]
        + safe_float(weights.get("exon7_boundary_support", 1.5), 1.5) * work["exon7_boundary_support_score"]
        + safe_float(weights.get("target_binding_strength", 1.5), 1.5) * work["target_binding_strength_score"]
        + safe_float(weights.get("contact_density", 1.0), 1.0) * work["contact_density_score"]
        + safe_float(weights.get("hydrogen_bond_proxy", 0.8), 0.8) * work["hydrogen_bond_proxy_score"]
        + safe_float(weights.get("stacking_proxy", 0.7), 0.7) * work["stacking_proxy_score"]
    )
    if score_mode in {"feature_based", "dry_run_input_analysis"}:
        positive += 2.0 * work["feature_based_support_score"]
    elif score_mode == "hybrid_pose_and_feature":
        positive += missing_pose.astype(float) * 2.0 * work["feature_based_support_score"]
        positive += (~missing_pose).astype(float) * 1.5 * work["pose_support_score"]
    else:
        positive += 1.5 * work["pose_support_score"]

    negative = (
        safe_float(weights.get("splice_site_occlusion_penalty", 2.0), 2.0) * work["splice_site_occlusion_penalty"]
        + safe_float(weights.get("off_target_risk_penalty", 1.5), 1.5) * off_target
        + safe_float(weights.get("missing_pose_penalty", 0.5), 0.5) * missing_pose.astype(float)
    )
    if used_dry_run_inputs:
        negative += 0.10
    denominator = max(1e-9, float(sum(abs(safe_float(value, 0.0)) for value in weights.values())) + 2.0)
    score = ((positive - negative) / denominator).clip(lower=0.0, upper=1.0)
    work["predicted_splicing_rescue_score"] = score.round(6)
    work["splicing_rescue_score_percentile"] = _percentiles(work["predicted_splicing_rescue_score"]).round(3)
    work["splicing_rescue_tier"] = [
        _tier(safe_float(score_value), safe_float(percentile, 0.0), minimum_score)
        for score_value, percentile in zip(work["predicted_splicing_rescue_score"], work["splicing_rescue_score_percentile"], strict=False)
    ]
    if "pose_analysis_status" not in work:
        work["pose_analysis_status"] = "missing_pose_feature_based_fallback"
    if "run_mode" not in work:
        work["run_mode"] = "NA"
    work["is_dry_run_input"] = bool(used_dry_run_inputs)
    work["notes"] = [
        (
            "DRY-RUN / SIMULATED DOCKING INPUTS - NOT SCIENTIFIC RESULTS. Predicted splicing-rescue proxy score only."
            if used_dry_run_inputs
            else "Predicted splicing-rescue proxy score only; not a measured splicing-rescue result."
        )
        for _ in range(len(work))
    ]
    for column in RESCUE_COLUMNS:
        if column not in work:
            work[column] = np.nan if column in NUMERIC_RESCUE_COLUMNS else "NA"
    return coerce_numeric_columns(work[RESCUE_COLUMNS], NUMERIC_RESCUE_COLUMNS)
