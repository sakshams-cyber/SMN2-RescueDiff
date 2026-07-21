"""Final SRSI ranking for Module 5."""

from __future__ import annotations

import pandas as pd

from .confidence_assigner import assign_confidence
from .splicing_feature_scorer import numeric_series, zscore


FINAL_COLUMNS = [
    "final_rank",
    "molecule_id",
    "canonical_smiles",
    "source_generator",
    "target_deltaG",
    "target_binding_strength_score",
    "best_decoy_deltaG",
    "weighted_mean_decoy_deltaG",
    "weighted_selectivity_gap",
    "percent_decoys_stronger_than_target",
    "off_target_risk_score",
    "predicted_splicing_rescue_score",
    "splice_site_occlusion_penalty",
    "molecular_weight",
    "logp",
    "hbd",
    "hba",
    "rotatable_bonds",
    "formal_charge",
    "qed",
    "sa_score",
    "lipinski_pass",
    "pains_alert",
    "pre_docking_candidate_score",
    "srsi_score",
    "srsi_percentile",
    "candidate_tier",
    "confidence_level",
    "recommended_next_step",
    "is_dry_run_input",
    "evidence_summary",
    "notes",
]


def _bool_series(df: pd.DataFrame, column: str, default: bool = False) -> pd.Series:
    if column not in df:
        return pd.Series([default] * len(df), index=df.index)
    return df[column].astype(str).str.lower().isin({"true", "1", "yes"})


def _percentiles(series: pd.Series) -> pd.Series:
    return series.rank(method="average", pct=True) * 100.0 if len(series) else series


def _candidate_tier(row: pd.Series, config: dict) -> str:
    tiers = config.get("srsi", {}).get("candidate_tiers", {})
    percentile = float(row.get("srsi_percentile", 0.0))
    rescue = float(row.get("predicted_splicing_rescue_score", 0.0))
    risk = float(pd.to_numeric(pd.Series([row.get("off_target_risk_score", 1.0)]), errors="coerce").fillna(1.0).iloc[0])
    if pd.isna(row.get("srsi_score")) or rescue <= 0:
        return "insufficient_evidence"
    if risk >= 0.80 or rescue < 0.20:
        return "not_recommended"
    if percentile >= float(tiers.get("high_priority_min_srsi_percentile", 90)) and rescue >= 0.50 and risk <= 0.65:
        return "high_priority_computational_candidate"
    if percentile >= float(tiers.get("medium_priority_min_srsi_percentile", 70)) and rescue >= 0.35 and risk <= 0.75:
        return "medium_priority_computational_candidate"
    if percentile >= float(tiers.get("low_priority_min_srsi_percentile", 50)) and rescue >= 0.25:
        return "low_priority_computational_candidate"
    return "not_recommended"


def _recommended_next_step(row: pd.Series) -> str:
    if bool(row.get("is_dry_run_input", False)):
        return "requires real docking before interpretation"
    status = str(row.get("pose_analysis_status", ""))
    if any(token in status for token in ["missing", "fallback", "failed", "no_receptor"]):
        return "requires pose validation"
    if float(row.get("off_target_risk_score", 1.0)) >= 0.80:
        return "not recommended due to off-target risk"
    if float(row.get("predicted_splicing_rescue_score", 0.0)) < 0.25:
        return "not recommended due to poor splicing-rescue proxy score"
    return "future experimental validation required"


def _evidence_summary(row: pd.Series) -> str:
    parts = [
        f"target_deltaG={row.get('target_deltaG', 'NA')}",
        f"weighted_selectivity_gap={row.get('weighted_selectivity_gap', 'NA')}",
        f"splicing_rescue_proxy={row.get('predicted_splicing_rescue_score', 'NA')}",
        f"off_target_risk={row.get('off_target_risk_score', 'NA')}",
        f"confidence={row.get('confidence_level', 'NA')}",
    ]
    if bool(row.get("is_dry_run_input", False)):
        parts.append("dry-run/simulated input")
    return "; ".join(parts)


def calculate_srsi(candidates: pd.DataFrame, rescue_scores: pd.DataFrame, config: dict, used_dry_run_inputs: bool) -> pd.DataFrame:
    """Merge evidence and calculate final SRSI rankings."""
    work = candidates.copy()
    rescue_cols = [column for column in rescue_scores.columns if column not in work.columns or column == "molecule_id"]
    work = work.merge(rescue_scores[rescue_cols], on="molecule_id", how="left")
    for column in ["canonical_smiles", "source_generator"]:
        alt = f"{column}_chem"
        if column not in work:
            work[column] = work.get(alt, "NA")
    weights = config.get("srsi", {}).get("weights", {})
    binding_strength = -numeric_series(work, "target_deltaG", default=0.0)
    weighted_gap = numeric_series(work, "weighted_selectivity_gap", default=0.0)
    rescue = numeric_series(work, "predicted_splicing_rescue_score", default=0.0)
    qed = numeric_series(work, "qed", default=0.0)
    pre_docking = numeric_series(work, "pre_docking_candidate_score", default=0.0)
    off_target = numeric_series(work, "off_target_risk_score", default=1.0)
    sa_score = numeric_series(work, "sa_score", default=0.0)
    occlusion = numeric_series(work, "splice_site_occlusion_penalty", default=0.0)
    lipinski = _bool_series(work, "lipinski_pass", default=False).astype(float)
    pains = _bool_series(work, "pains_alert", default=False).astype(float)
    srsi = (
        float(weights.get("smn2_binding_strength", 1.0)) * zscore(binding_strength)
        + float(weights.get("splicing_rescue_score", 2.0)) * zscore(rescue)
        + float(weights.get("weighted_selectivity_gap", 1.5)) * zscore(weighted_gap)
        + float(weights.get("qed", 0.7)) * zscore(qed)
        + float(weights.get("lipinski_bonus", 0.4)) * lipinski
        + float(weights.get("pre_docking_candidate_score", 0.5)) * zscore(pre_docking)
        - float(weights.get("off_target_risk_penalty", 1.5)) * zscore(off_target)
        - float(weights.get("sa_score_penalty", 0.5)) * zscore(sa_score)
        - float(weights.get("pains_penalty", 1.0)) * pains
        - float(weights.get("splice_site_occlusion_penalty", 1.2)) * zscore(occlusion)
    )
    if used_dry_run_inputs:
        # Preserve relative ranking for pipeline testing while making the caveat
        # explicit through confidence/recommended-next-step fields.
        srsi = srsi - 0.25
    work["srsi_score"] = srsi.round(6)
    work["srsi_percentile"] = _percentiles(work["srsi_score"]).round(3)
    work["is_dry_run_input"] = bool(used_dry_run_inputs)
    work["confidence_level"] = assign_confidence(work, config, used_dry_run_inputs)
    work["candidate_tier"] = [_candidate_tier(row, config) for _, row in work.iterrows()]
    work["recommended_next_step"] = [_recommended_next_step(row) for _, row in work.iterrows()]
    work = work.sort_values("srsi_score", ascending=False, na_position="last").reset_index(drop=True)
    work["final_rank"] = range(1, len(work) + 1)
    work["evidence_summary"] = [_evidence_summary(row) for _, row in work.iterrows()]
    work["notes"] = [
        (
            "DRY-RUN / SIMULATED DOCKING INPUTS - NOT SCIENTIFIC RESULTS. SRSI is a computational prioritization score only."
            if used_dry_run_inputs
            else "SRSI is a computational prioritization score only; not proof of biological activity."
        )
        for _ in range(len(work))
    ]
    for column in FINAL_COLUMNS:
        if column not in work:
            work[column] = "NA"
    return work[FINAL_COLUMNS]

