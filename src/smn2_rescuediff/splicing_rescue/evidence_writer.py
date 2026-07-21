"""Evidence trace writers for top Module 5 candidates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def _json_safe(value: Any) -> Any:
    if isinstance(value, (list, dict, tuple)):
        return value
    try:
        missing = pd.isna(value)
        if isinstance(missing, bool) and missing:
            return "NA"
    except Exception:
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _record_subset(row: pd.Series, columns: list[str]) -> dict[str, Any]:
    return {column: _json_safe(row.get(column, "NA")) for column in columns}


def _why_ranked_high(row: pd.Series) -> list[str]:
    reasons: list[str] = []
    if pd.to_numeric(pd.Series([row.get("target_deltaG")]), errors="coerce").fillna(0).iloc[0] <= -7:
        reasons.append("Strong predicted SMN2 target binding")
    if pd.to_numeric(pd.Series([row.get("weighted_selectivity_gap")]), errors="coerce").fillna(0).iloc[0] > 0:
        reasons.append("Positive weighted selectivity gap against HuRDA decoys")
    if pd.to_numeric(pd.Series([row.get("predicted_splicing_rescue_score")]), errors="coerce").fillna(0).iloc[0] >= 0.5:
        reasons.append("High predicted splicing-rescue proxy score")
    if bool(str(row.get("lipinski_pass", "")).lower() in {"true", "1", "yes"}):
        reasons.append("Acceptable Lipinski-style chemical property profile")
    if not reasons:
        reasons.append("Ranked by the aggregate SRSI prioritization score")
    return reasons


def _main_concerns(row: pd.Series) -> list[str]:
    concerns = ["Computational prediction only", "Requires experimental validation before biological interpretation"]
    if bool(row.get("is_dry_run_input", False)):
        concerns.insert(0, "Dry-run/simulated docking input; requires real docking before interpretation")
    if pd.to_numeric(pd.Series([row.get("off_target_risk_score")]), errors="coerce").fillna(1).iloc[0] >= 0.6:
        concerns.append("Elevated HuRDA off-target risk proxy")
    if str(row.get("pose_analysis_status", "")).lower().find("fallback") >= 0:
        concerns.append("Pose-based contact evidence unavailable or incomplete")
    return concerns


def build_evidence_trace(final_ranked: pd.DataFrame, top_n: int) -> dict[str, Any]:
    """Build per-candidate evidence trace for top candidates."""
    evidence: dict[str, Any] = {}
    for _, row in final_ranked.head(top_n).iterrows():
        molecule_id = str(row.get("molecule_id", "NA"))
        evidence[molecule_id] = {
            "rank": int(row.get("final_rank", 0)),
            "canonical_smiles": _json_safe(row.get("canonical_smiles", "NA")),
            "why_ranked_high": _why_ranked_high(row),
            "main_concerns": _main_concerns(row),
            "module_3_evidence": _record_subset(
                row,
                [
                    "source_generator",
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
                ],
            ),
            "module_4_evidence": _record_subset(
                row,
                [
                    "target_deltaG",
                    "best_decoy_deltaG",
                    "weighted_mean_decoy_deltaG",
                    "weighted_selectivity_gap",
                    "percent_decoys_stronger_than_target",
                    "off_target_risk_score",
                    "is_dry_run_input",
                ],
            ),
            "module_5_evidence": _record_subset(
                row,
                [
                    "predicted_splicing_rescue_score",
                    "splice_site_occlusion_penalty",
                    "srsi_score",
                    "srsi_percentile",
                    "candidate_tier",
                    "confidence_level",
                    "recommended_next_step",
                    "evidence_summary",
                ],
            ),
        }
    return evidence


def write_evidence_json(final_ranked: pd.DataFrame, top_n: int, path: str | Path) -> dict[str, Any]:
    """Write top-candidate evidence trace JSON."""
    evidence = build_evidence_trace(final_ranked, top_n)
    Path(path).write_text(json.dumps(evidence, indent=2, default=str), encoding="utf-8")
    return evidence


def write_warnings_json(warnings: list[str], path: str | Path) -> None:
    """Write scoring warnings to JSON."""
    payload = {"warnings": warnings}
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
