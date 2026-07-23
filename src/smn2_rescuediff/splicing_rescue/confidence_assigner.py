"""Assign cautious confidence levels to Module 5 candidate rankings."""

from __future__ import annotations

import pandas as pd


CONFIDENCE_ORDER = ["very_low", "low", "moderate", "high"]


def _lower_confidence(level: str) -> str:
    if level not in CONFIDENCE_ORDER:
        return "very_low"
    idx = max(0, CONFIDENCE_ORDER.index(level) - 1)
    return CONFIDENCE_ORDER[idx]


def _raise_confidence(level: str) -> str:
    if level not in CONFIDENCE_ORDER:
        return "low"
    idx = min(len(CONFIDENCE_ORDER) - 1, CONFIDENCE_ORDER.index(level) + 1)
    return CONFIDENCE_ORDER[idx]


def _as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _numeric_value(row: pd.Series, columns: list[str], default: float = 0.0) -> float:
    for column in columns:
        value = pd.to_numeric(pd.Series([row.get(column, None)]), errors="coerce").iloc[0]
        if pd.notna(value):
            return float(value)
    return default


def assign_confidence(df: pd.DataFrame, config: dict, used_dry_run_inputs: bool) -> pd.Series:
    """Assign confidence levels without implying experimental validation."""
    cfg = config.get("confidence", {})
    if not bool(cfg.get("assign_confidence_level", True)):
        return pd.Series(["not_assigned"] * len(df), index=df.index)
    levels = []
    for _, row in df.iterrows():
        run_mode = str(row.get("run_mode", "NA"))
        valid_decoys = _numeric_value(row, ["num_decoys_with_valid_scores", "num_decoys_with_scores"], default=0.0)
        weighted_gap = pd.to_numeric(pd.Series([row.get("weighted_selectivity_gap", None)]), errors="coerce").iloc[0]
        has_real_decoy_selectivity = _as_bool(row.get("has_real_decoy_selectivity", False)) or (
            valid_decoys > 0 and not _as_bool(row.get("is_dry_run_score", False)) and pd.notna(weighted_gap)
        )
        if has_real_decoy_selectivity and run_mode in {"", "NA", "nan"}:
            run_mode = "target_plus_decoy_docking"
        if used_dry_run_inputs or _as_bool(row.get("is_dry_run_input", False)):
            level = str(cfg.get("dry_run_confidence", "very_low"))
        else:
            if run_mode == "target_plus_decoy_docking" or has_real_decoy_selectivity:
                small_panel = float(cfg.get("moderate_confidence_min_decoys_with_valid_scores", 100))
                large_panel = float(cfg.get("high_confidence_min_decoys_with_valid_scores", 500))
                ablation_enabled = bool(config.get("ablation_analysis", {}).get("enabled", True))
                status = str(row.get("pose_analysis_status", ""))
                if valid_decoys >= large_panel and status == "pose_based_proxy" and ablation_enabled:
                    level = str(cfg.get("large_decoy_pose_ablation_confidence", "high"))
                elif valid_decoys >= small_panel:
                    level = str(cfg.get("target_plus_decoy_docking_confidence", "moderate"))
                elif valid_decoys > 0:
                    level = str(cfg.get("target_plus_small_decoy_panel_confidence", "moderate"))
                else:
                    level = str(cfg.get("target_only_docking_confidence", "low"))
            elif run_mode == "target_only_docking":
                level = str(cfg.get("target_only_docking_confidence", "low"))
            else:
                level = "low"
        status = str(row.get("pose_analysis_status", ""))
        if bool(cfg.get("pose_based_scoring_confidence_bonus", True)) and status == "pose_based_proxy" and not used_dry_run_inputs and level == "low" and run_mode != "target_only_docking":
            level = _raise_confidence(level)
        if bool(cfg.get("missing_pose_confidence_penalty", True)) and any(token in status for token in ["missing", "fallback", "failed", "no_receptor"]):
            level = _lower_confidence(level)
        levels.append(level)
    return pd.Series(levels, index=df.index)
