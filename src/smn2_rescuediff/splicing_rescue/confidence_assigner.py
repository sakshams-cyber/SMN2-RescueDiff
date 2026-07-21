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


def assign_confidence(df: pd.DataFrame, config: dict, used_dry_run_inputs: bool) -> pd.Series:
    """Assign confidence levels without implying experimental validation."""
    cfg = config.get("confidence", {})
    if not bool(cfg.get("assign_confidence_level", True)):
        return pd.Series(["not_assigned"] * len(df), index=df.index)
    levels = []
    for _, row in df.iterrows():
        run_mode = str(row.get("run_mode", "NA"))
        if used_dry_run_inputs or bool(row.get("is_dry_run_input", False)):
            level = str(cfg.get("dry_run_confidence", "very_low"))
        else:
            if run_mode == "target_plus_decoy_docking":
                valid_decoys = pd.to_numeric(pd.Series([row.get("num_decoys_with_valid_scores", 0)]), errors="coerce").fillna(0).iloc[0]
                small_panel = float(cfg.get("moderate_confidence_min_decoys_with_valid_scores", 100))
                large_panel = float(cfg.get("high_confidence_min_decoys_with_valid_scores", 500))
                ablation_enabled = bool(config.get("ablation_analysis", {}).get("enabled", True))
                status = str(row.get("pose_analysis_status", ""))
                if valid_decoys >= large_panel and status == "pose_based_proxy" and ablation_enabled:
                    level = str(cfg.get("large_decoy_pose_ablation_confidence", "high"))
                elif valid_decoys >= small_panel:
                    level = str(cfg.get("target_plus_decoy_docking_confidence", "moderate"))
                else:
                    level = str(cfg.get("target_plus_small_decoy_panel_confidence", "moderate"))
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
