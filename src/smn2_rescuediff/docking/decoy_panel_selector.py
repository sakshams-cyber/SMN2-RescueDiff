"""HuRDA decoy panel selection for counter-screening."""

from __future__ import annotations

from typing import Any

import pandas as pd


PANEL_COLUMNS = [
    "decoy_id",
    "decoy_category",
    "motif_type",
    "chromosome",
    "start",
    "end",
    "strand",
    "gene_name",
    "transcript_id",
    "sequence_rna",
    "sequence_length",
    "smn2_similarity_score",
    "bias_match_score",
    "biological_weight",
    "selected_for_docking",
    "selection_reason",
    "decoy_receptor_file",
    "notes",
]


def select_decoy_panel(decoys: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Select a representative HuRDA panel by configured category targets."""
    panel_cfg = config.get("decoy_panel", {})
    if not bool(panel_cfg.get("use_decoy_panel", True)):
        return pd.DataFrame(columns=PANEL_COLUMNS)
    total = int(panel_cfg.get("total_decoys_for_docking", 500))
    category_targets = panel_cfg.get("category_targets", {})
    if decoys.empty:
        raise ValueError("No decoys are available for panel selection.")
    work = decoys.copy()
    for col in ["smn2_similarity_score", "bias_match_score", "biological_weight"]:
        if col not in work.columns:
            work[col] = 0.0
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0.0)
    selected_frames: list[pd.DataFrame] = []
    for category, count in category_targets.items():
        subset = work[work["decoy_category"] == category].copy()
        if subset.empty or int(count) <= 0:
            continue
        subset = subset.sort_values(
            ["smn2_similarity_score", "biological_weight", "bias_match_score"],
            ascending=False,
        ).head(int(count))
        subset["selection_reason"] = "Selected by category target, SMN2 similarity, biological weight, and bias-match score."
        selected_frames.append(subset)
    selected = pd.concat(selected_frames, ignore_index=True) if selected_frames else pd.DataFrame()
    if len(selected) < total:
        already = set(selected["decoy_id"]) if not selected.empty and "decoy_id" in selected else set()
        fill = work[~work["decoy_id"].isin(already)].sort_values(
            ["smn2_similarity_score", "biological_weight", "bias_match_score"],
            ascending=False,
        ).head(total - len(selected))
        fill = fill.copy()
        fill["selection_reason"] = "Fill-in decoy selected to reach configured panel size."
        selected = pd.concat([selected, fill], ignore_index=True) if not fill.empty else selected
    selected = selected.head(total).copy()
    selected["selected_for_docking"] = True
    selected["decoy_receptor_file"] = "NA"
    selected["notes"] = "Sequence-based decoy panel; receptor structures are not assumed at this stage."
    for column in PANEL_COLUMNS:
        if column not in selected.columns:
            selected[column] = "NA"
    return selected[PANEL_COLUMNS]

