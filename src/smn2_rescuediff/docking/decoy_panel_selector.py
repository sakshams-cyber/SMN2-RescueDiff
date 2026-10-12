"""HuRDA decoy panel selection for counter-screening."""

from __future__ import annotations

from pathlib import Path
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


def _has_prepared_pdbqt(value: object, project_root: Path) -> bool:
    """Return whether a receptor value names an existing, nonempty PDBQT file."""
    text = str(value).strip()
    if not text or text.lower() in {"na", "nan", "none", "null"}:
        return False
    path = Path(text)
    if path.suffix.lower() != ".pdbqt":
        return False
    resolved = path if path.is_absolute() else project_root / path
    return resolved.is_file() and resolved.stat().st_size > 0


def select_decoy_panel(
    decoys: pd.DataFrame,
    config: dict[str, Any],
    project_root: str | Path | None = None,
) -> pd.DataFrame:
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
    include_categories = panel_cfg.get("include_categories") or []
    if include_categories:
        work = work[work["decoy_category"].isin(include_categories)].copy()
    min_similarity = panel_cfg.get("min_smn2_similarity")
    if min_similarity is not None:
        work = work[work["smn2_similarity_score"] >= float(min_similarity)].copy()
    min_weight = panel_cfg.get("min_biological_weight")
    if min_weight is not None:
        work = work[work["biological_weight"] >= float(min_weight)].copy()
    if work.empty:
        raise ValueError("No HuRDA decoys remain after configured panel filters.")
    receptor_column = str(panel_cfg.get("decoy_receptor_file_column", "decoy_receptor_file"))
    allow_sequence_only = bool(panel_cfg.get("allow_sequence_only_decoy_panel", True))
    if not allow_sequence_only:
        if receptor_column not in work.columns:
            raise ValueError(
                "Decoy docking requires prepared receptor PDBQT files, but the configured "
                f"column is missing: {receptor_column}"
            )
        root = Path(project_root).resolve() if project_root is not None else Path.cwd().resolve()
        work = work[work[receptor_column].map(lambda value: _has_prepared_pdbqt(value, root))].copy()
        if work.empty:
            raise ValueError("No HuRDA decoys with existing, nonempty receptor PDBQT files remain after filtering.")
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
    if not allow_sequence_only and len(selected) < total:
        raise ValueError(
            f"Requested {total} dockable HuRDA decoys, but only {len(selected)} decoys with prepared receptor PDBQT files are available."
        )
    selected["selected_for_docking"] = True
    if receptor_column in selected.columns:
        selected["decoy_receptor_file"] = selected[receptor_column].fillna("NA").astype(str)
    else:
        selected["decoy_receptor_file"] = "NA"
    selected["notes"] = (
        "Hard decoy panel selected by category, SMN2 similarity, biological weight, and bias-match score. "
        + (
            "Sequence-only rows may remain counter-screen candidates."
            if allow_sequence_only
            else "Every selected row has an existing, nonempty receptor PDBQT file."
        )
    )
    for column in PANEL_COLUMNS:
        if column not in selected.columns:
            selected[column] = "NA"
    return selected[PANEL_COLUMNS]
