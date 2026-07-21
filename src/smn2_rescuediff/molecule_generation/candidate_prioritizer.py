"""Select high-quality, diverse pre-docking candidates for Module 4."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .reward_functions import rna_ligand_like_score


def add_candidate_scores(molecules: pd.DataFrame) -> pd.DataFrame:
    """Add RNA-ligand-like and pre-docking candidate scores."""
    if molecules.empty:
        return molecules.copy()
    result = molecules.copy()
    result["rna_ligand_like_score"] = [rna_ligand_like_score(row) for _, row in result.iterrows()]
    qed = pd.to_numeric(result.get("qed", 0), errors="coerce").fillna(0)
    novelty = result.get("is_novel_vs_training", False).astype(bool).astype(float) if "is_novel_vs_training" in result else 0.0
    lipinski = result.get("lipinski_pass", False).astype(bool).astype(float) if "lipinski_pass" in result else 0.0
    pains_free = (~result.get("pains_alert", False).astype(bool)).astype(float) if "pains_alert" in result else 1.0
    charge = pd.to_numeric(result.get("formal_charge", 0), errors="coerce").fillna(0).abs()
    charge_score = (1.0 - (charge / 3.0)).clip(lower=0.0, upper=1.0)
    result["pre_docking_candidate_score"] = (
        0.30 * result["rna_ligand_like_score"]
        + 0.25 * qed
        + 0.15 * novelty
        + 0.15 * lipinski
        + 0.10 * pains_free
        + 0.05 * charge_score
    ).round(6)
    return result


def select_docking_candidates(molecules: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Select top candidates with optional scaffold diversity."""
    if molecules.empty:
        return molecules.copy()
    cfg = config.get("candidate_prioritization", {})
    max_count = int(cfg.get("max_candidates_to_export_for_docking", 10000))
    work = add_candidate_scores(molecules)
    work = work.sort_values("pre_docking_candidate_score", ascending=False).copy()
    if bool(cfg.get("diversity_selection_enabled", True)) and "bemis_murcko_scaffold" in work:
        selected_rows = []
        scaffold_seen: set[str] = set()
        for _, row in work.iterrows():
            scaffold = str(row.get("bemis_murcko_scaffold", "NA"))
            if scaffold not in scaffold_seen or len(scaffold_seen) >= max_count // 2:
                selected_rows.append(row)
                scaffold_seen.add(scaffold)
            if len(selected_rows) >= max_count:
                break
        selected = pd.DataFrame(selected_rows)
    else:
        selected = work.head(max_count).copy()
    work["selected_for_docking"] = work["canonical_smiles"].isin(set(selected["canonical_smiles"])) if not selected.empty else False
    selected = work[work["selected_for_docking"]].head(max_count).copy()
    return selected

