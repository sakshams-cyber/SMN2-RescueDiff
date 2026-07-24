"""Select high-quality, diverse pre-docking candidates for Module 4."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .chemical_filter import require_rdkit
from .novelty_analysis import _fingerprint
from .reward_functions import rna_ligand_like_score


def _numeric(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df:
        return pd.Series([default] * len(df), index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").fillna(default)


def _bool01(df: pd.DataFrame, column: str, default: bool = False) -> pd.Series:
    if column not in df:
        return pd.Series([default] * len(df), index=df.index).astype(float)
    return df[column].astype(str).str.lower().isin({"true", "1", "yes"}).astype(float)


def _range_score(values: pd.Series, low: float, high: float) -> pd.Series:
    midpoint = (low + high) / 2.0
    half_width = max((high - low) / 2.0, 1.0)
    return (1.0 - ((values - midpoint).abs() / half_width)).clip(lower=0.0, upper=1.0)


def add_candidate_scores(molecules: pd.DataFrame) -> pd.DataFrame:
    """Add quality-weighted RNA-ligand-like and pre-docking candidate scores."""
    if molecules.empty:
        return molecules.copy()
    result = molecules.copy()
    result["rna_ligand_like_score"] = [rna_ligand_like_score(row) for _, row in result.iterrows()]
    qed = _numeric(result, "qed", 0.0).clip(0, 1)
    mw = _numeric(result, "molecular_weight", 0.0)
    logp = _numeric(result, "logp", 0.0)
    hbd = _numeric(result, "hbd", 0.0)
    hba = _numeric(result, "hba", 0.0)
    rotatable = _numeric(result, "rotatable_bonds", 0.0)
    tpsa = _numeric(result, "tpsa", 0.0)
    novelty_bool = _bool01(result, "is_novel_vs_training", False)
    max_tanimoto = _numeric(result, "max_tanimoto_to_training", 1.0)
    result["novelty_vs_training_set"] = ((0.5 * novelty_bool) + (0.5 * (1.0 - max_tanimoto).clip(0, 1))).round(6)
    lipinski = _bool01(result, "lipinski_pass", False)
    pains_free = 1.0 - _bool01(result, "pains_alert", False)
    charge = _numeric(result, "formal_charge", 0.0).abs()
    charge_score = (1.0 - (charge / 3.0)).clip(lower=0.0, upper=1.0)
    synthetic = _numeric(result, "synthetic_accessibility_proxy", 0.5).clip(0, 1)
    property_fit = (
        0.24 * _range_score(mw, 220, 520)
        + 0.18 * _range_score(logp, 0.0, 4.5)
        + 0.16 * _range_score(hbd, 0, 5)
        + 0.16 * _range_score(hba, 2, 10)
        + 0.12 * _range_score(rotatable, 0, 8)
        + 0.14 * _range_score(tpsa, 35, 140)
    ).clip(0, 1)
    result["chemical_quality_score"] = (
        0.30 * qed
        + 0.22 * lipinski
        + 0.18 * pains_free
        + 0.15 * property_fit
        + 0.10 * synthetic
        + 0.05 * charge_score
    ).round(6)
    result["pre_docking_candidate_score"] = (
        0.32 * result["rna_ligand_like_score"]
        + 0.28 * result["chemical_quality_score"]
        + 0.18 * result["novelty_vs_training_set"]
        + 0.12 * qed
        + 0.10 * charge_score
    ).round(6)
    result["final_generation_selection_score"] = (
        0.40 * result["pre_docking_candidate_score"]
        + 0.25 * result["chemical_quality_score"]
        + 0.20 * result["rna_ligand_like_score"]
        + 0.15 * result["novelty_vs_training_set"]
    ).round(6)
    return result


def _diverse_by_tanimoto(work: pd.DataFrame, max_count: int, threshold: float) -> pd.DataFrame:
    """Greedily select high-scoring molecules below a Tanimoto redundancy threshold."""
    if work.empty:
        return work.copy()
    require_rdkit()
    from rdkit import DataStructs

    selected_rows = []
    selected_fps = []
    for _, row in work.iterrows():
        try:
            fp = _fingerprint(str(row["canonical_smiles"]))
        except Exception:
            continue
        if selected_fps:
            sims = DataStructs.BulkTanimotoSimilarity(fp, selected_fps)
            if max(sims) > threshold:
                continue
        selected_rows.append(row)
        selected_fps.append(fp)
        if len(selected_rows) >= max_count:
            break
    if len(selected_rows) < max_count:
        selected_ids = {str(row["canonical_smiles"]) for row in selected_rows}
        for _, row in work.iterrows():
            if str(row["canonical_smiles"]) in selected_ids:
                continue
            selected_rows.append(row)
            selected_ids.add(str(row["canonical_smiles"]))
            if len(selected_rows) >= max_count:
                break
    return pd.DataFrame(selected_rows)


def select_docking_candidates(molecules: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Select top quality-weighted candidates with optional Morgan/Tanimoto diversity."""
    if molecules.empty:
        return molecules.copy()
    cfg = config.get("candidate_prioritization", {})
    max_count = int(cfg.get("max_candidates_to_export_for_docking", 10000))
    work = add_candidate_scores(molecules)
    score_column = "final_generation_selection_score" if "final_generation_selection_score" in work else "pre_docking_candidate_score"
    min_quality = float(cfg.get("min_final_generation_selection_score", 0.0))
    work = work[pd.to_numeric(work[score_column], errors="coerce").fillna(0.0) >= min_quality].copy()
    work = work.sort_values(score_column, ascending=False).copy()
    if bool(cfg.get("morgan_diversity_selection_enabled", cfg.get("diversity_selection_enabled", True))):
        threshold = float(cfg.get("tanimoto_similarity_threshold_for_diversity", 0.85))
        selected = _diverse_by_tanimoto(work, max_count=max_count, threshold=threshold)
    elif bool(cfg.get("diversity_selection_enabled", True)) and "bemis_murcko_scaffold" in work:
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
