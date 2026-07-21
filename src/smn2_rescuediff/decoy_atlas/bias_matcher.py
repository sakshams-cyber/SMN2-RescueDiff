"""Bias matching, hard-decoy selection, and duplicate filtering."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .target_loader import TargetReference


def select_smn2_like_hard_decoys(candidates: pd.DataFrame, config: dict[str, Any], target_count: int) -> pd.DataFrame:
    """Select SMN2-like hard decoys from scored splice-site candidates."""
    if target_count <= 0 or candidates.empty:
        return pd.DataFrame()
    matching = config.get("feature_matching", {})
    min_similarity = float(matching.get("min_smn2_similarity_for_hard_decoys", 0.45))
    max_similarity = float(matching.get("max_smn2_similarity_for_hard_decoys", 0.90))
    scored = candidates[
        (candidates["smn2_similarity_score"] >= min_similarity)
        & (candidates["smn2_similarity_score"] <= max_similarity)
    ].copy()
    if scored.empty:
        scored = candidates.copy()
    selected = scored.sort_values(["smn2_similarity_score", "bias_match_score"], ascending=False).head(target_count).copy()
    selected["decoy_category"] = "smn2_like_hard_decoy"
    selected["motif_type"] = selected["motif_type"].map(lambda value: f"smn2_like_{value}")
    selected["window_name"] = selected["window_name"].map(lambda value: f"hard_{value}")
    selected["notes"] = selected["notes"].astype(str) + " Selected as an SMN2-like hard decoy using simple sequence-feature similarity."
    return selected


def select_random_bias_matched_windows(
    candidates: pd.DataFrame,
    reference: TargetReference,
    config: dict[str, Any],
    target_count: int,
) -> pd.DataFrame:
    """Select random transcriptome windows that approximately match SMN2 target biases."""
    if target_count <= 0 or candidates.empty:
        return pd.DataFrame()
    matching = config.get("feature_matching", {})
    gc_tolerance = float(matching.get("gc_tolerance", 0.08))
    length_must_match = bool(matching.get("length_must_match", True))
    scored = candidates.copy()
    filtered = scored[(scored["GC_content"] - reference.gc_content).abs() <= gc_tolerance]
    if length_must_match:
        filtered = filtered[filtered["sequence_length"] == reference.sequence_length]
    if filtered.empty:
        filtered = scored
    selected = filtered.sort_values(["bias_match_score", "smn2_similarity_score"], ascending=False).head(target_count).copy()
    selected["notes"] = selected["notes"].astype(str) + " Selected by simple GC/base-composition/k-mer bias matching."
    return selected


def remove_duplicate_and_exact_smn2_matches(
    decoys: pd.DataFrame,
    smn2_sequences: set[str],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove duplicate RNA sequences and exact SMN2 target matches."""
    if decoys.empty:
        empty = decoys.copy()
        empty["is_duplicate_removed"] = []
        empty["is_exact_smn2_match"] = []
        return empty, empty
    atlas_cfg = config.get("atlas", {})
    remove_duplicates = bool(atlas_cfg.get("remove_duplicate_sequences", True))
    remove_exact = bool(atlas_cfg.get("remove_exact_smn2_matches", True))
    priority = {
        "smn2_like_hard_decoy": 0,
        "five_prime_splice_site": 1,
        "three_prime_splice_site": 2,
        "exonic_regulatory": 3,
        "intronic_regulatory": 4,
        "random_bias_matched": 5,
        "mutated_smn2_control": 6,
        "scrambled_smn2_control": 7,
    }
    work = decoys.copy()
    work["_original_order"] = range(len(work))
    work["_category_priority"] = work["decoy_category"].map(lambda value: priority.get(value, 99))
    work = work.sort_values(["_category_priority", "smn2_similarity_score", "_original_order"], ascending=[True, False, True])
    exact = work["sequence_rna"].isin(smn2_sequences) if remove_exact else pd.Series(False, index=work.index)
    duplicate = work["sequence_rna"].duplicated(keep="first") if remove_duplicates else pd.Series(False, index=work.index)
    work["is_exact_smn2_match"] = exact
    work["is_duplicate_removed"] = duplicate
    removed = work[exact | duplicate].copy()
    kept = work[~(exact | duplicate)].copy()
    kept["is_exact_smn2_match"] = False
    kept["is_duplicate_removed"] = False
    columns_to_drop = ["_original_order", "_category_priority"]
    return kept.drop(columns=columns_to_drop), removed.drop(columns=columns_to_drop)

