"""SMN2 similarity and bias-match scoring for decoy motifs."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .feature_calculator import hamming_distance, kmer_jaccard_similarity, levenshtein_distance
from .target_loader import TargetReference


def _bounded(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def length_similarity(length: int, reference_length: int) -> float:
    """Normalized length similarity from 0 to 1."""
    if not length or not reference_length:
        return 0.0
    return _bounded(1.0 - abs(length - reference_length) / max(length, reference_length))


def frequency_similarity(row: pd.Series, reference: TargetReference) -> float:
    """Average similarity of A/U/G/C frequencies."""
    diffs = [
        abs(float(row["A_freq"]) - reference.a_freq),
        abs(float(row["U_freq"]) - reference.u_freq),
        abs(float(row["G_freq"]) - reference.g_freq),
        abs(float(row["C_freq"]) - reference.c_freq),
    ]
    return _bounded(1.0 - sum(diffs) / 2.0)


def score_decoy_row(row: pd.Series, reference: TargetReference, kmer_size: int = 3) -> dict[str, Any]:
    """Calculate distance and normalized similarity scores against SMN2."""
    sequence = str(row["sequence_rna"])
    reference_sequence = reference.sequence_rna
    if len(sequence) == len(reference_sequence):
        hamming: int | str = hamming_distance(sequence, reference_sequence)
    else:
        hamming = "NA"
    levenshtein = levenshtein_distance(sequence, reference_sequence)
    kmer_jaccard = kmer_jaccard_similarity(sequence, reference_sequence, kmer_size)
    gc_similarity = _bounded(1.0 - abs(float(row["GC_content"]) - reference.gc_content))
    splice_similarity = _bounded(1.0 - abs(float(row["simple_splice_strength_score"]) - reference.simple_splice_strength_score))
    u1_similarity = _bounded(1.0 - abs(float(row["u1_complementarity_score"]) - reference.u1_complementarity_score))
    len_similarity = length_similarity(int(row["sequence_length"]), reference.sequence_length)
    smn2_similarity = (
        0.25 * gc_similarity
        + 0.25 * kmer_jaccard
        + 0.20 * splice_similarity
        + 0.20 * u1_similarity
        + 0.10 * len_similarity
    )
    base_frequency_similarity = frequency_similarity(row, reference)
    bias_match_score = (
        0.30 * gc_similarity
        + 0.25 * base_frequency_similarity
        + 0.20 * kmer_jaccard
        + 0.15 * len_similarity
        + 0.10 * ((splice_similarity + u1_similarity) / 2.0)
    )
    return {
        "hamming_distance_to_smn2": hamming,
        "levenshtein_distance_to_smn2": levenshtein,
        "kmer_jaccard_similarity_to_smn2": round(kmer_jaccard, 6),
        "smn2_similarity_score": round(_bounded(smn2_similarity), 6),
        "bias_match_score": round(_bounded(bias_match_score), 6),
    }


def add_similarity_scores(decoys: pd.DataFrame, reference: TargetReference, kmer_size: int = 3) -> pd.DataFrame:
    """Append SMN2 similarity and bias-match scores."""
    if decoys.empty:
        return decoys.copy()
    score_rows = [score_decoy_row(row, reference, kmer_size) for _, row in decoys.iterrows()]
    scores = pd.DataFrame(score_rows, index=decoys.index)
    without_existing = decoys.drop(columns=[col for col in scores.columns if col in decoys.columns], errors="ignore")
    return pd.concat([without_existing, scores], axis=1)


def add_biological_weights(decoys: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Attach configured biological relevance weights to decoy categories."""
    weights = config.get("biological_weights", {})
    result = decoys.copy()
    result["biological_weight"] = result["decoy_category"].map(lambda category: float(weights.get(category, 0.0)))
    return result

