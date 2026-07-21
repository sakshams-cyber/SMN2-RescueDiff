"""Reusable RNA sequence features for HuRDA decoys."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

import pandas as pd
from Bio.Seq import Seq


RNA_BASES = set("AUGC")
DNA_BASES = set("ATGCN")
WC_PAIRS = {("A", "U"), ("U", "A"), ("G", "C"), ("C", "G")}
WOBBLE_PAIRS = {("G", "U"), ("U", "G")}


def dna_to_rna(sequence: str, strand: str = "+") -> str:
    """Convert DNA or RNA text to RNA, reverse-complementing negative strand DNA."""
    seq = sequence.upper().replace(" ", "").replace("\n", "")
    if "T" in seq or "N" in seq:
        invalid = set(seq) - DNA_BASES
        if invalid:
            raise ValueError(f"Invalid DNA characters encountered: {sorted(invalid)}")
        if strand == "-":
            seq = str(Seq(seq).reverse_complement())
        seq = seq.replace("T", "U")
    elif strand == "-":
        dna_like = seq.replace("U", "T")
        seq = str(Seq(dna_like).reverse_complement()).replace("T", "U")
    return seq


def validate_rna(sequence: str) -> bool:
    """Return true when a sequence contains only A/U/G/C."""
    return bool(sequence) and not (set(sequence.upper()) - RNA_BASES)


def kmer_counts(sequence: str, k: int) -> dict[str, int]:
    """Count k-mers in a sequence."""
    if len(sequence) < k:
        return {}
    return dict(Counter(sequence[i : i + k] for i in range(len(sequence) - k + 1)))


def kmer_set(sequence: str, k: int) -> set[str]:
    """Return the set of observed k-mers."""
    return set(kmer_counts(sequence, k))


def hamming_distance(left: str, right: str) -> int:
    """Hamming distance for equal-length strings."""
    if len(left) != len(right):
        raise ValueError("Hamming distance requires equal-length sequences.")
    return sum(a != b for a, b in zip(left, right))


def levenshtein_distance(left: str, right: str) -> int:
    """Compute Levenshtein edit distance with a memory-efficient dynamic program."""
    if left == right:
        return 0
    if len(left) == len(right):
        return hamming_distance(left, right)
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, left_base in enumerate(left, start=1):
        current = [i]
        for j, right_base in enumerate(right, start=1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            substitute = previous[j - 1] + (left_base != right_base)
            current.append(min(insert, delete, substitute))
        previous = current
    return previous[-1]


def kmer_jaccard_similarity(left: str, right: str, k: int = 3) -> float:
    """Jaccard similarity between observed k-mer sets."""
    left_set = kmer_set(left, k)
    right_set = kmer_set(right, k)
    if not left_set and not right_set:
        return 1.0
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def simple_splice_strength_score(sequence: str) -> float:
    """Exploratory splice-like motif score, not a validated splice model."""
    if not sequence:
        return 0.0
    motifs = ["GU", "AG", "CAG", "GUAAGU", "GUGAGU"]
    hits = sum(sequence.count(motif) for motif in motifs)
    return min(1.0, hits / max(1.0, len(sequence) / 10.0))


def u1_complementarity_score(sequence: str, u1_seed: str = "ACUUACCU") -> float:
    """Simple U1 seed complementarity score normalized from 0 to 1."""
    query = sequence.upper().replace("T", "U")[: len(u1_seed)]
    seed = u1_seed.upper().replace("T", "U")
    if not query or not seed:
        return 0.0
    total = 0.0
    for base, seed_base in zip(query, seed):
        if (base, seed_base) in WC_PAIRS:
            total += 1.0
        elif (base, seed_base) in WOBBLE_PAIRS:
            total += 0.5
    return round(total / len(seed), 4)


def sequence_feature_row(sequence: str, u1_seed: str = "ACUUACCU") -> dict[str, Any]:
    """Calculate sequence features required by the final atlas schema."""
    seq = sequence.upper().replace("T", "U")
    if not validate_rna(seq):
        raise ValueError(f"Invalid RNA sequence encountered: {seq[:30]}...")
    length = len(seq)
    counts = Counter(seq)
    return {
        "sequence_length": length,
        "A_freq": counts.get("A", 0) / length,
        "U_freq": counts.get("U", 0) / length,
        "G_freq": counts.get("G", 0) / length,
        "C_freq": counts.get("C", 0) / length,
        "GC_content": (counts.get("G", 0) + counts.get("C", 0)) / length,
        "purine_fraction": (counts.get("A", 0) + counts.get("G", 0)) / length,
        "pyrimidine_fraction": (counts.get("C", 0) + counts.get("U", 0)) / length,
        "dinucleotide_counts_json": json.dumps(kmer_counts(seq, 2), sort_keys=True),
        "kmer3_counts_json": json.dumps(kmer_counts(seq, 3), sort_keys=True),
        "simple_splice_strength_score": simple_splice_strength_score(seq),
        "u1_complementarity_score": u1_complementarity_score(seq, u1_seed),
    }


def add_sequence_features(decoys: pd.DataFrame, u1_seed: str = "ACUUACCU") -> pd.DataFrame:
    """Append sequence feature columns to a decoy table."""
    if decoys.empty:
        return decoys.copy()
    feature_rows = [sequence_feature_row(seq, u1_seed) for seq in decoys["sequence_rna"]]
    features = pd.DataFrame(feature_rows, index=decoys.index)
    without_existing = decoys.drop(columns=[col for col in features.columns if col in decoys.columns], errors="ignore")
    return pd.concat([without_existing, features], axis=1)

