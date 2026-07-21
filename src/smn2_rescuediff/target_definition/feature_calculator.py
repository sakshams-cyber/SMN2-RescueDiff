"""Exploratory RNA sequence feature calculations."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .annotation_parser import SelectedTarget
from .sequence_extractor import TargetWindow


WC_PAIRS = {("A", "U"), ("U", "A"), ("G", "C"), ("C", "G")}
WOBBLE_PAIRS = {("G", "U"), ("U", "G")}


def kmer_counts(sequence: str, k: int) -> dict[str, int]:
    return dict(Counter(sequence[i : i + k] for i in range(max(0, len(sequence) - k + 1))))


def motif_density(sequence: str) -> float:
    motifs = ["GAAGAA", "AGG", "UUUU", "CAGG", "AUG"]
    if not sequence:
        return 0.0
    hits = sum(sequence.count(motif) for motif in motifs)
    return hits / len(sequence)


def simple_splice_strength(sequence: str) -> float:
    if not sequence:
        return 0.0
    motifs = ["GU", "AG", "CAG", "GURAGU".replace("R", "A")]
    hits = sum(sequence.count(motif) for motif in motifs)
    return min(1.0, hits / max(1, len(sequence) / 10))


def u1_complementarity_score(sequence: str, u1_seed: str) -> float:
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


def contains_boundary(window: TargetWindow, target: SelectedTarget) -> bool:
    if target.strand == "-":
        five_boundary = target.start
        three_boundary = target.end
    else:
        five_boundary = target.end
        three_boundary = target.start
    return window.start <= five_boundary <= window.end or window.start <= three_boundary <= window.end


def calculate_features(windows: list[TargetWindow], target: SelectedTarget, u1_seed: str) -> pd.DataFrame:
    rows = []
    for window in windows:
        seq = window.sequence_rna
        length = len(seq)
        counts = Counter(seq)
        rows.append(
            {
                "target_id": window.target_id,
                "gene_symbol": target.gene_symbol,
                "gene_id": target.gene_id,
                "transcript_id": target.transcript_id,
                "chromosome": window.chromosome,
                "start": window.start,
                "end": window.end,
                "strand": window.strand,
                "window_name": window.window_name,
                "window_type": window.window_type,
                "sequence_rna": seq,
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
                "simple_motif_density": motif_density(seq),
                "simple_splice_strength_score": simple_splice_strength(seq),
                "u1_complementarity_score": u1_complementarity_score(seq, u1_seed),
                "contains_splice_boundary": contains_boundary(window, target),
                "distance_to_exon7_start": min(abs(window.start - target.start), abs(window.end - target.start)),
                "distance_to_exon7_end": min(abs(window.start - target.end), abs(window.end - target.end)),
                "notes": window.notes,
            }
        )
    return pd.DataFrame(rows)


def write_feature_plot(features: pd.DataFrame, output_path: str | Path) -> None:
    ordered = features.sort_values("start")
    labels = ordered["window_name"].tolist()
    x = range(len(ordered))
    plt.figure(figsize=(10, 4.5))
    plt.plot(list(x), ordered["GC_content"], marker="o", label="GC content")
    plt.plot(list(x), ordered["simple_motif_density"], marker="s", label="Motif density")
    plt.xticks(list(x), labels, rotation=35, ha="right", fontsize=8)
    plt.ylabel("Score")
    plt.ylim(0, max(1.0, float(ordered[["GC_content", "simple_motif_density"]].max().max()) + 0.1))
    plt.title("SMN2 target window exploratory sequence features")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
