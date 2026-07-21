"""Generate SMN2-derived computational controls."""

from __future__ import annotations

import random
from collections import Counter
from typing import Any

import pandas as pd

from .feature_calculator import validate_rna
from .splice_site_extractor import base_decoy_row


BASES = ["A", "U", "G", "C"]


def _target_row(target_windows: pd.DataFrame, rng: random.Random, index: int) -> pd.Series:
    return target_windows.iloc[index % len(target_windows)] if len(target_windows) else pd.Series(dtype=object)


def _metadata_value(source: pd.Series, key: str) -> str:
    value = source.get(key, "NA")
    if value is None or value == "":
        return "NA"
    return str(value)


def generate_scrambled_controls(target_windows: pd.DataFrame, *, target_count: int, random_seed: int) -> pd.DataFrame:
    """Generate composition-preserved scrambled SMN2 controls."""
    if target_count <= 0 or target_windows.empty:
        return pd.DataFrame()
    rng = random.Random(random_seed)
    rows: list[dict[str, Any]] = []
    attempts = 0
    while len(rows) < target_count and attempts < target_count * 50:
        attempts += 1
        source = _target_row(target_windows, rng, attempts)
        original = str(source["sequence_rna"]).upper().replace("T", "U")
        sequence_list = list(original)
        rng.shuffle(sequence_list)
        scrambled = "".join(sequence_list)
        if scrambled == original or not validate_rna(scrambled):
            continue
        rows.append(
            base_decoy_row(
                decoy_category="scrambled_smn2_control",
                motif_type="composition_preserved_scrambled_smn2",
                chromosome=str(source.get("chrom", "NA")),
                start=_metadata_value(source, "start"),
                end=_metadata_value(source, "end"),
                strand=_metadata_value(source, "strand"),
                gene_id="SMN2",
                gene_name="SMN2",
                transcript_id=str(source.get("transcript_id", "NA")),
                transcript_type="NA",
                source_feature="module1_target_window",
                window_name=f"scrambled_{source.get('window_name', 'smn2_window')}",
                sequence_rna=scrambled,
                contains_splice_boundary=False,
                distance_to_nearest_exon_boundary="NA",
                notes="Scrambled SMN2 control preserving length and base composition.",
            )
        )
    return pd.DataFrame(rows)


def generate_mutated_controls(target_windows: pd.DataFrame, *, target_count: int, random_seed: int) -> pd.DataFrame:
    """Generate SMN2 target variants with one to three substitutions."""
    if target_count <= 0 or target_windows.empty:
        return pd.DataFrame()
    rng = random.Random(random_seed)
    rows: list[dict[str, Any]] = []
    attempts = 0
    while len(rows) < target_count and attempts < target_count * 50:
        attempts += 1
        source = _target_row(target_windows, rng, attempts)
        original = str(source["sequence_rna"]).upper().replace("T", "U")
        if not validate_rna(original):
            continue
        mutated = list(original)
        n_mutations = 1 + (attempts % 3)
        positions = rng.sample(range(len(mutated)), k=min(n_mutations, len(mutated)))
        for pos in positions:
            choices = [base for base in BASES if base != mutated[pos]]
            mutated[pos] = rng.choice(choices)
        sequence = "".join(mutated)
        if sequence == original:
            continue
        rows.append(
            base_decoy_row(
                decoy_category="mutated_smn2_control",
                motif_type=f"{n_mutations}_base_substitution_smn2_control",
                chromosome=str(source.get("chrom", "NA")),
                start=_metadata_value(source, "start"),
                end=_metadata_value(source, "end"),
                strand=_metadata_value(source, "strand"),
                gene_id="SMN2",
                gene_name="SMN2",
                transcript_id=str(source.get("transcript_id", "NA")),
                transcript_type="NA",
                source_feature="module1_target_window",
                window_name=f"mutated_{source.get('window_name', 'smn2_window')}",
                sequence_rna=sequence,
                contains_splice_boundary=False,
                distance_to_nearest_exon_boundary="NA",
                notes="Computational SMN2 substitution control; no biological effect is implied.",
            )
        )
    return pd.DataFrame(rows)


def preserves_base_composition(original: str, scrambled: str) -> bool:
    """Check composition preservation for tests and QC."""
    return Counter(original) == Counter(scrambled)
