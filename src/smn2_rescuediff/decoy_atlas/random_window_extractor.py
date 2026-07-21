"""Random transcriptome window generation for bias-matched decoys."""

from __future__ import annotations

import random
from typing import Any

import pandas as pd
from pyfaidx import Fasta

from .splice_site_extractor import base_decoy_row, fetch_rna


def extract_random_transcriptome_windows(
    exons: pd.DataFrame,
    genome: Fasta,
    config: dict[str, Any],
    *,
    target_count: int,
    random_seed: int,
    overgenerate_factor: int = 4,
) -> pd.DataFrame:
    """Extract random annotated transcript windows for later bias matching."""
    if target_count <= 0:
        return pd.DataFrame()
    rng = random.Random(random_seed)
    size = int(config.get("windows", {}).get("random_window_nt", 51))
    pool = exons[exons["end"] - exons["start"] + 1 >= size].reset_index(drop=True)
    if pool.empty:
        return pd.DataFrame()
    desired = max(target_count, target_count * overgenerate_factor)
    rows: list[dict[str, Any]] = []
    attempts = 0
    max_attempts = desired * 20
    while len(rows) < desired and attempts < max_attempts:
        attempts += 1
        exon = pool.iloc[rng.randrange(len(pool))]
        exon_start = int(exon["start"])
        exon_end = int(exon["end"])
        start = rng.randint(exon_start, exon_end - size + 1)
        end = start + size - 1
        sequence = fetch_rna(genome, exon["chromosome"], start, end, exon["strand"])
        if sequence is None:
            continue
        distance = min(abs(start - exon_start), abs(end - exon_end))
        rows.append(
            base_decoy_row(
                decoy_category="random_bias_matched",
                motif_type="random_annotated_transcript_window",
                chromosome=exon["chromosome"],
                start=start,
                end=end,
                strand=exon["strand"],
                gene_id=exon["gene_id"],
                gene_name=exon["gene_name"],
                transcript_id=exon["transcript_id"],
                transcript_type=exon["transcript_type"],
                source_feature="exon",
                window_name="random_transcriptome_51nt",
                sequence_rna=sequence,
                contains_splice_boundary=False,
                distance_to_nearest_exon_boundary=int(distance),
                notes="Random annotated transcript window retained only if it matches simple SMN2 target biases.",
            )
        )
    return pd.DataFrame(rows)

