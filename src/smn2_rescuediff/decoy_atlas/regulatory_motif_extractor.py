"""Candidate exonic and intronic regulatory-window extraction."""

from __future__ import annotations

import random
from typing import Any

import pandas as pd
from pyfaidx import Fasta

from .gtf_parser import derive_introns
from .splice_site_extractor import base_decoy_row, fetch_rna


def extract_exonic_regulatory_windows(
    exons: pd.DataFrame,
    genome: Fasta,
    config: dict[str, Any],
    *,
    target_count: int,
    random_seed: int,
) -> pd.DataFrame:
    """Extract candidate exonic regulatory windows near exon ends."""
    if target_count <= 0:
        return pd.DataFrame()
    rng = random.Random(random_seed)
    size = int(config.get("windows", {}).get("regulatory_motif_window_nt", 51))
    candidates = exons[exons["end"] - exons["start"] + 1 >= size].copy()
    candidates = candidates.sample(frac=1.0, random_state=random_seed) if len(candidates) else candidates
    rows: list[dict[str, Any]] = []
    for _, exon in candidates.iterrows():
        starts = [int(exon["start"]), int(exon["end"]) - size + 1]
        rng.shuffle(starts)
        for start in starts:
            end = start + size - 1
            sequence = fetch_rna(genome, exon["chromosome"], start, end, exon["strand"])
            if sequence is None:
                continue
            distance = min(abs(start - int(exon["start"])), abs(end - int(exon["end"])))
            rows.append(
                base_decoy_row(
                    decoy_category="exonic_regulatory",
                    motif_type="candidate_exonic_regulatory_window",
                    chromosome=exon["chromosome"],
                    start=start,
                    end=end,
                    strand=exon["strand"],
                    gene_id=exon["gene_id"],
                    gene_name=exon["gene_name"],
                    transcript_id=exon["transcript_id"],
                    transcript_type=exon["transcript_type"],
                    source_feature="exon",
                    window_name="candidate_exonic_regulatory_51nt",
                    sequence_rna=sequence,
                    contains_splice_boundary=False,
                    distance_to_nearest_exon_boundary=int(distance),
                    notes="Candidate exonic regulatory window near a splice junction; not a confirmed ESE or ESS motif.",
                )
            )
            if len(rows) >= target_count:
                return pd.DataFrame(rows)
    return pd.DataFrame(rows)


def extract_intronic_regulatory_windows(
    exons: pd.DataFrame,
    genome: Fasta,
    config: dict[str, Any],
    *,
    target_count: int,
    random_seed: int,
) -> pd.DataFrame:
    """Extract candidate intronic regulatory windows near splice junctions."""
    if target_count <= 0:
        return pd.DataFrame()
    rng = random.Random(random_seed)
    size = int(config.get("windows", {}).get("regulatory_motif_window_nt", 51))
    introns = derive_introns(exons)
    if introns.empty:
        return pd.DataFrame()
    candidates = introns[introns["end"] - introns["start"] + 1 >= size].copy()
    candidates = candidates.sample(frac=1.0, random_state=random_seed) if len(candidates) else candidates
    rows: list[dict[str, Any]] = []
    for _, intron in candidates.iterrows():
        starts = [int(intron["start"]), int(intron["end"]) - size + 1]
        rng.shuffle(starts)
        for start in starts:
            end = start + size - 1
            sequence = fetch_rna(genome, intron["chromosome"], start, end, intron["strand"])
            if sequence is None:
                continue
            distance = min(abs(start - int(intron["start"])), abs(end - int(intron["end"])))
            rows.append(
                base_decoy_row(
                    decoy_category="intronic_regulatory",
                    motif_type="candidate_intronic_regulatory_window",
                    chromosome=intron["chromosome"],
                    start=start,
                    end=end,
                    strand=intron["strand"],
                    gene_id=intron["gene_id"],
                    gene_name=intron["gene_name"],
                    transcript_id=intron["transcript_id"],
                    transcript_type=intron["transcript_type"],
                    source_feature="derived_intron",
                    window_name="candidate_intronic_regulatory_51nt",
                    sequence_rna=sequence,
                    contains_splice_boundary=False,
                    distance_to_nearest_exon_boundary=int(distance),
                    notes="Candidate intronic regulatory window near a splice junction; not a confirmed ISE or ISS motif.",
                )
            )
            if len(rows) >= target_count:
                return pd.DataFrame(rows)
    return pd.DataFrame(rows)

