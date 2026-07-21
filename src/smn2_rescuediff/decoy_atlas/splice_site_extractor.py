"""Splice-site decoy extraction from annotated transcript exons."""

from __future__ import annotations

import random
from typing import Any

import pandas as pd
from pyfaidx import Fasta

from .feature_calculator import dna_to_rna, validate_rna
from .gtf_parser import transcript_exon_groups


def centered_bounds(center: int, size: int) -> tuple[int, int]:
    """Return 1-based inclusive bounds centered on a genomic coordinate."""
    left = (size - 1) // 2
    right = size - left - 1
    return center - left, center + right


def fetch_rna(genome: Fasta, chromosome: str, start: int, end: int, strand: str) -> str | None:
    """Fetch an oriented RNA sequence, returning None for invalid bounds/bases."""
    if chromosome not in genome:
        return None
    chrom_len = len(genome[chromosome])
    if start < 1 or end > chrom_len or start > end:
        return None
    sequence = dna_to_rna(str(genome[chromosome][start - 1 : end]), strand)
    if not validate_rna(sequence):
        return None
    return sequence


def base_decoy_row(
    *,
    decoy_category: str,
    motif_type: str,
    chromosome: str,
    start: int | str,
    end: int | str,
    strand: str,
    gene_id: str,
    gene_name: str,
    transcript_id: str,
    transcript_type: str,
    source_feature: str,
    window_name: str,
    sequence_rna: str,
    contains_splice_boundary: bool,
    distance_to_nearest_exon_boundary: int | str,
    notes: str,
) -> dict[str, Any]:
    """Create a base decoy record before feature scoring."""
    return {
        "decoy_category": decoy_category,
        "motif_type": motif_type,
        "chromosome": chromosome,
        "start": start,
        "end": end,
        "strand": strand,
        "gene_id": gene_id or "NA",
        "gene_name": gene_name or "NA",
        "transcript_id": transcript_id or "NA",
        "transcript_type": transcript_type or "NA",
        "source_feature": source_feature,
        "window_name": window_name,
        "sequence_rna": sequence_rna,
        "contains_splice_boundary": contains_splice_boundary,
        "distance_to_nearest_exon_boundary": distance_to_nearest_exon_boundary,
        "notes": notes,
    }


def _splice_boundaries(exons: pd.DataFrame, site: str) -> list[dict[str, Any]]:
    boundaries: list[dict[str, Any]] = []
    for transcript_id, group in transcript_exon_groups(exons):
        if len(group) < 2:
            continue
        for idx in range(len(group)):
            if site == "five_prime" and idx == len(group) - 1:
                continue
            if site == "three_prime" and idx == 0:
                continue
            exon = group.iloc[idx]
            strand = exon["strand"]
            if site == "five_prime":
                boundary = int(exon["end"]) if strand != "-" else int(exon["start"])
            else:
                boundary = int(exon["start"]) if strand != "-" else int(exon["end"])
            boundaries.append(
                {
                    "chromosome": exon["chromosome"],
                    "boundary": boundary,
                    "strand": strand,
                    "gene_id": exon["gene_id"],
                    "gene_name": exon["gene_name"],
                    "transcript_id": transcript_id,
                    "transcript_type": exon["transcript_type"],
                }
            )
    return boundaries


def _site_window_bounds(boundary: int, strand: str, site: str, window_name: str, window_cfg: dict[str, Any]) -> tuple[int, int]:
    if site == "five_prime":
        if window_name == "core":
            exon_nt = int(window_cfg.get("core_exon_nt", 3))
            intron_nt = int(window_cfg.get("core_intron_nt", 6))
            if strand == "-":
                return boundary - intron_nt, boundary + exon_nt - 1
            return boundary - exon_nt + 1, boundary + intron_nt
        size = int(window_cfg["local_window_nt"] if window_name == "local" else window_cfg["extended_window_nt"])
        return centered_bounds(boundary, size)
    if window_name == "core":
        intron_nt = int(window_cfg.get("intron_nt", 20))
        exon_nt = int(window_cfg.get("exon_nt", 3))
        if strand == "-":
            return boundary - exon_nt + 1, boundary + intron_nt
        return boundary - intron_nt, boundary + exon_nt - 1
    size = int(window_cfg["local_window_nt"] if window_name == "local" else window_cfg["extended_window_nt"])
    return centered_bounds(boundary, size)


def extract_splice_site_decoys(
    exons: pd.DataFrame,
    genome: Fasta,
    config: dict[str, Any],
    *,
    site: str,
    target_count: int,
    random_seed: int,
    category: str | None = None,
) -> pd.DataFrame:
    """Extract 5 prime or 3 prime splice-site decoys."""
    if target_count <= 0:
        return pd.DataFrame()
    rng = random.Random(random_seed)
    windows_cfg = config.get("windows", {})
    if site == "five_prime":
        site_cfg = windows_cfg.get("five_prime_splice_site", {})
        decoy_category = category or "five_prime_splice_site"
        motif_type = "five_prime_splice_site"
        source_feature = "exon_to_intron_boundary"
    elif site == "three_prime":
        site_cfg = windows_cfg.get("three_prime_splice_site", {})
        decoy_category = category or "three_prime_splice_site"
        motif_type = "three_prime_splice_site"
        source_feature = "intron_to_exon_boundary"
    else:
        raise ValueError("site must be 'five_prime' or 'three_prime'")

    boundaries = _splice_boundaries(exons, site)
    rng.shuffle(boundaries)
    rows: list[dict[str, Any]] = []
    window_names = ["core", "local", "extended"]
    for boundary in boundaries:
        rng.shuffle(window_names)
        for window_name in window_names:
            start, end = _site_window_bounds(
                int(boundary["boundary"]),
                str(boundary["strand"]),
                site,
                window_name,
                site_cfg,
            )
            sequence = fetch_rna(genome, boundary["chromosome"], start, end, boundary["strand"])
            if sequence is None:
                continue
            rows.append(
                base_decoy_row(
                    decoy_category=decoy_category,
                    motif_type=motif_type,
                    chromosome=boundary["chromosome"],
                    start=start,
                    end=end,
                    strand=boundary["strand"],
                    gene_id=boundary["gene_id"],
                    gene_name=boundary["gene_name"],
                    transcript_id=boundary["transcript_id"],
                    transcript_type=boundary["transcript_type"],
                    source_feature=source_feature,
                    window_name=f"{motif_type}_{window_name}",
                    sequence_rna=sequence,
                    contains_splice_boundary=True,
                    distance_to_nearest_exon_boundary=0,
                    notes="Candidate splice-site decoy extracted from annotated transcript exon boundary.",
                )
            )
            if len(rows) >= target_count:
                return pd.DataFrame(rows)
    return pd.DataFrame(rows)
