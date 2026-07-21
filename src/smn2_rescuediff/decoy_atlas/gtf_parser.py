"""GTF parsing utilities for transcriptome decoy extraction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


class GTFParserError(ValueError):
    """Raised when the GTF annotation cannot support decoy extraction."""


def parse_attributes(raw: str) -> dict[str, str]:
    """Parse GTF/GFF attributes into a simple dictionary."""
    attrs: dict[str, str] = {}
    for part in raw.strip().strip(";").split(";"):
        item = part.strip()
        if not item:
            continue
        if "=" in item and " " not in item.split("=", 1)[0]:
            key, value = item.split("=", 1)
        else:
            pieces = item.split(None, 1)
            if len(pieces) != 2:
                continue
            key, value = pieces
        attrs[key.strip()] = value.strip().strip('"')
    return attrs


def _first(attrs: dict[str, str], names: list[str]) -> str:
    for name in names:
        value = attrs.get(name)
        if value:
            return value
    return ""


def read_gtf_exons(annotation_path: str | Path, protein_coding_only: bool = True) -> pd.DataFrame:
    """Read exon rows from a GTF/GFF file.

    The parser keeps only fields needed for transcript-level splice-site and
    regulatory-window extraction. Missing exon numbers are inferred later.
    """
    rows: list[dict[str, Any]] = []
    with Path(annotation_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9:
                continue
            chrom, source, feature, start, end, score, strand, frame, attributes = fields
            if feature.lower() != "exon":
                continue
            attrs = parse_attributes(attributes)
            transcript_type = _first(attrs, ["transcript_type", "transcript_biotype", "gene_type", "gene_biotype"])
            if protein_coding_only and transcript_type and transcript_type != "protein_coding":
                continue
            exon_number_raw = _first(attrs, ["exon_number", "exon"])
            try:
                exon_number = int(exon_number_raw) if exon_number_raw else None
            except ValueError:
                exon_number = None
            rows.append(
                {
                    "chromosome": chrom,
                    "source_feature": source,
                    "start": int(start),
                    "end": int(end),
                    "strand": strand,
                    "gene_id": _first(attrs, ["gene_id", "geneID", "gene"]),
                    "gene_name": _first(attrs, ["gene_name", "gene", "Name"]),
                    "transcript_id": _first(attrs, ["transcript_id", "transcript", "Parent"]),
                    "transcript_type": transcript_type or "NA",
                    "exon_number": exon_number,
                }
            )
    if not rows:
        raise GTFParserError(
            f"No exon rows were parsed from {annotation_path}. Check that the file is a GTF/GFF annotation and contains exon features."
        )
    exons = pd.DataFrame(rows)
    exons = exons[exons["transcript_id"].astype(bool)].copy()
    if exons.empty:
        raise GTFParserError("No exon rows with transcript_id were found in the annotation.")
    return infer_exon_order(exons)


def infer_exon_order(exons: pd.DataFrame) -> pd.DataFrame:
    """Infer missing exon numbers by transcript order and strand."""
    exons = exons.copy()
    exons["exon_number"] = pd.to_numeric(exons["exon_number"], errors="coerce")
    for transcript_id, group in exons.groupby("transcript_id", sort=False):
        if group["exon_number"].notna().all():
            continue
        strand = group["strand"].iloc[0]
        ascending = strand != "-"
        ordered = group.sort_values("start", ascending=ascending).index
        for number, idx in enumerate(ordered, start=1):
            if pd.isna(exons.at[idx, "exon_number"]):
                exons.at[idx, "exon_number"] = number
    exons["exon_number"] = exons["exon_number"].astype(int)
    return exons


def transcript_exon_groups(exons: pd.DataFrame):
    """Yield transcript exons sorted in transcript order."""
    for transcript_id, group in exons.groupby("transcript_id", sort=False):
        strand = group["strand"].iloc[0]
        ascending = strand != "-"
        yield transcript_id, group.sort_values("start", ascending=ascending).reset_index(drop=True)


def derive_introns(exons: pd.DataFrame) -> pd.DataFrame:
    """Derive intron intervals between adjacent exons for each transcript."""
    rows: list[dict[str, Any]] = []
    for transcript_id, group in transcript_exon_groups(exons):
        if len(group) < 2:
            continue
        for idx in range(len(group) - 1):
            current = group.iloc[idx]
            nxt = group.iloc[idx + 1]
            intron_start = min(int(current["end"]), int(nxt["end"])) + 1
            intron_end = max(int(current["start"]), int(nxt["start"])) - 1
            if intron_start > intron_end:
                continue
            rows.append(
                {
                    "chromosome": current["chromosome"],
                    "start": intron_start,
                    "end": intron_end,
                    "strand": current["strand"],
                    "gene_id": current["gene_id"],
                    "gene_name": current["gene_name"],
                    "transcript_id": transcript_id,
                    "transcript_type": current["transcript_type"],
                    "source_feature": "derived_intron",
                    "upstream_exon_end": int(current["end"]),
                    "downstream_exon_start": int(nxt["start"]),
                }
            )
    return pd.DataFrame(rows)
