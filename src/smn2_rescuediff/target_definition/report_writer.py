"""Output metadata and Markdown report generation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .annotation_parser import SelectedTarget


LIMITATIONS = [
    "This module defines computational target windows only; it does not prove splicing rescue or drug activity.",
    "Simple sequence scores are exploratory and should be replaced or complemented by validated splicing models.",
    "SMN1 and SMN2 are highly similar, so genome build and annotation consistency must be checked carefully.",
]


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    text_df = df.fillna("").astype(str)
    headers = list(text_df.columns)
    rows = text_df.values.tolist()
    header_line = "| " + " | ".join(headers) + " |"
    divider = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_line, divider, *body])


def write_metadata(
    config: dict[str, Any],
    target: SelectedTarget,
    target_windows: list[str],
    structures: pd.DataFrame,
    warnings: list[str],
    output_path: str | Path,
) -> None:
    metadata = {
        "project": config.get("project_name", "SMN2-RescueDiff"),
        "module": "target_definition",
        "gene_symbol": target.gene_symbol,
        "selected_transcript_id": target.transcript_id,
        "selected_exon": target.exon_number,
        "genome_fasta": config.get("inputs", {}).get("genome_fasta"),
        "annotation_gtf": config.get("inputs", {}).get("annotation_gtf"),
        "target_windows_created": target_windows,
        "structures_requested": structures["pdb_id"].tolist() if not structures.empty else [],
        "date_created": datetime.now(timezone.utc).isoformat(),
        "warnings": warnings,
        "limitations": LIMITATIONS,
    }
    Path(output_path).write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def write_report(
    config: dict[str, Any],
    target: SelectedTarget,
    features: pd.DataFrame,
    structures: pd.DataFrame,
    transcript_summary: pd.DataFrame,
    warnings: list[str],
    output_path: str | Path,
) -> None:
    feature_table = _markdown_table(features[
        ["window_name", "start", "end", "sequence_length", "GC_content", "u1_complementarity_score", "contains_splice_boundary"]
    ])
    structures_table = _markdown_table(structures) if not structures.empty else "No structures were requested."
    transcript_table = _markdown_table(transcript_summary)
    warning_text = "\n".join(f"- {warning}" for warning in warnings) if warnings else "- No warnings were recorded."
    limitation_text = "\n".join(f"- {item}" for item in LIMITATIONS)

    report = f"""# {config.get("project_name", "SMN2-RescueDiff")} Target Definition Report

## Purpose

This report documents Part 1 of the pipeline: reproducible definition of SMN2 exon 7 splicing-related RNA target windows for downstream computational analyses.

## Biological Rationale

Spinal muscular atrophy is associated with insufficient functional SMN protein. SMN2 is a paralogous backup gene, but exon 7 is frequently skipped during splicing. This module focuses on exon 7 and nearby splice-site context, including the 5 prime splice site, 3 prime splice site, and an approximate U1 snRNA recognition region.

## Selected Target

- Gene symbol: {target.gene_symbol}
- Gene ID: {target.gene_id}
- Transcript ID: {target.transcript_id}
- Exon: {target.exon_number}
- Coordinates: {target.chromosome}:{target.start}-{target.end} ({target.strand})
- Annotation confidence: {target.annotation_confidence}
- Notes: {target.notes}

## Candidate Transcript Summary

{transcript_table}

## Target Windows Created

{feature_table}

## U1 Complementarity Summary

The U1 complementarity score is a simple normalized Watson-Crick and optional G-U wobble score against the configured U1 seed. It is intended as an exploratory feature, not as a validated splice-site model.

## Structure Files

{structures_table}

## Warnings And Ambiguities

{warning_text}

## Limitations

{limitation_text}

## Next Steps

Later modules can use these target windows for RNA decoy generation, docking, motif analysis, molecule generation, and cautious splicing-rescue scoring.

This module defines the SMN2 exon 7 splicing-related RNA target for downstream molecule generation, docking, decoy counter-screening, and splicing-rescue scoring. The extracted target windows do not prove therapeutic activity; they provide a reproducible target definition for later computational analysis.
"""
    Path(output_path).write_text(report, encoding="utf-8")
