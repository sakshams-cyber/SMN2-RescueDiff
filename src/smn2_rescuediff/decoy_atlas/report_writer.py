"""Markdown report generation for HuRDA-180K."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


LIMITATIONS = [
    "HuRDA decoys are candidate RNA motifs for future computational counter-screening, not experimentally confirmed binders.",
    "The atlas does not prove off-target binding, toxicity, clinical safety, or SMN2 exon 7 rescue.",
    "SMN2-like and bias-matched scores are simple sequence-derived rankings that should be complemented by later structural and experimental evidence.",
    "Coverage depends on the input genome build and annotation quality.",
]


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    text = df.fillna("").astype(str)
    headers = list(text.columns)
    rows = text.values.tolist()
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *["| " + " | ".join(row) + " |" for row in rows],
        ]
    )


def write_report(
    *,
    config: dict[str, Any],
    summary: dict[str, Any],
    category_counts: pd.DataFrame,
    feature_summary: pd.DataFrame,
    input_paths: dict[str, str],
    output_files: list[str],
    report_path: str | Path,
) -> None:
    """Write the requested cautious Markdown report."""
    limitations = "\n".join(f"- {item}" for item in LIMITATIONS)
    outputs = "\n".join(f"- `{item}`" for item in output_files)
    warnings = summary.get("warnings") or []
    warning_text = "\n".join(f"- {warning}" for warning in warnings) if warnings else "- No warnings were recorded."
    report = f"""# {config.get("project_name", "SMN2-RescueDiff")} - {config.get("module_name", "HuRDA-180K Human RNA Decoy Atlas")}

## Purpose

HuRDA-180K is a transcriptome-scale human RNA decoy atlas for future SMN2 splicing-modifier counter-screening. It creates biologically motivated RNA motif decoys that can later help estimate whether candidate molecules are selective for the SMN2 exon 7 target or resemble many other human RNA motifs.

## Why Decoys Are Needed

SMN2 exon 7 target binding alone is not enough for a selective design hypothesis. Candidate molecules may also bind similar RNA motifs elsewhere in the human transcriptome. This module therefore builds splice-site, regulatory-window, SMN2-like, scrambled, mutated, and random bias-matched candidate decoys for downstream off-target scoring.

## Inputs Used

- Genome FASTA: `{input_paths["genome_fasta"]}`
- Annotation GTF/GFF: `{input_paths["annotation_gtf"]}`
- Module 1 target FASTA: `{input_paths["smn2_target_fasta"]}`
- Module 1 target profile: `{input_paths["smn2_target_profile"]}`
- Module 1 target metadata: `{input_paths["smn2_target_metadata"]}`

## Decoy Categories Generated

{_markdown_table(category_counts)}

## Feature Calculations

HuRDA calculates sequence length, nucleotide frequencies, GC content, purine and pyrimidine fractions, dinucleotide counts, 3-mer counts, simple splice-like motif score, U1 complementarity score, SMN2 distances, k-mer Jaccard similarity, SMN2 similarity score, and bias-match score.

{_markdown_table(feature_summary)}

## Bias-Matching Strategy

Bias matching uses simple sequence features from the Module 1 SMN2 reference target, including length, GC content, base composition, 3-mer overlap, splice-like score, and U1 complementarity. These scores rank candidate decoys by similarity to SMN2-relevant sequence properties and do not imply binding or biological activity.

## SMN2-Like Hard-Decoy Strategy

SMN2-like hard decoys are selected from splice-site candidate pools using normalized SMN2 similarity. Exact SMN2 target-window matches are removed when configured.

## Split Strategy

Train, validation, and test labels are assigned deterministically by random seed and stratified by decoy category. Holdout groups include hard SMN2-like motifs, chromosome-level holdouts, and gene-level holdouts when available.

## QC Summary

- Total decoys retained: {summary.get("total_decoys")}
- Target decoy count: {summary.get("target_total_decoys")}
- Reached target count: {summary.get("reached_target_total")}
- Duplicates removed: {summary.get("duplicates_removed")}
- Exact SMN2 matches removed: {summary.get("exact_smn2_matches_removed")}
- Split counts: {summary.get("split_counts")}
- Holdout counts: {summary.get("holdout_counts")}

## Warnings

{warning_text}

## Output Files

{outputs}

## Limitations

{limitations}

## Next Steps

Later modules can use HuRDA for molecule-generation filtering, docking counter-screens, transcriptome-wide off-target scoring, and SRSI score development.

HuRDA-180K provides a transcriptome-scale RNA decoy atlas for future SMN2 small-molecule counter-screening. The atlas does not prove off-target binding or toxicity; instead, it provides a reproducible set of biologically motivated RNA motifs that future modules can use to estimate transcriptome-wide off-target risk.
"""
    Path(report_path).write_text(report, encoding="utf-8")

