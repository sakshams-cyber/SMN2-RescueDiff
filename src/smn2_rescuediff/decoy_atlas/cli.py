"""Command-line interface for HuRDA-180K decoy atlas generation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from pyfaidx import Fasta

from .atlas_writer import category_counts, feature_summary, write_atlas_outputs
from .bias_matcher import (
    remove_duplicate_and_exact_smn2_matches,
    select_random_bias_matched_windows,
    select_smn2_like_hard_decoys,
)
from .config import ensure_output_dirs, load_config, resolve_paths, validate_input_paths
from .feature_calculator import add_sequence_features
from .gtf_parser import read_gtf_exons
from .random_window_extractor import extract_random_transcriptome_windows
from .regulatory_motif_extractor import extract_exonic_regulatory_windows, extract_intronic_regulatory_windows
from .report_writer import write_report
from .similarity_scorer import add_biological_weights, add_similarity_scores
from .smn2_control_generator import generate_mutated_controls, generate_scrambled_controls
from .splice_site_extractor import extract_splice_site_decoys
from .split_strategy import assign_splits
from .target_loader import TargetReference, load_smn2_inputs


def _message(log: list[str], text: str) -> None:
    print(text)
    log.append(text)


def _score(decoys: pd.DataFrame, reference: TargetReference, config: dict[str, Any]) -> pd.DataFrame:
    if decoys.empty:
        return decoys
    kmer_size = int(config.get("feature_matching", {}).get("kmer_size", 3))
    scored = add_sequence_features(decoys)
    scored = add_similarity_scores(scored, reference, kmer_size)
    return add_biological_weights(scored, config)


def _concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    non_empty = [frame for frame in frames if frame is not None and not frame.empty]
    if not non_empty:
        return pd.DataFrame()
    return pd.concat(non_empty, ignore_index=True)


def _target(config: dict[str, Any], key: str) -> int:
    return int(config.get("decoy_targets", {}).get(key, 0))


def run(config_path: str | Path) -> int:
    """Run the complete HuRDA decoy atlas pipeline."""
    config = load_config(config_path)
    paths = resolve_paths(config, config_path)
    validate_input_paths(paths)
    ensure_output_dirs(paths)

    warnings: list[str] = []
    generation_log: list[str] = []
    seed = int(config.get("atlas", {}).get("random_seed", 42))
    target_total = int(config.get("atlas", {}).get("target_total_decoys", 180000))

    _message(generation_log, "Loading Module 1 SMN2 target-definition outputs.")
    target_fasta, target_profile, target_metadata, reference = load_smn2_inputs(
        paths.smn2_target_fasta,
        paths.smn2_target_profile,
        paths.smn2_target_metadata,
    )
    smn2_sequences = set(target_fasta["sequence_rna"].astype(str).str.upper().str.replace("T", "U", regex=False))
    _message(generation_log, f"Reference target window: {reference.window_name} ({reference.sequence_length} nt).")

    _message(generation_log, "Parsing annotation exon structure.")
    exons = read_gtf_exons(paths.annotation_gtf, protein_coding_only=True)
    _message(generation_log, f"Parsed {len(exons):,} protein-coding or compatible exon rows.")

    genome = Fasta(str(paths.genome_fasta), as_raw=True, sequence_always_upper=True)
    try:
        _message(generation_log, "Extracting 5 prime splice-site decoys.")
        five_prime = extract_splice_site_decoys(
            exons,
            genome,
            config,
            site="five_prime",
            target_count=_target(config, "five_prime_splice_site_decoys"),
            random_seed=seed + 1,
        )
        _message(generation_log, f"5 prime splice-site decoys generated: {len(five_prime):,}.")

        _message(generation_log, "Extracting 3 prime splice-site decoys.")
        three_prime = extract_splice_site_decoys(
            exons,
            genome,
            config,
            site="three_prime",
            target_count=_target(config, "three_prime_splice_site_decoys"),
            random_seed=seed + 2,
        )
        _message(generation_log, f"3 prime splice-site decoys generated: {len(three_prime):,}.")

        if five_prime.empty and three_prime.empty:
            raise ValueError("No splice-site decoys could be extracted. Check the genome FASTA, GTF exon annotations, and chromosome names.")

        _message(generation_log, "Scoring splice-site candidate pool for SMN2-like hard decoys.")
        hard_target = _target(config, "smn2_like_hard_decoys")
        hard_pool_count = max(hard_target * 4, hard_target)
        hard_pool = _concat(
            [
                extract_splice_site_decoys(
                    exons,
                    genome,
                    config,
                    site="five_prime",
                    target_count=hard_pool_count // 2,
                    random_seed=seed + 3,
                    category="smn2_like_candidate_pool",
                ),
                extract_splice_site_decoys(
                    exons,
                    genome,
                    config,
                    site="three_prime",
                    target_count=hard_pool_count - hard_pool_count // 2,
                    random_seed=seed + 4,
                    category="smn2_like_candidate_pool",
                ),
            ]
        )
        hard_scored = _score(hard_pool, reference, config)
        hard_decoys = select_smn2_like_hard_decoys(hard_scored, config, hard_target)
        _message(generation_log, f"SMN2-like hard decoys selected: {len(hard_decoys):,}.")

        _message(generation_log, "Extracting candidate exonic regulatory windows.")
        exonic_regulatory = extract_exonic_regulatory_windows(
            exons,
            genome,
            config,
            target_count=_target(config, "exonic_regulatory_decoys"),
            random_seed=seed + 5,
        )
        _message(generation_log, f"Candidate exonic regulatory windows generated: {len(exonic_regulatory):,}.")

        _message(generation_log, "Extracting candidate intronic regulatory windows.")
        intronic_regulatory = extract_intronic_regulatory_windows(
            exons,
            genome,
            config,
            target_count=_target(config, "intronic_regulatory_decoys"),
            random_seed=seed + 6,
        )
        _message(generation_log, f"Candidate intronic regulatory windows generated: {len(intronic_regulatory):,}.")

        _message(generation_log, "Generating scrambled SMN2 controls.")
        scrambled = generate_scrambled_controls(
            target_fasta,
            target_count=_target(config, "scrambled_smn2_controls"),
            random_seed=seed + 7,
        )
        _message(generation_log, f"Scrambled SMN2 controls generated: {len(scrambled):,}.")

        _message(generation_log, "Generating mutated SMN2 controls.")
        mutated = generate_mutated_controls(
            target_fasta,
            target_count=_target(config, "mutated_smn2_controls"),
            random_seed=seed + 8,
        )
        _message(generation_log, f"Mutated SMN2 controls generated: {len(mutated):,}.")

        _message(generation_log, "Extracting random transcriptome windows for bias matching.")
        random_target = _target(config, "random_bias_matched_windows")
        random_pool = extract_random_transcriptome_windows(
            exons,
            genome,
            config,
            target_count=random_target,
            random_seed=seed + 9,
            overgenerate_factor=4,
        )
        random_scored = _score(random_pool, reference, config)
        random_matched = select_random_bias_matched_windows(random_scored, reference, config, random_target)
        _message(generation_log, f"Random bias-matched windows selected: {len(random_matched):,}.")
    finally:
        genome.close()

    _message(generation_log, "Combining and scoring final decoy categories.")
    combined = _concat(
        [
            five_prime,
            three_prime,
            hard_decoys,
            exonic_regulatory,
            intronic_regulatory,
            scrambled,
            mutated,
            random_matched,
        ]
    )
    if combined.empty:
        raise ValueError("No decoys were generated. Check the configured inputs and decoy target counts.")
    scored = _score(combined, reference, config)

    _message(generation_log, "Removing duplicate RNA sequences and exact SMN2 target matches when configured.")
    deduplicated, removed = remove_duplicate_and_exact_smn2_matches(scored, smn2_sequences, config)
    _message(generation_log, f"Retained decoys after filtering: {len(deduplicated):,}. Removed rows: {len(removed):,}.")

    max_total = int(config.get("atlas", {}).get("max_total_decoys", 220000))
    if len(deduplicated) > max_total:
        deduplicated = deduplicated.sort_values(["biological_weight", "smn2_similarity_score"], ascending=False).head(max_total).copy()
        warnings.append(f"Atlas exceeded max_total_decoys={max_total}; retained top decoys by biological weight and SMN2 similarity.")

    if len(deduplicated) < target_total:
        warnings.append(
            f"Generated {len(deduplicated):,} retained decoys, below the configured target_total_decoys={target_total:,}. "
            "This can occur with small fixture genomes or restrictive filtering."
        )

    _message(generation_log, "Assigning train/validation/test splits and holdout groups.")
    split = assign_splits(deduplicated, config, seed)

    _message(generation_log, "Writing HuRDA atlas outputs, figures, and QC summaries.")
    summary = write_atlas_outputs(split, removed, config, paths.output_dir, paths.figure_dir, generation_log, warnings)

    output_files = [
        "data/processed/decoy_atlas/hurda_180k_full.tsv",
        "data/processed/decoy_atlas/hurda_180k_full.fasta",
        "data/processed/decoy_atlas/hurda_train.tsv",
        "data/processed/decoy_atlas/hurda_validation.tsv",
        "data/processed/decoy_atlas/hurda_test.tsv",
        "data/processed/decoy_atlas/hurda_hard_smn2_like_holdout.tsv",
        "data/processed/decoy_atlas/hurda_chromosome_holdout.tsv",
        "data/processed/decoy_atlas/hurda_gene_holdout.tsv",
        "data/processed/decoy_atlas/hurda_summary.json",
        "data/processed/decoy_atlas/hurda_feature_summary.tsv",
        "data/processed/decoy_atlas/hurda_decoy_category_counts.tsv",
        "data/processed/decoy_atlas/hurda_removed_duplicates.tsv",
        "data/processed/decoy_atlas/hurda_generation_log.txt",
    ]
    input_paths = {
        "genome_fasta": str(paths.genome_fasta),
        "annotation_gtf": str(paths.annotation_gtf),
        "smn2_target_fasta": str(paths.smn2_target_fasta),
        "smn2_target_profile": str(paths.smn2_target_profile),
        "smn2_target_metadata": str(paths.smn2_target_metadata),
    }
    write_report(
        config=config,
        summary=summary,
        category_counts=category_counts(split),
        feature_summary=feature_summary(split),
        input_paths=input_paths,
        output_files=output_files,
        report_path=paths.report_path,
    )

    print("\nHuRDA-180K decoy atlas generation complete.")
    print(f"Retained decoys: {summary['total_decoys']:,}")
    print(f"Reached 180K target: {summary['reached_target_total']}")
    print(f"Full TSV: {paths.output_dir / 'hurda_180k_full.tsv'}")
    print(f"Full FASTA: {paths.output_dir / 'hurda_180k_full.fasta'}")
    print(f"Report: {paths.report_path}")
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"- {warning}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the HuRDA-180K human RNA decoy atlas.")
    parser.add_argument("--config", required=True, help="Path to configs/decoy_atlas.yaml")
    args = parser.parse_args(argv)
    try:
        return run(args.config)
    except Exception as exc:  # noqa: BLE001 - CLI should report actionable errors.
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

