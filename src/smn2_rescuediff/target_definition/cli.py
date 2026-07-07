"""Command-line interface for SMN2 target definition."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .annotation_parser import read_exons, select_target
from .config import ensure_output_dirs, load_config, resolve_paths, validate_input_paths
from .feature_calculator import calculate_features, write_feature_plot
from .report_writer import write_metadata, write_report
from .sequence_extractor import build_windows, write_fasta
from .structure_fetcher import fetch_structures


def run(config_path: str | Path) -> int:
    config = load_config(config_path)
    paths = resolve_paths(config, config_path)
    validate_input_paths(paths)
    ensure_output_dirs(paths)

    warnings: list[str] = []
    exons = read_exons(paths.annotation_gtf)
    target, transcript_summary = select_target(exons, config)
    windows = build_windows(paths.genome_fasta, target, config)

    fasta_path = paths.output_dir / "smn2_target_windows.fasta"
    profile_path = paths.output_dir / "smn2_target_profile.tsv"
    metadata_path = paths.output_dir / "smn2_target_metadata.json"
    structure_metadata_path = paths.output_dir / "structure_metadata.tsv"
    transcript_summary_path = paths.output_dir / "smn2_transcript_candidates.tsv"
    figure_path = paths.figure_dir / "smn2_target_feature_profile.png"

    write_fasta(windows, target, fasta_path)
    u1_seed = config.get("u1_snRNA", {}).get("canonical_5prime_u1_seed", "ACUUACCU")
    features = calculate_features(windows, target, u1_seed)
    features.to_csv(profile_path, sep="\t", index=False)
    transcript_summary.to_csv(transcript_summary_path, sep="\t", index=False)
    write_feature_plot(features, figure_path)

    structures = fetch_structures(config, paths.structure_dir)
    structures.to_csv(structure_metadata_path, sep="\t", index=False)
    failed = structures[structures["download_status"] == "failed"] if not structures.empty else structures
    if failed is not None and not failed.empty:
        warnings.append("One or more configured PDB structures could not be downloaded; sequence target definition completed.")

    write_metadata(config, target, features["window_name"].tolist(), structures, warnings, metadata_path)
    write_report(config, target, features, structures, transcript_summary, warnings, paths.report_path)

    print("SMN2 target definition complete.")
    print(f"Selected transcript: {target.transcript_id}")
    print(f"Exon {target.exon_number}: {target.chromosome}:{target.start}-{target.end} ({target.strand})")
    print(f"Target windows: {len(windows)}")
    print(f"FASTA: {fasta_path}")
    print(f"Profile TSV: {profile_path}")
    print(f"Metadata JSON: {metadata_path}")
    print(f"Report: {paths.report_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Define SMN2 exon 7 splicing-related RNA target windows.")
    parser.add_argument("--config", required=True, help="Path to configs/target_definition.yaml")
    args = parser.parse_args(argv)
    try:
        return run(args.config)
    except Exception as exc:  # noqa: BLE001 - CLI should show actionable user-facing errors.
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
