#!/usr/bin/env python3
"""Build a frozen, validated release of the results discussed in peer review.

The builder copies observed inputs and outputs into a Git-trackable directory. It
does not fill missing docking jobs, alter scores, or overwrite source results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".json", ".md", ".smi", ".tsv", ".txt", ".yaml", ".yml"}


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(path: Path) -> Path:
    """Require an existing nonempty file or directory."""
    if not path.exists():
        raise FileNotFoundError(f"Required release source is missing: {path}")
    if path.is_file() and path.stat().st_size == 0:
        raise ValueError(f"Required release source is empty: {path}")
    return path


def as_false(series: pd.Series) -> pd.Series:
    """Return a mask for values serialized as false."""
    return series.astype(str).str.strip().str.lower().isin({"false", "0", "no", "n"})


class ReleaseBuilder:
    """Copy portable release files while recording source and release hashes."""

    def __init__(self, output: Path) -> None:
        self.output = output
        self.records: list[dict[str, Any]] = []

    def copy(self, source: Path, relative_destination: str | Path, *, portable_text: bool = True) -> Path:
        source = require(source)
        destination = self.output / relative_destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        sanitized = bool(portable_text and source.suffix.lower() in TEXT_SUFFIXES)
        if sanitized:
            text = source.read_text(encoding="utf-8", errors="replace")
            text = text.replace(str(PROJECT_ROOT) + "/", "")
            destination.write_text(text, encoding="utf-8")
        else:
            shutil.copy2(source, destination)
        self.records.append(
            {
                "source_path": str(source.relative_to(PROJECT_ROOT)),
                "source_sha256": sha256(source),
                "release_path": str(destination.relative_to(self.output)),
                "release_sha256": sha256(destination),
                "bytes": destination.stat().st_size,
                "portable_path_sanitization": sanitized,
            }
        )
        return destination

    def copy_tree(self, source: Path, relative_destination: str | Path) -> None:
        source = require(source)
        for path in sorted(item for item in source.rglob("*") if item.is_file() and item.name != ".DS_Store"):
            self.copy(path, Path(relative_destination) / path.relative_to(source))

    def write_text(self, relative_destination: str | Path, text: str) -> Path:
        destination = self.output / relative_destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text, encoding="utf-8")
        return destination

    def write_json(self, relative_destination: str | Path, payload: dict[str, Any]) -> Path:
        return self.write_text(relative_destination, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def generation_audit(builder: ReleaseBuilder) -> dict[str, Any]:
    """Audit the frozen production generator and re-annotate controls post hoc."""
    from smn2_rescuediff.molecule_generation.benchmark_controls import (
        annotate_benchmark_similarity,
        load_benchmark_controls,
    )
    from smn2_rescuediff.molecule_generation.diversity_analysis import pairwise_tanimoto_sample

    run = PROJECT_ROOT / "data/processed/molecule_generation/runs/module3_final_final_rigorous_x10"
    config_path = require(run / "molecule_generation_config_used.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    original_summary = json.loads(require(run / "generated_molecule_summary.json").read_text(encoding="utf-8"))
    training_summary = json.loads(require(run / "training_dataset_summary.json").read_text(encoding="utf-8"))
    filtered = pd.read_csv(require(run / "generated_molecules_filtered.tsv"), sep="\t")
    docking = pd.read_csv(require(run / "generated_molecules_for_docking.tsv"), sep="\t")
    valid = pd.read_csv(require(run / "generated_molecules_valid.tsv"), sep="\t")
    training_log = pd.read_csv(require(run / "model_training_log.tsv"), sep="\t")
    original_controls = pd.read_csv(require(run / "benchmark_controls_held_out.tsv"), sep="\t")
    current_controls_path = require(PROJECT_ROOT / "data/raw/molecules/benchmark_controls/splicing_benchmark_controls.smi")
    current_controls = load_benchmark_controls(current_controls_path)

    filtered_reannotated, filtered_control_summary, filtered_control_warnings = annotate_benchmark_similarity(
        filtered, current_controls, config
    )
    docking_reannotated, docking_control_summary, docking_control_warnings = annotate_benchmark_similarity(
        docking, current_controls, config
    )
    diversity_values = pairwise_tanimoto_sample(filtered, config)

    requested = int(config.get("sampling", {}).get("num_raw_to_generate", 0))
    emitted = int(original_summary["raw_generated_count"])
    valid_unique = int(valid["canonical_smiles"].nunique())
    filtered_unique = int(filtered["canonical_smiles"].nunique())
    novel = int(filtered["is_novel_vs_training"].astype(str).str.lower().isin({"true", "1", "yes"}).sum())
    filtered_scaffolds = filtered["bemis_murcko_scaffold"].dropna().astype(str)
    docking_scaffolds = docking["bemis_murcko_scaffold"].dropna().astype(str)
    train_losses = pd.to_numeric(training_log["train_loss"], errors="coerce").dropna()
    validation_losses = pd.to_numeric(training_log["validation_loss"], errors="coerce").dropna()

    audit = {
        "run_name": "module3_final_final_rigorous_x10",
        "model_type": "unconditional autoregressive SELFIES Transformer decoder",
        "target_conditioned_generation": False,
        "diffusion_model": False,
        "training": {
            "raw_rows_loaded": int(training_summary["training_molecules_raw"]),
            "valid_rows": int(training_summary["training_molecules_valid"]),
            "invalid_rows": int(training_summary["training_molecules_invalid"]),
            "unique_rows_used": int(training_summary["training_molecules_cleaned"]),
            "weighted_exposure_rows": int(training_summary["training_molecules_weighted"]),
            "epochs": int(len(training_log)),
            "final_train_loss": float(train_losses.iloc[-1]),
            "final_validation_loss": float(validation_losses.iloc[-1]),
            "best_validation_loss": float(validation_losses.min()),
        },
        "generation": {
            "raw_samples_requested": requested,
            "unique_generated_strings_emitted": emitted,
            "emitted_fraction_of_requested": emitted / requested if requested else None,
            "valid_emitted_strings": int(len(valid)),
            "validity_fraction_of_emitted_strings": len(valid) / emitted if emitted else None,
            "validity_denominator_note": (
                "The sampler removed duplicate generated strings before RDKit validation. "
                "This validity rate is not a rate over all sampling attempts."
            ),
            "unique_valid_molecules": valid_unique,
            "unique_valid_fraction": valid_unique / len(valid) if len(valid) else None,
            "filtered_molecules": int(len(filtered)),
            "unique_filtered_molecules": filtered_unique,
            "rejected_molecules": int(original_summary["rejected_count"]),
            "novel_filtered_molecules": novel,
            "novel_filtered_fraction": novel / len(filtered) if len(filtered) else None,
            "selected_for_docking": int(len(docking)),
        },
        "diversity": {
            "filtered_unique_scaffold_labels": int(filtered_scaffolds.nunique()),
            "filtered_scaffold_label_fraction": float(filtered_scaffolds.nunique() / len(filtered_scaffolds)),
            "docking_unique_scaffold_labels": int(docking_scaffolds.nunique()),
            "docking_scaffold_label_fraction": float(docking_scaffolds.nunique() / len(docking_scaffolds)),
            "sampled_pair_count": int(len(diversity_values)),
            "mean_pairwise_morgan_tanimoto": float(pd.Series(diversity_values).mean()),
            "median_pairwise_morgan_tanimoto": float(pd.Series(diversity_values).median()),
        },
        "benchmark_controls": {
            "controls_in_historical_generation_run": original_controls["benchmark_control_name"].astype(str).tolist(),
            "controls_in_current_posthoc_audit": current_controls["benchmark_control_name"].astype(str).tolist(),
            "historical_control_count": int(len(original_controls)),
            "current_posthoc_control_count": int(len(current_controls)),
            "filtered_posthoc_summary": filtered_control_summary,
            "docking_posthoc_summary": docking_control_summary,
            "warnings": filtered_control_warnings + docking_control_warnings,
            "interpretation": (
                "The frozen July generation run screened similarity against branaplam only. "
                "Risdiplam was added to the benchmark-control file later and is evaluated here post hoc; "
                "this does not retroactively change the historical generation run."
            ),
        },
        "source_limitations": [
            "The configured RBIND and SMMRNA files contributed zero rows to this run.",
            "Training was therefore based on the available ChEMBL and ZINC subset files, not an RNA-ligand-enriched corpus.",
            "Novelty fingerprints were limited to the first 20,000 canonical training molecules, as recorded by the run warning.",
            "The existing n-gram log records sample generation only; it is not a controlled generator-performance baseline.",
        ],
    }

    builder.copy(config_path, "generation/molecule_generation_config_used.yaml")
    builder.copy(run / "generated_molecule_summary.json", "generation/original_generated_molecule_summary.json")
    builder.copy(run / "training_dataset_summary.json", "generation/training_dataset_summary.json")
    builder.copy(run / "quality_training/source_balance_report.tsv", "generation/source_balance_report.tsv")
    builder.copy(run / "model_training_log.tsv", "generation/model_training_log.tsv")
    builder.copy(run / "benchmark_controls_held_out.tsv", "generation/benchmark_controls_held_out_original.tsv")
    builder.copy(current_controls_path, "generation/splicing_benchmark_controls_current.smi")
    builder.copy(PROJECT_ROOT / "data/processed/molecule_generation/baseline_generation_log.tsv", "generation/ngram_sampling_log.tsv")
    builder.copy(run / "generated_molecules_for_docking.tsv", "generation/generated_molecules_for_docking_original.tsv")
    docking_reannotated.to_csv(
        builder.output / "generation/generated_molecules_for_docking_posthoc_two_controls.tsv", sep="\t", index=False
    )
    builder.write_json("generation/generation_statistics_audit.json", audit)

    source_rows = []
    for source in training_summary.get("source_balance", []):
        path = Path(str(source.get("resolved_path", "")))
        row = {
            "source_name": source.get("source_name"),
            "configured_path": source.get("path"),
            "source_type": source.get("source_type"),
            "source_weight": source.get("source_weight"),
            "loaded_rows": source.get("loaded_rows"),
            "valid_rows": source.get("valid_rows"),
            "invalid_rows": source.get("invalid_rows"),
            "unique_used_rows": source.get("unique_used_rows"),
            "weighted_used_rows": source.get("weighted_used_rows"),
            "file_exists": path.exists(),
            "file_bytes": path.stat().st_size if path.exists() else 0,
            "file_sha256": sha256(path) if path.exists() else "missing",
        }
        source_rows.append(row)
    pd.DataFrame(source_rows).to_csv(builder.output / "generation/training_source_file_manifest.tsv", sep="\t", index=False)
    return audit


def target_docking_audit(builder: ReleaseBuilder) -> dict[str, Any]:
    """Validate and copy the five-conformer target docking experiment."""
    merged = PROJECT_ROOT / "data/processed/docking/target_conformer_runs/target5_500_v1_complete_merged"
    source_run = PROJECT_ROOT / "data/processed/docking/target_conformer_runs/target5_500_v1"
    long_scores = pd.read_csv(require(merged / "target_conformer_scores_long.tsv"), sep="\t")
    if len(long_scores) != 2500:
        raise ValueError(f"Expected 2,500 target-conformer rows; observed {len(long_scores)}")
    if long_scores["molecule_id"].nunique() != 500 or long_scores["conformer_id"].nunique() != 5:
        raise ValueError("Target-conformer table is not an exact 500 molecule x 5 conformer panel.")
    if not (long_scores["dock_status"].astype(str) == "completed").all():
        raise ValueError("Target-conformer release contains non-completed jobs.")
    if not as_false(long_scores["is_dry_run_score"]).all():
        raise ValueError("Target-conformer release contains dry-run scores.")

    for filename in [
        "conformer_panel_used.tsv",
        "merge_provenance.json",
        "missing_jobs_after_merge.tsv",
        "rigid_tree_repair_molecules_lower_confidence.tsv",
        "selected_molecules.tsv",
        "target_conformer_robustness_scores_final.tsv",
        "target_conformer_scores_long.tsv",
        "top150_for_hurda_100_decoy_docking.tsv",
    ]:
        builder.copy(merged / filename, f"target_docking/{filename}")
    builder.copy(source_run / "docking_config_used.yaml", "target_docking/docking_config_used.yaml")
    builder.copy(source_run / "logs/failed_docking_jobs.tsv", "target_docking/failed_docking_jobs_original.tsv")
    builder.copy_tree(PROJECT_ROOT / "data/processed/docking/receptors/target_conformers", "target_docking/receptors")
    builder.copy_tree(source_run / "ligands", "target_docking/ligands")
    return {
        "rows": int(len(long_scores)),
        "molecules": int(long_scores["molecule_id"].nunique()),
        "conformers": int(long_scores["conformer_id"].nunique()),
        "all_completed": True,
        "all_real_vina_scores": True,
    }


def decoy_docking_audit(builder: ReleaseBuilder) -> dict[str, Any]:
    """Validate and copy the observed partial HuRDA structural decoy screen."""
    merged = PROJECT_ROOT / "data/processed/docking/stage_b_decoy_150x100_after_target5_partial_merged"
    source_53 = PROJECT_ROOT / "data/processed/docking/stage_b_decoy_150x53_after_target5"
    source_47 = PROJECT_ROOT / "data/processed/docking/stage_b_decoy_150x47_after_target5"
    scores = pd.read_csv(require(merged / "decoy_docking_scores.tsv"), sep="\t")
    missing = pd.read_csv(require(merged / "missing_jobs.tsv"), sep="\t")

    observed = {
        "rows": int(len(scores)),
        "molecules": int(scores["molecule_id"].nunique()),
        "decoys": int(scores["decoy_id"].nunique()),
        "missing_pairs_relative_to_150x100": int(len(missing)),
    }
    expected = {"rows": 14259, "molecules": 147, "decoys": 97, "missing_pairs_relative_to_150x100": 741}
    if observed != expected:
        raise ValueError(f"Decoy-screen counts changed. Expected {expected}; observed {observed}")
    if scores.duplicated(["molecule_id", "decoy_id"]).any():
        raise ValueError("Duplicate molecule-decoy score pairs are present.")
    if not (scores["dock_status"].astype(str) == "completed").all():
        raise ValueError("Merged decoy score table includes non-completed rows.")
    if not as_false(scores["is_dry_run_score"]).all():
        raise ValueError("Merged decoy score table includes dry-run scores.")
    if pd.to_numeric(scores["decoy_deltaG"], errors="coerce").isna().any():
        raise ValueError("Merged decoy score table includes nonnumeric scores.")

    for filename in [
        "decoy_docking_scores.tsv",
        "molecule_decoy_coverage.tsv",
        "missing_jobs.tsv",
        "merge_provenance.json",
    ]:
        builder.copy(merged / filename, f"decoy_docking/{filename}")
    for label, source in [("panel_53", source_53), ("panel_47", source_47)]:
        run_name = "decoy_150x53_after_target5" if label == "panel_53" else "decoy_150x47_after_target5"
        run = source / "runs" / run_name
        for filename in [
            "docking_config_used.yaml",
            "selected_decoy_docking_panel.tsv",
            "selected_molecules_for_docking.tsv",
            "ligand_preparation_summary.tsv",
            "receptor_preparation_summary.tsv",
            "run_metrics.json",
        ]:
            builder.copy(run / filename, f"decoy_docking/source_runs/{label}/{filename}")
        builder.copy(source / "logs/failed_docking_jobs.tsv", f"decoy_docking/source_runs/{label}/failed_docking_jobs.tsv")
        builder.copy(source / "logs/docking_run_log.txt", f"decoy_docking/source_runs/{label}/docking_run_log.txt")
    builder.copy_tree(PROJECT_ROOT / "data/processed/docking/receptors/decoys", "decoy_docking/receptors")
    builder.copy_tree(source_53 / "runs/decoy_150x53_after_target5/ligands", "decoy_docking/ligands")
    return {
        **observed,
        "intended_design": "150 molecules x 100 decoys (15,000 pairs)",
        "observed_design": "147 molecules x 97 decoys (14,259 pairs)",
        "complete_150x100": False,
        "all_observed_rows_completed": True,
        "all_observed_scores_real_vina": True,
        "missing_scores_imputed": False,
    }


def final_scoring_copy(builder: ReleaseBuilder) -> dict[str, Any]:
    """Copy the audited Stage C inputs and Module 5 tables without reinterpreting them."""
    stage_c = PROJECT_ROOT / "data/processed/docking/stage_c_selectivity_150_partial_after_target5"
    inputs = PROJECT_ROOT / "data/processed/final_scoring/stage_c_147_inputs_20261011"
    scoring = PROJECT_ROOT / "data/processed/final_scoring/stage_c_147"
    builder.copy_tree(stage_c, "final_scoring/stage_c_selection")
    builder.copy_tree(inputs, "final_scoring/staged_inputs")
    for relative in [
        "RUN_SUMMARY.md",
        "final_scoring_summary.json",
        "run_metrics.json",
        "run_provenance.json",
        "splicing_rescue_scoring_config_used.yaml",
        "task2_validation_metrics.json",
        "logs/module5_run_log.txt",
        "reports/final_candidate_report.md",
        "reports/splicing_rescue_scoring_report.md",
        "evidence/scoring_warnings.json",
        "evidence/top_candidate_evidence.json",
        "tables/ablation_rank_comparison.tsv",
        "tables/benchmark_control_reference_scores.tsv",
        "tables/final_srsi_ranked_candidates.tsv",
        "tables/pose_contact_features.tsv",
        "tables/smn2_splicing_landmarks.tsv",
        "tables/splicing_rescue_scores.tsv",
        "tables/top_10_final_candidates.tsv",
    ]:
        builder.copy(scoring / relative, f"final_scoring/module5/{relative}")
    ranked = pd.read_csv(require(scoring / "tables/final_srsi_ranked_candidates.tsv"), sep="\t")
    return {
        "ranked_candidates": int(len(ranked)),
        "interpretation": "Composite scores are descriptive computational rankings, not validated binding or exon-rescue measurements.",
    }


def redocking_copy(builder: ReleaseBuilder) -> dict[str, Any]:
    """Copy the completed four-structure redocking validation and all score tables."""
    root = PROJECT_ROOT / "data/processed/redocking_validation"
    metadata = json.loads(require(root / "analysis/analysis_metadata.json").read_text(encoding="utf-8"))
    if metadata.get("docking_run_metadata", {}).get("completed_jobs") != 225:
        raise ValueError("Redocking job count is not the frozen 225-job result.")
    if metadata.get("pose_rows") != 4488 or metadata.get("pose_to_reference_rmsd_rows") != 64535:
        raise ValueError("Redocking mode or RMSD counts changed from the frozen result.")
    for filename in [
        "redocking_protocol.yaml",
        "protocol_lock.json",
        "structure_metadata.tsv",
        "fresh_ligand_preparation_summary.tsv",
        "preparation_input_sha256.tsv",
        "preparation_summary.json",
        "report_metadata.json",
    ]:
        builder.copy(root / filename, f"redocking/{filename}")
    for filename in [
        "analysis_metadata.json",
        "redocking_validation_summary.tsv",
        "model1_contact_comparison.tsv",
        "deposited_ensemble_pairwise_ligand_rmsd.tsv",
        "rna_alignment_qc.tsv",
        "pose_recovery_summary.tsv",
        "pose_to_ensemble_rmsd_long.tsv",
    ]:
        builder.copy(root / "analysis" / filename, f"redocking/analysis/{filename}")
    for filename in ["docking_job_manifest.tsv", "failed_docking_jobs.tsv", "redocking_mode_scores.tsv", "run_metadata.json"]:
        builder.copy(root / "docking" / filename, f"redocking/docking/{filename}")
    builder.copy_tree(root / "structures", "redocking/structures")
    builder.copy_tree(root / "prepared", "redocking/prepared")
    builder.copy(PROJECT_ROOT / "REDOCKING_REPORT.md", "redocking/REDOCKING_REPORT.md")
    return {
        "structures": int(metadata["structures_analyzed"]),
        "jobs": int(metadata["docking_run_metadata"]["completed_jobs"]),
        "failed_jobs": int(metadata["docking_run_metadata"]["failed_jobs"]),
        "retained_modes": int(metadata["pose_rows"]),
        "symmetry_corrected_rmsd_rows": int(metadata["pose_to_reference_rmsd_rows"]),
        "protocol_sha256": metadata["protocol_sha256"],
    }


def command_version(command: list[str]) -> str:
    """Return one-line command version text without making release creation fragile."""
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"unavailable: {exc}"
    return (result.stdout or result.stderr).strip().splitlines()[0] if (result.stdout or result.stderr).strip() else "unavailable"


def write_release_readme(builder: ReleaseBuilder, validation: dict[str, Any]) -> None:
    """Write the human-facing release scope and non-overclaiming statement."""
    generation = validation["generation"]
    decoys = validation["decoy_docking"]
    redocking = validation["redocking"]
    readme = f"""# SMN2-RescueDiff TCBB Reproducibility Snapshot

This directory freezes the code-adjacent inputs, configurations, score tables, failure records, and validation outputs available for the experiment discussed in peer review. It preserves observed results and does not fabricate or impute missing jobs.

## Scope Corrections

- The molecular generator is an **unconditional autoregressive SELFIES Transformer decoder**. It is not a diffusion model and is not conditioned on SMN2 sequence or structure.
- The production run requested {generation['generation']['raw_samples_requested']:,} samples, emitted {generation['generation']['unique_generated_strings_emitted']:,} unique generated strings after sampler deduplication, retained {generation['generation']['filtered_molecules']:,} after filtering, and exported {generation['generation']['selected_for_docking']:,} candidates.
- The historical generation run used ChEMBL and ZINC subset files. Configured RBIND and SMMRNA files contributed zero rows.
- The intended HuRDA experiment was 150 molecules x 100 decoys, but the frozen observed table is **147 molecules x 97 decoys = {decoys['rows']:,} completed real Vina pairs**. There are {decoys['missing_pairs_relative_to_150x100']:,} explicitly recorded missing pairs relative to 15,000. Missing scores were not imputed.
- A newer complete 200 x 100 run is a separate experiment and is not used to fill this snapshot.
- Four-structure redocking completed {redocking['jobs']} real Vina jobs and retained {redocking['retained_modes']:,} modes. Top-ranked near-native recovery was mixed and does not validate broad RNA selectivity inference.
- SRSI, contact, target, and decoy scores are computational descriptors. They do not establish binding, exon 7 inclusion, toxicity, safety, or clinical efficacy.

## Contents

- `generation/`: original run summaries, training/source accounting, losses, docking export, and post hoc two-control benchmark annotation.
- `target_docking/`: exact five-conformer score tables, selected molecules, receptor PDBQT files, prepared ligand PDBQT files, configuration, and failure records.
- `decoy_docking/`: the observed 14,259-pair table, missing-pair table, source-run configs/logs, 100 prepared HuRDA receptor PDBQT files, and prepared ligand PDBQT files.
- `final_scoring/`: staged selectivity inputs and Module 5 SRSI/contact/ablation tables for audit, with no claim that these scores are biologically validated.
- `redocking/`: locked protocol, deposited/prepared structures, job/mode scores, RMSD tables, contact comparison, and report.
- `release_validation.json`: machine-readable count and status checks.
- `SOURCE_FILE_MANIFEST.tsv`: source and release hashes, including path-sanitization status.
- `CHECKSUMS.sha256`: hashes of every released file.

Absolute local paths in text tables were replaced with repository-relative paths. `SOURCE_FILE_MANIFEST.tsv` records both the original source hash and released-file hash. Large generated docking pose directories are not duplicated here; the released inputs, protocol, job manifest, and mode-score/RMSD tables support rerunning and auditing the reported validation.

## Interpretation

The release resolves the stale-code contradiction by pairing the current real-decoy Vina implementation with the exact structural receptor panel and observed per-pair output. It does not make the docking method more accurate. The redocking result remains mixed, so target/selectivity rankings should be presented as exploratory computational outputs and a methodological caution.
"""
    builder.write_text("README.md", readme)


def write_checksums(builder: ReleaseBuilder) -> None:
    """Write release-wide source manifest and checksums last."""
    if builder.records:
        manifest = pd.DataFrame(builder.records).sort_values("release_path")
        manifest.to_csv(builder.output / "SOURCE_FILE_MANIFEST.tsv", sep="\t", index=False)
    files = sorted(path for path in builder.output.rglob("*") if path.is_file() and path.name != "CHECKSUMS.sha256")
    lines = [f"{sha256(path)}  {path.relative_to(builder.output)}" for path in files]
    builder.write_text("CHECKSUMS.sha256", "\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reproducibility/tcbb_2026")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing release directory.")
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Release directory already exists; refusing to overwrite: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    builder = ReleaseBuilder(output)
    validation = {
        "generation": generation_audit(builder),
        "target_docking": target_docking_audit(builder),
        "decoy_docking": decoy_docking_audit(builder),
        "final_scoring": final_scoring_copy(builder),
        "redocking": redocking_copy(builder),
        "environment": {
            "python": command_version([sys.executable, "--version"]),
            "vina": command_version([str(Path(sys.executable).parent / "vina"), "--version"]),
            "open_babel": command_version([str(Path(sys.executable).parent / "obabel"), "-V"]),
        },
        "scientific_interpretation": (
            "Exploratory computational workflow and docking-methodology caution only; "
            "not evidence of binding, splicing rescue, safety, or efficacy."
        ),
    }
    builder.write_json("release_validation.json", validation)
    write_release_readme(builder, validation)
    write_checksums(builder)
    print(json.dumps({"release": str(output), "validation": validation}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
