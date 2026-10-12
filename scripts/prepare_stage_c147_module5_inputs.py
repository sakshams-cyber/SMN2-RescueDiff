#!/usr/bin/env python3
"""Prepare explicit, non-imputed Module 5 inputs for the Stage C cohort.

The Module 5 pose parser consumes pose_file and receptor_file columns directly;
it does not discover poses through inputs.pose_dir.  This utility therefore
creates small adapter tables for the 147 Stage C molecules with numeric
conservative selectivity evidence and assigns the fixed SMN2_CONF_01 pose to
each molecule.  Original docking outputs are only read, never changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd
import yaml


EXPECTED_ENSEMBLE_ROWS = 150
EXPECTED_COHORT_ROWS = 147
EXPECTED_DECOYS_PER_MOLECULE = 97
POSE_CONFORMER = "SMN2_CONF_01"
RANDOM_SEED = 42


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path) -> None:
    """Fail if an input is absent or empty."""
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Required input file is missing or empty: {path}")


def false_mask(series: pd.Series) -> pd.Series:
    """Recognize explicit false values without treating missing data as false."""
    return series.astype(str).str.strip().str.lower().isin({"false", "0", "no"})


def project_path(root: Path, value: str) -> Path:
    """Resolve a configured path against the project root."""
    path = Path(value)
    return path if path.is_absolute() else root / path


def configured_input_files(root: Path, config_path: Path) -> list[Path]:
    """Collect file-valued Module 5 inputs from the exact config used."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    inputs = config.get("inputs", {})
    paths: list[Path] = [config_path]
    for key, value in inputs.items():
        if key == "benchmark_controls" and isinstance(value, dict):
            for nested in value.values():
                paths.append(project_path(root, str(nested)))
        elif key != "pose_dir" and isinstance(value, str):
            paths.append(project_path(root, value))
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ensemble-selectivity",
        default="data/processed/docking/stage_c_selectivity_150_partial_after_target5/ensemble_selectivity_scores.tsv",
    )
    parser.add_argument(
        "--target-long",
        default="data/processed/docking/target_conformer_runs/target5_500_v1_complete_merged/target_conformer_scores_long.tsv",
    )
    parser.add_argument(
        "--decoy-scores",
        default="data/processed/docking/stage_b_decoy_150x100_after_target5_partial_merged/decoy_docking_scores.tsv",
    )
    parser.add_argument("--config", default="configs/splicing_rescue_scoring_stageC147.yaml")
    parser.add_argument(
        "--output-dir",
        default="data/processed/final_scoring/stage_c_147_inputs_20261011",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    ensemble_path = project_path(root, args.ensemble_selectivity)
    target_long_path = project_path(root, args.target_long)
    decoy_scores_path = project_path(root, args.decoy_scores)
    config_path = project_path(root, args.config)
    output_dir = project_path(root, args.output_dir)

    for path in [ensemble_path, target_long_path, decoy_scores_path, config_path]:
        require_file(path)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to write into existing output directory: {output_dir}")

    ensemble = pd.read_csv(ensemble_path, sep="\t")
    if len(ensemble) != EXPECTED_ENSEMBLE_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ENSEMBLE_ROWS} Stage C rows, observed {len(ensemble)} in {ensemble_path}"
        )
    required_ensemble = {
        "molecule_id",
        "mean_target_deltaG",
        "weighted_mean_selectivity_gap",
        "conservative_selectivity_gap",
        "ensemble_off_target_risk_score",
        "decoys_valid",
        "percent_decoys_stronger_than_mean_target",
    }
    missing = sorted(required_ensemble - set(ensemble.columns))
    if missing:
        raise ValueError(f"Stage C ensemble table lacks required columns: {missing}")

    conservative_gap = pd.to_numeric(ensemble["conservative_selectivity_gap"], errors="coerce")
    cohort = ensemble.loc[conservative_gap.notna()].copy()
    if len(cohort) != EXPECTED_COHORT_ROWS or cohort["molecule_id"].astype(str).nunique() != EXPECTED_COHORT_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_COHORT_ROWS} unique molecules with numeric conservative_selectivity_gap; "
            f"observed {len(cohort)} rows and {cohort['molecule_id'].astype(str).nunique()} unique IDs"
        )
    decoy_counts = pd.to_numeric(cohort["decoys_valid"], errors="coerce")
    if decoy_counts.isna().any() or not decoy_counts.eq(EXPECTED_DECOYS_PER_MOLECULE).all():
        raise ValueError(
            f"Expected exactly {EXPECTED_DECOYS_PER_MOLECULE} valid decoys for every cohort molecule; "
            f"observed counts {sorted(decoy_counts.dropna().unique().tolist())}"
        )

    target_long = pd.read_csv(target_long_path, sep="\t")
    required_target = {
        "molecule_id",
        "conformer_id",
        "receptor_file",
        "target_deltaG",
        "dock_status",
        "is_dry_run_score",
        "pose_file",
    }
    missing = sorted(required_target - set(target_long.columns))
    if missing:
        raise ValueError(f"Target conformer table lacks required columns: {missing}")
    cohort_ids = set(cohort["molecule_id"].astype(str))
    pose_rows = target_long.loc[
        target_long["molecule_id"].astype(str).isin(cohort_ids)
        & target_long["conformer_id"].astype(str).eq(POSE_CONFORMER)
    ].copy()
    if len(pose_rows) != EXPECTED_COHORT_ROWS or pose_rows["molecule_id"].astype(str).nunique() != EXPECTED_COHORT_ROWS:
        raise ValueError(
            f"Expected one {POSE_CONFORMER} row for each cohort molecule; observed {len(pose_rows)} rows"
        )
    if not pose_rows["dock_status"].astype(str).str.lower().eq("completed").all():
        bad = pose_rows.loc[
            ~pose_rows["dock_status"].astype(str).str.lower().eq("completed"), "molecule_id"
        ].astype(str).tolist()
        raise ValueError(f"Incomplete {POSE_CONFORMER} target docking rows: {bad}")
    if not false_mask(pose_rows["is_dry_run_score"]).all():
        bad = pose_rows.loc[~false_mask(pose_rows["is_dry_run_score"]), "molecule_id"].astype(str).tolist()
        raise ValueError(f"Dry-run or missing target-score flags found for: {bad}")
    if pd.to_numeric(pose_rows["target_deltaG"], errors="coerce").isna().any():
        raise ValueError(f"Missing {POSE_CONFORMER} target_deltaG values; no values will be imputed")

    for column in ["pose_file", "receptor_file"]:
        for raw_path in pose_rows[column].astype(str):
            require_file(Path(raw_path))

    decoy_scores = pd.read_csv(decoy_scores_path, sep="\t")
    required_decoy = {"molecule_id", "dock_status", "decoy_deltaG", "is_dry_run_score", "run_mode"}
    missing = sorted(required_decoy - set(decoy_scores.columns))
    if missing:
        raise ValueError(f"Decoy score table lacks required columns: {missing}")
    relevant_decoys = decoy_scores.loc[decoy_scores["molecule_id"].astype(str).isin(cohort_ids)].copy()
    observed_counts = relevant_decoys.groupby(relevant_decoys["molecule_id"].astype(str)).size()
    if len(relevant_decoys) != EXPECTED_COHORT_ROWS * EXPECTED_DECOYS_PER_MOLECULE:
        raise ValueError(
            f"Expected {EXPECTED_COHORT_ROWS * EXPECTED_DECOYS_PER_MOLECULE} decoy rows; "
            f"observed {len(relevant_decoys)}"
        )
    if len(observed_counts) != EXPECTED_COHORT_ROWS or not observed_counts.eq(EXPECTED_DECOYS_PER_MOLECULE).all():
        raise ValueError("At least one Stage C molecule lacks the exact 97-row decoy panel")
    if not relevant_decoys["dock_status"].astype(str).str.lower().eq("completed").all():
        raise ValueError("At least one Stage C decoy job is not completed")
    if not false_mask(relevant_decoys["is_dry_run_score"]).all():
        raise ValueError("At least one Stage C decoy row is dry-run or lacks an explicit false dry-run flag")
    if pd.to_numeric(relevant_decoys["decoy_deltaG"], errors="coerce").isna().any():
        raise ValueError("At least one Stage C decoy score is missing; no values will be imputed")
    run_modes = relevant_decoys["run_mode"].dropna().astype(str).unique().tolist()
    if run_modes != ["target_plus_decoy_docking"]:
        raise ValueError(f"Unexpected Stage C decoy run_mode values: {run_modes}")

    output_dir.mkdir(parents=True)

    pose_columns = [
        "molecule_id",
        "conformer_id",
        "receptor_file",
        "target_deltaG",
        "dock_status",
        "is_dry_run_score",
        "pose_file",
    ]
    pose_adapter = pose_rows[pose_columns].rename(columns={"target_deltaG": "pose_target_deltaG"})
    target_adapter = cohort[["molecule_id", "mean_target_deltaG"]].merge(
        pose_adapter, on="molecule_id", how="inner", validate="one_to_one"
    )
    target_adapter["target_deltaG"] = pd.to_numeric(target_adapter["mean_target_deltaG"], errors="raise")
    target_adapter["target_score_metric"] = "mean_target_deltaG_across_5_conformers"
    target_adapter["pose_selection_rule"] = f"fixed_{POSE_CONFORMER}_for_all_molecules"
    target_adapter.to_csv(output_dir / "target_scores_stage_c147.tsv", sep="\t", index=False)

    selectivity_adapter = cohort.copy()
    selectivity_adapter["target_deltaG"] = pd.to_numeric(
        selectivity_adapter["mean_target_deltaG"], errors="raise"
    )
    selectivity_adapter["weighted_selectivity_gap"] = pd.to_numeric(
        selectivity_adapter["weighted_mean_selectivity_gap"], errors="raise"
    )
    selectivity_adapter["off_target_risk_score"] = pd.to_numeric(
        selectivity_adapter["ensemble_off_target_risk_score"], errors="raise"
    )
    selectivity_adapter["num_decoys_with_scores"] = decoy_counts.astype(int)
    selectivity_adapter["num_decoys_with_valid_scores"] = decoy_counts.astype(int)
    selectivity_adapter["percent_decoys_stronger_than_target"] = pd.to_numeric(
        selectivity_adapter["percent_decoys_stronger_than_mean_target"], errors="raise"
    )
    selectivity_adapter["is_dry_run_score"] = False
    selectivity_adapter["run_mode"] = run_modes[0]
    selectivity_adapter.to_csv(output_dir / "selectivity_scores_stage_c147.tsv", sep="\t", index=False)

    cohort[["molecule_id"]].to_csv(output_dir / "candidates_stage_c147.tsv", sep="\t", index=False)
    target_adapter[
        [
            "molecule_id",
            "conformer_id",
            "pose_file",
            "receptor_file",
            "pose_target_deltaG",
            "target_deltaG",
            "target_score_metric",
            "pose_selection_rule",
        ]
    ].to_csv(output_dir / "selected_pose_manifest.tsv", sep="\t", index=False)

    files_to_hash = {
        ensemble_path.resolve(),
        target_long_path.resolve(),
        decoy_scores_path.resolve(),
        *[Path(path).resolve() for path in target_adapter["pose_file"].astype(str)],
        *[Path(path).resolve() for path in target_adapter["receptor_file"].astype(str)],
    }
    for path in configured_input_files(root, config_path):
        require_file(path)
        files_to_hash.add(path.resolve())
    hash_rows = [{"path": str(path), "sha256": sha256(path)} for path in sorted(files_to_hash)]
    pd.DataFrame(hash_rows).to_csv(output_dir / "input_file_sha256.tsv", sep="\t", index=False)

    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    summary = {
        "random_seed": RANDOM_SEED,
        "git_commit": git_commit,
        "ensemble_rows": int(len(ensemble)),
        "cohort_rows": int(len(cohort)),
        "excluded_missing_conservative_selectivity_gap": ensemble.loc[
            conservative_gap.isna(), "molecule_id"
        ].astype(str).tolist(),
        "pose_conformer": POSE_CONFORMER,
        "pose_selection_rule": "A single fixed conformer was used for every molecule; no per-molecule score-based pose selection was performed.",
        "target_score_metric": "mean_target_deltaG_across_5_conformers",
        "decoys_per_molecule": EXPECTED_DECOYS_PER_MOLECULE,
        "decoy_rows_validated": int(len(relevant_decoys)),
        "missing_values_imputed": False,
        "pose_files_copied": False,
        "pose_file_mapping": "Explicit source pose_file and receptor_file paths are stored in target_scores_stage_c147.tsv.",
        "input_hash_manifest": str(output_dir / "input_file_sha256.tsv"),
    }
    (output_dir / "staging_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
