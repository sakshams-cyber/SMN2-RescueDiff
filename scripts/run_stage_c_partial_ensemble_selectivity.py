#!/usr/bin/env python3
"""Stage C partial HuRDA selectivity analysis after five-conformer target docking.

This script intentionally treats the planned 150 x 100 HuRDA campaign as a
theoretical design target and the observed partial decoy table as the real
evidence. It never fabricates missing decoy scores.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DECOY_SCORES = PROJECT_ROOT / "data/processed/docking/stage_b_decoy_150x100_after_target5_partial_merged/decoy_docking_scores.tsv"
DEFAULT_TARGET_SUMMARY = PROJECT_ROOT / "data/processed/docking/target_conformer_runs/target5_500_v1_complete_merged/target_conformer_robustness_scores_final.tsv"
DEFAULT_TARGET_LONG = PROJECT_ROOT / "data/processed/docking/target_conformer_runs/target5_500_v1_complete_merged/target_conformer_scores_long.tsv"
DEFAULT_TOP150 = PROJECT_ROOT / "data/processed/docking/target_conformer_runs/target5_500_v1_complete_merged/top150_for_hurda_100_decoy_docking.tsv"
DEFAULT_LOWER_CONFIDENCE = PROJECT_ROOT / "data/processed/docking/target_conformer_runs/target5_500_v1_complete_merged/rigid_tree_repair_molecules_lower_confidence.tsv"
DEFAULT_PANEL = PROJECT_ROOT / "data/processed/docking/target_conformer_runs/target5_500_v1_complete_merged/conformer_panel_used.tsv"
DEFAULT_MISSING_TARGET_JOBS = PROJECT_ROOT / "data/processed/docking/target_conformer_runs/target5_500_v1_complete_merged/missing_jobs_after_merge.tsv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/processed/docking/stage_c_selectivity_150_partial_after_target5"

EXPECTED_ZERO_COVERAGE = {"MolGen_005024", "MolGen_012374", "MolGen_018383"}


class StageCValidationError(RuntimeError):
    """Raised when observed inputs disagree with declared Stage-C assumptions."""


def read_tsv(path: Path, *, required_columns: set[str], name: str) -> pd.DataFrame:
    """Read a TSV and fail clearly on missing files/columns."""
    if not path.exists():
        raise FileNotFoundError(f"{name} is missing: {path}")
    df = pd.read_csv(path, sep="\t", low_memory=False)
    missing = required_columns - set(df.columns)
    if missing:
        raise StageCValidationError(f"{name} is missing required columns {sorted(missing)}: {path}")
    return df


def valid_real_decoy_scores(decoy_scores: pd.DataFrame) -> pd.DataFrame:
    """Return completed, numeric, non-dry-run HuRDA decoy rows."""
    work = decoy_scores.copy()
    work["decoy_deltaG_numeric"] = pd.to_numeric(work["decoy_deltaG"], errors="coerce")
    mask = work["dock_status"].astype(str).str.strip().str.lower().eq("completed") & work["decoy_deltaG_numeric"].notna()
    if "is_dry_run_score" in work:
        mask &= work["is_dry_run_score"].astype(str).str.strip().str.lower().isin({"false", "0", "no"})
    return work[mask].copy()


def valid_real_target_long(target_long: pd.DataFrame, top_ids: set[str]) -> pd.DataFrame:
    """Return completed, numeric, non-dry-run target-conformer rows for the top-150 set."""
    work = target_long[target_long["molecule_id"].astype(str).isin(top_ids)].copy()
    work["target_deltaG_numeric"] = pd.to_numeric(work["target_deltaG"], errors="coerce")
    mask = work["target_deltaG_numeric"].notna()
    if "dock_status" in work:
        mask &= work["dock_status"].astype(str).str.strip().str.lower().eq("completed")
    if "is_dry_run_score" in work:
        mask &= work["is_dry_run_score"].astype(str).str.strip().str.lower().isin({"false", "0", "no"})
    return work[mask].copy()


def weighted_mean_decoy(valid_rows: pd.DataFrame) -> float:
    """Calculate weighted mean decoy deltaG using existing row weights when available."""
    values = pd.to_numeric(valid_rows["decoy_deltaG"], errors="coerce")
    if "decoy_biological_weight" not in valid_rows:
        return float(values.mean())
    weights = pd.to_numeric(valid_rows["decoy_biological_weight"], errors="coerce").fillna(1.0)
    return float((values * weights).sum() / max(float(weights.sum()), 1e-12))


def ensemble_off_target_risk(conservative_gap: float, percent_stronger: float, best_decoy_delta_g: float) -> float:
    """Ensemble-aware off-target risk in [0, 1].

    This adapts the existing Module 4 selectivity scorer logic by replacing the
    single-target best-decoy gap with the conservative ensemble gap:
    best_decoy_deltaG - worst_target_deltaG. More-negative Vina scores are
    stronger predicted binding, so small/negative conservative gaps, a high
    fraction of decoys stronger than the mean target score, and very strong
    absolute decoy binding all increase risk.
    """
    gap_component = max(0.0, min(1.0, (1.5 - conservative_gap) / 3.0))
    stronger_component = max(0.0, min(1.0, percent_stronger / 100.0))
    strong_decoy_component = max(0.0, min(1.0, (-best_decoy_delta_g - 5.0) / 4.0))
    return round(float(max(0.0, min(1.0, 0.45 * gap_component + 0.35 * stronger_component + 0.20 * strong_decoy_component))), 6)


def validate_inputs(
    *,
    top150: pd.DataFrame,
    target_summary: pd.DataFrame,
    target_long_valid: pd.DataFrame,
    decoy_scores: pd.DataFrame,
    decoy_valid: pd.DataFrame,
    lower_confidence: pd.DataFrame,
    panel: pd.DataFrame,
    missing_target_jobs: pd.DataFrame,
    expected_molecules: int,
    expected_conformers: int,
    planned_decoys: int,
    expected_observed_rows: int,
    expected_covered_molecules: int,
    expected_common_decoys: int,
) -> dict[str, Any]:
    """Validate observed evidence against explicit Stage-C assumptions."""
    top_ids = top150["molecule_id"].astype(str)
    if top_ids.nunique() != expected_molecules or len(top150) != expected_molecules:
        raise StageCValidationError(f"Top-150 molecule table must contain {expected_molecules} unique molecules; found rows={len(top150)}, unique={top_ids.nunique()}.")
    top_id_set = set(top_ids)

    target_summary_ids = set(target_summary["molecule_id"].astype(str))
    missing_target_summary = sorted(top_id_set - target_summary_ids)
    if missing_target_summary:
        raise StageCValidationError(f"Target summary is missing top-150 molecules: {missing_target_summary[:10]}")

    if len(panel) != expected_conformers or panel["conformer_id"].astype(str).nunique() != expected_conformers:
        raise StageCValidationError(f"Conformer panel must contain {expected_conformers} unique conformers; found rows={len(panel)}, unique={panel['conformer_id'].nunique()}.")
    if len(missing_target_jobs) != 0:
        raise StageCValidationError(f"Target missing job check must contain 0 rows; found {len(missing_target_jobs)}.")

    target_counts = target_long_valid.groupby("molecule_id")["conformer_id"].nunique()
    target_count_distribution = target_counts.value_counts().sort_index().to_dict()
    invalid_target_counts = target_counts[target_counts != expected_conformers]
    if len(target_counts) != expected_molecules or not invalid_target_counts.empty:
        raise StageCValidationError(
            f"Each top-150 molecule must have {expected_conformers} numeric target conformer scores; "
            f"observed distribution={target_count_distribution}."
        )

    if len(decoy_scores) != expected_observed_rows:
        raise StageCValidationError(f"Observed partial HuRDA row count changed: expected {expected_observed_rows}, observed {len(decoy_scores)}.")
    if len(decoy_valid) != expected_observed_rows:
        raise StageCValidationError(f"Valid completed real HuRDA row count changed: expected {expected_observed_rows}, observed valid={len(decoy_valid)}.")
    duplicate_pairs = int(decoy_valid.duplicated(["molecule_id", "decoy_id"]).sum())
    if duplicate_pairs:
        raise StageCValidationError(f"Duplicate molecule-decoy score pairs found in partial HuRDA data: {duplicate_pairs}.")

    decoy_molecules = set(decoy_valid["molecule_id"].astype(str))
    if not decoy_molecules.issubset(top_id_set):
        extra = sorted(decoy_molecules - top_id_set)
        raise StageCValidationError(f"HuRDA decoy table contains molecules outside authoritative top-150 set: {extra[:10]}")
    zero_coverage = sorted(top_id_set - decoy_molecules)
    if set(zero_coverage) != EXPECTED_ZERO_COVERAGE:
        raise StageCValidationError(f"Unexpected zero-coverage molecule set: {zero_coverage}")
    if len(decoy_molecules) != expected_covered_molecules:
        raise StageCValidationError(f"Expected {expected_covered_molecules} molecules with HuRDA scores; observed {len(decoy_molecules)}.")

    coverage = decoy_valid.groupby("molecule_id")["decoy_id"].nunique()
    coverage_distribution = {str(int(key)): int(value) for key, value in coverage.value_counts().sort_index().items()}
    if coverage_distribution != {str(expected_common_decoys): expected_covered_molecules}:
        raise StageCValidationError(f"Unexpected HuRDA coverage distribution: {coverage_distribution}")
    common_decoys = int((decoy_valid.groupby("decoy_id")["molecule_id"].nunique() == expected_covered_molecules).sum())
    if common_decoys != expected_common_decoys:
        raise StageCValidationError(f"Expected {expected_common_decoys} common observed HuRDA decoys; observed {common_decoys}.")

    if len(lower_confidence) != 2:
        raise StageCValidationError(f"Expected 2 lower-confidence target-repair molecules; observed {len(lower_confidence)}.")

    planned_rows = expected_molecules * planned_decoys
    return {
        "top150_unique_molecules": int(top_ids.nunique()),
        "target_conformer_count_distribution": {str(int(key)): int(value) for key, value in target_count_distribution.items()},
        "decoy_coverage_distribution": coverage_distribution,
        "zero_coverage_molecule_ids": zero_coverage,
        "common_observed_decoys": common_decoys,
        "duplicate_molecule_decoy_pairs": duplicate_pairs,
        "decoy_rows_theoretical_full_panel": planned_rows,
        "decoy_rows_observed": int(len(decoy_valid)),
        "decoy_rows_missing": int(planned_rows - len(decoy_valid)),
    }


def build_stage_c_scores(
    *,
    top150: pd.DataFrame,
    target_summary: pd.DataFrame,
    decoy_valid: pd.DataFrame,
    lower_confidence: pd.DataFrame,
    planned_decoys: int,
    expected_conformers: int,
    common_decoys: int,
    top_n_to_advance: int = 40,
) -> pd.DataFrame:
    """Build the molecule-level Stage-C table for all 150 molecules."""
    top = top150.drop_duplicates("molecule_id").copy()
    target_cols = [
        "molecule_id",
        "num_conformers_completed",
        "mean_target_deltaG",
        "median_target_deltaG",
        "best_target_deltaG",
        "worst_target_deltaG",
        "std_target_deltaG",
        "target_deltaG_range",
        "target_conformer_rank",
    ]
    target = target_summary[target_cols].drop_duplicates("molecule_id").copy()
    base_cols = ["molecule_id", "canonical_smiles"]
    optional_top_cols = [column for column in ["molecular_weight", "qed", "final_generation_selection_score"] if column in top.columns]
    result = top[base_cols + optional_top_cols].merge(target, on="molecule_id", how="left")
    lower_ids = set(lower_confidence["molecule_id"].astype(str))
    result["target_rigid_tree_repair_lower_confidence"] = result["molecule_id"].astype(str).isin(lower_ids)

    rows: list[dict[str, Any]] = []
    for _, row in result.iterrows():
        molecule_id = str(row["molecule_id"])
        target_ok = pd.to_numeric(pd.Series([row.get("num_conformers_completed")]), errors="coerce").iloc[0] == expected_conformers
        subset = decoy_valid[decoy_valid["molecule_id"].astype(str) == molecule_id].copy()
        decoy_values = pd.to_numeric(subset["decoy_deltaG"], errors="coerce") if not subset.empty else pd.Series(dtype=float)
        decoys_valid = int(decoy_values.notna().sum())
        unique_decoys = int(subset["decoy_id"].nunique()) if not subset.empty else 0
        payload = row.to_dict()
        payload.update(
            {
                "decoys_planned": planned_decoys,
                "decoys_valid": decoys_valid,
                "unique_decoys_observed": unique_decoys,
                "decoy_coverage_fraction": round(unique_decoys / planned_decoys, 6) if planned_decoys else np.nan,
                "best_decoy_deltaG": np.nan,
                "mean_decoy_deltaG": np.nan,
                "median_decoy_deltaG": np.nan,
                "worst_decoy_deltaG": np.nan,
                "std_decoy_deltaG": np.nan,
                "weighted_mean_decoy_deltaG": np.nan,
                "mean_selectivity_gap": np.nan,
                "weighted_mean_selectivity_gap": np.nan,
                "median_selectivity_gap": np.nan,
                "conservative_selectivity_gap": np.nan,
                "best_decoy_vs_mean_target_gap": np.nan,
                "percent_decoys_stronger_than_mean_target": np.nan,
                "ensemble_off_target_risk_score": np.nan,
                "evidence_status": "INSUFFICIENT_TARGET_ENSEMBLE_EVIDENCE",
                "eligible_for_partial_poc_ranking": False,
                "partial_poc_selectivity_rank": np.nan,
                "advance_to_ligand_decoy_control_audit": False,
                "missing_scores_imputed": False,
                "notes": "No artificial decoy or target scores were imputed.",
            }
        )
        if not target_ok:
            rows.append(payload)
            continue
        if decoys_valid == 0:
            payload["evidence_status"] = "INSUFFICIENT_DECOY_EVIDENCE"
            rows.append(payload)
            continue

        best_decoy = float(decoy_values.min())
        mean_decoy = float(decoy_values.mean())
        median_decoy = float(decoy_values.median())
        worst_decoy = float(decoy_values.max())
        std_decoy = float(decoy_values.std(ddof=0))
        weighted_mean = weighted_mean_decoy(subset)
        mean_target = float(row["mean_target_deltaG"])
        median_target = float(row["median_target_deltaG"])
        worst_target = float(row["worst_target_deltaG"])
        conservative_gap = best_decoy - worst_target
        percent_stronger = float((decoy_values < mean_target).sum() / decoys_valid * 100.0)

        payload.update(
            {
                "best_decoy_deltaG": round(best_decoy, 6),
                "mean_decoy_deltaG": round(mean_decoy, 6),
                "median_decoy_deltaG": round(median_decoy, 6),
                "worst_decoy_deltaG": round(worst_decoy, 6),
                "std_decoy_deltaG": round(std_decoy, 6),
                "weighted_mean_decoy_deltaG": round(weighted_mean, 6),
                "mean_selectivity_gap": round(mean_decoy - mean_target, 6),
                "weighted_mean_selectivity_gap": round(weighted_mean - mean_target, 6),
                "median_selectivity_gap": round(median_decoy - median_target, 6),
                "conservative_selectivity_gap": round(conservative_gap, 6),
                "best_decoy_vs_mean_target_gap": round(best_decoy - mean_target, 6),
                "percent_decoys_stronger_than_mean_target": round(percent_stronger, 6),
                "ensemble_off_target_risk_score": ensemble_off_target_risk(conservative_gap, percent_stronger, best_decoy),
                "evidence_status": "PARTIAL_HURDA_97_COMPLETE" if decoys_valid == common_decoys and unique_decoys == common_decoys else "PARTIAL_IRREGULAR_DECOY_COVERAGE",
            }
        )
        payload["eligible_for_partial_poc_ranking"] = payload["evidence_status"] == "PARTIAL_HURDA_97_COMPLETE"
        rows.append(payload)

    scores = pd.DataFrame(rows)
    eligible = scores[scores["eligible_for_partial_poc_ranking"]].copy()
    eligible = eligible.sort_values(
        [
            "weighted_mean_selectivity_gap",
            "conservative_selectivity_gap",
            "ensemble_off_target_risk_score",
            "mean_target_deltaG",
            "std_target_deltaG",
            "molecule_id",
        ],
        ascending=[False, False, True, True, True, True],
        kind="mergesort",
    )
    rank_map = {molecule_id: rank for rank, molecule_id in enumerate(eligible["molecule_id"].astype(str), start=1)}
    scores["partial_poc_selectivity_rank"] = scores["molecule_id"].astype(str).map(rank_map)
    scores["advance_to_ligand_decoy_control_audit"] = pd.to_numeric(scores["partial_poc_selectivity_rank"], errors="coerce").between(1, top_n_to_advance)
    return scores.sort_values(["eligible_for_partial_poc_ranking", "partial_poc_selectivity_rank", "molecule_id"], ascending=[False, True, True], na_position="last").reset_index(drop=True)


def write_outputs(
    *,
    output_dir: Path,
    scores: pd.DataFrame,
    validation: dict[str, Any],
    overwrite: bool,
) -> list[Path]:
    """Write Stage-C outputs, refusing accidental overwrite unless requested."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        output_dir / "ensemble_selectivity_scores.tsv",
        output_dir / "top40_for_ligand_decoy_control_audit.tsv",
        output_dir / "insufficient_decoy_evidence.tsv",
        output_dir / "stage_c_validation.json",
        output_dir / "stage_c_run_summary.tsv",
        output_dir / "stage_c_methodology.txt",
    ]
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        details = "\n".join(str(path) for path in existing)
        raise FileExistsError(f"Stage-C output files already exist. Use --overwrite to regenerate them explicitly:\n{details}")

    top40 = scores[scores["advance_to_ligand_decoy_control_audit"]].copy()
    insufficient = scores[scores["evidence_status"] == "INSUFFICIENT_DECOY_EVIDENCE"].copy()
    scores.to_csv(paths[0], sep="\t", index=False)
    top40.to_csv(paths[1], sep="\t", index=False)
    insufficient.to_csv(paths[2], sep="\t", index=False)
    paths[3].write_text(json.dumps(validation, indent=2, default=str), encoding="utf-8")
    pd.DataFrame(
        [
            {"metric": "molecules_total", "value": validation["molecules_total"]},
            {"metric": "decoy_rows_observed", "value": validation["decoy_rows_observed"]},
            {"metric": "decoy_rows_missing", "value": validation["decoy_rows_missing"]},
            {"metric": "eligible_molecules", "value": validation["eligible_molecules"]},
            {"metric": "top40_count", "value": validation["top40_count"]},
        ]
    ).to_csv(paths[4], sep="\t", index=False)
    paths[5].write_text(
        "\n".join(
            [
                "Stage C partial HuRDA ensemble selectivity methodology",
                "",
                "This analysis uses the actual completed HuRDA decoy docking rows and does not impute missing scores.",
                "The primary target metric is mean_target_deltaG across five completed SMN2 target conformers.",
                "The primary selectivity metric is weighted_mean_decoy_deltaG - mean_target_deltaG.",
                "The conservative selectivity metric is best_decoy_deltaG - worst_target_deltaG.",
                "Eligible molecules are restricted to evidence_status=PARTIAL_HURDA_97_COMPLETE.",
                "Ranking is preliminary computational Proof-of-Concept prioritization, not SRSI and not biological validation.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return paths


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Execute Stage C and return validation metadata."""
    top150 = read_tsv(args.top150, required_columns={"molecule_id", "canonical_smiles"}, name="Top-150 molecule table")
    target_summary = read_tsv(
        args.target_summary,
        required_columns={
            "molecule_id",
            "num_conformers_completed",
            "mean_target_deltaG",
            "median_target_deltaG",
            "best_target_deltaG",
            "worst_target_deltaG",
            "std_target_deltaG",
            "target_deltaG_range",
            "target_conformer_rank",
        },
        name="Target conformer robustness summary",
    )
    target_long = read_tsv(args.target_long, required_columns={"molecule_id", "conformer_id", "target_deltaG", "is_dry_run_score"}, name="Target conformer long scores")
    decoy_scores = read_tsv(args.decoy_scores, required_columns={"molecule_id", "decoy_id", "dock_status", "decoy_deltaG"}, name="Partial HuRDA decoy scores")
    lower_confidence = read_tsv(args.lower_confidence, required_columns={"molecule_id"}, name="Lower-confidence target repair molecules")
    panel = read_tsv(args.conformer_panel, required_columns={"conformer_id", "receptor_file"}, name="Conformer panel used")
    missing_target_jobs = read_tsv(args.missing_target_jobs, required_columns={"molecule_id", "conformer_id"}, name="Target missing-job check")

    top_ids = set(top150["molecule_id"].astype(str))
    target_long_valid = valid_real_target_long(target_long, top_ids)
    decoy_valid = valid_real_decoy_scores(decoy_scores)
    validation_observed = validate_inputs(
        top150=top150,
        target_summary=target_summary,
        target_long_valid=target_long_valid,
        decoy_scores=decoy_scores,
        decoy_valid=decoy_valid,
        lower_confidence=lower_confidence,
        panel=panel,
        missing_target_jobs=missing_target_jobs,
        expected_molecules=args.expected_molecules,
        expected_conformers=args.expected_conformers,
        planned_decoys=args.planned_decoys,
        expected_observed_rows=args.expected_observed_rows,
        expected_covered_molecules=args.expected_covered_molecules,
        expected_common_decoys=args.expected_common_decoys,
    )
    scores = build_stage_c_scores(
        top150=top150,
        target_summary=target_summary,
        decoy_valid=decoy_valid,
        lower_confidence=lower_confidence,
        planned_decoys=args.planned_decoys,
        expected_conformers=args.expected_conformers,
        common_decoys=args.expected_common_decoys,
        top_n_to_advance=args.top40_count,
    )

    status_counts = {str(key): int(value) for key, value in scores["evidence_status"].value_counts().sort_index().items()}
    eligible_count = int((scores["evidence_status"] == "PARTIAL_HURDA_97_COMPLETE").sum())
    top40_count = int(scores["advance_to_ligand_decoy_control_audit"].sum())
    if len(scores) != args.expected_molecules:
        raise StageCValidationError(f"Stage-C output must contain all {args.expected_molecules} molecules; observed {len(scores)}.")
    if eligible_count != args.expected_covered_molecules:
        raise StageCValidationError(f"Expected {args.expected_covered_molecules} eligible molecules; observed {eligible_count}.")
    if top40_count != args.top40_count:
        raise StageCValidationError(f"Expected top-40 shortlist count {args.top40_count}; observed {top40_count}.")

    validation = {
        "status": "passed",
        "analysis_type": "partial_HuRDA_proof_of_concept",
        "molecules_total": int(len(scores)),
        "target_conformers": int(args.expected_conformers),
        "decoy_rows_theoretical_full_panel": int(args.expected_molecules * args.planned_decoys),
        "decoy_rows_observed": int(len(decoy_valid)),
        "decoy_rows_missing": int(args.expected_molecules * args.planned_decoys - len(decoy_valid)),
        "molecules_with_HuRDA_scores": int(args.expected_covered_molecules),
        "molecules_without_HuRDA_scores": int(args.expected_molecules - args.expected_covered_molecules),
        "HuRDA_decoys_theoretical": int(args.planned_decoys),
        "HuRDA_decoys_observed_common_panel": int(args.expected_common_decoys),
        "eligible_molecules": eligible_count,
        "top40_count": top40_count,
        "missing_scores_imputed": False,
        "primary_target_metric": "mean_target_deltaG_across_5_conformers",
        "primary_selectivity_metric": "weighted_mean_decoy_deltaG_minus_mean_target_deltaG",
        "conservative_selectivity_metric": "best_decoy_deltaG_minus_worst_target_deltaG",
        "evidence_status_counts": status_counts,
        "input_files": {
            "decoy_scores": str(args.decoy_scores),
            "target_summary": str(args.target_summary),
            "target_long": str(args.target_long),
            "top150": str(args.top150),
            "lower_confidence": str(args.lower_confidence),
            "conformer_panel": str(args.conformer_panel),
            "missing_target_jobs": str(args.missing_target_jobs),
        },
        **validation_observed,
    }
    output_paths = write_outputs(output_dir=args.output_dir, scores=scores, validation=validation, overwrite=args.overwrite)
    validation["output_files"] = [str(path) for path in output_paths]
    (args.output_dir / "stage_c_validation.json").write_text(json.dumps(validation, indent=2, default=str), encoding="utf-8")
    print(json.dumps(validation, indent=2, default=str))
    return validation


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run Stage C partial HuRDA ensemble selectivity analysis.")
    parser.add_argument("--decoy-scores", type=Path, default=DEFAULT_DECOY_SCORES)
    parser.add_argument("--target-summary", type=Path, default=DEFAULT_TARGET_SUMMARY)
    parser.add_argument("--target-long", type=Path, default=DEFAULT_TARGET_LONG)
    parser.add_argument("--top150", type=Path, default=DEFAULT_TOP150)
    parser.add_argument("--lower-confidence", type=Path, default=DEFAULT_LOWER_CONFIDENCE)
    parser.add_argument("--conformer-panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--missing-target-jobs", type=Path, default=DEFAULT_MISSING_TARGET_JOBS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--expected-molecules", type=int, default=150)
    parser.add_argument("--expected-conformers", type=int, default=5)
    parser.add_argument("--planned-decoys", type=int, default=100)
    parser.add_argument("--expected-observed-rows", type=int, default=14259)
    parser.add_argument("--expected-covered-molecules", type=int, default=147)
    parser.add_argument("--expected-common-decoys", type=int, default=97)
    parser.add_argument("--top40-count", type=int, default=40)
    parser.add_argument("--overwrite", action="store_true", help="Explicitly overwrite existing Stage-C output files.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
