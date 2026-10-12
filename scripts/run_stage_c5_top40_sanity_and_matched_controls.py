#!/usr/bin/env python3
"""Stage C.5 Top-40 HuRDA sanity audit and matched ligand controls.

This script reuses existing five-conformer SMN2 target docking and partial
HuRDA decoy docking evidence. It does not run docking, repair missing decoy
jobs, invent scores, or impute missing docking results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, QED, rdFingerprintGenerator


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_STAGE_C_MASTER = PROJECT_ROOT / "data/processed/docking/stage_c_selectivity_150_partial_after_target5/ensemble_selectivity_scores.tsv"
DEFAULT_TOP40 = PROJECT_ROOT / "data/processed/docking/stage_c_selectivity_150_partial_after_target5/top40_for_ligand_decoy_control_audit.tsv"
DEFAULT_DECOY_SCORES = PROJECT_ROOT / "data/processed/docking/stage_b_decoy_150x100_after_target5_partial_merged/decoy_docking_scores.tsv"
DEFAULT_TARGET_SUMMARY = PROJECT_ROOT / "data/processed/docking/target_conformer_runs/target5_500_v1_complete_merged/target_conformer_robustness_scores_final.tsv"
DEFAULT_TARGET_LONG = PROJECT_ROOT / "data/processed/docking/target_conformer_runs/target5_500_v1_complete_merged/target_conformer_scores_long.tsv"
DEFAULT_SELECTED_MOLECULES = PROJECT_ROOT / "data/processed/docking/target_conformer_runs/target5_500_v1_complete_merged/selected_molecules.tsv"
DEFAULT_TOP150 = PROJECT_ROOT / "data/processed/docking/target_conformer_runs/target5_500_v1_complete_merged/top150_for_hurda_100_decoy_docking.tsv"
DEFAULT_LOWER_CONFIDENCE = PROJECT_ROOT / "data/processed/docking/target_conformer_runs/target5_500_v1_complete_merged/rigid_tree_repair_molecules_lower_confidence.tsv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/processed/docking/stage_c5_top40_sanity_and_matched_controls"

EXPECTED_ZERO_COVERAGE = {"MolGen_005024", "MolGen_012374", "MolGen_018383"}
EXPECTED_MISSING_HURDA_RECEPTORS = {"HuRDA_000051", "HuRDA_000052", "HuRDA_000053"}
MATCHING_DESCRIPTORS = [
    "molecular_weight",
    "logp",
    "hbd",
    "hba",
    "tpsa",
    "rotatable_bonds",
    "formal_charge",
    "qed",
    "synthetic_accessibility_proxy",
]


class StageC5ValidationError(RuntimeError):
    """Raised when Stage C.5 inputs violate declared audit assumptions."""


@dataclass(frozen=True)
class StageC5Inputs:
    """Validated source tables for Stage C.5."""

    stage_c_master: pd.DataFrame
    top40: pd.DataFrame
    decoy_scores: pd.DataFrame
    decoy_valid: pd.DataFrame
    target_summary: pd.DataFrame
    target_long: pd.DataFrame
    target_long_valid: pd.DataFrame
    selected_molecules: pd.DataFrame
    top150: pd.DataFrame
    lower_confidence: pd.DataFrame
    validation: dict[str, Any]
    source_paths: dict[str, Path]


def read_tsv(path: Path, *, required_columns: set[str], name: str) -> pd.DataFrame:
    """Read a TSV and fail clearly if the file or required columns are absent."""
    if not path.exists():
        raise FileNotFoundError(f"{name} is missing: {path}")
    df = pd.read_csv(path, sep="\t", low_memory=False)
    missing = required_columns - set(df.columns)
    if missing:
        raise StageC5ValidationError(f"{name} is missing required columns {sorted(missing)}: {path}")
    return df


def sha256_file(path: Path) -> str:
    """Return a SHA256 digest for provenance."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    """Return the current git commit if available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unavailable"


def numeric_bool_false(series: pd.Series) -> pd.Series:
    """Return True for values representing non-dry-run scores."""
    return series.astype(str).str.strip().str.lower().isin({"false", "0", "no"})


def valid_real_decoy_scores(decoy_scores: pd.DataFrame) -> pd.DataFrame:
    """Return completed, numeric, real HuRDA decoy scores."""
    work = decoy_scores.copy()
    work["decoy_deltaG_numeric"] = pd.to_numeric(work["decoy_deltaG"], errors="coerce")
    mask = work["dock_status"].astype(str).str.strip().str.lower().eq("completed")
    mask &= work["decoy_deltaG_numeric"].notna()
    if "is_dry_run_score" in work:
        mask &= numeric_bool_false(work["is_dry_run_score"])
    return work[mask].copy()


def valid_real_target_long(target_long: pd.DataFrame) -> pd.DataFrame:
    """Return completed, numeric, real target-conformer docking rows."""
    work = target_long.copy()
    work["target_deltaG_numeric"] = pd.to_numeric(work["target_deltaG"], errors="coerce")
    mask = work["target_deltaG_numeric"].notna()
    if "dock_status" in work:
        mask &= work["dock_status"].astype(str).str.strip().str.lower().eq("completed")
    if "is_dry_run_score" in work:
        mask &= numeric_bool_false(work["is_dry_run_score"])
    return work[mask].copy()


def load_and_validate_inputs(args: argparse.Namespace) -> StageC5Inputs:
    """Load source tables and validate Stage C.5 assumptions before analysis."""
    paths = {
        "stage_c_master": args.stage_c_master,
        "top40": args.top40,
        "decoy_scores": args.decoy_scores,
        "target_summary": args.target_summary,
        "target_long": args.target_long,
        "selected_molecules": args.selected_molecules,
        "top150": args.top150,
        "lower_confidence": args.lower_confidence,
    }
    stage_c = read_tsv(paths["stage_c_master"], required_columns={"molecule_id", "partial_poc_selectivity_rank"}, name="Stage-C master table")
    top40 = read_tsv(paths["top40"], required_columns={"molecule_id", "partial_poc_selectivity_rank"}, name="Top-40 table")
    decoy = read_tsv(
        paths["decoy_scores"],
        required_columns={"molecule_id", "decoy_id", "dock_status", "decoy_deltaG", "is_dry_run_score"},
        name="Partial HuRDA decoy scores",
    )
    target_summary = read_tsv(
        paths["target_summary"],
        required_columns={
            "molecule_id",
            "num_conformers_completed",
            "mean_target_deltaG",
            "median_target_deltaG",
            "best_target_deltaG",
            "worst_target_deltaG",
            "std_target_deltaG",
            "target_deltaG_range",
        },
        name="Five-conformer target summary",
    )
    target_long = read_tsv(
        paths["target_long"],
        required_columns={"molecule_id", "conformer_id", "target_deltaG", "is_dry_run_score"},
        name="Five-conformer target long table",
    )
    selected = read_tsv(paths["selected_molecules"], required_columns={"molecule_id", "canonical_smiles"}, name="500-molecule metadata")
    top150 = read_tsv(paths["top150"], required_columns={"molecule_id"}, name="Top-150 table")
    lower_confidence = read_tsv(paths["lower_confidence"], required_columns={"molecule_id"}, name="Target lower-confidence table")

    top40_ids = top40["molecule_id"].astype(str)
    if len(top40) != args.expected_top40 or top40_ids.nunique() != args.expected_top40:
        raise StageC5ValidationError(
            f"Top-40 file must contain {args.expected_top40} unique candidates; "
            f"observed rows={len(top40)}, unique={top40_ids.nunique()}."
        )
    if top40_ids.duplicated().any():
        raise StageC5ValidationError("Top-40 file contains duplicated molecule_id values.")

    stage_c_ids = set(stage_c["molecule_id"].astype(str))
    missing_stage_c = sorted(set(top40_ids) - stage_c_ids)
    if missing_stage_c:
        raise StageC5ValidationError(f"Top-40 candidates missing from Stage-C master: {missing_stage_c}")
    if len(stage_c) != args.expected_stage_c_molecules or stage_c["molecule_id"].astype(str).nunique() != args.expected_stage_c_molecules:
        raise StageC5ValidationError(
            f"Stage-C master must contain {args.expected_stage_c_molecules} unique molecules; "
            f"observed rows={len(stage_c)}, unique={stage_c['molecule_id'].nunique()}."
        )

    top150_ids = set(top150["molecule_id"].astype(str))
    if len(top150) != args.expected_stage_c_molecules or len(top150_ids) != args.expected_stage_c_molecules:
        raise StageC5ValidationError(
            f"Top-150 table must contain {args.expected_stage_c_molecules} unique molecules; "
            f"observed rows={len(top150)}, unique={len(top150_ids)}."
        )

    decoy_valid = valid_real_decoy_scores(decoy)
    top40_decoy = decoy_valid[decoy_valid["molecule_id"].astype(str).isin(set(top40_ids))].copy()
    duplicate_pairs = int(top40_decoy.duplicated(["molecule_id", "decoy_id"]).sum())
    if duplicate_pairs:
        raise StageC5ValidationError(f"Top-40 HuRDA data contains duplicate molecule-decoy pairs: {duplicate_pairs}.")
    if len(top40_decoy) != args.expected_top40_decoy_rows:
        raise StageC5ValidationError(
            f"Top-40 HuRDA rows must be {args.expected_top40_decoy_rows}; observed {len(top40_decoy)}."
        )
    coverage = top40_decoy.groupby("molecule_id")["decoy_id"].nunique()
    if len(coverage) != args.expected_top40 or not (coverage == args.expected_decoys_per_candidate).all():
        raise StageC5ValidationError(
            f"Every Top-40 molecule must have {args.expected_decoys_per_candidate} unique HuRDA decoys; "
            f"observed distribution={coverage.value_counts().sort_index().to_dict()}."
        )
    observed_decoys = set(top40_decoy["decoy_id"].astype(str))
    if len(observed_decoys) != args.expected_decoys_per_candidate:
        raise StageC5ValidationError(
            f"Expected {args.expected_decoys_per_candidate} observed HuRDA decoys for Top-40; found {len(observed_decoys)}."
        )
    numeric_by_candidate = top40_decoy.groupby("molecule_id")["decoy_deltaG_numeric"].apply(lambda values: values.notna().sum())
    if not (numeric_by_candidate == args.expected_decoys_per_candidate).all():
        raise StageC5ValidationError("At least one Top-40 molecule lacks 97 numeric HuRDA docking scores.")

    target_valid = valid_real_target_long(target_long)
    top40_target_valid = target_valid[target_valid["molecule_id"].astype(str).isin(set(top40_ids))]
    target_counts = top40_target_valid.groupby("molecule_id")["conformer_id"].nunique()
    if len(target_counts) != args.expected_top40 or not (target_counts == args.expected_target_conformers).all():
        raise StageC5ValidationError(
            f"Every Top-40 molecule must have {args.expected_target_conformers} valid target conformers; "
            f"observed distribution={target_counts.value_counts().sort_index().to_dict()}."
        )

    selected_ids = selected["molecule_id"].astype(str)
    if len(selected) != args.expected_target_population or selected_ids.nunique() != args.expected_target_population:
        raise StageC5ValidationError(
            f"Selected target population must contain {args.expected_target_population} unique molecules; "
            f"observed rows={len(selected)}, unique={selected_ids.nunique()}."
        )
    target_summary_ids = set(target_summary["molecule_id"].astype(str))
    missing_selected_targets = sorted(set(selected_ids) - target_summary_ids)
    if missing_selected_targets:
        raise StageC5ValidationError(f"Target summary missing selected molecules: {missing_selected_targets[:10]}")
    selected_target_counts = target_valid[target_valid["molecule_id"].astype(str).isin(set(selected_ids))].groupby("molecule_id")["conformer_id"].nunique()
    if len(selected_target_counts) != args.expected_target_population or not (selected_target_counts == args.expected_target_conformers).all():
        raise StageC5ValidationError(
            "The 500-molecule control population cannot be reconciled with complete five-conformer target results; "
            f"observed distribution={selected_target_counts.value_counts().sort_index().to_dict()}."
        )

    stage_c_zero = set(stage_c.loc[pd.to_numeric(stage_c.get("decoys_valid"), errors="coerce").fillna(0).eq(0), "molecule_id"].astype(str))
    if stage_c_zero and stage_c_zero != EXPECTED_ZERO_COVERAGE:
        raise StageC5ValidationError(f"Unexpected zero-HuRDA molecule set in Stage-C master: {sorted(stage_c_zero)}")

    validation = {
        "status": "passed",
        "no_new_vina_docking_performed": True,
        "top40_unique_candidates": int(top40_ids.nunique()),
        "stage_c_master_rows": int(len(stage_c)),
        "stage_c_master_unique_molecules": int(stage_c["molecule_id"].astype(str).nunique()),
        "top40_hurda_rows": int(len(top40_decoy)),
        "top40_decoys_per_candidate_distribution": {str(int(k)): int(v) for k, v in coverage.value_counts().sort_index().items()},
        "observed_hurda_decoys": int(len(observed_decoys)),
        "missing_hurda_receptors_not_repaired": sorted(EXPECTED_MISSING_HURDA_RECEPTORS),
        "top40_duplicate_molecule_decoy_pairs": duplicate_pairs,
        "top40_target_conformer_distribution": {str(int(k)): int(v) for k, v in target_counts.value_counts().sort_index().items()},
        "target_population_rows": int(len(selected)),
        "target_population_unique_molecules": int(selected_ids.nunique()),
        "control_pool_max_after_top40_removal": int(args.expected_target_population - args.expected_top40),
        "stage_c_zero_coverage_molecule_ids": sorted(stage_c_zero),
        "source_files": {
            name: {
                "path": str(path),
                "rows": int(read_tsv(path, required_columns=set(), name=name).shape[0]),
                "sha256": sha256_file(path),
            }
            for name, path in paths.items()
        },
    }
    return StageC5Inputs(stage_c, top40, decoy, decoy_valid, target_summary, target_long, target_valid, selected, top150, lower_confidence, validation, paths)


def candidate_target_summary(top40: pd.DataFrame, target_summary: pd.DataFrame) -> pd.DataFrame:
    """Return Top-40 target metrics with Stage-C ranks."""
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
    cols = ["molecule_id", "partial_poc_selectivity_rank", "weighted_mean_selectivity_gap", "conservative_selectivity_gap"]
    return top40[cols].merge(target_summary[target_cols], on="molecule_id", how="left")


def build_top40_hurda_audit(inputs: StageC5Inputs) -> pd.DataFrame:
    """Build one-row-per-candidate HuRDA decoy sanity diagnostics."""
    top_ids = set(inputs.top40["molecule_id"].astype(str))
    top_decoy = inputs.decoy_valid[inputs.decoy_valid["molecule_id"].astype(str).isin(top_ids)].copy()
    target = candidate_target_summary(inputs.top40, inputs.target_summary)
    rows: list[dict[str, Any]] = []
    metadata_cols = [
        "decoy_category",
        "motif_type",
        "decoy_biological_weight",
        "decoy_smn2_similarity_score",
        "decoy_receptor_file",
    ]
    for _, candidate in target.iterrows():
        molecule_id = str(candidate["molecule_id"])
        subset = top_decoy[top_decoy["molecule_id"].astype(str) == molecule_id].copy()
        values = subset["decoy_deltaG_numeric"].astype(float)
        strongest_idx = values.idxmin()
        strongest = subset.loc[strongest_idx]
        mean_target = float(candidate["mean_target_deltaG"])
        worst_target = float(candidate["worst_target_deltaG"])
        num_stronger_mean = int((values < mean_target).sum())
        num_stronger_worst = int((values < worst_target).sum())
        if num_stronger_worst == 0:
            driver = "NO_NEGATIVE_CONSERVATIVE_GAP"
        elif num_stronger_worst == 1:
            driver = "SINGLE_DECOY_DRIVEN"
        elif num_stronger_worst <= 5:
            driver = "FEW_DECOY_DRIVEN"
        else:
            driver = "BROAD_OFF_TARGET_PATTERN"
        payload = {
            "molecule_id": molecule_id,
            "stage_c_rank": int(float(candidate["partial_poc_selectivity_rank"])),
            "num_conformers_completed": int(candidate["num_conformers_completed"]),
            "mean_target_deltaG": float(candidate["mean_target_deltaG"]),
            "median_target_deltaG": float(candidate["median_target_deltaG"]),
            "best_target_deltaG": float(candidate["best_target_deltaG"]),
            "worst_target_deltaG": float(candidate["worst_target_deltaG"]),
            "std_target_deltaG": float(candidate["std_target_deltaG"]),
            "target_deltaG_range": float(candidate["target_deltaG_range"]),
            "target_conformer_rank": int(candidate["target_conformer_rank"]),
            "stage_c_weighted_mean_selectivity_gap": float(candidate["weighted_mean_selectivity_gap"]),
            "stage_c_conservative_selectivity_gap": float(candidate["conservative_selectivity_gap"]),
            "num_valid_decoys": int(values.notna().sum()),
            "min_decoy_deltaG": float(values.min()),
            "p05_decoy_deltaG": float(values.quantile(0.05)),
            "p10_decoy_deltaG": float(values.quantile(0.10)),
            "p25_decoy_deltaG": float(values.quantile(0.25)),
            "median_decoy_deltaG": float(values.median()),
            "mean_decoy_deltaG": float(values.mean()),
            "p75_decoy_deltaG": float(values.quantile(0.75)),
            "p90_decoy_deltaG": float(values.quantile(0.90)),
            "p95_decoy_deltaG": float(values.quantile(0.95)),
            "max_decoy_deltaG": float(values.max()),
            "std_decoy_deltaG": float(values.std(ddof=0)),
            "iqr_decoy_deltaG": float(values.quantile(0.75) - values.quantile(0.25)),
            "strongest_decoy_id": str(strongest["decoy_id"]),
            "strongest_decoy_deltaG": float(strongest["decoy_deltaG_numeric"]),
            "strongest_decoy_vs_mean_target_gap": float(strongest["decoy_deltaG_numeric"] - mean_target),
            "strongest_decoy_vs_worst_target_gap": float(strongest["decoy_deltaG_numeric"] - worst_target),
            "num_decoys_stronger_than_mean_target": num_stronger_mean,
            "percent_decoys_stronger_than_mean_target": float(num_stronger_mean / len(values) * 100.0),
            "num_decoys_stronger_than_worst_target": num_stronger_worst,
            "percent_decoys_stronger_than_worst_target": float(num_stronger_worst / len(values) * 100.0),
            "conservative_gap_driver": driver,
        }
        for col in metadata_cols:
            payload[f"strongest_{col}"] = strongest[col] if col in strongest.index else ""
        rows.append(payload)
    return pd.DataFrame(rows).sort_values("stage_c_rank").reset_index(drop=True)


def summarize_decoy_recurrence(inputs: StageC5Inputs, audit: pd.DataFrame) -> pd.DataFrame:
    """Summarize recurrent strong HuRDA decoys across Top-40 candidates."""
    top_ids = set(inputs.top40["molecule_id"].astype(str))
    top_decoy = inputs.decoy_valid[inputs.decoy_valid["molecule_id"].astype(str).isin(top_ids)].copy()
    top_decoy = top_decoy.merge(audit[["molecule_id", "mean_target_deltaG"]], on="molecule_id", how="left")
    top_decoy["_rank_within_candidate"] = top_decoy.groupby("molecule_id")["decoy_deltaG_numeric"].rank(method="first", ascending=True)
    meta_cols = [c for c in ["decoy_category", "motif_type", "decoy_biological_weight", "decoy_smn2_similarity_score", "decoy_receptor_file"] if c in top_decoy.columns]
    rows: list[dict[str, Any]] = []
    for decoy_id, subset in top_decoy.groupby("decoy_id", sort=True):
        first = subset.iloc[0]
        payload = {
            "decoy_id": decoy_id,
            "num_top40_scores": int(len(subset)),
            "num_times_strongest_decoy": int((subset["_rank_within_candidate"] == 1).sum()),
            "num_times_top3_strongest": int((subset["_rank_within_candidate"] <= 3).sum()),
            "num_times_top5_strongest": int((subset["_rank_within_candidate"] <= 5).sum()),
            "num_times_stronger_than_candidate_mean_target": int((subset["decoy_deltaG_numeric"] < subset["mean_target_deltaG"]).sum()),
            "mean_decoy_deltaG_across_top40": float(subset["decoy_deltaG_numeric"].mean()),
            "median_decoy_deltaG_across_top40": float(subset["decoy_deltaG_numeric"].median()),
        }
        for col in meta_cols:
            payload[col] = first[col]
        rows.append(payload)
    return pd.DataFrame(rows).sort_values(
        ["num_times_strongest_decoy", "num_times_top3_strongest", "mean_decoy_deltaG_across_top40", "decoy_id"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)


def summarize_category_or_motif(inputs: StageC5Inputs, audit: pd.DataFrame, column: str) -> pd.DataFrame:
    """Summarize decoy category or motif composition among strong interactions."""
    if column not in inputs.decoy_valid.columns:
        return pd.DataFrame()
    top_ids = set(inputs.top40["molecule_id"].astype(str))
    top_decoy = inputs.decoy_valid[inputs.decoy_valid["molecule_id"].astype(str).isin(top_ids)].copy()
    top_decoy = top_decoy.merge(audit[["molecule_id", "mean_target_deltaG"]], on="molecule_id", how="left")
    top_decoy["_rank_within_candidate"] = top_decoy.groupby("molecule_id")["decoy_deltaG_numeric"].rank(method="first", ascending=True)
    panel = top_decoy.drop_duplicates("decoy_id")[column].value_counts()
    strongest = top_decoy[top_decoy["_rank_within_candidate"] == 1][column].value_counts()
    top3 = top_decoy[top_decoy["_rank_within_candidate"] <= 3][column].value_counts()
    top5 = top_decoy[top_decoy["_rank_within_candidate"] <= 5][column].value_counts()
    stronger = top_decoy[top_decoy["decoy_deltaG_numeric"] < top_decoy["mean_target_deltaG"]][column].value_counts()
    labels = sorted(set(panel.index) | set(strongest.index) | set(top3.index) | set(top5.index) | set(stronger.index))
    rows = []
    for label in labels:
        panel_count = int(panel.get(label, 0))
        panel_fraction = panel_count / max(float(panel.sum()), 1.0)
        strongest_count = int(strongest.get(label, 0))
        top3_count = int(top3.get(label, 0))
        top5_count = int(top5.get(label, 0))
        stronger_count = int(stronger.get(label, 0))
        rows.append(
            {
                column: label,
                "panel_decoy_count": panel_count,
                "panel_fraction": panel_fraction,
                "strongest_decoy_count": strongest_count,
                "strongest_fraction": strongest_count / max(float(strongest.sum()), 1.0),
                "strongest_enrichment_ratio_descriptive": (strongest_count / max(float(strongest.sum()), 1.0)) / panel_fraction if panel_fraction else np.nan,
                "top3_count": top3_count,
                "top3_fraction": top3_count / max(float(top3.sum()), 1.0),
                "top3_enrichment_ratio_descriptive": (top3_count / max(float(top3.sum()), 1.0)) / panel_fraction if panel_fraction else np.nan,
                "top5_count": top5_count,
                "top5_fraction": top5_count / max(float(top5.sum()), 1.0),
                "top5_enrichment_ratio_descriptive": (top5_count / max(float(top5.sum()), 1.0)) / panel_fraction if panel_fraction else np.nan,
                "stronger_than_mean_target_count": stronger_count,
                "stronger_than_mean_target_fraction": stronger_count / max(float(stronger.sum()), 1.0),
            }
        )
    return pd.DataFrame(rows).sort_values(["strongest_decoy_count", "top3_count", column], ascending=[False, False, True]).reset_index(drop=True)


def spearman_rank_correlation(x: pd.Series, y: pd.Series) -> float:
    """Compute Spearman correlation from average ranks without SciPy."""
    xy = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(xy) < 2 or xy["x"].nunique() < 2 or xy["y"].nunique() < 2:
        return float("nan")
    return float(xy["x"].rank(method="average").corr(xy["y"].rank(method="average"), method="pearson"))


def smn2_similarity_relationship(inputs: StageC5Inputs) -> dict[str, Any]:
    """Estimate descriptive Spearman relationships between decoy SMN2 similarity and docking strength."""
    if "decoy_smn2_similarity_score" not in inputs.decoy_valid.columns:
        return {"available": False, "reason": "decoy_smn2_similarity_score column is absent"}
    top_ids = set(inputs.top40["molecule_id"].astype(str))
    work = inputs.decoy_valid[inputs.decoy_valid["molecule_id"].astype(str).isin(top_ids)].copy()
    work["decoy_smn2_similarity_score_numeric"] = pd.to_numeric(work["decoy_smn2_similarity_score"], errors="coerce")
    work["decoy_docking_strength_positive"] = -work["decoy_deltaG_numeric"]
    valid = work.dropna(subset=["decoy_smn2_similarity_score_numeric", "decoy_docking_strength_positive"])
    if valid.empty:
        return {"available": False, "reason": "no numeric decoy similarity values"}
    per_candidate = []
    for molecule_id, subset in valid.groupby("molecule_id"):
        if subset["decoy_smn2_similarity_score_numeric"].nunique() > 1 and subset["decoy_docking_strength_positive"].nunique() > 1:
            corr = spearman_rank_correlation(subset["decoy_smn2_similarity_score_numeric"], subset["decoy_docking_strength_positive"])
            per_candidate.append(float(corr))
    return {
        "available": True,
        "interpretation": "Positive correlation means higher HuRDA-SMN2 sequence similarity tends to accompany stronger predicted decoy docking because strength is -deltaG.",
        "pooled_spearman_similarity_vs_strength": spearman_rank_correlation(valid["decoy_smn2_similarity_score_numeric"], valid["decoy_docking_strength_positive"]),
        "per_candidate_spearman_mean": float(np.nanmean(per_candidate)) if per_candidate else np.nan,
        "per_candidate_spearman_median": float(np.nanmedian(per_candidate)) if per_candidate else np.nan,
        "num_candidate_correlations": int(len(per_candidate)),
    }


def rdkit_mol(smiles: str, molecule_id: str) -> Chem.Mol:
    """Parse a SMILES string or fail clearly."""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise StageC5ValidationError(f"RDKit could not parse required molecule {molecule_id}: {smiles}")
    return mol


def compute_standard_descriptors(mol: Chem.Mol) -> dict[str, float]:
    """Compute standard descriptors if stored values are unavailable."""
    return {
        "molecular_weight": float(Descriptors.MolWt(mol)),
        "logp": float(Descriptors.MolLogP(mol)),
        "hbd": float(Descriptors.NumHDonors(mol)),
        "hba": float(Descriptors.NumHAcceptors(mol)),
        "tpsa": float(Descriptors.TPSA(mol)),
        "rotatable_bonds": float(Descriptors.NumRotatableBonds(mol)),
        "formal_charge": float(sum(atom.GetFormalCharge() for atom in mol.GetAtoms())),
        "qed": float(QED.qed(mol)),
    }


def compute_descriptors(population: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate or compute matching descriptors for all candidate/control molecules."""
    work = population.copy()
    mols: dict[str, Chem.Mol] = {}
    for row in work.itertuples(index=False):
        mols[str(row.molecule_id)] = rdkit_mol(str(row.canonical_smiles), str(row.molecule_id))

    descriptor_sources: dict[str, str] = {}
    computed_cache = {mid: compute_standard_descriptors(mol) for mid, mol in mols.items()}
    for descriptor in MATCHING_DESCRIPTORS:
        numeric = pd.to_numeric(work[descriptor], errors="coerce") if descriptor in work.columns else pd.Series([np.nan] * len(work))
        if int(numeric.notna().sum()) == len(work):
            work[descriptor] = numeric.astype(float)
            descriptor_sources[descriptor] = "stored_metadata"
            continue
        if descriptor in computed_cache[next(iter(computed_cache))]:
            work[descriptor] = work["molecule_id"].astype(str).map(lambda mid: computed_cache[mid][descriptor])
            descriptor_sources[descriptor] = "rdkit_computed"
            continue
        if descriptor == "synthetic_accessibility_proxy":
            raise StageC5ValidationError("synthetic_accessibility_proxy is required as the SA-like matching descriptor but is unavailable.")
        raise StageC5ValidationError(f"Descriptor {descriptor} is unavailable and cannot be computed.")
    return work, {
        "descriptor_sources": descriptor_sources,
        "descriptor_definitions": {
            "molecular_weight": "Stored metadata if complete; otherwise RDKit Descriptors.MolWt.",
            "logp": "Stored metadata if complete; otherwise RDKit Descriptors.MolLogP.",
            "hbd": "Stored metadata if complete; otherwise RDKit hydrogen bond donor count.",
            "hba": "Stored metadata if complete; otherwise RDKit hydrogen bond acceptor count.",
            "tpsa": "Stored metadata if complete; otherwise RDKit TPSA.",
            "rotatable_bonds": "Stored metadata if complete; otherwise RDKit rotatable bond count.",
            "formal_charge": "Stored metadata if complete; otherwise sum of atom formal charges.",
            "qed": "Stored metadata if complete; otherwise RDKit QED.",
            "synthetic_accessibility_proxy": "Stored Module-3 synthetic-accessibility proxy; used because sa_score is empty in the target5 metadata.",
        },
    }


def build_control_pool(inputs: StageC5Inputs, descriptors: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create the property-matching control pool from the existing 500-molecule campaign."""
    top_ids = set(inputs.top40["molecule_id"].astype(str))
    top_smiles = set(inputs.top40["canonical_smiles"].astype(str)) if "canonical_smiles" in inputs.top40.columns else set()
    target_complete = inputs.target_summary[pd.to_numeric(inputs.target_summary["num_conformers_completed"], errors="coerce").eq(5)].copy()
    target_complete_ids = set(target_complete["molecule_id"].astype(str))
    pool = descriptors[~descriptors["molecule_id"].astype(str).isin(top_ids)].copy()
    pool = pool[pool["molecule_id"].astype(str).isin(target_complete_ids)].copy()
    pool["is_exact_duplicate_of_top40_smiles"] = pool["canonical_smiles"].astype(str).isin(top_smiles)
    exact_duplicate_count = int(pool["is_exact_duplicate_of_top40_smiles"].sum())
    pool = pool[~pool["is_exact_duplicate_of_top40_smiles"]].copy()
    duplicate_smiles_count = int(pool.duplicated("canonical_smiles").sum())
    pool = pool.sort_values("molecule_id").drop_duplicates("canonical_smiles", keep="first").copy()
    return pool, {
        "initial_selected_molecules": int(len(inputs.selected_molecules)),
        "top40_removed": int(len(top_ids)),
        "controls_after_top40_removal": int(len(inputs.selected_molecules) - len(top_ids)),
        "exact_top40_smiles_duplicates_removed": exact_duplicate_count,
        "duplicate_control_smiles_removed": duplicate_smiles_count,
        "valid_control_pool_size": int(len(pool)),
    }


def zscore_population(descriptors: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """Z-score descriptor columns across candidates plus controls."""
    work = descriptors.copy()
    scaling: dict[str, dict[str, float]] = {}
    for col in columns:
        values = pd.to_numeric(work[col], errors="coerce")
        if values.isna().any():
            raise StageC5ValidationError(f"Cannot standardize descriptor {col}; missing numeric values remain.")
        mean = float(values.mean())
        std = float(values.std(ddof=0))
        if std == 0:
            std = 1.0
        work[f"z_{col}"] = (values - mean) / std
        scaling[col] = {"mean": mean, "std": std}
    return work, scaling


def fingerprint_map(population: pd.DataFrame) -> dict[str, Any]:
    """Build Morgan fingerprint map for molecules."""
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps: dict[str, Any] = {}
    for row in population.itertuples(index=False):
        mol = rdkit_mol(str(row.canonical_smiles), str(row.molecule_id))
        fps[str(row.molecule_id)] = generator.GetFingerprint(mol)
    return fps


def inspect_tanimoto_distribution(candidates: pd.DataFrame, controls: pd.DataFrame, fps: dict[str, Any]) -> dict[str, Any]:
    """Inspect candidate-control Tanimoto similarities before applying the guard."""
    all_sims: list[float] = []
    nearest: list[float] = []
    for cand in candidates.itertuples(index=False):
        sims = [float(DataStructs.TanimotoSimilarity(fps[str(cand.molecule_id)], fps[str(ctrl.molecule_id)])) for ctrl in controls.itertuples(index=False)]
        nearest.append(max(sims))
        all_sims.extend(sims)
    arr = np.array(all_sims, dtype=float)
    near = np.array(nearest, dtype=float)
    quantiles = [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
    return {
        "all_pairwise_count": int(len(arr)),
        "all_pairwise_quantiles": {str(q): float(np.quantile(arr, q)) for q in quantiles},
        "candidate_nearest_control_quantiles": {str(q): float(np.quantile(near, q)) for q in quantiles},
        "max_pairwise_tanimoto": float(arr.max()),
        "max_nearest_control_tanimoto": float(near.max()),
    }


def match_controls(
    *,
    candidates: pd.DataFrame,
    controls: pd.DataFrame,
    target_summary: pd.DataFrame,
    descriptors: list[str],
    fps: dict[str, Any],
    tanimoto_threshold: float,
    controls_per_candidate: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Select deterministic nearest-neighbor property-matched controls."""
    z_cols = [f"z_{col}" for col in descriptors]
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
    target = target_summary[target_cols].copy()
    assignments: list[dict[str, Any]] = []
    relaxation_thresholds = [tanimoto_threshold, 0.70, 0.80, 0.90, 1.0]
    relaxation_counts: dict[str, int] = {str(t): 0 for t in relaxation_thresholds}
    for cand in candidates.sort_values("partial_poc_selectivity_rank").itertuples(index=False):
        cand_id = str(cand.molecule_id)
        cand_vec = np.array([getattr(cand, z) for z in z_cols], dtype=float)
        rows: list[dict[str, Any]] = []
        chosen_threshold = tanimoto_threshold
        for threshold in relaxation_thresholds:
            rows = []
            for ctrl in controls.itertuples(index=False):
                ctrl_id = str(ctrl.molecule_id)
                if ctrl_id == cand_id or str(ctrl.canonical_smiles) == str(cand.canonical_smiles):
                    continue
                ctrl_vec = np.array([getattr(ctrl, z) for z in z_cols], dtype=float)
                tanimoto = float(DataStructs.TanimotoSimilarity(fps[cand_id], fps[ctrl_id]))
                if tanimoto > threshold:
                    continue
                abs_diffs = {f"abs_std_diff_{descriptor}": abs(float(getattr(cand, f"z_{descriptor}") - getattr(ctrl, f"z_{descriptor}"))) for descriptor in descriptors}
                rows.append(
                    {
                        "candidate_id": cand_id,
                        "control_id": ctrl_id,
                        "candidate_stage_c_rank": int(float(cand.partial_poc_selectivity_rank)),
                        "matching_distance": float(np.linalg.norm(cand_vec - ctrl_vec)),
                        "candidate_control_tanimoto": tanimoto,
                        "tanimoto_threshold_used": threshold,
                        "threshold_relaxed": bool(threshold != tanimoto_threshold),
                        **abs_diffs,
                    }
                )
            if len(rows) >= controls_per_candidate:
                chosen_threshold = threshold
                break
        if len(rows) < controls_per_candidate:
            raise StageC5ValidationError(
                f"Candidate {cand_id} has only {len(rows)} eligible matched controls after systematic threshold relaxation."
            )
        relaxation_counts[str(chosen_threshold)] += 1
        picked = sorted(rows, key=lambda row: (row["matching_distance"], row["candidate_control_tanimoto"], row["control_id"]))[:controls_per_candidate]
        for row in picked:
            row["match_rank_for_candidate"] = len([x for x in assignments if x["candidate_id"] == cand_id]) + 1
            assignments.append(row)

    result = pd.DataFrame(assignments)
    candidate_descriptor_cols = ["molecule_id", "canonical_smiles", *descriptors]
    cand_desc = candidates[candidate_descriptor_cols].rename(columns={"molecule_id": "candidate_id", "canonical_smiles": "candidate_canonical_smiles"})
    ctrl_desc = controls[candidate_descriptor_cols].rename(columns={"molecule_id": "control_id", "canonical_smiles": "control_canonical_smiles"})
    result = result.merge(cand_desc, on="candidate_id", how="left")
    result = result.merge(ctrl_desc, on="control_id", how="left", suffixes=("_candidate", "_control"))
    result = result.rename(columns={col: f"candidate_{col}" for col in descriptors})
    result = result.rename(columns={f"{col}_control": f"control_{col}" for col in descriptors})
    result = result.merge(target.add_prefix("candidate_"), left_on="candidate_id", right_on="candidate_molecule_id", how="left")
    result = result.merge(target.add_prefix("control_"), left_on="control_id", right_on="control_molecule_id", how="left")
    result = result.drop(columns=["candidate_molecule_id", "control_molecule_id"])

    reuse = result["control_id"].value_counts()
    result["control_reuse_count"] = result["control_id"].map(reuse).astype(int)
    return result.sort_values(["candidate_stage_c_rank", "match_rank_for_candidate"]).reset_index(drop=True), {
        "controls_per_candidate_requested": controls_per_candidate,
        "tanimoto_threshold_initial": tanimoto_threshold,
        "relaxation_thresholds": relaxation_thresholds,
        "candidate_threshold_use_counts": relaxation_counts,
        "number_control_assignments": int(len(result)),
        "number_unique_controls": int(result["control_id"].nunique()),
        "control_reuse_distribution": {str(int(k)): int(v) for k, v in reuse.value_counts().sort_index().items()},
    }


def evaluate_matching_quality(assignments: pd.DataFrame, descriptors: list[str]) -> dict[str, Any]:
    """Calculate global matching quality diagnostics."""
    diagnostics: dict[str, Any] = {
        "matching_distance_mean": float(assignments["matching_distance"].mean()),
        "matching_distance_median": float(assignments["matching_distance"].median()),
        "matching_distance_max": float(assignments["matching_distance"].max()),
        "candidate_control_tanimoto_mean": float(assignments["candidate_control_tanimoto"].mean()),
        "candidate_control_tanimoto_median": float(assignments["candidate_control_tanimoto"].median()),
        "candidate_control_tanimoto_max": float(assignments["candidate_control_tanimoto"].max()),
        "absolute_standardized_difference_mean_by_descriptor": {},
        "absolute_standardized_difference_median_by_descriptor": {},
    }
    for descriptor in descriptors:
        col = f"abs_std_diff_{descriptor}"
        diagnostics["absolute_standardized_difference_mean_by_descriptor"][descriptor] = float(assignments[col].mean())
        diagnostics["absolute_standardized_difference_median_by_descriptor"][descriptor] = float(assignments[col].median())
    return diagnostics


def compare_candidates_to_controls(assignments: pd.DataFrame, audit: pd.DataFrame) -> pd.DataFrame:
    """Summarize target ensemble enrichment for each candidate matched set."""
    rows: list[dict[str, Any]] = []
    audit_cols = [
        "molecule_id",
        "stage_c_rank",
        "stage_c_weighted_mean_selectivity_gap",
        "stage_c_conservative_selectivity_gap",
        "conservative_gap_driver",
        "num_decoys_stronger_than_mean_target",
        "num_decoys_stronger_than_worst_target",
    ]
    audit_map = audit[audit_cols].set_index("molecule_id").to_dict("index")
    for candidate_id, subset in assignments.groupby("candidate_id", sort=False):
        candidate_mean = float(subset["candidate_mean_target_deltaG"].iloc[0])
        candidate_std = float(subset["candidate_std_target_deltaG"].iloc[0])
        control_means = pd.to_numeric(subset["control_mean_target_deltaG"], errors="coerce")
        control_stds = pd.to_numeric(subset["control_std_target_deltaG"], errors="coerce")
        num_beats = int((candidate_mean < control_means).sum())
        metadata = audit_map[candidate_id]
        rows.append(
            {
                "candidate_id": candidate_id,
                "stage_c_rank": metadata["stage_c_rank"],
                "num_matched_controls": int(len(subset)),
                "candidate_mean_target_deltaG": candidate_mean,
                "candidate_median_target_deltaG": float(subset["candidate_median_target_deltaG"].iloc[0]),
                "candidate_best_target_deltaG": float(subset["candidate_best_target_deltaG"].iloc[0]),
                "candidate_worst_target_deltaG": float(subset["candidate_worst_target_deltaG"].iloc[0]),
                "candidate_std_target_deltaG": candidate_std,
                "candidate_target_deltaG_range": float(subset["candidate_target_deltaG_range"].iloc[0]),
                "control_mean_mean_target_deltaG": float(control_means.mean()),
                "control_median_mean_target_deltaG": float(control_means.median()),
                "control_best_mean_target_deltaG": float(control_means.min()),
                "control_worst_mean_target_deltaG": float(control_means.max()),
                "candidate_minus_control_median_target_difference": float(candidate_mean - control_means.median()),
                "target_enrichment_gap": float(control_means.median() - candidate_mean),
                "number_controls_candidate_beats": num_beats,
                "percent_controls_candidate_beats": float(num_beats / len(subset) * 100.0),
                "candidate_percentile_within_matched_set": float((num_beats + 1) / (len(subset) + 1) * 100.0),
                "control_median_std_target_deltaG": float(control_stds.median()),
                "robustness_advantage": float(control_stds.median() - candidate_std),
                "matching_distance_mean": float(subset["matching_distance"].mean()),
                "matching_distance_median": float(subset["matching_distance"].median()),
                "matching_distance_max": float(subset["matching_distance"].max()),
                "candidate_control_tanimoto_mean": float(subset["candidate_control_tanimoto"].mean()),
                "candidate_control_tanimoto_max": float(subset["candidate_control_tanimoto"].max()),
                "stage_c_weighted_mean_selectivity_gap": metadata["stage_c_weighted_mean_selectivity_gap"],
                "stage_c_conservative_selectivity_gap": metadata["stage_c_conservative_selectivity_gap"],
                "conservative_gap_driver": metadata["conservative_gap_driver"],
                "num_decoys_stronger_than_mean_target": metadata["num_decoys_stronger_than_mean_target"],
                "num_decoys_stronger_than_worst_target": metadata["num_decoys_stronger_than_worst_target"],
            }
        )
    return pd.DataFrame(rows).sort_values("stage_c_rank").reset_index(drop=True)


def bootstrap_ci(values: np.ndarray, *, seed: int, statistic: str, n_bootstrap: int) -> dict[str, float]:
    """Bootstrap a 95% confidence interval over candidate matched sets."""
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        if statistic == "median":
            stats.append(float(np.median(sample)))
        else:
            stats.append(float(np.mean(sample)))
    return {"lower": float(np.quantile(stats, 0.025)), "upper": float(np.quantile(stats, 0.975))}


def random_sign_permutation_test(values: np.ndarray, *, seed: int, n_permutations: int) -> dict[str, Any]:
    """Two-sided paired random sign-flip test for mean gap different from zero."""
    rng = np.random.default_rng(seed)
    observed = abs(float(values.mean()))
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_permutations, len(values)), replace=True)
    null_stats = np.abs((signs * values).mean(axis=1))
    p_value = float((np.sum(null_stats >= observed) + 1) / (n_permutations + 1))
    return {
        "paired_test_name": "random_sign_flip_permutation_test_on_candidate_level_mean_gap",
        "test_statistic_observed_abs_mean_gap": observed,
        "p_value": p_value,
        "n_permutations": n_permutations,
    }


def compute_group_statistics(
    comparison: pd.DataFrame,
    assignments: pd.DataFrame,
    matching_diagnostics: dict[str, Any],
    matching_metadata: dict[str, Any],
    tanimoto_distribution: dict[str, Any],
    *,
    seed: int,
    n_bootstrap: int,
    n_permutations: int,
) -> dict[str, Any]:
    """Compute group-level matched-control statistics without pseudo-replication."""
    gaps = pd.to_numeric(comparison["target_enrichment_gap"], errors="coerce").dropna().to_numpy(dtype=float)
    if len(gaps) != len(comparison):
        raise StageC5ValidationError("Missing target_enrichment_gap values in candidate-level comparison.")
    test = random_sign_permutation_test(gaps, seed=seed, n_permutations=n_permutations)
    std = float(gaps.std(ddof=1)) if len(gaps) > 1 else np.nan
    return {
        "number_candidates": int(len(comparison)),
        "number_control_assignments": int(len(assignments)),
        "number_unique_controls": int(assignments["control_id"].nunique()),
        "controls_per_candidate_distribution": {str(int(k)): int(v) for k, v in assignments.groupby("candidate_id").size().value_counts().sort_index().items()},
        "descriptor_matching_diagnostics": matching_diagnostics,
        "matching_metadata": matching_metadata,
        "tanimoto_distribution_before_guard": tanimoto_distribution,
        "median_target_enrichment_gap": float(np.median(gaps)),
        "mean_target_enrichment_gap": float(np.mean(gaps)),
        "median_target_enrichment_gap_bootstrap_95ci": bootstrap_ci(gaps, seed=seed, statistic="median", n_bootstrap=n_bootstrap),
        "mean_target_enrichment_gap_bootstrap_95ci": bootstrap_ci(gaps, seed=seed + 1, statistic="mean", n_bootstrap=n_bootstrap),
        **test,
        "paired_effect_size_cohen_dz_mean_gap": float(np.mean(gaps) / std) if std and not np.isnan(std) else np.nan,
        "number_candidates_outperforming_matched_control_median": int((comparison["target_enrichment_gap"] > 0).sum()),
        "percentage_candidates_outperforming_matched_control_median": float((comparison["target_enrichment_gap"] > 0).mean() * 100.0),
        "candidate_matched_sets_are_primary_units": True,
    }


def unique_control_table(assignments: pd.DataFrame) -> pd.DataFrame:
    """Return one row per unique matched control molecule."""
    target_cols = [
        "control_id",
        "control_canonical_smiles",
        "control_reuse_count",
        "control_num_conformers_completed",
        "control_mean_target_deltaG",
        "control_median_target_deltaG",
        "control_best_target_deltaG",
        "control_worst_target_deltaG",
        "control_std_target_deltaG",
        "control_target_deltaG_range",
        "control_target_conformer_rank",
    ]
    return assignments[target_cols].drop_duplicates("control_id").sort_values(["control_reuse_count", "control_id"], ascending=[False, True]).reset_index(drop=True)


def make_optional_figures(
    output_dir: Path,
    audit: pd.DataFrame,
    recurrence: pd.DataFrame,
    comparison: pd.DataFrame,
    assignments: pd.DataFrame,
) -> list[str]:
    """Create optional matplotlib figures without making plotting required."""
    try:
        os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="smn2_stage_c5_mpl_"))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    created: list[str] = []

    def save_current(name: str) -> None:
        path = figure_dir / name
        plt.tight_layout()
        plt.savefig(path, dpi=180)
        plt.close()
        created.append(str(path))

    plt.figure(figsize=(6, 4))
    comparison["target_enrichment_gap"].hist(bins=16)
    plt.xlabel("target_enrichment_gap")
    plt.ylabel("Candidates")
    save_current("candidate_matched_control_target_enrichment_distribution.png")

    plt.figure(figsize=(8, 4))
    recurrence.head(15).plot(x="decoy_id", y="num_times_strongest_decoy", kind="bar", legend=False)
    plt.ylabel("Times strongest decoy")
    plt.xticks(rotation=60, ha="right")
    save_current("top40_strongest_decoy_frequency.png")

    plt.figure(figsize=(6, 4))
    audit["num_decoys_stronger_than_worst_target"].hist(bins=range(0, int(audit["num_decoys_stronger_than_worst_target"].max()) + 3))
    plt.xlabel("Decoys stronger than worst target conformer")
    plt.ylabel("Candidates")
    save_current("concerning_hurda_decoys_per_candidate.png")

    plt.figure(figsize=(6, 4))
    plt.scatter(comparison["stage_c_weighted_mean_selectivity_gap"], comparison["target_enrichment_gap"])
    plt.xlabel("Stage-C weighted selectivity gap")
    plt.ylabel("Matched-control target enrichment gap")
    save_current("stage_c_gap_vs_matched_control_enrichment.png")

    plt.figure(figsize=(8, 4))
    descriptor_cols = [col for col in assignments.columns if col.startswith("abs_std_diff_")]
    assignments[descriptor_cols].mean().sort_values().plot(kind="bar")
    plt.ylabel("Mean absolute standardized difference")
    plt.xticks(rotation=60, ha="right")
    save_current("descriptor_balance_after_matching.png")
    return created


def write_methodology(
    path: Path,
    *,
    descriptors: list[str],
    tanimoto_threshold: float,
    seed: int,
    n_bootstrap: int,
    n_permutations: int,
) -> None:
    """Write human-readable methodology metadata."""
    text = f"""Stage C.5 methodology

Analysis scope:
- Reused existing five-conformer SMN2 target docking results and partial HuRDA decoy docking results.
- No new Vina docking was performed.
- Missing HuRDA scores were not repaired, invented, or imputed.

Top-40 HuRDA sanity audit:
- Each Top-40 molecule was required to have exactly 97 completed, numeric, non-dry-run HuRDA scores.
- Conservative gap driver rule: 0 decoys stronger than the molecule's worst target conformer = NO_NEGATIVE_CONSERVATIVE_GAP; 1 = SINGLE_DECOY_DRIVEN; 2-5 = FEW_DECOY_DRIVEN; >5 = BROAD_OFF_TARGET_PATTERN.
- More-negative docking deltaG was treated as stronger predicted docking.

Matched ligand controls:
- Controls were selected only from the existing 500-molecule target5 population after removing Top-40 candidates and exact duplicate candidate structures.
- Matching descriptors: {', '.join(descriptors)}.
- Continuous descriptors were z-score standardized across the candidate plus control population.
- Matching distance: Euclidean distance in standardized descriptor space.
- Morgan fingerprint settings: RDKit Morgan radius=2, fpSize=2048.
- Initial maximum candidate-control Tanimoto threshold: {tanimoto_threshold}.
- Controls were allowed to be reused across candidates; reuse counts are reported.
- Target enrichment gap = median matched-control mean_target_deltaG - candidate mean_target_deltaG, so positive values mean stronger predicted SMN2 ensemble docking for the candidate than the matched-control median.

Group statistics:
- The 40 candidate matched sets were treated as the independent units.
- Bootstrap seed: {seed}; bootstrap resamples: {n_bootstrap}.
- Paired test: random sign-flip permutation test on candidate-level target_enrichment_gap, permutations: {n_permutations}.

Scientific caution:
- These outputs are computational candidate-prioritization evidence only.
- A predicted worst-case docking concern is not proof of biological off-target activity.
- Matched-control enrichment is not proof of therapeutic superiority and requires experimental validation.
"""
    path.write_text(text)


def write_outputs(
    *,
    output_dir: Path,
    overwrite: bool,
    audit: pd.DataFrame,
    recurrence: pd.DataFrame,
    category_summary: pd.DataFrame,
    motif_summary: pd.DataFrame,
    assignments: pd.DataFrame,
    unique_controls: pd.DataFrame,
    comparison: pd.DataFrame,
    group_stats: dict[str, Any],
    validation: dict[str, Any],
    run_summary: pd.DataFrame,
    methodology_kwargs: dict[str, Any],
) -> None:
    """Write all Stage C.5 outputs, refusing accidental overwrites by default."""
    output_dir.mkdir(parents=True, exist_ok=True)
    required = {
        "top40_hurda_sanity_audit.tsv": audit,
        "hurda_decoy_recurrence_summary.tsv": recurrence,
        "matched_ligand_controls.tsv": assignments,
        "matched_control_unique_molecules.tsv": unique_controls,
        "candidate_vs_matched_controls.tsv": comparison,
        "stage_c5_run_summary.tsv": run_summary,
    }
    if not category_summary.empty:
        required["hurda_decoy_category_summary.tsv"] = category_summary
    if not motif_summary.empty:
        required["hurda_motif_summary.tsv"] = motif_summary
    json_outputs = {
        "matched_control_group_statistics.json": group_stats,
        "stage_c5_validation.json": validation,
    }
    output_paths = [output_dir / name for name in [*required.keys(), *json_outputs.keys(), "stage_c5_methodology.txt"]]
    existing = [path for path in output_paths if path.exists()]
    if existing and not overwrite:
        raise StageC5ValidationError(
            "Stage C.5 output files already exist. Rerun with --overwrite to replace only Stage C.5 outputs: "
            + ", ".join(str(path) for path in existing)
        )
    for name, df in required.items():
        df.to_csv(output_dir / name, sep="\t", index=False)
    for name, payload in json_outputs.items():
        (output_dir / name).write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    write_methodology(output_dir / "stage_c5_methodology.txt", **methodology_kwargs)


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Execute Stage C.5 analysis."""
    inputs = load_and_validate_inputs(args)

    top40_ids = set(inputs.top40["molecule_id"].astype(str))
    population = inputs.selected_molecules.copy()
    population, descriptor_meta = compute_descriptors(population)
    population, scaling = zscore_population(population, MATCHING_DESCRIPTORS)
    candidates = population[population["molecule_id"].astype(str).isin(top40_ids)].copy()
    candidates = candidates.merge(inputs.top40[["molecule_id", "partial_poc_selectivity_rank"]], on="molecule_id", how="left")
    if len(candidates) != args.expected_top40:
        raise StageC5ValidationError(f"Descriptor population did not recover all Top-40 candidates; recovered {len(candidates)}.")
    controls, control_pool_meta = build_control_pool(inputs, population)
    if len(controls) < args.controls_per_candidate:
        raise StageC5ValidationError(f"Control pool is too small for matching: {len(controls)}")

    fps = fingerprint_map(population)
    tanimoto_distribution = inspect_tanimoto_distribution(candidates, controls, fps)
    audit = build_top40_hurda_audit(inputs)
    recurrence = summarize_decoy_recurrence(inputs, audit)
    category_summary = summarize_category_or_motif(inputs, audit, "decoy_category")
    motif_summary = summarize_category_or_motif(inputs, audit, "motif_type")
    similarity_relationship = smn2_similarity_relationship(inputs)

    assignments, matching_meta = match_controls(
        candidates=candidates,
        controls=controls,
        target_summary=inputs.target_summary,
        descriptors=MATCHING_DESCRIPTORS,
        fps=fps,
        tanimoto_threshold=args.max_tanimoto,
        controls_per_candidate=args.controls_per_candidate,
    )
    if len(assignments) != args.expected_top40 * args.controls_per_candidate:
        raise StageC5ValidationError(f"Expected 400 matched-control assignments; observed {len(assignments)}.")
    if set(assignments["control_id"].astype(str)) & top40_ids:
        raise StageC5ValidationError("Matched controls include Top-40 candidates.")
    if not pd.to_numeric(assignments["control_num_conformers_completed"], errors="coerce").eq(args.expected_target_conformers).all():
        raise StageC5ValidationError("At least one matched control lacks five target conformer scores.")
    if (assignments["candidate_control_tanimoto"] > assignments["tanimoto_threshold_used"]).any():
        raise StageC5ValidationError("A matched control violates its recorded Tanimoto threshold.")

    matching_diagnostics = evaluate_matching_quality(assignments, MATCHING_DESCRIPTORS)
    comparison = compare_candidates_to_controls(assignments, audit)
    group_stats = compute_group_statistics(
        comparison,
        assignments,
        matching_diagnostics,
        matching_meta,
        tanimoto_distribution,
        seed=args.random_seed,
        n_bootstrap=args.n_bootstrap,
        n_permutations=args.n_permutations,
    )
    unique_controls = unique_control_table(assignments)

    validation = dict(inputs.validation)
    validation.update(
        {
            "git_commit": git_commit(),
            "script_path": str(Path(__file__).resolve()),
            "random_seed": args.random_seed,
            "descriptor_sources": descriptor_meta["descriptor_sources"],
            "descriptor_scaling": scaling,
            "matching_descriptors": MATCHING_DESCRIPTORS,
            "control_pool": control_pool_meta,
            "matched_assignments": int(len(assignments)),
            "unique_matched_controls": int(unique_controls["control_id"].nunique()),
            "controls_per_candidate_distribution": {str(int(k)): int(v) for k, v in assignments.groupby("candidate_id").size().value_counts().sort_index().items()},
            "control_reuse_distribution": group_stats["matching_metadata"]["control_reuse_distribution"],
            "all_controls_outside_top40": bool(not (set(assignments["control_id"].astype(str)) & top40_ids)),
            "all_controls_have_five_target_conformers": bool(pd.to_numeric(assignments["control_num_conformers_completed"], errors="coerce").eq(args.expected_target_conformers).all()),
            "all_tanimoto_similarities_reported": bool(assignments["candidate_control_tanimoto"].notna().all()),
            "all_descriptor_distances_reported": bool(assignments["matching_distance"].notna().all()),
            "missing_scores_imputed": False,
            "new_docking_jobs_run": 0,
            "smn2_similarity_relationship": similarity_relationship,
        }
    )

    run_summary = pd.DataFrame(
        [
            {"metric": "top40_candidates", "value": args.expected_top40},
            {"metric": "top40_hurda_rows", "value": len(inputs.decoy_valid[inputs.decoy_valid["molecule_id"].astype(str).isin(top40_ids)])},
            {"metric": "observed_hurda_decoys", "value": args.expected_decoys_per_candidate},
            {"metric": "matched_control_assignments", "value": len(assignments)},
            {"metric": "unique_controls", "value": unique_controls["control_id"].nunique()},
            {"metric": "median_target_enrichment_gap", "value": group_stats["median_target_enrichment_gap"]},
            {"metric": "percentage_candidates_outperforming_control_median", "value": group_stats["percentage_candidates_outperforming_matched_control_median"]},
            {"metric": "no_new_vina_docking_performed", "value": True},
        ]
    )

    created_figures = make_optional_figures(args.output_dir, audit, recurrence, comparison, assignments) if args.make_figures else []
    validation["optional_figures"] = created_figures

    write_outputs(
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        audit=audit,
        recurrence=recurrence,
        category_summary=category_summary,
        motif_summary=motif_summary,
        assignments=assignments,
        unique_controls=unique_controls,
        comparison=comparison,
        group_stats=group_stats,
        validation=validation,
        run_summary=run_summary,
        methodology_kwargs={
            "descriptors": MATCHING_DESCRIPTORS,
            "tanimoto_threshold": args.max_tanimoto,
            "seed": args.random_seed,
            "n_bootstrap": args.n_bootstrap,
            "n_permutations": args.n_permutations,
        },
    )
    return {
        "validation": validation,
        "group_stats": group_stats,
        "audit": audit,
        "recurrence": recurrence,
        "comparison": comparison,
        "assignments": assignments,
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-c-master", type=Path, default=DEFAULT_STAGE_C_MASTER)
    parser.add_argument("--top40", type=Path, default=DEFAULT_TOP40)
    parser.add_argument("--decoy-scores", type=Path, default=DEFAULT_DECOY_SCORES)
    parser.add_argument("--target-summary", type=Path, default=DEFAULT_TARGET_SUMMARY)
    parser.add_argument("--target-long", type=Path, default=DEFAULT_TARGET_LONG)
    parser.add_argument("--selected-molecules", type=Path, default=DEFAULT_SELECTED_MOLECULES)
    parser.add_argument("--top150", type=Path, default=DEFAULT_TOP150)
    parser.add_argument("--lower-confidence", type=Path, default=DEFAULT_LOWER_CONFIDENCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--expected-top40", type=int, default=40)
    parser.add_argument("--expected-stage-c-molecules", type=int, default=150)
    parser.add_argument("--expected-target-population", type=int, default=500)
    parser.add_argument("--expected-target-conformers", type=int, default=5)
    parser.add_argument("--expected-decoys-per-candidate", type=int, default=97)
    parser.add_argument("--expected-top40-decoy-rows", type=int, default=3880)
    parser.add_argument("--controls-per-candidate", type=int, default=10)
    parser.add_argument("--max-tanimoto", type=float, default=0.60)
    parser.add_argument("--random-seed", type=int, default=61453)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--n-permutations", type=int, default=20000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-figures", dest="make_figures", action="store_false")
    parser.set_defaults(make_figures=True)
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    result = run(args)
    validation = result["validation"]
    group_stats = result["group_stats"]
    print("Stage C.5 Top-40 sanity audit + matched controls complete.")
    print(json.dumps(
        {
            "top40_hurda_rows": validation["top40_hurda_rows"],
            "observed_hurda_decoys": validation["observed_hurda_decoys"],
            "matched_assignments": validation["matched_assignments"],
            "unique_controls": validation["unique_matched_controls"],
            "median_target_enrichment_gap": group_stats["median_target_enrichment_gap"],
            "percentage_candidates_outperforming_control_median": group_stats["percentage_candidates_outperforming_matched_control_median"],
            "no_new_vina_docking_performed": validation["no_new_vina_docking_performed"],
        },
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
