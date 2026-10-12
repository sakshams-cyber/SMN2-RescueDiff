#!/usr/bin/env python3
"""Recompute reviewer-raised geometry, score, and provenance diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdMolDescriptors
from scipy.stats import spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RELEASE = PROJECT_ROOT / "reproducibility" / "tcbb_2026"
DEFAULT_OUTPUT = RELEASE / "reviewer_revision_2026_10"
BOX_CENTER = np.asarray([24.896, 20.352, 24.973], dtype=float)
BOX_SIZE = np.asarray([44.963, 34.551, 44.652], dtype=float)


def read_coordinates(path: Path, first_model_only: bool = False) -> np.ndarray:
    """Read PDB/PDBQT coordinates, optionally stopping after the first model."""
    rows: list[list[float]] = []
    model_seen = False
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("MODEL"):
            if model_seen and first_model_only:
                break
            model_seen = True
            continue
        if first_model_only and model_seen and line.startswith("ENDMDL"):
            break
        if not line.startswith(("ATOM", "HETATM")):
            continue
        try:
            rows.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        except ValueError:
            continue
    if not rows:
        raise ValueError(f"No parseable atom coordinates: {path}")
    return np.asarray(rows, dtype=float)


def eta_squared(frame: pd.DataFrame, group: str, value: str) -> float:
    """Return one-way variance explained by a categorical grouping."""
    grand = float(frame[value].mean())
    total = float(((frame[value] - grand) ** 2).sum())
    between = sum(len(part) * float(part[value].mean() - grand) ** 2 for _, part in frame.groupby(group))
    return float(between / total) if total else float("nan")


def molecule_descriptors(smiles: str) -> dict[str, float | int]:
    """Calculate the manuscript descriptor set directly from a SMILES string."""
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"RDKit could not parse benchmark SMILES: {smiles}")
    return {
        "molecular_weight": float(Descriptors.MolWt(molecule)),
        "qed": float(QED.qed(molecule)),
        "logp": float(Crippen.MolLogP(molecule)),
        "hbd": int(Lipinski.NumHDonors(molecule)),
        "hba": int(Lipinski.NumHAcceptors(molecule)),
        "rotatable_bonds": int(Lipinski.NumRotatableBonds(molecule)),
        "tpsa": float(rdMolDescriptors.CalcTPSA(molecule)),
        "heavy_atom_count": int(molecule.GetNumHeavyAtoms()),
    }


def audit_decoys(output: Path, seed: int) -> dict[str, Any]:
    """Audit historical fixed-box decoy scores and write sensitivity tables."""
    scores = pd.read_csv(RELEASE / "decoy_docking" / "decoy_docking_scores.tsv", sep="\t", low_memory=False)
    lower = BOX_CENTER - BOX_SIZE / 2.0
    upper = BOX_CENTER + BOX_SIZE / 2.0
    receptor_rows: list[dict[str, Any]] = []
    receptor_root = RELEASE / "decoy_docking" / "receptors"
    unique_receptors = scores[["decoy_id", "decoy_category", "decoy_receptor_file"]].drop_duplicates()
    for row in unique_receptors.itertuples(index=False):
        receptor = receptor_root / Path(row.decoy_receptor_file).name
        coordinates = read_coordinates(receptor)
        inside = np.all((coordinates >= lower) & (coordinates <= upper), axis=1)
        receptor_rows.append(
            {
                "decoy_id": row.decoy_id,
                "decoy_category": row.decoy_category,
                "receptor_file": str(receptor.relative_to(PROJECT_ROOT)),
                "atom_count": int(len(coordinates)),
                "receptor_atoms_in_fixed_box": int(inside.sum()),
                "receptor_atoms_in_fixed_box_fraction": float(inside.mean()),
                "centroid_x": float(coordinates[:, 0].mean()),
                "centroid_y": float(coordinates[:, 1].mean()),
                "centroid_z": float(coordinates[:, 2].mean()),
            }
        )
    geometry = pd.DataFrame(receptor_rows)
    receptor_scores = (
        scores.groupby("decoy_id", as_index=False)
        .agg(
            mean_decoy_score=("decoy_deltaG", "mean"),
            median_decoy_score=("decoy_deltaG", "median"),
            strongest_decoy_score=("decoy_deltaG", "min"),
            zero_score_count=("decoy_deltaG", lambda values: int((values == 0).sum())),
            zero_score_fraction=("decoy_deltaG", lambda values: float((values == 0).mean())),
            molecule_count=("molecule_id", "nunique"),
        )
        .merge(geometry, on="decoy_id", how="left")
    )
    receptor_scores.to_csv(output / "decoy_receptor_geometry_and_zero_score_qc.tsv", sep="\t", index=False)

    rho, p_value = spearmanr(
        receptor_scores["receptor_atoms_in_fixed_box_fraction"], receptor_scores["mean_decoy_score"]
    )
    under_ten = receptor_scores["receptor_atoms_in_fixed_box_fraction"] < 0.10
    strongest_counts = (
        scores.loc[scores.groupby("molecule_id")["decoy_deltaG"].idxmin(), "decoy_id"]
        .value_counts()
        .rename_axis("decoy_id")
        .reset_index(name="times_strongest_decoy")
    )
    strongest_counts.to_csv(output / "strongest_decoy_frequency.tsv", sep="\t", index=False)

    target = pd.read_csv(
        RELEASE / "final_scoring" / "stage_c_selection" / "ensemble_selectivity_scores.tsv", sep="\t"
    ).set_index("molecule_id")
    common = sorted(set(scores["molecule_id"]) & set(target.index))
    ordered_decoys = receptor_scores.sort_values("mean_decoy_score")["decoy_id"].tolist()
    sensitivity_rows: list[dict[str, Any]] = []
    for removed_count in (0, 1, 3):
        removed = ordered_decoys[:removed_count]
        subset = scores[~scores["decoy_id"].isin(removed)]
        best = subset.groupby("molecule_id")["decoy_deltaG"].min().loc[common]
        gap = best - target.loc[common, "worst_target_deltaG"]
        sensitivity_rows.append(
            {
                "analysis": "conservative_gap_receptor_removal",
                "receptors_removed": removed_count,
                "removed_decoy_ids": ",".join(removed),
                "molecules": len(gap),
                "median_gap": float(gap.median()),
                "minimum_gap": float(gap.min()),
                "maximum_gap": float(gap.max()),
                "nonnegative_gap_count": int((gap >= 0).sum()),
                "nonnegative_gap_fraction": float((gap >= 0).mean()),
            }
        )
    pd.DataFrame(sensitivity_rows).to_csv(output / "decoy_conservative_gap_sensitivity.tsv", sep="\t", index=False)

    mean_with_zeros = scores.groupby("molecule_id")["decoy_deltaG"].mean().loc[common]
    nonzero = scores[scores["decoy_deltaG"] != 0]
    mean_without_zeros = nonzero.groupby("molecule_id")["decoy_deltaG"].mean().loc[common]
    zero_sensitivity = pd.DataFrame(
        {
            "molecule_id": common,
            "mean_target_score": target.loc[common, "mean_target_deltaG"].to_numpy(),
            "mean_decoy_score_with_zeros": mean_with_zeros.to_numpy(),
            "mean_decoy_score_excluding_zeros": mean_without_zeros.to_numpy(),
        }
    )
    zero_sensitivity["mean_gap_with_zeros"] = (
        zero_sensitivity["mean_decoy_score_with_zeros"] - zero_sensitivity["mean_target_score"]
    )
    zero_sensitivity["mean_gap_excluding_zeros"] = (
        zero_sensitivity["mean_decoy_score_excluding_zeros"] - zero_sensitivity["mean_target_score"]
    )
    zero_sensitivity.to_csv(output / "decoy_zero_score_sensitivity.tsv", sep="\t", index=False)

    matrix = scores.pivot(index="molecule_id", columns="decoy_id", values="decoy_deltaG").loc[common]
    rng = np.random.default_rng(seed)
    panel_rows: list[dict[str, Any]] = []
    for panel_size in (5, 10, 20, 50, matrix.shape[1]):
        replicates = 1 if panel_size == matrix.shape[1] else 2000
        for replicate in range(replicates):
            selected = matrix.columns if replicates == 1 else rng.choice(matrix.columns, size=panel_size, replace=False)
            gap = matrix.loc[:, selected].min(axis=1) - target.loc[common, "worst_target_deltaG"]
            panel_rows.append(
                {
                    "panel_size": panel_size,
                    "replicate": replicate,
                    "median_conservative_gap": float(gap.median()),
                    "nonnegative_gap_fraction": float((gap >= 0).mean()),
                }
            )
    panel_null = pd.DataFrame(panel_rows)
    panel_null.to_csv(output / "worst_decoy_panel_size_null.tsv", sep="\t", index=False)

    sample_rows: list[dict[str, Any]] = []
    for is_zero, group in scores.groupby(scores["decoy_deltaG"].eq(0)):
        sampled = group.sample(n=min(250, len(group)), random_state=seed)
        for row in sampled.itertuples(index=False):
            pose = PROJECT_ROOT / str(row.pose_file)
            receptor = receptor_root / Path(row.decoy_receptor_file).name
            record = {
                "molecule_id": row.molecule_id,
                "decoy_id": row.decoy_id,
                "decoy_deltaG": float(row.decoy_deltaG),
                "is_zero_score": bool(is_zero),
                "pose_exists": pose.exists(),
                "minimum_pose_receptor_distance": np.nan,
            }
            if pose.exists():
                ligand_coordinates = read_coordinates(pose, first_model_only=True)
                receptor_coordinates = read_coordinates(receptor)
                distances = np.linalg.norm(
                    ligand_coordinates[:, None, :] - receptor_coordinates[None, :, :], axis=2
                )
                record["minimum_pose_receptor_distance"] = float(distances.min())
            sample_rows.append(record)
    pose_sample = pd.DataFrame(sample_rows)
    pose_sample.to_csv(output / "zero_score_pose_qc_sample.tsv", sep="\t", index=False)

    plt.figure(figsize=(7.2, 4.8))
    scatter = plt.scatter(
        receptor_scores["receptor_atoms_in_fixed_box_fraction"],
        receptor_scores["mean_decoy_score"],
        c=receptor_scores["zero_score_fraction"],
        cmap="viridis",
        edgecolors="black",
        linewidths=0.35,
    )
    plt.colorbar(scatter, label="Exact-zero score fraction")
    plt.xlabel("Fraction of receptor atoms inside historical fixed box")
    plt.ylabel("Mean Vina score (kcal/mol)")
    plt.title("Historical decoy scores track receptor placement")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output / "decoy_grid_geometry_bias.pdf")
    plt.savefig(output / "decoy_grid_geometry_bias.png", dpi=240)
    plt.close()

    panel_summary = (
        panel_null.groupby("panel_size", as_index=False)
        .agg(
            median_gap=("median_conservative_gap", "median"),
            lower_gap=("median_conservative_gap", lambda x: float(np.quantile(x, 0.025))),
            upper_gap=("median_conservative_gap", lambda x: float(np.quantile(x, 0.975))),
        )
        .sort_values("panel_size")
    )
    plt.figure(figsize=(6.8, 4.5))
    plt.plot(panel_summary["panel_size"], panel_summary["median_gap"], marker="o", color="#0b7285")
    plt.fill_between(
        panel_summary["panel_size"], panel_summary["lower_gap"], panel_summary["upper_gap"], color="#99e9f2", alpha=0.6
    )
    plt.axhline(0, color="black", linewidth=1)
    plt.xlabel("Random decoy-panel size")
    plt.ylabel("Median worst-decoy gap (kcal/mol)")
    plt.title("Worst-decoy statistic depends on panel size")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output / "worst_decoy_panel_size_sensitivity.pdf")
    plt.savefig(output / "worst_decoy_panel_size_sensitivity.png", dpi=240)
    plt.close()

    pose_group = pose_sample.groupby("is_zero_score")["minimum_pose_receptor_distance"].median()
    return {
        "decoy_score_rows": int(len(scores)),
        "decoy_receptors": int(scores["decoy_id"].nunique()),
        "exact_zero_scores": int((scores["decoy_deltaG"] == 0).sum()),
        "exact_zero_fraction": float((scores["decoy_deltaG"] == 0).mean()),
        "receptors_under_10_percent_in_fixed_box": int(under_ten.sum()),
        "fixed_box_coverage_score_spearman_rho": float(rho),
        "fixed_box_coverage_score_spearman_p": float(p_value),
        "mean_score_receptors_under_10_percent_in_box": float(receptor_scores.loc[under_ten, "mean_decoy_score"].mean()),
        "mean_score_other_receptors": float(receptor_scores.loc[~under_ten, "mean_decoy_score"].mean()),
        "variance_explained_by_receptor": eta_squared(scores, "decoy_id", "decoy_deltaG"),
        "variance_explained_by_ligand": eta_squared(scores, "molecule_id", "decoy_deltaG"),
        "unique_biological_weights": sorted(scores["decoy_biological_weight"].dropna().unique().tolist()),
        "strongest_decoy_for_all_molecules": str(strongest_counts.iloc[0]["decoy_id"]),
        "strongest_decoy_frequency": int(strongest_counts.iloc[0]["times_strongest_decoy"]),
        "sample_median_minimum_distance_zero_scores": float(pose_group.get(True, np.nan)),
        "sample_median_minimum_distance_nonzero_scores": float(pose_group.get(False, np.nan)),
        "mean_gap_with_zeros_median": float(zero_sensitivity["mean_gap_with_zeros"].median()),
        "mean_gap_excluding_zeros_median": float(zero_sensitivity["mean_gap_excluding_zeros"].median()),
    }


def audit_target_size_and_descriptors(output: Path) -> dict[str, Any]:
    """Quantify molecular-size confounding and correct the leading descriptors."""
    molecules = pd.read_csv(RELEASE / "target_docking" / "selected_molecules.tsv", sep="\t", low_memory=False)
    target = pd.read_csv(
        RELEASE / "target_docking" / "target_conformer_robustness_scores_final.tsv", sep="\t", low_memory=False
    )
    joined = target.merge(molecules, on="molecule_id", how="left", suffixes=("", "_molecule"))
    joined["molecular_weight"] = pd.to_numeric(joined["molecular_weight"], errors="coerce")
    joined["heavy_atom_count"] = joined["canonical_smiles"].map(
        lambda value: Chem.MolFromSmiles(str(value)).GetNumHeavyAtoms()
    )
    joined["vina_ligand_efficiency"] = -joined["mean_target_deltaG"] / joined["heavy_atom_count"]
    slope, intercept = np.polyfit(joined["molecular_weight"], joined["mean_target_deltaG"], 1)
    joined["target_score_expected_from_mw"] = slope * joined["molecular_weight"] + intercept
    joined["target_score_mw_residual"] = joined["mean_target_deltaG"] - joined["target_score_expected_from_mw"]
    joined["ligand_efficiency_rank"] = joined["vina_ligand_efficiency"].rank(ascending=False, method="min")
    joined["mw_residual_rank"] = joined["target_score_mw_residual"].rank(ascending=True, method="min")
    columns = [
        "molecule_id",
        "canonical_smiles",
        "molecular_weight",
        "heavy_atom_count",
        "mean_target_deltaG",
        "worst_target_deltaG",
        "vina_ligand_efficiency",
        "target_score_expected_from_mw",
        "target_score_mw_residual",
        "ligand_efficiency_rank",
        "mw_residual_rank",
    ]
    joined[columns].to_csv(output / "target_score_size_sensitivity.tsv", sep="\t", index=False)
    rho, p_value = spearmanr(joined["molecular_weight"], joined["mean_target_deltaG"])

    generation = pd.read_csv(RELEASE / "generation" / "generated_molecules_for_docking_original.tsv", sep="\t")
    candidate = generation[generation["molecule_id"] == "MolGen_038292"].iloc[0]
    descriptors = molecule_descriptors(str(candidate["canonical_smiles"]))
    descriptor_row = {
        "molecule_id": "MolGen_038292",
        "canonical_smiles": candidate["canonical_smiles"],
        **descriptors,
    }
    selectivity = pd.read_csv(
        RELEASE / "final_scoring" / "stage_c_selection" / "ensemble_selectivity_scores.tsv", sep="\t"
    ).set_index("molecule_id")
    stronger_percent = float(
        selectivity.loc["MolGen_038292", "percent_decoys_stronger_than_mean_target"]
    )
    descriptor_row["decoys_stronger_than_mean_target_count"] = int(round(stronger_percent * 97 / 100))
    descriptor_row["decoy_panel_size"] = 97
    descriptor_row["fraction_decoys_stronger_than_mean_target"] = stronger_percent / 100.0
    descriptor_row["percent_decoys_stronger_than_mean_target"] = stronger_percent
    pd.DataFrame([descriptor_row]).to_csv(output / "molgen_038292_corrected_descriptors.tsv", sep="\t", index=False)

    benchmark_input = pd.read_csv(
        PROJECT_ROOT / "data/processed/benchmark_controls/risdiplam_branaplam_2x100/benchmark_controls_for_docking.tsv",
        sep="\t",
    )
    benchmark_rows = []
    for row in benchmark_input.itertuples(index=False):
        name = getattr(row, "benchmark_control_name", getattr(row, "molecule_id"))
        benchmark_rows.append(
            {"benchmark_control_name": name, "canonical_smiles": row.canonical_smiles, **molecule_descriptors(row.canonical_smiles)}
        )
    pd.DataFrame(benchmark_rows).to_csv(output / "benchmark_control_rdkit_descriptors.tsv", sep="\t", index=False)

    plt.figure(figsize=(6.8, 4.7))
    plt.scatter(joined["molecular_weight"], joined["mean_target_deltaG"], s=18, alpha=0.45, color="#1971c2")
    x_line = np.linspace(joined["molecular_weight"].min(), joined["molecular_weight"].max(), 100)
    plt.plot(x_line, slope * x_line + intercept, color="#c92a2a", linewidth=2, label="Linear size trend")
    lead = joined[joined["molecule_id"] == "MolGen_038292"].iloc[0]
    plt.scatter([lead["molecular_weight"]], [lead["mean_target_deltaG"]], color="#f08c00", edgecolor="black", s=90, label="MolGen_038292")
    plt.xlabel("Molecular weight (Da)")
    plt.ylabel("Mean target Vina score (kcal/mol)")
    plt.title("Target score retains a molecular-size dependence")
    plt.legend(frameon=False)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output / "target_score_molecular_weight_confounding.pdf")
    plt.savefig(output / "target_score_molecular_weight_confounding.png", dpi=240)
    plt.close()

    lead_size = joined[joined["molecule_id"] == "MolGen_038292"].iloc[0]
    return {
        "target_molecules": int(len(joined)),
        "target_score_mw_spearman_rho": float(rho),
        "target_score_mw_spearman_p": float(p_value),
        "target_score_mw_linear_slope": float(slope),
        "lead_candidate_ligand_efficiency": float(lead_size["vina_ligand_efficiency"]),
        "lead_candidate_ligand_efficiency_rank_of_500": int(lead_size["ligand_efficiency_rank"]),
        "lead_candidate_mw_residual": float(lead_size["target_score_mw_residual"]),
        "lead_candidate_mw_residual_rank_of_500": int(lead_size["mw_residual_rank"]),
        "lead_candidate_descriptors": descriptor_row,
    }


def audit_generation_flow(output: Path) -> dict[str, Any]:
    """Reconcile the generated-string, canonicalization, and filtering denominators."""
    run = PROJECT_ROOT / "data/processed/molecule_generation/runs/module3_final_final_rigorous_x10"
    valid = pd.read_csv(run / "generated_molecules_valid.tsv", sep="\t", low_memory=False)
    filtered = pd.read_csv(run / "generated_molecules_filtered.tsv", sep="\t", low_memory=False)
    rejected = pd.read_csv(run / "generated_molecules_rejected.tsv", sep="\t", low_memory=False)
    canonical_duplicates = int(valid["is_duplicate"].astype(bool).sum())
    unique_filter_failures = int((~rejected["is_duplicate"].astype(bool)).sum())
    rows = [
        ("raw_samples_requested", 50000, "sampler request"),
        ("unique_strings_emitted", len(valid), "deduplicated generated strings presented to RDKit"),
        ("rdkit_parse_valid_strings", int(valid["is_valid"].astype(bool).sum()), "denominator: unique emitted strings"),
        ("canonicalization_collapses", canonical_duplicates, "valid strings mapping to an already observed canonical graph"),
        ("unique_canonical_molecules", len(valid) - canonical_duplicates, "unique emitted strings minus canonicalization collapses"),
        ("unique_canonical_filter_failures", unique_filter_failures, "unique canonical molecules failing chemical filters"),
        ("retained_unique_molecules", len(filtered), "unique canonical molecules passing chemical filters"),
        ("rejected_records_total", len(rejected), "canonical duplicates plus unique chemical-filter failures"),
        ("exported_for_docking", 1000, "diversity-selected docking export"),
    ]
    flow = pd.DataFrame(rows, columns=["stage", "count", "definition"])
    flow.to_csv(output / "generation_flow_reconciliation.tsv", sep="\t", index=False)
    return {row.stage: int(row.count) for row in flow.itertuples(index=False)}


def audit_cross_target_and_benchmarks(output: Path) -> dict[str, Any]:
    """Record the locally available cross-target and benchmark campaigns."""
    cross_path = PROJECT_ROOT / "data/processed/docking/cross_target_offtargets/top40x20_v1/cross_target_docking_scores.tsv"
    cross = pd.read_csv(cross_path, sep="\t", low_memory=False)
    cross_summary = (
        cross.groupby(["receptor_id", "receptor_class"], as_index=False)
        .agg(
            jobs=("molecule_id", "size"),
            completed_jobs=("dock_status", lambda x: int((x == "completed").sum())),
            mean_offtarget_score=("offtarget_deltaG", "mean"),
            minimum_offtarget_score=("offtarget_deltaG", "min"),
            maximum_offtarget_score=("offtarget_deltaG", "max"),
        )
    )
    cross_summary.to_csv(output / "cross_target_40x20_summary.tsv", sep="\t", index=False)

    benchmark_root = PROJECT_ROOT / "data/processed/benchmark_controls/risdiplam_branaplam_2x100/module4_2x100_v2"
    benchmark_target = pd.read_csv(benchmark_root / "smn2_target_docking_scores.tsv", sep="\t")
    benchmark_decoy = pd.read_csv(benchmark_root / "decoy_docking_scores.tsv", sep="\t")
    benchmark_selectivity = pd.read_csv(benchmark_root / "molecule_selectivity_scores.tsv", sep="\t")
    benchmark_selectivity.to_csv(output / "benchmark_controls_2x100_selectivity_summary.tsv", sep="\t", index=False)
    return {
        "cross_target_rows": int(len(cross)),
        "cross_target_molecules": int(cross["molecule_id"].nunique()),
        "cross_target_receptors": int(cross["receptor_id"].nunique()),
        "cross_target_completed": int((cross["dock_status"] == "completed").sum()),
        "cross_target_real_scores": int((cross["is_dry_run_score"] == False).sum()),  # noqa: E712
        "benchmark_target_rows": int(len(benchmark_target)),
        "benchmark_decoy_rows": int(len(benchmark_decoy)),
        "benchmark_molecules": int(benchmark_decoy["molecule_id"].nunique()),
        "benchmark_warning": "Benchmark decoy scores share the historical fixed-box coordinate confound and are released for audit, not inference.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    metrics = {
        "analysis_scope": "Post hoc audit of released historical results; no score was imputed or replaced.",
        "decoy_geometry_and_score_audit": audit_decoys(output, args.seed),
        "target_size_and_descriptor_audit": audit_target_size_and_descriptors(output),
        "generation_flow": audit_generation_flow(output),
        "released_campaigns": audit_cross_target_and_benchmarks(output),
    }
    (output / "reviewer_revision_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
