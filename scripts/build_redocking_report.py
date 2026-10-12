#!/usr/bin/env python3
"""Build the final redocking validation report from locked result tables."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/processed/redocking_validation"
DOCKING = BASE / "docking"
ANALYSIS = BASE / "analysis"
REPORT = ROOT / "REDOCKING_REPORT.md"
PROTOCOL_PATH = BASE / "redocking_protocol.yaml"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: str | Path) -> str:
    return str(Path(path).resolve().relative_to(ROOT))


def fmt(value: Any, digits: int = 3) -> str:
    if pd.isna(value):
        return "not applicable"
    return f"{float(value):.{digits}f}"


def bool_text(value: Any) -> str:
    if isinstance(value, str):
        value = value.strip().lower() in {"true", "1", "yes"}
    return "yes" if bool(value) else "no"


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def residue_from_contact(contact: str) -> str:
    fields = contact.split(":")
    return ":".join(fields[:2])


def conclusion(summary: pd.DataFrame) -> tuple[str, str]:
    success = {
        row.pdb_id: bool(row.headline_within_2A_of_any_deposited_model)
        and bool(row.operational_meaningful_contact_recovery)
        for row in summary.itertuples(index=False)
    }
    target_ids = [pdb_id for pdb_id in ("8R62", "8R8P", "8R63") if pdb_id in success]
    target_values = [success[pdb_id] for pdb_id in target_ids]
    control_success = success.get("5KX9", False)
    if target_values and all(target_values):
        label = "Reading 1"
        prose = (
            "The preregistered protocol recovered the top-scored bound poses and meaningful "
            "RNA-contact patterns for the SMN splicing-modifier benchmarks. The docking "
            "instrument is therefore demonstrably capable on this target under this validation, "
            "and the manuscript's negative worst-case selectivity result is attributable to the "
            "tested candidates rather than a complete inability to resolve the target poses."
        )
    elif target_values and not any(target_values):
        if control_success:
            label = "Mixed result"
            prose = (
                "The preregistered protocol recovered the unrelated 5KX9 RNA-ligand control but "
                "did not recover the SMN splicing-modifier complexes by both pose and contact "
                "criteria. The instrument can resolve at least one structured RNA complex, but "
                "it is not validated for this SMN target; every target-derived docking quantity "
                "in the manuscript is therefore bounded by that target-specific failure."
            )
        else:
            label = "Reading 2"
            prose = (
                "The preregistered protocol did not recover the SMN target poses with meaningful "
                "contact recovery and also failed the unrelated RNA-ligand positive control. The "
                "instrument is not validated for resolving these RNA-ligand poses, so every "
                "docking-derived selectivity quantity in the manuscript is bounded by this "
                "measurement failure rather than establishing a molecular property of the candidates."
            )
    else:
        label = "Mixed result"
        prose = (
            "The preregistered protocol produced mixed recovery across the SMN benchmarks. It "
            "therefore shows some RNA-pose capability but does not support treating the docking "
            "instrument as uniformly validated for the target; manuscript claims must remain "
            "bounded by the failed benchmark complexes."
        )
    return label, prose


def main() -> int:
    if REPORT.exists():
        raise FileExistsError(f"Refusing to overwrite existing report: {REPORT}")
    protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    lock = json.loads((BASE / "protocol_lock.json").read_text(encoding="utf-8"))
    run = json.loads((DOCKING / "run_metadata.json").read_text(encoding="utf-8"))
    analysis_metadata = json.loads((ANALYSIS / "analysis_metadata.json").read_text(encoding="utf-8"))
    if sha256(PROTOCOL_PATH) != lock["protocol_sha256"]:
        raise RuntimeError("Protocol lock mismatch while building report.")
    if run["failed_jobs"] != 0 or run["status"] != "completed":
        raise RuntimeError("Cannot build a completed report from an incomplete docking run.")
    summary = pd.read_csv(ANALYSIS / "redocking_validation_summary.tsv", sep="\t")
    structures = pd.read_csv(BASE / "structure_metadata.tsv", sep="\t")
    scores = pd.read_csv(DOCKING / "redocking_mode_scores.tsv", sep="\t")
    contacts = pd.read_csv(ANALYSIS / "model1_contact_comparison.tsv", sep="\t")
    alignments = pd.read_csv(ANALYSIS / "rna_alignment_qc.tsv", sep="\t")

    all_inputs: set[Path] = {
        PROTOCOL_PATH,
        BASE / "protocol_lock.json",
        BASE / "structure_metadata.tsv",
        BASE / "model_preparation_summary.tsv",
        BASE / "fresh_ligand_preparation_summary.tsv",
        DOCKING / "redocking_mode_scores.tsv",
        ROOT / "scripts/prepare_redocking_validation.py",
        ROOT / "scripts/run_redocking_validation.py",
        ROOT / "scripts/analyze_redocking_validation.py",
        Path(__file__).resolve(),
    }
    manifest = pd.read_csv(DOCKING / "docking_job_manifest.tsv", sep="\t")
    all_inputs.update(Path(value) for value in manifest["receptor_file"].unique())
    all_inputs.update(Path(value) for value in manifest["ligand_file"].unique())
    all_inputs.update(Path(value) for value in scores["pose_file"].unique())
    all_inputs.update((BASE / "structures").glob("*.pdb"))
    all_inputs.update((BASE / "components").glob("*.cif"))
    all_inputs.update((BASE / "prepared").glob("*/model_*/receptor_rna.pdb"))
    all_inputs.update((BASE / "prepared").glob("*/model_*/reference_ligand_deposited.pdb"))
    all_inputs.update((BASE / "prepared/ligands_from_smiles").glob("*.sdf"))
    hash_rows = [
        {"path": relative(path), "sha256": sha256(path)}
        for path in sorted(all_inputs, key=lambda item: str(item))
    ]
    pd.DataFrame(hash_rows).to_csv(BASE / "final_input_sha256.tsv", sep="\t", index=False)

    display = pd.DataFrame(
        {
            "PDB ID": summary["pdb_id"],
            "ligand": summary["ligand_name"],
            "ensemble models": summary["ensemble_models"].astype(int),
            "native contacts": summary["native_model1_contact_atoms"].astype(int),
            "top-mode RMSD to model 1 (A)": summary["headline_model1_rmsd_angstrom"].map(fmt),
            "best RMSD across ensemble (A)": summary[
                "headline_minimum_ensemble_rmsd_angstrom"
            ].map(fmt),
            "best RMSD over all modes (A)": summary[
                "best_all_modes_minimum_ensemble_rmsd_angstrom"
            ].map(fmt),
            "seeds under 2.0 A": summary.apply(
                lambda row: f"{int(row['rank1_seed_successes_below_2A'])}/{int(row['primary_model1_seeds'])}",
                axis=1,
            ),
            "contact recall": summary["contact_recall"].map(fmt),
            "contact Jaccard": summary["contact_jaccard"].map(fmt),
        }
    )

    structure_display = structures[
        [
            "pdb_id",
            "ligand_name",
            "ligand_component_id",
            "experimental_method",
            "ensemble_models",
            "formula",
            "formula_weight",
            "heavy_atom_count",
            "preparation_status",
        ]
    ].copy()
    structure_display.columns = [
        "PDB ID",
        "ligand",
        "component",
        "deposited method",
        "models",
        "formula",
        "formula weight",
        "heavy atoms",
        "status",
    ]

    grid_rows = []
    for pdb_id, item in protocol["structures"].items():
        grid = item["grid_box"]
        grid_rows.append(
            {
                "PDB ID": pdb_id,
                "center x/y/z (A)": f"{grid['center_x']}, {grid['center_y']}, {grid['center_z']}",
                "size x/y/z (A)": f"{grid['size_x']}, {grid['size_y']}, {grid['size_z']}",
            }
        )

    score_counts = (
        scores.groupby("pdb_id")
        .agg(jobs=("pose_file", "nunique"), retained_modes=("mode", "size"), best_affinity=("affinity_kcal_mol", "min"), worst_retained_affinity=("affinity_kcal_mol", "max"))
        .reset_index()
    )
    score_counts.columns = [
        "PDB ID",
        "completed jobs",
        "retained modes",
        "best affinity (kcal/mol)",
        "worst retained affinity (kcal/mol)",
    ]
    score_counts["best affinity (kcal/mol)"] = score_counts["best affinity (kcal/mol)"].map(fmt)
    score_counts["worst retained affinity (kcal/mol)"] = score_counts[
        "worst retained affinity (kcal/mol)"
    ].map(fmt)

    label, conclusion_text = conclusion(summary)
    lines = [
        "# RNA-Ligand Redocking Validation",
        "",
        "This report validates the AutoDock Vina protocol used by SMN2-RescueDiff against public RNA-ligand complexes. It reports the preregistered result whether positive, negative, or mixed; it does not infer biological activity from docking.",
        "",
        "## Task 1 - Prepare",
        "",
        markdown_table(structure_display),
        "",
        "The prompt described all required structures as solution NMR. The deposited records disagree: 8R62 and the added branaplam complex 8R63 are solution NMR, 8R8P is solid-state NMR, and 5KX9 is X-ray diffraction. The deposited metadata govern this report. Pairwise ensemble spread is therefore not applicable to the single-model 5KX9 structure.",
        "",
        "Branaplam was added from RCSB PDB 8R63, component Y53. No predicted or homology structure was substituted.",
        "",
        "Receptor preparation audited `smn2_rescuediff.docking.receptor_preparer.prepare_receptor`, which validates paths and metadata but does not create PDBQT. Open Babel 3.2.1 therefore supplied the missing conversion step. Each ligand-free RNA receptor was converted with Gasteiger charges and rigid PDBQT output (`-xr`). The existing production SMN2 receptor was not reused because audit showed that it contains deposited Y59 atoms.",
        "",
        "Deposited ligand coordinates were retained only as references. Docking used independent RDKit ETKDGv3 conformers generated from RCSB component descriptors with seed 61453, MMFF/UFF optimization, and Open Babel ligand PDBQT conversion. Risdiplam's RCSB CACTVS SMILES failed RDKit kekulization, so the RCSB InChI from the same component dictionary was converted to a valid RDKit canonical SMILES before the conformer was generated.",
        "",
        "### Locked Grid Boxes",
        "",
        markdown_table(pd.DataFrame(grid_rows)),
        "",
        "Each center is the model-1 deposited ligand heavy-atom centroid. Each box dimension is the corresponding ligand extent plus 16 A, providing 8 A padding on both sides.",
        "",
        "### Exact Commands",
        "",
        "```bash",
        "conda run -n smn2-docking python -c \"from rdkit import Chem; print('ok')\"",
        "conda run -n smn2-docking vina --version",
        "conda run -n smn2-docking python -m pip install spyrmsd==0.9.0",
        "conda run -n smn2-docking env PYTHONPATH=src MPLCONFIGDIR=/tmp/smn2-mpl python scripts/prepare_redocking_validation.py",
        "conda run -n smn2-docking env PYTHONPATH=src MPLCONFIGDIR=/tmp/smn2-mpl python scripts/run_redocking_validation.py",
        "conda run -n smn2-docking env PYTHONPATH=src MPLCONFIGDIR=/tmp/smn2-mpl python scripts/analyze_redocking_validation.py",
        "conda run -n smn2-docking env PYTHONPATH=src python scripts/build_redocking_report.py",
        "```",
        "",
        f"The locked protocol SHA-256 is `{lock['protocol_sha256']}` and the recorded git commit is `{protocol['git_commit']}`. File-level hashes for every downloaded structure, component dictionary, prepared docking input, pose input, protocol, and validation script are in [`data/processed/redocking_validation/final_input_sha256.tsv`](data/processed/redocking_validation/final_input_sha256.tsv). Per-job receptor and ligand hashes and all 225 exact Vina commands are retained in the individual `job_metadata.json` files and the docking manifest.",
        "",
        "### Preparation and Pre-RMSD Failures",
        "",
        "1. The first preparation attempt retained the RCSB CACTVS risdiplam SMILES verbatim; RDKit rejected it during kekulization. That failed attempt is preserved under `data/processed/redocking_validation/preparation_attempt_001_failed/`. No docking or RMSD was performed from it.",
        "2. The second preparation attempt omitted Open Babel's rigid PDBQT flag. All 225 Vina invocations stopped at receptor parsing on `ROOT` records before search, so no scores or poses were produced. It is preserved under `data/processed/redocking_validation/preparation_attempt_002_failed_nonrigid_receptors/`.",
        "3. Both corrections were made before any RMSD was calculated. The final protocol was then relocked and was not changed after results were observed.",
        "",
        "## Task 2 - Redock",
        "",
        f"Vina version: `{protocol['software']['vina']}`. Exhaustiveness: {protocol['docking']['exhaustiveness']}. Requested modes: {protocol['docking']['num_modes']}. Energy range: {protocol['docking']['energy_range']} kcal/mol. Seeds: {', '.join(str(seed) for seed in protocol['docking']['seeds'])}. CPU per job: {protocol['docking']['cpu_per_job']}. Parallel workers: {protocol['docking']['parallel_workers']}. All deposited receptor models were docked, with model 1 designated primary.",
        "",
        markdown_table(score_counts),
        "",
        f"Final run status: {run['completed_jobs']}/{run['total_jobs']} jobs completed; {run['failed_jobs']} failed; {run['modes_scored']} retained modes; all scores real and non-dry-run: {bool_text(run['all_scores_are_real'])}. Every retained affinity is reported in [`data/processed/redocking_validation/docking/redocking_mode_scores.tsv`](data/processed/redocking_validation/docking/redocking_mode_scores.tsv).",
        "",
        "## Task 3 - Heavy-Atom RMSD Against the Ensemble",
        "",
        "RMSD was calculated with `spyrmsd.symmrmsd` using atomic-number and connectivity graph isomorphism, with `center=False` and `minimize=False`. Thus symmetry and atom-order differences were corrected, but docked ligands were not superposed onto the answer. Before ligand comparison, common receptor RNA heavy atoms aligned each deposited ensemble model to the receptor model used for docking. The complete pose-by-reference table is [`data/processed/redocking_validation/analysis/pose_to_ensemble_rmsd_long.tsv`](data/processed/redocking_validation/analysis/pose_to_ensemble_rmsd_long.tsv).",
        "",
        markdown_table(display),
        "",
    ]

    for row in summary.itertuples(index=False):
        status = "is" if row.headline_within_2A_of_any_deposited_model else "is not"
        lines.append(
            f"**{row.pdb_id} {row.ligand_name}:** The top-scored mode {status} within 2.0 A of at least one deposited model (best ensemble RMSD {fmt(row.headline_minimum_ensemble_rmsd_angstrom)} A; model-1 RMSD {fmt(row.headline_model1_rmsd_angstrom)} A). The best pose over all retained primary-run modes was mode {int(row.best_all_modes_rank)} at seed {int(row.best_all_modes_seed)}, RMSD {fmt(row.best_all_modes_minimum_ensemble_rmsd_angstrom)} A to deposited model {int(row.best_all_modes_closest_reference_model)}. Rank-1 seed success was {int(row.rank1_seed_successes_below_2A)}/{int(row.primary_model1_seeds)} ({fmt(row.rank1_seed_success_fraction_below_2A, 2)}). Deposited ensemble pairwise ligand RMSD min/median/max was {fmt(row.ensemble_pairwise_ligand_rmsd_min_angstrom)}/{fmt(row.ensemble_pairwise_ligand_rmsd_median_angstrom)}/{fmt(row.ensemble_pairwise_ligand_rmsd_max_angstrom)} A."
        )
        lines.append("")

    lines.extend(
        [
            f"RNA alignment QC used {int(alignments['common_rna_heavy_atoms'].min())}-{int(alignments['common_rna_heavy_atoms'].max())} common heavy atoms; alignment RMSD ranged from {fmt(alignments['rna_alignment_rmsd_angstrom'].min(), 4)} to {fmt(alignments['rna_alignment_rmsd_angstrom'].max(), 4)} A.",
            "",
            "## Task 4 - Contact Recovery",
            "",
            "A contact is a ligand heavy atom within 4.0 A of a receptor heavy atom. Contact identities retain RNA chain, residue number, insertion code, residue name, and atom name. The top-scored pose is the lowest-affinity mode across the five model-1 seed runs.",
            "",
        ]
    )
    contact_display = pd.DataFrame(
        {
            "PDB ID": summary["pdb_id"],
            "native contacts": summary["native_model1_contact_atoms"].astype(int),
            "predicted contacts": summary["headline_predicted_contact_atoms"].astype(int),
            "recovered": summary["recovered_native_contact_atoms"].astype(int),
            "precision": summary["contact_precision"].map(fmt),
            "recall": summary["contact_recall"].map(fmt),
            "Jaccard": summary["contact_jaccard"].map(fmt),
            "meaningful by locked criterion": summary[
                "operational_meaningful_contact_recovery"
            ].map(bool_text),
        }
    )
    lines.extend([markdown_table(contact_display), ""])
    for pdb_id in summary["pdb_id"]:
        subset = contacts[contacts["pdb_id"] == pdb_id].copy()
        native_residues = {
            residue_from_contact(row.contact_atom)
            for row in subset.itertuples(index=False)
            if bool(row.in_deposited_model1_contact_set)
        }
        predicted_residues = {
            residue_from_contact(row.contact_atom)
            for row in subset.itertuples(index=False)
            if bool(row.in_headline_predicted_contact_set)
        }
        native_only = sorted(native_residues - predicted_residues)
        predicted_only = sorted(predicted_residues - native_residues)
        lines.append(
            f"- **{pdb_id}:** Native-contact residues missed by the docked pose: {', '.join(native_only) if native_only else 'none'}. Docked-only contact residues: {', '.join(predicted_only) if predicted_only else 'none'}."
        )
    lines.extend(
        [
            "",
            "Atom-level contact membership is reported in [`data/processed/redocking_validation/analysis/model1_contact_comparison.tsv`](data/processed/redocking_validation/analysis/model1_contact_comparison.tsv). The locked operational criterion for meaningful contact recovery was recall >= 0.50 and Jaccard >= 0.33.",
            "",
            "## Task 5 - What It Means for the Paper",
            "",
            f"**Conclusion: {label}.** {conclusion_text}",
            "",
            "This validation addresses pose and contact recovery under one fixed Vina preparation and search protocol. It does not establish binding affinity accuracy, biological splicing rescue, toxicity, or clinical performance. The added branaplam result is a fourth benchmark and does not alter the preregistered requirement to report all three required structures.",
            "",
            "## NUMBERS FOR THE MANUSCRIPT",
            "",
            f"- AutoDock Vina: {protocol['software']['vina']}.",
            f"- Locked protocol SHA-256: `{lock['protocol_sha256']}`.",
            f"- Redocking jobs: {run['completed_jobs']}/{run['total_jobs']} completed; {run['failed_jobs']} failed.",
            f"- Retained real Vina modes: {run['modes_scored']}.",
        ]
    )
    for row in summary.itertuples(index=False):
        lines.append(
            f"- {row.pdb_id} {row.ligand_name}: top-mode model-1 RMSD {fmt(row.headline_model1_rmsd_angstrom)} A; best top-mode ensemble RMSD {fmt(row.headline_minimum_ensemble_rmsd_angstrom)} A; best all-mode ensemble RMSD {fmt(row.best_all_modes_minimum_ensemble_rmsd_angstrom)} A (rank {int(row.best_all_modes_rank)}, seed {int(row.best_all_modes_seed)}); rank-1 seeds <2.0 A {int(row.rank1_seed_successes_below_2A)}/{int(row.primary_model1_seeds)}; native/recovered contacts {int(row.native_model1_contact_atoms)}/{int(row.recovered_native_contact_atoms)}; contact precision {fmt(row.contact_precision)}, recall {fmt(row.contact_recall)}, Jaccard {fmt(row.contact_jaccard)}; ensemble pairwise RMSD min/median/max {fmt(row.ensemble_pairwise_ligand_rmsd_min_angstrom)}/{fmt(row.ensemble_pairwise_ligand_rmsd_median_angstrom)}/{fmt(row.ensemble_pairwise_ligand_rmsd_max_angstrom)} A."
        )
    lines.extend(["", conclusion_text, ""])

    REPORT.write_text("\n".join(lines), encoding="utf-8")
    report_metadata = {
        "report": str(REPORT),
        "report_sha256": sha256(REPORT),
        "protocol_sha256": lock["protocol_sha256"],
        "analysis_metadata": analysis_metadata,
        "conclusion_label": label,
        "hashed_input_files": len(hash_rows),
    }
    (BASE / "report_metadata.json").write_text(
        json.dumps(report_metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report_metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
