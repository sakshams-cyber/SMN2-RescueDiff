#!/usr/bin/env python3
"""Prepare public RNA-ligand complexes for preregistered redocking.

This stage performs no docking and no RMSD calculation. It splits every
deposited model, prepares RNA receptors with the same direct Open Babel PDB to
PDBQT procedure visible in the existing HuRDA receptor files, prepares fresh
ligand conformers through the repository's Module 4 ligand preparer, and locks
all docking and analysis parameters before any pose recovery result is seen.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from rdkit import Chem

from smn2_rescuediff.docking.ligand_preparer import prepare_ligands


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/processed/redocking_validation"
STRUCTURES = OUT / "structures"
COMPONENTS = OUT / "components"
PREPARED = OUT / "prepared"
LOGS = OUT / "logs/preparation"
PROTOCOL_PATH = OUT / "redocking_protocol.yaml"
LOCK_PATH = OUT / "protocol_lock.json"

STRUCTURE_SPECS = [
    {"pdb_id": "8R62", "ligand_id": "Y59", "ligand_name": "Risdiplam"},
    {"pdb_id": "8R8P", "ligand_id": "YB3", "ligand_name": "SMN-CX"},
    {"pdb_id": "5KX9", "ligand_id": "6YG", "ligand_name": "Ribocil analogue 6YG"},
    {"pdb_id": "8R63", "ligand_id": "Y53", "ligand_name": "Branaplam"},
]

RNA_RESIDUES = {"A", "C", "G", "U", "I", "PSU"}
VINA_SEEDS = [61453, 61459, 61463, 61469, 61471]
CONFORMER_SEED = 61453


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_nonempty(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Required input file is missing or empty: {path}")


def command_version(command: list[str]) -> str:
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return (result.stdout or result.stderr).strip()


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def pdb_models(path: Path) -> tuple[list[list[str]], str]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    method = " ".join(line[10:].strip() for line in lines if line.startswith("EXPDTA")) or "UNKNOWN"
    has_model_records = any(line.startswith("MODEL ") for line in lines)
    if not has_model_records:
        return [[line for line in lines if line.startswith(("ATOM  ", "HETATM"))]], method
    models: list[list[str]] = []
    current: list[str] | None = None
    for line in lines:
        if line.startswith("MODEL "):
            current = []
        elif line.startswith("ENDMDL"):
            if current is not None:
                models.append(current)
            current = None
        elif current is not None and line.startswith(("ATOM  ", "HETATM")):
            current.append(line)
    return models, method


def atom_element(line: str) -> str:
    element = line[76:78].strip()
    if element:
        return element.upper()
    name = re.sub(r"[^A-Za-z]", "", line[12:16]).upper()
    return name[:1]


def heavy_ligand_atoms(model: list[str], ligand_id: str) -> list[str]:
    atoms = [
        line
        for line in model
        if line.startswith("HETATM") and line[17:20].strip() == ligand_id and atom_element(line) != "H"
    ]
    if not atoms:
        raise ValueError(f"No deposited heavy atoms found for ligand {ligand_id}")
    return atoms


def xyz(line: str) -> tuple[float, float, float]:
    return float(line[30:38]), float(line[38:46]), float(line[46:54])


def grid_from_ligand(model: list[str], ligand_id: str) -> dict[str, float]:
    coords = [xyz(line) for line in heavy_ligand_atoms(model, ligand_id)]
    columns = list(zip(*coords))
    center = [sum(values) / len(values) for values in columns]
    extent = [max(values) - min(values) for values in columns]
    # Eight Angstrom padding is applied on both sides of each axis.
    sizes = [value + 16.0 for value in extent]
    return {
        "center_x": round(center[0], 6),
        "center_y": round(center[1], 6),
        "center_z": round(center[2], 6),
        "size_x": round(sizes[0], 6),
        "size_y": round(sizes[1], 6),
        "size_z": round(sizes[2], 6),
    }


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else [value]


def first_scalar(value: Any, field_name: str) -> Any:
    """Return a required scalar from an mmCIF value that may be list-wrapped."""
    values = as_list(value)
    if len(values) != 1:
        raise ValueError(f"Expected one value for {field_name}, found {len(values)}")
    return values[0]


def component_metadata(path: Path) -> dict[str, Any]:
    data = MMCIF2Dict(str(path))
    types = as_list(data["_pdbx_chem_comp_descriptor.type"])
    programs = as_list(data["_pdbx_chem_comp_descriptor.program"])
    descriptors = as_list(data["_pdbx_chem_comp_descriptor.descriptor"])
    candidates = list(zip(types, programs, descriptors))
    selected = next(
        (
            descriptor
            for descriptor_type, program, descriptor in candidates
            if descriptor_type == "SMILES_CANONICAL" and program == "CACTVS"
        ),
        None,
    )
    if not selected:
        raise ValueError(f"No CACTVS canonical SMILES in component record: {path}")
    smiles_source = "RCSB chemical component CIF; SMILES_CANONICAL from CACTVS"
    if Chem.MolFromSmiles(str(selected)) is None:
        inchi = next(
            (
                descriptor
                for descriptor_type, program, descriptor in candidates
                if descriptor_type == "InChI" and program == "InChI"
            ),
            None,
        )
        molecule = Chem.MolFromInchi(str(inchi)) if inchi else None
        if molecule is None:
            raise ValueError(
                f"CACTVS SMILES failed RDKit parsing and no usable RCSB InChI was found: {path}"
            )
        selected = Chem.MolToSmiles(molecule, canonical=True)
        smiles_source = (
            "RCSB chemical component CIF; InChI converted to RDKit canonical SMILES "
            "because the deposited CACTVS SMILES failed RDKit parsing"
        )
    elements = as_list(data["_chem_comp_atom.type_symbol"])
    return {
        "component_name": str(first_scalar(data["_chem_comp.name"], "_chem_comp.name")),
        "formula": str(first_scalar(data["_chem_comp.formula"], "_chem_comp.formula")),
        "formula_weight": float(
            first_scalar(data["_chem_comp.formula_weight"], "_chem_comp.formula_weight")
        ),
        "canonical_smiles": str(selected),
        "smiles_source": smiles_source,
        "component_heavy_atom_count": int(sum(str(element).upper() != "H" for element in elements)),
    }


def write_pdb(lines: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines + ["END"]) + "\n", encoding="utf-8")


def convert_with_obabel(
    input_path: Path,
    output_path: Path,
    log_path: Path,
    *,
    partial_charge: bool,
    rigid_pdbqt: bool = False,
) -> None:
    command = ["obabel", str(input_path), "-O", str(output_path)]
    if partial_charge:
        command.extend(["--partialcharge", "gasteiger"])
    if rigid_pdbqt:
        if output_path.suffix.lower() != ".pdbqt":
            raise ValueError("The Open Babel rigid-output flag is valid only for PDBQT output.")
        command.append("-xr")
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "COMMAND: " + " ".join(command) + "\n\nSTDOUT:\n" + result.stdout + "\nSTDERR:\n" + result.stderr,
        encoding="utf-8",
    )
    if result.returncode != 0 or not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(f"Open Babel conversion failed; see {log_path}")


def main() -> int:
    if PREPARED.exists() or PROTOCOL_PATH.exists() or LOCK_PATH.exists():
        raise FileExistsError(
            "Refusing to overwrite an existing preparation or locked protocol under "
            f"{OUT}. Use a new validation directory for a new experiment."
        )
    for spec in STRUCTURE_SPECS:
        require_nonempty(STRUCTURES / f"{spec['pdb_id']}.pdb")
        require_nonempty(COMPONENTS / f"{spec['ligand_id']}.cif")

    vina_version = command_version(["vina", "--version"])
    obabel_version = command_version(["obabel", "-V"])
    PREPARED.mkdir(parents=True)
    LOGS.mkdir(parents=True)

    structure_rows: list[dict[str, Any]] = []
    prep_rows: list[dict[str, Any]] = []
    protocol_structures: dict[str, Any] = {}
    molecule_rows: list[dict[str, str]] = []

    for spec in STRUCTURE_SPECS:
        pdb_id = spec["pdb_id"]
        ligand_id = spec["ligand_id"]
        pdb_path = STRUCTURES / f"{pdb_id}.pdb"
        component_path = COMPONENTS / f"{ligand_id}.cif"
        models, method = pdb_models(pdb_path)
        metadata = component_metadata(component_path)
        model1_heavy = heavy_ligand_atoms(models[0], ligand_id)
        pdb_heavy_count = len(model1_heavy)
        if pdb_heavy_count != metadata["component_heavy_atom_count"]:
            raise ValueError(
                f"Heavy-atom count mismatch for {pdb_id}/{ligand_id}: "
                f"PDB={pdb_heavy_count}, component={metadata['component_heavy_atom_count']}"
            )
        grid = grid_from_ligand(models[0], ligand_id)
        structure_status = "prepared"
        failure_notes: list[str] = []

        for index, model in enumerate(models, start=1):
            model_dir = PREPARED / pdb_id / f"model_{index:02d}"
            receptor_lines = [
                line
                for line in model
                if line.startswith("ATOM  ")
                or (line.startswith("HETATM") and line[17:20].strip() in RNA_RESIDUES)
            ]
            ligand_lines = [
                line
                for line in model
                if line.startswith("HETATM") and line[17:20].strip() == ligand_id
            ]
            receptor_pdb = model_dir / "receptor_rna.pdb"
            receptor_pdbqt = model_dir / "receptor_rna.pdbqt"
            reference_pdb = model_dir / "reference_ligand_deposited.pdb"
            reference_sdf = model_dir / "reference_ligand_deposited.sdf"
            reference_pdbqt = model_dir / "reference_ligand_deposited.pdbqt"
            try:
                if not receptor_lines:
                    raise ValueError(f"No RNA receptor atoms found in {pdb_id} model {index}")
                if not ligand_lines:
                    raise ValueError(f"No {ligand_id} atoms found in {pdb_id} model {index}")
                write_pdb(receptor_lines, receptor_pdb)
                write_pdb(ligand_lines, reference_pdb)
                convert_with_obabel(
                    receptor_pdb,
                    receptor_pdbqt,
                    LOGS / f"{pdb_id}_model_{index:02d}_receptor_obabel.log",
                    partial_charge=True,
                    rigid_pdbqt=True,
                )
                convert_with_obabel(
                    reference_pdb,
                    reference_sdf,
                    LOGS / f"{pdb_id}_model_{index:02d}_reference_sdf_obabel.log",
                    partial_charge=False,
                )
                convert_with_obabel(
                    reference_pdb,
                    reference_pdbqt,
                    LOGS / f"{pdb_id}_model_{index:02d}_reference_pdbqt_obabel.log",
                    partial_charge=True,
                )
                status = "prepared"
                note = "Deposited ligand is reference-only and will not be docked."
            except Exception as exc:  # noqa: BLE001
                status = "failed"
                note = str(exc)
                structure_status = "failed"
                failure_notes.append(f"model {index}: {exc}")
            prep_rows.append(
                {
                    "pdb_id": pdb_id,
                    "ligand_id": ligand_id,
                    "model_index": index,
                    "receptor_pdb": str(receptor_pdb),
                    "receptor_pdbqt": str(receptor_pdbqt),
                    "reference_ligand_pdb": str(reference_pdb),
                    "reference_ligand_sdf": str(reference_sdf),
                    "reference_ligand_pdbqt": str(reference_pdbqt),
                    "preparation_status": status,
                    "notes": note,
                }
            )

        molecule_rows.append(
            {
                "molecule_id": f"{pdb_id}_{ligand_id}",
                "canonical_smiles": metadata["canonical_smiles"],
            }
        )
        structure_rows.append(
            {
                "pdb_id": pdb_id,
                "ligand_name": spec["ligand_name"],
                "ligand_component_id": ligand_id,
                "experimental_method": method,
                "ensemble_models": len(models),
                "formula": metadata["formula"],
                "formula_weight": metadata["formula_weight"],
                "heavy_atom_count": pdb_heavy_count,
                "canonical_smiles": metadata["canonical_smiles"],
                "smiles_source": metadata["smiles_source"],
                "source_pdb": str(pdb_path),
                "source_component_cif": str(component_path),
                "preparation_status": structure_status,
                "failure_notes": " | ".join(failure_notes),
            }
        )
        protocol_structures[pdb_id] = {
            "ligand_name": spec["ligand_name"],
            "ligand_component_id": ligand_id,
            "experimental_method": method,
            "ensemble_models": len(models),
            "grid_box": grid,
        }

    ligand_config = {
        "ligand_preparation": {
            "add_hydrogens": True,
            "generate_3d_conformers": True,
            "optimize_geometry": True,
            "max_conformer_attempts": 20,
            "prepare_pdbqt": True,
            "open_babel_executable": "obabel",
        }
    }
    fresh_dir = PREPARED / "ligands_from_smiles"
    fresh = prepare_ligands(pd.DataFrame(molecule_rows), fresh_dir, ligand_config)
    fresh.to_csv(OUT / "fresh_ligand_preparation_summary.tsv", sep="\t", index=False)
    failed_fresh = fresh[
        fresh["pdbqt_file"].astype(str).isin({"", "NA", "nan"})
        | ~fresh["pdbqt_file"].astype(str).map(lambda value: Path(value).is_file())
    ]
    if not failed_fresh.empty:
        failed_ids = failed_fresh["molecule_id"].astype(str).tolist()
        for row in structure_rows:
            if f"{row['pdb_id']}_{row['ligand_component_id']}" in failed_ids:
                row["preparation_status"] = "failed"
                row["failure_notes"] = (row["failure_notes"] + " | " if row["failure_notes"] else "") + "Fresh ligand PDBQT preparation failed."

    structure_table = pd.DataFrame(structure_rows)
    preparation_table = pd.DataFrame(prep_rows)
    structure_table.to_csv(OUT / "structure_metadata.tsv", sep="\t", index=False)
    preparation_table.to_csv(OUT / "model_preparation_summary.tsv", sep="\t", index=False)

    protocol = {
        "protocol_name": "SMN2-RescueDiff RNA redocking validation",
        "locked_before_docking_and_rmsd": True,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "software": {
            "vina": vina_version,
            "open_babel": obabel_version,
            "rmsd": "spyrmsd.symmrmsd with center=false and minimize=false",
        },
        "ligand_preparation": {
            "function": "smn2_rescuediff.docking.ligand_preparer.prepare_ligands",
            "fresh_conformer_seed": CONFORMER_SEED,
            "embedding": "RDKit ETKDGv3",
            "optimization": "MMFF with UFF fallback",
            "pdbqt_conversion": "Open Babel with Gasteiger partial charges",
            "smiles_selection": (
                "RCSB component SMILES_CANONICAL from CACTVS; when RDKit cannot parse "
                "that string, convert the RCSB component InChI to RDKit canonical SMILES"
            ),
            "deposited_coordinates_used_for_docking": False,
        },
        "receptor_preparation": {
            "repository_function_reviewed": "smn2_rescuediff.docking.receptor_preparer.prepare_receptor",
            "repository_function_limitation": "Validation and metadata only; it does not create receptor PDBQT files.",
            "conversion": "obabel receptor_rna.pdb -O receptor_rna.pdbqt --partialcharge gasteiger -xr",
            "rigid_output_reason": (
                "Open Babel writes ROOT/BRANCH ligand records unless -xr is used; Vina rejects "
                "those records in a rigid receptor."
            ),
            "production_receptor_audit": (
                "The existing SMN2 production receptor was not reused because it contains Y59 "
                "deposited-ligand atoms; this validation uses ligand-free RNA receptors."
            ),
            "included": "RNA polymer atoms, including PSU",
            "excluded": "Deposited ligand, solvent, monatomic ions, and non-RNA heterogens",
        },
        "grid_definition": {
            "center": "Centroid of deposited ligand heavy atoms in model 1",
            "padding_angstrom_each_side": 8.0,
            "size_formula": "heavy-atom coordinate extent + 16.0 Angstrom on each axis",
        },
        "docking": {
            "exhaustiveness": 32,
            "num_modes": 20,
            "energy_range": 10,
            "cpu_per_job": 1,
            "parallel_workers": 4,
            "seeds": VINA_SEEDS,
            "primary_receptor_model": 1,
            "ensemble_docking": "all deposited models",
            "timeout_seconds_per_job": 1800,
        },
        "analysis": {
            "rmsd_threshold_angstrom_strictly_less_than": 2.0,
            "symmetry_correction": "spyrmsd graph-isomorphism correction",
            "center_before_rmsd": False,
            "minimize_by_ligand_superposition": False,
            "receptor_alignment": "Reference ensemble RNA heavy atoms aligned to each docked receptor model before ligand RMSD",
            "contact_cutoff_angstrom": 4.0,
            "contact_identity": "RNA chain, residue number, insertion code, residue name, and receptor atom name",
            "operational_meaningful_contact_recall_minimum": 0.5,
            "operational_meaningful_contact_jaccard_minimum": 0.33,
            "headline_pose": "Lowest-affinity mode across the five model-1 seed runs",
            "seed_success": "Each seed's rank-1 model-1 mode has RMSD <2.0 Angstrom to at least one aligned deposited ensemble model",
        },
        "structures": protocol_structures,
    }
    PROTOCOL_PATH.write_text(yaml.safe_dump(protocol, sort_keys=False), encoding="utf-8")
    protocol_hash = sha256(PROTOCOL_PATH)
    lock = {
        "protocol_path": str(PROTOCOL_PATH),
        "protocol_sha256": protocol_hash,
        "locked_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": protocol["git_commit"],
        "note": "Any parameter change requires a new output directory and must be reported as a later experiment.",
    }
    LOCK_PATH.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")

    input_files = [
        *(STRUCTURES / f"{spec['pdb_id']}.pdb" for spec in STRUCTURE_SPECS),
        *(COMPONENTS / f"{spec['ligand_id']}.cif" for spec in STRUCTURE_SPECS),
        Path(__file__).resolve(),
        ROOT / "src/smn2_rescuediff/docking/ligand_preparer.py",
        ROOT / "src/smn2_rescuediff/docking/receptor_preparer.py",
    ]
    hash_rows = [{"path": str(path), "sha256": sha256(path)} for path in input_files]
    pd.DataFrame(hash_rows).to_csv(OUT / "preparation_input_sha256.tsv", sep="\t", index=False)

    summary = {
        "status": "completed_with_failures" if (structure_table["preparation_status"] != "prepared").any() else "completed",
        "structures_attempted": int(len(structure_table)),
        "structures_prepared": int((structure_table["preparation_status"] == "prepared").sum()),
        "models_attempted": int(len(preparation_table)),
        "models_prepared": int((preparation_table["preparation_status"] == "prepared").sum()),
        "protocol_sha256": protocol_hash,
        "git_commit": protocol["git_commit"],
        "vina_version": vina_version,
        "open_babel_version": obabel_version,
    }
    (OUT / "preparation_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
