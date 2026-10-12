#!/usr/bin/env python3
"""Analyze locked RNA-ligand redocking results without ligand superposition."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import yaml
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from rdkit import Chem
from scipy.optimize import linear_sum_assignment
from spyrmsd import rmsd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/processed/redocking_validation"
DOCKING = BASE / "docking"
ANALYSIS = BASE / "analysis"
PROTOCOL_PATH = BASE / "redocking_protocol.yaml"
LOCK_PATH = BASE / "protocol_lock.json"

AD4_ATOMIC_NUMBERS = {
    "H": 1,
    "HD": 1,
    "C": 6,
    "A": 6,
    "N": 7,
    "NA": 7,
    "O": 8,
    "OA": 8,
    "F": 9,
    "P": 15,
    "S": 16,
    "SA": 16,
    "CL": 17,
    "BR": 35,
    "I": 53,
}

STRUCTURES = {
    "8R62": {"ligand_id": "Y59", "molecule_id": "8R62_Y59"},
    "8R8P": {"ligand_id": "YB3", "molecule_id": "8R8P_YB3"},
    "5KX9": {"ligand_id": "6YG", "molecule_id": "5KX9_6YG"},
    "8R63": {"ligand_id": "Y53", "molecule_id": "8R63_Y53"},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else [value]


def validate_inputs() -> tuple[dict[str, Any], str, dict[str, Any]]:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    protocol_hash = sha256(PROTOCOL_PATH)
    if protocol_hash != lock.get("protocol_sha256"):
        raise RuntimeError("Protocol hash differs from the pre-docking lock; refusing analysis.")
    protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    run = json.loads((DOCKING / "run_metadata.json").read_text(encoding="utf-8"))
    if run.get("protocol_sha256") != protocol_hash:
        raise RuntimeError("Docking run used a different protocol hash.")
    if run.get("failed_jobs") != 0 or run.get("completed_jobs") != run.get("total_jobs"):
        raise RuntimeError(f"Docking matrix is incomplete: {run}")
    if not run.get("all_scores_are_real"):
        raise RuntimeError("Docking metadata does not certify all scores as real Vina scores.")
    if ANALYSIS.exists():
        raise FileExistsError(f"Refusing to overwrite existing analysis: {ANALYSIS}")
    return protocol, protocol_hash, run


def pdb_element(line: str) -> str:
    element = line[76:78].strip().upper()
    if element:
        return element
    name = "".join(character for character in line[12:16] if character.isalpha()).upper()
    if name.startswith("CL"):
        return "CL"
    if name.startswith("BR"):
        return "BR"
    return name[:1]


def pdb_xyz(line: str) -> np.ndarray:
    return np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])


def receptor_atoms(path: Path) -> dict[tuple[str, int, str, str, str], np.ndarray]:
    atoms: dict[tuple[str, int, str, str, str], np.ndarray] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")) or pdb_element(line) == "H":
            continue
        identity = (
            line[21].strip(),
            int(line[22:26]),
            line[26].strip(),
            line[17:20].strip(),
            line[12:16].strip(),
        )
        if identity in atoms:
            raise ValueError(f"Duplicate receptor atom identity in {path}: {identity}")
        atoms[identity] = pdb_xyz(line)
    if not atoms:
        raise ValueError(f"No receptor heavy atoms parsed from {path}")
    return atoms


def kabsch_transform(
    moving_atoms: dict[tuple[str, int, str, str, str], np.ndarray],
    target_atoms: dict[tuple[str, int, str, str, str], np.ndarray],
) -> tuple[np.ndarray, np.ndarray, int, float]:
    common = sorted(set(moving_atoms) & set(target_atoms))
    if len(common) < 3:
        raise ValueError("Fewer than three common RNA heavy atoms are available for alignment.")
    moving = np.vstack([moving_atoms[key] for key in common])
    target = np.vstack([target_atoms[key] for key in common])
    moving_center = moving.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (moving - moving_center).T @ (target - target_center)
    left, _, right_t = np.linalg.svd(covariance)
    rotation = left @ right_t
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right_t
    translation = target_center - moving_center @ rotation
    aligned = moving @ rotation + translation
    alignment_rmsd = float(np.sqrt(np.mean(np.sum((aligned - target) ** 2, axis=1))))
    return rotation, translation, len(common), alignment_rmsd


def component_graph(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    data = MMCIF2Dict(str(path))
    atom_ids = [str(value) for value in as_list(data["_chem_comp_atom.atom_id"])]
    elements = [str(value).upper() for value in as_list(data["_chem_comp_atom.type_symbol"])]
    heavy_ids = [atom_id for atom_id, element in zip(atom_ids, elements) if element != "H"]
    heavy_elements = [element for element in elements if element != "H"]
    index = {atom_id: position for position, atom_id in enumerate(heavy_ids)}
    adjacency = np.zeros((len(heavy_ids), len(heavy_ids)), dtype=int)
    for atom1, atom2 in zip(
        as_list(data["_chem_comp_bond.atom_id_1"]),
        as_list(data["_chem_comp_bond.atom_id_2"]),
    ):
        if atom1 in index and atom2 in index:
            first, second = index[atom1], index[atom2]
            adjacency[first, second] = adjacency[second, first] = 1
    periodic_table = Chem.GetPeriodicTable()
    atomic_numbers = np.array([periodic_table.GetAtomicNumber(element) for element in heavy_elements])
    return heavy_ids, atomic_numbers, adjacency


def reference_ligand_coords(path: Path, heavy_atom_ids: list[str]) -> np.ndarray:
    by_name: dict[str, np.ndarray] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(("ATOM  ", "HETATM")) and pdb_element(line) != "H":
            by_name[line[12:16].strip()] = pdb_xyz(line)
    missing = set(heavy_atom_ids) - set(by_name)
    if missing:
        raise ValueError(f"Deposited ligand is missing component atoms in {path}: {sorted(missing)}")
    return np.vstack([by_name[atom_id] for atom_id in heavy_atom_ids])


def pdbqt_atoms(path: Path) -> tuple[np.ndarray, np.ndarray, list[int]]:
    coordinates: list[np.ndarray] = []
    atomic_numbers: list[int] = []
    heavy_positions: list[int] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        atom_type = line.split()[-1].upper()
        if atom_type not in AD4_ATOMIC_NUMBERS:
            raise ValueError(f"Unsupported AutoDock atom type {atom_type} in {path}")
        coordinates.append(pdb_xyz(line))
        atomic_numbers.append(AD4_ATOMIC_NUMBERS[atom_type])
        if atomic_numbers[-1] > 1:
            heavy_positions.append(len(coordinates) - 1)
    return np.vstack(coordinates), np.array(atomic_numbers), heavy_positions


def fresh_ligand_graph(sdf_path: Path, pdbqt_path: Path) -> tuple[np.ndarray, np.ndarray, list[int]]:
    molecule = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=True)[0]
    if molecule is None:
        raise ValueError(f"RDKit could not read prepared ligand SDF: {sdf_path}")
    conformer = molecule.GetConformer()
    sdf_indices = [atom.GetIdx() for atom in molecule.GetAtoms() if atom.GetAtomicNum() > 1]
    sdf_coordinates = np.array(
        [
            [
                conformer.GetAtomPosition(index).x,
                conformer.GetAtomPosition(index).y,
                conformer.GetAtomPosition(index).z,
            ]
            for index in sdf_indices
        ]
    )
    sdf_numbers = np.array([molecule.GetAtomWithIdx(index).GetAtomicNum() for index in sdf_indices])
    pdbqt_coordinates, pdbqt_numbers_all, heavy_positions = pdbqt_atoms(pdbqt_path)
    pdbqt_coordinates = pdbqt_coordinates[heavy_positions]
    pdbqt_numbers = pdbqt_numbers_all[heavy_positions]
    distances = np.linalg.norm(
        pdbqt_coordinates[:, np.newaxis, :] - sdf_coordinates[np.newaxis, :, :], axis=2
    )
    distances[pdbqt_numbers[:, np.newaxis] != sdf_numbers[np.newaxis, :]] = 1e6
    pdbqt_order, sdf_order = linear_sum_assignment(distances)
    if len(pdbqt_order) != len(sdf_indices) or float(distances[pdbqt_order, sdf_order].max()) > 0.01:
        raise ValueError(f"Could not map PDBQT atoms to SDF atoms for {sdf_path}")
    sdf_for_pdbqt = np.empty(len(sdf_indices), dtype=int)
    sdf_for_pdbqt[pdbqt_order] = np.array(sdf_indices)[sdf_order]
    adjacency = Chem.GetAdjacencyMatrix(molecule)[np.ix_(sdf_for_pdbqt, sdf_for_pdbqt)].astype(int)
    numbers = np.array([molecule.GetAtomWithIdx(int(index)).GetAtomicNum() for index in sdf_for_pdbqt])
    return numbers, adjacency, heavy_positions


def graphs_are_isomorphic(
    numbers1: np.ndarray, adjacency1: np.ndarray, numbers2: np.ndarray, adjacency2: np.ndarray
) -> bool:
    def graph(numbers: np.ndarray, adjacency: np.ndarray) -> nx.Graph:
        result = nx.from_numpy_array(adjacency)
        nx.set_node_attributes(result, {i: int(value) for i, value in enumerate(numbers)}, "element")
        return result

    return nx.is_isomorphic(
        graph(numbers1, adjacency1),
        graph(numbers2, adjacency2),
        node_match=lambda first, second: first["element"] == second["element"],
    )


def read_pose_models(path: Path, heavy_positions: list[int]) -> dict[int, np.ndarray]:
    models: dict[int, np.ndarray] = {}
    current_mode: int | None = None
    atom_coordinates: list[np.ndarray] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("MODEL"):
            current_mode = int(line.split()[1])
            atom_coordinates = []
        elif line.startswith(("ATOM  ", "HETATM")) and current_mode is not None:
            atom_coordinates.append(pdb_xyz(line))
        elif line.startswith("ENDMDL") and current_mode is not None:
            if max(heavy_positions) >= len(atom_coordinates):
                raise ValueError(f"Pose atom count differs from ligand PDBQT in {path}")
            models[current_mode] = np.vstack(atom_coordinates)[heavy_positions]
            current_mode = None
    if not models:
        raise ValueError(f"No pose models parsed from {path}")
    return models


def contact_set(
    receptor: dict[tuple[str, int, str, str, str], np.ndarray], ligand: np.ndarray, cutoff: float
) -> set[tuple[str, int, str, str, str]]:
    identities = list(receptor)
    receptor_coordinates = np.vstack([receptor[identity] for identity in identities])
    distances = np.linalg.norm(
        receptor_coordinates[:, np.newaxis, :] - ligand[np.newaxis, :, :], axis=2
    )
    return {identities[index] for index in np.where((distances <= cutoff).any(axis=1))[0]}


def contact_label(identity: tuple[str, int, str, str, str]) -> str:
    chain, residue_number, insertion_code, residue_name, atom_name = identity
    residue = f"{residue_number}{insertion_code}" if insertion_code else str(residue_number)
    return f"{chain}:{residue_name}{residue}:{atom_name}"


def main() -> int:
    protocol, protocol_hash, docking_run = validate_inputs()
    ANALYSIS.mkdir(parents=True, exist_ok=False)
    scores = pd.read_csv(DOCKING / "redocking_mode_scores.tsv", sep="\t")
    if scores.empty or scores["is_dry_run_score"].astype(str).str.lower().isin({"true", "1"}).any():
        raise RuntimeError("Missing scores or dry-run scores detected.")

    rmsd_rows: list[dict[str, Any]] = []
    pose_rows: list[dict[str, Any]] = []
    alignment_rows: list[dict[str, Any]] = []
    pairwise_rows: list[dict[str, Any]] = []
    contact_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    threshold = float(protocol["analysis"]["rmsd_threshold_angstrom_strictly_less_than"])
    contact_cutoff = float(protocol["analysis"]["contact_cutoff_angstrom"])
    recall_minimum = float(protocol["analysis"]["operational_meaningful_contact_recall_minimum"])
    jaccard_minimum = float(protocol["analysis"]["operational_meaningful_contact_jaccard_minimum"])

    for pdb_id, spec in STRUCTURES.items():
        ligand_id = spec["ligand_id"]
        molecule_id = spec["molecule_id"]
        model_count = int(protocol["structures"][pdb_id]["ensemble_models"])
        component_path = BASE / "components" / f"{ligand_id}.cif"
        heavy_ids, reference_numbers, reference_adjacency = component_graph(component_path)
        fresh_base = BASE / "prepared/ligands_from_smiles" / molecule_id
        fresh_numbers, fresh_adjacency, heavy_positions = fresh_ligand_graph(
            fresh_base.with_suffix(".sdf"), fresh_base.with_suffix(".pdbqt")
        )
        if not graphs_are_isomorphic(
            reference_numbers, reference_adjacency, fresh_numbers, fresh_adjacency
        ):
            raise ValueError(f"Fresh and deposited ligand graphs are not isomorphic for {pdb_id}")

        receptors = {
            model: receptor_atoms(
                BASE / "prepared" / pdb_id / f"model_{model:02d}" / "receptor_rna.pdb"
            )
            for model in range(1, model_count + 1)
        }
        reference_ligands = {
            model: reference_ligand_coords(
                BASE
                / "prepared"
                / pdb_id
                / f"model_{model:02d}"
                / "reference_ligand_deposited.pdb",
                heavy_ids,
            )
            for model in range(1, model_count + 1)
        }

        aligned_references_by_target: dict[int, dict[int, np.ndarray]] = {}
        for target_model in range(1, model_count + 1):
            aligned_references_by_target[target_model] = {}
            for reference_model in range(1, model_count + 1):
                rotation, translation, common_atoms, alignment_rmsd = kabsch_transform(
                    receptors[reference_model], receptors[target_model]
                )
                aligned_references_by_target[target_model][reference_model] = (
                    reference_ligands[reference_model] @ rotation + translation
                )
                alignment_rows.append(
                    {
                        "pdb_id": pdb_id,
                        "target_receptor_model": target_model,
                        "reference_ensemble_model": reference_model,
                        "common_rna_heavy_atoms": common_atoms,
                        "rna_alignment_rmsd_angstrom": alignment_rmsd,
                    }
                )

        model1_references = aligned_references_by_target[1]
        for first, second in combinations(range(1, model_count + 1), 2):
            value = float(
                rmsd.symmrmsd(
                    model1_references[first],
                    model1_references[second],
                    reference_numbers,
                    reference_numbers,
                    reference_adjacency,
                    reference_adjacency,
                    center=False,
                    minimize=False,
                )
            )
            pairwise_rows.append(
                {
                    "pdb_id": pdb_id,
                    "ensemble_model_1": first,
                    "ensemble_model_2": second,
                    "symmetry_corrected_ligand_rmsd_angstrom": value,
                }
            )

        structure_scores = scores[scores["pdb_id"] == pdb_id].copy()
        pose_cache: dict[tuple[int, int], dict[int, np.ndarray]] = {}
        for (model_index, seed), group in structure_scores.groupby(["model_index", "seed"]):
            model_index, seed = int(model_index), int(seed)
            pose_file = Path(group.iloc[0]["pose_file"])
            pose_cache[(model_index, seed)] = read_pose_models(pose_file, heavy_positions)
            aligned_references = aligned_references_by_target[model_index]
            reference_models = sorted(aligned_references)
            reference_coordinate_list = [aligned_references[model] for model in reference_models]
            for score in group.to_dict(orient="records"):
                mode = int(score["mode"])
                pose = pose_cache[(model_index, seed)][mode]
                values = rmsd.symmrmsd(
                    pose,
                    reference_coordinate_list,
                    fresh_numbers,
                    reference_numbers,
                    fresh_adjacency,
                    reference_adjacency,
                    center=False,
                    minimize=False,
                )
                if np.isscalar(values):
                    values = [float(values)]
                values = [float(value) for value in values]
                for reference_model, value in zip(reference_models, values):
                    rmsd_rows.append(
                        {
                            "pdb_id": pdb_id,
                            "ligand_name": score["ligand_name"],
                            "target_receptor_model": model_index,
                            "seed": seed,
                            "mode": mode,
                            "affinity_kcal_mol": float(score["affinity_kcal_mol"]),
                            "reference_ensemble_model": reference_model,
                            "symmetry_corrected_heavy_atom_rmsd_angstrom": value,
                            "recovered_below_2A": value < threshold,
                            "pose_file": str(pose_file),
                        }
                    )
                best_index = int(np.argmin(values))
                pose_rows.append(
                    {
                        "pdb_id": pdb_id,
                        "ligand_name": score["ligand_name"],
                        "target_receptor_model": model_index,
                        "seed": seed,
                        "mode": mode,
                        "affinity_kcal_mol": float(score["affinity_kcal_mol"]),
                        "minimum_ensemble_rmsd_angstrom": values[best_index],
                        "closest_reference_ensemble_model": reference_models[best_index],
                        "recovered_below_2A": values[best_index] < threshold,
                        "pose_file": str(pose_file),
                    }
                )

        structure_poses = pd.DataFrame([row for row in pose_rows if row["pdb_id"] == pdb_id])
        structure_rmsd = pd.DataFrame([row for row in rmsd_rows if row["pdb_id"] == pdb_id])
        primary = structure_poses[structure_poses["target_receptor_model"] == 1].copy()
        rank1 = primary[primary["mode"] == 1].copy()
        per_seed_any = primary.groupby("seed", as_index=False)["minimum_ensemble_rmsd_angstrom"].min()
        headline = primary.sort_values(
            ["affinity_kcal_mol", "seed", "mode"], ascending=[True, True, True]
        ).iloc[0]
        headline_model1_rmsd = float(
            structure_rmsd[
                (structure_rmsd["target_receptor_model"] == 1)
                & (structure_rmsd["seed"] == int(headline["seed"]))
                & (structure_rmsd["mode"] == int(headline["mode"]))
                & (structure_rmsd["reference_ensemble_model"] == 1)
            ]["symmetry_corrected_heavy_atom_rmsd_angstrom"].iloc[0]
        )
        best_primary = primary.sort_values(
            ["minimum_ensemble_rmsd_angstrom", "seed", "mode"], ascending=[True, True, True]
        ).iloc[0]
        headline_pose = pose_cache[(1, int(headline["seed"]))][int(headline["mode"])]
        native_contacts = contact_set(receptors[1], reference_ligands[1], contact_cutoff)
        predicted_contacts = contact_set(receptors[1], headline_pose, contact_cutoff)
        intersection = native_contacts & predicted_contacts
        union = native_contacts | predicted_contacts
        recall = len(intersection) / len(native_contacts) if native_contacts else float("nan")
        precision = len(intersection) / len(predicted_contacts) if predicted_contacts else float("nan")
        jaccard = len(intersection) / len(union) if union else float("nan")
        for identity in sorted(union):
            contact_rows.append(
                {
                    "pdb_id": pdb_id,
                    "contact_atom": contact_label(identity),
                    "in_deposited_model1_contact_set": identity in native_contacts,
                    "in_headline_predicted_contact_set": identity in predicted_contacts,
                    "shared_contact": identity in intersection,
                }
            )

        ensemble_best = structure_poses.groupby("target_receptor_model")[
            "minimum_ensemble_rmsd_angstrom"
        ].min()
        pairwise_values = [
            row["symmetry_corrected_ligand_rmsd_angstrom"]
            for row in pairwise_rows
            if row["pdb_id"] == pdb_id
        ]
        summary_rows.append(
            {
                "pdb_id": pdb_id,
                "ligand_name": protocol["structures"][pdb_id]["ligand_name"],
                "experimental_method": protocol["structures"][pdb_id]["experimental_method"],
                "ensemble_models": model_count,
                "primary_model1_seeds": int(rank1["seed"].nunique()),
                "rank1_seed_successes_below_2A": int(
                    (rank1["minimum_ensemble_rmsd_angstrom"] < threshold).sum()
                ),
                "rank1_seed_success_fraction_below_2A": float(
                    (rank1["minimum_ensemble_rmsd_angstrom"] < threshold).mean()
                ),
                "any_mode_seed_successes_below_2A": int(
                    (per_seed_any["minimum_ensemble_rmsd_angstrom"] < threshold).sum()
                ),
                "best_primary_model1_rmsd_angstrom": float(
                    primary["minimum_ensemble_rmsd_angstrom"].min()
                ),
                "best_rank1_primary_model1_rmsd_angstrom": float(
                    rank1["minimum_ensemble_rmsd_angstrom"].min()
                ),
                "headline_seed": int(headline["seed"]),
                "headline_mode": int(headline["mode"]),
                "headline_affinity_kcal_mol": float(headline["affinity_kcal_mol"]),
                "headline_model1_rmsd_angstrom": headline_model1_rmsd,
                "headline_minimum_ensemble_rmsd_angstrom": float(
                    headline["minimum_ensemble_rmsd_angstrom"]
                ),
                "headline_closest_reference_model": int(
                    headline["closest_reference_ensemble_model"]
                ),
                "headline_within_2A_of_any_deposited_model": bool(
                    headline["minimum_ensemble_rmsd_angstrom"] < threshold
                ),
                "best_all_modes_seed": int(best_primary["seed"]),
                "best_all_modes_rank": int(best_primary["mode"]),
                "best_all_modes_minimum_ensemble_rmsd_angstrom": float(
                    best_primary["minimum_ensemble_rmsd_angstrom"]
                ),
                "best_all_modes_closest_reference_model": int(
                    best_primary["closest_reference_ensemble_model"]
                ),
                "receptor_models_with_any_pose_below_2A": int((ensemble_best < threshold).sum()),
                "native_model1_contact_atoms": len(native_contacts),
                "headline_predicted_contact_atoms": len(predicted_contacts),
                "recovered_native_contact_atoms": len(intersection),
                "shared_contact_atoms": len(intersection),
                "contact_recall": recall,
                "contact_precision": precision,
                "contact_jaccard": jaccard,
                "operational_meaningful_contact_recovery": bool(
                    recall >= recall_minimum and jaccard >= jaccard_minimum
                ),
                "ensemble_pairwise_ligand_rmsd_min_angstrom": (
                    float(min(pairwise_values)) if pairwise_values else np.nan
                ),
                "ensemble_pairwise_ligand_rmsd_median_angstrom": (
                    float(np.median(pairwise_values)) if pairwise_values else np.nan
                ),
                "ensemble_pairwise_ligand_rmsd_max_angstrom": (
                    float(max(pairwise_values)) if pairwise_values else np.nan
                ),
            }
        )

    pd.DataFrame(rmsd_rows).to_csv(ANALYSIS / "pose_to_ensemble_rmsd_long.tsv", sep="\t", index=False)
    pd.DataFrame(pose_rows).to_csv(ANALYSIS / "pose_recovery_summary.tsv", sep="\t", index=False)
    pd.DataFrame(alignment_rows).to_csv(ANALYSIS / "rna_alignment_qc.tsv", sep="\t", index=False)
    pd.DataFrame(pairwise_rows).to_csv(
        ANALYSIS / "deposited_ensemble_pairwise_ligand_rmsd.tsv", sep="\t", index=False
    )
    pd.DataFrame(contact_rows).to_csv(ANALYSIS / "model1_contact_comparison.tsv", sep="\t", index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(ANALYSIS / "redocking_validation_summary.tsv", sep="\t", index=False)
    metadata = {
        "status": "completed",
        "protocol_sha256": protocol_hash,
        "docking_run_metadata": docking_run,
        "structures_analyzed": len(summary_rows),
        "pose_rows": len(pose_rows),
        "pose_to_reference_rmsd_rows": len(rmsd_rows),
        "all_rmsd_values_symmetry_corrected": True,
        "ligand_superposition_used": False,
        "rna_alignment_used": True,
        "contact_cutoff_angstrom": contact_cutoff,
        "software_versions": {
            "spyrmsd": importlib.metadata.version("spyrmsd"),
            "rdkit": importlib.metadata.version("rdkit"),
            "numpy": importlib.metadata.version("numpy"),
            "scipy": importlib.metadata.version("scipy"),
            "networkx": importlib.metadata.version("networkx"),
        },
    }
    (ANALYSIS / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
