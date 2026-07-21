"""Parse simple docking pose/receptor files for Module 5 contact proxies."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd


POSE_PARSE_COLUMNS = [
    "molecule_id",
    "pose_file",
    "receptor_file",
    "ligand_atom_count",
    "receptor_atom_count",
    "ligand_center_x",
    "ligand_center_y",
    "ligand_center_z",
    "nearest_receptor_atom_distance",
    "nearest_receptor_residue_or_base",
    "pose_parse_status",
    "notes",
]


def _to_float(text: str) -> float | None:
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _parse_pdb_like(path: Path) -> list[dict[str, Any]]:
    atoms: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        x = _to_float(line[30:38].strip())
        y = _to_float(line[38:46].strip())
        z = _to_float(line[46:54].strip())
        if x is None or y is None or z is None:
            parts = line.split()
            coords = [_to_float(item) for item in parts if _to_float(item) is not None]
            if len(coords) < 3:
                continue
            x, y, z = coords[-3:]
        atoms.append(
            {
                "x": float(x),
                "y": float(y),
                "z": float(z),
                "atom_name": line[12:16].strip() or "NA",
                "residue": line[17:20].strip() or "NA",
                "record": line[:6].strip(),
            }
        )
    return atoms


def _parse_sdf(path: Path) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if len(lines) < 4:
        return []
    try:
        atom_count = int(lines[3][0:3])
    except ValueError:
        parts = lines[3].split()
        atom_count = int(parts[0]) if parts and parts[0].isdigit() else 0
    atoms: list[dict[str, Any]] = []
    for line in lines[4 : 4 + atom_count]:
        parts = line.split()
        if len(parts) < 4:
            continue
        x, y, z = _to_float(parts[0]), _to_float(parts[1]), _to_float(parts[2])
        if x is None or y is None or z is None:
            continue
        atoms.append({"x": x, "y": y, "z": z, "atom_name": parts[3], "residue": "LIG", "record": "SDF"})
    return atoms


def _parse_mol2(path: Path) -> list[dict[str, Any]]:
    atoms: list[dict[str, Any]] = []
    in_atoms = False
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("@<TRIPOS>ATOM"):
            in_atoms = True
            continue
        if line.startswith("@<TRIPOS>") and in_atoms:
            break
        if not in_atoms:
            continue
        parts = line.split()
        if len(parts) < 6:
            continue
        x, y, z = _to_float(parts[2]), _to_float(parts[3]), _to_float(parts[4])
        if x is None or y is None or z is None:
            continue
        atoms.append({"x": x, "y": y, "z": z, "atom_name": parts[1], "residue": parts[5], "record": "MOL2"})
    return atoms


def parse_atom_file(path: str | Path) -> list[dict[str, Any]]:
    """Parse PDB/PDBQT/SDF/MOL2 atom coordinates."""
    p = Path(str(path))
    if not p.exists() or str(path).lower() in {"na", "nan", ""}:
        return []
    suffix = p.suffix.lower()
    if suffix in {".pdb", ".pdbqt"}:
        return _parse_pdb_like(p)
    if suffix == ".sdf":
        return _parse_sdf(p)
    if suffix == ".mol2":
        return _parse_mol2(p)
    return []


def _center(atoms: list[dict[str, Any]]) -> tuple[Any, Any, Any]:
    if not atoms:
        return "NA", "NA", "NA"
    return (
        sum(atom["x"] for atom in atoms) / len(atoms),
        sum(atom["y"] for atom in atoms) / len(atoms),
        sum(atom["z"] for atom in atoms) / len(atoms),
    )


def _nearest(ligand_atoms: list[dict[str, Any]], receptor_atoms: list[dict[str, Any]]) -> tuple[Any, Any]:
    if not ligand_atoms or not receptor_atoms:
        return "NA", "NA"
    best = float("inf")
    residue = "NA"
    for ligand in ligand_atoms:
        for receptor in receptor_atoms:
            distance = math.dist((ligand["x"], ligand["y"], ligand["z"]), (receptor["x"], receptor["y"], receptor["z"]))
            if distance < best:
                best = distance
                residue = receptor.get("residue", "NA")
    return round(best, 4), residue


def parse_pose_for_candidate(row: pd.Series, allow_missing: bool = True) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Parse one candidate pose/receptor pair."""
    molecule_id = str(row.get("molecule_id", "NA"))
    pose_file = str(row.get("pose_file", row.get("ligand_file", "NA")))
    receptor_file = str(row.get("receptor_file", "NA"))
    if pose_file.lower() in {"na", "nan", ""} or not Path(pose_file).exists():
        status = "missing_pose_feature_based_fallback" if allow_missing else "missing_pose"
        return (
            {
                "molecule_id": molecule_id,
                "pose_file": pose_file if pose_file else "NA",
                "receptor_file": receptor_file if receptor_file else "NA",
                "ligand_atom_count": 0,
                "receptor_atom_count": 0,
                "ligand_center_x": "NA",
                "ligand_center_y": "NA",
                "ligand_center_z": "NA",
                "nearest_receptor_atom_distance": "NA",
                "nearest_receptor_residue_or_base": "NA",
                "pose_parse_status": status,
                "notes": "Pose file missing; using feature-based splicing-rescue fallback.",
            },
            [],
            [],
        )
    ligand_atoms = parse_atom_file(pose_file)
    receptor_atoms = parse_atom_file(receptor_file)
    center_x, center_y, center_z = _center(ligand_atoms)
    nearest_distance, nearest_residue = _nearest(ligand_atoms, receptor_atoms)
    status = "parsed_pose" if ligand_atoms else "pose_parse_failed_feature_based_fallback"
    notes = "Pose parsed for proxy contact analysis." if ligand_atoms else "Pose file could not be parsed; using feature-based fallback."
    return (
        {
            "molecule_id": molecule_id,
            "pose_file": pose_file,
            "receptor_file": receptor_file,
            "ligand_atom_count": len(ligand_atoms),
            "receptor_atom_count": len(receptor_atoms),
            "ligand_center_x": round(center_x, 4) if isinstance(center_x, float) else center_x,
            "ligand_center_y": round(center_y, 4) if isinstance(center_y, float) else center_y,
            "ligand_center_z": round(center_z, 4) if isinstance(center_z, float) else center_z,
            "nearest_receptor_atom_distance": nearest_distance,
            "nearest_receptor_residue_or_base": nearest_residue,
            "pose_parse_status": status,
            "notes": notes,
        },
        ligand_atoms,
        receptor_atoms,
    )


def parse_candidate_poses(candidates: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]]]:
    """Parse pose files for all candidates."""
    allow_missing = bool(config.get("pose_analysis", {}).get("allow_missing_pose_files", True))
    rows: list[dict[str, Any]] = []
    atom_cache: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}
    for _, row in candidates.iterrows():
        parsed, ligand_atoms, receptor_atoms = parse_pose_for_candidate(row, allow_missing=allow_missing)
        rows.append(parsed)
        atom_cache[str(row.get("molecule_id", "NA"))] = (ligand_atoms, receptor_atoms)
    return pd.DataFrame(rows, columns=POSE_PARSE_COLUMNS), atom_cache

