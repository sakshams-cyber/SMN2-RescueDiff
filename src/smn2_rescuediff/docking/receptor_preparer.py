"""Receptor validation and metadata creation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SUPPORTED_FORMATS = {".pdb", ".cif", ".mmcif", ".pdbqt"}


def validate_docking_box(config: dict[str, Any]) -> dict[str, float]:
    """Validate docking box fields."""
    box = config.get("receptor", {}).get("docking_box", {})
    required = ["center_x", "center_y", "center_z", "size_x", "size_y", "size_z"]
    missing = [field for field in required if field not in box]
    if missing:
        raise ValueError(f"Docking box is missing required fields: {missing}")
    return {field: float(box[field]) for field in required}


def read_pdbqt_atom_coordinates(receptor_file: str | Path) -> np.ndarray:
    """Read ATOM/HETATM coordinates from a receptor PDBQT file."""
    path = Path(receptor_file)
    if path.suffix.lower() != ".pdbqt" or not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Decoy receptor must be a non-empty PDBQT file: {path}")
    coordinates: list[list[float]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        try:
            coordinates.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        except ValueError as exc:
            raise ValueError(f"Invalid atom coordinate in receptor PDBQT {path}: {line[:54]}") from exc
    if not coordinates:
        raise ValueError(f"No ATOM/HETATM coordinates were found in receptor PDBQT: {path}")
    return np.asarray(coordinates, dtype=float)


def receptor_atoms_in_box_fraction(coordinates: np.ndarray, box: dict[str, float]) -> float:
    """Return the fraction of receptor atoms contained in an axis-aligned Vina box."""
    center = np.asarray([box["center_x"], box["center_y"], box["center_z"]], dtype=float)
    size = np.asarray([box["size_x"], box["size_y"], box["size_z"]], dtype=float)
    if np.any(size <= 0):
        raise ValueError("Docking-box dimensions must be positive.")
    lower = center - size / 2.0
    upper = center + size / 2.0
    return float(np.mean(np.all((coordinates >= lower) & (coordinates <= upper), axis=1)))


def receptor_bounds_box(
    receptor_file: str | Path,
    *,
    padding_angstrom: float = 6.0,
    minimum_size_angstrom: float = 20.0,
    maximum_size_angstrom: float = 126.0,
) -> tuple[dict[str, float], np.ndarray]:
    """Build a receptor-specific blind-docking box from PDBQT coordinate bounds."""
    coordinates = read_pdbqt_atom_coordinates(receptor_file)
    if padding_angstrom < 0 or minimum_size_angstrom <= 0 or maximum_size_angstrom <= 0:
        raise ValueError("Decoy-box padding and size limits must be non-negative and positive, respectively.")
    lower = coordinates.min(axis=0)
    upper = coordinates.max(axis=0)
    center = (lower + upper) / 2.0
    size = np.maximum((upper - lower) + 2.0 * padding_angstrom, minimum_size_angstrom)
    if np.any(size > maximum_size_angstrom):
        raise ValueError(
            f"Receptor-specific box for {receptor_file} exceeds the configured maximum size "
            f"({size.tolist()} > {maximum_size_angstrom} A)."
        )
    box = {
        "center_x": float(center[0]),
        "center_y": float(center[1]),
        "center_z": float(center[2]),
        "size_x": float(size[0]),
        "size_y": float(size[1]),
        "size_z": float(size[2]),
    }
    return box, coordinates


def resolve_decoy_docking_box(
    config: dict[str, Any], receptor_file: str | Path
) -> tuple[dict[str, float], dict[str, Any]]:
    """Resolve and validate the box used for one decoy receptor."""
    panel = config.get("decoy_panel", {})
    mode = str(panel.get("docking_box_mode", "fixed_target_box")).strip().lower()
    if mode == "fixed_target_box":
        box = validate_docking_box(config)
        coordinates = read_pdbqt_atom_coordinates(receptor_file)
    elif mode == "per_receptor_bounds":
        box, coordinates = receptor_bounds_box(
            receptor_file,
            padding_angstrom=float(panel.get("docking_box_padding_angstrom", 6.0)),
            minimum_size_angstrom=float(panel.get("docking_box_min_size_angstrom", 20.0)),
            maximum_size_angstrom=float(panel.get("docking_box_max_size_angstrom", 126.0)),
        )
    else:
        raise ValueError(
            "decoy_panel.docking_box_mode must be fixed_target_box or per_receptor_bounds; "
            f"received {mode!r}."
        )
    coverage = receptor_atoms_in_box_fraction(coordinates, box)
    minimum_coverage = float(panel.get("minimum_receptor_atom_coverage", 0.95))
    if coverage + 1e-12 < minimum_coverage:
        raise ValueError(
            f"Docking box covers only {coverage:.3f} of receptor atoms for {receptor_file}; "
            f"configured minimum is {minimum_coverage:.3f}. Recenter or re-box this receptor before docking."
        )
    qc = {
        "receptor_file": str(receptor_file),
        "docking_box_mode": mode,
        "atom_count": int(len(coordinates)),
        "receptor_atoms_in_box_fraction": coverage,
        **box,
    }
    return box, qc


def config_with_docking_box(config: dict[str, Any], box: dict[str, float]) -> dict[str, Any]:
    """Return a shallow config copy containing a job-specific receptor box."""
    job_config = dict(config)
    receptor = dict(config.get("receptor", {}))
    receptor["docking_box"] = dict(box)
    job_config["receptor"] = receptor
    return job_config


def prepare_receptor(config: dict[str, Any], receptor_file: str | Path, run_mode: str) -> pd.DataFrame:
    """Validate receptor path and write receptor-preparation metadata."""
    box = validate_docking_box(config)
    path = Path(receptor_file)
    suffix = path.suffix.lower()
    exists = path.exists()
    status = "ready" if exists and suffix in SUPPORTED_FORMATS else "missing"
    notes = ""
    if not exists:
        notes = "Sequence-only target windows cannot be used for real molecular docking without a 3D receptor structure."
        if run_mode not in {"dry_run", "prepare_only", "analyze_existing"}:
            raise FileNotFoundError(f"SMN2 receptor file is missing for real docking mode: {path}. {notes}")
    elif suffix not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported receptor file format: {suffix}. Supported: {sorted(SUPPORTED_FORMATS)}")
    if run_mode == "dry_run" and not exists:
        status = "dry_run_missing_receptor"
        notes += " Dry-run simulation will continue with simulated scores only."
    return pd.DataFrame(
        [
            {
                "receptor_id": "SMN2_target",
                "receptor_type": "SMN2_RNA_target",
                "receptor_name": "SMN2 exon 7 splicing-related RNA target",
                "source_file": str(path),
                "prepared_file": str(path) if exists else "NA",
                "format": suffix.lstrip(".") if suffix else "NA",
                "box_center_x": box["center_x"],
                "box_center_y": box["center_y"],
                "box_center_z": box["center_z"],
                "box_size_x": box["size_x"],
                "box_size_y": box["size_y"],
                "box_size_z": box["size_z"],
                "prep_status": status,
                "notes": notes,
            }
        ]
    )
