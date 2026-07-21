"""Receptor validation and metadata creation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

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

