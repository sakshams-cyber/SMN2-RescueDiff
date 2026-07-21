"""Configuration loading and validation for the HuRDA decoy atlas."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class DecoyAtlasConfigError(ValueError):
    """Raised when the decoy atlas configuration is invalid."""


@dataclass(frozen=True)
class DecoyAtlasPaths:
    config_path: Path
    project_root: Path
    genome_fasta: Path
    annotation_gtf: Path
    smn2_target_fasta: Path
    smn2_target_profile: Path
    smn2_target_metadata: Path
    output_dir: Path
    report_path: Path
    figure_dir: Path


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    path = Path(config_path).resolve()
    if not path.exists():
        raise DecoyAtlasConfigError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise DecoyAtlasConfigError("Decoy atlas config must contain a YAML mapping.")
    return config


def resolve_paths(config: dict[str, Any], config_path: str | Path) -> DecoyAtlasPaths:
    """Resolve config paths relative to the project root."""
    path = Path(config_path).resolve()
    project_root = path.parent.parent

    def resolve(value: str | None, field: str) -> Path:
        if not value:
            raise DecoyAtlasConfigError(f"Missing required path field: {field}")
        candidate = Path(value)
        return candidate if candidate.is_absolute() else project_root / candidate

    inputs = config.get("inputs", {})
    outputs = config.get("outputs", {})
    return DecoyAtlasPaths(
        config_path=path,
        project_root=project_root,
        genome_fasta=resolve(inputs.get("genome_fasta"), "inputs.genome_fasta"),
        annotation_gtf=resolve(inputs.get("annotation_gtf"), "inputs.annotation_gtf"),
        smn2_target_fasta=resolve(inputs.get("smn2_target_fasta"), "inputs.smn2_target_fasta"),
        smn2_target_profile=resolve(inputs.get("smn2_target_profile"), "inputs.smn2_target_profile"),
        smn2_target_metadata=resolve(inputs.get("smn2_target_metadata"), "inputs.smn2_target_metadata"),
        output_dir=resolve(outputs.get("output_dir"), "outputs.output_dir"),
        report_path=resolve(outputs.get("report_path"), "outputs.report_path"),
        figure_dir=resolve(outputs.get("figure_dir"), "outputs.figure_dir"),
    )


def validate_input_paths(paths: DecoyAtlasPaths) -> None:
    """Fail clearly if required Module 1 or raw data inputs are missing."""
    required = [
        ("genome FASTA", paths.genome_fasta),
        ("annotation GTF/GFF", paths.annotation_gtf),
        ("Module 1 SMN2 target FASTA", paths.smn2_target_fasta),
        ("Module 1 SMN2 target profile TSV", paths.smn2_target_profile),
        ("Module 1 SMN2 target metadata JSON", paths.smn2_target_metadata),
    ]
    missing = [f"{label}: {path}" for label, path in required if not path.exists()]
    if missing:
        details = "\n".join(f"- {item}" for item in missing)
        raise FileNotFoundError(
            "Required decoy-atlas input file(s) are missing. Run Module 1 first and update configs/decoy_atlas.yaml:\n"
            f"{details}"
        )


def ensure_output_dirs(paths: DecoyAtlasPaths) -> None:
    """Create output directories, raising an actionable error on failure."""
    try:
        paths.output_dir.mkdir(parents=True, exist_ok=True)
        paths.report_path.parent.mkdir(parents=True, exist_ok=True)
        paths.figure_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OSError(f"Could not create decoy atlas output directories: {exc}") from exc

