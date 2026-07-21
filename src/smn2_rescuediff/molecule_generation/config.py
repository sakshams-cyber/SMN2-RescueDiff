"""Configuration loading for Module 3 molecule generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class MoleculeGenerationConfigError(ValueError):
    """Raised when the molecule-generation configuration is invalid."""


@dataclass(frozen=True)
class MoleculeGenerationPaths:
    config_path: Path
    project_root: Path
    training_smiles: list[Path]
    smn2_target_profile: Path
    smn2_target_metadata: Path
    decoy_atlas_summary: Path
    output_dir: Path
    checkpoint_dir: Path
    log_dir: Path
    report_path: Path
    figure_dir: Path
    raw_molecule_dir: Path


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load the YAML configuration."""
    path = Path(config_path).resolve()
    if not path.exists():
        raise MoleculeGenerationConfigError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise MoleculeGenerationConfigError("Molecule-generation config must contain a YAML mapping.")
    return config


def find_project_root(config_path: Path) -> Path:
    """Find the repository root for configs stored in configs/ or configs/presets/."""
    for parent in config_path.parents:
        if (parent / "src" / "smn2_rescuediff").exists() and (parent / "configs").exists():
            return parent
    return config_path.parent.parent


def resolve_paths(config: dict[str, Any], config_path: str | Path) -> MoleculeGenerationPaths:
    """Resolve config paths relative to the repository root."""
    path = Path(config_path).resolve()
    project_root = find_project_root(path)

    def resolve(value: str | None, field: str) -> Path:
        if not value:
            raise MoleculeGenerationConfigError(f"Missing required path field: {field}")
        candidate = Path(value)
        return candidate if candidate.is_absolute() else project_root / candidate

    def optional_context(name: str) -> str | None:
        inputs = config.get("inputs", {})
        optional = inputs.get("optional_context", {})
        return optional.get(name) or inputs.get(name)

    inputs = config.get("inputs", {})
    outputs = config.get("outputs", {})
    raw_training = inputs.get("training_smiles")
    if isinstance(raw_training, list):
        training = [resolve(str(item), "inputs.training_smiles") for item in raw_training]
    elif raw_training:
        training = [resolve(str(raw_training), "inputs.training_smiles")]
    else:
        raise MoleculeGenerationConfigError("Missing required path field: inputs.training_smiles")
    output_dir = resolve(outputs.get("output_dir"), "outputs.output_dir")
    return MoleculeGenerationPaths(
        config_path=path,
        project_root=project_root,
        training_smiles=training,
        smn2_target_profile=resolve(optional_context("smn2_target_profile"), "inputs.optional_context.smn2_target_profile"),
        smn2_target_metadata=resolve(optional_context("smn2_target_metadata"), "inputs.optional_context.smn2_target_metadata"),
        decoy_atlas_summary=resolve(optional_context("decoy_atlas_summary"), "inputs.optional_context.decoy_atlas_summary"),
        output_dir=output_dir,
        checkpoint_dir=resolve(outputs.get("checkpoint_dir") or str(output_dir / "checkpoints"), "outputs.checkpoint_dir"),
        log_dir=resolve(outputs.get("log_dir") or str(output_dir / "logs"), "outputs.log_dir"),
        report_path=resolve(outputs.get("report_path"), "outputs.report_path"),
        figure_dir=resolve(outputs.get("figure_dir"), "outputs.figure_dir"),
        raw_molecule_dir=training[0].parent,
    )


def ensure_output_dirs(paths: MoleculeGenerationPaths) -> None:
    """Create output directories for generated molecules, figures, and reports."""
    try:
        paths.output_dir.mkdir(parents=True, exist_ok=True)
        paths.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        paths.log_dir.mkdir(parents=True, exist_ok=True)
        paths.figure_dir.mkdir(parents=True, exist_ok=True)
        paths.report_path.parent.mkdir(parents=True, exist_ok=True)
        paths.raw_molecule_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OSError(f"Could not create molecule-generation output directories: {exc}") from exc


def optional_context_status(paths: MoleculeGenerationPaths) -> dict[str, bool]:
    """Record whether optional project context files exist."""
    return {
        "smn2_target_profile_exists": paths.smn2_target_profile.exists(),
        "smn2_target_metadata_exists": paths.smn2_target_metadata.exists(),
        "decoy_atlas_summary_exists": paths.decoy_atlas_summary.exists(),
    }
