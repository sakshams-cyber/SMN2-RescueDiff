"""Configuration handling for Module 4 docking and counter-screening."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class DockingConfigError(ValueError):
    """Raised when docking configuration is invalid."""


@dataclass(frozen=True)
class DockingPaths:
    config_path: Path
    project_root: Path
    molecules_filtered: Path
    smn2_target_fasta: Path
    smn2_target_profile: Path
    smn2_target_metadata: Path
    decoy_atlas_full: Path
    decoy_atlas_summary: Path
    raw_structure_dir: Path
    output_dir: Path
    ligand_dir: Path
    receptor_dir: Path
    pose_dir: Path
    log_dir: Path
    simulation_dir: Path
    report_path: Path
    figure_dir: Path
    live_log_file: Path
    live_state_file: Path
    smn2_receptor_file: Path


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load YAML config."""
    path = Path(config_path).resolve()
    if not path.exists():
        raise DockingConfigError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise DockingConfigError("Docking config must contain a YAML mapping.")
    return config


def resolve_paths(config: dict[str, Any], config_path: str | Path) -> DockingPaths:
    """Resolve all config paths relative to repo root."""
    path = Path(config_path).resolve()
    project_root = path.parent.parent

    def resolve(value: str | None, field: str) -> Path:
        if not value:
            raise DockingConfigError(f"Missing required path field: {field}")
        candidate = Path(value)
        return candidate if candidate.is_absolute() else project_root / candidate

    inputs = config.get("inputs", {})
    outputs = config.get("outputs", {})
    live = config.get("live_simulation", {})
    receptor = config.get("receptor", {})
    return DockingPaths(
        config_path=path,
        project_root=project_root,
        molecules_filtered=resolve(inputs.get("molecules_filtered"), "inputs.molecules_filtered"),
        smn2_target_fasta=resolve(inputs.get("smn2_target_fasta"), "inputs.smn2_target_fasta"),
        smn2_target_profile=resolve(inputs.get("smn2_target_profile"), "inputs.smn2_target_profile"),
        smn2_target_metadata=resolve(inputs.get("smn2_target_metadata"), "inputs.smn2_target_metadata"),
        decoy_atlas_full=resolve(inputs.get("decoy_atlas_full"), "inputs.decoy_atlas_full"),
        decoy_atlas_summary=resolve(inputs.get("decoy_atlas_summary"), "inputs.decoy_atlas_summary"),
        raw_structure_dir=resolve(inputs.get("raw_structure_dir"), "inputs.raw_structure_dir"),
        output_dir=resolve(outputs.get("output_dir"), "outputs.output_dir"),
        ligand_dir=resolve(outputs.get("ligand_dir"), "outputs.ligand_dir"),
        receptor_dir=resolve(outputs.get("receptor_dir"), "outputs.receptor_dir"),
        pose_dir=resolve(outputs.get("pose_dir"), "outputs.pose_dir"),
        log_dir=resolve(outputs.get("log_dir"), "outputs.log_dir"),
        simulation_dir=resolve(outputs.get("simulation_dir"), "outputs.simulation_dir"),
        report_path=resolve(outputs.get("report_path"), "outputs.report_path"),
        figure_dir=resolve(outputs.get("figure_dir"), "outputs.figure_dir"),
        live_log_file=resolve(live.get("live_log_file"), "live_simulation.live_log_file"),
        live_state_file=resolve(live.get("live_state_file"), "live_simulation.live_state_file"),
        smn2_receptor_file=resolve(receptor.get("smn2_receptor_file"), "receptor.smn2_receptor_file"),
    )


def ensure_output_dirs(paths: DockingPaths) -> None:
    """Create output directories."""
    for directory in [
        paths.output_dir,
        paths.ligand_dir,
        paths.receptor_dir,
        paths.pose_dir,
        paths.log_dir,
        paths.simulation_dir,
        paths.report_path.parent,
        paths.figure_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def run_mode(config: dict[str, Any]) -> str:
    """Return validated run mode."""
    mode = str(config.get("run", {}).get("run_mode", "dry_run")).strip()
    allowed = {"dry_run", "prepare_only", "target_only_docking", "target_plus_decoy_docking", "analyze_existing"}
    if mode not in allowed:
        raise DockingConfigError(f"run.run_mode must be one of {sorted(allowed)}")
    return mode


def validate_required_inputs(paths: DockingPaths, config: dict[str, Any]) -> list[str]:
    """Validate required input files and return warnings."""
    mode = run_mode(config)
    warnings: list[str] = []
    required = [
        ("Module 3 filtered molecules", paths.molecules_filtered),
        ("Module 1 target FASTA", paths.smn2_target_fasta),
        ("Module 1 target profile", paths.smn2_target_profile),
        ("Module 1 target metadata", paths.smn2_target_metadata),
    ]
    if config.get("decoy_panel", {}).get("use_decoy_panel", True):
        required.extend(
            [
                ("Module 2 HuRDA atlas", paths.decoy_atlas_full),
                ("Module 2 HuRDA summary", paths.decoy_atlas_summary),
            ]
        )
    missing = [f"{label}: {path}" for label, path in required if not path.exists()]
    if missing and mode != "analyze_existing":
        details = "\n".join(f"- {item}" for item in missing)
        raise FileNotFoundError(f"Required Module 4 input file(s) are missing:\n{details}")
    if not paths.raw_structure_dir.exists():
        warnings.append("Raw structure directory is missing; real receptor discovery may be unavailable.")
    return warnings

