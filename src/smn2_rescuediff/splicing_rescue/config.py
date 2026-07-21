"""Configuration handling for Module 5 final splicing-rescue scoring."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class SplicingRescueConfigError(ValueError):
    """Raised when Module 5 configuration is missing or invalid."""


@dataclass(frozen=True)
class SplicingRescuePaths:
    """Resolved Module 5 input and output paths."""

    config_path: Path
    project_root: Path
    smn2_target_fasta: Path
    smn2_target_profile: Path
    smn2_target_metadata: Path
    decoy_atlas_full: Path
    decoy_atlas_summary: Path
    generated_molecules_filtered: Path
    generated_molecules_for_docking: Path
    molecule_generation_summary: Path
    target_docking_scores: Path
    decoy_docking_scores: Path
    molecule_selectivity_scores: Path
    candidates_for_splicing_rescue: Path
    docking_summary: Path
    pose_dir: Path
    output_dir: Path
    table_dir: Path
    evidence_dir: Path
    log_dir: Path
    figure_dir: Path
    splicing_report_path: Path
    final_report_path: Path


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load and minimally validate YAML config."""
    path = Path(config_path).resolve()
    if not path.exists():
        raise SplicingRescueConfigError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise SplicingRescueConfigError("Module 5 config must contain a YAML mapping.")
    return config


def find_project_root(config_path: Path) -> Path:
    """Find the repository root for configs stored in configs/ or configs/presets/."""
    for parent in config_path.parents:
        if (parent / "src" / "smn2_rescuediff").exists() and (parent / "configs").exists():
            return parent
    return config_path.parent.parent


def resolve_paths(config: dict[str, Any], config_path: str | Path) -> SplicingRescuePaths:
    """Resolve all configured paths relative to the repository root."""
    path = Path(config_path).resolve()
    project_root = find_project_root(path)
    inputs = config.get("inputs", {})
    outputs = config.get("outputs", {})

    def resolve(value: str | None, field: str) -> Path:
        if not value:
            raise SplicingRescueConfigError(f"Missing required path field: {field}")
        candidate = Path(value)
        return candidate if candidate.is_absolute() else project_root / candidate

    output_dir = resolve(outputs.get("output_dir"), "outputs.output_dir")
    table_dir = resolve(outputs.get("table_dir") or str(output_dir / "tables"), "outputs.table_dir")
    evidence_dir = resolve(outputs.get("evidence_dir") or str(output_dir / "evidence"), "outputs.evidence_dir")
    log_dir = resolve(outputs.get("log_dir") or str(output_dir / "logs"), "outputs.log_dir")
    return SplicingRescuePaths(
        config_path=path,
        project_root=project_root,
        smn2_target_fasta=resolve(inputs.get("smn2_target_fasta"), "inputs.smn2_target_fasta"),
        smn2_target_profile=resolve(inputs.get("smn2_target_profile"), "inputs.smn2_target_profile"),
        smn2_target_metadata=resolve(inputs.get("smn2_target_metadata"), "inputs.smn2_target_metadata"),
        decoy_atlas_full=resolve(inputs.get("decoy_atlas_full"), "inputs.decoy_atlas_full"),
        decoy_atlas_summary=resolve(inputs.get("decoy_atlas_summary"), "inputs.decoy_atlas_summary"),
        generated_molecules_filtered=resolve(inputs.get("generated_molecules_filtered"), "inputs.generated_molecules_filtered"),
        generated_molecules_for_docking=resolve(inputs.get("generated_molecules_for_docking"), "inputs.generated_molecules_for_docking"),
        molecule_generation_summary=resolve(inputs.get("molecule_generation_summary"), "inputs.molecule_generation_summary"),
        target_docking_scores=resolve(inputs.get("target_docking_scores"), "inputs.target_docking_scores"),
        decoy_docking_scores=resolve(inputs.get("decoy_docking_scores"), "inputs.decoy_docking_scores"),
        molecule_selectivity_scores=resolve(inputs.get("molecule_selectivity_scores"), "inputs.molecule_selectivity_scores"),
        candidates_for_splicing_rescue=resolve(inputs.get("candidates_for_splicing_rescue"), "inputs.candidates_for_splicing_rescue"),
        docking_summary=resolve(inputs.get("docking_summary"), "inputs.docking_summary"),
        pose_dir=resolve(inputs.get("pose_dir"), "inputs.pose_dir"),
        output_dir=output_dir,
        table_dir=table_dir,
        evidence_dir=evidence_dir,
        log_dir=log_dir,
        figure_dir=resolve(outputs.get("figure_dir"), "outputs.figure_dir"),
        splicing_report_path=resolve(outputs.get("splicing_report_path"), "outputs.splicing_report_path"),
        final_report_path=resolve(outputs.get("final_report_path"), "outputs.final_report_path"),
    )


def ensure_output_dirs(paths: SplicingRescuePaths) -> None:
    """Create Module 5 output directories."""
    try:
        for path in [
            paths.output_dir,
            paths.table_dir,
            paths.evidence_dir,
            paths.log_dir,
            paths.figure_dir,
            paths.splicing_report_path.parent,
            paths.final_report_path.parent,
        ]:
            path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OSError(f"Could not create Module 5 output directories: {exc}") from exc


def validate_weights(config: dict[str, Any]) -> None:
    """Fail clearly when SRSI or splicing weights are missing or invalid."""
    for section, key in [
        ("srsi", "weights"),
        ("splicing_rescue_scoring", "weights"),
    ]:
        weights = config.get(section, {}).get(key, {})
        if not isinstance(weights, dict) or not weights:
            raise SplicingRescueConfigError(f"{section}.{key} must be a non-empty mapping.")
        for name, value in weights.items():
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise SplicingRescueConfigError(f"Invalid numeric weight at {section}.{key}.{name}: {value}") from exc
            if number < 0 and "penalty" not in str(name).lower():
                raise SplicingRescueConfigError(f"Only penalty weights may be negative: {section}.{key}.{name}")


def save_config_used(config: dict[str, Any], path: str | Path) -> None:
    """Write the exact Module 5 config used."""
    Path(path).write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
