"""Configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when the target definition configuration is invalid."""


@dataclass(frozen=True)
class PipelinePaths:
    config_path: Path
    project_root: Path
    genome_fasta: Path
    annotation_gtf: Path
    output_dir: Path
    report_path: Path
    figure_dir: Path
    structure_dir: Path


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path).resolve()
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ConfigError("Config file must contain a YAML mapping.")
    return config


def resolve_paths(config: dict[str, Any], config_path: str | Path) -> PipelinePaths:
    path = Path(config_path).resolve()
    project_root = path.parent.parent

    def resolve(value: str, field: str) -> Path:
        if not value:
            raise ConfigError(f"Missing required path field: {field}")
        candidate = Path(value)
        return candidate if candidate.is_absolute() else project_root / candidate

    return PipelinePaths(
        config_path=path,
        project_root=project_root,
        genome_fasta=resolve(config.get("inputs", {}).get("genome_fasta"), "inputs.genome_fasta"),
        annotation_gtf=resolve(config.get("inputs", {}).get("annotation_gtf"), "inputs.annotation_gtf"),
        output_dir=resolve(config.get("outputs", {}).get("output_dir"), "outputs.output_dir"),
        report_path=resolve(config.get("outputs", {}).get("report_path"), "outputs.report_path"),
        figure_dir=resolve(config.get("outputs", {}).get("figure_dir"), "outputs.figure_dir"),
        structure_dir=resolve(config.get("structures", {}).get("output_dir"), "structures.output_dir"),
    )


def validate_input_paths(paths: PipelinePaths) -> None:
    missing = []
    if not paths.genome_fasta.exists():
        missing.append(f"genome FASTA: {paths.genome_fasta}")
    if not paths.annotation_gtf.exists():
        missing.append(f"annotation GTF/GFF: {paths.annotation_gtf}")
    if missing:
        details = "\n".join(f"- {item}" for item in missing)
        raise FileNotFoundError(
            "Required input file(s) are missing. Provide valid paths in configs/target_definition.yaml:\n"
            f"{details}"
        )


def ensure_output_dirs(paths: PipelinePaths) -> None:
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    paths.report_path.parent.mkdir(parents=True, exist_ok=True)
    paths.figure_dir.mkdir(parents=True, exist_ok=True)
    paths.structure_dir.mkdir(parents=True, exist_ok=True)
