"""Build cleaned, canonical training datasets from one or more SMILES files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .chemical_filter import canonicalize_smiles, require_rdkit
from .quality_training import build_quality_training_dataset
from .smiles_cleaner import clean_smiles_strings, iter_smiles_lines
from .smiles_loader import TOY_WARNING, create_example_training_smiles


def normalize_training_paths(value: str | list[str]) -> list[str]:
    """Accept a single path or list of paths from config."""
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _allowed_atoms(mol: object, allowed: set[str]) -> bool:
    return all(atom.GetSymbol() in allowed for atom in mol.GetAtoms())


def build_training_dataset(config: dict[str, Any], project_root: Path, output_dir: Path) -> tuple[list[str], dict[str, Any], pd.DataFrame, list[str]]:
    """Load, validate, canonicalize, and merge configured training SMILES files."""
    require_rdkit()
    inputs = config.get("inputs", {})
    data_cfg = config.get("data_processing", {})
    manifest = inputs.get("training_source_manifest")
    if manifest:
        accepted, summary, rejected_df, warnings = build_quality_training_dataset(
            manifest_path=str(manifest),
            project_root=project_root,
            output_dir=output_dir,
            data_cfg=data_cfg,
        )
        min_warning = int(data_cfg.get("min_training_molecules_warning", 1000))
        if summary.get("training_molecules_cleaned", 0) < min_warning:
            warnings.append(
                "WARNING: This quality-weighted training dataset is small for final molecule generation. "
                "Interpret the run as pilot/debug unless larger source files are provided."
            )
        cleaned_path = output_dir / "training_smiles_merged_cleaned.smi"
        cleaned_path.parent.mkdir(parents=True, exist_ok=True)
        cleaned_path.write_text("\n".join(accepted) + "\n", encoding="utf-8")
        rejected_path = output_dir / "training_dataset_rejected.tsv"
        rejected_path.parent.mkdir(parents=True, exist_ok=True)
        rejected_df.to_csv(rejected_path, sep="\t", index=False)
        summary["warnings"] = warnings
        summary_path = output_dir / "training_dataset_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return accepted, summary, rejected_df, warnings

    raw_value = inputs.get("training_smiles", [])
    paths = normalize_training_paths(raw_value) if raw_value else []
    warnings: list[str] = []
    resolved: list[Path] = []
    for item in paths:
        path = Path(item)
        resolved.append(path if path.is_absolute() else project_root / path)
    existing = [path for path in resolved if path.exists()]
    if not existing:
        fallback = create_example_training_smiles(project_root / "data" / "raw" / "molecules")
        existing = [fallback]
        warnings.append(TOY_WARNING)
    raw_smiles: list[str] = []
    raw_count_by_file: dict[str, int] = {}
    for path in existing:
        lines = path.read_text(encoding="utf-8").splitlines()
        extracted = iter_smiles_lines(lines)
        raw_count_by_file[str(path)] = len(extracted)
        raw_smiles.extend(extracted)

    max_len = int(data_cfg.get("max_input_smiles_length", 160))
    cleaned_text = clean_smiles_strings(raw_smiles, max_length=max_len)
    remove_salts = bool(data_cfg.get("remove_salts", True))
    remove_mixtures = bool(data_cfg.get("remove_mixtures", True))
    canonicalize = bool(data_cfg.get("canonicalize_smiles", True))
    remove_duplicates = bool(data_cfg.get("remove_duplicates", True))
    min_mw = float(data_cfg.get("min_molecular_weight_training", 100))
    max_mw = float(data_cfg.get("max_molecular_weight_training", 900))
    allowed_atoms = set(data_cfg.get("allowed_atoms", ["C", "H", "N", "O", "S", "P", "F", "Cl", "Br", "I"]))

    Chem, Crippen, Descriptors, *_ = require_rdkit()
    accepted: list[str] = []
    rejected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for smiles in cleaned_text:
        if remove_mixtures and "." in smiles:
            rejected.append({"raw_smiles": smiles, "reason": "mixture_or_salt"})
            continue
        canonical, mol = canonicalize_smiles(smiles, remove_salts=remove_salts)
        if canonical is None or mol is None:
            rejected.append({"raw_smiles": smiles, "reason": "rdkit_invalid"})
            continue
        final_smiles = canonical if canonicalize else smiles
        mw = float(Descriptors.MolWt(mol))
        if mw < min_mw or mw > max_mw:
            rejected.append({"raw_smiles": smiles, "canonical_smiles": canonical, "reason": "training_molecular_weight_out_of_range"})
            continue
        if not _allowed_atoms(mol, allowed_atoms):
            rejected.append({"raw_smiles": smiles, "canonical_smiles": canonical, "reason": "disallowed_atom"})
            continue
        if remove_duplicates and final_smiles in seen:
            rejected.append({"raw_smiles": smiles, "canonical_smiles": canonical, "reason": "duplicate"})
            continue
        seen.add(final_smiles)
        accepted.append(final_smiles)

    min_warning = int(data_cfg.get("min_training_molecules_warning", 1000))
    if len(accepted) < min_warning:
        warnings.append(
            "WARNING: This training dataset is too small for scientific molecule generation. "
            "This run should be treated as a pipeline test only."
        )
    if not accepted:
        raise ValueError("No valid training molecules remained after cleaning. Provide larger valid SMILES files or loosen data_processing filters.")

    output_dir.mkdir(parents=True, exist_ok=True)
    cleaned_path = output_dir / "training_smiles_merged_cleaned.smi"
    cleaned_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_path.write_text("\n".join(accepted) + "\n", encoding="utf-8")
    rejected_df = pd.DataFrame(rejected, columns=["raw_smiles", "canonical_smiles", "reason"])
    rejected_path = output_dir / "training_dataset_rejected.tsv"
    rejected_path.parent.mkdir(parents=True, exist_ok=True)
    rejected_df.to_csv(rejected_path, sep="\t", index=False)
    summary = {
        "training_files": [str(path) for path in existing],
        "training_molecules_raw": int(len(raw_smiles)),
        "training_molecules_after_text_cleaning": int(len(cleaned_text)),
        "training_molecules_cleaned": int(len(accepted)),
        "raw_count_by_file": raw_count_by_file,
        "rejected_count": int(len(rejected_df)),
        "warnings": warnings,
    }
    summary_path = output_dir / "training_dataset_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return accepted, summary, rejected_df, warnings
