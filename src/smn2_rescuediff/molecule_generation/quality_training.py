"""Quality-weighted training source manifest support for Module 3."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd

from .chemical_filter import calculate_properties, canonicalize_smiles, require_rdkit
from .smiles_cleaner import iter_smiles_lines


MANIFEST_COLUMNS = ["source_name", "path", "source_type", "source_weight", "max_rows", "quality_tier", "notes"]


def _resolve(path: str | Path, project_root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else project_root / candidate


def _allowed_atoms(mol: object, allowed: set[str]) -> bool:
    return all(atom.GetSymbol() in allowed for atom in mol.GetAtoms())


def _as_int(value: object, default: int | None = None) -> int | None:
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if not text or text.lower() in {"na", "n/a", "none", "null"}:
        return default
    return int(float(text))


def _as_weight(value: object) -> float:
    try:
        weight = float(value)
    except (TypeError, ValueError):
        weight = 1.0
    return max(0.1, weight) if math.isfinite(weight) else 1.0


def load_training_source_manifest(manifest_path: str | Path, project_root: Path) -> pd.DataFrame:
    """Load and validate the quality training-source manifest."""
    path = _resolve(manifest_path, project_root)
    if not path.exists():
        raise FileNotFoundError(f"Training source manifest is missing: {path}")
    manifest = pd.read_csv(path, sep="\t")
    missing = set(MANIFEST_COLUMNS) - set(manifest.columns)
    if missing:
        raise ValueError(f"Training source manifest is missing columns: {sorted(missing)}")
    manifest = manifest[MANIFEST_COLUMNS].copy()
    manifest["source_weight"] = manifest["source_weight"].map(_as_weight)
    manifest["resolved_path"] = [str(_resolve(item, project_root)) for item in manifest["path"]]
    return manifest


def build_quality_training_dataset(
    *,
    manifest_path: str | Path,
    project_root: Path,
    output_dir: Path,
    data_cfg: dict[str, Any],
    exclude_canonical_smiles: set[str] | None = None,
) -> tuple[list[str], dict[str, Any], pd.DataFrame, list[str]]:
    """Build unique and weighted quality-training tables from a source manifest."""
    require_rdkit()
    Chem, Crippen, Descriptors, *_ = require_rdkit()
    manifest = load_training_source_manifest(manifest_path, project_root)
    quality_dir = output_dir / "quality_training"
    quality_dir.mkdir(parents=True, exist_ok=True)

    max_len = int(data_cfg.get("max_input_smiles_length", 160))
    remove_salts = bool(data_cfg.get("remove_salts", True))
    remove_mixtures = bool(data_cfg.get("remove_mixtures", True))
    min_mw = float(data_cfg.get("min_molecular_weight_training", 100))
    max_mw = float(data_cfg.get("max_molecular_weight_training", 900))
    allowed_atoms = set(data_cfg.get("allowed_atoms", ["C", "H", "N", "O", "S", "P", "F", "Cl", "Br", "I"]))

    unique: dict[str, dict[str, Any]] = {}
    weighted_rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    raw_total = 0
    valid_total = 0
    duplicate_total = 0
    benchmark_holdout_total = 0
    held_out = exclude_canonical_smiles or set()

    for source in manifest.to_dict(orient="records"):
        source_name = str(source["source_name"])
        source_path = Path(str(source["resolved_path"]))
        source_weight = _as_weight(source.get("source_weight", 1.0))
        max_rows = _as_int(source.get("max_rows"), None)
        source_raw = source_valid = source_invalid = source_duplicates = source_benchmark_holdouts = source_used_weighted = 0
        if not source_path.exists():
            warnings.append(f"Training source missing and skipped: {source_name} ({source_path})")
            source_rows.append(
                {
                    **source,
                    "loaded_rows": 0,
                    "valid_rows": 0,
                    "invalid_rows": 0,
                    "benchmark_holdout_rows": 0,
                    "deduplicated_rows": 0,
                    "unique_used_rows": 0,
                    "weighted_used_rows": 0,
                }
            )
            continue

        extracted = iter_smiles_lines(source_path.read_text(encoding="utf-8").splitlines())
        if max_rows is not None:
            extracted = extracted[:max_rows]
        for raw_smiles in extracted:
            source_raw += 1
            raw_total += 1
            smiles = str(raw_smiles).strip().split()[0]
            if not smiles or len(smiles) > max_len:
                source_invalid += 1
                rejected.append({"source_name": source_name, "raw_smiles": raw_smiles, "canonical_smiles": "", "reason": "empty_or_too_long"})
                continue
            if remove_mixtures and "." in smiles:
                source_invalid += 1
                rejected.append({"source_name": source_name, "raw_smiles": smiles, "canonical_smiles": "", "reason": "mixture_or_salt"})
                continue
            canonical, mol = canonicalize_smiles(smiles, remove_salts=remove_salts)
            if canonical is None or mol is None:
                source_invalid += 1
                rejected.append({"source_name": source_name, "raw_smiles": smiles, "canonical_smiles": "", "reason": "rdkit_invalid"})
                continue
            mw = float(Descriptors.MolWt(mol))
            if mw < min_mw or mw > max_mw:
                source_invalid += 1
                rejected.append({"source_name": source_name, "raw_smiles": smiles, "canonical_smiles": canonical, "reason": "training_molecular_weight_out_of_range"})
                continue
            if not _allowed_atoms(mol, allowed_atoms):
                source_invalid += 1
                rejected.append({"source_name": source_name, "raw_smiles": smiles, "canonical_smiles": canonical, "reason": "disallowed_atom"})
                continue
            if canonical in held_out:
                source_benchmark_holdouts += 1
                benchmark_holdout_total += 1
                rejected.append({"source_name": source_name, "raw_smiles": smiles, "canonical_smiles": canonical, "reason": "held_out_benchmark_control"})
                continue

            source_valid += 1
            valid_total += 1
            props = calculate_properties(mol, calculate_qed=True, pains_catalog=None)
            if canonical in unique:
                duplicate_total += 1
                source_duplicates += 1
                record = unique[canonical]
                record["source_names"] = ";".join(sorted(set(str(record["source_names"]).split(";") + [source_name])))
                record["source_types"] = ";".join(sorted(set(str(record["source_types"]).split(";") + [str(source["source_type"])])))
                record["quality_tiers"] = ";".join(sorted(set(str(record["quality_tiers"]).split(";") + [str(source["quality_tier"])])))
                record["total_source_weight"] = round(float(record["total_source_weight"]) + source_weight, 6)
                record["duplicate_source_count"] = int(record["duplicate_source_count"]) + 1
            else:
                unique[canonical] = {
                    "canonical_smiles": canonical,
                    "source_names": source_name,
                    "source_types": str(source["source_type"]),
                    "quality_tiers": str(source["quality_tier"]),
                    "primary_source_name": source_name,
                    "primary_source_type": str(source["source_type"]),
                    "primary_quality_tier": str(source["quality_tier"]),
                    "source_notes": str(source.get("notes", "")),
                    "source_weight": source_weight,
                    "total_source_weight": source_weight,
                    "duplicate_source_count": 0,
                    **props,
                }

            repeat_count = max(1, int(round(source_weight)))
            for repeat_idx in range(repeat_count):
                weighted_rows.append(
                    {
                        "canonical_smiles": canonical,
                        "source_name": source_name,
                        "source_type": str(source["source_type"]),
                        "quality_tier": str(source["quality_tier"]),
                        "source_weight": source_weight,
                        "weighted_repeat_index": repeat_idx + 1,
                    }
                )
                source_used_weighted += 1

        source_rows.append(
            {
                **source,
                "loaded_rows": source_raw,
                "valid_rows": source_valid,
                "invalid_rows": source_invalid,
                "benchmark_holdout_rows": source_benchmark_holdouts,
                "deduplicated_rows": source_duplicates,
                "unique_used_rows": source_valid - source_duplicates,
                "weighted_used_rows": source_used_weighted,
            }
        )

    unique_df = pd.DataFrame(unique.values()).sort_values(["primary_quality_tier", "canonical_smiles"]).reset_index(drop=True)
    weighted_df = pd.DataFrame(weighted_rows)
    source_report = pd.DataFrame(source_rows)
    rejected_df = pd.DataFrame(rejected, columns=["source_name", "raw_smiles", "canonical_smiles", "reason"])
    if unique_df.empty:
        raise ValueError("No valid training molecules remained after quality manifest cleaning.")

    unique_df.to_csv(quality_dir / "quality_training_unique.tsv", sep="\t", index=False)
    weighted_df.to_csv(quality_dir / "quality_training_weighted.tsv", sep="\t", index=False)
    source_report.to_csv(quality_dir / "source_balance_report.tsv", sep="\t", index=False)
    rejected_df.to_csv(quality_dir / "quality_training_rejected.tsv", sep="\t", index=False)

    summary = {
        "training_source_manifest": str(_resolve(manifest_path, project_root)),
        "training_files": manifest["resolved_path"].astype(str).tolist(),
        "training_molecules_raw": int(raw_total),
        "training_molecules_valid": int(valid_total),
        "training_molecules_invalid": int(len(rejected_df)),
        "training_molecules_deduplicated": int(duplicate_total),
        "training_molecules_held_out_benchmark_controls": int(benchmark_holdout_total),
        "training_molecules_cleaned": int(len(unique_df)),
        "training_molecules_weighted": int(len(weighted_df)),
        "source_balance": source_report.to_dict(orient="records"),
        "warnings": warnings,
    }
    return weighted_df["canonical_smiles"].astype(str).tolist(), summary, rejected_df, warnings
