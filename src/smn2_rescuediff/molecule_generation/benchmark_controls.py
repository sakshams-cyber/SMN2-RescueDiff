"""Held-out benchmark splicing-modifier controls for Module 3."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .chemical_filter import canonicalize_smiles, require_rdkit
from .novelty_analysis import _fingerprint


BENCHMARK_COLUMNS = [
    "benchmark_control_name",
    "canonical_smiles",
    "raw_smiles",
    "source_path",
]


def _resolve(path: str | Path, project_root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else project_root / candidate


def _configured_path(config: dict[str, Any], project_root: Path) -> Path | None:
    controls = config.get("inputs", {}).get("benchmark_controls", {})
    if not isinstance(controls, dict):
        return None
    path = controls.get("splicing_controls")
    return _resolve(str(path), project_root) if path else None


def _parse_control_line(line: str, idx: int, remove_salts: bool) -> dict[str, str] | None:
    text = line.strip()
    if not text or text.startswith("#"):
        return None
    tokens = text.split()
    if not tokens:
        return None

    candidate_positions = [0, *range(1, len(tokens))]
    for position in candidate_positions:
        raw_smiles = tokens[position]
        canonical, _mol = canonicalize_smiles(raw_smiles, remove_salts=remove_salts)
        if canonical is None:
            continue
        if position == 0:
            name = " ".join(tokens[1:]).strip() or f"benchmark_control_{idx:03d}"
        else:
            name = " ".join(tokens[:position] + tokens[position + 1 :]).strip() or f"benchmark_control_{idx:03d}"
        return {
            "benchmark_control_name": name,
            "canonical_smiles": canonical,
            "raw_smiles": raw_smiles,
        }
    return None


def load_benchmark_controls(path: str | Path, *, remove_salts: bool = True) -> pd.DataFrame:
    """Load held-out benchmark controls from a flexible .smi-like file.

    The preferred format is ``SMILES name``. For convenience, ``name SMILES`` is
    also accepted when RDKit can identify the SMILES token unambiguously.
    """
    require_rdkit()
    p = Path(path)
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for idx, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        parsed = _parse_control_line(line, idx, remove_salts)
        if parsed is None:
            continue
        if parsed["canonical_smiles"] in seen:
            continue
        seen.add(parsed["canonical_smiles"])
        parsed["source_path"] = str(p)
        rows.append(parsed)
    return pd.DataFrame(rows, columns=BENCHMARK_COLUMNS)


def load_configured_benchmark_controls(
    config: dict[str, Any],
    project_root: Path,
    *,
    remove_salts: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Load benchmark controls named in config, returning warnings instead of crashing on absence."""
    warnings: list[str] = []
    path = _configured_path(config, project_root)
    if path is None:
        return pd.DataFrame(columns=BENCHMARK_COLUMNS), warnings
    if not path.exists():
        warnings.append(
            f"Benchmark splicing-control file is configured but missing: {path}. "
            "Benchmark controls were not loaded and cannot be used for holdout filtering."
        )
        return pd.DataFrame(columns=BENCHMARK_COLUMNS), warnings
    controls = load_benchmark_controls(path, remove_salts=remove_salts)
    if controls.empty:
        warnings.append(f"Benchmark splicing-control file contained no valid RDKit-parsable controls: {path}.")
    return controls, warnings


def benchmark_control_set(controls: pd.DataFrame) -> set[str]:
    """Return canonical held-out benchmark SMILES for training exclusion."""
    if controls.empty or "canonical_smiles" not in controls:
        return set()
    return set(controls["canonical_smiles"].dropna().astype(str))


def annotate_benchmark_similarity(
    molecules: pd.DataFrame,
    controls: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    """Annotate generated candidates by nearest held-out benchmark-control similarity."""
    result = molecules.copy()
    warnings: list[str] = []
    cfg = config.get("benchmark_similarity", {})
    enabled = bool(cfg.get("enabled", False))
    threshold = float(cfg.get("max_allowed_tanimoto_to_benchmark", 0.70))
    exclude = bool(cfg.get("exclude_too_similar_candidates", True))

    for column, default in [
        ("nearest_benchmark_control_name", "NA"),
        ("nearest_benchmark_tanimoto", np.nan),
        ("is_too_similar_to_benchmark", False),
    ]:
        if column not in result:
            result[column] = default

    if result.empty:
        return result, {"enabled": enabled, "benchmark_controls_loaded": int(len(controls)), "too_similar_count": 0}, warnings
    if not enabled:
        return result, {"enabled": False, "benchmark_controls_loaded": int(len(controls)), "too_similar_count": 0}, warnings
    if controls.empty:
        warnings.append("Benchmark similarity filtering is enabled, but no valid held-out benchmark controls were loaded.")
        return result, {"enabled": True, "benchmark_controls_loaded": 0, "too_similar_count": 0}, warnings

    require_rdkit()
    from rdkit import DataStructs

    control_records = []
    for _, row in controls.iterrows():
        smiles = str(row.get("canonical_smiles", "")).strip()
        if not smiles or smiles == "NA":
            continue
        try:
            control_records.append((str(row.get("benchmark_control_name", "benchmark_control")), smiles, _fingerprint(smiles)))
        except Exception:
            continue
    if not control_records:
        warnings.append("Benchmark similarity filtering is enabled, but no benchmark-control fingerprints could be calculated.")
        return result, {"enabled": True, "benchmark_controls_loaded": int(len(controls)), "too_similar_count": 0}, warnings

    names: list[str] = []
    sims: list[float] = []
    flags: list[bool] = []
    control_fps = [record[2] for record in control_records]
    for smiles in result.get("canonical_smiles", pd.Series([""] * len(result), index=result.index)).astype(str):
        try:
            fp = _fingerprint(smiles)
            similarities = DataStructs.BulkTanimotoSimilarity(fp, control_fps)
        except Exception:
            similarities = []
        if not similarities:
            names.append("NA")
            sims.append(np.nan)
            flags.append(False)
            continue
        best_idx = int(np.argmax(similarities))
        best = float(similarities[best_idx])
        names.append(control_records[best_idx][0])
        sims.append(round(best, 6))
        flags.append(best > threshold)

    result["nearest_benchmark_control_name"] = names
    result["nearest_benchmark_tanimoto"] = sims
    result["is_too_similar_to_benchmark"] = flags
    too_similar_count = int(sum(flags))
    summary = {
        "enabled": True,
        "benchmark_controls_loaded": int(len(control_records)),
        "max_allowed_tanimoto_to_benchmark": threshold,
        "exclude_too_similar_candidates": exclude,
        "too_similar_count": too_similar_count,
        "candidate_count": int(len(result)),
    }
    if too_similar_count:
        warnings.append(
            f"{too_similar_count} generated candidates exceeded the benchmark-control Tanimoto threshold "
            f"({threshold:.2f}); final docking export exclusion is {exclude}."
        )
    return result, summary, warnings
