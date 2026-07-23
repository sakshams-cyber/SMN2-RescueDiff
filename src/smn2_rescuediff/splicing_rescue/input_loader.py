"""Input loading and candidate table assembly for Module 5."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import SplicingRescuePaths


class SplicingRescueInputError(ValueError):
    """Raised when required Module 5 inputs are unusable."""


def read_json(path: str | Path, required: bool = True) -> dict[str, Any]:
    """Read JSON metadata, returning an empty dict for optional missing files."""
    p = Path(path)
    if not p.exists():
        if required:
            raise FileNotFoundError(f"Required JSON input is missing: {p}")
        return {}
    with p.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_tsv(path: str | Path, *, required: bool = True, require_molecule_id: bool = False, nrows: int | None = None) -> pd.DataFrame:
    """Read a TSV file with clear validation errors."""
    p = Path(path)
    if not p.exists():
        if required:
            raise FileNotFoundError(f"Required TSV input is missing: {p}")
        return pd.DataFrame()
    df = pd.read_csv(p, sep="\t", nrows=nrows)
    if require_molecule_id and "molecule_id" not in df.columns:
        raise SplicingRescueInputError(f"Input table has no molecule_id column: {p}")
    return df


def _as_bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def _as_false_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"false", "0", "no"})


def _first_numeric_column(df: pd.DataFrame, columns: list[str]) -> pd.Series | None:
    for column in columns:
        if column in df:
            return pd.to_numeric(df[column], errors="coerce")
    return None


def _selectivity_gap_from_available_columns(df: pd.DataFrame) -> pd.Series | None:
    explicit_gap = _first_numeric_column(df, ["selectivity_gap", "mean_selectivity_gap"])
    if explicit_gap is not None:
        return explicit_gap
    if "target_deltaG" not in df:
        return None
    target = pd.to_numeric(df["target_deltaG"], errors="coerce")
    decoy = _first_numeric_column(df, ["decoy_deltaG", "mean_decoy_deltaG", "weighted_mean_decoy_deltaG", "best_decoy_deltaG"])
    if decoy is None:
        return None
    return decoy - target


def normalize_selectivity_table(selectivity: pd.DataFrame) -> pd.DataFrame:
    """Normalize Module 4 selectivity exports without inventing missing decoy evidence."""
    df = selectivity.copy()

    if "weighted_mean_selectivity_gap" in df:
        weighted_mean_gap = pd.to_numeric(df["weighted_mean_selectivity_gap"], errors="coerce")
        if "weighted_selectivity_gap" in df:
            df["weighted_selectivity_gap"] = pd.to_numeric(df["weighted_selectivity_gap"], errors="coerce").combine_first(weighted_mean_gap)
        else:
            df["weighted_selectivity_gap"] = weighted_mean_gap

    if "num_decoys_with_scores" in df:
        decoy_count = pd.to_numeric(df["num_decoys_with_scores"], errors="coerce")
        if "num_decoys_with_valid_scores" in df:
            df["num_decoys_with_valid_scores"] = pd.to_numeric(df["num_decoys_with_valid_scores"], errors="coerce").combine_first(decoy_count)
        else:
            df["num_decoys_with_valid_scores"] = decoy_count

    gap = _selectivity_gap_from_available_columns(df)
    if gap is not None:
        df["selectivity_gap"] = gap

    if "num_decoys_with_valid_scores" not in df:
        df["num_decoys_with_valid_scores"] = pd.NA

    real_decoy_evidence = (
        pd.to_numeric(df.get("num_decoys_with_valid_scores", pd.Series([pd.NA] * len(df), index=df.index)), errors="coerce").fillna(0) > 0
    )
    if "is_dry_run_score" in df:
        real_decoy_evidence &= _as_false_series(df["is_dry_run_score"])
    real_decoy_evidence &= pd.to_numeric(df.get("weighted_selectivity_gap", pd.Series([pd.NA] * len(df), index=df.index)), errors="coerce").notna()
    df["has_real_decoy_selectivity"] = real_decoy_evidence
    if "run_mode" not in df:
        df["run_mode"] = "NA"
    missing_run_mode = df["run_mode"].isna() | df["run_mode"].astype(str).isin(["", "NA", "nan"])
    df.loc[real_decoy_evidence & missing_run_mode, "run_mode"] = "target_plus_decoy_docking"
    return df


def detect_dry_run(target_scores: pd.DataFrame, selectivity: pd.DataFrame, docking_summary: dict[str, Any]) -> bool:
    """Detect whether Module 4 docking inputs are dry-run or simulated."""
    if str(docking_summary.get("run_mode", "")).lower() == "dry_run":
        return True
    if "is_dry_run_score" in target_scores and _as_bool_series(target_scores["is_dry_run_score"]).any():
        return True
    if "is_dry_run_score" in selectivity and _as_bool_series(selectivity["is_dry_run_score"]).any():
        return True
    return False


def load_all_inputs(paths: SplicingRescuePaths) -> dict[str, Any]:
    """Load all Module 1-4 inputs needed by Module 5.

    The filtered Module 3 file, target docking file, and selectivity table are
    required for scoring. The docking-ready molecule export may be absent in
    older Module 3 runs; when absent, filtered molecules are used with a warning.
    """
    warnings: list[str] = []
    target_profile = read_tsv(paths.smn2_target_profile, required=True)
    target_metadata = read_json(paths.smn2_target_metadata, required=True)
    decoy_summary = read_json(paths.decoy_atlas_summary, required=True)
    molecule_summary = read_json(paths.molecule_generation_summary, required=False)
    docking_summary = read_json(paths.docking_summary, required=True)
    molecules = read_tsv(paths.generated_molecules_filtered, required=True, require_molecule_id=True)
    docking_ready = read_tsv(paths.generated_molecules_for_docking, required=False, require_molecule_id=True)
    if docking_ready.empty:
        warnings.append(
            f"Optional docking-ready Module 3 export is missing or empty: {paths.generated_molecules_for_docking}. "
            "Using generated_molecules_filtered.tsv as the molecule feature source."
        )
    target_scores = read_tsv(paths.target_docking_scores, required=True, require_molecule_id=True)
    if target_scores.empty or "target_deltaG" not in target_scores.columns:
        raise SplicingRescueInputError("All docking scores are missing: target_docking_scores has no target_deltaG values.")
    selectivity = normalize_selectivity_table(read_tsv(paths.molecule_selectivity_scores, required=True, require_molecule_id=True))
    if selectivity.empty:
        raise SplicingRescueInputError("No molecules can be scored: molecule_selectivity_scores.tsv is empty.")
    candidates_for_rescue = read_tsv(paths.candidates_for_splicing_rescue, required=False, require_molecule_id=True)
    # Load only a small metadata preview of the full HuRDA atlas by default; the
    # Module 4 selectivity table already contains molecule-level decoy evidence.
    decoy_atlas_preview = read_tsv(paths.decoy_atlas_full, required=False, nrows=1000)
    decoy_scores_available = Path(paths.decoy_docking_scores).exists()
    if not decoy_scores_available:
        warnings.append("Decoy docking score file is unavailable; selectivity table will be used as the decoy evidence summary.")
    return {
        "target_profile": target_profile,
        "target_metadata": target_metadata,
        "decoy_summary": decoy_summary,
        "decoy_atlas_preview": decoy_atlas_preview,
        "molecule_summary": molecule_summary,
        "docking_summary": docking_summary,
        "molecules": molecules,
        "docking_ready": docking_ready,
        "target_scores": target_scores,
        "selectivity": selectivity,
        "candidates_for_rescue": candidates_for_rescue,
        "warnings": warnings,
        "used_dry_run_inputs": detect_dry_run(target_scores, selectivity, docking_summary),
    }


def assemble_candidate_table(inputs: dict[str, Any], max_candidates: int) -> pd.DataFrame:
    """Merge Module 3 chemistry with Module 4 docking/selectivity evidence."""
    molecules = inputs["molecules"].copy()
    target_scores = inputs["target_scores"].copy()
    selectivity = inputs["selectivity"].copy()
    candidates_for_rescue = inputs.get("candidates_for_rescue", pd.DataFrame()).copy()

    if "molecule_id" not in molecules or "molecule_id" not in target_scores or "molecule_id" not in selectivity:
        raise SplicingRescueInputError("Input tables must contain molecule_id columns.")
    target_scores = target_scores.sort_values("target_deltaG", na_position="last").drop_duplicates("molecule_id")
    selectivity = selectivity.drop_duplicates("molecule_id")
    chemistry = molecules.drop_duplicates("molecule_id")

    candidate_ids: set[str]
    if not candidates_for_rescue.empty:
        candidate_ids = set(candidates_for_rescue["molecule_id"].astype(str))
    else:
        candidate_ids = set(selectivity["molecule_id"].astype(str))
    merged = selectivity[selectivity["molecule_id"].astype(str).isin(candidate_ids)].copy()
    if merged.empty:
        merged = selectivity.copy()
    target_cols = [column for column in target_scores.columns if column not in merged.columns or column in {"molecule_id", "pose_file", "receptor_file", "ligand_file", "target_id"}]
    merged = merged.merge(target_scores[target_cols], on="molecule_id", how="left", suffixes=("", "_target"))
    chemistry_cols = [column for column in chemistry.columns if column not in merged.columns or column == "molecule_id"]
    merged = merged.merge(chemistry[chemistry_cols], on="molecule_id", how="left", suffixes=("", "_chem"))
    if "canonical_smiles" not in merged and "canonical_smiles_chem" in merged:
        merged["canonical_smiles"] = merged["canonical_smiles_chem"]
    if "canonical_smiles_chem" in merged:
        merged["canonical_smiles"] = merged["canonical_smiles"].fillna(merged["canonical_smiles_chem"])
    merged["_binding_sort"] = pd.to_numeric(merged.get("target_deltaG", pd.Series(dtype=float)), errors="coerce")
    merged = merged.sort_values("_binding_sort", ascending=True, na_position="last").drop(columns=["_binding_sort"]).head(max_candidates)
    if merged.empty:
        raise SplicingRescueInputError("No molecules can be scored after merging Module 3 and Module 4 tables.")
    if pd.to_numeric(merged.get("target_deltaG", pd.Series(dtype=float)), errors="coerce").isna().all():
        raise SplicingRescueInputError("All docking scores are missing after candidate merge.")
    return merged.reset_index(drop=True)
