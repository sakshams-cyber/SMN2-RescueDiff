"""Separate benchmark-control reference scoring for Module 5."""

from __future__ import annotations

from typing import Any

import pandas as pd


BENCHMARK_REFERENCE_COLUMNS = [
    "benchmark_control_name",
    "canonical_smiles",
    "matched_molecule_id",
    "target_deltaG",
    "weighted_mean_decoy_deltaG",
    "weighted_selectivity_gap",
    "selectivity_gap",
    "off_target_risk_score",
    "num_decoys_with_scores",
    "num_decoys_with_valid_scores",
    "srsi_score",
    "confidence_level",
    "is_dry_run_score",
    "reference_status",
    "notes",
]


def _matches(df: pd.DataFrame, canonical_smiles: str, control_name: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    masks = []
    if "canonical_smiles" in df:
        masks.append(df["canonical_smiles"].astype(str) == canonical_smiles)
    if "molecule_id" in df:
        masks.append(df["molecule_id"].astype(str).str.lower() == control_name.lower())
    if not masks:
        return pd.DataFrame()
    mask = masks[0]
    for item in masks[1:]:
        mask = mask | item
    return df[mask].copy()


def _best_row(df: pd.DataFrame, sort_column: str = "target_deltaG") -> pd.Series | None:
    if df.empty:
        return None
    if sort_column in df:
        return df.assign(_sort=pd.to_numeric(df[sort_column], errors="coerce")).sort_values("_sort", ascending=True, na_position="last").iloc[0]
    return df.iloc[0]


def _value(row: pd.Series | None, column: str) -> Any:
    if row is None:
        return pd.NA
    return row.get(column, pd.NA)


def assemble_benchmark_control_scores(inputs: dict[str, Any], final_ranked: pd.DataFrame) -> pd.DataFrame:
    """Build a separate reference-baseline table for held-out benchmark controls."""
    controls = inputs.get("benchmark_controls", pd.DataFrame()).copy()
    if controls.empty:
        return pd.DataFrame(columns=BENCHMARK_REFERENCE_COLUMNS)

    target_scores = inputs.get("target_scores", pd.DataFrame()).copy()
    selectivity = inputs.get("selectivity", pd.DataFrame()).copy()
    ranked = final_ranked.copy()
    rows: list[dict[str, Any]] = []
    for _, control in controls.iterrows():
        name = str(control.get("benchmark_control_name", "benchmark_control"))
        canonical = str(control.get("canonical_smiles", ""))
        target_row = _best_row(_matches(target_scores, canonical, name), "target_deltaG")
        selectivity_row = _best_row(_matches(selectivity, canonical, name), "weighted_selectivity_gap")
        ranked_row = _best_row(_matches(ranked, canonical, name), "final_rank")
        matched_id = _value(target_row, "molecule_id")
        if pd.isna(matched_id) or str(matched_id) in {"", "NA", "nan"}:
            matched_id = _value(selectivity_row, "molecule_id")
        if pd.isna(matched_id) or str(matched_id) in {"", "NA", "nan"}:
            matched_id = _value(ranked_row, "molecule_id")

        has_scores = target_row is not None or selectivity_row is not None or ranked_row is not None
        rows.append(
            {
                "benchmark_control_name": name,
                "canonical_smiles": canonical,
                "matched_molecule_id": matched_id,
                "target_deltaG": _value(target_row, "target_deltaG"),
                "weighted_mean_decoy_deltaG": _value(selectivity_row, "weighted_mean_decoy_deltaG"),
                "weighted_selectivity_gap": _value(selectivity_row, "weighted_selectivity_gap"),
                "selectivity_gap": _value(selectivity_row, "selectivity_gap"),
                "off_target_risk_score": _value(selectivity_row, "off_target_risk_score"),
                "num_decoys_with_scores": _value(selectivity_row, "num_decoys_with_scores"),
                "num_decoys_with_valid_scores": _value(selectivity_row, "num_decoys_with_valid_scores"),
                "srsi_score": _value(ranked_row, "srsi_score"),
                "confidence_level": _value(ranked_row, "confidence_level"),
                "is_dry_run_score": _value(target_row, "is_dry_run_score"),
                "reference_status": "scored_reference_baseline" if has_scores else "no_module4_scores_available",
                "notes": (
                    "Held-out benchmark control; computational reference values only."
                    if has_scores
                    else "Held-out benchmark control; no Module 4 docking/selectivity score was available."
                ),
            }
        )
    return pd.DataFrame(rows, columns=BENCHMARK_REFERENCE_COLUMNS)
