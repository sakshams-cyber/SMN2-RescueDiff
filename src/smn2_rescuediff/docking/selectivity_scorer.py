"""Target-vs-decoy selectivity scoring."""

from __future__ import annotations

from typing import Any

import pandas as pd


SELECTIVITY_COLUMNS = [
    "molecule_id",
    "canonical_smiles",
    "target_deltaG",
    "best_decoy_deltaG",
    "mean_decoy_deltaG",
    "median_decoy_deltaG",
    "weighted_mean_decoy_deltaG",
    "worst_case_decoy_category",
    "num_decoys_tested",
    "num_decoys_with_valid_scores",
    "selectivity_gap_best_decoy",
    "selectivity_gap_mean_decoy",
    "weighted_selectivity_gap",
    "percent_decoys_stronger_than_target",
    "off_target_risk_score",
    "target_binding_rank",
    "selectivity_rank",
    "advance_to_splicing_rescue",
    "run_mode",
    "is_dry_run_score",
    "notes",
]


def _valid_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").dropna()


def calculate_selectivity_scores(target_scores: pd.DataFrame, decoy_scores: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Calculate molecule-level target/decoy selectivity metrics."""
    scoring = config.get("selectivity_scoring", {})
    rows: list[dict[str, Any]] = []
    for _, target in target_scores.iterrows():
        molecule_id = target["molecule_id"]
        target_delta = pd.to_numeric(pd.Series([target.get("target_deltaG", None)]), errors="coerce").iloc[0]
        subset = decoy_scores[decoy_scores["molecule_id"] == molecule_id].copy() if not decoy_scores.empty else pd.DataFrame()
        valid = subset[pd.to_numeric(subset.get("decoy_deltaG", pd.Series(dtype=float)), errors="coerce").notna()].copy()
        decoy_values = _valid_numeric(valid["decoy_deltaG"]) if not valid.empty else pd.Series(dtype=float)
        num_tested = len(subset)
        num_valid = len(decoy_values)
        if pd.isna(target_delta):
            row = {
                "molecule_id": molecule_id,
                "canonical_smiles": target.get("canonical_smiles", "NA"),
                "target_deltaG": "NA",
                "best_decoy_deltaG": "NA",
                "mean_decoy_deltaG": "NA",
                "median_decoy_deltaG": "NA",
                "weighted_mean_decoy_deltaG": "NA",
                "worst_case_decoy_category": "NA",
                "num_decoys_tested": num_tested,
                "num_decoys_with_valid_scores": num_valid,
                "selectivity_gap_best_decoy": "NA",
                "selectivity_gap_mean_decoy": "NA",
                "weighted_selectivity_gap": "NA",
                "percent_decoys_stronger_than_target": "NA",
                "off_target_risk_score": 1.0,
                "target_binding_rank": "NA",
                "selectivity_rank": "NA",
                "advance_to_splicing_rescue": False,
                "run_mode": target.get("run_mode", "NA"),
                "is_dry_run_score": target.get("is_dry_run_score", False),
                "notes": "No valid target docking score.",
            }
            rows.append(row)
            continue
        if num_valid:
            best_decoy = float(decoy_values.min())
            mean_decoy = float(decoy_values.mean())
            median_decoy = float(decoy_values.median())
            weights = pd.to_numeric(valid.get("decoy_biological_weight", pd.Series([1.0] * len(valid))), errors="coerce").fillna(1.0)
            weighted_mean = float((pd.to_numeric(valid["decoy_deltaG"], errors="coerce") * weights).sum() / max(weights.sum(), 1e-9))
            worst_idx = pd.to_numeric(valid["decoy_deltaG"], errors="coerce").idxmin()
            worst_category = str(valid.loc[worst_idx, "decoy_category"])
            gap_best = best_decoy - float(target_delta)
            gap_mean = mean_decoy - float(target_delta)
            weighted_gap = weighted_mean - float(target_delta)
            stronger = float((decoy_values < float(target_delta)).sum() / num_valid * 100.0)
            gap_component = max(0.0, min(1.0, (1.5 - gap_best) / 3.0))
            stronger_component = stronger / 100.0
            strong_decoy_component = max(0.0, min(1.0, (-best_decoy - 5.0) / 4.0))
            risk = max(0.0, min(1.0, 0.45 * gap_component + 0.35 * stronger_component + 0.20 * strong_decoy_component))
        else:
            best_decoy = mean_decoy = median_decoy = weighted_mean = "NA"
            worst_category = "NA"
            gap_best = gap_mean = weighted_gap = "NA"
            stronger = "NA"
            risk = 0.5
        advance = (
            float(target_delta) <= float(scoring.get("advance_target_deltaG_threshold", -7.0))
            and isinstance(weighted_gap, float)
            and weighted_gap >= float(scoring.get("advance_weighted_gap_threshold", 1.0))
            and risk <= float(scoring.get("advance_off_target_risk_threshold", 0.40))
            and num_valid >= int(scoring.get("advance_min_decoys_with_scores", 20))
        )
        note = "Advance flag based on simulated values only." if bool(target.get("is_dry_run_score", False)) else "Computed from docking scores."
        rows.append(
            {
                "molecule_id": molecule_id,
                "canonical_smiles": target.get("canonical_smiles", "NA"),
                "target_deltaG": round(float(target_delta), 3),
                "best_decoy_deltaG": round(best_decoy, 3) if isinstance(best_decoy, float) else best_decoy,
                "mean_decoy_deltaG": round(mean_decoy, 3) if isinstance(mean_decoy, float) else mean_decoy,
                "median_decoy_deltaG": round(median_decoy, 3) if isinstance(median_decoy, float) else median_decoy,
                "weighted_mean_decoy_deltaG": round(weighted_mean, 3) if isinstance(weighted_mean, float) else weighted_mean,
                "worst_case_decoy_category": worst_category,
                "num_decoys_tested": num_tested,
                "num_decoys_with_valid_scores": num_valid,
                "selectivity_gap_best_decoy": round(gap_best, 3) if isinstance(gap_best, float) else gap_best,
                "selectivity_gap_mean_decoy": round(gap_mean, 3) if isinstance(gap_mean, float) else gap_mean,
                "weighted_selectivity_gap": round(weighted_gap, 3) if isinstance(weighted_gap, float) else weighted_gap,
                "percent_decoys_stronger_than_target": round(stronger, 3) if isinstance(stronger, float) else stronger,
                "off_target_risk_score": round(risk, 6),
                "target_binding_rank": "NA",
                "selectivity_rank": "NA",
                "advance_to_splicing_rescue": bool(advance),
                "run_mode": target.get("run_mode", "NA"),
                "is_dry_run_score": target.get("is_dry_run_score", False),
                "notes": note,
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(columns=SELECTIVITY_COLUMNS)
    result["_target_sort"] = pd.to_numeric(result["target_deltaG"], errors="coerce")
    result["_selectivity_sort"] = pd.to_numeric(result["weighted_selectivity_gap"], errors="coerce")
    result["target_binding_rank"] = result["_target_sort"].rank(method="min", ascending=True, na_option="bottom").astype(int)
    result["selectivity_rank"] = result["_selectivity_sort"].rank(method="min", ascending=False, na_option="bottom").astype(int)
    result = result.drop(columns=["_target_sort", "_selectivity_sort"])
    return result[SELECTIVITY_COLUMNS]

