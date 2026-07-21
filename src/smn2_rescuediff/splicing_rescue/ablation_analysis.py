"""Ablation analyses for Module 5 SRSI ranking."""

from __future__ import annotations

import pandas as pd

from .splicing_feature_scorer import numeric_series, zscore


ABLATION_COLUMNS = [
    "molecule_id",
    "binding_only_rank",
    "binding_plus_selectivity_rank",
    "binding_plus_splicing_rescue_rank",
    "full_srsi_rank",
    "rank_change_binding_to_full",
    "target_deltaG",
    "weighted_selectivity_gap",
    "predicted_splicing_rescue_score",
    "srsi_score",
    "interpretation",
]


def _rank_desc(values: pd.Series) -> pd.Series:
    return values.rank(method="min", ascending=False, na_option="bottom").astype(int)


def run_ablation_analysis(final_ranked: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Compare simple binding-only rankings with full SRSI rankings."""
    if not bool(config.get("ablation_analysis", {}).get("enabled", True)):
        return pd.DataFrame(columns=ABLATION_COLUMNS)
    work = final_ranked.copy()
    binding_strength = -numeric_series(work, "target_deltaG", default=0.0)
    selectivity = numeric_series(work, "weighted_selectivity_gap", default=0.0)
    rescue = numeric_series(work, "predicted_splicing_rescue_score", default=0.0)
    risk = numeric_series(work, "off_target_risk_score", default=1.0)
    work["binding_only_rank"] = _rank_desc(binding_strength)
    work["binding_plus_selectivity_rank"] = _rank_desc(zscore(binding_strength) + zscore(selectivity) - zscore(risk))
    work["binding_plus_splicing_rescue_rank"] = _rank_desc(zscore(binding_strength) + zscore(rescue))
    work["full_srsi_rank"] = work.get("final_rank", pd.Series(range(1, len(work) + 1), index=work.index)).astype(int)
    work["rank_change_binding_to_full"] = work["binding_only_rank"] - work["full_srsi_rank"]

    def interpret(row: pd.Series) -> str:
        change = int(row.get("rank_change_binding_to_full", 0))
        if change > 10:
            return "Full SRSI ranked this molecule higher after selectivity, splicing proxy, and chemical evidence were included."
        if change < -10:
            return "Full SRSI ranked this molecule lower despite binding because other evidence reduced priority."
        return "Full SRSI ranking was broadly similar to binding-only ranking for this molecule."

    work["interpretation"] = [interpret(row) for _, row in work.iterrows()]
    for column in ABLATION_COLUMNS:
        if column not in work:
            work[column] = "NA"
    return work[ABLATION_COLUMNS].sort_values("full_srsi_rank").reset_index(drop=True)

