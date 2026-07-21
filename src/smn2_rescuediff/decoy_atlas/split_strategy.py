"""Reproducible train/validation/test and holdout splitting."""

from __future__ import annotations

import random
from typing import Any

import pandas as pd


def assign_splits(decoys: pd.DataFrame, config: dict[str, Any], random_seed: int) -> pd.DataFrame:
    """Assign train/validation/test labels and holdout groups."""
    if decoys.empty:
        return decoys.copy()
    split_cfg = config.get("splits", {})
    train_fraction = float(split_cfg.get("train_fraction", 0.70))
    validation_fraction = float(split_cfg.get("validation_fraction", 0.10))
    result = decoys.copy()
    result["train_val_test_split"] = "train"

    rng = random.Random(random_seed)
    for _, group in result.groupby("decoy_category", sort=False):
        indices = list(group.index)
        rng.shuffle(indices)
        n_total = len(indices)
        n_train = int(round(n_total * train_fraction))
        n_validation = int(round(n_total * validation_fraction))
        validation_indices = indices[n_train : n_train + n_validation]
        test_indices = indices[n_train + n_validation :]
        result.loc[validation_indices, "train_val_test_split"] = "validation"
        result.loc[test_indices, "train_val_test_split"] = "test"

    result["holdout_group"] = "none"
    if bool(split_cfg.get("create_hard_decoy_holdout", True)):
        hard = result[result["decoy_category"] == "smn2_like_hard_decoy"].sort_values("smn2_similarity_score", ascending=False)
        if not hard.empty:
            n_holdout = max(1, int(round(len(hard) * 0.10)))
            result.loc[hard.head(n_holdout).index, "holdout_group"] = "hard_smn2_like_holdout"

    if bool(split_cfg.get("create_chromosome_holdout", True)):
        eligible = result[(result["holdout_group"] == "none") & (result["chromosome"].astype(str) != "NA")]
        chromosomes = sorted(eligible["chromosome"].dropna().astype(str).unique())
        if chromosomes:
            chromosome = rng.choice(chromosomes)
            result.loc[(result["holdout_group"] == "none") & (result["chromosome"] == chromosome), "holdout_group"] = "chromosome_holdout"

    if bool(split_cfg.get("create_gene_holdout", True)):
        eligible = result[(result["holdout_group"] == "none") & (~result["gene_name"].isin(["NA", "SMN2", ""]))]
        genes = sorted(eligible["gene_name"].dropna().astype(str).unique())
        if genes:
            n_genes = max(1, int(round(len(genes) * 0.03)))
            holdout_genes = set(rng.sample(genes, k=min(n_genes, len(genes))))
            result.loc[(result["holdout_group"] == "none") & (result["gene_name"].isin(holdout_genes)), "holdout_group"] = "gene_holdout"

    return result

