"""Matplotlib figures for the advanced molecule-generation module."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib-cache"))

import pandas as pd

from .scaffold_analysis import scaffold_frequency


def _plt():
    import matplotlib.pyplot as plt

    return plt


def _numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[column], errors="coerce").dropna()


def _placeholder(path: Path, title: str, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt = _plt()
    plt.figure(figsize=(7, 4.5))
    plt.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def write_generation_figures(
    *,
    training_log: pd.DataFrame,
    rl_log: pd.DataFrame,
    valid: pd.DataFrame,
    filtered: pd.DataFrame,
    docking_candidates: pd.DataFrame,
    raw_count: int,
    diversity_values: list[float],
    figure_dir: str | Path,
    toy_dataset: bool,
) -> None:
    """Write all required Module 3 figures."""
    plt = _plt()
    out = Path(figure_dir)
    out.mkdir(parents=True, exist_ok=True)
    suffix = "\nToy dataset test run - not scientific generation" if toy_dataset else ""

    for column, filename, title in [
        ("train_loss", "training_loss_curve.png", "Training loss curve"),
        ("validation_loss", "validation_loss_curve.png", "Validation loss curve"),
    ]:
        if column in training_log and training_log[column].notna().any():
            plt.figure(figsize=(7, 4.5))
            plt.plot(training_log["epoch"], training_log[column], marker="o")
            plt.xlabel("Epoch")
            plt.ylabel(column.replace("_", " "))
            plt.title(title + suffix)
            plt.tight_layout()
            plt.savefig(out / filename, dpi=200)
            plt.close()
        else:
            _placeholder(out / filename, title + suffix, "Training was not run or no loss values were available.")

    if "mean_reward" in rl_log and rl_log["mean_reward"].notna().any():
        plt.figure(figsize=(7, 4.5))
        plt.plot(rl_log["episode"], rl_log["mean_reward"], marker="o")
        plt.xlabel("Episode")
        plt.ylabel("Mean reward")
        plt.title("RL chemical-plausibility reward curve" + suffix)
        plt.tight_layout()
        plt.savefig(out / "rl_reward_curve.png", dpi=200)
        plt.close()
    else:
        _placeholder(out / "rl_reward_curve.png", "RL reward curve" + suffix, "Reward fine-tuning was disabled or not run.")

    unique_valid = valid["canonical_smiles"].nunique() if "canonical_smiles" in valid else 0
    novel = int((filtered.get("is_novel_vs_training", pd.Series(dtype=bool)) == True).sum()) if not filtered.empty else 0  # noqa: E712
    labels = ["raw", "valid", "unique_valid", "novel_filtered", "filtered", "docking_export"]
    values = [raw_count, len(valid), unique_valid, novel, len(filtered), len(docking_candidates)]
    plt.figure(figsize=(8, 4.5))
    plt.bar(labels, values)
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Count")
    plt.title("Validity, uniqueness, novelty, and export summary" + suffix)
    plt.tight_layout()
    plt.savefig(out / "validity_uniqueness_novelty_summary.png", dpi=200)
    plt.close()

    for column, filename, title, xlabel in [
        ("molecular_weight", "molecular_weight_distribution.png", "Molecular weight distribution", "Molecular weight"),
        ("logp", "logp_distribution.png", "LogP distribution", "LogP"),
        ("qed", "qed_distribution.png", "QED distribution", "QED"),
        ("rna_ligand_like_score", "rna_ligand_like_score_distribution.png", "RNA-ligand-like score distribution", "RNA-ligand-like score"),
        ("pre_docking_candidate_score", "pre_docking_candidate_score_distribution.png", "Pre-docking candidate score distribution", "Pre-docking candidate score"),
    ]:
        values = _numeric(filtered, column)
        if values.empty:
            _placeholder(out / filename, title + suffix, f"No numeric {xlabel} values available.")
        else:
            plt.figure(figsize=(7, 4.5))
            values.hist(bins=40)
            plt.xlabel(xlabel)
            plt.ylabel("Count")
            plt.title(title + suffix)
            plt.tight_layout()
            plt.savefig(out / filename, dpi=200)
            plt.close()

    scaffolds = scaffold_frequency(filtered).head(20)
    if scaffolds.empty:
        _placeholder(out / "scaffold_frequency_top20.png", "Top 20 scaffold frequency" + suffix, "No scaffolds available.")
    else:
        plt.figure(figsize=(9, 4.8))
        plt.bar(scaffolds["bemis_murcko_scaffold"].astype(str), scaffolds["count"])
        plt.xticks(rotation=60, ha="right", fontsize=7)
        plt.ylabel("Count")
        plt.title("Top 20 Bemis-Murcko scaffolds" + suffix)
        plt.tight_layout()
        plt.savefig(out / "scaffold_frequency_top20.png", dpi=200)
        plt.close()

    if diversity_values:
        plt.figure(figsize=(7, 4.5))
        plt.hist(diversity_values, bins=40)
        plt.xlabel("Pairwise Tanimoto similarity")
        plt.ylabel("Sampled pair count")
        plt.title("Diversity distribution" + suffix)
        plt.tight_layout()
        plt.savefig(out / "diversity_distribution.png", dpi=200)
        plt.close()
    else:
        _placeholder(out / "diversity_distribution.png", "Diversity distribution" + suffix, "Diversity calculation was disabled or insufficient molecules were available.")
