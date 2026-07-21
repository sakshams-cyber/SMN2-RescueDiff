"""Static matplotlib figures for docking and counter-screening."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


def _numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[column], errors="coerce").dropna()


def _placeholder(path: Path, title: str, message: str) -> None:
    plt.figure(figsize=(7, 4.5))
    plt.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def write_figures(
    target_scores: pd.DataFrame,
    decoy_scores: pd.DataFrame,
    selectivity: pd.DataFrame,
    live_log_file: str | Path,
    figure_dir: str | Path,
    config: dict[str, Any],
) -> None:
    """Write all required Module 4 static figures."""
    out = Path(figure_dir)
    out.mkdir(parents=True, exist_ok=True)
    run_mode = str(config.get("run", {}).get("run_mode", "dry_run"))
    label = "Dry-run simulated scores" if run_mode == "dry_run" else "Docking scores"

    for df, column, filename, title, xlabel in [
        (target_scores, "target_deltaG", "smn2_docking_score_distribution.png", "SMN2 target docking score distribution", "SMN2 target deltaG"),
        (decoy_scores, "decoy_deltaG", "decoy_docking_score_distribution.png", "Decoy docking score distribution", "Decoy deltaG"),
        (selectivity, "weighted_selectivity_gap", "selectivity_gap_distribution.png", "Weighted selectivity gap distribution", "Weighted selectivity gap"),
    ]:
        values = _numeric(df, column)
        if values.empty:
            _placeholder(out / filename, title, "No numeric scores available.")
        else:
            plt.figure(figsize=(7, 4.5))
            values.hist(bins=40)
            plt.xlabel(xlabel)
            plt.ylabel("Count")
            plt.title(f"{title}\n{label}")
            plt.tight_layout()
            plt.savefig(out / filename, dpi=200)
            plt.close()

    x = _numeric(selectivity, "target_deltaG")
    y = _numeric(selectivity, "mean_decoy_deltaG")
    if x.empty or y.empty:
        _placeholder(out / "target_vs_decoy_binding_scatter.png", "Target vs decoy binding scatter", "No paired target/decoy scores available.")
    else:
        paired = selectivity.copy()
        plt.figure(figsize=(7, 5))
        plt.scatter(pd.to_numeric(paired["target_deltaG"], errors="coerce"), pd.to_numeric(paired["mean_decoy_deltaG"], errors="coerce"), alpha=0.75)
        plt.xlabel("SMN2 target deltaG")
        plt.ylabel("Mean decoy deltaG")
        plt.title(f"Target vs decoy binding\n{label}")
        plt.tight_layout()
        plt.savefig(out / "target_vs_decoy_binding_scatter.png", dpi=200)
        plt.close()

    if decoy_scores.empty or "decoy_category" not in decoy_scores:
        _placeholder(out / "off_target_risk_by_decoy_category.png", "Off-target risk by decoy category", "No decoy scores available.")
    else:
        grouped = decoy_scores.copy()
        grouped["decoy_deltaG"] = pd.to_numeric(grouped["decoy_deltaG"], errors="coerce")
        means = grouped.groupby("decoy_category")["decoy_deltaG"].mean().dropna().sort_values()
        plt.figure(figsize=(8, 4.5))
        plt.bar(means.index.astype(str), means.values)
        plt.xticks(rotation=35, ha="right", fontsize=8)
        plt.ylabel("Mean decoy deltaG")
        plt.title(f"Average decoy binding by category\n{label}")
        plt.tight_layout()
        plt.savefig(out / "off_target_risk_by_decoy_category.png", dpi=200)
        plt.close()

    heat_cfg = config.get("figures", {})
    max_molecules = int(heat_cfg.get("max_heatmap_molecules", 50))
    max_decoys = int(heat_cfg.get("max_heatmap_decoys", 50))
    if decoy_scores.empty:
        _placeholder(out / "top_candidate_binding_heatmap.png", "Top candidate binding heatmap", "No decoy scores available.")
    else:
        top_mols = selectivity.sort_values("weighted_selectivity_gap", ascending=False).head(max_molecules)["molecule_id"].tolist()
        top_decoys = decoy_scores["decoy_id"].drop_duplicates().head(max_decoys).tolist()
        matrix = decoy_scores[decoy_scores["molecule_id"].isin(top_mols) & decoy_scores["decoy_id"].isin(top_decoys)]
        pivot = matrix.pivot_table(index="molecule_id", columns="decoy_id", values="decoy_deltaG", aggfunc="min")
        if pivot.empty:
            _placeholder(out / "top_candidate_binding_heatmap.png", "Top candidate binding heatmap", "No matrix values available.")
        else:
            plt.figure(figsize=(10, 6))
            plt.imshow(pivot.fillna(0).values, aspect="auto")
            plt.yticks(range(len(pivot.index)), pivot.index, fontsize=6)
            plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=90, fontsize=5)
            plt.colorbar(label="Decoy deltaG")
            plt.title(f"Top candidate decoy binding heatmap\n{label}")
            plt.tight_layout()
            plt.savefig(out / "top_candidate_binding_heatmap.png", dpi=200)
            plt.close()

    log_path = Path(live_log_file)
    scores = []
    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event_type") == "job_completed" and event.get("current_score") is not None:
                scores.append(float(event["current_score"]))
    if not scores:
        _placeholder(out / "docking_progress_over_time.png", "Docking progress over time", "No completed score events available.")
    else:
        plt.figure(figsize=(8, 4.5))
        plt.plot(range(1, len(scores) + 1), scores)
        plt.xlabel("Completed docking jobs")
        plt.ylabel("Latest deltaG")
        plt.title(f"Docking progress over time\n{label}")
        plt.tight_layout()
        plt.savefig(out / "docking_progress_over_time.png", dpi=200)
        plt.close()

