"""Matplotlib figure generation for Module 5."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib-cache"))

import pandas as pd


def _plt():
    import matplotlib.pyplot as plt

    return plt


def _numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[column], errors="coerce").dropna()


def _title(text: str, used_dry_run_inputs: bool) -> str:
    if used_dry_run_inputs:
        return f"{text}\nDry-run input analysis - not scientific results"
    return text


def _placeholder(path: Path, title: str, message: str) -> None:
    plt = _plt()
    plt.figure(figsize=(7, 4.5))
    plt.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _histogram(df: pd.DataFrame, column: str, path: Path, title: str, xlabel: str, used_dry_run_inputs: bool) -> None:
    values = _numeric(df, column)
    if values.empty:
        _placeholder(path, _title(title, used_dry_run_inputs), f"No numeric {xlabel} values available.")
        return
    plt = _plt()
    plt.figure(figsize=(7, 4.5))
    plt.hist(values, bins=min(40, max(5, len(values))))
    plt.xlabel(xlabel)
    plt.ylabel("Candidate count")
    plt.title(_title(title, used_dry_run_inputs))
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _scatter(df: pd.DataFrame, x: str, y: str, path: Path, title: str, xlabel: str, ylabel: str, used_dry_run_inputs: bool) -> None:
    x_values = _numeric(df, x)
    y_values = _numeric(df, y)
    both = pd.DataFrame({"x": pd.to_numeric(df.get(x, pd.Series(dtype=float)), errors="coerce"), "y": pd.to_numeric(df.get(y, pd.Series(dtype=float)), errors="coerce")}).dropna()
    if both.empty:
        _placeholder(path, _title(title, used_dry_run_inputs), "Insufficient numeric values for scatter plot.")
        return
    plt = _plt()
    plt.figure(figsize=(7, 4.5))
    plt.scatter(both["x"], both["y"], alpha=0.75)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(_title(title, used_dry_run_inputs))
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def write_figures(
    *,
    final_ranked: pd.DataFrame,
    rescue_scores: pd.DataFrame,
    ablation: pd.DataFrame,
    figure_dir: str | Path,
    config: dict,
    used_dry_run_inputs: bool,
) -> None:
    """Create all required Module 5 figures."""
    if not bool(config.get("figures", {}).get("make_figures", True)):
        return
    out = Path(figure_dir)
    out.mkdir(parents=True, exist_ok=True)
    _histogram(
        rescue_scores,
        "predicted_splicing_rescue_score",
        out / "splicing_rescue_score_distribution.png",
        "Predicted splicing-rescue proxy score distribution",
        "Predicted splicing-rescue proxy score",
        used_dry_run_inputs,
    )
    _histogram(
        final_ranked,
        "srsi_score",
        out / "srsi_score_distribution.png",
        "SRSI score distribution",
        "SRSI score",
        used_dry_run_inputs,
    )
    _scatter(
        final_ranked,
        "target_deltaG",
        "predicted_splicing_rescue_score",
        out / "target_binding_vs_splicing_rescue_scatter.png",
        "Target binding vs predicted splicing-rescue proxy",
        "Target deltaG (lower is stronger)",
        "Predicted splicing-rescue proxy score",
        used_dry_run_inputs,
    )
    _scatter(
        final_ranked,
        "weighted_selectivity_gap",
        "predicted_splicing_rescue_score",
        out / "selectivity_gap_vs_splicing_rescue_scatter.png",
        "Selectivity gap vs predicted splicing-rescue proxy",
        "Weighted selectivity gap",
        "Predicted splicing-rescue proxy score",
        used_dry_run_inputs,
    )
    _scatter(
        final_ranked,
        "off_target_risk_score",
        "srsi_score",
        out / "off_target_risk_vs_srsi_scatter.png",
        "Off-target risk vs SRSI",
        "HuRDA off-target risk proxy",
        "SRSI score",
        used_dry_run_inputs,
    )

    plt = _plt()
    top_n = int(config.get("figures", {}).get("max_candidates_waterfall", 10))
    top = final_ranked.head(top_n).copy()
    if top.empty:
        _placeholder(out / "top_10_candidate_score_breakdown.png", _title("Top candidate score breakdown", used_dry_run_inputs), "No candidates available.")
    else:
        labels = top["molecule_id"].astype(str).tolist()
        rescue = pd.to_numeric(top.get("predicted_splicing_rescue_score", 0), errors="coerce").fillna(0)
        binding = pd.to_numeric(top.get("target_binding_strength_score", 0), errors="coerce").fillna(0)
        selectivity = pd.to_numeric(top.get("weighted_selectivity_gap", 0), errors="coerce").fillna(0)
        selectivity_scaled = (selectivity - selectivity.min()) / max(selectivity.max() - selectivity.min(), 1e-9)
        plt.figure(figsize=(10, 5))
        plt.bar(labels, binding, label="Target binding")
        plt.bar(labels, rescue, bottom=binding, label="Splicing proxy")
        plt.bar(labels, selectivity_scaled, bottom=binding + rescue, label="Selectivity scaled")
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Stacked normalized score components")
        plt.title(_title("Top 10 candidate score breakdown", used_dry_run_inputs))
        plt.legend()
        plt.tight_layout()
        plt.savefig(out / "top_10_candidate_score_breakdown.png", dpi=200)
        plt.close()

    heat_n = int(config.get("figures", {}).get("max_candidates_heatmap", 50))
    heat = final_ranked.head(heat_n).copy()
    heat_cols = [
        "target_binding_strength_score",
        "predicted_splicing_rescue_score",
        "weighted_selectivity_gap",
        "off_target_risk_score",
        "qed",
        "srsi_score",
    ]
    matrix = heat[heat_cols].apply(pd.to_numeric, errors="coerce") if not heat.empty and set(heat_cols).issubset(heat.columns) else pd.DataFrame()
    if matrix.empty:
        _placeholder(out / "top_candidate_heatmap.png", _title("Top candidate heatmap", used_dry_run_inputs), "No complete numeric score matrix available.")
    else:
        matrix = matrix.fillna(matrix.median(numeric_only=True)).fillna(0)
        for column in matrix.columns:
            lo, hi = matrix[column].min(), matrix[column].max()
            matrix[column] = 0.5 if hi == lo else (matrix[column] - lo) / (hi - lo)
        plt.figure(figsize=(8, max(4.5, len(matrix) * 0.18)))
        plt.imshow(matrix.values, aspect="auto", interpolation="nearest")
        plt.yticks(range(len(heat)), heat["molecule_id"].astype(str), fontsize=6)
        plt.xticks(range(len(heat_cols)), heat_cols, rotation=45, ha="right", fontsize=8)
        plt.colorbar(label="Scaled value")
        plt.title(_title("Top candidate heatmap", used_dry_run_inputs))
        plt.tight_layout()
        plt.savefig(out / "top_candidate_heatmap.png", dpi=200)
        plt.close()

    if ablation.empty:
        _placeholder(out / "ablation_rank_shift_plot.png", _title("Ablation rank shift", used_dry_run_inputs), "Ablation analysis disabled or unavailable.")
    else:
        subset = ablation.head(top_n)
        plt.figure(figsize=(8, 4.5))
        plt.bar(subset["molecule_id"].astype(str), pd.to_numeric(subset["rank_change_binding_to_full"], errors="coerce").fillna(0))
        plt.axhline(0, color="black", linewidth=0.8)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Binding-only rank minus full SRSI rank")
        plt.title(_title("Ablation rank shift plot", used_dry_run_inputs))
        plt.tight_layout()
        plt.savefig(out / "ablation_rank_shift_plot.png", dpi=200)
        plt.close()

    if "confidence_level" in final_ranked and not final_ranked.empty:
        counts = final_ranked["confidence_level"].value_counts()
        plt.figure(figsize=(7, 4.5))
        plt.bar(counts.index.astype(str), counts.values)
        plt.xlabel("Confidence level")
        plt.ylabel("Candidate count")
        plt.title(_title("Candidate confidence summary", used_dry_run_inputs))
        plt.tight_layout()
        plt.savefig(out / "candidate_confidence_summary.png", dpi=200)
        plt.close()
    else:
        _placeholder(out / "candidate_confidence_summary.png", _title("Candidate confidence summary", used_dry_run_inputs), "No confidence labels available.")

