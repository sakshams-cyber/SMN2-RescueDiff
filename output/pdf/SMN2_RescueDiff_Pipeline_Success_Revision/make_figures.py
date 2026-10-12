#!/usr/bin/env python3
"""Regenerate manuscript figures from the bundled audit tables."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "figure_data"
FIGURES = ROOT / "figures"

NAVY = "#173B57"
TEAL = "#177E89"
LIGHT_TEAL = "#DCEFF0"
GREEN = "#3B8C6E"
LIGHT_GREEN = "#E5F2EB"
GOLD = "#C5842B"
ORANGE = "#E07A2E"
INK = "#17202A"
MUTED = "#5B6872"
GRID = "#D6DEE3"


def save(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def pipeline_figure() -> None:
    modules = [
        ("Module 1", "SMN2 target definition", "Genome + GTF\nTarget windows", "Fixed target profile\n5 target conformers"),
        ("Module 2", "HuRDA decoy atlas", "Target profile\nHuman RNA motifs", ">180,000 sequence decoys\n100 structural receptors"),
        ("Module 3", "Molecule generation", "ChEMBL + ZINC\nControls held out", "49,660 unique strings\n40,477 filter-pass"),
        ("Module 4", "Structural screening", "1,000-molecule export\nTarget + hard decoys", "2,500 target jobs\n14,259 decoy jobs"),
        ("Module 5", "SRSI + ablation", "Docking + chemistry\nPose/contact proxies", "147 x 97 common matrix\nAuditable shortlist"),
        ("Validation", "Controls + redocking", "Matched ligands\n4 experimental complexes", "Target-score enrichment\nCalibration boundary"),
    ]
    fig, ax = plt.subplots(figsize=(11.2, 4.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.97, "SMN2-RescueDiff: staged inputs, outputs, and audited scale", ha="center", va="top", fontsize=14, fontweight="bold", color=NAVY)
    xs = [0.02, 0.345, 0.67]
    ys = [0.54, 0.08]
    w, h = 0.305, 0.34
    for idx, (module, name, input_text, output_text) in enumerate(modules):
        row, col = divmod(idx, 3)
        x, y = xs[col], ys[row]
        ax.add_patch(Rectangle((x, y), w, h, facecolor="white", edgecolor=NAVY, linewidth=1.2))
        ax.add_patch(Rectangle((x, y + h - 0.09), w, 0.09, facecolor=NAVY, edgecolor=NAVY))
        ax.text(x + 0.012, y + h - 0.045, f"{module}: {name}", color="white", fontsize=8.8, fontweight="bold", va="center")
        ax.add_patch(Rectangle((x + 0.012, y + 0.13), w - 0.024, 0.105, facecolor=LIGHT_TEAL, edgecolor=TEAL, linewidth=0.8))
        ax.text(x + 0.022, y + 0.215, "INPUT", color=TEAL, fontsize=7.6, fontweight="bold", va="top")
        ax.text(x + 0.022, y + 0.19, input_text, color=INK, fontsize=8.1, va="top", linespacing=1.15)
        ax.add_patch(Rectangle((x + 0.012, y + 0.012), w - 0.024, 0.105, facecolor=LIGHT_GREEN, edgecolor=GREEN, linewidth=0.8))
        ax.text(x + 0.022, y + 0.097, "OUTPUT", color=GREEN, fontsize=7.6, fontweight="bold", va="top")
        ax.text(x + 0.022, y + 0.072, output_text, color=INK, fontsize=8.1, va="top", linespacing=1.15)
    for x0, x1, y in [(0.327, 0.345, 0.71), (0.652, 0.67, 0.71), (0.327, 0.345, 0.25), (0.652, 0.67, 0.25)]:
        ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=12, linewidth=1.5, color=ORANGE))
    ax.add_patch(FancyArrowPatch((0.822, 0.54), (0.822, 0.42), arrowstyle="-|>", mutation_scale=12, linewidth=1.5, color=ORANGE))
    save(fig, "Fig_pipeline")


def generation_figure() -> None:
    audit = json.loads((DATA / "generation_statistics_audit.json").read_text())
    values = [
        audit["training"]["unique_rows_used"],
        audit["generation"]["unique_generated_strings_emitted"],
        audit["generation"]["filtered_molecules"],
        audit["generation"]["selected_for_docking"],
        500,
    ]
    labels = ["Valid unique training", "Unique generated strings", "Chemical filter-pass", "Docking export", "5-conformer campaign"]
    palette = [NAVY, "#245C7A", TEAL, GREEN, GOLD]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    y = np.arange(len(labels))
    ax.barh(y, values, color=palette, height=0.62)
    ax.set_xscale("log")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Observed molecule count (log scale)")
    ax.set_title("Audited generation-to-docking progression", loc="left", fontweight="bold", color=NAVY)
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for yi, value in zip(y, values, strict=True):
        ax.text(value * 1.08, yi, f"{value:,}", va="center", fontsize=9, color=INK)
    ax.spines[["top", "right", "left"]].set_visible(False)
    save(fig, "Fig_generation_audit")


def _desirability_down(value: float, target: float, upper: float) -> float:
    return float(np.clip((upper - value) / (upper - target), 0, 1))


def _desirability_up(value: float, lower: float, target: float) -> float:
    return float(np.clip((value - lower) / (target - lower), 0, 1))


def stage_c_figure() -> None:
    df = pd.read_csv(DATA / "top40_for_ligand_decoy_control_audit.tsv", sep="\t").head(10)
    values = df[["mean_target_deltaG", "weighted_mean_selectivity_gap", "conservative_selectivity_gap", "ensemble_off_target_risk_score"]].to_numpy(float)
    desir = np.column_stack([
        [_desirability_down(v, -10, -6) for v in values[:, 0]],
        [_desirability_up(v, 0, 8) for v in values[:, 1]],
        [_desirability_up(v, -3, 1) for v in values[:, 2]],
        [_desirability_down(v, 0.30, 0.80) for v in values[:, 3]],
    ])
    fig, ax = plt.subplots(figsize=(5.3, 4.2))
    image = ax.imshow(desir, cmap=colors.LinearSegmentedColormap.from_list("audit", ["#F5F7F7", "#8FC3C7", "#0B4F5C"]), vmin=0, vmax=1, aspect="auto")
    labels = ["Target Vina\nscore down", "Weighted\ngap up", "Conservative\ngap up", "Off-target\nrisk down"]
    ax.set_xticks(np.arange(4), labels)
    ax.set_yticks(np.arange(10), df["molecule_id"].str.replace("MolGen_", "", regex=False))
    ax.set_ylabel("Candidate ID")
    ax.set_title("Stage C selectivity profile: top ten", loc="left", fontweight="bold", color=NAVY)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j]:.3f}", ha="center", va="center", fontsize=7.6, color="white" if desir[i, j] > 0.48 else INK)
    cbar = fig.colorbar(image, ax=ax, orientation="horizontal", fraction=0.07, pad=0.16)
    cbar.set_label("Prespecified desirability", fontsize=8)
    ax.tick_params(axis="both", length=0, labelsize=8)
    save(fig, "Fig_stageC_top10_heatmap")


def _interval_desirability(value: float, low: float, high: float, alpha: float, beta: float) -> float:
    if low <= value <= high:
        return 1.0
    if value < low:
        return 0.0 if alpha == 0 else float(np.clip((value - (low - alpha)) / alpha, 0, 1))
    return float(np.clip(((high + beta) - value) / beta, 0, 1))


def benchmark_figure() -> None:
    names = ["Risdiplam", "Branaplam", "MolGen_038292"]
    row_names = ["Target Vina score", "Weighted gap", "Conservative gap", "Off-target risk", "Stronger-decoy fraction", "QED", "Molecular weight", "cLogP", "H-bond donors", "H-bond acceptors", "Rotatable bonds", "tPSA", "Overall desirability"]
    values = np.array([
        [-8.613, -8.321, -8.929], [6.309, 6.071, 6.622], [-0.980, -1.240, -1.017],
        [0.664, 0.600, 0.581], [0.124, 0.165, 0.093], [0.590, 0.622, 0.498],
        [401.5, 393.5, 504.0], [1.51, 2.70, 4.41], [1, 3, 6], [6, 6, 11],
        [2, 4, 9], [78.1, 95.9, 152.0], [0.656, 0.663, 0.626],
    ], dtype=float)
    desir = np.zeros_like(values)
    for j in range(3):
        desir[:, j] = [
            _desirability_down(values[0, j], -10, -6), _desirability_up(values[1, j], 0, 8),
            _desirability_up(values[2, j], -3, 1), _desirability_down(values[3, j], .3, .8),
            _desirability_down(values[4, j], .05, .5), _desirability_up(values[5, j], .3, .8),
            _interval_desirability(values[6, j], 220, 520, 70, 80), _interval_desirability(values[7, j], 0, 4.5, 1, 1),
            _interval_desirability(values[8, j], 0, 5, 0, 2), _interval_desirability(values[9, j], 2, 10, 2, 2),
            _interval_desirability(values[10, j], 0, 8, 0, 2), _interval_desirability(values[11, j], 35, 140, 15, 40),
            values[12, j],
        ]
    fig, ax = plt.subplots(figsize=(5.3, 5.5))
    image = ax.imshow(desir, cmap=colors.LinearSegmentedColormap.from_list("audit", ["#F5F7F7", "#8FC3C7", "#0B4F5C"]), vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(3), names)
    ax.xaxis.tick_top()
    ax.set_yticks(np.arange(len(row_names)), row_names)
    ax.set_title("Generated candidate benchmarked under one protocol", loc="left", fontweight="bold", color=NAVY, pad=28)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            fmt = ".1f" if i in {6, 11} else (".0f" if i in {8, 9, 10} else ".3f")
            ax.text(j, i, format(values[i, j], fmt), ha="center", va="center", fontsize=7.6, color="white" if desir[i, j] > 0.48 else INK)
    cbar = fig.colorbar(image, ax=ax, orientation="horizontal", fraction=0.05, pad=0.10)
    cbar.set_label("Prespecified desirability", fontsize=8)
    ax.tick_params(axis="both", length=0, labelsize=8)
    save(fig, "Fig_benchmark_heatmap")


def matched_control_figure() -> None:
    df = pd.read_csv(DATA / "candidate_vs_matched_controls.tsv", sep="\t").sort_values("stage_c_rank")
    y = np.arange(1, len(df) + 1)
    candidate = df["candidate_mean_target_deltaG"].to_numpy(float)
    control = df["control_median_mean_target_deltaG"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(5.3, 5.2))
    for yi, a, b in zip(y, candidate, control, strict=True):
        ax.plot([a, b], [yi, yi], color="#9AB0BA", linewidth=0.8, zorder=1)
    ax.scatter(candidate, y, s=14, color=TEAL, label="Candidate", zorder=2)
    ax.scatter(control, y, s=14, color=GOLD, label="Matched-control median", zorder=2)
    ax.invert_yaxis()
    ax.set_xlabel("Mean SMN2 Vina score (kcal/mol; lower is more favorable)")
    ax.set_ylabel("Stage C rank")
    ax.set_title("Matched ligand-control target-score enrichment", loc="left", fontweight="bold", color=NAVY)
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.legend(frameon=False, ncol=2, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.19))
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "Fig_ligand_control_enrichment")


def ablation_figure() -> None:
    df = pd.read_csv(DATA / "ablation_rank_comparison.tsv", sep="\t")
    shift = (df["binding_only_rank"] - df["full_srsi_rank"]).abs()
    fig, ax = plt.subplots(figsize=(5.1, 4.1))
    scatter = ax.scatter(df["binding_only_rank"], df["full_srsi_rank"], c=shift, cmap="plasma", s=20, alpha=0.85, edgecolor="none")
    ax.plot([1, 147], [1, 147], color=MUTED, linewidth=1, linestyle="--")
    ax.set_xlim(0, 150)
    ax.set_ylim(150, 0)
    ax.set_xlabel("Target-score-only rank")
    ax.set_ylabel("Full SRSI rank")
    ax.set_title("SRSI ablation across 147 candidates", loc="left", fontweight="bold", color=NAVY)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.05, pad=0.04)
    cbar.set_label("Absolute rank shift", fontsize=8)
    save(fig, "Fig_srsi_ablation")


def main() -> None:
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})
    pipeline_figure()
    generation_figure()
    stage_c_figure()
    benchmark_figure()
    matched_control_figure()
    ablation_figure()


if __name__ == "__main__":
    main()
