#!/usr/bin/env python3
"""Generate manuscript figures exclusively from released evidence tables."""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[2]
AUDIT = ROOT / "reproducibility" / "tcbb_2026" / "reviewer_revision_2026_10"
RELEASE = ROOT / "reproducibility" / "tcbb_2026"
FIGURES = PACKAGE / "figures"
FIGURE_DATA = PACKAGE / "figure_data"


def save(name: str) -> None:
    plt.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    plt.savefig(FIGURES / f"{name}.png", dpi=240, bbox_inches="tight")
    plt.close()


def pipeline_figure() -> None:
    fig, ax = plt.subplots(figsize=(12.5, 5.6))
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 5.6)
    ax.axis("off")
    colors = ["#d0ebff", "#d3f9d8", "#fff3bf", "#ffe8cc", "#e5dbff", "#c5f6fa"]
    modules = [
        ("1", "Target definition", "sequence + structure\nprovenance"),
        ("2", "HuRDA atlas", "sequence atlas: 180k+\nstructural subset explicit"),
        ("3", "Generation", "unconditional model\nchemical triage"),
        ("4", "Docking", "5-conformer target\ncoordinate-aware boxes"),
        ("5", "Evidence audit", "geometry + status QC\nsize + null analyses"),
        ("6", "Prioritization", "target robustness first\nhypotheses only"),
    ]
    for i, (number, title, detail) in enumerate(modules):
        x = 0.25 + i * 2.05
        box = FancyBboxPatch((x, 2.65), 1.75, 1.55, boxstyle="round,pad=0.04,rounding_size=0.08",
                             linewidth=1.2, edgecolor="#343a40", facecolor=colors[i])
        ax.add_patch(box)
        ax.text(x + 0.16, 3.88, f"MODULE {number}", fontsize=8, fontweight="bold", color="#495057")
        ax.text(x + 0.16, 3.52, title, fontsize=10, fontweight="bold", color="#212529")
        ax.text(x + 0.16, 2.92, detail, fontsize=7.5, color="#343a40", va="bottom")
        if i < len(modules) - 1:
            ax.add_patch(FancyArrowPatch((x + 1.76, 3.42), (x + 2.03, 3.42), arrowstyle="-|>",
                                         mutation_scale=12, linewidth=1.2, color="#495057"))
    gate = FancyBboxPatch((0.8, 0.55), 10.9, 1.15, boxstyle="round,pad=0.05,rounding_size=0.08",
                          linewidth=1.4, edgecolor="#c92a2a", facecolor="#fff5f5")
    ax.add_patch(gate)
    ax.text(1.05, 1.39, "EVIDENCE CONTRACT AT EVERY TRANSITION", fontsize=10, fontweight="bold", color="#c92a2a")
    ax.text(1.05, 0.84,
            "schema + config + input hashes + run folder + failure table + non-dry-run flag + geometry/QC gate + cautious claim boundary",
            fontsize=9.1, color="#343a40")
    for i in range(6):
        x = 1.12 + i * 2.05
        ax.add_patch(FancyArrowPatch((x, 2.62), (x, 1.72), arrowstyle="-|>", mutation_scale=11,
                                     linewidth=1.0, color="#c92a2a"))
    ax.text(6.25, 5.05, "SMN2-RescueDiff as a failure-aware computational biology workflow",
            ha="center", fontsize=14, fontweight="bold")
    save("Fig_pipeline_evidence_contracts")


def generation_figure() -> None:
    flow = pd.read_csv(AUDIT / "generation_flow_reconciliation.tsv", sep="\t")
    baseline = pd.read_csv(AUDIT / "generator_baseline_comparison.tsv", sep="\t")
    flow.to_csv(FIGURE_DATA / "generation_flow_reconciliation.tsv", sep="\t", index=False)
    baseline.to_csv(FIGURE_DATA / "generator_baseline_comparison.tsv", sep="\t", index=False)

    selected = flow[flow["stage"].isin([
        "raw_samples_requested", "unique_strings_emitted", "unique_canonical_molecules",
        "retained_unique_molecules", "exported_for_docking"
    ])].copy()
    labels = ["Requested", "Emitted\nvalid strings", "Unique canonical", "Filter pass", "Docking export"]
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.5), gridspec_kw={"width_ratios": [1.25, 1]})
    axes[0].plot(range(len(selected)), selected["count"], marker="o", linewidth=2.4, color="#1971c2")
    axes[0].fill_between(range(len(selected)), selected["count"], color="#d0ebff", alpha=0.75)
    axes[0].set_xticks(range(len(selected)), labels)
    axes[0].set_ylabel("Molecule/string records")
    axes[0].set_title("Production generation accounting")
    axes[0].grid(axis="y", alpha=0.25)
    for idx, count in enumerate(selected["count"]):
        axes[0].text(idx, count + 1400, f"{int(count):,}", ha="center", fontsize=8)

    values = baseline.set_index("generator")[["valid_per_requested_fraction", "filter_pass_per_requested_fraction"]]
    values.plot.bar(ax=axes[1], color=["#2f9e44", "#f08c00"], width=0.72)
    axes[1].set_ylabel("Fraction of 50,000 requested samples")
    axes[1].set_xlabel("")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Controlled generator baseline")
    axes[1].tick_params(axis="x", rotation=0)
    axes[1].legend(["RDKit-valid", "Filter pass"], frameon=False, fontsize=8)
    axes[1].grid(axis="y", alpha=0.25)
    for container in axes[1].containers:
        axes[1].bar_label(container, fmt="%.3f", padding=2, fontsize=8)
    fig.tight_layout()
    save("Fig_generation_and_baseline")


def copy_audit_figures() -> None:
    mapping = {
        "decoy_grid_geometry_bias": "Fig_decoy_geometry_bias",
        "target_score_molecular_weight_confounding": "Fig_target_size_confounding",
        "worst_decoy_panel_size_sensitivity": "Fig_decoy_panel_sensitivity",
    }
    for source, destination in mapping.items():
        for suffix in (".pdf", ".png"):
            shutil.copy2(AUDIT / f"{source}{suffix}", FIGURES / f"{destination}{suffix}")


def target_robustness_figure() -> None:
    scores = pd.read_csv(RELEASE / "target_docking" / "target_conformer_robustness_scores_final.tsv", sep="\t")
    scores.to_csv(FIGURE_DATA / "target_conformer_robustness_scores.tsv", sep="\t", index=False)
    lead = scores[scores["molecule_id"] == "MolGen_038292"].iloc[0]
    plt.figure(figsize=(7.0, 4.8))
    scatter = plt.scatter(scores["mean_target_deltaG"], scores["worst_target_deltaG"],
                          c=scores["target_deltaG_range"], cmap="cividis", s=24, alpha=0.65,
                          edgecolors="none")
    plt.colorbar(scatter, label="Five-conformer score range (kcal/mol)")
    plt.scatter([lead["mean_target_deltaG"]], [lead["worst_target_deltaG"]], s=100,
                color="#e03131", edgecolor="black", label="MolGen_038292")
    plt.plot([-14, -4], [-14, -4], color="#868e96", linestyle="--", linewidth=1)
    plt.xlabel("Mean target Vina score (kcal/mol)")
    plt.ylabel("Worst target Vina score (kcal/mol)")
    plt.title("Primary five-conformer target robustness screen")
    plt.legend(frameon=False)
    plt.grid(alpha=0.2)
    save("Fig_target_robustness")


def redocking_figure() -> None:
    data = pd.read_csv(RELEASE / "redocking" / "analysis" / "redocking_validation_summary.tsv", sep="\t")
    data.to_csv(FIGURE_DATA / "redocking_validation_summary.tsv", sep="\t", index=False)
    plt.figure(figsize=(7.1, 4.7))
    labels = data["ligand_name"].str.replace(" analogue 6YG", "\nanalogue", regex=False)
    colors = ["#2f9e44" if value < 2 else "#e03131" for value in data["headline_model1_rmsd_angstrom"]]
    bars = plt.bar(labels, data["headline_model1_rmsd_angstrom"], color=colors, edgecolor="#343a40", linewidth=0.5)
    plt.axhline(2.0, color="#212529", linestyle="--", linewidth=1.3, label="2 Angstrom rank-1 criterion")
    plt.ylabel("Rank-1 pose RMSD to model 1 (Angstrom)")
    plt.title("Mixed redocking calibration")
    plt.legend(frameon=False)
    plt.grid(axis="y", alpha=0.25)
    plt.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
    save("Fig_redocking_calibration")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    FIGURE_DATA.mkdir(parents=True, exist_ok=True)
    pipeline_figure()
    generation_figure()
    copy_audit_figures()
    target_robustness_figure()
    redocking_figure()
    print(f"Wrote figures to {FIGURES}")


if __name__ == "__main__":
    main()
