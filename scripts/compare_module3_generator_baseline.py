#!/usr/bin/env python3
"""Compare the production Transformer with a controlled 50k n-gram baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "reproducibility" / "tcbb_2026" / "reviewer_revision_2026_10"
RUNS = PROJECT_ROOT / "data" / "processed" / "molecule_generation" / "runs"


def load_summary(run_name: str) -> dict:
    path = RUNS / run_name / "generated_molecule_summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing generator summary: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def row(label: str, run_name: str, summary: dict) -> dict:
    requested = 50_000
    emitted = int(summary.get("raw_generated_count", 0))
    valid = int(summary.get("valid_count", 0))
    unique_valid = summary.get("valid_unique_count")
    if unique_valid is None:
        unique_valid = valid - int(summary.get("valid_duplicate_count", 0))
    filtered = int(summary.get("filtered_count", 0))
    props = summary.get("property_summary", {})
    return {
        "generator": label,
        "run_name": run_name,
        "representation": summary.get("representation", ""),
        "architecture": summary.get("model_architecture", ""),
        "requested_samples": requested,
        "emitted_records": emitted,
        "valid_records": valid,
        "unique_valid_records": int(unique_valid),
        "filter_pass_records": filtered,
        "valid_per_requested_fraction": valid / requested,
        "filter_pass_per_requested_fraction": filtered / requested,
        "mean_qed_filter_pass": props.get("qed", {}).get("mean"),
        "mean_chemical_quality_filter_pass": props.get("chemical_quality_score", {}).get("mean"),
        "mean_novelty_filter_pass": props.get("novelty_vs_training_set", {}).get("mean"),
        "interpretation_scope": (
            "Controlled engineering baseline using the same requested sample count and training-source configuration; "
            "not an external biological validation."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    transformer_name = "module3_final_final_rigorous_x10"
    baseline_name = "module3_ngram_baseline_50k"
    comparison = pd.DataFrame(
        [
            row("Transformer + SELFIES", transformer_name, load_summary(transformer_name)),
            row("Token 3-gram + SMILES", baseline_name, load_summary(baseline_name)),
        ]
    )
    comparison.to_csv(output / "generator_baseline_comparison.tsv", sep="\t", index=False)

    plot = comparison.set_index("generator")[["valid_per_requested_fraction", "filter_pass_per_requested_fraction"]]
    ax = plot.plot.bar(figsize=(7.2, 4.7), color=["#1971c2", "#f08c00"], width=0.72)
    ax.set_ylabel("Fraction of 50,000 requested samples")
    ax.set_xlabel("")
    ax.set_ylim(0, 1.05)
    ax.set_title("Controlled generator baseline")
    ax.legend(["RDKit-valid", "Chemical-filter pass"], frameon=False)
    ax.grid(axis="y", alpha=0.25)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=2, fontsize=8)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output / "generator_baseline_comparison.pdf")
    plt.savefig(output / "generator_baseline_comparison.png", dpi=240)
    plt.close()

    transformer = comparison.iloc[0]
    baseline = comparison.iloc[1]
    metrics = {
        "comparison_scope": comparison.iloc[0]["interpretation_scope"],
        "transformer_valid_per_requested_fraction": transformer["valid_per_requested_fraction"],
        "ngram_valid_per_requested_fraction": baseline["valid_per_requested_fraction"],
        "transformer_filter_pass_per_requested_fraction": transformer["filter_pass_per_requested_fraction"],
        "ngram_filter_pass_per_requested_fraction": baseline["filter_pass_per_requested_fraction"],
        "transformer_filter_pass_count": int(transformer["filter_pass_records"]),
        "ngram_filter_pass_count": int(baseline["filter_pass_records"]),
    }
    (output / "generator_baseline_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
