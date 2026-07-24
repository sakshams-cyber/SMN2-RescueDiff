"""Figures, summary JSON, and Markdown report for Module 3."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib-cache"))

import pandas as pd


LIMITATIONS = [
    "Generated molecules are computational candidates only and are not experimentally validated compounds.",
    "This module does not perform SMN2 RNA docking, HuRDA off-target counter-screening, transcriptome-wide safety assessment, synthesis planning, or final SRSI ranking.",
    "Neural generation, SELFIES validity, reward diagnostics, chemical filters, PAINS flags, QED, Lipinski rules, novelty, and diversity metrics are computational prioritization signals, not evidence of SMN2 binding, exon 7 rescue, safety, or efficacy.",
    "Toy example SMILES are only for software smoke testing and should not be used for scientific conclusions.",
]


def _plt():
    import matplotlib.pyplot as plt

    return plt


def _numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[column], errors="coerce").dropna()


def _placeholder_figure(path: Path, title: str, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt = _plt()
    plt.figure(figsize=(7, 4.5))
    plt.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def write_figures(
    *,
    training_log: pd.DataFrame,
    valid: pd.DataFrame,
    filtered: pd.DataFrame,
    rejected: pd.DataFrame,
    raw_count: int,
    diversity_values: list[float],
    figure_dir: str | Path,
) -> None:
    """Create all requested matplotlib figures."""
    plt = _plt()
    out = Path(figure_dir)
    out.mkdir(parents=True, exist_ok=True)

    if "train_loss" in training_log and training_log["train_loss"].notna().any():
        plt.figure(figsize=(7, 4.5))
        plt.plot(training_log["epoch"], training_log["train_loss"], marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Training loss")
        plt.title("SMILES RNN training loss")
        plt.tight_layout()
        plt.savefig(out / "training_loss_curve.png", dpi=200)
        plt.close()
    else:
        _placeholder_figure(out / "training_loss_curve.png", "SMILES RNN training loss", "Neural training was not run for this configuration.")

    for column, filename, title, xlabel in [
        ("molecular_weight", "molecular_weight_distribution.png", "Molecular weight distribution", "Molecular weight"),
        ("logp", "logp_distribution.png", "LogP distribution", "LogP"),
        ("qed", "qed_distribution.png", "QED distribution", "QED"),
    ]:
        values = _numeric(filtered if not filtered.empty else valid, column)
        if values.empty:
            _placeholder_figure(out / filename, title, f"No numeric {xlabel} values were available.")
        else:
            plt.figure(figsize=(7, 4.5))
            values.hist(bins=40)
            plt.xlabel(xlabel)
            plt.ylabel("Count")
            plt.title(title)
            plt.tight_layout()
            plt.savefig(out / filename, dpi=200)
            plt.close()

    if "lipinski_pass" in valid and not valid.empty:
        counts = valid["lipinski_pass"].value_counts(dropna=False)
        plt.figure(figsize=(6, 4.5))
        plt.bar([str(item) for item in counts.index], counts.values)
        plt.xlabel("Lipinski pass")
        plt.ylabel("Count")
        plt.title("Lipinski pass/fail counts")
        plt.tight_layout()
        plt.savefig(out / "lipinski_pass_rate.png", dpi=200)
        plt.close()
    else:
        _placeholder_figure(out / "lipinski_pass_rate.png", "Lipinski pass/fail counts", "No valid molecules were available.")

    unique_count = int(valid["canonical_smiles"].nunique()) if "canonical_smiles" in valid else 0
    novel_count = int((filtered.get("is_novel_vs_training", pd.Series(dtype=bool)) == True).sum()) if not filtered.empty else 0  # noqa: E712
    summary_labels = ["raw", "valid", "unique_valid", "novel_filtered", "filtered"]
    summary_values = [raw_count, len(valid), unique_count, novel_count, len(filtered)]
    plt.figure(figsize=(7, 4.5))
    plt.bar(summary_labels, summary_values)
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("Count")
    plt.title("Validity, uniqueness, and novelty summary")
    plt.tight_layout()
    plt.savefig(out / "validity_uniqueness_novelty_summary.png", dpi=200)
    plt.close()

    if diversity_values:
        plt.figure(figsize=(7, 4.5))
        plt.hist(diversity_values, bins=40)
        plt.xlabel("Pairwise Tanimoto similarity")
        plt.ylabel("Sampled pair count")
        plt.title("Diversity distribution")
        plt.tight_layout()
        plt.savefig(out / "diversity_distribution.png", dpi=200)
        plt.close()
    else:
        _placeholder_figure(out / "diversity_distribution.png", "Diversity distribution", "Diversity calculation was disabled or insufficient molecules were available.")


def build_summary(
    *,
    config: dict[str, Any],
    context_status: dict[str, bool],
    training_path: str,
    toy_warning_used: bool,
    raw_count: int,
    valid: pd.DataFrame,
    filtered: pd.DataFrame,
    rejected: pd.DataFrame,
    warnings: list[str],
) -> dict[str, Any]:
    """Build the machine-readable Module 3 summary."""
    validity = len(valid) / raw_count if raw_count else 0.0
    uniqueness = valid["canonical_smiles"].nunique() / len(valid) if len(valid) and "canonical_smiles" in valid else 0.0
    novelty = (filtered["is_novel_vs_training"] == True).sum() / len(filtered) if len(filtered) and "is_novel_vs_training" in filtered else 0.0  # noqa: E712
    numeric_summary = {}
    for column in [
        "molecular_weight",
        "logp",
        "qed",
        "max_tanimoto_to_training",
        "novelty_vs_training_set",
        "rna_ligand_like_score",
        "chemical_quality_score",
        "pre_docking_candidate_score",
        "final_generation_selection_score",
    ]:
        values = _numeric(filtered, column)
        numeric_summary[column] = {
            "min": float(values.min()) if not values.empty else None,
            "max": float(values.max()) if not values.empty else None,
            "mean": float(values.mean()) if not values.empty else None,
        }
    return {
        "project": config.get("project_name", "SMN2-RescueDiff"),
        "module": config.get("module_name", "Deep Neural Molecular Generation Model"),
        "training_smiles_path": training_path,
        "toy_training_file_used": toy_warning_used,
        "context_files": context_status,
        "generator_mode": config.get("generation", {}).get("generator_mode", "neural_transformer"),
        "representation": config.get("tokenization", {}).get("representation", "selfies"),
        "model_architecture": config.get("model", {}).get("architecture", "transformer_decoder"),
        "raw_generated_count": int(raw_count),
        "valid_count": int(len(valid)),
        "filtered_count": int(len(filtered)),
        "rejected_count": int(len(rejected)),
        "validity_fraction": validity,
        "unique_valid_fraction": uniqueness,
        "novel_filtered_fraction": float(novelty),
        "source_generator_counts": filtered["source_generator"].value_counts().to_dict() if "source_generator" in filtered else {},
        "property_summary": numeric_summary,
        "warnings": warnings,
        "limitations": LIMITATIONS,
    }


def write_summary_json(summary: dict[str, Any], path: str | Path) -> None:
    """Write summary JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def write_report(
    *,
    config: dict[str, Any],
    summary: dict[str, Any],
    output_files: list[str],
    report_path: str | Path,
) -> None:
    """Write the cautious Module 3 Markdown report."""
    outputs = "\n".join(f"- `{item}`" for item in output_files)
    warnings = summary.get("warnings") or []
    warning_text = "\n".join(f"- {warning}" for warning in warnings) if warnings else "- No warnings were recorded."
    limitations = "\n".join(f"- {item}" for item in LIMITATIONS)
    context = summary.get("context_files", {})
    property_summary = json.dumps(summary.get("property_summary", {}), indent=2)
    figures = "\n".join(
        [
            "- `figures/molecule_generation/training_loss_curve.png`",
            "- `figures/molecule_generation/validation_loss_curve.png`",
            "- `figures/molecule_generation/rl_reward_curve.png`",
            "- `figures/molecule_generation/molecular_weight_distribution.png`",
            "- `figures/molecule_generation/logp_distribution.png`",
            "- `figures/molecule_generation/qed_distribution.png`",
            "- `figures/molecule_generation/rna_ligand_like_score_distribution.png`",
            "- `figures/molecule_generation/chemical_quality_score_distribution.png`",
            "- `figures/molecule_generation/pre_docking_candidate_score_distribution.png`",
            "- `figures/molecule_generation/final_generation_selection_score_distribution.png`",
            "- `figures/molecule_generation/scaffold_frequency_top20.png`",
            "- `figures/molecule_generation/validity_uniqueness_novelty_summary.png`",
            "- `figures/molecule_generation/diversity_distribution.png`",
        ]
    )
    report = f"""# {config.get("project_name", "SMN2-RescueDiff")} - {config.get("module_name", "Deep Neural Molecular Generation Model")}

## Purpose

Module 3 trains and samples a neural molecular language model to produce computational candidate molecules for future SMN2 RNA-target docking and HuRDA transcriptome-aware off-target counter-screening. The primary model is a Transformer decoder trained on curated molecular strings, with a GRU/LSTM fallback path and SELFIES support for more robust molecular-string generation.

The output is a prioritized pre-docking candidate library. It does not establish RNA binding, exon 7 rescue, toxicity, safety, or clinical utility.

## Required Inputs

- Training SMILES files used: `{summary.get("training_smiles_path")}`
- Module 1 target profile available: {context.get("smn2_target_profile_exists")}
- Module 1 target metadata available: {context.get("smn2_target_metadata_exists")}
- Module 2 HuRDA summary available: {context.get("decoy_atlas_summary_exists")}

Recommended real training sources include ChEMBL, ZINC subsets, RNA-binding ligand collections where available, approved drug-like molecule libraries, and project-specific compound collections. Toy fallback data are only for validating the software path.

## Dataset And Representation

- Raw training molecules read: {summary.get("training_molecules_raw", "NA")}
- Valid training molecules: {summary.get("training_molecules_valid", "NA")}
- Invalid training molecules: {summary.get("training_molecules_invalid", "NA")}
- Deduplicated training records: {summary.get("training_molecules_deduplicated", "NA")}
- Cleaned canonical training molecules: {summary.get("training_molecules_cleaned", "NA")}
- Weighted training rows used for model exposure: {summary.get("training_molecules_weighted", "NA")}
- Molecular representation: {summary.get("representation", config.get("tokenization", {}).get("representation", "selfies"))}
- Generator mode: {summary.get("generator_mode", config.get("generation", {}).get("generator_mode", "neural_transformer"))}

The preprocessing step removes invalid records, optionally removes salts and mixtures, canonicalizes molecules with RDKit, filters training molecules by configurable molecular-weight and atom constraints, removes duplicates, and writes rejected training records for auditability.

## Neural Model

- Architecture: {summary.get("model_architecture", config.get("model", {}).get("architecture", "transformer_decoder"))}
- Epochs completed: {summary.get("epochs_completed", "NA")}
- Token vocabulary written: `token_vocabulary.json`
- Training log written: `model_training_log.tsv`

The Transformer decoder is intended as the primary generator. The recurrent GRU/LSTM path remains available as a fallback and for small-hardware runs. Sampling uses configurable temperature, top-k, top-p, and maximum-length controls.

## Reward Diagnostics

Reward fine-tuning diagnostics use chemical plausibility and RNA-ligand-like features only. They intentionally avoid docking-derived rewards and do not claim SMN2 target activity. The run writes `rl_finetuning_log.tsv` and, when enabled, lightweight reward-diagnostic checkpoint markers.

## Chemical Filtering Criteria

The module uses RDKit to parse and canonicalize decoded SMILES, remove invalid molecules, optionally remove salts, remove duplicates, calculate chemical properties, apply configurable molecular-weight, LogP, hydrogen-bond, rotatable-bond, formal-charge, Lipinski, and PAINS filters, and retain a clean candidate library.

## Prioritization

- Raw generated SMILES: {summary.get("raw_generated_count")}
- Valid molecules: {summary.get("valid_count")}
- Filtered molecules retained: {summary.get("filtered_count")}
- Rejected molecules: {summary.get("rejected_count")}
- Validity fraction: {summary.get("validity_fraction"):.4f}
- Unique-valid fraction: {summary.get("unique_valid_fraction"):.4f}
- Novel-filtered fraction: {summary.get("novel_filtered_fraction"):.4f}
- Candidates exported for docking: {summary.get("selected_for_docking_count", "NA")}

Filtered molecules receive scaffold annotations, novelty scores against the training set, diversity cluster identifiers, RNA-ligand-like scores, chemical quality scores, and final generation selection scores. The docking export is ranked by quality-weighted score and diversity-selected with Morgan-fingerprint/Tanimoto filtering when enabled. These values support downstream prioritization but are not substitutes for Module 4 docking or experimental validation.

## Property Summary

```json
{property_summary}
```

## Diversity Analysis

Molecules are assigned scaffold-based diversity cluster identifiers. A runtime-capped sample of pairwise Morgan-fingerprint Tanimoto similarities is used for the diversity distribution figure.

## Figures

{figures}

## Warnings

{warning_text}

## Output Files

{outputs}

## Limitations

{limitations}

## Next Steps

The next module can use these filtered computational candidate molecules for SMN2 target docking preparation, HuRDA decoy counter-screening, and transcriptome-aware scoring. No synthesis, biological testing, or clinical use is implied by these outputs.
"""
    p = Path(report_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(report, encoding="utf-8")
