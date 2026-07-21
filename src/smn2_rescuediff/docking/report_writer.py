"""Markdown report and summary output for Module 4."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


LIMITATIONS = [
    "Docking scores are model-based computational predictions and do not prove experimental binding.",
    "Dry-run scores are simulated for pipeline testing and visualization only.",
    "This module does not prove splicing rescue, safety, therapeutic efficacy, or clinical relevance.",
    "Sequence-only RNA target windows cannot replace real 3D receptor structures for scientific docking.",
    "No chemical synthesis or biological testing instructions are provided.",
]


def build_summary(
    config: dict[str, Any],
    selected_molecules: pd.DataFrame,
    target_scores: pd.DataFrame,
    decoy_scores: pd.DataFrame,
    selectivity: pd.DataFrame,
    warnings: list[str],
) -> dict[str, Any]:
    """Build docking summary JSON."""
    target_values = pd.to_numeric(target_scores.get("target_deltaG", pd.Series(dtype=float)), errors="coerce")
    decoy_values = pd.to_numeric(decoy_scores.get("decoy_deltaG", pd.Series(dtype=float)), errors="coerce")
    return {
        "project": config.get("project_name", "SMN2-RescueDiff"),
        "module": config.get("module_name", "SMN2 Target Docking and HuRDA Off-Target Counter-Screening"),
        "run_mode": config.get("run", {}).get("run_mode", "dry_run"),
        "dry_run_warning": "DRY-RUN SIMULATION ONLY — NOT SCIENTIFIC DOCKING RESULTS" if config.get("run", {}).get("run_mode") == "dry_run" else "",
        "molecules_evaluated": int(len(selected_molecules)),
        "target_scores": {
            "count": int(target_values.notna().sum()),
            "min": float(target_values.min()) if target_values.notna().any() else None,
            "mean": float(target_values.mean()) if target_values.notna().any() else None,
        },
        "decoy_scores": {
            "count": int(decoy_values.notna().sum()),
            "min": float(decoy_values.min()) if decoy_values.notna().any() else None,
            "mean": float(decoy_values.mean()) if decoy_values.notna().any() else None,
        },
        "advanced_candidates": int((selectivity.get("advance_to_splicing_rescue", pd.Series(dtype=bool)) == True).sum()) if not selectivity.empty else 0,  # noqa: E712
        "warnings": warnings,
        "limitations": LIMITATIONS,
    }


def write_summary_json(summary: dict[str, Any], path: str | Path) -> None:
    """Write summary JSON."""
    Path(path).write_text(json.dumps(summary, indent=2), encoding="utf-8")


def write_report(
    *,
    config: dict[str, Any],
    summary: dict[str, Any],
    receptor_summary: pd.DataFrame,
    ligand_summary: pd.DataFrame,
    decoy_panel: pd.DataFrame,
    output_files: list[str],
    report_path: str | Path,
) -> None:
    """Write cautious Markdown report."""
    run_mode = summary.get("run_mode", "dry_run")
    dry_note = ""
    if run_mode == "dry_run":
        dry_note = "\n\n**This run used dry-run simulation mode. The reported scores are simulated for pipeline testing and visualization only and should not be interpreted as scientific docking results.**\n"
    warnings = summary.get("warnings") or []
    warning_text = "\n".join(f"- {warning}" for warning in warnings) if warnings else "- No warnings were recorded."
    outputs = "\n".join(f"- `{item}`" for item in output_files)
    limitations = "\n".join(f"- {item}" for item in LIMITATIONS)
    receptor_status = receptor_summary.to_dict(orient="records") if not receptor_summary.empty else []
    ligand_counts = ligand_summary["prep_status"].value_counts().to_dict() if "prep_status" in ligand_summary else {}
    decoy_counts = decoy_panel["decoy_category"].value_counts().to_dict() if "decoy_category" in decoy_panel else {}
    report = f"""# {config.get("project_name", "SMN2-RescueDiff")} - {config.get("module_name", "SMN2 Target Docking and HuRDA Off-Target Counter-Screening")}

## Purpose

This module estimates computational target binding and off-target RNA selectivity for generated candidate molecules. Docking scores are model-based predictions and do not prove experimental binding, splicing rescue, safety, or therapeutic activity.{dry_note}

## Inputs Used

- Module 1 SMN2 target FASTA/profile/metadata
- Module 2 HuRDA decoy atlas and summary
- Module 3 filtered generated molecules
- Optional receptor structures from the configured receptor path and raw structure directory

## Run Mode

Run mode: `{run_mode}`

## Receptor Preparation Summary

```json
{json.dumps(receptor_status, indent=2)}
```

## Ligand Preparation Summary

Ligand preparation status counts:

```json
{json.dumps(ligand_counts, indent=2)}
```

## Decoy Panel Selection Strategy

The HuRDA panel is selected by configured category targets, then prioritized by SMN2 similarity score, biological weight, and bias-match score. If decoy receptor structures are unavailable, the panel remains a sequence-based counter-screening panel for future structural modeling.

```json
{json.dumps(decoy_counts, indent=2)}
```

## Docking Engine

Configured engine: `{config.get("docking_engine", {}).get("engine", "vina")}`. Real Vina docking requires prepared receptor and ligand PDBQT files plus a confirmed docking box. Dry-run mode uses deterministic simulated scores and is not scientific docking.

## SMN2 Target Docking Summary

Target score summary:

```json
{json.dumps(summary.get("target_scores", {}), indent=2)}
```

## Decoy Counter-Screening Summary

Decoy score summary:

```json
{json.dumps(summary.get("decoy_scores", {}), indent=2)}
```

## Selectivity Score Definitions

Because more negative docking energy indicates stronger predicted binding, selectivity gaps are calculated as decoy score minus target score. Larger positive gaps indicate stronger predicted target selectivity.

## Candidate Advancement Criteria

Default advancement requires target deltaG at or below the configured threshold, weighted selectivity gap above the configured threshold, off-target risk at or below the configured threshold, and enough valid decoy scores. In dry-run mode, the flag is based on simulated values only.

Advanced candidate count: {summary.get("advanced_candidates")}

## Live Simulation Dashboard

The live simulation writes JSONL events and a current-state JSON file. The dashboard can replay or monitor these files:

```bash
streamlit run dashboard/docking_live_app.py
```

## Warnings

{warning_text}

## Output Files

{outputs}

## Limitations

{limitations}

## Next Steps

Module 5 will be Predicted Splicing-Rescue Scoring. Module 4 addresses binding and selectivity; Module 5 will evaluate whether top molecules bind in a position predicted to support SMN2 exon 7 inclusion, U1 snRNA recognition, and splice-site stabilization.
"""
    Path(report_path).write_text(report, encoding="utf-8")

