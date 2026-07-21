"""Summary JSON and Markdown reports for Module 5."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


LIMITATIONS = [
    "Docking scores are computational predictions and do not prove experimental binding.",
    "Predicted splicing-rescue scores are proxy scores, not measured exon 7 inclusion results.",
    "SRSI is a prioritization metric, not proof of biological activity, safety, or therapeutic effect.",
    "Top candidates require experimental validation before any biological interpretation.",
    "Dry-run results are simulated pipeline tests and are not scientific results.",
    "This module does not provide chemical synthesis instructions or human, animal, or cell testing procedures.",
]


def write_table(df: pd.DataFrame, path: str | Path) -> None:
    """Write a TSV table."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)


def build_summary(
    *,
    config: dict[str, Any],
    candidates_loaded: int,
    final_ranked: pd.DataFrame,
    used_dry_run_inputs: bool,
    pose_contact_features: pd.DataFrame,
    warnings: list[str],
) -> dict[str, Any]:
    """Build final Module 5 summary JSON."""
    tier_counts = final_ranked.get("candidate_tier", pd.Series(dtype=str)).value_counts().to_dict() if not final_ranked.empty else {}
    top = final_ranked.iloc[0] if not final_ranked.empty else None
    pose_status = pose_contact_features.get("pose_analysis_status", pd.Series(dtype=str)).astype(str) if not pose_contact_features.empty else pd.Series(dtype=str)
    return {
        "module": "Predicted Splicing-Rescue Scoring and Final SRSI Ranking",
        "run_name": config.get("run", {}).get("run_name", "final_scoring_001"),
        "num_candidates_loaded": int(candidates_loaded),
        "num_candidates_scored": int(len(final_ranked)),
        "num_high_priority_candidates": int(tier_counts.get("high_priority_computational_candidate", 0)),
        "num_medium_priority_candidates": int(tier_counts.get("medium_priority_computational_candidate", 0)),
        "num_low_priority_candidates": int(tier_counts.get("low_priority_computational_candidate", 0)),
        "num_not_recommended": int(tier_counts.get("not_recommended", 0)),
        "top_candidate_id": str(top.get("molecule_id")) if top is not None else None,
        "top_candidate_srsi": float(top.get("srsi_score")) if top is not None and pd.notna(top.get("srsi_score")) else None,
        "used_real_docking_inputs": bool(not used_dry_run_inputs),
        "used_dry_run_inputs": bool(used_dry_run_inputs),
        "pose_based_scoring_count": int((pose_status == "pose_based_proxy").sum()),
        "feature_based_fallback_count": int(pose_status.str.contains("fallback|missing|failed|no_receptor", case=False, regex=True).sum()) if len(pose_status) else 0,
        "warnings": warnings,
        "limitations": LIMITATIONS,
    }


def write_summary_json(summary: dict[str, Any], path: str | Path) -> None:
    """Write final scoring summary JSON."""
    Path(path).write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")


def _top_table(df: pd.DataFrame, top_n: int) -> str:
    columns = [
        "final_rank",
        "molecule_id",
        "target_deltaG",
        "weighted_selectivity_gap",
        "off_target_risk_score",
        "predicted_splicing_rescue_score",
        "srsi_score",
        "candidate_tier",
        "confidence_level",
    ]
    if df.empty:
        return "No candidates were ranked."
    available = [column for column in columns if column in df.columns]
    return _markdown_table(df.head(top_n)[available])


def _markdown_table(df: pd.DataFrame) -> str:
    """Render a small markdown table without optional pandas dependencies."""
    if df.empty:
        return "No rows available."
    rows = df.fillna("NA").astype(str).values.tolist()
    headers = [str(column) for column in df.columns]
    header = "| " + " | ".join(headers) + " |"
    divider = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, divider, *body])


def _evidence_snippets(evidence: dict[str, Any], top_n: int) -> str:
    if not evidence:
        return "No top-candidate evidence was available."
    blocks: list[str] = []
    for molecule_id, item in list(evidence.items())[:top_n]:
        reasons = "; ".join(item.get("why_ranked_high", []))
        concerns = "; ".join(item.get("main_concerns", []))
        blocks.append(f"- `{molecule_id}` rank {item.get('rank')}: {reasons}. Concerns: {concerns}.")
    return "\n".join(blocks)


def write_final_candidate_report(
    *,
    config: dict[str, Any],
    summary: dict[str, Any],
    final_ranked: pd.DataFrame,
    ablation: pd.DataFrame,
    evidence: dict[str, Any],
    output_files: list[str],
    path: str | Path,
) -> None:
    """Write the final project-style candidate report."""
    top_n = int(config.get("run", {}).get("top_n_report_candidates", 10))
    dry_run_notice = "\n\n**DRY-RUN / SIMULATED DOCKING INPUTS - NOT SCIENTIFIC RESULTS.**\n" if summary.get("used_dry_run_inputs") else ""
    output_text = "\n".join(f"- `{item}`" for item in output_files)
    warning_text = "\n".join(f"- {item}" for item in summary.get("warnings", [])) or "- No warnings were recorded."
    limitations = "\n".join(f"- {item}" for item in LIMITATIONS)
    ablation_note = "Ablation analysis was not available." if ablation.empty else _top_table(ablation.rename(columns={"full_srsi_rank": "final_rank"}), min(top_n, len(ablation)))
    report = f"""# SMN2-RescueDiff Final Candidate Report

## Predicted Splicing-Rescue Scoring And Final SRSI Ranking
{dry_run_notice}
## Purpose

This final module integrates SMN2 pre-mRNA target definition, HuRDA RNA decoy counter-screening, generated molecule features, and docking/selectivity outputs to rank computational candidate molecules by SRSI: Splicing Rescue Selectivity Index.

The top-ranked molecules identified by SMN2-RescueDiff are computational candidates only. Their rankings are based on predicted SMN2 RNA binding, HuRDA off-target counter-screening, predicted splicing-rescue proxy scoring, and chemical property filtering. These candidates are not experimentally validated SMN2 splicing modifiers and should be interpreted as hypotheses for future validation.

## Inputs From Modules 1-4

- Module 1 target FASTA/profile/metadata were used to define SMN2 exon 7 splicing-related RNA landmarks.
- Module 2 HuRDA metadata were used as transcriptome-scale decoy context.
- Module 3 molecule features provided chemical quality, novelty, and pre-docking candidate evidence.
- Module 4 docking/selectivity outputs provided target binding and HuRDA off-target evidence.

## Molecules Evaluated

- Candidates loaded: {summary.get("num_candidates_loaded")}
- Candidates scored: {summary.get("num_candidates_scored")}
- High-priority computational candidates: {summary.get("num_high_priority_candidates")}
- Medium-priority computational candidates: {summary.get("num_medium_priority_candidates")}
- Low-priority computational candidates: {summary.get("num_low_priority_candidates")}

## SRSI Method

SRSI combines target binding strength, predicted splicing-rescue support, weighted HuRDA selectivity gap, QED, Lipinski-style chemical quality, pre-docking candidate score, off-target risk, synthetic-accessibility proxy when available, PAINS alerts, and splice-site occlusion penalty. Because docking energies are more negative for stronger predicted binding, SMN2 binding strength is calculated as `-1 * target_deltaG`.

Binding alone is intentionally insufficient. A candidate with strong SMN2 target docking can rank lower if HuRDA decoy binding is also strong, if predicted splice-site occlusion is high, or if the splicing-rescue proxy score is weak.

## Predicted Splicing-Rescue Proxy

The predicted splicing-rescue score rewards target binding near beneficial exon 7 splicing-related landmarks, U1-support regions, exon 7 boundary-support windows, contact-density proxies when poses are available, and favorable selectivity. It penalizes off-target risk, missing pose evidence, dry-run inputs, and potential splice-site occlusion. It is a computational proxy score, not a measured rescue result.

## Top {top_n} Candidates

{_top_table(final_ranked, top_n)}

## Candidate Evidence Summaries

{_evidence_snippets(evidence, top_n)}

## Ablation Analysis

The ablation comparison contrasts binding-only ranking with rankings that include selectivity, splicing-rescue proxy evidence, and full SRSI. It is designed to show why strong target docking alone is not sufficient for final prioritization.

{ablation_note}

## Output Files

{output_text}

## Warnings

{warning_text}

## Limitations

{limitations}

## Conclusion

Module 5 produces a final ranked list of computational candidate molecules with SMN2 RNA target binding evidence, HuRDA off-target selectivity evidence, predicted splicing-rescue proxy evidence, chemical quality evidence, SRSI score, candidate tier, confidence level, and evidence summary. These outputs are hypotheses for future validation and are not proof of SMN2 splicing rescue, safety, or therapeutic activity.
"""
    Path(path).write_text(report, encoding="utf-8")


def write_splicing_method_report(
    *,
    config: dict[str, Any],
    landmarks: pd.DataFrame,
    pose_contact_features: pd.DataFrame,
    summary: dict[str, Any],
    path: str | Path,
) -> None:
    """Write Module 5 method-focused report."""
    dry_run_notice = "\n\n**DRY-RUN / SIMULATED DOCKING INPUTS - NOT SCIENTIFIC RESULTS.**\n" if summary.get("used_dry_run_inputs") else ""
    warnings = "\n".join(f"- {item}" for item in summary.get("warnings", [])) or "- No warnings were recorded."
    limitations = "\n".join(f"- {item}" for item in LIMITATIONS)
    landmark_table = _markdown_table(landmarks.head(20)) if not landmarks.empty else "No landmarks were inferred."
    pose_counts = (
        _markdown_table(pose_contact_features.get("pose_analysis_status", pd.Series(dtype=str)).value_counts().rename_axis("pose_analysis_status").reset_index(name="count"))
        if not pose_contact_features.empty
        else "No pose/contact features were calculated."
    )
    report = f"""# Splicing-Rescue Scoring Method Report
{dry_run_notice}
## Why Binding Alone Is Insufficient

Strong predicted binding to the SMN2 pre-mRNA exon 7 splicing-related RNA target is useful only when it is selective and plausibly compatible with splice-site recognition. A molecule can dock strongly yet still be a poor computational candidate if it binds HuRDA decoys strongly or appears likely to occlude essential donor/acceptor recognition regions.

## Landmark Annotation

Module 5 infers target landmarks from Module 1 target window names, target profile fields, splice-boundary labels, exon 7 distance fields, and metadata. Exact biological contact coordinates are not invented when unavailable.

{landmark_table}

## Pose And Contact Features

When pose files are available, the module parses PDB, PDBQT, SDF, or MOL2 coordinates and calculates geometric contact proxies, hydrogen-bond-like proxies, stacking-like proxies, nearest receptor atom distance, pose support score, and splice-site occlusion penalty.

Pose/contact status counts:

{pose_counts}

## Feature-Based Fallback

If pose files are unavailable or cannot be parsed, the module uses a feature-based fallback from target docking strength, HuRDA selectivity gap, off-target risk, and landmark availability. Fallback scores are intentionally conservative and recorded with `pose_analysis_status`.

## Predicted Splicing-Rescue Score

The score combines beneficial landmark proximity, U1-support-region proximity, exon 7 boundary support, target binding strength, contact density, hydrogen-bond proxy, stacking proxy, off-target risk penalty, missing-pose penalty, and splice-site occlusion penalty. It is normalized to a 0-1 range.

## Confidence Levels

Dry-run inputs receive very low confidence. Target-only docking receives low confidence. Target-plus-decoy docking receives moderate confidence. Pose-based scoring can increase confidence when real docking is used, while missing pose evidence decreases confidence.

## Warnings

{warnings}

## Limitations

{limitations}
"""
    Path(path).write_text(report, encoding="utf-8")
