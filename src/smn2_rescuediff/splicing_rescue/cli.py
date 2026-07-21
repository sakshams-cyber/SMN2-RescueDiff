"""CLI for Module 5 predicted splicing-rescue scoring and SRSI ranking."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from .ablation_analysis import run_ablation_analysis
from .config import ensure_output_dirs, load_config, resolve_paths, save_config_used, validate_weights
from .contact_analyzer import analyze_contacts
from .evidence_writer import write_evidence_json, write_warnings_json
from .figure_writer import write_figures
from .input_loader import assemble_candidate_table, load_all_inputs
from .pose_parser import parse_candidate_poses
from .report_writer import (
    build_summary,
    write_final_candidate_report,
    write_splicing_method_report,
    write_summary_json,
    write_table,
)
from .rescue_score_calculator import calculate_splicing_rescue_scores
from .splicing_feature_scorer import infer_feature_based_components
from .srsi_ranker import calculate_srsi
from .target_landmark_annotator import annotate_landmarks


RUN_MODE_LABEL = "DRY-RUN / SIMULATED DOCKING INPUTS - NOT SCIENTIFIC RESULTS"


def _validate_scoring_mode(config: dict[str, Any], used_dry_run_inputs: bool) -> None:
    mode = str(config.get("splicing_rescue_scoring", {}).get("score_mode", "hybrid_pose_and_feature"))
    allowed = {"pose_based", "feature_based", "hybrid_pose_and_feature", "dry_run_input_analysis"}
    if mode not in allowed:
        raise ValueError(f"splicing_rescue_scoring.score_mode must be one of {sorted(allowed)}")
    if used_dry_run_inputs and not bool(config.get("run", {}).get("allow_dry_run_inputs", True)):
        raise ValueError("Dry-run Module 4 inputs were detected, but run.allow_dry_run_inputs is false.")
    if used_dry_run_inputs and mode != "dry_run_input_analysis":
        config.setdefault("splicing_rescue_scoring", {})["score_mode"] = "dry_run_input_analysis"


def _display_path(paths, path: str | Path) -> str:
    p = Path(path)
    try:
        return str(p.relative_to(paths.project_root))
    except ValueError:
        return str(p)


def _output_file_list(paths) -> list[str]:
    return [
        _display_path(paths, paths.table_dir / "smn2_splicing_landmarks.tsv"),
        _display_path(paths, paths.table_dir / "pose_contact_features.tsv"),
        _display_path(paths, paths.table_dir / "splicing_rescue_scores.tsv"),
        _display_path(paths, paths.table_dir / "final_srsi_ranked_candidates.tsv"),
        _display_path(paths, paths.table_dir / "top_10_final_candidates.tsv"),
        _display_path(paths, paths.table_dir / "ablation_rank_comparison.tsv"),
        _display_path(paths, paths.evidence_dir / "top_candidate_evidence.json"),
        _display_path(paths, paths.evidence_dir / "scoring_warnings.json"),
        _display_path(paths, paths.output_dir / "final_scoring_summary.json"),
        _display_path(paths, paths.output_dir / "splicing_rescue_scoring_config_used.yaml"),
        _display_path(paths, paths.log_dir / "module5_run_log.txt"),
        _display_path(paths, paths.splicing_report_path),
        _display_path(paths, paths.final_report_path),
        _display_path(paths, paths.figure_dir / "splicing_rescue_score_distribution.png"),
        _display_path(paths, paths.figure_dir / "srsi_score_distribution.png"),
        _display_path(paths, paths.figure_dir / "target_binding_vs_splicing_rescue_scatter.png"),
        _display_path(paths, paths.figure_dir / "selectivity_gap_vs_splicing_rescue_scatter.png"),
        _display_path(paths, paths.figure_dir / "off_target_risk_vs_srsi_scatter.png"),
        _display_path(paths, paths.figure_dir / "top_10_candidate_score_breakdown.png"),
        _display_path(paths, paths.figure_dir / "top_candidate_heatmap.png"),
        _display_path(paths, paths.figure_dir / "ablation_rank_shift_plot.png"),
        _display_path(paths, paths.figure_dir / "candidate_confidence_summary.png"),
    ]


def run(config_path: str | Path) -> int:
    """Run Module 5 final SRSI scoring."""
    config = load_config(config_path)
    validate_weights(config)
    paths = resolve_paths(config, config_path)
    ensure_output_dirs(paths)
    run_log: list[str] = []

    def log(message: str) -> None:
        print(message)
        run_log.append(message)

    log("Loading Module 1-4 inputs.")
    inputs = load_all_inputs(paths)
    warnings = list(inputs.get("warnings", []))
    used_dry_run_inputs = bool(inputs.get("used_dry_run_inputs", False))
    _validate_scoring_mode(config, used_dry_run_inputs)
    if used_dry_run_inputs:
        warnings.append(RUN_MODE_LABEL)
        log(RUN_MODE_LABEL)

    max_candidates = int(config.get("run", {}).get("max_candidates_to_score", 10000))
    candidates = assemble_candidate_table(inputs, max_candidates=max_candidates)
    log(f"Assembled {len(candidates)} candidate molecules for Module 5 scoring.")

    landmarks, landmark_warnings = annotate_landmarks(inputs["target_profile"], inputs["target_metadata"], config)
    warnings.extend(landmark_warnings)
    write_table(landmarks, paths.table_dir / "smn2_splicing_landmarks.tsv")

    log("Parsing docking pose files when available.")
    pose_parse, atom_cache = parse_candidate_poses(candidates, config)
    contacts = analyze_contacts(candidates, pose_parse, atom_cache, landmarks, config)
    write_table(contacts, paths.table_dir / "pose_contact_features.tsv")
    fallback_count = int(contacts["pose_analysis_status"].astype(str).str.contains("fallback|missing|failed|no_receptor", case=False, regex=True).sum()) if not contacts.empty else 0
    if fallback_count:
        warnings.append(f"{fallback_count} candidates used feature-based fallback because pose/contact evidence was unavailable or incomplete.")

    log("Calculating predicted splicing-rescue proxy scores.")
    feature_scores = infer_feature_based_components(candidates, landmarks, config)
    rescue_scores = calculate_splicing_rescue_scores(candidates, contacts, feature_scores, config, used_dry_run_inputs)
    write_table(rescue_scores, paths.table_dir / "splicing_rescue_scores.tsv")

    log("Calculating final SRSI scores and candidate tiers.")
    final_ranked = calculate_srsi(candidates, rescue_scores, config, used_dry_run_inputs)
    if final_ranked.empty:
        raise ValueError("No molecules can be scored after SRSI calculation.")
    write_table(final_ranked, paths.table_dir / "final_srsi_ranked_candidates.tsv")
    top_report_n = int(config.get("run", {}).get("top_n_report_candidates", 10))
    write_table(final_ranked.head(top_report_n), paths.table_dir / "top_10_final_candidates.tsv")

    log("Running ablation analysis.")
    ablation = run_ablation_analysis(final_ranked, config)
    write_table(ablation, paths.table_dir / "ablation_rank_comparison.tsv")

    log("Writing evidence traces, figures, and reports.")
    top_final_n = int(config.get("run", {}).get("top_n_final_candidates", 50))
    evidence = write_evidence_json(final_ranked, top_final_n, paths.evidence_dir / "top_candidate_evidence.json")
    write_warnings_json(warnings, paths.evidence_dir / "scoring_warnings.json")
    summary = build_summary(
        config=config,
        candidates_loaded=len(candidates),
        final_ranked=final_ranked,
        used_dry_run_inputs=used_dry_run_inputs,
        pose_contact_features=contacts,
        warnings=warnings,
    )
    write_summary_json(summary, paths.output_dir / "final_scoring_summary.json")
    save_config_used(config, paths.output_dir / "splicing_rescue_scoring_config_used.yaml")
    write_figures(
        final_ranked=final_ranked,
        rescue_scores=rescue_scores,
        ablation=ablation,
        figure_dir=paths.figure_dir,
        config=config,
        used_dry_run_inputs=used_dry_run_inputs,
    )
    output_files = _output_file_list(paths)
    write_final_candidate_report(
        config=config,
        summary=summary,
        final_ranked=final_ranked,
        ablation=ablation,
        evidence=evidence,
        output_files=output_files,
        path=paths.final_report_path,
    )
    write_splicing_method_report(
        config=config,
        landmarks=landmarks,
        pose_contact_features=contacts,
        summary=summary,
        path=paths.splicing_report_path,
    )
    (paths.log_dir / "module5_run_log.txt").write_text("\n".join(run_log + warnings) + "\n", encoding="utf-8")

    top_candidate = final_ranked.iloc[0]
    print("\nModule 5 complete: Predicted splicing-rescue scoring and final SRSI ranking finished.\n")
    print(f"Candidates loaded: {summary['num_candidates_loaded']}")
    print(f"Candidates scored: {summary['num_candidates_scored']}")
    print(f"High-priority computational candidates: {summary['num_high_priority_candidates']}")
    print(f"Top candidate: {summary['top_candidate_id']}")
    print(f"Top SRSI score: {top_candidate['srsi_score']}")
    print("\nFinal ranked candidates:")
    print(_display_path(paths, paths.table_dir / "final_srsi_ranked_candidates.tsv"))
    print("\nTop 10 candidates:")
    print(_display_path(paths, paths.table_dir / "top_10_final_candidates.tsv"))
    print("\nSplicing-rescue scores:")
    print(_display_path(paths, paths.table_dir / "splicing_rescue_scores.tsv"))
    print("\nFinal report:")
    print(_display_path(paths, paths.final_report_path))
    print("\nFigures:")
    print(_display_path(paths, paths.figure_dir))
    print("\nOptional dashboard:")
    print("streamlit run dashboard/final_candidate_explorer.py")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse CLI args and run Module 5."""
    parser = argparse.ArgumentParser(description="Run Module 5 predicted splicing-rescue scoring and final SRSI ranking.")
    parser.add_argument("--config", required=True, help="Path to configs/splicing_rescue_scoring.yaml")
    args = parser.parse_args(argv)
    try:
        return run(args.config)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
