"""Command-line interface for Module 4 docking and counter-screening."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .config import ensure_output_dirs, load_config, resolve_paths, run_mode, validate_required_inputs
from .decoy_panel_selector import select_decoy_panel
from .dry_run_docker import dry_run_decoy_score, dry_run_target_score
from .figure_writer import write_figures
from .input_loader import load_decoy_atlas, load_molecules
from .ligand_preparer import prepare_ligands
from .live_logger import LiveLogger
from .molecule_selector import select_molecules_for_docking
from .receptor_preparer import prepare_receptor
from .report_writer import build_summary, write_report, write_summary_json
from .selectivity_scorer import calculate_selectivity_scores
from .vina_runner import run_vina, validate_vina_input_files


TARGET_COLUMNS = [
    "molecule_id",
    "canonical_smiles",
    "target_id",
    "target_name",
    "receptor_file",
    "ligand_file",
    "dock_status",
    "target_deltaG",
    "best_pose_rank",
    "num_poses",
    "pose_file",
    "engine",
    "run_mode",
    "is_dry_run_score",
    "notes",
]

DECOY_COLUMNS = [
    "molecule_id",
    "canonical_smiles",
    "decoy_id",
    "decoy_category",
    "motif_type",
    "decoy_biological_weight",
    "decoy_smn2_similarity_score",
    "decoy_receptor_file",
    "ligand_file",
    "dock_status",
    "decoy_deltaG",
    "best_pose_rank",
    "num_poses",
    "pose_file",
    "engine",
    "run_mode",
    "is_dry_run_score",
    "notes",
]


def _write(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)


def _run_name(config: dict[str, Any], mode: str) -> str:
    run = config.setdefault("run", {})
    stage = str(config.get("staged_docking", {}).get("stage_name", "")).strip()
    name = str(run.get("run_name") or stage or f"{mode}_run").strip()
    run["run_name"] = name
    return name


def _run_paths(stable_paths, config: dict[str, Any], mode: str):
    run_dir = stable_paths.output_dir / "runs" / _run_name(config, mode)
    return replace(
        stable_paths,
        output_dir=run_dir,
        ligand_dir=run_dir / "ligands",
        pose_dir=run_dir / "poses",
        log_dir=run_dir / "logs",
        simulation_dir=run_dir / "simulation",
        report_path=run_dir / "reports" / stable_paths.report_path.name,
        figure_dir=run_dir / "figures",
        live_log_file=run_dir / "simulation" / "live_docking_events.jsonl",
        live_state_file=run_dir / "simulation" / "current_docking_state.json",
    )


def _copy_file_if_exists(source: Path, destination: Path) -> None:
    if source.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _sync_run_outputs_to_stable(run_paths, stable_paths) -> None:
    for filename in [
        "selected_molecules_for_docking.tsv",
        "ligand_preparation_summary.tsv",
        "receptor_preparation_summary.tsv",
        "selected_decoy_docking_panel.tsv",
        "smn2_target_docking_scores.tsv",
        "decoy_docking_scores.tsv",
        "molecule_selectivity_scores.tsv",
        "candidates_for_splicing_rescue.tsv",
        "docking_summary.json",
        "docking_config_used.yaml",
    ]:
        _copy_file_if_exists(run_paths.output_dir / filename, stable_paths.output_dir / filename)
    for filename in ["docking_run_log.txt", "failed_docking_jobs.tsv"]:
        _copy_file_if_exists(run_paths.log_dir / filename, stable_paths.log_dir / filename)
    for filename in ["live_docking_events.jsonl", "current_docking_state.json", "docking_simulation_summary.json"]:
        _copy_file_if_exists(run_paths.simulation_dir / filename, stable_paths.simulation_dir / filename)
    for filename in ["RUN_SUMMARY.md", "run_metrics.json"]:
        _copy_file_if_exists(run_paths.output_dir / filename, stable_paths.output_dir / filename)
    _copy_file_if_exists(run_paths.report_path, stable_paths.report_path)
    if run_paths.figure_dir.exists():
        stable_paths.figure_dir.mkdir(parents=True, exist_ok=True)
        for figure in run_paths.figure_dir.glob("*"):
            if figure.is_file():
                shutil.copy2(figure, stable_paths.figure_dir / figure.name)


def _write_run_artifacts(run_paths, config: dict[str, Any], summary: dict[str, Any]) -> None:
    metrics = dict(summary)
    metrics["run_name"] = _run_name(config, str(summary.get("run_mode", "run")))
    metrics["run_folder"] = str(run_paths.output_dir)
    (run_paths.output_dir / "run_metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    lines = [
        f"# Module 4 Run Summary: {metrics['run_name']}",
        "",
        f"- Run folder: `{run_paths.output_dir}`",
        f"- Run mode: `{summary.get('run_mode', 'NA')}`",
        f"- Stage: `{summary.get('stage_name', 'none')}`",
        f"- Molecules evaluated: {summary.get('molecules_evaluated', 'NA')}",
        f"- Target scores: {summary.get('target_scores', {}).get('count', 'NA')}",
        f"- Decoy scores: {summary.get('decoy_scores', {}).get('count', 'NA')}",
        f"- Advanced candidates: {summary.get('advanced_candidates', 'NA')}",
        "",
        "## Stable Downstream Outputs",
        "",
        "- `data/processed/docking/smn2_target_docking_scores.tsv`",
        "- `data/processed/docking/decoy_docking_scores.tsv`",
        "- `data/processed/docking/molecule_selectivity_scores.tsv`",
        "- `data/processed/docking/candidates_for_splicing_rescue.tsv`",
        "",
        "## Limitations",
        "",
        "Docking scores are computational predictions. Dry-run scores are simulated and are not scientific results.",
    ]
    (run_paths.output_dir / "RUN_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_existing_table(path: str | Path, columns: list[str]) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=columns)
    existing = pd.read_csv(p, sep="\t")
    for column in columns:
        if column not in existing.columns:
            existing[column] = "NA"
    return existing[columns]


def _filter_molecules_from_stage_scores(molecules: pd.DataFrame, config: dict[str, Any], project_root: Path) -> pd.DataFrame:
    """Restrict molecules to the top prior-stage target scores for staged docking."""
    stage_cfg = config.get("staged_docking", {})
    if not bool(stage_cfg.get("select_top_from_target_scores", False)):
        return molecules
    score_path = Path(str(stage_cfg.get("target_scores_path", "data/processed/docking/smn2_target_docking_scores.tsv")))
    score_path = score_path if score_path.is_absolute() else project_root / score_path
    if not score_path.exists():
        raise FileNotFoundError(f"Staged docking target-score input is missing: {score_path}")
    scores = pd.read_csv(score_path, sep="\t")
    if "molecule_id" not in scores or "target_deltaG" not in scores:
        raise ValueError(f"Staged docking target-score file must contain molecule_id and target_deltaG: {score_path}")
    top_n = int(stage_cfg.get("top_molecules_from_target_scores", config.get("run", {}).get("max_molecules_to_dock", 100)))
    ranked = scores.copy()
    ranked["_target_deltaG"] = pd.to_numeric(ranked["target_deltaG"], errors="coerce")
    ranked = ranked[ranked["_target_deltaG"].notna()].sort_values("_target_deltaG", ascending=True).head(top_n)
    molecule_order = {molecule_id: idx for idx, molecule_id in enumerate(ranked["molecule_id"].astype(str).tolist())}
    selected = molecules[molecules["molecule_id"].astype(str).isin(molecule_order)].copy()
    selected["_stage_rank"] = selected["molecule_id"].astype(str).map(molecule_order)
    return selected.sort_values("_stage_rank").drop(columns=["_stage_rank"]).reset_index(drop=True)


def _is_false_series(series: pd.Series) -> pd.Series:
    """Return a boolean mask for values that represent false."""
    return series.astype(str).str.strip().str.lower().isin({"false", "0", "no", "n"})


def _benchmark_molecule_selection_from_target_scores(
    config: dict[str, Any],
    project_root: Path,
) -> tuple[pd.DataFrame, str]:
    """Select benchmark decoy-docking molecules directly from prior target scores."""
    stage_cfg = config.get("staged_docking", {})
    score_path = Path(str(stage_cfg.get("target_scores_path", "data/processed/docking/smn2_target_docking_scores.tsv")))
    score_path = score_path if score_path.is_absolute() else project_root / score_path
    if not score_path.exists():
        raise FileNotFoundError(f"Staged benchmark target-score input is missing: {score_path}")

    scores = pd.read_csv(score_path, sep="\t")
    required = {"molecule_id", "canonical_smiles", "target_deltaG", "dock_status"}
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f"Staged benchmark target-score file is missing required columns {sorted(missing)}: {score_path}")

    top_n = int(stage_cfg.get("top_molecules_from_target_scores", config.get("run", {}).get("max_molecules_to_dock", 100)))
    max_molecules = int(config.get("run", {}).get("max_molecules_to_dock", top_n))
    requested = min(top_n, max_molecules)
    if requested <= 0:
        raise ValueError("staged_docking.top_molecules_from_target_scores and run.max_molecules_to_dock must request at least one molecule.")

    ranked = scores.copy()
    ranked["_target_deltaG"] = pd.to_numeric(ranked["target_deltaG"], errors="coerce")
    ranked = ranked[ranked["_target_deltaG"].notna()].copy()
    ranked = ranked[ranked["dock_status"].astype(str).str.lower() == "completed"].copy()

    require_real = bool(config.get("run", {}).get("require_real_docking_for_scientific_results", False))
    if require_real:
        if "is_dry_run_score" not in ranked.columns:
            raise ValueError("Staged benchmark requires real docking scores, but target_scores_path has no is_dry_run_score column.")
        ranked = ranked[_is_false_series(ranked["is_dry_run_score"])].copy()

    ranked = ranked.sort_values("_target_deltaG", ascending=True)
    ranked = ranked.drop_duplicates(subset=["molecule_id"], keep="first")
    available = len(ranked)
    if available < requested:
        raise ValueError(
            "Staged benchmark requested "
            f"{requested} molecules, but only {available} eligible target-score rows were available after filtering for "
            "dock_status=completed"
            + (", is_dry_run_score=false" if require_real else "")
            + ", and numeric target_deltaG. "
            "Provide more completed Stage A target scores or lower top_molecules_from_target_scores/max_molecules_to_dock."
        )

    selected = ranked.head(requested).copy().reset_index(drop=True)
    selected["stage_target_deltaG"] = selected["_target_deltaG"]
    selected["stage_target_rank"] = selected.index + 1
    selected["selection_source"] = str(score_path)
    selected["selection_notes"] = (
        "Selected directly from staged_docking.target_scores_path by ascending target_deltaG; "
        "benchmark mode ignored advancement filters and prior decoy/selectivity outputs."
    )
    selected = selected.drop(columns=["_target_deltaG"])

    base_columns = ["molecule_id", "canonical_smiles"]
    extra_columns = [column for column in selected.columns if column not in base_columns]
    message = (
        f"Benchmark staged selection chose {len(selected)} molecules from {score_path} by ascending target_deltaG "
        f"(requested=min(top_molecules_from_target_scores={top_n}, max_molecules_to_dock={max_molecules})={requested}; "
        "kept dock_status=completed"
        + (" and is_dry_run_score=false" if require_real else "")
        + "; ignored advancement filters, prior selectivity files, previous decoy outputs, and candidate lists)."
    )
    return selected[base_columns + extra_columns], message


def _prior_stage_target_scores(config: dict[str, Any], project_root: Path, selected: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Load prior target-only scores for Stage B/C decoy runs."""
    stage_cfg = config.get("staged_docking", {})
    if not bool(stage_cfg.get("reuse_prior_target_scores", False)):
        return pd.DataFrame(columns=TARGET_COLUMNS)
    score_path = Path(str(stage_cfg.get("target_scores_path", "data/processed/docking/smn2_target_docking_scores.tsv")))
    score_path = score_path if score_path.is_absolute() else project_root / score_path
    existing = _read_existing_table(score_path, TARGET_COLUMNS)
    if existing.empty:
        raise FileNotFoundError(f"No prior Stage A target scores are available for staged docking: {score_path}")
    selected_ids = set(selected["molecule_id"].astype(str))
    existing = existing[existing["molecule_id"].astype(str).isin(selected_ids)].copy()
    existing = existing[existing["dock_status"] == "completed"].copy()
    existing["run_mode"] = mode
    existing["notes"] = existing["notes"].astype(str) + " Reused prior-stage target score for staged decoy docking."
    return existing[TARGET_COLUMNS]


def _ligand_file_for(molecule_id: str, ligand_summary: pd.DataFrame, require_pdbqt: bool = False) -> str:
    row = ligand_summary[ligand_summary["molecule_id"] == molecule_id]
    if row.empty:
        return "NA"
    record = row.iloc[0]
    if require_pdbqt:
        pdbqt_file = str(record.get("pdbqt_file", "NA"))
        return pdbqt_file if pdbqt_file != "NA" and Path(pdbqt_file).exists() else "NA"
    return str(record.get("sdf_file", "NA"))


def _target_row(molecule: pd.Series, ligand_file: str, receptor_file: str, result, mode: str) -> dict[str, Any]:
    data = result.to_dict()
    return {
        "molecule_id": molecule["molecule_id"],
        "canonical_smiles": molecule["canonical_smiles"],
        "target_id": "SMN2_exon7",
        "target_name": "SMN2 exon 7 splicing-related RNA target",
        "receptor_file": receptor_file,
        "ligand_file": ligand_file,
        "dock_status": data["dock_status"],
        "target_deltaG": data["delta_g"],
        "best_pose_rank": data["best_pose_rank"],
        "num_poses": data["num_poses"],
        "pose_file": data["pose_file"],
        "engine": data["engine"],
        "run_mode": mode,
        "is_dry_run_score": data["is_dry_run_score"],
        "notes": data["notes"],
    }


def _decoy_row(molecule: pd.Series, decoy: pd.Series, ligand_file: str, result, mode: str) -> dict[str, Any]:
    data = result.to_dict()
    return {
        "molecule_id": molecule["molecule_id"],
        "canonical_smiles": molecule["canonical_smiles"],
        "decoy_id": decoy.get("decoy_id", "NA"),
        "decoy_category": decoy.get("decoy_category", "NA"),
        "motif_type": decoy.get("motif_type", "NA"),
        "decoy_biological_weight": decoy.get("biological_weight", decoy.get("decoy_biological_weight", "NA")),
        "decoy_smn2_similarity_score": decoy.get("smn2_similarity_score", "NA"),
        "decoy_receptor_file": decoy.get("decoy_receptor_file", "NA"),
        "ligand_file": ligand_file,
        "dock_status": data["dock_status"],
        "decoy_deltaG": data["delta_g"],
        "best_pose_rank": data["best_pose_rank"],
        "num_poses": data["num_poses"],
        "pose_file": data["pose_file"],
        "engine": data["engine"],
        "run_mode": mode,
        "is_dry_run_score": data["is_dry_run_score"],
        "notes": data["notes"],
    }


def _empty_outputs(paths, config: dict[str, Any], selected: pd.DataFrame, ligand_summary: pd.DataFrame, receptor_summary: pd.DataFrame, decoy_panel: pd.DataFrame, warnings: list[str]) -> dict[str, Any]:
    target_scores = pd.DataFrame(columns=TARGET_COLUMNS)
    decoy_scores = pd.DataFrame(columns=DECOY_COLUMNS)
    selectivity = calculate_selectivity_scores(target_scores, decoy_scores, config)
    _write(target_scores, paths.output_dir / "smn2_target_docking_scores.tsv")
    _write(decoy_scores, paths.output_dir / "decoy_docking_scores.tsv")
    _write(selectivity, paths.output_dir / "molecule_selectivity_scores.tsv")
    _write(selectivity.head(0), paths.output_dir / "candidates_for_splicing_rescue.tsv")
    _write(pd.DataFrame(columns=["molecule_id", "target_id", "status", "notes"]), paths.log_dir / "failed_docking_jobs.tsv")
    summary = build_summary(config, selected, target_scores, decoy_scores, selectivity, warnings)
    write_summary_json(summary, paths.output_dir / "docking_summary.json")
    (paths.output_dir / "docking_config_used.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    write_figures(target_scores, decoy_scores, selectivity, paths.live_log_file, paths.figure_dir, config)
    write_report(
        config=config,
        summary=summary,
        receptor_summary=receptor_summary,
        ligand_summary=ligand_summary,
        decoy_panel=decoy_panel,
        output_files=_output_file_list(),
        report_path=paths.report_path,
    )
    return summary


def _output_file_list() -> list[str]:
    return [
        "data/processed/docking/selected_molecules_for_docking.tsv",
        "data/processed/docking/ligand_preparation_summary.tsv",
        "data/processed/docking/receptor_preparation_summary.tsv",
        "data/processed/docking/selected_decoy_docking_panel.tsv",
        "data/processed/docking/smn2_target_docking_scores.tsv",
        "data/processed/docking/decoy_docking_scores.tsv",
        "data/processed/docking/molecule_selectivity_scores.tsv",
        "data/processed/docking/candidates_for_splicing_rescue.tsv",
        "data/processed/docking/docking_summary.json",
        "data/processed/docking/docking_config_used.yaml",
        "data/processed/docking/logs/docking_run_log.txt",
        "data/processed/docking/logs/failed_docking_jobs.tsv",
        "data/processed/docking/simulation/live_docking_events.jsonl",
        "data/processed/docking/simulation/current_docking_state.json",
        "data/processed/docking/simulation/docking_simulation_summary.json",
    ]


def run(config_path: str | Path) -> int:
    """Run Module 4."""
    config = load_config(config_path)
    stable_paths = resolve_paths(config, config_path)
    mode = run_mode(config)
    _run_name(config, mode)
    paths = _run_paths(stable_paths, config, mode)
    ensure_output_dirs(stable_paths)
    ensure_output_dirs(paths)
    warnings = validate_required_inputs(stable_paths, config)
    seed = int(config.get("run", {}).get("random_seed", 42))
    live_cfg = config.get("live_simulation", {})
    logger = LiveLogger(paths.live_log_file, paths.live_state_file, enabled=bool(live_cfg.get("enable_live_logging", True)))
    run_log: list[str] = []

    def log(message: str) -> None:
        print(message)
        run_log.append(message)

    logger.event("module_started", run_mode=mode, status="running", message="Module 4 docking and counter-screening started.")
    logger.state(run_status="running", run_mode=mode, total_jobs=0, jobs_completed=0, last_message="Module started")

    if mode == "dry_run":
        warning = "WARNING: Running in dry-run mode. These are simulated docking scores and are not scientific results."
        warnings.append("DRY-RUN SIMULATION ONLY — NOT SCIENTIFIC DOCKING RESULTS")
        log(warning)

    if mode == "analyze_existing":
        target_scores = pd.read_csv(stable_paths.output_dir / "smn2_target_docking_scores.tsv", sep="\t")
        decoy_scores = pd.read_csv(stable_paths.output_dir / "decoy_docking_scores.tsv", sep="\t")
        selected = pd.read_csv(stable_paths.output_dir / "selected_molecules_for_docking.tsv", sep="\t")
        ligand_summary = pd.read_csv(stable_paths.output_dir / "ligand_preparation_summary.tsv", sep="\t")
        receptor_summary = pd.read_csv(stable_paths.output_dir / "receptor_preparation_summary.tsv", sep="\t")
        decoy_panel = pd.read_csv(stable_paths.output_dir / "selected_decoy_docking_panel.tsv", sep="\t")
        selectivity = calculate_selectivity_scores(target_scores, decoy_scores, config)
        _write(target_scores, paths.output_dir / "smn2_target_docking_scores.tsv")
        _write(decoy_scores, paths.output_dir / "decoy_docking_scores.tsv")
        _write(selected, paths.output_dir / "selected_molecules_for_docking.tsv")
        _write(ligand_summary, paths.output_dir / "ligand_preparation_summary.tsv")
        _write(receptor_summary, paths.output_dir / "receptor_preparation_summary.tsv")
        _write(decoy_panel, paths.output_dir / "selected_decoy_docking_panel.tsv")
        _write(selectivity, paths.output_dir / "molecule_selectivity_scores.tsv")
        candidates = selectivity[selectivity["advance_to_splicing_rescue"] == True][  # noqa: E712
            ["molecule_id", "canonical_smiles", "target_deltaG", "weighted_selectivity_gap", "off_target_risk_score", "advance_to_splicing_rescue", "run_mode", "notes"]
        ] if "advance_to_splicing_rescue" in selectivity else selectivity.head(0)
        _write(candidates, paths.output_dir / "candidates_for_splicing_rescue.tsv")
    else:
        log("Loading Module 3 molecule library.")
        molecules = load_molecules(paths.molecules_filtered)
        stage_cfg = config.get("staged_docking", {})
        if bool(stage_cfg.get("select_top_from_target_scores", False)) and bool(stage_cfg.get("ignore_advance_filters_for_benchmark", False)):
            selected, selection_message = _benchmark_molecule_selection_from_target_scores(config, paths.project_root)
            log(selection_message)
        else:
            molecules = _filter_molecules_from_stage_scores(molecules, config, paths.project_root)
            selected = select_molecules_for_docking(
                molecules,
                int(config.get("run", {}).get("max_molecules_to_dock", 500)),
                seed,
            )
            if bool(stage_cfg.get("select_top_from_target_scores", False)):
                log(
                    f"Scientific staged selection retained {len(selected)} molecules after target-score filtering "
                    "and the configured molecule-selection cap."
                )
        _write(selected, paths.output_dir / "selected_molecules_for_docking.tsv")

        logger.event("ligand_preparation_started", run_mode=mode, status="running", message="Preparing ligands with RDKit.")
        logger.state(last_message="Preparing ligands", total_jobs=len(selected), jobs_completed=0)
        ligand_summary = prepare_ligands(selected, paths.ligand_dir, config)
        _write(ligand_summary, paths.output_dir / "ligand_preparation_summary.tsv")
        logger.event("ligand_preparation_completed", run_mode=mode, status="completed", message="Ligand preparation completed.")

        logger.event("receptor_validation_started", run_mode=mode, status="running", message="Validating SMN2 receptor configuration.")
        receptor_summary = prepare_receptor(config, paths.smn2_receptor_file, mode)
        _write(receptor_summary, paths.output_dir / "receptor_preparation_summary.tsv")
        logger.event("receptor_validation_completed", run_mode=mode, status="completed", message="Receptor validation completed.")

        if config.get("decoy_panel", {}).get("use_decoy_panel", True):
            decoys = load_decoy_atlas(paths.decoy_atlas_full)
            decoy_panel = select_decoy_panel(decoys, config)
        else:
            decoy_panel = pd.DataFrame()
        _write(decoy_panel, paths.output_dir / "selected_decoy_docking_panel.tsv")
        logger.event("decoy_panel_selected", run_mode=mode, status="completed", message=f"Selected {len(decoy_panel)} HuRDA decoys.")

        (paths.output_dir / "docking_config_used.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

        if mode == "prepare_only":
            warnings.append("prepare_only mode completed setup without running docking.")
            summary = _empty_outputs(paths, config, selected, ligand_summary, receptor_summary, decoy_panel, warnings)
            _write_run_artifacts(paths, config, summary)
            _sync_run_outputs_to_stable(paths, stable_paths)
            logger.event("module_completed", run_mode=mode, status="completed", message="Prepare-only run completed.")
            logger.state(run_status="completed", last_message="Prepare-only run completed")
            return 0

        target_rows: list[dict[str, Any]] = []
        decoy_rows: list[dict[str, Any]] = []
        failed_rows: list[dict[str, Any]] = []
        resume = bool(config.get("run", {}).get("resume_existing_results", False))
        completed_target_molecules: set[str] = set()
        completed_decoy_pairs: set[tuple[str, str]] = set()
        prior_targets = _prior_stage_target_scores(config, paths.project_root, selected, mode)
        if not prior_targets.empty:
            target_rows.extend(prior_targets.to_dict(orient="records"))
            completed_target_molecules.update(prior_targets["molecule_id"].astype(str).tolist())
            log(f"Reusing {len(prior_targets)} prior Stage A target scores for this staged decoy run.")
        if resume:
            existing_targets = _read_existing_table(paths.output_dir / "smn2_target_docking_scores.tsv", TARGET_COLUMNS)
            existing_decoys = _read_existing_table(paths.output_dir / "decoy_docking_scores.tsv", DECOY_COLUMNS)
            existing_failed = _read_existing_table(paths.log_dir / "failed_docking_jobs.tsv", ["molecule_id", "target_id", "status", "notes"])
            already_loaded_targets = {str(row["molecule_id"]) for row in target_rows}
            existing_targets = existing_targets[~existing_targets["molecule_id"].astype(str).isin(already_loaded_targets)].copy()
            target_rows.extend(existing_targets.to_dict(orient="records"))
            decoy_rows.extend(existing_decoys.to_dict(orient="records"))
            failed_rows.extend(existing_failed.to_dict(orient="records"))
            if not existing_targets.empty:
                completed_target_molecules.update(existing_targets.loc[existing_targets["dock_status"] == "completed", "molecule_id"].astype(str))
            if not existing_decoys.empty:
                done = existing_decoys[existing_decoys["dock_status"] == "completed"]
                completed_decoy_pairs = set(zip(done["molecule_id"].astype(str), done["decoy_id"].astype(str)))
            if completed_target_molecules or completed_decoy_pairs:
                log(
                    "Resuming existing docking results: "
                    f"{len(completed_target_molecules)} completed target jobs and {len(completed_decoy_pairs)} completed decoy jobs will be skipped."
                )
        run_decoys = mode in {"dry_run", "target_plus_decoy_docking"}
        total_jobs = len(selected) * (1 + (len(decoy_panel) if run_decoys else 0))
        completed_jobs = 0
        best_target = None

        for _, molecule in selected.iterrows():
            ligand_file = _ligand_file_for(str(molecule["molecule_id"]), ligand_summary, require_pdbqt=mode != "dry_run")
            molecule_id = str(molecule["molecule_id"])
            if molecule_id in completed_target_molecules:
                logger.event(
                    "job_skipped",
                    run_mode=mode,
                    molecule_id=molecule_id,
                    target_type="SMN2_target",
                    target_id="SMN2_exon7",
                    status="completed",
                    message=f"Skipped completed SMN2 target docking for {molecule_id}.",
                )
            else:
                completed_jobs += 1
                logger.event(
                    "job_started",
                    run_mode=mode,
                    molecule_id=molecule["molecule_id"],
                    canonical_smiles=molecule.get("canonical_smiles"),
                    target_type="SMN2_target",
                    target_id="SMN2_exon7",
                    job_index=completed_jobs,
                    total_jobs=total_jobs,
                    status="running",
                    message=f"Docking {molecule['molecule_id']} against SMN2 target",
                )
                try:
                    if mode == "dry_run":
                        result = dry_run_target_score(molecule, seed)
                        receptor_file = "NA"
                    else:
                        pose_file = str(paths.pose_dir / f"{molecule['molecule_id']}_smn2_out.pdbqt")
                        validate_vina_input_files(str(paths.smn2_receptor_file), ligand_file)
                        result = run_vina(
                            receptor_file=str(paths.smn2_receptor_file),
                            ligand_file=ligand_file,
                            pose_file=pose_file,
                            config=config,
                        )
                        receptor_file = str(paths.smn2_receptor_file)
                    target_rows.append(_target_row(molecule, ligand_file, receptor_file, result, mode))
                    score = result.delta_g if isinstance(result.delta_g, float) else None
                    if score is not None:
                        best_target = score if best_target is None else min(best_target, score)
                    logger.event(
                        "job_completed",
                        run_mode=mode,
                        molecule_id=molecule["molecule_id"],
                        canonical_smiles=molecule.get("canonical_smiles"),
                        target_type="SMN2_target",
                        target_id="SMN2_exon7",
                        job_index=completed_jobs,
                        total_jobs=total_jobs,
                        status=result.status,
                        current_score=result.delta_g if isinstance(result.delta_g, float) else None,
                        best_target_score=best_target,
                        message=f"Completed SMN2 target docking for {molecule['molecule_id']}",
                    )
                except Exception as exc:
                    failed_rows.append({"molecule_id": molecule["molecule_id"], "target_id": "SMN2_exon7", "status": "failed", "notes": str(exc)})
                    logger.event("job_failed", run_mode=mode, molecule_id=molecule["molecule_id"], target_type="SMN2_target", target_id="SMN2_exon7", status="failed", message=str(exc))
            logger.state(
                current_molecule_id=molecule["molecule_id"],
                current_smiles=molecule.get("canonical_smiles"),
                current_target_id="SMN2_exon7",
                current_target_type="SMN2_target",
                current_decoy_category=None,
                jobs_completed=completed_jobs,
                total_jobs=total_jobs,
                latest_score=target_rows[-1]["target_deltaG"] if target_rows else None,
                best_target_score_so_far=best_target,
                last_message=f"Completed SMN2 target docking for {molecule['molecule_id']}",
            )

            if run_decoys and not decoy_panel.empty:
                for _, decoy in decoy_panel.iterrows():
                    decoy_id = str(decoy.get("decoy_id", "NA"))
                    if (molecule_id, decoy_id) in completed_decoy_pairs:
                        logger.event(
                            "job_skipped",
                            run_mode=mode,
                            molecule_id=molecule_id,
                            target_type="decoy",
                            target_id=decoy_id,
                            decoy_category=decoy.get("decoy_category"),
                            status="completed",
                            message=f"Skipped completed decoy docking for {molecule_id} vs {decoy_id}.",
                        )
                        continue
                    completed_jobs += 1
                    logger.event(
                        "job_started",
                        run_mode=mode,
                        molecule_id=molecule["molecule_id"],
                        canonical_smiles=molecule.get("canonical_smiles"),
                        target_type="decoy",
                        target_id=decoy.get("decoy_id"),
                        decoy_category=decoy.get("decoy_category"),
                        job_index=completed_jobs,
                        total_jobs=total_jobs,
                        status="running",
                        message=f"Docking {molecule['molecule_id']} against {decoy.get('decoy_id')}",
                    )
                    if mode == "dry_run":
                        result = dry_run_decoy_score(molecule, decoy, seed)
                    else:
                        from .docking_engine import DockingResult

                        decoy_receptor = str(decoy.get("decoy_receptor_file", "NA"))
                        try:
                            validate_vina_input_files(decoy_receptor, ligand_file)
                            pose_file = str(paths.pose_dir / f"{molecule['molecule_id']}_{decoy_id}_out.pdbqt")
                            result = run_vina(receptor_file=decoy_receptor, ligand_file=ligand_file, pose_file=pose_file, config=config)
                        except Exception as exc:
                            failed_rows.append(
                                {
                                    "molecule_id": molecule["molecule_id"],
                                    "target_id": decoy.get("decoy_id"),
                                    "status": "failed",
                                    "notes": str(exc),
                                }
                            )
                            result = DockingResult("not_run", "NA", "NA", "NA", "NA", "vina", mode, False, str(exc))
                    decoy_rows.append(_decoy_row(molecule, decoy, ligand_file, result, mode))
                    logger.event(
                        "job_completed",
                        run_mode=mode,
                        molecule_id=molecule["molecule_id"],
                        canonical_smiles=molecule.get("canonical_smiles"),
                        target_type="decoy",
                        target_id=decoy.get("decoy_id"),
                        decoy_category=decoy.get("decoy_category"),
                        job_index=completed_jobs,
                        total_jobs=total_jobs,
                        status=result.status,
                        current_score=result.delta_g if isinstance(result.delta_g, float) else None,
                        best_target_score=best_target,
                        message=f"Completed decoy docking for {molecule['molecule_id']} vs {decoy.get('decoy_id')}",
                    )
                    logger.state(
                        current_molecule_id=molecule["molecule_id"],
                        current_smiles=molecule.get("canonical_smiles"),
                        current_target_id=decoy.get("decoy_id"),
                        current_target_type="decoy",
                        current_decoy_category=decoy.get("decoy_category"),
                        jobs_completed=completed_jobs,
                        total_jobs=total_jobs,
                        latest_score=result.delta_g if isinstance(result.delta_g, float) else None,
                        best_target_score_so_far=best_target,
                        last_message=f"Completed decoy docking for {molecule['molecule_id']}",
                    )

        target_scores = pd.DataFrame(target_rows, columns=TARGET_COLUMNS)
        decoy_scores = pd.DataFrame(decoy_rows, columns=DECOY_COLUMNS)
        selectivity = calculate_selectivity_scores(target_scores, decoy_scores, config)
        _write(target_scores, paths.output_dir / "smn2_target_docking_scores.tsv")
        _write(decoy_scores, paths.output_dir / "decoy_docking_scores.tsv")
        _write(selectivity, paths.output_dir / "molecule_selectivity_scores.tsv")
        candidates = selectivity[selectivity["advance_to_splicing_rescue"] == True][  # noqa: E712
            ["molecule_id", "canonical_smiles", "target_deltaG", "weighted_selectivity_gap", "off_target_risk_score", "advance_to_splicing_rescue", "run_mode", "notes"]
        ]
        _write(candidates, paths.output_dir / "candidates_for_splicing_rescue.tsv")
        _write(pd.DataFrame(failed_rows, columns=["molecule_id", "target_id", "status", "notes"]), paths.log_dir / "failed_docking_jobs.tsv")

    summary = build_summary(config, selected, target_scores, decoy_scores, selectivity, warnings)
    write_summary_json(summary, paths.output_dir / "docking_summary.json")
    write_summary_json(summary, paths.simulation_dir / "docking_simulation_summary.json")
    (paths.output_dir / "docking_config_used.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    write_figures(target_scores, decoy_scores, selectivity, paths.live_log_file, paths.figure_dir, config)
    write_report(
        config=config,
        summary=summary,
        receptor_summary=receptor_summary,
        ligand_summary=ligand_summary,
        decoy_panel=decoy_panel,
        output_files=_output_file_list(),
        report_path=paths.report_path,
    )
    (paths.log_dir / "docking_run_log.txt").write_text("\n".join(run_log + warnings) + "\n", encoding="utf-8")
    _write_run_artifacts(paths, config, summary)
    _sync_run_outputs_to_stable(paths, stable_paths)
    logger.event("scores_updated", run_mode=mode, status="completed", message="Selectivity scores and figures written.")
    logger.event("module_completed", run_mode=mode, status="completed", message="Module 4 complete.")
    logger.state(run_status="completed", jobs_completed=summary.get("target_scores", {}).get("count", 0) + summary.get("decoy_scores", {}).get("count", 0), last_message="Module 4 complete")

    print("\nModule 4 complete: SMN2 docking and HuRDA counter-screening finished.\n")
    print(f"Run mode: {mode}")
    print(f"Molecules evaluated: {len(selected)}")
    print(f"SMN2 target docking results: {paths.output_dir / 'smn2_target_docking_scores.tsv'}")
    print(f"Decoy docking results: {paths.output_dir / 'decoy_docking_scores.tsv'}")
    print(f"Selectivity scores: {paths.output_dir / 'molecule_selectivity_scores.tsv'}")
    print(f"Candidates for Module 5: {paths.output_dir / 'candidates_for_splicing_rescue.tsv'}")
    print(f"Report: {paths.report_path}")
    print(f"Figures: {paths.figure_dir}")
    print(f"Live simulation log: {paths.live_log_file}\n")
    print("To view live simulation dashboard:")
    print("streamlit run dashboard/docking_live_app.py")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run SMN2 docking and HuRDA counter-screening.")
    parser.add_argument("--config", required=True, help="Path to configs/docking_counter_screening.yaml")
    args = parser.parse_args(argv)
    try:
        return run(args.config)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
