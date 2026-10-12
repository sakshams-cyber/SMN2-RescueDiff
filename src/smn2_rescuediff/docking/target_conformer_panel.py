"""Permanent five-conformer SMN2 target docking support for Module 4."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .docking_engine import DockingResult
from .input_loader import load_molecules
from .ligand_preparer import prepare_ligands
from .molecule_selector import select_molecules_for_docking
from .vina_runner import run_vina, validate_vina_input_files


CONFORMER_MANIFEST_COLUMNS = ["conformer_id", "receptor_file", "source_structure", "source_model"]
LONG_COLUMNS = [
    "molecule_id",
    "conformer_id",
    "receptor_file",
    "target_deltaG",
    "dock_status",
    "is_dry_run_score",
    "pose_file",
]
ROBUSTNESS_COLUMNS = [
    "molecule_id",
    "num_conformers_completed",
    "mean_target_deltaG",
    "median_target_deltaG",
    "best_target_deltaG",
    "worst_target_deltaG",
    "std_target_deltaG",
    "target_deltaG_range",
]


def _resolve(path: str | Path, project_root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else project_root / candidate


def conformer_panel_enabled(config: dict[str, Any]) -> bool:
    """Return whether the permanent target-conformer panel mode is enabled."""
    return bool(config.get("target_conformer_panel", {}).get("enabled", False))


def _write(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)


def load_target_conformer_panel(config: dict[str, Any], project_root: Path) -> pd.DataFrame:
    """Load and validate the fixed SMN2 receptor-conformer manifest."""
    panel_cfg = config.get("target_conformer_panel", {})
    manifest_value = panel_cfg.get("manifest")
    if not manifest_value:
        raise ValueError("target_conformer_panel.manifest is required when target_conformer_panel.enabled is true.")
    manifest_path = _resolve(str(manifest_value), project_root)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Target conformer manifest is missing: {manifest_path}")
    panel = pd.read_csv(manifest_path, sep="\t")
    missing = set(CONFORMER_MANIFEST_COLUMNS) - set(panel.columns)
    if missing:
        raise ValueError(f"Target conformer manifest is missing columns: {sorted(missing)}")
    panel = panel[CONFORMER_MANIFEST_COLUMNS].copy()
    expected = int(panel_cfg.get("expected_conformer_count", 5))
    if len(panel) != expected:
        raise ValueError(f"Target conformer manifest must contain exactly {expected} receptors; found {len(panel)}.")
    if bool(panel_cfg.get("require_all_conformers", True)) and panel["conformer_id"].astype(str).duplicated().any():
        raise ValueError("Target conformer manifest contains duplicate conformer_id values.")
    resolved_files: list[str] = []
    for _, row in panel.iterrows():
        receptor = _resolve(str(row["receptor_file"]), project_root)
        if receptor.suffix.lower() != ".pdbqt":
            raise ValueError(f"Target conformer receptor must be a PDBQT file: {receptor}")
        if not receptor.exists() or receptor.stat().st_size == 0:
            raise FileNotFoundError(f"Target conformer receptor file is missing or empty: {receptor}")
        resolved_files.append(str(receptor))
    panel["receptor_file"] = resolved_files
    return panel


def _strict_select_molecules(molecules: pd.DataFrame, requested: int, seed: int) -> pd.DataFrame:
    if requested <= 0:
        raise ValueError("run.max_molecules_to_dock must be positive.")
    if len(molecules) < requested:
        raise ValueError(f"Fewer molecules than requested are available: requested {requested}, available {len(molecules)}.")
    selected = select_molecules_for_docking(molecules, requested, seed)
    if len(selected) != requested:
        raise ValueError(f"Module 4 conformer-panel mode must select exactly {requested} molecules; selected {len(selected)}.")
    return selected.reset_index(drop=True)


def _check_output_overwrite(output_dir: Path, config: dict[str, Any]) -> None:
    if bool(config.get("run", {}).get("overwrite_existing", False)):
        return
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Configured conformer-panel output directory already exists and is not empty: {output_dir}. "
            "Set run.overwrite_existing: true or choose a new outputs.output_dir."
        )


def _ligand_file_for(molecule_id: str, ligand_summary: pd.DataFrame) -> str:
    rows = ligand_summary[ligand_summary["molecule_id"].astype(str) == str(molecule_id)]
    if rows.empty:
        return "NA"
    value = str(rows.iloc[0].get("pdbqt_file", "NA"))
    return value if value != "NA" and Path(value).exists() else "NA"


def _long_row(molecule_id: str, conformer_id: str, receptor_file: str, result: DockingResult) -> dict[str, Any]:
    return {
        "molecule_id": molecule_id,
        "conformer_id": conformer_id,
        "receptor_file": receptor_file,
        "target_deltaG": result.delta_g,
        "dock_status": result.status,
        "is_dry_run_score": result.is_dry_run_score,
        "pose_file": result.pose_file,
        "notes": result.notes,
        "engine": result.engine,
    }


def calculate_target_conformer_robustness(long_scores: pd.DataFrame, expected_conformer_count: int) -> pd.DataFrame:
    """Aggregate long target-conformer scores into robust target-binding metrics."""
    if long_scores.empty:
        return pd.DataFrame(columns=ROBUSTNESS_COLUMNS)
    work = long_scores.copy()
    work["_score"] = pd.to_numeric(work["target_deltaG"], errors="coerce")
    completed = work[(work["dock_status"].astype(str) == "completed") & work["_score"].notna()].copy()
    rows: list[dict[str, Any]] = []
    for molecule_id, group in completed.groupby("molecule_id", sort=False):
        scores = group["_score"].astype(float)
        rows.append(
            {
                "molecule_id": molecule_id,
                "num_conformers_completed": int(len(scores)),
                "mean_target_deltaG": round(float(scores.mean()), 6),
                "median_target_deltaG": round(float(scores.median()), 6),
                "best_target_deltaG": round(float(scores.min()), 6),
                "worst_target_deltaG": round(float(scores.max()), 6),
                "std_target_deltaG": round(float(scores.std(ddof=0)), 6),
                "target_deltaG_range": round(float(scores.max() - scores.min()), 6),
            }
        )
    robustness = pd.DataFrame(rows, columns=ROBUSTNESS_COLUMNS)
    for molecule_id in work["molecule_id"].astype(str).drop_duplicates():
        if molecule_id not in set(robustness["molecule_id"].astype(str)):
            robustness = pd.concat(
                [
                    robustness,
                    pd.DataFrame([{"molecule_id": molecule_id, "num_conformers_completed": 0}], columns=ROBUSTNESS_COLUMNS),
                ],
                ignore_index=True,
            )
    robustness["all_conformers_completed"] = robustness["num_conformers_completed"] == int(expected_conformer_count)
    return robustness


def filter_target_conformer_candidates(robustness: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Rank robust target binders without letting one favorable conformer dominate."""
    if robustness.empty:
        return robustness.copy()
    panel_cfg = config.get("target_conformer_panel", {})
    filter_cfg = config.get("target_conformer_filter", {})
    expected = int(panel_cfg.get("expected_conformer_count", 5))
    work = robustness.copy()
    if bool(filter_cfg.get("require_all_conformers_completed", panel_cfg.get("require_all_conformers", True))):
        work = work[work["num_conformers_completed"] == expected].copy()
    work = work.sort_values(
        ["worst_target_deltaG", "mean_target_deltaG", "target_deltaG_range", "std_target_deltaG"],
        ascending=[True, True, True, True],
        na_position="last",
    ).reset_index(drop=True)
    if work.empty:
        return work
    top_n = filter_cfg.get("top_n_to_advance")
    top_fraction = filter_cfg.get("top_fraction_to_advance")
    if top_n is not None:
        limit = int(top_n)
    elif top_fraction is not None:
        limit = max(1, int(len(work) * float(top_fraction)))
    else:
        limit = len(work)
    work = work.head(max(0, limit)).copy()
    work["target_conformer_rank"] = range(1, len(work) + 1)
    work["advance_reason"] = "Ranked by worst_target_deltaG, then mean_target_deltaG, target_deltaG_range, and std_target_deltaG."
    return work


def _summary(
    *,
    config: dict[str, Any],
    selected: pd.DataFrame,
    panel: pd.DataFrame,
    long_scores: pd.DataFrame,
    robustness: pd.DataFrame,
    passed: pd.DataFrame,
    failed_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "project": config.get("project_name", "SMN2-RescueDiff"),
        "module": config.get("module_name", "SMN2 Target Docking and HuRDA Off-Target Counter-Screening"),
        "run_mode": "target_conformer_panel_docking",
        "run_name": config.get("run", {}).get("run_name", "target5"),
        "molecules_selected": int(len(selected)),
        "conformers_in_panel": int(len(panel)),
        "total_target_jobs": int(len(selected) * len(panel)),
        "completed_target_jobs": int((long_scores.get("dock_status", pd.Series(dtype=str)) == "completed").sum()) if not long_scores.empty else 0,
        "failed_target_jobs": int(len(failed_rows)),
        "molecules_with_all_conformers_completed": int((robustness.get("all_conformers_completed", pd.Series(dtype=bool)) == True).sum()) if not robustness.empty else 0,  # noqa: E712
        "molecules_advanced": int(len(passed)),
        "is_dry_run_score": False,
        "ranking_policy": "worst_target_deltaG ascending, then mean_target_deltaG ascending, then smaller range and standard deviation.",
        "limitations": [
            "Multi-conformer target docking is computational evidence only and does not prove binding or splicing rescue.",
            "The conformer filter reduces dependence on a single unusually favorable receptor pose but still requires experimental validation.",
        ],
    }


def run_target_conformer_panel(config: dict[str, Any], stable_paths, mode: str) -> int:
    """Run permanent five-conformer target docking."""
    if mode == "dry_run":
        raise ValueError("target_conformer_panel docking requires real Vina docking; dry-run scores are not allowed.")
    seed = int(config.get("run", {}).get("random_seed", 42))
    output_dir = stable_paths.output_dir
    _check_output_overwrite(output_dir, config)
    output_dir.mkdir(parents=True, exist_ok=True)
    ligand_dir = output_dir / "ligands"
    log_dir = output_dir / "logs"
    ligand_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    panel = load_target_conformer_panel(config, stable_paths.project_root)
    molecules = load_molecules(stable_paths.molecules_filtered)
    requested = int(config.get("run", {}).get("max_molecules_to_dock", 500))
    selected = _strict_select_molecules(molecules, requested, seed)
    ligand_summary = prepare_ligands(selected, ligand_dir, config)

    _write(selected, output_dir / "selected_molecules.tsv")
    _write(selected, output_dir / "selected_molecules_for_docking.tsv")
    _write(panel, output_dir / "conformer_panel_used.tsv")
    _write(ligand_summary, output_dir / "ligand_preparation_summary.tsv")
    (output_dir / "config_used.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (output_dir / "docking_config_used.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    long_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []
    print(f"Running permanent target conformer panel: {len(selected)} molecules x {len(panel)} SMN2 receptors.")
    for _, conformer in panel.iterrows():
        conformer_id = str(conformer["conformer_id"])
        receptor_file = str(conformer["receptor_file"])
        conformer_dir = output_dir / conformer_id
        pose_dir = conformer_dir / "poses"
        pose_dir.mkdir(parents=True, exist_ok=True)
        conformer_rows: list[dict[str, Any]] = []
        for _, molecule in selected.iterrows():
            molecule_id = str(molecule["molecule_id"])
            ligand_file = _ligand_file_for(molecule_id, ligand_summary)
            pose_file = str(pose_dir / f"{molecule_id}_{conformer_id}_out.pdbqt")
            try:
                validate_vina_input_files(receptor_file, ligand_file)
                result = run_vina(receptor_file=receptor_file, ligand_file=ligand_file, pose_file=pose_file, config=config)
            except Exception as exc:  # noqa: BLE001
                result = DockingResult("failed", "NA", "NA", "NA", "NA", "vina", "target_conformer_panel_docking", False, str(exc))
                failed_rows.append({"molecule_id": molecule_id, "conformer_id": conformer_id, "target_id": conformer_id, "status": "failed", "notes": str(exc)})
            row = _long_row(molecule_id, conformer_id, receptor_file, result)
            conformer_rows.append(row)
            long_rows.append(row)
        _write(pd.DataFrame(conformer_rows), conformer_dir / "target_docking_scores.tsv")

    long_scores = pd.DataFrame(long_rows)
    for column in LONG_COLUMNS:
        if column not in long_scores:
            long_scores[column] = "NA"
    robustness = calculate_target_conformer_robustness(long_scores, int(config.get("target_conformer_panel", {}).get("expected_conformer_count", 5)))
    passed = filter_target_conformer_candidates(robustness, config)
    _write(long_scores[LONG_COLUMNS + [column for column in ["notes", "engine"] if column in long_scores]], output_dir / "target_conformer_scores_long.tsv")
    _write(robustness, output_dir / "target_conformer_robustness_scores.tsv")
    _write(passed, output_dir / "molecules_passed_target_conformer_filter.tsv")
    _write(pd.DataFrame(failed_rows, columns=["molecule_id", "conformer_id", "target_id", "status", "notes"]), log_dir / "failed_docking_jobs.tsv")

    summary = _summary(config=config, selected=selected, panel=panel, long_scores=long_scores, robustness=robustness, passed=passed, failed_rows=failed_rows)
    (output_dir / "docking_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (output_dir / "run_metrics.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (output_dir / "RUN_SUMMARY.md").write_text(
        "\n".join(
            [
                f"# Module 4 Target Conformer Panel Run: {summary['run_name']}",
                "",
                f"- Run folder: `{output_dir}`",
                f"- Molecules selected: {summary['molecules_selected']}",
                f"- Conformers in fixed panel: {summary['conformers_in_panel']}",
                f"- Completed target jobs: {summary['completed_target_jobs']} / {summary['total_target_jobs']}",
                f"- Molecules advanced: {summary['molecules_advanced']}",
                "",
                "Ranking uses the worst conformer score first, then the mean target score, score range, and standard deviation.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (log_dir / "docking_run_log.txt").write_text(
        f"Target conformer panel docking completed for {len(selected)} molecules and {len(panel)} conformers.\n",
        encoding="utf-8",
    )
    print("\nModule 4 complete: target conformer panel docking finished.\n")
    print(f"Selected molecules: {output_dir / 'selected_molecules.tsv'}")
    print(f"Long conformer scores: {output_dir / 'target_conformer_scores_long.tsv'}")
    print(f"Robustness scores: {output_dir / 'target_conformer_robustness_scores.tsv'}")
    print(f"Advanced molecules: {output_dir / 'molecules_passed_target_conformer_filter.tsv'}")
    return 0
