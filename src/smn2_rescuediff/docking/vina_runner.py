"""AutoDock Vina subprocess wrapper."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

from .docking_engine import DockingResult
from .score_parser import parse_vina_best_score


def vina_available(executable: str = "vina") -> bool:
    """Return whether the configured Vina executable is available."""
    return shutil.which(executable) is not None


def validate_vina_input_files(receptor_file: str, ligand_file: str) -> None:
    """Validate receptor and ligand files before launching Vina."""
    receptor = Path(str(receptor_file))
    ligand = Path(str(ligand_file))
    if not receptor_file or str(receptor_file) == "NA" or receptor.suffix.lower() != ".pdbqt" or not receptor.exists():
        raise FileNotFoundError(f"Receptor file must be an existing PDBQT file for Vina: {receptor_file}")
    if not ligand_file or str(ligand_file) == "NA" or ligand.suffix.lower() != ".pdbqt" or not ligand.exists():
        raise FileNotFoundError("Ligand PDBQT file is missing; Vina requires PDBQT, not SDF.")


def run_vina(
    *,
    receptor_file: str,
    ligand_file: str,
    pose_file: str,
    config: dict[str, Any],
) -> DockingResult:
    """Run Vina for one receptor-ligand pair."""
    engine_cfg = config.get("docking_engine", {})
    receptor_cfg = config.get("receptor", {})
    executable = str(engine_cfg.get("vina_executable", "vina"))
    if not vina_available(executable):
        raise FileNotFoundError("AutoDock Vina executable was not found. Install Vina or use run.run_mode: dry_run.")
    validate_vina_input_files(receptor_file, ligand_file)
    box = receptor_cfg.get("docking_box", {})
    command = [
        executable,
        "--receptor",
        receptor_file,
        "--ligand",
        ligand_file,
        "--out",
        pose_file,
        "--center_x",
        str(box["center_x"]),
        "--center_y",
        str(box["center_y"]),
        "--center_z",
        str(box["center_z"]),
        "--size_x",
        str(box["size_x"]),
        "--size_y",
        str(box["size_y"]),
        "--size_z",
        str(box["size_z"]),
        "--exhaustiveness",
        str(engine_cfg.get("exhaustiveness", 8)),
        "--num_modes",
        str(engine_cfg.get("num_modes", 10)),
        "--energy_range",
        str(engine_cfg.get("energy_range", 3)),
    ]
    seed_policy = str(engine_cfg.get("random_seed_policy", "")).strip().lower()
    if seed_policy == "fixed_run_seed":
        command.extend(["--seed", str(config.get("run", {}).get("random_seed", 42))])
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=int(engine_cfg.get("timeout_seconds_per_job", 120)),
        )
    except subprocess.TimeoutExpired:
        return DockingResult("timeout", "NA", "NA", "NA", "NA", "vina", "real_docking", False, "Vina job timed out.")
    score = parse_vina_best_score(completed.stdout)
    if completed.returncode != 0 or score is None:
        return DockingResult(
            "failed",
            "NA",
            "NA",
            "NA",
            "NA",
            "vina",
            "real_docking",
            False,
            f"Vina failed or score could not be parsed. stderr={completed.stderr[:300]}",
        )
    return DockingResult("completed", score, 1, int(engine_cfg.get("num_modes", 10)), pose_file, "vina", "real_docking", False, "Real Vina docking score parsed.")
