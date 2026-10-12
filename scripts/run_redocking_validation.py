#!/usr/bin/env python3
"""Run the locked RNA-ligand redocking matrix with complete provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/processed/redocking_validation"
PROTOCOL_PATH = BASE / "redocking_protocol.yaml"
LOCK_PATH = BASE / "protocol_lock.json"
DEFAULT_OUTPUT = BASE / "docking"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nonempty(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def validate_protocol_lock() -> tuple[dict[str, Any], str]:
    if not nonempty(PROTOCOL_PATH) or not nonempty(LOCK_PATH):
        raise FileNotFoundError("The locked redocking protocol or lock file is missing.")
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    actual_hash = sha256(PROTOCOL_PATH)
    if actual_hash != lock.get("protocol_sha256"):
        raise RuntimeError(
            "The redocking protocol changed after locking. Refusing to dock; create a new experiment."
        )
    protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if not protocol.get("locked_before_docking_and_rmsd"):
        raise RuntimeError("Protocol does not declare a pre-analysis lock.")
    return protocol, actual_hash


def parse_scores(pose_path: Path) -> list[tuple[int, float, float, float]]:
    rows: list[tuple[int, float, float, float]] = []
    mode = 0
    for line in pose_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("MODEL"):
            try:
                mode = int(line.split()[1])
            except (IndexError, ValueError):
                mode += 1
        elif line.startswith("REMARK VINA RESULT:"):
            fields = line.split()
            if len(fields) < 6:
                raise ValueError(f"Malformed Vina score line in {pose_path}: {line}")
            if mode < 1:
                mode = len(rows) + 1
            rows.append((mode, float(fields[3]), float(fields[4]), float(fields[5])))
    if not rows:
        raise ValueError(f"No Vina result records found in {pose_path}")
    return rows


def run_job(job: dict[str, Any], protocol_hash: str, timeout: int) -> dict[str, Any]:
    job_dir = Path(job["job_dir"])
    job_dir.mkdir(parents=True, exist_ok=False)
    pose_path = job_dir / "docked_poses.pdbqt"
    stdout_path = job_dir / "vina_stdout.log"
    stderr_path = job_dir / "vina_stderr.log"
    metadata_path = job_dir / "job_metadata.json"

    command = [
        "vina",
        "--receptor",
        job["receptor_file"],
        "--ligand",
        job["ligand_file"],
        "--center_x",
        str(job["center_x"]),
        "--center_y",
        str(job["center_y"]),
        "--center_z",
        str(job["center_z"]),
        "--size_x",
        str(job["size_x"]),
        "--size_y",
        str(job["size_y"]),
        "--size_z",
        str(job["size_z"]),
        "--exhaustiveness",
        str(job["exhaustiveness"]),
        "--num_modes",
        str(job["num_modes"]),
        "--energy_range",
        str(job["energy_range"]),
        "--seed",
        str(job["seed"]),
        "--cpu",
        str(job["cpu"]),
        "--out",
        str(pose_path),
    ]
    started = datetime.now(timezone.utc)
    metadata = {
        **job,
        "protocol_sha256": protocol_hash,
        "receptor_sha256": sha256(Path(job["receptor_file"])),
        "ligand_sha256": sha256(Path(job["ligand_file"])),
        "command": shlex.join(command),
        "started_utc": started.isoformat(),
    }
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
        stdout_path.write_text(result.stdout, encoding="utf-8")
        stderr_path.write_text(result.stderr, encoding="utf-8")
        metadata["return_code"] = result.returncode
        if result.returncode != 0:
            raise RuntimeError(
                f"Vina returned code {result.returncode}: "
                f"{(result.stderr or result.stdout).strip()[-1000:]}"
            )
        if not nonempty(pose_path):
            raise RuntimeError("Vina reported success but did not write a non-empty pose file.")
        scores = parse_scores(pose_path)
        mode_rows = [
            {
                "pdb_id": job["pdb_id"],
                "ligand_name": job["ligand_name"],
                "molecule_id": job["molecule_id"],
                "model_index": job["model_index"],
                "seed": job["seed"],
                "mode": mode,
                "affinity_kcal_mol": affinity,
                "rmsd_lb_from_best": rmsd_lb,
                "rmsd_ub_from_best": rmsd_ub,
                "receptor_file": job["receptor_file"],
                "ligand_file": job["ligand_file"],
                "pose_file": str(pose_path),
                "is_dry_run_score": False,
                "dock_status": "completed",
                "protocol_sha256": protocol_hash,
            }
            for mode, affinity, rmsd_lb, rmsd_ub in scores
        ]
        metadata.update(
            {
                "status": "completed",
                "completed_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": (datetime.now(timezone.utc) - started).total_seconds(),
                "pose_file": str(pose_path),
                "modes_written": len(mode_rows),
                "best_affinity_kcal_mol": min(row["affinity_kcal_mol"] for row in mode_rows),
            }
        )
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        return {"status": "completed", "job": job, "mode_rows": mode_rows, "metadata": metadata}
    except Exception as exc:  # noqa: BLE001
        metadata.update(
            {
                "status": "failed",
                "completed_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": (datetime.now(timezone.utc) - started).total_seconds(),
                "failure_reason": str(exc),
            }
        )
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        return {"status": "failed", "job": job, "mode_rows": [], "metadata": metadata}


def build_jobs(protocol: dict[str, Any], output_dir: Path) -> list[dict[str, Any]]:
    preparation = pd.read_csv(BASE / "model_preparation_summary.tsv", sep="\t", dtype=str)
    ligands = pd.read_csv(BASE / "fresh_ligand_preparation_summary.tsv", sep="\t", dtype=str)
    structures = pd.read_csv(BASE / "structure_metadata.tsv", sep="\t", dtype=str)
    failures = structures[structures["preparation_status"] != "prepared"]
    if not failures.empty:
        raise RuntimeError(
            "Not all structures passed preparation: "
            + ", ".join(failures["pdb_id"].astype(str))
        )
    ligand_map = dict(zip(ligands["molecule_id"], ligands["pdbqt_file"]))
    name_map = dict(zip(structures["pdb_id"], structures["ligand_name"]))
    settings = protocol["docking"]
    jobs: list[dict[str, Any]] = []
    for row in preparation.to_dict(orient="records"):
        if row["preparation_status"] != "prepared":
            raise RuntimeError(f"A receptor model failed preparation: {row}")
        pdb_id = row["pdb_id"]
        model_index = int(row["model_index"])
        molecule_id = f"{pdb_id}_{row['ligand_id']}"
        receptor_file = Path(row["receptor_pdbqt"])
        ligand_file = Path(ligand_map.get(molecule_id, ""))
        if receptor_file.suffix.lower() != ".pdbqt" or not nonempty(receptor_file):
            raise FileNotFoundError(f"Invalid receptor PDBQT for {pdb_id} model {model_index}")
        if ligand_file.suffix.lower() != ".pdbqt" or not nonempty(ligand_file):
            raise FileNotFoundError(f"Invalid fresh ligand PDBQT for {molecule_id}")
        grid = protocol["structures"][pdb_id]["grid_box"]
        for seed in settings["seeds"]:
            jobs.append(
                {
                    "job_id": f"{pdb_id}_model_{model_index:02d}_seed_{seed}",
                    "pdb_id": pdb_id,
                    "ligand_name": name_map[pdb_id],
                    "molecule_id": molecule_id,
                    "model_index": model_index,
                    "seed": int(seed),
                    "receptor_file": str(receptor_file.resolve()),
                    "ligand_file": str(ligand_file.resolve()),
                    "job_dir": str(
                        output_dir / "jobs" / pdb_id / f"model_{model_index:02d}" / f"seed_{seed}"
                    ),
                    **grid,
                    "exhaustiveness": int(settings["exhaustiveness"]),
                    "num_modes": int(settings["num_modes"]),
                    "energy_range": int(settings["energy_range"]),
                    "cpu": int(settings["cpu_per_job"]),
                }
            )
    expected = sum(int(item["ensemble_models"]) for item in protocol["structures"].values()) * len(
        settings["seeds"]
    )
    if len(jobs) != expected:
        raise RuntimeError(f"Expected {expected} docking jobs but constructed {len(jobs)}")
    return jobs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing docking output: {output_dir}. Use a new directory."
        )
    protocol, protocol_hash = validate_protocol_lock()
    jobs = build_jobs(protocol, output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(jobs).to_csv(output_dir / "docking_job_manifest.tsv", sep="\t", index=False)
    run_metadata = {
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_path": str(PROTOCOL_PATH),
        "protocol_sha256": protocol_hash,
        "vina_version": protocol["software"]["vina"],
        "total_jobs": len(jobs),
        "parallel_workers": int(protocol["docking"]["parallel_workers"]),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8"
    )

    results: list[dict[str, Any]] = []
    workers = int(protocol["docking"]["parallel_workers"])
    timeout = int(protocol["docking"]["timeout_seconds_per_job"])
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_job, job, protocol_hash, timeout): job for job in jobs}
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            print(
                f"[{completed:03d}/{len(jobs):03d}] {result['job']['job_id']}: {result['status']}",
                flush=True,
            )

    mode_rows = [row for result in results for row in result["mode_rows"]]
    failed_rows = [
        {
            **result["job"],
            "failure_reason": result["metadata"].get("failure_reason", "Unknown failure"),
            "metadata_file": str(Path(result["job"]["job_dir"]) / "job_metadata.json"),
        }
        for result in results
        if result["status"] == "failed"
    ]
    pd.DataFrame(mode_rows).to_csv(output_dir / "redocking_mode_scores.tsv", sep="\t", index=False)
    failure_columns = [*jobs[0].keys(), "failure_reason", "metadata_file"]
    pd.DataFrame(failed_rows, columns=failure_columns).to_csv(
        output_dir / "failed_docking_jobs.tsv", sep="\t", index=False
    )
    completed_jobs = sum(result["status"] == "completed" for result in results)
    run_metadata.update(
        {
            "status": "completed" if not failed_rows else "completed_with_failures",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "completed_jobs": completed_jobs,
            "failed_jobs": len(failed_rows),
            "modes_scored": len(mode_rows),
            "all_scores_are_real": bool(mode_rows)
            and all(not row["is_dry_run_score"] for row in mode_rows),
        }
    )
    (output_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(run_metadata, indent=2))
    return 0 if not failed_rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
