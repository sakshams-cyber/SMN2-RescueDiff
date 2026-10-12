from pathlib import Path
import argparse
import re
import shutil
import subprocess

import pandas as pd
import yaml


parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
args = parser.parse_args()

CONFIG = Path(args.config)

with CONFIG.open() as handle:
    cfg = yaml.safe_load(handle)

molecule_file = Path(cfg["inputs"]["molecules"])
receptor_file = Path(cfg["inputs"]["receptor_manifest"])

output_dir = Path(cfg["outputs"]["output_dir"])
score_file = Path(cfg["outputs"]["score_file"])
pose_root = Path(cfg["outputs"]["pose_dir"])
log_root = Path(cfg["outputs"]["log_dir"])

output_dir.mkdir(parents=True, exist_ok=True)
pose_root.mkdir(parents=True, exist_ok=True)
log_root.mkdir(parents=True, exist_ok=True)

vina = shutil.which("vina")

if vina is None:
    raise SystemExit("AutoDock Vina executable not found.")

molecules = pd.read_csv(
    molecule_file,
    sep="\t",
    dtype=str,
    keep_default_na=False,
)

receptors = pd.read_csv(
    receptor_file,
    sep="\t",
    dtype=str,
    keep_default_na=False,
)

if len(molecules) != cfg["run"]["expected_molecules"]:
    raise SystemExit(
        f"Expected {cfg['run']['expected_molecules']} molecules; "
        f"found {len(molecules)}."
    )

if len(receptors) != cfg["run"]["expected_receptors"]:
    raise SystemExit(
        f"Expected {cfg['run']['expected_receptors']} receptors; "
        f"found {len(receptors)}."
    )

ligand_columns = [
    "ligand_pdbqt_file",
    "prepared_ligand_pdbqt",
    "pdbqt_file",
    "ligand_file",
]

ligand_col = next(
    (x for x in ligand_columns if x in molecules.columns),
    None,
)

if ligand_col is None:
    raise SystemExit(
        "Top-40 input does not contain a prepared ligand PDBQT column."
    )

required_receptor_columns = {
    "receptor_id",
    "receptor_class",
    "receptor_file",
    "center_x",
    "center_y",
    "center_z",
    "size_x",
    "size_y",
    "size_z",
}

missing = required_receptor_columns - set(receptors.columns)

if missing:
    raise SystemExit(
        f"Receptor manifest missing columns: {sorted(missing)}"
    )

exhaustiveness = int(cfg["run"]["exhaustiveness"])
num_modes = int(cfg["run"]["num_modes"])
energy_range = float(cfg["run"]["energy_range"])
seed = int(cfg["run"]["seed"])


def safe(value):
    return re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        str(value),
    )


def parse_score(stdout, pose):
    for line in stdout.splitlines():
        match = re.match(
            r"^\s*1\s+(-?\d+(?:\.\d+)?)\s+",
            line,
        )

        if match:
            return float(match.group(1))

    if pose.exists():
        text = pose.read_text(errors="ignore")

        match = re.search(
            r"REMARK\s+VINA\s+RESULT:\s+"
            r"(-?\d+(?:\.\d+)?)",
            text,
        )

        if match:
            return float(match.group(1))

    return None


previous = pd.DataFrame()

if score_file.exists():
    previous = pd.read_csv(
        score_file,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

completed_pairs = set()

if len(previous):
    previous["_numeric"] = pd.to_numeric(
        previous["offtarget_deltaG"],
        errors="coerce",
    )

    mask = (
        previous["dock_status"].eq("completed")
        & previous["_numeric"].notna()
    )

    completed_pairs = set(
        zip(
            previous.loc[mask, "molecule_id"],
            previous.loc[mask, "receptor_id"],
        )
    )

    rows = previous.drop(
        columns=["_numeric"],
        errors="ignore",
    ).to_dict("records")
else:
    rows = []


total = len(molecules) * len(receptors)
counter = 0

for _, mol in molecules.iterrows():

    molecule_id = mol["molecule_id"].strip()
    ligand = Path(mol[ligand_col].strip())

    if not ligand.exists():
        raise SystemExit(
            f"Missing PDBQT for {molecule_id}: {ligand}"
        )

    for _, rec in receptors.iterrows():

        counter += 1

        receptor_id = rec["receptor_id"].strip()

        pair = (molecule_id, receptor_id)

        if pair in completed_pairs:
            continue

        receptor = Path(rec["receptor_file"].strip())

        if not receptor.exists():
            raise SystemExit(
                f"Missing receptor PDBQT: {receptor}"
            )

        receptor_pose_dir = (
            pose_root / safe(receptor_id)
        )

        receptor_log_dir = (
            log_root / safe(receptor_id)
        )

        receptor_pose_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        receptor_log_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        pose = (
            receptor_pose_dir
            / f"{safe(molecule_id)}.pdbqt"
        )

        log = (
            receptor_log_dir
            / f"{safe(molecule_id)}.log"
        )

        command = [
            vina,

            "--receptor",
            str(receptor),

            "--ligand",
            str(ligand),

            "--center_x",
            rec["center_x"],

            "--center_y",
            rec["center_y"],

            "--center_z",
            rec["center_z"],

            "--size_x",
            rec["size_x"],

            "--size_y",
            rec["size_y"],

            "--size_z",
            rec["size_z"],

            "--exhaustiveness",
            str(exhaustiveness),

            "--num_modes",
            str(num_modes),

            "--energy_range",
            str(energy_range),

            "--seed",
            str(seed),

            "--out",
            str(pose),
        ]

        print(
            f"[{counter}/{total}] "
            f"{molecule_id} × {receptor_id}"
        )

        status = "failed"
        affinity = ""
        error = ""

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=1800,
            )

            log.write_text(
                "COMMAND:\n"
                + " ".join(command)
                + "\n\nSTDOUT:\n"
                + result.stdout
                + "\n\nSTDERR:\n"
                + result.stderr
            )

            score = parse_score(
                result.stdout,
                pose,
            )

            if (
                result.returncode == 0
                and score is not None
                and pose.exists()
                and pose.stat().st_size > 0
            ):
                status = "completed"
                affinity = score

            else:
                error = (
                    result.stderr.strip()
                    or
                    result.stdout.strip()[-1500:]
                )

        except Exception as exc:
            error = str(exc)

        rows = [
            r for r in rows
            if not (
                r.get("molecule_id") == molecule_id
                and
                r.get("receptor_id") == receptor_id
            )
        ]

        rows.append({
            "molecule_id": molecule_id,
            "receptor_id": receptor_id,
            "receptor_class": rec["receptor_class"],
            "receptor_file": str(receptor),
            "ligand_pdbqt_file": str(ligand),
            "offtarget_deltaG": affinity,
            "dock_status": status,
            "is_dry_run_score": False,
            "pose_file": str(pose),
            "log_file": str(log),
            "error_message": error,
        })

        pd.DataFrame(rows).to_csv(
            score_file,
            sep="\t",
            index=False,
        )


results = pd.DataFrame(rows)

results["_numeric"] = pd.to_numeric(
    results["offtarget_deltaG"],
    errors="coerce",
)

valid = (
    results["dock_status"].eq("completed")
    & results["_numeric"].notna()
)

good = results[valid].copy()

counts = good.groupby(
    "molecule_id"
)["receptor_id"].nunique()

print("\nCROSS-TARGET VALIDATION")
print("-----------------------")
print("Expected jobs:", total)
print("Recorded jobs:", len(results))
print("Completed real jobs:", len(good))
print(
    "Unique molecules:",
    good["molecule_id"].nunique(),
)
print(
    "Unique receptors:",
    good["receptor_id"].nunique(),
)
print(
    "Molecules completing all receptors:",
    int(
        (
            counts
            == len(receptors)
        ).sum()
    ),
)

if len(good) != total:
    failed = results[~valid]

    failed.to_csv(
        output_dir / "failed_jobs.tsv",
        sep="\t",
        index=False,
    )

    raise SystemExit(
        f"Incomplete run: {len(good)}/{total} "
        "real docking jobs completed."
    )

print("\nCROSS-TARGET OFF-TARGET DOCKING: PASSED")
