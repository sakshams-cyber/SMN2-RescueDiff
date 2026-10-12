from pathlib import Path
import argparse
import json
import re
import shutil
import subprocess
from datetime import datetime

import pandas as pd
import yaml


parser = argparse.ArgumentParser()

parser.add_argument("--scores", required=True)
parser.add_argument("--config", required=True)
parser.add_argument("--expected-decoys", required=True, type=int)
parser.add_argument("--repair-dir", required=True)

args = parser.parse_args()


SCORES = Path(args.scores)
CONFIG = Path(args.config)
EXPECTED_DECOYS = args.expected_decoys
REPAIR_DIR = Path(args.repair_dir)

POSE_DIR = REPAIR_DIR / "poses"
LOG_DIR = REPAIR_DIR / "logs"

REPAIR_SCORE_FILE = (
    REPAIR_DIR / "repair_decoy_docking_scores.tsv"
)

FAILED_FILE = REPAIR_DIR / "failed_jobs.tsv"

COMPLETE_DIR = REPAIR_DIR.parent / "repaired_complete"

COMPLETE_FILE = (
    COMPLETE_DIR / "decoy_docking_scores.tsv"
)


# ============================================================
# BASIC CHECKS
# ============================================================

if not SCORES.exists():
    raise SystemExit(f"Score file missing:\n{SCORES}")

if not CONFIG.exists():
    raise SystemExit(f"Config missing:\n{CONFIG}")

vina = shutil.which("vina")

if vina is None:
    raise SystemExit(
        "AutoDock Vina executable was not found."
    )

POSE_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
COMPLETE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD ORIGINAL RESULTS
# ============================================================

original = pd.read_csv(
    SCORES,
    sep="\t",
    dtype=str,
    keep_default_na=False,
    low_memory=False,
)

required = {
    "molecule_id",
    "decoy_id",
    "decoy_deltaG",
    "dock_status",
}

missing_columns = required - set(original.columns)

if missing_columns:
    raise SystemExit(
        f"Missing required columns: "
        f"{sorted(missing_columns)}"
    )

original["molecule_id"] = (
    original["molecule_id"]
    .astype(str)
    .str.strip()
)

original["decoy_id"] = (
    original["decoy_id"]
    .astype(str)
    .str.strip()
)

original["_score_numeric"] = pd.to_numeric(
    original["decoy_deltaG"],
    errors="coerce",
)

valid_original = (
    original["dock_status"]
    .astype(str)
    .str.strip()
    .str.lower()
    .eq("completed")
    &
    original["_score_numeric"].notna()
)

if "is_dry_run_score" in original.columns:
    valid_original &= (
        original["is_dry_run_score"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["false", "0"])
    )


# ============================================================
# VERIFY EXPECTED GRID CAN BE RECONSTRUCTED
# ============================================================

molecules = sorted(
    original["molecule_id"]
    .replace("", pd.NA)
    .dropna()
    .unique()
)

decoys = sorted(
    original["decoy_id"]
    .replace("", pd.NA)
    .dropna()
    .unique()
)

print("\nORIGINAL RUN")
print("------------")
print("Recorded rows:", len(original))
print(
    "Valid completed rows:",
    int(valid_original.sum()),
)
print("Unique molecules:", len(molecules))
print("Unique decoys:", len(decoys))

if len(molecules) != 150:
    raise SystemExit(
        f"\nExpected 150 molecule IDs in the original "
        f"table, found {len(molecules)}.\n"
        "Do not continue automatically."
    )

if len(decoys) != EXPECTED_DECOYS:
    raise SystemExit(
        f"\nExpected {EXPECTED_DECOYS} decoy IDs in "
        f"the original table, found {len(decoys)}.\n"
        "Do not continue automatically."
    )

expected_grid = pd.MultiIndex.from_product(
    [molecules, decoys],
    names=["molecule_id", "decoy_id"],
).to_frame(index=False)

valid_pairs = (
    original.loc[
        valid_original,
        ["molecule_id", "decoy_id"],
    ]
    .drop_duplicates()
)

missing_jobs = expected_grid.merge(
    valid_pairs.assign(_valid=True),
    on=["molecule_id", "decoy_id"],
    how="left",
)

missing_jobs = missing_jobs[
    missing_jobs["_valid"].isna()
][
    ["molecule_id", "decoy_id"]
].reset_index(drop=True)

print("\nExpected jobs:", 150 * EXPECTED_DECOYS)
print("Jobs requiring repair:", len(missing_jobs))


# ============================================================
# SAVE REPAIR MANIFEST
# ============================================================

missing_jobs.to_csv(
    REPAIR_DIR / "repair_manifest.tsv",
    sep="\t",
    index=False,
)


if len(missing_jobs) == 0:
    print("\nNothing needs repair.")
else:
    print("\nRepair manifest:")
    print(REPAIR_DIR / "repair_manifest.tsv")


# ============================================================
# FIND INPUT PATH COLUMNS
# ============================================================

ligand_columns = [
    "ligand_pdbqt_file",
    "prepared_ligand_pdbqt",
    "pdbqt_file",
    "ligand_file",
]

receptor_columns = [
    "decoy_receptor_file",
    "receptor_file",
    "receptor_pdbqt_file",
]

ligand_col = next(
    (
        c for c in ligand_columns
        if c in original.columns
    ),
    None,
)

receptor_col = next(
    (
        c for c in receptor_columns
        if c in original.columns
    ),
    None,
)

if ligand_col is None:
    raise SystemExit(
        "\nNo prepared ligand PDBQT path column "
        "was found."
    )

if receptor_col is None:
    raise SystemExit(
        "\nNo decoy receptor PDBQT path column "
        "was found."
    )

print("\nUsing ligand column:", ligand_col)
print("Using receptor column:", receptor_col)


# ============================================================
# CREATE LOOKUP TABLES
# ============================================================

def first_existing(values):
    for value in values:
        value = str(value).strip()

        if not value:
            continue

        p = Path(value)

        if p.exists():
            return str(p)

    return ""


ligand_lookup = {}

for molecule_id, group in original.groupby(
    "molecule_id"
):
    ligand_lookup[molecule_id] = first_existing(
        group[ligand_col].tolist()
    )


receptor_lookup = {}

for decoy_id, group in original.groupby(
    "decoy_id"
):
    receptor_lookup[decoy_id] = first_existing(
        group[receptor_col].tolist()
    )


# ============================================================
# READ VINA SETTINGS FROM ORIGINAL CONFIG
# ============================================================

with CONFIG.open() as handle:
    cfg = yaml.safe_load(handle)


def flatten(value, prefix=""):
    result = {}

    if isinstance(value, dict):
        for key, child in value.items():

            path = (
                f"{prefix}.{key}"
                if prefix
                else str(key)
            )

            result.update(
                flatten(child, path)
            )

    else:
        result[prefix] = value

    return result


flat = flatten(cfg)


def resolve(names, default=None, required=False):

    for name in names:
        for key, value in flat.items():

            if key.lower().endswith(
                name.lower()
            ):
                return value

    if required:

        print("\nPotential docking settings:")

        for key, value in flat.items():
            if any(
                word in key.lower()
                for word in [
                    "center",
                    "size",
                    "box",
                    "exhaust",
                ]
            ):
                print(key, "=", value)

        raise SystemExit(
            f"\nCould not find config setting: {names}"
        )

    return default


center_x = float(
    resolve(
        [
            "docking_box.center_x",
            "box.center_x",
            "center_x",
        ],
        required=True,
    )
)

center_y = float(
    resolve(
        [
            "docking_box.center_y",
            "box.center_y",
            "center_y",
        ],
        required=True,
    )
)

center_z = float(
    resolve(
        [
            "docking_box.center_z",
            "box.center_z",
            "center_z",
        ],
        required=True,
    )
)

size_x = float(
    resolve(
        [
            "docking_box.size_x",
            "box.size_x",
            "size_x",
        ],
        required=True,
    )
)

size_y = float(
    resolve(
        [
            "docking_box.size_y",
            "box.size_y",
            "size_y",
        ],
        required=True,
    )
)

size_z = float(
    resolve(
        [
            "docking_box.size_z",
            "box.size_z",
            "size_z",
        ],
        required=True,
    )
)

exhaustiveness = int(
    resolve(
        [
            "vina.exhaustiveness",
            "docking.exhaustiveness",
            "exhaustiveness",
        ],
        default=8,
    )
)

num_modes = int(
    resolve(
        [
            "vina.num_modes",
            "docking.num_modes",
            "num_modes",
        ],
        default=9,
    )
)

energy_range = float(
    resolve(
        [
            "vina.energy_range",
            "docking.energy_range",
            "energy_range",
        ],
        default=3,
    )
)

seed = int(
    resolve(
        [
            "vina.seed",
            "docking.seed",
            "random_seed",
            "seed",
        ],
        default=42,
    )
)


print("\nVina settings")
print("-------------")
print(
    "Center:",
    center_x,
    center_y,
    center_z,
)
print(
    "Size:",
    size_x,
    size_y,
    size_z,
)
print("Exhaustiveness:", exhaustiveness)
print("Num modes:", num_modes)
print("Energy range:", energy_range)
print("Seed:", seed)


# ============================================================
# SCORE PARSER
# ============================================================

def parse_affinity(stdout, pose_path):

    for line in stdout.splitlines():

        match = re.match(
            r"^\s*1\s+"
            r"(-?\d+(?:\.\d+)?)\s+",
            line,
        )

        if match:
            return float(
                match.group(1)
            )

    if pose_path.exists():

        text = pose_path.read_text(
            errors="ignore"
        )

        match = re.search(
            r"REMARK\s+VINA\s+RESULT:\s+"
            r"(-?\d+(?:\.\d+)?)",
            text,
        )

        if match:
            return float(
                match.group(1)
            )

    return None


# ============================================================
# LOAD PREVIOUS REPAIR RESULTS FOR RESUME
# ============================================================

repair_rows = []

completed_repair_pairs = set()

if REPAIR_SCORE_FILE.exists():

    previous = pd.read_csv(
        REPAIR_SCORE_FILE,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    previous["_numeric"] = pd.to_numeric(
        previous["decoy_deltaG"],
        errors="coerce",
    )

    previous_valid = (
        previous["dock_status"]
        .str.strip()
        .str.lower()
        .eq("completed")
        &
        previous["_numeric"].notna()
    )

    for _, row in previous[
        previous_valid
    ].iterrows():

        completed_repair_pairs.add(
            (
                row["molecule_id"],
                row["decoy_id"],
            )
        )

    repair_rows = (
        previous.drop(
            columns=["_numeric"],
            errors="ignore",
        )
        .to_dict("records")
    )

    print(
        "\nResuming existing repair."
    )
    print(
        "Already completed repair pairs:",
        len(completed_repair_pairs),
    )


# ============================================================
# REPAIR ONLY MISSING/FAILED PAIRS
# ============================================================

total = len(missing_jobs)

for job_number, row in enumerate(
    missing_jobs.itertuples(index=False),
    start=1,
):

    molecule_id = str(
        row.molecule_id
    ).strip()

    decoy_id = str(
        row.decoy_id
    ).strip()

    pair = (
        molecule_id,
        decoy_id,
    )

    if pair in completed_repair_pairs:
        continue

    ligand = Path(
        ligand_lookup.get(
            molecule_id,
            "",
        )
    )

    receptor = Path(
        receptor_lookup.get(
            decoy_id,
            "",
        )
    )

    safe_mol = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        molecule_id,
    )

    safe_decoy = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        decoy_id,
    )

    pose_dir = (
        POSE_DIR / safe_decoy
    )

    log_dir = (
        LOG_DIR / safe_decoy
    )

    pose_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pose = (
        pose_dir
        / f"{safe_mol}.pdbqt"
    )

    log = (
        log_dir
        / f"{safe_mol}.log"
    )

    print(
        f"[{job_number}/{total}] "
        f"{molecule_id} × {decoy_id}"
    )

    status = "failed"
    score = ""
    error = ""

    if not ligand.exists():

        error = (
            "Prepared ligand PDBQT "
            f"not found: {ligand}"
        )

    elif not receptor.exists():

        error = (
            "Decoy receptor PDBQT "
            f"not found: {receptor}"
        )

    else:

        command = [
            vina,
            "--receptor",
            str(receptor),
            "--ligand",
            str(ligand),
            "--center_x",
            str(center_x),
            "--center_y",
            str(center_y),
            "--center_z",
            str(center_z),
            "--size_x",
            str(size_x),
            "--size_y",
            str(size_y),
            "--size_z",
            str(size_z),
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
                + result.stderr,
                errors="ignore",
            )

            affinity = parse_affinity(
                result.stdout,
                pose,
            )

            if (
                result.returncode == 0
                and affinity is not None
                and pose.exists()
                and pose.stat().st_size > 0
            ):

                status = "completed"
                score = affinity

            else:

                error = (
                    result.stderr.strip()
                    or
                    result.stdout.strip()[-1500:]
                    or
                    f"Vina return code "
                    f"{result.returncode}"
                )

        except Exception as exc:

            error = str(exc)


    new_record = {
        "molecule_id": molecule_id,
        "decoy_id": decoy_id,
        "decoy_deltaG": score,
        "dock_status": status,
        "is_dry_run_score": False,
        ligand_col: str(ligand),
        receptor_col: str(receptor),
        "pose_file": str(pose),
        "log_file": str(log),
        "error_message": error,
        "_result_source":
            "direct_failed_job_repair",
    }

    # Replace previous record for same pair if present.
    repair_rows = [
        r for r in repair_rows
        if not (
            str(
                r.get(
                    "molecule_id",
                    ""
                )
            ).strip()
            == molecule_id
            and
            str(
                r.get(
                    "decoy_id",
                    ""
                )
            ).strip()
            == decoy_id
        )
    ]

    repair_rows.append(
        new_record
    )

    pd.DataFrame(
        repair_rows
    ).to_csv(
        REPAIR_SCORE_FILE,
        sep="\t",
        index=False,
    )


# ============================================================
# VALIDATE REPAIR
# ============================================================

if REPAIR_SCORE_FILE.exists():

    repair = pd.read_csv(
        REPAIR_SCORE_FILE,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

else:

    repair = pd.DataFrame()


if len(repair):

    repair["_score_numeric"] = pd.to_numeric(
        repair["decoy_deltaG"],
        errors="coerce",
    )

    valid_repair = (
        repair["dock_status"]
        .str.strip()
        .str.lower()
        .eq("completed")
        &
        repair["_score_numeric"].notna()
    )

else:

    repair["_score_numeric"] = []
    valid_repair = pd.Series(
        dtype=bool
    )


repair_completed = (
    repair[valid_repair].copy()
    if len(repair)
    else repair.copy()
)


print("\nREPAIR RESULTS")
print("--------------")
print(
    "Jobs needing repair:",
    len(missing_jobs),
)
print(
    "Successfully repaired:",
    len(repair_completed),
)
print(
    "Still failed:",
    len(missing_jobs)
    - len(repair_completed),
)


if len(repair):

    still_failed = repair[
        ~valid_repair
    ].copy()

    still_failed.to_csv(
        FAILED_FILE,
        sep="\t",
        index=False,
    )

else:

    still_failed = repair


# ============================================================
# MERGE ORIGINAL COMPLETED + REPAIR COMPLETED
# ============================================================

original_good = (
    original[
        valid_original
    ].copy()
)

original_good["decoy_deltaG"] = (
    original_good[
        "_score_numeric"
    ].astype(float)
)

original_good = original_good.drop(
    columns=[
        "_score_numeric"
    ],
    errors="ignore",
)

original_good["_source_priority"] = 0

if "_result_source" not in original_good.columns:
    original_good[
        "_result_source"
    ] = "original_decoy_run"


if len(repair_completed):

    repair_completed[
        "decoy_deltaG"
    ] = repair_completed[
        "_score_numeric"
    ].astype(float)

    repair_completed = (
        repair_completed.drop(
            columns=[
                "_score_numeric"
            ],
            errors="ignore",
        )
    )

    repair_completed[
        "_source_priority"
    ] = 1

    combined = pd.concat(
        [
            original_good,
            repair_completed,
        ],
        ignore_index=True,
        sort=False,
    )

else:

    combined = original_good.copy()


combined = combined.sort_values(
    [
        "molecule_id",
        "decoy_id",
        "_source_priority",
    ]
)

combined = combined.drop_duplicates(
    [
        "molecule_id",
        "decoy_id",
    ],
    keep="first",
)

combined = combined.drop(
    columns=[
        "_source_priority"
    ],
    errors="ignore",
)


# ============================================================
# FINAL COMPLETENESS CHECK
# ============================================================

complete_pairs = combined[
    [
        "molecule_id",
        "decoy_id",
    ]
].drop_duplicates()

final_missing = expected_grid.merge(
    complete_pairs.assign(
        _present=True
    ),
    on=[
        "molecule_id",
        "decoy_id",
    ],
    how="left",
)

final_missing = final_missing[
    final_missing["_present"].isna()
][
    [
        "molecule_id",
        "decoy_id",
    ]
]

final_missing.to_csv(
    COMPLETE_DIR
    / "missing_jobs_after_repair.tsv",
    sep="\t",
    index=False,
)

counts = combined.groupby(
    "molecule_id"
)["decoy_id"].nunique()

expected_rows = (
    150 * EXPECTED_DECOYS
)

print("\nFINAL RUN VALIDATION")
print("--------------------")

print(
    "Completed rows:",
    len(combined),
)

print(
    "Unique molecules:",
    combined[
        "molecule_id"
    ].nunique(),
)

print(
    "Unique decoys:",
    combined[
        "decoy_id"
    ].nunique(),
)

print(
    f"Molecules with all "
    f"{EXPECTED_DECOYS}:",
    int(
        (
            counts
            == EXPECTED_DECOYS
        ).sum()
    ),
)

print(
    "Missing pairs:",
    len(final_missing),
)


if (
    len(combined)
    != expected_rows
    or
    combined[
        "molecule_id"
    ].nunique()
    != 150
    or
    combined[
        "decoy_id"
    ].nunique()
    != EXPECTED_DECOYS
    or
    len(final_missing)
    != 0
):

    print(
        "\nREPAIR IS NOT COMPLETE."
    )

    print(
        "\nInspect:"
    )

    print(FAILED_FILE)

    print(
        COMPLETE_DIR
        / "missing_jobs_after_repair.tsv"
    )

    raise SystemExit(1)


combined = combined.sort_values(
    [
        "molecule_id",
        "decoy_id",
    ]
).reset_index(drop=True)

combined.to_csv(
    COMPLETE_FILE,
    sep="\t",
    index=False,
)


provenance = {
    "created_at":
        datetime.now().isoformat(),
    "original_score_file":
        str(SCORES),
    "repair_score_file":
        str(REPAIR_SCORE_FILE),
    "expected_decoys":
        EXPECTED_DECOYS,
    "expected_rows":
        expected_rows,
    "completed_rows":
        len(combined),
    "repaired_jobs":
        len(repair_completed),
    "complete":
        True,
}

with (
    COMPLETE_DIR
    / "repair_provenance.json"
).open("w") as handle:

    json.dump(
        provenance,
        handle,
        indent=2,
    )


print(
    "\n================================="
)
print(
    "DECOY RUN REPAIR: PASSED"
)
print(
    "================================="
)

print(
    "\nComplete repaired file:"
)
print(
    COMPLETE_FILE
)
