from __future__ import annotations

from pathlib import Path
import inspect
import itertools
import json
import math
import re
import shutil

import numpy as np
import pandas as pd
import yaml

from smn2_rescuediff.splicing_rescue.srsi_ranker import (
    calculate_srsi,
)


# ============================================================
# SETTINGS
# ============================================================

FINAL_TABLE = Path(
    "data/processed/final_scoring/"
    "judge_grade_200x100/tables/"
    "final_srsi_ranked_candidates.tsv"
)

OUT = Path(
    "data/processed/interpretability/"
    "module5_shapley"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

RANDOM_SEED = 42


# ============================================================
# LOAD EXISTING PROJECT RESULT
# ============================================================

if not FINAL_TABLE.exists():
    raise SystemExit(
        f"Missing final SRSI table:\n{FINAL_TABLE}"
    )

df = pd.read_csv(
    FINAL_TABLE,
    sep="\t",
    low_memory=False,
)

if "molecule_id" not in df.columns:
    raise SystemExit(
        "Final table has no molecule_id column."
    )

if "srsi_score" not in df.columns:
    raise SystemExit(
        "Final table has no srsi_score column."
    )

if df["molecule_id"].duplicated().any():
    raise SystemExit(
        "Duplicate molecule_id values detected."
    )

ids = df["molecule_id"].astype(str).tolist()


# ============================================================
# DRY-RUN SAFETY CHECK
# ============================================================

def truthy(value):
    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
        "y",
    }


dry_columns = [
    c for c in df.columns
    if "dry" in c.lower()
]

for col in dry_columns:

    if df[col].apply(truthy).any():

        raise SystemExit(
            f"STOP: {col} contains dry-run values. "
            "Do not use simulated scores for the paper."
        )


# Also inspect textual warnings.
for col in df.select_dtypes(
    include=["object"]
).columns:

    text = (
        df[col]
        .astype(str)
        .str.upper()
    )

    if text.str.contains(
        "DRY-RUN / SIMULATED",
        regex=False,
    ).any():

        raise SystemExit(
            f"STOP: simulated-result warning found "
            f"in column {col}."
        )


# ============================================================
# FIND THE CONFIG USED FOR THIS RESULT
# ============================================================

search_root = FINAL_TABLE.parent.parent

config_candidates = []

direct = (
    search_root /
    "splicing_rescue_scoring_config_used.yaml"
)

if direct.exists():
    config_candidates.append(direct)

config_candidates.extend(
    sorted(
        search_root.rglob(
            "splicing_rescue_scoring_config_used.yaml"
        )
    )
)

# Remove duplicates.
config_candidates = list(
    dict.fromkeys(config_candidates)
)

if not config_candidates:

    broader = Path(
        "data/processed/final_scoring"
    )

    config_candidates = sorted(
        broader.rglob(
            "splicing_rescue_scoring_config_used.yaml"
        )
    )

if not config_candidates:

    raise SystemExit(
        "Could not locate "
        "splicing_rescue_scoring_config_used.yaml"
    )


# Prefer the config nearest to our table.
config_path = min(
    config_candidates,
    key=lambda p:
        len(p.parts)
        + (
            100
            if "judge_grade_200x100"
            not in str(p)
            else 0
        )
)

print("Using final table:")
print(FINAL_TABLE)

print("\nUsing scoring config:")
print(config_path)


with config_path.open() as handle:
    config = yaml.safe_load(handle)


# Freeze copies used for interpretability.
shutil.copy2(
    config_path,
    OUT / "scoring_config_used.yaml",
)

source = inspect.getsource(
    calculate_srsi
)

(
    OUT /
    "calculate_srsi_source_snapshot.py"
).write_text(
    source,
    encoding="utf-8",
)


# ============================================================
# RUN EXISTING SRSI FUNCTION
# ============================================================

def run_existing_srsi(frame: pd.DataFrame):

    # All rescue evidence is already present in
    # final_srsi_ranked_candidates.tsv.
    # We therefore provide only molecule_id as the merge key.
    rescue_stub = frame[
        ["molecule_id"]
    ].copy()

    result = calculate_srsi(
        frame.copy(),
        rescue_stub,
        config,
        used_dry_run_inputs=False,
    )

    result["molecule_id"] = (
        result["molecule_id"]
        .astype(str)
    )

    score = (
        result
        .set_index("molecule_id")
        ["srsi_score"]
        .reindex(ids)
    )

    return pd.to_numeric(
        score,
        errors="coerce",
    ).to_numpy(dtype=float)


# ============================================================
# REPRODUCIBILITY CHECK
# ============================================================

stored = pd.to_numeric(
    df["srsi_score"],
    errors="coerce",
).to_numpy(dtype=float)

recomputed = run_existing_srsi(df)

valid = (
    np.isfinite(stored)
    & np.isfinite(recomputed)
)

if not valid.any():
    raise SystemExit(
        "No valid stored/recomputed SRSI scores."
    )

max_abs_difference = float(
    np.max(
        np.abs(
            stored[valid]
            - recomputed[valid]
        )
    )
)

mean_abs_difference = float(
    np.mean(
        np.abs(
            stored[valid]
            - recomputed[valid]
        )
    )
)

print("\nSRSI reproduction check")
print("-----------------------")
print(
    "Max absolute difference:",
    max_abs_difference,
)
print(
    "Mean absolute difference:",
    mean_abs_difference,
)


# Allow only tiny rounding differences.
if max_abs_difference > 1e-4:

    raise SystemExit(
        "\nSTOP: calculate_srsi() did not reproduce "
        "the stored final score closely enough.\n"
        "Do not run Shapley until the matching "
        "scoring config/result pair is identified."
    )


# ============================================================
# DETECT INPUT FEATURES ACTUALLY READ BY calculate_srsi()
# ============================================================

patterns = [
    r'numeric_series\(\s*work\s*,\s*["\']([^"\']+)["\']',
    r'work\.get\(\s*["\']([^"\']+)["\']',
    r'work\[\s*["\']([^"\']+)["\']\s*\]',
]

detected = set()

for pattern in patterns:
    detected.update(
        re.findall(
            pattern,
            source,
        )
    )


# Include known real SRSI evidence columns if
# present AND referenced by the scoring source.
known_inputs = [
    "target_deltaG",
    "target_binding_strength_score",
    "predicted_splicing_rescue_score",
    "weighted_selectivity_gap",
    "qed",
    "QED",
    "lipinski_pass",
    "lipinski_score",
    "chemical_quality_score",
    "pre_docking_candidate_score",
    "off_target_risk_score",
    "sa_score",
    "pains_alert_count",
    "pains_penalty",
    "splice_site_occlusion_penalty",
    "pose_quality_score",
]

for col in known_inputs:
    if (
        col in df.columns
        and col in source
    ):
        detected.add(col)


# Outputs / metadata must never be players.
exclude = {
    "molecule_id",
    "canonical_smiles",
    "srsi_score",
    "srsi_percentile",
    "final_rank",
    "candidate_tier",
    "confidence_level",
    "recommendation",
    "notes",

    # Descriptive/provenance fields, not SRSI evidence inputs.
    "evidence_summary",
    "recommended_next_step",
    "is_dry_run_input",
}

features = sorted(
    c for c in detected
    if c in df.columns
    and c not in exclude
)


if not features:
    raise SystemExit(
        "No SRSI input features were detected."
    )


# ============================================================
# NEGATIVE CONTROL
# ============================================================

# This column is deliberately NOT used by calculate_srsi().
# Therefore its Shapley value should be exactly/nearly zero.

rng = np.random.default_rng(
    RANDOM_SEED
)

df["random_control_feature"] = (
    rng.normal(
        size=len(df)
    )
)

features.append(
    "random_control_feature"
)


print("\nDetected SRSI evidence inputs:")
for feature in features:
    print(" ", feature)

print(
    "\nNumber of Shapley players:",
    len(features),
)


# Keep exact calculation computationally reasonable.
if len(features) > 12:
    raise SystemExit(
        "More than 12 scoring inputs were detected. "
        "Stop and inspect detected features before "
        "running exact coalition Shapley."
    )


# ============================================================
# BASELINE MASKING
# ============================================================

def baseline_value(series: pd.Series):

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    # Treat as numeric if most values are numeric.
    if numeric.notna().mean() >= 0.80:

        return float(
            numeric.mean()
        )

    mode = (
        series
        .dropna()
        .mode()
    )

    if len(mode):
        return mode.iloc[0]

    return ""


baselines = {
    feature:
        baseline_value(df[feature])
    for feature in features
}


baseline_table = pd.DataFrame(
    {
        "feature": list(baselines),
        "baseline_value": [
            baselines[f]
            for f in baselines
        ],
    }
)

baseline_table.to_csv(
    OUT /
    "shapley_baselines.tsv",
    sep="\t",
    index=False,
)


# ============================================================
# COALITION VALUE FUNCTION
# ============================================================

M = len(features)

full_original = df.copy()

baseline_frame = df.copy()

for feature in features:

    baseline_frame[feature] = (
        baselines[feature]
    )


cache = {}


def coalition_value(mask: int):

    if mask in cache:
        return cache[mask]

    frame = baseline_frame.copy()

    for j, feature in enumerate(features):

        if mask & (1 << j):

            frame[feature] = (
                full_original[feature]
            )

    value = run_existing_srsi(
        frame
    )

    cache[mask] = value

    return value


# Calculate all 2^M coalition outputs.
for mask in range(
    1 << M
):
    coalition_value(mask)


print(
    "\nCoalitions evaluated:",
    len(cache),
)


# ============================================================
# EXACT SHAPLEY VALUES
# ============================================================

factorial_M = math.factorial(M)

phi = np.zeros(
    (
        len(df),
        M,
    ),
    dtype=float,
)


for j in range(M):

    bit = 1 << j

    for mask in range(
        1 << M
    ):

        if mask & bit:
            continue

        s = mask.bit_count()

        weight = (
            math.factorial(s)
            * math.factorial(
                M - s - 1
            )
            / factorial_M
        )

        marginal = (
            coalition_value(
                mask | bit
            )
            - coalition_value(mask)
        )

        phi[:, j] += (
            weight
            * marginal
        )


# ============================================================
# SHAPLEY EFFICIENCY CHECK
# ============================================================

empty_value = coalition_value(0)

full_mask = (
    (1 << M)
    - 1
)

full_value = coalition_value(
    full_mask
)

reconstructed = (
    empty_value
    + phi.sum(axis=1)
)

efficiency_error = np.abs(
    reconstructed
    - full_value
)

max_efficiency_error = float(
    np.nanmax(
        efficiency_error
    )
)

print(
    "\nMax Shapley efficiency error:",
    max_efficiency_error,
)


if max_efficiency_error > 1e-5:

    raise SystemExit(
        "Shapley efficiency check failed."
    )


# ============================================================
# LONG RESULTS
# ============================================================

rows = []

for i, molecule_id in enumerate(ids):

    for j, feature in enumerate(features):

        value = float(
            phi[i, j]
        )

        rows.append(
            {
                "molecule_id":
                    molecule_id,
                "feature":
                    feature,
                "shapley_value":
                    value,
                "absolute_shapley_value":
                    abs(value),
            }
        )


long = pd.DataFrame(rows)

long.to_csv(
    OUT /
    "module5_shapley_values_long.tsv",
    sep="\t",
    index=False,
)


# ============================================================
# FEATURE-LEVEL SUMMARY
# ============================================================

summary = (
    long.groupby("feature")
    .agg(
        mean_abs_shapley=(
            "absolute_shapley_value",
            "mean",
        ),
        sd_abs_shapley=(
            "absolute_shapley_value",
            "std",
        ),
        mean_signed_shapley=(
            "shapley_value",
            "mean",
        ),
        median_abs_shapley=(
            "absolute_shapley_value",
            "median",
        ),
        positive_fraction=(
            "shapley_value",
            lambda x:
                float(
                    (x > 0).mean()
                ),
        ),
        n_molecules=(
            "molecule_id",
            "nunique",
        ),
    )
    .reset_index()
)


total_importance = (
    summary[
        "mean_abs_shapley"
    ].sum()
)

summary[
    "relative_importance_percent"
] = np.where(
    total_importance > 0,
    (
        summary[
            "mean_abs_shapley"
        ]
        / total_importance
        * 100.0
    ),
    0.0,
)


summary = summary.sort_values(
    "mean_abs_shapley",
    ascending=False,
).reset_index(drop=True)

summary["rank"] = range(
    1,
    len(summary) + 1,
)


# ============================================================
# MODULE ORIGIN LABELS
# ============================================================

def origin(feature):

    f = feature.lower()

    if feature == "random_control_feature":
        return "Negative control"

    if (
        "target" in f
        and "off_target" not in f
    ):
        return "Module 1 → Module 4"

    if (
        "selectivity" in f
        or "off_target" in f
    ):
        return "Module 2/4"

    if (
        "qed" in f
        or "lipinski" in f
        or "pre_docking" in f
        or "sa_score" in f
        or "pains" in f
        or "chemical" in f
    ):
        return "Module 3"

    if (
        "rescue" in f
        or "pose" in f
        or "occlusion" in f
    ):
        return "Module 5"

    return "Integrated evidence"


summary[
    "pipeline_origin"
] = summary[
    "feature"
].apply(origin)


def direction(value):

    if abs(value) < 1e-10:
        return "Neutral"

    return (
        "Raises SRSI on average"
        if value > 0
        else "Lowers SRSI on average"
    )


summary[
    "mean_direction"
] = summary[
    "mean_signed_shapley"
].apply(direction)


summary.to_csv(
    OUT /
    "module5_shapley_summary_full.tsv",
    sep="\t",
    index=False,
)


# ============================================================
# PAPER-SIZED TABLE
# ============================================================

real = summary[
    summary["feature"]
    != "random_control_feature"
].head(7)

control = summary[
    summary["feature"]
    == "random_control_feature"
].head(1)

paper = pd.concat(
    [
        real,
        control,
    ],
    ignore_index=True,
)


paper = paper[
    [
        "rank",
        "feature",
        "pipeline_origin",
        "mean_abs_shapley",
        "relative_importance_percent",
        "mean_signed_shapley",
        "mean_direction",
    ]
].copy()


paper = paper.rename(
    columns={
        "rank":
            "Rank",
        "feature":
            "Evidence input",
        "pipeline_origin":
            "Pipeline origin",
        "mean_abs_shapley":
            "Mean |Shapley|",
        "relative_importance_percent":
            "Relative contribution (%)",
        "mean_signed_shapley":
            "Mean signed Shapley",
        "mean_direction":
            "Average effect",
    }
)


for col in [
    "Mean |Shapley|",
    "Relative contribution (%)",
    "Mean signed Shapley",
]:

    paper[col] = (
        pd.to_numeric(
            paper[col],
            errors="coerce",
        )
        .round(4)
    )


paper.to_csv(
    OUT /
    "module5_shapley_paper_table.tsv",
    sep="\t",
    index=False,
)

paper.to_csv(
    OUT /
    "module5_shapley_paper_table.csv",
    index=False,
)


# ============================================================
# VALIDATION RECORD
# ============================================================

validation = {
    "analysis_type":
        "Exact coalition Shapley attribution "
        "of existing Module 5 calculate_srsi()",

    "surrogate_model_trained":
        False,

    "source_table":
        str(FINAL_TABLE),

    "scoring_config":
        str(config_path),

    "molecules":
        len(df),

    "features":
        features,

    "num_features":
        M,

    "coalitions_evaluated":
        int(1 << M),

    "max_srsi_reproduction_difference":
        max_abs_difference,

    "mean_srsi_reproduction_difference":
        mean_abs_difference,

    "max_shapley_efficiency_error":
        max_efficiency_error,

    "negative_control":
        "random_control_feature",

    "interpretation":
        (
            "Shapley values quantify contributions "
            "to the existing computational SRSI "
            "ranking only; they do not demonstrate "
            "biological causality or experimental "
            "splicing rescue."
        ),
}


(
    OUT /
    "shapley_validation.json"
).write_text(
    json.dumps(
        validation,
        indent=2,
    ),
    encoding="utf-8",
)


print("\n========================================")
print("SHAPLEY ANALYSIS COMPLETE")
print("========================================")

print("\nPaper-sized table:\n")

print(
    paper.to_string(
        index=False
    )
)

print("\nFiles written:")

for name in [
    "module5_shapley_values_long.tsv",
    "module5_shapley_summary_full.tsv",
    "module5_shapley_paper_table.tsv",
    "module5_shapley_paper_table.csv",
    "shapley_baselines.tsv",
    "shapley_validation.json",
    "calculate_srsi_source_snapshot.py",
    "scoring_config_used.yaml",
]:
    print(
        OUT / name
    )
