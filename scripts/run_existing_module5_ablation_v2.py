from pathlib import Path
import inspect
import numpy as np
import pandas as pd
import yaml

from smn2_rescuediff.splicing_rescue import ablation_analysis as aa


FINAL = Path(
    "data/processed/final_scoring/"
    "judge_grade_200x100/tables/"
    "final_srsi_ranked_candidates.tsv"
)

CONFIG = Path(
    "data/processed/final_scoring/"
    "judge_grade_200x100/"
    "splicing_rescue_scoring_config_used.yaml"
)

OUT = Path(
    "data/processed/interpretability/"
    "module5_ablation"
)

OUT.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# LOAD FINAL EXISTING RESULTS
# ------------------------------------------------------------

if not FINAL.exists():
    raise SystemExit(f"Missing final table:\n{FINAL}")

df = pd.read_csv(
    FINAL,
    sep="\t",
    low_memory=False,
)

print("Final result table:")
print(FINAL)
print("\nRows:", len(df))
print("Unique molecules:", df["molecule_id"].nunique())


# ------------------------------------------------------------
# FIND THE REAL EXISTING ABLATION FUNCTION
# ------------------------------------------------------------

matches = []

for name, obj in vars(aa).items():

    if not inspect.isfunction(obj):
        continue

    if obj.__module__ != aa.__name__:
        continue

    doc = inspect.getdoc(obj) or ""

    if (
        "Compare simple binding-only rankings"
        in doc
        or
        "full SRSI rankings"
        in doc
    ):
        matches.append(
            (name, obj, inspect.signature(obj))
        )


print("\nMatched ablation functions:")

for name, _, sig in matches:
    print(" ", name, sig)


if len(matches) != 1:
    raise SystemExit(
        "\nExpected exactly one existing ablation function."
    )


function_name, function, signature = matches[0]

print("\nUsing:")
print(function_name, signature)


# ------------------------------------------------------------
# LOAD CONFIG ONLY IF REQUIRED
# ------------------------------------------------------------

config = None

if CONFIG.exists():
    with CONFIG.open() as f:
        config = yaml.safe_load(f)


# ------------------------------------------------------------
# BUILD ARGUMENTS ACCORDING TO REAL SIGNATURE
# ------------------------------------------------------------

args = []
kwargs = {}

parameters = list(signature.parameters.values())

for i, param in enumerate(parameters):

    if i == 0:
        args.append(df.copy())
        continue

    name = param.name.lower()

    if "config" in name:

        if config is None:
            raise SystemExit(
                f"{function_name} requires config, "
                f"but config was not found:\n{CONFIG}"
            )

        args.append(config)

    elif "dry" in name:

        args.append(False)

    elif param.default is not inspect.Parameter.empty:

        # Let default apply.
        continue

    else:

        raise SystemExit(
            f"Unknown required parameter "
            f"{param.name!r} in {function_name}{signature}"
        )


# ------------------------------------------------------------
# RUN EXISTING PROJECT ABLATION
# ------------------------------------------------------------

ablation = function(
    *args,
    **kwargs,
)

if not isinstance(ablation, pd.DataFrame):
    raise SystemExit(
        "Existing ablation function did not "
        "return a DataFrame."
    )

if ablation.empty:
    raise SystemExit(
        "Existing ablation function returned "
        "an empty DataFrame."
    )


raw = OUT / "module5_ablation_results_raw.tsv"

ablation.to_csv(
    raw,
    sep="\t",
    index=False,
)


print("\nAblation rows:", len(ablation))

print("\nAblation columns:")
for c in ablation.columns:
    print(" ", c)


# ------------------------------------------------------------
# IDENTIFY RANK VARIANTS
# ------------------------------------------------------------

rank_columns = [
    c for c in ablation.columns
    if c.endswith("_rank")
]

if "full_srsi_rank" not in rank_columns:
    raise SystemExit(
        "full_srsi_rank is missing."
    )


for col in rank_columns:

    ablation[col] = pd.to_numeric(
        ablation[col],
        errors="coerce",
    )


# ------------------------------------------------------------
# NAMES FOR PAPER
# ------------------------------------------------------------

LABELS = {
    "binding_only_rank":
        "Target binding only",

    "binding_plus_selectivity_rank":
        "Target + RNA selectivity",

    "binding_plus_splicing_rescue_rank":
        "Target + splicing-rescue proxy",

    "full_srsi_rank":
        "Full SRSI",
}


def pretty(col):

    return LABELS.get(
        col,
        col
        .replace("_rank", "")
        .replace("_", " ")
        .title()
    )


# ------------------------------------------------------------
# GLOBAL ABLATION SUMMARY
# ------------------------------------------------------------

full_col = "full_srsi_rank"

TOP_N = min(
    10,
    len(ablation),
)

full_top = set(
    ablation
    .nsmallest(
        TOP_N,
        full_col,
    )["molecule_id"]
    .astype(str)
)


rows = []

for col in rank_columns:

    # Avoid selecting full_srsi_rank twice when
    # the current ablation variant is Full SRSI.
    selected_cols = ["molecule_id", col]

    if col != full_col:
        selected_cols.append(full_col)

    sub = (
        ablation[selected_cols]
        .dropna()
        .copy()
    )

    if sub.empty:
        continue

    if col == full_col:
        shift = pd.Series(
            0.0,
            index=sub.index,
        )
        rho = 1.0

    else:
        shift = (
            sub[col]
            - sub[full_col]
        )

        rho = sub[col].corr(
            sub[full_col],
            method="spearman",
        )

    top = set(
        sub
        .nsmallest(
            TOP_N,
            col,
        )["molecule_id"]
        .astype(str)
    )

    top_overlap = (
        len(top & full_top)
        / TOP_N
        * 100.0
    )

    rows.append(
        {
            "Ablation variant":
                pretty(col),

            "Spearman rho vs full SRSI":
                rho,

            f"Top-{TOP_N} overlap (%)":
                top_overlap,

            "Mean absolute rank shift":
                shift.abs().mean(),

            "Median absolute rank shift":
                shift.abs().median(),

            "Maximum absolute rank shift":
                shift.abs().max(),

            "Molecules evaluated":
                len(sub),
        }
    )


summary = pd.DataFrame(rows)


ORDER = {
    "Target binding only": 1,
    "Target + RNA selectivity": 2,
    "Target + splicing-rescue proxy": 3,
    "Full SRSI": 99,
}


summary["_order"] = (
    summary["Ablation variant"]
    .map(ORDER)
    .fillna(50)
)

summary = (
    summary
    .sort_values("_order")
    .drop(columns="_order")
    .reset_index(drop=True)
)


numeric_cols = [
    "Spearman rho vs full SRSI",
    f"Top-{TOP_N} overlap (%)",
    "Mean absolute rank shift",
    "Median absolute rank shift",
    "Maximum absolute rank shift",
]

for col in numeric_cols:

    summary[col] = (
        pd.to_numeric(
            summary[col],
            errors="coerce",
        )
        .round(3)
    )


paper_tsv = (
    OUT /
    "module5_ablation_paper_table.tsv"
)

paper_csv = (
    OUT /
    "module5_ablation_paper_table.csv"
)

summary.to_csv(
    paper_tsv,
    sep="\t",
    index=False,
)

summary.to_csv(
    paper_csv,
    index=False,
)


# ------------------------------------------------------------
# TOP MOLECULE RANK CHANGES
# ------------------------------------------------------------

if (
    "binding_only_rank"
    in ablation.columns
):

    changes = ablation.copy()

    changes[
        "binding_to_full_rank_change"
    ] = (
        changes["binding_only_rank"]
        - changes["full_srsi_rank"]
    )

    changes[
        "absolute_rank_change"
    ] = (
        changes[
            "binding_to_full_rank_change"
        ]
        .abs()
    )

    keep = [
        c for c in [
            "molecule_id",
            "binding_only_rank",
            "full_srsi_rank",
            "binding_to_full_rank_change",
            "absolute_rank_change",
            "target_deltaG",
            "weighted_selectivity_gap",
            "predicted_splicing_rescue_score",
            "off_target_risk_score",
            "qed",
            "srsi_score",
        ]
        if c in changes.columns
    ]

    changes[
        keep
    ].sort_values(
        "absolute_rank_change",
        ascending=False,
    ).head(20).to_csv(
        OUT /
        "largest_ablation_rank_changes.tsv",
        sep="\t",
        index=False,
    )


print("\n======================================")
print("ABLATION ANALYSIS COMPLETE")
print("======================================")

print("\nPaper-ready table:\n")

print(
    summary.to_string(
        index=False
    )
)

print("\nSaved:")
print(raw)
print(paper_tsv)
print(paper_csv)

if (
    OUT /
    "largest_ablation_rank_changes.tsv"
).exists():

    print(
        OUT /
        "largest_ablation_rank_changes.tsv"
    )
