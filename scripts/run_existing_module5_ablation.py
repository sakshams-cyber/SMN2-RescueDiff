from pathlib import Path
import inspect
import pandas as pd
import numpy as np

from smn2_rescuediff.splicing_rescue import ablation_analysis as aa


FINAL = Path(
    "data/processed/final_scoring/"
    "judge_grade_200x100/tables/"
    "final_srsi_ranked_candidates.tsv"
)

OUT = Path(
    "data/processed/interpretability/"
    "module5_ablation"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


# ------------------------------------------------------------
# LOAD EXISTING FINAL RESULTS
# ------------------------------------------------------------

if not FINAL.exists():
    raise SystemExit(
        f"Missing final SRSI file:\n{FINAL}"
    )

df = pd.read_csv(
    FINAL,
    sep="\t",
    low_memory=False,
)

if "molecule_id" not in df.columns:
    raise SystemExit(
        "No molecule_id column."
    )

if "srsi_score" not in df.columns:
    raise SystemExit(
        "No srsi_score column."
    )

print("Final result table:")
print(FINAL)

print("\nRows:", len(df))
print(
    "Unique molecules:",
    df["molecule_id"].nunique(),
)


# ------------------------------------------------------------
# FIND EXISTING PROJECT ABLATION FUNCTION
# ------------------------------------------------------------

functions = []

for name, obj in vars(aa).items():

    if not inspect.isfunction(obj):
        continue

    if obj.__module__ != aa.__name__:
        continue

    if name.startswith("_"):
        continue

    sig = inspect.signature(obj)

    required = [
        p
        for p in sig.parameters.values()
        if p.default is inspect.Parameter.empty
        and p.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]

    # Existing ablation function should accept
    # the ranked dataframe as its only required argument.
    if len(required) <= 1:
        functions.append(
            (name, obj, sig)
        )


print("\nCandidate ablation functions:")

for name, _, sig in functions:
    print(" ", name, sig)


if len(functions) != 1:

    raise SystemExit(
        "\nCould not uniquely identify the existing "
        "ablation function.\n"
        "Send the function list above if this happens."
    )


function_name, ablation_function, signature = functions[0]

print(
    "\nUsing existing project function:",
    function_name,
    signature,
)


# ------------------------------------------------------------
# RUN EXISTING ABLATION ANALYSIS
# ------------------------------------------------------------

ablation = ablation_function(
    df.copy()
)

if not isinstance(
    ablation,
    pd.DataFrame,
):

    raise SystemExit(
        "Existing ablation function did not "
        "return a DataFrame."
    )

if ablation.empty:

    raise SystemExit(
        "Ablation result is empty."
    )


raw_path = (
    OUT /
    "module5_ablation_results_raw.tsv"
)

ablation.to_csv(
    raw_path,
    sep="\t",
    index=False,
)


print("\nAblation columns:")

for col in ablation.columns:
    print(" ", col)


# ------------------------------------------------------------
# IDENTIFY RANK COLUMNS
# ------------------------------------------------------------

rank_columns = [
    c
    for c in ablation.columns
    if c.endswith("_rank")
]


if "full_srsi_rank" not in rank_columns:

    raise SystemExit(
        "Expected full_srsi_rank "
        "was not produced."
    )


full_col = "full_srsi_rank"


# ------------------------------------------------------------
# CLEAN RANK VALUES
# ------------------------------------------------------------

for col in rank_columns:

    ablation[col] = pd.to_numeric(
        ablation[col],
        errors="coerce",
    )


valid_full = (
    ablation[full_col]
    .notna()
)

ablation = (
    ablation[
        valid_full
    ]
    .copy()
)


# ------------------------------------------------------------
# HUMAN-READABLE NAMES
# ------------------------------------------------------------

NAME_MAP = {
    "binding_only_rank":
        "Target binding only",

    "binding_plus_selectivity_rank":
        "Target + RNA selectivity",

    "binding_plus_splicing_rescue_rank":
        "Target + splicing-rescue proxy",

    "full_srsi_rank":
        "Full SRSI",
}


def readable_name(column):

    if column in NAME_MAP:
        return NAME_MAP[column]

    name = column

    if name.endswith("_rank"):
        name = name[:-5]

    name = name.replace(
        "_",
        " ",
    )

    return name.title()


# ------------------------------------------------------------
# PAPER-GRADE SUMMARY METRICS
# ------------------------------------------------------------

rows = []

N = len(ablation)

TOP_N = min(
    10,
    N,
)

full_top = set(
    ablation
    .nsmallest(
        TOP_N,
        full_col,
    )["molecule_id"]
    .astype(str)
)


for rank_col in rank_columns:

    subset = ablation[
        [
            "molecule_id",
            rank_col,
            full_col,
        ]
    ].dropna().copy()

    if subset.empty:
        continue

    rank = subset[rank_col]
    full = subset[full_col]

    abs_shift = (
        rank - full
    ).abs()

    if rank_col == full_col:

        rho = 1.0

    else:

        rho = rank.corr(
            full,
            method="spearman",
        )

    variant_top = set(
        subset
        .nsmallest(
            TOP_N,
            rank_col,
        )["molecule_id"]
        .astype(str)
    )

    overlap = (
        100.0
        * len(
            variant_top
            & full_top
        )
        / TOP_N
    )

    rows.append(
        {
            "Ablation variant":
                readable_name(
                    rank_col
                ),

            "Rank column":
                rank_col,

            "Spearman rho vs full SRSI":
                float(rho)
                if pd.notna(rho)
                else np.nan,

            f"Top-{TOP_N} overlap with full (%)":
                overlap,

            "Mean absolute rank shift":
                float(
                    abs_shift.mean()
                ),

            "Median absolute rank shift":
                float(
                    abs_shift.median()
                ),

            "Maximum absolute rank shift":
                float(
                    abs_shift.max()
                ),

            "Molecules evaluated":
                len(subset),
        }
    )


summary = pd.DataFrame(rows)


# ------------------------------------------------------------
# ORDER VARIANTS FROM SIMPLEST TO FULL
# ------------------------------------------------------------

ORDER = {
    "Target binding only": 1,
    "Target + RNA selectivity": 2,
    "Target + splicing-rescue proxy": 3,
    "Full SRSI": 99,
}

summary["_order"] = (
    summary[
        "Ablation variant"
    ]
    .map(ORDER)
    .fillna(50)
)

summary = (
    summary
    .sort_values("_order")
    .drop(columns="_order")
    .reset_index(drop=True)
)


# ------------------------------------------------------------
# ROUND FOR PAPER
# ------------------------------------------------------------

paper = summary.copy()

for col in [
    "Spearman rho vs full SRSI",
    f"Top-{TOP_N} overlap with full (%)",
    "Mean absolute rank shift",
    "Median absolute rank shift",
    "Maximum absolute rank shift",
]:

    paper[col] = (
        pd.to_numeric(
            paper[col],
            errors="coerce",
        )
        .round(3)
    )


# Remove internal rank column from paper table.
paper = paper.drop(
    columns=["Rank column"]
)


paper_tsv = (
    OUT /
    "module5_ablation_paper_table.tsv"
)

paper_csv = (
    OUT /
    "module5_ablation_paper_table.csv"
)

paper.to_csv(
    paper_tsv,
    sep="\t",
    index=False,
)

paper.to_csv(
    paper_csv,
    index=False,
)


# ------------------------------------------------------------
# LARGEST RANK CHANGES
# ------------------------------------------------------------

if (
    "binding_only_rank"
    in ablation.columns
):

    shifts = ablation.copy()

    shifts[
        "rank_change_binding_to_full"
    ] = (
        shifts[
            "binding_only_rank"
        ]
        - shifts[
            "full_srsi_rank"
        ]
    )

    shifts[
        "absolute_rank_change"
    ] = (
        shifts[
            "rank_change_binding_to_full"
        ]
        .abs()
    )

    keep = [
        c
        for c in [
            "molecule_id",
            "binding_only_rank",
            "full_srsi_rank",
            "rank_change_binding_to_full",
            "absolute_rank_change",
            "target_deltaG",
            "weighted_selectivity_gap",
            "predicted_splicing_rescue_score",
            "off_target_risk_score",
            "srsi_score",
        ]
        if c in shifts.columns
    ]

    largest = (
        shifts[
            keep
        ]
        .sort_values(
            "absolute_rank_change",
            ascending=False,
        )
        .head(15)
    )

    largest.to_csv(
        OUT /
        "largest_binding_to_full_rank_shifts.tsv",
        sep="\t",
        index=False,
    )


print("\n======================================")
print("ABLATION ANALYSIS COMPLETE")
print("======================================")

print("\nPaper table:\n")

print(
    paper.to_string(
        index=False
    )
)

print("\nSaved:")
print(raw_path)
print(paper_tsv)
print(paper_csv)

if (
    OUT /
    "largest_binding_to_full_rank_shifts.tsv"
).exists():

    print(
        OUT /
        "largest_binding_to_full_rank_shifts.tsv"
    )
