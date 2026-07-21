"""Write HuRDA atlas tables, FASTA files, summaries, and figures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


FINAL_COLUMNS = [
    "decoy_id",
    "decoy_category",
    "motif_type",
    "chromosome",
    "start",
    "end",
    "strand",
    "gene_id",
    "gene_name",
    "transcript_id",
    "transcript_type",
    "source_feature",
    "window_name",
    "sequence_rna",
    "sequence_length",
    "A_freq",
    "U_freq",
    "G_freq",
    "C_freq",
    "GC_content",
    "purine_fraction",
    "pyrimidine_fraction",
    "dinucleotide_counts_json",
    "kmer3_counts_json",
    "hamming_distance_to_smn2",
    "levenshtein_distance_to_smn2",
    "kmer_jaccard_similarity_to_smn2",
    "smn2_similarity_score",
    "simple_splice_strength_score",
    "u1_complementarity_score",
    "contains_splice_boundary",
    "distance_to_nearest_exon_boundary",
    "bias_match_score",
    "biological_weight",
    "train_val_test_split",
    "holdout_group",
    "is_duplicate_removed",
    "is_exact_smn2_match",
    "notes",
]


def assign_decoy_ids(decoys: pd.DataFrame) -> pd.DataFrame:
    """Assign stable HuRDA decoy identifiers."""
    result = decoys.reset_index(drop=True).copy()
    result["decoy_id"] = [f"HuRDA_{idx:06d}" for idx in range(1, len(result) + 1)]
    return result


def normalize_final_columns(decoys: pd.DataFrame) -> pd.DataFrame:
    """Ensure the full atlas schema exists and appears in the required order."""
    result = decoys.copy()
    for column in FINAL_COLUMNS:
        if column not in result.columns:
            result[column] = "NA"
    return result[FINAL_COLUMNS]


def write_fasta(decoys: pd.DataFrame, path: str | Path) -> None:
    """Write the full atlas FASTA file."""
    with Path(path).open("w", encoding="utf-8") as handle:
        for _, row in decoys.iterrows():
            header = (
                f">{row['decoy_id']}|category={row['decoy_category']}|gene={row['gene_name']}|"
                f"transcript={row['transcript_id']}|chrom={row['chromosome']}|start={row['start']}|"
                f"end={row['end']}|strand={row['strand']}"
            )
            handle.write(header + "\n")
            sequence = str(row["sequence_rna"])
            for idx in range(0, len(sequence), 80):
                handle.write(sequence[idx : idx + 80] + "\n")


def category_counts(decoys: pd.DataFrame) -> pd.DataFrame:
    """Count decoys by category."""
    if decoys.empty:
        return pd.DataFrame(columns=["decoy_category", "count"])
    return decoys["decoy_category"].value_counts().rename_axis("decoy_category").reset_index(name="count")


def feature_summary(decoys: pd.DataFrame) -> pd.DataFrame:
    """Summarize numeric atlas features for QC."""
    columns = [
        "GC_content",
        "smn2_similarity_score",
        "bias_match_score",
        "simple_splice_strength_score",
        "u1_complementarity_score",
        "sequence_length",
    ]
    rows = []
    for column in columns:
        if column not in decoys:
            continue
        series = pd.to_numeric(decoys[column], errors="coerce")
        rows.append(
            {
                "feature": column,
                "min": float(series.min()) if not series.empty else None,
                "max": float(series.max()) if not series.empty else None,
                "mean": float(series.mean()) if not series.empty else None,
            }
        )
    return pd.DataFrame(rows)


def build_summary(
    decoys: pd.DataFrame,
    removed: pd.DataFrame,
    config: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    """Build machine-readable QC summary JSON."""
    target_total = int(config.get("atlas", {}).get("target_total_decoys", 180000))
    split_counts = decoys["train_val_test_split"].value_counts().to_dict() if "train_val_test_split" in decoys else {}
    holdout_counts = decoys["holdout_group"].value_counts().to_dict() if "holdout_group" in decoys else {}
    gc = pd.to_numeric(decoys.get("GC_content", pd.Series(dtype=float)), errors="coerce")
    sim = pd.to_numeric(decoys.get("smn2_similarity_score", pd.Series(dtype=float)), errors="coerce")
    return {
        "project": config.get("project_name", "SMN2-RescueDiff"),
        "module": config.get("module_name", "HuRDA-180K Human RNA Decoy Atlas"),
        "total_decoys": int(len(decoys)),
        "target_total_decoys": target_total,
        "reached_target_total": bool(len(decoys) >= target_total),
        "category_counts": category_counts(decoys).to_dict(orient="records"),
        "duplicates_removed": int(removed.get("is_duplicate_removed", pd.Series(dtype=bool)).sum()) if not removed.empty else 0,
        "exact_smn2_matches_removed": int(removed.get("is_exact_smn2_match", pd.Series(dtype=bool)).sum()) if not removed.empty else 0,
        "gc_content": {"min": float(gc.min()) if len(gc) else None, "max": float(gc.max()) if len(gc) else None, "mean": float(gc.mean()) if len(gc) else None},
        "smn2_similarity_score": {"min": float(sim.min()) if len(sim) else None, "max": float(sim.max()) if len(sim) else None, "mean": float(sim.mean()) if len(sim) else None},
        "split_counts": split_counts,
        "holdout_counts": holdout_counts,
        "warnings": warnings,
    }


def write_figures(decoys: pd.DataFrame, figure_dir: str | Path) -> None:
    """Create required single-panel matplotlib figures."""
    path = Path(figure_dir)
    path.mkdir(parents=True, exist_ok=True)
    counts = category_counts(decoys)
    plt.figure(figsize=(9, 4.5))
    plt.bar(counts["decoy_category"], counts["count"])
    plt.xticks(rotation=35, ha="right", fontsize=8)
    plt.ylabel("Decoy count")
    plt.title("HuRDA decoy category counts")
    plt.tight_layout()
    plt.savefig(path / "decoy_category_counts.png", dpi=200)
    plt.close()

    histogram_specs = [
        ("GC_content", "gc_content_distribution.png", "GC content"),
        ("smn2_similarity_score", "smn2_similarity_distribution.png", "SMN2 similarity score"),
        ("bias_match_score", "bias_match_score_distribution.png", "Bias-match score"),
        ("simple_splice_strength_score", "splice_strength_distribution.png", "Simple splice-site strength score"),
        ("u1_complementarity_score", "u1_complementarity_distribution.png", "U1 complementarity score"),
    ]
    for column, filename, title in histogram_specs:
        plt.figure(figsize=(7, 4.5))
        pd.to_numeric(decoys[column], errors="coerce").dropna().hist(bins=40)
        plt.xlabel(title)
        plt.ylabel("Count")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(path / filename, dpi=200)
        plt.close()


def write_atlas_outputs(
    decoys: pd.DataFrame,
    removed: pd.DataFrame,
    config: dict[str, Any],
    output_dir: str | Path,
    figure_dir: str | Path,
    generation_log: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    """Write all requested HuRDA output files."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    final = normalize_final_columns(assign_decoy_ids(decoys))
    final.to_csv(out / "hurda_180k_full.tsv", sep="\t", index=False)
    write_fasta(final, out / "hurda_180k_full.fasta")
    final[final["train_val_test_split"] == "train"].to_csv(out / "hurda_train.tsv", sep="\t", index=False)
    final[final["train_val_test_split"] == "validation"].to_csv(out / "hurda_validation.tsv", sep="\t", index=False)
    final[final["train_val_test_split"] == "test"].to_csv(out / "hurda_test.tsv", sep="\t", index=False)
    final[final["holdout_group"] == "hard_smn2_like_holdout"].to_csv(out / "hurda_hard_smn2_like_holdout.tsv", sep="\t", index=False)
    final[final["holdout_group"] == "chromosome_holdout"].to_csv(out / "hurda_chromosome_holdout.tsv", sep="\t", index=False)
    final[final["holdout_group"] == "gene_holdout"].to_csv(out / "hurda_gene_holdout.tsv", sep="\t", index=False)
    category_counts(final).to_csv(out / "hurda_decoy_category_counts.tsv", sep="\t", index=False)
    feature_summary(final).to_csv(out / "hurda_feature_summary.tsv", sep="\t", index=False)
    removed.to_csv(out / "hurda_removed_duplicates.tsv", sep="\t", index=False)
    (out / "hurda_generation_log.txt").write_text("\n".join(generation_log) + "\n", encoding="utf-8")
    summary = build_summary(final, removed, config, warnings)
    (out / "hurda_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_figures(final, figure_dir)
    return summary

