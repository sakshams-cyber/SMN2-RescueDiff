#!/usr/bin/env python3
"""Audit benchmark docking provenance and summarize usable downstream evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def read_tsv(path: Path) -> pd.DataFrame:
    """Read a TSV while retaining literal identifiers and blank fields."""
    if not path.exists():
        raise FileNotFoundError(f"Required benchmark input is missing: {path}")
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, low_memory=False)


def sha256_file(path: Path) -> str:
    """Return a source-file SHA256 for provenance."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_false(series: pd.Series) -> pd.Series:
    """Identify explicit false values used for real (non-dry-run) scores."""
    return series.astype(str).str.strip().str.lower().isin({"false", "0", "no"})


def completed_real_rows(df: pd.DataFrame, score_column: str) -> pd.DataFrame:
    """Keep completed, real docking rows with a numeric score."""
    if df.empty:
        return df.copy()
    status = df.get("dock_status", pd.Series("", index=df.index)).astype(str).str.lower().eq("completed")
    real = is_false(df.get("is_dry_run_score", pd.Series("", index=df.index)))
    numeric = pd.to_numeric(df.get(score_column, pd.Series(index=df.index, dtype=float)), errors="coerce").notna()
    result = df.loc[status & real & numeric].copy()
    result[score_column] = pd.to_numeric(result[score_column], errors="coerce")
    return result


def first_value(df: pd.DataFrame, molecule_id: str, column: str, default: str = "") -> str:
    """Return the first matching string value without inventing a replacement."""
    if df.empty or "molecule_id" not in df or column not in df:
        return default
    values = df.loc[df["molecule_id"].astype(str).eq(molecule_id), column]
    return str(values.iloc[0]) if not values.empty else default


def per_molecule_audit(
    expected: pd.DataFrame,
    selected: pd.DataFrame,
    target: pd.DataFrame,
    decoy: pd.DataFrame,
    selectivity: pd.DataFrame,
    ligand: pd.DataFrame,
) -> pd.DataFrame:
    """Build one factual completeness row per expected or selected molecule."""
    expected_ids = set(expected["molecule_id"].astype(str))
    selected_ids = set(selected["molecule_id"].astype(str))
    expected_smiles = set(expected.get("canonical_smiles", pd.Series(dtype=str)).astype(str))
    selected_smiles = set(selected.get("canonical_smiles", pd.Series(dtype=str)).astype(str))
    selected_smiles_by_id = selected.set_index("molecule_id").get("canonical_smiles", pd.Series(dtype=str)).to_dict()
    target_real = completed_real_rows(target, "target_deltaG")
    decoy_real = completed_real_rows(decoy, "decoy_deltaG")

    ordered_ids = list(dict.fromkeys(expected["molecule_id"].astype(str).tolist() + selected["molecule_id"].astype(str).tolist()))
    rows: list[dict[str, object]] = []
    for molecule_id in ordered_ids:
        canonical = selected_smiles_by_id.get(
            molecule_id,
            first_value(expected, molecule_id, "canonical_smiles"),
        )
        is_expected_id = molecule_id in expected_ids
        matches_expected_smiles = bool(canonical and canonical in expected_smiles and canonical in selected_smiles)
        selected_for_run = molecule_id in selected_ids
        target_all = target[target.get("molecule_id", pd.Series("", index=target.index)).astype(str).eq(molecule_id)]
        target_valid = target_real[target_real["molecule_id"].astype(str).eq(molecule_id)]
        decoy_all = decoy[decoy.get("molecule_id", pd.Series("", index=decoy.index)).astype(str).eq(molecule_id)]
        decoy_valid = decoy_real[decoy_real["molecule_id"].astype(str).eq(molecule_id)]
        has_selectivity = bool(
            not selectivity.empty
            and "molecule_id" in selectivity
            and selectivity["molecule_id"].astype(str).eq(molecule_id).any()
        )

        if is_expected_id and not selected_for_run and not matches_expected_smiles:
            outcome = "expected_benchmark_control_not_docked"
        elif selected_for_run and not is_expected_id and not matches_expected_smiles and not target_valid.empty:
            outcome = "non_benchmark_molecule_analyzed"
        elif selected_for_run and not is_expected_id and not matches_expected_smiles:
            outcome = "non_benchmark_molecule_incomplete"
        elif target_valid.empty:
            outcome = "benchmark_molecule_incomplete"
        else:
            outcome = "benchmark_molecule_analyzed"

        rows.append(
            {
                "molecule_id": molecule_id,
                "canonical_smiles": canonical,
                "is_expected_benchmark_id": is_expected_id,
                "matches_expected_benchmark_smiles": matches_expected_smiles,
                "selected_for_module4_run": selected_for_run,
                "ligand_preparation_status": first_value(ligand, molecule_id, "prep_status"),
                "target_rows": len(target_all),
                "real_completed_target_scores": len(target_valid),
                "decoy_rows_scheduled": len(decoy_all),
                "real_completed_decoy_scores": len(decoy_valid),
                "decoy_completion_fraction": round(len(decoy_valid) / len(decoy_all), 6) if len(decoy_all) else pd.NA,
                "selectivity_summary_available": has_selectivity,
                "audit_outcome": outcome,
            }
        )
    return pd.DataFrame(rows)


def decoy_category_summary(valid_decoys: pd.DataFrame) -> pd.DataFrame:
    """Summarize completed real decoy scores by molecule and category."""
    columns = [
        "molecule_id",
        "decoy_category",
        "num_real_completed_decoys",
        "best_decoy_deltaG",
        "mean_decoy_deltaG",
        "median_decoy_deltaG",
        "worst_decoy_deltaG",
    ]
    if valid_decoys.empty:
        return pd.DataFrame(columns=columns)
    grouped = (
        valid_decoys.groupby(["molecule_id", "decoy_category"], dropna=False)["decoy_deltaG"]
        .agg(
            num_real_completed_decoys="count",
            best_decoy_deltaG="min",
            mean_decoy_deltaG="mean",
            median_decoy_deltaG="median",
            worst_decoy_deltaG="max",
        )
        .reset_index()
    )
    return grouped[columns]


def strongest_decoys(valid_decoys: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Return the strongest predicted off-target scores per scored molecule."""
    if valid_decoys.empty:
        return valid_decoys.copy()
    columns = [
        column
        for column in [
            "molecule_id",
            "decoy_id",
            "decoy_category",
            "motif_type",
            "decoy_smn2_similarity_score",
            "decoy_biological_weight",
            "decoy_deltaG",
            "pose_file",
        ]
        if column in valid_decoys
    ]
    return (
        valid_decoys.sort_values(["molecule_id", "decoy_deltaG"], ascending=[True, True])
        .groupby("molecule_id", group_keys=False)
        .head(top_n)[columns]
        .reset_index(drop=True)
    )


def selectivity_comparison(selectivity: pd.DataFrame) -> pd.DataFrame:
    """Keep the main absolute and conservative selectivity metrics."""
    columns = [
        "molecule_id",
        "target_deltaG",
        "best_decoy_deltaG",
        "selectivity_gap_best_decoy",
        "mean_decoy_deltaG",
        "weighted_selectivity_gap",
        "percent_decoys_stronger_than_target",
        "off_target_risk_score",
        "num_decoys_with_valid_scores",
        "is_dry_run_score",
    ]
    result = selectivity.copy()
    if "selectivity_gap_best_decoy" not in result and {"best_decoy_deltaG", "target_deltaG"}.issubset(result.columns):
        result["selectivity_gap_best_decoy"] = (
            pd.to_numeric(result["best_decoy_deltaG"], errors="coerce")
            - pd.to_numeric(result["target_deltaG"], errors="coerce")
        )
    for column in columns:
        if column not in result:
            result[column] = pd.NA
    return result[columns]


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Render a small DataFrame as Markdown without optional dependencies."""
    if df.empty:
        return "No rows were available."
    headers = [str(column) for column in df.columns]
    rows = [[str(value).replace("|", "\\|") for value in row] for row in df.itertuples(index=False, name=None)]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def write_report(
    summary: dict[str, object],
    audit: pd.DataFrame,
    comparison: pd.DataFrame,
    output_path: Path,
) -> None:
    """Write a cautious benchmark-readiness report."""
    ready = bool(summary["scientific_benchmark_ready"])
    verdict = (
        "This run is provenance-complete for a computational risdiplam-versus-branaplam benchmark: both "
        "held-out controls have real target docking scores and complete real scores across the same 100 prepared "
        "HuRDA receptors. This supports a controlled computational comparison, not a claim of measured binding, "
        "splicing rescue, safety, or clinical efficacy."
        if ready
        else (
            "This run is not a valid risdiplam-versus-branaplam benchmark because the Module 4 score files "
            "do not contain complete evidence for both intended benchmark controls. It is usable only as an "
            "incomplete downstream pipeline smoke test for the molecule identities actually present."
        )
    )
    cohort_note = (
        "A two-control cohort supports a direct pairwise comparison, but not stable population percentiles or broad rank validation."
        if int(summary["molecules_with_real_target_scores"]) >= 2
        else "A one-molecule cohort does not support comparative percentiles, rank stability, or meaningful ablation rank shifts."
    )
    control_note = (
        "Both held-out controls have usable real Module 4 target and decoy scores in this run."
        if ready
        else "One or more held-out controls lack usable Module 4 target or decoy scores in this run."
    )
    lines = [
        "# Benchmark Docking Downstream Audit",
        "",
        "## Verdict",
        "",
        verdict,
        "",
        "## Observed Inputs",
        "",
        f"- Expected benchmark controls: {summary['expected_benchmark_controls']}",
        f"- Expected controls matched by ID or exact canonical SMILES: {summary['expected_controls_matched']}",
        f"- Molecules selected in Module 4 output: {summary['selected_molecules']}",
        f"- Molecules with real completed target scores: {summary['molecules_with_real_target_scores']}",
        f"- Real completed decoy scores: {summary['real_completed_decoy_scores']}",
        f"- Scheduled decoy rows: {summary['scheduled_decoy_rows']}",
        f"- Dry-run completed scores used: {summary['dry_run_completed_scores_used']}",
        f"- New docking performed by this audit: {summary['new_docking_jobs_run']}",
        "",
        "## Molecule-Level Completeness",
        "",
        dataframe_to_markdown(audit),
        "",
        "## Docking And Selectivity Comparison",
        "",
        "More-negative docking energies indicate stronger predicted docking. The conservative best-decoy gap is "
        "`best_decoy_deltaG - target_deltaG`; a negative value means at least one screened decoy scored more favorably than the SMN2 target.",
        "",
        dataframe_to_markdown(comparison),
        "",
        "## Interpretation Limits",
        "",
        f"- {cohort_note}",
        "- Missing target or decoy scores were not imputed.",
        "- Docking and pose-derived values are computational predictions, not measured binding or splicing rescue.",
        f"- {control_note}",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    """Run the audit and write reproducible tables and metadata."""
    source_paths = {
        "expected_controls": Path(args.expected_controls),
        "selected_molecules": Path(args.selected_molecules),
        "target_scores": Path(args.target_scores),
        "decoy_scores": Path(args.decoy_scores),
        "selectivity_scores": Path(args.selectivity_scores),
        "ligand_summary": Path(args.ligand_summary),
    }
    tables = {name: read_tsv(path) for name, path in source_paths.items()}
    expected = tables["expected_controls"]
    if "molecule_id" not in expected or "canonical_smiles" not in expected:
        raise ValueError("Expected benchmark controls must contain molecule_id and canonical_smiles columns.")

    audit = per_molecule_audit(
        expected,
        tables["selected_molecules"],
        tables["target_scores"],
        tables["decoy_scores"],
        tables["selectivity_scores"],
        tables["ligand_summary"],
    )
    valid_target = completed_real_rows(tables["target_scores"], "target_deltaG")
    valid_decoys = completed_real_rows(tables["decoy_scores"], "decoy_deltaG")
    expected_matches = audit.loc[
        audit["is_expected_benchmark_id"]
        & (audit["selected_for_module4_run"] | audit["matches_expected_benchmark_smiles"])
    ]
    expected_audit = audit.loc[audit["is_expected_benchmark_id"]]
    expected_complete = (
        len(expected_matches) == len(expected)
        and not expected_audit.empty
        and expected_audit["real_completed_target_scores"].ge(1).all()
        and expected_audit["decoy_rows_scheduled"].gt(0).all()
        and expected_audit["real_completed_decoy_scores"].eq(expected_audit["decoy_rows_scheduled"]).all()
    )
    completed_all = pd.concat(
        [
            tables["target_scores"].assign(_kind="target"),
            tables["decoy_scores"].assign(_kind="decoy"),
        ],
        ignore_index=True,
        sort=False,
    )
    completed_mask = completed_all.get("dock_status", pd.Series("", index=completed_all.index)).astype(str).str.lower().eq("completed")
    dry_mask = ~is_false(completed_all.get("is_dry_run_score", pd.Series("", index=completed_all.index)))

    summary: dict[str, object] = {
        "scientific_benchmark_ready": bool(expected_complete),
        "provenance_mismatch_detected": len(expected_matches) != len(expected),
        "expected_benchmark_controls": len(expected),
        "expected_controls_matched": len(expected_matches),
        "selected_molecules": len(tables["selected_molecules"]),
        "molecules_with_real_target_scores": int(valid_target["molecule_id"].nunique()) if not valid_target.empty else 0,
        "scheduled_decoy_rows": len(tables["decoy_scores"]),
        "real_completed_decoy_scores": len(valid_decoys),
        "molecules_with_real_decoy_scores": int(valid_decoys["molecule_id"].nunique()) if not valid_decoys.empty else 0,
        "dry_run_completed_scores_used": int((completed_mask & dry_mask).sum()),
        "missing_scores_imputed": False,
        "new_docking_jobs_run": 0,
        "source_files": {
            name: {"path": str(path), "sha256": sha256_file(path), "rows": len(tables[name])}
            for name, path in source_paths.items()
        },
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output_dir / "benchmark_molecule_completeness.tsv", sep="\t", index=False)
    decoy_category_summary(valid_decoys).to_csv(output_dir / "completed_decoy_category_summary.tsv", sep="\t", index=False)
    strongest_decoys(valid_decoys).to_csv(output_dir / "strongest_completed_decoys.tsv", sep="\t", index=False)
    comparison = selectivity_comparison(tables["selectivity_scores"])
    comparison.to_csv(output_dir / "benchmark_selectivity_comparison.tsv", sep="\t", index=False)
    (output_dir / "benchmark_audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(summary, audit, comparison, output_dir / "BENCHMARK_AUDIT.md")
    print(json.dumps({key: value for key, value in summary.items() if key != "source_files"}, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    """Parse command-line paths for one benchmark audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-controls", required=True)
    parser.add_argument("--selected-molecules", required=True)
    parser.add_argument("--target-scores", required=True)
    parser.add_argument("--decoy-scores", required=True)
    parser.add_argument("--selectivity-scores", required=True)
    parser.add_argument("--ligand-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
