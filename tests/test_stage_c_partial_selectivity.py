from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT / "scripts" / "run_stage_c_partial_ensemble_selectivity.py"
SPEC = importlib.util.spec_from_file_location("stage_c_partial", SCRIPT)
stage_c = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(stage_c)


def _write(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
    return path


def _write_empty(path: Path, columns: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=columns).to_csv(path, sep="\t", index=False)
    return path


def _fixture_args(tmp_path: Path) -> argparse.Namespace:
    top_rows = [
        {"molecule_id": "MolA", "canonical_smiles": "CCO"},
        {"molecule_id": "MolB", "canonical_smiles": "CCN"},
        {"molecule_id": "MolMissing", "canonical_smiles": "CCC"},
    ]
    target_rows = [
        {"molecule_id": "MolA", "num_conformers_completed": 2, "mean_target_deltaG": -8.0, "median_target_deltaG": -8.0, "best_target_deltaG": -8.5, "worst_target_deltaG": -7.5, "std_target_deltaG": 0.5, "target_deltaG_range": 1.0, "target_conformer_rank": 1},
        {"molecule_id": "MolB", "num_conformers_completed": 2, "mean_target_deltaG": -7.0, "median_target_deltaG": -7.0, "best_target_deltaG": -7.2, "worst_target_deltaG": -6.8, "std_target_deltaG": 0.2, "target_deltaG_range": 0.4, "target_conformer_rank": 2},
        {"molecule_id": "MolMissing", "num_conformers_completed": 2, "mean_target_deltaG": -9.0, "median_target_deltaG": -9.0, "best_target_deltaG": -9.1, "worst_target_deltaG": -8.9, "std_target_deltaG": 0.1, "target_deltaG_range": 0.2, "target_conformer_rank": 3},
    ]
    long_rows = []
    for molecule_id, scores in {"MolA": [-8.5, -7.5], "MolB": [-7.2, -6.8], "MolMissing": [-9.1, -8.9]}.items():
        for idx, score in enumerate(scores, start=1):
            long_rows.append({"molecule_id": molecule_id, "conformer_id": f"C{idx}", "target_deltaG": score, "dock_status": "completed", "is_dry_run_score": False})
    decoy_rows = [
        {"molecule_id": "MolA", "canonical_smiles": "CCO", "decoy_id": "D1", "dock_status": "completed", "decoy_deltaG": -6.0, "decoy_biological_weight": 1.0, "is_dry_run_score": False},
        {"molecule_id": "MolA", "canonical_smiles": "CCO", "decoy_id": "D2", "dock_status": "completed", "decoy_deltaG": -6.5, "decoy_biological_weight": 2.0, "is_dry_run_score": False},
        {"molecule_id": "MolB", "canonical_smiles": "CCN", "decoy_id": "D1", "dock_status": "completed", "decoy_deltaG": -7.4, "decoy_biological_weight": 1.0, "is_dry_run_score": False},
        {"molecule_id": "MolB", "canonical_smiles": "CCN", "decoy_id": "D2", "dock_status": "completed", "decoy_deltaG": -6.9, "decoy_biological_weight": 1.0, "is_dry_run_score": False},
    ]
    return argparse.Namespace(
        decoy_scores=_write(tmp_path / "decoys.tsv", decoy_rows),
        target_summary=_write(tmp_path / "target_summary.tsv", target_rows),
        target_long=_write(tmp_path / "target_long.tsv", long_rows),
        top150=_write(tmp_path / "top.tsv", top_rows),
        lower_confidence=_write(tmp_path / "repair.tsv", [{"molecule_id": "MolB"}, {"molecule_id": "MolMissing"}]),
        conformer_panel=_write(tmp_path / "panel.tsv", [{"conformer_id": "C1", "receptor_file": "r1.pdbqt"}, {"conformer_id": "C2", "receptor_file": "r2.pdbqt"}]),
        missing_target_jobs=_write_empty(tmp_path / "missing.tsv", ["molecule_id", "conformer_id"]),
        output_dir=tmp_path / "out",
        expected_molecules=3,
        expected_conformers=2,
        planned_decoys=2,
        expected_observed_rows=4,
        expected_covered_molecules=2,
        expected_common_decoys=2,
        top40_count=1,
        overwrite=False,
    )


def test_stage_c_partial_selectivity_keeps_zero_coverage_without_imputation(tmp_path, monkeypatch):
    monkeypatch.setattr(stage_c, "EXPECTED_ZERO_COVERAGE", {"MolMissing"})
    validation = stage_c.run(_fixture_args(tmp_path))
    assert validation["status"] == "passed"
    assert validation["missing_scores_imputed"] is False
    scores = pd.read_csv(tmp_path / "out" / "ensemble_selectivity_scores.tsv", sep="\t")
    assert len(scores) == 3
    missing = scores[scores["molecule_id"] == "MolMissing"].iloc[0]
    assert missing["evidence_status"] == "INSUFFICIENT_DECOY_EVIDENCE"
    assert pd.isna(missing["weighted_mean_selectivity_gap"])
    assert pd.isna(missing["partial_poc_selectivity_rank"])
    top = pd.read_csv(tmp_path / "out" / "top40_for_ligand_decoy_control_audit.tsv", sep="\t")
    assert len(top) == 1
    insufficient = pd.read_csv(tmp_path / "out" / "insufficient_decoy_evidence.tsv", sep="\t")
    assert insufficient["molecule_id"].tolist() == ["MolMissing"]
    payload = json.loads((tmp_path / "out" / "stage_c_validation.json").read_text(encoding="utf-8"))
    assert payload["evidence_status_counts"] == {"INSUFFICIENT_DECOY_EVIDENCE": 1, "PARTIAL_HURDA_97_COMPLETE": 2}


def test_stage_c_validation_stops_on_changed_observed_row_count(tmp_path, monkeypatch):
    monkeypatch.setattr(stage_c, "EXPECTED_ZERO_COVERAGE", {"MolMissing"})
    args = _fixture_args(tmp_path)
    args.expected_observed_rows = 5
    with pytest.raises(stage_c.StageCValidationError, match="row count changed"):
        stage_c.run(args)
