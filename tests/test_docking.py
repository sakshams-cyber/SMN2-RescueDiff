from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from smn2_rescuediff.docking.cli import (
    _benchmark_molecule_selection_from_target_scores,
    _filter_molecules_from_stage_scores,
    _ligand_file_for,
    _prior_stage_target_scores,
    run,
)
from smn2_rescuediff.docking.config import load_config
from smn2_rescuediff.docking.decoy_panel_selector import select_decoy_panel
from smn2_rescuediff.docking.dry_run_docker import dry_run_target_score
from smn2_rescuediff.docking.input_loader import load_molecules
from smn2_rescuediff.docking.ligand_preparer import prepare_ligands
from smn2_rescuediff.docking.live_logger import LiveLogger
from smn2_rescuediff.docking.molecule_selector import select_molecules_for_docking
from smn2_rescuediff.docking.selectivity_scorer import calculate_selectivity_scores
from smn2_rescuediff.docking.vina_runner import validate_vina_input_files


def _molecules(path: Path) -> Path:
    df = pd.DataFrame(
        [
            {"molecule_id": "Mol_001", "canonical_smiles": "CCO", "molecular_weight": 46.07, "logp": -0.1, "qed": 0.4, "lipinski_pass": True, "filter_pass": True},
            {"molecule_id": "Mol_002", "canonical_smiles": "CCN", "molecular_weight": 45.08, "logp": -0.2, "qed": 0.45, "lipinski_pass": True, "filter_pass": True},
            {"molecule_id": "Mol_003", "canonical_smiles": "c1ccccc1", "molecular_weight": 78.11, "logp": 1.6, "qed": 0.44, "lipinski_pass": True, "filter_pass": True},
        ]
    )
    df.to_csv(path, sep="\t", index=False)
    return path


def _decoys(path: Path) -> Path:
    rows = []
    categories = ["smn2_like_hard_decoy", "five_prime_splice_site", "random_bias_matched"]
    for idx in range(9):
        category = categories[idx % len(categories)]
        rows.append(
            {
                "decoy_id": f"HuRDA_{idx + 1:06d}",
                "decoy_category": category,
                "motif_type": category,
                "chromosome": "chrMini",
                "start": idx + 1,
                "end": idx + 20,
                "strand": "+",
                "gene_name": f"GENE{idx}",
                "transcript_id": f"TX{idx}",
                "sequence_rna": "AUGCUAUGCU",
                "sequence_length": 10,
                "smn2_similarity_score": 0.9 - idx * 0.05,
                "bias_match_score": 0.8 - idx * 0.03,
                "biological_weight": 1.0 if category == "smn2_like_hard_decoy" else 0.5,
            }
        )
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
    return path


def _target_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    fasta = tmp_path / "smn2_target_windows.fasta"
    profile = tmp_path / "smn2_target_profile.tsv"
    metadata = tmp_path / "smn2_target_metadata.json"
    fasta.write_text(">SMN2|window=target\nAUGGUAAGU\n", encoding="utf-8")
    pd.DataFrame([{"window_name": "target", "sequence_rna": "AUGGUAAGU"}]).to_csv(profile, sep="\t", index=False)
    metadata.write_text(json.dumps({"module": "target_definition"}), encoding="utf-8")
    return fasta, profile, metadata


def test_docking_config_loads_correctly():
    config = load_config(PROJECT / "configs" / "docking_counter_screening.yaml")
    assert config["module_name"] == "SMN2 Target Docking and HuRDA Off-Target Counter-Screening"
    assert config["run"]["run_mode"] in {"dry_run", "prepare_only", "target_only_docking", "target_plus_decoy_docking", "analyze_existing"}


def test_molecule_input_loads_correctly(tmp_path):
    path = _molecules(tmp_path / "molecules.tsv")
    df = load_molecules(path)
    assert len(df) == 3
    assert "canonical_smiles" in df.columns


def test_molecule_selection_respects_max(tmp_path):
    df = load_molecules(_molecules(tmp_path / "molecules.tsv"))
    selected = select_molecules_for_docking(df, 2, 42)
    assert len(selected) == 2


def test_rdkit_ligand_preparation_works_for_valid_smiles(tmp_path):
    df = pd.DataFrame([{"molecule_id": "Mol_001", "canonical_smiles": "CCO"}])
    summary = prepare_ligands(df, tmp_path, {"ligand_preparation": {"prepare_pdbqt": False}})
    assert summary.iloc[0]["prep_status"] == "prepared"
    assert Path(summary.iloc[0]["sdf_file"]).exists()


def test_ligand_preparer_creates_pdbqt_with_obabel_when_available(tmp_path, monkeypatch):
    fake_obabel = tmp_path / "obabel"
    fake_obabel.write_text(
        "#!/usr/bin/env python3\n"
        "import pathlib, sys\n"
        "out = sys.argv[sys.argv.index('-O') + 1]\n"
        "pathlib.Path(out).write_text('REMARK fake pdbqt\\n')\n",
        encoding="utf-8",
    )
    fake_obabel.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ.get('PATH', '')}")
    df = pd.DataFrame([{"molecule_id": "Mol_001", "canonical_smiles": "CCO"}])
    summary = prepare_ligands(df, tmp_path / "ligands", {"ligand_preparation": {"prepare_pdbqt": True}})
    pdbqt_file = Path(summary.iloc[0]["pdbqt_file"])
    assert pdbqt_file.exists()
    assert pdbqt_file.suffix == ".pdbqt"


def test_invalid_smiles_handled_gracefully(tmp_path):
    df = pd.DataFrame([{"molecule_id": "Mol_bad", "canonical_smiles": "C1CC"}])
    summary = prepare_ligands(df, tmp_path, {"ligand_preparation": {}})
    assert summary.iloc[0]["prep_status"] == "failed"


def test_vina_ligand_selection_requires_pdbqt_and_never_falls_back_to_sdf(tmp_path):
    sdf_file = tmp_path / "Mol_001.sdf"
    sdf_file.write_text("sdf", encoding="utf-8")
    pdbqt_file = tmp_path / "Mol_001.pdbqt"
    pdbqt_file.write_text("pdbqt", encoding="utf-8")
    summary = pd.DataFrame([{"molecule_id": "Mol_001", "sdf_file": str(sdf_file), "pdbqt_file": str(pdbqt_file)}])
    assert _ligand_file_for("Mol_001", summary, require_pdbqt=True) == str(pdbqt_file)
    missing_summary = pd.DataFrame([{"molecule_id": "Mol_001", "sdf_file": str(sdf_file), "pdbqt_file": "NA"}])
    assert _ligand_file_for("Mol_001", missing_summary, require_pdbqt=True) == "NA"


def test_vina_validation_rejects_sdf_ligand_with_exact_message(tmp_path):
    receptor = tmp_path / "target.pdbqt"
    receptor.write_text("receptor", encoding="utf-8")
    ligand = tmp_path / "ligand.sdf"
    ligand.write_text("sdf", encoding="utf-8")
    try:
        validate_vina_input_files(str(receptor), str(ligand))
    except FileNotFoundError as exc:
        assert str(exc) == "Ligand PDBQT file is missing; Vina requires PDBQT, not SDF."
    else:
        raise AssertionError("SDF ligand should not pass Vina input validation.")


def test_decoy_panel_selector_respects_category_targets(tmp_path):
    decoys = pd.read_csv(_decoys(tmp_path / "decoys.tsv"), sep="\t")
    panel = select_decoy_panel(
        decoys,
        {
            "decoy_panel": {
                "total_decoys_for_docking": 4,
                "category_targets": {"smn2_like_hard_decoy": 2, "five_prime_splice_site": 1},
            }
        },
    )
    assert len(panel) == 4
    assert (panel["decoy_category"] == "smn2_like_hard_decoy").sum() >= 2


def test_decoy_panel_selector_filters_hard_decoys_and_preserves_receptor_file(tmp_path):
    decoys = pd.read_csv(_decoys(tmp_path / "decoys.tsv"), sep="\t")
    decoys["decoy_receptor_file"] = [str(tmp_path / f"decoy_{idx}.pdbqt") for idx in range(len(decoys))]
    panel = select_decoy_panel(
        decoys,
        {
            "decoy_panel": {
                "use_decoy_panel": True,
                "total_decoys_for_docking": 2,
                "include_categories": ["smn2_like_hard_decoy", "five_prime_splice_site"],
                "min_smn2_similarity": 0.75,
                "min_biological_weight": 0.5,
                "category_targets": {"smn2_like_hard_decoy": 1, "five_prime_splice_site": 1},
                "decoy_receptor_file_column": "decoy_receptor_file",
            }
        },
    )
    assert len(panel) == 2
    assert set(panel["decoy_category"]).issubset({"smn2_like_hard_decoy", "five_prime_splice_site"})
    assert panel["smn2_similarity_score"].min() >= 0.75
    assert panel["decoy_receptor_file"].str.endswith(".pdbqt").all()


def test_staged_docking_reuses_prior_target_scores(tmp_path):
    molecules = pd.DataFrame(
        [
            {"molecule_id": "Mol_001", "canonical_smiles": "CCO"},
            {"molecule_id": "Mol_002", "canonical_smiles": "CCN"},
            {"molecule_id": "Mol_003", "canonical_smiles": "CCC"},
        ]
    )
    score_path = tmp_path / "stage_a" / "smn2_target_docking_scores.tsv"
    score_path.parent.mkdir()
    pd.DataFrame(
        [
            {"molecule_id": "Mol_001", "canonical_smiles": "CCO", "target_id": "SMN2_exon7", "target_name": "target", "receptor_file": "r.pdbqt", "ligand_file": "l1.pdbqt", "dock_status": "completed", "target_deltaG": -6.0, "best_pose_rank": 1, "num_poses": 1, "pose_file": "p1.pdbqt", "engine": "vina", "run_mode": "target_only_docking", "is_dry_run_score": False, "notes": "stage a"},
            {"molecule_id": "Mol_002", "canonical_smiles": "CCN", "target_id": "SMN2_exon7", "target_name": "target", "receptor_file": "r.pdbqt", "ligand_file": "l2.pdbqt", "dock_status": "completed", "target_deltaG": -8.5, "best_pose_rank": 1, "num_poses": 1, "pose_file": "p2.pdbqt", "engine": "vina", "run_mode": "target_only_docking", "is_dry_run_score": False, "notes": "stage a"},
            {"molecule_id": "Mol_003", "canonical_smiles": "CCC", "target_id": "SMN2_exon7", "target_name": "target", "receptor_file": "r.pdbqt", "ligand_file": "l3.pdbqt", "dock_status": "completed", "target_deltaG": -7.0, "best_pose_rank": 1, "num_poses": 1, "pose_file": "p3.pdbqt", "engine": "vina", "run_mode": "target_only_docking", "is_dry_run_score": False, "notes": "stage a"},
        ]
    ).to_csv(score_path, sep="\t", index=False)
    config = {
        "run": {"max_molecules_to_dock": 2},
        "staged_docking": {
            "select_top_from_target_scores": True,
            "target_scores_path": str(score_path),
            "top_molecules_from_target_scores": 2,
            "reuse_prior_target_scores": True,
        },
    }
    selected = _filter_molecules_from_stage_scores(molecules, config, tmp_path)
    assert selected["molecule_id"].tolist() == ["Mol_002", "Mol_003"]
    prior = _prior_stage_target_scores(config, tmp_path, selected, "target_plus_decoy_docking")
    assert prior["molecule_id"].tolist() == ["Mol_002", "Mol_003"]
    assert prior["run_mode"].tolist() == ["target_plus_decoy_docking", "target_plus_decoy_docking"]
    assert prior["is_dry_run_score"].tolist() == [False, False]


def test_benchmark_staged_selection_uses_target_scores_directly(tmp_path):
    score_path = tmp_path / "stage_a" / "smn2_target_docking_scores.tsv"
    score_path.parent.mkdir()
    pd.DataFrame(
        [
            {"molecule_id": "Mol_001", "canonical_smiles": "CCO", "dock_status": "completed", "target_deltaG": -7.2, "is_dry_run_score": False},
            {"molecule_id": "Mol_002", "canonical_smiles": "CCN", "dock_status": "completed", "target_deltaG": -8.9, "is_dry_run_score": False},
            {"molecule_id": "Mol_003", "canonical_smiles": "CCC", "dock_status": "failed", "target_deltaG": -10.0, "is_dry_run_score": False},
            {"molecule_id": "Mol_004", "canonical_smiles": "CCCl", "dock_status": "completed", "target_deltaG": -8.1, "is_dry_run_score": True},
            {"molecule_id": "Mol_005", "canonical_smiles": "CNC", "dock_status": "completed", "target_deltaG": -7.8, "is_dry_run_score": False},
            {"molecule_id": "Mol_006", "canonical_smiles": "COC", "dock_status": "completed", "target_deltaG": -6.5, "is_dry_run_score": False},
        ]
    ).to_csv(score_path, sep="\t", index=False)
    config = {
        "run": {"max_molecules_to_dock": 3, "require_real_docking_for_scientific_results": True},
        "staged_docking": {
            "select_top_from_target_scores": True,
            "ignore_advance_filters_for_benchmark": True,
            "target_scores_path": str(score_path),
            "top_molecules_from_target_scores": 100,
        },
    }
    selected, message = _benchmark_molecule_selection_from_target_scores(config, tmp_path)
    assert selected["molecule_id"].tolist() == ["Mol_002", "Mol_005", "Mol_001"]
    assert selected["stage_target_deltaG"].tolist() == [-8.9, -7.8, -7.2]
    assert "ignored advancement filters" in message


def test_benchmark_staged_selection_fails_when_request_is_underfilled(tmp_path):
    score_path = tmp_path / "stage_a" / "smn2_target_docking_scores.tsv"
    score_path.parent.mkdir()
    pd.DataFrame(
        [
            {"molecule_id": "Mol_001", "canonical_smiles": "CCO", "dock_status": "completed", "target_deltaG": -7.2, "is_dry_run_score": False},
            {"molecule_id": "Mol_002", "canonical_smiles": "CCN", "dock_status": "completed", "target_deltaG": -8.9, "is_dry_run_score": False},
        ]
    ).to_csv(score_path, sep="\t", index=False)
    config = {
        "run": {"max_molecules_to_dock": 100, "require_real_docking_for_scientific_results": True},
        "staged_docking": {
            "select_top_from_target_scores": True,
            "ignore_advance_filters_for_benchmark": True,
            "target_scores_path": str(score_path),
            "top_molecules_from_target_scores": 100,
        },
    }
    with pytest.raises(ValueError, match="requested 100 molecules, but only 2 eligible"):
        _benchmark_molecule_selection_from_target_scores(config, tmp_path)


def test_dry_run_docker_deterministic_scores():
    molecule = pd.Series({"molecule_id": "Mol_001", "qed": 0.5, "molecular_weight": 250, "logp": 2.0})
    left = dry_run_target_score(molecule, 42)
    right = dry_run_target_score(molecule, 42)
    assert left.delta_g == right.delta_g
    assert left.is_dry_run_score is True


def test_selectivity_gap_calculation_is_correct():
    target = pd.DataFrame([{"molecule_id": "M1", "canonical_smiles": "CCO", "target_deltaG": -9.0, "run_mode": "dry_run", "is_dry_run_score": True}])
    decoys = pd.DataFrame(
        [
            {"molecule_id": "M1", "decoy_deltaG": -6.0, "decoy_category": "random_bias_matched", "decoy_biological_weight": 1.0},
            {"molecule_id": "M1", "decoy_deltaG": -8.5, "decoy_category": "smn2_like_hard_decoy", "decoy_biological_weight": 1.0},
        ]
    )
    scores = calculate_selectivity_scores(target, decoys, {"selectivity_scoring": {"advance_min_decoys_with_scores": 1}})
    assert scores.iloc[0]["selectivity_gap_best_decoy"] == 0.5


def test_off_target_risk_score_between_zero_and_one():
    target = pd.DataFrame([{"molecule_id": "M1", "canonical_smiles": "CCO", "target_deltaG": -7.0, "run_mode": "dry_run", "is_dry_run_score": True}])
    decoys = pd.DataFrame([{"molecule_id": "M1", "decoy_deltaG": -7.5, "decoy_category": "hard", "decoy_biological_weight": 1.0}])
    scores = calculate_selectivity_scores(target, decoys, {"selectivity_scoring": {}})
    assert 0 <= scores.iloc[0]["off_target_risk_score"] <= 1


def test_live_jsonl_event_logging_and_state_update(tmp_path):
    logger = LiveLogger(tmp_path / "events.jsonl", tmp_path / "state.json", enabled=True)
    logger.event("module_started", run_mode="dry_run", status="running", message="start")
    state = logger.state(run_status="running", jobs_completed=1, total_jobs=2, last_message="half")
    assert (tmp_path / "events.jsonl").read_text(encoding="utf-8").strip()
    assert state["percent_complete"] == 50.0


def test_docking_cli_runs_in_dry_run_mode_on_tiny_inputs(tmp_path):
    molecules = _molecules(tmp_path / "molecules.tsv")
    decoys = _decoys(tmp_path / "decoys.tsv")
    decoy_summary = tmp_path / "hurda_summary.json"
    decoy_summary.write_text(json.dumps({"total_decoys": 9}), encoding="utf-8")
    fasta, profile, metadata = _target_files(tmp_path)
    raw_structures = tmp_path / "raw_structures"
    raw_structures.mkdir()
    output_dir = tmp_path / "docking"
    config = tmp_path / "docking_counter_screening.yaml"
    config.write_text(
        f"""
project_name: SMN2-RescueDiff
module_name: SMN2 Target Docking and HuRDA Off-Target Counter-Screening
inputs:
  molecules_filtered: {molecules}
  smn2_target_fasta: {fasta}
  smn2_target_profile: {profile}
  smn2_target_metadata: {metadata}
  decoy_atlas_full: {decoys}
  decoy_atlas_summary: {decoy_summary}
  raw_structure_dir: {raw_structures}
outputs:
  output_dir: {output_dir}
  ligand_dir: {output_dir / "ligands"}
  receptor_dir: {output_dir / "receptors"}
  pose_dir: {output_dir / "poses"}
  log_dir: {output_dir / "logs"}
  simulation_dir: {output_dir / "simulation"}
  report_path: {tmp_path / "report.md"}
  figure_dir: {tmp_path / "figures"}
run:
  random_seed: 13
  run_mode: dry_run
  max_molecules_to_dock: 2
  allow_dry_run_simulation: true
docking_engine:
  engine: vina
  vina_executable: vina
  exhaustiveness: 1
  num_modes: 1
  energy_range: 3
  timeout_seconds_per_job: 5
receptor:
  receptor_mode: user_provided
  smn2_receptor_file: {output_dir / "receptors" / "smn2_target.pdbqt"}
  receptor_format: pdbqt
  docking_box:
    center_x: 0
    center_y: 0
    center_z: 0
    size_x: 20
    size_y: 20
    size_z: 20
  require_user_to_confirm_box: false
ligand_preparation:
  input_smiles_column: canonical_smiles
  add_hydrogens: true
  generate_3d_conformers: true
  optimize_geometry: true
  max_conformer_attempts: 5
  output_format: sdf
  prepare_pdbqt: false
decoy_panel:
  use_decoy_panel: true
  total_decoys_for_docking: 3
  category_targets:
    smn2_like_hard_decoy: 1
    five_prime_splice_site: 1
    random_bias_matched: 1
  allow_sequence_only_decoy_panel: true
  decoy_receptor_mode: sequence_panel_only
selectivity_scoring:
  advance_target_deltaG_threshold: -7.0
  advance_weighted_gap_threshold: 1.0
  advance_off_target_risk_threshold: 0.40
  advance_min_decoys_with_scores: 1
live_simulation:
  enable_live_logging: true
  live_log_file: {output_dir / "simulation" / "live_docking_events.jsonl"}
  live_state_file: {output_dir / "simulation" / "current_docking_state.json"}
figures:
  make_figures: true
  max_heatmap_molecules: 5
  max_heatmap_decoys: 5
""",
        encoding="utf-8",
    )
    assert run(config) == 0
    assert (output_dir / "smn2_target_docking_scores.tsv").exists()
    assert (output_dir / "decoy_docking_scores.tsv").exists()
    assert (output_dir / "molecule_selectivity_scores.tsv").exists()
    assert (output_dir / "candidates_for_splicing_rescue.tsv").exists()
    assert (output_dir / "simulation" / "live_docking_events.jsonl").exists()
    assert (output_dir / "RUN_SUMMARY.md").exists()
    assert (output_dir / "run_metrics.json").exists()
    run_dir = output_dir / "runs" / "dry_run_run"
    assert (run_dir / "smn2_target_docking_scores.tsv").exists()
    assert (run_dir / "RUN_SUMMARY.md").exists()
    assert (run_dir / "run_metrics.json").exists()
    assert (tmp_path / "figures" / "target_vs_decoy_binding_scatter.png").exists()


def test_real_vina_mode_logs_missing_ligand_pdbqt_without_using_sdf(tmp_path, monkeypatch):
    fake_vina = tmp_path / "vina"
    fake_vina.write_text("#!/bin/sh\necho should_not_run >&2\nexit 99\n", encoding="utf-8")
    fake_vina.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ.get('PATH', '')}")
    molecules = _molecules(tmp_path / "molecules.tsv")
    decoys = _decoys(tmp_path / "decoys.tsv")
    decoy_summary = tmp_path / "hurda_summary.json"
    decoy_summary.write_text(json.dumps({"total_decoys": 9}), encoding="utf-8")
    fasta, profile, metadata = _target_files(tmp_path)
    raw_structures = tmp_path / "raw_structures"
    raw_structures.mkdir()
    output_dir = tmp_path / "docking_real"
    receptor = output_dir / "receptors" / "smn2_target.pdbqt"
    receptor.parent.mkdir(parents=True)
    receptor.write_text("REMARK receptor pdbqt\n", encoding="utf-8")
    config = tmp_path / "docking_counter_screening_real.yaml"
    config.write_text(
        f"""
project_name: SMN2-RescueDiff
module_name: SMN2 Target Docking and HuRDA Off-Target Counter-Screening
inputs:
  molecules_filtered: {molecules}
  smn2_target_fasta: {fasta}
  smn2_target_profile: {profile}
  smn2_target_metadata: {metadata}
  decoy_atlas_full: {decoys}
  decoy_atlas_summary: {decoy_summary}
  raw_structure_dir: {raw_structures}
outputs:
  output_dir: {output_dir}
  ligand_dir: {output_dir / "ligands"}
  receptor_dir: {output_dir / "receptors"}
  pose_dir: {output_dir / "poses"}
  log_dir: {output_dir / "logs"}
  simulation_dir: {output_dir / "simulation"}
  report_path: {tmp_path / "real_report.md"}
  figure_dir: {tmp_path / "real_figures"}
run:
  random_seed: 13
  run_mode: target_only_docking
  max_molecules_to_dock: 1
docking_engine:
  engine: vina
  vina_executable: {fake_vina}
  exhaustiveness: 1
  num_modes: 1
  energy_range: 3
  timeout_seconds_per_job: 5
receptor:
  receptor_mode: user_provided
  smn2_receptor_file: {receptor}
  receptor_format: pdbqt
  docking_box:
    center_x: 0
    center_y: 0
    center_z: 0
    size_x: 20
    size_y: 20
    size_z: 20
  require_user_to_confirm_box: false
ligand_preparation:
  input_smiles_column: canonical_smiles
  add_hydrogens: true
  generate_3d_conformers: true
  optimize_geometry: true
  max_conformer_attempts: 5
  output_format: sdf
  prepare_pdbqt: false
decoy_panel:
  use_decoy_panel: false
  allow_sequence_only_decoy_panel: true
  decoy_receptor_mode: sequence_panel_only
selectivity_scoring:
  advance_target_deltaG_threshold: -7.0
  advance_weighted_gap_threshold: 1.0
  advance_off_target_risk_threshold: 0.40
  advance_min_decoys_with_scores: 1
live_simulation:
  enable_live_logging: true
  live_log_file: {output_dir / "simulation" / "live_docking_events.jsonl"}
  live_state_file: {output_dir / "simulation" / "current_docking_state.json"}
figures:
  make_figures: true
  max_heatmap_molecules: 5
  max_heatmap_decoys: 5
""",
        encoding="utf-8",
    )
    assert run(config) == 0
    failed = pd.read_csv(output_dir / "logs" / "failed_docking_jobs.tsv", sep="\t")
    assert failed.iloc[0]["notes"] == "Ligand PDBQT file is missing; Vina requires PDBQT, not SDF."
    target_scores = pd.read_csv(output_dir / "smn2_target_docking_scores.tsv", sep="\t")
    assert target_scores.empty
    ligand_summary = pd.read_csv(output_dir / "ligand_preparation_summary.tsv", sep="\t", keep_default_na=False)
    assert Path(ligand_summary.iloc[0]["sdf_file"]).suffix == ".sdf"
    assert str(ligand_summary.iloc[0]["pdbqt_file"]) == "NA"


def test_dashboard_file_exists_after_module_generation():
    assert (PROJECT / "dashboard" / "docking_live_app.py").exists()
