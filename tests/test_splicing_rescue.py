from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from smn2_rescuediff.splicing_rescue.ablation_analysis import run_ablation_analysis
from smn2_rescuediff.splicing_rescue.cli import run
from smn2_rescuediff.splicing_rescue.config import load_config, resolve_paths, validate_weights
from smn2_rescuediff.splicing_rescue.evidence_writer import write_evidence_json
from smn2_rescuediff.splicing_rescue.input_loader import assemble_candidate_table, load_all_inputs
from smn2_rescuediff.splicing_rescue.pose_parser import parse_pose_for_candidate
from smn2_rescuediff.splicing_rescue.rescue_score_calculator import calculate_splicing_rescue_scores
from smn2_rescuediff.splicing_rescue.splicing_feature_scorer import infer_feature_based_components, selectivity_gap_score
from smn2_rescuediff.splicing_rescue.srsi_ranker import calculate_srsi
from smn2_rescuediff.splicing_rescue.target_landmark_annotator import annotate_landmarks


def _write_tsv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


def _toy_project(tmp_path: Path) -> Path:
    target = tmp_path / "data" / "processed" / "target_definition"
    decoy = tmp_path / "data" / "processed" / "decoy_atlas"
    mol = tmp_path / "data" / "processed" / "molecule_generation"
    dock = tmp_path / "data" / "processed" / "docking"
    poses = dock / "poses"
    for path in [target, decoy, mol, dock, poses]:
        path.mkdir(parents=True, exist_ok=True)
    (target / "smn2_target_windows.fasta").write_text(">exon7\nAUGGUAAGU\n", encoding="utf-8")
    _write_tsv(
        target / "smn2_target_profile.tsv",
        [
            {
                "target_id": "SMN2_exon7",
                "window_name": "exon7",
                "window_type": "exon",
                "sequence_rna": "AUGGUAAGU",
                "contains_splice_boundary": True,
                "distance_to_exon7_start": 0,
                "distance_to_exon7_end": 0,
            },
            {
                "target_id": "SMN2_5ss",
                "window_name": "five_prime_splice_site_core",
                "window_type": "splice_site",
                "sequence_rna": "AUGGUAAGU",
                "contains_splice_boundary": True,
                "distance_to_exon7_start": 108,
                "distance_to_exon7_end": 2,
            },
        ],
    )
    (target / "smn2_target_metadata.json").write_text(json.dumps({"selected_transcript_id": "ENST_TEST", "selected_exon": 7}), encoding="utf-8")
    _write_tsv(decoy / "hurda_180k_full.tsv", [{"decoy_id": "D1", "decoy_category": "smn2_like_hard_decoy", "biological_weight": 1.0}])
    (decoy / "hurda_summary.json").write_text(json.dumps({"total_decoys": 1}), encoding="utf-8")
    molecule_rows = [
        {
            "molecule_id": "MolA",
            "source_generator": "toy",
            "canonical_smiles": "CCO",
            "molecular_weight": 46.07,
            "logp": -0.1,
            "hbd": 1,
            "hba": 1,
            "rotatable_bonds": 0,
            "formal_charge": 0,
            "qed": 0.4,
            "sa_score": "NA",
            "lipinski_pass": True,
            "pains_alert": False,
            "pre_docking_candidate_score": 0.45,
        },
        {
            "molecule_id": "MolB",
            "source_generator": "toy",
            "canonical_smiles": "c1ccncc1",
            "molecular_weight": 79.1,
            "logp": 1.1,
            "hbd": 0,
            "hba": 1,
            "rotatable_bonds": 0,
            "formal_charge": 0,
            "qed": 0.55,
            "sa_score": "NA",
            "lipinski_pass": True,
            "pains_alert": False,
            "pre_docking_candidate_score": 0.7,
        },
    ]
    _write_tsv(mol / "generated_molecules_filtered.tsv", molecule_rows)
    _write_tsv(mol / "generated_molecules_for_docking.tsv", molecule_rows)
    (mol / "generated_molecule_summary.json").write_text(json.dumps({"filtered_count": 2}), encoding="utf-8")
    _write_tsv(
        dock / "smn2_target_docking_scores.tsv",
        [
            {
                "molecule_id": "MolA",
                "canonical_smiles": "CCO",
                "target_deltaG": -7.8,
                "target_id": "SMN2_exon7",
                "pose_file": "NA",
                "receptor_file": "NA",
                "ligand_file": "NA",
                "run_mode": "dry_run",
                "is_dry_run_score": True,
            },
            {
                "molecule_id": "MolB",
                "canonical_smiles": "c1ccncc1",
                "target_deltaG": -6.3,
                "target_id": "SMN2_exon7",
                "pose_file": "NA",
                "receptor_file": "NA",
                "ligand_file": "NA",
                "run_mode": "dry_run",
                "is_dry_run_score": True,
            },
        ],
    )
    _write_tsv(
        dock / "decoy_docking_scores.tsv",
        [
            {"molecule_id": "MolA", "decoy_id": "D1", "decoy_deltaG": -5.8, "is_dry_run_score": True},
            {"molecule_id": "MolB", "decoy_id": "D1", "decoy_deltaG": -6.9, "is_dry_run_score": True},
        ],
    )
    _write_tsv(
        dock / "molecule_selectivity_scores.tsv",
        [
            {
                "molecule_id": "MolA",
                "canonical_smiles": "CCO",
                "target_deltaG": -7.8,
                "best_decoy_deltaG": -5.8,
                "weighted_mean_decoy_deltaG": -5.8,
                "weighted_selectivity_gap": 2.0,
                "percent_decoys_stronger_than_target": 0.0,
                "off_target_risk_score": 0.10,
                "run_mode": "dry_run",
                "is_dry_run_score": True,
            },
            {
                "molecule_id": "MolB",
                "canonical_smiles": "c1ccncc1",
                "target_deltaG": -6.3,
                "best_decoy_deltaG": -6.9,
                "weighted_mean_decoy_deltaG": -6.9,
                "weighted_selectivity_gap": -0.6,
                "percent_decoys_stronger_than_target": 100.0,
                "off_target_risk_score": 0.80,
                "run_mode": "dry_run",
                "is_dry_run_score": True,
            },
        ],
    )
    _write_tsv(dock / "candidates_for_splicing_rescue.tsv", [{"molecule_id": "MolA", "target_deltaG": -7.8}])
    (dock / "docking_summary.json").write_text(json.dumps({"run_mode": "dry_run"}), encoding="utf-8")
    config = tmp_path / "configs" / "splicing_rescue_scoring.yaml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(
        f"""
project_name: SMN2-RescueDiff
module_name: Predicted Splicing-Rescue Scoring and Final SRSI Ranking
inputs:
  smn2_target_fasta: {target / "smn2_target_windows.fasta"}
  smn2_target_profile: {target / "smn2_target_profile.tsv"}
  smn2_target_metadata: {target / "smn2_target_metadata.json"}
  decoy_atlas_full: {decoy / "hurda_180k_full.tsv"}
  decoy_atlas_summary: {decoy / "hurda_summary.json"}
  generated_molecules_filtered: {mol / "generated_molecules_filtered.tsv"}
  generated_molecules_for_docking: {mol / "generated_molecules_for_docking.tsv"}
  molecule_generation_summary: {mol / "generated_molecule_summary.json"}
  target_docking_scores: {dock / "smn2_target_docking_scores.tsv"}
  decoy_docking_scores: {dock / "decoy_docking_scores.tsv"}
  molecule_selectivity_scores: {dock / "molecule_selectivity_scores.tsv"}
  candidates_for_splicing_rescue: {dock / "candidates_for_splicing_rescue.tsv"}
  docking_summary: {dock / "docking_summary.json"}
  pose_dir: {poses}
outputs:
  output_dir: {tmp_path / "out" / "final_scoring"}
  table_dir: {tmp_path / "out" / "final_scoring" / "tables"}
  evidence_dir: {tmp_path / "out" / "final_scoring" / "evidence"}
  log_dir: {tmp_path / "out" / "final_scoring" / "logs"}
  figure_dir: {tmp_path / "figures" / "final_scoring"}
  splicing_report_path: {tmp_path / "reports" / "splicing_rescue_scoring_report.md"}
  final_report_path: {tmp_path / "reports" / "final_candidate_report.md"}
run:
  random_seed: 1
  run_name: pytest_final
  overwrite_existing: true
  allow_dry_run_inputs: true
  require_real_docking_for_scientific_claims: true
  max_candidates_to_score: 10
  top_n_final_candidates: 2
  top_n_report_candidates: 2
splicing_landmarks:
  use_target_profile_landmarks: true
  infer_landmarks_from_window_names: true
  beneficial_landmarks: [five_prime_splice_site, u1_snRNA_recognition_region]
  caution_landmarks: [core_donor_dinucleotide, core_acceptor_dinucleotide]
pose_analysis:
  enabled: true
  allow_missing_pose_files: true
  contact_distance_angstrom: 4.5
  hydrogen_bond_distance_angstrom: 3.5
  stacking_distance_angstrom: 5.0
splicing_rescue_scoring:
  score_mode: hybrid_pose_and_feature
  normalize_scores: true
  weights:
    beneficial_landmark_proximity: 2.0
    u1_support_region_proximity: 2.0
    exon7_boundary_support: 1.5
    target_binding_strength: 1.5
    contact_density: 1.0
    hydrogen_bond_proxy: 0.8
    stacking_proxy: 0.7
    splice_site_occlusion_penalty: 2.0
    off_target_risk_penalty: 1.5
    missing_pose_penalty: 0.5
  thresholds:
    strong_target_binding_deltaG: -7.0
    minimum_selectivity_gap: 1.0
    maximum_off_target_risk_score: 0.40
    minimum_splicing_rescue_score_for_advancement: 0.60
srsi:
  enabled: true
  use_z_score_normalization: true
  weights:
    smn2_binding_strength: 1.0
    splicing_rescue_score: 2.0
    weighted_selectivity_gap: 1.5
    qed: 0.7
    lipinski_bonus: 0.4
    pre_docking_candidate_score: 0.5
    off_target_risk_penalty: 1.5
    sa_score_penalty: 0.5
    pains_penalty: 1.0
    splice_site_occlusion_penalty: 1.2
  candidate_tiers:
    high_priority_min_srsi_percentile: 90
    medium_priority_min_srsi_percentile: 70
    low_priority_min_srsi_percentile: 50
confidence:
  assign_confidence_level: true
  dry_run_confidence: very_low
  target_only_docking_confidence: low
  target_plus_decoy_docking_confidence: moderate
  pose_based_scoring_confidence_bonus: true
  missing_pose_confidence_penalty: true
ablation_analysis:
  enabled: true
figures:
  make_figures: true
  max_candidates_heatmap: 5
  max_candidates_waterfall: 2
""",
        encoding="utf-8",
    )
    return config


def _pipeline_objects(tmp_path: Path):
    config_path = _toy_project(tmp_path)
    config = load_config(config_path)
    paths = resolve_paths(config, config_path)
    inputs = load_all_inputs(paths)
    candidates = assemble_candidate_table(inputs, 10)
    landmarks, _warnings = annotate_landmarks(inputs["target_profile"], inputs["target_metadata"], config)
    contacts = pd.DataFrame(
        [
            {
                "molecule_id": "MolA",
                "pose_support_score": 0.0,
                "splice_site_occlusion_penalty": 0.0,
                "pose_analysis_status": "missing_pose_feature_based_fallback",
                "num_close_contacts": 0,
                "hydrogen_bond_proxy_count": 0,
                "stacking_proxy_count": 0,
            }
        ]
    )
    features = infer_feature_based_components(candidates, landmarks, config)
    rescue = calculate_splicing_rescue_scores(candidates, contacts, features, config, True)
    final = calculate_srsi(candidates, rescue, config, True)
    return config, paths, inputs, candidates, landmarks, rescue, final


def test_splicing_rescue_config_loads(tmp_path):
    config_path = _toy_project(tmp_path)
    config = load_config(config_path)
    validate_weights(config)
    assert config["module_name"] == "Predicted Splicing-Rescue Scoring and Final SRSI Ranking"


def test_module3_module4_table_merge_works(tmp_path):
    _config, _paths, inputs, candidates, _landmarks, _rescue, _final = _pipeline_objects(tmp_path)
    assert list(candidates["molecule_id"]) == ["MolA"]
    assert "molecular_weight" in candidates.columns
    assert "weighted_selectivity_gap" in candidates.columns


def test_target_landmarks_are_created(tmp_path):
    _config, _paths, _inputs, _candidates, landmarks, _rescue, _final = _pipeline_objects(tmp_path)
    assert not landmarks.empty
    assert "u1_snRNA_recognition_region" in set(landmarks["landmark_name"])


def test_missing_landmarks_are_handled_safely(tmp_path):
    config_path = _toy_project(tmp_path)
    config = load_config(config_path)
    unknown = pd.DataFrame([{"target_id": "x", "window_name": "unknown", "sequence_rna": "AUG"}])
    landmarks, warnings = annotate_landmarks(unknown, {}, config)
    assert landmarks.loc[0, "landmark_type"] == "NA"
    assert warnings


def test_pose_parser_handles_missing_pose_files():
    parsed, ligand_atoms, receptor_atoms = parse_pose_for_candidate(pd.Series({"molecule_id": "MolX", "pose_file": "NA", "receptor_file": "NA"}))
    assert parsed["pose_parse_status"] == "missing_pose_feature_based_fallback"
    assert ligand_atoms == []
    assert receptor_atoms == []


def test_feature_based_fallback_scores(tmp_path):
    _config, _paths, _inputs, candidates, landmarks, _rescue, _final = _pipeline_objects(tmp_path)
    features = infer_feature_based_components(candidates, landmarks, _config)
    assert features["feature_based_support_score"].between(0, 1).all()


def test_splicing_rescue_score_between_zero_and_one(tmp_path):
    _config, _paths, _inputs, _candidates, _landmarks, rescue, _final = _pipeline_objects(tmp_path)
    assert rescue["predicted_splicing_rescue_score"].between(0, 1).all()


def test_selectivity_gap_logic():
    scores = selectivity_gap_score(pd.Series([2.0, -1.0]), minimum_gap=1.0)
    assert scores.iloc[0] > scores.iloc[1]


def test_srsi_score_and_tiers_are_calculated(tmp_path):
    _config, _paths, _inputs, _candidates, _landmarks, _rescue, final = _pipeline_objects(tmp_path)
    assert "srsi_score" in final.columns
    assert final["candidate_tier"].notna().all()


def test_dry_run_inputs_are_flagged(tmp_path):
    _config, _paths, inputs, _candidates, _landmarks, _rescue, final = _pipeline_objects(tmp_path)
    assert inputs["used_dry_run_inputs"] is True
    assert final["is_dry_run_input"].astype(bool).all()


def test_ablation_ranking_works(tmp_path):
    config, _paths, _inputs, _candidates, _landmarks, _rescue, final = _pipeline_objects(tmp_path)
    ablation = run_ablation_analysis(final, config)
    assert "binding_only_rank" in ablation.columns
    assert "rank_change_binding_to_full" in ablation.columns


def test_evidence_json_is_written(tmp_path):
    _config, _paths, _inputs, _candidates, _landmarks, _rescue, final = _pipeline_objects(tmp_path)
    path = tmp_path / "evidence.json"
    evidence = write_evidence_json(final, 1, path)
    assert path.exists()
    assert list(evidence)


def test_splicing_rescue_cli_runs_on_tiny_inputs(tmp_path):
    config_path = _toy_project(tmp_path)
    assert run(config_path) == 0
    config = load_config(config_path)
    table_dir = Path(config["outputs"]["table_dir"])
    evidence_dir = Path(config["outputs"]["evidence_dir"])
    assert (table_dir / "final_srsi_ranked_candidates.tsv").exists()
    assert (table_dir / "splicing_rescue_scores.tsv").exists()
    assert (table_dir / "top_10_final_candidates.tsv").exists()
    assert (table_dir / "ablation_rank_comparison.tsv").exists()
    assert (evidence_dir / "top_candidate_evidence.json").exists()
    final = pd.read_csv(table_dir / "final_srsi_ranked_candidates.tsv", sep="\t")
    assert {"molecule_id", "srsi_score", "candidate_tier", "confidence_level"}.issubset(final.columns)

