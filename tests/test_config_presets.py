from __future__ import annotations

import sys
from pathlib import Path

import yaml
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from smn2_rescuediff.docking.config import load_config as load_docking_config
from smn2_rescuediff.docking.config import resolve_paths as resolve_docking_paths
from smn2_rescuediff.molecule_generation.config import load_config as load_molecule_config
from smn2_rescuediff.molecule_generation.config import resolve_paths as resolve_molecule_paths
from smn2_rescuediff.splicing_rescue.config import load_config as load_splicing_config
from smn2_rescuediff.splicing_rescue.config import resolve_paths as resolve_splicing_paths
from smn2_rescuediff.splicing_rescue.confidence_assigner import assign_confidence


PRESETS = [
    "quick_test.yaml",
    "pilot_real.yaml",
    "final_rigorous.yaml",
    "module3_quick_test.yaml",
    "module3_pilot_real.yaml",
    "module3_final_rigorous.yaml",
    "module3_final_final_rigorous_x10.yaml",
    "module4_target_100.yaml",
    "module4_target_1000.yaml",
    "module4_decoy_stage1.yaml",
    "module4_decoy_stage2.yaml",
    "module5_target_only_final.yaml",
    "module5_decoy_final.yaml",
]


def _preset(name: str) -> Path:
    return PROJECT / "configs" / "presets" / name


def test_all_preset_configs_are_valid_yaml():
    for name in PRESETS:
        path = _preset(name)
        assert path.exists(), f"Missing preset: {name}"
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert isinstance(data, dict)
        assert data["project_name"] == "SMN2-RescueDiff"


def test_module3_presets_encode_debug_pilot_and_final_scales():
    quick = load_molecule_config(_preset("module3_quick_test.yaml"))
    pilot = load_molecule_config(_preset("module3_pilot_real.yaml"))
    final = load_molecule_config(_preset("module3_final_rigorous.yaml"))
    final_x10 = load_molecule_config(_preset("module3_final_final_rigorous_x10.yaml"))

    assert quick["generation"]["generator_mode"] == "debug_ngram"
    assert quick["training"]["enabled"] is False
    assert quick["sampling"]["num_raw_to_generate"] == 500
    assert pilot["generation"]["generator_mode"] == "neural_gru"
    assert pilot["model"]["architecture"] == "gru"
    assert pilot["candidate_prioritization"]["max_candidates_to_export_for_docking"] == 1000
    assert final["generation"]["generator_mode"] == "neural_transformer"
    assert final["inputs"]["training_smiles"] == [
        "data/raw/molecules/chembl_training_smiles.smi",
        "data/raw/molecules/zinc_training_smiles.smi",
        "data/raw/molecules/rna_ligand_training_smiles.smi",
    ]
    assert 10 <= final["training"]["epochs"] <= 20
    assert 100000 <= final["sampling"]["num_raw_to_generate"] <= 250000
    assert final["run"]["overwrite_existing"] is False
    assert final["candidate_prioritization"]["max_candidates_to_export_for_docking"] == 50000
    assert final_x10["inputs"]["training_source_manifest"] == "configs/molecule_training_sources_quality.tsv"
    assert final_x10["training"]["epochs"] == 10
    assert final_x10["sampling"]["num_raw_to_generate"] == 50000
    assert 500 <= final_x10["candidate_prioritization"]["max_candidates_to_export_for_docking"] <= 1000
    assert final_x10["rl_finetuning"]["enabled"] is False
    assert final_x10["candidate_prioritization"]["morgan_diversity_selection_enabled"] is True


def test_module4_presets_use_target_only_and_staged_decoy_panels():
    target_100 = load_docking_config(_preset("module4_target_100.yaml"))
    target_1000 = load_docking_config(_preset("module4_target_1000.yaml"))
    stage1 = load_docking_config(_preset("module4_decoy_stage1.yaml"))
    stage2 = load_docking_config(_preset("module4_decoy_stage2.yaml"))

    assert target_100["run"]["run_mode"] == "target_only_docking"
    assert target_100["run"]["max_molecules_to_dock"] == 100
    assert target_100["decoy_panel"]["use_decoy_panel"] is False
    assert target_1000["run"]["max_molecules_to_dock"] == 1000
    assert target_1000["staged_docking"]["stage_name"] == "stage_a_target_all"
    assert target_1000["run"]["resume_existing_results"] is True
    assert stage1["run"]["run_mode"] == "target_plus_decoy_docking"
    assert stage1["run"]["max_molecules_to_dock"] == 200
    assert stage1["decoy_panel"]["total_decoys_for_docking"] == 100
    assert stage1["staged_docking"]["reuse_prior_target_scores"] is True
    assert stage2["run"]["run_mode"] == "target_plus_decoy_docking"
    assert 50 <= stage2["run"]["max_molecules_to_dock"] <= 100
    assert 500 <= stage2["decoy_panel"]["total_decoys_for_docking"] <= 1000
    assert stage2["decoy_panel"]["total_decoys_for_docking"] < 180000
    assert stage2["decoy_panel"]["include_categories"][0] == "smn2_like_hard_decoy"
    assert stage2["decoy_panel"]["min_biological_weight"] >= 0.6


def test_module5_presets_reject_dry_run_inputs_and_encode_confidence_context():
    target_only = load_splicing_config(_preset("module5_target_only_final.yaml"))
    decoy_final = load_splicing_config(_preset("module5_decoy_final.yaml"))

    assert target_only["run"]["allow_dry_run_inputs"] is False
    assert target_only["splicing_rescue_scoring"]["score_mode"] == "feature_based"
    assert target_only["srsi"]["weights"]["weighted_selectivity_gap"] == 0.4
    assert decoy_final["run"]["allow_dry_run_inputs"] is False
    assert decoy_final["splicing_rescue_scoring"]["score_mode"] == "hybrid_pose_and_feature"
    assert decoy_final["confidence"]["target_plus_decoy_docking_confidence"] == "moderate"
    assert decoy_final["confidence"]["large_decoy_pose_ablation_confidence"] == "high"
    assert decoy_final["ablation_analysis"]["enabled"] is True


def test_module5_high_confidence_requires_large_decoy_panel_pose_and_ablation():
    config = load_splicing_config(_preset("module5_decoy_final.yaml"))
    rows = pd.DataFrame(
        [
            {"run_mode": "dry_run", "is_dry_run_input": True, "num_decoys_with_valid_scores": 750, "pose_analysis_status": "pose_based_proxy"},
            {"run_mode": "target_only_docking", "is_dry_run_input": False, "num_decoys_with_valid_scores": 0, "pose_analysis_status": "pose_based_proxy"},
            {"run_mode": "target_plus_decoy_docking", "is_dry_run_input": False, "num_decoys_with_valid_scores": 100, "pose_analysis_status": "missing_pose"},
            {"run_mode": "target_plus_decoy_docking", "is_dry_run_input": False, "num_decoys_with_valid_scores": 750, "pose_analysis_status": "pose_based_proxy"},
        ]
    )
    confidence = assign_confidence(rows, config, used_dry_run_inputs=False).tolist()
    assert confidence == ["very_low", "low", "low", "high"]


def test_pipeline_preset_manifests_group_module_configs():
    quick = yaml.safe_load(_preset("quick_test.yaml").read_text(encoding="utf-8"))
    pilot = yaml.safe_load(_preset("pilot_real.yaml").read_text(encoding="utf-8"))
    final = yaml.safe_load(_preset("final_rigorous.yaml").read_text(encoding="utf-8"))

    assert quick["preset_name"] == "quick_test"
    assert pilot["preset_name"] == "pilot_real"
    assert final["preset_name"] == "final_rigorous"
    assert final["module3_config"] == "configs/presets/module3_final_rigorous.yaml"
    assert "stage_c_top75_x750_hard_decoys" in final["module4_configs"]
    assert final["recommended_counts"]["maximum_direct_hurda_decoys_to_dock"] < 180000


def test_nested_preset_paths_resolve_from_project_root():
    molecule_path = _preset("module3_quick_test.yaml")
    docking_path = _preset("module4_target_100.yaml")
    splicing_path = _preset("module5_target_only_final.yaml")

    molecule_paths = resolve_molecule_paths(load_molecule_config(molecule_path), molecule_path)
    docking_paths = resolve_docking_paths(load_docking_config(docking_path), docking_path)
    splicing_paths = resolve_splicing_paths(load_splicing_config(splicing_path), splicing_path)

    assert molecule_paths.project_root == PROJECT
    assert docking_paths.project_root == PROJECT
    assert splicing_paths.project_root == PROJECT
    assert molecule_paths.output_dir == PROJECT / "data" / "processed" / "molecule_generation"
    assert docking_paths.output_dir == PROJECT / "data" / "processed" / "docking" / "quick_target_100"
    assert splicing_paths.output_dir == PROJECT / "data" / "processed" / "final_scoring"
