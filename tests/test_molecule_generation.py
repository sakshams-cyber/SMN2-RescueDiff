from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from smn2_rescuediff.molecule_generation.candidate_prioritizer import add_candidate_scores, select_docking_candidates
from smn2_rescuediff.molecule_generation.chemical_filter import (
    calculate_properties,
    mol_from_smiles,
    process_generated_smiles,
)
from smn2_rescuediff.molecule_generation.cli import run
from smn2_rescuediff.molecule_generation.config import load_config, resolve_paths
from smn2_rescuediff.molecule_generation.dataset_builder import build_training_dataset
from smn2_rescuediff.molecule_generation.molecular_tokenizer import MolecularTokenizer, smiles_tokens, tokenize_molecule
from smn2_rescuediff.molecule_generation.neural_sampler import sample_neural_strings
from smn2_rescuediff.molecule_generation.report_writer import build_summary, write_summary_json
from smn2_rescuediff.molecule_generation.representation_converter import (
    representation_to_smiles,
    selfies_available,
    smiles_to_representation,
)
from smn2_rescuediff.molecule_generation.rnn_model import build_rnn_model
from smn2_rescuediff.molecule_generation.transformer_model import build_transformer_model


def _advanced_filter_config() -> dict:
    return {
        "run": {"random_seed": 42},
        "chemical_filters": {
            "min_molecular_weight": 20,
            "max_molecular_weight": 500,
            "min_logp": -5,
            "max_logp": 8,
            "max_hbd": 10,
            "max_hba": 20,
            "max_rotatable_bonds": 20,
            "max_formal_charge_abs": 2,
            "require_lipinski_pass": False,
            "remove_pains": False,
            "remove_invalid_smiles": True,
            "remove_duplicates": True,
            "remove_salts": True,
        },
        "analysis": {
            "calculate_qed": True,
            "calculate_novelty_against_training": True,
            "calculate_diversity": True,
            "pairwise_diversity_sample_size": 50,
            "tanimoto_similarity_threshold_duplicate_like": 0.95,
            "max_training_fingerprints_for_novelty": 100,
        },
    }


def _tiny_cli_config(tmp_path: Path) -> Path:
    training = tmp_path / "training_smiles.smi"
    training.write_text(
        "\n".join(
            [
                "CCO",
                "CCN",
                "CCOC",
                "CCCO",
                "c1ccccc1",
                "CC(=O)O",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    context = tmp_path / "context.json"
    context.write_text(json.dumps({"ok": True}), encoding="utf-8")
    profile = tmp_path / "profile.tsv"
    profile.write_text("window_name\tsequence_rna\nx\tAUG\n", encoding="utf-8")
    output_dir = tmp_path / "out"
    figure_dir = tmp_path / "figures"
    report = tmp_path / "report.md"
    config = tmp_path / "molecule_generation.yaml"
    config.write_text(
        f"""
project_name: SMN2-RescueDiff
module_name: Deep Neural Molecular Generation Model
inputs:
  training_smiles:
    - {training}
  optional_context:
    smn2_target_profile: {profile}
    smn2_target_metadata: {context}
    decoy_atlas_summary: {context}
outputs:
  output_dir: {output_dir}
  checkpoint_dir: {output_dir / "checkpoints"}
  log_dir: {output_dir / "logs"}
  report_path: {report}
  figure_dir: {figure_dir}
run:
  random_seed: 3
  device: cpu
  run_name: pytest_tiny
  overwrite_existing: true
  save_config_used: true
data_processing:
  min_training_molecules_warning: 100
  max_input_smiles_length: 80
  remove_salts: true
  remove_mixtures: true
  canonicalize_smiles: true
  remove_duplicates: true
  require_rdkit_valid: true
  allowed_atoms: [C, H, N, O, S, P, F, Cl, Br, I]
  min_molecular_weight_training: 20
  max_molecular_weight_training: 500
tokenization:
  representation: selfies
  fallback_to_smiles_if_selfies_missing: true
  add_start_end_tokens: true
generation:
  generator_mode: inference_only
model:
  architecture: transformer_decoder
  embedding_dim: 16
  num_layers: 1
  num_heads: 2
  feedforward_dim: 32
  hidden_dim: 16
  dropout: 0.0
  max_sequence_length: 80
training:
  enabled: false
  epochs: 1
  batch_size: 4
  learning_rate: 0.001
  weight_decay: 0.0
  validation_fraction: 0.2
  early_stopping_patience: 1
  gradient_clip_norm: 1.0
  save_checkpoint_every_epoch: false
  resume_from_checkpoint: null
sampling:
  num_raw_to_generate: 6
  batch_size_generate: 4
  temperature: 0.8
  top_k: 10
  top_p: 0.95
  max_generated_length: 80
  remove_duplicate_generated_strings: false
rl_finetuning:
  enabled: false
  episodes: 1
  samples_per_episode: 4
  save_rl_checkpoints: false
chemical_filters:
  min_molecular_weight: 20
  max_molecular_weight: 500
  min_logp: -5
  max_logp: 8
  max_hbd: 10
  max_hba: 20
  max_rotatable_bonds: 20
  max_formal_charge_abs: 2
  require_lipinski_pass: false
  remove_pains: false
  remove_invalid_smiles: true
  remove_duplicates: true
  remove_salts: true
candidate_prioritization:
  enabled: true
  max_candidates_to_export_for_docking: 5
  diversity_selection_enabled: true
analysis:
  calculate_qed: true
  calculate_sa_score_if_available: true
  calculate_morgan_fingerprints: true
  calculate_diversity: true
  calculate_novelty_against_training: true
  calculate_scaffolds: true
  pairwise_diversity_sample_size: 20
  tanimoto_similarity_threshold_duplicate_like: 0.95
""",
        encoding="utf-8",
    )
    return config


def test_molecule_generation_config_loads():
    config = load_config(PROJECT / "configs" / "molecule_generation.yaml")
    assert config["module_name"] == "Deep Neural Molecular Generation Model"
    assert config["generation"]["generator_mode"] == "neural_transformer"
    assert config["tokenization"]["representation"] in {"selfies", "smiles"}
    assert isinstance(config["inputs"]["training_smiles"], list)


def test_training_smiles_loads_and_cleans(tmp_path):
    training = tmp_path / "training.smi"
    training.write_text("CCO\nCCO duplicate\ninvalid\nCC.Cl\nc1ccccc1 benzene\n", encoding="utf-8")
    config_path = _tiny_cli_config(tmp_path)
    config = load_config(config_path)
    config["inputs"]["training_smiles"] = [str(training)]
    output_dir = tmp_path / "dataset_out"
    cleaned, summary, rejected, warnings = build_training_dataset(config, tmp_path, output_dir)
    assert cleaned == ["CCO", "c1ccccc1"]
    assert summary["training_molecules_raw"] == 5
    assert not rejected.empty
    assert any("too small" in warning for warning in warnings)


def test_quality_training_manifest_builds_weighted_source_tables(tmp_path):
    source_a = tmp_path / "chembl.smi"
    source_b = tmp_path / "rna.smi"
    source_a.write_text("CCO\nCCN\nCC.Cl\ninvalid\n", encoding="utf-8")
    source_b.write_text("CCO\nc1ccncc1\nCCOC\n", encoding="utf-8")
    manifest = tmp_path / "sources.tsv"
    manifest.write_text(
        "source_name\tpath\tsource_type\tsource_weight\tmax_rows\tquality_tier\tnotes\n"
        f"chembl\t{source_a}\tbroad\t1.0\t100\ttier2\tbaseline\n"
        f"rna\t{source_b}\trna_ligand_like\t3.0\t100\ttier1\toverweight\n",
        encoding="utf-8",
    )
    config_path = _tiny_cli_config(tmp_path)
    config = load_config(config_path)
    config["inputs"].pop("training_smiles", None)
    config["inputs"]["training_source_manifest"] = str(manifest)
    config["data_processing"]["min_molecular_weight_training"] = 20
    cleaned, summary, rejected, warnings = build_training_dataset(config, tmp_path, tmp_path / "quality_out")
    assert len(set(cleaned)) == 4
    assert len(cleaned) > len(set(cleaned))
    assert summary["training_molecules_cleaned"] == 4
    assert summary["training_molecules_weighted"] > summary["training_molecules_cleaned"]
    assert not rejected.empty
    quality_dir = tmp_path / "quality_out" / "quality_training"
    assert (quality_dir / "quality_training_unique.tsv").exists()
    assert (quality_dir / "quality_training_weighted.tsv").exists()
    assert (quality_dir / "source_balance_report.tsv").exists()


def test_selfies_conversion_round_trips_known_smiles():
    pytest.importorskip("selfies")
    assert selfies_available()
    converted, rejected = smiles_to_representation(["CCO", "c1ccccc1"], "selfies")
    assert rejected == []
    decoded = [representation_to_smiles(item, "selfies") for item in converted]
    assert all(mol_from_smiles(item) is not None for item in decoded)


def test_tokenizer_supports_smiles_and_selfies():
    pytest.importorskip("selfies")
    assert smiles_tokens("ClCBr[NH4+]") == ["Cl", "C", "Br", "[NH4+]"]
    smiles_tokenizer = MolecularTokenizer.build(["CCO", "ClCBr"], representation="smiles")
    assert smiles_tokenizer.decode(smiles_tokenizer.encode("ClCBr")) == "ClCBr"
    selfies_strings, _ = smiles_to_representation(["CCO"], "selfies")
    selfies_tokenizer = MolecularTokenizer.build(selfies_strings, representation="selfies")
    encoded = selfies_tokenizer.encode(selfies_strings[0])
    assert tokenize_molecule(selfies_strings[0], "selfies")
    assert selfies_tokenizer.decode(encoded) == selfies_strings[0]


def test_transformer_forward_pass_shape():
    torch = pytest.importorskip("torch")
    model = build_transformer_model(vocab_size=12, embedding_dim=16, num_layers=1, num_heads=2, feedforward_dim=32, dropout=0.0, max_sequence_length=16)
    x = torch.tensor([[1, 4, 5, 2], [1, 6, 2, 0]], dtype=torch.long)
    logits = model(x, padding_mask=x.eq(0))
    assert tuple(logits.shape) == (2, 4, 12)


def test_gru_fallback_forward_pass_shape():
    torch = pytest.importorskip("torch")
    model = build_rnn_model(vocab_size=12, embedding_dim=8, hidden_dim=10, num_layers=1, dropout=0.0, architecture="gru")
    x = torch.tensor([[1, 4, 5, 2]], dtype=torch.long)
    logits = model(x)
    assert tuple(logits.shape) == (1, 4, 12)


def test_neural_sampler_produces_molecular_strings():
    pytest.importorskip("torch")
    tokenizer = MolecularTokenizer.build(["CCO", "CCN", "CCC"], representation="smiles")
    model = build_rnn_model(vocab_size=len(tokenizer.token_to_id), embedding_dim=8, hidden_dim=8, num_layers=1, dropout=0.0, architecture="gru")
    config = {
        "run": {"random_seed": 4},
        "model": {"architecture": "gru"},
        "sampling": {"max_generated_length": 12, "temperature": 1.0, "top_k": 0, "top_p": 1.0, "remove_duplicate_generated_strings": False},
    }
    generated = sample_neural_strings(model, tokenizer, config, count=4, device_name="cpu")
    assert len(generated) == 4
    assert {"raw_generated_string", "decoded_smiles", "representation"}.issubset(generated.columns)


def test_rdkit_validates_decoded_known_examples():
    pytest.importorskip("selfies")
    converted, _ = smiles_to_representation(["CCO", "CCN"], "selfies")
    decoded = [representation_to_smiles(item, "selfies") for item in converted]
    assert all(mol_from_smiles(item) is not None for item in decoded)


def test_chemical_filters_remove_invalid_and_bad_molecules():
    generated = pd.DataFrame(
        {
            "source_generator": ["test", "test", "test"],
            "representation": ["smiles", "smiles", "smiles"],
            "raw_generated_string": ["CCO", "C1CC", "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC"],
            "decoded_smiles": ["CCO", "C1CC", "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC"],
        }
    )
    config = _advanced_filter_config()
    config["chemical_filters"]["max_molecular_weight"] = 100
    valid, filtered, rejected = process_generated_smiles(generated, config)
    assert len(valid) == 2
    assert len(filtered) == 1
    reasons = ";".join(rejected["rejection_reason"].astype(str))
    assert "invalid_smiles" in reasons
    assert "molecular_weight_above_max" in reasons


def test_duplicate_canonical_smiles_are_removed():
    generated = pd.DataFrame(
        {
            "source_generator": ["test", "test", "test"],
            "representation": ["smiles", "smiles", "smiles"],
            "raw_generated_string": ["CCO", "OCC", "CCN"],
            "decoded_smiles": ["CCO", "OCC", "CCN"],
        }
    )
    valid, filtered, rejected = process_generated_smiles(generated, _advanced_filter_config())
    assert len(valid) == 3
    assert len(filtered) == 2
    assert "duplicate_canonical_smiles" in set(rejected["rejection_reason"])


def test_candidate_scoring_adds_prioritization_columns():
    mol = mol_from_smiles("c1ccncc1")
    props = calculate_properties(mol)
    df = pd.DataFrame([{**props, "canonical_smiles": "c1ccncc1", "is_valid": True, "is_duplicate": False, "is_novel_vs_training": True}])
    scored = add_candidate_scores(df)
    assert "rna_ligand_like_score" in scored
    assert "chemical_quality_score" in scored
    assert "final_generation_selection_score" in scored
    assert "pre_docking_candidate_score" in scored
    assert 0 <= float(scored["final_generation_selection_score"].iloc[0]) <= 1


def test_quality_weighted_docking_selection_exports_diverse_top_candidates():
    smiles = ["c1ccncc1", "c1ccccc1", "CCOC(=O)N", "CCN(CC)CC", "CCCCCCCCCCCC"]
    rows = []
    for item in smiles:
        mol = mol_from_smiles(item)
        rows.append({**calculate_properties(mol), "canonical_smiles": item, "is_valid": True, "is_duplicate": False, "is_novel_vs_training": True, "max_tanimoto_to_training": 0.2})
    selected = select_docking_candidates(
        pd.DataFrame(rows),
        {
            "candidate_prioritization": {
                "max_candidates_to_export_for_docking": 3,
                "morgan_diversity_selection_enabled": True,
                "tanimoto_similarity_threshold_for_diversity": 0.75,
            }
        },
    )
    assert len(selected) == 3
    assert selected["final_generation_selection_score"].notna().all()


def test_summary_json_contains_required_fields(tmp_path):
    generated = pd.DataFrame(
        {
            "source_generator": ["test"],
            "representation": ["smiles"],
            "raw_generated_string": ["CCO"],
            "decoded_smiles": ["CCO"],
        }
    )
    valid, filtered, rejected = process_generated_smiles(generated, _advanced_filter_config())
    summary = build_summary(
        config={
            "project_name": "SMN2-RescueDiff",
            "module_name": "Deep Neural Molecular Generation Model",
            "generation": {"generator_mode": "inference_only"},
            "tokenization": {"representation": "smiles"},
            "model": {"architecture": "transformer_decoder"},
        },
        context_status={"smn2_target_profile_exists": True, "smn2_target_metadata_exists": True, "decoy_atlas_summary_exists": True},
        training_path="tiny.smi",
        toy_warning_used=True,
        raw_count=1,
        valid=valid,
        filtered=filtered,
        rejected=rejected,
        warnings=["toy"],
    )
    path = tmp_path / "summary.json"
    write_summary_json(summary, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["module"] == "Deep Neural Molecular Generation Model"
    assert payload["generator_mode"] == "inference_only"
    assert "property_summary" in payload


def test_resolve_paths_accepts_training_smiles_list(tmp_path):
    config_path = _tiny_cli_config(tmp_path)
    config = load_config(config_path)
    paths = resolve_paths(config, config_path)
    assert len(paths.training_smiles) == 1
    assert paths.output_dir == tmp_path / "out"


def test_molecule_generation_cli_runs_on_tiny_test_mode(tmp_path):
    config_path = _tiny_cli_config(tmp_path)
    config = load_config(config_path)
    assert run(config_path) == 0
    output_dir = Path(config["outputs"]["output_dir"])
    figure_dir = Path(config["outputs"]["figure_dir"])
    report = Path(config["outputs"]["report_path"])
    assert (output_dir / "generated_strings_raw.tsv").exists()
    assert (output_dir / "generated_smiles_raw.smi").exists()
    assert (output_dir / "generated_molecules_filtered.tsv").exists()
    assert (output_dir / "generated_molecules_for_docking.tsv").exists()
    assert (output_dir / "generated_molecule_summary.json").exists()
    assert (output_dir / "token_vocabulary.json").exists()
    assert (output_dir / "tokenizer_metadata.json").exists()
    assert (output_dir / "RUN_SUMMARY.md").exists()
    assert (output_dir / "run_metrics.json").exists()
    run_dir = output_dir / "runs" / "pytest_tiny"
    assert (run_dir / "generated_molecules_for_docking.tsv").exists()
    assert (run_dir / "molecule_generation_config_used.yaml").exists()
    assert (run_dir / "RUN_SUMMARY.md").exists()
    assert (run_dir / "run_metrics.json").exists()
    assert report.exists()
    assert (figure_dir / "training_loss_curve.png").exists()
    assert (figure_dir / "validity_uniqueness_novelty_summary.png").exists()
    filtered = pd.read_csv(output_dir / "generated_molecules_filtered.tsv", sep="\t")
    assert {
        "molecule_id",
        "representation",
        "raw_generated_string",
        "decoded_smiles",
        "canonical_smiles",
        "rna_ligand_like_score",
        "pre_docking_candidate_score",
        "selected_for_docking",
    }.issubset(filtered.columns)


def test_molecule_generation_cli_writes_quality_training_outputs_from_manifest(tmp_path):
    source = tmp_path / "rna_source.smi"
    source.write_text("CCO\nCCN\nCCOC\nc1ccncc1\n", encoding="utf-8")
    manifest = tmp_path / "quality_sources.tsv"
    manifest.write_text(
        "source_name\tpath\tsource_type\tsource_weight\tmax_rows\tquality_tier\tnotes\n"
        f"rna\t{source}\trna_ligand_like\t2.0\t100\ttier1\ttiny test\n",
        encoding="utf-8",
    )
    config_path = _tiny_cli_config(tmp_path)
    config = load_config(config_path)
    config["inputs"].pop("training_smiles", None)
    config["inputs"]["training_source_manifest"] = str(manifest)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    assert run(config_path) == 0
    output_dir = Path(config["outputs"]["output_dir"])
    assert (output_dir / "quality_training" / "quality_training_unique.tsv").exists()
    assert (output_dir / "quality_training" / "quality_training_weighted.tsv").exists()
    assert (output_dir / "quality_training" / "source_balance_report.tsv").exists()
    docking = pd.read_csv(output_dir / "generated_molecules_for_docking.tsv", sep="\t")
    assert {"chemical_quality_score", "final_generation_selection_score", "tpsa", "synthetic_accessibility_proxy"}.issubset(docking.columns)
