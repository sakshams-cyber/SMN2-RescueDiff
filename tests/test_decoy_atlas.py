from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from smn2_rescuediff.decoy_atlas.bias_matcher import remove_duplicate_and_exact_smn2_matches
from smn2_rescuediff.decoy_atlas.cli import run
from smn2_rescuediff.decoy_atlas.config import load_config
from smn2_rescuediff.decoy_atlas.feature_calculator import (
    dna_to_rna,
    hamming_distance,
    kmer_jaccard_similarity,
    levenshtein_distance,
    sequence_feature_row,
)
from smn2_rescuediff.decoy_atlas.split_strategy import assign_splits
from smn2_rescuediff.decoy_atlas.smn2_control_generator import (
    generate_mutated_controls,
    generate_scrambled_controls,
    preserves_base_composition,
)
from smn2_rescuediff.decoy_atlas.target_loader import read_fasta


FIXTURES = PROJECT / "tests" / "fixtures"


def _write_target_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    fasta = tmp_path / "smn2_target_windows.fasta"
    profile = tmp_path / "smn2_target_profile.tsv"
    metadata = tmp_path / "smn2_target_metadata.json"
    sequence = "AUGGUAAGU"
    fasta.write_text(
        ">SMN2|transcript_id=SMN2_TX1|window=five_prime_splice_site_core|chrom=chrMini|start=195|end=203|strand=+\n"
        f"{sequence}\n",
        encoding="utf-8",
    )
    features = sequence_feature_row(sequence)
    pd.DataFrame(
        [
            {
                "target_id": "SMN2_TEST",
                "gene_symbol": "SMN2",
                "gene_id": "GENE1",
                "transcript_id": "SMN2_TX1",
                "chromosome": "chrMini",
                "start": 195,
                "end": 203,
                "strand": "+",
                "window_name": "five_prime_splice_site_core",
                "window_type": "splice_site",
                "sequence_rna": sequence,
                **features,
                "notes": "fixture",
            }
        ]
    ).to_csv(profile, sep="\t", index=False)
    metadata.write_text(json.dumps({"project": "SMN2-RescueDiff", "module": "target_definition"}), encoding="utf-8")
    return fasta, profile, metadata


def test_config_loads_correctly():
    config = load_config(PROJECT / "configs" / "decoy_atlas.yaml")
    assert config["module_name"] == "HuRDA-180K Human RNA Decoy Atlas"
    assert config["atlas"]["target_total_decoys"] == 180000


def test_smn2_target_fasta_loads_correctly(tmp_path):
    fasta, _, _ = _write_target_files(tmp_path)
    records = read_fasta(fasta)
    assert records.iloc[0]["window_name"] == "five_prime_splice_site_core"
    assert records.iloc[0]["sequence_rna"] == "AUGGUAAGU"


def test_rna_conversion_changes_t_to_u():
    assert dna_to_rna("ATGC", "+") == "AUGC"
    assert dna_to_rna("ATGC", "-") == "GCAU"


def test_gc_content_calculation_is_correct():
    features = sequence_feature_row("AUGC")
    assert features["GC_content"] == pytest.approx(0.5)


def test_kmer_jaccard_similarity_works():
    assert kmer_jaccard_similarity("AUGCU", "AUGAA", 2) > 0
    assert kmer_jaccard_similarity("AAAA", "CCCC", 2) == 0


def test_hamming_distance_equal_length():
    assert hamming_distance("AUGC", "AUGU") == 1


def test_levenshtein_distance_unequal_length():
    assert levenshtein_distance("AUGC", "AUGCUU") == 2


def test_scrambled_controls_preserve_length_and_base_composition(tmp_path):
    fasta, _, _ = _write_target_files(tmp_path)
    targets = read_fasta(fasta)
    controls = generate_scrambled_controls(targets, target_count=3, random_seed=7)
    assert not controls.empty
    original = targets.iloc[0]["sequence_rna"]
    scrambled = controls.iloc[0]["sequence_rna"]
    assert len(scrambled) == len(original)
    assert preserves_base_composition(original, scrambled)


def test_mutated_controls_differ_from_original(tmp_path):
    fasta, _, _ = _write_target_files(tmp_path)
    targets = read_fasta(fasta)
    controls = generate_mutated_controls(targets, target_count=3, random_seed=8)
    assert not controls.empty
    assert controls.iloc[0]["sequence_rna"] != targets.iloc[0]["sequence_rna"]


def test_duplicate_removal_works():
    decoys = pd.DataFrame(
        [
            {"decoy_category": "five_prime_splice_site", "sequence_rna": "AUGC", "smn2_similarity_score": 0.5},
            {"decoy_category": "three_prime_splice_site", "sequence_rna": "AUGC", "smn2_similarity_score": 0.4},
            {"decoy_category": "random_bias_matched", "sequence_rna": "CCCC", "smn2_similarity_score": 0.1},
        ]
    )
    kept, removed = remove_duplicate_and_exact_smn2_matches(decoys, {"UUUU"}, {"atlas": {"remove_duplicate_sequences": True}})
    assert len(kept) == 2
    assert len(removed) == 1


def test_train_validation_test_split_labels_are_created():
    decoys = pd.DataFrame(
        {
            "decoy_category": ["five_prime_splice_site"] * 20,
            "sequence_rna": [f"AUGC{i}" for i in range(20)],
            "chromosome": ["chrMini"] * 20,
            "gene_name": [f"GENE{i}" for i in range(20)],
            "smn2_similarity_score": [0.5] * 20,
        }
    )
    split = assign_splits(decoys, {"splits": {"train_fraction": 0.7, "validation_fraction": 0.1}}, 42)
    assert {"train", "validation", "test"}.issubset(set(split["train_val_test_split"]))
    assert "holdout_group" in split.columns


def test_decoy_atlas_cli_runs_on_tiny_fixture_files(tmp_path):
    fasta, profile, metadata = _write_target_files(tmp_path)
    output_dir = tmp_path / "decoy_out"
    figure_dir = tmp_path / "figures"
    report = tmp_path / "decoy_report.md"
    config = tmp_path / "decoy_atlas.yaml"
    config.write_text(
        f"""
project_name: SMN2-RescueDiff
module_name: HuRDA-180K Human RNA Decoy Atlas
inputs:
  genome_fasta: {FIXTURES / "mini_genome.fa"}
  annotation_gtf: {FIXTURES / "mini_annotation.gtf"}
  smn2_target_fasta: {fasta}
  smn2_target_profile: {profile}
  smn2_target_metadata: {metadata}
outputs:
  output_dir: {output_dir}
  report_path: {report}
  figure_dir: {figure_dir}
atlas:
  target_total_decoys: 20
  max_total_decoys: 50
  random_seed: 11
  remove_duplicate_sequences: true
  remove_exact_smn2_matches: true
windows:
  five_prime_splice_site:
    core_exon_nt: 3
    core_intron_nt: 6
    local_window_nt: 21
    extended_window_nt: 31
  three_prime_splice_site:
    intron_nt: 6
    exon_nt: 3
    local_window_nt: 21
    extended_window_nt: 31
  regulatory_motif_window_nt: 9
  random_window_nt: 9
decoy_targets:
  five_prime_splice_site_decoys: 4
  three_prime_splice_site_decoys: 4
  smn2_like_hard_decoys: 3
  exonic_regulatory_decoys: 3
  intronic_regulatory_decoys: 3
  scrambled_smn2_controls: 2
  mutated_smn2_controls: 2
  random_bias_matched_windows: 3
feature_matching:
  gc_tolerance: 0.5
  length_must_match: true
  min_smn2_similarity_for_hard_decoys: 0.0
  max_smn2_similarity_for_hard_decoys: 1.0
  kmer_size: 3
splits:
  train_fraction: 0.70
  validation_fraction: 0.10
  test_fraction: 0.20
  create_chromosome_holdout: true
  create_hard_decoy_holdout: true
  create_gene_holdout: true
biological_weights:
  five_prime_splice_site: 1.0
  smn2_like_hard_decoy: 1.0
  three_prime_splice_site: 0.8
  exonic_regulatory: 0.7
  intronic_regulatory: 0.6
  random_bias_matched: 0.3
  scrambled_smn2_control: 0.1
  mutated_smn2_control: 0.2
""",
        encoding="utf-8",
    )
    assert run(config) == 0
    assert (output_dir / "hurda_180k_full.tsv").exists()
    assert (output_dir / "hurda_180k_full.fasta").exists()
    assert (output_dir / "hurda_summary.json").exists()
    assert (output_dir / "hurda_removed_duplicates.tsv").exists()
    assert report.exists()
    assert (figure_dir / "decoy_category_counts.png").exists()
    atlas = pd.read_csv(output_dir / "hurda_180k_full.tsv", sep="\t")
    assert "decoy_id" in atlas.columns
    assert {"train", "validation", "test"} & set(atlas["train_val_test_split"])
