from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from smn2_rescuediff.target_definition.annotation_parser import read_exons, select_target
from smn2_rescuediff.target_definition.cli import run
from smn2_rescuediff.target_definition.feature_calculator import calculate_features, u1_complementarity_score
from smn2_rescuediff.target_definition.sequence_extractor import TargetWindow, build_windows, dna_to_rna


FIXTURES = PROJECT / "tests" / "fixtures"


def test_gtf_parser_reads_core_fields():
    exons = read_exons(FIXTURES / "mini_annotation.gtf")
    smn2 = exons[exons["gene_symbol"] == "SMN2"]
    assert smn2["transcript_id"].iloc[0] == "SMN2_TX1"
    assert smn2[smn2["exon_number"] == 7]["start"].iloc[0] == 150


def test_exon_ordering_positive_and_negative_strands():
    exons = read_exons(FIXTURES / "mini_annotation.gtf")
    neg = exons[exons["gene_symbol"] == "NEG"].sort_values("exon_number")
    assert neg.iloc[0]["start"] == 270
    assert neg.iloc[1]["start"] == 250


def test_sequence_extraction_reverse_complement_and_rna_conversion():
    assert dna_to_rna("ATGC", "+") == "AUGC"
    assert dna_to_rna("ATGC", "-") == "GCAU"


def test_build_windows_uses_negative_strand_orientation():
    exons = read_exons(FIXTURES / "mini_annotation.gtf")
    target, _ = select_target(
        exons,
        {
            "target": {
                "gene_symbol": "NEG",
                "transcript_id": "NEG_TX1",
                "exon_number": 1,
            },
            "windows": {},
        },
    )
    windows = build_windows(FIXTURES / "mini_genome.fa", target, {"windows": {}})
    exon = next(window for window in windows if window.window_name == "exon7")
    assert exon.sequence_rna == dna_to_rna("CGTACGTACGT", "-")


def test_feature_calculator_gc_content_and_u1_range():
    target_window = TargetWindow(
        target_id="x",
        window_name="test",
        window_type="unit",
        chromosome="chrMini",
        start=1,
        end=4,
        strand="+",
        sequence_rna="AUGC",
        notes="unit test",
    )
    target = type(
        "Target",
        (),
        {
            "gene_symbol": "SMN2",
            "gene_id": "GENE1",
            "transcript_id": "TX",
            "start": 1,
            "end": 4,
            "strand": "+",
        },
    )()
    features = calculate_features([target_window], target, "ACUUACCU")
    assert features["GC_content"].iloc[0] == pytest.approx(0.5)
    score = u1_complementarity_score("UGAAUGGA", "ACUUACCU")
    assert 0 <= score <= 1


def test_cli_runs_on_fixture_data(tmp_path):
    config_path = tmp_path / "target_definition.yaml"
    output_dir = tmp_path / "out"
    report = tmp_path / "report.md"
    figure_dir = tmp_path / "figures"
    structures = tmp_path / "structures"
    config_path.write_text(
        f"""
project_name: SMN2-RescueDiff
inputs:
  genome_fasta: {FIXTURES / "mini_genome.fa"}
  annotation_gtf: {FIXTURES / "mini_annotation.gtf"}
target:
  gene_symbol: SMN2
  transcript_id: SMN2_TX1
  exon_number: 7
windows:
  core_window_nt: 21
  local_window_nt: 51
  extended_window_nt: 101
  splice_site_core_5prime:
    exon_nt: 3
    intron_nt: 6
  splice_site_core_3prime:
    intron_nt: 20
    exon_nt: 3
u1_snRNA:
  canonical_5prime_u1_seed: "ACUUACCU"
structures:
  fetch_pdb: false
  pdb_ids:
    - "6HMI"
  output_dir: {structures}
outputs:
  output_dir: {output_dir}
  report_path: {report}
  figure_dir: {figure_dir}
""",
        encoding="utf-8",
    )
    assert run(config_path) == 0
    assert (output_dir / "smn2_target_windows.fasta").exists()
    assert (output_dir / "smn2_target_profile.tsv").exists()
    assert (output_dir / "smn2_target_metadata.json").exists()
    assert (output_dir / "structure_metadata.tsv").exists()
    assert report.exists()
    assert (figure_dir / "smn2_target_feature_profile.png").exists()

    profile = pd.read_csv(output_dir / "smn2_target_profile.tsv", sep="\t")
    assert set(["exon7", "five_prime_splice_site_core"]).issubset(set(profile["window_name"]))
