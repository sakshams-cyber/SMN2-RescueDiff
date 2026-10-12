from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_stage_c5_top40_sanity_and_matched_controls.py"
SPEC = importlib.util.spec_from_file_location("stage_c5", SCRIPT_PATH)
stage_c5 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = stage_c5
SPEC.loader.exec_module(stage_c5)


def test_conservative_gap_driver_counts_are_explicit() -> None:
    top40 = pd.DataFrame(
        {
            "molecule_id": ["cand_single", "cand_few"],
            "partial_poc_selectivity_rank": [1, 2],
            "weighted_mean_selectivity_gap": [2.0, 2.0],
            "conservative_selectivity_gap": [-1.0, -2.0],
        }
    )
    target_summary = pd.DataFrame(
        {
            "molecule_id": ["cand_single", "cand_few"],
            "num_conformers_completed": [5, 5],
            "mean_target_deltaG": [-8.0, -8.0],
            "median_target_deltaG": [-8.0, -8.0],
            "best_target_deltaG": [-8.4, -8.4],
            "worst_target_deltaG": [-7.5, -7.5],
            "std_target_deltaG": [0.2, 0.2],
            "target_deltaG_range": [0.9, 0.9],
            "target_conformer_rank": [1, 2],
        }
    )
    decoy_valid = pd.DataFrame(
        {
            "molecule_id": ["cand_single"] * 3 + ["cand_few"] * 3,
            "decoy_id": ["d1", "d2", "d3", "d1", "d2", "d3"],
            "decoy_deltaG_numeric": [-8.2, -2.0, -1.0, -8.6, -8.2, -8.1],
            "decoy_deltaG": [-8.2, -2.0, -1.0, -8.6, -8.2, -8.1],
            "decoy_category": ["hard"] * 6,
            "motif_type": ["motif"] * 6,
            "decoy_biological_weight": [1.0] * 6,
            "decoy_smn2_similarity_score": [0.8] * 6,
            "decoy_receptor_file": ["r.pdbqt"] * 6,
        }
    )
    inputs = SimpleNamespace(top40=top40, target_summary=target_summary, decoy_valid=decoy_valid)

    audit = stage_c5.build_top40_hurda_audit(inputs)

    by_id = audit.set_index("molecule_id")
    assert by_id.loc["cand_single", "conservative_gap_driver"] == "SINGLE_DECOY_DRIVEN"
    assert by_id.loc["cand_single", "num_decoys_stronger_than_worst_target"] == 1
    assert by_id.loc["cand_few", "conservative_gap_driver"] == "FEW_DECOY_DRIVEN"
    assert by_id.loc["cand_few", "num_decoys_stronger_than_worst_target"] == 3


def test_control_matching_excludes_candidates_and_reports_reuse() -> None:
    rows = [
        ("cand", "CCOc1ccccc1", 300, 2.0, 1, 3, 60, 3, 0, 0.8, 0.7, 1),
        ("ctrl_a", "CCNc1ccccc1", 302, 2.1, 1, 3, 61, 3, 0, 0.8, 0.7, 2),
        ("ctrl_b", "CCOc1ccncc1", 298, 2.0, 1, 3, 58, 3, 0, 0.8, 0.7, 3),
        ("ctrl_c", "CCOc1ccc(F)cc1", 304, 2.2, 1, 3, 59, 3, 0, 0.8, 0.7, 4),
    ]
    columns = [
        "molecule_id",
        "canonical_smiles",
        "molecular_weight",
        "logp",
        "hbd",
        "hba",
        "tpsa",
        "rotatable_bonds",
        "formal_charge",
        "qed",
        "synthetic_accessibility_proxy",
        "partial_poc_selectivity_rank",
    ]
    population = pd.DataFrame(rows, columns=columns)
    population, _ = stage_c5.zscore_population(population, stage_c5.MATCHING_DESCRIPTORS)
    candidates = population[population["molecule_id"].eq("cand")].copy()
    controls = population[~population["molecule_id"].eq("cand")].copy()
    target_summary = pd.DataFrame(
        {
            "molecule_id": population["molecule_id"],
            "num_conformers_completed": [5, 5, 5, 5],
            "mean_target_deltaG": [-8.0, -7.5, -7.4, -7.3],
            "median_target_deltaG": [-8.0, -7.5, -7.4, -7.3],
            "best_target_deltaG": [-8.3, -7.8, -7.7, -7.6],
            "worst_target_deltaG": [-7.7, -7.2, -7.1, -7.0],
            "std_target_deltaG": [0.2, 0.2, 0.2, 0.2],
            "target_deltaG_range": [0.6, 0.6, 0.6, 0.6],
            "target_conformer_rank": [1, 2, 3, 4],
        }
    )
    fps = stage_c5.fingerprint_map(population)

    assignments, metadata = stage_c5.match_controls(
        candidates=candidates,
        controls=controls,
        target_summary=target_summary,
        descriptors=stage_c5.MATCHING_DESCRIPTORS,
        fps=fps,
        tanimoto_threshold=1.0,
        controls_per_candidate=2,
    )

    assert len(assignments) == 2
    assert "cand" not in set(assignments["control_id"])
    assert assignments["candidate_control_tanimoto"].notna().all()
    assert assignments["matching_distance"].notna().all()
    assert metadata["number_control_assignments"] == 2
