"""Deterministic dry-run docking simulator."""

from __future__ import annotations

import hashlib
import math
import random
from typing import Any

import pandas as pd

from .docking_engine import DockingResult


DRY_RUN_NOTE = "Simulated score for pipeline testing only."


CATEGORY_CENTERS = {
    "smn2_like_hard_decoy": -6.2,
    "five_prime_splice_site": -5.9,
    "three_prime_splice_site": -5.5,
    "exonic_regulatory": -5.0,
    "intronic_regulatory": -4.9,
    "scrambled_smn2_control": -4.2,
    "mutated_smn2_control": -4.5,
    "random_bias_matched": -4.4,
}


def _stable_rng(seed: int, *parts: str) -> random.Random:
    key = "|".join([str(seed), *parts]).encode("utf-8")
    value = int(hashlib.sha256(key).hexdigest()[:16], 16)
    return random.Random(value)


def _quality_adjustment(molecule: pd.Series) -> float:
    qed = float(pd.to_numeric(pd.Series([molecule.get("qed", 0.5)]), errors="coerce").fillna(0.5).iloc[0])
    mw = float(pd.to_numeric(pd.Series([molecule.get("molecular_weight", 350)]), errors="coerce").fillna(350).iloc[0])
    logp = float(pd.to_numeric(pd.Series([molecule.get("logp", 2.5)]), errors="coerce").fillna(2.5).iloc[0])
    mw_bonus = max(0.0, 1.0 - abs(mw - 350.0) / 350.0)
    logp_bonus = max(0.0, 1.0 - abs(logp - 2.5) / 5.0)
    return -0.65 * qed - 0.25 * mw_bonus - 0.20 * logp_bonus


def dry_run_target_score(molecule: pd.Series, seed: int) -> DockingResult:
    """Generate deterministic simulated SMN2 target docking score."""
    rng = _stable_rng(seed, str(molecule["molecule_id"]), "SMN2")
    score = -6.75 + _quality_adjustment(molecule) + rng.gauss(0, 0.45)
    return DockingResult("completed", round(score, 3), 1, 1, "NA", "dry_run_simulator", "dry_run", True, DRY_RUN_NOTE)


def dry_run_decoy_score(molecule: pd.Series, decoy: pd.Series, seed: int) -> DockingResult:
    """Generate deterministic simulated decoy docking score."""
    category = str(decoy.get("decoy_category", "random_bias_matched"))
    center = CATEGORY_CENTERS.get(category, -4.5)
    similarity = float(pd.to_numeric(pd.Series([decoy.get("smn2_similarity_score", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
    weight = float(pd.to_numeric(pd.Series([decoy.get("biological_weight", 0.3)]), errors="coerce").fillna(0.3).iloc[0])
    rng = _stable_rng(seed, str(molecule["molecule_id"]), str(decoy.get("decoy_id", "")), category)
    score = center + _quality_adjustment(molecule) * 0.45 - 0.75 * similarity - 0.25 * math.sqrt(max(weight, 0.0)) + rng.gauss(0, 0.55)
    return DockingResult("completed", round(score, 3), 1, 1, "NA", "dry_run_simulator", "dry_run", True, DRY_RUN_NOTE)

