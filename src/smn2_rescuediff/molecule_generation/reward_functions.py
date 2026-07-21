"""Pre-docking chemical plausibility and RNA-ligand-like reward functions."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd


def _between(value: float, low: float, high: float) -> float:
    if low <= value <= high:
        return 1.0
    distance = min(abs(value - low), abs(value - high))
    width = max(high - low, 1.0)
    return max(0.0, 1.0 - distance / width)


def rna_ligand_like_score(row: pd.Series) -> float:
    """Score pre-docking chemical features associated with RNA-ligand-like candidates."""
    mw = float(pd.to_numeric(pd.Series([row.get("molecular_weight", 0)]), errors="coerce").fillna(0).iloc[0])
    logp = float(pd.to_numeric(pd.Series([row.get("logp", 0)]), errors="coerce").fillna(0).iloc[0])
    hbd = float(pd.to_numeric(pd.Series([row.get("hbd", 0)]), errors="coerce").fillna(0).iloc[0])
    hba = float(pd.to_numeric(pd.Series([row.get("hba", 0)]), errors="coerce").fillna(0).iloc[0])
    aromatic = float(pd.to_numeric(pd.Series([row.get("num_aromatic_rings", 0)]), errors="coerce").fillna(0).iloc[0])
    charge = abs(float(pd.to_numeric(pd.Series([row.get("formal_charge", 0)]), errors="coerce").fillna(0).iloc[0]))
    smiles = str(row.get("canonical_smiles", ""))
    hetero_bonus = 1.0 if any(atom in smiles for atom in ["N", "O", "S", "P"]) else 0.0
    nitrogen_heterocycle = 1.0 if "n" in smiles or "N" in smiles and "1" in smiles else 0.0
    score = (
        0.20 * _between(mw, 220, 520)
        + 0.18 * _between(logp, 0.0, 4.5)
        + 0.16 * _between(hbd, 0, 5)
        + 0.16 * _between(hba, 2, 10)
        + 0.12 * _between(aromatic, 1, 4)
        + 0.10 * hetero_bonus
        + 0.08 * nitrogen_heterocycle
    )
    score *= max(0.0, 1.0 - 0.15 * max(0.0, charge - 1.0))
    return round(max(0.0, min(1.0, score)), 6)


def chemical_plausibility_reward(row: pd.Series, weights: dict[str, float] | None = None) -> float:
    """Reward computationally plausible, diverse, RNA-ligand-like candidates without docking claims."""
    weights = weights or {}
    reward = 0.0
    reward += float(weights.get("validity", 2.0)) if bool(row.get("is_valid", False)) else -2.0
    reward += float(weights.get("uniqueness", 1.0)) if not bool(row.get("is_duplicate", False)) else float(weights.get("duplicate_penalty", -1.0))
    reward += float(weights.get("novelty", 1.0)) if bool(row.get("is_novel_vs_training", False)) else 0.0
    reward += float(weights.get("qed", 1.5)) * float(pd.to_numeric(pd.Series([row.get("qed", 0)]), errors="coerce").fillna(0).iloc[0])
    reward += float(weights.get("lipinski", 1.0)) if bool(row.get("lipinski_pass", False)) else 0.0
    reward += float(weights.get("molecular_weight_range", 0.8)) * _between(float(row.get("molecular_weight", 0)), 150, 650)
    reward += float(weights.get("logp_range", 0.8)) * _between(float(row.get("logp", 0)), -2, 6)
    reward += float(weights.get("hbond_balance", 0.6)) * _between(float(row.get("hbd", 0)) + float(row.get("hba", 0)), 2, 14)
    reward += float(weights.get("aromatic_ring_balance", 0.5)) * _between(float(row.get("num_aromatic_rings", 0)), 0, 4)
    reward += float(weights.get("heteroatom_presence", 0.7)) * (1.0 if any(atom in str(row.get("canonical_smiles", "")) for atom in ["N", "O", "S", "P"]) else 0.0)
    reward += float(weights.get("nitrogen_heterocycle_bonus", 0.8)) * (1.0 if "n" in str(row.get("canonical_smiles", "")) else 0.0)
    if bool(row.get("pains_alert", False)):
        reward += float(weights.get("pains_penalty", -2.0))
    if abs(float(row.get("formal_charge", 0))) > 2:
        reward += float(weights.get("excessive_charge_penalty", -1.0))
    if float(row.get("molecular_weight", 0)) > 650:
        reward += float(weights.get("large_molecule_penalty", -1.0))
    return float(reward) if math.isfinite(float(reward)) else -10.0

