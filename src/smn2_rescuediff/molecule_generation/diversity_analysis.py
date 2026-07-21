"""Diversity analysis for generated candidate molecules."""

from __future__ import annotations

import random
from typing import Any

import pandas as pd

from .chemical_filter import require_rdkit
from .novelty_analysis import _fingerprint


def assign_diversity_clusters(molecules: pd.DataFrame) -> pd.DataFrame:
    """Assign fast scaffold-based diversity cluster IDs."""
    if molecules.empty:
        return molecules.copy()
    Chem, *_ = require_rdkit()
    from rdkit.Chem.Scaffolds import MurckoScaffold

    result = molecules.copy()
    cluster_map: dict[str, int] = {}
    cluster_ids: list[str] = []
    for smiles in result["canonical_smiles"]:
        mol = Chem.MolFromSmiles(str(smiles))
        scaffold = ""
        if mol is not None:
            try:
                scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
            except Exception:
                scaffold = ""
        key = scaffold or str(smiles)
        if key not in cluster_map:
            cluster_map[key] = len(cluster_map) + 1
        cluster_ids.append(f"cluster_{cluster_map[key]:05d}")
    result["diversity_cluster_id"] = cluster_ids
    return result


def pairwise_tanimoto_sample(molecules: pd.DataFrame, config: dict[str, Any]) -> list[float]:
    """Calculate a runtime-capped sample of pairwise Tanimoto similarities."""
    if molecules.empty or len(molecules) < 2 or not bool(config.get("analysis", {}).get("calculate_diversity", True)):
        return []
    from rdkit import DataStructs

    seed = int(config.get("run", {}).get("random_seed", config.get("generation", {}).get("random_seed", 42)))
    max_sample = int(config.get("analysis", {}).get("pairwise_diversity_sample_size", config.get("analysis", {}).get("max_pairwise_diversity_sample", 2000)))
    rng = random.Random(seed)
    smiles = molecules["canonical_smiles"].dropna().astype(str).tolist()
    if len(smiles) > max_sample:
        smiles = rng.sample(smiles, max_sample)
    fps = [_fingerprint(item) for item in smiles]
    similarities: list[float] = []
    max_pairs = min(50000, len(fps) * (len(fps) - 1) // 2)
    attempts = 0
    while len(similarities) < max_pairs and attempts < max_pairs * 3:
        attempts += 1
        i = rng.randrange(len(fps))
        j = rng.randrange(len(fps))
        if i == j:
            continue
        similarities.append(float(DataStructs.TanimotoSimilarity(fps[i], fps[j])))
    return similarities
