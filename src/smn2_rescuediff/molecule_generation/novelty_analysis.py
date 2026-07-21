"""Novelty analysis against canonical training SMILES."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .chemical_filter import canonicalize_training_smiles, require_rdkit


def _fingerprint(smiles: str):
    Chem, *_ = require_rdkit()
    try:
        from rdkit.Chem import rdFingerprintGenerator

        mol = Chem.MolFromSmiles(smiles)
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        return generator.GetFingerprint(mol)
    except Exception:
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles(smiles)
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)


def add_novelty_scores(molecules: pd.DataFrame, training_smiles: list[str], config: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    """Annotate novelty and max Tanimoto similarity to the training set."""
    if molecules.empty:
        return molecules.copy(), []
    analysis = config.get("analysis", {})
    if not bool(analysis.get("calculate_novelty_against_training", True)):
        result = molecules.copy()
        result["is_novel_vs_training"] = "NA"
        result["max_tanimoto_to_training"] = "NA"
        return result, ["Novelty calculation disabled in config."]

    warnings: list[str] = []
    threshold = float(analysis.get("tanimoto_similarity_threshold_duplicate_like", 0.95))
    max_training_fps = int(analysis.get("max_training_fingerprints_for_novelty", 10000))
    canonical_training = canonicalize_training_smiles(training_smiles, remove_salts=bool(config.get("chemical_filters", {}).get("remove_salts", True)))
    training_set = set(canonical_training)
    if len(canonical_training) > max_training_fps:
        canonical_training = canonical_training[:max_training_fps]
        warnings.append(f"Novelty fingerprints limited to first {max_training_fps:,} canonical training molecules for runtime control.")
    training_fps = [_fingerprint(smiles) for smiles in canonical_training]
    from rdkit import DataStructs

    result = molecules.copy()
    novel_flags = []
    max_sims = []
    for smiles in result["canonical_smiles"]:
        fp = _fingerprint(str(smiles))
        similarities = DataStructs.BulkTanimotoSimilarity(fp, training_fps) if training_fps else []
        max_sim = max(similarities) if similarities else 0.0
        max_sims.append(round(float(max_sim), 6))
        novel_flags.append(str(smiles) not in training_set and max_sim < threshold)
    result["is_novel_vs_training"] = novel_flags
    result["max_tanimoto_to_training"] = max_sims
    return result, warnings

