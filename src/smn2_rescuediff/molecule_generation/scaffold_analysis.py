"""Bemis-Murcko scaffold analysis."""

from __future__ import annotations

import pandas as pd

from .chemical_filter import require_rdkit


def add_scaffolds(molecules: pd.DataFrame) -> pd.DataFrame:
    """Add Bemis-Murcko scaffolds when RDKit can calculate them."""
    if molecules.empty:
        return molecules.copy()
    Chem, *_ = require_rdkit()
    from rdkit.Chem.Scaffolds import MurckoScaffold

    result = molecules.copy()
    scaffolds: list[str] = []
    for smiles in result["canonical_smiles"]:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            scaffolds.append("NA")
            continue
        try:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        except Exception:
            scaffold = ""
        scaffolds.append(scaffold or "acyclic_or_no_scaffold")
    result["bemis_murcko_scaffold"] = scaffolds
    return result


def scaffold_frequency(molecules: pd.DataFrame) -> pd.DataFrame:
    """Return scaffold frequency table."""
    if molecules.empty or "bemis_murcko_scaffold" not in molecules:
        return pd.DataFrame(columns=["bemis_murcko_scaffold", "count"])
    return molecules["bemis_murcko_scaffold"].value_counts().rename_axis("bemis_murcko_scaffold").reset_index(name="count")

