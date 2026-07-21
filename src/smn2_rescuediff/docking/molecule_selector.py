"""Molecule subset selection for docking runs."""

from __future__ import annotations

import pandas as pd


def select_molecules_for_docking(molecules: pd.DataFrame, max_molecules: int, random_seed: int) -> pd.DataFrame:
    """Select a deterministic, manageable molecule subset."""
    if max_molecules <= 0:
        raise ValueError("run.max_molecules_to_dock must be positive.")
    work = molecules.copy()
    sort_columns = [col for col in ["qed", "molecular_weight", "molecule_id"] if col in work.columns]
    if "qed" in sort_columns:
        work["_qed_sort"] = pd.to_numeric(work["qed"], errors="coerce").fillna(-1.0)
        work = work.sort_values(["_qed_sort", "molecule_id"], ascending=[False, True])
        work = work.drop(columns=["_qed_sort"])
    elif sort_columns:
        work = work.sort_values(sort_columns)
    else:
        work = work.sample(frac=1.0, random_state=random_seed)
    selected = work.head(max_molecules).copy().reset_index(drop=True)
    if selected.empty:
        raise ValueError("No molecules selected for docking.")
    return selected

