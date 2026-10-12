"""Input loading utilities for Module 4."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def load_molecules(path: str | Path) -> pd.DataFrame:
    """Load Module 3 filtered molecules."""
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )
    required = {"molecule_id", "canonical_smiles"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Molecule input is missing required columns: {sorted(missing)}")
    if "filter_pass" in df.columns:
        mask = df["filter_pass"].astype(str).str.lower().isin(["true", "1", "yes"])
        if mask.any():
            df = df[mask].copy()
    df = df[df["canonical_smiles"].notna() & (df["canonical_smiles"].astype(str) != "NA")].copy()
    if df.empty:
        raise ValueError("No molecules are available for docking after filtering.")
    return df.reset_index(drop=True)


def load_decoy_atlas(path: str | Path) -> pd.DataFrame:
    """Load HuRDA atlas table."""
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )
    required = {"decoy_id", "decoy_category", "sequence_rna"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"HuRDA decoy atlas is missing required columns: {sorted(missing)}")
    return df


def load_json_if_exists(path: str | Path) -> dict[str, Any]:
    """Load a JSON file if present."""
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def read_fasta_records(path: str | Path) -> pd.DataFrame:
    """Read a small FASTA into a DataFrame."""
    rows: list[dict[str, str]] = []
    header = None
    chunks: list[str] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            if text.startswith(">"):
                if header is not None:
                    rows.append({"header": header, "sequence": "".join(chunks)})
                header = text[1:]
                chunks = []
            else:
                chunks.append(text)
    if header is not None:
        rows.append({"header": header, "sequence": "".join(chunks)})
    return pd.DataFrame(rows)

