"""SMILES text cleaning helpers for large molecule datasets."""

from __future__ import annotations


def iter_smiles_lines(lines: list[str]) -> list[str]:
    """Extract the first whitespace-delimited SMILES token from input lines."""
    cleaned: list[str] = []
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        cleaned.append(line.split()[0])
    return cleaned


def clean_smiles_strings(smiles: list[str], max_length: int | None = None) -> list[str]:
    """Remove blanks, comments, duplicates, and optionally length outliers."""
    seen: set[str] = set()
    cleaned: list[str] = []
    for item in smiles:
        value = str(item).strip()
        if not value or value.startswith("#"):
            continue
        value = value.split()[0]
        if max_length is not None and len(value) > max_length:
            continue
        if value in seen:
            continue
        seen.add(value)
        cleaned.append(value)
    return cleaned

