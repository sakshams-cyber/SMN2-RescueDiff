"""Training SMILES loading and cleaning."""

from __future__ import annotations

from pathlib import Path


TOY_SMILES = [
    "CCO",
    "CCN",
    "CC(=O)O",
    "c1ccccc1",
    "CCOC(=O)C",
    "CC(C)O",
    "CCS",
    "CCN(CC)CC",
    "O=C(O)c1ccccc1",
    "CC(C)N",
    "COc1ccccc1",
    "CCOc1ccccc1",
    "CC(=O)Oc1ccccc1C(=O)O",
    "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "COc1ccc2[nH]cc(CCN)c2c1",
]

TOY_WARNING = (
    "This toy SMILES file is only for testing the pipeline and should not be used for real scientific results."
)


def create_example_training_smiles(raw_molecule_dir: str | Path) -> Path:
    """Create a tiny toy SMILES file when the configured training set is missing."""
    directory = Path(raw_molecule_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "example_training_smiles.smi"
    if not path.exists():
        path.write_text("\n".join(TOY_SMILES) + "\n", encoding="utf-8")
    return path


def load_training_smiles(path: str | Path, raw_molecule_dir: str | Path) -> tuple[list[str], Path, list[str]]:
    """Load training SMILES, creating a toy example if the configured file is missing."""
    warnings: list[str] = []
    smiles_path = Path(path)
    if not smiles_path.exists():
        smiles_path = create_example_training_smiles(raw_molecule_dir)
        warnings.append(TOY_WARNING)
    rows = []
    with smiles_path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            rows.append(line.split()[0])
    cleaned = clean_smiles(rows)
    if not cleaned:
        raise ValueError(f"No non-empty training SMILES were found in {smiles_path}")
    return cleaned, smiles_path, warnings


def clean_smiles(smiles: list[str], max_length: int | None = None) -> list[str]:
    """Basic text cleaning before RDKit validation."""
    seen: set[str] = set()
    cleaned: list[str] = []
    for item in smiles:
        value = item.strip()
        if not value or value.startswith("#"):
            continue
        if max_length is not None and len(value) > max_length:
            continue
        if value not in seen:
            seen.add(value)
            cleaned.append(value)
    return cleaned


def write_smiles(smiles: list[str], path: str | Path) -> None:
    """Write one SMILES string per line."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(smiles) + ("\n" if smiles else ""), encoding="utf-8")
