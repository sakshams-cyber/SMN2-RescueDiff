"""RDKit validation, canonicalization, property calculation, and filtering."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def rdkit_available() -> bool:
    """Return whether RDKit can be imported."""
    try:
        import rdkit  # noqa: F401
        return True
    except Exception:
        return False


def require_rdkit():
    """Import RDKit modules or raise a clear installation message."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdMolDescriptors
        from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
        from rdkit.Chem.SaltRemover import SaltRemover

        return Chem, Crippen, Descriptors, Lipinski, QED, rdMolDescriptors, FilterCatalog, FilterCatalogParams, SaltRemover
    except Exception as exc:
        raise ImportError(
            "RDKit is required for Module 3 chemical validation and filtering. "
            "Install it with `pip install rdkit` in the project environment, then rerun the module."
        ) from exc


def _pains_catalog():
    Chem, Crippen, Descriptors, Lipinski, QED, rdMolDescriptors, FilterCatalog, FilterCatalogParams, SaltRemover = require_rdkit()
    params = FilterCatalogParams()
    for catalog_name in [
        FilterCatalogParams.FilterCatalogs.PAINS_A,
        FilterCatalogParams.FilterCatalogs.PAINS_B,
        FilterCatalogParams.FilterCatalogs.PAINS_C,
    ]:
        try:
            params.AddCatalog(catalog_name)
        except Exception:
            continue
    return FilterCatalog(params)


def mol_from_smiles(smiles: str):
    """Parse a SMILES string with RDKit."""
    Chem, *_ = require_rdkit()
    if not smiles or not str(smiles).strip():
        return None
    return Chem.MolFromSmiles(str(smiles).strip())


def canonicalize_smiles(smiles: str, remove_salts: bool = True) -> tuple[str | None, object | None]:
    """Parse, optionally strip salts/fragments, and return canonical SMILES."""
    Chem, Crippen, Descriptors, Lipinski, QED, rdMolDescriptors, FilterCatalog, FilterCatalogParams, SaltRemover = require_rdkit()
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None, None
    if remove_salts:
        try:
            remover = SaltRemover()
            mol = remover.StripMol(mol, dontRemoveEverything=True)
        except Exception:
            pass
    if mol is None or mol.GetNumAtoms() == 0:
        return None, None
    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    return canonical, mol


def lipinski_pass(mw: float, logp: float, hbd: int, hba: int) -> bool:
    """Basic Lipinski rule-of-five pass/fail."""
    failures = int(mw > 500) + int(logp > 5) + int(hbd > 5) + int(hba > 10)
    return failures <= 1


def calculate_properties(mol: object, calculate_qed: bool = True, pains_catalog: object | None = None) -> dict[str, Any]:
    """Calculate RDKit chemical properties for a molecule."""
    Chem, Crippen, Descriptors, Lipinski, QED, rdMolDescriptors, *_ = require_rdkit()
    mw = float(Descriptors.MolWt(mol))
    logp = float(Crippen.MolLogP(mol))
    hbd = int(Lipinski.NumHDonors(mol))
    hba = int(Lipinski.NumHAcceptors(mol))
    rotatable = int(Lipinski.NumRotatableBonds(mol))
    tpsa = float(rdMolDescriptors.CalcTPSA(mol))
    formal_charge = int(Chem.GetFormalCharge(mol))
    num_rings = int(mol.GetRingInfo().NumRings())
    aromatic_rings = int(rdMolDescriptors.CalcNumAromaticRings(mol))
    qed_value: float | str = float(QED.qed(mol)) if calculate_qed else "NA"
    pains_alert = False
    if pains_catalog is not None:
        try:
            pains_alert = bool(pains_catalog.HasMatch(mol))
        except Exception:
            pains_alert = False
    return {
        "molecular_weight": mw,
        "logp": logp,
        "hbd": hbd,
        "hba": hba,
        "rotatable_bonds": rotatable,
        "tpsa": tpsa,
        "formal_charge": formal_charge,
        "num_rings": num_rings,
        "num_aromatic_rings": aromatic_rings,
        "qed": qed_value,
        "sa_score": "NA",
        "lipinski_pass": lipinski_pass(mw, logp, hbd, hba),
        "pains_alert": pains_alert,
        "pains_flag": pains_alert,
        "synthetic_accessibility_proxy": round(
            max(0.0, min(1.0, 1.0 - 0.0012 * max(0.0, mw - 250.0) - 0.035 * rotatable - 0.04 * max(0, num_rings - 3))),
            6,
        ),
    }


def _filter_reasons(row: dict[str, Any], filters: dict[str, Any]) -> list[str]:
    reasons = []
    if row["molecular_weight"] < float(filters.get("min_molecular_weight", 150)):
        reasons.append("molecular_weight_below_min")
    if row["molecular_weight"] > float(filters.get("max_molecular_weight", 650)):
        reasons.append("molecular_weight_above_max")
    if row["logp"] < float(filters.get("min_logp", -2.0)):
        reasons.append("logp_below_min")
    if row["logp"] > float(filters.get("max_logp", 6.0)):
        reasons.append("logp_above_max")
    if row["hbd"] > int(filters.get("max_hbd", 8)):
        reasons.append("hbd_above_max")
    if row["hba"] > int(filters.get("max_hba", 15)):
        reasons.append("hba_above_max")
    if row["rotatable_bonds"] > int(filters.get("max_rotatable_bonds", 12)):
        reasons.append("rotatable_bonds_above_max")
    if abs(row["formal_charge"]) > int(filters.get("max_formal_charge_abs", 999)):
        reasons.append("formal_charge_abs_above_max")
    if bool(filters.get("require_lipinski_pass", False)) and not bool(row["lipinski_pass"]):
        reasons.append("lipinski_rule_fail")
    if bool(filters.get("remove_pains", True)) and bool(row["pains_alert"]):
        reasons.append("pains_alert")
    return reasons


def process_generated_smiles(generated: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Validate, canonicalize, deduplicate, and chemically filter generated SMILES."""
    filters = config.get("chemical_filters", {})
    analysis = config.get("analysis", {})
    remove_salts = bool(filters.get("remove_salts", True))
    remove_duplicates = bool(filters.get("remove_duplicates", True))
    calculate_qed = bool(analysis.get("calculate_qed", True))
    pains_catalog = _pains_catalog() if bool(filters.get("remove_pains", True)) else None
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for idx, row in generated.reset_index(drop=True).iterrows():
        decoded = str(row.get("decoded_smiles", row.get("raw_smiles", ""))).strip()
        raw_generated = str(row.get("raw_generated_string", row.get("raw_smiles", decoded))).strip()
        representation = str(row.get("representation", "smiles"))
        base = {
            "molecule_id": f"MolGen_{idx + 1:06d}",
            "source_generator": row.get("source_generator", "NA"),
            "representation": representation,
            "raw_generated_string": raw_generated,
            "decoded_smiles": decoded,
            "raw_smiles": decoded,
            "canonical_smiles": "NA",
            "valid_smiles": False,
            "is_valid": False,
            "is_duplicate": False,
            "is_novel_vs_training": "NA",
            "molecular_weight": "NA",
            "logp": "NA",
            "hbd": "NA",
            "hba": "NA",
            "rotatable_bonds": "NA",
            "tpsa": "NA",
            "formal_charge": "NA",
            "num_rings": "NA",
            "num_aromatic_rings": "NA",
            "qed": "NA",
            "sa_score": "NA",
            "lipinski_pass": "NA",
            "pains_alert": "NA",
            "pains_flag": "NA",
            "synthetic_accessibility_proxy": "NA",
            "filter_pass": False,
            "rejection_reason": "",
            "max_tanimoto_to_training": "NA",
            "bemis_murcko_scaffold": "NA",
            "rna_ligand_like_score": "NA",
            "pre_docking_candidate_score": "NA",
            "diversity_cluster_id": "NA",
            "selected_for_docking": False,
            "notes": "Generated computational candidate; not experimentally validated.",
        }
        canonical, mol = canonicalize_smiles(decoded, remove_salts=remove_salts)
        if canonical is None or mol is None:
            base["rejection_reason"] = "invalid_smiles"
            rows.append(base)
            continue
        props = calculate_properties(mol, calculate_qed=calculate_qed, pains_catalog=pains_catalog)
        base.update(props)
        base["canonical_smiles"] = canonical
        base["is_valid"] = True
        base["valid_smiles"] = True
        is_duplicate = canonical in seen
        base["is_duplicate"] = is_duplicate
        reasons = []
        if is_duplicate and remove_duplicates:
            reasons.append("duplicate_canonical_smiles")
        else:
            seen.add(canonical)
        reasons.extend(_filter_reasons(base, filters))
        base["filter_pass"] = not reasons
        base["rejection_reason"] = ";".join(reasons)
        rows.append(base)
    all_rows = pd.DataFrame(rows)
    valid = all_rows[all_rows["is_valid"] == True].copy()  # noqa: E712
    filtered = all_rows[all_rows["filter_pass"] == True].copy()  # noqa: E712
    rejected = all_rows[all_rows["filter_pass"] != True].copy()  # noqa: E712
    return valid, filtered, rejected


def canonicalize_training_smiles(smiles: list[str], remove_salts: bool = True) -> list[str]:
    """Canonicalize valid training SMILES for novelty calculations."""
    canonical: list[str] = []
    seen: set[str] = set()
    for item in smiles:
        value, _mol = canonicalize_smiles(item, remove_salts=remove_salts)
        if value and value not in seen:
            seen.add(value)
            canonical.append(value)
    if not canonical:
        raise ValueError("No valid training SMILES were found after RDKit canonicalization.")
    return canonical


def write_table(df: pd.DataFrame, path: str | Path) -> None:
    """Write a DataFrame as a TSV file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)
