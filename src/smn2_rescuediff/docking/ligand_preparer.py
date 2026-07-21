"""RDKit ligand preparation for docking inputs."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd


def require_rdkit():
    """Import RDKit modules or raise a clear error."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        return Chem, AllChem
    except Exception as exc:
        raise ImportError("RDKit is required for ligand preparation. Install rdkit in the project environment.") from exc


def obabel_available(executable: str = "obabel") -> bool:
    """Return whether Open Babel's obabel executable is available."""
    return shutil.which(executable) is not None


def _convert_to_pdbqt_with_obabel(
    *,
    sdf_file: Path,
    pdb_file: Path,
    pdbqt_file: Path,
    executable: str,
) -> tuple[bool, str]:
    """Convert SDF/PDB to ligand PDBQT with Open Babel when available."""
    if not obabel_available(executable):
        return False, "Open Babel obabel executable was not found; ligand PDBQT was not created."
    attempts = [sdf_file, pdb_file]
    errors: list[str] = []
    for input_file in attempts:
        if not input_file.exists():
            continue
        command = [
            executable,
            str(input_file),
            "-O",
            str(pdbqt_file),
            "--partialcharge",
            "gasteiger",
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode == 0 and pdbqt_file.exists() and pdbqt_file.stat().st_size > 0:
            return True, f"Created ligand PDBQT with Open Babel from {input_file.name}."
        errors.append(f"{input_file.name}: {completed.stderr.strip() or completed.stdout.strip() or 'Open Babel conversion failed'}")
    return False, "Open Babel PDBQT conversion failed. " + " | ".join(errors)


def prepare_ligands(molecules: pd.DataFrame, ligand_dir: str | Path, config: dict[str, Any]) -> pd.DataFrame:
    """Prepare SDF/PDB ligand files and optional PDBQT files from canonical SMILES."""
    Chem, AllChem = require_rdkit()
    out = Path(ligand_dir)
    out.mkdir(parents=True, exist_ok=True)
    prep_cfg = config.get("ligand_preparation", {})
    add_h = bool(prep_cfg.get("add_hydrogens", True))
    generate_3d = bool(prep_cfg.get("generate_3d_conformers", True))
    optimize = bool(prep_cfg.get("optimize_geometry", True))
    attempts = int(prep_cfg.get("max_conformer_attempts", 20))
    prepare_pdbqt = bool(prep_cfg.get("prepare_pdbqt", True))
    obabel_executable = str(prep_cfg.get("open_babel_executable", "obabel"))
    rows: list[dict[str, Any]] = []
    for _, row in molecules.iterrows():
        molecule_id = str(row["molecule_id"])
        smiles = str(row["canonical_smiles"])
        base = {
            "molecule_id": molecule_id,
            "canonical_smiles": smiles,
            "rdkit_parse_status": "failed",
            "conformer_status": "NA",
            "optimization_status": "NA",
            "sdf_file": "NA",
            "pdb_file": "NA",
            "pdbqt_file": "NA",
            "prep_status": "failed",
            "notes": "",
        }
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            base["notes"] = "Invalid SMILES; ligand preparation skipped."
            rows.append(base)
            continue
        base["rdkit_parse_status"] = "parsed"
        if add_h:
            mol = Chem.AddHs(mol)
        conformer_ok = True
        if generate_3d:
            params = AllChem.ETKDGv3()
            params.randomSeed = 61453
            status = AllChem.EmbedMolecule(mol, params)
            if status != 0:
                status = AllChem.EmbedMolecule(mol, maxAttempts=attempts, randomSeed=61453)
            conformer_ok = status == 0
            base["conformer_status"] = "generated" if conformer_ok else "failed"
        if conformer_ok and optimize:
            try:
                status = AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
                base["optimization_status"] = "optimized" if status in (0, 1) else f"mmff_status_{status}"
            except Exception:
                try:
                    status = AllChem.UFFOptimizeMolecule(mol, maxIters=200)
                    base["optimization_status"] = "uff_optimized" if status in (0, 1) else f"uff_status_{status}"
                except Exception as exc:
                    base["optimization_status"] = "failed"
                    base["notes"] += f"Geometry optimization failed: {exc}. "
        if not conformer_ok:
            rows.append(base)
            continue
        sdf_file = out / f"{molecule_id}.sdf"
        pdb_file = out / f"{molecule_id}.pdb"
        writer = Chem.SDWriter(str(sdf_file))
        writer.write(mol)
        writer.close()
        Chem.MolToPDBFile(mol, str(pdb_file))
        base["sdf_file"] = str(sdf_file)
        base["pdb_file"] = str(pdb_file)
        if prepare_pdbqt:
            pdbqt_file = out / f"{molecule_id}.pdbqt"
            converted, note = _convert_to_pdbqt_with_obabel(
                sdf_file=sdf_file,
                pdb_file=pdb_file,
                pdbqt_file=pdbqt_file,
                executable=obabel_executable,
            )
            if converted:
                base["pdbqt_file"] = str(pdbqt_file)
            base["notes"] += note + " "
        base["prep_status"] = "prepared"
        rows.append(base)
    return pd.DataFrame(rows)
