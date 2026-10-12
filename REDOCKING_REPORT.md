# RNA-Ligand Redocking Validation

This report validates the AutoDock Vina protocol used by SMN2-RescueDiff against public RNA-ligand complexes. It reports the preregistered result whether positive, negative, or mixed; it does not infer biological activity from docking.

## Task 1 - Prepare

| PDB ID | ligand | component | deposited method | models | formula | formula weight | heavy atoms | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8R62 | Risdiplam | Y59 | SOLUTION NMR | 14 | C22 H24 N7 O | 402.472 | 30 | prepared |
| 8R8P | SMN-CX | YB3 | SOLID-STATE NMR | 15 | C21 H21 F N6 | 376.43 | 28 | prepared |
| 5KX9 | Ribocil analogue 6YG | 6YG | X-RAY DIFFRACTION | 1 | C19 H21 N5 O2 S | 383.467 | 27 | prepared |
| 8R63 | Branaplam | Y53 | SOLUTION NMR | 15 | C22 H27 N5 O2 | 393.482 | 29 | prepared |

The prompt described all required structures as solution NMR. The deposited records disagree: 8R62 and the added branaplam complex 8R63 are solution NMR, 8R8P is solid-state NMR, and 5KX9 is X-ray diffraction. The deposited metadata govern this report. Pairwise ensemble spread is therefore not applicable to the single-model 5KX9 structure.

Branaplam was added from RCSB PDB 8R63, component Y53. No predicted or homology structure was substituted.

Receptor preparation audited `smn2_rescuediff.docking.receptor_preparer.prepare_receptor`, which validates paths and metadata but does not create PDBQT. Open Babel 3.2.1 therefore supplied the missing conversion step. Each ligand-free RNA receptor was converted with Gasteiger charges and rigid PDBQT output (`-xr`). The existing production SMN2 receptor was not reused because audit showed that it contains deposited Y59 atoms.

Deposited ligand coordinates were retained only as references. Docking used independent RDKit ETKDGv3 conformers generated from RCSB component descriptors with seed 61453, MMFF/UFF optimization, and Open Babel ligand PDBQT conversion. Risdiplam's RCSB CACTVS SMILES failed RDKit kekulization, so the RCSB InChI from the same component dictionary was converted to a valid RDKit canonical SMILES before the conformer was generated.

### Locked Grid Boxes

| PDB ID | center x/y/z (A) | size x/y/z (A) |
| --- | --- | --- |
| 8R62 | 33.0209, 17.991567, 30.289333 | 31.815, 21.53, 23.376 |
| 8R8P | 8.930821, 29.857, 30.511607 | 20.475, 19.683, 32.939 |
| 5KX9 | 27.734481, 38.909778, 21.570074 | 19.326, 25.682, 24.841 |
| 8R63 | 9.960103, 38.603138, 31.099241 | 22.195, 25.392, 29.393 |

Each center is the model-1 deposited ligand heavy-atom centroid. Each box dimension is the corresponding ligand extent plus 16 A, providing 8 A padding on both sides.

### Exact Commands

```bash
conda run -n smn2-docking python -c "from rdkit import Chem; print('ok')"
conda run -n smn2-docking vina --version
conda run -n smn2-docking python -m pip install spyrmsd==0.9.0
conda run -n smn2-docking env PYTHONPATH=src MPLCONFIGDIR=/tmp/smn2-mpl python scripts/prepare_redocking_validation.py
conda run -n smn2-docking env PYTHONPATH=src MPLCONFIGDIR=/tmp/smn2-mpl python scripts/run_redocking_validation.py
conda run -n smn2-docking env PYTHONPATH=src MPLCONFIGDIR=/tmp/smn2-mpl python scripts/analyze_redocking_validation.py
conda run -n smn2-docking env PYTHONPATH=src python scripts/build_redocking_report.py
```

The locked protocol SHA-256 is `93a483b1490b68ab6f5496688aa03569875be4cc3b3d5ae125793cb45abb77a6` and the recorded git commit is `7366492f66caff531fa3c4c7e09d872986904cbe`. File-level hashes for every downloaded structure, component dictionary, prepared docking input, pose input, protocol, and validation script are in [`data/processed/redocking_validation/final_input_sha256.tsv`](data/processed/redocking_validation/final_input_sha256.tsv). Per-job receptor and ligand hashes and all 225 exact Vina commands are retained in the individual `job_metadata.json` files and the docking manifest.

### Preparation and Pre-RMSD Failures

1. The first preparation attempt retained the RCSB CACTVS risdiplam SMILES verbatim; RDKit rejected it during kekulization. That failed attempt is preserved under `data/processed/redocking_validation/preparation_attempt_001_failed/`. No docking or RMSD was performed from it.
2. The second preparation attempt omitted Open Babel's rigid PDBQT flag. All 225 Vina invocations stopped at receptor parsing on `ROOT` records before search, so no scores or poses were produced. It is preserved under `data/processed/redocking_validation/preparation_attempt_002_failed_nonrigid_receptors/`.
3. Both corrections were made before any RMSD was calculated. The final protocol was then relocked and was not changed after results were observed.

## Task 2 - Redock

Vina version: `AutoDock Vina v1.2.7`. Exhaustiveness: 32. Requested modes: 20. Energy range: 10 kcal/mol. Seeds: 61453, 61459, 61463, 61469, 61471. CPU per job: 1. Parallel workers: 4. All deposited receptor models were docked, with model 1 designated primary.

| PDB ID | completed jobs | retained modes | best affinity (kcal/mol) | worst retained affinity (kcal/mol) |
| --- | --- | --- | --- | --- |
| 5KX9 | 5 | 99 | -9.800 | -8.710 |
| 8R62 | 70 | 1399 | -8.056 | -7.028 |
| 8R63 | 75 | 1499 | -7.384 | -6.219 |
| 8R8P | 75 | 1491 | -7.600 | -2.780 |

Final run status: 225/225 jobs completed; 0 failed; 4488 retained modes; all scores real and non-dry-run: yes. Every retained affinity is reported in [`data/processed/redocking_validation/docking/redocking_mode_scores.tsv`](data/processed/redocking_validation/docking/redocking_mode_scores.tsv).

## Task 3 - Heavy-Atom RMSD Against the Ensemble

RMSD was calculated with `spyrmsd.symmrmsd` using atomic-number and connectivity graph isomorphism, with `center=False` and `minimize=False`. Thus symmetry and atom-order differences were corrected, but docked ligands were not superposed onto the answer. Before ligand comparison, common receptor RNA heavy atoms aligned each deposited ensemble model to the receptor model used for docking. The complete pose-by-reference table is [`data/processed/redocking_validation/analysis/pose_to_ensemble_rmsd_long.tsv`](data/processed/redocking_validation/analysis/pose_to_ensemble_rmsd_long.tsv).

| PDB ID | ligand | ensemble models | native contacts | top-mode RMSD to model 1 (A) | best RMSD across ensemble (A) | best RMSD over all modes (A) | seeds under 2.0 A | contact recall | contact Jaccard |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8R62 | Risdiplam | 14 | 29 | 12.587 | 11.923 | 2.478 | 0/5 | 0.000 | 0.000 |
| 8R8P | SMN-CX | 15 | 22 | 7.169 | 6.986 | 1.856 | 0/5 | 0.273 | 0.162 |
| 5KX9 | Ribocil analogue 6YG | 1 | 59 | 6.173 | 6.173 | 4.384 | 0/5 | 0.271 | 0.198 |
| 8R63 | Branaplam | 15 | 21 | 1.307 | 1.244 | 1.137 | 1/5 | 0.810 | 0.630 |

**8R62 Risdiplam:** The top-scored mode is not within 2.0 A of at least one deposited model (best ensemble RMSD 11.923 A; model-1 RMSD 12.587 A). The best pose over all retained primary-run modes was mode 14 at seed 61453, RMSD 2.478 A to deposited model 7. Rank-1 seed success was 0/5 (0.00). Deposited ensemble pairwise ligand RMSD min/median/max was 0.068/1.309/1.861 A.

**8R8P SMN-CX:** The top-scored mode is not within 2.0 A of at least one deposited model (best ensemble RMSD 6.986 A; model-1 RMSD 7.169 A). The best pose over all retained primary-run modes was mode 20 at seed 61469, RMSD 1.856 A to deposited model 5. Rank-1 seed success was 0/5 (0.00). Deposited ensemble pairwise ligand RMSD min/median/max was 0.156/0.677/1.147 A.

**5KX9 Ribocil analogue 6YG:** The top-scored mode is not within 2.0 A of at least one deposited model (best ensemble RMSD 6.173 A; model-1 RMSD 6.173 A). The best pose over all retained primary-run modes was mode 15 at seed 61471, RMSD 4.384 A to deposited model 1. Rank-1 seed success was 0/5 (0.00). Deposited ensemble pairwise ligand RMSD min/median/max was not applicable/not applicable/not applicable A.

**8R63 Branaplam:** The top-scored mode is within 2.0 A of at least one deposited model (best ensemble RMSD 1.244 A; model-1 RMSD 1.307 A). The best pose over all retained primary-run modes was mode 2 at seed 61469, RMSD 1.137 A to deposited model 14. Rank-1 seed success was 1/5 (0.20). Deposited ensemble pairwise ligand RMSD min/median/max was 0.087/0.753/1.844 A.

RNA alignment QC used 466-2321 common heavy atoms; alignment RMSD ranged from 0.0000 to 2.6486 A.

## Task 4 - Contact Recovery

A contact is a ligand heavy atom within 4.0 A of a receptor heavy atom. Contact identities retain RNA chain, residue number, insertion code, residue name, and atom name. The top-scored pose is the lowest-affinity mode across the five model-1 seed runs.

| PDB ID | native contacts | predicted contacts | recovered | precision | recall | Jaccard | meaningful by locked criterion |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8R62 | 29 | 17 | 0 | 0.000 | 0.000 | 0.000 | no |
| 8R8P | 22 | 21 | 6 | 0.286 | 0.273 | 0.162 | no |
| 5KX9 | 59 | 38 | 16 | 0.421 | 0.271 | 0.198 | no |
| 8R63 | 21 | 23 | 17 | 0.739 | 0.810 | 0.630 | yes |

- **8R62:** Native-contact residues missed by the docked pose: A:C8, A:C9, B:A14, B:G15. Docked-only contact residues: A:A3, A:C4, A:PSU5, A:PSU6, B:A17, B:A18, B:G19, B:U20.
- **8R8P:** Native-contact residues missed by the docked pose: B:A14, B:G13, B:G15. Docked-only contact residues: A:A7, A:PSU5, A:PSU6, A:U10.
- **5KX9:** Native-contact residues missed by the docked pose: Y:U61. Docked-only contact residues: X:G32, X:G33, X:G41, Y:A102, Y:G101, Y:G93.
- **8R63:** Native-contact residues missed by the docked pose: none. Docked-only contact residues: none.

Atom-level contact membership is reported in [`data/processed/redocking_validation/analysis/model1_contact_comparison.tsv`](data/processed/redocking_validation/analysis/model1_contact_comparison.tsv). The locked operational criterion for meaningful contact recovery was recall >= 0.50 and Jaccard >= 0.33.

## Task 5 - What It Means for the Paper

**Conclusion: Mixed result.** The preregistered protocol produced mixed recovery across the SMN benchmarks. It therefore shows some RNA-pose capability but does not support treating the docking instrument as uniformly validated for the target; manuscript claims must remain bounded by the failed benchmark complexes.

This validation addresses pose and contact recovery under one fixed Vina preparation and search protocol. It does not establish binding affinity accuracy, biological splicing rescue, toxicity, or clinical performance. The added branaplam result is a fourth benchmark and does not alter the preregistered requirement to report all three required structures.

## NUMBERS FOR THE MANUSCRIPT

- AutoDock Vina: AutoDock Vina v1.2.7.
- Locked protocol SHA-256: `93a483b1490b68ab6f5496688aa03569875be4cc3b3d5ae125793cb45abb77a6`.
- Redocking jobs: 225/225 completed; 0 failed.
- Retained real Vina modes: 4488.
- 8R62 Risdiplam: top-mode model-1 RMSD 12.587 A; best top-mode ensemble RMSD 11.923 A; best all-mode ensemble RMSD 2.478 A (rank 14, seed 61453); rank-1 seeds <2.0 A 0/5; native/recovered contacts 29/0; contact precision 0.000, recall 0.000, Jaccard 0.000; ensemble pairwise RMSD min/median/max 0.068/1.309/1.861 A.
- 8R8P SMN-CX: top-mode model-1 RMSD 7.169 A; best top-mode ensemble RMSD 6.986 A; best all-mode ensemble RMSD 1.856 A (rank 20, seed 61469); rank-1 seeds <2.0 A 0/5; native/recovered contacts 22/6; contact precision 0.286, recall 0.273, Jaccard 0.162; ensemble pairwise RMSD min/median/max 0.156/0.677/1.147 A.
- 5KX9 Ribocil analogue 6YG: top-mode model-1 RMSD 6.173 A; best top-mode ensemble RMSD 6.173 A; best all-mode ensemble RMSD 4.384 A (rank 15, seed 61471); rank-1 seeds <2.0 A 0/5; native/recovered contacts 59/16; contact precision 0.421, recall 0.271, Jaccard 0.198; ensemble pairwise RMSD min/median/max not applicable/not applicable/not applicable A.
- 8R63 Branaplam: top-mode model-1 RMSD 1.307 A; best top-mode ensemble RMSD 1.244 A; best all-mode ensemble RMSD 1.137 A (rank 2, seed 61469); rank-1 seeds <2.0 A 1/5; native/recovered contacts 21/17; contact precision 0.739, recall 0.810, Jaccard 0.630; ensemble pairwise RMSD min/median/max 0.087/0.753/1.844 A.

The preregistered protocol produced mixed recovery across the SMN benchmarks. It therefore shows some RNA-pose capability but does not support treating the docking instrument as uniformly validated for the target; manuscript claims must remain bounded by the failed benchmark complexes.
