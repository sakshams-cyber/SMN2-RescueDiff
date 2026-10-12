# SMN2-RescueDiff Manuscript Validation Report

This report records two completed analyses and one research-only scoping task requested for manuscript review. No docking score, descriptor, or structure was imputed. No new AutoDock Vina docking was run.

## Task 1: Stage C.5 Selection-Effect Null

### What was run

RDKit was verified before analysis:

```bash
.venv/bin/python -c "from rdkit import Chem; print('ok')"
```

The requested destination was already an existing output directory. To comply with the non-overwrite rule, the analytical parameters were left unchanged and only the output path was moved to a new directory:

```bash
MPLCONFIGDIR=/tmp/smn2-rescuediff-mpl .venv/bin/python stage_c5_null.py \
  --selected-molecules data/processed/docking/target_conformer_runs/target5_500_v1_complete_merged/selected_molecules.tsv \
  --target-summary data/processed/docking/target_conformer_runs/target5_500_v1_complete_merged/target_conformer_robustness_scores_final.tsv \
  --top40 data/processed/docking/stage_c_selectivity_150_partial_after_target5/top40_for_ligand_decoy_control_audit.tsv \
  --replicates 2000 \
  --out data/processed/docking/stage_c5_selection_null_seed61453_r2000_20261011/stage_c5_selection_null.json
```

The Morgan guard remained enabled with a Tanimoto ceiling of 0.6, radius 2, and 2048 bits. The random seed was 61453. The git commit was `7366492f66caff531fa3c4c7e09d872986904cbe`.

### Sanity checks

| Check | Observed | Result |
|---|---:|---|
| Null pool size is 460 | 460 | PASS |
| Completed replicates are 2000 | 2000 | PASS |
| Failed-to-match replicates | 0 | PASS |
| Null mean is within +/-0.15 | -0.04917127624999998 | PASS |
| Observed mean gap is 0.7292 | 0.7292 | PASS |

### Results

- Null mean: -0.04917127624999998
- Null standard deviation: 0.09931632853161437
- Minimum cohort mean: -0.3593750000000001
- Maximum cohort mean: 0.3666924999999999
- Observed mean gap: 0.7292
- Observed z-score versus null: 7.83729410619754
- Cohorts at or above observed: 0
- Empirical p-value, observed greater than or equal to null: 0.0004997501249375312
- Replicates completed: 2000

| Null quantile | Value |
|---:|---:|
| 0.025 | -0.22676200000000016 |
| 0.50 | -0.05555874999999995 |
| 0.95 | 0.12601275 |
| 0.975 | 0.17292493749999988 |
| 0.99 | 0.22379912499999985 |

The empirical p-value uses the script's finite-sampling correction; zero of 2000 sampled cohorts reached the observed gap. One matching descriptor, `formal_charge`, had zero variance in the null pool and was excluded from standardized distance matching by the supplied script. Parameters were not changed in response.

### Provenance

| Input | SHA-256 |
|---|---|
| `/Users/saksham/Documents/Regeneron ISEF SMA Project/SMN2-RescueDiff/stage_c5_null.py` | `a9648b23dddb7bc3f55f875c3fdab700f2b0489fb43e8600c59e3b8137b70868` |
| `/Users/saksham/Documents/Regeneron ISEF SMA Project/SMN2-RescueDiff/data/processed/docking/target_conformer_runs/target5_500_v1_complete_merged/selected_molecules.tsv` | `f38208d39242aa302d7dabebb97805a046899224329f6b2a2b5301a1886db6c2` |
| `/Users/saksham/Documents/Regeneron ISEF SMA Project/SMN2-RescueDiff/data/processed/docking/target_conformer_runs/target5_500_v1_complete_merged/target_conformer_robustness_scores_final.tsv` | `db8e2ae293869bf4e538cd82de23e9952abaad4886fdc69f727a719c1d19da77` |
| `/Users/saksham/Documents/Regeneron ISEF SMA Project/SMN2-RescueDiff/data/processed/docking/stage_c_selectivity_150_partial_after_target5/top40_for_ligand_decoy_control_audit.tsv` | `0ad8da2e0e7f38c01d696c2100133a863e929b17c31d580c9cbc278595b706d4` |

Output JSON SHA-256: `1032a19c6e7c709bff5c59390850607e3ef2cdb8ed45b0bd023ad10674f43040`.

## Task 2: Pose Contact Features for the Stage C Cohort

### Investigation

1. `pose_parser.py` does not search `inputs.pose_dir`. `parse_pose_for_candidate` reads `pose_file` (or `ligand_file`) and `receptor_file` directly from each candidate row. The existing loader therefore cannot map a molecule ID into the five nested conformer directories by itself.
2. `ensemble_selectivity_scores.tsv` has `molecule_id` and `weighted_mean_selectivity_gap`; the loader already maps the latter to `weighted_selectivity_gap`. It does not map `mean_target_deltaG` to `target_deltaG`, `ensemble_off_target_risk_score` to `off_target_risk_score`, or `decoys_valid` to the valid-decoy count fields. An explicit adapter was required.
3. Copying poses into a flat directory was unnecessary. The minimal fix was a 147-row target-score adapter containing explicit source `pose_file` and `receptor_file` paths. `SMN2_CONF_01` was fixed for every molecule to preserve one receptor geometry and avoid per-molecule, score-driven pose selection. The five-conformer `mean_target_deltaG` remained the target score used by Module 5.

The original Stage C table contained 150 rows. Exactly 147 had numeric `conservative_selectivity_gap`; the excluded IDs were `MolGen_005024, MolGen_012374, MolGen_018383`. Each retained molecule had 97 completed real decoy scores, for 14259 validated decoy rows. No missing values were imputed.

The protected YAML sections matched the reference config exactly: `pose_analysis` PASS; `splicing_rescue_scoring.weights` PASS; `splicing_rescue_scoring.thresholds` PASS; `srsi.weights` PASS.

### What was run

```bash
PYTHONPATH=src .venv/bin/python scripts/prepare_stage_c147_module5_inputs.py \
  --config configs/splicing_rescue_scoring_stageC147.yaml \
  --output-dir data/processed/final_scoring/stage_c_147_inputs_20261011
```

```bash
MPLCONFIGDIR=/tmp/smn2-rescuediff-mpl PYTHONPATH=src .venv/bin/python \
  -m smn2_rescuediff.splicing_rescue.cli \
  --config configs/splicing_rescue_scoring_stageC147.yaml
```

Config: `configs/splicing_rescue_scoring_stageC147.yaml`. Stable output: `data/processed/final_scoring/stage_c_147/tables/pose_contact_features.tsv`. Run-specific output: `data/processed/final_scoring/stage_c_147/runs/module5_stage_c_147_pose_contacts/tables/pose_contact_features.tsv`.

### Validation

- Output rows: 147
- Unique molecule IDs: 147
- IDs exactly equal the Stage C 147: true
- Unexpected IDs: []
- Missing Stage C IDs: []
- Earlier 200-candidate rows: 200
- Molecule overlap with the earlier 200-candidate pose table: 0

| pose_analysis_status | Count |
|---|---:|
| pose_based_proxy | 147 |

### Contact-feature distributions

All values below were computed from the written 147-row TSV. Range is maximum minus minimum.

| Feature | Median | Minimum | Maximum | Range | Missing |
|---|---:|---:|---:|---:|---:|
| num_close_contacts | 1631.0 | 1301.0 | 2153.0 | 852.0 | 0 |
| num_contacts_with_beneficial_landmarks | 1223.0 | 975.0 | 1614.0 | 639.0 | 0 |
| num_contacts_with_caution_landmarks | 570.0 | 455.0 | 753.0 | 298.0 | 0 |
| minimum_distance_to_u1_region | 1.6534 | 0.9704 | 2.1387 | 1.1683 | 0 |
| hydrogen_bond_proxy_count | 197.0 | 107.0 | 323.0 | 216.0 | 0 |
| stacking_proxy_count | 522.0 | 323.0 | 716.0 | 393.0 | 0 |
| pose_support_score | 1.0 | 0.927586 | 1.0 | 0.07241399999999998 | 0 |
| splice_site_occlusion_penalty | 0.349666 | 0.349254 | 0.35 | 0.0007459999999999689 | 0 |

The occlusion near-degeneracy holds. In the Stage C cohort it spans 0.349254 to 0.35, a range of 0.0007459999999999689. The earlier 200-candidate table has the same observed minimum, maximum, and range. Despite 120 distinct rounded values, this feature remains too narrowly distributed to contribute meaningful rank discrimination under ordinary normalization without careful sensitivity analysis. The `pose_support_score` also saturates at 1.0, with a median of 1.0; that is another limitation of the current proxy construction.

These contact features are geometric proxies based on ligand-receptor atom-pair distances. They are not experimentally observed contacts and do not establish splicing rescue.

### Provenance

- Random seed: 42
- Git commit: `7366492f66caff531fa3c4c7e09d872986904cbe`
- Config SHA-256: `bffb2a8ff87f89eceeb6a38ce940c6a39a6972c7f3eb63cb9b8983b15746847f`
- Input files hashed: 165
- Input hash manifest: `/Users/saksham/Documents/Regeneron ISEF SMA Project/SMN2-RescueDiff/data/processed/final_scoring/stage_c_147_inputs_20261011/input_file_sha256.tsv`
- Input hash manifest SHA-256: `f376b4ae2e5b81b251a47a23e1664aa08f3e7c88b50996cb25265e3d052a80ac`
- Pose-contact TSV SHA-256: `9746cf29ad86151dcbd1bd3ab16d258b42e5b2076bf5225534f13754952c98a7`

## Task 3: Redocking Validation Scope

### Search performed

Research only; no docking or coordinate preparation was run. Search date: 2026-10-11. Local random seed: not applicable. Local input files: none. Git commit at scoping time: `7366492f66caff531fa3c4c7e09d872986904cbe`.

Searches covered RCSB PDB and the linked primary literature using the terms `risdiplam RNA SMN2 structure`, `branaplam RNA structure`, `SMN2 RNA risdiplam structure`, and `RNA small molecule complex redocking benchmark`. The primary structural paper for the 2024 SMN2/U1 complexes is [Malard et al., Nucleic Acids Research](https://doi.org/10.1093/nar/gkae201); the SMN-C5 structure is described by [Campagne et al., Nature Chemical Biology](https://doi.org/10.1038/s41589-019-0384-5).

### Candidate structures

For NMR entries, crystallographic resolution is not applicable; deposited ensemble size is reported instead. "Drug-like" is used here as a benchmarking classification for a discrete medicinal-chemistry small molecule rather than an ion, solvent, or native metabolite. It is not a clinical claim.

| PDB | Resolution / ensemble | Method | Ligand | RNA target | Drug-like for benchmark? |
|---|---|---|---|---|---|
| [8R62](https://www.rcsb.org/structure/8R62) | N/A; 14 deposited models | Solution NMR | Risdiplam | Human SMN2 exon 7 5' splice-site/U1 snRNA duplex with the A-1 bulge | Yes; medicinal small-molecule splicing modifier |
| [8R63](https://www.rcsb.org/structure/8R63) | N/A; 15 deposited models | Solution NMR | Branaplam | Human 5' splice-site/U1 snRNA duplex with the same bulged-adenine epitope | Yes; medicinal small-molecule splicing modifier |
| [6HMO](https://www.rcsb.org/structure/6HMO) | N/A; 20 deposited models | Solution NMR | SMN-C5 | Human SMN2 exon 7 5' splice-site/U1 snRNA duplex | Yes; research splicing-modifier scaffold |
| [8CF2](https://www.rcsb.org/structure/8CF2) | N/A; 14 deposited models | Solution NMR | SMN-CY | A-1-bulged 5' splice-site/U1 snRNA duplex | Yes; research splicing-modifier scaffold |
| [8R8P](https://www.rcsb.org/structure/8R8P) | N/A; 15 deposited models | Solid-state NMR in the RCSB method field | SMN-CX | SMN2 exon 7 5' splice-site/U1 snRNA duplex | Yes; research splicing-modifier scaffold |
| [5KX9](https://www.rcsb.org/structure/5KX9) | 2.90 Angstrom | X-ray diffraction | Ribocil analogue 6YG | *Fusobacterium nucleatum* FMN riboswitch | Yes; synthetic antibacterial RNA inhibitor |

RCSB titles 8R8P as a "Solution structure" while its experimental-method field says `SOLID-STATE NMR`; this metadata inconsistency should be resolved from the deposited experimental records before using 8R8P in a formal benchmark.

### Feasibility and minimum credible set

A redocking benchmark is feasible with public data, including the exact benchmark-control ligands risdiplam and branaplam bound to the relevant SMN2-like U1/5' splice-site RNA epitope. The smallest credible set is four structures: 8R62, 8R63, and 6HMO as the in-domain SMN2/U1 set, plus 5KX9 as an orthogonal X-ray RNA-ligand positive control. A stronger extension would add 8CF2 and, after resolving its method metadata, 8R8P.

The benchmark should remove each bound ligand, redock it with the frozen and predeclared receptor-preparation protocol, and evaluate symmetry-aware heavy-atom pose RMSD against the deposited pose. For the NMR entries, every deposited conformer should be treated explicitly or a conformer-selection rule should be fixed before docking. Success should be reported as top-1 and top-N pose recovery at a preregistered RMSD threshold, not as Vina score agreement. The X-ray control helps distinguish a failure caused by the SMN2 NMR ensemble from a general RNA-docking failure.

This benchmark can validate pose-recovery behavior for the chosen protocol. It cannot by itself validate Vina affinity rankings, exon 7 rescue, selectivity, or biological activity.

## NUMBERS FOR THE MANUSCRIPT

| Label | Value |
|---|---:|
| Stage C.5 null pool size | 460 |
| Stage C.5 replicates completed | 2000 |
| Stage C.5 replicates failed to match | 0 |
| Stage C.5 null mean | -0.04917127624999998 |
| Stage C.5 null standard deviation | 0.09931632853161437 |
| Stage C.5 null 0.025 quantile | -0.22676200000000016 |
| Stage C.5 null 0.50 quantile | -0.05555874999999995 |
| Stage C.5 null 0.95 quantile | 0.12601275 |
| Stage C.5 null 0.975 quantile | 0.17292493749999988 |
| Stage C.5 null 0.99 quantile | 0.22379912499999985 |
| Stage C.5 maximum null cohort mean | 0.3666924999999999 |
| Stage C.5 observed mean gap | 0.7292 |
| Stage C.5 observed z-score versus null | 7.83729410619754 |
| Stage C.5 empirical p-value | 0.0004997501249375312 |
| Stage C pose-feature cohort size | 147 |
| Stage C pose-based proxy count | 147 |
| Stage C num_close_contacts median [min, max] | 1631.0 [1301.0, 2153.0] |
| Stage C beneficial contacts median [min, max] | 1223.0 [975.0, 1614.0] |
| Stage C caution contacts median [min, max] | 570.0 [455.0, 753.0] |
| Stage C minimum U1 distance median [min, max], Angstrom | 1.6534 [0.9704, 2.1387] |
| Stage C hydrogen-bond proxy count median [min, max] | 197.0 [107.0, 323.0] |
| Stage C stacking proxy count median [min, max] | 522.0 [323.0, 716.0] |
| Stage C pose-support score median [min, max] | 1.0 [0.927586, 1.0] |
| Stage C occlusion penalty median [min, max] | 0.349666 [0.349254, 0.35] |
| Stage C occlusion penalty range | 0.0007459999999999689 |
| Minimum credible redocking benchmark size | 4 structures |
| Minimum credible redocking PDB IDs | 8R62, 8R63, 6HMO, 5KX9 |
