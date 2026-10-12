# TCBB Reviewer Response and Implementation Ledger

This ledger maps every actionable point in the two October 2026 evaluator reports to a repository or manuscript action. Historical outputs were preserved. Counts were recomputed from source tables; no expected value was fabricated to make a check pass.

Status definitions: **implemented** means code/data/text changed and verified; **withdrawn** means an unsupported claim was removed and the historical artifact retained for audit; **partial** means the available repository evidence was released but cannot validate biology; **outstanding experiment** means the requested evidence does not exist and was not invented.

## Repository and Evidence Findings

| Evaluator concern | Status | Action |
|---|---|---|
| One fixed SMN2 box was reused for unrelated HuRDA coordinate frames | **Implemented + withdrawn** | Added per-receptor bounds boxes and fixed-box atom-coverage validation in Module 4; added a receptor-specific 150 x 100 preset. Historical scores are quarantined from selectivity inference. |
| Quantify box placement bias | **Implemented** | Released receptor atom coverage, rho = -0.8469, p = 8.36e-28, group means, and variance decomposition. |
| Investigate 4,815 exact-zero scores | **Implemented** | Released counts by receptor, zero-score sensitivity, and a 500-pose distance audit. Sampled zero-score poses had median minimum receptor distance 32.65 Angstrom versus 2.58 Angstrom for nonzero poses. Zeros were not silently deleted. |
| Explain strongest-decoy domination | **Implemented** | Released decoy-frequency and removal sensitivity. HuRDA_000041 was strongest for all 147 molecules; removing it changed the median conservative gap from -1.95 to -0.50 kcal/mol. |
| Add worst-of-panel null/sensitivity | **Implemented** | Released 2,000-replicate random-panel analyses at 5, 10, 20, and 50 decoys plus the full panel. |
| Redo selectivity with valid boxes | **Code/preset implemented; outstanding compute** | Corrective per-receptor redocking is configured but not represented as completed. All historical target-decoy and SRSI selectivity claims are withdrawn until it runs. |
| Benchmark-control values were hard-coded/unreleased | **Implemented + qualified** | Released the complete local v2 risdiplam/branaplam target and 2 x 100 files. Their decoy values share the box defect and are audit-only. New paper figures are table-derived. |
| Correct MolGen_038292 properties | **Implemented** | RDKit recomputation reports MW 377.448, QED 0.8563, cLogP 2.7055, HBD 2, HBA 6, rotatable bonds 3, tPSA 79.38, and 28 heavy atoms. |
| Correct stronger-decoy value | **Implemented** | Reported as 1/97 = 0.010309 fraction = 1.0309%, never 0.093. Because historical decoy geometry is invalid, this is retained only as an audit correction. |
| “Biologically weighted” gap was a plain mean | **Implemented** | The manuscript states all observed weights were 1.0 and retires the biological-weighting interpretation. |
| Target conformers are first five 8R62 NMR models | **Implemented** | Explicitly stated as a narrow ligand-bound ensemble, not five independent targets. |
| Residual molecular-size confounding | **Implemented** | Released MW-score rho = -0.5054, ligand efficiency, and molecular-weight residual ranks for all 500 molecules. MolGen_038292 ranks 136/500 by ligand efficiency and 3/500 by MW residual. |
| Release claimed 40 x 20 cross-target campaign | **Implemented + qualified** | Released all 800 rows, receptors, logs, poses, panel provenance, and config. It is described as engineering/exploratory because heterogeneous receptor Vina scores are not calibrated as commensurate energies. |
| Reconcile generation counts and denominators | **Implemented** | Added a nine-stage flow table: 49,660 emitted/valid strings, 171 canonical collapses, 49,489 unique canonical molecules, 9,012 unique filter failures, 40,477 retained, 9,183 total rejected records, 1,000 exported. |
| Add a generator baseline | **Implemented** | Ran a controlled 50,000-sample token 3-gram baseline. It yielded 915 valid and 163 filter-pass molecules versus 49,660 and 40,477 for Transformer/SELFIES. Interpretation is limited to engineering yield. |
| Missing checkpoint/tokenizer/config | **Implemented** | Released the 41 MB production checkpoint, vocabulary, tokenizer metadata, config, training log, source balance, and dataset summary. |
| Missing exact ChEMBL/ZINC release IDs, dates, licenses, and corpus | **Partial** | Existing hashes and source accounting are preserved. Exact upstream release/licensing metadata could not be recovered reliably, so third-party corpus redistribution was not fabricated. |
| Add a license | **Implemented** | Added an MIT `LICENSE`. |
| Default branch stale and pinned artifact hard to find | **Implemented in documentation; branch alignment at publication step** | Root README leads with the corrective audit and evidence paths. The final publication commit is intended to be merged to `main` without force-pushing. |
| Target score rows with blank engine provenance | **Partial** | Historical rows remain unchanged rather than being silently relabeled. The manuscript identifies this as a legacy merge metadata gap. |
| Preserve auditable pose evidence | **Implemented** | Released all 800 cross-target poses and all locally retained benchmark-control poses; historical HuRDA target/decoy release remains intact. |

## Manuscript Interpretation and Methodology

| Evaluator concern | Status | Action |
|---|---|---|
| Headline enrichment is selection-circular | **Implemented** | Target-only five-conformer robustness is the primary comparator. Matched controls are explicitly labeled within-cohort selection behavior, not predictive validation. |
| Account for reused controls/scaffolds | **Implemented in interpretation; outstanding independent cohort** | Existing sign-flip result is demoted because 400 assignments reuse 183 controls. Independent, predeclared external ligands/scaffolds remain outstanding. |
| Redocking calibration is mixed | **Implemented** | Reports one of four rank-1 recoveries below 2 Angstrom and labels calibration weak/mixed. Lower-ranked near-native modes are evidence about sampling, not validation. |
| Expand to 20–30 diverse RNA complexes with positives/negatives, AUC/enrichment | **Outstanding experiment** | No defensible local dataset supports this. It is listed as the highest-priority computational validation, not simulated. |
| Test RNA-aware rescoring and alternative folds | **Outstanding experiment** | The corrective roadmap includes RNA-aware reranking and broader ensembles; no result is claimed. |
| Separate Stage C lexicographic ranking from SRSI | **Implemented** | The paper identifies them as different historical rules. Neither is the primary inference after the box audit. |
| Specify one primary rule | **Implemented** | Primary rule is complete five-conformer target robustness, ordered by worst target score, mean score, range, then standard deviation. |
| SRSI double-counts target score | **Implemented** | SRSI is labeled a post hoc heuristic and removed from primary efficacy/selectivity claims. |
| “Splicing-rescue proxy” overstates biology | **Implemented** | Renamed geometric contact proxy; explicitly notes fixed-fraction landmark assignment, saturation, and lack of exon-inclusion calibration. |
| Remove or justify saturated contact terms | **Implemented in interpretation** | Saturated/near-constant terms are excluded from the primary target-only rule and retained only for historical audit. |
| Candidate called stronger/competitive from raw Vina scores | **Implemented** | Language changed to score-based prioritization only; no binding, clinical, or superiority claim. Size-controlled ranks are reported. |
| All conservative gaps negative / no robust selectivity | **Withdrawn as inference** | Historical negative gaps are reported as artifacts of the box failure, not evidence for or against true selectivity. No corrected selectivity claim is made. |
| HuRDA-180K wording implies structural coverage | **Implemented** | Sequence atlas (180,000+) and structural subset (100 prepared; 97 common historical) are kept separate throughout. |
| Unconditional generator / no RNA data contribution / “RescueDiff” ambiguity | **Implemented** | Near title and Methods, the paper states the model is an unconditional autoregressive Transformer, not diffusion or target-conditioned; RNA-focused sources contributed zero rows historically. |
| Center contribution on TCBB innovation | **Implemented** | Paper reframed around evidence contracts, coordinate-aware validation, run-level provenance, missingness preservation, falsification, and staged structural evaluation as a reusable computational-biology contribution. |
| Soften “positive screening signal” conclusion | **Implemented** | Replaced by a narrower claim: successful execution plus successful detection/containment of invalid evidence. |
| Direct experimental binding and exon-7 inclusion validation | **Outstanding experiment** | No wet-lab data exist. The paper gives a prospective validation plan and does not claim activity. |
| Independent prospective validation | **Outstanding experiment** | Explicitly required before biological conclusions; no synthetic labels or fabricated measurements were introduced. |

## Verification

- Module 4 regression suite includes per-receptor box and fixed-box coverage checks.
- Audit scripts regenerate all reported corrective statistics from repository tables.
- Manuscript figures are generated from released TSV/JSON files; no benchmark scores are hard-coded.
- The final manuscript package includes its source, figure data, figures, preview PDF, change report, and checksum manifest.
