# Manuscript Change Report

## New Center of the Paper

The manuscript now presents SMN2-RescueDiff as a failure-aware computational biology pipeline. Its primary contribution is an evidence-contract architecture that preserves configuration, provenance, missingness, execution status, geometry checks, and interpretation boundaries across modules. The paper does not convert the project into a positive drug-discovery story unsupported by the data. Instead, it identifies successful pipeline engineering and successful internal falsification as the defensible contribution.

## Major Scientific Corrections

1. The historical 14,259-job HuRDA decoy campaign is withdrawn from biological selectivity inference. The paper reports the fixed-box coordinate failure, atom-coverage statistics, score correlation, variance decomposition, zero-score pose audit, strongest-receptor domination, and panel-size sensitivity.
2. Target-only five-conformer robustness is now the one primary ranking rule. Stage C lexicographic ranking and SRSI are explicitly separate historical heuristics.
3. The former “splicing-rescue proxy” is renamed “geometric contact proxy” and stripped of mechanistic interpretation.
4. Matched-control enrichment is demoted to within-cohort selection behavior. The target-only cohort performs at least as well, and control reuse is disclosed.
5. Redocking is described as mixed/weak calibration: one of four complexes meets the rank-1 2 Angstrom criterion.
6. MolGen_038292 descriptors are corrected from its repository SMILES: MW 377.448, QED 0.8563, cLogP 2.7055, HBD 2, HBA 6, rotatable bonds 3, tPSA 79.38, heavy atoms 28.
7. The stronger-decoy audit quantity is corrected to 1/97 = 0.010309 fraction = 1.0309%, while also being labeled non-inferential because the decoy experiment failed QC.
8. Molecular-size confounding is quantified. Target score versus MW has Spearman rho -0.5054; MolGen_038292 ranks 136/500 by Vina ligand efficiency and 3/500 by MW-adjusted residual.
9. The first five 8R62 NMR models are identified as a narrow ligand-associated ensemble, not five independent targets.
10. All historical decoy weights were 1.0, so “biologically weighted” is corrected to an unweighted mean.

## Reproducibility Additions

1. Generation accounting now separates 49,660 emitted valid strings, 171 canonical collapses, 49,489 unique canonical molecules, 9,012 unique filter failures, 40,477 retained molecules, 9,183 total rejected records, and 1,000 exports.
2. A controlled 50,000-sample token 3-gram baseline was run and released. It produced 915 valid and 163 filter-pass molecules, compared with 49,660 and 40,477 for Transformer/SELFIES.
3. The previously missing 40 x 20 cross-target campaign is released with all 800 rows, logs, poses, receptors, configuration, and provenance.
4. The complete local risdiplam/branaplam v2 target and 2 x 100 files are released; its decoy portion is labeled audit-only because it shares the box defect.
5. The production checkpoint, vocabulary, tokenizer metadata, configuration, training log, source balance, and dataset summary are released.
6. An MIT license, root audit notice, release documentation, and exhaustive repository reviewer-response ledger were added.
7. Module 4 now supports receptor-specific bounds boxes and fixed-box atom-coverage validation, with regression tests.

## Criticisms That Require New Evidence

The paper does not pretend that text or code can satisfy every evaluator request. The following remain explicit future experiments:

- corrected receptor-specific HuRDA redocking and pose QC;
- a prespecified benchmark of 20–30 diverse RNA-ligand complexes with nonbinders, enrichment/AUC, alternative folds, and RNA-aware rescoring;
- independent, scaffold-separated validation ligands;
- direct RNA binding and SMN2 exon-7 inclusion experiments;
- exact recovery of historical ChEMBL/ZINC release identifiers, retrieval dates, and redistribution licenses.

No synthetic values or expected counts were substituted for these missing data.

## Relationship to the Previous Paper

The SMA motivation, modular pipeline design, target definition, HuRDA concept, generator description, chemical filtering, five-conformer target docking, and cautious clinical boundary are retained. The abstract, ranking interpretation, decoy results, benchmark comparison, discussion, and conclusion were revised where the evaluator findings made the previous wording inaccurate.
