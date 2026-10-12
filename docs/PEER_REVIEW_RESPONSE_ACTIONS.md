# Peer-Review Response Actions

This document separates repository changes from manuscript edits. All numerical statements below are derived from existing result files; no expected count was substituted for an observed count.

## Changes Implemented In The Repository

1. Real HuRDA decoy docking now requires an existing, nonempty receptor PDBQT file and a ligand PDBQT file. The real `target_plus_decoy_docking` branch validates both inputs and calls AutoDock Vina for each molecule-decoy pair.
2. An automated regression test exercises the real-decoy branch and fails if the decoy receptor is not passed to Vina, if the ligand is not PDBQT, or if the output is marked as a dry-run score.
3. The five-conformer target protocol, prepared receptor panel, exact score tables, source-run configurations, ligand inputs, and failure records are included in the reproducibility snapshot.
4. The partial HuRDA experiment is released with all 14,259 observed score rows, the 741-pair missing-job table, coverage, source-run configurations, preparation summaries, logs, receptor inputs, and ligand inputs.
5. The four-structure redocking protocol and all mode-score, RMSD, alignment, and contact-comparison tables are released with checksums.
6. Module 3 reports an explicit validity denominator, requested-versus-emitted sample counts, training/validation losses, scaffold counts, and sampled fingerprint diversity. It labels the generator as unconditional and autoregressive.
7. The release builder stops if the frozen target, decoy, or redocking counts differ from their observed values. It does not impute missing docking scores.

## Audited Generation Statistics

The frozen production run is `module3_final_final_rigorous_x10`.

| Quantity | Observed value |
|---|---:|
| Model | Unconditional autoregressive SELFIES Transformer decoder |
| Training epochs | 10 |
| Raw training rows loaded | 150,000 |
| Valid training rows | 149,534 |
| Invalid training rows | 466 |
| Unique training rows used | 149,534 |
| Weighted exposure rows | 224,092 |
| ChEMBL valid rows | 74,558 |
| ZINC valid rows | 74,976 |
| RBIND rows used | 0 |
| SMMRNA rows used | 0 |
| Samples requested | 50,000 |
| Unique generated strings emitted after sampler deduplication | 49,660 |
| RDKit-valid emitted strings | 49,660 |
| Filtered molecules retained | 40,477 |
| Rejected molecules | 9,183 |
| Novel filtered molecules under the configured approximation | 40,406 |
| Candidates exported for docking | 1,000 |
| Final training loss | 0.6287246889571854 |
| Final validation loss | 0.5722038051621481 |

The reported 100% validity is the fraction of the 49,660 deduplicated strings emitted by the SELFIES sampler that RDKit parsed. It is not a success rate over all model sampling attempts. Novelty fingerprints were limited to the first 20,000 canonical training molecules, so novelty is an approximation under the recorded configuration.

The July generation run held out and compared against branaplam. Risdiplam was added to the benchmark-control file afterward. The reproducibility builder performs a clearly labeled post hoc two-control similarity audit without changing the historical run or its selected molecules.

The repository contains an n-gram sampling log, but it is not a controlled performance comparison. No claim should be made that the Transformer outperforms n-gram, GRU, random-library sampling, or another generator unless those baselines are rerun under matched sample counts and identical filters.

## Structural Docking Counts

### Target Conformer Panel

- 500 generated molecules
- 5 fixed SMN2 receptor conformers
- 2,500 completed real Vina score rows
- No dry-run scores in the released long table

### HuRDA Decoy Screen

- Intended design: 150 molecules by 100 decoys = 15,000 pairs
- Observed released design: 147 molecules by 97 decoys = 14,259 completed real Vina pairs
- Explicitly missing relative to the intended design: 741 pairs
- Duplicate observed molecule-decoy pairs: 0
- Missing scores imputed: no

The paper should not call this a complete 15,000-job experiment. A newer complete 200 by 100 result is a separate experiment and must be described as such if used.

## Redocking Reconciliation

The apparently conflicting RMSDs refer to different reference definitions:

| Complex | Rank-1 pose vs deposited model 1 | Best RMSD of that same rank-1 pose across deposited ensemble | Best retained mode across all runs |
|---|---:|---:|---:|
| 8R62 risdiplam | 12.587 A | 11.923 A | 2.478 A |
| 8R8P SMN-CX | 7.169 A | 6.986 A | 1.856 A |
| 5KX9 ribocil analogue | 6.173 A | 6.173 A | 4.384 A |
| 8R63 branaplam | 1.307 A | 1.244 A | 1.137 A |

The manuscript must label the reference definition next to each RMSD. The primary interpretation remains mixed: only branaplam had a near-native rank-1 result under the specified criterion, while the other three rank-1 poses failed.

## Manuscript-Only Revisions Required

These changes require access to the paper source and are not made by repository code:

1. Change the title and abstract to describe an unconditional molecular generator followed by RNA virtual screening. Remove wording that implies target-conditioned generation or a diffusion model.
2. Replace “binding free energy” with “Vina docking score” or “predicted docking energy” throughout.
3. State that the ChEMBL/ZINC production run had no RNA-ligand rows from the configured RBIND or SMMRNA sources. Add exact database versions, download dates, licenses, and subset-construction methods from the author records.
4. Replace the 15,000-job decoy claim with the observed 14,259-pair, 147-by-97 partial result and describe the 741 missing pairs.
5. Reconcile the redocking table and prose using the explicitly labeled model-1 and ensemble-reference RMSDs above.
6. Reframe target, selectivity, worst-case gap, contact, and SRSI results as exploratory descriptors. Do not infer target engagement, nonselectivity, exon 7 rescue, toxicity, safety, or efficacy.
7. Fully define every SRSI term, normalization, coefficient, threshold, and missing-value rule in the article. Include the ablation results and acknowledge when target-only ranking performs similarly or better.
8. Define every contact and clash cutoff. Report severe-clash counts and avoid biological interpretation of saturated or nearly constant proxy terms.
9. Add ligand-efficiency and size-controlled analyses before comparing differently sized ligands. Raw Vina scores alone should not support “better binder” language.
10. Discuss the extreme-value sensitivity of the worst-case gap to decoy-panel size, receptor-panel size, and scoring noise. The failure of known controls under this metric prevents treating it as validated selectivity.
11. Expand validation to a prespecified, broader RNA-ligand benchmark with seed, receptor-preparation, ligand-property, and method sensitivity analyses before making predictive claims.
12. Narrow the conclusion to a reproducible computational workflow and docking-methodology caution unless direct RNA binding and exon 7 inclusion experiments are added.

## Suggested Conclusion Scope

The current evidence supports a reproducible workflow for generating and structurally counter-screening computational molecules, together with a caution that favorable mean docking scores can coexist with poor conservative separation and failed pose recovery. It does not establish SMN2 binding, selective RNA recognition, exon 7 rescue, therapeutic activity, or clinical superiority over benchmark splicing modifiers.
