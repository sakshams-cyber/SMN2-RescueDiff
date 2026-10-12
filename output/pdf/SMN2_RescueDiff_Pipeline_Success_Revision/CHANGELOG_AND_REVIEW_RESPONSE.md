# Manuscript change log and reviewer-response ledger

## Revision objective

The original manuscript was revised to present the principal result as an auditable, executable candidate-prioritization pipeline rather than as a failed attempt to establish docking selectivity. The revision preserves most of the original Methods, Results, equations, staged workflow, and scientific narrative. It does not convert computational scores into claims of binding, splicing rescue, safety, or clinical benefit.

## Preservation audit

A word-level ordered comparison between the original Overleaf source and this revision found that 7,676 of the original 9,737 source words remain in matching order, or 78.8% of the original text. The revised source contains 10,315 words. The additions are concentrated in the audited generation statistics, matched-control analysis, SRSI ablation, redocking validation, reproducibility record, and reviewer-requested limitations; this was a selective revision, not a wholesale rewrite.

## Major changes made

1. **Title and contribution.** Replaced the diffusion-model and drug-discovery framing with an accurate description of an unconditional autoregressive generator and RNA-decoy docking pipeline.
2. **Abstract.** Added audited generation, target-docking, decoy-docking, matched-control, and redocking results. The abstract now states both the demonstrated pipeline successes and the calibration limits.
3. **Generator description.** Identified the model as an unconditional autoregressive SELFIES Transformer. Removed target-conditioned and diffusion-model implications.
4. **Training-data provenance.** Reported 150,000 inspected source rows, 149,534 valid unique ChEMBL/ZINC molecules, 466 invalid rows, and 224,092 weighted training exposures. Disclosed that configured R-BIND and SMMRNA sources contributed zero rows.
5. **Generation audit.** Added the requested-versus-observed sequence counts: 50,000 requested, 49,660 unique strings emitted, 49,489 unique canonical molecules, 40,477 filter-pass molecules, 9,183 rejected molecules, 40,406 approximately novel molecules under the recorded 20,000-fingerprint comparison, and 1,000 exported molecules.
6. **Diversity audit.** Added 977 scaffold labels among the 1,000 exported molecules and the sampled pairwise Morgan Tanimoto mean/median of 0.121/0.115.
7. **Benchmark-control separation.** Stated explicitly that risdiplam and branaplam were held-out controls and were not used for training. Retained them as computational reference baselines only.
8. **Docking terminology.** Replaced unsupported affinity and binding-energy language with `AutoDock Vina score`, `comparative docking score`, or `predicted score` where appropriate.
9. **Target docking completeness.** Reported completion of all 2,500 molecule-conformer jobs for 500 molecules across five fixed SMN2 receptors.
10. **Hard-decoy audit.** Reported 15,000 intended molecule-decoy jobs, 14,259 completed real-Vina jobs, 741 missingness records, and the complete 147-by-97 candidate-decoy matrix used for uniform comparisons.
11. **Matched ligand-decoy controls.** Added the complementary size/profile-control analysis: 40 candidates, ten matched controls per candidate, 183 distinct controls, mean candidate enrichment of 0.729 kcal/mol, and an observed cohort mean exceeding all 2,000 random cohorts.
12. **SRSI ablation.** Added full-versus-component rank correlations, top-ten overlap, top-forty overlap, median rank shift, and maximum rank shift. The text now states that the analysis demonstrates rank sensitivity, not biological superiority of the composite score.
13. **Redocking validation.** Reconciled and reported 45 deposited receptor models, five seeds, 225 completed jobs, and 4,488 retained poses. Added separate model-1 rank-1 RMSD, ensemble-minimum rank-1 RMSD, best sampled RMSD, rank of best mode, contact recall, and contact Jaccard values.
14. **RMSD discrepancy.** Explained that branaplam's 1.307 A value is model-1 rank-1 RMSD, whereas 1.244 A is the minimum rank-1 RMSD across the deposited ensemble. They are no longer presented as contradictory values.
15. **Contact-feature formulas.** Added distance cutoffs, normalization equations, feature weights, and the fixed beneficial/caution fractions. Labeled these quantities as engineering proxies rather than mechanistic contacts.
16. **Variance definition.** Defined the target-conformer standard deviation with denominator `K-1` and removed the previously undefined symbol.
17. **Clinical wording.** Corrected the SPR1NT description from prenatal treatment to presymptomatic treatment initiated within six weeks after birth.
18. **Figures.** Regenerated the pipeline diagram, generation audit, Stage C selectivity profile, benchmark comparison, matched-control enrichment, and SRSI ablation. Figures now use Vina-score labels rather than measured-affinity notation.
19. **Discussion and conclusion.** Reframed the study's success as a reproducible, traceable, large-scale pipeline that narrows candidates and exposes failure modes. Retained explicit statements that target enrichment does not prove binding and that worst-case target-decoy separation was weak.
20. **Reproducibility.** Added the public branch, exact commit, release contents, software versions, logs, checksums, receptor files, PDBQT inputs, complete score tables, and failure records.
21. **AI-assistance disclosure.** Expanded the acknowledgment to identify ChatGPT, Claude, OpenAI Codex, and Grammarly roles while assigning responsibility for claims to the author.

## Reviewer criticisms followed through

| Criticism | Action in this revision | Status |
|---|---|---|
| The claimed diffusion model did not match the implementation. | Corrected the title, abstract, and Methods to `unconditional autoregressive SELFIES Transformer`; removed diffusion claims. | Resolved |
| Target-conditioned generation was not demonstrated. | Explicitly states that generation was not conditioned on SMN2, structure, docking, or transcriptomic features. | Resolved by correction |
| The manuscript did not provide reproducible code/data access. | Added public branch, exact commit, release path, inputs, logs, complete tables, failures, and checksums. | Resolved for the audited snapshot |
| Generation counts and source balance were unclear. | Added complete source, training, generation, filtering, novelty, diversity, and export counts. | Resolved |
| R-BIND/SMMRNA use was overstated. | Disclosed that those configured sources contributed zero rows; conclusions no longer attribute results to RNA-ligand-focused training. | Resolved |
| Benchmark compounds may have leaked into training. | Documents held-out processing and states that benchmark controls were not used for training. | Resolved for recorded inputs |
| Docking was described as binding affinity or free energy. | Replaced the language with Vina score and computational prioritization terminology. | Resolved |
| The decoy analysis was incomplete and panel size was unclear. | Reports the planned and completed job counts, exact complete matrix, and panel-size limitations. | Resolved descriptively |
| The decoy panel cannot establish transcriptome-wide selectivity. | Removed transcriptome-wide selectivity claims and labels the analysis a staged pilot panel. | Resolved by limiting the claim |
| Raw Vina comparisons can be confounded by ligand size and receptor composition. | Added explicit caveats and a matched ligand-control analysis; calls for ligand-efficiency and receptor-normalized follow-up. | Partially addressed |
| SRSI weights and thresholds were insufficiently justified. | Added exact anchors, tolerances, equations, and weights; labels them exploratory rather than calibrated. | Addressed transparently; calibration remains open |
| SRSI's incremental value was not tested. | Added four-configuration ablation and rank-stability analysis. | Resolved computationally; biological value remains untested |
| Contact features appeared ad hoc and possibly saturated. | Added implementation details and disclosed saturation/clash risks and non-mechanistic interpretation. | Resolved descriptively |
| The manuscript contradicted itself about redocking. | Replaced the contradiction with a dedicated redocking Methods/Results section. | Resolved |
| Pose validation used too few systems and inconsistent RMSDs. | Added four-complex, five-seed, ensemble-aware results and separated model-1 from ensemble-minimum metrics. | Partially addressed |
| A successful docking example was missing. | Branaplam rank-1 recovery at 1.307 A with 81% contact recall is reported alongside three failures. | Resolved without hiding failures |
| Risdiplam's lower-RMSD pose was sampled but ranked poorly. | Reports rank-14 recovery and interprets it as a scoring/reranking limitation. | Resolved transparently |
| Matched controls were reused and statistical independence was overstated. | Reports 183 distinct controls across 400 assignments and describes reuse as a dependence limitation. | Resolved descriptively |
| The matched-control null was not an external-library baseline. | States that the random cohort and target-only analyses address selection effects but are not an external chemical-library control. | Resolved by limiting the claim |
| The model architecture lacked controlled baselines. | Added this as a limitation and removed claims that performance derives from the Transformer architecture. | Unresolved experimentally |
| Training release identifiers, download dates, and licenses were missing. | Explicitly records that the audited run did not capture them rather than inventing provenance. | Unresolved; requires source-record recovery |
| Novelty was computed against only part of the corpus. | Describes the exact first-20,000-fingerprint approximation and avoids a full-corpus novelty claim. | Resolved by correction |
| A broad 20-30 complex RNA-docking benchmark was needed. | Added as a prespecified next step. | Unresolved; requires new docking study |
| Alternative RNA-aware scoring or rescoring was needed. | Redocking identifies sampled-but-misranked poses and motivates RNA-aware reranking. | Unresolved; requires method development |
| Ligand-efficiency and receptor-normalized scores were needed. | Added to limitations and next steps. | Unresolved; requires new analysis |
| Wet-lab validation was absent. | The revised paper states that binding, exon inclusion, cellular efficacy, toxicity, and therapeutic performance remain untested. | Unresolved; requires experiments |
| The SPR1NT trial was mischaracterized as prenatal. | Corrected to presymptomatic postnatal treatment within six weeks after birth. | Resolved |
| The paper used `validated` too broadly. | Uses `tested`, `calibrated`, `recovered`, or `computationally evaluated` and reserves validation claims for their exact scope. | Resolved |

## Criticisms that cannot be closed by manuscript editing

The following require new evidence and are intentionally not presented as completed work:

- biochemical measurement of candidate-SMN2 RNA binding;
- cellular SMN2 exon 7 inclusion assays;
- cytotoxicity, pharmacology, or in vivo studies;
- a preregistered, larger RNA-ligand pose benchmark;
- controlled n-gram, recurrent-network, library-sampling, or alternative-generator baselines;
- RNA-aware pose rescoring and calibrated ligand-efficiency/receptor-normalized comparisons;
- cluster-aware uncertainty with independent external chemical-library controls;
- recovery of exact ChEMBL/ZINC source release identifiers, download dates, and license records for the historical training run.

## Integrity statement

No docking value was fabricated, imputed, or altered to make the pipeline appear successful. Expected counts were treated as validation targets, discrepancies were reported, and previous successful outputs were preserved. The positive framing rests on reproducible execution, candidate narrowing, target-score enrichment, quality-control gates, and transparent failure analysis—not on unmeasured biological efficacy.
