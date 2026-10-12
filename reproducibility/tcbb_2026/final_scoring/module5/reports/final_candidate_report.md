# SMN2-RescueDiff Final Candidate Report

## Predicted Splicing-Rescue Scoring And Final SRSI Ranking

## Purpose

This final module integrates SMN2 pre-mRNA target definition, HuRDA RNA decoy counter-screening, generated molecule features, and docking/selectivity outputs to rank computational candidate molecules by SRSI: Splicing Rescue Selectivity Index.

The top-ranked molecules identified by SMN2-RescueDiff are computational candidates only. Their rankings are based on predicted SMN2 RNA binding, HuRDA off-target counter-screening, predicted splicing-rescue proxy scoring, and chemical property filtering. These candidates are not experimentally validated SMN2 splicing modifiers and should be interpreted as hypotheses for future validation.

## Inputs From Modules 1-4

- Module 1 target FASTA/profile/metadata were used to define SMN2 exon 7 splicing-related RNA landmarks.
- Module 2 HuRDA metadata were used as transcriptome-scale decoy context.
- Module 3 molecule features provided chemical quality, novelty, and pre-docking candidate evidence.
- Module 4 docking/selectivity outputs provided target binding and HuRDA off-target evidence.

## Molecules Evaluated

- Candidates loaded: 147
- Candidates scored: 147
- High-priority computational candidates: 0
- Medium-priority computational candidates: 3
- Low-priority computational candidates: 71

## SRSI Method

SRSI combines target binding strength, predicted splicing-rescue support, weighted HuRDA selectivity gap, QED, Lipinski-style chemical quality, pre-docking candidate score, off-target risk, synthetic-accessibility proxy when available, PAINS alerts, and splice-site occlusion penalty. Because docking energies are more negative for stronger predicted binding, SMN2 binding strength is calculated as `-1 * target_deltaG`.

Binding alone is intentionally insufficient. A candidate with strong SMN2 target docking can rank lower if HuRDA decoy binding is also strong, if predicted splice-site occlusion is high, or if the splicing-rescue proxy score is weak.

Missing selectivity handling: 0 ranked candidates have missing `weighted_selectivity_gap` values. Blank selectivity fields indicate that no usable real decoy selectivity value was available for that molecule; values were not fabricated.

## Predicted Splicing-Rescue Proxy

The predicted splicing-rescue score rewards target binding near beneficial exon 7 splicing-related landmarks, U1-support regions, exon 7 boundary-support windows, contact-density proxies when poses are available, and favorable selectivity. It penalizes off-target risk, missing pose evidence, dry-run inputs, and potential splice-site occlusion. It is a computational proxy score, not a measured rescue result.

## Top 10 Candidates

| final_rank | molecule_id | target_deltaG | weighted_selectivity_gap | off_target_risk_score | predicted_splicing_rescue_score | nearest_benchmark_control_name | nearest_benchmark_tanimoto | srsi_score | candidate_tier | confidence_level |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | MolGen_048239 | -8.4194 | 6.266287 | 0.460658 | 0.359175 | branaplam | 0.096774 | 24.113413 | medium_priority_computational_candidate | moderate |
| 2 | MolGen_038292 | -8.928999999999998 | 6.621515 | 0.581158 | 0.376852 | branaplam | 0.094737 | 20.829101 | medium_priority_computational_candidate | moderate |
| 3 | MolGen_038175 | -8.7186 | 6.419453 | 0.657216 | 0.350244 | branaplam | 0.115789 | 14.16985 | medium_priority_computational_candidate | moderate |
| 4 | MolGen_048471 | -8.4502 | 6.278412 | 0.596466 | 0.340064 | branaplam | 0.108696 | 12.957373 | low_priority_computational_candidate | moderate |
| 5 | MolGen_030154 | -7.7244 | 5.73157 | 0.492566 | 0.303171 | branaplam | 0.131868 | 12.777051 | low_priority_computational_candidate | moderate |
| 6 | MolGen_020104 | -8.428799999999999 | 6.179685 | 0.657216 | 0.328983 | branaplam | 0.119048 | 10.926726 | low_priority_computational_candidate | moderate |
| 7 | MolGen_021804 | -8.0608 | 6.106748 | 0.579058 | 0.314261 | branaplam | 0.103448 | 10.744834 | low_priority_computational_candidate | moderate |
| 8 | MolGen_022148 | -8.656 | 6.390203 | 0.657216 | 0.345611 | branaplam | 0.09 | 9.977359 | low_priority_computational_candidate | moderate |
| 9 | MolGen_030809 | -8.422799999999999 | 6.149882 | 0.660825 | 0.327971 | branaplam | 0.1 | 9.944393 | low_priority_computational_candidate | moderate |
| 10 | MolGen_047257 | -8.6 | 6.239625 | 0.657216 | 0.341534 | branaplam | 0.1 | 9.739493 | low_priority_computational_candidate | moderate |

## Held-Out Benchmark Controls

Benchmark controls were not used for model training.

Known splicing modifiers provided through `inputs.benchmark_controls.splicing_controls` are reported separately as reference baselines when Module 4 docking or selectivity scores are available for them. These comparisons are limited to computational values such as docking score, selectivity gap, off-target-risk proxy, and SRSI. They do not show that any generated molecule is clinically better than risdiplam or another benchmark control.

| benchmark_control_name | target_deltaG | weighted_selectivity_gap | selectivity_gap | off_target_risk_score | srsi_score | confidence_level | reference_status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| risdiplam | NA | NA | NA | NA | NA | NA | no_module4_scores_available |
| branaplam | NA | NA | NA | NA | NA | NA | no_module4_scores_available |

## Candidate Evidence Summaries

- `MolGen_048239` rank 1: Strong predicted SMN2 target binding; Positive weighted selectivity gap against HuRDA decoys; Acceptable Lipinski-style chemical property profile. Concerns: Computational prediction only; Requires experimental validation before biological interpretation.
- `MolGen_038292` rank 2: Strong predicted SMN2 target binding; Positive weighted selectivity gap against HuRDA decoys; Acceptable Lipinski-style chemical property profile. Concerns: Computational prediction only; Requires experimental validation before biological interpretation.
- `MolGen_038175` rank 3: Strong predicted SMN2 target binding; Positive weighted selectivity gap against HuRDA decoys; Acceptable Lipinski-style chemical property profile. Concerns: Computational prediction only; Requires experimental validation before biological interpretation; Elevated HuRDA off-target risk proxy.
- `MolGen_048471` rank 4: Strong predicted SMN2 target binding; Positive weighted selectivity gap against HuRDA decoys; Acceptable Lipinski-style chemical property profile. Concerns: Computational prediction only; Requires experimental validation before biological interpretation.
- `MolGen_030154` rank 5: Strong predicted SMN2 target binding; Positive weighted selectivity gap against HuRDA decoys; Acceptable Lipinski-style chemical property profile. Concerns: Computational prediction only; Requires experimental validation before biological interpretation.
- `MolGen_020104` rank 6: Strong predicted SMN2 target binding; Positive weighted selectivity gap against HuRDA decoys; Acceptable Lipinski-style chemical property profile. Concerns: Computational prediction only; Requires experimental validation before biological interpretation; Elevated HuRDA off-target risk proxy.
- `MolGen_021804` rank 7: Strong predicted SMN2 target binding; Positive weighted selectivity gap against HuRDA decoys; Acceptable Lipinski-style chemical property profile. Concerns: Computational prediction only; Requires experimental validation before biological interpretation.
- `MolGen_022148` rank 8: Strong predicted SMN2 target binding; Positive weighted selectivity gap against HuRDA decoys; Acceptable Lipinski-style chemical property profile. Concerns: Computational prediction only; Requires experimental validation before biological interpretation; Elevated HuRDA off-target risk proxy.
- `MolGen_030809` rank 9: Strong predicted SMN2 target binding; Positive weighted selectivity gap against HuRDA decoys; Acceptable Lipinski-style chemical property profile. Concerns: Computational prediction only; Requires experimental validation before biological interpretation; Elevated HuRDA off-target risk proxy.
- `MolGen_047257` rank 10: Strong predicted SMN2 target binding; Positive weighted selectivity gap against HuRDA decoys; Acceptable Lipinski-style chemical property profile. Concerns: Computational prediction only; Requires experimental validation before biological interpretation; Elevated HuRDA off-target risk proxy.

## Ablation Analysis

The ablation comparison contrasts binding-only ranking with rankings that include selectivity, splicing-rescue proxy evidence, and full SRSI. It is designed to show why strong target docking alone is not sufficient for final prioritization.

| final_rank | molecule_id | target_deltaG | weighted_selectivity_gap | predicted_splicing_rescue_score | srsi_score |
| --- | --- | --- | --- | --- | --- |
| 1 | MolGen_048239 | -8.4194 | 6.266287 | 0.359175 | 24.113413 |
| 2 | MolGen_038292 | -8.928999999999998 | 6.621515 | 0.376852 | 20.829101 |
| 3 | MolGen_038175 | -8.7186 | 6.419453 | 0.350244 | 14.16985 |
| 4 | MolGen_048471 | -8.4502 | 6.278412 | 0.340064 | 12.957373 |
| 5 | MolGen_030154 | -7.7244 | 5.73157 | 0.303171 | 12.777051 |
| 6 | MolGen_020104 | -8.428799999999999 | 6.179685 | 0.328983 | 10.926726 |
| 7 | MolGen_021804 | -8.0608 | 6.106748 | 0.314261 | 10.744834 |
| 8 | MolGen_022148 | -8.656 | 6.390203 | 0.345611 | 9.977359 |
| 9 | MolGen_030809 | -8.422799999999999 | 6.149882 | 0.327971 | 9.944393 |
| 10 | MolGen_047257 | -8.6 | 6.239625 | 0.341534 | 9.739493 |

## Output Files

- `data/processed/final_scoring/stage_c_147/runs/module5_stage_c_147_pose_contacts/tables/smn2_splicing_landmarks.tsv`
- `data/processed/final_scoring/stage_c_147/runs/module5_stage_c_147_pose_contacts/tables/pose_contact_features.tsv`
- `data/processed/final_scoring/stage_c_147/runs/module5_stage_c_147_pose_contacts/tables/splicing_rescue_scores.tsv`
- `data/processed/final_scoring/stage_c_147/runs/module5_stage_c_147_pose_contacts/tables/final_srsi_ranked_candidates.tsv`
- `data/processed/final_scoring/stage_c_147/runs/module5_stage_c_147_pose_contacts/tables/top_10_final_candidates.tsv`
- `data/processed/final_scoring/stage_c_147/runs/module5_stage_c_147_pose_contacts/tables/benchmark_control_reference_scores.tsv`
- `data/processed/final_scoring/stage_c_147/runs/module5_stage_c_147_pose_contacts/tables/ablation_rank_comparison.tsv`
- `data/processed/final_scoring/stage_c_147/runs/module5_stage_c_147_pose_contacts/evidence/top_candidate_evidence.json`
- `data/processed/final_scoring/stage_c_147/runs/module5_stage_c_147_pose_contacts/evidence/scoring_warnings.json`
- `data/processed/final_scoring/stage_c_147/runs/module5_stage_c_147_pose_contacts/final_scoring_summary.json`
- `data/processed/final_scoring/stage_c_147/runs/module5_stage_c_147_pose_contacts/splicing_rescue_scoring_config_used.yaml`
- `data/processed/final_scoring/stage_c_147/runs/module5_stage_c_147_pose_contacts/logs/module5_run_log.txt`
- `data/processed/final_scoring/stage_c_147/runs/module5_stage_c_147_pose_contacts/reports/splicing_rescue_scoring_report.md`
- `data/processed/final_scoring/stage_c_147/runs/module5_stage_c_147_pose_contacts/reports/final_candidate_report.md`
- `data/processed/final_scoring/stage_c_147/runs/module5_stage_c_147_pose_contacts/figures/splicing_rescue_score_distribution.png`
- `data/processed/final_scoring/stage_c_147/runs/module5_stage_c_147_pose_contacts/figures/srsi_score_distribution.png`
- `data/processed/final_scoring/stage_c_147/runs/module5_stage_c_147_pose_contacts/figures/target_binding_vs_splicing_rescue_scatter.png`
- `data/processed/final_scoring/stage_c_147/runs/module5_stage_c_147_pose_contacts/figures/selectivity_gap_vs_splicing_rescue_scatter.png`
- `data/processed/final_scoring/stage_c_147/runs/module5_stage_c_147_pose_contacts/figures/off_target_risk_vs_srsi_scatter.png`
- `data/processed/final_scoring/stage_c_147/runs/module5_stage_c_147_pose_contacts/figures/top_10_candidate_score_breakdown.png`
- `data/processed/final_scoring/stage_c_147/runs/module5_stage_c_147_pose_contacts/figures/top_candidate_heatmap.png`
- `data/processed/final_scoring/stage_c_147/runs/module5_stage_c_147_pose_contacts/figures/ablation_rank_shift_plot.png`
- `data/processed/final_scoring/stage_c_147/runs/module5_stage_c_147_pose_contacts/figures/candidate_confidence_summary.png`

## Warnings

- U1 support landmark was inferred from the 5 prime splice-site window; exact U1 coordinates were not available.

## Limitations

- Docking scores are computational predictions and do not prove experimental binding.
- Predicted splicing-rescue scores are proxy scores, not measured exon 7 inclusion results.
- SRSI is a prioritization metric, not proof of biological activity, safety, or therapeutic effect.
- Top candidates require experimental validation before any biological interpretation.
- Dry-run results are simulated pipeline tests and are not scientific results.
- This module does not provide chemical synthesis instructions or human, animal, or cell testing procedures.

## Conclusion

Module 5 produces a final ranked list of computational candidate molecules with SMN2 RNA target binding evidence, HuRDA off-target selectivity evidence, predicted splicing-rescue proxy evidence, chemical quality evidence, SRSI score, candidate tier, confidence level, and evidence summary. These outputs are hypotheses for future validation and are not proof of SMN2 splicing rescue, safety, or therapeutic activity.
