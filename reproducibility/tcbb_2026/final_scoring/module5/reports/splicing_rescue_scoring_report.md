# Splicing-Rescue Scoring Method Report

## Why Binding Alone Is Insufficient

Strong predicted binding to the SMN2 pre-mRNA exon 7 splicing-related RNA target is useful only when it is selective and plausibly compatible with splice-site recognition. A molecule can dock strongly yet still be a poor computational candidate if it binds HuRDA decoys strongly or appears likely to occlude essential donor/acceptor recognition regions.

## Landmark Annotation

Module 5 infers target landmarks from Module 1 target window names, target profile fields, splice-boundary labels, exon 7 distance fields, and metadata. Exact biological contact coordinates are not invented when unavailable.

| landmark_id | landmark_name | window_id | sequence_rna | start_relative_to_exon7 | end_relative_to_exon7 | landmark_type | is_beneficial_target_region | is_caution_region | biological_reason | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SMN2_LANDMARK_001 | exon_7 | SMN2_ENST00000628696.2_exon7 | AUAAUUCCCCCACCACCUCCCAUAUGUCCAGAUUCUCUUGAUGAUGCUGAUGCUUUGGGAAGUAUGUUAAUUUCAUGGUACAUGAGUGGCUAUCAUACUGGCUAUUAUAUG | 0 | 0 | exon_7 | False | False | Annotated SMN2 exon 7 RNA window. | Coordinates are relative/annotated from Module 1 when available; no exact biological coordinate was invented. |
| SMN2_LANDMARK_002 | core_donor_dinucleotide | SMN2_ENST00000628696.2_five_prime_splice_site_core | AUGGUAAGU | 108 | 2 | core_donor_region | False | True | Core 5 prime splice-site window; useful but sensitive to occlusion. | Coordinates are relative/annotated from Module 1 when available; no exact biological coordinate was invented. |
| SMN2_LANDMARK_003 | core_acceptor_dinucleotide | SMN2_ENST00000628696.2_three_prime_splice_site_core | AUAUCUUUUUCUGUCUCCAGAUA | 2 | 108 | core_acceptor_region | False | True | Core 3 prime splice-site window; useful but sensitive to occlusion. | Coordinates are relative/annotated from Module 1 when available; no exact biological coordinate was invented. |
| SMN2_LANDMARK_004 | core_donor_dinucleotide | SMN2_ENST00000628696.2_core_21nt_approx_5prime_splice_region | GCUAUUAUAUGGUAAGUAAUC | 100 | 10 | core_donor_region | False | True | Core 5 prime splice-site window; useful but sensitive to occlusion. | Coordinates are relative/annotated from Module 1 when available; no exact biological coordinate was invented. |
| SMN2_LANDMARK_005 | five_prime_splice_site | SMN2_ENST00000628696.2_local_51nt_approx_5prime_splice_region | GUGGCUAUCAUACUGGCUAUUAUAUGGUAAGUAAUCACUCAGCAUCUUUUC | 85 | 25 | five_prime_splice_site | True | False | 5 prime splice-site-adjacent region relevant to U1-associated splice-site recognition. | Coordinates are relative/annotated from Module 1 when available; no exact biological coordinate was invented. |
| SMN2_LANDMARK_006 | five_prime_splice_site | SMN2_ENST00000628696.2_extended_101nt_approx_5prime_splice_region | AGUAUGUUAAUUUCAUGGUACAUGAGUGGCUAUCAUACUGGCUAUUAUAUGGUAAGUAAUCACUCAGCAUCUUUUCCUGACAAUUUUUUUGUAGUUAUGUG | 60 | 50 | five_prime_splice_site | True | False | 5 prime splice-site-adjacent region relevant to U1-associated splice-site recognition. | Coordinates are relative/annotated from Module 1 when available; no exact biological coordinate was invented. |
| SMN2_LANDMARK_007 | u1_snRNA_recognition_region | SMN2_ENST00000628696.2_five_prime_splice_site_core | AUGGUAAGU | 108 | 2 | u1_snRNA_recognition_region | True | False | Inferred from 5 prime splice-site target window as a U1-associated recognition support region. | Inferred label only; exact U1-contact coordinates require structural/experimental annotation. |

## Pose And Contact Features

When pose files are available, the module parses PDB, PDBQT, SDF, or MOL2 coordinates and calculates geometric contact proxies, hydrogen-bond-like proxies, stacking-like proxies, nearest receptor atom distance, pose support score, and splice-site occlusion penalty.

Pose/contact status counts:

| pose_analysis_status | count |
| --- | --- |
| pose_based_proxy | 147 |

## Feature-Based Fallback

If pose files are unavailable or cannot be parsed, the module uses a feature-based fallback from target docking strength, HuRDA selectivity gap, off-target risk, and landmark availability. Fallback scores are intentionally conservative and recorded with `pose_analysis_status`.

Selectivity evidence note: 0 candidates had missing `weighted_selectivity_gap` values. Missing selectivity values are left blank in numeric outputs and mean that usable real decoy selectivity evidence was not available.

## Predicted Splicing-Rescue Score

The score combines beneficial landmark proximity, U1-support-region proximity, exon 7 boundary support, target binding strength, contact density, hydrogen-bond proxy, stacking proxy, off-target risk penalty, missing-pose penalty, and splice-site occlusion penalty. It is normalized to a 0-1 range.

## Confidence Levels

Dry-run inputs receive very low confidence. Real target-only docking receives low confidence because no off-target RNA counter-screen has been performed. Real target-plus-small-decoy-panel docking receives moderate confidence. High confidence requires real target docking, a large hard-decoy panel with valid decoy scores, pose-based scoring evidence, and ablation analysis. These confidence labels are still computational evidence labels, not experimental validation.

## Warnings

- U1 support landmark was inferred from the 5 prime splice-site window; exact U1 coordinates were not available.

## Limitations

- Docking scores are computational predictions and do not prove experimental binding.
- Predicted splicing-rescue scores are proxy scores, not measured exon 7 inclusion results.
- SRSI is a prioritization metric, not proof of biological activity, safety, or therapeutic effect.
- Top candidates require experimental validation before any biological interpretation.
- Dry-run results are simulated pipeline tests and are not scientific results.
- This module does not provide chemical synthesis instructions or human, animal, or cell testing procedures.
