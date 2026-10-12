#!/usr/bin/env python3
"""Build REPORT.md from written validation artifacts and record provenance."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "REPORT.md"
PROVENANCE_DIR = ROOT / "data/processed/manuscript_validation_report_20261011"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Required report input is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def main() -> int:
    if REPORT.exists():
        raise FileExistsError(f"Refusing to overwrite existing report: {REPORT}")
    if PROVENANCE_DIR.exists():
        raise FileExistsError(f"Refusing to write into existing output directory: {PROVENANCE_DIR}")

    task1_result_path = ROOT / "data/processed/docking/stage_c5_selection_null_seed61453_r2000_20261011/stage_c5_selection_null.json"
    task1_provenance_path = ROOT / "data/processed/docking/stage_c5_selection_null_seed61453_r2000_20261011/run_provenance.json"
    task2_metrics_path = ROOT / "data/processed/final_scoring/stage_c_147/task2_validation_metrics.json"
    task2_staging_path = ROOT / "data/processed/final_scoring/stage_c_147_inputs_20261011/staging_summary.json"
    task2_provenance_path = ROOT / "data/processed/final_scoring/stage_c_147/run_provenance.json"
    reference_config_path = ROOT / "data/processed/final_scoring/judge_grade_200x100/splicing_rescue_scoring_config_used.yaml"
    stage_c_config_path = ROOT / "configs/splicing_rescue_scoring_stageC147.yaml"
    report_inputs = [
        task1_result_path,
        task1_provenance_path,
        task2_metrics_path,
        task2_staging_path,
        task2_provenance_path,
        reference_config_path,
        stage_c_config_path,
    ]
    for path in report_inputs:
        if not path.is_file():
            raise FileNotFoundError(f"Required report input is missing: {path}")

    task1 = load_json(task1_result_path)
    task1_provenance = load_json(task1_provenance_path)
    task2 = load_json(task2_metrics_path)
    task2_staging = load_json(task2_staging_path)
    task2_provenance = load_json(task2_provenance_path)
    reference_config = yaml.safe_load(reference_config_path.read_text(encoding="utf-8"))
    stage_c_config = yaml.safe_load(stage_c_config_path.read_text(encoding="utf-8"))

    config_checks = {
        "pose_analysis": reference_config["pose_analysis"] == stage_c_config["pose_analysis"],
        "splicing_rescue_scoring.weights": (
            reference_config["splicing_rescue_scoring"]["weights"]
            == stage_c_config["splicing_rescue_scoring"]["weights"]
        ),
        "splicing_rescue_scoring.thresholds": (
            reference_config["splicing_rescue_scoring"]["thresholds"]
            == stage_c_config["splicing_rescue_scoring"]["thresholds"]
        ),
        "srsi.weights": reference_config["srsi"]["weights"] == stage_c_config["srsi"]["weights"],
    }
    if not all(config_checks.values()):
        raise ValueError(f"Stage C config changed a protected reference block: {config_checks}")

    quantile_lines = "\n".join(
        f"| {key} | {value(number)} |" for key, number in task1["null_quantiles"].items()
    )
    feature_lines = "\n".join(
        "| {name} | {median} | {minimum} | {maximum} | {span} | {missing} |".format(
            name=name,
            median=value(stats["median"]),
            minimum=value(stats["minimum"]),
            maximum=value(stats["maximum"]),
            span=value(stats["range"]),
            missing=value(stats["count_missing"]),
        )
        for name, stats in task2["feature_statistics"].items()
    )
    status_lines = "\n".join(
        f"| {name} | {count} |" for name, count in task2["pose_analysis_status_distribution"].items()
    )
    input_hash_lines = "\n".join(
        f"| `{item['path']}` | `{item['sha256']}` |" for item in task1_provenance["input_files"]
    )

    t1_pass_a = task1["null_pool_size"] == 460
    t1_pass_b = task1["replicates_completed"] == 2000
    t1_pass_c = abs(task1["null_mean"]) <= 0.15
    t1_pass_d = task1["observed_mean_gap"] == 0.7292

    occ = task2["occlusion_near_degeneracy"]
    text = f"""# SMN2-RescueDiff Manuscript Validation Report

This report records two completed analyses and one research-only scoping task requested for manuscript review. No docking score, descriptor, or structure was imputed. No new AutoDock Vina docking was run.

## Task 1: Stage C.5 Selection-Effect Null

### What was run

RDKit was verified before analysis:

```bash
.venv/bin/python -c "from rdkit import Chem; print('ok')"
```

The requested destination was already an existing output directory. To comply with the non-overwrite rule, the analytical parameters were left unchanged and only the output path was moved to a new directory:

```bash
MPLCONFIGDIR=/tmp/smn2-rescuediff-mpl .venv/bin/python stage_c5_null.py \\
  --selected-molecules data/processed/docking/target_conformer_runs/target5_500_v1_complete_merged/selected_molecules.tsv \\
  --target-summary data/processed/docking/target_conformer_runs/target5_500_v1_complete_merged/target_conformer_robustness_scores_final.tsv \\
  --top40 data/processed/docking/stage_c_selectivity_150_partial_after_target5/top40_for_ligand_decoy_control_audit.tsv \\
  --replicates 2000 \\
  --out data/processed/docking/stage_c5_selection_null_seed61453_r2000_20261011/stage_c5_selection_null.json
```

The Morgan guard remained enabled with a Tanimoto ceiling of {value(task1['max_tanimoto'])}, radius {value(task1['morgan_radius'])}, and {value(task1['morgan_bits'])} bits. The random seed was {value(task1['random_seed'])}. The git commit was `{task1_provenance['git_commit']}`.

### Sanity checks

| Check | Observed | Result |
|---|---:|---|
| Null pool size is 460 | {value(task1['null_pool_size'])} | {'PASS' if t1_pass_a else 'FAIL'} |
| Completed replicates are 2000 | {value(task1['replicates_completed'])} | {'PASS' if t1_pass_b else 'FAIL'} |
| Failed-to-match replicates | {value(task1['replicates_failed_to_match'])} | {'PASS' if task1['replicates_failed_to_match'] == 0 else 'FAIL'} |
| Null mean is within +/-0.15 | {value(task1['null_mean'])} | {'PASS' if t1_pass_c else 'FAIL'} |
| Observed mean gap is 0.7292 | {value(task1['observed_mean_gap'])} | {'PASS' if t1_pass_d else 'FAIL'} |

### Results

- Null mean: {value(task1['null_mean'])}
- Null standard deviation: {value(task1['null_sd'])}
- Minimum cohort mean: {value(task1['null_min'])}
- Maximum cohort mean: {value(task1['null_max'])}
- Observed mean gap: {value(task1['observed_mean_gap'])}
- Observed z-score versus null: {value(task1['observed_z_vs_null'])}
- Cohorts at or above observed: {value(task1['cohorts_at_or_above_observed'])}
- Empirical p-value, observed greater than or equal to null: {value(task1['empirical_p_observed_ge_null'])}
- Replicates completed: {value(task1['replicates_completed'])}

| Null quantile | Value |
|---:|---:|
{quantile_lines}

The empirical p-value uses the script's finite-sampling correction; zero of {value(task1['replicates_completed'])} sampled cohorts reached the observed gap. One matching descriptor, `formal_charge`, had zero variance in the null pool and was excluded from standardized distance matching by the supplied script. Parameters were not changed in response.

### Provenance

| Input | SHA-256 |
|---|---|
{input_hash_lines}

Output JSON SHA-256: `{task1_provenance['output_files'][0]['sha256']}`.

## Task 2: Pose Contact Features for the Stage C Cohort

### Investigation

1. `pose_parser.py` does not search `inputs.pose_dir`. `parse_pose_for_candidate` reads `pose_file` (or `ligand_file`) and `receptor_file` directly from each candidate row. The existing loader therefore cannot map a molecule ID into the five nested conformer directories by itself.
2. `ensemble_selectivity_scores.tsv` has `molecule_id` and `weighted_mean_selectivity_gap`; the loader already maps the latter to `weighted_selectivity_gap`. It does not map `mean_target_deltaG` to `target_deltaG`, `ensemble_off_target_risk_score` to `off_target_risk_score`, or `decoys_valid` to the valid-decoy count fields. An explicit adapter was required.
3. Copying poses into a flat directory was unnecessary. The minimal fix was a 147-row target-score adapter containing explicit source `pose_file` and `receptor_file` paths. `SMN2_CONF_01` was fixed for every molecule to preserve one receptor geometry and avoid per-molecule, score-driven pose selection. The five-conformer `mean_target_deltaG` remained the target score used by Module 5.

The original Stage C table contained {value(task2_staging['ensemble_rows'])} rows. Exactly {value(task2_staging['cohort_rows'])} had numeric `conservative_selectivity_gap`; the excluded IDs were `{', '.join(task2_staging['excluded_missing_conservative_selectivity_gap'])}`. Each retained molecule had {value(task2_staging['decoys_per_molecule'])} completed real decoy scores, for {value(task2_staging['decoy_rows_validated'])} validated decoy rows. No missing values were imputed.

The protected YAML sections matched the reference config exactly: `pose_analysis` PASS; `splicing_rescue_scoring.weights` PASS; `splicing_rescue_scoring.thresholds` PASS; `srsi.weights` PASS.

### What was run

```bash
PYTHONPATH=src .venv/bin/python scripts/prepare_stage_c147_module5_inputs.py \\
  --config configs/splicing_rescue_scoring_stageC147.yaml \\
  --output-dir data/processed/final_scoring/stage_c_147_inputs_20261011
```

```bash
MPLCONFIGDIR=/tmp/smn2-rescuediff-mpl PYTHONPATH=src .venv/bin/python \\
  -m smn2_rescuediff.splicing_rescue.cli \\
  --config configs/splicing_rescue_scoring_stageC147.yaml
```

Config: `configs/splicing_rescue_scoring_stageC147.yaml`. Stable output: `data/processed/final_scoring/stage_c_147/tables/pose_contact_features.tsv`. Run-specific output: `data/processed/final_scoring/stage_c_147/runs/module5_stage_c_147_pose_contacts/tables/pose_contact_features.tsv`.

### Validation

- Output rows: {value(task2['row_count'])}
- Unique molecule IDs: {value(task2['unique_molecule_count'])}
- IDs exactly equal the Stage C 147: {value(task2['pose_ids_equal_stage_c_147'])}
- Unexpected IDs: {value(task2['unexpected_pose_ids'])}
- Missing Stage C IDs: {value(task2['missing_stage_c_ids'])}
- Earlier 200-candidate rows: {value(task2['earlier_200_row_count'])}
- Molecule overlap with the earlier 200-candidate pose table: {value(task2['earlier_200_overlap_count'])}

| pose_analysis_status | Count |
|---|---:|
{status_lines}

### Contact-feature distributions

All values below were computed from the written 147-row TSV. Range is maximum minus minimum.

| Feature | Median | Minimum | Maximum | Range | Missing |
|---|---:|---:|---:|---:|---:|
{feature_lines}

The occlusion near-degeneracy holds. In the Stage C cohort it spans {value(occ['stage_c_147_minimum'])} to {value(occ['stage_c_147_maximum'])}, a range of {value(occ['stage_c_147_range'])}. The earlier 200-candidate table has the same observed minimum, maximum, and range. Despite {value(occ['stage_c_147_unique_values'])} distinct rounded values, this feature remains too narrowly distributed to contribute meaningful rank discrimination under ordinary normalization without careful sensitivity analysis. The `pose_support_score` also saturates at {value(task2['feature_statistics']['pose_support_score']['maximum'])}, with a median of {value(task2['feature_statistics']['pose_support_score']['median'])}; that is another limitation of the current proxy construction.

These contact features are geometric proxies based on ligand-receptor atom-pair distances. They are not experimentally observed contacts and do not establish splicing rescue.

### Provenance

- Random seed: {value(task2_provenance['random_seed'])}
- Git commit: `{task2_provenance['git_commit']}`
- Config SHA-256: `{task2_provenance['config_used']['sha256']}`
- Input files hashed: {value(task2_provenance['input_file_sha256_manifest']['files_hashed'])}
- Input hash manifest: `{task2_provenance['input_file_sha256_manifest']['path']}`
- Input hash manifest SHA-256: `{task2_provenance['input_file_sha256_manifest']['sha256']}`
- Pose-contact TSV SHA-256: `{task2['pose_contact_features_sha256']}`

## Task 3: Redocking Validation Scope

### Search performed

Research only; no docking or coordinate preparation was run. Search date: 2026-10-11. Local random seed: not applicable. Local input files: none. Git commit at scoping time: `{task2_provenance['git_commit']}`.

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
| Stage C.5 null pool size | {value(task1['null_pool_size'])} |
| Stage C.5 replicates completed | {value(task1['replicates_completed'])} |
| Stage C.5 replicates failed to match | {value(task1['replicates_failed_to_match'])} |
| Stage C.5 null mean | {value(task1['null_mean'])} |
| Stage C.5 null standard deviation | {value(task1['null_sd'])} |
| Stage C.5 null 0.025 quantile | {value(task1['null_quantiles']['0.025'])} |
| Stage C.5 null 0.50 quantile | {value(task1['null_quantiles']['0.50'])} |
| Stage C.5 null 0.95 quantile | {value(task1['null_quantiles']['0.95'])} |
| Stage C.5 null 0.975 quantile | {value(task1['null_quantiles']['0.975'])} |
| Stage C.5 null 0.99 quantile | {value(task1['null_quantiles']['0.99'])} |
| Stage C.5 maximum null cohort mean | {value(task1['null_max'])} |
| Stage C.5 observed mean gap | {value(task1['observed_mean_gap'])} |
| Stage C.5 observed z-score versus null | {value(task1['observed_z_vs_null'])} |
| Stage C.5 empirical p-value | {value(task1['empirical_p_observed_ge_null'])} |
| Stage C pose-feature cohort size | {value(task2['row_count'])} |
| Stage C pose-based proxy count | {value(task2['pose_analysis_status_distribution']['pose_based_proxy'])} |
| Stage C num_close_contacts median [min, max] | {value(task2['feature_statistics']['num_close_contacts']['median'])} [{value(task2['feature_statistics']['num_close_contacts']['minimum'])}, {value(task2['feature_statistics']['num_close_contacts']['maximum'])}] |
| Stage C beneficial contacts median [min, max] | {value(task2['feature_statistics']['num_contacts_with_beneficial_landmarks']['median'])} [{value(task2['feature_statistics']['num_contacts_with_beneficial_landmarks']['minimum'])}, {value(task2['feature_statistics']['num_contacts_with_beneficial_landmarks']['maximum'])}] |
| Stage C caution contacts median [min, max] | {value(task2['feature_statistics']['num_contacts_with_caution_landmarks']['median'])} [{value(task2['feature_statistics']['num_contacts_with_caution_landmarks']['minimum'])}, {value(task2['feature_statistics']['num_contacts_with_caution_landmarks']['maximum'])}] |
| Stage C minimum U1 distance median [min, max], Angstrom | {value(task2['feature_statistics']['minimum_distance_to_u1_region']['median'])} [{value(task2['feature_statistics']['minimum_distance_to_u1_region']['minimum'])}, {value(task2['feature_statistics']['minimum_distance_to_u1_region']['maximum'])}] |
| Stage C hydrogen-bond proxy count median [min, max] | {value(task2['feature_statistics']['hydrogen_bond_proxy_count']['median'])} [{value(task2['feature_statistics']['hydrogen_bond_proxy_count']['minimum'])}, {value(task2['feature_statistics']['hydrogen_bond_proxy_count']['maximum'])}] |
| Stage C stacking proxy count median [min, max] | {value(task2['feature_statistics']['stacking_proxy_count']['median'])} [{value(task2['feature_statistics']['stacking_proxy_count']['minimum'])}, {value(task2['feature_statistics']['stacking_proxy_count']['maximum'])}] |
| Stage C pose-support score median [min, max] | {value(task2['feature_statistics']['pose_support_score']['median'])} [{value(task2['feature_statistics']['pose_support_score']['minimum'])}, {value(task2['feature_statistics']['pose_support_score']['maximum'])}] |
| Stage C occlusion penalty median [min, max] | {value(task2['feature_statistics']['splice_site_occlusion_penalty']['median'])} [{value(task2['feature_statistics']['splice_site_occlusion_penalty']['minimum'])}, {value(task2['feature_statistics']['splice_site_occlusion_penalty']['maximum'])}] |
| Stage C occlusion penalty range | {value(task2['feature_statistics']['splice_site_occlusion_penalty']['range'])} |
| Minimum credible redocking benchmark size | 4 structures |
| Minimum credible redocking PDB IDs | 8R62, 8R63, 6HMO, 5KX9 |
"""

    REPORT.write_text(text, encoding="utf-8")
    PROVENANCE_DIR.mkdir(parents=True)
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    provenance = {
        "random_seed": None,
        "random_seed_note": "Report assembly is deterministic and uses no random sampling.",
        "git_commit": git_commit,
        "input_files": [{"path": str(path), "sha256": sha256(path)} for path in report_inputs],
        "output_files": [{"path": str(REPORT), "sha256": sha256(REPORT)}],
    }
    (PROVENANCE_DIR / "run_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(REPORT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
