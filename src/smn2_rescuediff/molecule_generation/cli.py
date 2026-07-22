"""Command-line interface for Module 3 deep neural molecule generation."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .baseline_generator import NGramSmilesGenerator
from .candidate_prioritizer import add_candidate_scores, select_docking_candidates
from .chemical_filter import process_generated_smiles, require_rdkit, write_table
from .config import ensure_output_dirs, load_config, optional_context_status, resolve_paths
from .dataset_builder import build_training_dataset
from .diversity_analysis import assign_diversity_clusters, pairwise_tanimoto_sample
from .figure_writer import write_generation_figures
from .molecular_tokenizer import MolecularTokenizer
from .neural_sampler import sample_neural_strings, write_raw_generation_outputs
from .neural_trainer import train_neural_model
from .novelty_analysis import add_novelty_scores
from .representation_converter import choose_representation, representation_to_smiles, smiles_to_representation
from .report_writer import build_summary, write_report, write_summary_json
from .rl_finetuner import run_reward_finetuning
from .scaffold_analysis import add_scaffolds
from .smiles_loader import TOY_WARNING
from .transformer_model import torch_available


FINAL_FILTERED_COLUMNS = [
    "molecule_id",
    "source_generator",
    "representation",
    "raw_generated_string",
    "decoded_smiles",
    "canonical_smiles",
    "is_valid",
    "is_duplicate",
    "is_novel_vs_training",
    "molecular_weight",
    "logp",
    "hbd",
    "hba",
    "rotatable_bonds",
    "formal_charge",
    "num_rings",
    "num_aromatic_rings",
    "qed",
    "sa_score",
    "lipinski_pass",
    "pains_alert",
    "filter_pass",
    "rejection_reason",
    "max_tanimoto_to_training",
    "bemis_murcko_scaffold",
    "rna_ligand_like_score",
    "pre_docking_candidate_score",
    "diversity_cluster_id",
    "selected_for_docking",
    "notes",
]


def _generation_mode(config: dict[str, Any]) -> str:
    mode = str(config.get("generation", {}).get("generator_mode", "neural_transformer")).lower()
    allowed = {"neural_transformer", "neural_gru", "neural_transformer_with_rl", "inference_only", "debug_ngram"}
    if mode not in allowed:
        raise ValueError(f"generation.generator_mode must be one of {sorted(allowed)}")
    return mode


def _run_name(config: dict[str, Any]) -> str:
    run = config.setdefault("run", {})
    name = str(run.get("run_name") or "neural_run").strip()
    run["run_name"] = name
    return name


def _run_output_paths(paths, config: dict[str, Any]):
    run_name = _run_name(config)
    run_dir = paths.output_dir / "runs" / run_name
    return run_dir, run_dir / "checkpoints", run_dir / "logs"


def _write_text(path: str | Path, text: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def _write_tsv(df: pd.DataFrame, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, sep="\t", index=False)


def _copy2(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _write_run_artifacts(run_dir: Path, config: dict[str, Any], summary: dict[str, Any], output_files: list[str]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics = dict(summary)
    metrics["run_name"] = _run_name(config)
    metrics["run_folder"] = str(run_dir)
    metrics["output_files"] = output_files
    _write_text(run_dir / "run_metrics.json", json.dumps(metrics, indent=2, default=str))
    lines = [
        f"# Module 3 Run Summary: {_run_name(config)}",
        "",
        f"- Run folder: `{run_dir}`",
        f"- Generator mode: `{config.get('generation', {}).get('generator_mode', 'NA')}`",
        f"- Raw generated molecules: {summary.get('raw_generated_count', 'NA')}",
        f"- Valid molecules: {summary.get('valid_count', 'NA')}",
        f"- Filtered molecules: {summary.get('filtered_count', 'NA')}",
        f"- Docking exports: {summary.get('selected_for_docking_count', 'NA')}",
        f"- Toy/small training warning used: {summary.get('toy_training_file_used', 'NA')}",
        "",
        "## Stable Downstream Outputs",
        "",
        "- `data/processed/molecule_generation/generated_molecules_filtered.tsv`",
        "- `data/processed/molecule_generation/generated_molecules_for_docking.tsv`",
        "- `data/processed/molecule_generation/generated_molecule_summary.json`",
        "",
        "## Limitations",
        "",
        "Generated molecules are computational candidates only and do not prove binding, splicing rescue, safety, or therapeutic activity.",
    ]
    _write_text(run_dir / "RUN_SUMMARY.md", "\n".join(lines) + "\n")


def _sync_run_outputs_to_stable(run_dir: Path, stable_dir: Path, output_files: list[str]) -> None:
    stable_dir.mkdir(parents=True, exist_ok=True)
    for item in output_files:
        source = Path(item)
        if source.exists() and source.is_file():
            _copy2(source, stable_dir / source.name)
    for filename in ["RUN_SUMMARY.md", "run_metrics.json"]:
        source = run_dir / filename
        if source.exists():
            _copy2(source, stable_dir / filename)


def _sync_run_report_and_figures(run_dir: Path, stable_report: Path, stable_figure_dir: Path) -> None:
    run_report = run_dir / "reports" / stable_report.name
    if run_report.exists():
        _copy2(run_report, stable_report)
    run_figure_dir = run_dir / "figures"
    if run_figure_dir.exists():
        stable_figure_dir.mkdir(parents=True, exist_ok=True)
        for figure in run_figure_dir.glob("*"):
            if figure.is_file():
                _copy2(figure, stable_figure_dir / figure.name)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for column in FINAL_FILTERED_COLUMNS:
        if column not in result.columns:
            result[column] = "NA"
    return result[FINAL_FILTERED_COLUMNS]


def _debug_ngram_generate(training_smiles: list[str], representation: str, config: dict[str, Any], count: int) -> pd.DataFrame:
    seed = int(config.get("run", {}).get("random_seed", 42))
    sampling = config.get("sampling", {})
    generator = NGramSmilesGenerator(ngram_size=3, random_seed=seed)
    generator.fit(training_smiles)
    smiles = generator.generate(count, max_length=int(sampling.get("max_generated_length", 160)), temperature=float(sampling.get("temperature", 0.8)))
    return pd.DataFrame(
        {
            "source_generator": "debug_ngram",
            "representation": "smiles",
            "raw_generated_string": smiles,
            "decoded_smiles": smiles,
        }
    )


def _training_surrogate_generate(training_strings: list[str], representation: str, count: int) -> pd.DataFrame:
    """Fallback inference source when training is disabled and no model checkpoint is available."""
    rows = []
    for idx in range(count):
        value = training_strings[idx % len(training_strings)]
        rows.append(
            {
                "source_generator": "inference_only_training_seed",
                "representation": representation,
                "raw_generated_string": value,
                "decoded_smiles": representation_to_smiles(value, representation),
            }
        )
    return pd.DataFrame(rows)


def run(config_path: str | Path) -> int:
    """Run the upgraded Module 3 pipeline."""
    config = load_config(config_path)
    _run_name(config)
    paths = resolve_paths(config, config_path)
    ensure_output_dirs(paths)
    output_dir, checkpoint_dir, log_dir = _run_output_paths(paths, config)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    seed = int(config.get("run", {}).get("random_seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    require_rdkit()

    warnings: list[str] = []
    mode = _generation_mode(config)
    if mode == "neural_gru":
        config.setdefault("model", {})["architecture"] = "gru"
    if mode == "neural_transformer_with_rl":
        config.setdefault("rl_finetuning", {})["enabled"] = True

    print("Building merged cleaned training dataset.")
    canonical_training, dataset_summary, _rejected, dataset_warnings = build_training_dataset(config, paths.project_root, output_dir)
    warnings.extend(dataset_warnings)
    toy_dataset = TOY_WARNING in warnings or dataset_summary.get("training_molecules_cleaned", 0) < int(config.get("data_processing", {}).get("min_training_molecules_warning", 1000))

    representation, representation_warnings = choose_representation(config)
    warnings.extend(representation_warnings)
    training_strings, conversion_rejected = smiles_to_representation(canonical_training, representation)
    if conversion_rejected:
        _write_tsv(pd.DataFrame(conversion_rejected), output_dir / "representation_conversion_rejected.tsv")
        warnings.append(f"{len(conversion_rejected)} molecules could not be converted to {representation}.")

    tokenizer = MolecularTokenizer.build(
        training_strings,
        representation=representation,
        add_start_end_tokens=bool(config.get("tokenization", {}).get("add_start_end_tokens", True)),
    )
    tokenizer.to_json(output_dir / "token_vocabulary.json")
    tokenizer.write_metadata(output_dir / "tokenizer_metadata.json")

    if bool(config.get("run", {}).get("save_config_used", True)):
        _write_text(output_dir / "molecule_generation_config_used.yaml", yaml.safe_dump(config, sort_keys=False))

    requested_raw = int(config.get("sampling", {}).get("num_raw_to_generate", 100000))
    if toy_dataset and requested_raw > 1000:
        warnings.append("Toy/small training dataset detected; capping raw generation to 1000 to avoid presenting a toy run as a large scientific library.")
        requested_raw = 1000

    model = None
    training_log = pd.DataFrame(columns=["epoch", "train_loss", "validation_loss", "learning_rate", "device", "checkpoint_path", "notes"])
    device_name = "cpu"
    if mode in {"neural_transformer", "neural_gru", "neural_transformer_with_rl"}:
        if not torch_available():
            raise ImportError("PyTorch is required for neural generation modes. Install torch or use generation.generator_mode: debug_ngram.")
        print(f"Training neural molecular language model ({config.get('model', {}).get('architecture', 'transformer_decoder')}).")
        model, training_log, training_warnings, device_name = train_neural_model(training_strings, tokenizer, config, checkpoint_dir)
        warnings.extend(training_warnings)
        _write_tsv(training_log, output_dir / "model_training_log.tsv")
        print("Sampling neural candidate molecules.")
        raw_generated = sample_neural_strings(model, tokenizer, config, requested_raw, device_name)
    elif mode == "inference_only":
        warnings.append("Inference-only mode used training-set representation strings as a smoke-test source because no trained checkpoint loader was requested.")
        raw_generated = _training_surrogate_generate(training_strings, representation, requested_raw)
        _write_tsv(training_log, output_dir / "model_training_log.tsv")
    else:
        warnings.append("debug_ngram mode is for troubleshooting only and is not the neural-first scientific path.")
        raw_generated = _debug_ngram_generate(canonical_training, representation, config, requested_raw)
        _write_tsv(training_log, output_dir / "model_training_log.tsv")

    if raw_generated.empty:
        raise ValueError("No raw molecular strings were generated.")
    write_raw_generation_outputs(raw_generated, output_dir)

    print("Validating and filtering generated molecules with RDKit.")
    valid, filtered, rejected = process_generated_smiles(raw_generated, config)
    write_table(valid, output_dir / "generated_molecules_valid.tsv")
    write_table(rejected, output_dir / "generated_molecules_rejected.tsv")
    if filtered.empty:
        raise ValueError("No molecules passed chemical filters. Use a larger dataset, train longer, or loosen chemical_filters.")

    filtered, novelty_warnings = add_novelty_scores(filtered, canonical_training, config)
    warnings.extend(novelty_warnings)
    filtered = add_scaffolds(filtered)
    filtered = assign_diversity_clusters(filtered)
    filtered = add_candidate_scores(filtered)
    docking_candidates = select_docking_candidates(filtered, config) if bool(config.get("candidate_prioritization", {}).get("enabled", True)) else filtered.copy()
    selected_set = set(docking_candidates["canonical_smiles"]) if not docking_candidates.empty else set()
    filtered["selected_for_docking"] = filtered["canonical_smiles"].isin(selected_set)
    filtered = _normalize_columns(filtered)
    docking_candidates = _normalize_columns(filtered[filtered["selected_for_docking"] == True].copy())  # noqa: E712
    write_table(filtered, output_dir / "generated_molecules_filtered.tsv")
    write_table(docking_candidates, output_dir / "generated_molecules_for_docking.tsv")

    diversity_values = pairwise_tanimoto_sample(filtered, config)
    rl_log, rl_warnings = run_reward_finetuning(filtered, config, checkpoint_dir)
    warnings.extend(rl_warnings)
    _write_tsv(rl_log, output_dir / "rl_finetuning_log.tsv")

    summary = build_summary(
        config=config,
        context_status=optional_context_status(paths),
        training_path=", ".join(dataset_summary.get("training_files", [])),
        toy_warning_used=toy_dataset,
        raw_count=len(raw_generated),
        valid=valid,
        filtered=filtered,
        rejected=rejected,
        warnings=warnings,
    )
    summary.update(
        {
            "module": "Deep Neural Molecular Generation Model",
            "run_name": config.get("run", {}).get("run_name", "neural_run"),
            "training_files": dataset_summary.get("training_files", []),
            "training_molecules_raw": dataset_summary.get("training_molecules_raw", 0),
            "training_molecules_cleaned": dataset_summary.get("training_molecules_cleaned", 0),
            "representation": representation,
            "model_architecture": config.get("model", {}).get("architecture", "transformer_decoder"),
            "epochs_completed": int(len(training_log)),
            "unique_count": int(filtered["canonical_smiles"].nunique()) if "canonical_smiles" in filtered else 0,
            "novel_count": int((filtered["is_novel_vs_training"] == True).sum()) if "is_novel_vs_training" in filtered else 0,  # noqa: E712
            "selected_for_docking_count": int(len(docking_candidates)),
        }
    )
    write_summary_json(summary, output_dir / "generated_molecule_summary.json")
    write_generation_figures(
        training_log=training_log,
        rl_log=rl_log,
        valid=valid,
        filtered=filtered,
        docking_candidates=docking_candidates,
        raw_count=len(raw_generated),
        diversity_values=diversity_values,
        figure_dir=output_dir / "figures",
        toy_dataset=toy_dataset,
    )

    output_files = [
        str(output_dir / "training_smiles_merged_cleaned.smi"),
        str(output_dir / "training_dataset_summary.json"),
        str(output_dir / "training_dataset_rejected.tsv"),
        str(output_dir / "token_vocabulary.json"),
        str(output_dir / "tokenizer_metadata.json"),
        str(output_dir / "generated_strings_raw.tsv"),
        str(output_dir / "generated_smiles_raw.smi"),
        str(output_dir / "generated_smiles_raw.tsv"),
        str(output_dir / "generated_molecules_valid.tsv"),
        str(output_dir / "generated_molecules_filtered.tsv"),
        str(output_dir / "generated_molecules_rejected.tsv"),
        str(output_dir / "generated_molecules_for_docking.tsv"),
        str(output_dir / "generated_molecule_summary.json"),
        str(output_dir / "model_training_log.tsv"),
        str(output_dir / "rl_finetuning_log.tsv"),
        str(output_dir / "molecule_generation_config_used.yaml"),
    ]
    run_report_path = output_dir / "reports" / paths.report_path.name
    run_report_path.parent.mkdir(parents=True, exist_ok=True)
    write_report(config=config, summary=summary, output_files=output_files, report_path=run_report_path)
    _write_run_artifacts(output_dir, config, summary, output_files)

    # Convenience copies for Module 4, while preserving run-specific outputs.
    if output_dir != paths.output_dir:
        _sync_run_outputs_to_stable(output_dir, paths.output_dir, output_files)
        _sync_run_report_and_figures(output_dir, paths.report_path, paths.figure_dir)

    print("\nModule 3 complete: Deep neural molecule generation finished.\n")
    print(f"Run name: {config.get('run', {}).get('run_name', 'neural_run')}")
    print(f"Training molecules cleaned: {summary['training_molecules_cleaned']}")
    print(f"Representation: {representation.upper()}")
    print(f"Model architecture: {summary['model_architecture']}")
    print(f"Raw generated molecules: {summary['raw_generated_count']}")
    print(f"Valid molecules: {summary['valid_count']}")
    print(f"Filtered molecules: {summary['filtered_count']}")
    print(f"Candidates exported for docking: {summary['selected_for_docking_count']}\n")
    print("Filtered candidates:")
    print(output_dir / "generated_molecules_filtered.tsv")
    print("\nDocking-ready candidates:")
    print(output_dir / "generated_molecules_for_docking.tsv")
    print("\nReport:")
    print(paths.report_path)
    print("\nFigures:")
    print(paths.figure_dir)
    print("\nNext step:")
    print("Module 4 will dock these candidates against the SMN2 target and selected HuRDA RNA decoys.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Module 3 deep neural molecule generation.")
    parser.add_argument("--config", required=True, help="Path to configs/molecule_generation.yaml")
    args = parser.parse_args(argv)
    try:
        return run(args.config)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
