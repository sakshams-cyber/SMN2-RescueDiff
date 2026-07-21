"""Autoregressive neural molecule sampling."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .molecular_tokenizer import MolecularTokenizer
from .representation_converter import representation_to_smiles
from .transformer_model import require_torch


def _filter_logits(logits, top_k: int | None, top_p: float | None):
    torch, _ = require_torch()
    filtered = logits.clone()
    if top_k and top_k > 0 and top_k < filtered.size(-1):
        values, _indices = torch.topk(filtered, top_k)
        cutoff = values[..., -1, None]
        filtered[filtered < cutoff] = float("-inf")
    if top_p and 0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)
        remove = cumulative > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits[remove] = float("-inf")
        filtered.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
    return filtered


def sample_neural_strings(
    model: object,
    tokenizer: MolecularTokenizer,
    config: dict[str, Any],
    count: int,
    device_name: str,
) -> pd.DataFrame:
    """Sample generated representation strings and decoded SMILES."""
    torch, _ = require_torch()
    sampling = config.get("sampling", {})
    max_len = int(sampling.get("max_generated_length", 160))
    temperature = float(sampling.get("temperature", 0.8))
    top_k = int(sampling.get("top_k", 50))
    top_p = float(sampling.get("top_p", 0.95))
    seed = int(config.get("run", {}).get("random_seed", 42)) + 917
    torch.manual_seed(seed)
    device = torch.device(device_name)
    model.eval()
    rows: list[dict[str, str]] = []
    architecture = str(config.get("model", {}).get("architecture", "transformer_decoder"))
    with torch.no_grad():
        for idx in range(max(0, count)):
            ids = [tokenizer.start_id]
            hidden = None
            for _ in range(max_len):
                x = torch.tensor([ids], dtype=torch.long, device=device)
                if architecture == "transformer_decoder":
                    logits = model(x)[:, -1, :]
                else:
                    logits = model(x)[:, -1, :]
                logits = logits / max(temperature, 1e-6)
                logits = _filter_logits(logits, top_k, top_p)
                probs = torch.softmax(logits, dim=-1)
                next_id = int(torch.multinomial(probs, num_samples=1).item())
                if next_id == tokenizer.end_id:
                    break
                if next_id != tokenizer.pad_id and next_id != tokenizer.start_id:
                    ids.append(next_id)
            generated = tokenizer.decode(ids[1:])
            rows.append(
                {
                    "source_generator": f"neural_{architecture}",
                    "representation": tokenizer.representation,
                    "raw_generated_string": generated,
                    "decoded_smiles": representation_to_smiles(generated, tokenizer.representation),
                }
            )
    df = pd.DataFrame(rows)
    if bool(sampling.get("remove_duplicate_generated_strings", True)) and not df.empty:
        df = df.drop_duplicates(subset=["raw_generated_string"]).reset_index(drop=True)
    return df


def write_raw_generation_outputs(raw: pd.DataFrame, output_dir: str | Path) -> None:
    """Write raw generated strings and decoded SMILES outputs."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw.to_csv(out / "generated_strings_raw.tsv", sep="\t", index=False)
    smiles = raw["decoded_smiles"].fillna("").astype(str).tolist() if "decoded_smiles" in raw else []
    (out / "generated_smiles_raw.smi").write_text("\n".join(smiles) + ("\n" if smiles else ""), encoding="utf-8")
    pd.DataFrame({"source_generator": raw.get("source_generator", "NA"), "raw_smiles": smiles}).to_csv(
        out / "generated_smiles_raw.tsv",
        sep="\t",
        index=False,
    )

