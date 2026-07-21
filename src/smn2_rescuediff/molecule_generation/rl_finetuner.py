"""Reward-based fine-tuning summary for pre-docking molecular quality."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .reward_functions import chemical_plausibility_reward


def run_reward_finetuning(
    generated: pd.DataFrame,
    config: dict[str, Any],
    checkpoint_dir: str | Path,
) -> tuple[pd.DataFrame, list[str]]:
    """Record reward-guided fine-tuning diagnostics.

    This intentionally avoids docking rewards. If full policy-gradient updates are
    added later, this function can become the optimization loop while preserving
    the same log schema.
    """
    rl_cfg = config.get("rl_finetuning", {})
    if not bool(rl_cfg.get("enabled", False)):
        return pd.DataFrame(columns=["episode", "mean_reward", "validity_rate", "uniqueness_rate", "novelty_rate", "mean_qed", "mean_molecular_weight", "mean_logp", "checkpoint_path", "notes"]), []
    warnings = ["Reward fine-tuning logged chemical plausibility rewards only; no docking or SMN2-binding reward was used."]
    episodes = int(rl_cfg.get("episodes", 5))
    sample_size = int(rl_cfg.get("samples_per_episode", 5000))
    weights = config.get("reward_weights", {})
    rows = []
    if generated.empty:
        return pd.DataFrame(rows), warnings
    for episode in range(1, episodes + 1):
        sample = generated.sample(n=min(sample_size, len(generated)), random_state=episode).copy()
        rewards = [chemical_plausibility_reward(row, weights) for _, row in sample.iterrows()]
        checkpoint = Path(checkpoint_dir) / f"rl_reward_episode_{episode:03d}.txt"
        if bool(rl_cfg.get("save_rl_checkpoints", True)):
            checkpoint.write_text("Reward-only diagnostic checkpoint; no policy weights updated.\n", encoding="utf-8")
        rows.append(
            {
                "episode": episode,
                "mean_reward": sum(rewards) / max(1, len(rewards)),
                "validity_rate": float((sample.get("is_valid", False) == True).mean()) if "is_valid" in sample else 0.0,  # noqa: E712
                "uniqueness_rate": float((sample.get("is_duplicate", False) == False).mean()) if "is_duplicate" in sample else 0.0,  # noqa: E712
                "novelty_rate": float((sample.get("is_novel_vs_training", False) == True).mean()) if "is_novel_vs_training" in sample else 0.0,  # noqa: E712
                "mean_qed": float(pd.to_numeric(sample.get("qed", pd.Series(dtype=float)), errors="coerce").mean()),
                "mean_molecular_weight": float(pd.to_numeric(sample.get("molecular_weight", pd.Series(dtype=float)), errors="coerce").mean()),
                "mean_logp": float(pd.to_numeric(sample.get("logp", pd.Series(dtype=float)), errors="coerce").mean()),
                "checkpoint_path": str(checkpoint) if checkpoint.exists() else "NA",
                "notes": "Chemical plausibility and RNA-ligand-like reward diagnostic only.",
            }
        )
    return pd.DataFrame(rows), warnings

