"""Training loop for the optional PyTorch SMILES RNN."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .rnn_model import build_model, require_torch
from .tokenizer import SmilesTokenizer


def _make_batches(encoded: list[list[int]], batch_size: int, pad_id: int):
    torch, _ = require_torch()
    for idx in range(0, len(encoded), batch_size):
        batch = encoded[idx : idx + batch_size]
        max_len = max(len(item) for item in batch)
        padded = [item + [pad_id] * (max_len - len(item)) for item in batch]
        tensor = torch.tensor(padded, dtype=torch.long)
        yield tensor[:, :-1], tensor[:, 1:]


def train_rnn(
    smiles: list[str],
    tokenizer: SmilesTokenizer,
    config: dict[str, Any],
    output_dir: str | Path,
) -> tuple[object, pd.DataFrame]:
    """Train the optional neural SMILES generator."""
    torch, nn = require_torch()
    training_cfg = config.get("training", {})
    seed = int(config.get("generation", {}).get("random_seed", 42))
    torch.manual_seed(seed)
    encoded = [tokenizer.encode(item, add_special=True) for item in smiles]
    if not encoded:
        raise ValueError("No encoded SMILES are available for RNN training.")
    batch_size = int(training_cfg.get("batch_size", 128))
    model = build_model(
        vocab_size=len(tokenizer.token_to_id),
        embedding_dim=int(training_cfg.get("embedding_dim", 128)),
        hidden_dim=int(training_cfg.get("hidden_dim", 256)),
        num_layers=int(training_cfg.get("num_layers", 2)),
        dropout=float(training_cfg.get("dropout", 0.2)),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training_cfg.get("learning_rate", 0.001)))
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
    rows = []
    best_loss = float("inf")
    patience = int(training_cfg.get("early_stopping_patience", 3))
    patience_used = 0
    epochs = int(training_cfg.get("epochs", 10))
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for inputs, targets in _make_batches(encoded, batch_size, tokenizer.pad_id):
            optimizer.zero_grad()
            logits, _ = model(inputs)
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        mean_loss = sum(losses) / max(1, len(losses))
        rows.append({"epoch": epoch, "train_loss": mean_loss})
        if mean_loss < best_loss:
            best_loss = mean_loss
            patience_used = 0
            torch.save(model.state_dict(), Path(output_dir) / "smiles_rnn_model.pt")
        else:
            patience_used += 1
            if patience_used >= patience:
                break
    return model, pd.DataFrame(rows)

