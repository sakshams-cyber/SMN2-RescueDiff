"""Sampling utilities for optional neural SMILES generation."""

from __future__ import annotations

from typing import Any

from .rnn_model import require_torch
from .tokenizer import END, START, SmilesTokenizer


def sample_from_rnn(
    model: object,
    tokenizer: SmilesTokenizer,
    count: int,
    max_length: int,
    temperature: float,
    random_seed: int,
) -> list[str]:
    """Sample raw SMILES strings from a trained RNN model."""
    torch, _ = require_torch()
    torch.manual_seed(random_seed)
    model.eval()
    generated: list[str] = []
    with torch.no_grad():
        for _ in range(max(0, count)):
            token_id = torch.tensor([[tokenizer.start_id]], dtype=torch.long)
            hidden = None
            output_ids: list[int] = []
            for _step in range(max_length):
                logits, hidden = model(token_id, hidden)
                logits = logits[:, -1, :] / max(temperature, 1e-6)
                probabilities = torch.softmax(logits, dim=-1)
                next_id = int(torch.multinomial(probabilities, num_samples=1).item())
                token = tokenizer.id_to_token.get(next_id, "")
                if token == END:
                    break
                if token != START:
                    output_ids.append(next_id)
                token_id = torch.tensor([[next_id]], dtype=torch.long)
            generated.append(tokenizer.decode(output_ids))
    return generated

