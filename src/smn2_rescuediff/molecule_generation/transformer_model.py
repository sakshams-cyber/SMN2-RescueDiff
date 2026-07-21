"""PyTorch Transformer decoder-style molecular language model."""

from __future__ import annotations

import math


def torch_available() -> bool:
    """Return whether PyTorch can be imported."""
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def require_torch():
    """Import PyTorch or raise an actionable error."""
    try:
        import torch
        import torch.nn as nn
        return torch, nn
    except Exception as exc:
        raise ImportError(
            "PyTorch is required for neural molecular generation. Install torch or use inference/debug settings that do not train neural models."
        ) from exc


class PositionalEncoding:
    """Factory wrapper for sinusoidal positional encodings."""

    @staticmethod
    def build(max_len: int, embedding_dim: int):
        torch, _ = require_torch()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * (-math.log(10000.0) / embedding_dim))
        pe = torch.zeros(max_len, embedding_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)


def build_causal_mask(size: int, device):
    """Create a causal attention mask."""
    torch, _ = require_torch()
    return torch.triu(torch.ones(size, size, device=device) * float("-inf"), diagonal=1)


def build_transformer_model(
    vocab_size: int,
    embedding_dim: int = 256,
    num_layers: int = 4,
    num_heads: int = 8,
    feedforward_dim: int = 1024,
    dropout: float = 0.1,
    max_sequence_length: int = 160,
):
    """Build a decoder-style autoregressive Transformer using causal self-attention."""
    torch, nn = require_torch()

    class TransformerMolecularLM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
            self.register_buffer("positional_encoding", PositionalEncoding.build(max_sequence_length + 1, embedding_dim))
            layer = nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=num_heads,
                dim_feedforward=feedforward_dim,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
            self.dropout = nn.Dropout(dropout)
            self.output = nn.Linear(embedding_dim, vocab_size)

        def forward(self, token_ids, padding_mask=None):
            seq_len = token_ids.size(1)
            x = self.embedding(token_ids) * math.sqrt(embedding_dim)
            x = x + self.positional_encoding[:, :seq_len, :]
            x = self.dropout(x)
            causal_mask = build_causal_mask(seq_len, token_ids.device)
            hidden = self.transformer(x, mask=causal_mask, src_key_padding_mask=padding_mask)
            return self.output(hidden)

    return TransformerMolecularLM()

