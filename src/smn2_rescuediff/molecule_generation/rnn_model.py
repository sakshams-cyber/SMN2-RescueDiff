"""Optional PyTorch recurrent molecular language model."""

from __future__ import annotations


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
            "PyTorch is required for neural SMILES generation. Install it with a platform-appropriate command, "
            "or set generation.generator_mode: debug_ngram in configs/molecule_generation.yaml for software smoke tests."
        ) from exc


def build_model(vocab_size: int, embedding_dim: int, hidden_dim: int, num_layers: int, dropout: float):
    """Create a GRU-based next-token model.

    This function is retained for backward compatibility with the original
    Module 3 implementation.
    """
    return build_rnn_model(vocab_size, embedding_dim, hidden_dim, num_layers, dropout, architecture="gru")


def build_rnn_model(
    vocab_size: int,
    embedding_dim: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    architecture: str = "gru",
):
    """Create a GRU or LSTM next-token molecular language model."""
    torch, nn = require_torch()
    rnn_class = nn.LSTM if architecture.lower() == "lstm" else nn.GRU

    class MolecularRNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
            self.rnn = rnn_class(
                embedding_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.output = nn.Linear(hidden_dim, vocab_size)

        def forward(self, token_ids, hidden=None):
            embedded = self.embedding(token_ids)
            outputs, hidden = self.rnn(embedded, hidden)
            logits = self.output(outputs)
            return logits

    return MolecularRNN()
