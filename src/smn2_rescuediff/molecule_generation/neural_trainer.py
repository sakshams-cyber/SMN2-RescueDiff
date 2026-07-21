"""Shared PyTorch trainer for Transformer and recurrent molecule LMs."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import pandas as pd

from .molecular_tokenizer import MolecularTokenizer
from .rnn_model import build_rnn_model
from .transformer_model import build_transformer_model, require_torch


def select_device(requested: str = "auto") -> tuple[str, list[str]]:
    """Select CUDA, Apple MPS, or CPU."""
    torch, _ = require_torch()
    warnings: list[str] = []
    if requested != "auto":
        return requested, warnings
    if torch.cuda.is_available():
        return "cuda", warnings
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", warnings
    warnings.append("GPU/MPS unavailable; using CPU. Neural training may be slower.")
    return "cpu", warnings


def _pad_batch(batch: list[list[int]], pad_id: int):
    torch, _ = require_torch()
    max_len = max(len(item) for item in batch)
    padded = [item + [pad_id] * (max_len - len(item)) for item in batch]
    tensor = torch.tensor(padded, dtype=torch.long)
    return tensor[:, :-1], tensor[:, 1:]


def _batches(encoded: list[list[int]], batch_size: int, pad_id: int, shuffle: bool, seed: int):
    order = list(range(len(encoded)))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        batch = [encoded[idx] for idx in order[start : start + batch_size]]
        yield _pad_batch(batch, pad_id)


def build_neural_model(config: dict[str, Any], vocab_size: int):
    """Build configured neural model."""
    model_cfg = config.get("model", {})
    architecture = str(model_cfg.get("architecture", "transformer_decoder")).lower()
    if architecture == "transformer_decoder":
        return build_transformer_model(
            vocab_size=vocab_size,
            embedding_dim=int(model_cfg.get("embedding_dim", 256)),
            num_layers=int(model_cfg.get("num_layers", 4)),
            num_heads=int(model_cfg.get("num_heads", 8)),
            feedforward_dim=int(model_cfg.get("feedforward_dim", 1024)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            max_sequence_length=int(model_cfg.get("max_sequence_length", 160)),
        )
    if architecture in {"gru", "lstm", "neural_gru"}:
        return build_rnn_model(
            vocab_size=vocab_size,
            embedding_dim=int(model_cfg.get("embedding_dim", 256)),
            hidden_dim=int(model_cfg.get("hidden_dim", 256)),
            num_layers=int(model_cfg.get("num_layers", 2)),
            dropout=float(model_cfg.get("dropout", 0.2)),
            architecture=str(model_cfg.get("fallback_architecture", "gru")),
        )
    raise ValueError(f"Unsupported model architecture: {architecture}")


def train_neural_model(
    training_strings: list[str],
    tokenizer: MolecularTokenizer,
    config: dict[str, Any],
    checkpoint_dir: str | Path,
) -> tuple[object | None, pd.DataFrame, list[str], str]:
    """Train configured neural model and save checkpoints."""
    warnings: list[str] = []
    training_cfg = config.get("training", {})
    if not bool(training_cfg.get("enabled", True)):
        return None, pd.DataFrame(columns=["epoch", "train_loss", "validation_loss", "learning_rate", "device", "checkpoint_path", "notes"]), warnings, "cpu"
    torch, nn = require_torch()
    seed = int(config.get("run", {}).get("random_seed", 42))
    torch.manual_seed(seed)
    device_name, device_warnings = select_device(str(config.get("run", {}).get("device", "auto")))
    warnings.extend(device_warnings)
    device = torch.device(device_name)
    encoded = [tokenizer.encode(item)[: int(config.get("model", {}).get("max_sequence_length", 160))] for item in training_strings]
    encoded = [item if item[-1] == tokenizer.end_id else [*item, tokenizer.end_id] for item in encoded if len(item) >= 2]
    if not encoded:
        raise ValueError("No tokenized training strings are available.")
    rng = random.Random(seed)
    rng.shuffle(encoded)
    val_fraction = float(training_cfg.get("validation_fraction", 0.1))
    val_count = max(1, int(len(encoded) * val_fraction)) if len(encoded) > 2 else 1
    validation = encoded[:val_count]
    train = encoded[val_count:] or encoded
    model = build_neural_model(config, len(tokenizer.token_to_id)).to(device)
    resume = training_cfg.get("resume_from_checkpoint")
    if resume:
        model.load_state_dict(torch.load(resume, map_location=device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg.get("learning_rate", 0.0005)),
        weight_decay=float(training_cfg.get("weight_decay", 0.00001)),
    )
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    best_val = float("inf")
    patience = int(training_cfg.get("early_stopping_patience", 5))
    patience_used = 0
    epochs = int(training_cfg.get("epochs", 20))
    batch_size = int(training_cfg.get("batch_size", 128))
    grad_clip = float(training_cfg.get("gradient_clip_norm", 1.0))
    architecture = str(config.get("model", {}).get("architecture", "transformer_decoder"))
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for x, y in _batches(train, batch_size, tokenizer.pad_id, True, seed + epoch):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            padding_mask = x.eq(tokenizer.pad_id) if architecture == "transformer_decoder" else None
            logits = model(x, padding_mask=padding_mask) if architecture == "transformer_decoder" else model(x)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y in _batches(validation, batch_size, tokenizer.pad_id, False, seed):
                x = x.to(device)
                y = y.to(device)
                padding_mask = x.eq(tokenizer.pad_id) if architecture == "transformer_decoder" else None
                logits = model(x, padding_mask=padding_mask) if architecture == "transformer_decoder" else model(x)
                val_loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                val_losses.append(float(val_loss.detach().cpu()))
        train_loss = sum(train_losses) / max(1, len(train_losses))
        validation_loss = sum(val_losses) / max(1, len(val_losses))
        epoch_path = checkpoint_path / f"{'transformer' if architecture == 'transformer_decoder' else architecture}_epoch_{epoch:03d}.pt"
        if bool(training_cfg.get("save_checkpoint_every_epoch", True)):
            torch.save({"model_state_dict": model.state_dict(), "config": config, "epoch": epoch}, epoch_path)
        if validation_loss < best_val:
            best_val = validation_loss
            patience_used = 0
            torch.save({"model_state_dict": model.state_dict(), "config": config, "epoch": epoch}, checkpoint_path / "best_neural_generator.pt")
            note = "best_checkpoint"
        else:
            patience_used += 1
            note = "early_stopping_patience_counted"
        rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "learning_rate": float(training_cfg.get("learning_rate", 0.0005)),
                "device": device_name,
                "checkpoint_path": str(epoch_path) if epoch_path.exists() else "NA",
                "notes": note,
            }
        )
        if patience_used >= patience:
            warnings.append(f"Early stopping triggered after epoch {epoch}.")
            break
    return model, pd.DataFrame(rows), warnings, device_name

