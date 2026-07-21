"""SMILES tokenizer with bracket, halogen, and special-token support."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


PAD = "<PAD>"
START = "<START>"
END = "<END>"
UNK = "<UNK>"


def tokenize_smiles(smiles: str) -> list[str]:
    """Tokenize common SMILES syntax while keeping bracket atoms and halogens intact."""
    tokens: list[str] = []
    idx = 0
    while idx < len(smiles):
        if smiles[idx] == "[":
            close = smiles.find("]", idx + 1)
            if close != -1:
                tokens.append(smiles[idx : close + 1])
                idx = close + 1
                continue
        if smiles[idx : idx + 2] in {"Cl", "Br"}:
            tokens.append(smiles[idx : idx + 2])
            idx += 2
            continue
        tokens.append(smiles[idx])
        idx += 1
    return tokens


@dataclass
class SmilesTokenizer:
    """Vocabulary and encode/decode helpers for SMILES generation."""

    token_to_id: dict[str, int]
    id_to_token: dict[int, str]

    @classmethod
    def build(cls, smiles: list[str]) -> "SmilesTokenizer":
        vocab = [PAD, START, END, UNK]
        observed = sorted({token for item in smiles for token in tokenize_smiles(item)})
        vocab.extend(token for token in observed if token not in vocab)
        token_to_id = {token: idx for idx, token in enumerate(vocab)}
        id_to_token = {idx: token for token, idx in token_to_id.items()}
        return cls(token_to_id=token_to_id, id_to_token=id_to_token)

    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD]

    @property
    def start_id(self) -> int:
        return self.token_to_id[START]

    @property
    def end_id(self) -> int:
        return self.token_to_id[END]

    def encode(self, smiles: str, add_special: bool = True) -> list[int]:
        tokens = tokenize_smiles(smiles)
        if add_special:
            tokens = [START, *tokens, END]
        return [self.token_to_id.get(token, self.token_to_id[UNK]) for token in tokens]

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        tokens = []
        for idx in ids:
            token = self.id_to_token.get(int(idx), UNK)
            if skip_special and token in {PAD, START, END}:
                continue
            tokens.append(token)
        return "".join(tokens)

    def to_json(self, path: str | Path) -> None:
        payload = {"token_to_id": self.token_to_id, "id_to_token": {str(k): v for k, v in self.id_to_token.items()}}
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "SmilesTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            token_to_id={str(k): int(v) for k, v in payload["token_to_id"].items()},
            id_to_token={int(k): str(v) for k, v in payload["id_to_token"].items()},
        )

