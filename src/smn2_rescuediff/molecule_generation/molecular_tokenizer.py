"""Tokenizer supporting SMILES and SELFIES molecular strings."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


PAD = "<PAD>"
START = "<START>"
END = "<END>"
UNK = "<UNK>"


def smiles_tokens(smiles: str) -> list[str]:
    """Tokenize SMILES while preserving common two-character and bracket tokens."""
    tokens: list[str] = []
    i = 0
    while i < len(smiles):
        if smiles[i] == "[":
            close = smiles.find("]", i + 1)
            if close != -1:
                tokens.append(smiles[i : close + 1])
                i = close + 1
                continue
        if smiles[i : i + 2] in {"Cl", "Br"}:
            tokens.append(smiles[i : i + 2])
            i += 2
            continue
        tokens.append(smiles[i])
        i += 1
    return tokens


def selfies_tokens(value: str) -> list[str]:
    """Tokenize SELFIES symbols."""
    import selfies

    return list(selfies.split_selfies(value))


def tokenize_molecule(value: str, representation: str) -> list[str]:
    """Tokenize a molecule string in the configured representation."""
    return selfies_tokens(value) if representation == "selfies" else smiles_tokens(value)


@dataclass
class MolecularTokenizer:
    """Vocabulary-backed molecular tokenizer."""

    token_to_id: dict[str, int]
    id_to_token: dict[int, str]
    representation: str
    add_start_end_tokens: bool = True

    @classmethod
    def build(cls, strings: list[str], representation: str = "smiles", add_start_end_tokens: bool = True) -> "MolecularTokenizer":
        vocab = [PAD, START, END, UNK]
        observed = sorted({token for item in strings for token in tokenize_molecule(item, representation)})
        vocab.extend(token for token in observed if token not in vocab)
        token_to_id = {token: idx for idx, token in enumerate(vocab)}
        id_to_token = {idx: token for token, idx in token_to_id.items()}
        return cls(token_to_id=token_to_id, id_to_token=id_to_token, representation=representation, add_start_end_tokens=add_start_end_tokens)

    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD]

    @property
    def start_id(self) -> int:
        return self.token_to_id[START]

    @property
    def end_id(self) -> int:
        return self.token_to_id[END]

    def encode(self, value: str, add_special: bool | None = None) -> list[int]:
        special = self.add_start_end_tokens if add_special is None else add_special
        tokens = tokenize_molecule(value, self.representation)
        if special:
            tokens = [START, *tokens, END]
        return [self.token_to_id.get(token, self.token_to_id[UNK]) for token in tokens]

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        tokens: list[str] = []
        for idx in ids:
            token = self.id_to_token.get(int(idx), UNK)
            if skip_special and token in {PAD, START, END}:
                continue
            tokens.append(token)
        return "".join(tokens)

    def to_json(self, path: str | Path) -> None:
        payload = {
            "representation": self.representation,
            "add_start_end_tokens": self.add_start_end_tokens,
            "token_to_id": self.token_to_id,
            "id_to_token": {str(k): v for k, v in self.id_to_token.items()},
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def write_metadata(self, path: str | Path) -> None:
        payload = {
            "representation": self.representation,
            "vocab_size": len(self.token_to_id),
            "pad_token": PAD,
            "start_token": START,
            "end_token": END,
            "unk_token": UNK,
            "tokenizer_notes": "SELFIES symbols are used when representation=selfies; otherwise common SMILES tokens are preserved.",
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

