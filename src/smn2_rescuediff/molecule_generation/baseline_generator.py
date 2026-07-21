"""Deterministic n-gram baseline SMILES generator."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass

from .tokenizer import END, START, tokenize_smiles


@dataclass
class NGramSmilesGenerator:
    """A lightweight token n-gram generator for baseline comparisons."""

    ngram_size: int = 3
    random_seed: int = 42

    def __post_init__(self) -> None:
        if self.ngram_size < 2:
            raise ValueError("ngram_size must be at least 2.")
        self.transitions: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
        self.rng = random.Random(self.random_seed)

    def fit(self, smiles: list[str]) -> None:
        """Fit token transition counts from training SMILES."""
        context_size = self.ngram_size - 1
        for item in smiles:
            tokens = [START] * context_size + tokenize_smiles(item) + [END]
            for idx in range(context_size, len(tokens)):
                context = tuple(tokens[idx - context_size : idx])
                self.transitions[context][tokens[idx]] += 1
        if not self.transitions:
            raise ValueError("No transitions were learned by the baseline generator.")

    def _sample_next(self, context: tuple[str, ...], temperature: float) -> str:
        counts = self.transitions.get(context)
        if not counts:
            context = tuple([START] * (self.ngram_size - 1))
            counts = self.transitions[context]
        tokens = list(counts.keys())
        weights = [float(count) ** (1.0 / max(temperature, 1e-6)) for count in counts.values()]
        return self.rng.choices(tokens, weights=weights, k=1)[0]

    def generate_one(self, max_length: int = 120, temperature: float = 1.0) -> str:
        """Generate a single raw SMILES string."""
        context_size = self.ngram_size - 1
        context = tuple([START] * context_size)
        tokens: list[str] = []
        for _ in range(max_length):
            token = self._sample_next(context, temperature)
            if token == END:
                break
            tokens.append(token)
            context = tuple([*context[1:], token])
        return "".join(tokens)

    def generate(self, count: int, max_length: int = 120, temperature: float = 1.0) -> list[str]:
        """Generate multiple raw SMILES strings."""
        return [self.generate_one(max_length=max_length, temperature=temperature) for _ in range(max(0, count))]

