"""SMILES/SELFIES representation conversion."""

from __future__ import annotations

from typing import Any


def selfies_available() -> bool:
    """Return whether SELFIES is installed."""
    try:
        import selfies  # noqa: F401
        return True
    except Exception:
        return False


def choose_representation(config: dict[str, Any]) -> tuple[str, list[str]]:
    """Choose configured representation, with optional SELFIES fallback."""
    token_cfg = config.get("tokenization", {})
    requested = str(token_cfg.get("representation", "smiles")).lower()
    warnings: list[str] = []
    if requested == "selfies" and not selfies_available():
        if bool(token_cfg.get("fallback_to_smiles_if_selfies_missing", True)):
            warnings.append("SELFIES was requested but the selfies package is missing; falling back to SMILES tokenization.")
            return "smiles", warnings
        raise ImportError("SELFIES was requested but the selfies package is not installed. Install `selfies` or use tokenization.representation: smiles.")
    if requested not in {"smiles", "selfies"}:
        raise ValueError("tokenization.representation must be either 'smiles' or 'selfies'.")
    return requested, warnings


def smiles_to_representation(smiles: list[str], representation: str) -> tuple[list[str], list[dict[str, str]]]:
    """Convert canonical SMILES into the training representation."""
    if representation == "smiles":
        return list(smiles), []
    import selfies

    converted: list[str] = []
    rejected: list[dict[str, str]] = []
    for item in smiles:
        try:
            converted.append(selfies.encoder(item))
        except Exception as exc:
            rejected.append({"canonical_smiles": item, "reason": f"selfies_encoding_failed: {exc}"})
    if not converted:
        raise ValueError("No molecules could be converted to SELFIES.")
    return converted, rejected


def representation_to_smiles(value: str, representation: str) -> str:
    """Decode a generated representation string to SMILES."""
    if representation == "smiles":
        return value
    import selfies

    try:
        return selfies.decoder(value)
    except Exception:
        return ""

