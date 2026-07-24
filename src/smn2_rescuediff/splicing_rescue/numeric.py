"""Numeric parsing helpers for Module 5."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


MISSING_NUMERIC_STRINGS = {"", "na", "n/a", "none", "null", "nan"}


def safe_float(value: Any, default: float = np.nan) -> float:
    """Convert a value to float while treating common missing tokens as NaN."""
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    if isinstance(value, str) and value.strip().lower() in MISSING_NUMERIC_STRINGS:
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def coerce_numeric_columns(df: pd.DataFrame, columns: set[str]) -> pd.DataFrame:
    """Coerce selected columns to numeric values, leaving missing entries blank on TSV export."""
    out = df.copy()
    for column in sorted(columns & set(out.columns)):
        out[column] = pd.to_numeric(out[column].replace(list(MISSING_NUMERIC_STRINGS), pd.NA), errors="coerce")
    return out
