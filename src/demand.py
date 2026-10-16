"""Demand data: seeded generation, CSV loading and robust interactive input."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from numpy.typing import NDArray

DEMAND_COLUMNS = ("D1_kWh", "D2_kWh")

# Weekly pattern, Monday..Sunday (illustrative): the clinic is busiest on
# weekdays (high daytime load D1), while critical equipment (refrigeration,
# oxygen, lab) runs all week with only a slight weekday bump.
_D1_WEEK = np.array([72, 74, 73, 72, 70, 54, 47], dtype=float)
_D2_WEEK = np.array([68, 69, 68, 68, 67, 66, 65], dtype=float)


def generate_demand(n_days: int = 30, seed: int = 42, start: str = "2026-01-05") -> pd.DataFrame:
    """Synthetic daily demand with a weekly pattern plus noise (seeded).

    ``start`` is a Monday, so day 0 maps to Monday. Loads are never negative.
    """
    if n_days <= 0:
        raise ValueError("n_days must be positive.")
    rng = np.random.default_rng(seed)
    dow = np.arange(n_days) % 7
    d1 = _D1_WEEK[dow] + rng.normal(0, 3.5, n_days)
    d2 = _D2_WEEK[dow] + rng.normal(0, 2.5, n_days)
    return pd.DataFrame(
        {
            "date": pd.date_range(start, periods=n_days, freq="D").strftime("%Y-%m-%d"),
            "D1_kWh": np.round(np.maximum(d1, 0), 2),
            "D2_kWh": np.round(np.maximum(d2, 0), 2),
        }
    )


def load_demand_csv(path: str | Path) -> tuple[pd.DataFrame, NDArray[np.float64]]:
    """Read a demand CSV; return the DataFrame and a 2 x days array (rows D1, D2).

    Raises ``ValueError`` for missing columns, non-numeric, missing, negative
    or empty data, and ``FileNotFoundError`` if the file does not exist.
    """
    df = pd.read_csv(path)
    missing = [c for c in DEMAND_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing column(s): {missing}")
    if df.empty:
        raise ValueError("CSV contains no rows.")
    values = df[list(DEMAND_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    if values.isna().any().any():
        raise ValueError("Demand columns must be numeric with no missing values.")
    if (values < 0).any().any():
        raise ValueError("Demand values cannot be negative.")
    return df, values.to_numpy(dtype=float).T


def prompt_nonnegative_number(
    prompt: str,
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[[str], None] = print,
) -> float:
    """Ask until the user types a finite, non-negative number.

    Rejects empty, non-numeric, non-finite (nan/inf) and negative entries and
    re-prompts. ``input_fn`` and ``print_fn`` can be replaced for testing.
    """
    while True:
        text = input_fn(prompt).strip()
        if text == "":
            print_fn("Value cannot be empty. Try again.")
            continue
        try:
            value = float(text)
        except ValueError:
            print_fn(f"'{text}' is not a number. Try again.")
            continue
        if not np.isfinite(value):
            print_fn("Value must be finite. Try again.")
            continue
        if value < 0:
            print_fn("Value cannot be negative. Try again.")
            continue
        return value
