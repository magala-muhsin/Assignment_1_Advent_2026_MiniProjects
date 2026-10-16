"""Seeded, bounded random-walk price model (UGX per kg)."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class PriceModel:
    """Weekly fish price as a random walk that cannot leave ``[lower, upper]``.

    Each week the price moves by a Normal(0, ``weekly_volatility``) shock and is
    then clipped to the bounds **at every step**, so a price that hits a bound
    can move back immediately (unlike clipping a whole unbounded path at the
    end, which would let the hidden walk drift far outside the bounds).

    Parameters
    ----------
    start, lower, upper:
        Starting price and bounds in UGX/kg (``lower <= start <= upper``).
    weekly_volatility:
        Standard deviation of the weekly price change in UGX/kg (> 0).
    seed:
        Default seed; ``simulate_paths`` restarts from it on every call, so the
        same seed always gives the same paths (useful for fair scenario comparisons).
    """

    def __init__(self, start: float = 12_000, lower: float = 9_000, upper: float = 16_000,
                 weekly_volatility: float = 400, seed: int = 2026) -> None:
        if not lower <= start <= upper:
            raise ValueError("Need lower <= start <= upper.")
        if lower <= 0:
            raise ValueError("Prices must be positive (lower > 0).")
        if weekly_volatility <= 0:
            raise ValueError("weekly_volatility must be positive.")
        self.start, self.lower, self.upper = float(start), float(lower), float(upper)
        self.weekly_volatility = float(weekly_volatility)
        self.seed = seed

    def __repr__(self) -> str:
        return (f"PriceModel(start={self.start:,.0f}, bounds=({self.lower:,.0f}, "
                f"{self.upper:,.0f}), vol={self.weekly_volatility:,.0f}, seed={self.seed})")

    def simulate_paths(self, n_paths: int, weeks: int,
                       seed: int | None = None) -> NDArray[np.float64]:
        """Simulate ``n_paths`` price paths of ``weeks`` weeks; shape ``(n_paths, weeks)``.

        Week 0 equals ``start``.
        """
        if n_paths < 1 or weeks < 1:
            raise ValueError("n_paths and weeks must be at least 1.")
        rng = np.random.default_rng(self.seed if seed is None else seed)
        shocks = rng.normal(0.0, self.weekly_volatility, size=(n_paths, weeks - 1))
        prices = np.empty((n_paths, weeks))
        prices[:, 0] = self.start
        for t in range(1, weeks):
            prices[:, t] = np.clip(prices[:, t - 1] + shocks[:, t - 1], self.lower, self.upper)
        return prices

    def simulate_path(self, weeks: int, seed: int | None = None) -> NDArray[np.float64]:
        """One price path of ``weeks`` weeks."""
        return self.simulate_paths(1, weeks, seed)[0]
