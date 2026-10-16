"""Dispatch analysis: feasibility handling, statistics, costs, sensitivity."""
from __future__ import annotations

import statistics
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import solve
from scipy.optimize import nnls

from src.microgrid import MicroGrid


@dataclass
class DispatchResult:
    """Outcome of solving many days and enforcing x, y >= 0.

    Attributes
    ----------
    raw:            exact solutions, shape (n_sources, days); may contain negatives.
    dispatch:       final non-negative solutions, same shape.
    infeasible:     boolean mask (days,) of days with a negative exact solution.
    residual_norm:  ||A @ dispatch - demands|| per day (kWh of load not matched).
    strategy:       "nnls" or "clip".
    """

    raw: NDArray[np.float64]
    dispatch: NDArray[np.float64]
    infeasible: NDArray[np.bool_]
    residual_norm: NDArray[np.float64]
    strategy: str

    @property
    def n_infeasible(self) -> int:
        return int(self.infeasible.sum())


def resolve_dispatch(
    grid: MicroGrid,
    demands: ArrayLike,
    strategy: str = "nnls",
    tol: float = 1e-9,
) -> DispatchResult:
    """Solve all days, flag physically infeasible ones and repair them.

    Strategies
    ----------
    ``"nnls"``: non-negative least squares (``scipy.optimize.nnls``): the
        non-negative (x, y) that minimises ``||A @ s - b||``.
    ``"clip"``: set negative entries to zero and keep the rest.
    """
    if strategy not in ("nnls", "clip"):
        raise ValueError("strategy must be 'nnls' or 'clip'.")
    B = np.asarray(demands, dtype=float)
    raw = grid.solve_batch(B)
    infeasible = np.any(raw < -tol, axis=0)
    dispatch = raw.copy()
    for j in np.flatnonzero(infeasible):
        if strategy == "nnls":
            dispatch[:, j], _ = nnls(grid.A, B[:, j])
        else:
            dispatch[:, j] = np.clip(raw[:, j], 0, None)
    dispatch = np.clip(dispatch, 0, None)  # removes tiny negative round-off
    residual = np.linalg.norm(grid.A @ dispatch - B, axis=0)
    return DispatchResult(raw, dispatch, infeasible, residual, strategy)


def source_statistics(values: ArrayLike) -> dict[str, float]:
    """Mean, sample variance, sample st. dev. and CV (%) using ``statistics``."""
    vals = [float(v) for v in values]
    if len(vals) < 2:
        raise ValueError("Need at least two values.")
    mean = statistics.mean(vals)
    if mean == 0:
        raise ValueError("Coefficient of variation is undefined for a zero mean.")
    sd = statistics.stdev(vals)
    return {
        "mean": mean,
        "variance": statistics.variance(vals),
        "stdev": sd,
        "cv_percent": sd / mean * 100,
    }


def energy_costs(
    solar: ArrayLike,
    battery: ArrayLike,
    solar_rate: float = 150.0,
    battery_rate: float = 450.0,
) -> dict[str, NDArray[np.float64] | float]:
    """Daily and monthly cost in UGX given per-kWh rates."""
    if solar_rate < 0 or battery_rate < 0:
        raise ValueError("Rates cannot be negative.")
    s = np.asarray(solar, dtype=float)
    b = np.asarray(battery, dtype=float)
    if s.shape != b.shape:
        raise ValueError("solar and battery must have the same shape.")
    solar_cost, battery_cost = s * solar_rate, b * battery_rate
    daily = solar_cost + battery_cost
    return {
        "daily": daily,
        "solar_total": float(solar_cost.sum()),
        "battery_total": float(battery_cost.sum()),
        "monthly_total": float(daily.sum()),
    }


@dataclass
class SensitivityResult:
    """Monte Carlo sensitivity of the exact solution to demand errors."""

    mean_move: NDArray[np.float64]   # kWh, per source
    p95_move: NDArray[np.float64]    # kWh, per source
    amplification_max: float
    amplification_mean: float
    condition_number: float
    draws: NDArray[np.float64]       # (n_draws, n_sources, days) perturbed solutions
    baseline: NDArray[np.float64]    # (n_sources, days)


def sensitivity_analysis(
    grid: MicroGrid,
    demands: ArrayLike,
    pct: float = 0.05,
    n_draws: int = 1000,
    seed: int = 2026,
) -> SensitivityResult:
    """Perturb every demand by independent uniform +/- ``pct`` and re-solve.

    All ``n_draws x days`` systems are solved in a single vectorised call.
    ``amplification`` = (relative change in solution) / (relative change in
    demands), both in the 2-norm; it can never exceed the condition number.
    """
    if not 0 < pct < 1:
        raise ValueError("pct must be between 0 and 1.")
    if n_draws < 1:
        raise ValueError("n_draws must be at least 1.")
    B = np.asarray(demands, dtype=float)
    base = grid.solve_batch(B)
    n, days = B.shape
    rng = np.random.default_rng(seed)
    pert = B[None] * rng.uniform(1 - pct, 1 + pct, size=(n_draws, n, days))
    flat = pert.transpose(1, 0, 2).reshape(n, -1)
    sol = solve(grid.A, flat).reshape(n, n_draws, days).transpose(1, 0, 2)

    moves = np.abs(sol - base[None])                     # (draws, n, days)
    rel_in = np.linalg.norm(pert - B[None], axis=1) / np.linalg.norm(B, axis=0)[None]
    rel_out = np.linalg.norm(sol - base[None], axis=1) / np.linalg.norm(base, axis=0)[None]
    amp = rel_out / rel_in
    return SensitivityResult(
        mean_move=moves.mean(axis=(0, 2)),
        p95_move=np.percentile(moves, 95, axis=(0, 2)),
        amplification_max=float(amp.max()),
        amplification_mean=float(amp.mean()),
        condition_number=grid.condition_number,
        draws=sol,
        baseline=base,
    )
