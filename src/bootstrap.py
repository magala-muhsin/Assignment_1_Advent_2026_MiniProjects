"""Residual-bootstrap prediction intervals for the CAGR model."""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.forecasters import CAGRForecaster


def bootstrap_cagr_interval(
    years: ArrayLike,
    values: ArrayLike,
    horizon: int,
    n_resamples: int = 2000,
    level: float = 0.95,
    seed: int = 42,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Prediction interval for a CAGR forecast via residual bootstrap.

    Method
    ------
    1. Fit CAGR; compute *relative* residuals ``y/fitted - 1``. The first and
       last residuals are zero by construction (the curve passes through both
       endpoints), so they are dropped, and the rest are centred.
    2. For each resample: draw residuals to build a synthetic history around
       the fitted curve, refit CAGR on it (parameter uncertainty), project
       forward, and multiply by a freshly drawn residual per future year
       (irreducible noise).
    3. Return the ``(1-level)/2`` and ``(1+level)/2`` percentiles per horizon.
    """
    if not 0 < level < 1:
        raise ValueError("level must be between 0 and 1.")
    if n_resamples < 1:
        raise ValueError("n_resamples must be at least 1.")

    model = CAGRForecaster().fit(years, values)
    if len(model.values) < 4:
        raise ValueError("Need at least 4 observations to bootstrap residuals.")
    model._check_ready(horizon)  # validates horizon

    rng = np.random.default_rng(seed)
    fitted = model.fitted_values()
    rel = (model.values / fitted - 1)[1:-1]
    rel = rel - rel.mean()

    n, span = len(fitted), model.years[-1] - model.years[0]
    synthetic = fitted * (1 + rng.choice(rel, size=(n_resamples, n)))
    rates = (synthetic[:, -1] / synthetic[:, 0]) ** (1 / span) - 1
    steps = np.arange(1, horizon + 1)
    paths = synthetic[:, [-1]] * (1 + rates[:, None]) ** steps
    paths *= 1 + rng.choice(rel, size=(n_resamples, horizon))

    tail = (1 - level) / 2 * 100
    lower, upper = np.percentile(paths, [tail, 100 - tail], axis=0)
    return lower, upper
