"""Forecasting models sharing one abstract interface (``fit`` / ``predict``)."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike, NDArray


def fibonacci_ratios(n: int) -> NDArray[np.float64]:
    """First ``n`` ratios F(k+1)/F(k): 1/1, 2/1, 3/2, 5/3, 8/5, ..."""
    if n <= 0:
        raise ValueError("n must be a positive integer.")
    fib = [1, 1]
    while len(fib) < n + 1:
        fib.append(fib[-1] + fib[-2])
    return np.array([fib[i + 1] / fib[i] for i in range(n)], dtype=float)


class Forecaster(ABC):
    """Abstract base class: ``fit(years, values)`` then ``predict(horizon)``."""

    def __init__(self) -> None:
        self.years: NDArray[np.float64] | None = None
        self.values: NDArray[np.float64] | None = None

    # ------------------------------------------------------------- shared logic
    def _store_data(self, years: ArrayLike, values: ArrayLike) -> None:
        years_arr = np.asarray(years, dtype=float)
        values_arr = np.asarray(values, dtype=float)
        if years_arr.ndim != 1 or values_arr.ndim != 1:
            raise ValueError("Years and values must be one-dimensional.")
        if len(years_arr) != len(values_arr):
            raise ValueError("Years and values must have equal length.")
        if len(years_arr) < 2:
            raise ValueError("At least two observations are required.")
        if not (np.all(np.isfinite(years_arr)) and np.all(np.isfinite(values_arr))):
            raise ValueError("Years and values must be finite.")
        if np.any(values_arr < 0):
            raise ValueError("Values cannot be negative.")
        if np.any(np.diff(years_arr) <= 0):
            raise ValueError("Years must be strictly increasing.")
        self.years, self.values = years_arr, values_arr

    def _check_ready(self, horizon: int) -> NDArray[np.float64]:
        """Validate fit state and horizon; return the future years."""
        if self.years is None or self.values is None:
            raise RuntimeError(f"{type(self).__name__}: call fit() before predict().")
        if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)):
            raise TypeError("horizon must be a positive integer.")
        if horizon <= 0:
            raise ValueError("horizon must be a positive integer.")
        return self.years[-1] + np.arange(1, horizon + 1)

    def future_years(self, horizon: int) -> NDArray[np.int_]:
        """Calendar years covered by ``predict(horizon)``."""
        return self._check_ready(horizon).astype(int)

    # -------------------------------------------------------------- interface
    @abstractmethod
    def fit(self, years: ArrayLike, values: ArrayLike) -> "Forecaster":
        """Fit on history and return ``self``."""

    @abstractmethod
    def predict(self, horizon: int) -> NDArray[np.float64]:
        """Forecast ``horizon`` steps beyond the last training year."""

    @abstractmethod
    def fitted_values(self) -> NDArray[np.float64]:
        """In-sample fitted values at the training years (NaN if not defined)."""


class LinearTrendForecaster(Forecaster):
    """Straight-line trend fitted with ``np.polyfit`` (least squares)."""

    def fit(self, years: ArrayLike, values: ArrayLike) -> "LinearTrendForecaster":
        self._store_data(years, values)
        self.slope, self.intercept = np.polyfit(self.years, self.values, deg=1)
        return self

    def predict(self, horizon: int) -> NDArray[np.float64]:
        return self.slope * self._check_ready(horizon) + self.intercept

    def fitted_values(self) -> NDArray[np.float64]:
        self._check_ready(1)
        return self.slope * self.years + self.intercept


class CAGRForecaster(Forecaster):
    """Exponential growth at the endpoint CAGR, projected from the last value."""

    def fit(self, years: ArrayLike, values: ArrayLike) -> "CAGRForecaster":
        self._store_data(years, values)
        if self.values[0] <= 0 or self.values[-1] <= 0:
            raise ValueError("CAGR needs positive first and last values.")
        span = self.years[-1] - self.years[0]
        self.rate: float = float((self.values[-1] / self.values[0]) ** (1 / span) - 1)
        return self

    def predict(self, horizon: int) -> NDArray[np.float64]:
        steps = self._check_ready(horizon) - self.years[-1]
        return self.values[-1] * (1 + self.rate) ** steps

    def fitted_values(self) -> NDArray[np.float64]:
        self._check_ready(1)
        return self.values[0] * (1 + self.rate) ** (self.years - self.years[0])


class FibonacciRatioForecaster(Forecaster):
    """Scale the last value by successive Fibonacci ratios 1/1, 2/1, 3/2, 5/3, ...

    Parameters
    ----------
    compound:
        ``False`` (default): every ratio multiplies the *same* last observed value
        (ratio k applied to the last value). ``True``: ratios are chained
        (cumulative product), i.e. growth compounds year after year.
    """

    def __init__(self, compound: bool = False) -> None:
        super().__init__()
        self.compound = compound

    def fit(self, years: ArrayLike, values: ArrayLike) -> "FibonacciRatioForecaster":
        self._store_data(years, values)
        return self

    def predict(self, horizon: int) -> NDArray[np.float64]:
        self._check_ready(horizon)
        ratios = fibonacci_ratios(horizon)
        if self.compound:
            ratios = np.cumprod(ratios)
        return self.values[-1] * ratios

    def fitted_values(self) -> NDArray[np.float64]:
        self._check_ready(1)
        return np.full_like(self.values, np.nan)
