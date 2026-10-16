"""Forecast accuracy metrics and model comparison helpers."""
from __future__ import annotations

from typing import Callable, Mapping

import numpy as np
from numpy.typing import ArrayLike

from src.district import DistrictPopulation
from src.forecasters import Forecaster


def _prepare(actual: ArrayLike, predicted: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    if a.shape != p.shape:
        raise ValueError("actual and predicted must have the same shape.")
    if a.size == 0:
        raise ValueError("Inputs must not be empty.")
    return a, p


def mae(actual: ArrayLike, predicted: ArrayLike) -> float:
    """Mean absolute error (same units as the data)."""
    a, p = _prepare(actual, predicted)
    return float(np.mean(np.abs(a - p)))


def rmse(actual: ArrayLike, predicted: ArrayLike) -> float:
    """Root mean squared error (penalises large misses more than MAE)."""
    a, p = _prepare(actual, predicted)
    return float(np.sqrt(np.mean((a - p) ** 2)))


def mape(actual: ArrayLike, predicted: ArrayLike) -> float:
    """Mean absolute percentage error, in percent. Undefined if any actual is 0."""
    a, p = _prepare(actual, predicted)
    if np.any(a == 0):
        raise ValueError("MAPE is undefined when an actual value is zero.")
    return float(np.mean(np.abs((a - p) / a)) * 100)


def compare_models(
    district: DistrictPopulation,
    model_factories: Mapping[str, Callable[[], Forecaster]],
    last_train_year: int = 2021,
) -> dict[str, dict[str, float]]:
    """Train each model on data up to ``last_train_year``; score on the rest."""
    tr_years, tr_vals, te_years, te_vals = district.split(last_train_year)
    scores: dict[str, dict[str, float]] = {}
    for name, factory in model_factories.items():
        pred = factory().fit(tr_years, tr_vals).predict(len(te_years))
        scores[name] = {
            "MAE": mae(te_vals, pred),
            "RMSE": rmse(te_vals, pred),
            "MAPE": mape(te_vals, pred),
        }
    return scores


def select_best(scores: Mapping[str, Mapping[str, float]], metric: str = "MAPE") -> str:
    """Name of the model with the lowest ``metric``."""
    if not scores:
        raise ValueError("No scores to choose from.")
    return min(scores, key=lambda name: scores[name][metric])
