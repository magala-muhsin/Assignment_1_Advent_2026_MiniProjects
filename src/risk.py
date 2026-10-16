"""Revenue calculation and risk assessment for the export cooperative."""
from __future__ import annotations

import statistics
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

KG_PER_TONNE = 1_000


def weekly_revenue(catch_tonnes: ArrayLike, prices: ArrayLike) -> NDArray[np.float64]:
    """Weekly revenue in UGX = harvest (kg) x price (UGX/kg).

    ``prices`` may be 1-D (one path) or 2-D (paths x weeks).
    """
    catch = np.asarray(catch_tonnes, dtype=float)
    p = np.asarray(prices, dtype=float)
    if catch.ndim != 1:
        raise ValueError("catch_tonnes must be 1-D.")
    if p.shape[-1] != catch.shape[0]:
        raise ValueError("prices and catch must cover the same number of weeks.")
    if np.any(catch < 0) or np.any(p < 0):
        raise ValueError("Catch and prices cannot be negative.")
    return p * catch * KG_PER_TONNE


def annual_revenue(catch_tonnes: ArrayLike, price_paths: ArrayLike) -> NDArray[np.float64]:
    """Total revenue per simulated path (UGX), summed over all weeks supplied."""
    return weekly_revenue(catch_tonnes, np.atleast_2d(price_paths)).sum(axis=1)


def revenue_by_year(catch_tonnes: ArrayLike, price_paths: ArrayLike,
                    weeks_per_year: int = 52) -> NDArray[np.float64]:
    """Revenue per path per year, shape ``(paths, years)``; weeks must be a whole number of years."""
    w = weekly_revenue(catch_tonnes, np.atleast_2d(price_paths))
    if w.shape[1] % weeks_per_year:
        raise ValueError("Number of weeks must be a multiple of weeks_per_year.")
    return w.reshape(w.shape[0], -1, weeks_per_year).sum(axis=2)


def describe_revenue(values: ArrayLike) -> dict[str, float]:
    """Mean, median, sample variance, st. dev. and CV, computed with ``statistics``.

    Units: mean/median/stdev in UGX, variance in **UGX squared**, CV unitless.
    """
    vals = [float(v) for v in values]
    if len(vals) < 2:
        raise ValueError("Need at least two values.")
    mean = statistics.mean(vals)
    if mean == 0:
        raise ValueError("CV is undefined when the mean revenue is zero.")
    sd = statistics.stdev(vals)
    return {"mean": mean, "median": statistics.median(vals),
            "variance": statistics.variance(vals), "stdev": sd, "cv": sd / mean}


@dataclass
class RiskReport:
    """Result of :meth:`RiskAssessor.assess` for one scenario (all money in UGX)."""

    mean: float
    stdev: float
    cv: float
    risk_class: str
    percentile_5: float          # revenue that is beaten in 95% of years
    value_at_risk: float         # mean - percentile_5: shortfall vs expectation
    var_fraction: float          # value_at_risk / mean


class RiskAssessor:
    """Classify revenue risk from the coefficient of variation (CV) of annual revenue.

    Rule (CV = st. dev. / mean, unitless, so it does not depend on the currency
    unit or on how large the catch is):

    - **Low**      CV < ``low``      (default 0.10)
    - **Moderate** ``low`` <= CV < ``high``
    - **High**     CV >= ``high``    (default 0.25)

    Justification: for roughly bell-shaped revenue the 5%-worst year falls about
    ``1.645 x CV`` below the mean. CV 0.10 therefore means a bad year loses
    about 16% of expected revenue (a small reserve covers it), and CV 0.25 means
    about 41% (a bad year could threaten fixed costs). The thresholds are a
    planning judgement, not a law, and can be changed through the constructor.
    """

    def __init__(self, low: float = 0.10, high: float = 0.25, var_level: float = 0.05) -> None:
        if not 0 < low < high:
            raise ValueError("Need 0 < low < high.")
        if not 0 < var_level < 0.5:
            raise ValueError("var_level must be between 0 and 0.5.")
        self.low, self.high, self.var_level = low, high, var_level

    def __repr__(self) -> str:
        return f"RiskAssessor(low={self.low}, high={self.high}, var_level={self.var_level})"

    def classify(self, cv: float) -> str:
        """Return ``'Low'``, ``'Moderate'`` or ``'High'`` for a CV."""
        if cv < 0:
            raise ValueError("CV cannot be negative.")
        if cv < self.low:
            return "Low"
        if cv < self.high:
            return "Moderate"
        return "High"

    def assess(self, annual_revenues: ArrayLike) -> RiskReport:
        """CV, risk class and Value-at-Risk from simulated annual revenues."""
        stats = describe_revenue(annual_revenues)
        p = float(np.percentile(np.asarray(annual_revenues, dtype=float), self.var_level * 100))
        return RiskReport(
            mean=stats["mean"], stdev=stats["stdev"], cv=stats["cv"],
            risk_class=self.classify(stats["cv"]), percentile_5=p,
            value_at_risk=stats["mean"] - p, var_fraction=(stats["mean"] - p) / stats["mean"],
        )
