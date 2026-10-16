"""Fish-stock domain model for Mini-Project 3 (Lake Victoria, Jinja cooperative)."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

WEEKS_PER_YEAR = 52


def fibonacci_sequence(n: int, first: int = 0, second: int = 1) -> list[int]:
    """First ``n`` Fibonacci numbers (default 0, 1, 1, 2, 3, ...)."""
    if n <= 0:
        raise ValueError("n must be a positive integer.")
    seq = [first, second]
    while len(seq) < n:
        seq.append(seq[-1] + seq[-2])
    return seq[:n]


class FishStock:
    """Discrete logistic growth with proportional harvesting.

    ::

        N(t+1) = N(t) + r * N(t) * (1 - N(t)/K) - h * N(t)

    Time step is one week; ``h`` is the fraction of the stock caught each week.

    Parameters
    ----------
    r:
        Intrinsic weekly growth rate (> 0).
    carrying_capacity:
        ``K``, the stock the lake supports without fishing, in tonnes (> 0).
    initial_stock:
        ``N(0)`` in tonnes (>= 0).
    """

    def __init__(self, r: float = 0.4, carrying_capacity: float = 10_000.0,
                 initial_stock: float = 4_000.0) -> None:
        if r <= 0:
            raise ValueError("r must be positive.")
        if carrying_capacity <= 0:
            raise ValueError("carrying_capacity must be positive.")
        if initial_stock < 0:
            raise ValueError("initial_stock cannot be negative.")
        self.r = float(r)
        self.K = float(carrying_capacity)
        self.initial_stock = float(initial_stock)

    def __repr__(self) -> str:
        return f"FishStock(r={self.r}, K={self.K:,.0f}, N0={self.initial_stock:,.0f})"

    # ------------------------------------------------------------- simulation
    def simulate(self, harvest_rate: float, weeks: int = WEEKS_PER_YEAR,
                 closed_weeks: int = 0) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Simulate ``weeks`` weeks.

        Parameters
        ----------
        harvest_rate:
            Weekly harvest fraction ``h`` in [0, 1].
        closed_weeks:
            Number of weeks at the *start of every 52-week year* with no
            harvesting (a closed season), 0 to 52.

        Returns
        -------
        stock:  length ``weeks + 1`` array of tonnes, starting at ``N(0)``.
        catch:  length ``weeks`` array of tonnes harvested each week.
        """
        if not 0 <= harvest_rate <= 1:
            raise ValueError("harvest_rate must be between 0 and 1.")
        if weeks <= 0:
            raise ValueError("weeks must be positive.")
        if not 0 <= closed_weeks <= WEEKS_PER_YEAR:
            raise ValueError("closed_weeks must be between 0 and 52.")

        stock = np.empty(weeks + 1)
        catch = np.empty(weeks)
        stock[0] = self.initial_stock
        for t in range(weeks):
            h = 0.0 if (t % WEEKS_PER_YEAR) < closed_weeks else harvest_rate
            catch[t] = h * stock[t]
            growth = self.r * stock[t] * (1 - stock[t] / self.K)
            stock[t + 1] = max(stock[t] + growth - catch[t], 0.0)
        return stock, catch

    # ------------------------------------------------------- theory (sustainability)
    @property
    def msy(self) -> float:
        """Maximum sustainable yield ``rK/4`` in tonnes per week."""
        return self.r * self.K / 4

    @property
    def msy_harvest_rate(self) -> float:
        """Weekly harvest rate that achieves MSY: ``r/2``."""
        return self.r / 2

    @property
    def msy_stock(self) -> float:
        """Stock level that supports MSY: ``K/2``."""
        return self.K / 2

    def equilibrium_stock(self, harvest_rate: float) -> float:
        """Long-run stock ``K(1 - h/r)`` (0 if ``h >= r``: the stock collapses)."""
        if not 0 <= harvest_rate <= 1:
            raise ValueError("harvest_rate must be between 0 and 1.")
        return max(0.0, self.K * (1 - harvest_rate / self.r))

    def equilibrium_catch(self, harvest_rate: float) -> float:
        """Long-run weekly catch ``h * K(1 - h/r)`` in tonnes."""
        return harvest_rate * self.equilibrium_stock(harvest_rate)

    def sustainability_status(self, harvest_rate: float) -> str:
        """``'sustainable'`` (h <= r/2), ``'overfishing'`` (r/2 < h < r) or ``'collapse'`` (h >= r)."""
        if not 0 <= harvest_rate <= 1:
            raise ValueError("harvest_rate must be between 0 and 1.")
        if harvest_rate >= self.r:
            return "collapse"
        if harvest_rate > self.msy_harvest_rate:
            return "overfishing"
        return "sustainable"
