"""The DistrictPopulation domain class."""
from __future__ import annotations

import statistics

import numpy as np
from numpy.typing import ArrayLike, NDArray


class DistrictPopulation:
    """Yearly population estimates (in thousands) for one district.

    Parameters
    ----------
    name:
        District name (non-empty).
    years:
        Strictly increasing integer calendar years.
    populations:
        Population estimates in thousands, one per year. Must be finite and
        non-negative.

    Raises
    ------
    ValueError
        If the inputs are empty, not 1-D, of unequal length, contain
        non-finite or negative values, or the years are not strictly
        increasing integers.
    """

    def __init__(self, name: str, years: ArrayLike, populations: ArrayLike) -> None:
        name = str(name).strip()
        if not name:
            raise ValueError("District name must be non-empty.")

        years_arr = np.asarray(years)
        pops_arr = np.asarray(populations, dtype=float)

        if years_arr.ndim != 1 or pops_arr.ndim != 1:
            raise ValueError("Years and populations must be one-dimensional.")
        if len(years_arr) == 0:
            raise ValueError("At least one observation is required.")
        if len(years_arr) != len(pops_arr):
            raise ValueError(
                f"Years and populations must have equal length "
                f"(got {len(years_arr)} and {len(pops_arr)})."
            )
        if not np.all(np.isfinite(years_arr.astype(float))):
            raise ValueError("Years must be finite.")
        if not np.all(years_arr == np.floor(years_arr)):
            raise ValueError("Years must be whole numbers.")
        if not np.all(np.isfinite(pops_arr)):
            raise ValueError("Populations must be finite numbers.")
        if np.any(pops_arr < 0):
            raise ValueError("Populations cannot be negative.")

        years_arr = years_arr.astype(int)
        if np.any(np.diff(years_arr) <= 0):
            raise ValueError("Years must be strictly increasing.")

        self.name: str = name
        self.years: NDArray[np.int_] = years_arr
        self.populations: NDArray[np.float64] = pops_arr

    # ------------------------------------------------------------------ dunders
    def __len__(self) -> int:
        return len(self.years)

    def __repr__(self) -> str:
        return (
            f"DistrictPopulation(name={self.name!r}, "
            f"years={self.years[0]}-{self.years[-1]}, n={len(self)})"
        )

    # ------------------------------------------------------------- descriptive
    def describe_statistics(self) -> dict[str, float]:
        """Mean, median, sample variance and sample st. dev. (``statistics`` module).

        ``statistics.variance``/``stdev`` use the *sample* formula (divide by n-1).
        Needs at least two observations.
        """
        vals = self.populations.tolist()
        return {
            "mean": statistics.mean(vals),
            "median": statistics.median(vals),
            "variance": statistics.variance(vals),
            "stdev": statistics.stdev(vals),
        }

    def describe_numpy(self, ddof: int = 0) -> dict[str, float]:
        """Mean, median, variance and st. dev. using NumPy.

        ``ddof`` ("delta degrees of freedom") sets the divisor to ``n - ddof``.
        NumPy's default is 0 (population formula); ``ddof=1`` gives the sample
        formula and matches the ``statistics`` module.
        """
        return {
            "mean": float(np.mean(self.populations)),
            "median": float(np.median(self.populations)),
            "variance": float(np.var(self.populations, ddof=ddof)),
            "stdev": float(np.std(self.populations, ddof=ddof)),
        }

    # ------------------------------------------------------------------ growth
    def growth_rates(self) -> NDArray[np.float64]:
        """Year-on-year growth as fractions (0.04 = 4%); length ``len(self) - 1``."""
        if len(self) < 2:
            raise ValueError("At least two observations are needed for growth rates.")
        if np.any(self.populations[:-1] == 0):
            raise ValueError("Growth rate undefined when a population is zero.")
        return np.diff(self.populations) / self.populations[:-1]

    def cagr(self) -> float:
        """Compound annual growth rate from first to last observation (fraction)."""
        span = int(self.years[-1] - self.years[0])
        if span <= 0:
            raise ValueError("CAGR needs at least two distinct years.")
        if self.populations[0] <= 0 or self.populations[-1] <= 0:
            raise ValueError("CAGR needs positive first and last populations.")
        return float((self.populations[-1] / self.populations[0]) ** (1 / span) - 1)

    # ---------------------------------------------------------------- splitting
    def split(self, last_train_year: int) -> tuple[
        NDArray[np.int_], NDArray[np.float64], NDArray[np.int_], NDArray[np.float64]
    ]:
        """Chronological train/test split.

        Returns ``(train_years, train_pops, test_years, test_pops)`` where the
        training set ends at ``last_train_year`` (inclusive).
        """
        mask = self.years <= last_train_year
        if not mask.any() or mask.all():
            raise ValueError("last_train_year must leave both train and test data.")
        return (
            self.years[mask],
            self.populations[mask],
            self.years[~mask],
            self.populations[~mask],
        )
