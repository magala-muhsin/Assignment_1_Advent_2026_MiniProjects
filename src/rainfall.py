"""Domain classes for Mini-Project 4: regional rainfall and crop suitability."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

MONTHS: tuple[str, ...] = ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
                           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")

DROUGHT = "Drought risk"
WATERLOGGING = "Waterlogging risk"


class Region:
    """Twelve monthly rainfall totals (mm, January to December) for one region.

    Parameters
    ----------
    name:
        Region name (non-empty).
    rainfall:
        Exactly 12 finite, non-negative monthly totals in mm.
    """

    def __init__(self, name: str, rainfall: ArrayLike) -> None:
        name = str(name).strip()
        if not name:
            raise ValueError("Region name must be non-empty.")
        values = np.asarray(rainfall, dtype=float)
        if values.shape != (12,):
            raise ValueError("Rainfall must be a 1-D array of 12 monthly values.")
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError("Rainfall values must be finite and non-negative.")
        self.name: str = name
        self.rainfall: NDArray[np.float64] = values

    def __repr__(self) -> str:
        return f"Region({self.name!r}, annual={self.annual_total():,.0f} mm)"

    def __len__(self) -> int:
        return len(self.rainfall)

    def annual_total(self) -> float:
        """Sum of the 12 months, in mm."""
        return float(self.rainfall.sum())

    def monthly_mean(self) -> float:
        """Mean monthly rainfall, in mm."""
        return float(self.rainfall.mean())

    def wettest_month(self) -> tuple[str, float]:
        """(month name, mm) of the wettest month."""
        i = int(np.argmax(self.rainfall))
        return MONTHS[i], float(self.rainfall[i])

    def driest_month(self) -> tuple[str, float]:
        """(month name, mm) of the driest month."""
        i = int(np.argmin(self.rainfall))
        return MONTHS[i], float(self.rainfall[i])

    def coefficient_of_variation(self) -> float:
        """Standard deviation / mean of the 12 monthly totals (population formula).

        The 12 months are the *complete* annual cycle, so ``ddof=0`` is used.
        Unitless; a high value means a strongly seasonal climate.
        """
        mean = self.rainfall.mean()
        if mean == 0:
            raise ValueError("CV is undefined for a region with zero rainfall.")
        return float(self.rainfall.std(ddof=0) / mean)


@dataclass(frozen=True)
class CropRule:
    """Rainfall rule for one crop, derived from a *seasonal* water need.

    The suitable monthly band is the seasonal band spread evenly over the crop
    cycle: ``monthly = seasonal / cycle_months``. This is a transparent
    screening heuristic, not a validated agronomic model (see the notebook).

    Parameters
    ----------
    name:
        Crop name.
    seasonal_min_mm, seasonal_max_mm:
        Water need over the whole crop cycle, in mm.
    cycle_months:
        Length of the crop cycle in months (12 for a perennial such as coffee).
    source:
        Citation for the thresholds.
    """

    name: str
    seasonal_min_mm: float
    seasonal_max_mm: float
    cycle_months: int
    source: str

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Crop name must be non-empty.")
        if self.seasonal_min_mm < 0 or self.seasonal_max_mm <= self.seasonal_min_mm:
            raise ValueError("Need 0 <= seasonal_min_mm < seasonal_max_mm.")
        if not 1 <= self.cycle_months <= 12:
            raise ValueError("cycle_months must be between 1 and 12.")

    @property
    def monthly_min_mm(self) -> float:
        """Lower edge of the good monthly band (mm)."""
        return self.seasonal_min_mm / self.cycle_months

    @property
    def monthly_max_mm(self) -> float:
        """Upper edge of the good monthly band (mm)."""
        return self.seasonal_max_mm / self.cycle_months

    def classify(self, monthly_rainfall_mm: float) -> str:
        """``'Drought risk'``, ``'Good for <crop>'`` or ``'Waterlogging risk'``."""
        if monthly_rainfall_mm < 0:
            raise ValueError("Rainfall cannot be negative.")
        if monthly_rainfall_mm < self.monthly_min_mm:
            return DROUGHT
        if monthly_rainfall_mm > self.monthly_max_mm:
            return WATERLOGGING
        return f"Good for {self.name.lower()}"

    def classify_region(self, region: Region) -> list[str]:
        """Classification of each of the 12 months of ``region``."""
        return [self.classify(v) for v in region.rainfall]

    def cycle_total(self, region: Region, start_month: int) -> float:
        """Rain (mm) over one full crop cycle starting at ``start_month`` (0 = Jan).

        The calendar wraps around, so a cycle starting in November runs into
        the next year's January, February, ...
        """
        if not 0 <= start_month < 12:
            raise ValueError("start_month must be between 0 and 11.")
        idx = [(start_month + k) % 12 for k in range(self.cycle_months)]
        return float(region.rainfall[idx].sum())

    def planting_windows(self, region: Region) -> list[dict]:
        """Rain over a full cycle for each possible planting month.

        ``status`` is ``'adequate'`` (total inside the seasonal band),
        ``'deficit'`` (below it) or ``'excess'`` (above it). The lowest single
        month in the cycle is also reported.
        """
        out = []
        for m in range(12):
            total = self.cycle_total(region, m)
            idx = [(m + k) % 12 for k in range(self.cycle_months)]
            if total < self.seasonal_min_mm:
                status = "deficit"
            elif total > self.seasonal_max_mm:
                status = "excess"
            else:
                status = "adequate"
            out.append({"plant": MONTHS[m],
                        "cycle": f"{MONTHS[idx[0]]}-{MONTHS[idx[-1]]}",
                        "total_mm": total,
                        "lowest_month_mm": float(region.rainfall[idx].min()),
                        "status": status})
        return out


def default_crops() -> list[CropRule]:
    """Maize, beans and coffee with thresholds from the cited sources.

    - Maize: 500-800 mm per cycle (FAO: a medium-maturity grain crop needs
      500-800 mm depending on climate); a 4-month cycle is assumed.
    - Beans: 300-500 mm per cycle (FAO: 60-120-day crop); a 3-month cycle
      (about 90 days, inside FAO's range) is assumed.
    - Coffee: 1,200-1,800 mm per year, spread over 12 months (annual ranges
      reported for Robusta; they are general figures, not Uganda-specific).
    """
    return [
        CropRule("Maize", 500, 800, 4,
                 "FAO crop water information, Maize; Brouwer & Heibloem (1986), "
                 "FAO Irrigation Water Management Manual No. 3, Table 2 (500-800 mm)"),
        CropRule("Beans", 300, 500, 3,
                 "FAO crop water information, Bean (300-500 mm for a 60-120-day crop); "
                 "Brouwer & Heibloem (1986), Table 2"),
        CropRule("Coffee", 1200, 1800, 12,
                 "Annual rainfall 1,200-1,800 mm for Robusta (Brunini et al., ASIC 2025 poster); "
                 "UCDA Robusta Coffee Handbook (2019), section 2.4, should be checked for Uganda"),
    ]
