"""Rainy-season (peak) detection on a circular 12-month climatology."""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.signal import find_peaks


def detect_rainy_seasons(rainfall: ArrayLike, prominence_fraction: float = 0.30,
                         min_distance: int = 2) -> list[int]:
    """Month indices (0 = Jan) of distinct rainfall peaks.

    The 12 months are tiled three times so that a peak next to the
    December/January boundary is not missed, and only peaks in the middle copy
    are kept. A peak counts as a separate rainy season only if its
    **prominence** (how far it stands above the surrounding trough) is at least
    ``prominence_fraction`` of the wettest month. ``min_distance`` is the
    minimum number of months between peaks.
    """
    v = np.asarray(rainfall, dtype=float)
    if v.shape != (12,):
        raise ValueError("rainfall must have 12 monthly values.")
    if not np.all(np.isfinite(v)) or np.any(v < 0):
        raise ValueError("rainfall must be finite and non-negative.")
    if not 0 <= prominence_fraction < 1:
        raise ValueError("prominence_fraction must be in [0, 1).")
    if min_distance < 1:
        raise ValueError("min_distance must be at least 1.")
    if v.max() == 0 or np.ptp(v) == 0:
        return []
    peaks, _ = find_peaks(np.tile(v, 3), prominence=max(prominence_fraction * v.max(), 1e-9),
                          distance=min_distance)
    return [int(p) - 12 for p in peaks if 12 <= p < 24]


def classify_regime(peak_months: list[int]) -> str:
    """``'unimodal'``, ``'bimodal'``, ``'multimodal'`` or ``'no distinct season'``."""
    n = len(peak_months)
    if n == 0:
        return "no distinct season"
    return {1: "unimodal", 2: "bimodal"}.get(n, "multimodal")


def peak_count_sweep(rainfall: ArrayLike, fractions: list[float]) -> dict[float, int]:
    """Number of peaks detected at each prominence threshold (a sensitivity check)."""
    return {f: len(detect_rainy_seasons(rainfall, f)) for f in fractions}
