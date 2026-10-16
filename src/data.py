"""Illustrative district population data (thousands), 2015-2024.

Kampala, Wakiso and Gulu are given in the brief. Mbarara and Jinja are
illustrative series added for this project (synthetic, not UBOS figures).
"""
from __future__ import annotations

import numpy as np

from src.district import DistrictPopulation

YEARS: np.ndarray = np.arange(2015, 2025)

POPULATIONS: dict[str, list[float]] = {
    "Kampala": [1200, 1250, 1300, 1350, 1420, 1500, 1580, 1650, 1720, 1800],
    "Wakiso": [950, 1000, 1070, 1150, 1220, 1300, 1390, 1480, 1570, 1670],
    "Gulu": [320, 330, 345, 360, 375, 390, 410, 430, 455, 480],
    # Added districts: illustrative / synthetic.
    "Mbarara": [480, 495, 510, 530, 550, 575, 600, 630, 660, 695],
    "Jinja": [470, 480, 495, 510, 525, 545, 565, 590, 615, 645],
}


def load_districts() -> list[DistrictPopulation]:
    """Build one :class:`DistrictPopulation` per entry in ``POPULATIONS``."""
    return [DistrictPopulation(name, YEARS, vals) for name, vals in POPULATIONS.items()]
