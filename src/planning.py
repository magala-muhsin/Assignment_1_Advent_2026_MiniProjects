"""Primary-school classroom planning from population forecasts."""
from __future__ import annotations

import math


def classrooms_needed(
    population_thousands: float,
    school_age_share: float = 0.18,
    pupils_per_classroom: int = 53,
) -> int:
    """Classrooms required to seat all primary-age pupils (rounded up)."""
    if population_thousands < 0:
        raise ValueError("Population cannot be negative.")
    if not 0 <= school_age_share <= 1:
        raise ValueError("school_age_share must be between 0 and 1.")
    if pupils_per_classroom <= 0:
        raise ValueError("pupils_per_classroom must be positive.")
    pupils = population_thousands * 1000 * school_age_share
    return math.ceil(round(pupils / pupils_per_classroom, 9))


def additional_classrooms(
    population_now: float,
    population_future: float,
    school_age_share: float = 0.18,
    pupils_per_classroom: int = 53,
) -> int:
    """Extra classrooms needed by the future year.

    Assumption: the current stock of classrooms exactly meets current need
    (``classrooms_needed(population_now)``).
    """
    now = classrooms_needed(population_now, school_age_share, pupils_per_classroom)
    future = classrooms_needed(population_future, school_age_share, pupils_per_classroom)
    return future - now
