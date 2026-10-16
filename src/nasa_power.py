"""NASA POWER monthly rainfall: fetch, aggregate, validate and load (extension)."""
from __future__ import annotations

import calendar
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.rainfall import MONTHS

ENDPOINT = "https://power.larc.nasa.gov/api/temporal/daily/point"
NASA_LOCATIONS: dict[str, tuple[float, float]] = {   # (latitude, longitude)
    "Kampala": (0.3476, 32.5825),
    "Gulu": (2.7746, 32.2990),
    "Mbarara": (-0.6072, 30.6545),
}
REQUIRED_COLUMNS = ("region", "year", "month", "rainfall_mm")
MISSING = -999


def aggregate_daily_to_monthly(region: str, daily: Mapping[str, float]) -> pd.DataFrame:
    """Sum daily precipitation (mm/day, keyed ``YYYYMMDD``) into monthly totals.

    Days flagged ``-999`` (missing) or with a malformed date are skipped. The
    result has ``n_days`` (days counted) and ``complete`` (all days present).
    """
    rows = []
    for key, value in daily.items():
        if len(key) != 8 or not key.isdigit() or value == MISSING:
            continue
        rows.append((int(key[:4]), int(key[4:6]), float(value)))
    if not rows:
        raise ValueError("No valid daily values to aggregate.")
    df = pd.DataFrame(rows, columns=["year", "month", "mm"])
    out = df.groupby(["year", "month"], as_index=False).agg(rainfall_mm=("mm", "sum"), n_days=("mm", "size"))
    out["days_in_month"] = [calendar.monthrange(y, m)[1] for y, m in zip(out["year"], out["month"])]
    out["complete"] = out["n_days"] == out["days_in_month"]
    out.insert(0, "region", region)
    out["month_name"] = out["month"].map(lambda m: MONTHS[m - 1])
    return out.drop(columns="days_in_month")


def fetch_nasa_monthly(locations: Mapping[str, tuple[float, float]] = NASA_LOCATIONS,
                       start_year: int = 2014, end_year: int = 2023,
                       timeout: int = 90) -> pd.DataFrame:
    """Download daily PRECTOTCORR from NASA POWER and aggregate it to monthly totals.

    Needs internet access and the ``requests`` package. Gridded/reanalysis
    estimates, not rain-gauge records.
    """
    import requests  # imported here so the rest of the project works without it

    if end_year < start_year:
        raise ValueError("end_year must not be before start_year.")
    frames = []
    for name, (lat, lon) in locations.items():
        params = {"parameters": "PRECTOTCORR", "community": "AG", "longitude": lon,
                  "latitude": lat, "format": "JSON", "start": start_year, "end": end_year}
        response = requests.get(ENDPOINT, params=params, timeout=timeout)
        response.raise_for_status()
        daily = response.json()["properties"]["parameter"]["PRECTOTCORR"]
        frames.append(aggregate_daily_to_monthly(name, daily))
    return pd.concat(frames, ignore_index=True)


def load_nasa_monthly(path: str | Path) -> pd.DataFrame:
    """Read and validate a monthly NASA POWER CSV (columns region, year, month, rainfall_mm)."""
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing column(s): {missing}")
    if df.empty:
        raise ValueError("CSV contains no rows.")
    for col in ("year", "month", "rainfall_mm"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[["year", "month", "rainfall_mm"]].isna().any().any():
        raise ValueError("year, month and rainfall_mm must be numeric with no missing values.")
    if not df["month"].between(1, 12).all():
        raise ValueError("month must be between 1 and 12.")
    if (df["rainfall_mm"] < 0).any():
        raise ValueError("rainfall_mm cannot be negative.")
    return df


def monthly_matrix(df: pd.DataFrame, region: str) -> NDArray[np.float64]:
    """Rainfall as a (years x 12) array for one region; every month must be present."""
    sub = df[df["region"] == region]
    if sub.empty:
        raise ValueError(f"No data for region {region!r}.")
    table = sub.pivot_table(index="year", columns="month", values="rainfall_mm", aggfunc="sum")
    if table.shape[1] != 12 or table.isna().any().any():
        raise ValueError(f"Region {region!r} has missing months.")
    return table.to_numpy(dtype=float)
