"""Figures for Mini-Project 3."""
from __future__ import annotations

from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import ArrayLike


def plot_stock_trajectories(stocks: Mapping[float, ArrayLike], K: float) -> Figure:
    """Stock (tonnes) over time for each weekly harvest rate, with K and K/2 marked."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for h, s in stocks.items():
        ax.plot(np.arange(len(s)), s, lw=2, label=f"h = {h:.2f} per week")
    ax.axhline(K, color="gray", ls=":", label=f"Carrying capacity K = {K:,.0f} t")
    ax.axhline(K / 2, color="black", ls="--", lw=1, label="K/2 (stock at MSY)")
    ax.set_xlabel("Week")
    ax.set_ylabel("Fish stock (tonnes)")
    ax.set_title("Fish stock under alternative weekly harvest rates")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_revenue_histogram(revenues_ugx: ArrayLike, mean: float, p5: float, title: str) -> Figure:
    """Histogram of simulated annual revenue (UGX billions) with mean and 5th percentile."""
    rev = np.asarray(revenues_ugx, dtype=float) / 1e9
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(rev, bins=45, color="#2a9d8f", alpha=0.85, edgecolor="white")
    ax.axvline(mean / 1e9, color="black", ls="-", lw=1.5, label=f"Mean = UGX {mean / 1e9:,.1f} bn")
    ax.axvline(p5 / 1e9, color="#c1121f", ls="--", lw=2,
               label=f"5th percentile = UGX {p5 / 1e9:,.1f} bn")
    ax.set_xlabel("Annual revenue (UGX billions)")
    ax.set_ylabel("Number of simulated price paths")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_price_paths(paths: np.ndarray, lower: float, upper: float, n_show: int = 8) -> Figure:
    """A few simulated price paths with the bounds drawn."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for p in paths[:n_show]:
        ax.plot(np.arange(len(p)), p, lw=1.2, alpha=0.8)
    ax.axhline(lower, color="#c1121f", ls="--", label=f"Lower bound {lower:,.0f}")
    ax.axhline(upper, color="#c1121f", ls="--", label=f"Upper bound {upper:,.0f}")
    ax.set_xlabel("Week")
    ax.set_ylabel("Price (UGX per kg)")
    ax.set_title(f"{min(n_show, len(paths))} simulated weekly price paths (bounded random walk)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_closed_season(stock_open: ArrayLike, stock_closed: ArrayLike, weeks_closed: int, h: float) -> Figure:
    """Five-year stock with and without a closed season."""
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(stock_open, lw=2, label="No closed season")
    ax.plot(stock_closed, lw=2, label=f"{weeks_closed} weeks closed each year")
    n_years = (len(stock_open) - 1) // 52
    for y in range(n_years):
        ax.axvspan(y * 52, y * 52 + weeks_closed, color="#999999", alpha=0.18,
                   label="Closed weeks" if y == 0 else None)
    ax.set_xlabel("Week")
    ax.set_ylabel("Fish stock (tonnes)")
    ax.set_title(f"Effect of a closed season on the stock (h = {h:.2f} per week)")
    ax.grid(alpha=0.25)
    ax.legend(loc="center right")
    fig.tight_layout()
    return fig
