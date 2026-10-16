"""Figures for Mini-Project 2."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import ArrayLike


def plot_dispatch(
    solar: ArrayLike,
    battery: ArrayLike,
    daily_cost: ArrayLike,
    infeasible: ArrayLike | None = None,
) -> Figure:
    """Stacked bars of solar + battery (kWh) with daily cost (UGX) on a second axis.

    Days where the exact solution was infeasible and repaired are hatched.
    """
    solar, battery, cost = (np.asarray(a, dtype=float) for a in (solar, battery, daily_cost))
    days = np.arange(1, len(solar) + 1)
    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax1.bar(days, solar, color="#f4b400", label="Solar (kWh)")
    ax1.bar(days, battery, bottom=solar, color="#4285f4", label="Battery (kWh)")
    if infeasible is not None and np.any(infeasible):
        mask = np.asarray(infeasible, dtype=bool)
        ax1.bar(days[mask], (solar + battery)[mask], fill=False, hatch="//",
                edgecolor="black", linewidth=0, label="Infeasible day (repaired by NNLS)")
    ax1.set_xlabel("Day of month")
    ax1.set_ylabel("Energy supplied (kWh)")
    ax1.set_xticks(days)
    ax1.grid(axis="y", alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(days, cost, color="#d93025", marker="o", lw=1.8, label="Daily cost (UGX)")
    ax2.set_ylabel("Daily cost (UGX)")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper center", ncol=4, bbox_to_anchor=(0.5, -0.1), frameon=False)
    ax1.set_title("Kasese health-centre micro-grid: 30-day dispatch and daily energy cost")
    fig.tight_layout()
    return fig


def plot_sensitivity_cloud(draws: np.ndarray, baseline: np.ndarray, day: int, names=("Solar x", "Battery y")) -> Figure:
    """Scatter of 1,000 perturbed (x, y) solutions for one day around its exact solution."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(draws[:, 0, day], draws[:, 1, day], s=6, alpha=0.35, color="#4285f4", label="Perturbed solutions (+/-5% demand)")
    ax.scatter(*baseline[:2, day], color="#d93025", s=60, zorder=3, label="Unperturbed solution")
    ax.set_xlabel(f"{names[0]} (kWh)")
    ax.set_ylabel(f"{names[1]} (kWh)")
    ax.set_title(f"Day {day + 1}: how far x and y move under +/-5% demand error")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig
