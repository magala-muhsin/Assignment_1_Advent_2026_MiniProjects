"""Figures for Mini-Project 4."""
from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from src.rainfall import DROUGHT, MONTHS, WATERLOGGING, CropRule, Region
from src.nasa_power import monthly_matrix


def plot_rainfall_lines(regions: Sequence[Region]) -> Figure:
    """Monthly rainfall (mm) of all regions on one line chart."""
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for r in regions:
        ax.plot(MONTHS, r.rainfall, marker="o", lw=2, label=f"{r.name} ({r.annual_total():,.0f} mm/yr)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Monthly rainfall (mm)")
    ax.set_title("Monthly rainfall regimes (illustrative data)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_suitability_heatmap(regions: Sequence[Region], crops: Sequence[CropRule]) -> Figure:
    """Crop suitability by month and region: one panel per crop."""
    colours = ["#d95f59", "#56a36c", "#4a90c2"]   # drought, good, waterlogging
    fig, axes = plt.subplots(1, len(crops), figsize=(6 * len(crops), 3.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, crop in zip(axes, crops):
        grid = np.array([[0 if c == DROUGHT else 2 if c == WATERLOGGING else 1
                          for c in crop.classify_region(r)] for r in regions])
        ax.imshow(grid, aspect="auto", cmap=ListedColormap(colours), vmin=0, vmax=2)
        for i, r in enumerate(regions):
            for j, mm in enumerate(r.rainfall):
                ax.text(j, i, f"{mm:.0f}", ha="center", va="center", fontsize=7, color="white")
        ax.set_xticks(range(12), MONTHS, rotation=45)
        ax.set_yticks(range(len(regions)), [r.name for r in regions])
        ax.set_xlabel("Month")
        ax.set_title(f"{crop.name} (good: {crop.monthly_min_mm:.0f}-{crop.monthly_max_mm:.0f} mm/month)", fontsize=10)
    fig.legend(handles=[Patch(color=colours[0], label="Drought risk"),
                        Patch(color=colours[1], label="Good"),
                        Patch(color=colours[2], label="Waterlogging risk")],
               loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Crop suitability by month (numbers = monthly rainfall in mm)", fontweight="bold")
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    return fig


def plot_similarity_matrices(names: Sequence[str], cosine: np.ndarray, pearson: np.ndarray,
                             euclid: np.ndarray) -> Figure:
    """Three annotated heatmaps: cosine similarity, Pearson correlation, Euclidean distance."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    specs = [("Cosine similarity", cosine, "Blues", 0, 1), ("Pearson correlation", pearson, "RdBu_r", -1, 1),
             ("Euclidean distance (mm)", euclid, "Oranges", None, None)]
    for ax, (title, mat, cmap, lo, hi) in zip(axes, specs):
        im = ax.imshow(mat, cmap=cmap, vmin=lo, vmax=hi)
        ax.set_xticks(range(len(names)), names)
        ax.set_yticks(range(len(names)), names)
        ax.set_title(title)
        for i in range(len(names)):
            for j in range(len(names)):
                dark = im.norm(mat[i, j]) > 0.6
                ax.text(j, i, f"{mat[i, j]:.2f}" if title != "Euclidean distance (mm)" else f"{mat[i, j]:.0f}",
                        ha="center", va="center", fontsize=10, color="white" if dark else "black")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    return fig


def plot_monthly_boxplots(df: pd.DataFrame, regions: Sequence[str]) -> Figure:
    """Year-to-year spread of monthly rainfall (box plots per month, one panel per region)."""
    fig, axes = plt.subplots(1, len(regions), figsize=(5.4 * len(regions), 4.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, regions):
        ax.boxplot(monthly_matrix(df, name), tick_labels=MONTHS, showfliers=False)
        ax.set_title(name)
        ax.set_xlabel("Month")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Monthly rainfall total (mm)")
    fig.suptitle("Year-to-year variability of monthly rainfall (NASA POWER)", fontweight="bold")
    fig.tight_layout()
    return fig
