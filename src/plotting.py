"""Plotting helpers for the district forecasts figure."""
from __future__ import annotations

from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from src.district import DistrictPopulation
from src.forecasters import CAGRForecaster
from src.bootstrap import bootstrap_cagr_interval


def plot_district_forecasts(
    districts: list[DistrictPopulation],
    horizon: int = 5,
    last_train_year: int = 2021,
    n_resamples: int = 2000,
    seed: int = 42,
    models: Mapping[str, type] | None = None,
) -> Figure:
    """One subplot per district: actual, train fit, test prediction, forecast + 95% PI.

    ``models`` maps district name -> forecaster class (default: CAGR for all).
    """
    models = models or {}
    n = len(districts)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4.6 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for k, (ax, d) in enumerate(zip(axes, districts)):
        cls = models.get(d.name, CAGRForecaster)
        tr_y, tr_v, te_y, _ = d.split(last_train_year)
        val_model = cls().fit(tr_y, tr_v)
        full_model = cls().fit(d.years, d.populations)
        fc_years = full_model.future_years(horizon)
        forecast = full_model.predict(horizon)

        ax.axvspan(d.years[0] - 0.5, last_train_year + 0.5, color="#eaf2f8", alpha=0.8)
        ax.axvspan(last_train_year + 0.5, d.years[-1] + 0.5, color="#fff2df", alpha=0.8)
        ax.axvline(last_train_year + 0.5, color="#555555", ls=":", lw=1.5)

        ax.plot(d.years, d.populations, "o-", color="#222222", lw=2, ms=4.5, label="Actual")
        fitted = val_model.fitted_values()
        if not np.all(np.isnan(fitted)):
            ax.plot(tr_y, fitted, "--", color="#2673b8", lw=2, label="Fitted (train)")
        ax.plot(
            np.r_[tr_y[-1], te_y],
            np.r_[tr_v[-1], val_model.predict(len(te_y))],
            "--", color="#7b3fa0", lw=2, label="Test prediction (2022-24)",
        )
        if isinstance(full_model, CAGRForecaster):
            lo, hi = bootstrap_cagr_interval(
                d.years, d.populations, horizon, n_resamples=n_resamples, seed=seed + k
            )
            ax.fill_between(fc_years, lo, hi, color="#d97706", alpha=0.22, label="95% bootstrap PI")
        ax.plot(
            np.r_[d.years[-1], fc_years], np.r_[d.populations[-1], forecast],
            "s-", color="#d97706", ms=4, lw=2, label="Forecast (refit on all data)",
        )
        ax.set_title(d.name, fontweight="bold")
        ax.set_xlabel("Year")
        ax.set_ylabel("Population (thousands)")
        ax.set_xlim(d.years[0] - 0.5, fc_years[-1] + 0.5)
        ax.grid(alpha=0.25)

    for ax in axes[n:]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=9)
    fig.suptitle(
        "District population: actual, train/test backtest and 2025-2029 forecast\n"
        "Shaded background: blue = training (2015-2021), tan = test (2022-2024); dotted line = split",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    return fig
