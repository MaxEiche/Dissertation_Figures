#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Figure 3.13 — Front velocity vs total density (contour plots).

- Loads combined CSV from Front_Velocity_Combine.py
- Filters:
    * clones with lifetime > MIN_CLONE_LIFETIME_DAYS
    * |velocity| <= MAX_ABS_VELOCITY_UM_PER_DAY
    * per-sample EXCLUDE_SAMPLES from experiment_filters.py
- Outputs:
    (1) Main figure: all strategies pooled in a single contour plot
    (2) Supplementary figure: one contour panel per strategy in a 2x4 grid
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from scipy.stats import gaussian_kde
from numpy.linalg import LinAlgError

# --- thesis-wide helpers & filters ---
from figtools import use_style, set_size, savefig, therapy_color
from experiment_filters import EXCLUDE_SAMPLES, apply_sample_exclusions as apply_sample_exclusions_global

# =========================
# User configuration / paths
# =========================

FRONT_VEL_DIR = Path(
    r"Y:\Members\MaxE\Thesis\Organized Figures\Figure 3.13 Front Velocity"
)
INPUT_CSV = FRONT_VEL_DIR / "front_velocity_with_density_and_treatment.csv"

OUT_PDF_MAIN = FRONT_VEL_DIR / "Figure 3.13 FrontVelocity_vs_TotalDensity_all_strategies.pdf"
OUT_PDF_SUPP = FRONT_VEL_DIR / "Supplementary_FrontVelocity_vs_TotalDensity_per_strategy.pdf"

# =========================
# Filtering parameters
# =========================

MIN_CLONE_LIFETIME_DAYS = 20.0
MAX_ABS_VELOCITY_UM_PER_DAY = 200.0  # set to None to disable

GRID_N = 120
CONTOUR_QUANTILES = (0.65, 0.80, 0.90, 0.95)

SHOW_FIGURES = True  # set True for debugging if you like

# Strategy ordering / display names for supplementary grid
STRATEGY_DISPLAY_NAMES = {
    "CT": "CT",
    "AT50": "AT50",
    "AT100": "AT100",
    "Pulse": "Pulse",
    "Dens": "AT100+",
    "ATAgent1": "KPP-Agent",
    "ATAgent2": "LV-Agent",
}

STRATEGY_ORDER = [
    "CT",
    "AT50",
    "AT100",
    "Pulse",
    "Dens",
    "ATAgent1",
    "ATAgent2",
]

# =========================
# Helpers
# =========================

def infer_strategy_from_sample(sample: str) -> str:
    """
    Infer strategy from sample name, e.g. 'B3_AT100-B3' -> 'AT100'.
    If parsing fails, returns 'NA'.
    """
    try:
        after_underscore = sample.split("_", 1)[1]
        strat = after_underscore.split("-", 1)[0]
        return strat
    except Exception:
        return "NA"


def load_and_filter_data(path: Path) -> pd.DataFrame:
    """
    Load the combined CSV and apply filters:
    - filter extreme velocities
    - filter clones by lifetime
    - add 'strategy' column
    - apply manual sample exclusions/truncations via experiment_filters.py
    - drop rows without density/velocity
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")

    df = pd.read_csv(path)
    print(f"[load] {path.name}: {len(df)} rows")

    # Filter extreme velocities
    if MAX_ABS_VELOCITY_UM_PER_DAY is not None:
        before = len(df)
        df = df[np.abs(df["velocity_um_per_day"]) <= MAX_ABS_VELOCITY_UM_PER_DAY].copy()
        print(
            f"[filter] |velocity| <= {MAX_ABS_VELOCITY_UM_PER_DAY} µm/day: "
            f"{len(df)} / {before} rows"
        )

    # Clone lifetime filter (> MIN_CLONE_LIFETIME_DAYS)
    grp_life = (
        df.groupby(["plate", "sample", "clone_label"], as_index=False)
          .agg(
              first_day=("day_t", "min"),
              last_day=("day_tp1", "max"),
          )
    )
    grp_life["lifetime_days"] = grp_life["last_day"] - grp_life["first_day"]
    long_clones = grp_life[grp_life["lifetime_days"] > MIN_CLONE_LIFETIME_DAYS]
    print(
        f"[filter] clones with lifetime > {MIN_CLONE_LIFETIME_DAYS} days: "
        f"{len(long_clones)} unique clones"
    )

    df = df.merge(
        long_clones[["plate", "sample", "clone_label"]],
        on=["plate", "sample", "clone_label"],
        how="inner",
        validate="m:m",
    )
    print(f"[filter] after lifetime filter: {len(df)} rows")

    # Strategy label from sample
    df["strategy"] = df["sample"].astype(str).apply(infer_strategy_from_sample)

    # Apply experiment_filters truncations/exclusions
    df = apply_sample_exclusions_global(
        df,
        exclude_map=EXCLUDE_SAMPLES,
        column="sample",
        cutoff_column="day_tp1",
        strict=True,
    )

    # Keep only rows with both density and velocity
    before = len(df)
    df = df[df["mean_total_density"].notna() & df["velocity_um_per_day"].notna()].copy()
    print(f"[filter] rows with valid density and velocity: {len(df)} / {before}")

    return df


def compute_density_grid(
    x: np.ndarray,
    y: np.ndarray,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    grid_n: int = GRID_N,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Try to compute a smooth KDE density grid. If KDE fails,
    fall back to a simple 2D histogram-based grid.
    Returns (Xg, Yg, Z).
    """
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 5:
        raise ValueError("Not enough points for density estimation.")

    xx = np.linspace(xlim[0], xlim[1], grid_n)
    yy = np.linspace(ylim[0], ylim[1], grid_n)
    Xg, Yg = np.meshgrid(xx, yy)
    positions = np.vstack([Xg.ravel(), Yg.ravel()])
    pts = np.vstack([x, y])

    try:
        kde = gaussian_kde(pts)
        Z = kde(positions).reshape(Xg.shape)
        return Xg, Yg, Z
    except LinAlgError:
        # jitter and retry
        jitter = 1e-6
        pts_j = np.vstack([
            x + np.random.normal(0, jitter, size=x.size),
            y + np.random.normal(0, jitter, size=y.size),
        ])
        try:
            kde = gaussian_kde(pts_j)
            Z = kde(positions).reshape(Xg.shape)
            return Xg, Yg, Z
        except Exception:
            pass
    except Exception:
        pass

    # Fallback: 2D histogram
    H, xedges, yedges = np.histogram2d(x, y, bins=50, range=[xlim, ylim])
    Xc = 0.5 * (xedges[:-1] + xedges[1:])
    Yc = 0.5 * (yedges[:-1] + yedges[1:])
    Xg, Yg = np.meshgrid(Xc, Yc)
    return Xg, Yg, H.T


def filled_contours_on_ax(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    base_color: str = "#000000",
) -> None:
    """
    Draw filled contour density (white low-density region, colored bands) on
    an existing Axes, matching the style used in Figure 3.12 supplementary.
    """
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 10:
        print(f"[contour] Not enough points for filled contours (n={x.size}), skipping.")
        return

    try:
        Xg, Yg, Z = compute_density_grid(x, y, xlim, ylim)
    except ValueError:
        print("[contour] Not enough points after filter, skipping.")
        return

    vals = Z[np.isfinite(Z) & (Z > 0)]
    if vals.size == 0:
        print("[contour] All densities zero or NaN, skipping.")
        return

    core_levels = np.quantile(vals, CONTOUR_QUANTILES)
    levels = np.concatenate(
        ([vals.min() * 0.999], core_levels, [vals.max() * 1.001])
    )

    base_rgba = np.array(mcolors.to_rgba(base_color))
    n_regions = len(levels) - 1

    colors = []
    # lowest-density region: white
    colors.append((1.0, 1.0, 1.0, 1.0))
    for i in range(1, n_regions):
        alpha = 0.20 + 0.75 * (i / (n_regions - 1))
        rgba = base_rgba.copy()
        rgba[-1] = alpha
        colors.append(tuple(rgba))

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    ax.contourf(Xg, Yg, Z, levels=levels, colors=colors)
    ax.contour(Xg, Yg, Z, levels=core_levels, colors=[base_color], linewidths=0.8)


# =========================
# Plotting: main figure
# =========================

def make_main_figure(df: pd.DataFrame) -> None:
    """
    Single-panel main figure: all strategies pooled.
    """
    x_all = df["mean_total_density"].to_numpy(dtype=float)
    y_all = df["velocity_um_per_day"].to_numpy(dtype=float)

    xlim = (
        float(np.nanpercentile(x_all, 1)),
        float(np.nanpercentile(x_all, 99)),
    )
    ylim = (
        float(np.nanpercentile(y_all, 1)),
        float(np.nanpercentile(y_all, 99)),
    )
    print(f"[axes] main xlim={xlim}, ylim={ylim}")

    fig, ax = plt.subplots(1, 1)
    set_size(fig, width="thesis", ratio=0.7)
    fig.subplots_adjust(
        left=0.12,
        right=0.98,
        bottom=0.16,
        top=0.94,
    )

    base_color_all = therapy_color("AT50")
    filled_contours_on_ax(ax, x_all, y_all, xlim, ylim, base_color=base_color_all)

    # --- axis labels / title: adjust text as you like ---
    ax.set_xlabel("Mean Total Density")
    ax.set_ylabel("Front Velocity [µm/day]")
    ax.set_title("Front Velocity Decreases with Increasing Local Cell Density")  # <- change text if needed
    ax.grid(False)

    savefig(fig, OUT_PDF_MAIN, dpi=300, transparent=True)
    print(f"[main] Saved main figure to: {OUT_PDF_MAIN}")

    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)


# =========================
# Plotting: supplementary grid
# =========================

def make_supplementary_figure(df: pd.DataFrame) -> None:
    """
    Supplementary figure: one contour panel per strategy in a 2x4 grid.
    """
    strategies_present = sorted(df["strategy"].dropna().unique())
    print("[supp] strategies present in data:", strategies_present)

    xs_all = df["mean_total_density"].to_numpy(dtype=float)
    ys_all = df["velocity_um_per_day"].to_numpy(dtype=float)
    xlim = (
        float(np.nanpercentile(xs_all, 1)),
        float(np.nanpercentile(xs_all, 99)),
    )
    ylim = (
        float(np.nanpercentile(ys_all, 1)),
        float(np.nanpercentile(ys_all, 99)),
    )
    print(f"[supp] global xlim={xlim}, ylim={ylim}")

    fig, axes = plt.subplots(2, 4)
    set_size(fig, width="thesis", ratio=0.7)
    fig.subplots_adjust(
        left=0.07,
        right=0.98,
        bottom=0.10,
        top=0.94,
        wspace=0.35,
        hspace=0.40,
    )

    for idx, strat in enumerate(STRATEGY_ORDER):
        row, col = divmod(idx, 4)
        ax = axes[row, col]

        sub = df[df["strategy"] == strat]
        display_name = STRATEGY_DISPLAY_NAMES.get(strat, strat)

        if sub.empty:
            ax.text(
                0.5,
                0.5,
                "no data",
                ha="center",
                va="center",
                fontsize=8,
                color="gray",
                transform=ax.transAxes,
            )
            ax.set_title(display_name, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            continue

        x_s = sub["mean_total_density"].to_numpy(dtype=float)
        y_s = sub["velocity_um_per_day"].to_numpy(dtype=float)

        try:
            base_col = therapy_color(strat)
        except KeyError:
            base_col = "#000000"

        filled_contours_on_ax(ax, x_s, y_s, xlim, ylim, base_color=base_col)

        ax.set_title(display_name, fontsize=9)
        ax.set_xlabel("Mean Total Density", fontsize=8)
        ax.set_ylabel("Front Velocity [µm/day]", fontsize=8)
        ax.grid(False)

    # If you want a global title for the supplementary figure, uncomment:
    # fig.suptitle("Front velocity vs total density per strategy", fontsize=11)

    savefig(fig, OUT_PDF_SUPP, dpi=300, transparent=True)
    print(f"[supp] Saved supplementary figure to: {OUT_PDF_SUPP}")

    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)


# =========================
# Main
# =========================

def main():
    df = load_and_filter_data(INPUT_CSV)

    with use_style("figstyle.mplstyle"):
        make_main_figure(df)
        make_supplementary_figure(df)

    print("[done] Figure 3.13 (main + supplementary) generated.")


if __name__ == "__main__":
    main()