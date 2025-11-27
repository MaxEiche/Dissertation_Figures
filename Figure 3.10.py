#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Figure 3.10 — Plot all clone trajectories at once with slope-colored segments.

- Input CSV (from Figure_1d_analysis.py) columns:
    plate, replicate, clone_id, t_idx, day, dist_px, dist_um
- Uses 'day' as-is (no external remap by default).
- Shades day 0–2.
- Colors each segment by slope dy/dx:
    negative → blue, zero → black, positive → red
  Robust normalization (95th pct of |slope|) so extremes don’t wash colors.
  Exactly zero → pure black; near-zero → dark gray.

Prefs: thesis figure style, half-width (single column).
"""

from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, LinearSegmentedColormap

from figtools import use_style, new_figure, savefig

# --------------------------
# User parameters
# --------------------------
COMBINED_CSV = r"Y:\Members\MaxE\Thesis\Organized Figures\Figure 3.10\Data\Figure1d_distances_all_pulse.csv"
OUT_DIR      = r"Y:\Members\MaxE\Thesis\Organized Figures\Figure 3.10"

# Do NOT remap 'day' from external files (leave False)
APPLY_DAY_REMAP = False
RESULTS_DIR     = r"D:\2025_08_CoreSeed_Large_Testdataset\Results_today"  # only used if APPLY_DAY_REMAP=True

# Background shaded windows on x-axis (in 'day' units)
SHADED_WINDOWS: List[Tuple[float, float]] = [(0.0, 2.0)]

# Filters (None = include all)
FILTER_PLATES: List[str] | None = None
FILTER_REPLICATES: List[str] | None = None
FILTER_CLONE_IDS: List[int] | None = None

GRID_ALPHA      = 0.25
LINEWIDTH       = 2.6

STYLE_PATH = Path("figstyle.mplstyle")

# --------------------------
# Helpers
# --------------------------
def _find_combined_csv(path: str) -> Path:
    p = Path(path)
    if p.exists():
        return p
    alt = Path.cwd() / Path(path).name
    if alt.exists():
        return alt
    raise FileNotFoundError(f"Combined CSV not found:\n  {path}\nTried also:\n  {alt}")

def _load_df() -> pd.DataFrame:
    csv_path = _find_combined_csv(COMBINED_CSV)
    print(f"[info] Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)

    required = {"plate", "replicate", "clone_id", "t_idx", "day", "dist_um"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {sorted(missing)}")

    if FILTER_PLATES is not None:
        df = df[df["plate"].isin(FILTER_PLATES)]
    if FILTER_REPLICATES is not None:
        df = df[df["replicate"].isin(FILTER_REPLICATES)]
    if FILTER_CLONE_IDS is not None:
        df = df[df["clone_id"].isin(FILTER_CLONE_IDS)]

    df = df.sort_values(["plate", "replicate", "clone_id", "t_idx", "day"]).reset_index(drop=True)
    return df

def _remap_days_using_results(df: pd.DataFrame) -> pd.DataFrame:
    """Template stub — only if you later re-enable remap."""
    if not APPLY_DAY_REMAP:
        return df
    if not os.path.isdir(RESULTS_DIR):
        print(f"[warn] RESULTS_DIR not found: {RESULTS_DIR} → skipping remap.")
        return df
    # Implement your specific parsing here if ever needed.
    return df

def _ensure_outdir():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# --------------------------
# Main plotting
# --------------------------
def main():
    _ensure_outdir()
    df = _load_df()
    if APPLY_DAY_REMAP:
        df = _remap_days_using_results(df)

    with use_style(style_mpl=STYLE_PATH):
        # Half-width (single column), ratio ~ original 10x6 → 0.6
        fig, ax = new_figure(width="thesis", ratio=0.62)

        # Shade requested day windows
        for (x0, x1) in SHADED_WINDOWS:
            ax.axvspan(float(x0), float(x1), color="gray", alpha=0.15, zorder=0)

        # --- First pass: build segments & collect slopes for robust scaling ---
        group_key = ["plate", "replicate", "clone_id"]
        group_segments: Dict[tuple, np.ndarray] = {}
        group_slopes: Dict[tuple, np.ndarray] = {}
        all_slopes: List[float] = []

        for key, sub in df.groupby(group_key, sort=False):
            sub = sub.sort_values(["t_idx", "day"]).copy()
            sub = sub.drop_duplicates(subset=["day"], keep="first")

            x = sub["day"].to_numpy()
            y = sub["dist_um"].to_numpy(dtype=float)

            valid = ~np.isnan(y)
            x, y = x[valid], y[valid]
            if x.size < 2:
                continue

            dx = x[1:] - x[:-1]
            dy = y[1:] - y[:-1]

            seg_mask = dx > 0
            if not np.any(seg_mask):
                continue

            x0 = x[:-1][seg_mask]
            y0 = y[:-1][seg_mask]
            x1 = x[1:][seg_mask]
            y1 = y[1:][seg_mask]

            slopes = (y1 - y0) / (x1 - x0)  # dy/dx (µm per day)

            segments = np.stack(
                [np.stack([x0, y0], axis=1),
                 np.stack([x1, y1], axis=1)],
                axis=1
            )  # (N, 2, 2)

            group_segments[key] = segments
            group_slopes[key] = slopes
            if slopes.size:
                all_slopes.extend(slopes.tolist())

        # Robust global normalization (95th percentile of |slope|)
        if len(all_slopes) == 0:
            all_slopes = [0.0]
        global_max_abs_slope = float(np.percentile(np.abs(all_slopes), 95))
        if global_max_abs_slope <= 0:
            global_max_abs_slope = 1.0
        norm = Normalize(vmin=-global_max_abs_slope, vmax=global_max_abs_slope)

        # Colormap with *very narrow* pure-black center; near-zero = dark gray
        cmap = LinearSegmentedColormap.from_list(
            "blue_black_red_narrow",
            [(0.00, "blue"),
             (0.49, "#444444"),
             (0.50, "#000000"),  # exactly zero → black
             (0.51, "#444444"),
             (1.00, "red")]
        )

        # Define a small epsilon so near-zero slopes don’t collapse to black
        eps = 0.02 * global_max_abs_slope  # 2% of scale

        # --- Second pass: draw collections with adjusted near-zero slopes ---
        for key in group_segments.keys():
            segments = group_segments[key]
            slopes   = group_slopes[key]

            # Keep exact zeros black; push tiny non-zeros away from 0 to dark gray
            adj = slopes.copy()
            is_zero = adj == 0
            near = (~is_zero) & (np.abs(adj) < eps)
            adj[near] = np.sign(adj[near]) * eps

            lc = LineCollection(
                segments,
                cmap=cmap,
                norm=norm,
                linewidths=LINEWIDTH,
                capstyle="round",
                joinstyle="round",
                alpha=1.0
            )
            lc.set_array(adj)
            ax.add_collection(lc)

        # Axes, labels, grid
        ax.set_xlabel("Day")
        ax.set_ylabel("Distance to front [µm]")
        ax.tick_params(axis="both")  # label sizes etc. from figstyle
        ax.grid(True, which="major", alpha=GRID_ALPHA)

        # Limits
        if not df.empty:
            xmin = float(np.nanmin(df["day"].to_numpy()))
            xmax = float(np.nanmax(df["day"].to_numpy()))
            ymin = float(np.nanmin(df["dist_um"].to_numpy()))
            ymax = float(np.nanmax(df["dist_um"].to_numpy()))
            xr = xmax - xmin if xmax > xmin else 1.0
            yr = ymax - ymin if ymax > ymin else 1.0
            ax.set_xlim(xmin - 0.02 * xr, xmax + 0.02 * xr)
            ax.set_ylim(max(0.0, ymin - 0.05 * yr), ymax + 0.05 * yr)

        # Save PDF only
        out_pdf = Path(OUT_DIR) / "Figure1d_all_trajectories.pdf"
        savefig(fig, out_pdf, dpi=300, transparent=True)
        print(f"[done] Saved: {out_pdf}")

        plt.show()
        plt.close(fig)

if __name__ == "__main__":
    main()
