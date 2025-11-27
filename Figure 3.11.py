#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Figure 2c / 3.11 — Plot all clone trajectories (AT50 + AT100) with clean time handling.

Input CSV (from Figure_2c_analysis.py):
    plate, replicate, clone_id, t_idx, day, dist_px, dist_um

Key fixes:
- By default, we DO NOT remap 'day' from external files (uses Mon/Wed/Fri days from analysis).
- If APPLY_DAY_REMAP=True, we remap and then re-sort + dedupe by 'day' to avoid zig-zags.
- For each trajectory (plate, replicate, clone_id), we sort by day and drop duplicate days.

Visuals:
- One line per clone trajectory.
- Color encodes strategy (AT50 vs AT100) using central thesis color mapping.
- Legend shows which strategies are present.
- Uses thesis-wide figstyle (Arial, standard sizes, constrained layout).
"""

from __future__ import annotations
import os
import re
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from experiment_filters import apply_sample_exclusions

from figtools import use_style, new_figure, savefig, therapy_color

# --------------------------
# User parameters
# --------------------------
COMBINED_CSV = r"Y:\Members\MaxE\Thesis\Organized Figures\Figure 3.11\Figure2c_distances_all_AT50_AT100.csv"
OUT_DIR      = r"Y:\Members\MaxE\Thesis\Organized Figures\Figure 3.11"

# Day remap toggle (leave False to use analysis days)
APPLY_DAY_REMAP = False
RESULTS_DIR     = r"D:\2025_08_CoreSeed_Large_Testdataset\Results_today"  # only used if remap is enabled

# Filters (None = include all)
FILTER_PLATES: List[str] | None = None
FILTER_REPLICATES: List[str] | None = None
FILTER_CLONE_IDS: List[int] | None = None

# Visual prefs
LINEWIDTH       = 2.6
ALPHA_LINE      = 0.6
GRID_ALPHA      = 0.25

STYLE_PATH = Path("figstyle.mplstyle")

# --------------------------
# Helpers
# --------------------------
def _ensure_outdir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)

def _find_csv(path: str) -> Path:
    p = Path(path)
    if p.exists():
        return p
    alt = Path.cwd() / p.name
    if alt.exists():
        return alt
    raise FileNotFoundError(f"Combined CSV not found:\n  {path}\nTried:\n  {alt}")

def _load_df() -> pd.DataFrame:
    csv_path = _find_csv(COMBINED_CSV)
    print(f"[info] Loading: {csv_path}")
    df = pd.read_csv(csv_path)

    required = {"plate", "replicate", "clone_id", "t_idx", "day", "dist_um"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {sorted(missing)}")

    # Optional filters
    if FILTER_PLATES is not None:
        df = df[df["plate"].isin(FILTER_PLATES)]
    if FILTER_REPLICATES is not None:
        df = df[df["replicate"].isin(FILTER_REPLICATES)]
    if FILTER_CLONE_IDS is not None:
        df = df[df["clone_id"].isin(FILTER_CLONE_IDS)]

    # Extract strategy (AT50/AT100) from replicate name "<WELL>_<STRATEGY>-<WELL>"
    strat = df["replicate"].str.extract(
        r"^[A-H]\d{1,2}_(?P<strategy>[^-_]+)-[A-H]\d{1,2}$"
    )
    df["strategy"] = strat["strategy"].fillna("UNKNOWN")
    return df

def _remap_timepoints_to_real_days(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optional remap to per-plate 'real days' if you keep results_dXX.txt lists.
    This function is intentionally simple and defensive.
    """
    if not os.path.isdir(RESULTS_DIR):
        print(f"[warn] RESULTS_DIR not found: {RESULTS_DIR} → skipping remap.")
        return df

    # Build list of available days per plate by parsing filenames "results_dNN.txt"
    day_map = {}
    for plate in sorted(df["plate"].unique()):
        plate_dir = Path(RESULTS_DIR) / plate
        if not plate_dir.is_dir():
            continue
        days = []
        for f in plate_dir.glob("results_d*.txt"):
            m = re.search(r"results_d(\d+)\.txt$", f.name)
            if m:
                days.append(int(m.group(1)))
        if days:
            day_map[plate] = sorted(set(days))

    # Map within each (plate, replicate) by rank order of t_idx
    new_day = np.empty(len(df), dtype=float)
    new_day[:] = np.nan
    for (plate, rep), sub_idx in df.groupby(["plate", "replicate"]).groups.items():
        idx = np.array(list(sub_idx))
        sub = df.loc[idx].sort_values("t_idx")
        real_days = day_map.get(plate, [])
        if not real_days:
            new_day[idx] = df.loc[idx, "day"].to_numpy()  # keep as-is
            continue
        mapped = []
        for k, _ in enumerate(sub["t_idx"]):
            if k < len(real_days):
                mapped.append(real_days[k])
            else:
                mapped.append(real_days[-1] + (k - len(real_days) + 1))  # simple tail
        # write back in the same order as 'sub'
        new_day[sub.index.to_numpy()] = np.array(mapped, dtype=float)

    out = df.copy()
    out["day"] = new_day
    return out

# --------------------------
# Main
# --------------------------
def main():
    _ensure_outdir(OUT_DIR)
    df = _load_df()

    # Optional remap
    if APPLY_DAY_REMAP:
        df = _remap_timepoints_to_real_days(df)

    # Apply manual replicate exclusions / truncations
    df = apply_sample_exclusions(
        df,
        column="replicate",
        cutoff_column="day",
        strict=False,  # <- allow missing EXCLUDE_SAMPLES keys, just warn
    )

    # FINAL canonical order before plotting
    df = df.sort_values(["plate", "replicate", "clone_id", "day"]).reset_index(drop=True)


    with use_style(style_mpl=STYLE_PATH):
        # Full-width thesis figure
        fig, ax = new_figure(width="thesis", ratio=0.62)

        # One line per (plate, replicate, clone_id)
        group_keys = ["plate", "replicate", "clone_id"]
        for key, sub in df.groupby(group_keys, sort=False):
            # Clean per-trajectory: sort by day and drop duplicate days
            sub = sub.sort_values("day").drop_duplicates(subset=["day"], keep="first")

            x = sub["day"].to_numpy()
            y = sub["dist_um"].to_numpy(dtype=float)
            valid = ~np.isnan(y)
            x, y = x[valid], y[valid]
            if x.size < 2:
                continue

            # Strategy color via central mapping (AT50/AT100)
            strat = str(sub["strategy"].iloc[0])
            try:
                color = therapy_color(strat)
            except KeyError:
                color = "#666666"

            ax.plot(
                x,
                y,
                color=color,
                lw=LINEWIDTH,
                alpha=ALPHA_LINE,
                solid_capstyle="round",
                solid_joinstyle="round",
            )

        # Labels, grid, limits
        ax.set_xlabel("Day")
        ax.set_ylabel("Distance to front (µm)")
        ax.tick_params(axis="both")
        ax.grid(True, which="major", alpha=GRID_ALPHA)

        if not df.empty:
            xmin = float(np.nanmin(df["day"].to_numpy()))
            xmax = float(np.nanmax(df["day"].to_numpy()))
            ymin = float(np.nanmin(df["dist_um"].to_numpy()))
            ymax = float(np.nanmax(df["dist_um"].to_numpy()))
            xr = xmax - xmin if xmax > xmin else 1.0
            yr = ymax - ymin if ymax > ymin else 1.0
            ax.set_xlim(xmin - 0.02 * xr, xmax + 0.02 * xr)
            ax.set_ylim(max(0.0, ymin - 0.05 * yr), ymax + 0.05 * yr)

        # Simple legend for strategies present
        present = sorted(df["strategy"].dropna().unique().tolist())
        handles = []
        for s in present:
            if s == "UNKNOWN":
                continue
            try:
                col = therapy_color(s)
            except KeyError:
                col = "#666666"
            h = plt.Line2D(
                [0],
                [0],
                color=col,
                lw=LINEWIDTH,
                alpha=ALPHA_LINE,
                label=s,
            )
            handles.append(h)

        if handles:
            ax.legend(handles=handles, frameon=False, loc="best")

        # Save PDF only
        out_pdf = Path(OUT_DIR) / "Figure2c_AllClones_Trajectories_AT50_AT100.pdf"
        savefig(fig, out_pdf, dpi=300, transparent=True)
        print(f"[done] Saved: {out_pdf}")

        plt.show()
        plt.close(fig)

if __name__ == "__main__":
    main()
