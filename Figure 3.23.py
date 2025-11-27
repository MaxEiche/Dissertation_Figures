#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Figure 3.23 — Density vs TTP (PLOTTING ONLY)

Scatter: one dot per clone
  x = per-clone median or mean density (from upstream analysis)
  y = TTP day (first day with total_norm > 2.5), colony-level

Input CSV
  Y:\\Members\\MaxE\\Thesis\\Organized Figures\\Figure 3.23\\Figure5c_density_vs_TTP.csv

Output PDFs
  Figure 3.23 Density_vs_TTP_median.pdf
  Figure 3.23 Density_vs_TTP_mean.pdf

Formatting:
- with use_style("figstyle.mplstyle", font_family="Arial")
- new_figure(width="single", ratio=0.8)   # half-page column
- no tight_layout() / constrained_layout()
- ticks outward, no top/right spines (figtools.new_figure + format_axes)
- colours via therapy_color("NT", "CT", "AT50", "AT100", "Pulse", ...)
- apply EXCLUDE_SAMPLES (full exclusions) on 'sample'
- keep only longest-living clone per replicate (plate + sample)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from scipy.stats import spearmanr

from figtools import use_style, new_figure, savefig, therapy_color  # type: ignore
from experiment_filters import EXCLUDE_SAMPLES  # type: ignore


# ---------------------------------------------------------------------
# Paths and config
# ---------------------------------------------------------------------

DATA_DIR = Path(r"Y:\Members\MaxE\Thesis\Organized Figures\Figure 3.23")
IN_CSV = DATA_DIR / "Figure5c_density_vs_TTP.csv"

OUT_MEDIAN = DATA_DIR / "Figure 3.23 Density_vs_TTP_median.pdf"
OUT_MEAN = DATA_DIR / "Figure 3.23 Density_vs_TTP_mean.pdf"

# Plot toggles (stylistic only)
COLOR_BY_STRATEGY: bool = True
JITTER_FRAC_X: float = 0.01   # fraction of x-range
JITTER_FRAC_Y: float = 0.0    # keep 0.0 (days are discrete)
LOGX: bool = False
XLIM: Optional[Tuple[float, float]] = None
YLIM: Optional[Tuple[float, float]] = (0, 80)
POINT_SIZE: float = 10.0
ALPHA: float = 0.7
EDGELESS: bool = True
SEED: int = 7


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def strategy_color_from_palette(strategy: str) -> str:
    """
    Map the 'strategy' string in the CSV to your canonical therapy_color key.
    """
    s = strategy.strip().upper()
    mapping: Dict[str, str] = {
        "NT": "NT",
        "CT": "CT",
        "PULSE": "Pulse",
        "IT": "IT",
        "AT50": "AT50",
        "AT100": "AT100",
        "AT80": "AT80",
    }
    key = mapping.get(s, s)  # fall back to raw name if already matches
    return therapy_color(key)


def load_df() -> pd.DataFrame:
    """
    Load the clone-level density vs TTP CSV and apply basic cleanup.
    """
    if not IN_CSV.is_file():
        raise FileNotFoundError(f"Cannot find analysis CSV: {IN_CSV}")

    df = pd.read_csv(IN_CSV)

    needed = {
        "plate",
        "sample",
        "strategy",
        "clone_label",
        "median_density",
        "mean_density",
        "ttp_day",
    }
    missing = needed.difference(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    df["median_density"] = pd.to_numeric(df["median_density"], errors="coerce")
    df["mean_density"] = pd.to_numeric(df["mean_density"], errors="coerce")
    df["ttp_day"] = pd.to_numeric(df["ttp_day"], errors="coerce")

    df = df.dropna(subset=["median_density", "mean_density", "ttp_day"]).copy()
    return df


def apply_exclude_samples(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply EXCLUDE_SAMPLES to this clone-level dataset.

    For this aggregated CSV we:
      - Drop all samples whose EXCLUDE_SAMPLES entry is None (full exclusion).
      - Ignore truncated cutoffs (those were already handled in the
        original time-series analysis before generating this CSV).
    """
    full_exclude = {s for s, cutoff in EXCLUDE_SAMPLES.items() if cutoff is None}
    before = len(df)
    df2 = df[~df["sample"].isin(full_exclude)].copy()
    print(f"[EXCLUDE] After EXCLUDE_SAMPLES: {len(df2)} / {before} rows kept")
    return df2


def keep_longest_clone_per_replicate(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each replicate (plate + sample), keep only the clone with
    the maximal ttp_day (longest living clone).
    """
    required = {"plate", "sample", "ttp_day"}
    if not required.issubset(df.columns):
        raise ValueError(f"Expected columns {required} in the CSV.")

    idx = df.groupby(["plate", "sample"])["ttp_day"].idxmax()
    df_longest = df.loc[idx].copy()
    print(f"[FILTER] Longest clone per replicate: {len(df_longest)} rows")
    return df_longest


def make_scatter_plot(
    df: pd.DataFrame,
    x_col: str,
    xlabel: str,
    out_path: Path,
    title: str,
) -> None:
    """
    Single-column scatter plot:
        x = df[x_col]
        y = df["ttp_day"]
    Coloured by strategy, with jitter and Spearman rho printed to log.
    """
    if df.empty:
        raise RuntimeError(
            "No rows to plot after filtering; check input / EXCLUDE_SAMPLES"
        )

    rng = np.random.default_rng(SEED)

    # Ranges
    xmin, xmax = float(df[x_col].min()), float(df[x_col].max())
    xr = max(1e-12, xmax - xmin)

    with use_style("figstyle.mplstyle", font_family="Arial"):
        fig, ax = new_figure(width="single", ratio=0.8)

        # X tick formatting: plain, no 1e3 offset
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=False))
        ax.ticklabel_format(style="plain", axis="x")

        if COLOR_BY_STRATEGY:
            strategies = sorted(df["strategy"].astype(str).unique())
            handles: List[Line2D] = []

            for strat in strategies:
                sub = df[df["strategy"] == strat]
                if sub.empty:
                    continue

                x = sub[x_col].to_numpy()
                y = sub["ttp_day"].to_numpy()
                jitter_x = rng.uniform(-JITTER_FRAC_X, JITTER_FRAC_X, size=x.size) * xr

                color = strategy_color_from_palette(strat)

                ax.scatter(
                    x + jitter_x,
                    y,
                    s=POINT_SIZE,
                    alpha=ALPHA,
                    c=color,
                    edgecolors="none" if EDGELESS else "face",
                    label=strat,
                )

                handles.append(
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        linestyle="None",
                        markersize=4,
                        color=color,
                        label=strat,
                    )
                )

            if handles:
                ax.legend(
                    handles=handles,
                    frameon=False,
                    fontsize=7,
                    ncol=1,
                    loc="lower left",
                )
        else:
            x = df[x_col].to_numpy()
            y = df["ttp_day"].to_numpy()
            jitter_x = rng.uniform(-JITTER_FRAC_X, JITTER_FRAC_X, size=x.size) * xr
            ax.scatter(
                x + jitter_x,
                y,
                s=POINT_SIZE,
                alpha=ALPHA,
                c="black",
                edgecolors="none" if EDGELESS else "face",
            )

        # Labels & title
        ax.set_xlabel(xlabel)
        ax.set_ylabel("TTP")
        ax.set_title(title)

        # Spearman correlation on raw (unjittered) data – printed to log only
        xs = df[x_col].to_numpy(dtype=float)
        ys = df["ttp_day"].to_numpy(dtype=float)
        mask = np.isfinite(xs) & np.isfinite(ys)
        if np.count_nonzero(mask) >= 3:
            res = spearmanr(xs[mask], ys[mask], nan_policy="omit")
            rho, pval = res.statistic, res.pvalue
            print(
                f"[STATS] {x_col}: Spearman rho = {rho:.3f}, p = {pval:.3g}, n = {np.count_nonzero(mask)}"
            )
        else:
            print(f"[STATS] {x_col}: not enough valid points for Spearman (n < 3)")

        # Axis scaling and limits
        if LOGX:
            ax.set_xscale("log")

        if XLIM is not None:
            ax.set_xlim(*XLIM)
        if YLIM is not None:
            ax.set_ylim(*YLIM)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))

        # No tight_layout / constrained_layout
        savefig(fig, out_path, dpi=300, transparent=True)
        plt.show()
        plt.close(fig)
        print(f"[SAVE] Wrote: {out_path}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    df = load_df()
    df = apply_exclude_samples(df)

    # keep only the longest-living clone per replicate (plate + sample)
    df = keep_longest_clone_per_replicate(df)

    # 1) Median density vs TTP
    make_scatter_plot(
        df=df,
        x_col="median_density",
        xlabel="Median Clone Density",
        out_path=OUT_MEDIAN,
        title="Higher Median Density Leads to Higher TTP",
    )

    # 2) Mean density vs TTP
    #make_scatter_plot(
    #    df=df,
    #    x_col="mean_density",
    #    xlabel="Mean clone density (a.u.)",
    #    out_path=OUT_MEAN,
    #    title="Average clone density shows a similar trend with TTP",
    #)


if __name__ == "__main__":
    main()
