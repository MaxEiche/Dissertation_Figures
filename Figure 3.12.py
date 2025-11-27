#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Figure 3.12 — Density correlation + examples split into two subfigures.

Subfigure (a–b):
    (a) All sectors: total cells (normalized) vs resistant density change (abs).
    (b) All sectors: ring total density (raw) vs resistant density change (abs).
    Both shown as contour-only density plots.

Subfigure (c–f):
    Four example timelines (NT, CT, AT50, AT100) from ring-based dataset (2h):
        timepoint vs ring total density (blue) and resistant density change (red)
        using left/right y-axes.

Replicate handling:
    - 2f data: replicate exclusions + truncations via apply_sample_exclusions(..., column="sample", cutoff_column="day").
    - 2h data: full-sample exclusions only (samples with None in EXCLUDE_SAMPLES are dropped);
               truncation cutoffs (which are in days) are not applied because no day column is present.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

from scipy.stats import gaussian_kde, pearsonr, spearmanr
from numpy.linalg import LinAlgError

from figtools import use_style, set_size, savefig, therapy_color
from experiment_filters import apply_sample_exclusions, EXCLUDE_SAMPLES

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------

BASE_DIR = Path(r"Y:\Members\MaxE\Thesis\Organized Figures\Figure 3.12")
DATA_DIR = BASE_DIR / "Data"

CSV_2F = DATA_DIR / "Figure2f_ring_change_vs_total_cells_timeseries.csv"
CSV_2H = DATA_DIR / "Figure2h_ring_change_vs_ring_total_density_timeseries.csv"

OUT_PDF_TOP = BASE_DIR / "Figure 3.12ab Density Correlations.pdf"
OUT_PDF_EXAMPLES = BASE_DIR / "Figure 3.12c-f Density Examples.pdf"
SUPP_DIR = BASE_DIR / "supplementary"


STYLE_PATH = Path("figstyle.mplstyle")

# -------------------------------------------------------------------
# Shared config
# -------------------------------------------------------------------

CHANGE_TYPE = "abs"
GRID_N = 120
CONTOUR_QUANTILES = (0.5, 0.7, 0.85, 0.95)

Y_LIM_TOP = (-2.0, 5.0)

EXAMPLES_2H: Dict[str, Dict[str, object]] = {
    "NT": {
        "plate": "Plate_4",
        "sample": "B1_NT-B1",
        "clone_label": 2,
    },
    "CT": {
        "plate": "Plate_2",
        "sample": "A3_CT-A3",
        "clone_label": 2,
    },
    "AT50": {
        "plate": "Plate_4",
        "sample": "D6_AT50-D6",
        "clone_label": 4,
    },
    "AT100": {
        "plate": "Plate_1",
        "sample": "B2_AT100-B2",
        "clone_label": 4,
    },
}

MS_TIME_TOTAL = 18
MS_TIME_CHG = 16
ALPHA_PTS = 0.7

COLOR_TOTAL = "tab:blue"
COLOR_CHG = "tab:red"

STRATEGY_DISPLAY_NAMES = {
    "NT": "NT",
    "CT": "CT",
    "AT50": "AT50",
    "AT100": "AT100",
    "Dens": "AT100+",
    "ATAgent1": "KPP-Agent",
    "ATAgent2": "LV-Agent",
}

STRATEGY_ORDER = [
    "NT",
    "CT",
    "AT50",
    "AT100",
    "Dens",
    "ATAgent1",
    "ATAgent2",
]

# -------------------------------------------------------------------
# Helper for exclusions
# -------------------------------------------------------------------

def drop_excluded_samples_only(df: pd.DataFrame, column: str = "sample") -> pd.DataFrame:
    """Remove samples that are fully excluded (EXCLUDE_SAMPLES[sample] is None)."""
    bad = [s for s, cutoff in EXCLUDE_SAMPLES.items() if cutoff is None]
    mask = ~df[column].isin(bad)
    dropped = (~mask).sum()
    if dropped:
        print(f"[exclude] Dropped {dropped} rows by full-sample exclusions (no truncation).")
    else:
        print("[exclude INFO] No full-sample exclusions applied.")
    return df[mask].copy()

# -------------------------------------------------------------------
# Correlation helper (log only)
# -------------------------------------------------------------------

def print_correlations(label: str, x: np.ndarray, y: np.ndarray) -> None:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = x.size
    if n < 3:
        print(f"[corr] {label}: n={n} (too few points)")
        return
    try:
        r, pr = pearsonr(x, y)
    except Exception:
        r, pr = np.nan, np.nan
    try:
        rho, ps = spearmanr(x, y)
    except Exception:
        rho, ps = np.nan, np.nan

    def fmt(v: float) -> str:
        return f"{v:.2f}" if np.isfinite(v) else "NA"

    def fmtp(v: float) -> str:
        return f"{v:.2g}" if np.isfinite(v) else "NA"

    print(
        f"[corr] {label}: n={n}, rho={fmt(rho)} (p={fmtp(ps)}), r={fmt(r)} (p={fmtp(pr)})"
    )

# -------------------------------------------------------------------
# Density grid + filled contours
# -------------------------------------------------------------------

def force_column_axes_width(ax, col, total_cols=2, left=0.12, right=0.98, wspace=0.30):
    """
    Ensures axes in column 'col' occupy the same normalized horizontal space.
    """
    total_width = right - left
    single_width = (total_width - (total_cols - 1) * wspace) / total_cols
    x0 = left + col * (single_width + wspace)
    ax.set_position([x0, ax.get_position().y0, single_width, ax.get_position().height])

def compute_density_grid(
    x: np.ndarray,
    y: np.ndarray,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    grid_n: int = GRID_N,
):
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
        jitter = 1e-6
        pts_j = np.vstack([
            x + np.random.normal(0, jitter, size=x.size),
            y + np.random.normal(0, jitter, size=y.size),
        ])
        kde = gaussian_kde(pts_j)
        Z = kde(positions).reshape(Xg.shape)
        return Xg, Yg, Z
    except Exception:
        pass

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
    base_color: str = "#1f77b4",
) -> None:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 10:
        print(f"[contour] Not enough points for filled contours (n={x.size}), skipping.")
        return

    Xg, Yg, Z = compute_density_grid(x, y, xlim, ylim, grid_n=GRID_N)
    vals = Z[np.isfinite(Z) & (Z > 0)]
    if vals.size == 0:
        print("[contour] All densities zero or NaN, skipping.")
        return

    core_levels = np.quantile(vals, CONTOUR_QUANTILES)
    levels = np.concatenate(([vals.min() * 0.999], core_levels, [vals.max() * 1.001]))

    base_rgba = np.array(mcolors.to_rgba(base_color))
    n_regions = len(levels) - 1
    colors = []
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

# -------------------------------------------------------------------
# Data loading
# -------------------------------------------------------------------

def load_2f_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.is_file():
        raise FileNotFoundError(f"2f CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)

    required = {
        "plate",
        "sample",
        "strategy",
        "clone_label",
        "sector",
        "timepoint",
        "day",
        "total_cells_norm",
        "Mean_Resistant_Density_Change",
        "change_type",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"2f CSV missing columns: {sorted(missing)}")

    df["timepoint"] = pd.to_numeric(df["timepoint"], errors="coerce").astype(int)
    df["day"] = pd.to_numeric(df["day"], errors="coerce")
    df["total_cells_norm"] = pd.to_numeric(df["total_cells_norm"], errors="coerce")
    df["Mean_Resistant_Density_Change"] = pd.to_numeric(
        df["Mean_Resistant_Density_Change"], errors="coerce"
    )

    df = apply_sample_exclusions(df, column="sample", cutoff_column="day", strict=False)
    df = df[df["change_type"].astype(str) == str(CHANGE_TYPE)].copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["total_cells_norm", "Mean_Resistant_Density_Change"]
    )
    return df

def load_2h_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.is_file():
        raise FileNotFoundError(f"2h CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)

    required = {
        "plate",
        "sample",
        "strategy",
        "clone_label",
        "sector",
        "change_type",
        "timepoint",
        "Mean_Total_Density",
        "Mean_Resistant_Density_Change",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"2h CSV missing columns: {sorted(missing)}")

    df = drop_excluded_samples_only(df, column="sample")
    df = df[df["change_type"].astype(str) == str(CHANGE_TYPE)].copy()

    df["timepoint"] = pd.to_numeric(df["timepoint"], errors="coerce").astype(int)
    df["Mean_Total_Density"] = pd.to_numeric(df["Mean_Total_Density"], errors="coerce")
    df["Mean_Resistant_Density_Change"] = pd.to_numeric(
        df["Mean_Resistant_Density_Change"], errors="coerce"
    )

    df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["Mean_Total_Density", "Mean_Resistant_Density_Change"]
    )
    return df

# -------------------------------------------------------------------
# Top panels
# -------------------------------------------------------------------

def plot_panel_total_vs_change(ax: plt.Axes, df_2f: pd.DataFrame) -> None:
    xs = df_2f["total_cells_norm"].to_numpy()
    ys = df_2f["Mean_Resistant_Density_Change"].to_numpy()

    def pad(lo, hi):
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return (0.25, 3.0)
        d = hi - lo
        return (lo - 0.06 * d, hi + 0.06 * d)

    xlim = pad(np.nanmin(xs), np.nanmax(xs))
    ylim = Y_LIM_TOP

    filled_contours_on_ax(ax, xs, ys, xlim, ylim, base_color="#1f77b4")

    ax.set_xlabel("Total Cells (normalized)")
    ax.set_ylabel(f"Res. Density Change")
    ax.set_box_aspect(1)
    ax.grid(False)

    print_correlations("totalCells_vs_change", xs, ys)

def plot_panel_ringdensity_vs_change(ax: plt.Axes, df_2h: pd.DataFrame) -> None:
    xs = df_2h["Mean_Total_Density"].to_numpy()
    ys = df_2h["Mean_Resistant_Density_Change"].to_numpy()

    xmax = np.nanmax(xs)
    xlim = (0.0, max(15.0, float(xmax * 1.05)))
    ylim = Y_LIM_TOP

    filled_contours_on_ax(ax, xs, ys, xlim, ylim, base_color="#1f77b4")

    ax.set_xlabel("Total Density")
    ax.set_ylabel(f"Res. Density Change")
    ax.set_box_aspect(1)
    ax.grid(False)

    print_correlations("ringDensity_vs_change", xs, ys)

# -------------------------------------------------------------------
# Example timelines
# -------------------------------------------------------------------

def extract_example_2h(
    df_2h: pd.DataFrame, plate: str, sample: str, clone_label: int
) -> pd.DataFrame:
    sub = df_2h[
        (df_2h["plate"] == plate)
        & (df_2h["sample"] == sample)
        & (df_2h["clone_label"] == int(clone_label))
    ].copy()
    return sub.sort_values(["timepoint", "sector"]).reset_index(drop=True)

def compute_example_limits_2h(
    examples: Dict[str, Dict[str, object]], df_2h: pd.DataFrame
) -> Tuple[np.ndarray, Tuple[float, float], Tuple[float, float]]:
    tps: List[int] = []
    y1_vals: List[float] = []
    y2_vals: List[float] = []

    for spec in examples.values():
        sub = extract_example_2h(
            df_2h,
            plate=str(spec["plate"]),
            sample=str(spec["sample"]),
            clone_label=int(spec["clone_label"]),
        )
        if sub.empty:
            continue
        tps.extend(sub["timepoint"].unique().tolist())
        y1_vals.extend(sub["Mean_Total_Density"].tolist())
        y2_vals.extend(sub["Mean_Resistant_Density_Change"].tolist())

    if not tps:
        xticks = np.array([0, 1, 2], dtype=int)
    else:
        xticks = np.array(sorted(set(tps)), dtype=int)

    def _pad(vals, frac=0.05, default=(0.0, 1.0)):
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return default
        lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
        if hi <= lo:
            return (lo - 0.5, hi + 0.5)
        span = hi - lo
        return (lo - frac * span, hi + frac * span)

    y1_lim = _pad(y1_vals, frac=0.05, default=(0.0, 10.0))
    y2_lim = _pad(y2_vals, frac=0.05, default=(-2.0, 6.0))
    return xticks, y1_lim, y2_lim

def plot_single_example(
    axL: plt.Axes,
    sub: pd.DataFrame,
    xticks: np.ndarray,
    y1_lim: Tuple[float, float],
    y2_lim: Tuple[float, float],
    name: str,
) -> None:
    if sub.empty:
        axL.text(0.5, 0.5, "No data", transform=axL.transAxes,
                 ha="center", va="center")
        axL.set_axis_off()
        return

    grouped = sub.groupby("timepoint", as_index=False).agg(
        Mean_Total_Density=("Mean_Total_Density", "mean"),
        Mean_Resistant_Density_Change=("Mean_Resistant_Density_Change", "mean"),
    )

    axR = axL.twinx()

    axL.scatter(
        grouped["timepoint"],
        grouped["Mean_Total_Density"],
        s=MS_TIME_TOTAL,
        alpha=ALPHA_PTS,
        color=COLOR_TOTAL,
        label=None,
    )
    axR.scatter(
        grouped["timepoint"],
        grouped["Mean_Resistant_Density_Change"],
        s=MS_TIME_CHG,
        alpha=ALPHA_PTS,
        color=COLOR_CHG,
        label=None,
    )

    axL.set_xlim(min(xticks) - 0.5, max(xticks) + 0.5)
    # Global x-tick range (0, 5, 10, 15, ...)
    global_max = int(np.nanmax(xticks))  # xticks comes from compute_example_limits_2h()
    tick_positions = list(range(0, global_max + 1, 5))
    axL.set_xticks(tick_positions)

    axL.set_ylim(*y1_lim)
    axR.set_ylim(-0.7, 3.2)

    axL.set_xlabel("Days")
    axL.set_ylabel("Total density ", color=COLOR_TOTAL)
    axR.set_ylabel(f"Res. Density Change", color=COLOR_CHG)

    axL.tick_params(axis="y", colors=COLOR_TOTAL, labelsize=8)
    axR.tick_params(axis="y", colors=COLOR_CHG, labelsize=8)
    axL.tick_params(axis="x", labelsize=8)
    axL.xaxis.labelpad = 2
    axL.yaxis.labelpad = 2
    axR.yaxis.labelpad = 2

# -------------------------------------------------------------------
# Supplementary per-strategy contour plots
# -------------------------------------------------------------------

def make_supplementary_grid(
    df: pd.DataFrame,
    xcol: str,
    ycol: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
) -> None:

    fig, axes = plt.subplots(2, 4)
    set_size(fig, width="thesis", ratio=0.7)  # nice compact supplementary
    fig.subplots_adjust(wspace=0.35, hspace=0.40)

    # global axis limits (consistent across all strategies)
    xs_all = df[xcol].to_numpy()
    ys_all = df[ycol].to_numpy()
    xlim = (
        float(np.nanpercentile(xs_all, 1)),
        float(np.nanpercentile(xs_all, 99)),
    )
    ylim = (
        float(np.nanpercentile(ys_all, 1)),
        float(np.nanpercentile(ys_all, 99)),
    )

    for idx, strat in enumerate(STRATEGY_ORDER):

        ax = axes[idx // 4, idx % 4]
        sub = df[df["strategy"] == strat]

        display_name = STRATEGY_DISPLAY_NAMES.get(strat, strat)

        if sub.empty:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    fontsize=8, color="gray")
            ax.set_title(display_name)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        x = sub[xcol].to_numpy()
        y = sub[ycol].to_numpy()

        # try canonical strategy colour
        try:
            col = therapy_color(display_name)
        except Exception:
            col = "#1f77b4"

        filled_contours_on_ax(ax, x, y, xlim, ylim, base_color=col)

        ax.set_title(display_name, fontsize=9)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.grid(False)

    fig.tight_layout()
    savefig(fig, out_path, dpi=300, transparent=True)
    plt.close(fig)
    print(f"[supp] Saved: {out_path}")


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    BASE_DIR.mkdir(parents=True, exist_ok=True)

    df_2f = load_2f_data(CSV_2F)
    df_2h = load_2h_data(CSV_2H)

    with use_style(style_mpl=STYLE_PATH):
        # ---------------- Top figure (a, b) ----------------
        fig1, (ax1, ax2) = plt.subplots(1, 2)
        set_size(fig1, width="thesis", ratio=0.463)
        # unified margins & spacing
        fig1.subplots_adjust(
            left=0.12,
            right=0.98,
            bottom=0.18,
            top=0.95,
            wspace=0.30,
        )

        plot_panel_total_vs_change(ax1, df_2f)
        plot_panel_ringdensity_vs_change(ax2, df_2h)

        for label, ax in zip(("a", "b"), (ax1, ax2)):
            ax.text(
                -0.12,
                1.02,
                label,
                transform=ax.transAxes,
                fontweight="bold",
                fontsize=12,
                ha="left",
                va="bottom",
            )

        savefig(fig1, OUT_PDF_TOP, dpi=300, transparent=True)
        print(f"[done] Saved top panels to: {OUT_PDF_TOP}")
        plt.show()
        plt.close(fig1)

        # ---------------- Bottom figure (c–f) ----------------
        fig2, axes = plt.subplots(2, 2)
        set_size(fig2, width="thesis", ratio=0.6)
        fig2.subplots_adjust(
            left=0.12,
            right=0.98,
            bottom=0.18,
            top=0.95,
            wspace=0.30,  # same as fig1
            hspace=0.45,
        )

        xticks, y1_lim, y2_lim = compute_example_limits_2h(EXAMPLES_2H, df_2h)
        ex_names = list(EXAMPLES_2H.keys())

        for idx, name in enumerate(ex_names):
            row, col = divmod(idx, 2)
            axL = axes[row, col]
            spec = EXAMPLES_2H[name]
            sub = extract_example_2h(
                df_2h,
                plate=str(spec["plate"]),
                sample=str(spec["sample"]),
                clone_label=int(spec["clone_label"]),
            )
            plot_single_example(axL, sub, xticks, y1_lim, y2_lim, name)

            # Add panel letters c, d, e, f
            panel_labels = ["c", "d", "e", "f"]
            axL.text(
                -0.12,
                1.02,
                panel_labels[idx],
                transform=axL.transAxes,
                ha="left",
                va="bottom",
                fontsize=12,
                fontweight="bold"
            )

        savefig(fig2, OUT_PDF_EXAMPLES, dpi=300, transparent=True)
        print(f"[done] Saved example panels to: {OUT_PDF_EXAMPLES}")
        plt.show()
        plt.close(fig2)

    # --------------- Supplementary: resChange vs totalCells ----------------
    make_supplementary_grid(
        df=df_2f,
        xcol="total_cells_norm",
        ycol="Mean_Resistant_Density_Change",
        xlabel="Total cells (normalized)",
        ylabel="Res. density change",
        out_path=BASE_DIR / "Supplementary_resChange_vs_totalCells.pdf",
    )

    # --------------- Supplementary: resChange vs totalDensity --------------
    make_supplementary_grid(
        df=df_2h,
        xcol="Mean_Total_Density",
        ycol="Mean_Resistant_Density_Change",
        xlabel="Ring total density (raw)",
        ylabel="Res. density change",
        out_path=BASE_DIR / "Supplementary_resChange_vs_totalDensity.pdf",
    )


if __name__ == "__main__":
    main()