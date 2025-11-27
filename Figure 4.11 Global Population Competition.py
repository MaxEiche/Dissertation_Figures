#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Figure 4.11 – Global population competition under AT (topography style)

Standalone script that:
- Loads ME_A2-3 spheroid data (Plate 1 + 2) from the Excel file.
- Applies the same artifact-corrected normalization as in Figure_7.
- Adds treatment codes per day/replicate.
- Computes per-day relative resistant growth.
- Builds a per-point dataset with:
    g_normalized (susceptible population),
    relative_change (resistant growth per day),
    Treatment (NT, CT, AT100, AT80, AT50).

Produces TWO figures, styled like Figure 4.3:

1) Scatter + line-only KDE contours per treatment.
   -> Figure 4.11 GlobalPopulationCompetition_underAT.pdf

2) Filled KDE “blobs” per treatment, no points.
   -> Figure 4.11 GlobalPopulationCompetition_underAT_filled.pdf

Formatting:
- with use_style("figstyle.mplstyle", font_family="Arial")
- set_size(ax, width="thesis", ratio=0.70)  # 16 cm width
- no tight_layout() / constrained_layout()
- colours only via figtools.therapy_color(...)
- no replicate filtering via EXCLUDE_SAMPLES
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from scipy.stats import gaussian_kde

from figtools import use_style, set_size, savefig, therapy_color


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

DATA_XLS = Path(
    r"Y:\Members\MaxE\Thesis\Organized Figures\Figure 4.11 Global Competition\ME_A2-3 Data.xlsx"
)
OUT_PDF_SCATTER = DATA_XLS.parent / "Figure 4.11 GlobalPopulationCompetition_underAT.pdf"
OUT_PDF_FILLED = (
    DATA_XLS.parent / "Figure 4.11 GlobalPopulationCompetition_underAT_filled.pdf"
)


# ---------------------------------------------------------------------
# Sample layout (from Figure_7)
# ---------------------------------------------------------------------

sample_groups: Dict[str, List[str]] = {
    "No treatment": ["A1", "A2", "A3", "A4", "A5", "A6"],
    "Maximum-tolerable-dose": ["A7", "A8", "A9", "A10", "A11", "A12"],
    "Adaptive Therapy 100": [
        "B1",
        "B2",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7",
        "B8",
        "B9",
        "B10",
        "B11",
        "C1",
        "C2",
        "C3",
        "C4",
        "C5",
        "C6",
        "C7",
        "C8",
        "C9",
        "C10",
        "C11",
    ],
    "Adaptive Therapy 80": [
        "D1",
        "D2",
        "D3",
        "D4",
        "D5",
        "D6",
        "D7",
        "D8",
        "D9",
        "D10",
        "D11",
        "E1",
        "E2",
        "E3",
        "E4",
        "E5",
        "E6",
        "E7",
        "E8",
        "E9",
        "E10",
        "E11",
    ],
    "Adaptive Therapy 50": [
        "F1",
        "F2",
        "F3",
        "F4",
        "F5",
        "F6",
        "F7",
        "F8",
        "F9",
        "F10",
        "F11",
        "G1",
        "G2",
        "G3",
        "G4",
        "G5",
        "G6",
        "G7",
        "G8",
        "G9",
        "G10",
        "G11",
    ],
    "Resistant Control No Treatment": ["H1", "H2", "H3", "H4", "H5", "H6"],
    "Resistant Control Maximum-tolerable-dose": ["H7", "H8", "H9", "H10", "H11", "H12"],
    "Intermittent": ["B12", "C12", "D12", "E12", "F12", "G12"],
}


# ---------------------------------------------------------------------
# Helper: KDE contours (line-only) – same as Figure 4.3
# ---------------------------------------------------------------------

def add_kde_contours(
    ax: plt.Axes,
    x,
    y,
    probs=(0.1, 0.2, 0.3, 0.4),
    gridsize: int = 300,
    color="k",
    alpha: float = 0.9,
    linewidths: Tuple[float, float, float] = (2.0, 1.5, 1.0),
    bw_method=0.7,
    pad: float = 0.10,
) -> None:
    """
    KDE density contours enclosing a given probability mass for (x, y).

    - probs: cumulative probability masses to enclose (lower -> tighter).
    - bw_method: KDE bandwidth (None | float | 'scott' | 'silverman').
    - pad: fraction of data range added around the data before gridding.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 5:
        return

    xm, xM = np.nanmin(x[m]), np.nanmax(x[m])
    ym, yM = np.nanmin(y[m]), np.nanmax(y[m])
    xr = xM - xm if xM > xm else 1.0
    yr = yM - ym if yM > ym else 1.0
    xm -= pad * xr
    xM += pad * xr
    ym -= pad * yr
    yM += pad * yr

    xx = np.linspace(xm, xM, gridsize)
    yy = np.linspace(ym, yM, gridsize)
    X, Y = np.meshgrid(xx, yy)

    kde = gaussian_kde(np.vstack([x[m], y[m]]), bw_method=bw_method)
    Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

    z = Z.ravel()
    if not np.isfinite(z).any() or np.nanmin(Z) == np.nanmax(Z):
        return  # flat or invalid surface

    # Build contour levels for requested cumulative probs
    z_sorted = np.sort(z)[::-1]
    cdf = np.cumsum(z_sorted)
    cdf /= cdf[-1]
    levels = []
    for p in sorted(probs):
        idx = np.searchsorted(cdf, p)
        thr = z_sorted[idx] if idx < len(z_sorted) else z_sorted[-1]
        levels.append(float(thr))

    levels = np.sort(np.unique(levels))
    if levels.size == 0 or not np.isfinite(levels).all():
        return

    ax.contour(
        X,
        Y,
        Z,
        levels=levels,
        colors=color,
        linewidths=linewidths,
        alpha=alpha,
    )


# ---------------------------------------------------------------------
# Helper: filled contours – copied from Figure 4.3 and adapted
# ---------------------------------------------------------------------

def filled_contours_on_ax(
    ax: plt.Axes,
    x,
    y,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    base_color="#1f77b4",
    gridsize: int = 300,
    probs=(0.1, 0.2, 0.3, 0.4),
    bw_method=0.7,
    pad: float = 0.10,
) -> None:
    """
    Draw filled KDE contours for (x, y) on ax, using the SAME level construction
    as add_kde_contours (same 'probs'), but with semi-transparent fills
    in 'base_color' instead of points.

    This way, the contour lines in the filled plot coincide with those in
    the scatter+contour plot; only the fill style changes.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size < 5:
        print(f"[contour] Not enough points for filled contours (n={x.size}), skipping.")
        return

    # Same domain logic as in add_kde_contours
    xm, xM = np.nanmin(x), np.nanmax(x)
    ym, yM = np.nanmin(y), np.nanmax(y)
    xr = xM - xm if xM > xm else 1.0
    yr = yM - ym if yM > ym else 1.0
    xm -= pad * xr
    xM += pad * xr
    ym -= pad * yr
    yM += pad * yr

    xx = np.linspace(xm, xM, gridsize)
    yy = np.linspace(ym, yM, gridsize)
    Xg, Yg = np.meshgrid(xx, yy)

    kde = gaussian_kde(np.vstack([x, y]), bw_method=bw_method)
    Z = kde(np.vstack([Xg.ravel(), Yg.ravel()])).reshape(Xg.shape)

    z = Z.ravel()
    if not np.isfinite(z).any() or np.nanmin(Z) == np.nanmax(Z):
        print("[contour] Flat or invalid KDE surface, skipping.")
        return

    # Probability-based levels as in add_kde_contours
    z_sorted = np.sort(z)[::-1]
    cdf = np.cumsum(z_sorted)
    cdf /= cdf[-1]
    line_levels = []
    for p in sorted(probs):
        idx = np.searchsorted(cdf, p)
        thr = z_sorted[idx] if idx < len(z_sorted) else z_sorted[-1]
        line_levels.append(float(thr))
    line_levels = np.sort(np.unique(line_levels))

    if line_levels.size == 0 or not np.isfinite(line_levels).all():
        print("[contour] Invalid contour levels, skipping.")
        return

    # Fill bands: [level0, level1], ..., [last_level, z_max]
    z_max = float(np.nanmax(Z))
    fill_levels = np.concatenate([line_levels, [z_max + 1e-12]])

    base_rgba = np.array(mcolors.to_rgba(base_color))
    n_regions = len(fill_levels) - 1
    colors = []

    for i in range(n_regions):
        # lowest band = lightest, highest-density core = strongest colour
        alpha = 0.15 + 0.55 * (i / max(1, n_regions - 1))
        rgba = base_rgba.copy()
        rgba[-1] = alpha
        colors.append(tuple(rgba))

    ax.contourf(
        Xg,
        Yg,
        Z,
        levels=fill_levels,
        colors=colors,
        zorder=1,
    )

    ax.contour(
        Xg,
        Yg,
        Z,
        levels=line_levels,
        colors=[base_color],
        linewidths=0.8,
        zorder=2,
    )


# ---------------------------------------------------------------------
# Data extraction & normalization (de-artifacted, from Figure_7)
# ---------------------------------------------------------------------

def extract_and_normalize_remove_artifacts(data_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract g / r / total and normalize to baseline (Day 0),
    applying the same artifact corrections as in Figure_7.
    """
    extracted_data = {"Replicate": [], "Day": [], "g": [], "r": [], "t": []}

    # Every fifth column: replicate label, g, r, t pattern
    replicate_columns = range(3, data_df.shape[1], 5)
    g_columns = range(3, data_df.shape[1], 5)
    r_columns = range(5, data_df.shape[1], 5)
    t_columns = range(6, data_df.shape[1], 5)

    # Day labels from first column (row ≥ 3)
    days = data_df.iloc[3:, 0].dropna().astype(int).unique()

    for rep_col, g_col, r_col, t_col in zip(
        replicate_columns, g_columns, r_columns, t_columns
    ):
        replicate = data_df.iloc[1, rep_col]
        for day in days:
            # Skip artifact days exactly as in Figure_7
            if day in (29, 15, 27):
                continue

            day_data = data_df.loc[data_df.iloc[:, 0] == day]
            if day_data.empty:
                continue

            g_value = day_data.iloc[0, g_col]
            r_value = day_data.iloc[0, r_col]

            # Early-day offset corrections
            if day in range(-2, 6):  # -2..5
                g_value -= 40_000_000
                r_value -= 40_000_000
            elif day == 6:
                g_value -= 5_000_000
                r_value -= 5_000_000

            # Additional correction: subtract r/12 from g
            g_value -= r_value / 12.0

            extracted_data["Replicate"].append(replicate)
            extracted_data["Day"].append(day)
            extracted_data["g"].append(g_value)
            extracted_data["r"].append(r_value)
            extracted_data["t"].append(day_data.iloc[0, t_col])

    extracted_df = pd.DataFrame(extracted_data)

    # Normalize to baseline total at Day 0
    normalized_data = {
        "Replicate": [],
        "Day": [],
        "g_normalized": [],
        "r_normalized": [],
        "t_normalized": [],
    }

    baseline_t_values = (
        extracted_df[extracted_df["Day"] == 0].set_index("Replicate")["t"]
    )

    for _, row in extracted_df.iterrows():
        replicate = row["Replicate"]
        baseline_t = baseline_t_values.get(replicate, None)
        if baseline_t is None:
            continue

        normalized_data["Replicate"].append(replicate)
        normalized_data["Day"].append(row["Day"])
        normalized_data["g_normalized"].append(row["g"] / baseline_t)
        normalized_data["r_normalized"].append(row["r"] / baseline_t)
        normalized_data["t_normalized"].append(row["t"] / baseline_t)

    return pd.DataFrame(normalized_data)


def combine_normalized_with_treatment(
    normalized_df: pd.DataFrame,
    treatment_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach a numeric 'treatment' code to each (Day, Replicate) entry.

    treatment codes (from Figure_7):
        1 = treatment
        2 = on-study (no treatment)
        3,4,5 = death / dropout
        0 = after dropout
    """
    treatment_days = treatment_df.iloc[:, 0].dropna().unique()
    replicate_columns = treatment_df.columns[2:]

    treatment_dict: Dict[Tuple[int, str], float] = {}
    for _, row in treatment_df.iterrows():
        day_str = row.iloc[0]
        if day_str not in treatment_days:
            continue
        # Expect strings like "d-2", "d0", ...
        day = int(day_str[1:])
        for replicate in replicate_columns:
            treatment_dict[(day, replicate)] = row[replicate]

    normalized_df = normalized_df.copy()
    normalized_df["treatment"] = normalized_df.apply(
        lambda r: treatment_dict.get((r["Day"], r["Replicate"]), 0),
        axis=1,
    )
    return normalized_df


# ---------------------------------------------------------------------
# Build global competition dataset
# ---------------------------------------------------------------------

def build_global_competition_df(
    combined_df_plate1: pd.DataFrame,
    combined_df_plate2: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare the dataframe used for the global population competition plot.

    Steps:
        - Prefix replicates with plate ID (1/2).
        - Compute per-day relative resistant growth (ratio).
        - Restrict to on-study days (exclude 0,3,4,5).
        - Restrict relative_change to [0.75, 2].
        - Map replicates to treatment strategies via sample_groups.
        - Keep only NT, CT, AT100, AT80, AT50.
    """
    df1 = combined_df_plate1.copy()
    df2 = combined_df_plate2.copy()

    # Prefix replicate IDs
    df1["Replicate"] = df1["Replicate"].apply(lambda x: f"1{x}")
    df2["Replicate"] = df2["Replicate"].apply(lambda x: f"2{x}")
    df = pd.concat([df1, df2], ignore_index=True)

    # Per-day relative change of resistant population
    df["relative_change"] = (
        df.groupby("Replicate")["r_normalized"].pct_change(fill_method=None) + 1.0
    )

    # Keep relevant columns
    df = df[
        [
            "Replicate",
            "Day",
            "r_normalized",
            "t_normalized",
            "g_normalized",
            "treatment",
            "relative_change",
        ]
    ]

    # On-study days only (exclude death/dropout and 'after' states)
    df = df[~df["treatment"].isin([0.0, 3.0, 4.0, 5.0])]

    # Remove rows with NaN growth
    df = df.dropna(subset=["relative_change"])

    # Restrict to reasonable range, as in Figure_7
    df = df[df["relative_change"].between(0.75, 2.0)]

    # Map replicates to short treatment labels via sample_groups
    long_to_short = {
        "No treatment": "NT",
        "Maximum-tolerable-dose": "CT",
        "Adaptive Therapy 100": "AT100",
        "Adaptive Therapy 80": "AT80",
        "Adaptive Therapy 50": "AT50",
    }
    selected_longs = set(long_to_short.keys())

    rep_to_short: Dict[str, str] = {}
    for plate_prefix in ["1", "2"]:
        for long_name, wells in sample_groups.items():
            if long_name not in selected_longs:
                continue
            short = long_to_short[long_name]
            for well in wells:
                rep_to_short[plate_prefix + well] = short

    df["Treatment"] = df["Replicate"].map(rep_to_short)

    # Keep only the five strategies we want to show
    df = df[df["Treatment"].isin(long_to_short.values())].copy()

    return df


# ---------------------------------------------------------------------
# Plotting – 1) Scatter + line-only contours
# ---------------------------------------------------------------------

def plot_global_competition_scatter(df: pd.DataFrame) -> None:
    """
    Full-width cloud + KDE "topography": global population competition under AT.

    Version 1: points + line-only contours, like the first panel in Figure 4.3.
    """
    # Colors from figtools palette
    treatment_colors = {
        "AT100": therapy_color("AT100"),
        "AT80": therapy_color("AT80"),
        "AT50": therapy_color("AT50"),
        "NT": therapy_color("NT"),
        "CT": therapy_color("CT"),
    }

    # X = normalized susceptible (g), Y = relative resistant growth [%]
    # Axis limits: follow 4.11 logic (adapt if needed)
    xlim = (0.0, 4.0)
    ylim = (-10.0, 30.0)

    with use_style("figstyle.mplstyle", font_family="Arial"):
        fig, ax = plt.subplots()
        set_size(ax, width="thesis", ratio=0.70)

        # Panel letter
        ax.text(
            -0.15,
            1.05,
            "a",
            transform=ax.transAxes,
            fontweight="bold",
            va="bottom",
            ha="left",
        )

        # Axes cosmetics: no top/right spines, ticks outward
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(direction="out")

        # Order in legend: NT, CT, then adaptive strategies
        plot_order = ["NT", "CT", "AT100", "AT80", "AT50"]

        for t in plot_order:
            sel = df["Treatment"] == t
            if sel.sum() < 3:
                continue

            xg = df.loc[sel, "g_normalized"].to_numpy()
            yg = (df.loc[sel, "relative_change"].to_numpy() - 1.0) * 100.0
            color = treatment_colors.get(t, "k")

            # Scatter cloud
            ax.scatter(xg, yg, s=8, c=color, alpha=0.3, label=t)

            # KDE contour lines around the cloud
            add_kde_contours(
                ax,
                xg,
                yg,
                probs=(0.1, 0.2, 0.3, 0.4),
                gridsize=300,
                color=color,
                alpha=0.9,
                linewidths=(2.0, 1.5, 1.0),
                bw_method=0.7,
                pad=0.10,
            )

        ax.set_xlabel("Normalized susceptible population")
        ax.set_ylabel("Relative resistant growth [%]")
        ax.set_title("Global population competition under AT")

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        # Legend: frameless, compact
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(
                handles,
                labels,
                frameon=False,
                loc="best",
            )

        savefig(fig, OUT_PDF_SCATTER, dpi=300, transparent=True)
        plt.show()
        plt.close(fig)
        print(f"Saved scatter+contours global competition figure to: {OUT_PDF_SCATTER}")


# ---------------------------------------------------------------------
# Plotting – 2) Filled KDE blobs
# ---------------------------------------------------------------------

def plot_global_competition_filled(df: pd.DataFrame) -> None:
    """
    Full-width filled KDE blobs: global population competition under AT.

    Version 2: no points, only filled density “blobs” per treatment,
    plus contour lines and a legend with coloured squares,
    analogous to the second figure in Figure 4.3.
    """
    treatment_colors = {
        "AT100": therapy_color("AT100"),
        "AT80": therapy_color("AT80"),
        "AT50": therapy_color("AT50"),
        "NT": therapy_color("NT"),
        "CT": therapy_color("CT"),
    }

    xlim = (0.0, 4.0)
    ylim = (-10.0, 30.0)

    with use_style("figstyle.mplstyle", font_family="Arial"):
        fig, ax = plt.subplots()
        set_size(ax, width="thesis", ratio=0.70)

        # Panel letter
        ax.text(
            -0.15,
            1.05,
            "b",
            transform=ax.transAxes,
            fontweight="bold",
            va="bottom",
            ha="left",
        )

        # Axes cosmetics
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(direction="out")
        ax.set_facecolor("white")

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        plot_order = ["NT", "CT", "AT100", "AT80", "AT50"]

        for t in plot_order:
            sel = df["Treatment"] == t
            if sel.sum() < 3:
                continue

            xg = df.loc[sel, "g_normalized"].to_numpy()
            yg = (df.loc[sel, "relative_change"].to_numpy() - 1.0) * 100.0
            color = treatment_colors.get(t, "k")

            # Filled blobs, no dots
            filled_contours_on_ax(
                ax,
                xg,
                yg,
                xlim=xlim,
                ylim=ylim,
                base_color=color,
                gridsize=300,
                probs=(0.1, 0.2, 0.3, 0.4),
                bw_method=0.7,
                pad=0.10,
            )

        ax.set_xlabel("Normalized Susceptible Population")
        ax.set_ylabel("Relative Resistant Growth [%]")
        ax.set_title("Global Population Competition under AT (Non-Embedded)")

        # Legend: coloured squares, frameless
        handles = [
            plt.Line2D(
                [0],
                [0],
                marker="s",
                linestyle="",
                markerfacecolor=treatment_colors[t],
                markeredgecolor="none",
                label=t,
            )
            for t in plot_order
            if t in treatment_colors
        ]
        if handles:
            ax.legend(handles=handles, frameon=False, loc="best")

        savefig(fig, OUT_PDF_FILLED, dpi=300, transparent=True)
        plt.show()
        plt.close(fig)
        print(f"Saved filled-blobs global competition figure to: {OUT_PDF_FILLED}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    # Load Excel
    xls = pd.ExcelFile(DATA_XLS)

    # Plate data
    data_df_plate1 = pd.read_excel(xls, "Plate 1")
    data_df_plate2 = pd.read_excel(xls, "Plate 2")

    # Treatment sheets
    treatment_df_plate1 = pd.read_excel(xls, "Plate 1 Treatment")
    treatment_df_plate2 = pd.read_excel(xls, "Plate 2 Treatment")

    # Artifact-corrected normalization (as in Figure_7)
    norm1 = extract_and_normalize_remove_artifacts(data_df_plate1)
    norm2 = extract_and_normalize_remove_artifacts(data_df_plate2)

    # Add treatment codes
    comb1 = combine_normalized_with_treatment(norm1, treatment_df_plate1)
    comb2 = combine_normalized_with_treatment(norm2, treatment_df_plate2)

    # Build global competition dataframe
    df_global = build_global_competition_df(comb1, comb2)

    # Plot both variants
    plot_global_competition_scatter(df_global)
    plot_global_competition_filled(df_global)


if __name__ == "__main__":
    main()