# Figure 5.5 — S100 timecourse + correlation (thesis formatting)
# ---------------------------------------------------------------
# Uses: S100 shifted by 3 days (a reading on day t+3 is paired with cells on day t).
# S100 source values are in ng/L; convert to µg/L for plotting.
# Panels:
#   a) Timecourse (sus, res, total + S100 on right axis)
#   b) Correlation (total vs S100_ugL)
#
# Formatting follows global dissertation rules.

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from figtools import (
    PALETTE,
    POPULATION_COLORS,
    SECONDARY_LINEWIDTH,
    add_panel_label,
    format_axes,
    new_subplots,
    savefig,
    use_style,
)

# ============================================================
# Paths & data
# ============================================================

DATA_PATH = Path(
    r"Y:\Members\MaxE\Thesis\Organized Figures\Chapter_5\Figure 5.5 S100_Analysis\S100_Analysis.xlsx"
)
OUT_PDF = Path(
    r"Y:\Members\MaxE\Thesis\Organized Figures\Chapter_5\Figure 5.5 S100_Analysis\Figure_5.5_S100.pdf"
)

# A=Day, B=sus, C=res, D=total, E=S100 (ng/L)
df = pd.read_excel(
    DATA_PATH,
    sheet_name="Tabelle2",
    usecols="A:E",
    names=["Day", "sus", "res", "total", "S100_ngL"],
)

# 1 µg/L = 1000 ng/L. The raw values plateau at 30,000 ng/L,
# corresponding to the assay's 30 µg/L upper limit.
S100_LIMIT_UG_L = 30.0
LAG_DAYS = 3
df["S100_ugL"] = df["S100_ngL"] / 1000.0
df["S100_censored"] = df["S100_ugL"] >= S100_LIMIT_UG_L

# A reading taken on day t+3 is plotted/paired at cell-count day t.
df["S100_shifted_day"] = df["Day"] - LAG_DAYS

# Pair cell counts at t with S100 measured at t+3 using exact sampling days.
pairs = df[["Day", "total"]].merge(
    df[["S100_shifted_day", "S100_ugL", "S100_censored"]],
    left_on="Day",
    right_on="S100_shifted_day",
    how="inner",
    validate="one_to_one",
).dropna(subset=["total", "S100_ugL"])


def censored_kendall_tau(x, y, censored):
    """Kendall tau-b using only S100 orderings known despite upper censoring."""
    concordant = discordant = tied_x = tied_y = excluded = 0
    for i in range(len(x) - 1):
        for j in range(i + 1, len(x)):
            if censored[i] and censored[j]:
                # Both readings are only known to be at/above the limit.
                excluded += 1
                continue

            dx = np.sign(x[j] - x[i])
            if censored[i]:
                dy = -1  # y[i] is above the uncensored y[j]
            elif censored[j]:
                dy = 1   # uncensored y[i] is below y[j]
            else:
                dy = np.sign(y[j] - y[i])

            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                tied_x += 1
            elif dy == 0:
                tied_y += 1
            elif dx == dy:
                concordant += 1
            else:
                discordant += 1

    denominator = np.sqrt(
        (concordant + discordant + tied_x)
        * (concordant + discordant + tied_y)
    )
    tau = (concordant - discordant) / denominator if denominator else np.nan
    return tau, concordant, discordant, tied_x, tied_y, excluded

# ============================================================
# Build the combined figure
# ============================================================

with use_style("figstyle.mplstyle"):
    fig, (ax1, ax2) = new_subplots(1, 2, width="thesis", ratio=0.38)

    # --------------------------------------------------------
    # PANEL a — Timecourse (left)
    # --------------------------------------------------------
    ax1.plot(df["Day"], df["sus"], color=POPULATION_COLORS["sensitive"], label="Sensitive")
    ax1.plot(df["Day"], df["res"], color=POPULATION_COLORS["resistant"], label="Resistant")
    ax1.plot(df["Day"], df["total"], color=POPULATION_COLORS["total"], label="Total")

    ax1.set_xlabel("Time [Days]")
    ax1.set_ylabel("Cell count")
    ax1.set_xlim(0, 60)
    ax1.set_ylim(0, 110_000)

    # Right axis
    ax1_s100 = ax1.twinx()
    ax1_s100.plot(
        df["S100_shifted_day"],
        df["S100_ugL"].where(~df["S100_censored"]),
        linestyle=":",
        linewidth=SECONDARY_LINEWIDTH,
        color=PALETTE["cat4"],
        label="S100 (shifted)"
    )
    capped_timecourse = df["S100_censored"] & df["S100_shifted_day"].between(0, 60)
    ax1_s100.scatter(
        df.loc[capped_timecourse, "S100_shifted_day"],
        df.loc[capped_timecourse, "S100_ugL"],
        marker="^",
        s=24,
        color=PALETTE["cat4"],
        zorder=4,
        label="S100 at assay limit (≥30)",
    )
    ax1_s100.set_ylabel("S100 [µg/L]")
    ax1_s100.set_ylim(0, 31)
    format_axes(ax1_s100, spines=("top", "left"), tick_params={"right": True})

    # Combined legend
    lines = ax1.get_lines() + ax1_s100.get_lines()
    labels = [l.get_label() for l in lines]
    lines.append(
        Line2D([], [], marker="^", linestyle="none", color=PALETTE["cat4"],
               label="S100 at assay limit (≥30)")
    )
    labels.append("S100 at assay limit (≥30)")
    ax1.legend(lines, labels, frameon=False, loc="upper left")

    # Panel letter
    add_panel_label(ax1, "a")

    # --------------------------------------------------------
    # PANEL b — Correlation (right)
    # --------------------------------------------------------
    x = pairs["total"].to_numpy(dtype=float)
    y = pairs["S100_ugL"].to_numpy(dtype=float)
    censored = pairs["S100_censored"].to_numpy(dtype=bool)

    # Descriptive censored Kendall tau-b; do not report a p-value for this
    # single-replicate time series, whose observations are not independent.
    tau, concordant, discordant, tied_x, tied_y, excluded = censored_kendall_tau(
        x, y, censored
    )
    tau_label = (
        f"Censored Kendall tau-b = {tau:.2f}"
        if np.isfinite(tau)
        else "Association not estimable\n(all paired S100 values capped)"
    )
    print(
        f"[S100] {tau_label.replace(chr(10), ' ')}; "
        f"n={len(pairs)}, capped={censored.sum()}, "
        f"excluded both-capped pairs={excluded}. "
        "Descriptive only; no p-value (single time series)."
    )

    # Scatter (colored by day)
    sc = ax2.scatter(
        x, y,
        s=16,
        c=pairs["Day"],
        cmap="viridis",
        zorder=3
    )
    ax2.scatter(
        x[censored], y[censored],
        marker="^",
        s=34,
        facecolors="none",
        edgecolors="black",
        linewidths=0.8,
        zorder=4,
    )
    ax2.set_xlabel("Total cell number")
    ax2.set_ylabel("S100 [µg/L]")
    ax2.set_xlim(0, 110_000)
    ax2.set_ylim(0, 31)
    ax2.text(
        0.02, 0.96, f"τb = {tau:.2f}" if np.isfinite(tau) else "τb not estimable",
        transform=ax2.transAxes, ha="left", va="top",
    )

    # --------------------------------------------------------
    # Add colorbar for 'Day'
    # --------------------------------------------------------
    cbar = fig.colorbar(
        sc,
        ax=ax2,
        fraction=0.046,  # nice compact bar
        pad=0.04
    )
    cbar.set_label("Time [Days]")

    # Panel letter
    add_panel_label(ax2, "b")

    # Save
    savefig(fig, OUT_PDF, dpi=300, transparent=True, bbox_inches=None)
    savefig(fig, OUT_PDF.with_suffix(".png"), dpi=600, transparent=False, facecolor="white", bbox_inches=None)
    print(f"[Saved] {OUT_PDF}")
    plt.show()

    plt.close(fig)
