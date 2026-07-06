# Figure 5.5 — S100 timecourse + correlation (thesis formatting)
# ---------------------------------------------------------------
# Uses: S100 shifted by 3 days, converted to mg/L
# Panels:
#   a) Timecourse (sus, res, total + S100 on right axis)
#   b) Correlation (total vs S100_mgL)
#
# Formatting follows global dissertation rules.

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from figtools import POPULATION_COLORS, use_style, set_size, savefig

# ============================================================
# Paths & data
# ============================================================

DATA_PATH = Path(
    r"Y:\Members\MaxE\Thesis\Organized Figures\Chapter_5\Figure 5.5 S100_Analysis\S100_Analysis.xlsx"
)
OUT_PDF = Path(
    r"Y:\Members\MaxE\Thesis\Organized Figures\Chapter_5\Figure 5.5 S100_Analysis\Figure_5.5_S100.pdf"
)

# A=Day, B=sus, C=res, D=total, E=S100 (µg/L)
df = pd.read_excel(
    DATA_PATH,
    sheet_name="Tabelle2",
    usecols="A:E",
    names=["Day", "sus", "res", "total", "S100_ugL"],
)

# Shift S100 by –3 days & convert to mg/L
df["S100_shifted_day"] = df["Day"] - 3
df["S100_mgL"] = df["S100_ugL"] / 1000.0

# ============================================================
# Build the combined figure
# ============================================================

with use_style("figstyle.mplstyle"):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    set_size(fig, width="thesis", ratio=0.38)

    # Manual spacing
    fig.subplots_adjust(
        left=0.10,
        right=0.98,
        bottom=0.18,
        top=0.92,
        wspace=0.35
    )

    # --------------------------------------------------------
    # PANEL a — Timecourse (left)
    # --------------------------------------------------------
    ax1.plot(df["Day"], df["sus"], color=POPULATION_COLORS["sensitive"], label="Sensitive")
    ax1.plot(df["Day"], df["res"], color=POPULATION_COLORS["resistant"], label="Resistant")
    ax1.plot(df["Day"], df["total"], color=POPULATION_COLORS["total"], label="Total")

    ax1.set_xlabel("Time [days]")
    ax1.set_ylabel("Cell count")
    ax1.set_xlim(0, 60)
    ax1.set_ylim(0, 110_000)

    # Right axis
    ax1_s100 = ax1.twinx()
    ax1_s100.plot(
        df["S100_shifted_day"],
        df["S100_mgL"],
        linestyle=":",
        linewidth=1.2,
        color="red",
        label="S100 (shifted)"
    )
    ax1_s100.set_ylabel("S100 [mg/L]")
    ax1_s100.set_ylim(0, 31)

    # Combined legend
    lines = ax1.get_lines() + ax1_s100.get_lines()
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, frameon=False, fontsize=7, loc="upper left")

    # Panel letter
    ax1.text(-0.18, 1.04, "a", transform=ax1.transAxes, fontweight="bold")

    # --------------------------------------------------------
    # PANEL b — Correlation (right)
    # --------------------------------------------------------
    x = df["total"].to_numpy()
    y = df["S100_mgL"].to_numpy()

    # Pearson printed only
    r, p = pearsonr(x, y)
    print(f"[S100] Pearson r = {r:.2f}, p = {p:.2e}")

    # Scatter (colored by day)
    sc = ax2.scatter(
        x, y,
        s=18,
        c=df["Day"],
        cmap="viridis",
        zorder=3
    )

    # Regression line
    slope, intercept = np.polyfit(x, y, 1)
    x_fit = np.linspace(x.min(), x.max(), 200)
    y_fit = slope * x_fit + intercept
    ax2.plot(
        x_fit, y_fit,
        color="red",
        linestyle="--",
        linewidth=1.0,
        zorder=2
    )

    ax2.set_xlabel("Total cell number")
    ax2.set_ylabel("S100 [mg/L]")
    ax2.set_xlim(0, 110_000)
    ax2.set_ylim(0, 31)

    # --------------------------------------------------------
    # Add colorbar for 'Day'
    # --------------------------------------------------------
    # Make a little bit of extra room on the right
    fig.subplots_adjust(right=0.93)

    cbar = fig.colorbar(
        sc,
        ax=ax2,
        fraction=0.046,  # nice compact bar
        pad=0.04
    )
    cbar.set_label("Day (days)")

    # Panel letter
    ax2.text(-0.18, 1.04, "b", transform=ax2.transAxes, fontweight="bold")

    # Save
    savefig(fig, OUT_PDF, dpi=300, transparent=True)
    print(f"[Saved] {OUT_PDF}")
    plt.show()

    plt.close(fig)
