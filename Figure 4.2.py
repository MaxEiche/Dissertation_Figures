# Figure_4_5_combined.py — Spheroid sinking behaviour + resistant growth (side-by-side)
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from figtools import use_style, set_size, savefig
from experiment_filters import EXCLUDE_SAMPLES  # required by global pipeline

# -----------------------
# Paths
# -----------------------
EXCEL_PATH = Path(
    r"Y:\Members\MaxE\Thesis\Organized Figures\Figure 4.2\Spheroid Height DropSeeding.xlsx"
)
ROOT_INT = Path(r"Y:\Members\MaxE\Thesis\Organized Figures\Figure 4.2")
CSV_PATH = ROOT_INT / "intensity_sum2d_per_channel_per_scene.csv"

OUTDIR = Path(r"Y:\Members\MaxE\Thesis\Organized Figures\Figure 4.2")
OUTDIR.mkdir(parents=True, exist_ok=True)

# -----------------------
# Load data for panel B (sinking behaviour)
# -----------------------
df_raw = pd.read_excel(EXCEL_PATH, sheet_name="Tabelle2", header=0)
df_sink = df_raw.iloc[:, :3].copy()
df_sink.columns = ["Cooling_min", "Distance_um", "SD"]

for col in df_sink.columns:
    df_sink[col] = (
        df_sink[col]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .astype(float)
    )

df_sink = (
    df_sink.dropna(subset=["Cooling_min", "Distance_um"])
    .sort_values("Cooling_min")
    .reset_index(drop=True)
)

# -----------------------
# Load + prepare data for panel C (normalized timelines)
# -----------------------
SCENE = "E4"
CHANNEL_RESISTANT = "C0"
CHANNEL_SUSCEPTIBLE = "C1"

df_int = pd.read_csv(CSV_PATH)
sub = df_int[
    (df_int["scene"] == SCENE)
    & (df_int["channel"].isin([CHANNEL_RESISTANT, CHANNEL_SUSCEPTIBLE]))
].copy()

sub["day"] = pd.to_numeric(sub["day"], errors="coerce")
sub = sub.dropna(subset=["day", "intensity_sum_2d", "n_pixels_2d"])

sub["mean_intensity"] = sub["intensity_sum_2d"] / sub["n_pixels_2d"].replace(0, np.nan)

wide = (
    sub.groupby(["day", "channel"])["mean_intensity"]
       .mean()
       .unstack("channel")
       .sort_index()
)

wide = wide.rename(columns={
    CHANNEL_RESISTANT: "resistant",
    CHANNEL_SUSCEPTIBLE: "susceptible",
})

wide["total"] = wide["resistant"] + wide["susceptible"]

first_day = float(wide.index.min())
norm_den = float(wide.loc[first_day, "total"])
if not np.isfinite(norm_den) or norm_den == 0:
    raise ValueError(
        f"Normalization failed: total at first timepoint (day={first_day}) is {norm_den}"
    )

wide_norm = wide[["susceptible", "resistant", "total"]] / norm_den

# NOTE:
# This dataset has no per-replicate/sample ID column, so EXCLUDE_SAMPLES
# from experiment_filters cannot be applied in a meaningful way here.
# If a replicate/sample column is added later, filtering should be done
# before aggregation and plotting.

# -----------------------
# Combined plot
# -----------------------
with use_style("figstyle.mplstyle"):
    # Ensure Arial font globally (in case not already specified in figstyle)
    plt.rcParams["font.family"] = "Arial"

    fig, axes = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False)
    ax_left, ax_right = axes

    # Subplot labels: panels b and c
    panel_labels = ["b", "c"]
    for ax, label in zip(axes, panel_labels):
        ax.text(
            -0.15, 1.05,
            label,
            transform=ax.transAxes,
            fontweight="bold",
            va="bottom",
            ha="left",
        )

    # Make figure thesis-wide and reasonably flat
    set_size(ax_left, width="thesis", ratio=0.5)

    # Axes cosmetics: no top/right spines, ticks outward
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out")

    # --- Panel b: sinking behaviour ---
    ax_left.errorbar(
        df_sink["Cooling_min"].values,
        df_sink["Distance_um"].values,
        yerr=df_sink["SD"].values,
        fmt="-o",
        color="black",
        ecolor="black",
        capsize=3,
    )
    ax_left.set_title("Spheroid sinking behaviour")
    ax_left.set_xlabel("Cooling time [min]")
    ax_left.set_ylabel("Distance to medium [µm]")

    max_x = float(df_sink["Cooling_min"].max())
    ax_left.set_xticks(range(2, int(max_x) + 2, 2))

    ymax = float((df_sink["Distance_um"] + df_sink["SD"]).max())
    ax_left.set_ylim(0, ymax * 1.1 if ymax > 0 else 1.0)
    ax_left.margins(x=0.02, y=0.05)

    # --- Panel c: resistant growth after CT ---
    ax_right.plot(
        wide_norm.index,
        wide_norm["susceptible"],
        marker="o",
        color="#4169E1",  # blue
        label="Susceptible",
    )
    ax_right.plot(
        wide_norm.index,
        wide_norm["resistant"],
        marker="o",
        color="#DAA520",  # yellow/golden
        label="Resistant",
    )
    ax_right.plot(
        wide_norm.index,
        wide_norm["total"],
        marker="o",
        color="black",
        label="Total",
    )

    ax_right.set_title("Growth under CT")
    ax_right.set_xlabel("Time after treatment start [days]")
    ax_right.set_ylabel("Normalized channel intensity")
    ax_right.set_ylim(0, 1.3)

    max_day = int(np.nanmax(wide_norm.index))
    ax_right.set_xticks(np.arange(2, max_day + 2, 2))

    # Legend: frameless
    ax_right.legend(fontsize=6, frameon=False)
    ax_right.margins(x=0.02, y=0.05)

    # Global spacing (no tight_layout)
    fig.subplots_adjust(left=0.12, right=0.98, wspace=0.30)

    out_pdf = OUTDIR / "Figure 4.2.pdf"
    plt.show()
    savefig(fig, out_pdf, dpi=300, transparent=True)
    plt.close(fig)

print("Saved combined figure:", out_pdf)
