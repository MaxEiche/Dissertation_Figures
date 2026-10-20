#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Figure 4.16 – Distribution of resistant cell behaviour (CT vs AT)
Half-width (single column) thesis-formatted boxplot.
"""

from pathlib import Path
from itertools import combinations
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Thesis formatting
from figtools import (
    ANNOTATION_FONTSIZE,
    SECONDARY_LINEWIDTH,
    format_axes,
    new_figure,
    savefig,
    therapy_color,
    use_style,
)

# ======== USER OPTIONS ========
Y_AXIS_LABEL = "Distribution Score"

CT_COLOR = therapy_color("CT")
AT_COLOR = therapy_color("AT100")
# ===============================

# -------------------- Data --------------------
AT_vals = np.array([2, 3, 1, 1, 1, 2, 3, 2, 5, 2, 4, 3])
CT_vals = np.array([10, 6, 9, 6, 5, 8, 8, 5])

groups = [("CT", CT_vals), ("AT", AT_vals)]
AT, CT = AT_vals, CT_vals
all_vals = np.concatenate([CT_vals, AT_vals]).astype(float)

n_CT = len(CT)
n_AT = len(AT)

# -------------------- Exhaustive label-permutation test --------------------
pooled_ranks = stats.rankdata(all_vals, method="average")  # Retain ties as midranks.
observed_rank_sum = pooled_ranks[n_CT:].sum()
U = observed_rank_sum - n_AT * (n_AT + 1) / 2
null_centre = n_AT * (len(all_vals) + 1) / 2
observed_distance = abs(observed_rank_sum - null_centre)
permutation_count = extreme_count = 0
for allocation in combinations(range(len(all_vals)), n_AT):
    rank_sum = pooled_ranks[list(allocation)].sum()
    permutation_count += 1
    if abs(rank_sum - null_centre) >= observed_distance:
        extreme_count += 1
p = extreme_count / permutation_count
print(f"[Two-sided midrank label-permutation] U_AT = {U:.3g}, "
      f"extreme allocations = {extreme_count} / {permutation_count}, p = {p:.8g}, "
      f"n_CT = {n_CT}, n_AT = {n_AT}")


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

rng = np.random.default_rng(42)


def jitter(x, n, scale=0.04):
    return x + rng.normal(0, scale, size=n)


def safe_bracket_y(ax, values, pad_frac=0.03):
    ymin, ymax = ax.get_ylim()
    yrange = ymax - ymin
    ymax_data = float(np.nanmax(values))
    return ymax_data + pad_frac * yrange


def p_to_stars(pval: float) -> str:
    """Convert p-value to significance stars."""
    if pval < 0.001:
        return "***"
    elif pval < 0.01:
        return "**"
    elif pval < 0.05:
        return "*"
    else:
        return "ns"


def add_bracket_with_annotations(
    ax,
    x1: float,
    x2: float,
    y: float,
    stars: str,
    fontsize: float = ANNOTATION_FONTSIZE,
):
    """
    Draw comparison bracket between x1 and x2 at height y and place
    significance stars above the bracket.
    """
    ymin, ymax = ax.get_ylim()
    yrange = ymax - ymin

    # Vertical and horizontal coordinates for the bracket
    bar_y = y + 0.05 * yrange
    ax.plot(
        [x1, x1, x2, x2],
        [y, bar_y, bar_y, y],
        lw=SECONDARY_LINEWIDTH,
        c="black",
    )

    # Text positions
    text_y_top = bar_y + 0.00 * yrange
    # Stars above
    ax.text(
        (x1 + x2) / 2,
        text_y_top,
        stars,
        ha="center",
        va="bottom",
        fontsize=fontsize,
    )

    # Ensure stars are visible
    need = text_y_top + 0.05 * yrange
    if need > ymax:
        ax.set_ylim(ymin, need)


# ---------------------------------------------------------------------
# Final plot – thesis format (single width)
# ---------------------------------------------------------------------

with use_style("figstyle.mplstyle"):

    fig, ax = new_figure(width="single", ratio=0.7)

    data = [CT_vals, AT_vals]
    labels = ["CT", "AT"]        # short labels as requested
    positions = [1, 2]
    colors = [CT_COLOR, AT_COLOR]

    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.5,
        patch_artist=True,
        showfliers=True,
        medianprops=dict(color="black", linewidth=1.3),
        boxprops=dict(alpha=0.30, linewidth=SECONDARY_LINEWIDTH),
        whiskerprops=dict(linewidth=SECONDARY_LINEWIDTH),
        whis=(0, 100),
        capprops=dict(linewidth=SECONDARY_LINEWIDTH),
    )

    # Color boxes
    bp["boxes"][0].set_facecolor(CT_COLOR)
    bp["boxes"][1].set_facecolor(AT_COLOR)

    for w in bp["whiskers"]:
        w.set_color("black")
    for c in bp["caps"]:
        c.set_color("black")

    # Raw scatter points
    ax.plot(jitter(np.ones_like(CT_vals) * 1, len(CT_vals)), CT_vals,
            "o", alpha=0.8, color=CT_COLOR)
    ax.plot(jitter(np.ones_like(AT_vals) * 2, len(AT_vals)), AT_vals,
            "o", alpha=0.8, color=AT_COLOR)

    # Formatting
    format_axes(ax)

    ax.set_xticks(positions, labels)
    ax.set_ylabel(Y_AXIS_LABEL)

    # Bracket with stars
    y_box = safe_bracket_y(ax, all_vals) + 0.6
    stars = p_to_stars(p)
    add_bracket_with_annotations(ax, 1, 2, y=y_box, stars=stars)

    ax.set_ylim(0, 12.5)

    # Save next to this script
    OUT_PDF = r"Y:\Members\MaxE\Thesis\Organized Figures\Chapter_4\Figure 4.16 Isabel 2\Figure_4.16_DistributionBoxplot.pdf"
    savefig(fig, OUT_PDF, dpi=300, transparent=True, bbox_inches=None)
    savefig(fig, Path(OUT_PDF).with_suffix(".png"), dpi=600, transparent=False, facecolor="white", bbox_inches=None)
    plt.show()

    plt.close(fig)

print(f"[SAVE] Written: {OUT_PDF}")
