#!/usr/bin/env python3
"""
Figure: MAE comparison (density vs no-density) by strategy.

Input: a summary CSV created by export_mae_summary_for_thesis.py with columns:
    mode, replicate_id, strategy, mae_density, mae_no_density

This script:
  - builds panel (a): all strategies combined (density vs no-density)
  - builds panel (b): per-strategy pairs along the x-axis
  - performs Wilcoxon tests (paired) where possible, otherwise Mann–Whitney
  - always draws comparison brackets with stars or 'ns' and the n number

Adheres to dissertation figure standards:
  - with use_style("figstyle.mplstyle")
  - set_size(ax, width="thesis", ratio=...)
  - no tight_layout(), spacing via subplots_adjust
  - axes: no top/right spines, ticks outward, Arial
  - colours from figtools.therapy_color
"""

from __future__ import annotations

import csv
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, mannwhitneyu

from figtools import use_style, set_size, savefig, therapy_color  # type: ignore

# ---------------------------------------------------------------------
# USER SETTINGS (EDIT THESE PATHS PER FIGURE / MODE)
# ---------------------------------------------------------------------

# Y-axis limits for MAE
YMIN, YMAX = 0.0, 0.32

# Median line colour
MEDIAN_COLOR = "#FF991C" # orange

# Minimum number of values per side to include a strategy in panel (b)
MIN_VALUES_PER_SIDE = 1

# Order in which strategies appear on the right panel
DESIRED_ORDER = ["NT", "CT", "Pulse", "AT50", "AT100", "ATAgent1", "ATAgent2", "Dens"]

# Pretty labels
DISPLAY_LABEL = {
    "NT": "NT",
    "CT": "CT",
    "Pulse": "Pulse",
    "AT50": "AT50",
    "AT100": "AT100",
    "ATAgent1": "KPP-Agent",  # Agent1 = KPP
    "ATAgent2": "LV-Agent",   # Agent2 = LV
    "Dens": "AT100+",         # density-based schedule
}

# Colours from global therapy palette
STRAT_COLOR = {
    "NT":       therapy_color("NT"),
    "CT":       therapy_color("CT"),
    "Pulse":    therapy_color("Pulse"),
    "AT50":     therapy_color("AT50"),
    "AT100":    therapy_color("AT100"),
    "ATAgent1": therapy_color("KPP-Agent"),
    "ATAgent2": therapy_color("LV-Agent"),
    "Dens":     therapy_color("AT100+"),
}


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def significance_stars(p: float | None) -> str:
    """Return '***', '**', '*', or 'ns' based on p-value (None -> 'ns')."""
    if p is None or not np.isfinite(p):
        return "ns"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def load_summary(csv_path: Path):
    """
    Load summary CSV and return:
        by_strategy_pairs: strat -> list of (dens, nod)
        by_strategy_vals:  strat -> {'density': [...], 'no_density': [...]}
    """
    by_strategy_pairs: dict[str, list[tuple[float, float]]] = defaultdict(list)
    by_strategy_vals: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"density": [], "no_density": []}
    )

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            strat = row["strategy"]
            try:
                d = float(row["mae_density"])
                n = float(row["mae_no_density"])
            except (KeyError, ValueError):
                continue

            by_strategy_pairs[strat].append((d, n))
            by_strategy_vals[strat]["density"].append(d)
            by_strategy_vals[strat]["no_density"].append(n)

    return by_strategy_pairs, by_strategy_vals


def compute_p_values(by_strategy_pairs: dict[str, list[tuple[float, float]]]):
    """
    Compute per-strategy and combined p-values (paired if possible).
    Returns:
      strat_pvals: strat -> p
      combined_pval: float | None
    """
    strat_pvals: dict[str, float] = {}

    valid_strats = []
    for strat, pairs in by_strategy_pairs.items():
        if len(pairs) < MIN_VALUES_PER_SIDE:
            continue

        dens = np.array([p[0] for p in pairs], dtype=float)
        nod  = np.array([p[1] for p in pairs], dtype=float)

        # Paired test where possible
        try:
            stat, p = wilcoxon(dens, nod, alternative="two-sided", zero_method="wilcox")
            test_name = "Wilcoxon signed-rank (paired)"
        except ValueError:
            stat, p = mannwhitneyu(dens, nod, alternative="two-sided")
            test_name = "Mann–Whitney U (unpaired fallback)"

        strat_pvals[strat] = p
        valid_strats.append(strat)

        print(f"[{MODE_LABEL}] {strat}: n={len(dens)}  {test_name}  "
              f"stat={stat:.4g}  p={p:.3g}  med_d={np.median(dens):.3f}  med_n={np.median(nod):.3f}")

    # Combined test across all valid strategies
    all_pairs = []
    for strat in valid_strats:
        all_pairs.extend(by_strategy_pairs[strat])

    if len(all_pairs) < 2:
        print(f"[{MODE_LABEL}] Combined: not enough pairs.")
        return strat_pvals, None

    dens_all = np.array([p[0] for p in all_pairs], dtype=float)
    nod_all  = np.array([p[1] for p in all_pairs], dtype=float)

    try:
        stat, p_comb = wilcoxon(dens_all, nod_all, alternative="two-sided", zero_method="wilcox")
        name = "Wilcoxon signed-rank (paired)"
    except ValueError:
        stat, p_comb = mannwhitneyu(dens_all, nod_all, alternative="two-sided")
        name = "Mann–Whitney U (unpaired fallback)"

    print(f"[{MODE_LABEL}] COMBINED: n={len(dens_all)}  {name}  "
          f"stat={stat:.4g}  p={p_comb:.3g}  med_d={np.median(dens_all):.3f}  med_n={np.median(nod_all):.3f}")

    return strat_pvals, p_comb


# ---------------------------------------------------------------------
# Main plotting
# ---------------------------------------------------------------------

def main():
    print(f"[{MODE_LABEL}] Loading summary from {SUMMARY_CSV}")
    by_strategy_pairs, by_strategy_vals = load_summary(SUMMARY_CSV)

    # Strategies that actually have data
    present_strats = [
        s for s in DESIRED_ORDER
        if s in by_strategy_vals
        and len(by_strategy_vals[s]["density"]) >= MIN_VALUES_PER_SIDE
        and len(by_strategy_vals[s]["no_density"]) >= MIN_VALUES_PER_SIDE
    ]
    if not present_strats:
        print(f"[{MODE_LABEL}] No strategies with sufficient data.")
        return

    # Compute p-values
    strat_pvals, combined_pval = compute_p_values(by_strategy_pairs)

    # Prepare combined arrays for panel (a)
    all_d = [v for s in present_strats for v in by_strategy_vals[s]["density"]]
    all_n = [v for s in present_strats for v in by_strategy_vals[s]["no_density"]]

    with use_style("figstyle.mplstyle"):
        plt.rcParams["font.family"] = "Arial"

        # Two panels: left (overall), right (per-strategy)
        fig, (ax_all, ax_strat) = plt.subplots(
            1, 2,
            sharey=True,
            gridspec_kw={"width_ratios": [1, 4]},
        )

        # Thesis width, slim height
        set_size(ax_all, width="thesis", ratio=0.45)

        # Axes cosmetics
        for ax in (ax_all, ax_strat):
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            ax.tick_params(direction="out")

        # ---------------- Panel (a): all strategies combined ----------------
        if all_d and all_n:
            data_all = [all_d, all_n]
            pos_all = [1, 2]
            bp_all = ax_all.boxplot(
                data_all,
                positions=pos_all,
                widths=0.6,
                patch_artist=True,
                tick_labels=["Density", "No density"],
                showfliers=False,
            )

            # Boxes grey with black edges, medians yellow, whiskers/caps black
            for box in bp_all["boxes"]:
                box.set(facecolor="#DDDDDD", edgecolor="black", linewidth=1.0)
            for item in bp_all["whiskers"] + bp_all["caps"]:
                item.set(color="black", linewidth=1.0)
            for med in bp_all["medians"]:
                med.set(color=MEDIAN_COLOR, linewidth=1.5, zorder=2)
                # shift median line slightly to the left without changing its width
                x = med.get_xdata()
                if len(x) >= 2:
                    x1, x2 = x[0], x[-1]
                    shift = 0.0125 * (x2 - x1)  # ~3% of width to the left
                    med.set_xdata([x1 - shift, x2 - shift])


            # No title here
            ax_all.set_ylabel("MAE")
            ax_all.set_ylim(YMIN, YMAX)
            ax_all.grid(True, axis="y", linestyle=":", linewidth=0.8, alpha=0.7)

            # Significance annotation
            stars = significance_stars(combined_pval)
            x1, x2 = pos_all
            y = YMAX * 0.92
            ax_all.plot(
                [x1, x1, x2, x2],
                [y - 0.01, y, y, y - 0.01],
                lw=1.0,
                c="k",
            )
            ax_all.text(
                (x1 + x2) / 2.0,
                y + 0.005,
                stars,
                ha="center",
                va="bottom",
            )
            n_all = len(all_d)  # same as len(all_n)
            ax_all.text(
                (x1 + x2) / 2.0,
                y - 0.015,  # n number down or up here
                f"n = {n_all}",
                ha="center",
                va="top",
                fontsize=9,
            )

        # ---------------- Panel (b): per-strategy pairs ----------------
        base_gap = 2.4
        x_ticks = []
        x_ticklabels = []

        for i, strat in enumerate(present_strats):
            dens_vals = by_strategy_vals[strat]["density"]
            nod_vals  = by_strategy_vals[strat]["no_density"]
            if len(dens_vals) < MIN_VALUES_PER_SIDE or len(nod_vals) < MIN_VALUES_PER_SIDE:
                continue

            x1 = 1 + i * base_gap
            x2 = x1 + 0.9
            positions = [x1, x2]

            bp = ax_strat.boxplot(
                [dens_vals, nod_vals],
                positions=positions,
                widths=0.7,
                patch_artist=True,
                showfliers=False,
            )

            # Colour boxes in strategy colour, outlines black; medians yellow
            color = STRAT_COLOR.get(strat, "black")
            for box in bp["boxes"]:
                box.set(edgecolor="black", facecolor=color, linewidth=1.0)
            for item in bp["whiskers"] + bp["caps"]:
                item.set(color="black", linewidth=1.0)
            for med in bp["medians"]:
                med.set(color=MEDIAN_COLOR, linewidth=1.5, zorder=2)
                x = med.get_xdata()
                if len(x) >= 2:
                    x1, x2 = x[0], x[-1]
                    shift = 0.0125 * (x2 - x1)
                    med.set_xdata([x1 - shift, x2 - shift])


            # Strategy label
            x_center = (x1 + x2) / 2.0
            x_ticks.append(x_center)
            x_ticklabels.append(DISPLAY_LABEL.get(strat, strat))

            # Significance annotation (always drawn, 'ns' if not significant)
            dens_arr = np.array(dens_vals, dtype=float)
            p_val = strat_pvals.get(strat, None)
            stars = significance_stars(p_val)
            y = YMAX * 0.92
            ax_strat.plot(
                [x1, x1, x2, x2],
                [y - 0.01, y, y, y - 0.01],
                lw=1.0,
                c="k",
            )
            ax_strat.text(
                x_center,
                y + 0.005,
                stars,
                ha="center",
                va="bottom",
            )
            n_val = len(dens_arr)
            ax_strat.text(
                x_center,
                y - 0.015,  # n number down or up here
                f"n = {n_val}",
                ha="center",
                va="top",
                fontsize=9,
            )

        ax_strat.set_ylim(YMIN, YMAX)
        ax_strat.grid(True, axis="y", linestyle=":", linewidth=0.8, alpha=0.7)
        ax_strat.set_xticks(x_ticks)
        ax_strat.set_xticklabels(x_ticklabels, rotation=30, ha="right")
        ax_strat.set_ylabel("MAE")

        # Panel labels
        ax_all.text(
            -0.25, 1.05, "a",
            transform=ax_all.transAxes,
            fontweight="bold",
            va="bottom",
            ha="left",
        )
        ax_strat.text(
            -0.1, 1.05, "b",
            transform=ax_strat.transAxes,
            fontweight="bold",
            va="bottom",
            ha="left",
        )

        # Thesis-style layout
        fig.subplots_adjust(left=0.12, right=0.98, wspace=0.30)

        OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
        savefig(fig, OUT_PDF, dpi=300, transparent=True)
        plt.show()
        plt.close(fig)
        print(f"[{MODE_LABEL}] Saved PDF: {OUT_PDF}")


if __name__ == "__main__":
    for mode in ["res", "sen", "total"]:
        if mode == "res":
            SUMMARY_CSV = Path(r"Y:\Members\MaxE\Thesis\Organized Figures\Figure 3.14 KPP\mae_summary_res.csv")
            OUT_PDF = Path(r"Y:\Members\MaxE\Thesis\Organized Figures\Figure 3.14 KPP\Figure_MAE_res.pdf")
        elif mode == "sen":
            SUMMARY_CSV = Path(r"Y:\Members\MaxE\Thesis\Organized Figures\Figure 3.14 KPP\mae_summary_sen.csv")
            OUT_PDF = Path(r"Y:\Members\MaxE\Thesis\Organized Figures\Figure 3.14 KPP\Figure_MAE_sen.pdf")
        else:  # total
            SUMMARY_CSV = Path(r"Y:\Members\MaxE\Thesis\Organized Figures\Figure 3.14 KPP\mae_summary_total.csv")
            OUT_PDF = Path(r"Y:\Members\MaxE\Thesis\Organized Figures\Figure 3.14 KPP\Figure_MAE_total.pdf")

        MODE_LABEL = mode
        main()