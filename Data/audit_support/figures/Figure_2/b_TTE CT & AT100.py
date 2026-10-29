#!/usr/bin/env python3
# =============================================================================
# UPSTREAM DATA & ANALYSIS
# =============================================================================
# Experimental and simulation TTE source:
# - <EICHE_FISCHER_DATA_ROOT>/intermediate_analysis_data/combined/
#   figure_2/all_escape_margin.csv
# Upstream analysis: analysis/figure_2/Escape_Margin_Analysis.py
# =============================================================================
"""
TL;DR
-----
This script computes and plots time to escape (TTE) for CT and AT100
replicates using the canonical signed escape-margin table.

Detailed
--------
Inputs:
- Experimental and simulation rows from `all_escape_margin.csv`.
- Replicate exclusion/truncation rules from `experiment_filters.py`.

Outputs:
- A TTE comparison figure and the associated pairwise survival-style
  significance statistics.

How it works:
- Loads escape timing information per replicate and identifies the relevant
  sample/escape columns.
- Applies the shared exclusion and truncation logic so invalid or censored
  replicates are handled consistently.
- Builds replicate-level TTE tables and compares CT versus AT100 with the
  script’s significance logic.

Pipeline role:
- Figure 2 plotting script for escape timing.
"""

from __future__ import annotations

# =============================================================================
# DEPENDENCY OVERVIEW
# =============================================================================
# Files: canonical all_escape_margin.csv containing experimental and simulation
# signed escape-margin trajectories.
# Scripts: analysis/figure_2/Event_Time_Analysis.py, experiment_filters.py,
# strategy_colors.py, and figtools.py.
# Outputs: CT/AT100 TTE PDFs and pairwise comparison summaries.

# ---------------------------------------------------------------------
# EDITABLE FIGURE PARAMETERS
# ---------------------------------------------------------------------
# Width and height refer to the axes box: the physical x-axis and y-axis
# length. Margins are extra space for tick labels, axis labels, brackets,
# and other annotations.
FIGURE_AXES_WIDTH_MM = 17.5
FIGURE_AXES_HEIGHT_MM = 35.0
FIGURE_MARGINS_MM = (9.30, 1.24, 5.74, 3.28)  # left, right, bottom, top
TEXT_SIZE_PT = 7.0
TICK_SIZE_PT = 6.0
LEGEND_SIZE_PT = 5.0
LEGEND_VERTICAL_OFFSET_MM = 5.0
LEGEND_VERTICAL_OFFSET_DAYS = 10.0
DPI = 300
SHOW_FIGURE = True

import sys
import re
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import chi2

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import figtools as ft
from analysis.figure_2.Event_Time_Analysis import (
    ESCAPE_MARGIN_CSV,
    derive_tte,
    load_escape_margins,
)


OUT_DIR_EXTRA = ft.paper_figure_data_dir(__file__)
OUT_DIR_EXTRA.mkdir(parents=True, exist_ok=True)
pdf_path_extra = OUT_DIR_EXTRA / "Fig. 2 TTE CT & AT100.pdf"

STRATEGIES_KEEP = ["CT", "AT50", "AT100"]
YLIM = (0, 160)
EXPERIMENT_BAR_Y = (148, 156)
SIMULATION_COMPARISONS = (("CT", "AT50"), ("CT", "AT100"))
SIMULATION_BAR_Y = (132, 140)

# ---- Typography / strokes (match TTP) ----
ft.apply_nature_style(
    text_pt=TEXT_SIZE_PT,
    tick_pt=TICK_SIZE_PT,
    legend_pt=LEGEND_SIZE_PT,
    overrides={
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
    },
)

# Bracket + annotation styling (match TTP)
BRACKET_LW = 0.9
BRACKET_H_FRAC = 0.04
BRACKET_TOP_MARGIN_FRAC = 0.08
BRACKET_STEP_FRAC = 0.085
STARS_Y_OFFSET_H_FRAC = 0.45
STARS_FONTSIZE = 5
N_FONTSIZE = 5
GRID_ALPHA = 0.0
POINT_JITTER = 0.08
POINT_SIZE = 8
POINT_ALPHA = 0.85
EXPERIMENT_POINT_MARKER = "o"
SIMULATION_POINT_MARKER = "x"
SIMULATION_POINT_LINEWIDTH = 0.8
EXPERIMENT_X_OFFSET = -0.18
SIMULATION_X_OFFSET = 0.18
MEDIAN_COLOR = "#b000b5"
MEDIAN_LINE_HALF_WIDTH = 0.10
BOOTSTRAP_REPEATS = 10_000
BOOTSTRAP_SEED = 20260827
# -------------------------------------------------------------------


# ---------------------- STRATEGY COLORS IMPORT ----------------------
from strategy_colors import get_color
# -------------------------------------------------------------------


def load_simulation_tte(margins: pd.DataFrame | None = None) -> pd.DataFrame:
    if margins is None:
        margins = load_escape_margins(ESCAPE_MARGIN_CSV)
    out = derive_tte(
        margins,
        dataset_type="simulation",
        strategies=tuple(STRATEGIES_KEEP),
    )
    out = out[out["tte"].notna()].copy()
    print(f"[INFO] Simulation TTE: {len(out)} unique strategy/run rows loaded.")
    return out.loc[:, ["strategy", "tte", "escaped"]]


def prepare_tte_statistics(tte: pd.DataFrame) -> pd.DataFrame:
    """Restore truncation cutoffs as right-censor times for log-rank/KM use."""
    out = tte.copy()
    cutoff = out["excluded_reason"].str.extract(
        r"^no_escape_within_truncation_cutoff_([0-9.]+)$", expand=False
    )
    censor_mask = cutoff.notna() & out["tte"].isna()
    out.loc[censor_mask, "tte"] = np.minimum(
        out.loc[censor_mask, "last_observed_day"].to_numpy(float),
        cutoff.loc[censor_mask].to_numpy(float),
    )
    return out.loc[
        out["tte"].notna() & ~out["excluded_reason"].eq("excluded_by_experiment_filters")
    ].copy()


def p_to_stars(p: float) -> str:
    if not np.isfinite(p):
        return "n/a"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return "ns"


def kaplan_meier_median(durations: np.ndarray, events: np.ndarray) -> float:
    """Return the first time at which Kaplan--Meier survival is at most 0.5."""
    durations = np.asarray(durations, dtype=float)
    events = np.asarray(events, dtype=bool)
    valid = np.isfinite(durations)
    durations, events = durations[valid], events[valid]
    if durations.size == 0:
        return np.nan
    survival = 1.0
    for event_time in np.unique(durations[events]):
        at_risk = int(np.sum(durations >= event_time))
        observed_events = int(np.sum((durations == event_time) & events))
        if at_risk:
            survival *= 1.0 - observed_events / at_risk
        if survival <= 0.5:
            return float(event_time)
    return np.nan


def bootstrap_median_difference_ci(
    reference: np.ndarray,
    comparison: np.ndarray,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Return comparison-minus-reference median difference and 95% CI."""
    reference = np.asarray(reference, dtype=float)
    comparison = np.asarray(comparison, dtype=float)
    if reference.size == 0 or comparison.size == 0:
        return np.nan, np.nan, np.nan
    difference = float(np.median(comparison) - np.median(reference))
    reference_boot = rng.choice(reference, size=(BOOTSTRAP_REPEATS, reference.size), replace=True)
    comparison_boot = rng.choice(comparison, size=(BOOTSTRAP_REPEATS, comparison.size), replace=True)
    differences = np.median(comparison_boot, axis=1) - np.median(reference_boot, axis=1)
    low, high = np.quantile(differences, [0.025, 0.975])
    return difference, float(low), float(high)


def print_panel_tte_statistics(
    experimental: pd.DataFrame,
    simulation: pd.DataFrame,
) -> None:
    """Report censored time-to-event contrasts used in the plotted log-rank tests."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sources = {
        "Experiment": experimental.loc[:, ["strategy", "tte", "escaped"]],
        "Simulation": simulation.loc[:, ["strategy", "tte", "escaped"]],
    }
    print("[MANUSCRIPT STATS] TTE contrasts (log-rank; Kaplan--Meier median):")
    for source, table in sources.items():
        print(f"  {source}:")
        for reference_name, comparison_name in combinations(STRATEGIES_KEEP, 2):
            reference_rows = table.loc[table["strategy"] == reference_name]
            comparison_rows = table.loc[table["strategy"] == comparison_name]
            reference = reference_rows["tte"].to_numpy(float)
            comparison = comparison_rows["tte"].to_numpy(float)
            if reference.size < 2 or comparison.size < 2:
                print(f"    {reference_name} vs {comparison_name}: insufficient values")
                continue
            p_value = logrank_pvalue(
                reference, reference_rows["escaped"].to_numpy(bool),
                comparison, comparison_rows["escaped"].to_numpy(bool),
            )
            print(
                f"    {reference_name} vs {comparison_name} "
                f"({comparison_name} - {reference_name}): "
                f"n={reference.size}/{comparison.size}; "
                f"KM medians={kaplan_meier_median(reference, reference_rows['escaped']):.3g}/"
                f"{kaplan_meier_median(comparison, comparison_rows['escaped']):.3g} days; "
                f"log-rank p={p_value:.6g}"
            )


def is_significant(p: float) -> bool:
    return bool(np.isfinite(p) and p < 5e-2)


def logrank_pvalue(
    durations_a: np.ndarray,
    events_a: np.ndarray,
    durations_b: np.ndarray,
    events_b: np.ndarray,
) -> float:
    """
    Two-sample log-rank test p-value (chi-square with 1 df).
    durations_*: observed/censor times
    events_*: 1=event occurred, 0=right-censored
    """
    t_a = np.asarray(durations_a, dtype=float)
    t_b = np.asarray(durations_b, dtype=float)
    e_a = np.asarray(events_a, dtype=bool)
    e_b = np.asarray(events_b, dtype=bool)

    valid_a = np.isfinite(t_a)
    valid_b = np.isfinite(t_b)
    t_a = t_a[valid_a]
    t_b = t_b[valid_b]
    e_a = e_a[valid_a]
    e_b = e_b[valid_b]

    if t_a.size < 2 or t_b.size < 2:
        return np.nan

    event_times = np.unique(np.concatenate([t_a[e_a], t_b[e_b]]))
    if event_times.size == 0:
        return np.nan

    oe_sum = 0.0
    var_sum = 0.0

    for t in event_times:
        n1 = int(np.sum(t_a >= t))
        n2 = int(np.sum(t_b >= t))
        n = n1 + n2
        if n <= 1:
            continue

        d1 = int(np.sum((t_a == t) & e_a))
        d2 = int(np.sum((t_b == t) & e_b))
        d = d1 + d2
        if d == 0:
            continue

        e1 = d * (n1 / n)
        var = (n1 * n2 * d * (n - d)) / (n * n * (n - 1))
        oe_sum += (d1 - e1)
        var_sum += var

    if var_sum <= 0:
        return np.nan

    chisq = (oe_sum * oe_sum) / var_sum
    return float(chi2.sf(chisq, df=1))


def add_sig_bracket(
    ax,
    x1: float,
    x2: float,
    y: float,
    stars: str,
    h_frac: float = BRACKET_H_FRAC,
) -> None:
    """Draw a compact horizontal significance bar without end caps."""
    ax.hlines(y, x1, x2, lw=BRACKET_LW, color="black", clip_on=False)
    xm = (x1 + x2) / 2.0
    ax.text(xm, y + 2.0, stars,
            ha="center", va="center", fontsize=STARS_FONTSIZE, color="black")


def add_sig_bar(ax, x1: float, x2: float, y: float, stars: str) -> None:
    """Draw a compact horizontal significance bar without end caps."""
    ax.hlines(y, x1, x2, lw=BRACKET_LW, color="black", clip_on=False)
    ax.text(
        (x1 + x2) / 2.0,
        y + 2.0,
        stars,
        ha="center",
        va="center",
        fontsize=STARS_FONTSIZE,
        color="black",
    )


def plot_tte_boxplot(
    tte_df: pd.DataFrame,
    simulation_tte: pd.DataFrame,
    output_pdf: Path,
    label: str,
) -> None:
    """Plot TTE for selected strategies using observed escapes only."""
    present_strats = set(tte_df["strategy"]) | set(simulation_tte["strategy"])
    labels = [s for s in STRATEGIES_KEEP if s in present_strats]
    group_frames = [
        tte_df.loc[
            (tte_df["strategy"] == s) & tte_df["escaped"], ["tte", "escaped"]
        ].dropna(subset=["tte"]).copy()
        for s in labels
    ]
    groups = [g["tte"].to_numpy(dtype=float) for g in group_frames]
    simulation_groups = [
        simulation_tte.loc[
            (simulation_tte["strategy"] == s) & simulation_tte["escaped"], "tte"
        ].dropna()
        for s in labels
    ]
    experimental_statistics = [
        tte_df.loc[tte_df["strategy"] == s, ["tte", "escaped"]].dropna(subset=["tte"])
        for s in labels
    ]
    simulation_statistics = [
        simulation_tte.loc[
            simulation_tte["strategy"] == s, ["tte", "escaped"]
        ].dropna(subset=["tte"])
        for s in labels
    ]
    labels_with_n = [s for s in labels]

    if len(labels) < 2:
        print("[WARN] Need at least two strategies in STRATEGIES_KEEP with data.")
        return

    # Log rows excluded from boxplot (tte is NaN after exclusions/truncations)
    excluded_from_plot = tte_df[tte_df["tte"].isna()].copy()
    if not excluded_from_plot.empty:
        print("\n[INFO] Replicates excluded from boxplot by experiment filters:")
        for strat in labels:
            sub = excluded_from_plot[excluded_from_plot["strategy"] == strat]
            if sub.empty:
                continue
            print(f"  {strat}:")
            for _, r in sub.iterrows():
                reason = r.get("excluded_reason", "")
                print(f"    - {r['sample']}  ({reason})")
        print("")

    # Pairwise log-rank tests (handles right-censoring) for all strategy pairs.
    pairwise_results = []
    for i, j in combinations(range(len(labels)), 2):
        g_i = experimental_statistics[i]
        g_j = experimental_statistics[j]

        if len(g_i) >= 2 and len(g_j) >= 2:
            p = logrank_pvalue(
                durations_a=g_i["tte"].to_numpy(dtype=float),
                events_a=g_i["escaped"].to_numpy(dtype=bool),
                durations_b=g_j["tte"].to_numpy(dtype=float),
                events_b=g_j["escaped"].to_numpy(dtype=bool),
            )
            stars = p_to_stars(float(p)) if np.isfinite(p) else "n/a"
        else:
            p = np.nan
            stars = "n/a"

        pairwise_results.append(
            {
                "i": i,
                "j": j,
                "label_i": labels[i],
                "label_j": labels[j],
                "p": p,
                "stars": stars,
                "n_text": f"n={len(g_i)}/{len(g_j)}",
            }
        )

    print(f"[STATS] Pairwise log-rank tests ({label}):")
    if pairwise_results:
        for r in pairwise_results:
            p_str = f"{r['p']:.6g}" if np.isfinite(r["p"]) else "n/a"
            print(
                f"  {r['label_i']} vs {r['label_j']}: "
                f"p={p_str}, stars={r['stars']}, {r['n_text']}"
            )
    else:
        print("  No valid pairwise comparisons.")

    simulation_pairwise_results = []
    label_indices = {strategy: index for index, strategy in enumerate(labels)}
    for reference_name, comparison_name in SIMULATION_COMPARISONS:
        if reference_name not in label_indices or comparison_name not in label_indices:
            continue
        sim_i = simulation_statistics[label_indices[reference_name]]
        sim_j = simulation_statistics[label_indices[comparison_name]]
        p = (
            logrank_pvalue(
                sim_i["tte"].to_numpy(float), sim_i["escaped"].to_numpy(bool),
                sim_j["tte"].to_numpy(float), sim_j["escaped"].to_numpy(bool),
            )
            if len(sim_i) >= 2 and len(sim_j) >= 2
            else np.nan
        )
        simulation_pairwise_results.append(
            {
                "i": label_indices[reference_name],
                "j": label_indices[comparison_name],
                "label_i": reference_name,
                "label_j": comparison_name,
                "p": p,
                "stars": p_to_stars(p),
            }
        )

    print(f"[STATS] Simulation log-rank tests ({label}; requested contrasts):")
    for result in simulation_pairwise_results:
        p_text = f"{result['p']:.6g}" if np.isfinite(result["p"]) else "n/a"
        print(f"  {result['label_i']} vs {result['label_j']}: p={p_text}")

    # Summary counts
    for strat in labels:
        sub = tte_df[tte_df["strategy"] == strat]
        escaped = int(sub["escaped"].sum())
        total = int(len(sub))
        in_plot = int(sub["tte"].notna().sum())
        print(f"[INFO] {label} {strat}: escaped {escaped}/{total}, in_boxplot {in_plot}/{total}.")

    fig, ax = ft.new_figure(
        width=FIGURE_AXES_WIDTH_MM,
        height_mm=FIGURE_AXES_HEIGHT_MM,
        margins=FIGURE_MARGINS_MM,
        dpi=DPI,
        apply_style=False,
    )
    x_positions = np.arange(1, len(labels) + 1, dtype=float)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels_with_n)

    # Color x tick labels to match
    ticklabels = ax.get_xticklabels()
    for tl, strat in zip(ticklabels, labels):
        tl.set_color(get_color(strat, "#888888"))

    ax.set_ylabel("Time to escape, TTE [days]")
    ax.set_ylim(*YLIM)
    ax.grid(True, axis="y", alpha=GRID_ALPHA)
    ax.legend(
        handles=[
            Line2D(
                [], [], linestyle="none", marker=EXPERIMENT_POINT_MARKER,
                markerfacecolor="#888888", markeredgecolor="black",
                markeredgewidth=0.3, markersize=3.5, label="Exp",
            ),
            Line2D(
                [], [], linestyle="none", marker=SIMULATION_POINT_MARKER,
                markerfacecolor="#888888", markeredgecolor="black",
                markeredgewidth=SIMULATION_POINT_LINEWIDTH, markersize=3.5, label="Sim",
            ),
            Line2D(
                [], [], color=MEDIAN_COLOR, linewidth=1.1, label="KM median",
            ),
        ],
        loc="upper left",
        bbox_to_anchor=(
            0,
            1
            - LEGEND_VERTICAL_OFFSET_MM / FIGURE_AXES_HEIGHT_MM
            - LEGEND_VERTICAL_OFFSET_DAYS / (YLIM[1] - YLIM[0]),
        ),
        frameon=False,
        handlelength=1.0,
        handletextpad=0.3,
        borderaxespad=0.2,
        fontsize=LEGEND_SIZE_PT,
    )

    ax.tick_params(axis="x", labelsize=6, pad=1)
    ax.tick_params(axis="y", pad=1)

    ax.text(
        x_positions[0] - 0.55,
        -0.105,
        "n =",
        transform=ax.get_xaxis_transform(),
        ha="right",
        va="top",
        fontsize=N_FONTSIZE,
        clip_on=False,
    )
    for xpos, experimental_stats, simulation_stats in zip(
        x_positions, experimental_statistics, simulation_statistics
    ):
        ax.text(
            xpos,
            -0.105,
            f"({len(experimental_stats)}, {len(simulation_stats)})",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=N_FONTSIZE,
            clip_on=False,
        )

    # Overlay individual replicate values with slight x-jitter.
    rng = np.random.default_rng(7)
    for idx, vals in enumerate(groups, start=1):
        vals_arr = np.asarray(vals, dtype=float)
        strat = labels[idx - 1]
        col = get_color(strat, "#888888")
        if vals_arr.size > 0:
            x = idx + EXPERIMENT_X_OFFSET + rng.uniform(-POINT_JITTER, POINT_JITTER, size=vals_arr.size)
            ax.scatter(
                x,
                vals_arr,
                s=POINT_SIZE,
                c=col,
                alpha=POINT_ALPHA,
                edgecolors="black",
                linewidths=0.3,
                marker=EXPERIMENT_POINT_MARKER,
                zorder=2,
            )
            km_median = kaplan_meier_median(
                experimental_statistics[idx - 1]["tte"].to_numpy(float),
                experimental_statistics[idx - 1]["escaped"].to_numpy(bool),
            )
            if np.isfinite(km_median):
                ax.hlines(
                    km_median,
                    idx + EXPERIMENT_X_OFFSET - MEDIAN_LINE_HALF_WIDTH,
                    idx + EXPERIMENT_X_OFFSET + MEDIAN_LINE_HALF_WIDTH,
                    colors=MEDIAN_COLOR,
                    linewidth=1.1,
                    zorder=4,
                )

        sim_vals_arr = np.asarray(simulation_groups[idx - 1], dtype=float)
        if sim_vals_arr.size > 0:
            x_sim = idx + SIMULATION_X_OFFSET + rng.uniform(-POINT_JITTER, POINT_JITTER, size=sim_vals_arr.size)
            ax.scatter(
                x_sim,
                sim_vals_arr,
                s=POINT_SIZE,
                c=col,
                alpha=POINT_ALPHA,
                edgecolors="black",
                linewidths=SIMULATION_POINT_LINEWIDTH,
                marker=SIMULATION_POINT_MARKER,
                zorder=2,
            )
            km_median = kaplan_meier_median(
                simulation_statistics[idx - 1]["tte"].to_numpy(float),
                simulation_statistics[idx - 1]["escaped"].to_numpy(bool),
            )
            if np.isfinite(km_median):
                ax.hlines(
                    km_median,
                    idx + SIMULATION_X_OFFSET - MEDIAN_LINE_HALF_WIDTH,
                    idx + SIMULATION_X_OFFSET + MEDIAN_LINE_HALF_WIDTH,
                    colors=MEDIAN_COLOR,
                    linewidth=1.1,
                    zorder=4,
                )

    # Bracket placement for significant pairwise comparisons only.
    significant_results = [r for r in pairwise_results if is_significant(r["p"])]
    if significant_results:
        for y_bracket, r in zip(EXPERIMENT_BAR_Y, significant_results):
            add_sig_bracket(
                ax,
                r["i"] + 1 + EXPERIMENT_X_OFFSET,
                r["j"] + 1 + EXPERIMENT_X_OFFSET,
                y_bracket,
                stars=r["stars"],
            )

    for y, result in zip(SIMULATION_BAR_Y, simulation_pairwise_results):
        if is_significant(result["p"]):
            add_sig_bar(
                ax,
                result["i"] + 1 + SIMULATION_X_OFFSET,
                result["j"] + 1 + SIMULATION_X_OFFSET,
                y,
                result["stars"],
            )

    axes_w_mm, axes_h_mm = ft.axes_size_mm(ax)
    print(f"[FIGURE] Axes box: {axes_w_mm:.2f} x {axes_h_mm:.2f} mm")
    ft.savefig(fig, output_pdf)

    if SHOW_FIGURE and plt.get_backend().lower() != "agg":
        plt.show()
    plt.close(fig)

    print(f"[OK] Saved: {output_pdf}")


def main() -> None:
    margins = load_escape_margins(ESCAPE_MARGIN_CSV)
    tte_df = derive_tte(
        margins,
        dataset_type="experimental",
        strategies=tuple(STRATEGIES_KEEP),
        apply_truncation=True,
    )
    tte_df = prepare_tte_statistics(tte_df)
    simulation_tte = load_simulation_tte(margins)

    if tte_df.empty:
        print(f"[WARN] No replicates found for STRATEGIES_KEEP={STRATEGIES_KEEP}.")
        return

    # Print-only manuscript statistics; plotting behavior remains unchanged.
    print_panel_tte_statistics(tte_df, simulation_tte)
    plot_tte_boxplot(tte_df, simulation_tte, pdf_path_extra, label="filtered")


if __name__ == "__main__":
    main()
