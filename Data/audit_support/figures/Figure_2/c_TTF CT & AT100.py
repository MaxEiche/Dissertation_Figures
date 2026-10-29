#!/usr/bin/env python3
# =============================================================================
# UPSTREAM DATA & ANALYSIS
# =============================================================================
# Data inputs: configured by the Path(...) constants, CLI defaults, and imports below.
# Upstream scripts: determined by the imports and pipeline calls below.
# =============================================================================
"""
TL;DR
-----
This script computes and plots time to progression (TTP) for CT and AT100
replicates from replicate-level total-burden trajectories.

Detailed
--------
Inputs:
- Experimental `*_total_poisson_padded.h5` and simulation `*_simulation.h5`
  point-cloud trajectories.
- Replicate exclusion/truncation rules from `experiment_filters.py`.

Outputs:
- A TTP comparison figure and the associated strategy-level summary statistics.

How it works:
- Loads per-day total cell counts, normalizes each replicate to its day-1 total
  burden, and applies the shared replicate exclusion/truncation rules.
- Defines TTP as the first day the normalized total burden crosses the
  progression threshold.
- Compares CT and AT100 replicate TTP distributions and annotates the plot with
  the resulting significance statistics.

Pipeline role:
- Figure 2 plotting script for progression timing.
"""

from __future__ import annotations

# =============================================================================
# DEPENDENCY OVERVIEW
# =============================================================================
# Files: experimental *_total_poisson_padded.h5 and simulation *_simulation.h5
# point clouds.
# Scripts: analysis/figure_2/Event_Time_Analysis.py, experiment_filters.py,
# strategy_colors.py, and figtools.py.
# Outputs: CT/AT100 TTP PDFs and statistical summaries.

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
DPI = 300
SHOW_FIGURE = True

import sys
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import figtools as ft
from experiment_filters import EXCLUDE_SAMPLES
from analysis.figure_2.Event_Time_Analysis import (
    derive_progression,
    load_population_counts,
)


# ---------------------------------------------------------------------
# USER CONFIG
# ---------------------------------------------------------------------
OUT_DIR_EXTRA = ft.paper_figure_data_dir(__file__)
OUT_DIR_EXTRA.mkdir(parents=True, exist_ok=True)
OUT_PDF_EXTRA = OUT_DIR_EXTRA / "Fig. 2 TTP CT & AT100.pdf"
STRATEGIES_KEEP = ["CT", "AT50", "AT100"]
PROGRESSION_THRESHOLD = 0.5  # failure when resistant cells reach 50% of initial total


# ---------------------------------------------------------------------
# FIGURE STYLE
# ---------------------------------------------------------------------
YLIM = (0, 160)
EXPERIMENT_BAR_Y = (148, 156)
SIMULATION_COMPARISONS = (("CT", "AT50"), ("CT", "AT100"))
SIMULATION_BAR_Y = (132, 140)

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


# ---------------------------------------------------------------------
# IMPORT SHARED STRATEGY COLORS
# ---------------------------------------------------------------------
from strategy_colors import get_color


def collect_all_samples() -> pd.DataFrame:
    return load_population_counts(
        dataset_type="experimental",
        strategies=tuple(STRATEGIES_KEEP),
    )


def prepare_progression_statistics(
    counts: pd.DataFrame, *, apply_truncation: bool
) -> pd.DataFrame:
    """Return event times with non-progressors retained as right-censored rows."""
    out = derive_progression(
        counts, apply_truncation=apply_truncation, threshold=PROGRESSION_THRESHOLD,
    )
    observed = counts.copy()
    if apply_truncation:
        rules = observed["sample"].map(
            lambda sample: EXCLUDE_SAMPLES.get(str(sample), "not_listed")
        )
        keep = rules.ne(None)
        cutoff = pd.to_numeric(
            rules.where(rules.ne("not_listed")), errors="coerce"
        )
        keep &= cutoff.isna() | observed["day"].le(cutoff)
        observed = observed.loc[keep]
    keys = ["dataset_type", "source", "plate", "sample", "strategy"]
    last_observed = (
        observed.groupby(keys, as_index=False)["day"]
        .max()
        .rename(columns={"day": "last_observed_day"})
    )
    out = out.merge(last_observed, on=keys, how="left")
    out["ttp"] = out["ttp"].fillna(out["last_observed_day"])
    return out.loc[out["ttp"].notna()].copy()


def load_simulation_ttp() -> pd.DataFrame:
    counts = load_population_counts(
        dataset_type="simulation",
        strategies=tuple(STRATEGIES_KEEP),
    )
    out = prepare_progression_statistics(counts, apply_truncation=False)
    print(f"[INFO] Simulation TTP: {len(out)} unique strategy/run rows loaded.")
    return out


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


def print_panel_ttf_statistics(
    experimental: pd.DataFrame,
    simulation: pd.DataFrame,
) -> None:
    """Report censored time-to-event contrasts used in the plotted log-rank tests."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sources = {
        "Experiment": experimental.loc[:, ["strategy", "ttp", "progressed"]],
        "Simulation": simulation.loc[:, ["strategy", "ttp", "progressed"]],
    }
    print("[MANUSCRIPT STATS] TTF contrasts (log-rank; Kaplan--Meier median):")
    for source, table in sources.items():
        print(f"  {source}:")
        for reference_name, comparison_name in combinations(STRATEGIES_KEEP, 2):
            reference_rows = table.loc[table["strategy"] == reference_name]
            comparison_rows = table.loc[table["strategy"] == comparison_name]
            reference = reference_rows["ttp"].to_numpy(float)
            comparison = comparison_rows["ttp"].to_numpy(float)
            if reference.size < 2 or comparison.size < 2:
                print(f"    {reference_name} vs {comparison_name}: insufficient values")
                continue
            p_value = logrank_pvalue(
                reference, reference_rows["progressed"].to_numpy(bool),
                comparison, comparison_rows["progressed"].to_numpy(bool),
            )
            print(
                f"    {reference_name} vs {comparison_name} "
                f"({comparison_name} - {reference_name}): "
                f"n={reference.size}/{comparison.size}; "
                f"KM medians={kaplan_meier_median(reference, reference_rows['progressed']):.3g}/"
                f"{kaplan_meier_median(comparison, comparison_rows['progressed']):.3g} days; "
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
    """Two-sample log-rank p-value for observed and right-censored event times."""
    t_a = np.asarray(durations_a, dtype=float)
    t_b = np.asarray(durations_b, dtype=float)
    e_a = np.asarray(events_a, dtype=bool)
    e_b = np.asarray(events_b, dtype=bool)
    valid_a, valid_b = np.isfinite(t_a), np.isfinite(t_b)
    t_a, e_a = t_a[valid_a], e_a[valid_a]
    t_b, e_b = t_b[valid_b], e_b[valid_b]
    if t_a.size < 2 or t_b.size < 2:
        return np.nan
    event_times = np.unique(np.concatenate([t_a[e_a], t_b[e_b]]))
    if event_times.size == 0:
        return np.nan
    oe_sum = 0.0
    var_sum = 0.0
    for event_time in event_times:
        n_a, n_b = int(np.sum(t_a >= event_time)), int(np.sum(t_b >= event_time))
        n = n_a + n_b
        if n <= 1:
            continue
        d_a = int(np.sum((t_a == event_time) & e_a))
        d_b = int(np.sum((t_b == event_time) & e_b))
        d = d_a + d_b
        if d == 0:
            continue
        expected_a = d * n_a / n
        variance = n_a * n_b * d * (n - d) / (n * n * (n - 1))
        oe_sum += d_a - expected_a
        var_sum += variance
    return np.nan if var_sum <= 0 else float(chi2.sf((oe_sum * oe_sum) / var_sum, df=1))


def add_sig_bracket(
    ax,
    x1: float,
    x2: float,
    y: float,
    stars: str,
    h_frac: float = BRACKET_H_FRAC,
) -> None:
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


# ---------------------------------------------------------------------
# PLOT
# ---------------------------------------------------------------------
def plot_boxplot(
    ttp: pd.DataFrame,
    simulation_ttp: pd.DataFrame,
    output_pdf: Path,
    label: str,
) -> None:
    # Build groups dynamically from STRATEGIES_KEEP
    present_strats = set(ttp["strategy"]) | set(simulation_ttp["strategy"])
    labels = [s for s in STRATEGIES_KEEP if s in present_strats]
    groups = [
        ttp.loc[
            (ttp["strategy"] == s) & ttp["progressed"], "ttp"
        ].dropna()
        for s in labels
    ]
    simulation_groups = [
        simulation_ttp.loc[
            (simulation_ttp["strategy"] == s) & simulation_ttp["progressed"], "ttp"
        ].dropna()
        for s in labels
    ]
    experimental_statistics = [
        ttp.loc[ttp["strategy"] == s, ["ttp", "progressed"]].dropna(subset=["ttp"])
        for s in labels
    ]
    simulation_statistics = [
        simulation_ttp.loc[
            simulation_ttp["strategy"] == s, ["ttp", "progressed"]
        ].dropna(subset=["ttp"])
        for s in labels
    ]
    labels_with_n = [s for s in labels]

    # Print simple sample-size summary for all groups
    print(f"[STATS] TTP group sizes ({label}):")
    for s, vals, sim_vals in zip(labels, groups, simulation_groups):
        print(f"  {s}: experimental n={len(vals)}, simulation n={len(sim_vals)}")

    # Pairwise Mann–Whitney U for all strategy pairs
    pairwise_results = []
    for i, j in combinations(range(len(labels)), 2):
        vals_i = experimental_statistics[i]
        vals_j = experimental_statistics[j]

        if len(vals_i) >= 2 and len(vals_j) >= 2:
            p = logrank_pvalue(
                vals_i["ttp"].to_numpy(float), vals_i["progressed"].to_numpy(bool),
                vals_j["ttp"].to_numpy(float), vals_j["progressed"].to_numpy(bool),
            )
            stars = p_to_stars(float(p)) if np.isfinite(p) else "n/a"
        else:
            p = np.nan
            stars = "n/a"

        n_text = f"n={len(vals_i)}/{len(vals_j)}"

        pairwise_results.append(
            {
                "i": i,
                "j": j,
                "label_i": labels[i],
                "label_j": labels[j],
                "p": p,
                "stars": stars,
                "n_text": n_text,
            }
        )

    print("[STATS] Pairwise log-rank tests:")
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
        vals_i = simulation_statistics[label_indices[reference_name]]
        vals_j = simulation_statistics[label_indices[comparison_name]]
        p = (
            logrank_pvalue(
                vals_i["ttp"].to_numpy(float), vals_i["progressed"].to_numpy(bool),
                vals_j["ttp"].to_numpy(float), vals_j["progressed"].to_numpy(bool),
            )
            if len(vals_i) >= 2 and len(vals_j) >= 2
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

    ticklabels = ax.get_xticklabels()
    for tl, strat in zip(ticklabels, labels):
        tl.set_color(get_color(strat, "#888888"))

    ax.set_ylabel("Time to failure, TTF [days]")
    ax.set_ylim(*YLIM)
    ax.grid(True, axis="y", alpha=GRID_ALPHA)
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
                experimental_statistics[idx - 1]["ttp"].to_numpy(float),
                experimental_statistics[idx - 1]["progressed"].to_numpy(bool),
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
                simulation_statistics[idx - 1]["ttp"].to_numpy(float),
                simulation_statistics[idx - 1]["progressed"].to_numpy(bool),
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

    # Draw pairwise brackets only for statistically significant comparisons
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
    print(f"[OK] Saved: {output_pdf}")

    if SHOW_FIGURE and plt.get_backend().lower() != "agg":
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main() -> None:
    counts = collect_all_samples()
    if counts.empty:
        print("[ERROR] No samples loaded. Check paths / plate folders.")
        return
    ttp = prepare_progression_statistics(counts, apply_truncation=True)
    simulation_ttp = load_simulation_ttp()

    # Print-only manuscript statistics; plotting behavior remains unchanged.
    print_panel_ttf_statistics(ttp, simulation_ttp)

    for strat in STRATEGIES_KEEP:
        sub = ttp[ttp["strategy"] == strat]
        if len(sub) > 0:
            print(f"filtered {strat}: {int(sub['progressed'].sum())}/{len(sub)} progressed")

    plot_boxplot(ttp, simulation_ttp, OUT_PDF_EXTRA, label="filtered")


if __name__ == "__main__":
    main()
