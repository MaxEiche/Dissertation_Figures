#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Figure 3.8 â€“ Global population competition (Elias, absolute growth vs susceptible)

- x-axis: susceptible population (absolute cell counts)
- y-axis: absolute resistant change per day: R(t) âˆ’ R(tâˆ’1)
- One mean +/- SD cross per strategy (NT, CT, AT50, eAT100, AT100)
- Global linear regression over ALL individual timepoints from these strategies,
  shown as a dotted black line.
  Regression statistics (slope, intercept, r, p, stderr, n) are printed.

Formatting:
- use_style("figstyle.mplstyle", font_family="Arial")
- set_size(ax, width="single", ratio=0.70)
- fig.subplots_adjust(left=0.12, right=0.98, bottom=0.12, top=0.92)
- colours via therapy_color(...)
- no top/right spines, ticks outward
- frameless legend
- no panel label
"""

from __future__ import annotations

import warnings
from pathlib import Path
from data_paths import DATA_ROOT, output_dir
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

from figtools import (
    ERRORBAR_CAPSIZE,
    ERRORBAR_LINEWIDTH,
    SECONDARY_LINEWIDTH,
    new_figure,
    savefig,
    therapy_color,
    use_style,
)


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

DATA_DIR = DATA_ROOT / "elias_competition"

OUT_PDF = output_dir("figures", "figure_3_8") / "Figure 3.8_Elias_AbsoluteGrowth.pdf"


# ---------------------------------------------------------------------
# Strategy definitions / mapping
# ---------------------------------------------------------------------

# Short strategy code (after mapping) -> figtools therapy_color key
STRATEGY_MAP: Dict[str, str] = {
    "CD": "CT",      # MTD remapped to CD (plotted as CT colour)
    "NT": "NT",
    "AT50": "AT50",
    # Raw "AT80" is a legacy/internal filename token for the experimentally used eAT100 condition here.
    "AT80": "eAT100",
    "AT100": "AT100",
}

# Plot order and legend labels
PLOT_ORDER: List[Tuple[str, str]] = [
    ("NT", "NT"),
    ("CD", "CT"),
    ("AT50", "AT50"),
    ("AT80", "eAT100"),
    ("AT100", "AT100"),
]


def strategy_display_label(code: str) -> str:
    """Return the visible display label for an internal strategy code."""
    return next((lbl for strat_code, lbl in PLOT_ORDER if strat_code == code), code)

# Raw labels that should be excluded completely
EXCLUDED_RAW_STRATS = {"ATSerhii", "Inter", "5", "ncr"}

MAX_CATEGORICAL_TIMEPOINTS = 12


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def load_time_series_from_file(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load one txt file with columns:
        Day  Susceptible  Resistant
    Return arrays for days > 0.
    """
    data = pd.read_csv(
        path,
        delimiter=" ",
        header=None,
        names=["Day", "Susceptible", "Resistant"],
    )

    # Day is a string like '..._d00'; keep trailing part after 'd'
    data["Day"] = data["Day"].astype(str).apply(lambda x: x.split("d")[-1])
    day = pd.to_numeric(data["Day"], errors="coerce")

    sus = pd.to_numeric(data["Susceptible"], errors="coerce")
    res = pd.to_numeric(data["Resistant"], errors="coerce")

    m = day.notna() & sus.notna() & res.notna()
    day = day[m].to_numpy()
    sus = sus[m].to_numpy()
    res = res[m].to_numpy()

    # drop day 0 (Elias convention)
    mask = day > 0
    return day[mask], sus[mask], res[mask]


def parse_raw_strategy_from_filename(filename: str) -> str:
    """
    Extract the *raw* treatment strategy label from the filename.

    Example:
      Results_G418_a2a4_C2_AT50-C2.txt
      -> 'AT50'
    """
    stem = filename.split(".")[0]
    parts = stem.split("_")
    if len(parts) < 2:
        return "UNKNOWN"

    well_name = parts[-2] + "_" + parts[-1]  # e.g. C2_AT50-C2
    raw = well_name.split("-")[0].split("_")[-1]
    return raw.strip()


def map_raw_strategy(raw: str) -> str | None:
    """
    Map raw label from filenames to canonical short codes.

      MTD      â†’ CD
      nc       â†’ NT
      AT50     â†’ AT50
      AT80     -> eAT100
      AT100    â†’ AT100
      ATSerhii, Inter, 5, ncr â†’ excluded
    """
    if raw in EXCLUDED_RAW_STRATS:
        return None
    if raw == "MTD":
        return "CD"
    if raw == "nc":
        return "NT"
    if raw in {"AT50", "AT80", "AT100"}:
        return raw
    return None


# ---------------------------------------------------------------------
# Build per-point dataset: susceptible vs *absolute* resistant change
# ---------------------------------------------------------------------

def build_competition_points(data_dir: Path) -> pd.DataFrame:
    """
    Construct per-point dataset with:

        x = susceptible cells at time t
        y = absolute resistant change between t-1 and t:
              R_t - R_{t-1}

    Columns:
        Strategy, Label, Day, susceptible, growth_abs

    Additional metadata columns are included for console-only statistics:
        colony_id, source_file, raw_strategy, Day_start, susceptible_start
    """
    records: List[Dict[str, float | str]] = []

    for path in sorted(data_dir.glob("*.txt")):
        filename = path.name

        raw_strat = parse_raw_strategy_from_filename(filename)
        mapped_strat = map_raw_strategy(raw_strat)

        if mapped_strat is None:
            continue
        if mapped_strat not in STRATEGY_MAP:
            print(f"[WARN] Unknown mapped strategy '{mapped_strat}' in {filename}")
            continue

        try:
            day, sus, res = load_time_series_from_file(path)
        except Exception as exc:
            print(f"[WARN] Failed to load {path}: {exc}")
            continue

        if day.size < 2:
            continue

        # sort by day for safety
        order = np.argsort(day)
        day_sorted = day[order]
        sus_sorted = sus[order]
        res_sorted = res[order]

        # absolute resistant change: R_t - R_{t-1}
        res_series = pd.Series(res_sorted)
        day_series = pd.Series(day_sorted)
        sus_series = pd.Series(sus_sorted)
        growth_abs = res_series.diff()  # first entry NaN
        day_start = day_series.shift(1)
        susceptible_start = sus_series.shift(1)

        label = next((lbl for s, lbl in PLOT_ORDER if s == mapped_strat), mapped_strat)

        for d, s_val, g, d_start, s_start in zip(
            day_sorted,
            sus_sorted,
            growth_abs.to_numpy(),
            day_start.to_numpy(),
            susceptible_start.to_numpy(),
        ):
            if not np.isfinite(g) or not np.isfinite(s_val):
                continue

            records.append(
                {
                    "colony_id": path.stem,
                    "source_file": filename,
                    "raw_strategy": raw_strat,
                    "Strategy": mapped_strat,
                    "Label": label,
                    "Day": float(d),
                    "Day_start": float(d_start) if np.isfinite(d_start) else np.nan,
                    "susceptible": float(s_val),
                    "susceptible_start": float(s_start) if np.isfinite(s_start) else np.nan,
                    "growth_abs": float(g),
                }
            )

    df = pd.DataFrame.from_records(records)
    return df


# ---------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------

def compute_mean_sd(values: np.ndarray) -> Tuple[float, float]:
    """Mean and standard deviation (sample SD) for a 1D array."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    mean = float(np.nanmean(values))
    if values.size > 1:
        sd = float(np.nanstd(values, ddof=1))
    else:
        sd = 0.0
    return mean, sd


def prepare_gee_analysis_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Build the analysis-only dataframe for the colony-aware GEE.

    Uses the start timepoint of each consecutive interval:
      - predictor: susceptible_start
      - timepoint: Day_start
      - outcome: growth_abs = R_t - R_{t-1}
    """
    analysis_df = df.rename(
        columns={
            "Strategy": "treatment_strategy",
            "Day_start": "timepoint",
            "susceptible_start": "sensitive_cells",
            "growth_abs": "net_resistant_growth",
        }
    ).copy()

    relevant_columns = [
        "colony_id",
        "treatment_strategy",
        "timepoint",
        "sensitive_cells",
        "net_resistant_growth",
    ]

    missing_rows_dropped = int(analysis_df[relevant_columns].isna().any(axis=1).sum())
    analysis_df = analysis_df.dropna(subset=relevant_columns).copy()
    analysis_df["timepoint"] = analysis_df["timepoint"].astype(float)
    analysis_df["sensitive_cells"] = analysis_df["sensitive_cells"].astype(float)
    analysis_df["net_resistant_growth"] = analysis_df["net_resistant_growth"].astype(float)
    analysis_df["sensitive_cells_per_1000"] = analysis_df["sensitive_cells"] / 1000.0
    analysis_df["timepoint_centered"] = (
        analysis_df["timepoint"] - analysis_df["timepoint"].mean()
    )

    return analysis_df, missing_rows_dropped


def print_gee_analysis(df: pd.DataFrame) -> None:
    """Run the console-only colony-aware GEE analysis."""
    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
    except Exception:
        print(
            "\n===== GEE ANALYSIS =====\n"
            "statsmodels is required for the colony-aware GEE analysis.\n"
            "Install it in this environment, for example:\n"
            "  .\\venv312\\Scripts\\python.exe -m pip install statsmodels\n"
        )
        return

    analysis_df, missing_rows_dropped = prepare_gee_analysis_dataframe(df)

    print("\n===== GEE COLUMN IDENTIFICATION =====")
    print("Selected analysis columns:")
    print("  colony / biological replicate identifier: colony_id")
    print("  treatment strategy: treatment_strategy")
    print("  timepoint/day: timepoint")
    print("  sensitive-cell abundance: sensitive_cells")
    print("  net resistant growth between consecutive measurements: net_resistant_growth")
    print("\nFirst five relevant rows:")
    preview_columns = [
        "colony_id",
        "treatment_strategy",
        "timepoint",
        "sensitive_cells",
        "net_resistant_growth",
    ]
    print(analysis_df.loc[:, preview_columns].head().to_string(index=False))

    if analysis_df.empty:
        print("\n===== GEE ANALYSIS =====")
        print("No rows available after analysis filtering. GEE not fitted.")
        return

    distinct_timepoints = int(analysis_df["timepoint"].nunique())
    use_categorical_time = distinct_timepoints <= MAX_CATEGORICAL_TIMEPOINTS
    if use_categorical_time:
        time_term = "C(timepoint)"
        time_note = (
            f"timepoint treated as categorical fixed effect "
            f"({distinct_timepoints} distinct timepoints)."
        )
    else:
        time_term = "timepoint_centered"
        time_note = (
            f"{distinct_timepoints} distinct timepoints detected; treating timepoint "
            f"as a centered continuous covariate in the primary model."
        )

    primary_formula = (
        f"net_resistant_growth ~ sensitive_cells_per_1000 + "
        f"C(treatment_strategy) + {time_term}"
    )
    exploratory_formula = (
        f"net_resistant_growth ~ sensitive_cells_per_1000 * "
        f"C(treatment_strategy) + {time_term}"
    )

    gee_warnings: list[warnings.WarningMessage] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        primary_model = smf.gee(
            primary_formula,
            groups="colony_id",
            data=analysis_df,
            family=sm.families.Gaussian(),
            cov_struct=sm.cov_struct.Exchangeable(),
        )
        primary_result = primary_model.fit(cov_type="robust")
        gee_warnings.extend(caught)

    coef_table = pd.DataFrame(
        {
            "coefficient": primary_result.params,
            "robust_se": primary_result.bse,
            "ci_lower": primary_result.conf_int()[0],
            "ci_upper": primary_result.conf_int()[1],
            "z": primary_result.tvalues,
            "p_value": primary_result.pvalues,
        }
    )

    sensitive_term = "sensitive_cells_per_1000"
    sensitive_present = sensitive_term in coef_table.index
    sensitive_reliable = False
    if sensitive_present:
        sensitive_row = coef_table.loc[sensitive_term]
        sensitive_reliable = bool(np.isfinite(sensitive_row.to_numpy()).all())
    else:
        sensitive_row = None

    n_rows = int(len(analysis_df))
    n_colonies = int(analysis_df["colony_id"].nunique())
    obs_per_colony = analysis_df.groupby("colony_id").size()
    colonies_per_group = (
        analysis_df.groupby("treatment_strategy")["colony_id"].nunique().sort_index()
    )

    reg_strategies = {"NT", "CD", "AT50", "AT80", "AT100"}
    descriptive_df = df.loc[df["Strategy"].isin(reg_strategies), ["susceptible", "growth_abs"]]
    descriptive_df = descriptive_df.dropna()
    descriptive_n = int(len(descriptive_df))
    descriptive_r = np.nan
    descriptive_p = np.nan
    if descriptive_n >= 2:
        descriptive_reg = linregress(
            descriptive_df["susceptible"].to_numpy(),
            descriptive_df["growth_abs"].to_numpy(),
        )
        descriptive_r = descriptive_reg.rvalue
        descriptive_p = descriptive_reg.pvalue

    residuals = analysis_df["net_resistant_growth"] - primary_result.fittedvalues
    single_observation_colony = bool((obs_per_colony == 1).any())

    print("\n===== PRIMARY GEE ANALYSIS =====")
    print(f"Rows included in GEE analysis: {n_rows}")
    print(f"Unique colonies/replicates: {n_colonies}")
    print(
        "Observations per colony: "
        f"min={int(obs_per_colony.min())}, "
        f"median={float(obs_per_colony.median()):.1f}, "
        f"max={int(obs_per_colony.max())}"
    )
    print("Treatment groups and number of colonies per group:")
    for group_name, count in colonies_per_group.items():
        print(f"  {strategy_display_label(group_name)}: {int(count)} colonies")
    print(f"Timepoint handling: {time_note}")
    print(f"Exact GEE formula: {primary_formula}")
    print("Working correlation structure: Exchangeable")
    print("Covariance estimator: robust/sandwich")
    print("\nFull GEE coefficient table:")
    print(coef_table.to_string(float_format=lambda x: f"{x:.6g}"))

    print("\nSensitive-cell coefficient summary:")
    if sensitive_reliable and sensitive_row is not None:
        direction = "positive" if sensitive_row["coefficient"] > 0 else "negative"
        print(
            f"  Coefficient per additional 1000 sensitive cells: "
            f"{sensitive_row['coefficient']:.6g}"
        )
        print(
            f"  95% CI: [{sensitive_row['ci_lower']:.6g}, "
            f"{sensitive_row['ci_upper']:.6g}]"
        )
        print(f"  p-value: {sensitive_row['p_value']:.6g}")
        print(f"  Association direction: {direction}")
    else:
        print("  Sensitive-cell coefficient could not be estimated reliably.")

    print("\nDESCRIPTIVE POOLED CORRELATION ONLY â€” DOES NOT ACCOUNT FOR REPEATED MEASURES")
    if descriptive_n >= 2 and np.isfinite(descriptive_r) and np.isfinite(descriptive_p):
        print(f"  Pearson r: {descriptive_r:.6g}")
        print(f"  p-value: {descriptive_p:.6g}")
        print(f"  Observations: {descriptive_n}")
    else:
        print("  Not enough observations to compute pooled Pearson correlation.")

    print("\nDiagnostic section:")
    print(f"  Missing rows dropped: {missing_rows_dropped}")
    print(f"  Any colony with only one usable observation: {single_observation_colony}")
    print(f"  Residual mean: {float(np.mean(residuals)):.6g}")
    print(f"  Residual standard deviation: {float(np.std(residuals, ddof=1)):.6g}")
    if gee_warnings:
        print("  Warning: model fitting produced warnings:")
        for w in gee_warnings:
            message = str(w.message).strip()
            if message:
                print(f"    - {message}")
    else:
        print("  No model-fitting warnings captured.")
    if not bool(getattr(primary_result, "converged", True)):
        print("  Warning: primary GEE did not report convergence.")
    if not sensitive_reliable:
        print("  Warning: sensitive-cell coefficient may not be estimated reliably.")

    print("\n===== EXPLORATORY SECONDARY GEE =====")
    print("Exploratory only: interaction between sensitive-cell abundance and treatment strategy.")
    print(f"Exact GEE formula: {exploratory_formula}")

    exploratory_warnings: list[warnings.WarningMessage] = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            exploratory_model = smf.gee(
                exploratory_formula,
                groups="colony_id",
                data=analysis_df,
                family=sm.families.Gaussian(),
                cov_struct=sm.cov_struct.Exchangeable(),
            )
            exploratory_result = exploratory_model.fit(cov_type="robust")
            exploratory_warnings.extend(caught)

        try:
            wald_terms = exploratory_result.wald_test_terms(skip_single=False, scalar=True)
            wald_table = wald_terms.summary_frame()
            interaction_rows = [
                idx for idx in wald_table.index if "sensitive_cells_per_1000:C(treatment_strategy)" in str(idx)
            ]
            if interaction_rows:
                print("Omnibus/term-level Wald results for interaction terms:")
                print(wald_table.loc[interaction_rows].to_string(float_format=lambda x: f"{x:.6g}"))
            else:
                raise ValueError("Interaction term not available in Wald term table.")
        except Exception:
            interaction_mask = exploratory_result.params.index.str.contains(
                "sensitive_cells_per_1000:C\\(treatment_strategy\\)",
                regex=True,
            )
            interaction_table = pd.DataFrame(
                {
                    "coefficient": exploratory_result.params[interaction_mask],
                    "robust_se": exploratory_result.bse[interaction_mask],
                    "ci_lower": exploratory_result.conf_int().loc[interaction_mask, 0],
                    "ci_upper": exploratory_result.conf_int().loc[interaction_mask, 1],
                    "z": exploratory_result.tvalues[interaction_mask],
                    "p_value": exploratory_result.pvalues[interaction_mask],
                }
            )
            if interaction_table.empty:
                print("Interaction terms were not estimable.")
            else:
                print("Coefficient table for interaction terms:")
                print(interaction_table.to_string(float_format=lambda x: f"{x:.6g}"))

        if exploratory_warnings:
            print("Exploratory-model warnings:")
            for w in exploratory_warnings:
                message = str(w.message).strip()
                if message:
                    print(f"  - {message}")
    except Exception as exc:
        print(f"Exploratory interaction GEE could not be fitted: {exc}")

    print("\nTHESIS-READY PRIMARY RESULT")
    if sensitive_reliable and sensitive_row is not None:
        direction = "negative" if sensitive_row["coefficient"] < 0 else "positive"
        print(f"* GEE coefficient per 1000 sensitive cells: {sensitive_row['coefficient']:.6g}")
        print(
            f"* 95% CI: [{sensitive_row['ci_lower']:.6g}, "
            f"{sensitive_row['ci_upper']:.6g}]"
        )
        print(f"* p-value: {sensitive_row['p_value']:.6g}")
        print(f"* number of colonies: {n_colonies}")
        print(f"* number of observations: {n_rows}")
        print(
            "* Interpretation: This is an adjusted population-average "
            f"{direction} association accounting for repeated measurements within colonies."
        )
    else:
        print("* GEE coefficient per 1000 sensitive cells: not estimable")
        print("* 95% CI: not estimable")
        print("* p-value: not estimable")
        print(f"* number of colonies: {n_colonies}")
        print(f"* number of observations: {n_rows}")
        print(
            "* Interpretation: The adjusted population-average association accounting "
            "for repeated measurements within colonies could not be estimated reliably."
        )


def plot_growth_mean_sd_and_regression(ax: plt.Axes, df: pd.DataFrame) -> None:
    """
    Plot:
      - one mean Â± SD cross per strategy (no background dots)
      - global linear regression line (dotted black)
        using ALL individual timepoints from NT, CT, AT50, eAT100, AT100.

    y-axis shows *absolute* resistant change (cells).
    """
    if df.empty:
        raise RuntimeError("No data points to plot.")

    # -------- Regression on ALL individual points from selected strategies --------
    reg_strategies = {"NT", "CD", "AT50", "AT80", "AT100"}
    mask_reg = df["Strategy"].isin(reg_strategies)

    x_all = df.loc[mask_reg, "susceptible"].to_numpy()
    y_all = df.loc[mask_reg, "growth_abs"].to_numpy()

    m = np.isfinite(x_all) & np.isfinite(y_all)
    x_reg = x_all[m]
    y_reg = y_all[m]

    print(f"[INFO] Regression uses {x_reg.size} individual points.")

    if x_reg.size >= 2:
        reg = linregress(x_reg, y_reg)
        slope, intercept, r_value, p_value, stderr = reg
        print(
            "[INFO] DESCRIPTIVE POOLED CORRELATION ONLY â€” DOES NOT ACCOUNT FOR "
            "REPEATED MEASURES | "
            "Linear regression (all timepoints; absolute growth vs susceptible): "
            f"slope={slope:.3e}, intercept={intercept:.3f}, "
            f"r={r_value:.3f}, p={p_value:.3e}, stderr={stderr:.3e}"
        )

        x_line = np.linspace(0.0, 25000.0, 200)
        y_line = slope * x_line + intercept

        ax.plot(
            x_line,
            y_line,
            linestyle=":",
            color="black",
            linewidth=SECONDARY_LINEWIDTH,
            zorder=1,
        )
    else:
        print("[WARN] Not enough points for regression; skipping line.")

    # -------- Strategy-wise mean Â± SD crosses (absolute change) --------
    for strat_short, label in PLOT_ORDER:
        sel = df["Strategy"] == strat_short
        if not sel.any():
            continue

        x = df.loc[sel, "susceptible"].to_numpy()
        y = df.loc[sel, "growth_abs"].to_numpy()

        color_key = STRATEGY_MAP[strat_short]
        try:
            color = therapy_color(color_key)
        except KeyError:
            color = "k"

        sus_mean, sus_sd = compute_mean_sd(x)
        gr_mean, gr_sd = compute_mean_sd(y)

        if not np.isfinite(sus_mean) or not np.isfinite(gr_mean):
            continue

        # Symmetric SD errorbars
        xerr = np.array([[sus_sd], [sus_sd]])
        yerr = np.array([[gr_sd], [gr_sd]])

        ax.errorbar(
            [sus_mean],
            [gr_mean],
            xerr=xerr,
            yerr=yerr,
            fmt="o",
            elinewidth=ERRORBAR_LINEWIDTH,
            capsize=ERRORBAR_CAPSIZE,
            color=color,
            ecolor=color,
            alpha=0.95,
            linestyle="none",
            label=label,
            zorder=2,
        )

    # -------- Axis labels and limits --------
    ax.set_xlabel("Sensitive Population")
    ax.set_ylabel("Absolute Resistant Growth")

    # x-limits fixed for comparability
    ax.set_xlim(0, 26000)
    ax.set_ylim(-1600, 4000)

    # Legend: one entry per strategy (from crosses)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        seen = set()
        uniq_h, uniq_l = [], []
        for h, lbl in zip(handles, labels):
            if lbl in seen:
                continue
            seen.add(lbl)
            uniq_h.append(h)
            uniq_l.append(lbl)
        ax.legend(uniq_h, uniq_l, frameon=False, loc="best")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(f"DATA_DIR does not exist: {DATA_DIR}")

    df = build_competition_points(DATA_DIR)
    print("[INFO] Points per strategy (absolute growth):")
    if not df.empty:
        counts = df.groupby("Strategy")[["susceptible", "growth_abs"]].count()
        counts = counts.rename(index={code: strategy_display_label(code) for code in counts.index})
        print(counts)
        print_gee_analysis(df)

    with use_style("figstyle.mplstyle", font_family="Arial"):
        fig, ax = new_figure(width="single", ratio=0.70)

        plot_growth_mean_sd_and_regression(ax, df)

        savefig(fig, OUT_PDF, dpi=300, transparent=True)
        savefig(fig, OUT_PDF.with_suffix(".png"), dpi=600, transparent=False, facecolor="white")
        plt.show()
        plt.close(fig)

    print(f"[SAVE] Figure 3.8 written to: {OUT_PDF}")


if __name__ == "__main__":
    main()
