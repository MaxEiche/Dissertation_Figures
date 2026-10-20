#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Figure 4.11 - Resistant growth before versus after spheroid disruption.

The raw before/after boxplot is retained, but the primary inference is now based
on a collapse-restricted spheroid-clustered GEE with a paired within-spheroid
sensitivity analysis.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon
import statsmodels.api as sm
import statsmodels.formula.api as smf

from figtools import (
    ANNOTATION_FONTSIZE,
    NEUTRAL_STATE_COLORS,
    SECONDARY_LINEWIDTH,
    new_figure,
    savefig,
    use_style,
)


DATA_XLS = Path(
    r"Y:\Members\MaxE\Thesis\Organized Figures\Chapter_4\Figure 4.11 Spheroid Disruption\ME_A2-3 Data.xlsx"
)
OUT_PDF = DATA_XLS.parent / "Figure 4.11 ResGrowth_after_SpheroidDisruption.pdf"
OUT_STATS = DATA_XLS.parent / "Figure 4.11 ResGrowth_after_SpheroidDisruption_stats.json"

SAMPLE_GROUPS = {
    "No treatment": ["A1", "A2", "A3", "A4", "A5", "A6"],
    "Maximum-tolerable-dose": ["A7", "A8", "A9", "A10", "A11", "A12"],
    "Adaptive Therapy 100": [
        "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10", "B11",
        "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11",
    ],
    "Adaptive Therapy 80": [
        "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10", "D11",
        "E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9", "E10", "E11",
    ],
    "Adaptive Therapy 50": [
        "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10", "F11",
        "G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9", "G10", "G11",
    ],
    "Resistant Control No Treatment": ["H1", "H2", "H3", "H4", "H5", "H6"],
    "Resistant Control Maximum-tolerable-dose": ["H7", "H8", "H9", "H10", "H11", "H12"],
    "Intermittent": ["B12", "C12", "D12", "E12", "F12", "G12"],
}
GROUP_SHORT = {
    "No treatment": "NT",
    "Maximum-tolerable-dose": "CT",
    "Adaptive Therapy 100": "AT100",
    "Adaptive Therapy 80": "AT80",
    "Adaptive Therapy 50": "AT50",
    "Resistant Control No Treatment": "RC_NT",
    "Resistant Control Maximum-tolerable-dose": "RC_CT",
    "Intermittent": "INT",
}
MIXED_STRATEGIES = {"NT", "CT", "AT100", "AT80", "AT50"}


def extract_and_normalize_remove_artifacts(data_df: pd.DataFrame) -> pd.DataFrame:
    extracted_data = {"Replicate": [], "Day": [], "g": [], "r": [], "t": []}

    replicate_columns = range(3, data_df.shape[1], 5)
    g_columns = range(3, data_df.shape[1], 5)
    r_columns = range(5, data_df.shape[1], 5)
    t_columns = range(6, data_df.shape[1], 5)
    days = data_df.iloc[3:, 0].dropna().astype(int).unique()

    for rep_col, g_col, r_col, t_col in zip(replicate_columns, g_columns, r_columns, t_columns):
        replicate = data_df.iloc[1, rep_col]
        for day in days:
            if day in (29, 15, 27):
                continue

            day_data = data_df.loc[data_df.iloc[:, 0] == day]
            if day_data.empty:
                continue

            g_value = day_data.iloc[0, g_col]
            r_value = day_data.iloc[0, r_col]

            if day in range(-2, 6):
                g_value -= 40_000_000
                r_value -= 40_000_000
            elif day == 6:
                g_value -= 5_000_000
                r_value -= 5_000_000

            g_value -= r_value / 12.0

            extracted_data["Replicate"].append(replicate)
            extracted_data["Day"].append(day)
            extracted_data["g"].append(g_value)
            extracted_data["r"].append(r_value)
            extracted_data["t"].append(day_data.iloc[0, t_col])

    extracted_df = pd.DataFrame(extracted_data)
    baseline_t_values = extracted_df[extracted_df["Day"] == 0].set_index("Replicate")["t"]

    normalized_rows = []
    for _, row in extracted_df.iterrows():
        baseline_t = baseline_t_values.get(row["Replicate"])
        if baseline_t is None:
            continue
        normalized_rows.append(
            {
                "Replicate": row["Replicate"],
                "Day": row["Day"],
                "g_normalized": row["g"] / baseline_t,
                "r_normalized": row["r"] / baseline_t,
                "t_normalized": row["t"] / baseline_t,
            }
        )

    return pd.DataFrame(normalized_rows)


def combine_normalized_with_treatment(
    normalized_df: pd.DataFrame,
    treatment_df: pd.DataFrame,
) -> pd.DataFrame:
    treatment_days = treatment_df.iloc[:, 0].dropna().unique()
    replicate_columns = treatment_df.columns[2:]

    treatment_dict = {}
    for _, row in treatment_df.iterrows():
        day_str = row.iloc[0]
        if day_str not in treatment_days:
            continue
        day = int(str(day_str)[1:])
        for replicate in replicate_columns:
            treatment_dict[(day, replicate)] = row[replicate]

    normalized_df = normalized_df.copy()
    normalized_df["treatment"] = normalized_df.apply(
        lambda r: treatment_dict.get((r["Day"], r["Replicate"]), 0),
        axis=1,
    )
    return normalized_df


def add_fallapart_info(combined_df: pd.DataFrame, fallapart_df: pd.DataFrame) -> pd.DataFrame:
    fallapart_days = fallapart_df.iloc[:, 0].dropna().unique()
    replicate_columns = fallapart_df.columns[2:]

    fallapart_dict = {}
    for _, row in fallapart_df.iterrows():
        day_str = row.iloc[0]
        if day_str not in fallapart_days:
            continue
        day = int(str(day_str)[1:])
        for replicate in replicate_columns:
            fallapart_dict[(day, replicate)] = row[replicate]

    combined_df = combined_df.copy()
    combined_df["fallapart_info"] = combined_df.apply(
        lambda r: 6 if fallapart_dict.get((r["Day"], r["Replicate"]), 0) == 6 else 0,
        axis=1,
    )
    # Fix collapse from all original annotations, including omitted imaging days.
    collapse_days = {}
    for (day, replicate), code in fallapart_dict.items():
        if code == 6:
            collapse_days[replicate] = min(day, collapse_days.get(replicate, day))
    combined_df["collapse_day"] = combined_df["Replicate"].map(collapse_days)
    return combined_df


def build_group_map() -> dict[str, str]:
    rep_to_group = {}
    for plate_prefix in ["1", "2"]:
        for group_name, wells in SAMPLE_GROUPS.items():
            for well in wells:
                rep_to_group[plate_prefix + well] = GROUP_SHORT[group_name]
    return rep_to_group


def build_base_df(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    df1 = df1.copy()
    df2 = df2.copy()
    df1["Replicate"] = df1["Replicate"].apply(lambda x: f"1{x}")
    df2["Replicate"] = df2["Replicate"].apply(lambda x: f"2{x}")
    df = pd.concat([df1, df2], ignore_index=True)
    df = df.sort_values(["Replicate", "Day"])
    df["strategy"] = df["Replicate"].map(build_group_map())
    df["interval_days"] = df["Day"] - df.groupby("Replicate")["Day"].shift(1)
    df["relative_change"] = (
        df.groupby("Replicate")["r_normalized"].pct_change(fill_method=None) + 1.0
    )
    df = df[
        [
            "Replicate",
            "Day",
            "r_normalized",
            "t_normalized",
            "g_normalized",
            "treatment",
            "relative_change",
            "fallapart_info",
            "strategy",
            "collapse_day",
            "interval_days",
        ]
    ]
    df = df[~df["treatment"].isin([0.0, 3.0, 4.0, 5.0])]
    df = df.dropna(subset=["relative_change"])
    df = df[df["relative_change"].between(0.75, 2.0)].copy()
    df = df[df["interval_days"] > 0].copy()
    df["relative_change_pct"] = 100.0 * (df["relative_change"] - 1.0) / df["interval_days"]
    # Audit against the former alignment inferred from surviving analysis rows.
    old_collapse = df.loc[df["fallapart_info"] == 6].groupby("Replicate")["Day"].min()
    old_days = df["Replicate"].map(old_collapse)
    df["collapse_alignment_changed"] = df["collapse_day"].notna() & (
        old_days.isna() | df["collapse_day"].ne(old_days)
    )
    df.attrs["collapse_alignment_changed"] = {
        "all_spheroids": int(df.loc[df["collapse_alignment_changed"], "Replicate"].nunique()),
        "selected_strategies": int(df.loc[
            df["collapse_alignment_changed"] & df["strategy"].isin(MIXED_STRATEGIES),
            "Replicate",
        ].nunique()),
    }
    return df


def mean_ci(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    se = values.std(ddof=1) / np.sqrt(len(values))
    delta = 1.96 * se
    return mean, float(mean - delta), float(mean + delta)


def p_to_stars(pval: float) -> str:
    if pval < 1e-4:
        return "****"
    if pval < 1e-3:
        return "***"
    if pval < 0.01:
        return "**"
    if pval < 0.05:
        return "*"
    return "n.s."


def build_descriptive_plot_df(base_df: pd.DataFrame) -> pd.DataFrame:
    df = base_df[base_df["strategy"].isin(MIXED_STRATEGIES)].copy()
    df = df[df["collapse_day"].notna()].copy()
    df["adjusted_day"] = df["Day"] - df["collapse_day"]
    df["post_collapse"] = (df["adjusted_day"] >= 0).astype(int)
    both = df.groupby(["Replicate", "post_collapse"]).size().unstack()
    both = both.dropna().index
    df = df[df["Replicate"].isin(both)].copy()
    df["period"] = np.where(df["post_collapse"] == 1, "On/after observed collapse", "Before disruption")
    return df


def summarize_statistics(base_df: pd.DataFrame) -> dict:
    # Use the same fixed observed collapse day for descriptive statistics.
    descriptive = base_df[base_df["collapse_day"].notna()].copy()
    descriptive["adjusted_day"] = descriptive["Day"] - descriptive["collapse_day"]
    excluded_true = [
        "1" + r for r in SAMPLE_GROUPS["Resistant Control No Treatment"] + SAMPLE_GROUPS["Resistant Control Maximum-tolerable-dose"]
    ] + [
        "2" + r for r in SAMPLE_GROUPS["Resistant Control No Treatment"] + SAMPLE_GROUPS["Resistant Control Maximum-tolerable-dose"]
    ]
    descriptive = descriptive[~descriptive["Replicate"].isin(excluded_true)].copy()
    descriptive["period"] = np.where(descriptive["adjusted_day"] < 0, "Before disruption", "On/after observed collapse")
    before_vals = descriptive.loc[descriptive["period"] == "Before disruption", "relative_change_pct"]
    after_vals = descriptive.loc[descriptive["period"] == "On/after observed collapse", "relative_change_pct"]
    pooled_u, pooled_p = mannwhitneyu(before_vals, after_vals, alternative="two-sided")

    infer_df = build_descriptive_plot_df(base_df)
    paired = infer_df.groupby(["Replicate", "post_collapse"])["relative_change_pct"].mean().unstack()
    paired = paired.dropna()
    paired["diff"] = paired[1] - paired[0]
    paired_mean, paired_lo, paired_hi = mean_ci(paired["diff"].to_numpy())
    paired_w = wilcoxon(paired["diff"].to_numpy())

    gee_df = infer_df.copy()
    gee_df["adj_day_c"] = gee_df["adjusted_day"] - gee_df["adjusted_day"].mean()
    gee = smf.gee(
        "relative_change_pct ~ post_collapse + adj_day_c + C(strategy)",
        groups="Replicate",
        data=gee_df,
        cov_struct=sm.cov_struct.Exchangeable(),
        family=sm.families.Gaussian(),
    )
    gee_res = gee.fit()
    gee_ci = gee_res.conf_int().loc["post_collapse"]

    return {
        "dataset_definition": {
            "source_workbook": str(DATA_XLS),
            "included_strategies_for_primary_model": sorted(MIXED_STRATEGIES),
            "required_for_primary_model": [
                "Actual collapse annotation present",
                "Usable observations on both sides of first collapse",
                "No resistant controls",
                "No non-collapse alignment to last observation",
                "Fixed earliest original collapse annotation before growth/outcome filtering",
                "Endpoint before versus on/after collapse; transition intervals retained",
                "Outcome: 100 * (R_end / R_start - 1) / elapsed days",
            ],
            "verified_excluded_plate2_controls": [f"2H{i}" for i in range(1, 13)],
        },
        "collapse_alignment_changed": base_df.attrs["collapse_alignment_changed"],
        "primary_inference": {
            "label": "Primary spheroid-aware inference",
            "method": "GEE Gaussian, exchangeable working correlation, robust sandwich SE",
            "formula": "relative_change_pct ~ post_collapse + adj_day_c + C(strategy)",
            "cluster": "Replicate",
            "n_spheroids": int(gee_df["Replicate"].nunique()),
            "n_obs": int(len(gee_df)),
            "interval_length_counts": {
                str(int(days)): int(count)
                for days, count in gee_df["interval_days"].value_counts().sort_index().items()
            },
            "n_collapse_alignment_changed": int(gee_df.loc[
                gee_df["collapse_alignment_changed"], "Replicate"
            ].nunique()),
            "coefficient_units": "percentage points per day",
            "n_pre_obs": int((gee_df["post_collapse"] == 0).sum()),
            "n_post_obs": int((gee_df["post_collapse"] == 1).sum()),
            "coefficient_pct_points": float(gee_res.params["post_collapse"]),
            "ci_low": float(gee_ci[0]),
            "ci_high": float(gee_ci[1]),
            "p": float(gee_res.pvalues["post_collapse"]),
            "interpretation": "Adjusted on/after-minus-before observed collapse difference in resistant-fluorescence percentage change per day.",
        },
        "descriptive_pooled_summary": {
            "label": "Descriptive pooled summary only",
            "method": "Corrected pooled Mann-Whitney U (not primary inference)",
            "n_before": int(before_vals.size),
            "n_after": int(after_vals.size),
            "median_before_pct": float(np.median(before_vals)),
            "median_after_pct": float(np.median(after_vals)),
            "u": float(pooled_u),
            "p": float(pooled_p),
        },
        "sensitivity_analysis": {
            "label": "Paired spheroid-level sensitivity analysis",
            "method": "Within-spheroid post-minus-pre mean difference with Wilcoxon signed-rank test",
            "n_spheroids": int(len(paired)),
            "mean_diff_pct_points": float(paired_mean),
            "median_diff_pct_points": float(np.median(paired["diff"])),
            "ci_low": float(paired_lo),
            "ci_high": float(paired_hi),
            "p": float(paired_w.pvalue),
        },
    }


def plot_res_growth_after_spheroid_disruption(
    df1_fall: pd.DataFrame,
    df2_fall: pd.DataFrame,
    primary_p: float,
) -> None:
    df_plot = build_descriptive_plot_df(build_base_df(df1_fall, df2_fall))

    data_before = df_plot.loc[
        df_plot["period"] == "Before disruption", "relative_change_pct"
    ].to_numpy()
    data_after = df_plot.loc[
        df_plot["period"] == "On/after observed collapse", "relative_change_pct"
    ].to_numpy()
    positions = [0, 1]

    with use_style("figstyle.mplstyle"):
        fig, ax = new_figure(width="single", ratio=1.2)

        bp = ax.boxplot(
            [data_before, data_after],
            positions=positions,
            widths=0.6,
            showfliers=False,
            patch_artist=True,
        )

        # Keep a light-vs-dark ordering while avoiding a true black fill that
        # blends into the box outlines and median line.
        color_before = NEUTRAL_STATE_COLORS["inactive"]
        color_after = NEUTRAL_STATE_COLORS["active"]
        for patch, col in zip(bp["boxes"], [color_before, color_after]):
            patch.set(facecolor=col, alpha=0.8, edgecolor="black", linewidth=1.0)

        for element in ["whiskers", "caps", "medians"]:
            for item in bp[element]:
                item.set(color="black", linewidth=1.0)

        ax.set_xticks(positions)
        ax.set_xticklabels(["Before Disruption", "On/after observed collapse"])
        ax.set_xlabel("")
        ax.set_ylabel("Relative resistant-fluorescence change [% per day]")
        ax.set_ylim(-30, 70)

        stars = p_to_stars(primary_p)
        y = 62.0
        h = 3.0
        ax.plot(
            [positions[0], positions[0], positions[1], positions[1]],
            [y, y + h, y + h, y],
            color="black",
            linewidth=SECONDARY_LINEWIDTH,
        )
        ax.text(
            0.5 * (positions[0] + positions[1]),
            y + h + 0.5,
            stars,
            ha="center",
            va="bottom",
            fontsize=ANNOTATION_FONTSIZE,
        )

        savefig(fig, OUT_PDF, dpi=300, transparent=True)
        savefig(fig, OUT_PDF.with_suffix(".png"), dpi=600, transparent=False, facecolor="white")
        plt.show()
        plt.close(fig)
        print(f"Saved fallapart figure to: {OUT_PDF}")


def main() -> None:
    xls = pd.ExcelFile(DATA_XLS)
    data_df_plate1 = pd.read_excel(xls, "Plate 1")
    data_df_plate2 = pd.read_excel(xls, "Plate 2")
    treatment_df_plate1 = pd.read_excel(xls, "Plate 1 Treatment")
    treatment_df_plate2 = pd.read_excel(xls, "Plate 2 Treatment")
    fallapart_df_plate1 = pd.read_excel(xls, "Plate 1 Treatment_fallapart")
    fallapart_df_plate2 = pd.read_excel(xls, "Plate 2 Treatment_fallapart")

    norm1 = extract_and_normalize_remove_artifacts(data_df_plate1)
    norm2 = extract_and_normalize_remove_artifacts(data_df_plate2)

    comb1 = combine_normalized_with_treatment(norm1, treatment_df_plate1)
    comb2 = combine_normalized_with_treatment(norm2, treatment_df_plate2)

    comb1_fall = add_fallapart_info(comb1, fallapart_df_plate1)
    comb2_fall = add_fallapart_info(comb2, fallapart_df_plate2)

    stats_summary = summarize_statistics(build_base_df(comb1_fall, comb2_fall))

    primary = stats_summary["primary_inference"]
    print("[Primary spheroid-aware inference]")
    print(
        f"GEE, clustered by spheroid: post_collapse = {primary['coefficient_pct_points']:.3f} pp/day, "
        f"95% CI [{primary['ci_low']:.3f}, {primary['ci_high']:.3f}], "
        f"p = {primary['p']:.3e}, n_obs = {primary['n_obs']}, "
        f"n_spheroids = {primary['n_spheroids']}"
    )
    print(f"Primary-model interval lengths (days: count): {primary['interval_length_counts']}")
    changed = stats_summary["collapse_alignment_changed"]
    print(f"Spheroids with changed collapse alignment: all = {changed['all_spheroids']}, "
          f"selected strategies = {changed['selected_strategies']}, "
          f"included in primary model = {primary['n_collapse_alignment_changed']}")

    pooled = stats_summary["descriptive_pooled_summary"]
    print("[Descriptive pooled summary only]")
    print(
        f"Corrected pooled comparison retained only as display: n_before = {pooled['n_before']}, "
        f"n_after = {pooled['n_after']}, medians = {pooled['median_before_pct']:.3f}% vs "
        f"{pooled['median_after_pct']:.3f}% per day, "
        f"Mann-Whitney U = {pooled['u']:.3f}, p = {pooled['p']:.3e}"
    )

    sens = stats_summary["sensitivity_analysis"]
    print("[Sensitivity analysis]")
    print(
        f"Within-spheroid on/after-minus-before difference = {sens['mean_diff_pct_points']:.3f} pp/day "
        f"(median {sens['median_diff_pct_points']:.3f}), "
        f"95% CI [{sens['ci_low']:.3f}, {sens['ci_high']:.3f}], "
        f"Wilcoxon p = {sens['p']:.3e}, n = {sens['n_spheroids']}"
    )

    OUT_STATS.write_text(json.dumps(stats_summary, indent=2), encoding="utf-8")
    plot_res_growth_after_spheroid_disruption(
        comb1_fall,
        comb2_fall,
        primary_p=primary["p"],
    )


if __name__ == "__main__":
    main()
