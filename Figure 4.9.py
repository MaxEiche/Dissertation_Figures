#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Figure 4.9 - Recent treatment exposure versus resistant growth.

The raw pooled boxplot is retained as a descriptive visualization, but the
primary inference is now a spheroid-clustered GEE with a paired within-spheroid
sensitivity analysis written alongside the figure.
"""

from __future__ import annotations

import json
from pathlib import Path
from data_paths import DATA_ROOT, output_dir

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


DATA_XLS = DATA_ROOT / "low_adhesion" / "ME_A2-3 Data.xlsx"
OUT_PDF = output_dir("figures", "figure_4_9") / "Figure 4.9 TreatmentEffect_on_ResistantGrowth.pdf"
OUT_STATS = output_dir("figures", "figure_4_9") / "Figure 4.9 TreatmentEffect_on_ResistantGrowth_stats.json"

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


def build_group_map() -> dict[str, str]:
    rep_to_group = {}
    for plate_prefix in ["1", "2"]:
        for group_name, wells in SAMPLE_GROUPS.items():
            for well in wells:
                rep_to_group[plate_prefix + well] = GROUP_SHORT[group_name]
    return rep_to_group


def build_analysis_df(
    df1: pd.DataFrame, df2: pd.DataFrame,
    treatment_df1: pd.DataFrame, treatment_df2: pd.DataFrame,
) -> pd.DataFrame:
    # Use every treatment-sheet day, including days omitted from imaging.
    treatment_lookup = {}
    for prefix, sheet in (("1", treatment_df1), ("2", treatment_df2)):
        for _, row in sheet.iterrows():
            if pd.isna(row.iloc[0]):
                continue
            day = int(str(row.iloc[0])[1:])
            for well in sheet.columns[2:]:
                treatment_lookup[(prefix + str(well), day)] = row[well]

    df1 = df1.copy()
    df2 = df2.copy()
    df1["Replicate"] = df1["Replicate"].apply(lambda x: f"1{x}")
    df2["Replicate"] = df2["Replicate"].apply(lambda x: f"2{x}")
    df = pd.concat([df1, df2], ignore_index=True)
    df = df.sort_values(["Replicate", "Day"])
    df["strategy"] = df["Replicate"].map(build_group_map())
    df["day_start"] = df.groupby("Replicate")["Day"].shift(1)
    df["interval_days"] = df["Day"] - df["day_start"]
    df["relative_change"] = (
        df.groupby("Replicate")["r_normalized"].pct_change(fill_method=None) + 1.0
    )
    df = df[df["interval_days"] > 0].copy()

    def interval_state(row):
        # Imaging precedes treatment changes: exposure covers [start, end).
        codes = [
            treatment_lookup.get((row["Replicate"], day))
            for day in range(int(row["day_start"]), int(row["Day"]))
        ]
        if any(code not in (1, 2) for code in codes):
            return "unknown"
        return codes[0] if len(set(codes)) == 1 else "mixed"

    df["treatment"] = df.apply(interval_state, axis=1)
    exclusions = {}
    for scope, intervals in (
        ("all_strategies", df),
        ("primary_strategies", df[df["strategy"].isin(MIXED_STRATEGIES)]),
    ):
        exclusions[scope] = {
            "formed_intervals": int(len(intervals)),
            "mixed": int((intervals["treatment"] == "mixed").sum()),
            "unknown": int((intervals["treatment"] == "unknown").sum()),
        }
    df = df[df["treatment"].isin([1, 2])].copy()
    df = df.dropna(subset=["relative_change"])
    df = df[df["relative_change"].between(0.75, 2.0)].copy()
    df["relative_change_pct"] = 100.0 * (df["relative_change"] - 1.0) / df["interval_days"]
    df["treated_prev"] = (df["treatment"] == 1.0).astype(int)
    df["day_c"] = df["Day"] - df["Day"].mean()
    df.attrs["interval_exclusions"] = exclusions
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


def summarize_statistics(df: pd.DataFrame) -> dict:
    pooled_treat = df[df["treatment"] == 1.0]["relative_change_pct"]
    pooled_no = df[df["treatment"] == 2.0]["relative_change_pct"]
    pooled_u, pooled_p = mannwhitneyu(pooled_treat, pooled_no, alternative="two-sided")

    mix = df[df["strategy"].isin(MIXED_STRATEGIES)].copy()
    paired = mix.groupby(["Replicate", "treated_prev"])["relative_change_pct"].mean().unstack()
    paired = paired.dropna()
    paired["diff"] = paired[1] - paired[0]
    paired_mean, paired_lo, paired_hi = mean_ci(paired["diff"].to_numpy())
    paired_w = wilcoxon(paired["diff"].to_numpy())

    gee = smf.gee(
        "relative_change_pct ~ treated_prev + day_c + C(strategy)",
        groups="Replicate",
        data=mix,
        cov_struct=sm.cov_struct.Exchangeable(),
        family=sm.families.Gaussian(),
    )
    gee_res = gee.fit()
    gee_ci = gee_res.conf_int().loc["treated_prev"]

    return {
        "dataset_definition": {
            "source_workbook": str(DATA_XLS),
            "included_strategies_for_primary_model": sorted(MIXED_STRATEGIES),
            "filters": [
                "Removed artifact days 15, 27, 29",
                "Normalized to day-0 total fluorescence",
                "Aligned exposure to original treatment-sheet days in [day_start, Day)",
                "Excluded mixed intervals and unknown exposure (missing or codes 0, 3, 4, 5)",
                "Restricted relative_change to [0.75, 2.0]",
                "Outcome: 100 * (R_end / R_start - 1) / elapsed days",
            ],
        },
        "interval_exclusions_before_ratio_filter": df.attrs["interval_exclusions"],
        "primary_inference": {
            "label": "Primary spheroid-aware inference",
            "method": "GEE Gaussian, exchangeable working correlation, robust sandwich SE",
            "formula": "relative_change_pct ~ treated_prev + day_c + C(strategy)",
            "cluster": "Replicate",
            "n_spheroids": int(mix["Replicate"].nunique()),
            "n_obs": int(len(mix)),
            "interval_length_counts": {
                str(int(days)): int(count)
                for days, count in mix["interval_days"].value_counts().sort_index().items()
            },
            "coefficient_units": "percentage points per day",
            "coefficient_pct_points": float(gee_res.params["treated_prev"]),
            "ci_low": float(gee_ci[0]),
            "ci_high": float(gee_ci[1]),
            "p": float(gee_res.pvalues["treated_prev"]),
            "interpretation": "Adjusted treated-minus-untreated difference in resistant-fluorescence percentage change per day.",
        },
        "descriptive_pooled_summary": {
            "label": "Descriptive pooled summary only",
            "method": "Mann-Whitney U (not primary inference)",
            "n_treat": int(pooled_treat.size),
            "n_no": int(pooled_no.size),
            "median_treat_pct": float(np.median(pooled_treat)),
            "median_no_pct": float(np.median(pooled_no)),
            "u": float(pooled_u),
            "p": float(pooled_p),
        },
        "sensitivity_analysis": {
            "label": "Paired spheroid-level sensitivity analysis",
            "method": "Within-spheroid mean difference with Wilcoxon signed-rank test",
            "n_spheroids": int(len(paired)),
            "mean_diff_pct_points": float(paired_mean),
            "median_diff_pct_points": float(np.median(paired["diff"])),
            "ci_low": float(paired_lo),
            "ci_high": float(paired_hi),
            "p": float(paired_w.pvalue),
        },
    }


def plot_treatment_effect_on_resistant_growth(
    df: pd.DataFrame,
    primary_p: float,
) -> None:
    df = df[df["strategy"].isin(MIXED_STRATEGIES)].copy()

    df["treatment"] = df["treatment"].map(
        {
            1.0: "Treatment throughout interval",
            2.0: "No treatment throughout interval",
        }
    )
    df = df[
        df["treatment"].isin(
            ["Treatment throughout interval", "No treatment throughout interval"]
        )
    ].copy()

    data_no = df.loc[
        df["treatment"] == "No treatment throughout interval", "relative_change_pct"
    ].to_numpy()
    data_yes = df.loc[
        df["treatment"] == "Treatment throughout interval", "relative_change_pct"
    ].to_numpy()

    positions = [0, 1]

    with use_style("figstyle.mplstyle"):
        fig, ax = new_figure(width="single", ratio=1.2)

        bp = ax.boxplot(
            [data_no, data_yes],
            positions=positions,
            widths=0.6,
            showfliers=False,
            patch_artist=True,
        )

        # Keep a light-vs-dark ordering while avoiding a true black fill that
        # blends into the box outlines and median line.
        color_no = NEUTRAL_STATE_COLORS["inactive"]
        color_yes = NEUTRAL_STATE_COLORS["active"]
        for patch, col in zip(bp["boxes"], [color_no, color_yes]):
            patch.set(facecolor=col, alpha=0.8, edgecolor="black", linewidth=1.0)

        for element in ["whiskers", "caps", "medians"]:
            for item in bp[element]:
                item.set(color="black", linewidth=1.0)

        ax.set_xticks(positions)
        ax.set_xticklabels(["No Treatment", "Treatment"])
        ax.set_xlabel("")
        ax.set_ylabel("Relative resistant-fluorescence change [% per day]")
        ax.set_ylim(-30, 70)

        stars = p_to_stars(primary_p)
        y = 56
        h = 2
        ax.plot(
            [positions[0], positions[0], positions[1], positions[1]],
            [y, y + h, y + h, y],
            color="black",
            linewidth=SECONDARY_LINEWIDTH,
        )
        ax.text(
            0.5 * (positions[0] + positions[1]),
            y + h + 1,
            stars,
            ha="center",
            va="bottom",
            fontsize=ANNOTATION_FONTSIZE,
        )

        savefig(fig, OUT_PDF, dpi=300, transparent=True)
        savefig(fig, OUT_PDF.with_suffix(".png"), dpi=600, transparent=False, facecolor="white")
        plt.show()
        plt.close(fig)
        print(f"Saved treatment-effect figure to: {OUT_PDF}")


def main() -> None:
    xls = pd.ExcelFile(DATA_XLS)
    data_df_plate1 = pd.read_excel(xls, "Plate 1")
    data_df_plate2 = pd.read_excel(xls, "Plate 2")
    treatment_df_plate1 = pd.read_excel(xls, "Plate 1 Treatment")
    treatment_df_plate2 = pd.read_excel(xls, "Plate 2 Treatment")

    norm1 = extract_and_normalize_remove_artifacts(data_df_plate1)
    norm2 = extract_and_normalize_remove_artifacts(data_df_plate2)

    analysis_df = build_analysis_df(norm1, norm2, treatment_df_plate1, treatment_df_plate2)
    stats_summary = summarize_statistics(analysis_df)

    primary = stats_summary["primary_inference"]
    print("[Primary spheroid-aware inference]")
    print(
        f"GEE, clustered by spheroid: treated_prev = {primary['coefficient_pct_points']:.3f} pp/day, "
        f"95% CI [{primary['ci_low']:.3f}, {primary['ci_high']:.3f}], "
        f"p = {primary['p']:.3e}, n_obs = {primary['n_obs']}, "
        f"n_spheroids = {primary['n_spheroids']}"
    )
    print(f"Primary-model interval lengths (days: count): {primary['interval_length_counts']}")
    for scope, counts in stats_summary["interval_exclusions_before_ratio_filter"].items():
        print(f"[{scope}] Before ratio filtering: {counts['formed_intervals']} intervals; "
              f"excluded mixed = {counts['mixed']}, unknown = {counts['unknown']}")

    pooled = stats_summary["descriptive_pooled_summary"]
    print("[Descriptive pooled summary only]")
    print(
        f"Raw pooled comparison retained only as display: n_treat = {pooled['n_treat']}, "
        f"n_no = {pooled['n_no']}, medians = {pooled['median_treat_pct']:.3f}% vs "
        f"{pooled['median_no_pct']:.3f}% per day, "
        f"Mann-Whitney U = {pooled['u']:.3f}, p = {pooled['p']:.3e}"
    )

    sens = stats_summary["sensitivity_analysis"]
    print("[Sensitivity analysis]")
    print(
        f"Within-spheroid paired difference = {sens['mean_diff_pct_points']:.3f} pp/day "
        f"(median {sens['median_diff_pct_points']:.3f}), "
        f"95% CI [{sens['ci_low']:.3f}, {sens['ci_high']:.3f}], "
        f"Wilcoxon p = {sens['p']:.3e}, n = {sens['n_spheroids']}"
    )

    OUT_STATS.write_text(json.dumps(stats_summary, indent=2), encoding="utf-8")
    plot_treatment_effect_on_resistant_growth(
        analysis_df,
        primary_p=primary["p"],
    )


if __name__ == "__main__":
    main()
