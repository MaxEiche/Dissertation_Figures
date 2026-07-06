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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon
import statsmodels.api as sm
import statsmodels.formula.api as smf

from figtools import NEUTRAL_STATE_COLORS, new_figure, savefig, use_style


DATA_XLS = Path(
    r"Y:\Members\MaxE\Thesis\Organized Figures\Chapter_4\Figure 4.9 Treatment effect on res growth\ME_A2-3 Data.xlsx"
)
OUT_PDF = DATA_XLS.parent / "Figure 4.9 TreatmentEffect_on_ResistantGrowth.pdf"
OUT_STATS = DATA_XLS.parent / "Figure 4.9 TreatmentEffect_on_ResistantGrowth_stats.json"

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


def build_analysis_df(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    df1 = df1.copy()
    df2 = df2.copy()
    df1["Replicate"] = df1["Replicate"].apply(lambda x: f"1{x}")
    df2["Replicate"] = df2["Replicate"].apply(lambda x: f"2{x}")
    df = pd.concat([df1, df2], ignore_index=True)
    df["strategy"] = df["Replicate"].map(build_group_map())
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
            "strategy",
        ]
    ]
    df = df[~df["treatment"].isin([0.0, 3.0, 4.0, 5.0])]
    df = df.dropna(subset=["relative_change"])
    df = df[df["relative_change"].between(0.75, 2.0)].copy()
    df["relative_change_pct"] = (df["relative_change"] - 1.0) * 100.0
    df["treated_prev"] = (df["treatment"] == 1.0).astype(int)
    df["day_c"] = df["Day"] - df["Day"].mean()
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
    pooled_treat = df[df["treatment"] == 1.0]["relative_change"]
    pooled_no = df[df["treatment"] == 2.0]["relative_change"]
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
                "Excluded treatment codes 0, 3, 4, 5",
                "Restricted relative_change to [0.75, 2.0]",
            ],
        },
        "primary_inference": {
            "label": "Primary spheroid-aware inference",
            "method": "GEE Gaussian, exchangeable working correlation, robust sandwich SE",
            "formula": "relative_change_pct ~ treated_prev + day_c + C(strategy)",
            "cluster": "Replicate",
            "n_spheroids": int(mix["Replicate"].nunique()),
            "n_obs": int(len(mix)),
            "coefficient_pct_points": float(gee_res.params["treated_prev"]),
            "ci_low": float(gee_ci[0]),
            "ci_high": float(gee_ci[1]),
            "p": float(gee_res.pvalues["treated_prev"]),
            "interpretation": "Recent treatment exposure was associated with higher resistant growth in non-embedded spheroids.",
        },
        "descriptive_pooled_summary": {
            "label": "Descriptive pooled summary only",
            "method": "Mann-Whitney U (not primary inference)",
            "n_treat": int(pooled_treat.size),
            "n_no": int(pooled_no.size),
            "median_treat_pct": float(np.median((pooled_treat - 1.0) * 100.0)),
            "median_no_pct": float(np.median((pooled_no - 1.0) * 100.0)),
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
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    primary_p: float,
) -> None:
    df = build_analysis_df(df1, df2)
    df = df[df["strategy"].isin(MIXED_STRATEGIES)].copy()

    df["treatment"] = df["treatment"].map(
        {
            1.0: "Treatment the day before",
            2.0: "No treatment the day before",
        }
    )
    df = df[
        df["treatment"].isin(
            ["Treatment the day before", "No treatment the day before"]
        )
    ].copy()

    data_no = df.loc[
        df["treatment"] == "No treatment the day before", "relative_change_pct"
    ].to_numpy()
    data_yes = df.loc[
        df["treatment"] == "Treatment the day before", "relative_change_pct"
    ].to_numpy()

    positions = [0, 1]

    with use_style("figstyle.mplstyle", font_family="Arial"):
        fig, ax = new_figure(width=60, ratio=1.2)

        bp = ax.boxplot(
            [data_no, data_yes],
            positions=positions,
            widths=0.6,
            showfliers=False,
            patch_artist=True,
        )

        # Keep a light-vs-dark ordering while avoiding a true black fill that
        # blends into the box outlines and median line.
        color_no = "#C7C7C7"
        color_yes = "#4A4A4A"
        for patch, col in zip(bp["boxes"], [color_no, color_yes]):
            patch.set(facecolor=col, alpha=0.8, edgecolor="black", linewidth=1.0)

        for element in ["whiskers", "caps", "medians"]:
            for item in bp[element]:
                item.set(color="black", linewidth=1.0)

        ax.set_xticks(positions)
        ax.set_xticklabels(["No Treatment", "Treatment"])
        ax.set_xlabel("")
        ax.set_ylabel("Relative Res. Growth [%]")
        ax.set_ylim(-30, 70)

        stars = p_to_stars(primary_p)
        y = 56
        h = 2
        ax.plot(
            [positions[0], positions[0], positions[1], positions[1]],
            [y, y + h, y + h, y],
            color="black",
            linewidth=1.0,
        )
        ax.text(
            0.5 * (positions[0] + positions[1]),
            y + h + 1,
            stars,
            ha="center",
            va="bottom",
        )

        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.tick_params(direction="out")

        savefig(fig, OUT_PDF, dpi=300, transparent=True)
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

    comb1 = combine_normalized_with_treatment(norm1, treatment_df_plate1)
    comb2 = combine_normalized_with_treatment(norm2, treatment_df_plate2)

    stats_summary = summarize_statistics(build_analysis_df(comb1, comb2))

    primary = stats_summary["primary_inference"]
    print("[Primary spheroid-aware inference]")
    print(
        f"GEE, clustered by spheroid: treated_prev = {primary['coefficient_pct_points']:.3f} pp "
        f"[{primary['ci_low']:.3f}, {primary['ci_high']:.3f}], "
        f"p = {primary['p']:.3e}, n_spheroids = {primary['n_spheroids']}"
    )

    pooled = stats_summary["descriptive_pooled_summary"]
    print("[Descriptive pooled summary only]")
    print(
        f"Raw pooled comparison retained only as display: n_treat = {pooled['n_treat']}, "
        f"n_no = {pooled['n_no']}, medians = {pooled['median_treat_pct']:.3f}% vs "
        f"{pooled['median_no_pct']:.3f}%"
    )

    sens = stats_summary["sensitivity_analysis"]
    print("[Sensitivity analysis]")
    print(
        f"Within-spheroid paired difference = {sens['mean_diff_pct_points']:.3f} pp "
        f"(median {sens['median_diff_pct_points']:.3f}), "
        f"95% CI [{sens['ci_low']:.3f}, {sens['ci_high']:.3f}], "
        f"Wilcoxon p = {sens['p']:.3e}, n = {sens['n_spheroids']}"
    )

    OUT_STATS.write_text(json.dumps(stats_summary, indent=2), encoding="utf-8")
    plot_treatment_effect_on_resistant_growth(
        comb1,
        comb2,
        primary_p=primary["p"],
    )


if __name__ == "__main__":
    main()
