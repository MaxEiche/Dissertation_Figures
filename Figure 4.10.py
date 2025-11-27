#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Standalone figure script: Treatment effect on resistant growth (Figure 7X)

This script:
- Loads ME_A2-3 spheroid data (Plate 1 + 2) from the Excel file.
- Applies the same artifact-corrected normalisation as in Figure_7.
- Merges in treatment codes for each day.
- Computes per-day relative resistant growth depending on whether
  the previous day was on treatment or not.
- Plots a single half-width ("single") boxplot:
      left  = No treatment the day before
      right = Treatment the day before

Formatting rules:
- use_style("figstyle.mplstyle", font_family="Arial")
- use new_figure(width="single", ratio=0.7)
- no tight_layout() or constrained_layout()
- colours only via figtools.therapy_color (NT, CT)
- no replicate filtering via experiment_filters (EXCLUDE_SAMPLES)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

from figtools import use_style, new_figure, savefig, therapy_color


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

DATA_XLS = Path(
    r"Y:\Members\MaxE\Thesis\Organized Figures\Figure 4.10 Treatment effect on res growth\ME_A2-3 Data.xlsx"
)
OUT_PDF = DATA_XLS.parent / "Figure 4.10 TreatmentEffect_on_ResistantGrowth.pdf"


# ---------------------------------------------------------------------
# Core data processing (copied from Figure_7 logic)
# ---------------------------------------------------------------------

def extract_and_normalize_remove_artifacts(data_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract g/r/total and normalize to baseline (Day 0),
    applying the same crude artifact corrections as in Figure_7.
    """
    extracted_data = {"Replicate": [], "Day": [], "g": [], "r": [], "t": []}

    # Every fifth column: replicate, g, r, t pattern as in Figure_7
    replicate_columns = range(3, data_df.shape[1], 5)
    g_columns = range(3, data_df.shape[1], 5)
    r_columns = range(5, data_df.shape[1], 5)
    t_columns = range(6, data_df.shape[1], 5)

    # Day labels from first column
    days = data_df.iloc[3:, 0].dropna().astype(int).unique()

    for rep_col, g_col, r_col, t_col in zip(
        replicate_columns, g_columns, r_columns, t_columns
    ):
        replicate = data_df.iloc[1, rep_col]
        for day in days:
            # Skip artifact days (as in Figure_7)
            if day in (29, 15, 27):
                continue

            day_data = data_df.loc[data_df.iloc[:, 0] == day]
            if day_data.empty:
                continue

            g_value = day_data.iloc[0, g_col]
            r_value = day_data.iloc[0, r_col]

            # Early-day offset corrections
            if day in range(-2, 6):  # -2..5
                g_value -= 40_000_000
                r_value -= 40_000_000
            elif day == 6:
                g_value -= 5_000_000
                r_value -= 5_000_000

            # Additional correction: subtract r/12 from g
            g_value -= r_value / 12.0

            extracted_data["Replicate"].append(replicate)
            extracted_data["Day"].append(day)
            extracted_data["g"].append(g_value)
            extracted_data["r"].append(r_value)
            extracted_data["t"].append(day_data.iloc[0, t_col])

    extracted_df = pd.DataFrame(extracted_data)

    # Normalise to baseline total at Day 0
    normalized_data = {
        "Replicate": [],
        "Day": [],
        "g_normalized": [],
        "r_normalized": [],
        "t_normalized": [],
    }

    baseline_t_values = (
        extracted_df[extracted_df["Day"] == 0].set_index("Replicate")["t"]
    )

    for _, row in extracted_df.iterrows():
        replicate = row["Replicate"]
        baseline_t = baseline_t_values.get(replicate, None)
        if baseline_t is None:
            continue

        normalized_data["Replicate"].append(replicate)
        normalized_data["Day"].append(row["Day"])
        normalized_data["g_normalized"].append(row["g"] / baseline_t)
        normalized_data["r_normalized"].append(row["r"] / baseline_t)
        normalized_data["t_normalized"].append(row["t"] / baseline_t)

    return pd.DataFrame(normalized_data)


def combine_normalized_with_treatment(
    normalized_df: pd.DataFrame,
    treatment_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add a 'treatment' column to the normalized dataframe.

    treatment codes (as in Figure_7):
        1 = treatment
        2 = on-study (no treatment)
        3,4,5 = death / dropout
        0 = after dropout
    """
    treatment_days = treatment_df.iloc[:, 0].dropna().unique()
    replicate_columns = treatment_df.columns[2:]

    treatment_dict = {}
    for _, row in treatment_df.iterrows():
        day_str = row.iloc[0]
        if day_str not in treatment_days:
            continue
        # day strings like "d-2", "d0", ...
        day = int(day_str[1:])
        for replicate in replicate_columns:
            treatment_dict[(day, replicate)] = row[replicate]

    normalized_df = normalized_df.copy()
    normalized_df["treatment"] = normalized_df.apply(
        lambda r: treatment_dict.get((r["Day"], r["Replicate"]), 0),
        axis=1,
    )
    return normalized_df


# ---------------------------------------------------------------------
# Plot: Treatment effect on resistant growth
# ---------------------------------------------------------------------

def plot_treatment_effect_on_resistant_growth(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
) -> None:
    """
    Half-width boxplot: effect of treatment the day before on per-day
    relative resistant growth.

    Data logic identical to Figure_7::plot_fig7_treatment_effect_single,
    but with:
    - figtools.use_style + new_figure(width="single")
    - colours via therapy_color("NT") / therapy_color("CT")
    - no seaborn
    """
    df1 = df1.copy()
    df2 = df2.copy()

    # Prefix replicates by plate
    df1["Replicate"] = df1["Replicate"].apply(lambda x: f"1{x}")
    df2["Replicate"] = df2["Replicate"].apply(lambda x: f"2{x}")
    df = pd.concat([df1, df2], ignore_index=True)

    # Cumulative treatment (kept for consistency with original logic)
    df["cumulative_treatment"] = df.groupby(
        (df["treatment"] != df["treatment"].shift(1)).cumsum()
    ).apply(lambda x: (x["treatment"] == 1.0).cumsum()).reset_index(drop=True)
    df["cumulative_treatment"] = df.groupby("Replicate")["cumulative_treatment"].shift(
        fill_value=0
    )

    # Per-day resistant growth ratio (1.0 = no change, 1.1 = +10%, ...)
    df["relative_change"] = (
        df.groupby("Replicate")["r_normalized"].pct_change(fill_method=None) + 1.0
    )

    # Keep only relevant columns
    df = df[
        [
            "Replicate",
            "Day",
            "r_normalized",
            "t_normalized",
            "g_normalized",
            "treatment",
            "cumulative_treatment",
            "relative_change",
        ]
    ]

    # Exclude death / dropout / after-dropout states
    df = df[~df["treatment"].isin([0.0, 3.0, 4.0, 5.0])]

    # Remove rows with NaN growth
    df = df.dropna(subset=["relative_change"])

    # Exclude extreme outliers (same cut as in original code)
    df = df[df["relative_change"].between(0.75, 2.0)]

    # Split by treatment state on the CURRENT day:
    #   1.0 = treatment the day before the growth interval
    #   2.0 = no treatment the day before
    df_treat_1 = df[df["treatment"] == 1.0]
    df_treat_2 = df[df["treatment"] == 2.0]

    # Map to readable labels and restrict
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

    # Convert to percent growth per day
    df["relative_change_pct"] = (df["relative_change"] - 1.0) * 100.0

    # Mann–Whitney U test on the raw ratios (not percent)
    stat, p = mannwhitneyu(
        df_treat_1["relative_change"],
        df_treat_2["relative_change"],
        alternative="two-sided",
    )
    print(f"Mann–Whitney U: stat = {stat:.3f}, p = {p:.2E}")

    def p_to_stars(pval: float) -> str:
        if pval < 1e-4:
            return "****"
        elif pval < 1e-3:
            return "***"
        elif pval < 0.01:
            return "**"
        elif pval < 0.05:
            return "*"
        else:
            return "n.s."

    stars = p_to_stars(p)

    # Prepare data for boxplot
    data_no = df.loc[
        df["treatment"] == "No treatment the day before", "relative_change_pct"
    ].to_numpy()
    data_yes = df.loc[
        df["treatment"] == "Treatment the day before", "relative_change_pct"
    ].to_numpy()

    # X positions: 0 = No treatment, 1 = Treatment
    positions = [0, 1]

    with use_style("figstyle.mplstyle", font_family="Arial"):
        fig, ax = new_figure(width=60, ratio=1.2)

        # Boxplot (no fliers)
        bp = ax.boxplot(
            [data_no, data_yes],
            positions=positions,
            widths=0.6,
            showfliers=False,
            patch_artist=True,
        )

        # Colours from therapy palette:
        # map "No treatment" -> NT, "Treatment" -> CT
        color_no = therapy_color("NT")
        color_yes = therapy_color("CT")

        for patch, col in zip(bp["boxes"], [color_no, color_yes]):
            patch.set(facecolor=col, alpha=0.8, edgecolor="black", linewidth=1.0)

        # Whiskers, caps, medians in black
        for element in ["whiskers", "caps", "medians"]:
            for item in bp[element]:
                item.set(color="black", linewidth=1.0)

        # X tick labels
        ax.set_xticks(positions)
        ax.set_xticklabels(["No Treatment", "Treatment"])

        # Axis labels and limits
        ax.set_xlabel("")
        ax.set_ylabel("Relative Res. Growth [%]")
        ax.set_ylim(-30, 70)

        # Significance bracket + stars
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

        # Aesthetics: no top/right spines, ticks outward
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.tick_params(direction="out")

        savefig(fig, OUT_PDF, dpi=300, transparent=True)
        plt.show()
        plt.close(fig)
        print(f"Saved treatment-effect figure to: {OUT_PDF}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    # Load Excel
    xls = pd.ExcelFile(DATA_XLS)

    # Raw plate data
    data_df_plate1 = pd.read_excel(xls, "Plate 1")
    data_df_plate2 = pd.read_excel(xls, "Plate 2")

    # Treatment matrices
    treatment_df_plate1 = pd.read_excel(xls, "Plate 1 Treatment")
    treatment_df_plate2 = pd.read_excel(xls, "Plate 2 Treatment")

    # Artifact-corrected normalisation (same as Figure_7)
    norm1 = extract_and_normalize_remove_artifacts(data_df_plate1)
    norm2 = extract_and_normalize_remove_artifacts(data_df_plate2)

    # Add treatment codes
    comb1 = combine_normalized_with_treatment(norm1, treatment_df_plate1)
    comb2 = combine_normalized_with_treatment(norm2, treatment_df_plate2)

    # Plot
    plot_treatment_effect_on_resistant_growth(comb1, comb2)


if __name__ == "__main__":
    main()
