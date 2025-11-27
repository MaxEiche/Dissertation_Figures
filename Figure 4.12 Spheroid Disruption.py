#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Figure 4.12 – Resistant growth before vs. after spheroid disruption

Standalone script that:
- Loads ME_A2-3 spheroid data (Plate 1 + 2) from the Excel file.
- Applies the same artifact-corrected normalization as in Figure_7.
- Merges in treatment codes for each day.
- Adds spheroid disruption (“fallapart”) info.
- Computes per-day resistant growth and aligns days relative to disruption.
- Builds two groups: "Before disruption" vs "After disruption".
- Plots a half-width ("single") boxplot of relative resistant growth [%].

Formatting:
- with use_style("figstyle.mplstyle", font_family="Arial")
- new_figure(width="single", ratio=0.7)
- no tight_layout() / constrained_layout()
- colors only via figtools.therapy_color(...)
- no replicate filtering via EXCLUDE_SAMPLES
- no seaborn
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

from figtools import use_style, new_figure, savefig, therapy_color


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

DATA_XLS = Path(
    r"Y:\Members\MaxE\Thesis\Organized Figures\Figure 4.12 Spheroid Disruption\ME_A2-3 Data.xlsx"
)
OUT_PDF = (
    DATA_XLS.parent
    / "Figure 4.12 ResGrowth_after_SpheroidDisruption.pdf"
)


# ---------------------------------------------------------------------
# Sample layout (for excluding resistant-control wells, same as Figure_7)
# ---------------------------------------------------------------------

sample_groups: Dict[str, List[str]] = {
    "No treatment": ["A1", "A2", "A3", "A4", "A5", "A6"],
    "Maximum-tolerable-dose": ["A7", "A8", "A9", "A10", "A11", "A12"],
    "Adaptive Therapy 100": [
        "B1",
        "B2",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7",
        "B8",
        "B9",
        "B10",
        "B11",
        "C1",
        "C2",
        "C3",
        "C4",
        "C5",
        "C6",
        "C7",
        "C8",
        "C9",
        "C10",
        "C11",
    ],
    "Adaptive Therapy 80": [
        "D1",
        "D2",
        "D3",
        "D4",
        "D5",
        "D6",
        "D7",
        "D8",
        "D9",
        "D10",
        "D11",
        "E1",
        "E2",
        "E3",
        "E4",
        "E5",
        "E6",
        "E7",
        "E8",
        "E9",
        "E10",
        "E11",
    ],
    "Adaptive Therapy 50": [
        "F1",
        "F2",
        "F3",
        "F4",
        "F5",
        "F6",
        "F7",
        "F8",
        "F9",
        "F10",
        "F11",
        "G1",
        "G2",
        "G3",
        "G4",
        "G5",
        "G6",
        "G7",
        "G8",
        "G9",
        "G10",
        "G11",
    ],
    "Resistant Control No Treatment": ["H1", "H2", "H3", "H4", "H5", "H6"],
    "Resistant Control Maximum-tolerable-dose": ["H7", "H8", "H9", "H10", "H11", "H12"],
    "Intermittent": ["B12", "C12", "D12", "E12", "F12", "G12"],
}


# ---------------------------------------------------------------------
# Core data processing (same artifact logic as in Figure_7)
# ---------------------------------------------------------------------

def extract_and_normalize_remove_artifacts(data_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract g / r / total and normalize to baseline (Day 0),
    applying the same crude artifact corrections as in Figure_7.
    """
    extracted_data = {"Replicate": [], "Day": [], "g": [], "r": [], "t": []}

    # Every fifth column: replicate label, g, r, t pattern
    replicate_columns = range(3, data_df.shape[1], 5)
    g_columns = range(3, data_df.shape[1], 5)
    r_columns = range(5, data_df.shape[1], 5)
    t_columns = range(6, data_df.shape[1], 5)

    # Day labels from first column (row ≥ 3)
    days = data_df.iloc[3:, 0].dropna().astype(int).unique()

    for rep_col, g_col, r_col, t_col in zip(
        replicate_columns, g_columns, r_columns, t_columns
    ):
        replicate = data_df.iloc[1, rep_col]
        for day in days:
            # Skip artifact days exactly as in Figure_7
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

    # Normalize to baseline total at Day 0
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
    Attach a numeric 'treatment' code to each (Day, Replicate) entry.

    treatment codes (from Figure_7):
        1 = treatment
        2 = on-study (no treatment)
        3,4,5 = death / dropout
        0 = after dropout
    """
    treatment_days = treatment_df.iloc[:, 0].dropna().unique()
    replicate_columns = treatment_df.columns[2:]

    treatment_dict: Dict[Tuple[int, str], float] = {}
    for _, row in treatment_df.iterrows():
        day_str = row.iloc[0]
        if day_str not in treatment_days:
            continue
        # Expect strings like "d-2", "d0", ...
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
# Fallapart / disruption info (as in Figure_7)
# ---------------------------------------------------------------------

def add_fallapart_info(
    combined_df: pd.DataFrame,
    fallapart_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add a 'fallapart_info' column indicating spheroid disruption (code 6)
    based on the Plate X Treatment_fallapart sheets.
    """
    fallapart_days = fallapart_df.iloc[:, 0].dropna().unique()
    replicate_columns = fallapart_df.columns[2:]

    fallapart_dict: Dict[Tuple[int, str], float] = {}
    for _, row in fallapart_df.iterrows():
        day_str = row.iloc[0]
        if day_str not in fallapart_days:
            continue
        day = int(day_str[1:])  # "d-2", "d0", ...
        for replicate in replicate_columns:
            fallapart_dict[(day, replicate)] = row[replicate]

    combined_df = combined_df.copy()
    combined_df["fallapart_info"] = combined_df.apply(
        lambda r: 6 if fallapart_dict.get((r["Day"], r["Replicate"]), 0) == 6 else 0,
        axis=1,
    )
    return combined_df


# ---------------------------------------------------------------------
# Plot: resistant growth before vs after disruption
# ---------------------------------------------------------------------

def plot_res_growth_after_spheroid_disruption(
    df1_fall: pd.DataFrame,
    df2_fall: pd.DataFrame,
) -> None:
    """
    Half-width boxplot: resistant growth before vs. after spheroid disruption.

    Logic follows plot_fig7_fallapart_single from Figure_7:
    - prefix replicates by plate
    - compute per-day relative_change of r_normalized
    - keep only on-study days (treatment 1 or 2)
    - restrict relative_change to [0.75, 2]
    - align days relative to first fallapart day per replicate
    - group into "Before" vs "After" based on adjusted_day < 0 / ≥ 0
    - exclude resistant-control samples
    """

    df1 = df1_fall.copy()
    df2 = df2_fall.copy()

    # Prefix replicates with plate ID
    df1["Replicate"] = df1["Replicate"].apply(lambda x: f"1{x}")
    df2["Replicate"] = df2["Replicate"].apply(lambda x: f"2{x}")

    df = pd.concat([df1, df2], ignore_index=True)

    # Per-day relative change of resistant population
    df["relative_change"] = (
        df.groupby("Replicate")["r_normalized"].pct_change(fill_method=None) + 1.0
    )

    # Keep relevant columns (fallapart_info already present)
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
        ]
    ]

    # Remove death / dropout / after-dropout states
    df = df[~df["treatment"].isin([0.0, 3.0, 4.0, 5.0])]

    # Remove rows with NaN growth
    df = df.dropna(subset=["relative_change"])

    # Restrict to reasonable range (same as other growth plots)
    df = df[df["relative_change"].between(0.75, 2.0)]

    # Compute adjusted_day relative to fallapart:
    #  day 0 = first fallapart_info==6; if none, last day of replicate
    df["adjusted_day"] = np.nan
    for rep in df["Replicate"].unique():
        rep_df = df[df["Replicate"] == rep]
        fall_day = rep_df[rep_df["fallapart_info"] == 6]["Day"].min()
        if pd.notna(fall_day):
            df.loc[df["Replicate"] == rep, "adjusted_day"] = (
                df.loc[df["Replicate"] == rep, "Day"] - fall_day
            )
        else:
            last_day = rep_df["Day"].max()
            df.loc[df["Replicate"] == rep, "adjusted_day"] = (
                df.loc[df["Replicate"] == rep, "Day"] - last_day
            )

    # Adjusted sample groups to add plate prefixes (same logic as in Figure_7)
    adjusted_sample_groups: Dict[str, List[str]] = {}
    for sample, reps in sample_groups.items():
        # Original code: prefix with '1' for letter-start replicates
        adjusted_reps = ["1" + r if r[0].isalpha() else "2" + r for r in reps]
        adjusted_sample_groups[sample] = adjusted_reps

    # Exclude resistant-control wells
    excluded_samples = [
        "Resistant Control No Treatment",
        "Resistant Control Maximum-tolerable-dose",
    ]
    excluded_reps = [
        rep for sample in excluded_samples for rep in adjusted_sample_groups[sample]
    ]
    df = df[~df["Replicate"].isin(excluded_reps)]

    # Prepare plotting dataframe
    df_plot = df.copy()
    df_plot["relative_change_pct"] = (df_plot["relative_change"] - 1.0) * 100.0
    df_plot["period2"] = np.where(
        df_plot["adjusted_day"] < 0, "Before disruption", "After disruption"
    )

    # Mann–Whitney U test on raw ratios (not percent)
    before_vals = df_plot.loc[
        df_plot["period2"] == "Before disruption", "relative_change"
    ]
    after_vals = df_plot.loc[
        df_plot["period2"] == "After disruption", "relative_change"
    ]
    stat, p = mannwhitneyu(before_vals, after_vals, alternative="two-sided")
    print(f"Fallapart MWU: stat = {stat:.3f}, p = {p:.2E}")

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

    # ---- Plot ----
    with use_style("figstyle.mplstyle", font_family="Arial"):
        fig, ax = new_figure(width=60, ratio=1.2)

        # Data for the two boxes
        order = ["Before disruption", "After disruption"]
        data_before = df_plot.loc[
            df_plot["period2"] == "Before disruption", "relative_change_pct"
        ].to_numpy()
        data_after = df_plot.loc[
            df_plot["period2"] == "After disruption", "relative_change_pct"
        ].to_numpy()
        positions = [0, 1]

        # Boxplot
        bp = ax.boxplot(
            [data_before, data_after],
            positions=positions,
            widths=0.6,
            showfliers=False,
            patch_artist=True,
        )

        # Colors from therapy palette
        # "Before disruption" -> NT, "After disruption" -> CT
        color_before = therapy_color("NT")
        color_after = therapy_color("CT")

        for patch, col in zip(bp["boxes"], [color_before, color_after]):
            patch.set(facecolor=col, alpha=0.8, edgecolor="black", linewidth=1.0)

        # Whiskers, caps, medians in black
        for element in ["whiskers", "caps", "medians"]:
            for item in bp[element]:
                item.set(color="black", linewidth=1.0)

        # X tick labels
        ax.set_xticks(positions)
        ax.set_xticklabels(["Before Disruption", "After Disruption"])

        # Axis labels and limits
        ax.set_xlabel("")
        ax.set_ylabel("Relative Res. Growth [%]")
        ax.set_ylim(-30, 70)

        # Significance bracket + stars
        ymin, ymax = ax.get_ylim()
        y = ymax - 8.0  # baseline a bit below top
        h = 3.0
        ax.plot(
            [positions[0], positions[0], positions[1], positions[1]],
            [y, y + h, y + h, y],
            color="black",
            linewidth=1.0,
        )
        ax.text(
            0.5 * (positions[0] + positions[1]),
            y + h + 0.5,
            stars,
            ha="center",
            va="bottom",
        )

        # Aesthetics: no top/right spines, ticks outward
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.tick_params(direction="out")

        # No tight_layout / constrained_layout
        savefig(fig, OUT_PDF, dpi=300, transparent=True)
        plt.show()
        plt.close(fig)
        print(f"Saved fallapart figure to: {OUT_PDF}")


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

    # Fallapart sheets
    fallapart_df_plate1 = pd.read_excel(xls, "Plate 1 Treatment_fallapart")
    fallapart_df_plate2 = pd.read_excel(xls, "Plate 2 Treatment_fallapart")

    # Artifact-corrected normalization (same as Figure_7)
    norm1 = extract_and_normalize_remove_artifacts(data_df_plate1)
    norm2 = extract_and_normalize_remove_artifacts(data_df_plate2)

    # Add treatment codes
    comb1 = combine_normalized_with_treatment(norm1, treatment_df_plate1)
    comb2 = combine_normalized_with_treatment(norm2, treatment_df_plate2)

    # Add fallapart info
    comb1_fall = add_fallapart_info(comb1, fallapart_df_plate1)
    comb2_fall = add_fallapart_info(comb2, fallapart_df_plate2)

    # Plot
    plot_res_growth_after_spheroid_disruption(comb1_fall, comb2_fall)


if __name__ == "__main__":
    main()
