#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase diagram (scatter only) colored by resistant growth.

- Reads ME_A2-3 Data.xlsx (Plate 1 & 2, + treatment sheets)
- Uses de-artifacted normalized data (same as Figure 7 scripts)
- Computes per-day relative resistant growth
- Plots:
    (1) All strategies combined (scatter colored by rel. growth [%])
    (2) One scatter per strategy (IT, AT100, AT80, AT50, CT, NT)

Formatting:
- use_style("figstyle.mplstyle")
- width = 16 cm (thesis)
- no tight_layout() or constrained_layout()
- savefig(..., dpi=300, transparent=True)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import gaussian_kde  # (not strictly needed here, but ok to keep)
from figtools import use_style, set_size, savefig  # uses your figstyle.mplstyle

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

FILE_PATH = Path(
    r"Y:\Members\MaxE\Thesis\Organized Figures\Figure 4.11 Global Competition\ME_A2-3 Data.xlsx"
)
OUT_DIR = FILE_PATH.parent  # figures will be saved next to the Excel file

# ---------------------------------------------------------------------
# Sample groups (same as original script)
# ---------------------------------------------------------------------

sample_groups = {
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
# Data extraction / normalization helpers (unchanged logic)
# ---------------------------------------------------------------------

def extract_and_normalize(data_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract g, r, t from timecourse Excel sheet and normalize by t at Day 0.
    Same logic as in the original Test.py.
    """
    extracted_data = {"Replicate": [], "Day": [], "g": [], "r": [], "t": []}

    replicate_columns = range(3, data_df.shape[1], 5)
    g_columns = range(3, data_df.shape[1], 5)
    r_columns = range(5, data_df.shape[1], 5)
    t_columns = range(6, data_df.shape[1], 5)

    # Extract days from first column, skipping headers
    days = data_df.iloc[3:, 0].dropna().astype(int).unique()

    for rep_col, g_col, r_col, t_col in zip(replicate_columns, g_columns, r_columns, t_columns):
        replicate = data_df.iloc[1, rep_col]
        for day in days:
            day_data = data_df.loc[data_df.iloc[:, 0] == day]
            if day_data.empty:
                continue
            extracted_data["Replicate"].append(replicate)
            extracted_data["Day"].append(day)
            extracted_data["g"].append(day_data.iloc[0, g_col])
            extracted_data["r"].append(day_data.iloc[0, r_col])
            extracted_data["t"].append(day_data.iloc[0, t_col])

    extracted_df = pd.DataFrame(extracted_data)

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
        rep = row["Replicate"]
        baseline_t = baseline_t_values.get(rep, None)
        if baseline_t is None:
            continue
        normalized_data["Replicate"].append(rep)
        normalized_data["Day"].append(row["Day"])
        normalized_data["g_normalized"].append(row["g"] / baseline_t)
        normalized_data["r_normalized"].append(row["r"] / baseline_t)
        normalized_data["t_normalized"].append(row["t"] / baseline_t)

    return pd.DataFrame(normalized_data)


def extract_and_normalize_remove_artifacts(data_df: pd.DataFrame) -> pd.DataFrame:
    """
    De-artifacted extraction (same crude corrections as original).
    """
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

            g_val = day_data.iloc[0, g_col]
            r_val = day_data.iloc[0, r_col]

            if day in range(-2, 6):
                g_val -= 40_000_000
                r_val -= 40_000_000
            elif day == 6:
                g_val -= 5_000_000
                r_val -= 5_000_000

            # subtract r'/12 from g
            g_val -= r_val / 12.0

            extracted_data["Replicate"].append(replicate)
            extracted_data["Day"].append(day)
            extracted_data["g"].append(g_val)
            extracted_data["r"].append(r_val)
            extracted_data["t"].append(day_data.iloc[0, t_col])

    extracted_df = pd.DataFrame(extracted_data)

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
        rep = row["Replicate"]
        baseline_t = baseline_t_values.get(rep, None)
        if baseline_t is None:
            continue

        normalized_data["Replicate"].append(rep)
        normalized_data["Day"].append(row["Day"])
        normalized_data["g_normalized"].append(row["g"] / baseline_t)
        normalized_data["r_normalized"].append(row["r"] / baseline_t)
        normalized_data["t_normalized"].append(row["t"] / baseline_t)

    return pd.DataFrame(normalized_data)


def combine_normalized_with_treatment(
    normalized_df: pd.DataFrame, treatment_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Attach treatment codes to normalized time series.
    """
    treatment_days = treatment_df.iloc[:, 0].dropna().unique()
    replicate_columns = treatment_df.columns[2:]

    treatment_dict = {}
    for _, row in treatment_df.iterrows():
        day_str = row.iloc[0]
        if day_str not in treatment_days:
            continue
        day = int(day_str[1:])  # 'd-2' -> -2, 'd10' -> 10 (as in original)
        for rep in replicate_columns:
            treatment_dict[(day, rep)] = row[rep]

    normalized_df = normalized_df.copy()
    normalized_df["treatment"] = normalized_df.apply(
        lambda row: treatment_dict.get((row["Day"], row["Replicate"]), 0),
        axis=1,
    )
    return normalized_df


# ---------------------------------------------------------------------
# Phase scatter preparation
# ---------------------------------------------------------------------

def prepare_phase_dataframe(
    df1: pd.DataFrame, df2: pd.DataFrame
) -> pd.DataFrame:
    """
    Take combined_df_plate1_deartifacted, combined_df_plate2_deartifacted,
    compute relative resistant growth and restrict to on-study days etc.
    Returns a DF ready for phase scatter plots.
    """
    df1 = df1.copy()
    df2 = df2.copy()

    # prefix replicate IDs to distinguish plates
    df1["Replicate"] = df1["Replicate"].apply(lambda x: "1" + x)
    df2["Replicate"] = df2["Replicate"].apply(lambda x: "2" + x)

    df = pd.concat([df1, df2], ignore_index=True)

    # per-day relative change of resistant population
    df["relative_change"] = (
        df.groupby("Replicate")["r_normalized"]
        .pct_change(fill_method=None)
        + 1.0
    )

    df = df[
        [
            "Replicate",
            "Day",
            "r_normalized",
            "g_normalized",
            "treatment",
            "relative_change",
        ]
    ]

    # On-study days only, remove weird states
    df = df[~df["treatment"].isin([0.0, 3.0, 4.0, 5.0])]
    df = df.dropna(subset=["relative_change"])

    # Restrict to reasonable ratios
    df = df[df["relative_change"].between(0.75, 2.0)]

    # Relative growth in percent, clipped to [-10, 30]
    df["rel_growth_pct"] = (df["relative_change"] - 1.0) * 100.0
    df["rel_growth_pct_clipped"] = df["rel_growth_pct"].clip(-10, 30)

    # For log y-axis, avoid zero
    epsilon = 1e-6
    df["r_plot"] = df["r_normalized"].clip(lower=epsilon)

    # Map replicates to treatment groups
    included_long = [
        "Intermittent",
        "Adaptive Therapy 100",
        "Adaptive Therapy 80",
        "Adaptive Therapy 50",
        "Maximum-tolerable-dose",
        "No treatment",
    ]
    rep_to_long = {}
    for plate_prefix in ["1", "2"]:
        for long_name, wells in sample_groups.items():
            if long_name not in included_long:
                continue
            for well in wells:
                rep_to_long[plate_prefix + well] = long_name

    df["TreatmentLong"] = df["Replicate"].map(rep_to_long)
    df = df[~df["TreatmentLong"].isna()].copy()

    long_to_short = {
        "No treatment": "NT",
        "Maximum-tolerable-dose": "CT",
        "Adaptive Therapy 100": "AT100",
        "Adaptive Therapy 80": "AT80",
        "Adaptive Therapy 50": "AT50",
        "Intermittent": "IT",
    }
    df["TreatmentShort"] = df["TreatmentLong"].map(long_to_short)

    return df


# ---------------------------------------------------------------------
# Plotting: Scatter phase diagrams (overall + per strategy)
# ---------------------------------------------------------------------

def plot_phase_scatter_all_and_per_strategy(df_phase: pd.DataFrame) -> None:
    """
    Make:
      - one scatter phase diagram with all strategies combined,
      - one per strategy (IT, AT100, AT80, AT50, CT, NT) if present.
    """
    with use_style("figstyle.mplstyle"):
        # -------------------------
        # Overall (all strategies)
        # -------------------------
        fig, ax = plt.subplots()
        set_size(fig, width="thesis", ratio=0.7)
        fig.subplots_adjust(left=0.12, right=0.98, bottom=0.16, top=0.92)

        sc = ax.scatter(
            df_phase["g_normalized"],
            df_phase["r_plot"],
            c=df_phase["rel_growth_pct_clipped"],
            s=8,
            alpha=0.7,
            vmin=-10,
            vmax=30,
            cmap="coolwarm",
        )

        ax.set_xlabel("Normalized susceptible population")
        ax.set_ylabel("Normalized resistant population")
        ax.set_title("Phase diagram\nAll strategies combined")
        ax.set_xlim(0, 4)
        ax.set_ylim(0.02, 5)
        ax.set_yscale("log")

        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("Relative resistant growth [%]")

        out_all = OUT_DIR / "PhaseDiagram_scatter_all_strategies.pdf"
        savefig(fig, out_all, dpi=300, transparent=True)
        print(f"[saved] {out_all}")
        plt.show()
        plt.close(fig)

        # -------------------------
        # Per strategy
        # -------------------------
        order_short = ["IT", "AT100", "AT80", "AT50", "CT", "NT"]
        present_short = [
            s
            for s in order_short
            if s in df_phase["TreatmentShort"].dropna().unique()
        ]

        for short in present_short:
            sub = df_phase[df_phase["TreatmentShort"] == short]
            if sub.empty:
                continue

            fig, ax = plt.subplots()
            set_size(fig, width="thesis", ratio=0.7)
            fig.subplots_adjust(
                left=0.12, right=0.98, bottom=0.16, top=0.92
            )

            sc = ax.scatter(
                sub["g_normalized"],
                sub["r_plot"],
                c=sub["rel_growth_pct_clipped"],
                s=8,
                alpha=0.7,
                vmin=-10,
                vmax=30,
                cmap="coolwarm",
            )

            ax.set_xlabel("Normalized susceptible population")
            ax.set_ylabel("Normalized resistant population")
            ax.set_title(f"Phase diagram\n{short}")
            ax.set_xlim(0, 4)
            ax.set_ylim(0.02, 5)
            ax.set_yscale("log")

            cbar = fig.colorbar(sc, ax=ax)
            cbar.set_label("Relative resistant growth [%]")

            out_strat = OUT_DIR / f"PhaseDiagram_scatter_{short}.pdf"
            savefig(fig, out_strat, dpi=300, transparent=True)
            plt.show()
            print(f"[saved] {out_strat}")
            plt.close(fig)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    # Load Excel
    xls = pd.ExcelFile(FILE_PATH)

    data_df_plate1 = pd.read_excel(xls, "Plate 1")
    data_df_plate2 = pd.read_excel(xls, "Plate 2")
    treatment_df_plate1 = pd.read_excel(xls, "Plate 1 Treatment")
    treatment_df_plate2 = pd.read_excel(xls, "Plate 2 Treatment")

    # De-artifacted normalized data
    norm1_deart = extract_and_normalize_remove_artifacts(data_df_plate1)
    norm2_deart = extract_and_normalize_remove_artifacts(data_df_plate2)

    # Attach treatment
    combined1 = combine_normalized_with_treatment(norm1_deart, treatment_df_plate1)
    combined2 = combine_normalized_with_treatment(norm2_deart, treatment_df_plate2)

    # Prepare phase DF
    df_phase = prepare_phase_dataframe(combined1, combined2)

    # Plot overall + per strategy
    plot_phase_scatter_all_and_per_strategy(df_phase)

    print("[done] Phase scatter diagrams generated.")


if __name__ == "__main__":
    main()
