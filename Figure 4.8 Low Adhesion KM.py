#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Figure 4.8 – Kaplan–Meier + fraction of days on treatment (3D spheroids)

Panel (a): KM survival by treatment strategy, using the same event definition
           inherited from the original Figure_7 script:
           failure if t_normalized > 5 OR r_normalized > 0.1.
Panel (b): Per-replicate fraction of imaging days spent on treatment,
           computed from the Plate Treatment sheets (codes 1 = treatment,
           2 = on-study, 3/4/5 = death / dropout).

- No replicate filtering via experiment_filters (EXCLUDE_SAMPLES).
- Styling follows global thesis rules and matches the Figure 3.22 KM layout:
    * use_style("figstyle.mplstyle")
    * 16 cm width via set_size(..., width="thesis", ratio=...)
    * no tight_layout(), only fig.subplots_adjust(left=0.12, right=0.98, wspace=0.30)
    * colours via figtools.therapy_color(...)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from figtools import (  # type: ignore
    GRID_ALPHA,
    GRID_LINEWIDTH,
    add_panel_label,
    new_subplots,
    savefig,
    therapy_color,
    use_style,
)
from scipy.stats import chi2  # NEW
from scipy.stats import kruskal, mannwhitneyu  # NEW



# ---------------------------------------------------------------------
# Paths and basic config
# ---------------------------------------------------------------------

# Excel file with Plate 1/2 and treatment sheets
DATA_XLS = Path(
    r"Y:\Members\MaxE\Thesis\Organized Figures\Chapter_4\Figure 4.8 Low-Adhesion KM + Fractions\ME_A2-3 Data.xlsx"
)

# Output PDF for the combined figure
OUT_PDF = DATA_XLS.parent / "Figure 4.8_KM_plus_fraction.pdf"


# ---------------------------------------------------------------------
# Experimental layout: mapping wells -> treatment groups
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


def identify_sample(replicate: str) -> str | float:
    """
    Map replicate name to sample group.

    Replicates are sometimes prefixed with '1' / '2' (for Plate 1 / Plate 2).
    We strip a leading '1' or '2' to recover the well ID (A1, B3, ...),
    then look it up in sample_groups.
    """
    rep = replicate
    if rep.startswith("1") or rep.startswith("2"):
        rep = rep[1:]
    for sample, reps in sample_groups.items():
        if rep in reps:
            return sample
    return np.nan


# ---------------------------------------------------------------------
# Data extraction and processing (copied from Figure_7, de-artifacted)
# ---------------------------------------------------------------------

def extract_and_normalize_remove_artifacts(data_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract g / r / total and normalize to baseline (Day 0),
    applying the same artifact corrections as in the original Figure_7 script.
    """
    extracted_data = {"Replicate": [], "Day": [], "g": [], "r": [], "t": []}

    # Every fifth column pattern: [.., Replicate, ..., g, ..., r, ..., t, ...]
    replicate_columns = range(3, data_df.shape[1], 5)
    g_columns = range(3, data_df.shape[1], 5)
    r_columns = range(5, data_df.shape[1], 5)
    t_columns = range(6, data_df.shape[1], 5)

    # Day labels in first column (row ≥ 3)
    days = data_df.iloc[3:, 0].dropna().astype(int).unique()

    for rep_col, g_col, r_col, t_col in zip(
        replicate_columns, g_columns, r_columns, t_columns
    ):
        replicate = data_df.iloc[1, rep_col]
        for day in days:
            # Skip artifact days exactly as in the original script
            if day in (29, 15, 27):
                continue

            day_data = data_df.loc[data_df.iloc[:, 0] == day]
            if day_data.empty:
                continue

            g_value = day_data.iloc[0, g_col]
            r_value = day_data.iloc[0, r_col]

            # Early-day offset corrections
            if day in range(-2, 6):  # days -2 to 5
                g_value -= 40_000_000
                r_value -= 40_000_000
            elif day == 6:
                g_value -= 5_000_000
                r_value -= 5_000_000

            # Additional correction: subtract r'/12 from g
            g_value -= r_value / 12.0

            extracted_data["Replicate"].append(replicate)
            extracted_data["Day"].append(day)
            extracted_data["g"].append(g_value)
            extracted_data["r"].append(r_value)
            extracted_data["t"].append(day_data.iloc[0, t_col])

    extracted_df = pd.DataFrame(extracted_data)

    # Normalize by t at Day 0 (baseline total)
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
    Add a 'treatment' column to normalized_df based on the Plate Treatment sheet.

    treatment codes follow the original script:
        1 = treatment
        2 = on-study / no treatment
        3,4,5 = death / dropout
    """
    treatment_days = treatment_df.iloc[:, 0].dropna().unique()
    replicate_columns = treatment_df.columns[2:]

    treatment_dict: Dict[Tuple[int, str], float] = {}
    for _, row in treatment_df.iterrows():
        day_str = row.iloc[0]
        if day_str not in treatment_days:
            continue
        # day strings like "d-2", "d0", "d10"
        day = int(day_str[1:])
        for replicate in replicate_columns:
            treatment_dict[(day, replicate)] = row[replicate]

    normalized_df = normalized_df.copy()
    normalized_df["treatment"] = normalized_df.apply(
        lambda r: treatment_dict.get((r["Day"], r["Replicate"]), 0),
        axis=1,
    )
    return normalized_df


def process_replicate_data(
    normalized_df: pd.DataFrame,
    treatment_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each replicate, find:
        - First_Day  : first day where failure criterion holds
        - First_Relevant_Day : first day with treatment code 0 / 4 / 5 (dropout)

    Failure criterion is identical to the original Figure_7:
        failure if t_normalized > 5 OR r_normalized > 0.1 on two consecutive days.
    """
    results = []

    # --- failure day from normalized data ---
    for replicate in normalized_df["Replicate"].unique():
        rep_df = normalized_df[normalized_df["Replicate"] == replicate].sort_values(
            "Day"
        )

        # two-day condition on either total or resistant
        cond = (
            (rep_df["t_normalized"] > 5) & (rep_df["t_normalized"].shift(1) > 5)
        ) | ((rep_df["r_normalized"] > 0.1) & (rep_df["r_normalized"].shift(1) > 0.1))

        first_day = rep_df[cond]["Day"].min()
        results.append({"Replicate": replicate, "First_Day": first_day})

    result_df = pd.DataFrame(results)

    # --- dropout day from treatment data ---
    relevant_days = {}
    for replicate in treatment_df.columns[2:]:
        for _, row in treatment_df.iterrows():
            if row[replicate] in [0, 4, 5]:
                relevant_days[replicate] = row["Unnamed: 0"]
                break

    relevant_days_df = pd.DataFrame(
        list(relevant_days.items()),
        columns=["Replicate", "First_Relevant_Day"],
    )

    result_df = result_df.merge(relevant_days_df, on="Replicate", how="left")
    return result_df


def combine_replicate_data(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
    Prefix replicates by plate (1 / 2) and concatenate.
    """
    df1 = df1.copy()
    df2 = df2.copy()
    df1["Replicate"] = df1["Replicate"].apply(lambda x: f"1{x}")
    df2["Replicate"] = df2["Replicate"].apply(lambda x: f"2{x}")
    return pd.concat([df1, df2], ignore_index=True)


# ---------------------------------------------------------------------
# Kaplan–Meier helper (same maths as in Figure 3.22 KM script)
# ---------------------------------------------------------------------

def kaplan_meier_fit(
    times: List[float],
    events: List[int],
) -> Tuple[List[float], List[float]]:
    """
    Classic Kaplan–Meier survival curve (right-censored).
    Returns (time_points, survival_values).
    """
    from collections import defaultdict
    import math

    by_time = defaultdict(lambda: {"events": 0, "censored": 0})
    clean_pairs: List[Tuple[float, int]] = []

    for t, e in zip(times, events):
        if t is None or t == "":
            continue
        try:
            tf = float(t)
        except Exception:
            continue
        if math.isnan(tf):
            continue
        clean_pairs.append((tf, int(e)))

    if not clean_pairs:
        return [0.0], [1.0]

    for tf, e in clean_pairs:
        if e == 1:
            by_time[tf]["events"] += 1
        else:
            by_time[tf]["censored"] += 1

    unique_times = sorted(by_time.keys())
    n = len(clean_pairs)
    S = 1.0

    t_out = [0.0]
    S_out = [1.0]

    for t in unique_times:
        d = by_time[t]["events"]
        c = by_time[t]["censored"]
        if n > 0:
            if d > 0:
                S *= (1.0 - d / n)
                t_out.append(t)
                S_out.append(S)
            n -= (d + c)
        else:
            break

    return t_out, S_out

def logrank_test(times_a, events_a, times_b, events_b):
    """
    Simple two-sample log-rank test.
    events: 1=event, 0=censored
    returns (chi2_stat, p_value)
    """
    # Convert to numpy
    ta = np.asarray(times_a, dtype=float)
    ea = np.asarray(events_a, dtype=int)
    tb = np.asarray(times_b, dtype=float)
    eb = np.asarray(events_b, dtype=int)

    # Event times (where event occurred)
    event_times = np.unique(np.concatenate([ta[ea == 1], tb[eb == 1]]))
    if event_times.size == 0:
        return np.nan, np.nan

    Oa = 0.0
    Ea = 0.0
    Va = 0.0

    for t in event_times:
        # at risk just before t
        na = np.sum(ta >= t)
        nb = np.sum(tb >= t)
        n = na + nb
        if n <= 1:
            continue

        # events at t
        da = np.sum((ta == t) & (ea == 1))
        db = np.sum((tb == t) & (eb == 1))
        d = da + db
        if d == 0:
            continue

        # expected events in group A
        ea_t = d * (na / n)

        # variance (hypergeometric)
        v_t = (na * nb * d * (n - d)) / (n**2 * (n - 1))

        Oa += da
        Ea += ea_t
        Va += v_t

    if Va <= 0:
        return np.nan, np.nan

    z = (Oa - Ea)
    chi2_stat = (z * z) / Va
    p = chi2.sf(chi2_stat, df=1)
    return chi2_stat, p


# ---------------------------------------------------------------------
# Build KM + fraction-of-days datasets
# ---------------------------------------------------------------------

def build_km_input(
    normalized_df_plate1: pd.DataFrame,
    normalized_df_plate2: pd.DataFrame,
    treatment_df_plate1: pd.DataFrame,
    treatment_df_plate2: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reproduce the event coding from plot_fig7_km_and_cumulative in Figure_7.
    Returns a DataFrame with columns:
        Replicate, Event_Day, Event_Type
    """
    # Per-plate failure / dropout
    result_df1 = process_replicate_data(normalized_df_plate1, treatment_df_plate1)
    result_df2 = process_replicate_data(normalized_df_plate2, treatment_df_plate2)

    combined = combine_replicate_data(result_df1, result_df2)

    # Make numeric and extract day number from strings (e.g. 'd10')
    combined["First_Day"] = pd.to_numeric(combined["First_Day"], errors="coerce")
    combined["First_Relevant_Day"] = (
        combined["First_Relevant_Day"].astype(str).str.extract(r"(\d+)").astype(float)
    )

    # Event day: min(failure day, dropout day)
    combined["Event_Day"] = combined[["First_Day", "First_Relevant_Day"]].min(axis=1)

    # Event_Type logic copied from original script, including inversion
    combined["Event_Type"] = combined.apply(
        lambda row: 1
        if pd.isna(row["First_Day"])
        or row["First_Day"] <= row["First_Relevant_Day"]
        else 0,
        axis=1,
    )
    combined = combined.dropna(subset=["Event_Day", "Event_Type"]).copy()
    combined["Event_Type"] = combined["Event_Type"].apply(lambda x: 1 if x == 0 else 0)

    return combined[["Replicate", "Event_Day", "Event_Type"]]


def build_fraction_input(
    combined_df_plate1: pd.DataFrame,
    combined_df_plate2: pd.DataFrame,
) -> pd.DataFrame:
    """
    Fraction of days on treatment per replicate, using the exact logic from
    plot_treatment_violinplots / plot_fig7_km_and_cumulative:

    - Combine plates and prefix replicates with 1/2
    - Remove rows with NaNs
    - Exclude any replicate that ever has treatment code 4 or 5
    - Keep only treatment codes 1 and 2 for counting on-study days
    - Subtract 2 baseline days from the denominator
    - Exclude control samples for the final plot
    """
    df1 = combined_df_plate1.copy()
    df2 = combined_df_plate2.copy()
    df1["Replicate"] = df1["Replicate"].apply(lambda x: f"1{x}")
    df2["Replicate"] = df2["Replicate"].apply(lambda x: f"2{x}")
    combined = pd.concat([df1, df2], ignore_index=True)

    combined2 = combined.dropna()

    # Exclude replicates that ever visit states 4 or 5
    replicates_to_exclude = combined2[
        combined2["treatment"].isin([4.0, 5.0])
    ]["Replicate"].unique()
    filtered = combined2[~combined2["Replicate"].isin(replicates_to_exclude)].copy()

    # Exclude days with treatment codes 0 or 3 (death/dropout)
    filtered = filtered[~filtered["treatment"].isin([0.0, 3.0])]

    # Denominator: on-study days (codes 1 or 2)
    denom = (
        filtered[filtered["treatment"].isin([1.0, 2.0])]
        .groupby("Replicate")
        .size()
        .reset_index(name="n_days")
    )
    # Subtract 2 baseline days
    denom["n_days"] = denom["n_days"] - 2
    denom.loc[denom["n_days"] <= 0, "n_days"] = 1

    # Numerator: treated days (code 1)
    num = (
        filtered[filtered["treatment"] == 1.0]
        .groupby("Replicate")
        .size()
        .reset_index(name="treated_days")
    )

    grouped = pd.merge(denom, num, on="Replicate", how="left").fillna(
        {"treated_days": 0}
    )
    grouped["Sample"] = grouped["Replicate"].apply(identify_sample)

    # Exclude controls, as in the original violin
    excluded_samples = [
        "No treatment",
        "Resistant Control No Treatment",
        "Resistant Control Maximum-tolerable-dose",
    ]
    grouped = grouped[~grouped["Sample"].isin(excluded_samples)].copy()

    grouped["fraction_treated"] = grouped["treated_days"] / grouped["n_days"]

    # Add group-wise medians to sort by
    grouped["median"] = grouped.groupby("Sample")["fraction_treated"].transform(
        "median"
    )

    return grouped


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def plot_km_and_fraction(
    km_df: pd.DataFrame,
    frac_df: pd.DataFrame,
) -> None:
    """
    Create the combined figure:
        (a) KM survival per treatment
        (b) fraction of days on treatment per treatment
    """

    # Treatment groups we actually want to show
    excluded_groups = [
        "No treatment",
        "Resistant Control No Treatment",
        "Resistant Control Maximum-tolerable-dose",
    ]

    # Map long treatment names to canonical labels + colors
    # (colours must be taken from figtools.therapy_color)
    LONG_TO_CANON = {
        "Maximum-tolerable-dose": "CT",
        "Adaptive Therapy 100": "AT100",
        "Adaptive Therapy 80": "AT80",
        "Adaptive Therapy 50": "AT50",
        "Intermittent": "Pulse",  # adjust if your palette uses a different key
    }
    LONG_TO_DISPLAY = {
        "Maximum-tolerable-dose": "CT",
        "Adaptive Therapy 100": "AT100",
        "Adaptive Therapy 80": "AT80",
        "Adaptive Therapy 50": "AT50",
        "Intermittent": "IT",
    }

    # Strategies present in either KM data or fraction data
    present_long = sorted(
        {
            s
            for s in frac_df["Sample"].dropna().unique().tolist()
            if s not in excluded_groups
        }
    )

    # Build per-strategy KM series
    per_strat_times: Dict[str, List[float]] = {}
    per_strat_events: Dict[str, List[int]] = {}

    for long_name in present_long:
        wells = sample_groups.get(long_name, [])
        # Replicates are prefixed with plate identifier 1/2, so compare on [1:]
        mask = km_df["Replicate"].str[1:].isin(wells)
        sub = km_df[mask]
        if sub.empty:
            continue

        times = sub["Event_Day"].astype(float).tolist()
        events = sub["Event_Type"].astype(int).tolist()
        per_strat_times[long_name] = times
        per_strat_events[long_name] = events

        # ---------- Console stats: pairwise log-rank tests (KM) ----------
        km_groups = [g for g in present_long if g in per_strat_times]
        if len(km_groups) >= 2:
            print("\n[KM] Pairwise log-rank tests (Event_Day):")
            for i in range(len(km_groups)):
                for j in range(i + 1, len(km_groups)):
                    g1, g2 = km_groups[i], km_groups[j]
                    chi2_stat, p = logrank_test(
                        per_strat_times[g1], per_strat_events[g1],
                        per_strat_times[g2], per_strat_events[g2],
                    )
                    d1 = LONG_TO_DISPLAY.get(g1, g1)
                    d2 = LONG_TO_DISPLAY.get(g2, g2)
                    n1 = len(per_strat_times[g1])
                    n2 = len(per_strat_times[g2])
                    print(f"  {d1} (n={n1}) vs {d2} (n={n2}): chi2={chi2_stat:.3f}, p={p:.3e}")

    # For panel (b): use only treatments with fraction data
    frac_present_long = sorted(frac_df["Sample"].dropna().unique().tolist())
    frac_present_long = [g for g in frac_present_long if g not in excluded_groups]

    # Medians for ordering panel (b) high → low treated fraction
    medians: Dict[str, float] = {}
    for g in frac_present_long:
        med = frac_df.loc[frac_df["Sample"] == g, "fraction_treated"].median()
        medians[g] = med

    order_frac = sorted(
        frac_present_long, key=lambda s: medians.get(s, np.nan), reverse=True
    )

    # ---------- Console stats: fraction-treated (Kruskal + pairwise MWU) ----------
    if order_frac and len(order_frac) >= 2:
        groups = {g: (frac_df.loc[frac_df["Sample"] == g, "fraction_treated"].to_numpy() * 100.0)
                  for g in order_frac}
        arrays = [v for v in groups.values() if v.size > 0]
        if len(arrays) >= 2:
            H, p_kw = kruskal(*arrays)
            print(f"\n[Fraction-treated] Kruskal-Wallis: H={H:.3f}, p={p_kw:.3e}")

            print("[Fraction-treated] Pairwise Mann-Whitney U (two-sided):")
            for i in range(len(order_frac)):
                for j in range(i + 1, len(order_frac)):
                    g1, g2 = order_frac[i], order_frac[j]
                    a, b = groups[g1], groups[g2]
                    if a.size == 0 or b.size == 0:
                        continue
                    U, p = mannwhitneyu(a, b, alternative="two-sided")
                    d1 = LONG_TO_DISPLAY.get(g1, g1)
                    d2 = LONG_TO_DISPLAY.get(g2, g2)
                    print(f"  {d1} (n={a.size}) vs {d2} (n={b.size}): U={U:.1f}, p={p:.3e}")


    # ============== Plot ==========================
    with use_style("figstyle.mplstyle"):
        fig, axes = new_subplots(
            nrows=1,
            ncols=2,
            width="thesis",
            ratio=0.62,
            sharex=False,
            sharey=False,
            gridspec_kw={"width_ratios": [2, 1]},  # KM 2/3, fraction 1/3
        )
        ax_km, ax_frac = axes

        # Panel letters
        for ax, lab in zip(axes, ["a", "b"]):
            add_panel_label(ax, lab)

        # ---------- Panel (a): KM ----------
        for long_name in present_long:
            if long_name not in per_strat_times:
                continue

            canonical = LONG_TO_CANON.get(long_name, long_name)
            display = LONG_TO_DISPLAY.get(long_name, long_name)
            color = therapy_color(canonical)

            times = per_strat_times[long_name]
            events = per_strat_events[long_name]
            if not times:
                continue

            t_vals, S_vals = kaplan_meier_fit(times, events)

            # Extend curve to last observed time for that strategy
            last_time = max(times) if times else 0.0
            if t_vals[-1] < last_time:
                t_vals.append(last_time)
                S_vals.append(S_vals[-1])

            line, = ax_km.plot(
                t_vals,
                S_vals,
                label=f"{display} (n={len(times)})",
                drawstyle="steps-post",
                color=color,
            )

            # Mark final point
            ax_km.plot(
                t_vals[-1],
                S_vals[-1],
                marker="o",
                markersize=4,
                color=line.get_color(),
            )

            # Censoring ticks at appropriate y-levels
            censored_times = [t for t, e in zip(times, events) if e == 0]
            if censored_times:
                for ct in censored_times:
                    # survival value at censor time
                    y = S_vals[0]
                    for tv, sv in zip(t_vals, S_vals):
                        if tv <= ct:
                            y = sv
                        else:
                            break
                    ax_km.plot(
                        ct,
                        y,
                        marker="|",
                        markersize=8,
                        markeredgewidth=0.9,
                        color=line.get_color(),
                    )

        ax_km.set_xlabel("Time [days]")
        ax_km.set_ylabel("Survival")
        ax_km.set_ylim(0, 1.05)
        ax_km.set_xlim(left=0)
        ax_km.grid(
            True,
            alpha=GRID_ALPHA,
            linestyle="--",
            linewidth=GRID_LINEWIDTH,
        )

        ax_km.legend(frameon=False)

        # ---------- Panel (b): fraction of days on treatment ----------
        if order_frac:
            positions = np.arange(1, len(order_frac) + 1)

            # Convert to percent
            data_pct = [
                frac_df.loc[frac_df["Sample"] == g, "fraction_treated"].to_numpy()
                * 100.0
                for g in order_frac
            ]

            # Boxplot (no outliers; black boxes)
            bp = ax_frac.boxplot(
                data_pct,
                positions=positions,
                showfliers=False,
                widths=0.6,
            )

            for element in ["boxes", "whiskers", "caps", "medians"]:
                for item in bp[element]:
                    item.set(color="black", linewidth=1.0)

            # Jittered individual points, coloured by therapy type
            for i, long_name in enumerate(order_frac, start=1):
                arr = (
                    frac_df.loc[frac_df["Sample"] == long_name, "fraction_treated"]
                    .to_numpy()
                    * 100.0
                )
                if arr.size == 0:
                    continue
                x_jitter = np.random.normal(loc=i, scale=0.06, size=arr.size)
                canonical = LONG_TO_CANON.get(long_name, long_name)
                color = therapy_color(canonical)
                ax_frac.scatter(
                    x_jitter,
                    arr,
                    s=10,
                    alpha=0.7,
                    c=color,
                    edgecolors="none",
                )

            # X tick labels as short codes
            ax_frac.set_xticks(positions)
            ax_frac.set_xticklabels(
                [LONG_TO_DISPLAY.get(g, g) for g in order_frac],
                rotation=40,
                ha="right",
            )

            ax_frac.set_ylabel("Days under treatment [%]")
            ax_frac.set_ylim(-5, 105)

        # Preserve the intended 160 mm physical canvas; tight cropping would
        # otherwise slightly reduce the exported PDF width.
        savefig(fig, OUT_PDF, dpi=300, transparent=True, bbox_inches=None)
        plt.show()
        plt.close(fig)
        print(f"Saved Figure 4.8 KM + fraction panel to: {OUT_PDF}")


# ---------------------------------------------------------------------
# Main: load, process, plot
# ---------------------------------------------------------------------

def main() -> None:
    # Load Excel
    xls = pd.ExcelFile(DATA_XLS)

    # Plate 1 / 2 raw & treatment
    data_df_plate1 = pd.read_excel(xls, "Plate 1")
    data_df_plate2 = pd.read_excel(xls, "Plate 2")
    treatment_df_plate1 = pd.read_excel(xls, "Plate 1 Treatment")
    treatment_df_plate2 = pd.read_excel(xls, "Plate 2 Treatment")

    # De-artifacted normalized data
    norm1 = extract_and_normalize_remove_artifacts(data_df_plate1)
    norm2 = extract_and_normalize_remove_artifacts(data_df_plate2)

    # Combine with treatment codes (for fraction panel)
    comb1 = combine_normalized_with_treatment(norm1, treatment_df_plate1)
    comb2 = combine_normalized_with_treatment(norm2, treatment_df_plate2)

    # KM input: event coding
    km_df = build_km_input(norm1, norm2, treatment_df_plate1, treatment_df_plate2)

    # Fraction input: per-replicate treated fraction
    frac_df = build_fraction_input(comb1, comb2)

    # Plot combined figure
    plot_km_and_fraction(km_df, frac_df)


if __name__ == "__main__":
    main()
