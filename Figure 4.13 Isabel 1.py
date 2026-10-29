#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Figure 4.13 – CT vs AT time courses and KM (Isabel experiment)

Panels:
  a) CT:        Total, EGFP/Total, mScarlet/Total (median ± IQR)
  b) AT:        Total, EGFP/Total, mScarlet/Total (median ± IQR)
  c) Resistant Growth: mScarlet/Total (median ± IQR) for CT vs AT
  d) Spheroid Survival: KM of time to 200% mScarlet/Total

Data source:
  Data/isabel/Data_main_all_wells.xlsx

Sheets used:
  - "BTotal_normalized"
  - "BEGFP_normalized_to_total"
  - "BmScarlet_normalized_to_total"
  - "Meta"   (for valid wells)

Only wells that are both:
  - in WELL_GROUPS, and
  - present in Meta.WELL
are used for plotting.
"""

from __future__ import annotations

from pathlib import Path
from data_paths import DATA_ROOT, output_dir
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

from figtools import (
    POPULATION_COLORS,
    SECONDARY_LINEWIDTH,
    add_panel_label,
    format_axes,
    new_subplots,
    savefig,
    therapy_color,
    use_style,
)


# ---------------------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------------------

DATA_XLS = DATA_ROOT / "isabel" / "Data_main_all_wells.xlsx"

OUT_PDF = output_dir("figures", "figure_4_13") / "Figure 4.13_CT_AT_Timecourses_and_KM.pdf"

# Wells and treatment group (from original setup)
WELL_GROUPS: Dict[str, str] = {
    "D11": "AT100",
    "D7":  "AT100",
    "E10": "AT100",
    "E3":  "AT100",
    "E4":  "AT100",
    "E7":  "AT100",
    "E8":  "AT100",
    "F3":  "AT100",
    "G3":  "AT100",
    "G4":  "AT100",
    "G6":  "AT100",
    "G8":  "AT100",
    "B3":  "CT",
    "B4":  "CT",
    "C3":  "CT",
    "C6":  "CT",
    "C7":  "CT",
    "C8":  "CT",
    "F11": "CT",
    "G2":  "CT",
}

# KM configuration
KM_THRESHOLD = 200.0
KM_DIRECTION = "at_or_above"  # 'at_or_above', 'above', 'at_or_below', 'below'

# Resistant / susceptible colours (local for now, can be moved into figtools later)
COLOR_RESISTANT = POPULATION_COLORS["resistant"]
COLOR_SUSCEPTIBLE = POPULATION_COLORS["sensitive"]
COLOR_TOTAL = POPULATION_COLORS["total"]
COLOR_CT_GROUP = therapy_color("CT")
COLOR_AT_GROUP = therapy_color("AT100")


# ---------------------------------------------------------------------
# Data loading and helpers
# ---------------------------------------------------------------------

def load_valid_wells_from_meta(xls_path: Path) -> List[str]:
    """
    Read the 'Meta' sheet and return the list of wells that were used.

    Meta header (row 1):
      WELL  TREATMENT  CONCENTRATION  DAYSOFTREATMENT  To sort  Dispersion  Begin_migration
    """
    meta = pd.read_excel(xls_path, sheet_name="Meta")
    if "WELL" not in meta.columns:
        raise ValueError("Meta sheet has no 'WELL' column.")
    wells = (
        meta["WELL"]
        .astype(str)
        .str.upper()
        .str.strip()
        .tolist()
    )
    return wells


def load_metric_sheets(xls_path: Path) -> Dict[str, pd.DataFrame]:
    """
    Load and melt the three metric sheets from Data_main_all_wells.xlsx into
    long format with columns: WELL, TREATMENT_CONCENTRATION, DAY, VALUE.

    Sheets:
      - "BTotal_normalized"             -> key 'total_norm'
      - "BEGFP_normalized_to_total"     -> key 'egfp_norm_total'
      - "BmScarlet_normalized_to_total" -> key 'mscar_norm_total'
    """
    sheet_map = {
        "total_norm":       "BTotal_normalized",
        "egfp_norm_total":  "BEGFP_normalized_to_total",
        "mscar_norm_total": "BmScarlet_normalized_to_total",
    }

    xls = pd.ExcelFile(xls_path)
    out: Dict[str, pd.DataFrame] = {}

    for key, sheet_name in sheet_map.items():
        if sheet_name not in xls.sheet_names:
            raise FileNotFoundError(
                f"Sheet '{sheet_name}' not found in {xls_path.name}. "
                f"Available sheets: {xls.sheet_names}"
            )

        df = pd.read_excel(xls, sheet_name=sheet_name)
        # Expect columns: WELL, TREATMENT_CONCENTRATION, 0,2,4,...72
        if "WELL" not in df.columns:
            df = df.rename(columns={df.columns[0]: "WELL"})

        df["WELL"] = df["WELL"].astype(str).str.upper().str.strip()
        if "TREATMENT_CONCENTRATION" not in df.columns:
            df["TREATMENT_CONCENTRATION"] = ""

        day_cols = [
            c for c in df.columns
            if c not in ("WELL", "TREATMENT_CONCENTRATION")
        ]

        dfl = df.melt(
            id_vars=["WELL", "TREATMENT_CONCENTRATION"],
            value_vars=day_cols,
            var_name="DAY",
            value_name="VALUE",
        ).dropna(subset=["VALUE"])

        dfl["DAY"] = pd.to_numeric(dfl["DAY"], errors="coerce")
        dfl = dfl.dropna(subset=["DAY"]).copy()
        dfl["DAY"] = dfl["DAY"].astype(int)

        out[key] = dfl

    return out


def get_all_days(sheets: Dict[str, pd.DataFrame]) -> List[int]:
    """Return sorted unique list of all days present across all metrics."""
    days = set()
    for df in sheets.values():
        days.update(df["DAY"].unique())
    return sorted(int(d) for d in days)


def agg_med_iqr(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate long-format metric to median and IQR per day."""
    g = df.groupby("DAY")["VALUE"]
    med = g.median()
    q25 = g.quantile(0.25)
    q75 = g.quantile(0.75)
    return pd.DataFrame(
        {
            "DAY": med.index,
            "MED": med.values,
            "Q25": q25.values,
            "Q75": q75.values,
        }
    )


def collapse_group(label: str) -> str:
    """Collapse AT variants to 'AT' and keep CT as 'CT'."""
    lab = str(label).upper().strip()
    if lab.startswith("AT"):
        return "AT"
    return lab


def event_happened(v: float, thr: float, direction: str) -> bool:
    if direction == "at_or_above":
        return v >= thr
    if direction == "above":
        return v > thr
    if direction == "at_or_below":
        return v <= thr
    if direction == "below":
        return v < thr
    raise ValueError(f"Bad KM_DIRECTION: {direction!r}")


# ---------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------

def plot_combined_timecourse(
    ax: plt.Axes,
    sheets: Dict[str, pd.DataFrame],
    wells: List[str],
    title: str,
) -> None:
    """
    Plot Total, EGFP/Total, mScarlet/Total (median ± IQR) for a given group.
    """

    # (metric_key, label, color)
    series_def = [
        ("total_norm",       "Total",       COLOR_TOTAL),
        ("egfp_norm_total",  "Sensitive", COLOR_SUSCEPTIBLE),
        ("mscar_norm_total", "Resistant",   COLOR_RESISTANT),
    ]

    for metric_key, label, color in series_def:
        dfl = sheets.get(metric_key)
        if dfl is None or dfl.empty:
            continue
        sub = dfl[dfl["WELL"].isin(wells)]
        if sub.empty:
            continue
        agg = agg_med_iqr(sub)
        ax.plot(
            agg["DAY"],
            agg["MED"],
            color=color,
            label=label,
        )
        ax.fill_between(
            agg["DAY"],
            agg["Q25"],
            agg["Q75"],
            color=color,
            alpha=0.25,
        )

    # Reference line at 100%
    ax.axhline(
        100.0,
        color="0.7",
        linestyle="--",
        linewidth=SECONDARY_LINEWIDTH,
        zorder=0,
    )

    ax.set_title(title)
    ax.set_ylabel("Normalized TFI")
    ax.set_xlabel("Time [Days]")

    # X range and ticks: 0–72, ticks at 0,5,10,...,70
    ax.set_xlim(0, 72)
    ticks = list(range(0, 80, 10))  # 0,10,20,...,70
    ax.set_xticks(ticks)

    # y-limits fairly generous
    ax.set_ylim(0, 400)
    ax.set_yticks([0, 100, 200, 300])

    format_axes(ax)
    ax.legend(frameon=False, loc="upper left")


def plot_resistant_timecourse(
    ax: plt.Axes,
    sheets: Dict[str, pd.DataFrame],
    wells_ct: List[str],
    wells_at: List[str],
) -> None:
    """
    Panel c: mScarlet/Total (median ± IQR) for CT vs AT.
    """
    dfl = sheets.get("mscar_norm_total")
    if dfl is None or dfl.empty:
        return

    for label, wells, color, linestyle in [
        ("CT", wells_ct, COLOR_CT_GROUP, "-"),
        ("AT", wells_at, COLOR_AT_GROUP, "-"),
    ]:
        sub = dfl[dfl["WELL"].isin(wells)]
        if sub.empty:
            continue
        agg = agg_med_iqr(sub)
        ax.plot(
            agg["DAY"],
            agg["MED"],
            color=color,
            linestyle=linestyle,
            label=label,
        )
        ax.fill_between(
            agg["DAY"],
            agg["Q25"],
            agg["Q75"],
            color=color,
            alpha=0.25,
        )

    ax.axhline(
        100.0,
        color="0.7",
        linestyle="--",
        linewidth=SECONDARY_LINEWIDTH,
        zorder=0,
    )

    ax.set_ylabel("Normalized mScarlet TFI")
    ax.set_xlabel("Time [Days]")

    # X range and ticks: 0–72, ticks at 0,5,10,...,70
    ax.set_xlim(0, 72)
    ticks = list(range(0, 80, 10))  # 0,10,20,...,70
    ax.set_xticks(ticks)

    ax.set_ylim(0, 400)
    ax.set_yticks([0, 100, 200, 300])

    format_axes(ax)
    ax.legend(frameon=False, loc="upper left")


def compute_km_data(
    dfl_mscar: pd.DataFrame,
    well_groups: Dict[str, str],
) -> pd.DataFrame:
    """
    Compute KM durations/events from long-format mScar_norm_to_total data.

    For each well:
      - find first day where VALUE meets the KM_THRESHOLD condition
      - if found: duration = that day, event = 1
      - if not:   duration = last measured day, event = 0
      - group = 'CT' or 'AT' (collapse all AT variants)
    """
    rows = []

    for well, raw_group in well_groups.items():
        sub = dfl_mscar[dfl_mscar["WELL"] == well].sort_values("DAY")
        if sub.empty:
            continue

        vals = sub["VALUE"].to_numpy(dtype=float)
        days = sub["DAY"].to_numpy(dtype=float)

        idxs = np.where(
            [event_happened(v, KM_THRESHOLD, KM_DIRECTION) for v in vals]
        )[0]

        if idxs.size > 0:
            duration = float(days[int(idxs[0])])
            event = 1
        else:
            duration = float(days[-1])
            event = 0

        group = collapse_group(raw_group)
        if group not in {"CT", "AT"}:
            continue

        rows.append(
            {
                "well": well,
                "group": group,
                "duration": duration,
                "event": event,
            }
        )

    return pd.DataFrame(rows)


def plot_km_panel(
    ax: plt.Axes,
    km_df: pd.DataFrame,
) -> None:
    """
    Panel d: KM curves CT vs AT.
    """
    if km_df.empty:
        raise RuntimeError("KM dataframe is empty; no wells for CT/AT found.")

    PRETTY_LABEL = {"CT": "CT", "AT": "AT"}
    GROUP_COLOR = {"CT": COLOR_CT_GROUP, "AT": COLOR_AT_GROUP}

    km_groups = [g for g in ["CT", "AT"] if g in set(km_df["group"])]

    kmf = KaplanMeierFitter()
    for g in km_groups:
        sub = km_df[km_df["group"] == g]
        label = PRETTY_LABEL.get(g, g)
        color = GROUP_COLOR.get(g, None)
        kmf.fit(sub["duration"], event_observed=sub["event"], label=label)
        kmf.plot(ax=ax, ci_show=True, color=color)

    ax.set_xlabel("Time [Days]")
    ax.set_ylabel("Survival Probability")
    ax.set_ylim(0, 1.05)

    # X range and ticks: 0–72, ticks at 0,5,10,...,70
    ax.set_xlim(0, 72)
    ticks = list(range(0, 80, 10))  # 0,10,20,...,70
    ax.set_xticks(ticks)

    format_axes(ax)
    ax.legend(frameon=False, loc="lower left")

    # Log-rank test (printed to console)
    if len(km_groups) == 2:
        g1, g2 = km_groups
        d1 = km_df[km_df["group"] == g1]
        d2 = km_df[km_df["group"] == g2]
        lr = logrank_test(
            d1["duration"], d2["duration"],
            event_observed_A=d1["event"],
            event_observed_B=d2["event"],
        )
        print(
            f"[KM] Log-rank {PRETTY_LABEL[g1]} vs {PRETTY_LABEL[g2]}: "
            f"p = {lr.p_value:.3g}"
        )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    if not DATA_XLS.is_file():
        raise FileNotFoundError(f"Data Excel not found: {DATA_XLS}")

    # Load metric sheets (BTotal..., BEGFP..., BmScarlet...)
    sheets = load_metric_sheets(DATA_XLS)

    # Valid wells according to Meta
    valid_wells = set(load_valid_wells_from_meta(DATA_XLS))

    # Restrict WELL_GROUPS to wells present in Meta
    restricted_groups = {
        w: g for w, g in WELL_GROUPS.items()
        if w in valid_wells
    }

    if not restricted_groups:
        raise RuntimeError(
            "No wells from WELL_GROUPS are present in Meta.WELL. "
            "Check WELL_GROUPS and Meta sheet."
        )

    wells_ct = [w for w, g in restricted_groups.items() if collapse_group(g) == "CT"]
    wells_at = [w for w, g in restricted_groups.items() if collapse_group(g) == "AT"]

    print(f"[INFO] CT wells used: {wells_ct}")
    print(f"[INFO] AT wells used: {wells_at}")

    if not wells_ct or not wells_at:
        raise RuntimeError("After Meta filtering, no CT or AT wells remain for plotting.")

    # Filter melted sheets to only wells in restricted_groups
    for key in list(sheets.keys()):
        dfl = sheets[key]
        sheets[key] = dfl[dfl["WELL"].isin(restricted_groups.keys())].copy()

    dfl_mscar = sheets["mscar_norm_total"]
    km_df = compute_km_data(dfl_mscar, restricted_groups)

    with use_style("figstyle.mplstyle"):
        fig, axes = new_subplots(
            nrows=2,
            ncols=2,
            width="thesis",
            ratio=0.8,
            sharex=False,
            sharey=False,
        )
        ax_a, ax_b = axes[0]
        ax_c, ax_d = axes[1]

        # Panel a – CT time courses
        add_panel_label(ax_a, "a")
        plot_combined_timecourse(
            ax=ax_a,
            sheets=sheets,
            wells=wells_ct,
            title=f"CT (n = {len(wells_ct)})",
        )

        # Panel b – AT time courses
        add_panel_label(ax_b, "b")
        plot_combined_timecourse(
            ax=ax_b,
            sheets=sheets,
            wells=wells_at,
            title=f"AT (n = {len(wells_at)})",
        )

        # Panel c – Resistant growth
        add_panel_label(ax_c, "c")
        plot_resistant_timecourse(
            ax=ax_c,
            sheets=sheets,
            wells_ct=wells_ct,
            wells_at=wells_at,
        )

        # Panel d – KM (spheroid survival)
        add_panel_label(ax_d, "d")
        plot_km_panel(
            ax=ax_d,
            km_df=km_df,
        )

        savefig(fig, OUT_PDF, dpi=300, transparent=True)
        savefig(fig, OUT_PDF.with_suffix(".png"), dpi=600, transparent=False, facecolor="white")
        plt.show()
        plt.close(fig)

    print(f"[SAVE] Figure 4.13 written to: {OUT_PDF}")


if __name__ == "__main__":
    main()
