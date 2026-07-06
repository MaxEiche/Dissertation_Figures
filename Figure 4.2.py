# Figure 4.2.py — Embedded spheroid sensitive/resistant dynamics (topography)
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from figtools import use_style, set_size, savefig, therapy_color
from matplotlib import colors as mcolors


# ---------------------------------------------------------------------
# Style config
# ---------------------------------------------------------------------
STYLE_PATH = Path("figstyle.mplstyle")

# ---------------------------------------------------------------------
# KDE helper
# ---------------------------------------------------------------------
def add_kde_contours(
    ax,
    x,
    y,
    probs=(0.1, 0.2, 0.3, 0.4),
    gridsize=300,
    color="k",
    alpha=0.9,
    linewidths=(2.0, 1.5, 1.0),
    bw_method=0.7,
    pad=0.10,
):
    """
    KDE density contours enclosing a given probability mass for (x,y).

    - probs: cumulative probability masses to enclose (lower -> tighter).
    - bw_method: KDE bandwidth (None | float | 'scott' | 'silverman').
    - pad: fraction of data range added around the data before gridding.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 5:
        return

    xm, xM = np.nanmin(x[m]), np.nanmax(x[m])
    ym, yM = np.nanmin(y[m]), np.nanmax(y[m])
    xr = xM - xm if xM > xm else 1.0
    yr = yM - ym if yM > ym else 1.0
    xm -= pad * xr
    xM += pad * xr
    ym -= pad * yr
    yM += pad * yr

    xx = np.linspace(xm, xM, gridsize)
    yy = np.linspace(ym, yM, gridsize)
    X, Y = np.meshgrid(xx, yy)

    kde = gaussian_kde(np.vstack([x[m], y[m]]), bw_method=bw_method)
    Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

    z = Z.ravel()
    if not np.isfinite(z).any() or np.nanmin(Z) == np.nanmax(Z):
        return  # flat or invalid surface

    # Build contour levels for requested cumulative probs
    z_sorted = np.sort(z)[::-1]
    cdf = np.cumsum(z_sorted)
    cdf /= cdf[-1]
    levels = []
    for p in sorted(probs):
        idx = np.searchsorted(cdf, p)
        thr = z_sorted[idx] if idx < len(z_sorted) else z_sorted[-1]
        levels.append(float(thr))

    levels = np.sort(np.unique(levels))
    if levels.size == 0 or not np.isfinite(levels).all():
        return

    ax.contour(
        X,
        Y,
        Z,
        levels=levels,
        colors=color,
        linewidths=linewidths,
        alpha=alpha,
    )


# ---------------------------------------------------------------------
# Data loading and preprocessing
# ---------------------------------------------------------------------
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

# Path to the ME_A2-4 plate data
file_path = r"Y:\Members\MaxE\Thesis\Organized Figures\Chapter_4\Figure 4.2\ME_A2-4 Data.xlsx"
stats_path = Path(file_path).parent / "Figure_4_2_topography_relative_stats.json"
xls = pd.ExcelFile(file_path)

# Load plate data and treatment sheet
plate_df = pd.read_excel(xls)
treatment_sheet = "Treatment"
# IMPORTANT: no index_col here – we want the 'd-2' strings as first column
treatment_df = pd.read_excel(xls, treatment_sheet)

# Extract raw per-replicate time series for g, r', t
extracted_data = {"Replicate": [], "Day": [], "g": [], "r'": [], "t": []}

# Column indices: replicate label row is row 1; data rows start at row 3
replicate_columns = range(3, plate_df.shape[1], 5)
g_columns = range(3, plate_df.shape[1], 5)
r_prime_columns = range(5, plate_df.shape[1], 5)
t_columns = range(6, plate_df.shape[1], 5)

# Days from first column (skip header rows)
days = plate_df.iloc[3:, 0].dropna().astype(int).unique()

for rep_col, g_col, r_prime_col, t_col in zip(
    replicate_columns, g_columns, r_prime_columns, t_columns
):
    replicate = plate_df.iloc[1, rep_col]
    for day in days:
        day_data = plate_df.loc[plate_df.iloc[:, 0] == day]
        if not day_data.empty:
            extracted_data["Replicate"].append(replicate)
            extracted_data["Day"].append(day)
            extracted_data["g"].append(day_data.iloc[0, g_col])
            extracted_data["r'"].append(day_data.iloc[0, r_prime_col])
            extracted_data["t"].append(day_data.iloc[0, t_col])
        else:
            print(f"[Warning] No data found for day: {day}")

extracted_df = pd.DataFrame(extracted_data)

# ---------------------------------------------------------------------
# Bleedthrough correction (subtract fixed fraction of red from green)
# ---------------------------------------------------------------------
fraction = 0.06
extracted_df["g"] = extracted_df.apply(
    lambda row: row["g"] - fraction * row["r'"], axis=1
)

# ---------------------------------------------------------------------
# Normalisation by baseline t (day 2)
# ---------------------------------------------------------------------
normalized_data = {
    "Replicate": [],
    "Day": [],
    "g_normalized": [],
    "r'_normalized": [],
    "t_normalized": [],
}

baseline_t_values = (
    extracted_df[extracted_df["Day"] == 2].set_index("Replicate")["t"]
)

for _, row in extracted_df.iterrows():
    replicate = row["Replicate"]
    day = row["Day"]
    baseline_t = baseline_t_values.get(replicate, None)
    if baseline_t is None:
        continue

    g_normalized = row["g"] / baseline_t
    r_prime_normalized = row["r'"] / baseline_t
    t_normalized = row["t"] / baseline_t

    normalized_data["Replicate"].append(replicate)
    normalized_data["Day"].append(day)
    normalized_data["g_normalized"].append(g_normalized)
    normalized_data["r'_normalized"].append(r_prime_normalized)
    normalized_data["t_normalized"].append(t_normalized)

normalized_df = pd.DataFrame(normalized_data)

# Exclude day 32 from this analysis
normalized_df = normalized_df[normalized_df["Day"] != 32]

# ---------------------------------------------------------------------
# Treatment annotation per (Day, Replicate) from treatment sheet
# ---------------------------------------------------------------------
def combine_normalized_with_treatment(normalized_df, treatment_df):
    """
    Attach a 'treatment' code to each (Day, Replicate) in normalized_df
    using the treatment sheet. The first column is expected to hold day labels
    like 'd-2', 'd-4', ... but we also handle numeric values gracefully.
    """
    normalized_df = normalized_df.copy()

    # Replicate columns start after the first two columns (day label + maybe comment/time)
    replicate_columns = treatment_df.columns[2:]

    treatment_dict = {}
    for _, row in treatment_df.iterrows():
        day_val = row.iloc[0]
        if pd.isna(day_val):
            continue

        # Accept either 'd-2' style strings or numeric values
        if isinstance(day_val, str):
            # Expect something like 'd-2' -> 2
            try:
                day = int(day_val[1:])
            except (ValueError, IndexError):
                continue
        else:
            # e.g., already numeric or timestamp – try casting to int
            try:
                day = int(day_val)
            except (TypeError, ValueError):
                continue

        for replicate in replicate_columns:
            treatment_dict[(day, replicate)] = row[replicate]

    normalized_df["treatment"] = normalized_df.apply(
        lambda r: treatment_dict.get((r["Day"], r["Replicate"]), 0), axis=1
    )
    return normalized_df


combined_df = combine_normalized_with_treatment(normalized_df, treatment_df)

# ---------------------------------------------------------------------
# Main analysis + figure
# ---------------------------------------------------------------------
def correlate_red_growth(df1):
    df = df1.copy()

    # Prefix replicate IDs to match previous mapping scheme (1A1, 1B1, ...)
    df["Replicate"] = df["Replicate"].apply(lambda x: "1" + x)

    # Cumulative treatment (matches old logic: 1/4 count as 1, 3 counts double)
    df["cumulative_treatment"] = (
        df.groupby((df["treatment"] != df["treatment"].shift(1)).cumsum())
        .apply(
            lambda x: ((x["treatment"] == 1) | (x["treatment"] == 4)).cumsum()
            + (x["treatment"] == 3).cumsum() * 2
        )
        .reset_index(drop=True)
    )

    # Shift cumulative_treatment by one day so it refers to previous exposure
    df["cumulative_treatment"] = df.groupby("Replicate")[
        "cumulative_treatment"
    ].shift(fill_value=0)

    # Relative change of red channel per day + absolute change
    df["relative_change"] = (
        df.groupby("Replicate")[r"r'_normalized"].pct_change(fill_method=None) + 1.0
    )
    df["absolute_change"] = df.groupby("Replicate")[r"r'_normalized"].diff(periods=1)

    # Keep relevant columns
    df = df[
        [
            "Replicate",
            "Day",
            r"r'_normalized",
            "t_normalized",
            "g_normalized",
            "treatment",
            "cumulative_treatment",
            "relative_change",
            "absolute_change",
        ]
    ]

    # Remove rows without treatment or without change values
    df = df[~df["treatment"].isin([0.0])]
    df = df.dropna(subset=["relative_change", "absolute_change"])

    # Diagnostic: how many absolute-change zeros (for debugging only)
    zero_mask = df["absolute_change"] == 0
    num_zero = zero_mask.sum()
    num_total = df["absolute_change"].notna().sum()
    if num_total > 0:
        print(
            f"[Debug] Absolute-change zeros: {num_zero} / {num_total} "
            f"= {100 * num_zero / num_total:.1f}%"
        )

    # Restrict to valid replicates (full plate without calibration wells)
    valid_replicates = [
        "1A1",
        "1A2",
        "1A3",
        "1A4",
        "1A5",
        "1A6",
        "1A7",
        "1B1",
        "1B2",
        "1B3",
        "1B4",
        "1B5",
        "1B6",
        "1B7",
        "1C1",
        "1C2",
        "1C3",
        "1C4",
        "1C5",
        "1C6",
        "1C7",
        "1D1",
        "1D2",
        "1D3",
        "1D4",
        "1D5",
        "1D6",
        "1D7",
        "1E1",
        "1E2",
        "1E3",
        "1E4",
        "1E5",
        "1E6",
        "1E7",
        "1F1",
        "1F2",
        "1F3",
        "1F4",
        "1F5",
        "1F6",
        "1F7",
    ]
    df = df[df["Replicate"].isin(valid_replicates)]

    # -----------------------------------------------------------------
    # Assign therapy labels to replicates
    # Only keep AT100, AT80, AT50, NT for the figure
    # -----------------------------------------------------------------
    treatment_types = {
        "AT100": [
            "1A1",
            "1B1",
            "1C1",
            "1D1",
            "1E1",
            "1A2",
            "1B2",
            "1C2",
            "1D2",
            "1E2",
        ],
        "AT80": [
            "1A3",
            "1B3",
            "1C3",
            "1D3",
            "1E3",
            "1A4",
            "1B4",
            "1C4",
            "1D4",
            "1E4",
        ],
        "AT50": [
            "1A5",
            "1B5",
            "1C5",
            "1D5",
            "1E5",
            "1A6",
            "1B6",
            "1C6",
            "1D6",
            "1E6",
        ],
        "NT": [
            "1A7",
            "1B7",
            "1C7",
            "1D7",
            "1E7",
            "1F7",
        ],
    }

    df["Treatment"] = "Other"
    for name, reps in treatment_types.items():
        df.loc[df["Replicate"].isin(reps), "Treatment"] = name

    # Drop everything that is not part of the four treatment groups
    treatment_colors = {
        "AT100": therapy_color("AT100"),  # your figtools palette
        "AT80": therapy_color("AT80"),
        "AT50": therapy_color("AT50"),
        "NT": therapy_color("NT"),
    }
    df = df[df["Treatment"].isin(treatment_colors.keys())]

    # -----------------------------------------------------------------
    # Descriptive status for the pooled visualization
    # -----------------------------------------------------------------
    df_corr = df.copy()

    # Use the same y-units as in the plot (% per day)
    x = df_corr["g_normalized"].to_numpy()
    y = ((df_corr["relative_change"] - 1.0) * 100.0).to_numpy()

    # Keep finite pairs only (defensive)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    n = int(m.sum())

    n_spheroids = int(df_corr.loc[m, "Replicate"].nunique())
    print("[Descriptive pooled visualization only]")
    print(
        "The originally reported pooled correlation could not be reproduced "
        "from the available canonical code/data path. This figure is retained "
        "as a descriptive pooled visualization only."
    )
    print(f"Descriptive pooled observations retained in the plot: n = {n}")
    print(f"Embedded spheroids represented in the plot: n = {n_spheroids}")

    # -------------------------------------------------------------
    # Precompute plot vectors for relative change topographies
    # -------------------------------------------------------------
    jitter = False  # keep as before

    colors_rel = df["Treatment"].map(treatment_colors)
    valid_idx = colors_rel.notna()

    if jitter:
        x_rel = (
            df.loc[valid_idx, "g_normalized"]
            + np.random.normal(0, 0.1, size=valid_idx.sum())
        ).reset_index(drop=True)
        y_rel = (
            (df.loc[valid_idx, "relative_change"] - 1.0) * 100.0
            + np.random.normal(0, 10.0, size=valid_idx.sum())
        ).reset_index(drop=True)
    else:
        x_rel = df.loc[valid_idx, "g_normalized"].reset_index(drop=True)
        y_rel = (
            (df.loc[valid_idx, "relative_change"] - 1.0) * 100.0
        ).reset_index(drop=True)

    labels_rel = df.loc[valid_idx, "Treatment"].reset_index(drop=True).values
    unique_treatments = sorted(pd.unique(labels_rel))

    # Axis limits used for both plots
    xlim = (0.0, 5.5)
    ylim = (-10.0, 50.0)


    # -----------------------------------------------------------------
    # Final figure: relative-change topography per treatment
    # -----------------------------------------------------------------
    jitter = False  # no jitter by default for the final figure

    with use_style(STYLE_PATH):
        # Ensure Arial font (if not already in figstyle)
        plt.rcParams["font.family"] = "Arial"

        fig, ax = plt.subplots()
        set_size(ax, width="thesis", ratio=0.70)

        # Axes cosmetics: no top/right spines, ticks outward
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(direction="out")

        for t in unique_treatments:
            sel = labels_rel == t
            if sel.sum() < 3:
                continue

            xg = x_rel[sel]
            yg = y_rel[sel]
            color = treatment_colors.get(t, "k")

            # point cloud
            ax.scatter(xg, yg, s=8, c=color, alpha=0.3, label=t)

            # line-only KDE contours around the cloud
            add_kde_contours(
                ax,
                xg,
                yg,
                probs=(0.1, 0.2, 0.3, 0.4),
                gridsize=300,
                color=color,
                alpha=0.9,
                linewidths=(2.0, 1.5, 1.0),
                bw_method=0.7,
                pad=0.10,
            )

        ax.set_xlabel("Normalized Sensitive Population")
        ax.set_ylabel("Relative Resistant Growth [%]")
        ax.set_xlim(0, 6)
        ax.set_ylim(-10, 60)

        # Legend: frameless, reasonably small
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(
                handles,
                labels,
                frameon=False,
                loc="best",
            )

        out_pdf = Path(file_path).parent / "Figure_4_2_topography_relative.pdf"
        savefig(fig, out_pdf, dpi=300, transparent=True)
        plt.show()
        plt.close(fig)

        print(f"[Saved] {out_pdf.resolve()}")

        # -------------------------------------------------------------
        # Second figure: filled density blobs per treatment (no points)
        # -------------------------------------------------------------
        with use_style(STYLE_PATH):
            plt.rcParams["font.family"] = "Arial"

            fig2, ax2 = plt.subplots()
            set_size(ax2, width="thesis", ratio=0.70)

            # Axes cosmetics
            for spine in ("top", "right"):
                ax2.spines[spine].set_visible(False)
            ax2.tick_params(direction="out")
            ax2.set_facecolor("white")

            ax2.set_xlim(*xlim)
            ax2.set_ylim(*ylim)

            for t in unique_treatments:
                sel = (labels_rel == t)
                if sel.sum() < 3:
                    continue

                xg = x_rel[sel]
                yg = y_rel[sel]
                color = treatment_colors.get(t, "k")

                # no dots here, only filled KDE regions
                filled_contours_on_ax(ax2, xg, yg, xlim, ylim, base_color=color)

            ax2.set_xlabel("Normalized Sensitive Population")
            ax2.set_ylabel("Relative Resistant Growth [%]")

            # Legend: coloured squares, frameless, default fontsize
            handles = [
                plt.Line2D(
                    [0], [0],
                    marker="s",
                    linestyle="",
                    markerfacecolor=treatment_colors[t],
                    markeredgecolor="none",
                    label=t,
                )
                for t in unique_treatments
            ]
            if handles:
                ax2.legend(handles=handles, frameon=False, loc="best")

            out_pdf_filled = Path(file_path).parent / "Figure_4_2_topography_relative_filled.pdf"
            savefig(fig2, out_pdf_filled, dpi=300, transparent=True)
            plt.show()
            plt.close(fig2)
            print(f"[Saved] {out_pdf_filled.resolve()}")

    stats_payload = {
        "figure": "Figure 4.2",
        "dataset": str(Path(file_path)),
        "analysis_status": "descriptive_only",
        "descriptive_plot_observations": n,
        "descriptive_plot_spheroids": n_spheroids,
        "reported_original_pooled_correlation_reproduced": False,
        "interpretation": (
            "The originally reported pooled correlation could not be reproduced "
            "from the available canonical code/data path. This figure is retained "
            "as a descriptive pooled visualization only."
        ),
    }
    with stats_path.open("w", encoding="utf-8") as fh:
        json.dump(stats_payload, fh, indent=2)
    print(f"[Saved results] {stats_path.resolve()}")

def filled_contours_on_ax(
    ax,
    x,
    y,
    xlim,
    ylim,
    base_color="#1f77b4",
    gridsize=300,
    probs=(0.1, 0.2, 0.3, 0.4),
    bw_method=0.7,
    pad=0.10,
):
    """
    Draw filled KDE contours for (x, y) on ax, using the SAME level construction
    as add_kde_contours (same 'probs'), but with semi-transparent fills
    in 'base_color' instead of points.

    This way, the contour lines in the second plot coincide with those in
    the first plot; only the fill style changes.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size < 5:
        print(f"[contour] Not enough points for filled contours (n={x.size}), skipping.")
        return

    # Same domain logic as in add_kde_contours
    xm, xM = np.nanmin(x), np.nanmax(x)
    ym, yM = np.nanmin(y), np.nanmax(y)
    xr = xM - xm if xM > xm else 1.0
    yr = yM - ym if yM > ym else 1.0
    xm -= pad * xr
    xM += pad * xr
    ym -= pad * yr
    yM += pad * yr

    xx = np.linspace(xm, xM, gridsize)
    yy = np.linspace(ym, yM, gridsize)
    Xg, Yg = np.meshgrid(xx, yy)

    kde = gaussian_kde(np.vstack([x, y]), bw_method=bw_method)
    Z = kde(np.vstack([Xg.ravel(), Yg.ravel()])).reshape(Xg.shape)

    z = Z.ravel()
    if not np.isfinite(z).any() or np.nanmin(Z) == np.nanmax(Z):
        print("[contour] Flat or invalid KDE surface, skipping.")
        return

    # --- SAME probability-based levels as add_kde_contours ---
    z_sorted = np.sort(z)[::-1]
    cdf = np.cumsum(z_sorted)
    cdf /= cdf[-1]
    line_levels = []
    for p in sorted(probs):
        idx = np.searchsorted(cdf, p)
        thr = z_sorted[idx] if idx < len(z_sorted) else z_sorted[-1]
        line_levels.append(float(thr))
    line_levels = np.sort(np.unique(line_levels))

    if line_levels.size == 0 or not np.isfinite(line_levels).all():
        print("[contour] Invalid contour levels, skipping.")
        return

    # For filled regions, we define bands between min..first level..last level..max
    z_min = float(np.nanmin(Z))
    z_max = float(np.nanmax(Z))
    # We only fill BETWEEN successive inner line levels,
    # not the extreme outside region.
    # We want to fill all regions *between* levels, including the
    # highest-density core (above the last contour level).
    z_max = float(np.nanmax(Z))

    # Fill bands: [level0, level1], ..., [last_level, z_max]
    fill_levels = np.concatenate([line_levels, [z_max + 1e-12]])

    base_rgba = np.array(mcolors.to_rgba(base_color))
    n_regions = len(fill_levels) - 1
    colors = []

    for i in range(n_regions):
        # lowest band = lightest, highest-density core = strongest colour
        alpha = 0.15 + 0.55 * (i / max(1, n_regions - 1))
        rgba = base_rgba.copy()
        rgba[-1] = alpha
        colors.append(tuple(rgba))

    # Filled blobs for all interior regions, including the peak
    ax.contourf(
        Xg, Yg, Z,
        levels=fill_levels,
        colors=colors
    )

    # Contour lines: same levels as in the first figure
    ax.contour(
        Xg, Yg, Z,
        levels=line_levels,
        colors=[base_color],
        linewidths=0.8
    )


# ---------------------------------------------------------------------
# Run analysis
# ---------------------------------------------------------------------
if __name__ == "__main__":
    correlate_red_growth(combined_df)
