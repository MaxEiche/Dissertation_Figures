# ===============================================================
# === Figure 2.3: Relative Growth for G418 alpha-2 / alpha-4  ===
# ===============================================================

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Import unified figure tools
from figtools import (
    ERRORBAR_CAPSIZE,
    ERRORBAR_LINEWIDTH,
    add_panel_label,
    new_subplots,
    savefig,
    use_style,
)

# --------------------------
# GLOBAL PARAMETERS (tweak!)
# --------------------------
HEIGHT_RATIO = 0.45      # <---- ONE PLACE to adjust height of entire figure
LEGEND_COLS = 2          # <---- Can use 1, 2 or 3 depending on space

# -------------------
# Paths & parameters
# -------------------
STYLE_PATH = Path("figstyle.mplstyle")

XLSX = Path(
    r"Y:\Members\MaxE\Thesis\Organized Figures\Chapter_2\Figure_2.3\Plate_Reader_Experiments\2025_01_Plate_Reader_Results.xlsx"
)

SHEETS = ("G418 alpha-2", "G418 alpha-4")

SHEET_TITLES = {
    "G418 alpha-2": "Sensitive Cells",
    "G418 alpha-4": "Resistant Cells",
}

ROW_SLICE_MEANS = slice(281, 294)
ROW_SLICE_SDS = slice(296, 309)
COLS = [0, 1, 2, 3, 4]
COLNAMES = ["Condition", "0", "2", "5", "7"]
X_DAYS = np.array([0, 2, 5, 7], dtype=float)

# --------------------------
# Color palette
# --------------------------
PALETTES = {
    "purple_orange": ("#6A3D9A", "#FF7F0E"),
    "teal_magenta": ("#008B8B", "#C51B7D"),
    "navy_coral": ("#1F3A93", "#FF6F61"),
    "slate_rose": ("#3E4C59", "#D81B60"),
}
PALETTE_CHOICE = "purple_orange"

def make_diverging_cmap(n_colors, name="custom_div"):
    c0, c1 = PALETTES[PALETTE_CHOICE]
    cmap = LinearSegmentedColormap.from_list(name, [c0, c1])
    return [cmap(i) for i in np.linspace(0.05, 0.95, n_colors)]


# --------------------------
# Data helper functions
# --------------------------
def load_tables(sheet_name: str):
    df = pd.read_excel(XLSX, sheet_name=sheet_name, header=None)

    means = (
        df.iloc[ROW_SLICE_MEANS, COLS]
        .copy()
        .set_axis(COLNAMES, axis=1)
        .dropna(subset=["Condition"])
    )
    sds = (
        df.iloc[ROW_SLICE_SDS, COLS]
        .copy()
        .set_axis(COLNAMES, axis=1)
        .dropna(subset=["Condition"])
        .set_index("Condition")
    )

    for c in COLNAMES[1:]:
        means[c] = pd.to_numeric(means[c], errors="coerce")
        sds[c] = pd.to_numeric(sds[c], errors="coerce")

    return means, sds


# --------------------------
# Plotting
# --------------------------
with use_style(style_mpl=STYLE_PATH):
    fig, axes = new_subplots(
        1,
        2,
        width="thesis",
        ratio=HEIGHT_RATIO,
        sharex=True,
        sharey=True,
    )

    # Panel labels (a), (b)
    panel_labels = ["a", "b"]
    for ax, label in zip(axes, panel_labels):
        add_panel_label(ax, label)

    for ax in axes:
        ax.set_facecolor("white")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
        ax.set_yscale("log")
        ax.set_ylim(5e-3, 40)
        ax.set_xticks([0, 2, 5, 7])
        ax.set_xlabel("Day after Treatment Start")

    axes[0].set_ylabel("Relative Growth")

    for ax, sheet in zip(axes, SHEETS):
        means, sds = load_tables(sheet)
        conditions = means["Condition"].tolist()
        colors = make_diverging_cmap(len(conditions))

        for color, (_, row) in zip(colors, means.iterrows()):
            cond = row["Condition"]
            y = row[["0", "2", "5", "7"]].to_numpy()

            if cond in sds.index:
                yerr = sds.loc[cond, ["0", "2", "5", "7"]].to_numpy()
            else:
                yerr = np.zeros_like(y)

            ax.errorbar(
                X_DAYS,
                y,
                yerr=yerr,
                marker="o",
                elinewidth=ERRORBAR_LINEWIDTH,
                capsize=ERRORBAR_CAPSIZE,
                color=color,
                label=str(cond),
            )

        ax.set_title(SHEET_TITLES.get(sheet, sheet))

        leg = ax.legend(
            frameon=False,
            ncols=LEGEND_COLS,
            title="G418 concentration [mg/ml]",
            handlelength=1.4,
            columnspacing=0.8,
            handletextpad=0.4,
        )

    out_pdf = XLSX.parent / "Figure 2.3 RelativeGrowth_alpha2_alpha4_side_by_side.pdf"
    savefig(fig, out_pdf, dpi=300, transparent=True)

    plt.show()

    plt.close(fig)

print("Saved PDF:", out_pdf)
