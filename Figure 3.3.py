# === Thawing-time Dependency (Relative Drop vs # of Splits) ===
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from figtools import (
    ERRORBAR_LINEWIDTH,
    GRID_ALPHA,
    GRID_LINEWIDTH,
    new_figure,
    savefig,
    use_style,
)

# -------------------
# Paths & parameters
# -------------------
STYLE_PATH = Path("figstyle.mplstyle")

XLSX = Path(
    r"Y:\Members\MaxE\Thesis\Organized Figures\Chapter_3\Figure_3.3\Thawing_time_dependency_delay_quantification.xlsx"
)

SHEET = "Sheet1"
USECOLS = "AA:AC"
START_ROW = 7
N_ROWS = 5

# -------------------
# Load data
# -------------------
df = pd.read_excel(
    XLSX,
    sheet_name=SHEET,
    header=None,
    usecols=USECOLS,
    skiprows=START_ROW,
    nrows=N_ROWS,
    engine="openpyxl",
)

df.columns = ["splits_after_thaw", "relative_drop_mean", "relative_drop_sd"]

# Convert European decimal commas
for col in df.columns:
    df[col] = df[col].astype(str).str.replace(",", ".", regex=False)
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["splits_after_thaw", "relative_drop_mean"])

x = df["splits_after_thaw"].to_numpy()
y = df["relative_drop_mean"].to_numpy()
yerr = df["relative_drop_sd"].to_numpy()

# -------------------
# Plot (single column: 85 mm wide)
# -------------------
with use_style(style_mpl=STYLE_PATH):
    fig, ax = new_figure(width="single", ratio=0.7)

    ax.errorbar(
        x,
        y,
        yerr=yerr,
        marker="o",
        linestyle="-",
        color="black",
        markerfacecolor="black",
        markeredgecolor="black",
        elinewidth=ERRORBAR_LINEWIDTH,
    )

    ax.set_ylim(0, 0.6)

    ax.set_xlabel("Passages after Thawing")
    ax.set_ylabel("Normalized Size Decrease")

    # X ticks
    if len(x) > 0:
        xmin, xmax = int(np.floor(x.min())), int(np.ceil(x.max()))
        ax.set_xticks(np.arange(xmin, xmax + 1, 1))

    ax.grid(True, linestyle="--", linewidth=GRID_LINEWIDTH, alpha=GRID_ALPHA)

    out_pdf = XLSX.parent / "Thawing_time_dependency.pdf"
    savefig(fig, out_pdf, dpi=300, transparent=True)

    plt.show()

    plt.close(fig)

print("Saved PDF:", out_pdf)
# === END ===
