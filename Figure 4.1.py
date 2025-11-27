from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

from figtools import use_style, set_size, savefig  # unified figure tools

# ---------------------------------------------------
# Global layout parameters (easy to tweak)
# ---------------------------------------------------
HEIGHT_RATIO = 0.40      # overall height (smaller = flatter figure)
LEGEND_FONTSIZE = 5.5      # smaller legend text

# ---------------------------------------------------
# Paths & data loading
# ---------------------------------------------------
EXCEL_PATH = Path(
    r"Y:\Members\MaxE\Thesis\Organized Figures\Figure 4.1\Summary.xlsx"
)
OUTDIR = Path(
    r"Y:\Members\MaxE\Thesis\Organized Figures\Figure 4.1"
)
OUTDIR.mkdir(parents=True, exist_ok=True)

STYLE_PATH = Path("figstyle.mplstyle")

# Read the sheet 'Tabelle2'
df_raw = pd.read_excel(EXCEL_PATH, sheet_name="Tabelle2", header=0)

# Take columns A–F and rename
df = df_raw.iloc[:, :6].copy()
df.columns = [
    "Collagen",
    "Day",
    "Type",
    "Average_diameter",
    "Average_roundness",
    "Average_Z_roundness",
]

# Numeric cleanup
for col in ["Average_diameter", "Average_roundness", "Average_Z_roundness"]:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .astype(float)
    )

df["Day"] = pd.to_numeric(df["Day"], errors="coerce")

# Strip whitespace from categorical fields
df["Collagen"] = df["Collagen"].astype(str).str.strip()
df["Type"] = df["Type"].astype(str).str.strip()

print(df)

# ---------------------------------------------------
# Plot config
# ---------------------------------------------------
collagen_order = ["0 mg / ml Collagen", "0,5 mg / ml Collagen", "1 mg / ml Collagen"]
type_order = ["Juso", "SK"]
conditions = [(t, c) for t in type_order for c in collagen_order]

# Base hues: one per cell type
BASE = {
    "Juso": "#1B9E77",  # green
    "SK":   "#D95F02",  # orange/red
}

def shade(hex_color, shade_factor):
    """
    Blend base color toward white.
    shade_factor in [0,1]: 1 = base, lower = lighter.
    """
    r, g, b = to_rgb(hex_color)
    r = r * shade_factor + 1.0 * (1 - shade_factor)
    g = g * shade_factor + 1.0 * (1 - shade_factor)
    b = b * shade_factor + 1.0 * (1 - shade_factor)
    return (r, g, b)

# Lighter → darker with increasing collagen
SHADE_LEVEL = {
    "0 mg / ml Collagen": 0.3,
    "0,5 mg / ml Collagen": 0.6,
    "1 mg / ml Collagen": 1.0,
}

def color_for(type_name, collagen_label):
    return shade(BASE[type_name], SHADE_LEVEL[collagen_label])

# (column, title, y-label) in order of left→right panels
PLOTS = [
    ("Average_diameter",    "Spheroid Size",    "Diameter [µm]"),
    ("Average_roundness",   "Spheroid circularity",  "Planar circularity"),
    ("Average_Z_roundness", "Spheroid circularity","Vertical circularity"),
]

# ---------------------------------------------------
# Plotting: one figure with three horizontal subplots
# ---------------------------------------------------
with use_style(style_mpl=STYLE_PATH):
    # 1 row, 3 columns
    fig, axes = plt.subplots(nrows=1, ncols=3, sharex=True)
    axes = axes.flatten()

    # Add panel letters (a, b, c)
    panel_labels = ["a", "b", "c"]
    for ax, label in zip(axes, panel_labels):
        ax.text(
            -0.15, 1.05,  # position in axes fraction coordinates
            label,
            transform=ax.transAxes,
            fontweight="bold",
            va="bottom", ha="left"
        )

    # Make the overall figure thesis-wide and relatively flat
    set_size(axes[0], width="thesis", ratio=HEIGHT_RATIO)

    for ax, (col, title, ylab) in zip(axes, PLOTS):
        for (t, c) in conditions:
            sub = df[(df["Type"] == t) & (df["Collagen"] == c)].copy()
            if sub.empty:
                continue
            sub = sub.sort_values("Day")

            # shorter legend labels, e.g. "Juso, 0 mg/ml"
            coll_short = c.replace(" mg / ml Collagen", " mg/ml")

            ax.plot(
                sub["Day"].values,
                sub[col].values,
                label=f"{t}, {coll_short}",
                marker="o",
                color=color_for(t, c),
                # linewidth from figstyle.mplstyle (lines.linewidth)
            )

        ax.set_title(title)
        ax.set_xlabel("Day")
        ax.set_ylabel(ylab)

        # y-limits per metric
        if col == "Average_diameter":
            ax.set_ylim(0, 1500)
        else:
            ax.set_ylim(0, 1.1)

        ax.margins(x=0.02, y=0.05)

        # Per-axes legend: bottom-left, smaller, no title
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(
                handles,
                labels,
                loc="lower left",
                frameon=False,
                fontsize=LEGEND_FONTSIZE,
                ncol=2,
                handlelength=1.2,
                columnspacing=0.4,
                handletextpad=0.4,
            )

    fig.tight_layout()

    pdf_path = OUTDIR / "Figure 4.1 Circularity.pdf"
    savefig(fig, pdf_path, dpi=300, transparent=True)

    plt.show()

    plt.close(fig)

print(f"Saved combined figure to: {pdf_path.resolve()}")
# === END ===
