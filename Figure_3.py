from pathlib import Path
import pandas as pd

# Path to your Excel file
EXCEL_PATH = Path(r"Y:\Members\MaxE\Thesis\Data for Figures\Figure_3\Summary.xlsx")

# Read the sheet 'Tabelle2'
df_raw = pd.read_excel(EXCEL_PATH, sheet_name="Tabelle2", header=0)

# Optionally, rename columns explicitly (in case Excel headers differ)
df = df_raw.iloc[:, :6]  # ensure we only take columns A–F
df.columns = [
    "Collagen",
    "Day",
    "Type",
    "Average_diameter",
    "Average_roundness",
    "Average_Z_roundness",
]

# Clean up numeric columns (convert commas to dots etc.)
for col in ["Average_diameter", "Average_roundness", "Average_Z_roundness"]:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .astype(float)
    )

# Convert 'Day' to numeric if it isn’t already
df["Day"] = pd.to_numeric(df["Day"], errors="coerce")

# Strip whitespace from categorical fields
df["Collagen"] = df["Collagen"].astype(str).str.strip()
df["Type"] = df["Type"].astype(str).str.strip()

# Show the organized dataframe
print(df)

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from pathlib import Path

STYLE_PATH = Path("figstyle.mplstyle")
OUTDIR = Path(r'Y:\Members\MaxE\Thesis\Data for Figures\Figure_3')
OUTDIR.mkdir(parents=True, exist_ok=True)

collagen_order = ["0 mg / ml Collagen", "0,5 mg / ml Collagen", "1 mg / ml Collagen"]
type_order = ["Juso", "SK"]
conditions = [(t, c) for t in type_order for c in collagen_order]

# Base hues: one per Type
BASE = {
    "Juso": "#1B9E77",  # green
    "SK":   "#D95F02",  # red
}

# Shade helper: blend base color toward white; shade in [0,1], 1 = base color, lower = lighter
def shade(hex_color, shade_factor):
    r, g, b = to_rgb(hex_color)
    r = r * shade_factor + 1.0 * (1 - shade_factor)
    g = g * shade_factor + 1.0 * (1 - shade_factor)
    b = b * shade_factor + 1.0 * (1 - shade_factor)
    return (r, g, b)

# Shades per collagen level (lighter → darker with increasing collagen)
SHADE_LEVEL = {
    "0 mg / ml Collagen": 0.3,
    "0,5 mg / ml Collagen": 0.6,
    "1 mg / ml Collagen": 1.0,
}

def color_for(type_name, collagen_label):
    return shade(BASE[type_name], SHADE_LEVEL[collagen_label])

plots = [
    ("Average_diameter",    "Average diameter over time",    "Average diameter [µm]",    "avg_diameter"),
    ("Average_roundness",   "Planar circularity over time",   "Planar circularity",    "avg_roundness"),
    ("Average_Z_roundness", "Vertical circularity over time", "Vertical circularity",  "avg_z_roundness"),
]

with plt.style.context(str(STYLE_PATH)):
    for col, title, ylab, stub in plots:
        fig, ax = plt.subplots()

        for (t, c) in conditions:
            sub = df[(df["Type"] == t) & (df["Collagen"] == c)].copy()
            if sub.empty:
                continue
            sub = sub.sort_values("Day")
            ax.plot(
                sub["Day"].values,
                sub[col].values,
                label=f"{t} — {c}",
                marker="o",
                linewidth=1.6,
                color=color_for(t, c),
            )

        ax.set_title(title)
        ax.set_xlabel("Day")
        ax.set_ylabel(ylab)
        # ---- Y limits per your spec ----
        if col == "Average_diameter":
            ax.set_ylim(0, 1500)
        else:
            ax.set_ylim(0, 1.1)
        # --------------------------------
        ax.legend(ncol=2)
        ax.margins(x=0.02, y=0.05)

        png_path = OUTDIR / f"Figure3_{stub}.png"
        pdf_path = OUTDIR / f"Figure3_{stub}.pdf"
        fig.savefig(pdf_path)

        # If you want to see them in PyCharm interactively:
        plt.show()

        # If running headless / batch, keep them closed:
        plt.close(fig)

print(f"Saved figures to: {OUTDIR.resolve()}")
# === END REPLACEMENT ===