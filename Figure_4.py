# Figure_4.py
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ----- Paths -----
EXCEL_DIR = Path(r"Y:\Members\MaxE\Thesis\Data for Figures\Figure_4")
STYLE_PATH = Path("figstyle.mplstyle")           # adjust if your style file lives elsewhere
OUTDIR = EXCEL_DIR
OUTDIR.mkdir(parents=True, exist_ok=True)

EXCEL_PATH = Path(r"Y:\Members\MaxE\Thesis\Data for Figures\Figure_4\Spheroid Height DropSeeding.xlsx")
print(f"Reading: {EXCEL_PATH.name}")

# ----- Load data -----
df_raw = pd.read_excel(EXCEL_PATH, sheet_name="Tabelle2", header=0)

# Expecting columns:
# 'Cooling [min]' | 'Distance to medium [µm]' | 'SD'
df = df_raw.iloc[:, :3].copy()
df.columns = ["Cooling_min", "Distance_um", "SD"]

# Convert decimal commas to dots and to numeric
for c in ["Cooling_min", "Distance_um", "SD"]:
    df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ".", regex=False), errors="coerce")

# Clean + order
df = df.dropna(subset=["Cooling_min", "Distance_um"]).sort_values("Cooling_min").reset_index(drop=True)
print(df)

# ----- Plot -----
with plt.style.context(str(STYLE_PATH)):
    fig, ax = plt.subplots()

    # Black line + markers + SD whiskers
    ax.errorbar(
        df["Cooling_min"].values,
        df["Distance_um"].values,
        yerr=df["SD"].values if "SD" in df.columns else None,
        fmt="-o",
        color="black",
        ecolor="black",
        elinewidth=1.2,
        capsize=3,
        markersize=5.5,
        linewidth=1.6,
    )

    # Labels & title
    ax.set_title("Sinking behaviour over time")
    ax.set_xlabel("Cooling time [min]")
    ax.set_ylabel("Distance to medium [µm]")

    # X ticks at 2, 4, 6, ...
    max_x = float(df["Cooling_min"].max())
    ax.set_xticks(range(2, int(max_x) + 2, 2))

    # No legend
    # (no ax.legend() call)

    # Y limit from 0 to a bit above max mean+SD
    ymax = float((df["Distance_um"] + df.get("SD", 0)).max())
    ax.set_ylim(0, ymax * 1.1 if ymax > 0 else 1)

    ax.margins(x=0.02, y=0.05)

    # Save
    pdf_path = OUTDIR / "Figure4_sinking_behaviour.pdf"
    fig.savefig(pdf_path)

    # Show interactively in PyCharm
    plt.show()
    plt.close(fig)

print(f"Saved: {pdf_path}")