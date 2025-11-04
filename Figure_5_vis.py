# === VISUALIZATION: Resistant Growth after Continuous Therapy ===
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Paths
ROOT = Path(r"Y:\Members\MaxE\Thesis\Data for Figures\Figure_5")
CSV_PATH = ROOT / "intensity_sum2d_per_channel_per_scene.csv"
STYLE_PATH = Path("figstyle.mplstyle")

# Parameters
SCENE = "E4"
CHANNEL_RESISTANT = "C0"
CHANNEL_SUSCEPTIBLE = "C1"

# --- Load and prepare ---
df = pd.read_csv(CSV_PATH)

# Filter scene and channels
sub = df[(df["scene"] == SCENE) & (df["channel"].isin([CHANNEL_RESISTANT, CHANNEL_SUSCEPTIBLE]))].copy()
sub["day"] = pd.to_numeric(sub["day"], errors="coerce")
sub = sub.dropna(subset=["day", "intensity_sum_2d", "n_pixels_2d"])

# Mean intensity per pixel
sub["mean_intensity"] = sub["intensity_sum_2d"] / sub["n_pixels_2d"].replace(0, np.nan)

# Pivot → rows: day, columns: channel
wide = (
    sub.groupby(["day", "channel"])["mean_intensity"]
       .mean()
       .unstack("channel")
       .sort_index()
)
wide = wide.rename(columns={CHANNEL_RESISTANT: "resistant", CHANNEL_SUSCEPTIBLE: "susceptible"})

# Compute total and normalize to total at first timepoint
wide["total"] = wide["resistant"] + wide["susceptible"]
first_day = float(wide.index.min())
norm_den = float(wide.loc[first_day, "total"])
if not np.isfinite(norm_den) or norm_den == 0:
    raise ValueError(f"Normalization failed: total at first timepoint (day={first_day}) is {norm_den}")
wide_norm = wide[["susceptible", "resistant", "total"]] / norm_den

# --- Plot ---
with plt.style.context(str(STYLE_PATH)):
    fig, ax = plt.subplots()

    # Plot each channel line
    ax.plot(wide_norm.index, wide_norm["susceptible"], marker="o", color="#0072B2",
            linewidth=1.6, label="Susceptible")
    ax.plot(wide_norm.index, wide_norm["resistant"], marker="o", color="#E69F00",
            linewidth=1.6, label="Resistant")
    ax.plot(wide_norm.index, wide_norm["total"], marker="o", color="black",
            linewidth=2.0, label="Total")

    # Cosmetic details
    ax.set_title("Resistant Growth after Continuous Therapy")
    ax.set_xlabel("Day after Treatment Start")
    ax.set_ylabel("Normalized Channel Intensity")
    ax.set_ylim(0, 1.2)

    # X-ticks at 2, 4, 6, …
    max_day = int(np.nanmax(wide_norm.index))
    ax.set_xticks(np.arange(2, max_day + 2, 2))

    ax.legend()
    ax.margins(x=0.02, y=0.05)

    # Save outputs
    out_png = ROOT / f"{SCENE}_normalized_timelines.png"
    out_pdf = ROOT / f"{SCENE}_normalized_timelines.pdf"
    fig.savefig(out_png)
    fig.savefig(out_pdf)

    # Uncomment to view interactively in PyCharm
    # plt.show()
    plt.close(fig)

#print("Saved:", out_png)
print("Saved:", out_pdf)
# === END VISUALIZATION ===
