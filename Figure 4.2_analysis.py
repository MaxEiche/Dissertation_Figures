# sum_mip_intensity_per_scene.py
"""
Summarize 2D (MIP) total intensities from multi-scene, 3-channel CZI files.

What it does
------------
- Iterates over all .czi files in ROOT.
- For each file, iterates over all scenes.
- Loads data as CYX (collapsing T=1 and Z=1).
- Sums pixel intensities over Y and X for each channel (per scene).
- Writes a tidy CSV with one row per (file, day, scene, channel).

Why CYX?
--------
Your inspection shows dims (T=1, C=3, Z=1, Y=1849, X=1849). These are *already MIPs*.
Summing over Y,X in CYX is therefore the correct 2D total intensity.

Outputs
-------
- Console printout of the DataFrame
- CSV: intensity_sum2d_per_channel_per_scene.csv
- A tiny sanity report: which days were found, and any files where 'day' could not be parsed.

Requirements
------------
- Python 3.12 (recommended)
- aicsimageio==4.14.0
- numpy, pandas
- tqdm (optional; progress bars degrade gracefully if missing)

"""

from pathlib import Path
import re
import sys
from typing import Iterable, List, Optional, Tuple, Dict

import numpy as np
import pandas as pd

# --- Optional progress bars: fall back to no-op if tqdm is not installed ---
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x: Iterable, **kwargs):
        return x  # simple passthrough if tqdm is unavailable


# ----------------------------- CONFIG ---------------------------------------
ROOT = Path(r"Y:\Members\MaxE\Thesis\Organized Figures\Figure 4.2")
GLOB = "*.czi"
OUT_CSV = ROOT / "intensity_sum2d_per_channel_per_scene.csv"
# ----------------------------------------------------------------------------


def find_day_from_filename(name: str) -> Optional[int]:
    """
    Extract experimental day from a filename in a robust way.

    Accepted patterns (case-insensitive; separators allowed: space, '-', '_'):
        d0, d 0, d-0, d_0
        day0, day-0, day_0, day 0
        d11, day18, D6, Day-14, etc.

    Examples:
        '2024_04_04 d0 Doubledrop.czi'         -> 0
        '2024_04_04 Day_11 something.czi'      -> 11
        'myfile_D 6_scene.czi'                 -> 6
        'control.czi'                           -> None

    Parameters
    ----------
    name : str
        Filename or path string.

    Returns
    -------
    Optional[int]
        The parsed integer day, or None if no day-like token is found.
    """
    s = name.lower()
    # Normalize common separators to single spaces to simplify regex:
    s = re.sub(r"[_\-]+", " ", s)

    # Strict tokens with word boundaries to avoid grabbing parts of dates, etc.
    # Matches 'd 11' or 'day 11' (with any number of spaces)
    m = re.search(r"\b(?:d|day)\s*(\d{1,3})\b", s)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def per_scene_channel_sum_2d(path: Path) -> List[Tuple[str, str, float, int]]:
    """
    Load one .czi file and compute 2D (MIP) total intensities per scene & per channel.

    We request 'CYX' from AICSImage, which collapses T and Z (both are 1 in your data).
    Then we sum over Y and X for each channel, per scene.

    Returns
    -------
    List[Tuple[scene, channel_name, sum2d, n_pixels]]
    """
    from aicsimageio import AICSImage

    img = AICSImage(str(path))

    # Channel names if available; else fallback to C0, C1, ...
    try:
        ch_names = list(img.get_channel_names())
    except Exception:
        ch_names = None

    rows = []
    # Progress over scenes for this file
    for scene in tqdm(img.scenes, desc=f"Scenes in {path.name}", leave=False):
        img.set_scene(scene)

        # CYX collapses T and Z, giving a 3D array (C, Y, X)
        cyx = img.get_image_data("CYX")  # shape: (C, Y, X)
        cyx = np.asarray(cyx, dtype=np.float64)

        if cyx.ndim != 3:
            raise RuntimeError(f"Expected CYX (3D) but got shape {cyx.shape} for scene {scene} in {path.name}")

        C, Y, X = cyx.shape
        n_pixels = int(Y * X)
        names = ch_names if (ch_names and len(ch_names) == C) else [f"C{i}" for i in range(C)]

        # Sum Y,X per channel
        sums = cyx.reshape(C, -1).sum(axis=1)

        for cname, s in zip(names, sums):
            rows.append((scene, cname, float(s), n_pixels))

    return rows


def main() -> None:
    """
    Walk ROOT, parse days, compute per-scene/channel 2D intensity sums, and save CSV.
    Includes basic sanity reporting at the end.
    """
    files = sorted(ROOT.glob(GLOB))
    if not files:
        raise FileNotFoundError(f"No CZI files found in {ROOT}")

    all_rows: List[Dict] = []
    unmatched_day_files: List[str] = []

    for f in tqdm(files, desc="Files"):
        day = find_day_from_filename(f.name)
        if day is None:
            unmatched_day_files.append(f.name)

        try:
            scene_rows = per_scene_channel_sum_2d(f)
        except Exception as e:
            print(f"[ERROR] {f.name}: {e}", file=sys.stderr)
            continue

        for scene, ch, sum2d, npx in scene_rows:
            all_rows.append({
                "file": f.name,
                "path": str(f),
                "day": day,           # may be None if not found
                "scene": scene,
                "channel": ch,
                "intensity_sum_2d": sum2d,
                "n_pixels_2d": npx,
            })

    df = pd.DataFrame(all_rows).sort_values(["day", "scene", "channel"], na_position="last").reset_index(drop=True)

    # Show a preview and save
    pd.set_option("display.width", 180)
    pd.set_option("display.max_rows", 300)
    print(df)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV}")

    # ---- Sanity report ----
    unique_days = sorted({d for d in df["day"].dropna().unique().tolist()})
    print("\n=== Sanity report ===")
    print(f"Unique days parsed: {unique_days if unique_days else '—'}")
    if unmatched_day_files:
        print("Files with no day parsed (check naming like 'd0' / 'day11'):")
        for name in unmatched_day_files:
            print(f"  - {name}")
    else:
        print("All files had a recognizable day token (good).")


if __name__ == "__main__":
    main()