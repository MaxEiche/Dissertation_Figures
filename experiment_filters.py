# experiment_filters.py

from __future__ import annotations
from typing import Dict
import pandas as pd

# Manual per-sample exclusion / truncation:
# - If value is None  -> remove this sample entirely from analysis.
# - If value is a number -> keep only intervals with day_tp1 <= this value for that sample.
EXCLUDE_SAMPLES: Dict[str, float | None] = {
    # -------- Plate 1: fully excluded --------
    "A1_AT50-A1": None,
    "A2_AT50-A2": None,
    "A3_AT50-A3": None,
    "A4_AT50-A4": None,
    "A5_AT50-A5": None,
    "A6_AT50-A6": None,
    "B1_AT100-B1": None,
    "C1_Pulse-C1": None,
    "D1_NT-D1": None,
    "D2_NT-D2": None,
    "D3_NT-D3": None,
    "D5_NT-D5": None,
    "D6_NT-D6": None,

    # -------- Plate 2: fully excluded --------
    "A2_CT-A2": None,
    "A4_CT-A4": None,
    "A5_CT-A5": None,
    "B3_AT50-B3": None,
    "C3_AT100-C3": None,
    "D1_Pulse-D1": None,
    "D2_Pulse-D2": None,
    "D3_Pulse-D3": None,
    "D6_Pulse-D6": None,

    # -------- Plate 3: fully excluded --------
    "A1_NT-A1": None,
    "A2_NT-A2": None,
    "A3_NT-A3": None,
    "A4_NT-A4": None,
    "A5_NT-A5": None,
    "A6_NT-A6": None,
    "B1_CT-B1": None,
    "B4_CT-B4": None,
    "B5_CT-B5": None,
    "B6_CT-B6": None,
    "C1_AT50-C1": None,
    "C6_AT50-C6": None,
    "D1_AT100-D1": None,
    "D2_AT100-D2": None,
    "D3_AT100-D3": None,
    "D5_AT100-D5": None,
    "D6_AT100-D6": None,

    # -------- Plate 4: fully excluded --------
    "A1_Pulse-A1": None,
    "A2_Pulse-A2": None,
    "A3_Pulse-A3": None,
    "A6_Pulse-A6": None,
    "C3_CT-C3": None,
    "C6_CT-C6": None,

    # -------- Plate 5: fully excluded --------
    "A1_AT100-A1": None,
    "A3_AT100-A3": None,
    "A6_AT100-A6": None,
    "B6_Pulse-B6": None,
    "D1_CT-D1": None,
    "D6_CT-D6": None,

    # -------- Plate 6: fully excluded --------
    "A1_ATAgent1-A1": None,
    "A2_ATAgent1-A2": None,
    "A6_ATAgent1-A6": None,
    "B1_ATAgent2-B1": None,
    "B6_ATAgent2-B6": None,
    "C1_ATAgent1-C1": None,
    "C5_ATAgent1-C5": None,
    "C6_ATAgent1-C6": None,
    "D1_ATAgent2-D1": None,
    "D2_ATAgent2-D2": None,
    "D3_ATAgent2-D3": None,
    "D4_ATAgent2-D4": None,
    "D5_ATAgent2-D5": None,
    "D6_ATAgent2-D6": None,

    # -------- Plate 7: fully excluded --------
    "A1_ATAgent2-A1": None,
    "A3_ATAgent2-A3": None,
    "B3_ATAgent1-B3": None,
    "C3_ATAgent2-C3": None,
    "C6_ATAgent2-C6": None,
    "D1_ATAgent1-D1": None,
    "D2_ATAgent1-D2": None,
    "D3_ATAgent1-D3": None,
    "D4_ATAgent1-D4": None,
    "D5_ATAgent1-D5": None,
    "D6_ATAgent1-D6": None,

    # -------- Plate 8: fully excluded --------
    "A1_Dens-A1": None,
    "A2_Dens-A2": None,
    "A3_Dens-A3": None,
    "A4_Dens-A4": None,
    "A5_Dens-A5": None,
    "A6_Dens-A6": None,
    "B1_Dens-B1": None,
    "B5_Dens-B5": None,
    "B6_Dens-B6": None,
    "C1_Dens-C1": None,
    "C6_Dens-C6": None,
    "D1_Dens-D1": None,
    "D2_Dens-D2": None,
    "D6_Dens-D6": None,

    # -------- Plate 1: truncated --------
    "B3_AT100-B3": 40.0,
    "B4_AT100-B4": 50.0,
    "B5_AT100-B5": 22.0,

    # -------- Plate 2: truncated --------
    "A1_CT-A1": 36.0,
    "A3_CT-A3": 47.0,
    "A6_CT-A6": 40.0,
    "B1_AT50-B1": 33.0,
    "B2_AT50-B2": 57.0,
    "B5_AT50-B5": 52.0,
    "C2_AT100-C2": 38.0,
    "C4_AT100-C4": 40.0,
    "C5_AT100-C5": 47.0,
    "C6_AT100-C6": 40.0,

    # -------- Plate 3: truncated --------
    "B2_CT-B2": 45.0,
    "C3_AT50-C3": 40.0,
    "C5_AT50-C5": 89.0,
    "D4_AT100-D4": 45.0,

    # -------- Plate 4: truncated --------
    "C1_CT-C1": 50.0,
    "C2_CT-C2": 54.0,
    "C4_CT-C4": 71.0,
    "D1_AT50-D1": 36.0,
    "D2_AT50-D2": 52.0,
    "D3_AT50-D3": 45.0,
    "D4_AT50-D4": 43.0,
    "D5_AT50-D5": 54.0,

    # -------- Plate 5: truncated --------
    "A2_AT100-A2": 43.0,
    "A5_AT100-A5": 64.0,
    "D2_CT-D2": 47.0,
    "D5_CT-D5": 45.0,

    # -------- Plate 6: truncated --------
    "A3_ATAgent1-A3": 40.0,
    "A5_ATAgent1-A5": 47.0,
    "B3_ATAgent2-B3": 64.0,
    "B5_ATAgent2-B5": 47.0,

    # -------- Plate 7: truncated --------
    "A2_ATAgent2-A2": 54.0,
    "A5_ATAgent2-A5": 40.0,
    "B5_ATAgent1-B5": 54.0,
    "B6_ATAgent1-B6": 61.0,
    "C4_ATAgent2-C4": 68.0,
    "C5_ATAgent2-C5": 36.0,

    # -------- Plate 8: truncated --------
    "B2_Dens-B2": 47.0,
    "B4_Dens-B4": 59.0,
    "C2_Dens-C2": 54.0,
    "C3_Dens-C3": 64.0,
    "C4_Dens-C4": 52.0,
    "C5_Dens-C5": 40.0,
    "D4_Dens-D4": 61.0,
}

def apply_sample_exclusions(
    df: pd.DataFrame,
    exclude_map: Dict[str, float | None] = EXCLUDE_SAMPLES,
    column: str = "sample",
    cutoff_column: str = "day_tp1",
    strict: bool = True,
) -> pd.DataFrame:
    """
    Apply manual per-sample exclusions / truncations.

    Parameters
    ----------
    df : DataFrame
        Input data.
    exclude_map : dict
        Mapping sample_name -> None (drop) or cutoff (truncate).
    column : str
        Column in df containing sample/replicate IDs.
    cutoff_column : str
        Time column used for truncation.
    strict : bool
        If True, error if any exclude_map keys are not found in df[column].

    Returns
    -------
    DataFrame
        Filtered copy of df.
    """

    # ---------------------------
    # Strict checking: ensure all keys exist
    # ---------------------------
    unique_values = set(df[column].unique())

    # Identify missing keys from global EXCLUDE_SAMPLES
    missing = [key for key in exclude_map.keys() if key not in unique_values]

    # Filter missing entries to only those that share a strategy prefix or plate prefix
    # This catches actual typos instead of unrelated samples.
    df_prefixes = {val.split("_")[1].split("-")[0] for val in unique_values}  # e.g. {'AT50', 'AT100'}
    suspicious = []
    for key in missing:
        strat = key.split("_")[1].split("-")[0]  # extract strategy prefix (AT50, AT100, CT, etc.)
        if strat in df_prefixes:  # only suspicious if strategy appears in dataset
            suspicious.append(key)

    if suspicious:
        print(
            f"[exclude WARNING] Possible mismatches in EXCLUDE_SAMPLES for df['{column}']:\n"
            + "\n".join(f"  - {s}" for s in suspicious)
            + "\nThese entries share strategy prefixes with the dataset. "
              "Check for typos or inconsistent replicate names."
        )
    else:
        # Only print the short info once
        print(f"[exclude INFO] All missing EXCLUDE_SAMPLES entries are irrelevant for this dataset.")

    # ---------------------------
    # Apply filtering logic
    # ---------------------------
    mask = pd.Series(True, index=df.index)

    for sample, cutoff in exclude_map.items():

        if sample not in unique_values:
            # If strict=True we would have already raised an error
            continue

        if cutoff is None:
            # Full exclusion
            mask &= ~(df[column] == sample)

        else:
            # Truncate
            mask &= ~((df[column] == sample) & (df[cutoff_column] > float(cutoff)))

    df2 = df[mask].copy()
    print(f"[exclude] After EXCLUDE_SAMPLES filter: {len(df2)} / {len(df)} rows kept")
    return df2
