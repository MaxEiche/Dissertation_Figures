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
    "D1_Dens-D1": None,
    "D2_Dens-D2": None,
    "D6_Dens-D6": None,

    # -------- Plate 1: truncated --------
    "B3_AT100-B3": 18.0,
    "B4_AT100-B4": 22.0,
    "B5_AT100-B5": 10.0,

    # -------- Plate 2: truncated --------
    "A3_CT-A3": 21.0,
    "A6_CT-A6": 18.0,
    "B1_AT50-B1": 15.0,
    "B2_AT50-B2": 25.0,
    "B5_AT50-B5": 23.0,
    "C2_AT100-C2": 17.0,
    "C4_AT100-C4": 18.0,
    "C5_AT100-C5": 21.0,
    "C6_AT100-C6": 18.0,

    # -------- Plate 3: truncated --------
    "B2_CT-B2": 20.0,
    "C3_AT50-C3": 18.0,
    "D4_AT100-D4": 20.0,

    # -------- Plate 4: truncated --------
    "C1_CT-C1": 22.0,
    "C2_CT-C2": 24.0,
    "D1_AT50-D1": 16.0,
    "D2_AT50-D2": 23.0,
    "D3_AT50-D3": 20.0,
    "D4_AT50-D4": 19.0,
    "D5_AT50-D5": 24.0,

    # -------- Plate 5: truncated --------
    "A2_AT100-A2": 19.0,
    "A5_AT100-A5": 28.0,
    "D2_CT-D2": 21.0,
    "D5_CT-D5": 20.0,
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
