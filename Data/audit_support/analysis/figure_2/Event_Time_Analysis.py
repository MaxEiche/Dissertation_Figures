"""Shared Figure 2 TTE and progression-time input analysis.

TTE is derived from ``all_escape_margin.csv`` as the first observed day with
``signed_escape_margin > 0``. Population progression is derived directly from
the experimental ``*_total_poisson_padded.h5`` files and simulation
``*_simulation.h5`` files as the first day resistant cells reach 50% of the
initial total population.

Both observed event times use acquisition days (the first image is day 1),
matching the time axes of the escape/population trajectories. Experimental
endpoint cutoffs are expressed in the same acquisition-day convention.

This module writes no files. Figure scripts remain responsible for their own
plots, statistics, and plotted-value exports.
"""

from __future__ import annotations

from pathlib import Path
import re

import h5py
import numpy as np
import pandas as pd

from experiment_filters import EXCLUDE_SAMPLES
from strategy_colors import normalize_strategy
from data_paths import EXP_RAW_DATA, intermediate_dir


EXPERIMENT_ROOT = EXP_RAW_DATA
# Prepared point-cloud HDF5s are the canonical simulation population input.
# Density-map TIFFs are a separate downstream product and contain no HDF5
# trajectories for the TTF calculation.
SIMULATION_ROOT = intermediate_dir("simulation", "basic", "prepared_pointclouds", create=False)
ESCAPE_MARGIN_CSV = intermediate_dir("combined", "figure_2", create=False) / "all_escape_margin.csv"
EXPERIMENT_H5_SUFFIX = "_total_poisson_padded.h5"
SIMULATION_H5_SUFFIX = "_simulation.h5"
DEFAULT_STRATEGIES = ("CT", "AT50", "AT100")
PROGRESSION_THRESHOLD = 0.5
EXPERIMENT_END_DAY = 96.0
TTE_COLUMNS = [
    "dataset_type", "source", "plate", "sample", "strategy", "escape_day",
    "tte", "escaped", "last_observed_day", "excluded_reason",
]
PROGRESSION_COLUMNS = [
    "dataset_type", "source", "plate", "sample", "strategy", "ttp", "ttf",
    "progressed", "initial_total",
]


def _run_number(sample: str) -> float:
    match = re.match(r"^R(\d+)_", str(sample))
    return float(match.group(1)) if match else np.nan


def load_escape_margins(path: Path = ESCAPE_MARGIN_CSV) -> pd.DataFrame:
    """Load and validate the canonical experimental/simulation margin table."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Canonical escape-margin table not found: {path}. Run "
            "analysis/figure_2/Escape_Margin_Analysis.py first."
        )
    data = pd.read_csv(path)
    required = {
        "dataset_type", "plate", "sample", "strategy", "day",
        "signed_escape_margin",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}")
    data = data.copy()
    data["dataset_type"] = data["dataset_type"].astype(str).str.lower()
    data["strategy"] = data["strategy"].astype(str).map(normalize_strategy)
    data["day"] = pd.to_numeric(data["day"], errors="coerce")
    data["signed_escape_margin"] = pd.to_numeric(
        data["signed_escape_margin"], errors="coerce"
    )
    data = data.dropna(subset=["day", "signed_escape_margin"])
    return data.drop_duplicates(
        subset=["dataset_type", "plate", "sample", "day"], keep="last"
    ).sort_values(["dataset_type", "strategy", "plate", "sample", "day"])


def derive_tte(
    margins: pd.DataFrame,
    *,
    dataset_type: str,
    strategies: tuple[str, ...] = DEFAULT_STRATEGIES,
    apply_truncation: bool = True,
    simulation_min_run: int = 10,
) -> pd.DataFrame:
    """Return replicate-level TTE rows from signed daily escape margins."""
    kind = dataset_type.lower()
    subset = margins[
        margins["dataset_type"].eq(kind) & margins["strategy"].isin(strategies)
    ].copy()
    if kind == "simulation":
        subset["run_number"] = subset["sample"].map(_run_number)
        subset = subset[subset["run_number"] >= simulation_min_run]

    rows: list[dict[str, object]] = []
    group_columns = ["plate", "sample", "strategy"]
    for (plate, sample, strategy), group in subset.groupby(group_columns, sort=True):
        group = group.sort_values("day")
        last_day = float(group["day"].max())
        escaped_days = group.loc[group["signed_escape_margin"] > 0, "day"]
        escape_day = float(escaped_days.min()) if not escaped_days.empty else np.nan

        if kind == "experimental":
            rule = EXCLUDE_SAMPLES.get(str(sample), "not_listed")
            if rule is None:
                if apply_truncation:
                    rows.append({
                        "dataset_type": kind, "source": "In vitro", "plate": plate,
                        "sample": sample, "strategy": strategy, "escape_day": escape_day,
                        "tte": np.nan, "escaped": False, "last_observed_day": last_day,
                        "excluded_reason": "excluded_by_experiment_filters",
                    })
                continue
            if apply_truncation and rule != "not_listed":
                cutoff = float(rule)
                if not np.isfinite(escape_day) or escape_day > cutoff:
                    rows.append({
                        "dataset_type": kind, "source": "In vitro", "plate": plate,
                        "sample": sample, "strategy": strategy, "escape_day": escape_day,
                        "tte": np.nan, "escaped": False, "last_observed_day": last_day,
                        "excluded_reason": f"no_escape_within_truncation_cutoff_{cutoff:g}",
                    })
                    continue

        escaped = bool(np.isfinite(escape_day))
        rows.append({
            "dataset_type": kind,
            "source": "In vitro" if kind == "experimental" else "In silico",
            "plate": plate,
            "sample": sample,
            "strategy": strategy,
            "escape_day": escape_day,
            "tte": escape_day if escaped else last_day,
            "escaped": escaped,
            "last_observed_day": last_day,
            "excluded_reason": "" if escaped else "no_escape_in_margin_table",
        })
    return pd.DataFrame(rows, columns=TTE_COLUMNS)


def _numeric_sort_key(value: str) -> tuple[int, str]:
    match = re.search(r"\d+", value)
    return (int(match.group()) if match else 10**9, value)


def _coordinate_count(dataset: h5py.Dataset) -> int:
    shape = dataset.shape
    if len(shape) == 2 and shape[1] == 2:
        return int(shape[0])
    size = int(np.prod(shape, dtype=np.int64))
    return size // 2 if size >= 2 else 0


def _group_day_counts(h5: h5py.File, group_name: str) -> dict[int, int]:
    if group_name not in h5:
        return {}
    counts: dict[int, int] = {}
    group = h5[group_name]
    for key in sorted(group.keys(), key=_numeric_sort_key):
        match = re.search(r"\d+", key)
        if not match:
            continue
        day = int(match.group())
        obj = group[key]
        dataset: h5py.Dataset | None = None
        if isinstance(obj, h5py.Dataset):
            dataset = obj
        elif isinstance(obj, h5py.Group):
            if "block0_values" in obj:
                dataset = obj["block0_values"]
            elif "values" in obj:
                dataset = obj["values"]
        counts[day] = _coordinate_count(dataset) if dataset is not None else 0
    return counts


def _read_population_file(
    path: Path,
    *,
    dataset_type: str,
    plate: str,
    sample: str,
    strategy: str,
) -> pd.DataFrame:
    with h5py.File(path, "r") as h5:
        sensitive = _group_day_counts(h5, "sus")
        resistant = _group_day_counts(h5, "res")
    days = sorted(set(sensitive).union(resistant))
    if not days:
        return pd.DataFrame()
    first_day = min(days)
    rows = []
    for day in days:
        n_sensitive = int(sensitive.get(day, 0))
        n_resistant = int(resistant.get(day, 0))
        rows.append({
            "dataset_type": dataset_type,
            "source": "In vitro" if dataset_type == "experimental" else "In silico",
            "plate": plate,
            "sample": sample,
            "strategy": strategy,
            "day": float(day),
            "day_tp1": float(day - first_day),
            "sus": float(n_sensitive),
            "res": float(n_resistant),
            "total": float(n_sensitive + n_resistant),
            "input_file": str(path),
        })
    return pd.DataFrame(rows)


def load_population_counts(
    *,
    dataset_type: str,
    strategies: tuple[str, ...] = DEFAULT_STRATEGIES,
    experiment_root: Path = EXPERIMENT_ROOT,
    simulation_root: Path = SIMULATION_ROOT,
    simulation_min_run: int = 10,
) -> pd.DataFrame:
    """Read resistant/sensitive counts directly from the canonical HDF5 inputs."""
    frames: list[pd.DataFrame] = []
    kind = dataset_type.lower()
    if kind == "experimental":
        paths = sorted(experiment_root.glob(f"Plate_*/hdf5/*{EXPERIMENT_H5_SUFFIX}"))
        for path in paths:
            sample = path.name.removeprefix("G418_a2a4_").removesuffix(EXPERIMENT_H5_SUFFIX)
            strategy = normalize_strategy(sample.split("_", 1)[1].split("-", 1)[0])
            if strategy not in strategies:
                continue
            frame = _read_population_file(
                path, dataset_type=kind, plate=path.parent.parent.name,
                sample=sample, strategy=strategy,
            )
            if not frame.empty:
                frames.append(frame)
    elif kind == "simulation":
        paths = sorted(simulation_root.glob(f"Plate_*/hdf5/*{SIMULATION_H5_SUFFIX}"))
        for path in paths:
            sample = path.name.removesuffix(SIMULATION_H5_SUFFIX)
            if _run_number(sample) < simulation_min_run:
                continue
            strategy = normalize_strategy(path.parent.parent.name.removeprefix("Plate_"))
            if strategy not in strategies:
                continue
            frame = _read_population_file(
                path, dataset_type=kind, plate=path.parent.parent.name,
                sample=sample, strategy=strategy,
            )
            if not frame.empty:
                frames.append(frame)
    else:
        raise ValueError(f"Unknown dataset_type {dataset_type!r}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def derive_progression(
    counts: pd.DataFrame,
    *,
    apply_truncation: bool = True,
    threshold: float = PROGRESSION_THRESHOLD,
    experiment_end_day: float = EXPERIMENT_END_DAY,
) -> pd.DataFrame:
    """Derive the first 50%-resistant progression acquisition day per replicate."""
    if counts.empty:
        return pd.DataFrame(columns=PROGRESSION_COLUMNS)
    rows: list[dict[str, object]] = []
    keys = ["dataset_type", "source", "plate", "sample", "strategy"]
    for key, group in counts.groupby(keys, sort=True):
        dataset_type, source, plate, sample, strategy = key
        group = group.sort_values("day").copy()
        if dataset_type == "experimental":
            rule = EXCLUDE_SAMPLES.get(str(sample), "not_listed")
            if rule is None:
                continue
            if apply_truncation and rule != "not_listed":
                group = group[group["day"] <= float(rule)]
        if group.empty:
            continue
        initial_total = float(group.iloc[0]["total"])
        if initial_total <= 0 or not np.isfinite(initial_total):
            continue
        hit = group.loc[group["res"] / initial_total >= threshold, "day"]
        progressed = not hit.empty
        if progressed:
            event_time = float(hit.min())
        elif dataset_type == "simulation":
            event_time = np.nan
        else:
            rule = EXCLUDE_SAMPLES.get(str(sample), "not_listed")
            truncated = apply_truncation and rule not in {"not_listed", None}
            event_time = np.nan if truncated else float(experiment_end_day)
        rows.append({
            "dataset_type": dataset_type,
            "source": source,
            "plate": plate,
            "sample": sample,
            "strategy": strategy,
            "ttp": event_time,
            "ttf": event_time,
            "progressed": progressed,
            "initial_total": initial_total,
        })
    return pd.DataFrame(rows, columns=PROGRESSION_COLUMNS)
