#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Kaplan–Meier curves by treatment strategy + fraction-of-days-on-treatment.

Panel (a): KM survival by strategy (relative resistant threshold).
Panel (b): Per-replicate fraction of imaging days spent on treatment.

- Uses unified thesis styling via figstyle.mplstyle + figtools.
- Plots ONLY the relative threshold KM (THRESHOLD = 1.0).
- Uses canonical therapy colours from figtools.therapy_color.
- Applies EXCLUDE_SAMPLES from experiment_filters (full exclusion + truncation).
- Saves as a PDF for direct use in the dissertation.
"""

from __future__ import annotations

import csv
import math
import os
import re
from collections import defaultdict, namedtuple
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional

import numpy as np
import matplotlib.pyplot as plt

from figtools import use_style, savefig, therapy_color, set_size  # type: ignore
from experiment_filters import EXCLUDE_SAMPLES  # per-sample exclusion / truncation map

# ---------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------

# Root directory containing Plate_*/Results/Results_*.txt and treatment_schedule.txt
ROOTS: List[Path] = [Path(r"Y:\Members\MaxE\Thesis\Organized Figures\Figure 3.22 KM")]

# Name of the treatment schedule file in each Plate_* directory
SCHEDULE_BASENAME: str = "treatment_schedule.txt"

# Relative threshold for failure: resistant / baseline_total >= THRESHOLD
THRESHOLD: float = 1.0

# Whether wells that already fail at baseline are allowed
ALLOW_BASELINE_FAILURE: bool = False

# Strategies to include in the KM plot (case-insensitive)
INCLUDE_STRATEGIES: List[str] = ["AT50", "AT100", "CT", "ATAgent1", "ATAgent2", "Dens"]

# Map strategy labels (as parsed from filename or schedule) to canonical colours
STRATEGY_COLOR: Dict[str, str] = {
    "AT50":     therapy_color("AT50"),
    "AT100":    therapy_color("AT100"),
    "CT":       therapy_color("CT"),
    "ATAgent1": therapy_color("LV-Agent"),
    "ATAgent2": therapy_color("KPP-Agent"),
    "Dens":     therapy_color("AT100+"),
}

DISPLAY_NAME: Dict[str, str] = {
    "CT":       "CT",
    "AT50":     "AT50",
    "AT100":    "AT100",
    "Dens":     "AT100+",
    "ATAgent1": "KPP-Agent",
    "ATAgent2": "LV-Agent",
}

# Slightly thicker lines than default, but not huge
LINEWIDTH: float = 1.8

# Where to save outputs (inside the first ROOT)
SAVE_DIR: Path = ROOTS[0] / "KM_Plots"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

OUTFILE: Path = SAVE_DIR / f"km_plus_fraction_rel_{THRESHOLD:.1f}.pdf"
CSV_PATH: Path = SAVE_DIR / f"time_to_failure_by_well_rel_{THRESHOLD:.1f}.csv"
FAIL_CSV_PATH: Path = SAVE_DIR / f"failing_wells_rel_{THRESHOLD:.1f}.csv"

# ---------------------------------------------------------------------
# Regexes for filename and record parsing
# ---------------------------------------------------------------------

FNAME_RE = re.compile(
    r"""
    (?P<plate>Plate_\d+).*?
    [\\/]Results[\\/]Results_
    (?P<assay>[^_]+)_        # assay / experiment name
    (?P<cond>[^_]+)_         # condition
    (?P<well>[A-H]\d+)_      # well, e.g. A1
    (?P<strategy>[A-Za-z0-9]+)-  # treatment strategy, e.g. AT50, Dens, ATAgent1
    (?P<well2>[A-H]\d+)\.txt$    # second well label (ignored)
    """,
    re.IGNORECASE | re.VERBOSE,
)

DAY_RE = re.compile(r"(?:^|[_-])d(\d{1,2})$", re.IGNORECASE)

# For parsing schedules (fraction plot)
_RX_STRAT = re.compile(r"_([^_\-]+)-")  # capture between '_' and '-'
_RX_DIGITS = re.compile(r"(\d+)")

Record = namedtuple("Record", ["day", "total", "resistant"])

# ---------------------------------------------------------------------
# Helpers for KM
# ---------------------------------------------------------------------

def find_result_files(roots: Iterable[Path]) -> List[Path]:
    """Find all Results_*.txt files under the given root directories."""
    matches: List[Path] = []
    for root in roots:
        for dirpath, _, filenames in os.walk(root):
            if os.path.basename(dirpath).lower() == "results":
                for fn in filenames:
                    if fn.lower().startswith("results_") and fn.lower().endswith(".txt"):
                        matches.append(Path(dirpath) / fn)
    return matches


def parse_header_from_path(path: Path) -> Tuple[str, str, str] | None:
    """
    Extract (plate, well, strategy) from the Results_*.txt filename path.

    We keep `strategy` in the original case so it matches EXCLUDE_SAMPLES keys
    like 'A1_AT50-A1', 'A1_Dens-A1', 'A1_ATAgent1-A1'.
    """
    m = FNAME_RE.search(str(path))
    if not m:
        return None
    plate = m.group("plate")
    well = m.group("well").upper()          # 'A1', 'B3', ...
    strategy = m.group("strategy")          # keep original case: 'AT50', 'Dens', 'ATAgent1'
    return plate, well, strategy


def parse_file(filepath: Path) -> List[Record]:
    """
    Parse a Results_*.txt file into a list of Records (day, total, resistant).
    """
    records: List[Record] = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) < 3:
                continue

            rec_id = parts[0].strip().rstrip(",;")
            md = DAY_RE.search(rec_id)
            if not md:
                continue

            try:
                sensitive = int(parts[1])
                resistant = int(parts[2])
            except ValueError:
                continue

            total = sensitive + resistant
            day = int(md.group(1))
            records.append(Record(day=day, total=total, resistant=resistant))

    records.sort(key=lambda r: r.day)
    return records


def compute_failure_time(
    records: List[Record],
    threshold: float,
) -> Tuple[float | float("nan"), int, float | None]:
    """
    Compute time to failure given a list of records and a relative threshold.

    Returns
    -------
    (time_to_event_days, event_flag, baseline_total)
    """
    if not records:
        return (math.nan, 0, None)

    baseline = records[0]
    baseline_total = float(baseline.total) if baseline.total > 0 else float("nan")

    if math.isnan(baseline_total) or baseline_total == 0:
        return (math.nan, 0, None)

    start_idx = 0 if ALLOW_BASELINE_FAILURE else 1
    event_day: int | None = None

    for rec in records[start_idx:]:
        frac = rec.resistant / baseline_total
        if frac >= threshold:
            event_day = rec.day
            break

    day0 = records[0].day
    last_day = records[-1].day

    if event_day is not None:
        return (event_day - day0, 1, baseline_total)
    else:
        return (last_day - day0, 0, baseline_total)


def kaplan_meier_fit(times: List[float], events: List[int]) -> Tuple[List[float], List[float]]:
    """
    Classic Kaplan–Meier survival curve.
    """
    by_time: Dict[float, Dict[str, int]] = defaultdict(lambda: {"events": 0, "censored": 0})
    clean_pairs: List[Tuple[float, int]] = []

    for t, e in zip(times, events):
        if t is None or t == "":
            continue
        try:
            tf = float(t)
        except Exception:
            continue
        if math.isnan(tf):
            continue
        clean_pairs.append((tf, e))

    if not clean_pairs:
        return [0.0], [1.0]

    for tf, e in clean_pairs:
        if e == 1:
            by_time[tf]["events"] += 1
        else:
            by_time[tf]["censored"] += 1

    unique_times = sorted(by_time.keys())
    n = len(clean_pairs)
    S = 1.0

    t_out = [0.0]
    S_out = [1.0]

    for t in unique_times:
        d = by_time[t]["events"]
        c = by_time[t]["censored"]
        if n > 0:
            if d > 0:
                S *= (1.0 - d / n)
                t_out.append(t)
                S_out.append(S)
            n -= (d + c)
        else:
            break

    return t_out, S_out


def logrank_test(times1: List[float], events1: List[int],
                 times2: List[float], events2: List[int]) -> Tuple[float, float]:
    """
    Two-sample log-rank test (1 df) for right-censored data.
    """
    group1 = [(float(t), int(e)) for t, e in zip(times1, events1) if t is not None and t != ""]
    group2 = [(float(t), int(e)) for t, e in zip(times2, events2) if t is not None and t != ""]

    if not group1 or not group2:
        return float("nan"), float("nan")

    event_times = sorted({
        t for (t, e) in group1 if e == 1
    }.union({
        t for (t, e) in group2 if e == 1
    }))

    if not event_times:
        return float("nan"), float("nan")

    O1 = 0.0
    E1 = 0.0
    V1 = 0.0

    for t in event_times:
        n1 = sum(1 for (ti, _) in group1 if ti >= t)
        n2 = sum(1 for (ti, _) in group2 if ti >= t)
        N = n1 + n2
        if N <= 1:
            continue

        d1 = sum(1 for (ti, ei) in group1 if ti == t and ei == 1)
        d2 = sum(1 for (ti, ei) in group2 if ti == t and ei == 1)
        D = d1 + d2
        if D == 0:
            continue

        E1_t = D * (n1 / N)
        var_t = (n1 * n2 * D * (N - D)) / (N * N * (N - 1)) if N > 1 else 0.0

        O1 += d1
        E1 += E1_t
        V1 += var_t

    if V1 <= 0:
        return float("nan"), float("nan")

    chi2_stat = (O1 - E1) ** 2 / V1

    z = math.sqrt(chi2_stat)
    phi = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    p_value = 2.0 * (1.0 - phi)

    return chi2_stat, p_value


# ---------------------------------------------------------------------
# Helpers for fraction-of-days-on-treatment (panel b)
# ---------------------------------------------------------------------

def parse_strategy_from_sample_key(sample_key: str) -> Optional[str]:
    m = _RX_STRAT.search(sample_key)
    return m.group(1) if m else None


def parse_days(tokens: List[str]) -> Tuple[List[int], Optional[int]]:
    """
    Return (start_days, stop_day) from a schedule tokens list.
    Matches the logic from the test script.
    """
    starts: List[int] = []
    stop: Optional[int] = None
    for t in tokens:
        if not t:
            continue
        if "s" in t:
            m = _RX_DIGITS.search(t)
            if m:
                try:
                    stop = int(m.group(1))
                except ValueError:
                    pass
        else:
            if t.isdigit():
                starts.append(int(t))
            else:
                m = _RX_DIGITS.search(t)
                if m:
                    try:
                        starts.append(int(m.group(1)))
                    except ValueError:
                        pass
    return starts, stop


def compute_rep_stats(start_days: List[int], stop_day: Optional[int]) -> Tuple[set[int], int]:
    """
    Compute treated days set and number of imaging days.

    Tokens without 's' are interpreted as imaging days that are on treatment.
    Tokens with 's' (e.g. '43s') only mark the stop and are ignored here.

    Example:
    '1 2 3 ... 41 42 43s'  ->  treated_days = 42, imaging_days = 42, fraction = 1.0
    """
    # Unique treated days = numeric tokens
    treated: set[int] = set(start_days)

    # Total imaging days = last numeric day in the schedule
    imaging_days = max(treated) if treated else 0

    return treated, imaging_days



def find_schedule_files(roots: Iterable[Path]) -> List[Path]:
    """Find all treatment_schedule.txt files under the given roots."""
    matches: List[Path] = []
    for root in roots:
        for dirpath, _, filenames in os.walk(root):
            if SCHEDULE_BASENAME in filenames:
                matches.append(Path(dirpath) / SCHEDULE_BASENAME)
    return matches


def compute_per_strategy_fractions() -> Dict[str, List[float]]:
    """
    Parse treatment_schedule.txt files and compute per-replicate fractions of
    imaging days spent on treatment, grouped by strategy.

    Wells with EXCLUDE_SAMPLES[sample_key] is None are removed.
    """
    schedule_files = find_schedule_files(ROOTS)
    per_strategy_fracs: Dict[str, List[float]] = defaultdict(list)

    if not schedule_files:
        print("[FRAC] No schedule files found for fraction panel.")
        return per_strategy_fracs

    include_upper = [s.upper() for s in INCLUDE_STRATEGIES]

    for sched_path in schedule_files:
        print(f"[FRAC] read {sched_path}")
        with open(sched_path, "r", encoding="utf-8") as fh:
            for raw in fh:
                parts = raw.strip().split()
                if not parts:
                    continue

                sample_key = parts[0]           # e.g. "A1_AT50-A1"
                tokens = parts[1:]

                # Apply EXCLUDE_SAMPLES: fully excluded wells (None) are skipped
                cutoff = EXCLUDE_SAMPLES.get(sample_key, "KEEP")
                if cutoff is None:
                    continue

                # Parse strategy from sample_key
                strat = parse_strategy_from_sample_key(sample_key)
                if not strat:
                    print(f"  [FRAC warn] cannot parse strategy in '{sample_key}'")
                    continue

                # Map to canonical strategy label and enforce inclusion list
                canonical = next(
                    (s for s in INCLUDE_STRATEGIES if s.upper() == strat.upper()),
                    strat,
                )
                if canonical.upper() not in include_upper:
                    continue

                start_days, stop_day = parse_days(tokens)
                treated_set, imaging_days = compute_rep_stats(start_days, stop_day)
                treated_days = len(treated_set)

                if imaging_days <= 0:
                    continue

                frac = treated_days / imaging_days
                if math.isfinite(frac):
                    per_strategy_fracs[canonical].append(float(frac))

    # Debug info
    print("\n[FRAC] Per-strategy replicate counts and median fractions:")
    for strat in sorted(per_strategy_fracs.keys()):
        arr = np.array(per_strategy_fracs[strat], dtype=float)
        if arr.size == 0:
            continue
        print(f"  {strat}: n={arr.size}, median={np.nanmedian(arr):.3f}")

    return per_strategy_fracs


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    result_files = find_result_files(ROOTS)
    if not result_files:
        print("No Results_*.txt files found under:", ROOTS)
        return

    groups: Dict[str, List[dict]] = defaultdict(list)
    summary_rows: List[dict] = []

    include_upper = [s.upper() for s in INCLUDE_STRATEGIES]

    for fp in result_files:
        header = parse_header_from_path(fp)
        if not header:
            continue

        plate, well, strategy = header

        # Strategy inclusion filter (case-insensitive)
        if strategy.upper() not in include_upper:
            continue

        # Build sample key matching EXCLUDE_SAMPLES, e.g. "A1_AT50-A1"
        sample_key = f"{well}_{strategy}-{well}"

        # Check if this sample should be excluded or truncated
        cutoff = EXCLUDE_SAMPLES.get(sample_key, "KEEP")

        if cutoff is None:
            # Fully exclude: skip this well entirely
            continue

        # Parse time series for this well
        records = parse_file(fp)
        if not records:
            continue

        # Numeric cutoff (if any)
        cutoff_val: float | None = None
        if cutoff != "KEEP":
            try:
                cutoff_val = float(cutoff)
            except (TypeError, ValueError):
                cutoff_val = None

        # Apply truncation at cutoff_val (if defined)
        if cutoff_val is not None:
            records = [r for r in records if r.day <= cutoff_val]
            if not records:
                continue

        # Compute failure time on the (possibly truncated) records
        t_val, event_flag, baseline_total = compute_failure_time(records, THRESHOLD)

        # Convert to plotted time (days since baseline + 1)
        if not math.isnan(t_val):
            time_value = float(t_val) + 1.0

            # For truncated + censored samples, override time with the cutoff
            if cutoff_val is not None and event_flag == 0:
                time_value = cutoff_val
        else:
            time_value = ""

        row = {
            "plate": plate,
            "well": well,
            "strategy": strategy,
            "filepath": str(fp),
            "time": time_value,
            "event": int(event_flag),
            "baseline_total": int(baseline_total) if baseline_total and not math.isnan(baseline_total) else "",
            "well_id": f"{plate}:{well}",
            "sample_key": sample_key,
        }

        groups[strategy].append(row)
        summary_rows.append(row)

    if not summary_rows:
        print("No usable data after filtering.")
        return

    # -----------------------------------------------------------------
    # Save per-well summaries
    # -----------------------------------------------------------------
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Wrote per-well summary: {CSV_PATH}")

    with open(FAIL_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["plate", "well", "strategy", "event_time_days", "filepath", "sample_key"],
        )
        writer.writeheader()
        for r in summary_rows:
            if r["time"] != "" and r["event"] == 1:
                writer.writerow(
                    {
                        "plate": r["plate"],
                        "well": r["well"],
                        "strategy": r["strategy"],
                        "event_time_days": r["time"],
                        "filepath": r["filepath"],
                        "sample_key": r["sample_key"],
                    }
                )
    print(f"Wrote failing wells list: {FAIL_CSV_PATH}")

    # -----------------------------------------------------------------
    # Debug: per-strategy event / censor summary + log-rank tests
    # -----------------------------------------------------------------
    print("\n[KM DEBUG] Per-strategy events and censoring:")
    for strategy, rows in sorted(groups.items(), key=lambda kv: kv[0]):
        canonical = next(
            (s for s in INCLUDE_STRATEGIES if s.upper() == strategy.upper()),
            strategy,
        )
        events = [r for r in rows if r["time"] != "" and r["event"] == 1]
        censored = [r for r in rows if r["time"] != "" and r["event"] == 0]

        print(f"  Strategy {canonical}: "
              f"n_total = {len(events) + len(censored)} "
              f"(events = {len(events)}, censored = {len(censored)})")

        for r in sorted(events, key=lambda rr: float(rr["time"])):
            print(f"    EVENT    at t = {float(r['time']):.1f} d  -> {r['sample_key']}")

        for r in sorted(censored, key=lambda rr: float(rr['time'])):
            print(f"    CENSORED at t = {float(r['time']):.1f} d  -> {r['sample_key']}")

    # Log-rank tests
    survival_data: Dict[str, Tuple[List[float], List[int]]] = {}
    for strategy, rows in groups.items():
        times = [float(r["time"]) for r in rows if r["time"] != ""]
        events = [int(r["event"]) for r in rows if r["time"] != ""]
        if times:
            survival_data[strategy] = (times, events)

    strategies = sorted(survival_data.keys())

    print("\n[logrank] Pairwise log-rank tests (1 df):")
    if len(strategies) < 2:
        print("[logrank] Not enough groups for pairwise tests.")
    else:
        for i in range(len(strategies)):
            for j in range(i + 1, len(strategies)):
                s1, s2 = strategies[i], strategies[j]
                t1, e1 = survival_data[s1]
                t2, e2 = survival_data[s2]

                chi2_stat, p_val = logrank_test(t1, e1, t2, e2)

                canonical1 = next(
                    (s for s in INCLUDE_STRATEGIES if s.upper() == s1.upper()),
                    s1,
                )
                canonical2 = next(
                    (s for s in INCLUDE_STRATEGIES if s.upper() == s2.upper()),
                    s2,
                )

                print(
                    f"[logrank] {canonical1} vs {canonical2}: "
                    f"chi2 = {chi2_stat:.3f}, p = {p_val:.3g}, "
                    f"n_{canonical1} = {len(t1)}, n_{canonical2} = {len(t2)}"
                )

    # -----------------------------------------------------------------
    # Compute per-strategy treatment fractions (for panel b)
    # -----------------------------------------------------------------
    per_strategy_fracs = compute_per_strategy_fractions()

    # Order strategies consistently across both panels
    ordered_strategies = [
        s for s in INCLUDE_STRATEGIES
        if s in per_strategy_fracs or any(
            k.upper() == s.upper() for k in groups.keys()
        )
    ]

    # -----------------------------------------------------------------
    # Combined figure: panel (a) KM, panel (b) fraction of days on treatment
    # -----------------------------------------------------------------
    with use_style("figstyle.mplstyle"):
        plt.rcParams["font.family"] = "Arial"

        fig, axes = plt.subplots(
            1, 2,
            sharex=False,
            sharey=False,
            gridspec_kw={"width_ratios": [2, 1]},  # KM 2/3, fraction 1/3
        )
        ax_km, ax_frac = axes

        # Set overall size from left axis
        set_size(ax_km, width="thesis", ratio=0.62)

        # Panel labels
        labels = ["a", "b"]
        for ax, lab in zip(axes, labels):
            ax.text(
                -0.15, 1.05,
                lab,
                transform=ax.transAxes,
                fontweight="bold",
                va="bottom",
                ha="left",
            )

        # Common axis cosmetics
        for ax in axes:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(direction="out")

        # ---------- Panel (a): KM ----------
        for strategy in ordered_strategies:
            # Find rows for this strategy (case-insensitive)
            rows = []
            for s_key, data_rows in groups.items():
                if s_key.upper() == strategy.upper():
                    rows = data_rows
                    break
            if not rows:
                continue

            canonical = strategy  # canonical from INCLUDE_STRATEGIES
            color = STRATEGY_COLOR.get(canonical, "#000000")
            display = DISPLAY_NAME.get(canonical, canonical)

            # Extract times and event flags
            times = [float(r["time"]) for r in rows if r["time"] != ""]
            events = [int(r["event"]) for r in rows if r["time"] != ""]
            if not times:
                continue

            # KM fit
            t_vals, surv_vals = kaplan_meier_fit(times, events)

            # Extend to last observed time
            current_day = max(times) if times else 0
            if t_vals[-1] < current_day:
                t_vals.append(current_day)
                surv_vals.append(surv_vals[-1])

            # Main KM curve
            line, = ax_km.plot(
                t_vals,
                surv_vals,
                label=f"{display} (n={len(times)})",
                linewidth=LINEWIDTH,
                drawstyle="steps-post",
                color=color,
            )

            # Marker at last point
            ax_km.plot(
                t_vals[-1],
                surv_vals[-1],
                marker="o",
                color=line.get_color(),
                markersize=4,
            )

            # Censoring ticks
            censored_times = [t for t, e in zip(times, events) if e == 0]
            if censored_times:
                for ct in censored_times:
                    y = surv_vals[0]
                    for tv, sv in zip(t_vals, surv_vals):
                        if tv <= ct:
                            y = sv
                        else:
                            break
                    ax_km.plot(
                        ct,
                        y,
                        marker="|",
                        markersize=8,
                        markeredgewidth=LINEWIDTH * 0.5,
                        color=line.get_color(),
                    )

        # KM axis labels, limits, grid, legend
        ax_km.set_xlabel("Time [days]")
        ax_km.set_ylabel("Survival")
        ax_km.set_ylim(0, 1.05)
        ax_km.set_xlim(left=0)

        ax_km.grid(True, alpha=0.2, linestyle="--")

        ax_km.legend(
            frameon=False
        )


        # ---------- Panel (b): fraction of days on treatment ----------
        # Only keep strategies with data
        order_frac = [s for s in ordered_strategies if s in per_strategy_fracs]

        # Compute medians and sort high → low
        medians = {
            s: np.nanmedian(per_strategy_fracs[s]) for s in order_frac
            if len(per_strategy_fracs[s]) > 0
        }
        order_frac = sorted(order_frac, key=lambda s: medians.get(s, np.nan), reverse=True)

        if order_frac:
            # Convert fractions [0,1] → percent [0,100] for plotting
            data_pct = [np.array(per_strategy_fracs[s], dtype=float) * 100.0 for s in order_frac]

            positions = np.arange(1, len(order_frac) + 1)

            # Boxplot (no outliers)
            bp = ax_frac.boxplot(
                data_pct,
                positions=positions,
                showfliers=False,
                widths=0.6,
            )

            # Colour boxes black; jittered points carry therapy colours
            for element in ["boxes", "whiskers", "caps", "medians"]:
                for item in bp[element]:
                    item.set(color="black", linewidth=1.0)

            # Jittered points (in %)
            for i, strat in enumerate(order_frac, start=1):
                arr = np.array(per_strategy_fracs[strat], dtype=float) * 100.0
                if arr.size == 0:
                    continue
                x_jitter = np.random.normal(loc=i, scale=0.06, size=arr.size)
                ax_frac.scatter(
                    x_jitter,
                    arr,
                    s=10,
                    alpha=0.7,
                    c=STRATEGY_COLOR.get(strat, "black"),
                    edgecolors="none",
                )

            # X-ticks with display names
            ax_frac.set_xticks(positions)
            ax_frac.set_xticklabels(
                [DISPLAY_NAME.get(s, s) for s in order_frac],
                rotation=40,
                ha="right",
            )

            # Y-axis: percent
            ax_frac.set_ylabel("Days under Treatment [%]")
            ax_frac.set_ylim(-5, 105)

        # Layout according to thesis rules (no tight_layout)
        fig.subplots_adjust(left=0.12, right=0.98, wspace=0.30)

        savefig(fig, OUTFILE, dpi=300, transparent=True)
        plt.show()
        plt.close(fig)
        print(f"Saved combined KM + fraction figure: {OUTFILE}")


if __name__ == "__main__":
    main()
