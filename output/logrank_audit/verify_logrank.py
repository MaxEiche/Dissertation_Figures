"""Read-only numerical audit; writes only local audit artifacts.

Uses the analysis' actual event-time derivation and extracts pure statistical
functions without importing plotting modules (whose top-level code creates
output folders). Checks every raw log-rank p-value against lifelines.
"""
from __future__ import annotations

import ast
import csv
import json
import math
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2
from lifelines.statistics import logrank_test as reference_logrank

sys.stdout.reconfigure(encoding="utf-8")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from data_paths import DATA_ROOT, output_dir

CODE_ROOT = DATA_ROOT / "audit_support"
OUT = output_dir("logrank_audit")
sys.path.insert(0, str(CODE_ROOT))
from analysis.figure_2 import Event_Time_Analysis as events
from experiment_filters import EXCLUDE_SAMPLES


def pure_functions(relative_path, names, namespace):
    path = CODE_ROOT / relative_path
    tree = ast.parse(path.read_text(encoding="utf-8"))
    selected = [node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(selected) == len(names), (relative_path, names)
    unit = ast.Module(body=[ast.ImportFrom(module="__future__", names=[
        ast.alias(name="annotations")], level=0), *selected], type_ignores=[])
    exec(compile(ast.fix_missing_locations(unit), str(path), "exec"), namespace)
    return namespace


common = dict(np=np, pd=pd, chi2=chi2, EXCLUDE_SAMPLES=EXCLUDE_SAMPLES,
              derive_progression=events.derive_progression,
              PROGRESSION_THRESHOLD=0.5)
tte = pure_functions("figures/Figure_2/b_TTE CT & AT100.py",
                     {"prepare_tte_statistics", "logrank_pvalue"}, dict(common))
ttf = pure_functions("figures/Figure_2/c_TTF CT & AT100.py",
                     {"prepare_progression_statistics", "logrank_pvalue"}, dict(common))
km = pure_functions("figures/Figure_4/e_Figure_4_Kaplan_Meier_in_vitro_and_in_silico.py",
                    {"logrank_p_value"}, {"math": math})
rows = []


def compare(table, cohort, endpoint, time_column, event_column, strategies,
            function, correction, m):
    for a, b in combinations(strategies, 2):
        x, y = (table.loc[table.strategy.eq(s)] for s in (a, b))
        tx, ex = x[time_column].to_numpy(float), x[event_column].to_numpy(int)
        ty, ey = y[time_column].to_numpy(float), y[event_column].to_numpy(int)
        p = function(tx, ex, ty, ey)
        ref = reference_logrank(tx, ty, event_observed_A=ex,
                                event_observed_B=ey).p_value
        assert np.isclose(p, ref, rtol=1e-10, atol=1e-100), (cohort, a, b, p, ref)
        rows.append(dict(figure="3.17" if endpoint == "TTE" or cohort.startswith("figure2") else "3.22",
                         cohort=cohort, endpoint=endpoint, strategy_a=a,
                         strategy_b=b, n_a=len(x), events_a=int(sum(ex)),
                         n_b=len(y), events_b=int(sum(ey)), p_raw=p,
                         correction=correction, family_size=m,
                         p_reported=min(1.0, m*p) if correction == "Bonferroni" else p,
                         lifelines_p=ref))
        print(cohort, endpoint, a, b, f"raw={p:.10g}",
              f"reported={rows[-1]['p_reported']:.10g}", flush=True)


strategies = ["CT", "AT50", "AT100"]
print("Reading saved escape-margin table", flush=True)
margins = events.load_escape_margins()
for kind in ("experimental", "simulation"):
    table = events.derive_tte(margins, dataset_type=kind,
                             apply_truncation=kind == "experimental")
    table = tte["prepare_tte_statistics"](table)
    table.to_csv(OUT / f"figure317_tte_{kind}_audit_rows.csv", index=False)
    compare(table, f"figure2_{kind}", "TTE", "tte", "escaped", strategies,
            tte["logrank_pvalue"], "none", 3)
    print("Reading population-count HDF5 metadata:", kind, flush=True)
    counts = events.load_population_counts(dataset_type=kind)
    table = ttf["prepare_progression_statistics"](
        counts, apply_truncation=kind == "experimental")
    table.to_csv(OUT / f"figure317_ttf_{kind}_audit_rows.csv", index=False)
    compare(table, f"figure2_{kind}", "TTF", "ttp", "progressed", strategies,
            ttf["logrank_pvalue"], "none", 3)

base = DATA_ROOT / "audit" / "figure_data" / "Figure_4" / "kaplan_meier"
five = ["CT", "AT50", "AT100", "Dens", "ATAgent2"]
inputs = [
    ("time_to_failure_by_well_rel_0.5.csv", "figure4_experimental_displayed"),
    ("time_to_failure_by_well_rel_0.5_no_trunc.csv", "figure4_experimental_no_trunc_sensitivity"),
    ("time_to_failure_by_run_rel_0.5_in_silico.csv", "figure4_simulation_displayed"),
]
for name, cohort in inputs:
    table = pd.read_csv(base / name)
    compare(table, cohort, "TTF", "time", "event", five,
            km["logrank_p_value"], "Bonferroni", 10)

pd.DataFrame(rows).to_csv(OUT / "pairwise_logrank_audit.csv", index=False)
print(f"Verified {len(rows)} raw p-values against lifelines; saved {OUT / 'pairwise_logrank_audit.csv'}", flush=True)

import pymupdf
thesis_path = DATA_ROOT / "audit" / "reference" / "PhD_Thesis - 2026-10-26T164617.879.pdf"
doc = pymupdf.open(thesis_path)
selected_pages = [87, 88, 94, 180]
excerpts = []
for number in selected_pages:
    page = doc[number-1]
    excerpts.append(f"PDF page {number}\n{page.get_text()}")
    if number in (88, 94, 180):
        page.get_pixmap(matrix=pymupdf.Matrix(1.1, 1.1)).save(
            OUT / f"thesis_pdf_page_{number}.png")
(OUT / "dissertation_excerpts.txt").write_text(
    f"Source: {thesis_path}\n\n" + "\n\n".join(excerpts), encoding="utf-8")
