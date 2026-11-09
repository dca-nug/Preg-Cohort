"""
=============================================================================
 STEP 7 -- TABLE 1, BASELINE CHARACTERISTICS
=============================================================================

Builds one combined baseline table with a column per analytic level:

    visit level       pregnancy by visit cleaned.csv        step 6
    individual level  pregnancy by individual.csv           step 5
    pregnancy level   pregnancy by episode cleaned.csv      step 4

Rows: maternal age (mean +/- SD), extreme-age category, residence region,
insurance subsidy status, and pregnancy order. Percentages use each dataset's
own N as the denominator.

WHICH PREGNANCY-LEVEL FILE
--------------------------
The default input is the POST-quality-control file from step 4, not the
pre-QC file from step 3. Table 1 describes the analytic cohort, so it must
reflect both the age exclusion and the diagnostic-conflict exclusion. Pointing
this at the step 3 output produces a table whose N does not match the cohort
described anywhere else in the manuscript, and the mismatch is easy to miss
because both files exist side by side.

VALUE LABELS
------------
The 0/1 mappings below are not assumptions; they follow the definitions applied
in step 1:

    dom      = 0 when the province service code is in the Java-Bali set,
               1 otherwise. The label therefore reads Java-Bali, not Java.
    subsid   = 1 when the membership category is subsidised.
    age_risk = 1 when maternal age is below 20 or above 35 years.

The observed value domain of each variable is printed before the table is
built, so an unexpected code is caught here rather than mislabelled.

AUTOMATIC VALIDATION
--------------------
For every categorical block the script checks two things and reports failures:

    counts within the block sum to the column N
    percentages within the block sum to 100

These are the errors that a reader can verify with a calculator and that a
reviewer will check first. A block whose counts fall short of N indicates
either missing values that are not being reported or a transcription error;
either way it must be resolved or disclosed before submission. Where a block
legitimately has a different denominator, that denominator is stated in the
validation report rather than left implicit.

PREGNANCY ORDER
---------------
Pregnancy order is an episode attribute. At the visit level it is populated
only on termination encounters, and at the individual level only for women with
at least one reconstructed episode, so in both cases the counts do not sum to
the column N. Where coverage falls below NPREG_MIN_COVERAGE the row is left
blank rather than reported against a denominator it does not match. The actual
coverage of each dataset is printed, so the decision is visible.

INPUT
-----
    pregnancy by visit cleaned.csv, pregnancy by individual.csv,
    pregnancy by episode cleaned.csv

OUTPUT
------
    table1_baseline.csv             the table
    table1_validation.csv           per-block sum checks
    table1_value_domains.csv        observed values of each categorical variable
    benchmark_step7.csv             per-stage timing and memory
    environment_step7.txt           library versions and machine specification

USAGE
-----
    python 07_table1_baseline.py
=============================================================================
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from pipeline_utils import Bench, environment_report, warn_if_no_psutil

# Relative paths in the configuration block below are resolved against the
# folder holding this script, not the shell's working directory. Without this,
# launching from an editor's Run button resolves them against the editor's
# install directory instead, and every path fails.
os.chdir(Path(__file__).resolve().parent)

# =============================================================================
# CONFIGURATION
# =============================================================================

OUT_DIR = r"./output_step7"

SOURCES = [
    ("Visit-level dataset",      r"./output_step6/pregnancy by visit cleaned.csv"),
    ("Individual-level dataset", r"./output_step5/pregnancy by individual.csv"),
    ("Pregnancy-level dataset",  r"./output_step4/pregnancy by episode cleaned.csv"),
]

OUT_TABLE      = "table1_baseline.csv"
OUT_VALIDATION = "table1_validation.csv"
OUT_DOMAINS    = "table1_value_domains.csv"

AGE_RISK_LABELS = {0: "20-35 years",             1: "<20 or >35 years"}
DOM_LABELS      = {0: "Within Java-Bali islands", 1: "Outside Java-Bali islands"}
SUBSID_LABELS   = {1: "Subsidised",               0: "Non-subsidised"}

NEEDED = ["age", "age_risk", "dom", "subsid", "n_preg"]

# Pregnancy order is reported only when this share of rows carries a value.
NPREG_MIN_COVERAGE = 0.90

# Tolerance for the percentage sum check, in percentage points.
PCT_SUM_TOLERANCE = 0.05

NOT_APPLICABLE = "\u2014"          # em dash

# =============================================================================


def load(label: str, path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        sys.exit(f"[FATAL] {label}: file not found: {os.path.abspath(path)}")
    header = pd.read_csv(path, nrows=0)
    lut = {c.lower(): c for c in header.columns}
    missing = [c for c in NEEDED if c not in lut]
    if missing:
        print(f"\n[FATAL] {label}: missing columns {missing}")
        print("Columns present:")
        for c in header.columns:
            print(f"    {c}")
        sys.exit(1)
    d = pd.read_csv(path, usecols=[lut[c] for c in NEEDED])
    d.columns = d.columns.str.lower()
    for c in NEEDED:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    print(f"    {label:<26} {len(d):>12,} rows   <- {os.path.basename(path)}")
    return d


def report_domains(dfs: dict[str, pd.DataFrame], out_dir: str) -> pd.DataFrame:
    """Print and record the observed value domain of each categorical variable."""
    rows = []
    for label, d in dfs.items():
        print(f"\n    {label}  (N = {len(d):,})")
        for c in ["age_risk", "dom", "subsid"]:
            vc = d[c].value_counts(dropna=False).sort_index()
            shown = {("missing" if pd.isna(k) else int(k)): int(v)
                     for k, v in vc.items()}
            print(f"      {c:<9}: {shown}")
            for k, v in shown.items():
                rows.append({"dataset": label, "variable": c,
                             "value": k, "n": v})
            unexpected = [k for k in shown if k not in (0, 1, "missing")]
            if unexpected:
                print(f"      [WARNING] unexpected values in {c}: {unexpected}")
        n_miss = int(d["age"].isna().sum())
        print(f"      age      : mean={d['age'].mean():.2f} "
              f"sd={d['age'].std():.2f} missing={n_miss:,}")
        rows.append({"dataset": label, "variable": "age",
                     "value": "missing", "n": n_miss})
        cov = float(d["n_preg"].notna().mean())
        print(f"      n_preg   : coverage={cov * 100:.2f}%")
        rows.append({"dataset": label, "variable": "n_preg",
                     "value": "coverage_pct", "n": round(cov * 100, 2)})

    dom = pd.DataFrame(rows)
    dom.to_csv(os.path.join(out_dir, OUT_DOMAINS), index=False)
    return dom


def fmt_count(n: int, denom: int) -> str:
    if denom == 0:
        return "0 (0.00)"
    p = 100 * n / denom
    ptxt = "<0.01" if (n > 0 and round(p, 2) == 0.00) else f"{p:.2f}"
    return f"{n:,} ({ptxt})"


def build(dfs: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    npreg_vals = sorted({int(v) for d in dfs.values()
                         for v in d["n_preg"].dropna().unique()})

    # (label, kind, argument); kind in {header, meansd, cat}
    specs: list[tuple[str, str, object]] = [
        ("Maternal age, mean \u00b1 SD", "meansd", None),
        ("Maternal age risk category, n(%)", "header", "age_risk"),
        (f"    {AGE_RISK_LABELS[0]}", "cat", ("age_risk", 0)),
        (f"    {AGE_RISK_LABELS[1]}", "cat", ("age_risk", 1)),
        ("Residence region, n(%)", "header", "dom"),
        (f"    {DOM_LABELS[1]}", "cat", ("dom", 1)),
        (f"    {DOM_LABELS[0]}", "cat", ("dom", 0)),
        ("Insurance subsidy status, n(%)", "header", "subsid"),
        (f"    {SUBSID_LABELS[1]}", "cat", ("subsid", 1)),
        (f"    {SUBSID_LABELS[0]}", "cat", ("subsid", 0)),
        ("Pregnancy order, n(%)", "header", "n_preg"),
    ]
    specs += [(f"    {k}", "cat", ("n_preg", k)) for k in npreg_vals]

    table = {"Characteristics": [s[0] for s in specs]}
    checks: list[dict] = []

    for label, d in dfs.items():
        N = len(d)
        colname = f"{label} (n={N:,})"
        coverage = float(d["n_preg"].notna().mean())
        npreg_ok = coverage >= NPREG_MIN_COVERAGE
        if not npreg_ok:
            print(f"    {label}: pregnancy order left blank "
                  f"(coverage {coverage * 100:.2f}% < "
                  f"{NPREG_MIN_COVERAGE * 100:.0f}%)")

        col: list[str] = []
        block_totals: dict[str, list[tuple[int, float]]] = {}

        for lbl, kind, arg in specs:
            if kind == "header":
                col.append("")
                block_totals[str(arg)] = []
                continue
            if kind == "meansd":
                col.append(f"{d['age'].mean():.2f} \u00b1 {d['age'].std():.2f}")
                continue
            var, val = arg                      # type: ignore[misc]
            if var == "n_preg" and not npreg_ok:
                col.append(NOT_APPLICABLE)
                continue
            n = int((d[var] == val).sum())
            col.append(fmt_count(n, N))
            block_totals.setdefault(var, []).append((n, 100 * n / N if N else 0))

        table[colname] = col

        for var, entries in block_totals.items():
            if not entries:
                continue
            n_sum = sum(e[0] for e in entries)
            p_sum = sum(e[1] for e in entries)
            n_nonmissing = int(d[var].notna().sum())
            checks.append({
                "dataset": label, "block": var, "column_N": N,
                "counts_sum": n_sum, "counts_match_N": n_sum == N,
                "shortfall": N - n_sum,
                "non_missing_N": n_nonmissing,
                "counts_match_non_missing": n_sum == n_nonmissing,
                "percent_sum": round(p_sum, 3),
                "percent_sums_to_100": abs(p_sum - 100) <= PCT_SUM_TOLERANCE,
            })

    return pd.DataFrame(table), pd.DataFrame(checks)


def report_validation(checks: pd.DataFrame) -> None:
    print("\n    VALIDATION")
    bad = checks[~checks["counts_match_N"] | ~checks["percent_sums_to_100"]]
    if bad.empty:
        print("      every block sums to its column N and to 100 per cent")
        return
    for _, r in bad.iterrows():
        print(f"      {r['dataset']} / {r['block']}")
        print(f"        counts sum to {r['counts_sum']:,} "
              f"vs column N {r['column_N']:,}  "
              f"(shortfall {r['shortfall']:,})")
        print(f"        percentages sum to {r['percent_sum']:.2f}")
        if r["counts_match_non_missing"]:
            print(f"        the shortfall equals the number of missing values "
                  f"({r['column_N'] - r['non_missing_N']:,}); report this "
                  f"denominator explicitly or add a missing category")
        else:
            print("        the shortfall is NOT explained by missing values; "
                  "investigate before submission")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 74)
    print(" STEP 7 -- TABLE 1, BASELINE CHARACTERISTICS")
    print("=" * 74)
    print(environment_report(os.path.join(OUT_DIR, "environment_step7.txt")))
    warn_if_no_psutil()

    bench = Bench("step 7, Table 1")

    with bench.stage("Load the three analytic datasets", "pandas") as b:
        dfs = {label: load(label, path) for label, path in SOURCES}
        b["rows_out"] = sum(len(d) for d in dfs.values())

    with bench.stage("Report value domains", "pandas") as b:
        report_domains(dfs, OUT_DIR)
        b["rows_out"] = len(dfs)

    with bench.stage("Build and validate the table", "pandas") as b:
        table, checks = build(dfs)
        table.to_csv(os.path.join(OUT_DIR, OUT_TABLE), index=False)
        checks.to_csv(os.path.join(OUT_DIR, OUT_VALIDATION), index=False)
        report_validation(checks)
        b["rows_out"] = len(table)

    print("\n" + table.to_string(index=False))

    bench.finalise(os.path.join(OUT_DIR, "benchmark_step7.csv"),
                   rows_out=len(table))
    print(f"\nOutput written to {os.path.abspath(OUT_DIR)}")


if __name__ == "__main__":
    main()
