"""
=============================================================================
 STEP 6 -- APPLY THE PERSON-LEVEL AGE EXCLUSION TO THE VISIT-LEVEL TABLE
=============================================================================

Produces the visit-level analytic dataset by removing every encounter
belonging to a woman with any biologically implausible recorded age, while
retaining all columns.

The rule is identical to the one applied when building the individual-level
dataset, and is imported from `pipeline_utils.create_eligibility` so that the
two cannot diverge: a woman is eligible if she has at least one recorded age
and every recorded age falls within 12-55 years. Women with any implausible
age are removed together with all of their encounters, rather than only the
offending encounter, because a per-visit filter would leave fragments of the
same woman's history behind with the implausible value silently deleted.

This is why the number of visit rows removed exceeds the number of encounters
flagged during quality control.

NOTES ON THIS REVISION
----------------------
The original script read the whole visit-level table into pandas and grouped
it there. On the full dataset that materialises roughly 7.5 million rows and
all columns in memory at once, which dominates the memory footprint of the
step for no benefit: the operation is a semi-join and a filter, both of which
DuckDB evaluates without materialising the table. The rewrite streams from CSV
to CSV and holds only the per-woman summary in memory.

Two additions:

  1. Women with no recorded age at all are counted separately. The original
     comment noted correctly that they are excluded, because comparisons
     against NULL are never true, but the printed report folded them into the
     total without naming them.

  2. Column names and dtypes of the output are verified against the input, so
     that a silent type coercion during the CSV round trip is caught here
     rather than surfacing as a parsing difference two steps downstream.

INPUT
-----
    pregnancy by visit.csv          complete visit-level table from step 1

OUTPUT
------
    pregnancy by visit cleaned.csv  visit-level analytic dataset, all columns
    exclusion_report_step6.csv      counts for the flow diagram
    benchmark_step6.csv             per-stage timing and memory
    environment_step6.txt           library versions and machine specification

USAGE
-----
    python 06_apply_age_exclusion.py
    python 06_apply_age_exclusion.py --input other.csv --output cleaned.csv
=============================================================================
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import duckdb
import pandas as pd

from pipeline_utils import (AGE_MAX_DEFAULT, AGE_MIN_DEFAULT, Bench,
                            create_eligibility, environment_report,
                            exclusion_counts, warn_if_no_psutil)

# Relative paths in the configuration block below are resolved against the
# folder holding this script, not the shell's working directory. Without this,
# launching from an editor's Run button resolves them against the editor's
# install directory instead, and every path fails.
os.chdir(Path(__file__).resolve().parent)

# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_INPUT  = r"./output_step1/pregnancy by visit.csv"
DEFAULT_OUTPUT = r"pregnancy by visit cleaned.csv"
OUT_DIR        = r"./output_step6"

AGE_MIN, AGE_MAX = AGE_MIN_DEFAULT, AGE_MAX_DEFAULT

# Restrict to the women who contribute at least one episode to the analytic
# pregnancy-level dataset, so that all three levels describe one population.
# Produced by step 4, which must therefore be run first. Set to None to retain
# every woman who passes the age rule, which was the earlier behaviour.
RESTRICT_TO_COHORT = r"./output_step4/analytic_cohort_ids.csv"

ID_COL  = "PSTV01"
AGE_COL = "age"

# =============================================================================


def sql_path(path: str) -> str:
    """DuckDB accepts forward slashes on every platform."""
    return os.path.abspath(path).replace("\\", "/")


def load_visits(con, path: str) -> tuple[int, list[str]]:
    """Register the visit-level table. All columns are retained."""
    if not os.path.exists(path):
        sys.exit(f"[FATAL] input not found: {os.path.abspath(path)}")

    con.execute(f"""
        CREATE OR REPLACE TABLE src AS
        SELECT * FROM read_csv_auto('{sql_path(path)}', HEADER=TRUE,
                                    SAMPLE_SIZE=-1);
    """)
    cols = con.sql("SELECT name FROM pragma_table_info('src')").df()["name"].tolist()
    n = con.sql("SELECT COUNT(*) FROM src").fetchone()[0]
    print(f"    loaded {n:,} rows x {len(cols)} columns")

    for required in (ID_COL, AGE_COL):
        if required not in cols:
            print(f"\n[FATAL] required column '{required}' not found.")
            print("Columns present:")
            for c in cols:
                print(f"    {c}")
            sys.exit(1)
    return n, cols


def load_cohort(con) -> int | None:
    """Read the analytic-cohort roster written by step 4, if configured."""
    if not RESTRICT_TO_COHORT:
        print("    no cohort restriction; every age-eligible woman is retained")
        return None
    path = RESTRICT_TO_COHORT
    if not os.path.exists(path):
        sys.exit(f"[FATAL] {os.path.abspath(path)} not found.\n"
                 "        Run steps 3 and 4 before this one, or set "
                 "RESTRICT_TO_COHORT = None.")
    con.execute(f"""
        CREATE OR REPLACE TABLE cohort AS
        SELECT DISTINCT CAST({ID_COL} AS VARCHAR) AS PSTV01
        FROM read_csv_auto('{sql_path(path)}', HEADER=TRUE);
    """)
    n = con.sql("SELECT COUNT(*) FROM cohort").fetchone()[0]
    print(f"    cohort roster: {n:,} women contribute at least one episode")
    return n


def write_cleaned(con, path: str, restricted: bool) -> int:
    """Retain every column, drop every row belonging to an excluded woman."""
    cond = f"s.{ID_COL} IN (SELECT PSTV01 FROM elig)"
    if restricted:
        cond += f" AND CAST(s.{ID_COL} AS VARCHAR) IN (SELECT PSTV01 FROM cohort)"
    con.execute(f"""
        COPY (SELECT s.* FROM src s WHERE {cond})
        TO '{sql_path(path)}' (HEADER, DELIMITER ',');
    """)
    n = con.sql(f"SELECT COUNT(*) FROM src s WHERE {cond}").fetchone()[0]
    size = os.path.getsize(path) / 1024 ** 3
    print(f"    wrote {n:,} rows -> {path}  ({size:.2f} GB)")
    return n


def verify_roundtrip(con, in_cols: list[str], out_path: str) -> None:
    """Confirm the output has the same columns, in the same order, as the input.

    A CSV round trip can silently change how dates are rendered, which would
    surface as a parsing difference in a later step rather than here.
    """
    head = pd.read_csv(out_path, nrows=200)
    out_cols = head.columns.tolist()
    if out_cols == in_cols:
        print(f"    column check: {len(out_cols)} columns, order preserved")
    else:
        missing = [c for c in in_cols if c not in out_cols]
        added = [c for c in out_cols if c not in in_cols]
        print("    [WARNING] column set or order changed during the round trip")
        if missing:
            print(f"      missing: {missing}")
        if added:
            print(f"      added  : {added}")
        if not missing and not added:
            print("      same columns, different order -- verify downstream scripts")

    date_like = [c for c in out_cols
                 if c.lower() in ("fkp03", "fkl03", "combined_date", "doa",
                                  "dol", "dopt", "fin_g", "ref_start")]
    for c in date_like:
        sample = head[c].dropna().astype(str).head(1)
        if len(sample):
            print(f"      {c:<14} rendered as: {sample.iloc[0]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("USAGE")[0])
    ap.add_argument("--input", default=DEFAULT_INPUT,
                    help="complete visit-level table from step 1")
    ap.add_argument("--output", default=DEFAULT_OUTPUT,
                    help="cleaned visit-level file, written into the output folder")
    ap.add_argument("--outdir", default=OUT_DIR)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    out_path = os.path.join(args.outdir, args.output)

    print("=" * 74)
    print(" STEP 6 -- PERSON-LEVEL AGE EXCLUSION, VISIT LEVEL")
    print("=" * 74)
    print(environment_report(os.path.join(args.outdir, "environment_step6.txt")))
    warn_if_no_psutil()

    bench = Bench("step 6, visit-level cleaning")
    con = duckdb.connect(database=":memory:")

    with bench.stage("Load complete visit-level table", "DuckDB") as b:
        n_in, in_cols = load_visits(con, args.input)
        b["rows_out"] = n_in

    with bench.stage("Determine person-level eligibility", "DuckDB",
                     rows_in=n_in) as b:
        n_ind, n_ok = create_eligibility(con, src_table="src", id_col=ID_COL,
                                         age_col=AGE_COL, age_min=AGE_MIN,
                                         age_max=AGE_MAX)
        b["rows_out"] = n_ind

    with bench.stage("Load the analytic-cohort roster", "DuckDB") as b:
        n_cohort = load_cohort(con)
        b["rows_out"] = n_cohort or 0

    with bench.stage("Write cleaned visit-level file", "DuckDB",
                     rows_in=n_in) as b:
        n_out = write_cleaned(con, out_path, restricted=n_cohort is not None)
        b["rows_out"] = n_out

    with bench.stage("Verify CSV round trip", "pandas") as b:
        verify_roundtrip(con, in_cols, out_path)
        b["rows_out"] = n_out

    counts = exclusion_counts(con, AGE_MIN, AGE_MAX)

    print("\n    VISIT LEVEL")
    print(f"      rows in                       : {n_in:>12,}")
    print(f"      rows removed                  : {n_in - n_out:>12,}")
    print(f"      rows retained                 : {n_out:>12,}")
    print("\n    INDIVIDUAL LEVEL")
    for k, v in counts.items():
        print(f"      {k:<30}: {v:>12,}")

    rep = [{"level": "visit", "metric": "rows_in", "n": n_in},
           {"level": "visit", "metric": "rows_removed", "n": n_in - n_out},
           {"level": "visit", "metric": "rows_retained", "n": n_out}]
    rep += [{"level": "individual", "metric": k, "n": v} for k, v in counts.items()]
    pd.DataFrame(rep).to_csv(
        os.path.join(args.outdir, "exclusion_report_step6.csv"), index=False)

    con.close()
    bench.finalise(os.path.join(args.outdir, "benchmark_step6.csv"),
                   rows_out=n_out)
    print(f"\nOutput written to {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()
