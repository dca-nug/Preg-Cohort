"""
=============================================================================
 STEP 5 -- BUILD THE INDIVIDUAL-LEVEL DATASET
=============================================================================

Collapses the visit-level table to one row per woman and, in the same pass,
writes the visit-level analytic file after the person-level age exclusion.

Both outputs are derived from the complete visit-level table produced by
step 1. The individual-level and pregnancy-level datasets are built in
parallel from that same input, not sequentially from one another.

AGE EXCLUSION
-------------
Applied at the PERSON level, not the visit level: if any recorded age for a
woman falls outside 12-55 years, all of her encounters are removed, not only
the offending encounter. A per-visit filter would leave fragments of the same
woman's history in the data with an implausible age silently dropped.

This is why the number of visit rows removed exceeds the number of visits
flagged during quality control.

NOTES ON THIS REVISION
----------------------
The original script reproduced here read `pregnancy by visit.csv` four times,
once per query. It is now read once into a DuckDB table and queried from
there, which is the single largest cost in this step.

Three diagnostics were added:

  1. Individuals with no recorded age at all. These are excluded by the
     `MIN(age) >= 12 AND MAX(age) <= 55` predicate, because comparisons
     against NULL are never true, but the original report did not count them.

  2. Individuals whose residence changes across visits. `dom` is aggregated
     with MAX here ("ever outside Java-Bali"), whereas the pregnancy-level
     script aggregates it with MIN ("ever within Java-Bali"). The two
     conventions point in opposite directions and only differ for women with
     mixed residence, so the count of such women bounds the discrepancy.
     Set DOM_AGGREGATION to harmonise the two levels.

  3. Individuals whose subsidy status changes across visits, for the same
     reason. Both levels currently use MAX, so this is reported for
     completeness rather than because a conflict exists.

The retained `age` for an eligible woman is her MINIMUM recorded age, i.e.
age at earliest observed contact, and `age_risk` is derived from that value
rather than aggregated from per-visit indicators, so the two cannot conflict.

INPUT
-----
    pregnancy by visit.csv          from step 1
    analytic_cohort_ids.csv         from step 4 (cohort restriction)
    pregnancy by episode cleaned.csv
                                    from step 4 (pregnancy order)

OUTPUT
------
    pregnancy by visit - filtered.csv   visit-level analytic file
    pregnancy by individual.csv         individual-level analytic file
    exclusion_report_step5.csv          counts for the flow diagram
    aggregation_conflicts_step5.csv     mixed-residence / mixed-subsidy counts
    benchmark_step5.csv                 per-stage timing and memory
    environment_step5.txt               library versions and machine spec

USAGE
-----
    python 05_build_individual_level.py
=============================================================================
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

import duckdb
import pandas as pd

from pipeline_utils import (Bench, create_eligibility, environment_report,
                            exclusion_counts, warn_if_no_psutil)

# Relative paths in the configuration block below are resolved against the
# folder holding this script, not the shell's working directory. Without this,
# launching from an editor's Run button resolves them against the editor's
# install directory instead, and every path fails.
os.chdir(Path(__file__).resolve().parent)

# =============================================================================
# CONFIGURATION
# =============================================================================

VISIT_CSV = r"./output_step1/pregnancy by visit.csv"
OUT_DIR   = r"./output_step5"

OUT_VISIT_FILTERED = "pregnancy by visit - filtered.csv"
OUT_INDIVIDUAL     = "pregnancy by individual.csv"

# Biologically plausible maternal age, applied at the person level.
AGE_MIN, AGE_MAX = 12, 55

# Extreme maternal age indicator.
AGE_RISK_LOW, AGE_RISK_HIGH = 20, 35

# Aggregation of residence across a woman's visits.
#   "max" -> 1 if ever outside Java-Bali   (original individual-level rule)
#   "min" -> 0 if ever within Java-Bali    (rule used at the pregnancy level)
# These disagree only for women with mixed residence; the count is reported.
DOM_AGGREGATION = "max"

# Aggregation of subsidy status. Both levels currently use "max".
SUBSID_AGGREGATION = "max"

# Restrict to the women who contribute at least one episode to the analytic
# pregnancy-level dataset, so that all three levels describe one population.
# Produced by step 4, which must therefore be run first. Set to None to retain
# every woman who passes the age rule, which was the earlier behaviour.
#
# Without this, the individual-level dataset also contains women who carried no
# coded termination, and women whose every episode was removed by the washout
# or by quality control. Those women are non-cases by construction and enlarge
# the unexposed denominator, so a cross-level comparison would conflate the
# unit of aggregation with the population each level admits.
RESTRICT_TO_COHORT = r"./output_step4/analytic_cohort_ids.csv"

# Pregnancy order at the individual level: the highest order among the woman's
# episodes in the analytic pregnancy-level dataset written by step 4. With
# None, MAX(n_preg) over all of her visits is used instead (earlier behaviour),
# which still counts episodes later removed by the washout or quality control,
# so women at a given order can outnumber the episodes of that order.
N_PREG_FROM_EPISODES = r"./output_step4/pregnancy by episode cleaned.csv"

# =============================================================================


def load_visits(con, path: str) -> int:
    """Read the visit-level table once, with explicit casts."""
    if not os.path.exists(path):
        sys.exit(f"[FATAL] input not found: {os.path.abspath(path)}")

    con.execute(f"""
        CREATE OR REPLACE TABLE src AS
        SELECT
            CAST(PSTV01 AS VARCHAR)   AS PSTV01,
            TRY_CAST(age AS DOUBLE)   AS age,
            TRY_CAST(age_risk AS BIGINT) AS age_risk,
            TRY_CAST(dom AS BIGINT)   AS dom,
            TRY_CAST(subsid AS BIGINT) AS subsid,
            TRY_CAST(n_preg AS BIGINT) AS n_preg
        FROM read_csv_auto('{path}', HEADER=TRUE)
        WHERE PSTV01 IS NOT NULL;
    """)
    n = con.sql("SELECT COUNT(*) FROM src").fetchone()[0]
    print(f"    loaded {n:,} visit rows")
    return n


def load_cohort(con) -> int | None:
    """Read the analytic-cohort roster written by step 4, if configured."""
    if not RESTRICT_TO_COHORT:
        print("    no cohort restriction; every age-eligible woman is retained")
        return None
    if not os.path.exists(RESTRICT_TO_COHORT):
        sys.exit(f"[FATAL] {os.path.abspath(RESTRICT_TO_COHORT)} not found.\n"
                 "        Run steps 3 and 4 before this one, or set "
                 "RESTRICT_TO_COHORT = None.")
    con.execute(f"""
        CREATE OR REPLACE TABLE cohort AS
        SELECT DISTINCT CAST(PSTV01 AS VARCHAR) AS PSTV01
        FROM read_csv_auto('{os.path.abspath(RESTRICT_TO_COHORT).replace(chr(92), "/")}',
                           HEADER=TRUE);
    """)
    n = con.sql("SELECT COUNT(*) FROM cohort").fetchone()[0]
    print(f"    cohort roster: {n:,} women contribute at least one episode")
    return n


def apply_cohort_restriction(con) -> tuple[int, int]:
    """Narrow `elig` to the analytic cohort and report what that removes."""
    before = con.sql("SELECT COUNT(*) FROM elig").fetchone()[0]
    con.execute("""
        CREATE OR REPLACE TABLE elig AS
        SELECT e.PSTV01 FROM elig e
        WHERE CAST(e.PSTV01 AS VARCHAR) IN (SELECT PSTV01 FROM cohort);
    """)
    after = con.sql("SELECT COUNT(*) FROM elig").fetchone()[0]
    print(f"    age-eligible women                    : {before:>12,}")
    print(f"    also contributing an episode          : {after:>12,}")
    print(f"    removed, no surviving episode         : {before - after:>12,}"
          f"   ({(before - after) / before * 100:.2f}%)")
    return before, after


def build_eligibility(con) -> None:
    """Per-woman summary and eligibility, using the shared rule.

    The extra aggregates are attached here rather than in the shared helper so
    that the eligibility predicate itself lives in exactly one place
    (`pipeline_utils.create_eligibility`), shared with step 6.
    """
    create_eligibility(
        con, src_table="src", id_col="PSTV01", age_col="age",
        age_min=AGE_MIN, age_max=AGE_MAX,
        extra_aggs={
            "dom_min": "MIN(dom)", "dom_max": "MAX(dom)",
            "subsid_min": "MIN(subsid)", "subsid_max": "MAX(subsid)",
            "n_preg": "MAX(n_preg)",
        })


def write_filtered_visits(con, path: str) -> int:
    """Visit-level analytic file: visits of eligible women only."""
    con.execute(f"""
        COPY (
            SELECT s.PSTV01, s.age, s.age_risk, s.dom, s.subsid, s.n_preg
            FROM src s
            WHERE s.PSTV01 IN (SELECT PSTV01 FROM elig)
            ORDER BY s.PSTV01
        ) TO '{path}' (HEADER, DELIMITER ',');
    """)
    n = con.sql("""
        SELECT COUNT(*) FROM src WHERE PSTV01 IN (SELECT PSTV01 FROM elig)
    """).fetchone()[0]
    print(f"    wrote {n:,} rows -> {path}")
    return n


def load_episode_order(con) -> None:
    """Highest surviving pregnancy order per woman, from the step 4 dataset."""
    if not N_PREG_FROM_EPISODES:
        print("    pregnancy order: MAX(n_preg) over all visits")
        con.execute("CREATE OR REPLACE TABLE ep_order "
                    "(PSTV01 VARCHAR, n_preg BIGINT)")
        return
    if not os.path.exists(N_PREG_FROM_EPISODES):
        sys.exit(f"[FATAL] {os.path.abspath(N_PREG_FROM_EPISODES)} not found.\n"
                 "        Run step 4 first, or set N_PREG_FROM_EPISODES = None.")
    path = os.path.abspath(N_PREG_FROM_EPISODES).replace(chr(92), "/")
    con.execute(f"""
        CREATE OR REPLACE TABLE ep_order AS
        SELECT CAST(PSTV01 AS VARCHAR)           AS PSTV01,
               MAX(TRY_CAST(n_preg AS BIGINT))   AS n_preg
        FROM read_csv_auto('{path}', HEADER=TRUE)
        WHERE PSTV01 IS NOT NULL
        GROUP BY 1;
    """)
    n, n_lower = con.execute("""
        SELECT COUNT(*),
               COUNT(*) FILTER (WHERE o.n_preg < p.n_preg)
        FROM ep_order o JOIN per_ind p ON p.PSTV01 = o.PSTV01
    """).fetchone()
    print(f"    pregnancy order from analytic episodes: {n:,} women; "
          f"{n_lower:,} have a lower order than MAX(n_preg) over visits")


def write_individual(con, path: str) -> int:
    """Individual-level analytic file: one row per eligible woman.

    `age` is her minimum recorded age; `age_risk` is derived from that value.
    """
    dom_expr = f"dom_{'max' if DOM_AGGREGATION == 'max' else 'min'}"
    sub_expr = f"subsid_{'max' if SUBSID_AGGREGATION == 'max' else 'min'}"

    con.execute(f"""
        COPY (
            SELECT
                p.PSTV01,
                p.amin AS age,
                CASE WHEN p.amin IS NULL THEN NULL
                     WHEN p.amin < {AGE_RISK_LOW} OR p.amin > {AGE_RISK_HIGH} THEN 1
                     ELSE 0 END AS age_risk,
                p.{dom_expr} AS dom,
                p.{sub_expr} AS subsid,
                COALESCE(o.n_preg, p.n_preg) AS n_preg
            FROM per_ind p
            LEFT JOIN ep_order o ON o.PSTV01 = p.PSTV01
            WHERE p.PSTV01 IN (SELECT PSTV01 FROM elig)
            ORDER BY p.PSTV01
        ) TO '{path}' (HEADER, DELIMITER ',');
    """)
    n = con.sql("SELECT COUNT(*) FROM elig").fetchone()[0]
    print(f"    wrote {n:,} rows -> {path}")
    print(f"    dom aggregated with {DOM_AGGREGATION.upper()}, "
          f"subsid with {SUBSID_AGGREGATION.upper()}")
    return n


def exclusion_report(con, out_dir: str) -> pd.DataFrame:
    """Counts for the flow diagram, at both visit and individual level."""
    v_total, v_kept = con.sql("""
        SELECT (SELECT COUNT(*) FROM src),
               (SELECT COUNT(*) FROM src
                WHERE PSTV01 IN (SELECT PSTV01 FROM elig))
    """).fetchone()
    counts = exclusion_counts(con, AGE_MIN, AGE_MAX)

    print("\n    VISIT LEVEL")
    print(f"      total                         : {v_total:>12,}")
    print(f"      removed with excluded women   : {v_total - v_kept:>12,}")
    print(f"      retained                      : {v_kept:>12,}")
    print("\n    INDIVIDUAL LEVEL")
    for k, v in counts.items():
        print(f"      {k:<30}: {v:>12,}")

    rep = [{"level": "visit", "metric": "total", "n": v_total},
           {"level": "visit", "metric": "removed_with_excluded_women",
            "n": v_total - v_kept},
           {"level": "visit", "metric": "retained", "n": v_kept}]
    rep += [{"level": "individual", "metric": k, "n": v} for k, v in counts.items()]
    df = pd.DataFrame(rep)
    df.to_csv(os.path.join(out_dir, "exclusion_report_step5.csv"), index=False)
    return df


def aggregation_conflicts(con, out_dir: str) -> pd.DataFrame:
    """Women whose residence or subsidy status is not constant across visits.

    For these women the choice between MIN and MAX changes the assigned value,
    so this count bounds the discrepancy between the individual-level rule used
    here and the pregnancy-level rule.
    """
    row = con.sql("""
        SELECT
            COUNT(*) AS eligible,
            COUNT(*) FILTER (WHERE dom_min IS DISTINCT FROM dom_max) AS dom_mixed,
            COUNT(*) FILTER (WHERE subsid_min IS DISTINCT FROM subsid_max)
                AS subsid_mixed
        FROM per_ind
        WHERE PSTV01 IN (SELECT PSTV01 FROM elig)
    """).fetchone()
    n, dom_mixed, sub_mixed = row

    print("\n    AGGREGATION CONFLICTS (eligible women)")
    print(f"      residence not constant : {dom_mixed:>12,}"
          f"   ({dom_mixed / n * 100:.2f}% of {n:,})" if n else "")
    print(f"      subsidy not constant   : {sub_mixed:>12,}"
          f"   ({sub_mixed / n * 100:.2f}% of {n:,})" if n else "")
    if dom_mixed:
        print("      -> for these women MIN and MAX assign opposite residence.")
        print("         The pregnancy-level script uses MIN for dom; this script")
        print("         uses MAX. Harmonise via DOM_AGGREGATION if the")
        print("         cross-level residence comparison is to be interpreted.")

    df = pd.DataFrame([
        {"metric": "eligible_individuals", "n": n},
        {"metric": "residence_not_constant", "n": dom_mixed},
        {"metric": "subsidy_not_constant", "n": sub_mixed},
        {"metric": "dom_aggregation_used", "n": DOM_AGGREGATION},
        {"metric": "subsid_aggregation_used", "n": SUBSID_AGGREGATION},
    ])
    df.to_csv(os.path.join(out_dir, "aggregation_conflicts_step5.csv"), index=False)
    return df


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 74)
    print(" STEP 5 -- BUILD THE INDIVIDUAL-LEVEL DATASET")
    print("=" * 74)
    print(environment_report(os.path.join(OUT_DIR, "environment_step5.txt")))
    warn_if_no_psutil()

    bench = Bench("step 5, individual level")
    con = duckdb.connect(database=":memory:")

    with bench.stage("Load visit-level table", "DuckDB") as b:
        n_visits = load_visits(con, VISIT_CSV)
        b["rows_out"] = n_visits

    with bench.stage("Determine person-level eligibility", "DuckDB",
                     rows_in=n_visits) as b:
        build_eligibility(con)
        b["rows_out"] = con.sql("SELECT COUNT(*) FROM per_ind").fetchone()[0]

    with bench.stage("Apply the analytic-cohort restriction", "DuckDB") as b:
        n_cohort = load_cohort(con)
        if n_cohort is not None:
            apply_cohort_restriction(con)
        b["rows_out"] = con.sql("SELECT COUNT(*) FROM elig").fetchone()[0]

    with bench.stage("Write filtered visit-level file", "DuckDB",
                     rows_in=n_visits) as b:
        b["rows_out"] = write_filtered_visits(
            con, os.path.join(OUT_DIR, OUT_VISIT_FILTERED).replace("\\", "/"))

    with bench.stage("Pregnancy order from analytic episodes", "DuckDB") as b:
        load_episode_order(con)
        b["rows_out"] = con.execute("SELECT COUNT(*) FROM ep_order").fetchone()[0]

    with bench.stage("Aggregate to individual level", "DuckDB",
                     rows_in=n_visits) as b:
        n_ind = write_individual(
            con, os.path.join(OUT_DIR, OUT_INDIVIDUAL).replace("\\", "/"))
        b["rows_out"] = n_ind

    with bench.stage("Exclusion and conflict reports", "DuckDB") as b:
        exclusion_report(con, OUT_DIR)
        aggregation_conflicts(con, OUT_DIR)
        b["rows_out"] = n_ind

    con.close()
    bench.finalise(os.path.join(OUT_DIR, "benchmark_step5.csv"), rows_out=n_ind)
    print(f"\nOutput written to {os.path.abspath(OUT_DIR)}")


if __name__ == "__main__":
    main()