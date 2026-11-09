"""
=============================================================================
 STEP 15 -- VISIT-LEVEL COUNTING-PROCESS DATASET FOR SURVIVAL ANALYSIS
=============================================================================

Rebuilds a visit-level table in (tstart, tstop, event] counting-process form,
with administrative censoring at a fixed date, for time-to-event analysis of
all-cause mortality.

WHY THIS STEP EXISTS
--------------------
Neither existing visit-level file is usable for survival analysis:

    output_step1/pregnancy by visit.csv
        Complete and carries every date, but one row per encounter with no
        risk-set structure and no censoring.

    output_step5/pregnancy by visit - filtered.csv
        The visit-level ANALYTIC file, but it retains only
        PSTV01, age, age_risk, dom, subsid, n_preg. No dates, no diagnosis
        columns. Nothing in it can support a time axis.

This step therefore reads step 1's complete output and restricts it to the
analytic cohort roster written by step 4, so that the survival population is
the same population the cross-sectional analyses describe.

DESIGN DECISIONS, AND WHERE THEY ARE ARGUABLE
---------------------------------------------
1. TIME ORIGIN. Default is the first episode anchor (`fin_g`), i.e. the first
   recorded pregnancy termination. Follow-up before a woman's first observed
   pregnancy is not part of the risk set under this definition. Set
   TIME_ORIGIN = "first_visit" to start at first observed contact instead;
   that admits more person-time but mixes women who were never pregnant in the
   observation window into the denominator.

2. CENSORING. Administrative, at ADMIN_CENSOR_DATE, identical for everyone.
   This assumes continuous observability to that date. It is NOT the last
   observed visit: censoring at last contact is informative, because ceasing
   to attend correlates with health status in both directions. If the
   membership file carries a disenrollment date, that variable should replace
   this constant and the assumption should be dropped.

3. DEATH DATE GRANULARITY. The death variable in the membership file is a
   YEAR, not a date. A year cannot support a day-scale Cox model without an
   assumption. Two routes are provided:

     DEATH_DATE_RULE = "midyear"   assume 1 July of the death year, then treat
                                   the result as a date. Convenient, and the
                                   assumption must be stated in the Methods
                                   and varied in a sensitivity analysis
                                   ("yearstart" / "yearend" bracket it).

     DEATH_DATE_RULE = "discrete"  emit no assumed date. The person-period
                                   file (one row per woman per year at risk)
                                   is written instead, for discrete-time
                                   survival by pooled logistic or
                                   complementary log-log. This is the honest
                                   route when only a year is observed, and the
                                   one that survives review most easily.

   Both routes write the raw `death_year`, so the choice is not baked in.

4. TIME-VARYING COVARIATES. The interval file carries only the covariates that
   are cheap and unambiguous (attendance counts, cumulative pregnancy order,
   age, region, subsidy). Diagnoses are NOT pivoted into two hundred columns
   here. They are written as a long event log, `survival_dx_events.csv`, with
   one row per (patient, date, condition), which is the input
   `survival::tmerge` expects. Merging there rather than here means the
   analyst chooses which conditions enter the model and with what lag, and
   the choice is visible in the analysis script rather than buried in a
   build step.

5. AGE. `PSTV03` (date of birth) is dropped in step 1, so age is recovered as
   birth year = visit year - recorded age, taken as the modal value across a
   woman's encounters, then re-evaluated at each interval start. This
   reproduces step 1's own year-difference definition of age and is therefore
   consistent with `age_risk` elsewhere in the pipeline. It is a year-scale
   approximation, as it already was upstream.

DEATH VARIABLE AUDIT
--------------------
`PSTV18` is read into step 1 and joined into the visit table, but is never
used by any downstream step. Its meaning has not been verified anywhere in
this pipeline. Stage 2 below therefore prints its type, range, coverage and
value distribution, writes `death_variable_audit.csv`, and STOPS if the values
do not look like plausible calendar years. Confirm against the BPJS data
dictionary before trusting the output of this script.

INPUT
-----
    output_step1/pregnancy by visit.csv        complete visit-level table
    output_step4/analytic_cohort_ids.csv       cohort roster (optional)
    03_build_pregnancy_level.py                condition definitions

OUTPUT
------
    survival_intervals.csv          counting-process (tstart, tstop, event]
    survival_person_summary.csv     one row per woman: origin, exit, event
    survival_dx_events.csv          long diagnosis event log, for tmerge
    survival_person_period.csv      one row per woman-year (discrete route)
    death_variable_audit.csv        evidence that the death variable is a year
    followup_diagnostics.csv        person-time, events, exclusions
    benchmark_step15.csv            per-stage timing and memory
    environment_step15.txt          library versions and machine spec

USAGE
-----
    python 12_build_survival_visit_level.py

DOWNSTREAM, IN R
----------------
    base <- read.csv("survival_person_summary.csv")
    dx   <- read.csv("survival_dx_events.csv")
    iv   <- read.csv("survival_intervals.csv")

    fit <- coxph(Surv(tstart, tstop, event) ~ ht + dm + age_tstart + dom,
                 data = merged, id = PSTV01)

    # time-varying diagnosis history via tmerge:
    td <- tmerge(base, base, id = PSTV01,
                 death = event(followup_days, event))
    td <- tmerge(td, subset(dx, condition == "ht"), id = PSTV01,
                 ht = tdc(days_from_origin))
=============================================================================
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from pipeline_utils import Bench, environment_report, warn_if_no_psutil

# Relative paths in the configuration block below are resolved against the
# folder holding this script, not the shell's working directory.
os.chdir(Path(__file__).resolve().parent)

# =============================================================================
# CONFIGURATION
# =============================================================================

VISIT_CSV = r"./output_step1/pregnancy by visit.csv"
COHORT_IDS = r"./output_step4/analytic_cohort_ids.csv"   # None to skip
OUT_DIR = r"./output_step15"

STEP3_MODULE = "03_build_pregnancy_level"   # single source of condition sets

# --- identifiers and source columns ------------------------------------------
ID_COL = "PSTV01"
DEATH_COL = "PSTV18"          # believed to be year of death -- AUDITED BELOW
DX_COLS = ["FKP14A", "FKL15A", "FKL17A", "FKL24A"]
FKTP_DATE = "FKP03"           # primary care encounter date
FKRTL_DATE = "FKL03"          # hospital admission date
FKRTL_DATE_END = "FKL04"      # hospital discharge date, if present

# --- time axis ----------------------------------------------------------------
# Fixed administrative censoring. Every woman alive at this date is censored
# here. Must not exceed the coverage of the source extract; the observed
# maximum encounter date is printed and compared against it.
ADMIN_CENSOR_DATE = "2023-12-31"

# Earliest date admitted to the risk set. Encounters before this are used to
# establish diagnosis history but contribute no person-time.
STUDY_START_DATE = "2016-01-01"   # 2015 is the washout year in step 3

# "first_anchor" : origin = first pregnancy termination (fin_g). Default.
# "first_visit"  : origin = first observed encounter of any kind.
TIME_ORIGIN = "first_anchor"

# "midyear" | "yearstart" | "yearend" | "discrete"
# See design decision 3 in the header. "midyear" is the assumption most often
# reported; run the two brackets as a sensitivity analysis.
DEATH_DATE_RULE = "midyear"

# Person-period file for the discrete-time route. Written regardless of
# DEATH_DATE_RULE so the two analyses can be compared, unless set to False.
WRITE_PERSON_PERIOD = True

# Plausible calendar-year range for the audit in stage 2. Values outside this
# range in more than AUDIT_TOLERANCE of non-null records stop the run.
AUDIT_YEAR_MIN, AUDIT_YEAR_MAX = 1990, 2030
AUDIT_TOLERANCE = 0.01

# Minimum interval length in days. Intervals of zero length are invalid in a
# counting-process model and are collapsed into the following interval.
MIN_INTERVAL_DAYS = 1

AGE_RISK_LOW, AGE_RISK_HIGH = 20, 35

# =============================================================================


def sql_path(path: str) -> str:
    return os.path.abspath(path).replace("\\", "/")


def resolve(cols: list[str], want: str) -> str | None:
    """Case-insensitive column lookup; the step 1 output mixes conventions."""
    lut = {c.lower(): c for c in cols}
    return lut.get(want.lower())


# =============================================================================
# STAGE 1 -- LOAD
# =============================================================================


def s1_load(con) -> tuple[int, list[str]]:
    if not os.path.exists(VISIT_CSV):
        sys.exit(f"[FATAL] input not found: {os.path.abspath(VISIT_CSV)}\n"
                 "        Run 01_build_visit_level.py first. The step 5 "
                 "visit file cannot be used: it carries no dates.")

    con.execute(f"""
        CREATE OR REPLACE TABLE raw AS
        SELECT * FROM read_csv_auto('{sql_path(VISIT_CSV)}',
                                    HEADER=TRUE, SAMPLE_SIZE=-1);
    """)
    cols = con.sql("SELECT name FROM pragma_table_info('raw')").df()["name"].tolist()
    n = con.sql("SELECT COUNT(*) FROM raw").fetchone()[0]
    print(f"    loaded {n:,} visit rows x {len(cols)} columns")

    required = [ID_COL] + DX_COLS
    missing = [c for c in required if resolve(cols, c) is None]
    if missing:
        print(f"\n[FATAL] required columns not found: {missing}")
        print("Columns present:")
        for c in cols:
            print(f"    {c}")
        sys.exit(1)

    if resolve(cols, DEATH_COL) is None:
        print(f"\n[FATAL] the death variable '{DEATH_COL}' is absent from "
              f"{VISIT_CSV}.")
        print("        Step 1 selects PSTV18 from Kepesertaan and does not "
              "drop it, so its absence means step 1 was run with a different")
        print("        column specification. Re-run step 1, or point "
              "DEATH_COL at the correct variable.")
        print("Columns present:")
        for c in cols:
            print(f"    {c}")
        sys.exit(1)

    for c in (FKTP_DATE, FKRTL_DATE):
        if resolve(cols, c) is None:
            sys.exit(f"[FATAL] encounter date column '{c}' not found.")

    return n, cols


# =============================================================================
# STAGE 2 -- AUDIT THE DEATH VARIABLE
# =============================================================================


def s2_audit_death(con, cols: list[str], out_dir: str) -> pd.DataFrame:
    """Establish that DEATH_COL holds calendar years before anything is built
    on top of it. This stage exists because PSTV18 enters step 1 and is never
    used again anywhere in the pipeline, so its meaning is undocumented here.
    """
    dcol = resolve(cols, DEATH_COL)

    dtype = con.sql(f"""
        SELECT data_type FROM pragma_table_info('raw') WHERE name = '{dcol}'
    """).fetchone()[0]

    summary = con.sql(f'''
        SELECT
            COUNT(*)                                   AS n_rows,
            COUNT("{dcol}")                            AS n_nonnull,
            COUNT(DISTINCT "{ID_COL}")                 AS n_women,
            COUNT(DISTINCT CASE WHEN "{dcol}" IS NOT NULL
                  THEN "{ID_COL}" END)                 AS n_women_with_value
        FROM raw
    ''').df()

    print(f"    column            : {dcol}")
    print(f"    declared type     : {dtype}")
    print(f"    rows with a value : {int(summary.n_nonnull[0]):,} of "
          f"{int(summary.n_rows[0]):,}")
    print(f"    women with a value: {int(summary.n_women_with_value[0]):,} of "
          f"{int(summary.n_women[0]):,}"
          f"  ({summary.n_women_with_value[0] / summary.n_women[0] * 100:.2f}%)")

    # Value distribution, per woman rather than per visit, since a membership
    # attribute repeats across every encounter of the same woman.
    vals = con.sql(f'''
        SELECT v AS value, COUNT(*) AS n_women FROM (
            SELECT "{ID_COL}" AS pid, MAX(TRY_CAST("{dcol}" AS DOUBLE)) AS v
            FROM raw WHERE "{dcol}" IS NOT NULL GROUP BY 1
        ) GROUP BY 1 ORDER BY 1
    ''').df()

    if vals.empty:
        print(f"\n[FATAL] '{dcol}' is present but entirely null. It cannot be "
              "the death variable in this extract.")
        sys.exit(1)

    numeric = vals["value"].notna().all()
    if not numeric:
        print(f"\n[FATAL] '{dcol}' did not cast to a number. If it is a full "
              "date rather than a year, that is GOOD NEWS for the analysis:")
        print("        set DEATH_COL to it, remove the year assumption, and "
              "use a day-scale Cox model. Adjust this stage accordingly.")
        sys.exit(1)

    lo, hi = float(vals["value"].min()), float(vals["value"].max())
    n_women_valued = int(vals["n_women"].sum())
    out_of_range = int(vals.loc[(vals["value"] < AUDIT_YEAR_MIN)
                                | (vals["value"] > AUDIT_YEAR_MAX),
                                "n_women"].sum())
    frac_oob = out_of_range / n_women_valued

    print(f"    observed range    : {lo:g} to {hi:g}")
    print(f"    distinct values   : {len(vals):,}")
    print("    value distribution (women):")
    for _, r in vals.head(20).iterrows():
        print(f"      {r['value']:>10g} : {int(r['n_women']):>10,}")
    if len(vals) > 20:
        print(f"      ... {len(vals) - 20} further values in "
              "death_variable_audit.csv")

    audit = pd.DataFrame([
        {"item": "column", "value": dcol},
        {"item": "declared_type", "value": dtype},
        {"item": "n_women_total", "value": int(summary.n_women[0])},
        {"item": "n_women_with_value", "value": int(summary.n_women_with_value[0])},
        {"item": "min_value", "value": lo},
        {"item": "max_value", "value": hi},
        {"item": "n_distinct_values", "value": len(vals)},
        {"item": "n_women_outside_year_range", "value": out_of_range},
    ])
    audit.to_csv(os.path.join(out_dir, "death_variable_audit.csv"), index=False)
    vals.to_csv(os.path.join(out_dir, "death_variable_values.csv"), index=False)

    if frac_oob > AUDIT_TOLERANCE:
        print(f"\n[FATAL] {frac_oob * 100:.2f}% of women carry a value outside "
              f"{AUDIT_YEAR_MIN}-{AUDIT_YEAR_MAX}.")
        print(f"        '{dcol}' does not behave like a year of death. It may "
              "be a membership class, a region code, or a status flag.")
        print("        Check the BPJS data dictionary and set DEATH_COL "
              "correctly. Nothing further is built on an unverified variable.")
        print(f"        The observed distribution is in "
              f"{os.path.join(out_dir, 'death_variable_values.csv')}")
        sys.exit(1)

    if len(vals) < 3:
        print(f"\n[WARNING] '{dcol}' takes only {len(vals)} distinct values. "
              "A year of death in a multi-year extract should take more.")
        print("          Verify before proceeding; this is not stopped "
              "automatically, but it is a strong signal the variable is a "
              "flag rather than a year.")

    print(f"\n    '{dcol}' is consistent with a calendar year. Proceeding. "
          "This is a plausibility check, NOT confirmation of its meaning: "
          "verify it against the data dictionary.")
    return audit


# =============================================================================
# STAGE 3 -- ENCOUNTER TABLE
# =============================================================================


def s3_encounters(con, cols: list[str]) -> int:
    """One row per (woman, encounter date), with the source of the encounter.

    `combined_date` from step 1 is deliberately NOT used as the time axis.
    It is min(FKP03, FKL03) taken after a full outer join, so on a row that
    carries both a primary-care and a hospital date it assigns the earlier of
    the two to both. For a survival time axis the encounter dates are kept
    separate and a hospital visit is dated by FKL03.
    """
    id_c = resolve(cols, ID_COL)
    p_c = resolve(cols, FKTP_DATE)
    h_c = resolve(cols, FKRTL_DATE)
    hd_c = resolve(cols, FKRTL_DATE_END)
    d_c = resolve(cols, DEATH_COL)
    age_c = resolve(cols, "age")
    dom_c = resolve(cols, "dom")
    sub_c = resolve(cols, "subsid")
    fin_c = resolve(cols, "fin_g")

    if fin_c is None and TIME_ORIGIN == "first_anchor":
        sys.exit("[FATAL] TIME_ORIGIN = 'first_anchor' requires `fin_g`, "
                 "which is absent. Re-run step 1 or set TIME_ORIGIN = "
                 "'first_visit'.")

    los = (f'DATE_DIFF(\'day\', TRY_CAST("{h_c}" AS DATE), '
           f'TRY_CAST("{hd_c}" AS DATE))' if hd_c else "NULL")

    con.execute(f'''
        CREATE OR REPLACE TABLE enc AS
        SELECT
            CAST("{id_c}" AS VARCHAR)                      AS pid,
            TRY_CAST("{h_c}" AS DATE)                      AS hosp_date,
            TRY_CAST("{p_c}" AS DATE)                      AS prim_date,
            COALESCE(TRY_CAST("{h_c}" AS DATE),
                     TRY_CAST("{p_c}" AS DATE))            AS enc_date,
            CASE WHEN "{h_c}" IS NOT NULL THEN 1 ELSE 0 END AS is_hospital,
            {los}                                          AS los_days,
            TRY_CAST("{d_c}" AS INTEGER)                   AS death_year,
            TRY_CAST("{age_c}" AS DOUBLE)                  AS age,
            TRY_CAST("{dom_c}" AS INTEGER)                 AS dom,
            TRY_CAST("{sub_c}" AS INTEGER)                 AS subsid,
            {f'TRY_CAST("{fin_c}" AS DATE)' if fin_c else "NULL"} AS fin_g
        FROM raw
        WHERE "{id_c}" IS NOT NULL;
    ''')

    con.execute("DELETE FROM enc WHERE enc_date IS NULL;")
    n = con.sql("SELECT COUNT(*) FROM enc").fetchone()[0]
    obs_max, obs_min = con.sql(
        "SELECT MAX(enc_date), MIN(enc_date) FROM enc").fetchone()
    print(f"    {n:,} dated encounters, {obs_min} to {obs_max}")

    censor = pd.Timestamp(ADMIN_CENSOR_DATE).date()
    if obs_max and obs_max > censor:
        print(f"    [WARNING] encounters exist after ADMIN_CENSOR_DATE "
              f"({obs_max} > {censor}). Person-time after the censoring date "
              "is discarded; confirm the censoring date is correct.")
    if obs_max and (censor - obs_max).days > 180:
        print(f"    [WARNING] ADMIN_CENSOR_DATE is {(censor - obs_max).days} "
              f"days after the last observed encounter ({obs_max}). If the "
              "extract ends earlier than the censoring date, the tail of "
              "follow-up is person-time in which no event could have been "
              "observed, which biases the rate downward.")
    return n


# =============================================================================
# STAGE 4 -- PER-WOMAN TIME AXIS
# =============================================================================


def s4_person_axis(con) -> pd.DataFrame:
    """Origin, exit, event indicator and baseline covariates, one row per woman.

    Birth year is recovered as (encounter year - recorded age) and taken as
    the modal value, reproducing step 1's year-difference definition of age.
    """
    origin_expr = ("MIN(fin_g)" if TIME_ORIGIN == "first_anchor"
                   else "MIN(enc_date)")

    con.execute(f'''
        CREATE OR REPLACE TABLE birth AS
        SELECT pid, MODE(EXTRACT(YEAR FROM enc_date) - CAST(age AS INTEGER))
               AS birth_year
        FROM enc WHERE age IS NOT NULL GROUP BY pid;
    ''')

    con.execute(f'''
        CREATE OR REPLACE TABLE person AS
        SELECT
            e.pid,
            {origin_expr}                     AS origin_raw,
            MIN(e.enc_date)                   AS first_enc,
            MAX(e.enc_date)                   AS last_enc,
            MAX(e.death_year)                 AS death_year,
            MIN(e.dom)                        AS dom,
            MAX(e.subsid)                     AS subsid,
            b.birth_year
        FROM enc e LEFT JOIN birth b USING (pid)
        GROUP BY e.pid, b.birth_year;
    ''')

    # A woman may carry conflicting death years across her membership rows.
    conflict = con.sql('''
        SELECT COUNT(*) FROM (
            SELECT pid FROM enc WHERE death_year IS NOT NULL
            GROUP BY pid HAVING COUNT(DISTINCT death_year) > 1)
    ''').fetchone()[0]
    if conflict:
        print(f"    [WARNING] {conflict:,} women carry more than one distinct "
              "death year across their records. The maximum is used. "
              "Investigate before reporting.")

    start = pd.Timestamp(STUDY_START_DATE).date()
    censor = pd.Timestamp(ADMIN_CENSOR_DATE).date()

    rule = {"midyear": "07-01", "yearstart": "01-01", "yearend": "12-31"}
    if DEATH_DATE_RULE not in set(rule) | {"discrete"}:
        sys.exit("[FATAL] DEATH_DATE_RULE must be 'midyear', 'yearstart', "
                 "'yearend', or 'discrete'.")

    # A date is assigned even under the "discrete" rule, because the interval
    # file needs an exit date to exist at all. What "discrete" changes is which
    # file you then analyse: the person-period file carries only the year and
    # makes no within-year assumption, so under this rule the interval file is
    # a by-product and the person-period file is the analysis dataset.
    mmdd = rule.get(DEATH_DATE_RULE, "07-01")
    if DEATH_DATE_RULE == "discrete":
        print("    [NOTE] DEATH_DATE_RULE = 'discrete'. survival_intervals.csv "
              f"still dates deaths at {mmdd} of the death year, because an "
              "interval must end somewhere.")
        print("           Analyse survival_person_period.csv instead; it "
              "carries the year only.")
    death_expr = (f"CASE WHEN death_year IS NOT NULL THEN "
                  f"TRY_CAST(CAST(death_year AS VARCHAR) || '-{mmdd}' "
                  f"AS DATE) END")

    con.execute(f"""
        CREATE OR REPLACE TABLE person2 AS
        SELECT *,
            GREATEST(origin_raw, DATE '{start}')   AS origin,
            {death_expr}                           AS death_date,
            DATE '{censor}'                        AS censor_date
        FROM person;
    """)

    # Exit = death if it precedes administrative censoring, else censoring.
    con.execute("""
        CREATE OR REPLACE TABLE person3 AS
        SELECT *,
            CASE WHEN death_date IS NOT NULL AND death_date <= censor_date
                 THEN 1 ELSE 0 END                        AS event,
            CASE WHEN death_date IS NOT NULL AND death_date <= censor_date
                 THEN death_date ELSE censor_date END     AS exit_date
        FROM person2;
    """)

    n_all = con.sql("SELECT COUNT(*) FROM person3").fetchone()[0]
    n_noorigin = con.sql(
        "SELECT COUNT(*) FROM person3 WHERE origin IS NULL").fetchone()[0]
    n_dead_before = con.sql("""
        SELECT COUNT(*) FROM person3
        WHERE origin IS NOT NULL AND exit_date <= origin
    """).fetchone()[0]

    con.execute("""
        CREATE OR REPLACE TABLE person_final AS
        SELECT * FROM person3
        WHERE origin IS NOT NULL AND exit_date > origin;
    """)
    n_kept = con.sql("SELECT COUNT(*) FROM person_final").fetchone()[0]

    print(f"    women in the encounter table          : {n_all:>12,}")
    print(f"    no usable time origin                 : {n_noorigin:>12,}"
          + ("   (no episode anchor)" if TIME_ORIGIN == "first_anchor" else ""))
    print(f"    exit at or before origin, removed     : {n_dead_before:>12,}")
    print(f"    women contributing person-time        : {n_kept:>12,}")

    if n_kept == 0:
        sys.exit("[FATAL] no woman contributes person-time. Check "
                 "STUDY_START_DATE, ADMIN_CENSOR_DATE and TIME_ORIGIN.")

    excl = pd.DataFrame([
        {"metric": "women_in_encounter_table", "n": n_all},
        {"metric": "removed_no_time_origin", "n": n_noorigin},
        {"metric": "removed_exit_before_origin", "n": n_dead_before},
        {"metric": "women_analysed", "n": n_kept},
        {"metric": "time_origin_rule", "n": TIME_ORIGIN},
        {"metric": "death_date_rule", "n": DEATH_DATE_RULE},
        {"metric": "admin_censor_date", "n": ADMIN_CENSOR_DATE},
    ])
    return excl


# =============================================================================
# STAGE 5 -- COUNTING-PROCESS INTERVALS
# =============================================================================


def s5_intervals(con) -> int:
    """Split each woman's follow-up at every encounter date.

    Change points are the encounter dates strictly inside (origin, exit).
    Covariates are evaluated as of the start of each interval, so an encounter
    updates the covariates for the interval that BEGINS at it, never for the
    interval that ends there. This keeps the model free of the anticipation
    that arises when a covariate is assigned to time before it was observed.
    """
    con.execute("""
        CREATE OR REPLACE TABLE cutpoints AS
        SELECT DISTINCT p.pid, e.enc_date AS cut
        FROM person_final p
        JOIN enc e ON e.pid = p.pid
        WHERE e.enc_date > p.origin AND e.enc_date < p.exit_date
        UNION
        SELECT pid, origin AS cut FROM person_final;
    """)

    # Attendance at each cut date, for the covariates that update there.
    con.execute("""
        CREATE OR REPLACE TABLE cut_enc AS
        SELECT pid, enc_date AS cut,
               MAX(is_hospital)            AS hosp_visit_at_start,
               COUNT(*)                    AS n_records_at_start,
               MAX(los_days)               AS los_days_at_start
        FROM enc GROUP BY pid, enc_date;
    """)

    con.execute(f"""
        CREATE OR REPLACE TABLE iv AS
        SELECT
            c.pid,
            c.cut                                                 AS start_date,
            COALESCE(LEAD(c.cut) OVER (PARTITION BY c.pid ORDER BY c.cut),
                     p.exit_date)                                 AS stop_date,
            p.origin, p.exit_date, p.event, p.death_year,
            p.birth_year, p.dom, p.subsid,
            COALESCE(ce.hosp_visit_at_start, 0)                   AS hosp_visit,
            COALESCE(ce.n_records_at_start, 0)                    AS n_records,
            ce.los_days_at_start                                  AS los_days
        FROM cutpoints c
        JOIN person_final p USING (pid)
        LEFT JOIN cut_enc ce ON ce.pid = c.pid AND ce.cut = c.cut;
    """)

    # Cumulative attendance and pregnancy order as of each interval start.
    con.execute("""
        CREATE OR REPLACE TABLE iv2 AS
        SELECT *,
            DATE_DIFF('day', origin, start_date)                  AS tstart,
            DATE_DIFF('day', origin, stop_date)                   AS tstop,
            SUM(hosp_visit) OVER (PARTITION BY pid ORDER BY start_date
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cum_hosp_visits,
            ROW_NUMBER() OVER (PARTITION BY pid ORDER BY start_date)
                                                                  AS visit_seq,
            EXTRACT(YEAR FROM start_date) - birth_year            AS age_tstart
        FROM iv;
    """)

    # Pregnancy terminations accumulated before each interval start.
    con.execute("""
        CREATE OR REPLACE TABLE anchors AS
        SELECT DISTINCT pid, fin_g FROM enc WHERE fin_g IS NOT NULL;
    """)
    # Expressed as a join and an aggregate rather than a correlated subquery
    # per interval row. A woman carries at most a handful of anchors, so the
    # intermediate expands by a small factor; a correlated scalar subquery
    # would be evaluated once per interval and is not viable at this scale.
    con.execute("""
        CREATE OR REPLACE TABLE anchor_state AS
        SELECT i.pid, i.start_date,
               COUNT(*)       AS n_preg_todate,
               MAX(a.fin_g)   AS last_termination
        FROM iv2 i JOIN anchors a
          ON a.pid = i.pid AND a.fin_g <= i.start_date
        GROUP BY i.pid, i.start_date;
    """)
    con.execute("""
        CREATE OR REPLACE TABLE iv3 AS
        SELECT i.*,
               COALESCE(s.n_preg_todate, 0) AS n_preg_todate,
               s.last_termination
        FROM iv2 i
        LEFT JOIN anchor_state s
          ON s.pid = i.pid AND s.start_date = i.start_date;
    """)

    con.execute(f"""
        CREATE OR REPLACE TABLE intervals AS
        SELECT
            pid                                  AS {ID_COL},
            tstart, tstop,
            CASE WHEN stop_date = exit_date THEN event ELSE 0 END AS event,
            start_date, stop_date, origin, exit_date,
            death_year,
            age_tstart,
            CASE WHEN age_tstart IS NULL THEN NULL
                 WHEN age_tstart < {AGE_RISK_LOW}
                   OR age_tstart > {AGE_RISK_HIGH} THEN 1 ELSE 0 END
                                                 AS age_risk_tstart,
            dom, subsid,
            hosp_visit, n_records, los_days,
            cum_hosp_visits, visit_seq,
            n_preg_todate,
            DATE_DIFF('day', last_termination, start_date)
                                                 AS days_since_termination
        FROM iv3
        WHERE tstop - tstart >= {MIN_INTERVAL_DAYS}
        ORDER BY pid, tstart;
    """)

    n = con.sql("SELECT COUNT(*) FROM intervals").fetchone()[0]
    n_pid = con.sql(f'SELECT COUNT(DISTINCT "{ID_COL}") FROM intervals'
                    ).fetchone()[0]
    n_ev = con.sql("SELECT SUM(event) FROM intervals").fetchone()[0] or 0
    ptime = con.sql("SELECT SUM(tstop - tstart) FROM intervals").fetchone()[0] or 0

    print(f"    intervals                             : {n:>12,}")
    print(f"    women                                 : {n_pid:>12,}")
    print(f"    events (deaths)                       : {int(n_ev):>12,}")
    print(f"    person-years                          : {ptime / 365.25:>12,.0f}")
    if ptime:
        print(f"    crude mortality per 1,000 person-years: "
              f"{n_ev / (ptime / 365.25) * 1000:>12.2f}")

    # A woman whose every interval was dropped by MIN_INTERVAL_DAYS would lose
    # her event entirely, so the two counts are compared explicitly.
    n_ev_person = con.sql("SELECT SUM(event) FROM person_final").fetchone()[0] or 0
    if int(n_ev) != int(n_ev_person):
        print(f"    [WARNING] {int(n_ev_person) - int(n_ev):,} events present at "
              "the person level are absent from the interval file. This means "
              "an event fell in an interval shorter than "
              f"{MIN_INTERVAL_DAYS} day(s) and was discarded. Lower "
              "MIN_INTERVAL_DAYS or inspect those women.")
    return n


# =============================================================================
# STAGE 6 -- DIAGNOSIS EVENT LOG
# =============================================================================


def load_condition_map() -> tuple[pd.DataFrame, dict]:
    """Condition definitions imported from step 3, never restated here."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        m = importlib.import_module(STEP3_MODULE)
    except Exception as exc:
        sys.exit(f"[FATAL] could not import {STEP3_MODULE}.py: {exc}\n"
                 "        The condition definitions must come from step 3 so "
                 "that this file cannot describe a different feature set.")

    rows = []
    for group, d in (("chronic", m.chronic_conditions),
                     ("infectious", m.infectious_conditions),
                     ("pregnancy", m.pregnancy_conditions)):
        for cond, codes in d.items():
            for code in codes:
                rows.append({"code": code, "condition": cond, "group": group})
    exact = pd.DataFrame(rows)
    print(f"    {exact['condition'].nunique()} conditions from exact code "
          f"lists, {len(m.conditions_regex)} from patterns")
    return exact, dict(m.conditions_regex)


def s6_dx_events(con, cols: list[str], out_dir: str) -> int:
    exact, regexes = load_condition_map()
    con.register("cond_exact", exact)

    id_c = resolve(cols, ID_COL)
    p_c = resolve(cols, FKTP_DATE)
    h_c = resolve(cols, FKRTL_DATE)
    dxr = [resolve(cols, c) for c in DX_COLS]

    union = "\n            UNION ALL\n            ".join(
        f'SELECT CAST("{id_c}" AS VARCHAR) AS pid, '
        f'COALESCE(TRY_CAST("{h_c}" AS DATE), TRY_CAST("{p_c}" AS DATE)) AS dt, '
        f'TRIM(CAST("{c}" AS VARCHAR)) AS code FROM raw WHERE "{c}" IS NOT NULL'
        for c in dxr)

    con.execute(f"""
        CREATE OR REPLACE TABLE dx_long AS
        SELECT DISTINCT pid, dt, code FROM (
            {union}
        ) WHERE dt IS NOT NULL AND code <> '';
    """)
    n_dx = con.sql("SELECT COUNT(*) FROM dx_long").fetchone()[0]
    print(f"    distinct (woman, date, code) records  : {n_dx:>12,}")

    # Exact match first; step 1 and step 3 both match three-character codes
    # with `isin`, so the join reproduces their behaviour exactly.
    con.execute("""
        CREATE OR REPLACE TABLE dx_matched AS
        SELECT d.pid, d.dt, c.condition, c."group"
        FROM dx_long d JOIN cond_exact c ON d.code = c.code;
    """)

    # Pattern-matched conditions, appended with the same patterns as step 3.
    for cond, pat in regexes.items():
        con.execute(f"""
            INSERT INTO dx_matched
            SELECT pid, dt, '{cond}' AS condition, 'regex' AS "group"
            FROM dx_long WHERE regexp_matches(code, '{pat}');
        """)

    n_unmatched = con.sql("""
        SELECT COUNT(DISTINCT code) FROM dx_long d
        WHERE NOT EXISTS (SELECT 1 FROM cond_exact c WHERE c.code = d.code)
    """).fetchone()[0]
    print(f"    codes matching no condition (ignored) : {n_unmatched:>12,}")

    # Attach the time axis. Diagnoses BEFORE the origin are retained with a
    # negative offset: they are baseline history, not follow-up events, and
    # discarding them here would make a baseline covariate impossible to build.
    con.execute(f"""
        CREATE OR REPLACE TABLE dx_events AS
        SELECT
            m.pid                                    AS {ID_COL},
            m.dt                                     AS dx_date,
            m.condition,
            m."group",
            DATE_DIFF('day', p.origin, m.dt)         AS days_from_origin,
            CASE WHEN m.dt < p.origin THEN 'baseline'
                 WHEN m.dt <= p.exit_date THEN 'followup'
                 ELSE 'after_exit' END               AS period
        FROM dx_matched m
        JOIN person_final p ON p.pid = m.pid
        WHERE m.dt <= p.exit_date
        ORDER BY m.pid, m.dt;
    """)
    n = con.sql("SELECT COUNT(*) FROM dx_events").fetchone()[0]
    n_base = con.sql("SELECT COUNT(*) FROM dx_events WHERE period='baseline'"
                     ).fetchone()[0]
    print(f"    diagnosis events on the time axis     : {n:>12,}")
    print(f"      of which baseline (before origin)   : {n_base:>12,}")

    con.execute(f"""
        COPY (SELECT * FROM dx_events)
        TO '{sql_path(os.path.join(out_dir, "survival_dx_events.csv"))}'
        (HEADER, DELIMITER ',');
    """)
    return n


# =============================================================================
# STAGE 7 -- PERSON-PERIOD FILE (DISCRETE-TIME ROUTE)
# =============================================================================


def s7_person_period(con, out_dir: str) -> int:
    """One row per woman per calendar year at risk.

    This is the file to use when only a year of death is observed. The event
    indicator is 1 in the year matching `death_year` and 0 in every earlier
    year, with no assumption about where in the year the death occurred. Fit
    with pooled logistic regression or a complementary log-log link.
    """
    censor_year = pd.Timestamp(ADMIN_CENSOR_DATE).year

    con.execute(f"""
        CREATE OR REPLACE TABLE person_period AS
        WITH bounds AS (
            SELECT pid, EXTRACT(YEAR FROM origin) AS y0,
                   LEAST(COALESCE(death_year, {censor_year}), {censor_year})
                       AS y1,
                   death_year, birth_year, dom, subsid, origin
            FROM person_final
        ),
        yrs AS (
            -- UNNEST over the list-returning form of generate_series. The
            -- table-function form would need a lateral join against `bounds`,
            -- which is not portable across DuckDB versions.
            SELECT b.pid, b.death_year, b.birth_year, b.dom, b.subsid, b.origin,
                   CAST(UNNEST(generate_series(CAST(b.y0 AS INTEGER),
                                               CAST(b.y1 AS INTEGER)))
                        AS INTEGER) AS year
            FROM bounds b
            WHERE b.y1 >= b.y0
        )
        SELECT
            pid                                       AS {ID_COL},
            year,
            year - CAST(EXTRACT(YEAR FROM origin) AS INTEGER) AS period,
            CASE WHEN death_year = year THEN 1 ELSE 0 END      AS event,
            year - birth_year                         AS age_year,
            dom, subsid
        FROM yrs
        ORDER BY pid, year;
    """)
    n = con.sql("SELECT COUNT(*) FROM person_period").fetchone()[0]
    n_ev = con.sql("SELECT SUM(event) FROM person_period").fetchone()[0] or 0
    print(f"    woman-years                           : {n:>12,}")
    print(f"    events                                : {int(n_ev):>12,}")

    con.execute(f"""
        COPY (SELECT * FROM person_period)
        TO '{sql_path(os.path.join(out_dir, "survival_person_period.csv"))}'
        (HEADER, DELIMITER ',');
    """)
    return n


# =============================================================================
# MAIN
# =============================================================================


def restrict_cohort(con) -> None:
    if not COHORT_IDS:
        print("    no cohort restriction applied; every woman in the visit "
              "table is retained. The survival population will NOT match the "
              "population of the other analyses.")
        return
    if not os.path.exists(COHORT_IDS):
        sys.exit(f"[FATAL] {os.path.abspath(COHORT_IDS)} not found.\n"
                 "        Run step 4 first, or set COHORT_IDS = None and "
                 "state in the Methods that the survival population differs.")
    con.execute(f"""
        CREATE OR REPLACE TABLE cohort AS
        SELECT DISTINCT CAST("{ID_COL}" AS VARCHAR) AS pid
        FROM read_csv_auto('{sql_path(COHORT_IDS)}', HEADER=TRUE);
    """)
    n_cohort = con.sql("SELECT COUNT(*) FROM cohort").fetchone()[0]
    before = con.sql("SELECT COUNT(*) FROM raw").fetchone()[0]
    # Written to a new table and renamed rather than replaced in place:
    # CREATE OR REPLACE on a table that the same statement reads is refused.
    con.execute(f"""
        CREATE OR REPLACE TABLE raw_restricted AS SELECT * FROM raw
        WHERE CAST("{ID_COL}" AS VARCHAR) IN (SELECT pid FROM cohort);
    """)
    con.execute("DROP TABLE raw;")
    con.execute("ALTER TABLE raw_restricted RENAME TO raw;")
    after = con.sql("SELECT COUNT(*) FROM raw").fetchone()[0]
    print(f"    cohort roster                         : {n_cohort:>12,} women")
    print(f"    visit rows before restriction         : {before:>12,}")
    print(f"    visit rows retained                   : {after:>12,}")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 74)
    print(" STEP 15 -- VISIT-LEVEL COUNTING-PROCESS SURVIVAL DATASET")
    print("=" * 74)
    print(environment_report(os.path.join(OUT_DIR, "environment_step15.txt")))
    warn_if_no_psutil()
    print(f"\n  time origin        : {TIME_ORIGIN}")
    print(f"  study start        : {STUDY_START_DATE}")
    print(f"  administrative censoring : {ADMIN_CENSOR_DATE}")
    print(f"  death date rule    : {DEATH_DATE_RULE}")

    bench = Bench("step 15, survival dataset")
    con = duckdb.connect(database=":memory:")

    with bench.stage("Load complete visit-level table", "DuckDB") as b:
        n_rows, cols = s1_load(con)
        b["rows_out"] = n_rows

    with bench.stage("Audit the death variable", "DuckDB") as b:
        audit = s2_audit_death(con, cols, OUT_DIR)
        b["rows_out"] = len(audit)

    with bench.stage("Restrict to the analytic cohort", "DuckDB",
                     rows_in=n_rows) as b:
        restrict_cohort(con)
        b["rows_out"] = con.sql("SELECT COUNT(*) FROM raw").fetchone()[0]

    with bench.stage("Build the encounter table", "DuckDB") as b:
        b["rows_out"] = s3_encounters(con, cols)

    with bench.stage("Derive the per-woman time axis", "DuckDB") as b:
        excl = s4_person_axis(con)
        b["rows_out"] = con.sql("SELECT COUNT(*) FROM person_final").fetchone()[0]

    with bench.stage("Split follow-up into intervals", "DuckDB") as b:
        n_iv = s5_intervals(con)
        b["rows_out"] = n_iv

    with bench.stage("Build the diagnosis event log", "DuckDB") as b:
        b["rows_out"] = s6_dx_events(con, cols, OUT_DIR)

    if WRITE_PERSON_PERIOD:
        with bench.stage("Build the person-period file", "DuckDB") as b:
            b["rows_out"] = s7_person_period(con, OUT_DIR)

    with bench.stage("Export and diagnostics", "DuckDB") as b:
        con.execute(f"""
            COPY (SELECT * FROM intervals)
            TO '{sql_path(os.path.join(OUT_DIR, "survival_intervals.csv"))}'
            (HEADER, DELIMITER ',');
        """)
        con.execute(f"""
            COPY (
                SELECT pid AS "{ID_COL}", origin, exit_date, event, death_year,
                       birth_year, dom, subsid, first_enc, last_enc,
                       DATE_DIFF('day', origin, exit_date) AS followup_days
                FROM person_final ORDER BY pid
            )
            TO '{sql_path(os.path.join(OUT_DIR, "survival_person_summary.csv"))}'
            (HEADER, DELIMITER ',');
        """)

        ptime, n_ev, n_w = con.sql("""
            SELECT SUM(DATE_DIFF('day', origin, exit_date)),
                   SUM(event), COUNT(*) FROM person_final
        """).fetchone()
        diag = pd.concat([excl, pd.DataFrame([
            {"metric": "person_years", "n": round(ptime / 365.25, 1)},
            {"metric": "deaths", "n": int(n_ev or 0)},
            {"metric": "median_followup_days", "n": con.sql(
                "SELECT MEDIAN(DATE_DIFF('day', origin, exit_date)) "
                "FROM person_final").fetchone()[0]},
            {"metric": "interval_rows", "n": n_iv},
            {"metric": "intervals_per_woman", "n": round(n_iv / n_w, 2)},
        ])], ignore_index=True)
        diag.to_csv(os.path.join(OUT_DIR, "followup_diagnostics.csv"),
                    index=False)
        print("\n    FOLLOW-UP SUMMARY")
        for _, r in diag.iterrows():
            val = f"{r['n']:,}" if isinstance(r["n"], (int, float)) else str(r["n"])
            print(f"      {str(r['metric']):<32}: {val:>14}")
        b["rows_out"] = n_iv

    con.close()
    bench.finalise(os.path.join(OUT_DIR, "benchmark_step15.csv"),
                   rows_out=n_iv)
    print(f"\nOutput written to {os.path.abspath(OUT_DIR)}")
    print("  survival_intervals.csv       counting process (tstart, tstop, event]")
    print("  survival_person_summary.csv  one row per woman")
    print("  survival_dx_events.csv       diagnosis event log for tmerge")
    if WRITE_PERSON_PERIOD:
        print("  survival_person_period.csv   woman-year file, discrete-time route")
    print("  death_variable_audit.csv     evidence on the death variable")
    print("  followup_diagnostics.csv     person-time and exclusions")


if __name__ == "__main__":
    main()
