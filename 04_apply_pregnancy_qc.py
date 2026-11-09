"""
=============================================================================
 STEP 4 -- PREGNANCY-LEVEL QUALITY CONTROL
=============================================================================

Applies the two exclusions that define the analytic pregnancy-level dataset:

  1. Biologically implausible maternal age (outside 12-55 years).
  2. Implausible diagnostic combinations, in which a current abortive outcome
     co-occurs with a condition that cannot accompany it.

AGE EXCLUSION LEVEL
-------------------
At the visit and individual levels the exclusion is unambiguous: a woman with
any implausible recorded age is removed together with all of her encounters.
At the pregnancy level there are two defensible choices, and they give
different counts:

  "episode"  Drop only the episode whose maternal age is out of range. Maternal
             age at this level is the earliest age recorded within the episode,
             so a woman's other episodes may fall inside the plausible range and
             remain valid. This is what the working script did.

  "patient"  Drop every episode of a woman who has any out-of-range episode,
             matching the rule applied at the visit and individual levels.

Both counts are reported on every run regardless of which is applied, so the
consequence of the choice is visible rather than implicit. Whichever is used
must be described accurately in the Methods: the two are not interchangeable
and the resulting cohort size differs.

DIAGNOSTIC CONFLICTS
--------------------
An episode is flagged when `c_abortive = 1` and at least one of the columns in
CONFLICT_COLS is positive. Flagging is at the episode level: only the
conflicted episode is removed, not the woman's other episodes.

The full conflict column list is written to `conflict_breakdown_step4.csv`
together with, for each column, the number of episodes it flags and the number
it flags on its own. A manuscript that describes the rule with an abbreviated
list cannot be reproduced from that description, so the complete list belongs
in a supplementary table; this file provides it.

NOTES ON THIS REVISION
----------------------
  1. `age_risk` is recomputed from the retained `age` after both exclusions.
     Step 3 already derives it this way, so this is a guard against a stale
     column rather than a correction, and it is idempotent.
  2. The order of the two exclusions is unchanged: age first, then conflicts.
     Counts are therefore conditional on the preceding step, which is how the
     flow diagram reads them.
  3. Column presence is verified before use, and any CONFLICT_COLS entry absent
     from the input is listed rather than silently skipped.
  4. Episode-sequence exclusion, applied after the conflict exclusion
     (sequence_flags / sequence_exclusion). Anchors are read from step 1
     because the previous episode may already be gone from step 3's file.
     Removed: a delivery whose onset precedes the previous delivery (re-code);
     an abortive episode followed by a delivery whose onset precedes the
     abortive date by more than SEQ_TOLERANCE_DAYS; an abortive episode that
     absorbed a later delivery record. Overlaps whose CURRENT episode is
     abortive arise from the 140-day upper-limit duration and are counted,
     not removed. Steps 11 and 13 call the same two functions.

INPUT
-----
    pregnancy by episode.csv            from step 3

OUTPUT
------
    pregnancy by episode cleaned.csv    analytic pregnancy-level dataset
    exclusion_report_step4.csv          counts for the flow diagram
    conflict_breakdown_step4.csv        per-column conflict contribution
    benchmark_step4.csv                 per-stage timing and memory
    environment_step4.txt               library versions and machine spec

USAGE
-----
    python 04_apply_pregnancy_qc.py
=============================================================================
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

import duckdb
import pandas as pd

from pipeline_utils import (AGE_MAX_DEFAULT, AGE_MIN_DEFAULT, Bench,
                            environment_report, warn_if_no_psutil)

# Relative paths in the configuration block below are resolved against the
# folder holding this script, not the shell's working directory. Without this,
# launching from an editor's Run button resolves them against the editor's
# install directory instead, and every path fails.
os.chdir(Path(__file__).resolve().parent)

# =============================================================================
# CONFIGURATION
# =============================================================================

IN_PATH  = r"./output_step3/pregnancy by episode.csv"
OUT_DIR  = r"./output_step4"
OUT_NAME = r"pregnancy by episode cleaned.csv"

# Roster of the women who contribute at least one surviving episode. Steps 5
# and 3 read this so that all three analytic levels describe the same
# population; without it the visit- and individual-level datasets also contain
# women who never produced an episode, and the cross-level comparison would
# confound the unit of aggregation with the population each level admits.
OUT_COHORT = r"analytic_cohort_ids.csv"

AGE_MIN, AGE_MAX = AGE_MIN_DEFAULT, AGE_MAX_DEFAULT
AGE_RISK_LOW, AGE_RISK_HIGH = 20, 35

# "episode" reproduces the working script; "patient" matches the visit- and
# individual-level rule. See the header for the consequences of each.
AGE_EXCLUSION_LEVEL = "episode"

ID_COL       = "PSTV01"
ABORTIVE_COL = "c_abortive"

# Episode-sequence QC reads the visit-level anchors from step 1, because the
# previous episode of a woman may already be gone from step 3's file (2015
# washout) and the delivery dates inside an episode are not kept at step 3.
VISIT_CSV_PATH = r"./output_step1/pregnancy by visit.csv"

# An abortive episode followed by a delivery whose back-calculated onset falls
# more than this many days before the abortive date is treated as an abortive
# code recorded during the pregnancy that ended in that delivery. Up to 14 days
# is compatible with conception at the first ovulation after the abortion,
# since onset is dated from a notional last menstrual period.
SEQ_TOLERANCE_DAYS = 14

# Onset durations of step 1, used to recover the termination type.
DUR_TO_TYPE = {140: "abortive", 259: "preterm", 280: "term", 294: "post-term"}

# Conditions that cannot accompany a current abortive outcome.
CONFLICT_COLS = [
    'c_preecl', 'c_ecl', 'c_anh', 'c_previa', 'c_abrupt',
    'c_polyhydra', 'c_abnamnio', 'c_prom', 'c_prolong', 'c_preterm',
    'c_fail', 'c_abnforce', 'c_long', 'c_malpres', 'c_obspelvic', 'c_iph',
    'c_distress', 'c_umbilical', 'c_laceration', 'c_obstrau', 'c_pph',
    'c_retained', 'c_normal', 'c_instrum', 'c_caesar', 'c_assisted',
    'c_multiple', 'c_disprop', 'c_malpresent', 'c_abnorpelv', 'c_placental'
]

# =============================================================================


def sql_path(path: str) -> str:
    return os.path.abspath(path).replace("\\", "/")


def load(con, path: str) -> tuple[int, list[str]]:
    if not os.path.exists(path):
        sys.exit(f"[FATAL] input not found: {os.path.abspath(path)}")
    con.execute(f"""
        CREATE OR REPLACE TABLE raw AS
        SELECT * FROM read_csv_auto('{sql_path(path)}', HEADER=TRUE,
                                    SAMPLE_SIZE=-1);
    """)
    cols = con.sql("SELECT name FROM pragma_table_info('raw')").df()["name"].tolist()
    n = con.sql("SELECT COUNT(*) FROM raw").fetchone()[0]
    print(f"    loaded {n:,} episodes x {len(cols)} columns")

    lower = {c.lower(): c for c in cols}
    for required in (ID_COL, ABORTIVE_COL, "age"):
        if required.lower() not in lower:
            print(f"\n[FATAL] required column '{required}' not found.")
            print("Columns present:")
            for c in cols:
                print(f"    {c}")
            sys.exit(1)
    return n, cols


def resolve_conflict_cols(cols: list[str]) -> tuple[list[str], list[str]]:
    lower = {c.lower(): c for c in cols}
    present = [lower[c.lower()] for c in CONFLICT_COLS if c.lower() in lower]
    absent = [c for c in CONFLICT_COLS if c.lower() not in lower]
    print(f"    conflict columns present : {len(present)} of {len(CONFLICT_COLS)}")
    if absent:
        print("    absent from the input (not evaluated):")
        for c in absent:
            print(f"      {c}")
    return present, absent


def age_exclusion(con) -> dict:
    """Apply the configured rule and report both alternatives."""
    if AGE_EXCLUSION_LEVEL not in ("episode", "patient"):
        sys.exit("[FATAL] AGE_EXCLUSION_LEVEL must be 'episode' or 'patient'.")

    in_range = f"age BETWEEN {AGE_MIN} AND {AGE_MAX}"

    n_total = con.sql("SELECT COUNT(*) FROM raw").fetchone()[0]
    n_ep_drop = con.sql(f"""
        SELECT COUNT(*) FROM raw WHERE NOT ({in_range}) OR age IS NULL
    """).fetchone()[0]
    n_pt_drop, n_women_affected = con.sql(f"""
        WITH bad AS (
            SELECT DISTINCT {ID_COL} FROM raw
            WHERE NOT ({in_range}) OR age IS NULL
        )
        SELECT (SELECT COUNT(*) FROM raw WHERE {ID_COL} IN (SELECT {ID_COL} FROM bad)),
               (SELECT COUNT(*) FROM bad)
    """).fetchone()

    print(f"    episodes with age outside {AGE_MIN}-{AGE_MAX} or missing : "
          f"{n_ep_drop:,}")
    print(f"    women affected                              : {n_women_affected:,}")
    print(f"    episodes removed under 'episode' rule       : {n_ep_drop:,}")
    print(f"    episodes removed under 'patient' rule       : {n_pt_drop:,}")
    print(f"    difference between the two rules            : "
          f"{n_pt_drop - n_ep_drop:,}")
    print(f"    -> applying the '{AGE_EXCLUSION_LEVEL}' rule")

    if AGE_EXCLUSION_LEVEL == "episode":
        con.execute(f"""
            CREATE OR REPLACE TABLE age_filtered AS
            SELECT * FROM raw WHERE {in_range};
        """)
    else:
        con.execute(f"""
            CREATE OR REPLACE TABLE age_filtered AS
            SELECT * FROM raw
            WHERE {ID_COL} NOT IN (
                SELECT DISTINCT {ID_COL} FROM raw
                WHERE NOT ({in_range}) OR age IS NULL
            );
        """)

    n_after = con.sql("SELECT COUNT(*) FROM age_filtered").fetchone()[0]
    return {"episodes_total": n_total,
            "removed_age_rule_episode": n_ep_drop,
            "removed_age_rule_patient": n_pt_drop,
            "women_with_implausible_age": n_women_affected,
            "age_rule_applied": AGE_EXCLUSION_LEVEL,
            "episodes_after_age_exclusion": n_after}


def conflict_exclusion(con, present: list[str]) -> tuple[dict, pd.DataFrame]:
    """Remove episodes where an abortive outcome co-occurs with an
    incompatible condition, and quantify each column's contribution."""
    if not present:
        print("    [WARNING] no conflict columns present; no episode removed")
        con.execute("CREATE OR REPLACE TABLE final AS SELECT * FROM age_filtered;")
        n = con.sql("SELECT COUNT(*) FROM final").fetchone()[0]
        return {"removed_diagnostic_conflict": 0,
                "episodes_final": n}, pd.DataFrame()

    total_expr = " + ".join(f'COALESCE("{c}",0)' for c in present)
    predicate = f'(COALESCE("{ABORTIVE_COL}",0)=1 AND ({total_expr})>0)'

    n_before = con.sql("SELECT COUNT(*) FROM age_filtered").fetchone()[0]
    n_drop = con.sql(f"SELECT COUNT(*) FROM age_filtered WHERE {predicate}"
                     ).fetchone()[0]

    # Per-column contribution, computed on the conflicted subset only.
    rows = []
    for c in present:
        hit, alone = con.sql(f"""
            SELECT
              COUNT(*) FILTER (WHERE COALESCE("{c}",0) > 0),
              COUNT(*) FILTER (WHERE COALESCE("{c}",0) > 0 AND ({total_expr}) =
                               COALESCE("{c}",0))
            FROM age_filtered
            WHERE COALESCE("{ABORTIVE_COL}",0) = 1
        """).fetchone()
        rows.append({"conflict_column": c, "episodes_flagged": hit,
                     "episodes_flagged_by_this_column_alone": alone})
    breakdown = pd.DataFrame(rows).sort_values("episodes_flagged",
                                               ascending=False)

    con.execute(f"""
        CREATE OR REPLACE TABLE final AS
        SELECT * FROM age_filtered WHERE NOT {predicate};
    """)
    n_final = con.sql("SELECT COUNT(*) FROM final").fetchone()[0]

    n_abortive = con.sql(f"""
        SELECT COUNT(*) FROM age_filtered
        WHERE COALESCE("{ABORTIVE_COL}",0) = 1
    """).fetchone()[0]
    print(f"    episodes with a current abortive outcome : {n_abortive:,}")
    print(f"    of which conflicted                      : {n_drop:,}"
          + (f"  ({n_drop / n_abortive * 100:.2f}%)" if n_abortive else ""))
    print(f"    episodes remaining                       : {n_final:,}")
    print("    top contributing conflict columns:")
    for _, r in breakdown.head(8).iterrows():
        print(f"      {r['conflict_column']:<16} {r['episodes_flagged']:>8,}"
              f"   alone: {r['episodes_flagged_by_this_column_alone']:>8,}")

    return {"episodes_with_abortive_outcome": n_abortive,
            "removed_diagnostic_conflict": n_drop,
            "episodes_final": n_final}, breakdown


def sequence_flags(visit: pd.DataFrame, id_col: str = ID_COL,
                   date_col: str = "combined_date", dx_cols=None,
                   abort_codes=None, partus_codes=None,
                   tolerance: int = SEQ_TOLERANCE_DAYS) -> pd.DataFrame:
    """Episodes whose sequence with the woman's other terminations is not
    biologically possible. Returns one row per flagged episode:
    [id_col, n_preg, reason].

      overlap_delivery   a delivery whose onset precedes the previous
                         delivery's termination; the later record is taken
                         as a re-code of the earlier delivery
      overlap_abortive   an abortive episode followed by a delivery whose
                         onset precedes the abortive date by more than
                         `tolerance` days; the abortive code is taken as
                         recorded during that ongoing pregnancy
      abortive_with_delivery
                         an abortive episode that absorbed a later delivery
                         record (gap shorter than the separation window)

    Uses `doa`/`dol` when present; otherwise derives them from the diagnosis
    columns with the supplied code lists.
    """
    v = visit
    for c in (id_col, "n_preg", "fin_g", "ref_start"):
        if c not in v.columns:
            sys.exit(f"[FATAL] sequence QC needs column '{c}' in the visit table")
    if {"doa", "dol"} <= set(v.columns):
        doa = pd.to_datetime(v["doa"], errors="coerce")
        dol = pd.to_datetime(v["dol"], errors="coerce")
    else:
        if dx_cols is None or abort_codes is None or partus_codes is None:
            sys.exit("[FATAL] no doa/dol columns and no code lists supplied")
        d = pd.to_datetime(v[date_col], errors="coerce")
        doa = d.where(v[dx_cols].isin(abort_codes).any(axis=1))
        dol = d.where(v[dx_cols].isin(partus_codes).any(axis=1))

    key_id = v[id_col].astype(str)
    key_k = pd.to_numeric(v["n_preg"], errors="coerce")
    fin = pd.to_datetime(v["fin_g"], errors="coerce")
    rs = pd.to_datetime(v["ref_start"], errors="coerce")

    ep = (pd.DataFrame({id_col: key_id, "n_preg": key_k,
                        "fin_g": fin, "ref_start": rs})
            .loc[fin.notna() & key_k.notna()]
            .drop_duplicates([id_col, "n_preg"])
            .sort_values([id_col, "n_preg"])
            .reset_index(drop=True))
    ep["type"] = (ep["fin_g"] - ep["ref_start"]).dt.days.map(DUR_TO_TYPE)
    if ep["type"].isna().any():
        sys.exit("[FATAL] onset durations outside DUR_TO_TYPE; check step 1")
    g = ep.groupby(id_col)
    ep["prev_end"] = g["fin_g"].shift()
    ep["prev_type"] = g["type"].shift()
    ov_days = (ep["prev_end"] - ep["ref_start"]).dt.days
    is_deliv = ep["type"].ne("abortive")
    prev_deliv = ep["prev_type"].notna() & ep["prev_type"].ne("abortive")

    r1 = is_deliv & prev_deliv & (ov_days > 0)
    trig = is_deliv & ep["prev_type"].eq("abortive") & (ov_days > tolerance)
    r2 = (trig.groupby(ep[id_col]).shift(-1).astype("boolean")
              .fillna(False).astype(bool)) & ep["type"].eq("abortive")

    first_dol = (pd.DataFrame({id_col: key_id, "n_preg": key_k, "dol": dol})
                   .dropna(subset=["dol", "n_preg"])
                   .groupby([id_col, "n_preg"])["dol"].min())
    ep = ep.join(first_dol.rename("first_dol"), on=[id_col, "n_preg"])
    r3 = ep["type"].eq("abortive") & (ep["first_dol"] > ep["fin_g"])

    n_ab_ov = int((ep["type"].eq("abortive") & (ov_days > 0)).sum())
    print(f"    overlaps with a current abortive episode (kept): {n_ab_ov:,}")

    ep["reason"] = None
    ep.loc[r3, "reason"] = "abortive_with_delivery"
    ep.loc[r2, "reason"] = "overlap_abortive"
    ep.loc[r1, "reason"] = "overlap_delivery"
    out = ep.loc[ep["reason"].notna(), [id_col, "n_preg", "reason"]]
    out["n_preg"] = out["n_preg"].astype(int)
    print(f"    sequence flags on all anchors : "
          + ", ".join(f"{k} {n:,}" for k, n in out["reason"].value_counts().items()))
    return out.reset_index(drop=True)


def _key(expr: str) -> str:
    """Normalise an identifier for joining across CSV type inference."""
    return f"LTRIM(CAST({expr} AS VARCHAR), '0')"


def sequence_exclusion(con, flags: pd.DataFrame, id_col: str = ID_COL) -> dict:
    """Remove flagged episodes from table `final`; counts are of episodes
    still present after the age and conflict exclusions."""
    con.register("seq_df", flags)
    fid, sid = f'f."{id_col}"', f's."{id_col}"'
    on = (f"{_key(fid)} = {_key(sid)} AND "
          'CAST(ROUND(CAST(f."n_preg" AS DOUBLE)) AS BIGINT) = '
          'CAST(s."n_preg" AS BIGINT)')
    hit = con.sql(f"""
        SELECT s.reason, COUNT(*) AS n FROM final f JOIN seq_df s ON {on}
        GROUP BY s.reason
    """).df().set_index("reason")["n"].to_dict()
    n_before = con.sql("SELECT COUNT(*) FROM final").fetchone()[0]
    con.execute(f"""
        CREATE OR REPLACE TABLE final AS
        SELECT f.* FROM final f ANTI JOIN seq_df s ON {on};
    """)
    con.unregister("seq_df")
    n_after = con.sql("SELECT COUNT(*) FROM final").fetchone()[0]
    rep = {"removed_overlap_delivery": int(hit.get("overlap_delivery", 0)),
           "removed_overlap_abortive": int(hit.get("overlap_abortive", 0)),
           "removed_abortive_with_delivery":
               int(hit.get("abortive_with_delivery", 0)),
           "episodes_after_conflict": n_before,
           "episodes_final": n_after}
    print(f"    removed, delivery re-coded       : "
          f"{rep['removed_overlap_delivery']:>8,}")
    print(f"    removed, abortive code in an ongoing pregnancy (overlap) : "
          f"{rep['removed_overlap_abortive']:>8,}")
    print(f"    removed, abortive episode containing a delivery          : "
          f"{rep['removed_abortive_with_delivery']:>8,}")
    print(f"    episodes remaining               : {n_after:,}")
    return rep


def recompute_age_risk(con) -> None:
    """Derive age_risk from the retained age. Idempotent."""
    cols = con.sql("SELECT name FROM pragma_table_info('final')").df()["name"]
    if "age_risk" not in cols.tolist():
        con.execute("ALTER TABLE final ADD COLUMN age_risk INT")
    con.execute(f"""
        UPDATE final SET age_risk = CASE
            WHEN age IS NULL THEN NULL
            WHEN age < {AGE_RISK_LOW} OR age > {AGE_RISK_HIGH} THEN 1
            ELSE 0 END;
    """)
    n1, n = con.sql("""
        SELECT SUM(CASE WHEN age_risk = 1 THEN 1 ELSE 0 END), COUNT(*) FROM final
    """).fetchone()
    print(f"    extreme maternal age : {n1:,} of {n:,}  ({n1 / n * 100:.2f}%)")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 74)
    print(" STEP 4 -- PREGNANCY-LEVEL QUALITY CONTROL")
    print("=" * 74)
    print(environment_report(os.path.join(OUT_DIR, "environment_step4.txt")))
    warn_if_no_psutil()

    bench = Bench("step 4, pregnancy-level QC")
    con = duckdb.connect(database=":memory:")

    with bench.stage("Load pregnancy-level table", "DuckDB") as b:
        n_in, cols = load(con, IN_PATH)
        present, absent = resolve_conflict_cols(cols)
        b["rows_out"] = n_in

    with bench.stage("Age exclusion", "DuckDB", rows_in=n_in) as b:
        age_rep = age_exclusion(con)
        b["rows_out"] = age_rep["episodes_after_age_exclusion"]

    with bench.stage("Diagnostic conflict exclusion", "DuckDB",
                     rows_in=age_rep["episodes_after_age_exclusion"]) as b:
        conf_rep, breakdown = conflict_exclusion(con, present)
        b["rows_out"] = conf_rep["episodes_final"]

    with bench.stage("Episode-sequence exclusion", "pandas+DuckDB",
                     rows_in=conf_rep["episodes_final"]) as b:
        hdr = pd.read_csv(VISIT_CSV_PATH, nrows=0).columns
        use = [ID_COL, "n_preg", "fin_g", "ref_start"]
        kw = {}
        if {"doa", "dol"} <= set(hdr):
            use += ["doa", "dol"]
        else:
            import importlib
            s1 = importlib.import_module("01_build_visit_level")
            use += ["combined_date"] + list(s1.DX_COLS)
            kw = dict(dx_cols=s1.DX_COLS, abort_codes=s1.ABORT_CODES,
                      partus_codes=s1.PARTUS_CODES)
        visit = pd.read_csv(VISIT_CSV_PATH, usecols=use,
                            dtype={ID_COL: str}, low_memory=False)
        flags = sequence_flags(visit, **kw)
        del visit
        flags.to_csv(os.path.join(OUT_DIR, "sequence_flags_step4.csv"),
                     index=False)
        seq_rep = sequence_exclusion(con, flags)
        conf_rep["episodes_final"] = seq_rep.pop("episodes_final")
        b["rows_out"] = conf_rep["episodes_final"]

    with bench.stage("Recompute age_risk and export", "DuckDB",
                     rows_in=conf_rep["episodes_final"]) as b:
        recompute_age_risk(con)
        out_path = os.path.join(OUT_DIR, OUT_NAME)
        con.execute(f"""
            COPY final TO '{sql_path(out_path)}' (HEADER, DELIMITER ',');
        """)
        print(f"    wrote {conf_rep['episodes_final']:,} episodes -> {out_path}")

        roster_path = os.path.join(OUT_DIR, OUT_COHORT)
        con.execute(f"""
            COPY (SELECT DISTINCT "{ID_COL}" FROM final ORDER BY 1)
            TO '{sql_path(roster_path)}' (HEADER, DELIMITER ',');
        """)
        n_women = con.sql(f'SELECT COUNT(DISTINCT "{ID_COL}") FROM final'
                          ).fetchone()[0]
        conf_rep["women_in_analytic_cohort"] = n_women
        print(f"    wrote {n_women:,} patient identifiers -> {roster_path}")
        print(f"    steps 5 and 6 must be run AFTER this step and pointed at "
              f"that file")
        b["rows_out"] = conf_rep["episodes_final"]

    rep = {**age_rep, **{k: v for k, v in conf_rep.items()
                         if k != "episodes_final"},
           **seq_rep, "episodes_final": conf_rep["episodes_final"]}
    pd.DataFrame([{"metric": k, "value": v} for k, v in rep.items()]).to_csv(
        os.path.join(OUT_DIR, "exclusion_report_step4.csv"), index=False)
    if len(breakdown):
        breakdown.to_csv(os.path.join(OUT_DIR, "conflict_breakdown_step4.csv"),
                         index=False)
    if absent:
        pd.DataFrame({"absent_conflict_column": absent}).to_csv(
            os.path.join(OUT_DIR, "conflict_columns_absent_step4.csv"),
            index=False)

    print("\n    FLOW DIAGRAM COUNTS")
    for k, v in rep.items():
        val = f"{v:,}" if isinstance(v, (int, float)) else str(v)
        print(f"      {k:<36}: {val:>12}")

    con.close()
    bench.finalise(os.path.join(OUT_DIR, "benchmark_step4.csv"),
                   rows_out=conf_rep["episodes_final"])
    print(f"\nOutput written to {os.path.abspath(OUT_DIR)}")


if __name__ == "__main__":
    main()
