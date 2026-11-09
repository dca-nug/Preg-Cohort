"""
=============================================================================
 STEP 1 -- BUILD THE VISIT-LEVEL DATASET
=============================================================================

Integrates the four BPJS Kesehatan sample tables into a single visit-level
table, derives patient characteristics, anchors pregnancy episodes, and
back-calculates a reference start date for each episode.

This script reproduces the logic of the original exploratory notebook
(`Dataset_Generation_for_visit_level.ipynb`) without alteration, with four
additions required for release and for computational reporting:

  1. Per-stage instrumentation: wall-clock time, peak resident memory, and
     input/output row counts. Written to `benchmark_step1.csv`.
  2. `CREATE OR REPLACE TABLE` instead of `CREATE TABLE`, so that re-running
     the script on an existing database file does not fail.
  3. Duplicate-column removal after the full outer join is now detected
     dynamically rather than relying on DuckDB's positional suffix naming,
     which is not stable across DuckDB versions.
  4. `doa`, `dol`, and `dopt` are retained by default (KEEP_DOPT_COLUMNS).
     The notebook dropped them, which makes the raw termination dates
     unrecoverable downstream; `fin_g` holds only anchor dates, so any
     interval below the separation threshold has already been collapsed.
     Set KEEP_DOPT_COLUMNS = False to match the notebook byte for byte.
  5. The separation threshold is 182 days (26 weeks): the earliest return of
     ovulation after delivery in most women (6 weeks) plus the earliest
     gestation recorded as a delivery (20 weeks). Anchoring is implemented once,
     in `anchor_from_masks`, and reused by steps 11 (sensitivity) and 12
     (TMUCRD). The threshold may also depend on the type of the later
     termination record (SEPARATION_SENS, used by step 11 only).

The pipeline is fully deterministic: every transformation is rule-based and
no stochastic component is involved, so a given input yields an identical
output on repeated execution.

ARCHITECTURE
------------
Ingestion, joins, and column-wise derivations run in DuckDB, which evaluates
them out of core and avoids materialising intermediate copies. Episode
anchoring and the gestational back-calculation run in pandas, because they
depend on grouped, order-sensitive date arithmetic (`groupby().shift()` over
a per-patient chronological ordering) that is awkward to express in SQL.
The handover point is a single `SELECT *` into a DataFrame.

INPUT
-----
Four Stata files in DATA_DIR:
    FKTP_2023.dta               primary care visits
    FKRTL_2023.dta              hospital visits
    FKRTL_Sekunder_2023.dta     hospital secondary diagnoses
    Kepesertaan_2023.dta        membership / demographics

OUTPUT
------
    pregnancy by visit.csv      visit-level analytic table
    npreg_distribution_raw.csv  episode-order distribution before any cap
    benchmark_step1.csv         per-stage timing and memory
    environment_step1.txt       library versions and machine specification

USAGE
-----
    python 01_build_visit_level.py

A synthetic input generator is provided in `make_synthetic_input.py` so that
the pipeline can be executed end to end without access to BPJS Kesehatan
data, which cannot be redistributed.
=============================================================================
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import sys
import time

import duckdb
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

DATA_DIR = r"."                     # folder holding the four .dta files
OUT_DIR  = r"./output_step1"        # all output is written here
DB_PATH  = r"./fin_maternal.db"     # DuckDB database file

SRC_FKTP     = "FKTP_2023.dta"
SRC_FKRTL    = "FKRTL_2023.dta"
SRC_SEKUNDER = "FKRTL_Sekunder_2023.dta"
SRC_PESERTA  = "Kepesertaan_2023.dta"

OUT_VISIT_CSV = "pregnancy by visit.csv"
WRITE_PARQUET = False               # Parquet is smaller and faster to reload

# Retain doa / dol / dopt in the output (see note 4 in the header).
KEEP_DOPT_COLUMNS = True

# --- clinical and coding parameters -----------------------------------------
DX_COLS        = ["FKP14A", "FKL15A", "FKL17A", "FKL24A"]
ABORT_CODES    = ["O00", "O01", "O02", "O03", "O04", "O05", "O06", "O07"]
PARTUS_CODES   = ["O80", "O81", "O82", "O83", "O84"]
PRETERM_CODES  = ["O60"]
POSTTERM_CODES = ["O48"]

# Episode separation threshold, main analysis: uniform 182 days (26 weeks).
SEPARATION_DAYS = 182

# Sensitivity scheme (step 11): the threshold depends on the type of the LATER
# termination record. Values are the empirical troughs of the interval
# distributions (step 2) rounded to the nearest whole week:
#   next record a delivery : 259 days (37 weeks; trough 262 for abortive->delivery)
#   next record abortive   : 105 days (15 weeks; trough 108 for delivery->abortive)
SEPARATION_SENS = {"delivery": 259, "abortive": 105}

# Gestational duration assumed for the back-calculation of episode onset.
# abortive <20 wk, preterm <37 wk, term 40 wk, post-term >=42 wk.
DAYS_ABORT, DAYS_PRETERM, DAYS_POSTTERM, DAYS_TERM = 140, 259, 294, 280

# Java-Bali province service codes; everything else is coded as outside.
JAVA_BALI_CODES = (31, 32, 33, 34, 35, 36, 51)

# Subsidised membership categories (PSTV08).
SUBSIDISED_CODES = (2, 3)

# Age thresholds are NOT applied here. Biologically implausible ages are
# flagged but retained at this stage, because removing individual encounters
# would break the longitudinal linkage that episode reconstruction depends on.
AGE_MIN, AGE_MAX = 12, 55

# =============================================================================
# PIPELINE STAGES
# =============================================================================


def s1_ingest(con) -> dict:
    """Cells 0-2: read the four Stata files and register them as DuckDB tables."""
    counts = {}

    spec = [
        (SRC_FKTP, "fktp1",
         "PSTV01, FKP02, FKP03, FKP04, FKP05, FKP13, FKP14A"),
        (SRC_FKRTL, "fkrtl1",
         "PSTV01, FKP02, FKL02, FKL03, FKL04, FKL05, FKL09, FKL11, FKL14, "
         "FKL15A, FKL17A"),
        (SRC_SEKUNDER, "sek1",
         "FKL02, FKL24A"),
        (SRC_PESERTA, "peserta1",
         "PSTV01, PSTV03, PSTV08, PSTV18"),
    ]

    for fname, table, cols in spec:
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            sys.exit(f"[FATAL] input file not found: {os.path.abspath(path)}")
        df = pd.read_stata(path, convert_categoricals=False)
        con.register("src_temp", df)
        con.sql(f"CREATE OR REPLACE TABLE {table} AS SELECT {cols} FROM src_temp;")
        con.unregister("src_temp")
        counts[table] = len(df)
        print(f"    {fname:<28} -> {table:<10} {len(df):>12,} rows")
        del df

    return counts


def s2_merge(con) -> int:
    """Cell 3: attach secondary diagnoses, union primary care with hospital,
    then attach membership demographics."""
    # Hospital visits joined to their secondary diagnoses on the visit key.
    con.sql("""
        CREATE OR REPLACE TABLE rs AS
        SELECT p.*, s.FKL24A
        FROM fkrtl1 AS p
        LEFT JOIN sek1 AS s USING (FKL02);
    """)
    n_rs = con.sql("SELECT COUNT(*) FROM rs").fetchone()[0]
    print(f"    hospital visits + secondary diagnoses : {n_rs:>12,} rows")

    # Full outer join so that no encounter from either care level is lost.
    con.sql("""
        CREATE OR REPLACE TABLE klin AS
        SELECT
          COALESCE(CAST(f.PSTV01 AS VARCHAR), CAST(p.PSTV01 AS VARCHAR)) AS PSTV01,
          COALESCE(CAST(f.FKP02  AS VARCHAR), CAST(p.FKP02  AS VARCHAR)) AS FKP02,
          f.* EXCLUDE (FKP02),
          p.* EXCLUDE (FKP02)
        FROM rs AS f
        FULL OUTER JOIN fktp1 AS p
          ON CAST(f.PSTV01 AS VARCHAR) = CAST(p.PSTV01 AS VARCHAR)
         AND CAST(f.FKP02  AS VARCHAR) = CAST(p.FKP02  AS VARCHAR);
    """)
    n_klin = con.sql("SELECT COUNT(*) FROM klin").fetchone()[0]
    print(f"    combined clinical visits              : {n_klin:>12,} rows")

    con.sql("""
        CREATE OR REPLACE TABLE kia AS
        SELECT k.*, m.PSTV03, m.PSTV08, m.PSTV18
        FROM klin AS k
        LEFT JOIN peserta1 AS m USING (PSTV01);
    """)
    n_kia = con.sql("SELECT COUNT(*) FROM kia").fetchone()[0]
    print(f"    with membership demographics          : {n_kia:>12,} rows")
    return n_kia


def s3_drop_duplicate_columns(con) -> None:
    """Cell 5: remove the duplicate identifier columns produced by the join.

    The notebook dropped 'pstv01_1', 'pstv01_2', 'fkl02', and 'fkp02' by name.
    DuckDB's suffix naming for duplicated columns is not stable across
    versions, so duplicates are detected dynamically here.
    """
    cols = con.sql("SELECT name FROM pragma_table_info('kia')").df()["name"].tolist()
    dupes = [c for c in cols if re.fullmatch(r"(?i)pstv01_\d+", c)]
    # FKL02 and FKP02 served only as join keys.
    keys = [c for c in cols if c.lower() in ("fkl02", "fkp02")]

    for c in dupes + keys:
        con.sql(f'ALTER TABLE kia DROP COLUMN "{c}"')
        print(f"    dropped column: {c}")
    if not dupes:
        print("    no duplicated PSTV01 columns found "
              "(DuckDB naming may differ; verify column list)")


def s4_characteristics(con) -> None:
    """Cell 6: derive age, extreme-age indicator, region, and subsidy status."""
    con.sql("ALTER TABLE kia ADD COLUMN age INT")
    con.sql("""
        UPDATE kia SET age = CASE
            WHEN fkl03 IS NOT NULL THEN EXTRACT(YEAR FROM fkl03) - EXTRACT(YEAR FROM pstv03)
            WHEN fkp03 IS NOT NULL THEN EXTRACT(YEAR FROM fkp03) - EXTRACT(YEAR FROM pstv03)
        END;
    """)

    # age_risk = 1 for extreme maternal age (<20 or >35 years).
    con.sql("ALTER TABLE kia ADD COLUMN age_risk INT")
    con.sql("""
        UPDATE kia SET age_risk = CASE
            WHEN age IS NULL THEN NULL
            WHEN age < 20 OR age > 35 THEN 1
            ELSE 0
        END;
    """)

    # dom = 0 within Java-Bali, 1 outside.
    codes = ", ".join(str(c) for c in JAVA_BALI_CODES)
    con.sql("ALTER TABLE kia ADD COLUMN dom INT")
    con.sql(f"""
        UPDATE kia SET dom = CASE
            WHEN COALESCE(fkl05, fkp05) IN ({codes}) THEN 0
            WHEN fkl05 IS NULL AND fkp05 IS NULL THEN NULL
            ELSE 1
        END;
    """)

    sub = ", ".join(str(c) for c in SUBSIDISED_CODES)
    con.sql("ALTER TABLE kia ADD COLUMN subsid INT")
    con.sql(f"""
        UPDATE kia SET subsid = CASE
            WHEN (pstv08) IN ({sub}) THEN 1
            WHEN pstv08 IS NULL THEN NULL
            ELSE 0
        END;
    """)

    for c in ("pstv03", "fkl05", "fkp05", "pstv08"):
        con.sql(f'ALTER TABLE kia DROP COLUMN "{c}"')

    # Report, but do not act on, biologically implausible ages.
    q = con.sql(f"""
        SELECT
          SUM(CASE WHEN age IS NULL THEN 1 ELSE 0 END) AS age_missing,
          SUM(CASE WHEN age < {AGE_MIN} OR age > {AGE_MAX} THEN 1 ELSE 0 END) AS age_oob,
          SUM(CASE WHEN age_risk = 1 THEN 1 ELSE 0 END) AS extreme_age
        FROM kia
    """).fetchone()
    print(f"    age missing                  : {q[0] or 0:>12,}")
    print(f"    age outside {AGE_MIN}-{AGE_MAX} (flagged only): {q[1] or 0:>12,}")
    print(f"    extreme age (<20 or >35)     : {q[2] or 0:>12,}")


def s5_to_pandas(con) -> pd.DataFrame:
    """Cell 7: hand over to pandas for order-sensitive date arithmetic."""
    df = con.sql("SELECT * FROM kia").df()
    print(f"    materialised {len(df):,} rows x {len(df.columns)} columns")
    return df


def _separation_threshold(sep, later_is_abortive: pd.Series) -> pd.Series:
    """Threshold for each termination record. `sep` is an int (uniform) or a
    dict {"delivery": d, "abortive": a} keyed by the type of that record, i.e.
    the LATER record of the pair being compared."""
    if isinstance(sep, dict):
        return pd.Series(np.where(later_is_abortive, sep["abortive"],
                                  sep["delivery"]), index=later_is_abortive.index)
    return pd.Series(int(sep), index=later_is_abortive.index)


def anchor_from_masks(df: pd.DataFrame, sep, abortus_mask, partus_mask,
                      preterm_mask, postterm_mask,
                      days_abort: int = DAYS_ABORT,
                      days_preterm: int = DAYS_PRETERM,
                      days_postterm: int = DAYS_POSTTERM,
                      days_term: int = DAYS_TERM,
                      id_col: str = "PSTV01", date_col: str = "combined_date",
                      verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    """Anchor pregnancy episodes and back-calculate onset.

    The single implementation of the anchoring rule, used by step 1 (BPJS),
    step 11 (separation sensitivity) and step 12 (TMUCRD). The separation
    interval is measured against the immediately preceding termination record,
    not against the last accepted anchor; a chain of records each closer than
    the threshold is therefore absorbed into a single episode. The episode type
    is that of its first record, with an abortive code taking precedence when
    a record carries both.
    """
    d = pd.to_datetime(df[date_col], errors="coerce")
    df["doa"] = d.where(abortus_mask)
    df["dol"] = d.where(partus_mask)
    df["dopt"] = df["doa"].fillna(df["dol"])

    df = df.sort_values([id_col, "dopt"])
    ab = abortus_mask.reindex(df.index).fillna(False).astype(bool)
    pa = partus_mask.reindex(df.index).fillna(False).astype(bool)
    pre = preterm_mask.reindex(df.index).fillna(False).astype(bool)
    post = postterm_mask.reindex(df.index).fillna(False).astype(bool)

    prev = df.groupby(id_col, dropna=False)["dopt"].shift()
    gap = (df["dopt"] - prev).dt.days
    thr = _separation_threshold(sep, ab)
    is_anchor = df["dopt"].notna() & (prev.isna() | (gap >= thr))
    df["fin_g"] = df["dopt"].where(is_anchor)

    df["ref_start"] = pd.NaT
    idx = {"abortive": is_anchor & ab,
           "preterm": is_anchor & pa & ~ab & pre,
           "post-term": is_anchor & pa & ~ab & ~pre & post,
           "term": is_anchor & pa & ~ab & ~pre & ~post}
    days = {"abortive": days_abort, "preterm": days_preterm,
            "post-term": days_postterm, "term": days_term}
    for k, m in idx.items():
        df.loc[m, "ref_start"] = df.loc[m, "fin_g"] - pd.to_timedelta(days[k], unit="D")

    df["n_preg"] = (is_anchor.astype(int).groupby(df[id_col]).cumsum()
                    .where(df["dopt"].notna())).astype("Int64")

    diag = {"separation": str(sep),
            "records_abortive": int(ab.sum()), "records_delivery": int(pa.sum()),
            "records_with_dopt": int(df["dopt"].notna().sum()),
            "anchors": int(is_anchor.sum()),
            **{f"anchors_{k}": int(m.sum()) for k, m in idx.items()}}
    if verbose:
        print(f"    visits carrying a DOPT       : {diag['records_with_dopt']:>12,}")
        print(f"    episode anchors ({sep} d)     : {diag['anchors']:>12,}")
        for k in idx:
            print(f"      onset -{days[k]:>3} d  {k:<10} {diag['anchors_' + k]:>10,}")
    return df, diag


def anchor_episodes(df: pd.DataFrame, sep, dx_cols=DX_COLS,
                    abort_codes=ABORT_CODES, partus_codes=PARTUS_CODES,
                    preterm_codes=PRETERM_CODES, postterm_codes=POSTTERM_CODES,
                    days_abort: int = DAYS_ABORT, days_preterm: int = DAYS_PRETERM,
                    days_postterm: int = DAYS_POSTTERM, days_term: int = DAYS_TERM,
                    id_col: str = "PSTV01", date_col: str = "combined_date",
                    verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    """Build the code masks from the diagnosis columns and anchor episodes."""
    if date_col not in df.columns:
        for c in ("FKP03", "FKL03"):
            df[c] = pd.to_datetime(df[c], errors="coerce")
        df[date_col] = df[["FKP03", "FKL03"]].min(axis=1)
    m = {k: df[dx_cols].isin(v).any(axis=1) for k, v in
         (("ab", abort_codes), ("pa", partus_codes), ("pre", preterm_codes),
          ("post", postterm_codes))}
    return anchor_from_masks(df, sep, m["ab"], m["pa"], m["pre"], m["post"],
                             days_abort, days_preterm, days_postterm, days_term,
                             id_col=id_col, date_col=date_col, verbose=verbose)


def s6_episode_anchoring(df: pd.DataFrame) -> pd.DataFrame:
    """Cell 8: derive DOPT, anchor episodes, back-calculate onset, count episodes.

    The anchoring itself is in `anchor_from_masks` (see its docstring), so that
    steps 11 and 12 apply exactly the same rule.
    """
    for c in ("FKP03", "FKL03"):
        df[c] = pd.to_datetime(df[c], errors="coerce")

    df["combined_date"] = df[["FKP03", "FKL03"]].min(axis=1)

    abortus_mask  = df[DX_COLS].isin(ABORT_CODES).any(axis=1)
    partus_mask   = df[DX_COLS].isin(PARTUS_CODES).any(axis=1)
    preterm_mask  = df[DX_COLS].isin(PRETERM_CODES).any(axis=1)
    postterm_mask = df[DX_COLS].isin(POSTTERM_CODES).any(axis=1)

    if not abortus_mask.any() and not partus_mask.any():
        print("    [FATAL] no termination codes matched. The diagnosis columns "
              "may hold codes longer than three characters.")
        for c in DX_COLS:
            vals = df[c].dropna().astype(str)
            ex = vals[vals.str.startswith("O")].unique()[:12]
            print(f"      {c}: values beginning with O -> {list(ex)}")
        sys.exit(1)

    print(f"    visits with an abortive code : {int(abortus_mask.sum()):>12,}")
    print(f"    visits with a delivery code  : {int(partus_mask.sum()):>12,}")

    df, _ = anchor_from_masks(df, SEPARATION_DAYS, abortus_mask, partus_mask,
                              preterm_mask, postterm_mask)

    if not KEEP_DOPT_COLUMNS:
        df = df.drop(columns=["doa", "dol", "dopt"])
        print("    doa / dol / dopt dropped (KEEP_DOPT_COLUMNS = False)")
    else:
        print("    doa / dol / dopt retained")

    return df


def s7_export(df: pd.DataFrame) -> None:
    """Cell 13 plus the episode-order distribution that the notebook printed
    but never saved."""
    csv_path = os.path.join(OUT_DIR, OUT_VISIT_CSV)
    df.to_csv(csv_path, index=False)
    size = os.path.getsize(csv_path) / 1024 ** 3
    print(f"    wrote {csv_path}  ({size:.2f} GB)")

    if WRITE_PARQUET:
        pq = os.path.splitext(csv_path)[0] + ".parquet"
        df.to_parquet(pq, index=False)
        print(f"    wrote {pq}  "
              f"({os.path.getsize(pq) / 1024 ** 3:.2f} GB)")

    # Episode-order distribution before any cap is applied downstream. This is
    # what determines whether a MAX_N_PREG limit would silently discard episodes.
    ep = df.loc[df["fin_g"].notna(), ["PSTV01", "n_preg"]].dropna(subset=["n_preg"])
    dist = ep["n_preg"].astype(int).value_counts().sort_index()
    dist.rename("n_episodes").rename_axis("n_preg").to_csv(
        os.path.join(OUT_DIR, "npreg_distribution_raw.csv"))
    print(f"    highest episode order present : {int(dist.index.max())}")
    print("    episode-order distribution:")
    for k, n in dist.items():
        print(f"      {k:>3} : {n:>10,}")


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 74)
    print(" STEP 1 -- BUILD THE VISIT-LEVEL DATASET")
    print("=" * 74)
    print(environment_report(os.path.join(OUT_DIR, "environment_step1.txt")))
    warn_if_no_psutil()

    bench = Bench("step 1, visit level")
    con = duckdb.connect(DB_PATH)

    with bench.stage("Ingest source tables", "DuckDB") as b:
        counts = s1_ingest(con)
        b["rows_out"] = sum(counts.values())

    with bench.stage("Integrate visits and demographics", "DuckDB",
               rows_in=sum(counts.values())) as b:
        b["rows_out"] = s2_merge(con)

    with bench.stage("Remove duplicate join columns", "DuckDB") as b:
        s3_drop_duplicate_columns(con)
        b["rows_out"] = con.sql("SELECT COUNT(*) FROM kia").fetchone()[0]

    with bench.stage("Derive patient characteristics", "DuckDB") as b:
        s4_characteristics(con)
        b["rows_out"] = con.sql("SELECT COUNT(*) FROM kia").fetchone()[0]

    with bench.stage("Materialise to DataFrame", "DuckDB -> pandas") as b:
        df = s5_to_pandas(con)
        b["rows_out"] = len(df)

    with bench.stage("Anchor episodes and back-calculate onset", "pandas",
               rows_in=len(df)) as b:
        df = s6_episode_anchoring(df)
        b["rows_out"] = len(df)

    with bench.stage("Export visit-level table", "pandas", rows_in=len(df)) as b:
        s7_export(df)
        b["rows_out"] = len(df)

    con.close()
    bench.finalise(os.path.join(OUT_DIR, "benchmark_step1.csv"), rows_out=len(df))
    print(f"\nOutput written to {os.path.abspath(OUT_DIR)}")


if __name__ == "__main__":
    main()
