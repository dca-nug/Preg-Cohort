"""
=============================================================================
 SHARED INSTRUMENTATION
=============================================================================

Per-stage timing, peak resident memory, and environment reporting, used by
every script in the pipeline so that computational performance is measured
the same way throughout.

Peak memory is obtained by sampling the process resident set size in a
background thread rather than reading it at stage boundaries, so transient
peaks inside a stage are captured. If `psutil` is not installed the memory
columns are reported as missing and timing still works.
=============================================================================
"""

from __future__ import annotations

import os
import platform
import sys
import threading
import time
from contextlib import contextmanager

import numpy as np
import pandas as pd

try:
    import psutil
except ImportError:
    psutil = None


class PeakMemory:
    """Sample resident set size in a background thread and keep the maximum."""

    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.peak = 0
        self._stop = threading.Event()
        self._thread = None
        self._proc = psutil.Process(os.getpid()) if psutil else None

    def _run(self):
        while not self._stop.is_set():
            try:
                self.peak = max(self.peak, self._proc.memory_info().rss)
            except Exception:
                pass
            self._stop.wait(self.interval)

    def start(self):
        if self._proc is None:
            return self
        self.peak = self._proc.memory_info().rss
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> float:
        """Return peak RSS in gigabytes, or NaN if psutil is unavailable."""
        if self._proc is None:
            return float("nan")
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        return self.peak / 1024 ** 3


class Bench:
    """Collects one record per pipeline stage and renders the summary table."""

    def __init__(self, label: str):
        self.label = label
        self.rows: list[dict] = []
        self.t_start = time.perf_counter()

    @contextmanager
    def stage(self, name: str, engine: str, rows_in=None):
        print(f"\n[{len(self.rows) + 1}] {name}  ({engine})", flush=True)
        mon = PeakMemory().start()
        t0 = time.perf_counter()
        box = {"rows_out": None}
        try:
            yield box
        finally:
            elapsed = time.perf_counter() - t0
            peak = mon.stop()
            self.rows.append({
                "stage": name,
                "engine": engine,
                "rows_in": rows_in,
                "rows_out": box["rows_out"],
                "wall_clock_s": round(elapsed, 2),
                "peak_memory_gb": round(peak, 3) if peak == peak else None,
            })
            mem = f"{peak:.2f} GB" if peak == peak else "n/a"
            print(f"    done in {elapsed:,.2f} s   peak memory {mem}", flush=True)

    def finalise(self, out_csv: str, rows_out=None) -> pd.DataFrame:
        total = time.perf_counter() - self.t_start
        df = pd.DataFrame(self.rows)
        peak = df["peak_memory_gb"].max() if len(df) else None
        df.loc[len(df)] = {"stage": "TOTAL", "engine": "-", "rows_in": None,
                           "rows_out": rows_out, "wall_clock_s": round(total, 2),
                           "peak_memory_gb": peak}
        # Row counts are written as integers. A column holding a missing value
        # would otherwise be stored as floating point, and a spreadsheet reading
        # the file under a locale that treats the point as a thousands separator
        # renders 7533617.0 as 75336170, silently inflating every count tenfold.
        for c in ("rows_in", "rows_out"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
        df.to_csv(out_csv, index=False)
        print("\n" + "=" * 74)
        print(f" COMPUTATIONAL PERFORMANCE -- {self.label}")
        print("=" * 74)
        print(df.to_string(index=False))
        return df


def environment_report(path: str) -> str:
    """Collect library versions and machine specification for the Methods."""
    try:
        import duckdb
        duckdb_version = duckdb.__version__
    except ImportError:
        duckdb_version = "not installed"

    lines = [
        "ENVIRONMENT",
        "-" * 60,
        f"platform            : {platform.platform()}",
        f"python              : {sys.version.split()[0]}",
        f"pandas              : {pd.__version__}",
        f"numpy               : {np.__version__}",
        f"duckdb              : {duckdb_version}",
        f"psutil              : {psutil.__version__ if psutil else 'not installed'}",
        f"processor           : {platform.processor() or 'unknown'}",
    ]
    if psutil:
        lines += [
            f"physical cores      : {psutil.cpu_count(logical=False)}",
            f"logical cores       : {psutil.cpu_count(logical=True)}",
            f"total RAM (GB)      : {psutil.virtual_memory().total / 1024 ** 3:.1f}",
        ]
    text = "\n".join(lines)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    return text


def warn_if_no_psutil() -> None:
    if psutil is None:
        print("\n[WARNING] psutil is not installed; memory will not be recorded.")


# =============================================================================
# SHARED ELIGIBILITY RULE
# =============================================================================
#
# The person-level age exclusion is defined here once and imported by every
# script that applies it, so that the visit-level, individual-level, and
# pregnancy-level datasets cannot drift apart.
#
# Rule: a woman is eligible if she has at least one recorded age and every
# recorded age falls within [age_min, age_max]. A woman with any implausible
# age is removed together with all of her encounters and episodes, rather
# than only the offending encounter, because a per-visit filter would leave
# fragments of the same woman's history in the data.
#
# NULL handling: SQL MIN and MAX ignore NULL, so a woman with no recorded age
# has amin = amax = NULL, and `NULL >= age_min` is NULL rather than TRUE.
# Such women are therefore excluded, and counted separately in the report.

AGE_MIN_DEFAULT = 12
AGE_MAX_DEFAULT = 55


def create_eligibility(con, src_table: str = "src", id_col: str = "PSTV01",
                       age_col: str = "age",
                       age_min: int = AGE_MIN_DEFAULT,
                       age_max: int = AGE_MAX_DEFAULT,
                       extra_aggs: dict[str, str] | None = None,
                       verbose: bool = True) -> tuple[int, int]:
    """Create the tables `per_ind` (one row per woman) and `elig`.

    `extra_aggs` maps an output alias to an aggregate expression, allowing a
    caller to attach additional per-woman summaries without duplicating the
    eligibility logic. Returns (n_individuals, n_eligible).
    """
    extra = ""
    if extra_aggs:
        extra = ",\n            " + ",\n            ".join(
            f"{expr} AS {alias}" for alias, expr in extra_aggs.items())

    con.execute(f"""
        CREATE OR REPLACE TABLE per_ind AS
        SELECT
            {id_col} AS PSTV01,
            MIN(TRY_CAST({age_col} AS DOUBLE)) AS amin,
            MAX(TRY_CAST({age_col} AS DOUBLE)) AS amax,
            COUNT(*) AS n_visits,
            COUNT(TRY_CAST({age_col} AS DOUBLE)) AS n_visits_with_age{extra}
        FROM {src_table}
        WHERE {id_col} IS NOT NULL
        GROUP BY {id_col};
    """)
    con.execute(f"""
        CREATE OR REPLACE TABLE elig AS
        SELECT PSTV01 FROM per_ind
        WHERE amin >= {age_min} AND amax <= {age_max};
    """)
    n_ind = con.sql("SELECT COUNT(*) FROM per_ind").fetchone()[0]
    n_ok = con.sql("SELECT COUNT(*) FROM elig").fetchone()[0]
    if verbose:
        print(f"    individuals: {n_ind:,}   eligible ({age_min}-{age_max} y): {n_ok:,}")
    return n_ind, n_ok


def exclusion_counts(con, age_min: int = AGE_MIN_DEFAULT,
                     age_max: int = AGE_MAX_DEFAULT) -> dict:
    """Break down the person-level exclusion by reason."""
    row = con.sql(f"""
        SELECT
            (SELECT COUNT(*) FROM per_ind) AS individuals_total,
            (SELECT COUNT(*) FROM per_ind WHERE n_visits_with_age = 0) AS no_age,
            (SELECT COUNT(*) FROM per_ind WHERE amin < {age_min}) AS below_min,
            (SELECT COUNT(*) FROM per_ind
             WHERE amax > {age_max} AND NOT (amin < {age_min})) AS above_max,
            (SELECT COUNT(*) FROM elig) AS individuals_kept
    """).fetchone()
    total, no_age, below, above, kept = row
    return {"individuals_total": total, "dropped_no_recorded_age": no_age,
            f"dropped_age_below_{age_min}": below,
            f"dropped_age_above_{age_max}": above,
            "dropped_total": total - kept, "individuals_kept": kept}


# =============================================================================
# EPISODE ANCHORING
# =============================================================================
#
# Defined here once and used by both the main build (step 1) and the
# sensitivity driver (step 9), so that the 180-day column of the sensitivity
# analysis is guaranteed to reproduce the main analysis rather than merely
# resemble it.
#
# The separation interval is measured against the IMMEDIATELY PRECEDING
# termination record, not against the last accepted anchor. A chain of records
# in which every consecutive pair is closer than `gap` days is therefore
# absorbed into a single episode however long the total span. This is the
# behaviour of the original notebook and is preserved deliberately; the
# consequence is quantified by the interval-distribution analysis.

def anchor_from_masks(df, gap, abortus_mask, partus_mask, preterm_mask,
                     postterm_mask, days_abort, days_preterm, days_postterm,
                     days_term, id_col="PSTV01", date_col="combined_date",
                     verbose=True):
    """Anchor episodes and back-calculate onset from precomputed masks.

    This is the rule itself, separated from the code matching that produces the
    masks. The two data sources store diagnosis codes differently -- the
    development cohort holds three-character categories matched exactly, the
    external cohort holds full codes across two coding systems and needs prefix
    matching -- so the matching cannot be shared. The rule can, and is: the
    separation interval, the chained comparison against the preceding record,
    the four gestational offsets, and the episode numbering are identical for
    both cohorts because both call this function.

    Adds or overwrites: doa, dol, dopt, fin_g, ref_start, n_preg.
    Returns (df, diagnostics dict).
    """
    import pandas as _pd

    df["doa"] = df[date_col].where(abortus_mask)
    df["dol"] = df[date_col].where(partus_mask)
    df["dopt"] = df["doa"].fillna(df["dol"])

    df = df.sort_values([id_col, "dopt"])
    order = df.index
    abortus_mask = abortus_mask.reindex(order)
    partus_mask = partus_mask.reindex(order)
    preterm_mask = preterm_mask.reindex(order)
    postterm_mask = postterm_mask.reindex(order)

    prev = df.groupby(id_col, dropna=False)["dopt"].shift()
    gap_days = (df["dopt"] - prev).dt.days
    is_anchor = df["dopt"].notna() & (prev.isna() | (gap_days >= gap))
    df["fin_g"] = df["dopt"].where(is_anchor)

    at = df["fin_g"].notna()
    df["ref_start"] = _pd.NaT
    idx = {
        "abortive":  at & abortus_mask,
        "preterm":   at & partus_mask & ~abortus_mask & preterm_mask,
        "post-term": (at & partus_mask & ~abortus_mask & ~preterm_mask
                      & postterm_mask),
        "term":      (at & partus_mask & ~abortus_mask & ~preterm_mask
                      & ~postterm_mask),
    }
    days = {"abortive": days_abort, "preterm": days_preterm,
            "post-term": days_postterm, "term": days_term}
    onset = {}
    for name, sel in idx.items():
        df.loc[sel, "ref_start"] = (df.loc[sel, "fin_g"]
                                    - _pd.to_timedelta(days[name], unit="D"))
        onset[name] = int(sel.sum())

    df["n_preg"] = (is_anchor.astype(int)
                    .groupby(df[id_col]).cumsum()
                    .where(df["dopt"].notna())).astype("Int64")

    diag = {
        "gap_days": gap,
        "records_with_abortive_code": int(abortus_mask.sum()),
        "records_with_delivery_code": int(partus_mask.sum()),
        "records_with_dopt": int(df["dopt"].notna().sum()),
        "episode_anchors": int(is_anchor.sum()),
        "max_n_preg": int(df["n_preg"].max()) if is_anchor.any() else 0,
        **{f"onset_{k}": v for k, v in onset.items()},
    }
    if verbose:
        print(f"      anchors at gap {gap:>3} d : {diag['episode_anchors']:>10,}"
              f"   max order {diag['max_n_preg']}")
        for k, v in onset.items():
            print(f"        onset -{days[k]:>3} d  {k:<10} {v:>10,}")
    return df, diag


def anchor_episodes(df, gap, dx_cols, abort_codes, partus_codes,
                    preterm_codes, postterm_codes,
                    days_abort, days_preterm, days_postterm, days_term,
                    id_col="PSTV01", verbose=True):
    """Anchoring for the development cohort, which stores three-character
    diagnosis categories and is therefore matched exactly.

    A thin wrapper over `anchor_from_masks`, which holds the rule.
    """
    import pandas as _pd

    if "combined_date" in df.columns and df["combined_date"].notna().any():
        df["combined_date"] = _pd.to_datetime(df["combined_date"],
                                              errors="coerce")
    else:
        for c in ("FKP03", "FKL03"):
            if c in df.columns:
                df[c] = _pd.to_datetime(df[c], errors="coerce")
        have = [c for c in ("FKP03", "FKL03") if c in df.columns]
        if not have:
            raise KeyError("neither combined_date nor FKP03/FKL03 present")
        df["combined_date"] = df[have].min(axis=1)

    return anchor_from_masks(
        df, gap,
        abortus_mask=df[dx_cols].isin(abort_codes).any(axis=1),
        partus_mask=df[dx_cols].isin(partus_codes).any(axis=1),
        preterm_mask=df[dx_cols].isin(preterm_codes).any(axis=1),
        postterm_mask=df[dx_cols].isin(postterm_codes).any(axis=1),
        days_abort=days_abort, days_preterm=days_preterm,
        days_postterm=days_postterm, days_term=days_term,
        id_col=id_col, date_col="combined_date", verbose=verbose)