"""
=============================================================================
 STEP 9 -- CRUDE ODDS RATIOS, VISIT AND INDIVIDUAL LEVELS
=============================================================================

Computes the same association as step 8, between an abortive outcome and each
exposure, but at the two levels of aggregation that precede episode
reconstruction. Together with step 8 this produces the three-level comparison
reported in the manuscript.

    visit level       unit = encounter. Abortive code and exposure code
                      occurring on the SAME encounter.
    individual level  unit = woman. Ever-abortive against ever-exposed,
                      taking the maximum of each flag across her encounters.

The abortive outcome is not used as an exposure, so it has no row here. At the
pregnancy level it does, because there the before- and after-pregnancy phases
refer to a different episode than the outcome.

WHY THE CODE DICTIONARIES ARE IMPORTED
--------------------------------------
The earlier version of this analysis carried its own copy of every ICD code
list. The two copies happened to agree, but a cross-level comparison is only
interpretable if all three levels use identical exposure definitions, and
nothing enforced that. The definitions are now imported from step 3, so the
three levels cannot drift apart. Step 3 defines twelve conditions by regular
expression rather than by code list, and those are evaluated here with
DuckDB's `regexp_matches`, which performs the same unanchored search as the
pandas `str.contains` call used there.

The imported dictionaries are printed with their sizes on every run, so a
change in step 3 is visible here.

NOTES ON THIS REVISION
----------------------
  1. A `code` column is written alongside every label. The forest-plot step
     matched rows by display label, which fails silently when a label differs
     between scripts; matching by code removes that failure mode.
  2. `or_long_visit_individual.csv` is written in the long format the
     forest-plot step reads directly, so no formatted string has to be parsed
     back out of a spreadsheet cell.
  3. Two labels corrected: `hemorrh` maps to K64, haemorrhoids, and was
     labelled 'Gastrointestinal hemorrhage', which is K92; `dom` is defined
     against the Java-Bali province set, so its label now says Java-Bali.
  4. The estimability breakdown is reported for both levels, matching step 8.

INPUT
-----
    pregnancy by visit cleaned.csv       from step 6

OUTPUT
------
    or_visit_individual.xlsx             one sheet, both levels side by side
    or_long_visit_individual.csv         tidy long format for the forest plots
    or_vi_summary_counts.csv             estimability breakdown per level
    Supplementary_table_OR_by_level.xlsx copy-ready table: baseline
                                         characteristics and all conditions,
                                         visit / individual / pregnancy
                                         (before, during, after); needs step 8
    benchmark_step9.csv                 per-stage timing and memory
    environment_step9.txt               library versions and machine specification

USAGE
-----
    python 09_odds_ratios_visit_individual.py
    python 09_odds_ratios_visit_individual.py --report-only
        rebuild only the supplementary table from the existing CSVs
=============================================================================
"""

from __future__ import annotations

import importlib
import math
import os
import sys
from pathlib import Path

import duckdb
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

VISIT_FILE = Path(r"./output_step6/pregnancy by visit cleaned.csv")
OUT_DIR    = Path(r"./output_step8")          # same folder as step 8
OUT_EXCEL  = "or_visit_individual.xlsx"
OUT_LONG   = "or_long_visit_individual.csv"
SHEET_NAME = "visit_individual_OR"

STEP3_MODULE = "03_build_pregnancy_level"     # source of the code definitions

DX_COLS = ["FKP14A", "FKL15A", "FKL17A", "FKL24A"]
ABORT_CODES = ["O00", "O01", "O02", "O03", "O04", "O05", "O06", "O07"]

# Exposures that are episode attributes rather than diagnoses.
TRIO_ORDER = [
    ("age_risk", "Maternal age risk (<20 or >35 y)"),
    ("dom",      "Residence outside Java-Bali islands"),
    ("subsid",   "Subsidised"),
]

SPARSE_MIN_CELL = 5
NOT_ESTIMABLE = "\u2014"

# Labels are corrected relative to the earlier script; see note 3 above.
LABEL_OVERRIDES = {
    "hemorrh": "Haemorrhoids",
}

# =============================================================================


def import_definitions():
    """Pull the condition dictionaries from step 3 so they exist in one place."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        m = importlib.import_module(STEP3_MODULE)
    except Exception as exc:
        sys.exit(f"[FATAL] could not import {STEP3_MODULE}.py: {exc}\n"
                 "        It must sit in the same folder as this script.")
    need = ["chronic_conditions", "infectious_conditions",
            "pregnancy_conditions", "conditions_regex"]
    missing = [n for n in need if not hasattr(m, n)]
    if missing:
        sys.exit(f"[FATAL] {STEP3_MODULE}.py does not define {missing}")

    code_lists = {}
    for name in ["chronic_conditions", "infectious_conditions",
                 "pregnancy_conditions"]:
        d = getattr(m, name)
        code_lists.update(d)
        print(f"    {name:<24}: {len(d):>4} conditions")
    regexes = dict(getattr(m, "conditions_regex"))
    print(f"    {'conditions_regex':<24}: {len(regexes):>4} conditions")

    # The abortive outcome is not an exposure.
    code_lists.pop("abortive", None)
    print(f"    total exposures          : "
          f"{len(code_lists) + len(regexes) + len(TRIO_ORDER):>4} "
          f"(including {len(TRIO_ORDER)} baseline characteristics)")
    return code_lists, regexes


def build_labels(code_lists, regexes):
    """Display labels, taken from step 8 where available."""
    labels = dict(TRIO_ORDER)
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        s7 = importlib.import_module("08_odds_ratios")
        labels.update({c: lab for c, lab in
                       s7.CHRONIC_ORDER + s7.INFECTIOUS_ORDER + s7.PREGNANCY_ORDER})
        print("    display labels imported from 08_odds_ratios.py")
    except Exception:
        print("    [NOTE] 08_odds_ratios.py not importable; codes used as labels")
    labels.update(LABEL_OVERRIDES)
    for k in list(code_lists) + list(regexes):
        labels.setdefault(k, k)
    return labels


def in_list_sql(codes: list[str]) -> str:
    lst = ", ".join(f"'{c}'" for c in codes)
    return " OR ".join(f'"{dc}" IN ({lst})' for dc in DX_COLS)


def regex_sql(pattern: str) -> str:
    pat = pattern.replace("'", "''")
    return " OR ".join(
        f"regexp_matches(COALESCE(\"{dc}\", ''), '{pat}')" for dc in DX_COLS)


def or_stats(a, b, c, d) -> dict:
    """Crude odds ratio with a Woolf interval. No continuity correction.

    The odds ratio is undefined when any cell is empty, so the estimate is
    withheld rather than corrected into a number. Cells below SPARSE_MIN_CELL
    are estimated without correction and flagged, matching step 8.
    """
    a, b, c, d = int(a), int(b), int(c), int(d)
    # Any cell below the threshold, as in steps 8 and 13.
    sparse = min(a, b, c, d) < SPARSE_MIN_CELL
    if min(a, b, c, d) == 0:
        return {"odds_ratio": None, "ci_lower": None, "ci_upper": None,
                "text": NOT_ESTIMABLE, "significance": NOT_ESTIMABLE,
                "sparse": "Yes" if sparse else "No",
                "estimability": "not estimable (empty cell)",
                "a": a, "b": b, "c": c, "d": d}
    orr = (a * d) / (b * c)
    se = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    lo = math.exp(math.log(orr) - 1.96 * se)
    hi = math.exp(math.log(orr) + 1.96 * se)
    return {"odds_ratio": round(orr, 3), "ci_lower": round(lo, 3),
            "ci_upper": round(hi, 3),
            "text": f"{orr:.3f} ({lo:.3f}-{hi:.3f})" + ("+" if sparse else ""),
            "significance": "Significant" if (lo > 1 or hi < 1)
                            else "Not Significant",
            "sparse": "Yes" if sparse else "No",
            "estimability": "sparse" if sparse else "estimable",
            "a": a, "b": b, "c": c, "d": d}


# =============================================================================
# SUPPLEMENTARY TABLE: ODDS RATIOS ACROSS THE THREE DATASET LEVELS
# =============================================================================
# Combines this step's visit- and individual-level ORs with the pregnancy-level
# ORs from step 8 (or_long.csv, same folder) into one copy-ready table:
# baseline characteristics first, then every diagnostic condition in the order
# it is defined in step 3 (by ICD-10 chapter and code), never alphabetical.
# Rebuild without recomputing:   python 09_odds_ratios_visit_individual.py --report-only

PREG_LONG = "or_long.csv"                     # step 8 output, in OUT_DIR
FLAG      = "\u1d43"                         # superscript a, IJE footnote style
OUT_SUPP  = "Supplementary_table_OR_by_level.xlsx"

CHAR_LABEL = {"age_risk": "Maternal age <20 or >35 years (vs 20\u201335)",
              "dom": "Residence outside Java\u2013Bali (vs within)",
              "subsid": "Subsidised membership (vs non-subsidised)"}
GROUP_ORDER = [("chronic", "Chronic conditions"),
               ("infectious", "Infectious diseases"),
               ("pregnancy", "Pregnancy-related conditions")]

# The condition labels use US spelling; IJE uses UK English.
UK_SPELLING = {"Hemorrhage": "Haemorrhage", "hemorrhage": "haemorrhage",
               "Hemorrhagic": "Haemorrhagic", "hemorrhagic": "haemorrhagic",
               "Edema": "Oedema", "edema": "oedema", "Anemia": "Anaemia",
               "anemia": "anaemia", "Subsidized": "Subsidised",
               "subsidized": "subsidised", "Esophag": "Oesophag",
               "esophag": "oesophag", "Hematolog": "Haematolog",
               "hematolog": "haematolog", "Leukemia": "Leukaemia",
               "leukemia": "leukaemia", "Diarrhea": "Diarrhoea",
               "diarrhea": "diarrhoea", "Tumor": "Tumour", "tumor": "tumour"}


def uk(text: str) -> str:
    for us, gb in UK_SPELLING.items():
        text = text.replace(us, gb)
    return text


def _cell(row) -> tuple[str, bool]:
    """Formatted OR (95% CI) and whether it is flagged (any cell below five)."""
    if row is None or row["estimability"] not in ("estimable", "sparse") \
            or pd.isna(row["odds_ratio"]):
        return "\u2013", False
    return (f"{row['odds_ratio']:.2f} ({row['ci_lower']:.2f}\u2013"
            f"{row['ci_upper']:.2f})", row["estimability"] == "sparse")


def build_level_table(vi: pd.DataFrame, pl: pd.DataFrame) -> list[dict]:
    """Rows of the combined table; each value is (text, flagged)."""
    vlook = {(r["level"], r["code"]): r for _, r in vi.iterrows()}
    plook = {(r["sheet"], r["code"]): r for _, r in pl.iterrows()}
    rows = [{"kind": "header", "label": "Baseline characteristics"}]
    for _, r in pl[pl["sheet"] == "characteristics"].iterrows():
        rows.append({"kind": "char", "label": CHAR_LABEL.get(r["code"], uk(r["condition"])),
                     "visit": _cell(vlook.get(("visit", r["code"]))),
                     "individual": _cell(vlook.get(("individual", r["code"]))),
                     "pregnancy": _cell(r)})
    for g, title in GROUP_ORDER:
        rows.append({"kind": "header", "label": title})
        for _, r in pl[pl["sheet"] == f"b_{g}"].iterrows():      # definition order
            k = r["code"]
            rows.append({"kind": "dx", "label": uk(str(r["condition"])),
                         "visit": _cell(vlook.get(("visit", k))),
                         "individual": _cell(vlook.get(("individual", k))),
                         "before": _cell(plook.get((f"b_{g}", k))),
                         "during": _cell(plook.get((f"c_{g}", k))),
                         "after": _cell(plook.get((f"a_{g}", k)))})
    return rows


SUPP_TITLE = ("Odds ratios for abortive outcome across the visit-, individual- "
              "and pregnancy-level datasets")
SUPP_NOTES = [
    "CI, confidence interval; OR, odds ratio.",
    "Crude ORs with log-Wald 95% CIs. Visit level: one record per encounter, outcome an "
    "abortive code at that encounter. Individual level: one record per woman, outcome any "
    "abortive code. Pregnancy level: one record per episode, outcome an abortive "
    "termination, with each condition classified before, during or after the episode; "
    "baseline characteristics are episode-level and are not classified by period.",
    "\u1d43 Any cell of the 2x2 table below five observations; interpret with caution. "
    "\u2013: not estimable "
    "(empty cell), not applicable (chronic conditions during pregnancy; the abortive "
    "outcome itself at the visit and individual levels) or not estimable by design (after "
    "quality control, conditions incompatible with an abortive outcome cannot co-occur with "
    "it during pregnancy).",
    "Conditions are listed in the order of Supplementary Table S2.",
]


def write_level_table(out_dir: Path) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, Side
    vi = pd.read_csv(out_dir / OUT_LONG)
    p_path = out_dir / PREG_LONG
    if not p_path.exists():
        print(f"    [skip] {PREG_LONG} not found in {out_dir}; run step 8 first")
        return
    pl = pd.read_csv(p_path)
    rows = build_level_table(vi, pl)

    thin = Side(style="thin", color="000000")
    f = Font(name="Arial", size=10)
    fb = Font(name="Arial", size=10, bold=True)
    wb = Workbook(); ws = wb.active; ws.title = "OR by level"
    ws.cell(row=1, column=1, value=SUPP_TITLE).font = fb
    # two header rows: level, then period for the pregnancy level
    h1 = ["Characteristic or condition", "Visit level, OR (95% CI)",
          "Individual level, OR (95% CI)", "Pregnancy level, OR (95% CI)", "", ""]
    h2 = ["", "", "", "Before", "During", "After"]
    for j, v in enumerate(h1, 1):
        c = ws.cell(row=3, column=j, value=v or None); c.font = fb
        c.alignment = Alignment(wrap_text=True, vertical="bottom")
        c.border = Border(top=thin)
    ws.merge_cells(start_row=3, start_column=4, end_row=3, end_column=6)
    for j, v in enumerate(h2, 1):
        c = ws.cell(row=4, column=j, value=v or None); c.font = fb
        c.border = Border(bottom=thin)
    for j in (4, 5, 6):
        ws.cell(row=4, column=j).border = Border(top=thin, bottom=thin)
        ws.cell(row=4, column=j).alignment = Alignment(horizontal="center")
    ws.cell(row=3, column=4).alignment = Alignment(horizontal="center")

    i = 5
    for r in rows:
        if r["kind"] == "header":
            ws.cell(row=i, column=1, value=r["label"]).font = fb
            i += 1
            continue
        ws.cell(row=i, column=1, value="    " + r["label"]).font = f
        cells = [("visit", 2), ("individual", 3)]
        cells += [("pregnancy", 4)] if r["kind"] == "char" else \
                 [("before", 4), ("during", 5), ("after", 6)]
        for key, col in cells:
            text, flagged = r[key]
            c = ws.cell(row=i, column=col, value=text + FLAG if flagged else text)
            c.font = f
            c.alignment = Alignment(horizontal="center")
        if r["kind"] == "char":
            ws.merge_cells(start_row=i, start_column=4, end_row=i, end_column=6)
        i += 1
    for j in range(1, 7):
        ws.cell(row=i - 1, column=j).border = Border(bottom=thin)
    for k, n in enumerate(SUPP_NOTES):
        ws.cell(row=i + 1 + k, column=1, value=n).font = Font(name="Arial", size=9)
    for col, w in zip("ABCDEF", (52, 20, 20, 18, 18, 18)):
        ws.column_dimensions[col].width = w
    ws.freeze_panes = "B5"
    wb.save(out_dir / OUT_SUPP)
    n_dx = sum(r["kind"] == "dx" for r in rows)
    print(f"    supplementary table: 3 characteristics + {n_dx} conditions "
          f"-> {OUT_SUPP}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 74)
    print(" STEP 9 -- CRUDE ODDS RATIOS, VISIT AND INDIVIDUAL LEVELS")
    print("=" * 74)
    print(environment_report(str(OUT_DIR / "environment_step9.txt")))
    warn_if_no_psutil()

    if not VISIT_FILE.exists():
        sys.exit(f"[FATAL] input not found: {VISIT_FILE.resolve()}")

    bench = Bench("step 9, visit and individual odds ratios")

    with bench.stage("Import condition definitions", "python") as b:
        code_lists, regexes = import_definitions()
        labels = build_labels(code_lists, regexes)
        exposures = ([k for k, _ in TRIO_ORDER]
                     + list(code_lists) + list(regexes))
        b["rows_out"] = len(exposures)

    con = duckdb.connect()

    with bench.stage("Build per-encounter flag table", "DuckDB") as b:
        con.execute(f"""
            CREATE OR REPLACE TABLE raw AS
            SELECT * FROM read_csv_auto('{VISIT_FILE.as_posix()}',
                                        SAMPLE_SIZE=-1)
        """)
        cols = {c.lower(): c for c in con.execute(
            "SELECT name FROM pragma_table_info('raw')").fetchdf()["name"]}
        id_col = cols.get("pstv01") or cols.get("pstv")
        if id_col is None:
            sys.exit("[FATAL] patient identifier column not found.")
        for dc in DX_COLS:
            if dc.lower() not in cols:
                sys.exit(f"[FATAL] diagnosis column '{dc}' not found.")

        flags = [f"CASE WHEN {in_list_sql(ABORT_CODES)} THEN 1 ELSE 0 END AS ab"]
        used = []
        for k, _ in TRIO_ORDER:
            if k.lower() in cols:
                flags.append(f'CASE WHEN "{cols[k.lower()]}"=1 THEN 1 ELSE 0 END '
                             f'AS "e_{k}"')
                used.append(k)
            else:
                print(f"      [WARNING] baseline characteristic '{k}' absent")
        for k, codes in code_lists.items():
            flags.append(f'CASE WHEN {in_list_sql(codes)} THEN 1 ELSE 0 END '
                         f'AS "e_{k}"')
            used.append(k)
        for k, pat in regexes.items():
            flags.append(f'CASE WHEN {regex_sql(pat)} THEN 1 ELSE 0 END '
                         f'AS "e_{k}"')
            used.append(k)

        con.execute(f'CREATE OR REPLACE TABLE v AS SELECT "{id_col}" AS pid, '
                    f'{", ".join(flags)} FROM raw')
        con.execute("DROP TABLE raw")
        n_visits = con.execute("SELECT COUNT(*) FROM v").fetchone()[0]
        n_ab_visits = con.execute("SELECT SUM(ab) FROM v").fetchone()[0]
        print(f"    {n_visits:,} encounters, {int(n_ab_visits or 0):,} carrying "
              f"an abortive code, {len(used)} exposures")
        b["rows_out"] = n_visits

    with bench.stage("Collapse to person level", "DuckDB",
                     rows_in=n_visits) as b:
        maxes = ["MAX(ab) AS ab"] + [f'MAX("e_{k}") AS "e_{k}"' for k in used]
        con.execute(f'CREATE OR REPLACE TABLE ind AS SELECT pid, '
                    f'{", ".join(maxes)} FROM v GROUP BY pid')
        n_persons = con.execute("SELECT COUNT(*) FROM ind").fetchone()[0]
        n_ab_persons = con.execute("SELECT SUM(ab) FROM ind").fetchone()[0]
        print(f"    {n_persons:,} women, {int(n_ab_persons or 0):,} ever with "
              f"an abortive code")
        b["rows_out"] = n_persons

    def two_by_two(table: str) -> pd.Series:
        parts = []
        for k in used:
            e = f'"e_{k}"'
            parts += [
                f'SUM(CASE WHEN ab=1 AND {e}=1 THEN 1 ELSE 0 END) AS "a_{k}"',
                f'SUM(CASE WHEN ab=1 AND {e}=0 THEN 1 ELSE 0 END) AS "b_{k}"',
                f'SUM(CASE WHEN ab=0 AND {e}=1 THEN 1 ELSE 0 END) AS "c_{k}"',
                f'SUM(CASE WHEN ab=0 AND {e}=0 THEN 1 ELSE 0 END) AS "d_{k}"']
        return con.execute(f"SELECT {', '.join(parts)} FROM {table}"
                           ).fetchdf().iloc[0]

    with bench.stage("Compute odds ratios at both levels", "DuckDB") as b:
        vcell = two_by_two("v")
        icell = two_by_two("ind")

        wide, long = [], []
        for k in used:
            lab = labels.get(k, k)
            v = or_stats(vcell[f"a_{k}"], vcell[f"b_{k}"],
                         vcell[f"c_{k}"], vcell[f"d_{k}"])
            i = or_stats(icell[f"a_{k}"], icell[f"b_{k}"],
                         icell[f"c_{k}"], icell[f"d_{k}"])
            wide.append({
                "code": k, "condition": lab,
                "Visit OR (95% CI)": v["text"],
                "Visit significance": v["significance"],
                "Visit sparse": v["sparse"],
                "Visit estimability": v["estimability"],
                "Visit cell_a (abortive & condition present)": v["a"],
                "Visit cell_b (abortive & condition absent)": v["b"],
                "Visit cell_c (non-abortive & condition present)": v["c"],
                "Visit cell_d (non-abortive & condition absent)": v["d"],
                "Individual OR (95% CI)": i["text"],
                "Individual significance": i["significance"],
                "Individual sparse": i["sparse"],
                "Individual estimability": i["estimability"],
                "Individual cell_a (abortive & condition present)": i["a"],
                "Individual cell_b (abortive & condition absent)": i["b"],
                "Individual cell_c (non-abortive & condition present)": i["c"],
                "Individual cell_d (non-abortive & condition absent)": i["d"],
            })
            for level, st in (("visit", v), ("individual", i)):
                long.append({"level": level, "code": k, "condition": lab,
                             **{kk: st[kk] for kk in
                                ("odds_ratio", "ci_lower", "ci_upper",
                                 "significance", "sparse",
                                 "estimability", "a", "b", "c", "d")}})

        wide_df = pd.DataFrame(wide)
        long_df = pd.DataFrame(long)
        b["rows_out"] = len(long_df)

    with bench.stage("Export", "pandas", rows_in=len(long_df)) as b:
        long_df.to_csv(OUT_DIR / OUT_LONG, index=False)
        with pd.ExcelWriter(OUT_DIR / OUT_EXCEL, engine="openpyxl") as w:
            wide_df.to_excel(w, sheet_name=SHEET_NAME, index=False)
            ws = w.sheets[SHEET_NAME]
            for cc in ws.columns:
                width = max(len(str(c.value)) if c.value else 0 for c in cc)
                ws.column_dimensions[cc[0].column_letter].width = min(width + 2, 48)

        counts = (long_df.groupby(["level", "estimability"]).size()
                  .rename("n").reset_index())
        counts.to_csv(OUT_DIR / "or_vi_summary_counts.csv", index=False)

        print("\n    ESTIMABILITY")
        for level in ("visit", "individual"):
            print(f"      {level}")
            for _, r in counts[counts["level"] == level].iterrows():
                print(f"        {r['estimability']:<32}: {int(r['n']):>6,}")
        print(f"\n    wrote {OUT_LONG} and {OUT_EXCEL}")
        b["rows_out"] = len(long_df)

    con.close()
    write_level_table(OUT_DIR)
    bench.finalise(str(OUT_DIR / "benchmark_step9.csv"), rows_out=len(long_df))
    print(f"\nOutput written to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    if "--report-only" in sys.argv:
        write_level_table(OUT_DIR)
    else:
        main()