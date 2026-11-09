"""
=============================================================================
 STEP 8 -- CRUDE ODDS RATIOS, PREGNANCY LEVEL
=============================================================================

Computes the crude association between a current abortive outcome and every
diagnostic feature flag, and between the abortive outcome and the three
baseline characteristics, from 2 x 2 contingency tables.

    cell a   abortive outcome present, condition present
    cell b   abortive outcome present, condition absent
    cell c   abortive outcome absent,  condition present
    cell d   abortive outcome absent,  condition absent

The unadjusted odds ratio is a*d / (b*c), with Woolf logit confidence
intervals. No continuity correction is applied. The odds ratio is undefined
when any cell is empty, so those associations are reported as not estimable
rather than corrected into a number that would read as a real effect.

Associations in which cell a or cell c holds fewer than SPARSE_MIN_CELL
observations are computed without correction and flagged as sparse. The
flag is a warning about interval width, not a change to the estimate. Flagged
rows are excluded from the forest plots and from the cross-system concordance
assessment in step 13.

WHICH INPUT FILE
----------------
The default input is the POST-quality-control file from step 4. This matters
more here than anywhere else in the pipeline, and it has a consequence that
must be stated in the manuscript.

The quality-control rule removes every episode in which a current abortive
outcome co-occurs with one of the conditions listed in QC_CONFLICT_COLS.
In the resulting dataset, cell a is therefore ZERO BY CONSTRUCTION for the
current-phase association between the abortive outcome and each of those
conditions. Their current-phase odds ratios are not estimable because the
quality-control step defines them away, not because the data are sparse.

This script detects that situation and reports those cells as not estimable
by design, separately from cells that are empty for want of data. The
affected cells are listed in
`or_not_estimable_by_design.csv` so that the corresponding blanks in the
supplementary tables can be justified from code rather than filled in by hand.

Running this on the pre-QC file from step 3 instead would produce numbers in
those cells, but they would describe co-occurrences that the analytic cohort
does not contain.

NOTES ON THIS REVISION
----------------------
  1. Two labels were corrected. `hemorrh` maps to K64, haemorrhoids, and was
     previously labelled 'Gastrointestinal hemorrhage', which is K92. `dom` is
     defined against the Java-Bali province set, so its label now says
     Java-Bali rather than Java.
  2. `c_prev` was listed among the conflict columns but no condition
     dictionary defines it, so it never existed as a column; it has been
     removed from the list here and in step 4. Absent columns are listed
     rather than silently skipped.
  3. A tidy long-format CSV is written alongside the workbook. The forest-plot
     step reads that file rather than parsing formatted strings out of Excel
     cells, which is fragile where rich text is used for the sparse marker.
  4. Counts are reported: total associations tested, estimable, sparse, and
     not estimable by design, split into empty-cell and by-design causes.
     These are the denominators the manuscript needs, and they come from the
     code rather than from counting cells in the supplementary tables.
  5. No multiplicity correction is applied, matching the original analysis. The
     number of tests is reported so that this can be stated explicitly rather
     than left for a reviewer to count.

INPUT
-----
    pregnancy by episode cleaned.csv     from step 4

OUTPUT
------
    OR by episode.xlsx                   one sheet per category and phase
    or_long.csv                          tidy long format, one row per association
    or_summary_counts.csv                estimability breakdown
    or_not_estimable_by_design.csv       cells defined away by quality control
    or_columns_absent.csv                expected columns not present in the input
    benchmark_step8.csv                  per-stage timing and memory
    environment_step8.txt                library versions and machine specification

USAGE
-----
    python 08_odds_ratios.py
=============================================================================
"""

from __future__ import annotations

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

INPUT_FILE  = Path(r"./output_step4/pregnancy by episode cleaned.csv")
OUT_DIR     = Path(r"./output_step8")
OUTPUT_FILE = "OR by episode.xlsx"

OUTCOME_COL = "c_abortive"

# Sparse if either cell involving the condition falls below this count.
SPARSE_MIN_CELL = 5

# Conditions removed by the step 4 quality-control rule when they co-occur with
# a current abortive outcome. Their current-phase association is not estimable
# in the cleaned dataset by construction. Must match CONFLICT_COLS in step 4.
QC_CONFLICT_COLS = [
    'c_preecl', 'c_ecl', 'c_anh', 'c_previa', 'c_abrupt',
    'c_polyhydra', 'c_abnamnio', 'c_prom', 'c_prolong', 'c_preterm',
    'c_fail', 'c_abnforce', 'c_long', 'c_malpres', 'c_obspelvic', 'c_iph',
    'c_distress', 'c_umbilical', 'c_laceration', 'c_obstrau', 'c_pph',
    'c_retained', 'c_normal', 'c_instrum', 'c_caesar', 'c_assisted',
    'c_multiple', 'c_disprop', 'c_malpresent', 'c_abnorpelv', 'c_placental'
]

NOT_ESTIMABLE = "\u2014"          # em dash

# =============================================================================
# ROW ORDER AND LABELS  ---  follows Supplementary Table 8
# =============================================================================
TRIO_ORDER = [  
    ("age_risk", "Maternal age risk (<20 or >35 y)"),
    ("dom",      "Residence outside Java-Bali islands"),
    ("subsid",   "Subsidized"),
]

CHRONIC_ORDER = [
    ('dm',          'Diabetes mellitus'),
    ('malnut',      'Malnutrition'),
    ('nutri',       'Nutritional deficiencies'),
    ('obese',       'Obesity'),
    ('substance',   'Substance abuse'),
    ('schizo',      'Schizophrenia'),
    ('neurot',      'Neurotic disorders'),
    ('neu_deg',     'Degenerative diseases of nervous systems'),
    ('headache',    'Headache'),
    ('neuropathy',  'Neuropathy'),
    ('rhd',         'Rheumatic heart disease'),
    ('ht',          'Hypertensive disorders'),
    ('angin',      'Angina pectoris'),
    ('myocard',     'Myocardial infarction'),
    ('phd',         'Pulmonary heart disease'),
    ('carditis',    'Carditis and inflammatory cardiac conditions'),
    ('cmp',         'Cardiomyopathies'),
    ('arrythmia',   'Arrhythmias'),
    ('hf',          'Heart failure'),
    ('stroke',      'Stroke'),
    ('artery',      'Arterial diseases'),
    ('vein',        'Venous diseases'),
    ('chronic_res', 'Chronic respiratory diseases'),
    ('pul_edema',   'Pulmonary edema'),
    ('pleura',      'Pleural diseases'),
    ('oral',        'Oral diseases'),
    ('gastritis',   'Gastritis & duodenitis'),
    ('hernia',      'Hernias'),
    ('intestinal',  'Intestinal disorders'),
    ('hemorrh',     'Haemorrhoids'),   # K64. Previously labelled
                                   # 'Gastrointestinal hemorrhage', which is K92;
                                   # the data are present, the ST8 row was missing.
    ('periton',     'Peritonitis'),
    ('liver_fail',  'Liver failure'),
    ('liver',       'Chronic liver diseases'),
    ('gallbladder', 'Gallbladder/biliary diseases'),
    ('pancreas',    'Pancreatic disorders'),
    ('bullous',     'Bullous disorders'),
    ('atopic',      'Atopic dermatitis'),
    ('dermatitis',  'Dermatitis'),
    ('urticaria',   'Urticaria'),
    ('urolith',     'Urolithiasis'),
    ('endomet',     'Endometriosis'),
    ('femgen',      'Female genital tract disorders'),
    ('hypomen',     'Hypomenorrhea / Amenorrhea'),
    ('menorrh',     'Menorrhagia / Abnormal bleeding'),
    ('dysmen',      'Dysmenorrhea'),
    ('arthropathy', 'Arthropathies'),
    ('sysconn',     'Systemic connective tissue disorders'),
    ('dorsopathy',  'Dorsopathies'),
    ('muscle_dis',  'Muscle disorders'),
    ('synov_dis',   'Synovial/tendon disorders'),
    ('soft_dis',    'Bone & soft-tissue disorders'),
    ('renal_dis',   'Renal diseases'),
    ('renal_fail',  'Renal failure'),
    ('breast_dis',  'Breast disorders'),
    ('pid',         'Pelvic inflammatory disease'),
    ('poison',      'Poisoning'),
    ('toxic',       'Toxic effects'),
]

INFECTIOUS_ORDER = [
    ('typhoid',   'Typhoid fever'),
    ('cholera',   'Cholera'),
    ('v_age',     'Viral gastroenteritis'),
    ('b_age',     'Bacterial enteric infections'),
    ('p_age',     'Parasitic enteric infections'),
    ('tb',        'Tuberculosis'),
    ('myco',      'Mycobacterial infections'),
    ('lepto',     'Leptospirosis'),
    ('std',       'Sexually transmitted infections'),
    ('torch',     'TORCH infections'),
    ('v_skin',    'Viral skin infections'),
    ('hepatitis', 'Hepatitis'),
    ('hiv',       'HIV'),
    ('sepsis',    'Sepsis'),
    ('infla_cns', 'Inflammatory of the central nervous system'),
    ('urti',      'Upper respiratory infections'),
    ('lrti',      'Lower respiratory infections'),
    ('uti',       'Urinary tract infections'),
]

PREGNANCY_ORDER = [
    ('abortive',   'Abortive pregnancy'),
    ('preecl',     'Pre-eclampsia'),
    ('ecl',        'Eclampsia'),
    ('earlyhemo',  'Hemorrhage in early pregnancy'),
    ('heg',        'Excessive vomiting in pregnancy'),
    ('venpreg',    'Venous complications and haemorrhoids in pregnancy'),
    ('utipreg',    'Infections of genitourinary tract in pregnancy'),
    ('malpreg',    'Malnutrition in pregnancy'),
    ('multigest',  'Multiple gestation'),
    ('malpresent', 'Known or suspected malpresentation of fetus'),
    ('disprop',    'Known or suspected disproportion'),
    ('abnorpelv',  'Known or suspected abnormality of pelvic organs'),
    ('fetalprob',  'Known or suspected fetal abnormality or problems'),
    ('polyhydra',  'Polyhydramnios'),
    ('abnamnio',   'Other disorders of amniotic fluid and membranes'),
    ('prom',       'Premature rupture of membranes'),
    ('placental',  'Placental disorders'),
    ('previa',     'Placenta previa'),
    ('abrupt',     'Placental abruption'),
    ('anh',        'Antenatal hemorrhage'),
    ('prolong',    'Prolonged pregnancy'),
    ('preterm',    'Preterm childbirth'),
    ('fail',       'Failed induction of childbirth'),
    ('abnforce',   'Abnormalities of forces of childbirth'),
    ('long',       'Long childbirth'),
    ('malpres',    'Malposition or malpresentation of fetus'),
    ('obspelvic',  'Obstructed childbirth'),
    ('iph',        'Intrapartum hemorrhage'),
    ('distress',   'Fetal distress'),
    ('umbilical',  'Umbilical cord complications complicating childbirth'),
    ('laceration', 'Perineal laceration during delivery'),
    ('obstrau',    'Other obstetric trauma'),
    ('pph',        'Postpartum hemorrhage'),
    ('retained',   'Retained placenta and membranes'),
    ('normal',     'Single spontaneous delivery'),
    ('instrum',    'Single instrumented delivery'),
    ('caesar',     'Single delivery by caesarean section'),
    ('assisted',   'Other assisted delivery'),
    ('multiple',   'Multiple delivery'),
]

# Condition codes per category, in table order
CHRONIC    = [c for c, _ in CHRONIC_ORDER]
INFECTIOUS = [c for c, _ in INFECTIOUS_ORDER]
PREGNANCY  = [c for c, _ in PREGNANCY_ORDER]

LABELS = {c: lab for c, lab in CHRONIC_ORDER + INFECTIOUS_ORDER + PREGNANCY_ORDER}
LABELS.update({c: lab for c, lab in TRIO_ORDER})
ORDER_INDEX = {c: i for i, c in enumerate(CHRONIC + INFECTIOUS + PREGNANCY)}

# Chronic conditions have no current phase by construction; see step 3.
PHASE_MAP = {
    'chronic':    ['b_', 'a_'],
    'infectious': ['b_', 'c_', 'a_'],
    'pregnancy':  ['b_', 'c_', 'a_'],
}

# =============================================================================


def build_union_blocks(columns: list[str]) -> str:
    blocks = []
    for col in columns:
        qcol = f'"{col}"'
        blocks.append(f"""
            SELECT
              '{col}' AS var,
              SUM(CASE WHEN "{OUTCOME_COL}"=1 AND {qcol}=1 THEN 1 ELSE 0 END) AS a,
              SUM(CASE WHEN "{OUTCOME_COL}"=1 AND {qcol}=0 THEN 1 ELSE 0 END) AS b,
              SUM(CASE WHEN "{OUTCOME_COL}"=0 AND {qcol}=1 THEN 1 ELSE 0 END) AS c,
              SUM(CASE WHEN "{OUTCOME_COL}"=0 AND {qcol}=0 THEN 1 ELSE 0 END) AS d
            FROM final
            WHERE "{OUTCOME_COL}" IS NOT NULL AND {qcol} IS NOT NULL
        """)
    return " UNION ALL ".join(blocks)


def calc_or(db, columns: list[str]) -> pd.DataFrame:
    """Crude odds ratios with Woolf confidence intervals, uncorrected.

    The estimate is returned only when every cell is positive. Where any cell
    is empty the odds ratio is undefined and NULL is returned, and `annotate`
    marks the row as not estimable.
    """
    if not columns:
        return pd.DataFrame()

    sql = f"""
    WITH base AS ({build_union_blocks(columns)}),
    stats AS (
      SELECT var, a, b, c, d,
        (a+b) AS total_abortive,
        (c+d) AS total_non_abortive,
        CASE WHEN a>0 AND b>0 AND c>0 AND d>0
             THEN (a*d)::DOUBLE/(b*c) END AS or_raw
      FROM base
    ),
    ci AS (
      SELECT var, a, b, c, d, total_abortive, total_non_abortive,
        or_raw AS odds_ratio,
        CASE WHEN or_raw IS NOT NULL
             THEN EXP(LN(or_raw) - 1.96*SQRT(1.0/a+1.0/b+1.0/c+1.0/d))
        END AS ci_lower,
        CASE WHEN or_raw IS NOT NULL
             THEN EXP(LN(or_raw) + 1.96*SQRT(1.0/a+1.0/b+1.0/c+1.0/d))
        END AS ci_upper
      FROM stats
    )
    SELECT var, a, b, c, d, total_abortive, total_non_abortive,
      ROUND(odds_ratio,3) AS odds_ratio,
      ROUND(ci_lower,3)   AS ci_lower,
      ROUND(ci_upper,3)   AS ci_upper,
      CASE WHEN ci_lower IS NOT NULL AND ci_upper IS NOT NULL
            AND (ci_lower>1 OR ci_upper<1) THEN 'Significant'
           ELSE 'Not Significant' END AS significance,
      CASE WHEN a<{SPARSE_MIN_CELL} OR b<{SPARSE_MIN_CELL} OR c<{SPARSE_MIN_CELL}
                OR d<{SPARSE_MIN_CELL} THEN 'Yes' ELSE 'No' END AS sparse
    FROM ci
    """
    return db.execute(sql).fetchdf()


def annotate(df: pd.DataFrame) -> pd.DataFrame:
    """Attach phase, label, estimability, and the by-design flag."""
    if df.empty:
        return df
    df = df.copy()
    df["phase"] = df["var"].str.extract(r"^([bca])_")[0].fillna("none")
    df["code"] = df["var"].str.replace(r"^[bca]_", "", regex=True)
    df["condition"] = df["code"].map(LABELS).fillna(df["code"])
    df["_rank"] = df["code"].map(ORDER_INDEX).fillna(10 ** 6).astype(int)

    qc = {c.lower() for c in QC_CONFLICT_COLS}
    df["not_estimable_by_design"] = (
        (df["a"] == 0) & df["var"].str.lower().isin(qc))

    def status(r):
        if r["var"].lower() == OUTCOME_COL.lower():
            return "not estimable (outcome versus itself)"
        if r["not_estimable_by_design"]:
            return "not estimable by design"
        if min(r["a"], r["b"], r["c"], r["d"]) == 0:
            return "not estimable (empty cell)"
        if r["sparse"] == "Yes":
            return "sparse"
        return "estimable"

    df["estimability"] = df.apply(status, axis=1)
    return df.sort_values("_rank", kind="stable").reset_index(drop=True)


def format_for_sheet(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    def cell(r):
        if r["estimability"].startswith("not estimable") or pd.isna(
                r["odds_ratio"]):
            return NOT_ESTIMABLE
        return (f"{r['odds_ratio']:.3f} "
                f"({r['ci_lower']:.3f}\u2013{r['ci_upper']:.3f})")

    out = df.copy()
    out["OR (95% CI)"] = out.apply(cell, axis=1)
    out.loc[out["estimability"].str.startswith("not estimable"),
            "significance"] = NOT_ESTIMABLE
    out = out.rename(columns={
        'a': 'cell_a (abortive & condition present)',
        'b': 'cell_b (abortive & condition absent)',
        'c': 'cell_c (non-abortive & condition present)',
        'd': 'cell_d (non-abortive & condition absent)',
    })
    return out[[
        'condition', 'OR (95% CI)', 'significance', 'sparse', 'estimability',
        'cell_a (abortive & condition present)',
        'cell_b (abortive & condition absent)',
        'cell_c (non-abortive & condition present)',
        'cell_d (non-abortive & condition absent)',
    ]]


def mark_sparse_superscript(ws, df: pd.DataFrame) -> None:
    """Superscript '+' on sparse rows, matching the supplementary tables.
    Falls back to a plain '+' where rich text is unavailable."""
    if df.empty or 'sparse' not in df.columns:
        return
    or_col = list(df.columns).index('OR (95% CI)') + 1
    try:
        from openpyxl.cell.rich_text import CellRichText, TextBlock
        from openpyxl.cell.text import InlineFont
        rich_ok = True
    except Exception:
        rich_ok = False

    for i, (sp, est) in enumerate(zip(df['sparse'], df['estimability'])):
        if str(est).startswith("not estimable"):
            continue
        if str(sp).strip().lower() not in ('yes', 'y', '1', 'true'):
            continue
        c = ws.cell(row=i + 2, column=or_col)
        base = str(c.value)
        if rich_ok:
            try:
                c.value = CellRichText([
                    TextBlock(InlineFont(), base),
                    TextBlock(InlineFont(vertAlign='superscript'), '+')])
                continue
            except Exception:
                pass
        c.value = base + '+'


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 74)
    print(" STEP 8 -- CRUDE ODDS RATIOS, PREGNANCY LEVEL")
    print("=" * 74)
    print(environment_report(str(OUT_DIR / "environment_step8.txt")))
    warn_if_no_psutil()

    if not INPUT_FILE.exists():
        sys.exit(f"[FATAL] input not found: {INPUT_FILE.resolve()}")

    bench = Bench("step 8, odds ratios")
    db = duckdb.connect()

    with bench.stage("Load pregnancy-level dataset", "DuckDB") as b:
        db.execute(f"""
            CREATE OR REPLACE TABLE final AS
            SELECT * FROM read_csv_auto('{INPUT_FILE.as_posix()}',
                                        SAMPLE_SIZE=-1)
        """)
        n_ep = db.sql("SELECT COUNT(*) FROM final").fetchone()[0]
        col_names = db.sql("SELECT name FROM pragma_table_info('final')"
                           ).df()["name"].tolist()
        lut = {n.lower(): n for n in col_names}
        if OUTCOME_COL not in lut:
            sys.exit(f"[FATAL] outcome column '{OUTCOME_COL}' not found.")
        n_abort = db.sql(f'SELECT SUM("{lut[OUTCOME_COL]}") FROM final'
                         ).fetchone()[0]
        print(f"    episodes {n_ep:,}   with a current abortive outcome "
              f"{int(n_abort or 0):,}")
        b["rows_out"] = n_ep

    with bench.stage("Resolve sheets and columns", "python") as b:
        sheets: dict[str, list[str]] = {}
        absent: list[str] = []
        for cat, conds in [('chronic', CHRONIC), ('infectious', INFECTIOUS),
                           ('pregnancy', PREGNANCY)]:
            for phase in PHASE_MAP[cat]:
                name = f"{phase.strip('_')}_{cat}"
                cols = []
                for cond in conds:
                    key = f"{phase}{cond}".lower()
                    if key in lut:
                        cols.append(lut[key])
                    else:
                        absent.append(f"{phase}{cond}")
                sheets[name] = cols
                print(f"      {name:<18} {len(cols):>3} columns")
        trio = [lut[c] for c, _ in TRIO_ORDER if c in lut]
        missing_trio = [c for c, _ in TRIO_ORDER if c not in lut]
        if missing_trio:
            print(f"      [WARNING] baseline characteristics absent: {missing_trio}")
        if trio:
            sheets = {"characteristics": trio, **sheets}
            print(f"      {'characteristics':<18} {len(trio):>3} columns")
        b["rows_out"] = sum(len(v) for v in sheets.values())

    long_rows: list[pd.DataFrame] = []
    with bench.stage("Compute odds ratios and write workbook", "DuckDB") as b:
        with pd.ExcelWriter(OUT_DIR / OUTPUT_FILE, engine='openpyxl') as writer:
            for name, cols in sheets.items():
                if not cols:
                    print(f"      skipped {name}: no columns")
                    continue
                res = annotate(calc_or(db, cols))
                res.insert(0, "sheet", name)
                long_rows.append(res)
                sheet = format_for_sheet(res)
                sheet.to_excel(writer, sheet_name=name, index=False)
                ws = writer.sheets[name]
                for col_cells in ws.columns:
                    width = max(len(str(c.value)) if c.value else 0
                                for c in col_cells)
                    ws.column_dimensions[col_cells[0].column_letter].width = \
                        min(width + 2, 50)
                mark_sparse_superscript(ws, sheet)
                print(f"      {name:<18} {len(sheet):>3} rows")
        b["rows_out"] = sum(len(x) for x in long_rows)

    with bench.stage("Summarise estimability", "pandas") as b:
        long = pd.concat(long_rows, ignore_index=True)
        long.drop(columns=["_rank"]).to_csv(OUT_DIR / "or_long.csv", index=False)

        counts = (long["estimability"].value_counts()
                  .rename_axis("estimability").rename("n").reset_index())
        counts.loc[len(counts)] = {"estimability": "TOTAL tested", "n": len(long)}
        counts.to_csv(OUT_DIR / "or_summary_counts.csv", index=False)

        by_design = long.loc[long["not_estimable_by_design"],
                             ["sheet", "var", "condition", "a", "b", "c", "d"]]
        by_design.to_csv(OUT_DIR / "or_not_estimable_by_design.csv", index=False)
        if absent:
            pd.DataFrame({"absent_column": sorted(set(absent))}).to_csv(
                OUT_DIR / "or_columns_absent.csv", index=False)

        print("\n    ESTIMABILITY")
        for _, r in counts.iterrows():
            print(f"      {r['estimability']:<32}: {r['n']:>6,}")
        print(f"\n    No multiplicity correction was applied to "
              f"{len(long):,} tests.")
        if len(by_design):
            print(f"\n    Not estimable by design ({len(by_design)}): the "
                  f"quality-control rule removes these co-occurrences.")
            for _, r in by_design.head(10).iterrows():
                print(f"      {r['var']:<16} {r['condition'][:46]}")
            if len(by_design) > 10:
                print(f"      ... full list in or_not_estimable_by_design.csv")
        if absent:
            print(f"\n    Expected but absent columns ({len(set(absent))}): "
                  f"see or_columns_absent.csv")
        b["rows_out"] = len(long)

    db.close()
    bench.finalise(str(OUT_DIR / "benchmark_step8.csv"), rows_out=len(long))
    print(f"\nOutput written to {OUT_DIR.resolve()}")


if __name__ == '__main__':
    main()