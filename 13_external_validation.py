"""
=============================================================================
 STEP 13 -- EXTERNAL VALIDATION IN THE TMUCRD COHORT
=============================================================================

Applies the reconstruction rules unchanged to a structurally different data
source, and compares the direction of every estimable association against the
development cohort.

The point of this step is not to reproduce the Indonesian estimates. It is to
establish whether the rules transfer: whether the same termination anchors,
the same separation interval, and the same phase definitions yield a coherent
cohort in hospital electronic records from a high-income single-payer system,
having been developed on national insurance claims from a middle-income one.

DEFINITIONS ARE IMPORTED, NOT RESTATED
--------------------------------------
ICD-10 code sets come from step 3. ICD-9 code sets come from
`icd_crosswalk.csv`, which is generated from Supplementary Table 2 by
`make_icd_crosswalk.py`. Nothing is written twice, so the two cohorts cannot
end up comparing different definitions of the same condition — which is the
only way a directional comparison means anything.

TWO CODING SYSTEMS
------------------
The external cohort spans the transition from ICD-9-CM to ICD-10-CM, so both
code sets must be evaluated. They are not interchangeable and cannot be pooled
into one match list, because a small number of codes are valid in both systems
with unrelated meanings: ICD-9 V27x denotes the outcome of delivery, while
ICD-10 V27x denotes a motorcycle collision. Each recorded value is therefore
classified before matching:

    purely numeric, or V/E followed by digits   -> ICD-9
    a letter followed by digits, not V or E     -> ICD-10
    V or E followed by digits                   -> ambiguous; resolved by the
                                                   column, then by the record
                                                   date relative to TRANSITION

The number of flags contributed by each system is reported per condition, and
the counts either side of the transition are reported separately.

PREFIX MATCHING
---------------
The development cohort stores three-character diagnosis codes, so step 3 can
match them exactly. Hospital records store full codes, so a three-character
category has to be matched as a prefix: `O20` must match `O200` and `O20.0`.
Exact matching against full codes would return nothing, and every condition
flag would be zero — which would look like a sparse external cohort rather than
a matching failure. The observed code format is detected and printed, so this
cannot pass unnoticed.

WHY THE SPARSITY REPORT MATTERS
-------------------------------
Most associations are expected to be non-estimable in a cohort three orders of
magnitude smaller. But a condition can also appear absent because its codes
were never matched, or because it only existed under ICD-9 and the pre-
transition period was silently skipped. Those are failures, not findings, and
`external_sparsity_report.csv` separates them: for every condition it records
how many episodes were flagged, from which coding system, and on which side of
the transition.

INPUT
-----
    tmucrd_pregnancy_by_visit.csv        the external visit-level table, built
                                         with the same anchoring rules
    icd_crosswalk.csv                    from make_icd_crosswalk.py
    or_long.csv                          from step 8, the development estimates

OUTPUT
------
    tmucrd_pregnancy_by_episode.csv      external pregnancy-level dataset
    external_or_long.csv                 external odds ratios
    external_concordance.csv             direction agreement, pair by pair
    OR external.xlsx                     the two cohorts side by side, one sheet
                                         per category and phase, formatted for
                                         the supplementary table
    external_concordance_summary.csv     the counts reported in the manuscript
    external_forest.png / .pdf           forest plot of the estimable
                                         associations, both cohorts side by side
    external_sparsity_report.csv         why each condition was or was not estimable
    external_qc_report.csv               episodes removed by the quality-control rule
    external_washout_report.csv          episodes before and after the washout
    external_baseline.csv                baseline characteristics by hospital,
                                         laid out as the supplementary table
    external_baseline.xlsx               the same, ready to paste
    benchmark_step13.csv                 per-stage timing and memory
    environment_step13.txt               library versions and machine specification

USAGE
-----
    python 13_external_validation.py
=============================================================================
"""

from __future__ import annotations

import importlib
import os
import re
import sys
import warnings
from pathlib import Path

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

VISIT_FILE   = Path(r"./output_step12/tmucrd_pregnancy_by_visit.csv")
CROSSWALK    = Path(r"./icd_crosswalk.csv")
DEV_OR_LONG  = Path(r"./output_step8/or_long.csv")
OUT_DIR      = Path(r"./output_step13")

ID_COL = "CHR_NO"

# Name of the encounter date column in the external visit-level table. This is
# NOT the same as in the development cohort, which calls it `combined_date`;
# set it to whatever the external build produced. If the name is wrong the
# script stops immediately and lists the candidates it found, rather than
# failing later inside the phase calculation.
DATE_COL = "visit_date"

# Diagnosis columns. Names containing 'ICD10' are treated as ICD-10 only;
# the rest may hold either system depending on the record date.
DX_COLS = [
    "MDIAG_CODE", "EDIAG_CODE",
    "SDIAG_CODE1", "SDIAG_CODE2", "SDIAG_CODE3", "SDIAG_CODE4",
    "SDIAG_CODE5", "SDIAG_CODE6", "SDIAG_CODE7",
    "ESDIAG_CODE1", "ESDIAG_CODE2", "ESDIAG_CODE3", "ESDIAG_CODE4",
    "ICD10_CODE1_OUT", "ICD10_CODE2_OUT", "ICD10_CODE3_OUT",
    "ICD10_CODE4_OUT", "ICD10_CODE5_OUT", "ICD10_CODE6_OUT",
    "ICD10_CODE7_OUT", "ICD10_CODE8_OUT",
]

# Taiwan adopted ICD-10-CM on 1 January 2016.
TRANSITION = pd.Timestamp("2016-01-01")

OUTCOME = "c_abortive"

# Sheets included in the forest plot. Only associations estimable in both
# cohorts are drawn, since an estimate that exists on one side only cannot be
# compared. Add the chronic and infectious sheets here to widen it.
FIGURE_SHEETS = ["b_pregnancy", "c_pregnancy", "a_pregnancy"]

# Mark the rows whose estimates fall on opposite sides of the null.
FIGURE_MARK_DISCORDANT = False

FIGURE_COLOURS = {"dev": "#1f4e79", "ext": "#c0392b"}
PHASE_LABEL = {"b": "Before", "c": "During", "a": "After"}
# The abortive history rows are labelled by what they mean, not by the phase,
# so that they cannot be read as the outcome itself.
ABORTIVE_LABEL = {"b": "Previous abortive outcome", "a": "Subsequent abortive outcome"}
FIGURE_DPI = 450          # keeps the figure above IJE's 3600 px width
UK_SPELLING = {"Hemorrhage": "Haemorrhage", "hemorrhage": "haemorrhage",
               "Edema": "Oedema", "edema": "oedema", "Anemia": "Anaemia",
               "anemia": "anaemia", "Diarrhea": "Diarrhoea", "diarrhea": "diarrhoea"}


def _uk(text: str) -> str:
    for us, gb in UK_SPELLING.items():
        text = text.replace(us, gb)
    return text

# Hospital identifier to display name, from the loader's suffix mapping:
# _t = TMUH, _w = Wan Fang, _s = Shuang Ho.
HOSPITAL_NAMES = {
    0: "Taipei Medical University Hospital",
    1: "Wan Fang Hospital",
    2: "Shuang Ho Hospital",
}
AFTER_PREG_GRACE_DAYS = 30

# Washout year. First-order episodes terminating in the first year of data are
# removed, because that year cannot distinguish an incident pregnancy from one
# already in progress when observation began. Applied inside the per-order loop
# and to first-order episodes only, exactly as in step 3; the year differs
# because the two sources begin at different times. Set to None to disable.
WASHOUT_YEAR = 2013
AGE_RISK_LOW, AGE_RISK_HIGH = 20, 35
SPARSE_MIN_CELL = 5

STEP3_MODULE = "03_build_pregnancy_level"
STEP4_MODULE = "04_apply_pregnancy_qc"
STEP8_MODULE = "08_odds_ratios"

# Apply the pregnancy-level quality control to the external cohort as well.
# Without it the two cohorts are not treated alike: episodes in which an
# abortive outcome co-occurs with an incompatible condition are removed from
# the development cohort but retained here, so the current-phase association is
# empty by construction on one side and estimable on the other. Set to False to
# leave the external episodes unfiltered.
APPLY_QC_TO_EXTERNAL = True

# =============================================================================


def load_definitions():
    """ICD-10 sets from step 3, ICD-9 sets from the generated crosswalk."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        s4 = importlib.import_module(STEP3_MODULE)
        s7 = importlib.import_module(STEP8_MODULE)
    except Exception as exc:
        sys.exit(f"[FATAL] could not import the pipeline modules: {exc}")

    groups = {
        "chronic": dict(s4.chronic_conditions),
        "infectious": dict(s4.infectious_conditions),
        "pregnancy": dict(s4.pregnancy_conditions),
    }
    regexes = dict(s4.conditions_regex)

    if not CROSSWALK.exists():
        sys.exit(f"[FATAL] {CROSSWALK} not found. Generate it first:\n"
                 "        python make_icd_crosswalk.py --html supplementary.html")
    cw = pd.read_csv(CROSSWALK)
    icd9 = {r["code"]: [c for c in str(r["icd9"]).split(";") if c and c != "nan"]
            for _, r in cw.iterrows()}

    n10 = sum(len(v) for v in groups.values()) + len(regexes)
    have9 = sum(1 for k in
                (set().union(*[set(g) for g in groups.values()]) | set(regexes))
                if icd9.get(k))
    print(f"    ICD-10 definitions from {STEP3_MODULE}.py : {n10} conditions")
    print(f"    ICD-9 definitions from {CROSSWALK.name}   : {have9} conditions")
    missing9 = sorted(
        (set().union(*[set(g) for g in groups.values()]) | set(regexes))
        - {k for k, v in icd9.items() if v})
    if missing9:
        print(f"    [WARNING] no ICD-9 codes for {len(missing9)} conditions; "
              f"they cannot be detected before {TRANSITION.date()}:")
        for m in missing9[:12]:
            print(f"      {m}")
    return groups, regexes, icd9, s4, s7


def clean_code(value) -> str:
    """Normalise one recorded diagnosis value.

    Numeric ICD-9 columns are frequently exported as floating point, so a code
    arrives as '25000.0' rather than '25000'. Stripping the decimal point
    without removing the trailing zero would turn it into '250000', which still
    matches a three-digit prefix but would silently corrupt any longer one. The
    trailing '.0' is therefore removed before the decimal separator.
    """
    v = str(value).strip().upper()
    if v.endswith(".0"):
        v = v[:-2]
    return v.replace(".", "")


def classify_code(value: str) -> str:
    """Return 'icd9', 'icd10', or 'ambiguous' for one recorded value."""
    v = clean_code(value)
    if not v or v == "NAN":
        return "empty"
    if v[0].isdigit():
        return "icd9"
    if v[0] in ("V", "E") and v[1:2].isdigit():
        return "ambiguous"
    if v[0].isalpha() and v[1:2].isdigit():
        return "icd10"
    return "empty"


def inspect_codes(df: pd.DataFrame, dx_cols: list[str]) -> pd.DataFrame:
    """Report the observed code format so a matching failure cannot pass silently."""
    rows = []
    for c in dx_cols:
        vals = df[c].dropna().map(clean_code)
        vals = vals[(vals != "") & (vals != "NAN")]
        if not len(vals):
            rows.append({"column": c, "n_values": 0, "example": "",
                         "modal_length": None, "icd9": 0, "icd10": 0,
                         "ambiguous": 0})
            continue
        kinds = vals.map(classify_code)
        rows.append({
            "column": c, "n_values": len(vals),
            "example": "; ".join(vals.unique()[:4]),
            "modal_length": int(vals.str.len().mode().iloc[0]),
            "icd9": int((kinds == "icd9").sum()),
            "icd10": int((kinds == "icd10").sum()),
            "ambiguous": int((kinds == "ambiguous").sum()),
        })
    rep = pd.DataFrame(rows)
    print(f"    {'column':<20}{'n':>10}{'len':>5}{'icd9':>9}{'icd10':>9}"
          f"{'ambig':>8}   example")
    for _, r in rep.iterrows():
        print(f"    {r['column']:<20}{r['n_values']:>10,}"
              f"{str(r['modal_length'] or ''):>5}{r['icd9']:>9,}"
              f"{r['icd10']:>9,}{r['ambiguous']:>8,}   {r['example'][:34]}")
    modal = rep.loc[rep["n_values"] > 0, "modal_length"]
    if len(modal) and modal.max() > 3:
        print(f"    codes are longer than three characters, so category "
              f"matching uses a prefix")
    return rep


def build_matchers(groups, regexes, icd9, dx_cols):
    """Compile, for each condition, the prefix sets to test in each system."""
    matchers = {}
    for cat, d in groups.items():
        for cond, codes in d.items():
            matchers[cond] = {"category": cat, "kind": "list",
                              "icd10": [c.upper() for c in codes],
                              "icd9": [c.upper() for c in icd9.get(cond, [])]}
    for cond, pat in regexes.items():
        matchers[cond] = {"category": "chronic", "kind": "regex",
                          "regex": re.compile(pat),
                          "icd9": [c.upper() for c in icd9.get(cond, [])]}
    return matchers


def normalise(df: pd.DataFrame, dx_cols: list[str]) -> pd.DataFrame:
    """Upper-case, strip decimals, and classify every diagnosis value once."""
    norm, kind = {}, {}
    for c in dx_cols:
        s = df[c].map(lambda v: clean_code(v) if pd.notna(v) else pd.NA)
        s = pd.Series(s, index=df.index, dtype="string")
        norm[c] = s
        kind[c] = s.map(lambda v: classify_code(v) if pd.notna(v) else "empty")
    return pd.DataFrame(norm, index=df.index), pd.DataFrame(kind, index=df.index)


def match_condition(norm, kind, spec, icd10_only_cols, is_pre_transition):
    """Boolean series: does this record carry the condition?

    ICD-9 prefixes are tested only where the value is classified as ICD-9, or is
    ambiguous in a legacy column on a pre-transition record. ICD-10 prefixes are
    tested only where the value is classified as ICD-10, or is ambiguous in an
    ICD-10 column. This is what keeps ICD-9 V27x (outcome of delivery) apart
    from ICD-10 V27x (motorcycle collision).
    """
    hit9 = pd.Series(False, index=norm.index)
    hit10 = pd.Series(False, index=norm.index)

    for c in norm.columns:
        col = norm[c]
        k = kind[c]
        is_icd10_col = c in icd10_only_cols

        if spec["kind"] == "list" and spec["icd10"]:
            allow10 = (k == "icd10") | ((k == "ambiguous") & is_icd10_col)
            m = pd.Series(False, index=norm.index)
            for p in spec["icd10"]:
                m |= col.str.startswith(p, na=False)
            hit10 |= m & allow10
        elif spec["kind"] == "regex":
            allow10 = (k == "icd10") | ((k == "ambiguous") & is_icd10_col)
            m = col.str.match(spec["regex"], na=False)
            hit10 |= m.fillna(False) & allow10

        if spec["icd9"]:
            allow9 = (k == "icd9") | ((k == "ambiguous") & (not is_icd10_col)
                                      & is_pre_transition)
            m = pd.Series(False, index=norm.index)
            for p in spec["icd9"]:
                m |= col.str.startswith(p, na=False)
            hit9 |= m & allow9

    return hit9, hit10


NOT_ESTIMABLE_MARK = "\u2014"


def _fmt(o, lo, hi, sparse, estimability) -> str:
    """Format one estimate as it appears in the supplementary tables."""
    if str(estimability).startswith("not estimable") or pd.isna(o):
        return NOT_ESTIMABLE_MARK
    mark = "+" if str(sparse).lower() == "yes" else ""
    return f"{o:.3f} ({lo:.3f}\u2013{hi:.3f}){mark}"


def write_external_workbook(m: pd.DataFrame, out_dir: Path) -> None:
    """Development and external estimates side by side, ready to paste.

    The long-format files remain the machine-readable record; this workbook
    exists so the supplementary table can be assembled without reshaping a CSV
    by hand. A dash marks an estimate that is not estimable, and a trailing plus
    marks one resting on a sparse cell.
    """
    rows = []
    for _, r in m.iterrows():
        rows.append({
            "sheet": r["sheet_dev"],
            "condition": r["condition_dev"],
            "phase": r.get("phase_dev", ""),
            "BPJS Kesehatan": _fmt(r["odds_ratio_dev"], r["ci_lower_dev"],
                                   r["ci_upper_dev"], r.get("sparse_dev"),
                                   r["estimability_dev"]),
            "TMUCRD": _fmt(r["odds_ratio_ext"], r["ci_lower_ext"],
                           r["ci_upper_ext"], r.get("sparse_ext"),
                           r["estimability_ext"]),
            "estimable in both": "Yes" if r["estimable_both"] else "No",
            "direction": ("concordant" if r["estimable_both"] and r["concordant"]
                          else "discordant" if r["estimable_both"] else ""),
        })
    tbl = pd.DataFrame(rows)
    path = out_dir / "OR external.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        for sheet, g in tbl.groupby("sheet", sort=False):
            name = str(sheet)[:31]
            g.drop(columns=["sheet"]).to_excel(w, sheet_name=name, index=False)
            ws = w.sheets[name]
            for cc in ws.columns:
                width = max(len(str(c.value)) if c.value else 0 for c in cc)
                ws.column_dimensions[cc[0].column_letter].width = min(width + 2, 52)
    print(f"    wrote {path.name} ({tbl['sheet'].nunique()} sheets, "
          f"{len(tbl)} rows)")


def draw_forest(m: pd.DataFrame, s7, out_dir: Path) -> int:
    """Forest plot of the associations estimable in both cohorts.

    Only estimable rows are drawn. An estimate that exists on one side only
    cannot be compared, and plotting it invites the reader to compare it with a
    blank. The x axis is logarithmic because the two cohorts differ by orders
    of magnitude in size and therefore in the width of their intervals.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = m[m["sheet_dev"].isin(FIGURE_SHEETS) & m["estimable_both"]].copy()
    if d.empty:
        print("    no estimable association in the selected sheets; "
              "no figure drawn")
        return 0

    order = {c: i for i, (c, _) in enumerate(s7.PREGNANCY_ORDER
                                             + s7.CHRONIC_ORDER
                                             + s7.INFECTIOUS_ORDER)}
    phase_rank = {"b": 0, "c": 1, "a": 2}
    d["_code"] = d["var"].str.replace(r"^[bca]_", "", regex=True)
    d["_p"] = d["phase_dev"].map(phase_rank).fillna(9)
    d["_c"] = d["_code"].map(order).fillna(10 ** 6)
    d = d.sort_values(["_p", "_c"]).reset_index(drop=True)

    labels = [ABORTIVE_LABEL[r["phase_dev"]]
              if r["_code"] == "abortive" and r["phase_dev"] in ABORTIVE_LABEL
              else f"{_uk(str(r['condition_dev']))} "
                   f"({PHASE_LABEL.get(r['phase_dev'], r['phase_dev'])})"
              for _, r in d.iterrows()]

    n = len(d)
    fig, ax = plt.subplots(figsize=(9.5, max(4.0, 0.42 * n + 1.6)))
    y = list(range(n - 1, -1, -1))

    for i in range(n):
        if i % 2 == 0:
            ax.axhspan(y[i] - 0.5, y[i] + 0.5, color="0.94", zorder=0)

    for suffix, key, lab in (("dev", "dev", "BPJS Kesehatan"),
                             ("ext", "ext", "TMUCRD")):
        off = 0.16 if key == "dev" else -0.16
        c = FIGURE_COLOURS[key]
        o = d[f"odds_ratio_{suffix}"].astype(float)
        lo = d[f"ci_lower_{suffix}"].astype(float)
        hi = d[f"ci_upper_{suffix}"].astype(float)
        yy = [v + off for v in y]
        ax.errorbar(o, yy, xerr=[o - lo, hi - o], fmt="o", markersize=5,
                    color=c, ecolor=c, elinewidth=1.3, capsize=2.5,
                    markeredgecolor="white", markeredgewidth=0.6,
                    label=lab, zorder=3, linestyle="none")

    ax.axvline(1.0, color="0.35", ls="--", lw=1.1, zorder=2)
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlabel("Odds ratio (95% CI), log scale")
    ticks = [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]   # as in Figures 3 and 4
    lo_all = min(d["ci_lower_dev"].min(), d["ci_lower_ext"].min())
    hi_all = max(d["ci_upper_dev"].max(), d["ci_upper_ext"].max())
    ticks = [t for t in ticks if lo_all / 1.6 <= t <= hi_all * 1.6]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.set_xlim(lo_all / 1.6, hi_all * 1.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", length=0)

    if FIGURE_MARK_DISCORDANT:
        for i, (_, r) in enumerate(d.iterrows()):
            if not r["concordant"]:
                ax.text(ax.get_xlim()[1], y[i], "  \u2020", va="center",
                        ha="left", fontsize=9, color="0.25")

    # Legend below the axis so that it never covers a row.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=2,
              frameon=False, fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"external_forest.{ext}", dpi=FIGURE_DPI,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)

    n_disc = int((~d["concordant"]).sum())
    print(f"    forest plot: {n} associations drawn, "
          f"{n - n_disc} concordant and {n_disc} discordant")
    return n


def apply_external_age_exclusion(ep: pd.DataFrame) -> pd.DataFrame:
    """Remove episodes with an implausible maternal age.

    Applied to the episode, using the aggregated age, exactly as at the
    pregnancy level of the development cohort. Maternal age there is the
    earliest age recorded within the episode, so a woman's other episodes may
    fall within the plausible range and are retained.
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        s5 = importlib.import_module(STEP4_MODULE)
        lo, hi = int(s5.AGE_MIN), int(s5.AGE_MAX)
    except Exception:
        lo, hi = 12, 55
        print(f"    [NOTE] age bounds default to {lo}-{hi}")

    if "age" not in ep.columns:
        print("    [WARNING] no age column; the age exclusion is skipped")
        return ep
    age = pd.to_numeric(ep["age"], errors="coerce")
    bad = (age < lo) | (age > hi) | age.isna()
    n = int(bad.sum())
    print(f"    age exclusion, applied to the episode as in the development "
          f"cohort:")
    print(f"      bounds                     : {lo} to {hi} years")
    print(f"      episodes removed           : {n:,} "
          f"({n / len(ep) * 100:.2f}%)" if len(ep) else "")
    print(f"      episodes retained          : {len(ep) - n:,}")
    pd.DataFrame([
        {"metric": "episodes_total", "value": len(ep)},
        {"metric": "removed_age_rule_episode", "value": n},
        {"metric": "episodes_after_age_exclusion", "value": len(ep) - n},
    ]).to_csv(OUT_DIR / "external_age_qc_report.csv", index=False)
    return ep[~bad].reset_index(drop=True)


def apply_external_qc(ep: pd.DataFrame) -> pd.DataFrame:
    """Remove episodes in which an abortive outcome co-occurs with an
    incompatible condition, using the same column list as the development
    cohort so that both are filtered identically."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        s5 = importlib.import_module(STEP4_MODULE)
        conflict = list(s5.CONFLICT_COLS)
    except Exception as exc:
        print(f"    [WARNING] could not import {STEP4_MODULE}.py ({exc}); "
              "the external cohort is left unfiltered")
        return ep

    present = [c for c in conflict if c in ep.columns]
    absent = [c for c in conflict if c not in ep.columns]
    if not present or OUTCOME not in ep.columns:
        print("    [WARNING] conflict columns unavailable; no episode removed")
        return ep

    flagged = (ep[OUTCOME].fillna(0).astype(int) == 1) & \
        (ep[present].fillna(0).astype(int).sum(axis=1) > 0)
    n = int(flagged.sum())
    print(f"    quality control, applied as in the development cohort:")
    print(f"      conflict columns evaluated : {len(present)} of {len(conflict)}")
    if absent:
        print(f"      absent from this cohort    : {len(absent)}")
    print(f"      episodes removed           : {n:,} "
          f"({n / len(ep) * 100:.2f}%)")
    print(f"      episodes retained          : {len(ep) - n:,}")
    pd.DataFrame([
        {"metric": "episodes_before_conflict_qc", "value": len(ep)},
        {"metric": "conflict_columns_evaluated", "value": len(present)},
        {"metric": "conflict_columns_absent", "value": len(absent)},
        {"metric": "removed_diagnostic_conflict", "value": n},
        {"metric": "episodes_final", "value": len(ep) - n},
    ]).to_csv(OUT_DIR / "external_qc_report.csv", index=False)
    return ep[~flagged].reset_index(drop=True)


def apply_external_sequence_qc(ep: pd.DataFrame,
                               visit: pd.DataFrame) -> pd.DataFrame:
    """Episode-sequence QC of step 4, applied with the same function and
    tolerance as in the development cohort."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    s5 = importlib.import_module(STEP4_MODULE)
    if not {"doa", "dol"} <= set(visit.columns):
        print("    [WARNING] doa/dol absent from the external visit table; "
              "sequence QC skipped")
        return ep
    flags = s5.sequence_flags(visit, id_col=ID_COL, date_col=DATE_COL)
    k_ep = ep[ID_COL].astype(str).str.lstrip("0") + "|" + \
        pd.to_numeric(ep["n_preg"]).astype(int).astype(str)
    k_fl = flags[ID_COL].astype(str).str.lstrip("0") + "|" + \
        flags["n_preg"].astype(str)
    reason = k_ep.map(dict(zip(k_fl, flags["reason"])))
    counts = reason.value_counts()
    print("    episode-sequence QC, applied as in the development cohort:")
    for r in ("overlap_delivery", "overlap_abortive", "abortive_with_delivery"):
        print(f"      removed, {r:<24}: {int(counts.get(r, 0)):,}")
    print(f"      episodes retained          : {int(reason.isna().sum()):,}")
    pd.DataFrame([{"metric": f"removed_{r}", "value": int(counts.get(r, 0))}
                  for r in ("overlap_delivery", "overlap_abortive",
                            "abortive_with_delivery")]
                 + [{"metric": "episodes_final", "value": int(reason.isna().sum())}]
                 ).to_csv(OUT_DIR / "external_sequence_qc_report.csv", index=False)
    return ep[reason.isna().values].reset_index(drop=True)


def build_baseline(ep: pd.DataFrame) -> pd.DataFrame:
    """Baseline characteristics laid out as the supplementary table.

    Rows are characteristics, columns are hospitals plus a combined column.
    Every categorical block reports all of its categories, so the counts sum to
    the column total and a reader can verify the table with a calculator. The
    earlier version reported only the positive category of the extreme-age
    indicator and omitted pregnancy order entirely, which left both blocks
    unable to sum to their own denominator.
    """
    groups = [("All hospitals", ep)]
    if "hospital_id" in ep.columns:
        for h, g in sorted(ep.groupby(pd.to_numeric(ep["hospital_id"],
                                                    errors="coerce"))):
            groups.append((HOSPITAL_NAMES.get(int(h), f"Hospital {int(h)}"), g))

    orders = sorted(int(v) for v in ep["n_preg"].dropna().unique()) \
        if "n_preg" in ep.columns else []

    def pct(n, d):
        return f"{n:,} ({100 * n / d:.2f})" if d else "0 (0.00)"

    rows = [{"Characteristics": "Maternal age, mean \u00b1 SD"},
            {"Characteristics": "Maternal age risk category, n(%)"},
            {"Characteristics": "    20-35 years"},
            {"Characteristics": "    <20 or >35 years"},
            {"Characteristics": "Pregnancy order, n(%)"}]
    rows += [{"Characteristics": f"    {k}"} for k in orders]
    rows.append({"Characteristics": "Abortive outcome, n(%)"})

    for label, g in groups:
        n = len(g)
        col = f"{label} (n={n:,})"
        vals = []
        if "age" in g.columns:
            vals.append(f"{g['age'].mean():.2f} ({g['age'].std():.2f})")
        else:
            vals.append("")
        vals.append("")
        if "age_risk" in g.columns:
            ar = pd.to_numeric(g["age_risk"], errors="coerce")
            vals.append(pct(int((ar == 0).sum()), n))
            vals.append(pct(int((ar == 1).sum()), n))
        else:
            vals += ["", ""]
        vals.append("")
        for k in orders:
            vals.append(pct(int((g["n_preg"] == k).sum()), n))
        vals.append(pct(int(g[OUTCOME].sum()), n) if OUTCOME in g.columns else "")
        for r, v in zip(rows, vals):
            r[col] = v

    tbl = pd.DataFrame(rows)

    # Verify that every categorical block sums to its column total.
    for label, g in groups:
        n = len(g)
        col = f"{label} (n={n:,})"
        if "age_risk" in g.columns:
            ar = pd.to_numeric(g["age_risk"], errors="coerce")
            s_ar = int((ar == 0).sum()) + int((ar == 1).sum())
            if s_ar != n:
                print(f"    [WARNING] {label}: age risk sums to {s_ar:,} "
                      f"vs {n:,} (missing {n - s_ar:,})")
        if orders:
            s_np = sum(int((g["n_preg"] == k).sum()) for k in orders)
            if s_np != n:
                print(f"    [WARNING] {label}: pregnancy order sums to "
                      f"{s_np:,} vs {n:,} (missing {n - s_np:,})")
    return tbl


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 74)
    print(" STEP 13 -- EXTERNAL VALIDATION IN THE TMUCRD COHORT")
    print("=" * 74)
    print(environment_report(str(OUT_DIR / "environment_step13.txt")))
    warn_if_no_psutil()

    if not VISIT_FILE.exists():
        sys.exit(f"[FATAL] {VISIT_FILE} not found. Build it first with the "
                 "external cohort script, which must apply the same anchoring "
                 "rules as step 1.")

    bench = Bench("step 13, external validation")

    with bench.stage("Import definitions", "python") as b:
        groups, regexes, icd9, s4, s7 = load_definitions()
        b["rows_out"] = sum(len(g) for g in groups.values()) + len(regexes)

    with bench.stage("Load external visit-level table", "pandas") as b:
        # Read everything as text. Diagnosis columns that happen to be all
        # numeric would otherwise be parsed as floating point, and the codes
        # would arrive with a spurious decimal component.
        visit = pd.read_csv(VISIT_FILE, dtype=str, low_memory=False)
        cols = {c.upper(): c for c in visit.columns}

        def resolve(name):
            if name in visit.columns:
                return name
            return cols.get(name.upper())

        missing = [r for r in (ID_COL, DATE_COL, "fin_g", "ref_start", "n_preg")
                   if resolve(r) is None]
        if missing:
            print(f"\n[FATAL] required column(s) not found: {missing}")
            print("\nColumns present in the file:")
            for c in visit.columns:
                print(f"    {c}")
            if DATE_COL in missing:
                cand = []
                for c in visit.columns:
                    if re.search(r"date|dt$|_dt|day|time", c, re.I):
                        cand.append(c)
                        continue
                    sample = visit[c].dropna().astype(str).head(200)
                    if len(sample):
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            ok = pd.to_datetime(sample, errors="coerce",
                                                format="mixed").notna().mean()
                        if ok > 0.9:
                            cand.append(c)
                print(f"\nColumns that look like dates, any of which could be "
                      f"DATE_COL:")
                for c in cand or ["    (none found)"]:
                    print(f"    {c}")
                print("\nSet DATE_COL in the configuration block to the encounter "
                      "date used when")
                print("the external visit-level table was built, then rerun.")
            sys.exit(1)

        # Work with the resolved names from here on.
        ID_RESOLVED = resolve(ID_COL)
        DATE_RESOLVED = resolve(DATE_COL)
        if DATE_RESOLVED != DATE_COL:
            visit = visit.rename(columns={DATE_RESOLVED: DATE_COL})
        if ID_RESOLVED != ID_COL:
            visit = visit.rename(columns={ID_RESOLVED: ID_COL})
        dx = [cols[c.upper()] for c in DX_COLS if c.upper() in cols]
        missing_dx = [c for c in DX_COLS if c.upper() not in cols]
        if not dx:
            sys.exit("[FATAL] no diagnosis columns found; check DX_COLS.")
        if missing_dx:
            print(f"    [NOTE] {len(missing_dx)} configured diagnosis columns "
                  f"absent: {missing_dx[:6]}")
        for c in (DATE_COL, "fin_g", "ref_start"):
            if c in visit.columns:
                visit[c] = pd.to_datetime(visit[c], errors="coerce")
        for c in ("age", "n_preg", "dom", "subsid", "hospital_id", "age_risk",
                  "ref_year"):
            if c in visit.columns:
                visit[c] = pd.to_numeric(visit[c], errors="coerce")
        print(f"    {len(visit):,} rows, {len(dx)} diagnosis columns")
        b["rows_out"] = len(visit)

    with bench.stage("Inspect the recorded code format", "pandas") as b:
        fmt = inspect_codes(visit, dx)
        fmt.to_csv(OUT_DIR / "external_code_format.csv", index=False)
        b["rows_out"] = len(fmt)

    with bench.stage("Extract condition flags and collapse", "pandas",
                     rows_in=len(visit)) as b:
        matchers = build_matchers(groups, regexes, icd9, dx)
        norm, kind = normalise(visit, dx)
        icd10_only = {c for c in dx if "ICD10" in c.upper()}
        pre = visit[DATE_COL] < TRANSITION if DATE_COL in visit.columns \
            else pd.Series(False, index=visit.index)
        print(f"    records before {TRANSITION.date()} : {int(pre.sum()):,}")
        print(f"    records on or after               : {int((~pre).sum()):,}")

        # One boolean column per condition, evaluated once per encounter.
        hits, sparsity = {}, []
        for cond, spec in matchers.items():
            h9, h10 = match_condition(norm, kind, spec, icd10_only, pre)
            hit = h9 | h10
            hits[cond] = hit
            sparsity.append({
                "condition": cond, "category": spec["category"],
                "n_icd9_only": int((h9 & ~h10).sum()),
                "n_icd10_only": int((h10 & ~h9).sum()),
                "n_both": int((h9 & h10).sum()),
                "n_records_pre_transition": int((hit & pre).sum()),
                "n_records_post_transition": int((hit & ~pre).sum()),
                "has_icd9_definition": bool(spec.get("icd9")),
                "n_records_total": int(hit.sum())})
        hits = pd.DataFrame(hits, index=visit.index)
        sp = pd.DataFrame(sparsity).sort_values("n_records_total")
        sp.to_csv(OUT_DIR / "external_sparsity_report.csv", index=False)
        never = sp[sp["n_records_total"] == 0]
        print(f"    conditions never matched          : {len(never)} of {len(sp)}")

        # Phase assignment requires the episode's anchor and onset to be known
        # for EVERY encounter of that woman, not only for the anchor encounter
        # itself. The development pipeline maps them across the woman's records
        # before evaluating the phase boundaries, and the same mapping is
        # applied here. Evaluating the boundaries against the raw per-record
        # anchor date would leave every non-anchor encounter comparing against
        # a missing value, so only codes recorded on the termination encounter
        # itself would ever be flagged.
        orders = sorted(int(v) for v in
                        visit.loc[visit["fin_g"].notna(), "n_preg"].dropna().unique())
        print(f"    pregnancy orders present          : {orders}")

        grace = pd.Timedelta(days=AFTER_PREG_GRACE_DAYS)
        cat_of = {c: matchers[c]["category"] for c in matchers}
        frames, washout_total = [], 0
        for k in orders:
            women = visit.loc[visit["n_preg"] == k, ID_COL].unique()
            sel = visit[ID_COL].isin(women)
            v = visit[sel]
            h = hits[sel]

            anchors = (visit.loc[(visit["n_preg"] == k) & visit["fin_g"].notna(),
                                 [ID_COL, "fin_g", "ref_start"]]
                       .sort_values([ID_COL, "fin_g"])
                       .drop_duplicates(ID_COL, keep="first")
                       .set_index(ID_COL))
            ref = v[ID_COL].map(anchors["fin_g"])
            rs = v[ID_COL].map(anchors["ref_start"])
            d = v[DATE_COL]
            # Derived from the mapped anchor, so every record of the woman
            # carries the episode's year. Step 3 does the same, and the washout
            # below filters on it.
            ref_year = ref.dt.year

            ph = {
                "before_chronic": d <= ref,
                "before_inf_preg": d < rs,
                "after_chronic_inf": d > ref,
                "after_preg": (d - ref).dt.days > AFTER_PREG_GRACE_DAYS,
                "during_inf": (d >= rs) & (d <= ref),
                "during_preg": (d >= rs) & (d <= (ref + grace)),
            }

            cols = {ID_COL: v[ID_COL].values}
            for cond, cat in cat_of.items():
                m = h[cond]
                if cat == "chronic":
                    cols[f"b_{cond}"] = (m & ph["before_chronic"]).values
                    cols[f"a_{cond}"] = (m & ph["after_chronic_inf"]).values
                elif cat == "infectious":
                    cols[f"b_{cond}"] = (m & ph["before_inf_preg"]).values
                    cols[f"c_{cond}"] = (m & ph["during_inf"]).values
                    cols[f"a_{cond}"] = (m & ph["after_chronic_inf"]).values
                else:
                    cols[f"b_{cond}"] = (m & ph["before_inf_preg"]).values
                    cols[f"c_{cond}"] = (m & ph["during_preg"]).values
                    cols[f"a_{cond}"] = (m & ph["after_preg"]).values

            block = pd.DataFrame(cols)
            block["ref_year"] = ref_year.values

            # The washout exists ONLY in the first pass, as in step 3. Records
            # of a first-order episode terminating in the washout year are
            # dropped from this pass; the woman's higher-order episodes are
            # untouched because each pass starts again from the full table.
            if k == 1 and WASHOUT_YEAR is not None:
                n_before_w = block[ID_COL].nunique()
                block = block[block["ref_year"] != WASHOUT_YEAR]
                n_washout = n_before_w - block[ID_COL].nunique()
                washout_total += n_washout

            flag_cols = [c for c in block.columns
                         if c not in (ID_COL, "ref_year")]
            out = block.groupby(ID_COL)[flag_cols].max().astype(int).reset_index()

            demo = (v.loc[v["n_preg"] == k]
                    .groupby(ID_COL)
                    .agg({c: "min" for c in ("age", "dom", "hospital_id",
                                             "ref_year", "ref_start", "fin_g")
                          if c in v.columns})
                    .reset_index())
            out = out.merge(demo, on=ID_COL, how="left")
            out["n_preg"] = k
            frames.append(out)
            note = (f"   ({n_washout:,} removed by the {WASHOUT_YEAR} washout)"
                    if k == 1 and WASHOUT_YEAR is not None and n_washout else "")
            print(f"      order {k:>2} : {len(out):>8,} episodes from "
                  f"{len(women):,} women, {len(v):,} encounters{note}")

        ep = pd.concat(frames, ignore_index=True)
        if WASHOUT_YEAR is not None:
            print(f"    removed by the {WASHOUT_YEAR} washout : "
                  f"{washout_total:>10,}")
        # Saved for the flow diagram (step 16); until now these were printed only.
        pd.DataFrame([
            {"metric": "episodes_before_washout", "value": len(ep) + washout_total},
            {"metric": "removed_by_washout", "value": washout_total},
            {"metric": "washout_year", "value": WASHOUT_YEAR},
            {"metric": "episodes_after_washout", "value": len(ep)},
            {"metric": "women_after_washout", "value": ep[ID_COL].nunique()},
            {"metric": "abortive_after_washout",
             "value": int(ep[OUTCOME].sum()) if OUTCOME in ep.columns else None},
        ]).to_csv(OUT_DIR / "external_washout_report.csv", index=False)
        if "age" in ep.columns:
            ep["age_risk"] = ((ep["age"] < AGE_RISK_LOW)
                              | (ep["age"] > AGE_RISK_HIGH)).astype(int)
        ep = apply_external_age_exclusion(ep)
        if APPLY_QC_TO_EXTERNAL:
            ep = apply_external_qc(ep)
            ep = apply_external_sequence_qc(ep, visit)
        # Output column names follow step 3, so the two episode-level files can
        # be compared directly without renaming.
        ep = ep.rename(columns={"ref_start": "pregnancy_start",
                                "fin_g": "pregnancy_end"})
        ep.to_csv(OUT_DIR / "tmucrd_pregnancy_by_episode.csv", index=False)
        print(f"    {len(ep):,} episodes from {ep[ID_COL].nunique():,} patients")
        if OUTCOME not in ep.columns:
            sys.exit(f"[FATAL] outcome column '{OUTCOME}' was not produced.")
        print(f"    with a current abortive outcome    : "
              f"{int(ep[OUTCOME].sum()):,}")
        flag_cols_all = [c for c in ep.columns if c[:2] in ("b_", "c_", "a_")]
        nz = sum(1 for c in flag_cols_all if ep[c].sum() == 0)
        print(f"    flags never positive at episode level: {nz} of "
              f"{len(flag_cols_all)}")
        b["rows_out"] = len(ep)

    with bench.stage("Odds ratios, using step 8's calculation", "DuckDB",
                     rows_in=len(ep)) as b:
        con = duckdb.connect()
        con.register("ep_df", ep)
        con.execute("CREATE OR REPLACE TABLE final AS SELECT * FROM ep_df")
        con.unregister("ep_df")
        lut = {c.lower(): c for c in
               con.sql("SELECT name FROM pragma_table_info('final')").df()["name"]}
        frames = []
        for cat, conds in [("chronic", s7.CHRONIC), ("infectious", s7.INFECTIOUS),
                           ("pregnancy", s7.PREGNANCY)]:
            for phase_pref in s7.PHASE_MAP[cat]:
                cl = [lut[f"{phase_pref}{c}".lower()] for c in conds
                      if f"{phase_pref}{c}".lower() in lut]
                if not cl:
                    continue
                res = s7.annotate(s7.calc_or(con, cl))
                res.insert(0, "sheet", f"{phase_pref.strip('_')}_{cat}")
                frames.append(res)
        ext = pd.concat(frames, ignore_index=True)
        ext.drop(columns=["_rank"]).to_csv(OUT_DIR / "external_or_long.csv",
                                           index=False)
        con.close()
        print(f"    {len(ext):,} associations computed")
        b["rows_out"] = len(ext)

    with bench.stage("Compare direction against the development cohort",
                     "pandas") as b:
        if not DEV_OR_LONG.exists():
            sys.exit(f"[FATAL] {DEV_OR_LONG} not found. Run step 8 first.")
        dev = pd.read_csv(DEV_OR_LONG)
        m = dev.merge(ext, on="var", suffixes=("_dev", "_ext"))
        m["estimable_dev"] = ~m["estimability_dev"].astype(str).str.startswith(
            "not estimable") & m["sparse_dev"].eq("No")
        m["estimable_ext"] = ~m["estimability_ext"].astype(str).str.startswith(
            "not estimable") & m["sparse_ext"].eq("No")
        m["estimable_both"] = m["estimable_dev"] & m["estimable_ext"]
        m["concordant"] = ((m["odds_ratio_dev"] - 1)
                           * (m["odds_ratio_ext"] - 1)) > 0

        keep = ["sheet_dev", "var", "condition_dev", "phase_dev",
                "odds_ratio_dev", "ci_lower_dev", "ci_upper_dev",
                "estimability_dev", "odds_ratio_ext", "ci_lower_ext",
                "ci_upper_ext", "estimability_ext", "estimable_both",
                "concordant"]
        m[keep].to_csv(OUT_DIR / "external_concordance.csv", index=False)
        write_external_workbook(m, OUT_DIR)
        draw_forest(m, s7, OUT_DIR)

        n_pairs = len(m)
        est = m[m["estimable_both"]]
        n_conc = int(est["concordant"].sum())
        n_disc = len(est) - n_conc
        pct = round(n_conc / len(est) * 100, 1) if len(est) else float("nan")
        summary = pd.DataFrame([
            {"metric": "association-phase comparisons", "value": n_pairs},
            {"metric": "estimable in both cohorts", "value": len(est)},
            {"metric": "not estimable in both", "value": n_pairs - len(est)},
            {"metric": "directionally concordant", "value": n_conc},
            {"metric": "concordance percent", "value": pct},
            {"metric": "discordant", "value": n_disc},
        ])
        summary.to_csv(OUT_DIR / "external_concordance_summary.csv", index=False)

        print(f"\n    association-phase comparisons : {n_pairs:>6,}")
        print(f"    estimable in both cohorts     : {len(est):>6,}")
        print(f"    directionally concordant      : {n_conc:>6,}  ({pct}%)")
        print(f"    discordant                    : {n_disc:>6,}")
        if n_disc:
            print("\n    discordant associations:")
            for _, r in est[~est["concordant"]].iterrows():
                print(f"      {r['var']:<18} {str(r['condition_dev'])[:38]:<38} "
                      f"dev {r['odds_ratio_dev']:>8.3f}   "
                      f"ext {r['odds_ratio_ext']:>8.3f}")
        b["rows_out"] = n_pairs

    with bench.stage("Baseline characteristics", "pandas", rows_in=len(ep)) as b:
        baseline = build_baseline(ep)
        baseline.to_csv(OUT_DIR / "external_baseline.csv", index=False)
        with pd.ExcelWriter(OUT_DIR / "external_baseline.xlsx",
                            engine="openpyxl") as w:
            baseline.to_excel(w, sheet_name="Baseline", index=False)
            ws = w.sheets["Baseline"]
            for cc in ws.columns:
                width = max(len(str(c.value)) if c.value else 0 for c in cc)
                ws.column_dimensions[cc[0].column_letter].width = min(width + 2, 46)
        print(baseline.to_string(index=False))
        b["rows_out"] = len(baseline)

    bench.finalise(str(OUT_DIR / "benchmark_step13.csv"), rows_out=len(ep))
    print(f"\nOutput written to {OUT_DIR.resolve()}")


def figure_only() -> None:
    """Redraw the forest plot from external_concordance.csv without rerunning."""
    p = OUT_DIR / "external_concordance.csv"
    if not p.exists():
        sys.exit(f"[FATAL] {p} not found; run step 13 first")
    s7 = importlib.import_module(STEP8_MODULE)
    m = pd.read_csv(p)
    m["estimable_both"] = m["estimable_both"].astype(str).str.lower().eq("true")
    m["concordant"] = m["concordant"].astype(str).str.lower().eq("true")
    draw_forest(m, s7, OUT_DIR)


if __name__ == "__main__":
    if "--figure-only" in sys.argv:
        figure_only()
    else:
        main()