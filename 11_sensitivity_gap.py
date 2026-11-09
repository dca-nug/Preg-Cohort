"""
=============================================================================
 STEP 11 -- SENSITIVITY TO THE EPISODE SEPARATION RULE
=============================================================================

Repeats the whole pregnancy-level analysis under each separation scheme in
SCHEMES and assembles the comparison tables and figure:

    uniform 182 days        main analysis (26 weeks: earliest postpartum
                            ovulation in most women plus the earliest gestation
                            recorded as a delivery)
    259 / 105 days          threshold set by the type of the LATER termination
                            record: 259 days before a delivery, 105 days before
                            an abortive record (empirical troughs of step 2,
                            rounded to whole weeks)

For each scheme, the driver re-anchors the complete visit-level table and then
reruns the existing steps:

    anchoring          step 1  anchor_episodes
    feature extraction step 3  feature_pass
    quality control    step 4  age_exclusion, conflict_exclusion,
                               sequence_flags, sequence_exclusion
    odds ratios        step 8  calc_or, annotate

NOTHING IS REIMPLEMENTED HERE
-----------------------------
The previous arrangement used four separate scripts, each carrying its own copy
of the logic it needed: the sensitivity build duplicated `feature_pass`
verbatim, and separate scripts duplicated the odds-ratio calculation, the
Table 1 construction, and the figure. Four copies of the same logic must be
kept in step, and they had already fallen out of step: the duplicated
`feature_pass` still aggregated `age_risk` across visits while the main build
had moved to deriving it from the aggregated age.

This driver therefore imports and calls the same functions the main analysis
uses. The 182-day column is produced by exactly the code that produces the main
result, so it reproduces it by construction rather than by coincidence, and a
change to any step propagates here automatically.

THE FIGURE IS NOT HAND-TYPED
----------------------------
The earlier figure script held its counts as literal values, and its own
docstring recorded that three different pregnancy totals were in circulation:
551,448 in a spreadsheet, 551,369 in an earlier chart, and 551,488 from the
quality-control output, which is also the Table 1 denominator. Counts here are
read from the quality-control output of each run, so the three cannot diverge.
Deliveries and abortive outcomes are recounted from the episode files rather
than transcribed, and the reconciliation check that their sum equals the
pregnancy total is applied automatically.

WHAT THE SCHEME DOES AND DOES NOT AFFECT
-------------------------------------
Only the pregnancy-level dataset is rebuilt. Maternal age, extreme-age
category, residence, and subsidy status at the visit and individual levels are
properties of encounters and women, not of episode boundaries, and are
unaffected. Gravidity at the individual level IS scheme-dependent, so the
individual-level column is not reported per scheme here; the comparison table
covers the pregnancy level only, which is what the supplement reports.

INPUT
-----
    pregnancy by visit.csv               from step 1 (complete, not age-cleaned)

OUTPUT
------
    scheme_<id>/pregnancy by episode.csv          pre-QC episodes for that scheme
    scheme_<id>/pregnancy by episode cleaned.csv  analytic episodes for that scheme
    scheme_<id>/or_long.csv                       odds ratios for that scheme
    sensitivity_qc_counts.csv                    the quality-control table
    sensitivity_table1_pregnancy.csv             baseline characteristics by scheme
    sensitivity_or_comparison.csv                odds ratios side by side
    OR by scheme.xlsx                            the same, one sheet per category
                                                 and phase, formatted for the
                                                 supplementary table
    sensitivity_significance_switches.csv        pairs changing significance or direction
    scheme_sensitivity_counts.png                absolute counts by scheme
    scheme_sensitivity_relative.png              change relative to the main scheme
    Sensitivity_tables.xlsx                      copy-ready tables: main-text Table 3,
                                                 baseline by rule, OR by rule
    sensitivity_switch_summary.csv               reversals and CI switches among
                                                 associations estimable under both
    benchmark_step11.csv                          per-stage timing and memory
    environment_step11.txt                        library versions and machine spec

USAGE
-----
    python 11_sensitivity_gap.py
    python 11_sensitivity_gap.py --report-only   rebuild only Sensitivity_tables.xlsx
=============================================================================
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

VISIT_CSV = Path(r"./output_step1/pregnancy by visit.csv")
OUT_DIR   = Path(r"./output_step11")

# label -> (folder id, separation rule passed to step 1's anchor_episodes)
SCHEMES = {
    "182 d (uniform)": ("uniform_182", 182),
    "259/105 d (by later record)": ("later_259_105",
                                    {"delivery": 259, "abortive": 105}),
}
MAIN_GAP = "182 d (uniform)"  # the main analysis, used as the reference column

# Only associations estimable under EVERY scheme are compared, since a pair
# that is estimable in one scheme and not in another cannot be said to have
# changed significance.
COMPARE_ONLY_COMMON = True

DPI = 300
COLORS = {"Pregnancies": "#1f77b4", "Deliveries": "#ff7f0e",
          "Abortive outcomes": "#2ca02c"}

# =============================================================================


def load_steps():
    """Import the main-analysis modules so nothing is reimplemented."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    mods = {}
    for alias, name in [("s1", "01_build_visit_level"),
                        ("s4", "03_build_pregnancy_level"),
                        ("s5", "04_apply_pregnancy_qc"),
                        ("s7", "08_odds_ratios")]:
        try:
            mods[alias] = importlib.import_module(name)
        except Exception as exc:
            sys.exit(f"[FATAL] could not import {name}.py: {exc}\n"
                     "        All pipeline scripts must sit in one folder.")
    print("    imported 01, 04, 05, and 07; no logic is duplicated in this driver")
    print("      01 supplies the ICD code sets and gestational offsets")
    print("      04 supplies feature_pass, 05 the quality control, "
          "07 the odds ratios")
    return mods["s1"], mods["s4"], mods["s5"], mods["s7"]


def build_for_gap(visit_raw: pd.DataFrame, gap: str, sep, s1, s4, s5, s7,
                  out_dir: Path) -> dict:
    """Re-anchor under scheme `gap` (rule `sep`), rebuild, apply QC, and
    compute odds ratios."""
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  --- scheme {gap} " + "-" * 40)

    visit, diag = s1.anchor_episodes(
        visit_raw.copy(), sep,
        dx_cols=s1.DX_COLS, abort_codes=s1.ABORT_CODES,
        partus_codes=s1.PARTUS_CODES, preterm_codes=s1.PRETERM_CODES,
        postterm_codes=s1.POSTTERM_CODES,
        days_abort=s1.DAYS_ABORT, days_preterm=s1.DAYS_PRETERM,
        days_postterm=s1.DAYS_POSTTERM, days_term=s1.DAYS_TERM)

    # --- feature extraction, one pass per pregnancy order --------------------
    ep = visit.loc[visit["fin_g"].notna(), "n_preg"].dropna().astype(int)
    kmax = int(ep.max())
    frames, washout = [], 0
    for k in range(1, kmax + 1):
        if not (visit["n_preg"] == k).any():
            continue
        out, n_wash = s4.feature_pass(visit.copy(), k)
        washout += n_wash
        frames.append(out)
    episodes = pd.concat(frames, ignore_index=True).rename(
        columns={"ref_start": "pregnancy_start", "ref": "pregnancy_end"})
    pre_qc = out_dir / "pregnancy by episode.csv"
    episodes.to_csv(pre_qc, index=False)
    print(f"      episodes before QC   : {len(episodes):>10,}"
          f"   ({washout:,} removed by the {s4.WASHOUT_YEAR} washout)")

    # --- quality control, using step 4's own functions ------------------------
    con = duckdb.connect()
    con.register("ep_df", episodes)
    con.execute("CREATE OR REPLACE TABLE raw AS SELECT * FROM ep_df")
    con.unregister("ep_df")
    cols = con.sql("SELECT name FROM pragma_table_info('raw')").df()["name"].tolist()
    present, absent = s5.resolve_conflict_cols(cols)
    age_rep = s5.age_exclusion(con)
    conf_rep, breakdown = s5.conflict_exclusion(con, present)
    flags = s5.sequence_flags(visit, dx_cols=s1.DX_COLS,
                              abort_codes=s1.ABORT_CODES,
                              partus_codes=s1.PARTUS_CODES)
    seq_rep = s5.sequence_exclusion(con, flags)
    conf_rep["episodes_final"] = seq_rep.pop("episodes_final")
    s5.recompute_age_risk(con)

    cleaned = out_dir / "pregnancy by episode cleaned.csv"
    con.execute(f"COPY final TO '{cleaned.resolve().as_posix()}' "
                "(HEADER, DELIMITER ',')")

    # --- counts for the figure, recounted rather than transcribed ------------
    counts = con.sql("""
        SELECT COUNT(*) AS pregnancies,
               SUM(CASE WHEN COALESCE("c_abortive",0)=1 THEN 1 ELSE 0 END)
                   AS abortive,
               SUM(CASE WHEN COALESCE("c_abortive",0)=0 THEN 1 ELSE 0 END)
                   AS deliveries
        FROM final
    """).df().iloc[0]

    # --- odds ratios, using step 8's own functions ---------------------------
    con.execute("CREATE OR REPLACE TABLE final AS SELECT * FROM final")
    lut = {c.lower(): c for c in
           con.sql("SELECT name FROM pragma_table_info('final')").df()["name"]}
    sheets = {}
    for cat, conds in [('chronic', s7.CHRONIC), ('infectious', s7.INFECTIOUS),
                       ('pregnancy', s7.PREGNANCY)]:
        for phase in s7.PHASE_MAP[cat]:
            key = f"{phase.strip('_')}_{cat}"
            sheets[key] = [lut[f"{phase}{c}".lower()] for c in conds
                           if f"{phase}{c}".lower() in lut]
    trio = [lut[c] for c, _ in s7.TRIO_ORDER if c in lut]
    if trio:
        sheets = {"characteristics": trio, **sheets}

    or_frames = []
    for name, cl in sheets.items():
        if not cl:
            continue
        res = s7.annotate(s7.calc_or(con, cl))
        res.insert(0, "sheet", name)
        or_frames.append(res)
    or_long = pd.concat(or_frames, ignore_index=True)
    or_long["gap_days"] = gap
    or_long.drop(columns=["_rank"]).to_csv(out_dir / "or_long.csv", index=False)
    con.close()

    # --- baseline characteristics for this scheme ---------------------------
    final_df = pd.read_csv(cleaned, usecols=["age", "age_risk", "dom",
                                             "subsid", "n_preg"])
    print(f"      episodes after QC    : {len(final_df):>10,}")

    return {
        "gap": gap, "diag": diag, "washout": washout,
        "pre_qc_episodes": len(episodes),
        "age_rep": age_rep, "conf_rep": conf_rep, "seq_rep": seq_rep,
        "pregnancies": int(counts["pregnancies"]),
        "deliveries": int(counts["deliveries"]),
        "abortive": int(counts["abortive"]),
        "or_long": or_long, "baseline": final_df,
    }


NOT_ESTIMABLE = "\u2014"


def _or_text(r, suffix: str = "") -> str:
    """Format one estimate as it appears in the supplementary tables."""
    if str(r.get("estimability", "")).startswith("not estimable"):
        return NOT_ESTIMABLE
    o, lo, hi = r.get("odds_ratio"), r.get("ci_lower"), r.get("ci_upper")
    if pd.isna(o):
        return NOT_ESTIMABLE
    mark = "+" if str(r.get("sparse", "No")).lower() == "yes" else ""
    return f"{o:.3f} ({lo:.3f}\u2013{hi:.3f}){mark}"


def write_or_workbook(long: pd.DataFrame, sw: pd.DataFrame, out_dir: Path,
                      gaps: list[int]) -> None:
    """One sheet per category and phase, schemes side by side.

    The long-format file remains the machine-readable record; this workbook
    exists so the supplementary table can be assembled without reshaping a CSV
    by hand.
    """
    switch = {}
    if len(sw):
        for _, r in sw.iterrows():
            switch[r["var"]] = (bool(r["significance_switch"]),
                                bool(r["direction_switch"]))

    cells = {}
    order = []
    for _, r in long.iterrows():
        key = (r["sheet"], r["var"], r["condition"])
        if key not in cells:
            cells[key] = {}
            order.append(key)
        cells[key][r["gap_days"]] = _or_text(r, "")

    rows = []
    for sheet, var, cond in order:
        rec = {"sheet": sheet, "condition": cond}
        for g in gaps:
            rec[f"{g}"] = cells[(sheet, var, cond)].get(g, NOT_ESTIMABLE)
        sig, dirn = switch.get(var, (False, False))
        rec["significance switch"] = "Yes" if sig else "No"
        rec["direction switch"] = "Yes" if dirn else "No"
        rows.append(rec)

    tbl = pd.DataFrame(rows)
    path = out_dir / "OR by scheme.xlsx"
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


def qc_table(results: list[dict], out_dir: Path) -> pd.DataFrame:
    rows = []
    for r in results:
        g = r["gap"]
        rows.append({
            "QC category": "Episodes before quality control",
            f"{g}": r["pre_qc_episodes"]})
        rows.append({
            "QC category": "Age out of range",
            f"{g}": r["age_rep"]["removed_age_rule_episode"]})
        rows.append({
            "QC category": "Conflicting diagnosis",
            f"{g}": r["conf_rep"]["removed_diagnostic_conflict"]})
        for lab, k in (("Delivery re-coded (overlap)", "removed_overlap_delivery"),
                       ("Abortive code in an ongoing pregnancy (overlap)",
                        "removed_overlap_abortive"),
                       ("Abortive episode containing a delivery",
                        "removed_abortive_with_delivery")):
            rows.append({"QC category": lab, f"{g}": r["seq_rep"][k]})
        rows.append({
            "QC category": f"Removed by the washout year",
            f"{g}": r["washout"]})
        rows.append({
            "QC category": "Episodes in the analytic dataset",
            f"{g}": r["conf_rep"]["episodes_final"]})
    tbl = (pd.DataFrame(rows).groupby("QC category", sort=False).first()
           .reset_index())
    tbl.to_csv(out_dir / "sensitivity_qc_counts.csv", index=False)
    return tbl


def baseline_table(results: list[dict], out_dir: Path) -> pd.DataFrame:
    """Pregnancy-level baseline characteristics, gaps side by side."""
    npreg_vals = sorted({int(v) for r in results
                         for v in r["baseline"]["n_preg"].dropna().unique()})
    specs = [("Maternal age, mean \u00b1 SD", "meansd", None),
             ("Maternal age risk category, n(%)", "header", None),
             ("    20-35 years", "cat", ("age_risk", 0)),
             ("    <20 or >35 years", "cat", ("age_risk", 1)),
             ("Residence region, n(%)", "header", None),
             ("    Outside Java-Bali islands", "cat", ("dom", 1)),
             ("    Within Java-Bali islands", "cat", ("dom", 0)),
             ("Insurance subsidy status, n(%)", "header", None),
             ("    Subsidised", "cat", ("subsid", 1)),
             ("    Non-subsidised", "cat", ("subsid", 0)),
             ("Pregnancy order, n(%)", "header", None)]
    specs += [(f"    {k}", "cat", ("n_preg", k)) for k in npreg_vals]

    tbl = {"Characteristics": [s[0] for s in specs]}
    checks = []
    for r in results:
        d, N = r["baseline"], len(r["baseline"])
        tag = f"{r['gap']}" + (" (main)" if r["gap"] == MAIN_GAP else "")
        col, block = [], {}
        for lbl, kind, arg in specs:
            if kind == "header":
                col.append("")
            elif kind == "meansd":
                col.append(f"{d['age'].mean():.2f} \u00b1 {d['age'].std():.2f}")
            else:
                var, val = arg
                n = int((d[var] == val).sum())
                col.append(f"{n:,} ({100 * n / N:.2f})" if N else "0 (0.00)")
                block.setdefault(var, 0)
                block[var] += n
        tbl[f"{tag} (n={N:,})"] = col
        for var, s in block.items():
            checks.append({"gap": r["gap"], "block": var, "counts_sum": s,
                           "N": N, "matches": s == N, "shortfall": N - s})
    out = pd.DataFrame(tbl)
    out.to_csv(out_dir / "sensitivity_table1_pregnancy.csv", index=False)

    chk = pd.DataFrame(checks)
    bad = chk[~chk["matches"]]
    if len(bad):
        print("\n    [WARNING] blocks whose counts do not sum to N:")
        for _, r in bad.iterrows():
            print(f"      scheme {r['gap']} / {r['block']:<9} "
                  f"{r['counts_sum']:,} vs {r['N']:,} "
                  f"(shortfall {r['shortfall']:,})")
    else:
        print("\n    every baseline block sums to its column N")
    return out


def or_comparison(results: list[dict], out_dir: Path) -> tuple:
    """Odds ratios side by side, and the pairs that change significance."""
    long = pd.concat([r["or_long"] for r in results], ignore_index=True)
    long["significant"] = ~((long["ci_lower"] <= 1) & (long["ci_upper"] >= 1))
    long["above_one"] = long["odds_ratio"] > 1
    long["plottable"] = ~long["estimability"].astype(str).str.startswith(
        "not estimable")

    wide = long.pivot_table(index=["sheet", "var", "condition"],
                            columns="gap_days",
                            values=["odds_ratio", "ci_lower", "ci_upper"],
                            aggfunc="first")
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()
    wide.to_csv(out_dir / "sensitivity_or_comparison.csv", index=False)

    piv = long.pivot_table(index=["sheet", "var", "condition"],
                           columns="gap_days",
                           values=["significant", "above_one", "plottable"],
                           aggfunc="first")
    gaps = list(dict.fromkeys(long["gap_days"]))
    switches = []
    for idx, row in piv.iterrows():
        plott = [bool(row.get(("plottable", g), False)) for g in gaps]
        if COMPARE_ONLY_COMMON and not all(plott):
            continue
        sig = [bool(row.get(("significant", g), False)) for g in gaps]
        dirn = [bool(row.get(("above_one", g), False)) for g in gaps]
        if len(set(sig)) > 1 or len(set(dirn)) > 1:
            rec = {"sheet": idx[0], "var": idx[1], "condition": idx[2],
                   "significance_switch": len(set(sig)) > 1,
                   "direction_switch": len(set(dirn)) > 1}
            for g in gaps:
                rec[f"significant_{g}"] = row.get(("significant", g))
            switches.append(rec)
    sw = pd.DataFrame(switches)
    sw.to_csv(out_dir / "sensitivity_significance_switches.csv", index=False)

    n_common = int(sum(
        all(bool(row.get(("plottable", g), False)) for g in gaps)
        for _, row in piv.iterrows()))
    print(f"\n    associations estimable under every scheme : {n_common:,}")
    print(f"    changing significance across schemes      : "
          f"{int(sw['significance_switch'].sum()) if len(sw) else 0:,}")
    print(f"    changing direction across schemes         : "
          f"{int(sw['direction_switch'].sum()) if len(sw) else 0:,}")
    if len(sw):
        for _, r in sw.head(12).iterrows():
            flags = []
            if r["significance_switch"]:
                flags.append("significance")
            if r["direction_switch"]:
                flags.append("direction")
            print(f"      {r['var']:<16} {r['condition'][:40]:<40} "
                  f"{'+'.join(flags)}")
        if len(sw) > 12:
            print(f"      ... full list in "
                  f"sensitivity_significance_switches.csv")
    return wide, sw, n_common


def figures(results: list[dict], out_dir: Path) -> None:
    gaps = [r["gap"] for r in results]
    series = {
        "Pregnancies": [r["pregnancies"] for r in results],
        "Deliveries": [r["deliveries"] for r in results],
        "Abortive outcomes": [r["abortive"] for r in results],
    }
    for i, g in enumerate(gaps):
        p = series["Pregnancies"][i]
        s = series["Deliveries"][i] + series["Abortive outcomes"][i]
        if p != s:
            print(f"    [WARNING] scheme {g}: deliveries + abortive = {s:,} "
                  f"but pregnancies = {p:,} (difference {p - s:+,})")

    fig, ax = plt.subplots(figsize=(9.4, 6.4))
    x = range(len(gaps))
    for name, vals in series.items():
        ax.plot(x, vals, marker="o", markersize=9, lw=2,
                color=COLORS[name], label=name, zorder=3)
        for xi, v in zip(x, vals):
            ax.annotate(f"{v:,}", (xi, v), textcoords="offset points",
                        xytext=(0, 12), ha="center", fontsize=9)
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(g) for g in gaps])
    ax.set_xlim(-0.35, len(gaps) - 0.65)
    ax.set_ylim(0, max(series["Pregnancies"]) * 1.18)
    ax.set_xlabel("Episode separation rule")
    ax.set_ylabel("Count")
    ax.grid(True, ls=":", color="0.8", zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(out_dir / "scheme_sensitivity_counts.png", dpi=DPI)
    plt.close(fig)

    ref = gaps.index(MAIN_GAP) if MAIN_GAP in gaps else 0
    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    for name, vals in series.items():
        base = vals[ref]
        pct = [100 * (v - base) / base for v in vals]
        ax.plot(x, pct, marker="o", markersize=8, lw=2,
                color=COLORS[name], label=name, zorder=3)
        for xi, pv in zip(x, pct):
            ax.annotate(f"{pv:+.1f}%", (xi, pv), textcoords="offset points",
                        xytext=(0, 11 if pv >= 0 else -11), ha="center",
                        va="bottom" if pv >= 0 else "top", fontsize=9)
    ax.axhline(0, color="gray", ls="--", lw=1, zorder=2)
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(g) for g in gaps])
    ax.set_xlim(-0.35, len(gaps) - 0.65)
    ax.set_xlabel("Episode separation rule")
    ax.set_ylabel(f"Change vs {MAIN_GAP} (%)")
    ax.grid(True, axis="y", ls=":", color="0.8", zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(out_dir / "scheme_sensitivity_relative.png", dpi=DPI)
    plt.close(fig)
    print("    saved scheme_sensitivity_counts.png and "
          "scheme_sensitivity_relative.png")


# =============================================================================
# COPY-READY TABLES: MAIN-TEXT TABLE 3 AND SUPPLEMENTARY TABLES
# =============================================================================
# Built from the files this step writes (sensitivity_qc_counts.csv and, per
# scheme, the episode files and or_long.csv), so they can be rebuilt without
# rerunning the schemes:   python 11_sensitivity_gap.py --report-only
#
# Sheets in Sensitivity_tables.xlsx
#   Table 3            episodes removed at each stage under each rule, with the
#                      recorded abortive proportion before and after QC
#   Baseline by rule   pregnancy-level characteristics under each rule
#   OR by rule         every association under each rule, in the order of the
#                      condition list (ICD-10), never alphabetical
# "Estimable" here means what it means elsewhere in the paper: no empty cell
# and no cell below five. Flagged estimates are shown with a superscript a.

OUT_TABLES = "Sensitivity_tables.xlsx"
FLAG = "\u1d43"
EN = "\u2013"
GE = "\u2265"
SHEET_ORDER = ["characteristics", "b_chronic", "b_infectious", "b_pregnancy",
               "c_infectious", "c_pregnancy", "a_chronic", "a_infectious",
               "a_pregnancy"]
PERIOD = {"b": "Before pregnancy", "c": "During pregnancy", "a": "After pregnancy"}
GROUP = {"chronic": "Chronic conditions", "infectious": "Infectious diseases",
         "pregnancy": "Pregnancy-related conditions"}
CHAR_LABEL = {"age_risk": f"Maternal age <20 or >35 years (vs 20{EN}35)",
              "dom": f"Residence outside Java{EN}Bali (vs within)",
              "subsid": "Subsidised membership (vs non-subsidised)"}
UK_SPELLING = {"Hemorrhage": "Haemorrhage", "hemorrhage": "haemorrhage",
               "Edema": "Oedema", "edema": "oedema", "Anemia": "Anaemia",
               "anemia": "anaemia", "Subsidized": "Subsidised",
               "subsidized": "subsidised", "Esophag": "Oesophag",
               "esophag": "oesophag", "Hematolog": "Haematolog",
               "hematolog": "haematolog", "Leukemia": "Leukaemia",
               "leukemia": "leukaemia", "Diarrhea": "Diarrhoea",
               "diarrhea": "diarrhoea", "Tumor": "Tumour", "tumor": "tumour"}


def _uk(t: str) -> str:
    for us, gb in UK_SPELLING.items():
        t = t.replace(us, gb)
    return t


def _n(x) -> str:
    """IJE number style: no separator up to 9999, a space from 10 000."""
    x = int(round(float(x)))
    return f"{x:,}".replace(",", " ") if abs(x) >= 10000 else str(x)


def _n_pct(n, d) -> str:
    return f"{_n(n)} ({100 * n / d:.1f})" if d else _n(n)


def _or_cell(r) -> str:
    if r is None or str(r["estimability"]) not in ("estimable", "sparse") \
            or pd.isna(r["odds_ratio"]):
        return EN
    s = f"{r['odds_ratio']:.2f} ({r['ci_lower']:.2f}{EN}{r['ci_upper']:.2f})"
    return s + FLAG if r["estimability"] == "sparse" else s


def _scheme_dirs(out_dir: Path) -> list[tuple[str, Path]]:
    return [(g, out_dir / f"scheme_{folder}") for g, (folder, _) in SCHEMES.items()]


def _short(g: str) -> str:
    return "182-day rule (main analysis)" if g == MAIN_GAP else \
        "259/105-day rule (by type of later record)"


def table3(out_dir: Path) -> pd.DataFrame:
    qc = pd.read_csv(out_dir / "sensitivity_qc_counts.csv").set_index("QC category")
    rows = []
    cols = {}
    for g, d in _scheme_dirs(out_dir):
        c = qc[g]
        pre = pd.read_csv(d / "pregnancy by episode.csv", usecols=["c_abortive"])
        post = pd.read_csv(d / "pregnancy by episode cleaned.csv", usecols=["c_abortive"])
        a_pre, a_post = int((pre["c_abortive"] == 1).sum()), int((post["c_abortive"] == 1).sum())
        n_in, n_out = c["Episodes before quality control"], c["Episodes in the analytic dataset"]
        seq = sum(c[k] for k in ("Delivery re-coded (overlap)",
                                 "Abortive code in an ongoing pregnancy (overlap)",
                                 "Abortive episode containing a delivery"))
        cols[g] = [
            _n(n_in + c["Removed by the washout year"]),
            _n(c["Removed by the washout year"]),
            _n(n_in),
            _n(n_in - n_out),
            _n(c["Age out of range"]),
            _n(c["Conflicting diagnosis"]),
            _n(seq),
            _n(c["Delivery re-coded (overlap)"]),
            _n(c["Abortive code in an ongoing pregnancy (overlap)"]),
            _n(c["Abortive episode containing a delivery"]),
            _n(n_out),
            _n_pct(a_pre, n_in),
            _n_pct(a_post, n_out),
        ]
        if a_pre + 0 != len(pre[pre["c_abortive"] == 1]) or len(post) != n_out:
            print(f"    [WARNING] scheme {g}: episode files and QC counts disagree")
    labels = ["Episodes reconstructed",
              "Removed by the washout (first-order episodes ending in 2015)",
              "Episodes entering quality control",
              "Removed by quality control",
              "    Implausible maternal age",
              "    Diagnostic conflict",
              "    Implausible sequence",
              "        Delivery with onset before the previous delivery",
              "        Abortive code within an ongoing pregnancy",
              "        Abortive episode containing a later delivery",
              "Episodes in the analytic dataset",
              "Abortive episodes before quality control, n (%)",
              "Abortive episodes after quality control, n (%)"]
    t = pd.DataFrame({"Stage": labels})
    for g in cols:
        t[_short(g)] = cols[g]
    return t


def baseline_by_rule(out_dir: Path) -> pd.DataFrame:
    out = {"Characteristic": []}
    first = True
    for g, d in _scheme_dirs(out_dir):
        b = pd.read_csv(d / "pregnancy by episode cleaned.csv",
                        usecols=["age", "age_risk", "dom", "subsid", "n_preg"])
        N = len(b)
        npreg = b["n_preg"].clip(upper=5)
        spec = [
            ("Maternal age, years, mean (SD)", f"{b['age'].mean():.1f} ({b['age'].std():.1f})"),
            ("Maternal age group, n (%)", ""),
            (f"    20{EN}35 years", _n_pct((b["age_risk"] == 0).sum(), N)),
            ("    <20 or >35 years", _n_pct((b["age_risk"] == 1).sum(), N)),
            ("Residence, n (%)", ""),
            (f"    Outside Java{EN}Bali", _n_pct((b["dom"] == 1).sum(), N)),
            (f"    Within Java{EN}Bali", _n_pct((b["dom"] == 0).sum(), N)),
            ("Insurance membership, n (%)", ""),
            ("    Subsidised", _n_pct((b["subsid"] == 1).sum(), N)),
            ("    Non-subsidised", _n_pct((b["subsid"] == 0).sum(), N)),
            ("Pregnancy order, n (%)", ""),
        ] + [("    " + (GE if k == 5 else "") + str(k), _n_pct((npreg == k).sum(), N))
             for k in range(1, 6)]
        if first:
            out["Characteristic"] = [s[0] for s in spec]
            first = False
        out[f"{_short(g)} (n = {_n(N)} episodes)"] = [s[1] for s in spec]
    return pd.DataFrame(out)


def or_by_rule(out_dir: Path) -> tuple[pd.DataFrame, dict]:
    gaps = list(SCHEMES)
    frames = {g: pd.read_csv(d / "or_long.csv") for g, d in _scheme_dirs(out_dir)}
    main = frames[MAIN_GAP].copy()
    main["_p"] = range(len(main))
    main["_o"] = main["sheet"].map({s: i for i, s in enumerate(SHEET_ORDER)})
    main = main.sort_values(["_o", "_p"], kind="stable")
    look = {g: {(r["sheet"], r["var"]): r for _, r in f.iterrows()} for g, f in frames.items()}
    rows, n_both, n_dir, n_ci = [], 0, 0, 0
    for _, m in main.iterrows():
        key = (m["sheet"], m["var"])
        rr = [look[g].get(key) for g in gaps]
        both = all(r is not None and r["estimability"] == "estimable" for r in rr)
        d = c = None
        if both:
            n_both += 1
            above = [r["odds_ratio"] > 1 for r in rr]
            excl = [(r["ci_lower"] > 1) or (r["ci_upper"] < 1) for r in rr]
            d, c = len(set(above)) > 1, len(set(excl)) > 1
            n_dir += d; n_ci += c
        sh = m["sheet"]
        cond = CHAR_LABEL.get(m["var"], _uk(str(m["condition"]))) if sh == "characteristics" \
            else _uk(str(m["condition"]))
        rec = {"Period": "Baseline characteristics" if sh == "characteristics" else PERIOD[sh[0]],
               "Diagnostic group": "" if sh == "characteristics" else GROUP[sh.split("_", 1)[1]],
               "Condition": cond}
        for g, r in zip(gaps, rr):
            rec[f"OR (95% CI), {_short(g).split(' (')[0]}"] = _or_cell(r)
        rec["Direction reversed"] = EN if d is None else ("Yes" if d else "No")
        rec["CI moved across 1"] = EN if c is None else ("Yes" if c else "No")
        rows.append(rec)
    summ = {"estimable_under_both": n_both, "direction_reversed": n_dir,
            "ci_moved_across_1": n_ci}
    return pd.DataFrame(rows), summ


T3_TITLE = ("Pregnancy episodes removed at each stage under the main and alternative "
            "separation rules")
T3_NOTES = [
    "Main rule: a termination opened a new episode at least 182 days after the preceding "
    "termination record. Alternative rule: at least 259 days before a delivery record and "
    "105 days before an abortive record.",
    "Percentages are of episodes entering quality control (before) and of episodes in the "
    "analytic dataset (after).",
]
BASE_TITLE = "Characteristics of the pregnancy-level dataset under each separation rule"
BASE_NOTES = [
    "SD, standard deviation.",
    "Pregnancy order is the position of the episode in the woman's reconstructed sequence "
    "of pregnancies observed from 2015 to 2022, not parity; episode numbers are not "
    "reassigned after the washout or quality control.",
]
OR_TITLE = ("Pregnancy-level odds ratios for abortive outcome under the main and "
            "alternative separation rules")
OR_NOTES = [
    "CI, confidence interval; OR, odds ratio.",
    f"Crude ORs with log-Wald 95% CIs. {FLAG} Any cell of the 2x2 table below five "
    f"observations. {EN}: not estimable or not applicable. Direction reversed: OR on "
    "opposite sides of 1 under the two rules; CI moved across 1: 95% CI included 1 under "
    "one rule and excluded it under the other. Both are shown only for associations "
    "estimable under both rules.",
]


def _write_sheet(ws, df: pd.DataFrame, title: str, notes: list[str]) -> None:
    from openpyxl.styles import Alignment, Border, Font, Side
    thin = Side(style="thin", color="000000")
    f, fb = Font(name="Arial", size=10), Font(name="Arial", size=10, bold=True)
    ws.cell(row=1, column=1, value=title).font = fb
    for j, col in enumerate(df.columns, 1):
        c = ws.cell(row=3, column=j, value=col)
        c.font = fb
        c.alignment = Alignment(wrap_text=True, vertical="bottom")
        c.border = Border(top=thin, bottom=thin)
    for i, row in enumerate(df.itertuples(index=False), 4):
        for j, v in enumerate(row, 1):
            c = ws.cell(row=i, column=j, value=None if v == "" else v)
            c.font = f
            c.alignment = Alignment(wrap_text=j == 1, vertical="top",
                                    horizontal="left" if j == 1 else "center")
    last = 3 + len(df)
    for j in range(1, len(df.columns) + 1):
        ws.cell(row=last, column=j).border = Border(bottom=thin)
    for k, n in enumerate(notes):
        ws.cell(row=last + 2 + k, column=1, value=n).font = Font(name="Arial", size=9)
    for j, col in enumerate(df.columns, 1):
        w = max([len(str(col)) * 0.55] + [len(str(v)) for v in df.iloc[:, j - 1]])
        ws.column_dimensions[ws.cell(row=3, column=j).column_letter].width = min(max(w + 2, 12), 60)
    ws.freeze_panes = "A4"


def write_tables(out_dir: Path) -> None:
    from openpyxl import Workbook
    needed = [out_dir / "sensitivity_qc_counts.csv"] + \
        [d / f for _, d in _scheme_dirs(out_dir)
         for f in ("pregnancy by episode.csv", "pregnancy by episode cleaned.csv", "or_long.csv")]
    missing = [str(p) for p in needed if not p.exists()]
    if missing:
        print("    [skip] tables not written; missing:\n      " + "\n      ".join(missing))
        return
    wb = Workbook()
    ws = wb.active; ws.title = "Table 3"
    _write_sheet(ws, table3(out_dir), T3_TITLE, T3_NOTES)
    _write_sheet(wb.create_sheet("Baseline by rule"), baseline_by_rule(out_dir),
                 BASE_TITLE, BASE_NOTES)
    ort, summ = or_by_rule(out_dir)
    _write_sheet(wb.create_sheet("OR by rule"), ort, OR_TITLE, OR_NOTES)
    wb.save(out_dir / OUT_TABLES)
    print(f"    wrote {OUT_TABLES}: {', '.join(wb.sheetnames)}")
    print(f"    estimable (no cell <5) under both rules : {summ['estimable_under_both']}")
    print(f"      direction reversed                    : {summ['direction_reversed']}")
    print(f"      CI moved across 1                     : {summ['ci_moved_across_1']}")
    pd.DataFrame([summ]).to_csv(out_dir / "sensitivity_switch_summary.csv", index=False)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 74)
    print(" STEP 11 -- SENSITIVITY TO THE EPISODE SEPARATION RULE")
    print("=" * 74)
    print(environment_report(str(OUT_DIR / "environment_step11.txt")))
    warn_if_no_psutil()

    if not VISIT_CSV.exists():
        sys.exit(f"[FATAL] input not found: {VISIT_CSV.resolve()}")

    bench = Bench("step 11, separation-rule sensitivity")

    with bench.stage("Import the main-analysis modules", "python") as b:
        s1, s4, s5, s7 = load_steps()
        b["rows_out"] = 4

    with bench.stage("Load complete visit-level table", "pandas") as b:
        visit_raw = pd.read_csv(VISIT_CSV, low_memory=False)
        print(f"    {len(visit_raw):,} rows x {len(visit_raw.columns)} columns")
        b["rows_out"] = len(visit_raw)

    results = []
    for gap, (folder, sep) in SCHEMES.items():
        with bench.stage(f"Rebuild and analyse, scheme {gap}", "pandas + DuckDB",
                         rows_in=len(visit_raw)) as b:
            r = build_for_gap(visit_raw, gap, sep, s1, s4, s5, s7,
                              OUT_DIR / f"scheme_{folder}")
            results.append(r)
            b["rows_out"] = r["conf_rep"]["episodes_final"]

    with bench.stage("Assemble comparison tables and figures", "pandas") as b:
        qc = qc_table(results, OUT_DIR)
        print("\n    QUALITY CONTROL BY SCHEME")
        print(qc.to_string(index=False))
        baseline_table(results, OUT_DIR)
        wide, sw, n_common = or_comparison(results, OUT_DIR)
        long_all = pd.concat([r["or_long"] for r in results], ignore_index=True)
        write_or_workbook(long_all, sw, OUT_DIR, [r["gap"] for r in results])
        print()
        figures(results, OUT_DIR)
        write_tables(OUT_DIR)
        b["rows_out"] = len(wide)

    bench.finalise(str(OUT_DIR / "benchmark_step11.csv"),
                   rows_out=sum(r["conf_rep"]["episodes_final"]
                                for r in results))
    print(f"\nOutput written to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    if "--report-only" in sys.argv:
        write_tables(OUT_DIR)
    else:
        main()