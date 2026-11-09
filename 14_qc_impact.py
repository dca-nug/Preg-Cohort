"""
=============================================================================
 STEP 14 -- IMPACT OF EACH QUALITY-CONTROL RULE
=============================================================================

Recomputes the pregnancy-level odds ratios after each QC rule, in the order
step 4 applies them:

    stage 0   before QC                 step 3 output
    stage 1   after age exclusion       step 4 age_exclusion
    stage 2   after conflict exclusion  step 4 conflict_exclusion
    stage 3   after sequence exclusion  step 4 sequence_exclusion
                                        (= the analytic dataset of step 4)

The QC functions are imported from step 4 and the odds-ratio functions from
step 8, so this script cannot describe a rule or an estimator other than the
ones the main analysis applied.

It also assembles the QC yield of each rule in both systems: BPJS from this
run and step 3's washout report, TMUCRD from the reports written by step 13.

OUTPUT
------
    or_by_qc_stage.csv       every odds ratio at every stage (long format)
    qc_switches.csv          per variable, stage 0 versus stage 3: direction
                             and significance switches, OR ratio
    qc_stage_counts.csv      episodes and abortive episodes per stage
    qc_yield_by_system.csv   episodes removed by each rule, BPJS and TMUCRD
    Figure_qc_or_before_after.png
                             scatter of ORs before vs after QC (main-text figure)
    Supplementary_tables_qc.xlsx
                             copy-ready supplementary tables: QC summary, all
                             associations, ORs by stage, stage counts, yield

USAGE
-----
    python 14_qc_impact.py                 full run
    python 14_qc_impact.py --report-only   rebuild figure and workbook from
                                           the CSVs already in OUT_DIR
"""

import importlib
import os
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

os.chdir(Path(__file__).resolve().parent)
sys.path.insert(0, os.getcwd())

# =============================================================================
# CONFIGURATION
# =============================================================================

STEP3_CSV      = r"./output_step3/pregnancy by episode.csv"
STEP3_WASHOUT  = r"./output_step3/washout_report_step3.csv"
STEP13_DIR     = Path(r"./output_step13")
OUT_DIR        = Path(r"./output_step14")

STAGES = ["0 before QC", "1 after age", "2 after conflict", "3 after sequence"]

# =============================================================================


def sheets_for(con, s7) -> dict:
    """Same sheet and column resolution as step 8."""
    lut = {c.lower(): c for c in
           con.sql("SELECT name FROM pragma_table_info('final')").df()["name"]}
    sheets = {}
    for cat, conds in [("chronic", s7.CHRONIC), ("infectious", s7.INFECTIOUS),
                       ("pregnancy", s7.PREGNANCY)]:
        for phase in s7.PHASE_MAP[cat]:
            sheets[f"{phase.strip('_')}_{cat}"] = [
                lut[f"{phase}{c}".lower()] for c in conds
                if f"{phase}{c}".lower() in lut]
    trio = [lut[c] for c, _ in s7.TRIO_ORDER if c in lut]
    if trio:
        sheets = {"characteristics": trio, **sheets}
    return sheets


def or_at_stage(con, s7, stage: str) -> tuple[pd.DataFrame, dict]:
    frames = []
    for name, cols in sheets_for(con, s7).items():
        if cols:
            res = s7.annotate(s7.calc_or(con, cols))
            res.insert(0, "sheet", name)
            frames.append(res)
    out = pd.concat(frames, ignore_index=True)
    out.insert(0, "stage", stage)
    n, n_ab = con.sql(f"""
        SELECT COUNT(*), SUM(CASE WHEN COALESCE("{s7.OUTCOME_COL}",0)=1
                                  THEN 1 ELSE 0 END) FROM final
    """).fetchone()
    print(f"    {stage:<20} episodes {n:>9,}   abortive {int(n_ab or 0):>8,}")
    return out, {"stage": stage, "episodes": n, "abortive": int(n_ab or 0)}


def switches(long: pd.DataFrame) -> pd.DataFrame:
    keep = ["sheet", "var", "condition", "odds_ratio", "ci_lower", "ci_upper",
            "significance", "estimability"]
    a = long.loc[long["stage"] == STAGES[0], keep]
    b = long.loc[long["stage"] == STAGES[-1], keep]
    m = a.merge(b, on=["sheet", "var", "condition"], suffixes=("_pre", "_post"))
    both = m["odds_ratio_pre"].notna() & m["odds_ratio_post"].notna()
    m["direction_switch"] = both & (((m["odds_ratio_pre"] - 1)
                                     * (m["odds_ratio_post"] - 1)) < 0)
    m["significance_switch"] = both & (m["significance_pre"]
                                       != m["significance_post"])
    m["or_ratio_post_over_pre"] = (m["odds_ratio_post"]
                                   / m["odds_ratio_pre"]).where(both)
    m["estimability_changed"] = m["estimability_pre"] != m["estimability_post"]
    return m


def read_metrics(path) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    d = pd.read_csv(p)
    return dict(zip(d["metric"], d["value"]))


def yield_table(counts: dict, seq_rep: dict, age_rep: dict,
                conf_rep: dict) -> pd.DataFrame:
    w = read_metrics(STEP3_WASHOUT)
    bpjs = {
        "Removed by the washout year": w.get("removed_by_washout"),
        "Episodes entering QC": age_rep["episodes_total"],
        "Age out of range": age_rep["removed_age_rule_episode"],
        "Conflicting diagnosis": conf_rep["removed_diagnostic_conflict"],
        "Delivery re-coded (overlap)": seq_rep["removed_overlap_delivery"],
        "Abortive code in an ongoing pregnancy (overlap)":
            seq_rep["removed_overlap_abortive"],
        "Abortive episode containing a delivery":
            seq_rep["removed_abortive_with_delivery"],
        "Episodes in the analytic dataset": counts[STAGES[-1]],
    }
    a = read_metrics(STEP13_DIR / "external_age_qc_report.csv")
    c = read_metrics(STEP13_DIR / "external_qc_report.csv")
    s = read_metrics(STEP13_DIR / "external_sequence_qc_report.csv")
    tmu = {
        "Removed by the washout year": None,
        "Episodes entering QC": a.get("episodes_total"),
        "Age out of range": a.get("removed_age_rule_episode"),
        "Conflicting diagnosis": c.get("removed_diagnostic_conflict"),
        "Delivery re-coded (overlap)": s.get("removed_overlap_delivery"),
        "Abortive code in an ongoing pregnancy (overlap)":
            s.get("removed_overlap_abortive"),
        "Abortive episode containing a delivery":
            s.get("removed_abortive_with_delivery"),
        "Episodes in the analytic dataset": s.get("episodes_final"),
    }
    if not (a or c or s):
        print(f"    [note] no step 13 reports in {STEP13_DIR}; TMUCRD column empty")
    t = pd.DataFrame({"QC rule": list(bpjs), "BPJS Kesehatan": list(bpjs.values()),
                      "TMUCRD": [tmu[k] for k in bpjs]})
    for col in ("BPJS Kesehatan", "TMUCRD"):
        base = t.loc[t["QC rule"] == "Episodes entering QC", col].iloc[0]
        t[f"{col} %"] = (pd.to_numeric(t[col], errors="coerce") / base * 100
                         if base else None)
    return t


# =============================================================================
# REPORTING: FIGURE AND SUPPLEMENTARY-TABLE WORKBOOK
# =============================================================================
# Both are built from the CSVs written above, so they can be regenerated
# without rerunning the pipeline:   python 14_qc_impact.py --report-only

FIG_NAME  = "Figure_qc_or_before_after.png"
XLSX_NAME = "Supplementary_tables_qc.xlsx"
FIG_DPI   = 770            # 12 cm square at 770 dpi = 3637 px, above IJE's 3600 px

# Points to label: the smallest and largest ratio of ORs, and every reversal
# whose confidence interval excludes 1 both before and after QC. Offsets are
# in points from the marker; override per condition if labels collide.
LABEL_OFFSETS = {}         # e.g. {"Excessive vomiting in pregnancy": (60, -40)}

PERIOD = {"b": "Before pregnancy", "c": "During pregnancy", "a": "After pregnancy"}
GROUP  = {"chronic": "Chronic conditions", "infectious": "Infectious diseases",
          "pregnancy": "Pregnancy-related conditions"}
EST_LABEL = {"estimable": "Estimable",
             "sparse": "Flagged (cell <5)",
             "not estimable (empty cell)": "Not estimable (empty cell)",
             "not estimable by design": "Not estimable by design",
             "not estimable (outcome versus itself)": "Not estimable (outcome itself)"}
STAGE_LABEL = {"0 before QC": "Before QC", "1 after age": "After age exclusion",
               "2 after conflict": "After conflict exclusion",
               "3 after sequence": "After sequence exclusion (analytic dataset)"}
SHEET_ORDER = ["characteristics", "b_chronic", "b_infectious", "b_pregnancy",
               "c_infectious", "c_pregnancy", "a_chronic", "a_infectious",
               "a_pregnancy"]


# The pipeline's condition labels use US spelling; IJE uses UK English.
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


def _period(sheet: str) -> str:
    return "Baseline characteristics" if sheet == "characteristics" else PERIOD[sheet[0]]


def _group(sheet: str) -> str:
    return "" if sheet == "characteristics" else GROUP[sheet.split("_", 1)[1]]


def _fmt_or(o, lo, hi) -> str:
    if pd.isna(o):
        return "\u2013"
    if pd.isna(lo) or pd.isna(hi):
        return f"{o:.2f}"
    return f"{o:.2f} ({lo:.2f}\u2013{hi:.2f})"


def _excludes_1(lo, hi) -> bool:
    return (lo > 1) or (hi < 1)


def estimable_both(sw: pd.DataFrame) -> pd.DataFrame:
    """Associations estimable (not flagged, not empty) before and after QC."""
    return sw[(sw["estimability_pre"] == "estimable")
              & (sw["estimability_post"] == "estimable")].copy()


def qc_figure(sw: pd.DataFrame, path: Path) -> None:
    import textwrap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    b = estimable_both(sw)
    b["period"] = b["sheet"].map(_period)
    b["group"] = b["sheet"].map(lambda s: "Baseline" if s == "characteristics"
                                else s.split("_", 1)[1])
    col = {"Before pregnancy": "#0072B2", "During pregnancy": "#D55E00",
           "After pregnancy": "#009E73", "Baseline characteristics": "#555555"}
    mk = {"chronic": "o", "infectious": "^", "pregnancy": "s", "Baseline": "D"}

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, ax = plt.subplots(figsize=(12 / 2.54, 12 / 2.54))
    lo = 10 ** np.floor(np.log10(min(b.odds_ratio_pre.min(), b.odds_ratio_post.min())))
    hi = 1.5 * max(b.odds_ratio_pre.max(), b.odds_ratio_post.max())
    ax.fill_between([lo, 1], 1, hi, color="#eeeeee", zorder=0)
    ax.fill_between([1, hi], lo, 1, color="#eeeeee", zorder=0)
    ax.plot([lo, hi], [lo, hi], color="black", lw=0.8, ls="--", zorder=1)
    ax.axhline(1, color="black", lw=0.6, zorder=1)
    ax.axvline(1, color="black", lw=0.6, zorder=1)

    for p in ["Before pregnancy", "After pregnancy", "Baseline characteristics",
              "During pregnancy"]:
        for g, m in mk.items():
            d = b[(b.period == p) & (b.group == g)]
            if len(d):
                ax.scatter(d.odds_ratio_pre, d.odds_ratio_post, s=16, marker=m,
                           c=col[p], alpha=0.75, edgecolors="white",
                           linewidths=0.3, zorder=3)

    r = b["or_ratio_post_over_pre"]
    robust = b[b["direction_switch"]
               & b.apply(lambda x: _excludes_1(x.ci_lower_pre, x.ci_upper_pre)
                         and _excludes_1(x.ci_lower_post, x.ci_upper_post), axis=1)]
    lab = pd.concat([b.loc[[r.idxmin(), r.idxmax()]], robust]).drop_duplicates(
        subset=["sheet", "var"])
    for _, x in lab.iterrows():
        name = uk(str(x["condition"])).replace(" in pregnancy", "")
        text = "\n".join(textwrap.wrap(name, 22))
        off = LABEL_OFFSETS.get(x["condition"],
                                (45, -12) if x.or_ratio_post_over_pre < 1 else (-110, -12))
        ax.annotate(text, (x.odds_ratio_pre, x.odds_ratio_post), xytext=off,
                    textcoords="offset points", fontsize=7, zorder=4,
                    arrowprops=dict(arrowstyle="-", lw=0.5))

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ticks = [t for t in [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100] if lo <= t <= hi]
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.set_yticklabels([f"{t:g}" for t in ticks])
    ax.minorticks_off(); ax.set_aspect("equal")
    ax.set_xlabel("Odds ratio before quality control (log scale)")
    ax.set_ylabel("Odds ratio after quality control (log scale)")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    h = [Line2D([], [], ls="", marker="o", color=col[p], label=p) for p in
         ["Before pregnancy", "During pregnancy", "After pregnancy",
          "Baseline characteristics"]]
    h += [Line2D([], [], ls="", marker=mk[g], color="#777777", mfc="white",
                 label=GROUP[g].split()[0]) for g in ["chronic", "infectious", "pregnancy"]]
    ax.legend(handles=h, loc="upper left", fontsize=7, frameon=False,
              handletextpad=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)
    print(f"    figure: {len(b)} points, labelled {len(lab)} -> {path}")


def summary_table(sw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sh in SHEET_ORDER:
        d = sw[sw["sheet"] == sh]
        if d.empty:
            continue
        e = estimable_both(d)
        q = e["or_ratio_post_over_pre"].quantile([.25, .5, .75])
        rows.append({
            "Period": _period(sh), "Diagnostic group": _group(sh),
            "Associations, n": len(d),
            "Estimable before and after QC, n": len(e),
            "Estimability changed, n": int(d["estimability_changed"].sum()),
            "Direction reversed, n": int(e["direction_switch"].sum()),
            "CI moved across 1, n": int(e["significance_switch"].sum()),
            "Ratio of ORs after to before QC, median (IQR)":
                f"{q[.5]:.2f} ({q[.25]:.2f}\u2013{q[.75]:.2f})" if len(e) else "\u2013"})
    t = pd.DataFrame(rows)
    tot = {c: (t[c].sum() if c.endswith(", n") else "") for c in t.columns}
    tot.update({"Period": "Total", "Diagnostic group": "",
                "Ratio of ORs after to before QC, median (IQR)": "\u2013"})
    return pd.concat([t, pd.DataFrame([tot])], ignore_index=True)


def full_table(sw: pd.DataFrame) -> pd.DataFrame:
    d = sw.copy()
    # Keep the definition order within each period (step 3 list, by ICD-10),
    # never alphabetical.
    d["_p"] = range(len(d))
    d["_o"] = d["sheet"].map({s: i for i, s in enumerate(SHEET_ORDER)})
    d = d.sort_values(["_o", "_p"], kind="stable")
    est = (d["estimability_pre"] == "estimable") & (d["estimability_post"] == "estimable")
    yn = lambda s: s.map({True: "Yes", False: "No"}).where(est, "\u2013")
    return pd.DataFrame({
        "Period": d["sheet"].map(_period),
        "Diagnostic group": d["sheet"].map(_group),
        "Condition": d["condition"].astype(str).map(uk),
        "OR (95% CI) before QC": [_fmt_or(*v) for v in
                                  d[["odds_ratio_pre", "ci_lower_pre", "ci_upper_pre"]].values],
        "OR (95% CI) after QC": [_fmt_or(*v) for v in
                                 d[["odds_ratio_post", "ci_lower_post", "ci_upper_post"]].values],
        "Ratio of ORs after to before QC": d["or_ratio_post_over_pre"].where(est)
            .map(lambda v: "\u2013" if pd.isna(v) else f"{v:.2f}"),
        "Estimability before QC": d["estimability_pre"].map(EST_LABEL),
        "Estimability after QC": d["estimability_post"].map(EST_LABEL),
        "Direction reversed": yn(d["direction_switch"]),
        "CI moved across 1": yn(d["significance_switch"]),
    })


def stage_table(long: pd.DataFrame) -> pd.DataFrame:
    long = long.copy()
    long["cell"] = [_fmt_or(*v) for v in long[["odds_ratio", "ci_lower", "ci_upper"]].values]
    w = long.pivot_table(index=["sheet", "var", "condition"], columns="stage",
                         values="cell", aggfunc="first").reset_index()
    # pivot_table sorts rows alphabetically; restore the order in which the
    # conditions are defined (step 3 list, by ICD-10 chapter and code).
    first = long.drop_duplicates(["sheet", "var"]).reset_index(drop=True)
    pos = {k: i for i, k in enumerate(zip(first["sheet"], first["var"]))}
    w["_p"] = [pos[(a, v)] for a, v in zip(w["sheet"], w["var"])]
    w["_o"] = w["sheet"].map({s: i for i, s in enumerate(SHEET_ORDER)})
    w = w.sort_values(["_o", "_p"], kind="stable")
    names = STAGE_LABEL
    out = pd.DataFrame({"Period": w["sheet"].map(_period),
                        "Diagnostic group": w["sheet"].map(_group),
                        "Condition": w["condition"].astype(str).map(uk)})
    for s in STAGES:
        out[f"OR (95% CI), {names[s][0].lower() + names[s][1:]}"] = w[s].values
    return out


NOTES = {
    "summary": [
        "CI, confidence interval; IQR, interquartile range; OR, odds ratio; QC, quality control.",
        "Estimable: no cell of the 2x2 table empty or below five observations.",
        "Direction reversed: OR on opposite sides of 1 before and after QC; CI moved across 1: "
        "95% CI included 1 at one stage and excluded it at the other. Both, and the ratio of "
        "ORs, are calculated among associations estimable before and after QC.",
        "Chronic conditions were classified only before or after pregnancy.",
        "After QC, the conflict indicators cannot co-occur with an abortive outcome during "
        "pregnancy and are not estimable by design.",
    ],
    "full": [
        "CI, confidence interval; OR, odds ratio; QC, quality control.",
        "Pregnancy-level crude ORs for abortive outcome, log-Wald 95% CI. Flagged: any cell of "
        "the 2x2 table below five observations. Ratio of ORs and the two switch columns are "
        "shown only for associations estimable before and after QC.",
    ],
    "stage": [
        "CI, confidence interval; OR, odds ratio; QC, quality control.",
        "ORs recomputed after each exclusion, in the order applied in the analysis. "
        "\u2013, not estimable.",
    ],
}


def _write_sheet(ws, df: pd.DataFrame, title: str, notes: list[str]) -> None:
    from openpyxl.styles import Alignment, Border, Font, Side
    thin = Side(style="thin", color="000000")
    f, fb = Font(name="Arial", size=10), Font(name="Arial", size=10, bold=True)
    ws.cell(row=1, column=1, value=title).font = fb
    for j, c in enumerate(df.columns, 1):
        cell = ws.cell(row=3, column=j, value=c)
        cell.font = fb
        cell.alignment = Alignment(wrap_text=True, vertical="bottom")
        cell.border = Border(top=thin, bottom=thin)
    for i, row in enumerate(df.itertuples(index=False), 4):
        for j, v in enumerate(row, 1):
            v = None if (isinstance(v, float) and np.isnan(v)) else v
            v = int(v) if isinstance(v, (np.integer,)) else v
            cell = ws.cell(row=i, column=j, value=v)
            cell.font = fb if str(row[0]) == "Total" else f
            cell.alignment = Alignment(wrap_text=True, vertical="top")
    last = 3 + len(df)
    for j in range(1, len(df.columns) + 1):
        ws.cell(row=last, column=j).border = Border(bottom=thin)
    for k, n in enumerate(notes):
        ws.cell(row=last + 2 + k, column=1, value=n).font = Font(name="Arial", size=9)
    for j, c in enumerate(df.columns, 1):
        width = max([len(str(c)) * 0.6] + [len(str(v)) for v in df.iloc[:, j - 1]])
        ws.column_dimensions[ws.cell(row=3, column=j).column_letter].width = \
            min(max(width + 2, 10), 48)
    ws.freeze_panes = "A4"


def supplementary_workbook(sw: pd.DataFrame, long: pd.DataFrame,
                           counts: pd.DataFrame, yt: pd.DataFrame | None,
                           path: Path) -> None:
    from openpyxl import Workbook
    wb = Workbook()
    ws = wb.active; ws.title = "QC summary"
    _write_sheet(ws, summary_table(sw),
                 "Change in pregnancy-level odds ratios for abortive outcome after "
                 "quality control, by period and diagnostic group", NOTES["summary"])
    _write_sheet(wb.create_sheet("QC all associations"), full_table(sw),
                 "Pregnancy-level odds ratios for abortive outcome before and after "
                 "quality control", NOTES["full"])
    _write_sheet(wb.create_sheet("QC by stage"), stage_table(long),
                 "Pregnancy-level odds ratios for abortive outcome after each "
                 "quality-control exclusion", NOTES["stage"])
    c = counts.assign(stage=counts["stage"].map(STAGE_LABEL).fillna(counts["stage"]))
    c = c.rename(columns={"stage": "Stage", "episodes": "Episodes, n",
                               "abortive": "Abortive episodes, n"})
    c["Abortive, %"] = (c["Abortive episodes, n"] / c["Episodes, n"] * 100).round(1)
    _write_sheet(wb.create_sheet("QC stage counts"), c,
                 "Episodes and abortive episodes after each quality-control exclusion",
                 ["Values are computed by the pipeline; cells hold results, not formulas."])
    if yt is not None:
        y = yt.copy()
        for col in [x for x in y.columns if x.endswith("%")]:
            y[col] = pd.to_numeric(y[col], errors="coerce").round(2)
        _write_sheet(wb.create_sheet("QC yield by system"), y,
                     "Episodes removed by each quality-control rule, BPJS Kesehatan "
                     "and TMUCRD", ["Percentages are of episodes entering quality control."])
    wb.save(path)
    print(f"    workbook: {', '.join(wb.sheetnames)} -> {path}")


def write_reports(out_dir: Path) -> None:
    sw = pd.read_csv(out_dir / "qc_switches.csv")
    long = pd.read_csv(out_dir / "or_by_qc_stage.csv")
    counts = pd.read_csv(out_dir / "qc_stage_counts.csv")
    yp = out_dir / "qc_yield_by_system.csv"
    yt = pd.read_csv(yp) if yp.exists() else None
    b = estimable_both(sw)
    print(f"    estimable before and after QC (not flagged): {len(b):,}; "
          f"direction reversed {int(b['direction_switch'].sum())}, "
          f"CI moved across 1 {int(b['significance_switch'].sum())}")
    qc_figure(sw, out_dir / FIG_NAME)
    supplementary_workbook(sw, long, counts, yt, out_dir / XLSX_NAME)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 74)
    print(" STEP 14 -- IMPACT OF EACH QUALITY-CONTROL RULE")
    print("=" * 74)

    s5 = importlib.import_module("04_apply_pregnancy_qc")
    s7 = importlib.import_module("08_odds_ratios")

    con = duckdb.connect()
    n_in, cols = s5.load(con, STEP3_CSV)
    present, _ = s5.resolve_conflict_cols(cols)

    long, counts = [], []
    con.execute("CREATE OR REPLACE TABLE final AS SELECT * FROM raw")
    r, c = or_at_stage(con, s7, STAGES[0]); long.append(r); counts.append(c)

    age_rep = s5.age_exclusion(con)
    con.execute("CREATE OR REPLACE TABLE final AS SELECT * FROM age_filtered")
    r, c = or_at_stage(con, s7, STAGES[1]); long.append(r); counts.append(c)

    conf_rep, _ = s5.conflict_exclusion(con, present)
    r, c = or_at_stage(con, s7, STAGES[2]); long.append(r); counts.append(c)

    hdr = pd.read_csv(s5.VISIT_CSV_PATH, nrows=0).columns
    use = [s5.ID_COL, "n_preg", "fin_g", "ref_start"]
    kw = {}
    if {"doa", "dol"} <= set(hdr):
        use += ["doa", "dol"]
    else:
        s1 = importlib.import_module("01_build_visit_level")
        use += ["combined_date"] + list(s1.DX_COLS)
        kw = dict(dx_cols=s1.DX_COLS, abort_codes=s1.ABORT_CODES,
                  partus_codes=s1.PARTUS_CODES)
    visit = pd.read_csv(s5.VISIT_CSV_PATH, usecols=use,
                        dtype={s5.ID_COL: str}, low_memory=False)
    seq_rep = s5.sequence_exclusion(con, s5.sequence_flags(visit, **kw))
    del visit
    r, c = or_at_stage(con, s7, STAGES[3]); long.append(r); counts.append(c)
    con.close()

    long = pd.concat(long, ignore_index=True)
    long.drop(columns=["_rank"], errors="ignore").to_csv(
        OUT_DIR / "or_by_qc_stage.csv", index=False)
    pd.DataFrame(counts).to_csv(OUT_DIR / "qc_stage_counts.csv", index=False)

    sw = switches(long)
    sw.to_csv(OUT_DIR / "qc_switches.csv", index=False)
    est = sw["odds_ratio_pre"].notna() & sw["odds_ratio_post"].notna()
    print(f"    estimable before and after QC : {int(est.sum()):,}")
    print(f"      direction switches          : {int(sw['direction_switch'].sum()):,}")
    print(f"      significance switches       : {int(sw['significance_switch'].sum()):,}")
    print(f"      estimability changed        : {int(sw['estimability_changed'].sum()):,}")

    yt = yield_table({c["stage"]: c["episodes"] for c in counts},
                     seq_rep, age_rep, conf_rep)
    yt.to_csv(OUT_DIR / "qc_yield_by_system.csv", index=False)
    print("\n    QC YIELD BY SYSTEM")
    print("      " + yt.to_string(index=False).replace("\n", "\n      "))
    write_reports(OUT_DIR)
    print(f"\nOutput written to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    if "--report-only" in sys.argv:
        write_reports(OUT_DIR)
    else:
        main()