"""
=============================================================================
 STEP 16 -- FLOW DIAGRAMS (BPJS KESEHATAN AND TMUCRD)
=============================================================================

Draws the participant flow diagrams from the counts the pipeline has already
written, so the numbers in the figure cannot drift from the analysis:

    Figure_flow_bpjs.png     main-text flow diagram (BPJS Kesehatan)
    Figure_flow_tmucrd.png   supplementary flow diagram (TMUCRD)
    flow_counts.csv          every number drawn, with its source

Run after steps 1-5 and 12-14. Nothing is recomputed except one split the
earlier steps do not save: how many visit-level records were removed for an
implausible age and how many because the woman had no episode in the
analytic dataset. That split is counted here from the step 1 visit table.

COUNTS THAT ARE ONLY PRINTED, NOT SAVED
---------------------------------------
Some counts appear only in the console output of earlier steps: the row
counts of the four BPJS source tables (step 1) and, for TMUCRD, the episodes
before the washout and the number removed by it (step 13). Enter them in
MANUAL below. A count left as None is drawn without a number and listed as
missing at the end of the run.

Numbers are printed in IJE style: no separator up to 9999, a space from
10 000. Figures are at least 3600 px wide.

USAGE
-----
    python 16_flow_diagrams.py
=============================================================================
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import pandas as pd

os.chdir(Path(__file__).resolve().parent)

# =============================================================================
# CONFIGURATION
# =============================================================================

P = {
    "visit_csv":   Path(r"./output_step1/pregnancy by visit.csv"),
    "washout":     Path(r"./output_step3/washout_report_step3.csv"),
    "qc4":         Path(r"./output_step4/exclusion_report_step4.csv"),
    "cohort":      Path(r"./output_step4/analytic_cohort_ids.csv"),
    "excl5":       Path(r"./output_step5/exclusion_report_step5.csv"),
    "stages14":    Path(r"./output_step14/qc_stage_counts.csv"),
    "tm_build":    Path(r"./output_step12/tmucrd_build_report.csv"),
    "tm_visit":    Path(r"./output_step12/tmucrd_pregnancy_by_visit.csv"),
    "tm_episode":  Path(r"./output_step13/tmucrd_pregnancy_by_episode.csv"),
    "tm_qc":       Path(r"./output_step13/external_qc_report.csv"),
    "tm_seq":      Path(r"./output_step13/external_sequence_qc_report.csv"),
    "tm_washout":  Path(r"./output_step13/external_washout_report.csv"),
}
OUT_DIR = Path(r"./output_step16")
BPJS_ID, TM_ID = "PSTV01", "CHR_NO"
AGE_MIN, AGE_MAX = 12, 55
OUTCOME = "c_abortive"

# Counts printed to the console by earlier steps but not saved to a file.
MANUAL = {
    "bpjs_hospital_visits": 1648135,       # step 1 console
    "bpjs_secondary_dx": 1517325,          # step 1 console
    "bpjs_primary_visits": 5681686,        # step 1 console
    "bpjs_membership": 413819,             # step 1 console
    # Read from external_washout_report.csv when step 13 has written it;
    # otherwise enter them from the step 13 console output.
    "tm_episodes_before_washout": None,
    "tm_removed_by_washout": None,
    "tm_removed_age": 0,                   # step 13 console / qc_yield_by_system.csv
}

DPI = 460            # keeps both figures above 3600 px wide
FONT = 9

# =============================================================================


def n(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "n = ?"
    x = int(x)
    return "n = " + (f"{x:,}".replace(",", " ") if x >= 10000 else str(x))


def num(x) -> str:
    return n(x)[4:]


def kv(path: Path, key="metric", val="value") -> dict:
    d = pd.read_csv(path)
    return dict(zip(d[key].astype(str), d[val]))


def need(*paths: Path) -> None:
    miss = [str(p) for p in paths if not p.exists()]
    if miss:
        sys.exit("[FATAL] missing input:\n  " + "\n  ".join(miss))


# --------------------------------------------------------------------------- #
# Counts                                                                      #
# --------------------------------------------------------------------------- #

def bpjs_counts() -> dict:
    need(P["visit_csv"], P["washout"], P["qc4"], P["cohort"], P["excl5"], P["stages14"])
    import duckdb
    con = duckdb.connect()
    vp = P["visit_csv"].resolve().as_posix()
    cp = P["cohort"].resolve().as_posix()
    v_total, v_age, w_age, v_noep, w_noep = con.execute(f"""
        WITH v AS (SELECT CAST({BPJS_ID} AS VARCHAR) id, TRY_CAST(age AS DOUBLE) age
                   FROM read_csv_auto('{vp}', HEADER=TRUE) WHERE {BPJS_ID} IS NOT NULL),
        w AS (SELECT id, COUNT(*) nv,
                     (MIN(age) >= {AGE_MIN} AND MAX(age) <= {AGE_MAX}) ok
              FROM v GROUP BY id),
        c AS (SELECT DISTINCT CAST({BPJS_ID} AS VARCHAR) id
              FROM read_csv_auto('{cp}', HEADER=TRUE))
        SELECT SUM(nv),
               SUM(nv) FILTER (WHERE NOT COALESCE(ok, FALSE)),
               COUNT(*) FILTER (WHERE NOT COALESCE(ok, FALSE)),
               SUM(nv) FILTER (WHERE COALESCE(ok, FALSE) AND id NOT IN (SELECT id FROM c)),
               COUNT(*) FILTER (WHERE COALESCE(ok, FALSE) AND id NOT IN (SELECT id FROM c))
        FROM w
    """).fetchone()
    con.close()

    wo, q4 = kv(P["washout"]), kv(P["qc4"])
    e5 = pd.read_csv(P["excl5"])
    e5 = {(r.level, r.metric): r.n for r in e5.itertuples()}
    st = pd.read_csv(P["stages14"]).set_index("stage")
    ab = st["abortive"].tolist()            # before QC, after age, conflict, sequence
    ep = st["episodes"].tolist()

    c = {
        **{k: MANUAL[k] for k in MANUAL if k.startswith("bpjs_")},
        "visits_total": v_total,
        "visits_removed_age": v_age, "women_removed_age": w_age,
        "visits_removed_noep": v_noep, "women_removed_noep": w_noep,
        "visits_final": e5[("visit", "retained")],
        "women_total": e5[("individual", "individuals_total")],
        "women_final": e5[("individual", "individuals_kept")],
        "ep_reconstructed": int(wo["episodes_after_washout"]) + int(wo["removed_by_washout"]),
        "ep_washout": int(wo["removed_by_washout"]),
        "ep_after_washout": int(wo["episodes_after_washout"]),
        "women_after_washout": int(wo["women"]),
        "ep_age": int(q4["removed_age_rule_episode"]), "ab_age": ab[0] - ab[1],
        "ep_conflict": int(q4["removed_diagnostic_conflict"]), "ab_conflict": ab[1] - ab[2],
        "ep_seq": ep[2] - ep[3], "ab_seq": ab[2] - ab[3],
        "ep_seq_delivery": int(q4["removed_overlap_delivery"]),
        "ep_seq_ongoing": int(q4["removed_overlap_abortive"]),
        "ep_seq_absorbed": int(q4["removed_abortive_with_delivery"]),
        "ep_final": int(q4["episodes_final"]),
        "women_preg_final": int(q4["women_in_analytic_cohort"]),
        "ab_before": ab[0], "ab_final": ab[3],
    }
    if c["visits_total"] - c["visits_removed_age"] - c["visits_removed_noep"] != c["visits_final"]:
        print("    [WARNING] visit-level split does not add up to the step 5 total")
    if c["women_total"] - c["women_removed_age"] - c["women_removed_noep"] != c["women_final"]:
        print("    [WARNING] individual-level split does not add up to the step 5 total")
    return c


def tmucrd_counts() -> dict:
    need(P["tm_build"], P["tm_visit"], P["tm_episode"], P["tm_qc"], P["tm_seq"])
    b = pd.read_csv(P["tm_build"]).set_index("table")["rows"]
    v = pd.read_csv(P["tm_visit"], usecols=[TM_ID])
    e = pd.read_csv(P["tm_episode"], usecols=[TM_ID, OUTCOME])
    q, s = kv(P["tm_qc"]), kv(P["tm_seq"])
    w = kv(P["tm_washout"]) if P["tm_washout"].exists() else {}
    ep_pre = w.get("episodes_before_washout", MANUAL["tm_episodes_before_washout"])
    ep_wash = w.get("removed_by_washout", MANUAL["tm_removed_by_washout"])
    ab_final = int((e[OUTCOME] == 1).sum())
    seq_ab = int(s["removed_overlap_abortive"]) + int(s["removed_abortive_with_delivery"])
    pre = int(q["episodes_before_conflict_qc"]) + int(MANUAL["tm_removed_age"] or 0)
    return {
        "ipd": int(b.get("ipd_basic")), "opd": int(b.get("opd_basic")),
        "visits": len(v), "women": v[TM_ID].nunique(),
        "ep_reconstructed": None if ep_pre is None else int(ep_pre),
        "ep_washout": None if ep_wash is None else int(ep_wash),
        "ep_after_washout": pre,
        "ep_age": MANUAL["tm_removed_age"],
        "ep_conflict": int(q["removed_diagnostic_conflict"]),
        "ep_seq": int(s["removed_overlap_delivery"]) + seq_ab,
        "ep_seq_delivery": int(s["removed_overlap_delivery"]),
        "ep_seq_ongoing": int(s["removed_overlap_abortive"]),
        "ep_seq_absorbed": int(s["removed_abortive_with_delivery"]),
        "ep_final": len(e), "women_final": e[TM_ID].nunique(),
        # conflict removals are abortive by definition, as are the two
        # abortive sequence rules
        "ab_before": ab_final + int(q["removed_diagnostic_conflict"]) + seq_ab,
        "ab_final": ab_final,
    }


# --------------------------------------------------------------------------- #
# Drawing                                                                     #
# --------------------------------------------------------------------------- #

LINE_H = 2.1          # height of one text line in canvas units
PAD = 1.0


class Canvas:
    def __init__(self, width_in: float, height_units: float):
        self.H = height_units
        self.fig, self.ax = plt.subplots(figsize=(width_in, width_in * height_units / 100))
        self.ax.set_xlim(0, 100); self.ax.set_ylim(height_units, 0)
        self.ax.axis("off")
        plt.rcParams["font.family"] = "DejaVu Sans"

    def box(self, x: float, y: float, w: float, lines: list[str], bold_first=False,
            align="center") -> dict:
        h = len(lines) * LINE_H + 2 * PAD
        self.ax.add_patch(Rectangle((x - w / 2, y), w, h, fill=False, lw=1.0))
        for i, t in enumerate(lines):
            ty = y + PAD + (i + 0.5) * LINE_H
            tx = x if align == "center" else x - w / 2 + 1.2
            self.ax.text(tx, ty, t, ha=align, va="center", fontsize=FONT,
                         fontweight="bold" if (bold_first and i == 0) else "normal")
        return {"l": x - w / 2, "r": x + w / 2, "t": y, "b": y + h, "x": x,
                "y": y + h / 2}

    def arrow(self, p0, p1):
        self.ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=9,
                                          lw=1.0, color="black", shrinkA=0, shrinkB=0))

    def line(self, pts):
        xs, ys = zip(*pts)
        self.ax.plot(xs, ys, color="black", lw=1.0)

    def join(self, a: dict, b: dict, target: dict, y_mid: float):
        """Two boxes merging into one below them."""
        self.line([(a["x"], a["b"]), (a["x"], y_mid), (b["x"], y_mid), (b["x"], b["b"])])
        self.arrow((target["x"], y_mid), (target["x"], target["t"]))

    def down(self, a: dict, b: dict):
        self.arrow((a["x"], a["b"]), (b["x"], b["t"]))

    def side(self, src: dict, dst: dict, y: float):
        """Branch from the vertical line under src to a box at its side."""
        edge = dst["l"] if dst["l"] > src["x"] else dst["r"]
        self.arrow((src["x"], y), (edge, y))

    def save(self, path: Path):
        self.fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.15)
        plt.close(self.fig)


def draw_bpjs(c: dict, path: Path) -> None:
    """Layout of the original flow diagram: the visit level branches to the
    left, the individual level has its exclusions on the left and the
    pregnancy level on the right."""
    pct = lambda a, d: f"{100 * a / d:.1f}%"
    cv = Canvas(13, 116)
    hv = cv.box(13, 2, 18, ["Hospital visits", n(c["bpjs_hospital_visits"])])
    sd = cv.box(35, 2, 18, ["Hospital secondary", "diagnoses", n(c["bpjs_secondary_dx"])])
    ih = cv.box(24, 17, 18, ["Integrated hospital", "records"])
    cv.join(hv, sd, ih, 14.5)
    pc = cv.box(52, 17, 18, ["Primary care visits", n(c["bpjs_primary_visits"])])
    cl = cv.box(38, 29, 16, ["Clinical records"])
    cv.join(ih, pc, cl, 26.5)
    mb = cv.box(62, 27, 18, ["Membership records", n(c["bpjs_membership"]) + " women"])
    vl = cv.box(50, 39, 20, ["Visit-level dataset", n(c["visits_total"]) + " records",
                             n(c["women_total"]) + " women"])
    cv.join(cl, mb, vl, 36.5)

    # visit level: arrow to the left, exclusions hanging from that arrow
    v1 = cv.box(9, 39.5, 16, ["Analytic visit-level", "dataset",
                              n(c["visits_final"]), "records"])
    cv.arrow((vl["l"], vl["y"]), (v1["r"], vl["y"]))
    vq = cv.box(29, 53, 24, ["Excluded records",
                             f"Implausible age: {num(c['visits_removed_age'])}",
                             f"No analytic episode: {num(c['visits_removed_noep'])}"],
                bold_first=True, align="left")
    cv.arrow((29, vl["y"]), (29, vq["t"]))

    # split to the individual and pregnancy levels
    cv.line([(vl["x"], vl["b"]), (vl["x"], 64), (38, 64), (62, 64)])
    i0 = cv.box(38, 67, 20, ["Individual-level dataset", n(c["women_total"]) + " women"])
    p0 = cv.box(62, 67, 20, ["Pregnancy-level dataset", n(c["ep_reconstructed"]) + " episodes"])
    cv.arrow((38, 64), (38, i0["t"])); cv.arrow((62, 64), (62, i0["t"]))

    iq = cv.box(14, 80, 24, ["Excluded women",
                             f"Implausible age: {num(c['women_removed_age'])}",
                             f"No analytic episode: {num(c['women_removed_noep'])}"],
                bold_first=True, align="left")
    pq = cv.box(86.5, 78, 27, [
        "Excluded episodes (abortive)",
        f"2015 washout: {num(c['ep_washout'])}",
        f"Implausible age: {num(c['ep_age'])} ({num(c['ab_age'])})",
        f"Diagnostic conflict: {num(c['ep_conflict'])} ({num(c['ab_conflict'])})",
        f"Implausible sequence: {num(c['ep_seq'])} ({num(c['ab_seq'])})",
        f"  Delivery before previous: {num(c['ep_seq_delivery'])}",
        f"  Abortive in ongoing: {num(c['ep_seq_ongoing'])}",
        f"  Abortive with delivery: {num(c['ep_seq_absorbed'])}"],
        bold_first=True, align="left")
    i1 = cv.box(38, 101, 20, ["Analytic individual-level", "dataset",
                              n(c["women_final"]) + " women"])
    p1 = cv.box(62, 101, 20, ["Analytic pregnancy-level", "dataset",
                              n(c["ep_final"]) + " episodes",
                              n(c["women_preg_final"]) + " women",
                              f"Abortive: {num(c['ab_final'])} "
                              f"({pct(c['ab_final'], c['ep_final'])})"])
    cv.down(i0, i1); cv.down(p0, p1)
    cv.arrow((38, iq["y"]), (iq["r"], iq["y"]))
    cv.arrow((62, pq["y"]), (pq["l"], pq["y"]))
    cv.save(path)


def draw_tmucrd(c: dict, path: Path) -> None:
    """Same style as the pregnancy branch of the BPJS diagram."""
    pct = lambda a, d: f"{100 * a / d:.1f}%"
    cv = Canvas(10, 84)
    ip = cv.box(24, 2, 26, ["Inpatient records", n(c["ipd"])])
    op = cv.box(60, 2, 26, ["Outpatient records", n(c["opd"])])
    vl = cv.box(42, 15, 34, ["Visit-level dataset", "(women with a termination code)",
                             n(c["visits"]) + " records", n(c["women"]) + " women"])
    cv.join(ip, op, vl, 12.5)
    p0 = cv.box(42, 33, 30, ["Pregnancy-level dataset",
                             n(c["ep_reconstructed"]) + " episodes"])
    cv.down(vl, p0)
    wash = "?" if c["ep_washout"] is None else num(c["ep_washout"])
    q = cv.box(79, 44, 36, [
        "Excluded episodes (abortive)",
        f"2013 washout: {wash}",
        f"Implausible age: {num(c['ep_age'])}",
        f"Diagnostic conflict: {num(c['ep_conflict'])} ({num(c['ep_conflict'])})",
        f"Implausible sequence: {num(c['ep_seq'])} "
        f"({num(c['ep_seq_ongoing'] + c['ep_seq_absorbed'])})"],
        bold_first=True, align="left")
    a = cv.box(42, 64, 30, ["Analytic pregnancy-level", "dataset",
                            n(c["ep_final"]) + " episodes", n(c["women_final"]) + " women",
                            f"Abortive: {num(c['ab_final'])} "
                            f"({pct(c['ab_final'], c['ep_final'])})"])
    cv.down(p0, a)
    cv.arrow((42, q["y"]), (q["l"], q["y"]))
    cv.save(path)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    print("BPJS Kesehatan")
    b = bpjs_counts()
    draw_bpjs(b, OUT_DIR / "Figure_flow_bpjs.png")
    rows += [{"system": "BPJS", "count": k, "n": v} for k, v in b.items()]
    print(f"    saved Figure_flow_bpjs.png")
    print("TMUCRD")
    try:
        t = tmucrd_counts()
        draw_tmucrd(t, OUT_DIR / "Figure_flow_tmucrd.png")
        rows += [{"system": "TMUCRD", "count": k, "n": v} for k, v in t.items()]
        print(f"    saved Figure_flow_tmucrd.png")
    except SystemExit as e:
        print(f"    [skip] {e}")
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "flow_counts.csv", index=False)
    miss = df[df["n"].isna()]
    if len(miss):
        print("\n    counts still missing (enter them in MANUAL):")
        for r in miss.itertuples():
            print(f"      {r.system}: {r.count}")


if __name__ == "__main__":
    main()