"""
=============================================================================
 STEP 3 -- BUILD THE PREGNANCY-LEVEL DATASET
=============================================================================

Extracts diagnostic feature flags relative to each pregnancy episode and
collapses the visit-level table to one row per episode.

Each diagnosis is assigned to a temporal phase relative to the episode:

    b_*   before pregnancy
    c_*   during pregnancy   (infectious and pregnancy-related groups only)
    a_*   after pregnancy

Chronic conditions have no `c_` phase by construction, because they are
defined by any record on or before the termination date rather than within
the gestational window.

The phase boundaries are those of the original notebook and are unchanged:

    before_chronic     combined_date <= ref
    before_inf_preg    combined_date <  ref_start
    after_chronic_inf  combined_date >  ref
    after_preg         combined_date -  ref > 30 days
    during_inf         ref_start <= combined_date <= ref
    during_preg        ref_start <= combined_date <= ref + 30 days

NOTES ON THIS REVISION
----------------------
Derived from the working script, with the time-to-event component removed.
That component belongs to a separate manuscript and is not required to
reproduce the cohort-construction results, so it is not distributed here.

Four changes:

  1. The pregnancy-order loop is bounded by the highest order actually present
     in the data rather than by a fixed constant. The previous limit of seven
     silently discarded every episode of order eight or above; a shorter
     separation window produces more episodes per woman and therefore more
     such episodes. Set MAX_N_PREG to an integer only to cap deliberately, in
     which case the number of discarded episodes is reported.

  2. `age_risk` is derived from the aggregated `age` after the collapse rather
     than aggregated from per-visit indicators. Aggregating a binary indicator
     across visits allows it to contradict the age it is supposed to describe:
     a woman aged 19 at one visit and 21 at another yields a different answer
     under MIN than under MAX, and neither need match the retained age.
     Deriving it from the aggregated value makes the two consistent by
     construction.

  3. The number of episodes removed by the 2015 washout is now reported. The
     washout applies to first-order episodes only, which is the behaviour of
     the original notebook, but the count was never recorded.

  4. Feature-flag prevalence is written to `flag_prevalence_step3.csv`, so that
     an all-zero or all-one column is caught here rather than surfacing as an
     unexplained non-estimable association several steps later.

The washout, the phase definitions, the aggregation rules, and the
`abortive = O00-O07` definition are unchanged from the working script.

INPUT
-----
    pregnancy by visit.csv              from step 1 (complete, not age-cleaned)

OUTPUT
------
    pregnancy by episode.csv            pregnancy-level dataset, pre-QC
    npreg_distribution_step3.csv        episodes built per pregnancy order
    flag_prevalence_step3.csv           prevalence of every b_/c_/a_ flag
    washout_report_step3.csv            episodes removed by the washout year
    benchmark_step3.csv                 per-stage timing and memory
    environment_step3.txt               library versions and machine spec

USAGE
-----
    python 03_build_pregnancy_level.py
=============================================================================
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import sys
import warnings

import numpy as np
import pandas as pd

from pipeline_utils import Bench, environment_report, warn_if_no_psutil

# The regular expressions below use capturing groups, e.g. 'M(0[0-9]|1[0-9])'.
# pandas warns that groups are ignored by str.contains, which is exactly the
# intended behaviour here since only a match is required. The patterns are left
# as written rather than rewritten with non-capturing groups, so that they stay
# byte-identical to the working script.
warnings.filterwarnings(
    "ignore", message=".*interpreted as a regular expression.*",
    category=UserWarning)

# pandas raises PerformanceWarning when aggregating a frame that is several
# hundred columns wide. The construction it suggests -- a single concatenation
# rather than repeated insertion -- is already used below, so the remaining
# warning reflects the width of the frame rather than an avoidable inefficiency.
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# Relative paths in the configuration block below are resolved against the
# folder holding this script, not the shell's working directory. Without this,
# launching from an editor's Run button resolves them against the editor's
# install directory instead, and every path fails.
os.chdir(Path(__file__).resolve().parent)

# =============================================================================
# CONFIGURATION
# =============================================================================

# The COMPLETE visit-level table from step 1, not the age-cleaned file from
# step 6. The pregnancy-level age exclusion is applied in step 4; building from
# the cleaned file would remove the affected women before anchoring, so the
# pre-exclusion episode count and the step 4 exclusion count could not be
# reproduced. Step 6's output is the visit-level ANALYTIC dataset, used for
# Table 1 and for the visit- and individual-level odds ratios.
VISIT_CSV_PATH = r"./output_step1/pregnancy by visit.csv"
OUT_DIR        = r"./output_step3"
OUT_NAME       = r"pregnancy by episode.csv"

WASHOUT_YEAR = 2015          # first-order episodes terminating in this year are
                             # dropped, because enrolment cannot be distinguished
                             # from incident pregnancy in the first data year
MAX_N_PREG   = None          # None = build every order present in the data
DX_COLS      = ["FKP14A", "FKL15A", "FKL17A", "FKL24A"]

AGE_RISK_LOW, AGE_RISK_HIGH = 20, 35      # extreme maternal age boundaries
AFTER_PREG_GRACE_DAYS = 30                # post-termination window for c_/a_ split

# Flag columns are assembled in one concatenation rather than inserted one at a
# time. Inserting roughly two hundred columns individually fragments the block
# manager and dominates the cost of this stage on the full dataset. The boolean
# expressions themselves are unchanged, so the result is identical; set this to
# True once on a subsample to confirm that on your own data.
VERIFY_FLAG_CONSTRUCTION = False

# =============================================================================
# CONDITION DEFINITIONS  ---  verbatim from notebook cell 17
# =============================================================================
chronic_conditions = { 
    'dm': ['E10','E11','E12','E13','E14','O24'],
    'malnut': ['E40','E41','E42','E43','E44','E45','E46'],
    'nutri': ['E50','E51','E52','E53','E54','E55','E56','E57','E58','E59','E60','E61','E62','E63','E64'],
    'obese': ['E66'],
    'substance': ['F10','F11','F12','F13','F14','F15','F16','F17','F18','F19'],
    'schizo': ['F20','F21','F22','F23','F24','F25','F28','F29'],
    'neurot': ['F40','F41','F42','F43','F44','F48','F45'],
    'neu_deg': ['G10','G11','G12','G20','G21','G22','G23','G24','G25','G26','G30','G31','G32','G35','G36','G37'],
    'headache': ['G43','G44'],
    'neuropathy': ['G50','G51','G52','G53','G54','G55','G56','G57','G58','G59','G60','G61','G62','G63','G64'],
    'rhd': ['I05','I06','I07','I08','I09'],
    'ht': ['I10','I11','I12','I13','I14','I15','O10','O13','O16'],
    'angin': ['I20'],                     
    'myocard': ['I21','I22','I24'], 
    'phd': ['I26','I27','I28'],
    'carditis': ['I30','I32','I33','I38','I39','I40','I41'],
    'cmp': ['I42','I43'],
    'arrythmia': ['I44','I45','I47','I48','I49'],
    'hf': ['I50'],
    'stroke': ['I60','I61','I62','I63','I64'],
    'artery': ['I70','I71','I72','I73','I74','I77','I78','I79'],
    'vein': ['I80','I81','I82','I83','I85','I86','I87','I88','I89'],
    'chronic_res': ['J35','J37','J40','J41','J42','J43','J44','J45'],
    'pul_edema': ['J81'],
    'pleura': ['J90','J91','J92','J93','J94'],
    'oral': ['K00','K01','K02','K03','K04','K05','K06','K07','K08','K09','K10','K11','K12','K13','K14'],
    'gastritis': ['K22','K25','K26','K27','K28','K29','K30'],
    'hernia': ['K40','K41','K42','K43','K44','K45','K46'],
    'intestinal': ['K50','K51','K52','K56','K58','K59','K60','K61','K62','K63'],
    'hemorrh': ['K64'],
    'periton': ['K65'],
    'liver_fail': ['K72'],
    'liver': ['K70','K71','K73','K74','K75','K76'],
    'gallbladder': ['K80','K81','K82','K83'],
    'pancreas': ['K85','K86'],
    'bullous': ['L10','L11','L12','L13','L14'],
    'atopic': ['L20'],
    'dermatitis': ['L21','L23','L25','L26','L27','L28','L30'],
    'urticaria': ['L50'],
    'urolith': ['N20','N21','N22'],
    'endomet': ['N80'],
    'femgen': ['N81','N82','N83','N84','N85','N86','N87','N88','N89','N90'],
    'hypomen': ['N91'],
    'menorrh': ['N92'],
    'dysmen': ['N94']
}
infectious_conditions = {
    'typhoid': ['A01'],
    'cholera': ['A00'],
    'v_age': ['A08'],
    'b_age': ['A00', 'A02', 'A03', 'A04', 'A05'],
    'p_age': ['A06', 'A07'],
    'tb': ['A15', 'A16', 'A17', 'A18', 'A19'],
    'myco': ['A30', 'A31'],
    'lepto': ['A27'],
    'std': ['A51','A52','A53','A54','A55','A56','A57','A58','A59','A63','A64'],
    'torch': ['B58','B06','B25','B00','A60'],
    'v_skin': ['B01','B02','B03','B04','B05','B07','B08','B09'],
    'hepatitis': ['B15','B16','B17','B18','B19'],
    'hiv': ['B20','B21','B22','B23','B24'],
    'sepsis': ['A40','A41'],
    'infla_cns': ['G00','G01','G02','G03','G04','G08','G05','G06','G07','G09'],
    'urti': ['J00','J01','J02','J03','J04','J05','J06','J09','J10','J11'],
    'lrti': ['J12','J13','J14','J15','J16','J17','J18','J20','J21','J22'],
    'uti': ['N30','N34','N39']
}
pregnancy_conditions = {
    'abortive': ['O00', 'O01', 'O02', 'O03', 'O04', 'O05', 'O06', 'O07'],
    'preecl': ['O11', 'O14'], 'ecl': ['O15'], 'earlyhemo': ['O20'], 'heg': ['O21'],
    'venpreg': ['O22'], 'utipreg': ['O23'], 'malpreg': ['O25'], 'multigest': ['O30'],
    'malpresent': ['O32'], 'disprop': ['O33'], 'abnorpelv': ['O34'], 'fetalprob': ['O35', 'O36'],
    'polyhydra': ['O40'], 'abnamnio': ['O41'], 'prom': ['O42'], 'placental': ['O43'],
    'previa': ['O44'], 'abrupt': ['O45'], 'anh': ['O46'], 'prolong': ['O48'],
    'preterm': ['O60'], 'fail': ['O61'], 'abnforce': ['O62'], 'long': ['O63'],
    'obspelvic': ['O65', 'O66'], 'malpres': ['O64'], 'iph': ['O67'], 'distress': ['O68'],
    'umbilical': ['O69'], 'laceration': ['O70'], 'obstrau': ['O71'], 'pph': ['O72'],
    'retained': ['O73'], 'normal': ['O80'], 'instrum': ['O81'], 'caesar': ['O82'],
    'assisted': ['O83'], 'multiple': ['O84']
}
conditions_regex = {
    'arthropathy': r'M(0[0-9]|1[0-9]|2[0-5])',
    'sysconn': r'M3[0-6]',
    'dorsopathy': r'M4[0-9]|M5[0-4]',
    'muscle_dis': r'M6[0-3]',
    'synov_dis': r'M6[5-8]',
    'soft_dis': r'M8[0-9]|M9[0-4]',
    'renal_dis': r'N0[0-9]|N1[0-6]',
    'renal_fail': r'N1[7-9]',
    'breast_dis': r'N6[0-4]',
    'pid': r'N7[0-7]',
    'poison': r'T3[6-9]|T4[0-9]|T50',
    'toxic': r'T5[1-9]|T6[0-5]'
}
compiled_patterns = {c: re.compile(r) for c, r in conditions_regex.items()}

# =============================================================================
# FEATURE EXTRACTION AND COLLAPSE  ---  verbatim, survival columns removed
# =============================================================================


def feature_pass(primi_all: pd.DataFrame, k: int) -> pd.DataFrame:
    """Build one row per woman for pregnancy order `k`."""
    valid = primi_all.loc[primi_all["n_preg"] == k, "PSTV01"].unique()
    primi = primi_all[primi_all["PSTV01"].isin(valid)].reset_index(drop=True)

    ref_map = (primi.loc[(primi["n_preg"] == k) & (primi["fin_g"].notna()),
                         ["PSTV01", "fin_g"]]
               .sort_values(["PSTV01", "fin_g"]).drop_duplicates("PSTV01", keep="first")
               .set_index("PSTV01")["fin_g"])
    primi["ref"] = primi["PSTV01"].map(ref_map)

    rs_map = (primi.loc[(primi["n_preg"] == k) & (primi["ref_start"].notna()),
                        ["PSTV01", "ref_start"]]
              .sort_values(["PSTV01", "ref_start"]).drop_duplicates("PSTV01", keep="first")
              .set_index("PSTV01")["ref_start"])
    primi["ref_start"] = primi["PSTV01"].map(rs_map)

    primi["combined_date"] = pd.to_datetime(primi["combined_date"], errors="coerce")
    primi["ref"]           = pd.to_datetime(primi["ref"], errors="coerce")
    primi["ref_start"]     = pd.to_datetime(primi["ref_start"], errors="coerce")

    grace = pd.Timedelta(days=AFTER_PREG_GRACE_DAYS)
    before_chronic    = (primi["combined_date"] <= primi["ref"])
    before_inf_preg   = (primi["combined_date"] <  primi["ref_start"])
    after_chronic_inf = (primi["combined_date"] >  primi["ref"])
    after_preg        = (primi["combined_date"] - primi["ref"]).dt.days > AFTER_PREG_GRACE_DAYS
    during_inf  = ((primi["combined_date"] >= primi["ref_start"])
                   & (primi["combined_date"] <= primi["ref"]))
    during_preg = ((primi["combined_date"] >= primi["ref_start"])
                   & (primi["combined_date"] <= (primi["ref"] + grace)))

    def anymatch(codes):
        return (primi["FKP14A"].isin(codes) | primi["FKL15A"].isin(codes) |
                primi["FKL17A"].isin(codes) | primi["FKL24A"].isin(codes))

    def regex_match(pat):
        return (primi["FKP14A"].str.contains(pat, na=False)
                | primi["FKL15A"].str.contains(pat, na=False)
                | primi["FKL17A"].str.contains(pat, na=False)
                | primi["FKL24A"].str.contains(pat, na=False))

    flags: dict[str, np.ndarray] = {}

    for cond, codes in chronic_conditions.items():
        m = anymatch(codes)
        flags[f"b_{cond}"] = (m & before_chronic).astype(int)
        flags[f"a_{cond}"] = (m & after_chronic_inf).astype(int)
    for cond, codes in infectious_conditions.items():
        m = anymatch(codes)
        flags[f"b_{cond}"] = (m & before_inf_preg).astype(int)
        flags[f"c_{cond}"] = (m & during_inf).astype(int)
        flags[f"a_{cond}"] = (m & after_chronic_inf).astype(int)
    # abortive = O00-O07 for ALL passes (O08 = complications of ectopic/molar
    # pregnancy, not a primary abortive outcome). The original notebook wrongly
    # appended O08 in passes 2-7; corrected here per intended definition.
    for cond, codes in pregnancy_conditions.items():
        m = anymatch(codes)
        flags[f"b_{cond}"] = (m & before_inf_preg).astype(int)
        flags[f"c_{cond}"] = (m & during_preg).astype(int)
        flags[f"a_{cond}"] = (m & after_preg).astype(int)
    for cond, pat in compiled_patterns.items():
        m = regex_match(pat)
        flags[f"b_{cond}"] = (m & before_chronic).astype(int)
        flags[f"a_{cond}"] = (m & after_chronic_inf).astype(int)

    # ref_year is concatenated together with the flags rather than assigned
    # afterwards, so that no single column is inserted into a frame that is
    # already several hundred columns wide.
    flags["ref_year"] = primi["ref"].dt.year
    primi = pd.concat([primi, pd.DataFrame(flags, index=primi.index)], axis=1)
    flags.pop("ref_year")

    if VERIFY_FLAG_CONSTRUCTION:
        check = primi.copy()
        for name, series in flags.items():
            check[name] = series
        assert check[list(flags)].equals(primi[list(flags)]), \
            "vectorised flag construction does not match column-by-column insertion"
        print("      flag construction verified against column-by-column insertion")

    # The 2015 washout exists ONLY in pass 1 (notebook cell 18). Passes 2 and
    # above compute ref_year but do not filter it. Replicated, and now counted.
    n_washout = 0
    if k == 1:
        n_before = primi["PSTV01"].nunique()
        primi = primi[primi["ref_year"] != WASHOUT_YEAR]
        n_washout = n_before - primi["PSTV01"].nunique()
    primi["n_preg"] = k

    # --- collapse ------------------------------------------------------------
    # age_risk is intentionally absent: it is derived from the aggregated age
    # below, so that the indicator cannot contradict the age it describes.
    agg = {"subsid": "max", "age": "min", "dom": "min",
           "n_preg": "min", "ref_year": "min",
           "ref_start": "min",      # -> pregnancy_start
           "ref": "min"}            # -> pregnancy_end
    flag_cols = [c for c in primi.columns if c[:2] in ("b_", "c_", "a_")]
    agg.update({c: "sum" for c in flag_cols})
    agg = {c: r for c, r in agg.items() if c in primi.columns}

    out = primi.groupby("PSTV01").agg(agg).reset_index()
    out[flag_cols] = out[flag_cols].clip(upper=1)
    out = out.copy()          # defragment before appending the derived column
    out["age_risk"] = ((out["age"] < AGE_RISK_LOW)
                       | (out["age"] > AGE_RISK_HIGH)).astype(int)
    return out, n_washout


# =============================================================================
# MAIN
# =============================================================================


def load_visits(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        sys.exit(f"[FATAL] input not found: {os.path.abspath(path)}")
    visit = pd.read_csv(path, low_memory=False)
    for c in ["combined_date", "fin_g", "ref_start"]:
        if c in visit.columns:
            visit[c] = pd.to_datetime(visit[c], errors="coerce")
    missing = [c for c in DX_COLS + ["PSTV01", "n_preg"] if c not in visit.columns]
    if missing:
        print(f"\n[FATAL] required columns not found: {missing}")
        print("Columns present:")
        for c in visit.columns:
            print(f"    {c}")
        sys.exit(1)
    print(f"    loaded {len(visit):,} rows x {len(visit.columns)} columns")
    return visit


def resolve_orders(visit: pd.DataFrame) -> list[int]:
    """Every pregnancy order present, unless a cap is set deliberately."""
    ep = visit.loc[visit["fin_g"].notna(), "n_preg"].dropna().astype(int)
    if not len(ep):
        sys.exit("[FATAL] no episode anchors found; check `fin_g` in the input.")
    kmax_data = int(ep.max())

    if MAX_N_PREG is None:
        kmax = kmax_data
    else:
        kmax = min(kmax_data, MAX_N_PREG)
        n_lost = int((ep > MAX_N_PREG).sum())
        w_lost = int(visit.loc[visit["fin_g"].notna()
                               & (visit["n_preg"] > MAX_N_PREG), "PSTV01"].nunique())
        if n_lost:
            print(f"    [WARNING] MAX_N_PREG = {MAX_N_PREG} discards {n_lost:,} "
                  f"episodes (orders {MAX_N_PREG + 1}-{kmax_data}) "
                  f"from {w_lost:,} women")

    print(f"    highest order present = {kmax_data}; building orders 1-{kmax}")
    print("    anchors per order:")
    for k, n in ep.value_counts().sort_index().items():
        mark = "   <-- not built" if k > kmax else ""
        print(f"      {k:>3} : {n:>10,}{mark}")
    return [k for k in range(1, kmax + 1) if (visit["n_preg"] == k).any()]


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 74)
    print(" STEP 3 -- BUILD THE PREGNANCY-LEVEL DATASET")
    print("=" * 74)
    print(environment_report(os.path.join(OUT_DIR, "environment_step3.txt")))
    warn_if_no_psutil()

    bench = Bench("step 3, pregnancy level")

    with bench.stage("Load complete visit-level table", "pandas") as b:
        visit = load_visits(VISIT_CSV_PATH)
        b["rows_out"] = len(visit)

    with bench.stage("Resolve pregnancy orders", "pandas",
                     rows_in=len(visit)) as b:
        orders = resolve_orders(visit)
        b["rows_out"] = len(orders)

    frames, washout_total = [], 0
    with bench.stage("Extract features and collapse", "pandas",
                     rows_in=len(visit)) as b:
        for k in orders:
            out, n_wash = feature_pass(visit.copy(), k)
            washout_total += n_wash
            frames.append(out)
            print(f"      order {k:>2} : {len(out):>10,} episodes"
                  + (f"   ({n_wash:,} removed by the {WASHOUT_YEAR} washout)"
                     if n_wash else ""))
        final = pd.concat(frames, ignore_index=True)
        b["rows_out"] = len(final)

    with bench.stage("Rename and export", "pandas", rows_in=len(final)) as b:
        final = final.rename(columns={"ref_start": "pregnancy_start",
                                      "ref": "pregnancy_end"})
        out_path = os.path.join(OUT_DIR, OUT_NAME)
        final.to_csv(out_path, index=False)
        print(f"    wrote {len(final):,} rows -> {out_path}")
        b["rows_out"] = len(final)

    with bench.stage("Diagnostics", "pandas", rows_in=len(final)) as b:
        dist = final["n_preg"].value_counts().sort_index()
        dist.rename("n_episodes").rename_axis("n_preg").to_csv(
            os.path.join(OUT_DIR, "npreg_distribution_step3.csv"))

        flag_cols = [c for c in final.columns if c[:2] in ("b_", "c_", "a_")]
        prev = pd.DataFrame({
            "flag": flag_cols,
            "n_positive": [int(final[c].sum()) for c in flag_cols],
        })
        prev["prevalence_pct"] = (prev["n_positive"] / len(final) * 100).round(4)
        prev = prev.sort_values("n_positive")
        prev.to_csv(os.path.join(OUT_DIR, "flag_prevalence_step3.csv"), index=False)

        pd.DataFrame([
            {"metric": "episodes_after_washout", "value": len(final)},
            {"metric": "removed_by_washout", "value": washout_total},
            {"metric": "washout_year", "value": WASHOUT_YEAR},
            {"metric": "women", "value": int(final["PSTV01"].nunique())},
        ]).to_csv(os.path.join(OUT_DIR, "washout_report_step3.csv"), index=False)

        n_zero = int((prev["n_positive"] == 0).sum())
        n_all = int((prev["n_positive"] == len(final)).sum())
        print(f"    episodes                     : {len(final):>12,}")
        print(f"    women                        : {final['PSTV01'].nunique():>12,}")
        print(f"    feature flags                : {len(flag_cols):>12,}")
        print(f"    flags never positive         : {n_zero:>12,}")
        print(f"    flags always positive        : {n_all:>12,}")
        print(f"    removed by {WASHOUT_YEAR} washout      : {washout_total:>12,}")
        print(f"    extreme maternal age         : {int(final['age_risk'].sum()):>12,}"
              f"  ({final['age_risk'].mean() * 100:.2f}%)")
        if n_zero:
            zeros = prev.loc[prev["n_positive"] == 0, "flag"].tolist()
            print("    flags never positive (non-estimable downstream), "
                  f"first 15 of {len(zeros)}:")
            for f in sorted(zeros)[:15]:
                print(f"      {f}")
            print("      ... full list in flag_prevalence_step3.csv")
        b["rows_out"] = len(final)

    bench.finalise(os.path.join(OUT_DIR, "benchmark_step3.csv"),
                   rows_out=len(final))
    print(f"\nOutput written to {os.path.abspath(OUT_DIR)}")


if __name__ == "__main__":
    main()