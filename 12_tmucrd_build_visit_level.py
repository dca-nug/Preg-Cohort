"""
=============================================================================
 STEP 12 -- BUILD THE EXTERNAL VISIT-LEVEL TABLE (TMUCRD)
=============================================================================

Builds `tmucrd_pregnancy_by_visit.csv`, the external equivalent of step 1's
output, from the per-table CSVs written by the loader.

The point of the external cohort is that the reconstruction rules are applied
UNCHANGED. That claim only holds if the rules are literally the same code, so
this script imports them:

    separation interval, chained gap comparison,
    four gestational offsets, episode numbering   -> step 1 anchor_from_masks
    ICD-10 code sets, offsets, age bounds         -> step 1
    ICD-9 code sets                               -> icd_crosswalk.csv

What cannot be shared is the code MATCHING. The development cohort stores
three-character diagnosis categories and is matched exactly; this cohort stores
full codes across two coding systems and needs prefix matching. That difference
is a property of the data sources, not of the rules, and is stated as such.

CHANGES FROM THE EARLIER VERSION OF THIS SCRIPT
-----------------------------------------------
Four differences from the development cohort have been removed, because each
would have meant the rules were not in fact applied unchanged.

  1. Look-back window. Visits were previously discarded unless they fell within
     180 days before `ref_start`. In the development cohort the before-pregnancy
     phase for a chronic condition is every record on or before the anchor, with
     no lower bound. A 180-day cap suppresses chronic-condition prevalence
     systematically and is a plausible reason why so few associations were
     estimable here. The cap is gone; all visits of an included patient are
     retained and the phase boundaries alone decide what counts as before,
     during, or after.

  2. Follow-forward window. Visits more than 365 days after the anchor were
     likewise discarded. The development cohort has no such cap, so it is gone.

  3. Gestational offsets. Only two were applied, 140 days for abortive outcomes
     and 280 for every delivery. All four are now applied, matching step 1:
     140 abortive, 259 preterm, 294 post-term, 280 term. Preterm and post-term
     deliveries previously received a window shifted by up to 21 days relative
     to the development cohort.

  4. Age bounds. The filter was 15 to 55 years; it is now 12 to 55, imported
     from step 1.

One bug is also fixed. ICD-9 and ICD-10 prefixes were pooled into a single
match list, so ICD-10 V27x, a motorcycle collision, was counted as an outcome
of delivery because ICD-9 V27x denotes exactly that. Each recorded value is now
classified before matching.

Row-wise `DataFrame.apply` over the diagnosis columns has been replaced with
vectorised string operations.

INPUT
-----
    <TABLE_DIR>/ipd_basic.csv, opd_basic.csv, chr_basic.csv
        written by the loader from the SAS source tables
    icd_crosswalk.csv
    01_build_visit_level.py                for the rules and code sets

OUTPUT
------
    tmucrd_pregnancy_by_visit.csv          external visit-level table
    tmucrd_build_report.csv                counts at each stage
    tmucrd_anchor_diagnostics.csv          anchors and onset assignment by type

USAGE
-----
    python 12_tmucrd_build_visit_level.py
=============================================================================
"""

from __future__ import annotations

import importlib
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


os.chdir(Path(__file__).resolve().parent)

# =============================================================================
# CONFIGURATION
# =============================================================================

TABLE_DIR = Path(r"F:\Olahdata\TMUCRD June\outputs\tables")
OUT_DIR   = Path(r"./output_step12")
OUT_NAME  = "tmucrd_pregnancy_by_visit.csv"

CROSSWALK    = Path(r"./icd_crosswalk.csv")
STEP1_MODULE = "01_build_visit_level"

ID_COL   = "CHR_NO"
DATE_COL = "visit_date"

# Diagnosis columns per source table.
IPD_ICD_COLS = ["MDIAG_CODE", "EDIAG_CODE",
                "SDIAG_CODE1", "SDIAG_CODE2", "SDIAG_CODE3", "SDIAG_CODE4",
                "SDIAG_CODE5", "SDIAG_CODE6", "SDIAG_CODE7",
                "ESDIAG_CODE1", "ESDIAG_CODE2", "ESDIAG_CODE3", "ESDIAG_CODE4",
                "ICD10_CODE1_OUT", "ICD10_CODE2_OUT", "ICD10_CODE3_OUT",
                "ICD10_CODE4_OUT", "ICD10_CODE5_OUT", "ICD10_CODE6_OUT",
                "ICD10_CODE7_OUT", "ICD10_CODE8_OUT"]
OPD_ICD_COLS = ["MDIAG_CODE", "SDIAG_CODE1", "SDIAG_CODE2",
                "SDIAG_CODE3", "SDIAG_CODE4", "SDIAG_CODE5"]

# Candidate date columns, tried in order.
IPD_DATE_CANDIDATES = ["IPD_DATE", "CPD_DATE"]
OPD_DATE_CANDIDATES = ["OPD_DATE", "REG_DATE", "VISIT_DATE"]

# Hospital identifier to region. Taipei hospitals are 0, New Taipei is 1. This
# is a different construct from the development cohort's Java-Bali variable and
# is not comparable across systems; it is retained for the descriptive table.
HOSPITAL_TO_REGION = {0: 0, 1: 0, 2: 1}

# The washout is NOT applied here. In the development cohort it lives inside
# the per-order feature extraction, where it filters records on the mapped
# ref_year; step 13 does the same. Applying it here, by voiding columns in the
# visit-level table, would be a different mechanism and would leave behind the
# termination records that the separation rule had absorbed into the removed
# episode.

# Taiwan adopted ICD-10-CM on 1 January 2016. Used only to resolve values that
# are ambiguous between the two systems.
TRANSITION = pd.Timestamp("2016-01-01")

# Anchor conditions, resolved against the crosswalk for their ICD-9 codes.
ABORT_CONDITIONS    = ["abortive"]
PARTUS_CONDITIONS   = ["normal", "instrum", "caesar", "assisted", "multiple"]
PRETERM_CONDITIONS  = ["preterm"]
POSTTERM_CONDITIONS = ["prolong"]

# =============================================================================


def import_rules():
    """Code sets, offsets, and age bounds from step 1; ICD-9 from the crosswalk."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        s1 = importlib.import_module(STEP1_MODULE)
    except Exception as exc:
        sys.exit(f"[FATAL] could not import {STEP1_MODULE}.py: {exc}")
    if not CROSSWALK.exists():
        sys.exit(f"[FATAL] {CROSSWALK} not found. Generate it first:\n"
                 "        python make_icd_crosswalk.py --html supplementary.html")
    cw = pd.read_csv(CROSSWALK).set_index("code")

    def icd9(conds):
        out = []
        for c in conds:
            if c not in cw.index:
                print(f"    [WARNING] '{c}' has no crosswalk row; its ICD-9 "
                      f"codes cannot be matched")
                continue
            out += [x for x in str(cw.loc[c, "icd9"]).split(";")
                    if x and x != "nan"]
        return sorted(set(out))

    rules = {
        "gap": int(s1.SEPARATION_DAYS),
        "days_abort": int(s1.DAYS_ABORT), "days_preterm": int(s1.DAYS_PRETERM),
        "days_postterm": int(s1.DAYS_POSTTERM), "days_term": int(s1.DAYS_TERM),
        "age_min": int(s1.AGE_MIN), "age_max": int(s1.AGE_MAX),
        "abort10": [c.upper() for c in s1.ABORT_CODES],
        "partus10": [c.upper() for c in s1.PARTUS_CODES],
        "preterm10": [c.upper() for c in s1.PRETERM_CODES],
        "postterm10": [c.upper() for c in s1.POSTTERM_CODES],
        "abort9": icd9(ABORT_CONDITIONS),
        "partus9": icd9(PARTUS_CONDITIONS),
        "preterm9": icd9(PRETERM_CONDITIONS),
        "postterm9": icd9(POSTTERM_CONDITIONS),
        "age_risk_low": int(getattr(s1, "AGE_RISK_LOW", 20)),
        "age_risk_high": int(getattr(s1, "AGE_RISK_HIGH", 35)),
    }
    print(f"    separation interval  : {rules['gap']} days")
    print(f"    gestational offsets  : abortive {rules['days_abort']}, "
          f"preterm {rules['days_preterm']}, term {rules['days_term']}, "
          f"post-term {rules['days_postterm']}")
    print(f"    plausible age        : {rules['age_min']} to {rules['age_max']} years")
    for k in ("abort", "partus", "preterm", "postterm"):
        print(f"    {k:<9} ICD-10 {rules[k + '10']}")
        print(f"    {k:<9} ICD-9  {rules[k + '9']}")
    return rules


def clean_code(value) -> str:
    """Upper-case, drop a trailing '.0' from float-parsed numeric codes, then
    remove the decimal separator."""
    v = str(value).strip().upper()
    if v.endswith(".0"):
        v = v[:-2]
    return v.replace(".", "")


def classify(v: str) -> str:
    if not v or v == "NAN":
        return "empty"
    if v[0].isdigit():
        return "icd9"
    if v[0] in ("V", "E") and v[1:2].isdigit():
        return "ambiguous"
    if v[0].isalpha() and v[1:2].isdigit():
        return "icd10"
    return "empty"


def parse_roc_date(series: pd.Series) -> pd.Series:
    """TMUCRD stores some dates as Republic of China years, YYYMMDD, where the
    year is 1911 less than the Gregorian year. Values that are not in that form
    are parsed normally."""
    def one(val):
        if pd.isna(val):
            return pd.NaT
        try:
            n = int(float(val))
        except (TypeError, ValueError):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return pd.to_datetime(val, errors="coerce")
        s = str(n).zfill(7)
        if len(s) == 7:
            try:
                return pd.Timestamp(year=int(s[:3]) + 1911,
                                    month=int(s[3:5]), day=int(s[5:7]))
            except ValueError:
                return pd.NaT
        return pd.NaT
    return series.map(one)


def load_table(name: str, usecols=None) -> pd.DataFrame:
    path = TABLE_DIR / f"{name}.csv"
    if not path.exists():
        print(f"    [WARNING] {path.name} not found")
        return pd.DataFrame()
    df = pd.read_csv(path, usecols=usecols, dtype=str, low_memory=False)
    df.columns = [c.upper() for c in df.columns]
    return df


def load_visits() -> tuple[pd.DataFrame, list[str], list[dict]]:
    """All inpatient and outpatient encounters, one row per encounter."""
    frames, report = [], []
    spec = [("ipd_basic", IPD_ICD_COLS, IPD_DATE_CANDIDATES, "IPD"),
            ("opd_basic", OPD_ICD_COLS, OPD_DATE_CANDIDATES, "OPD")]

    for name, icd_cols, date_cands, tag in spec:
        df = load_table(name)
        if df.empty:
            report.append({"table": name, "rows": 0, "note": "not found"})
            continue
        date_col = next((c for c in date_cands if c in df.columns), None)
        if date_col is None:
            print(f"    [WARNING] {name}: no date column among {date_cands}")
            report.append({"table": name, "rows": len(df),
                           "note": "no date column"})
            continue
        present = [c for c in icd_cols if c in df.columns]
        missing = [c for c in icd_cols if c not in df.columns]
        if missing:
            print(f"    {name}: {len(missing)} configured diagnosis columns "
                  f"absent: {missing}")

        hosp = next((c for c in ("HOSPITAL_ID",) if c in df.columns), None)
        keep = [ID_COL, date_col] + present + ([hosp] if hosp else [])
        v = df[keep].copy().rename(columns={date_col: DATE_COL})
        if hosp:
            v = v.rename(columns={hosp: "hospital_id"})
        else:
            v["hospital_id"] = np.nan
        v["source"] = tag
        v[DATE_COL] = parse_roc_date(v[DATE_COL])
        frames.append(v)
        print(f"    {name:<12} {len(v):>10,} rows, {len(present)} diagnosis "
              f"columns, date from {date_col}")
        report.append({"table": name, "rows": len(v), "date_column": date_col,
                       "icd_columns": len(present), "note": "ok"})
        del df

    if not frames:
        sys.exit("[FATAL] no encounter data could be loaded from "
                 f"{TABLE_DIR.resolve()}")

    visits = pd.concat(frames, ignore_index=True, sort=False)
    visits[DATE_COL] = pd.to_datetime(visits[DATE_COL], errors="coerce")
    before = len(visits)
    visits = visits.dropna(subset=[ID_COL, DATE_COL])
    if before - len(visits):
        print(f"    dropped for missing identifier or date: "
              f"{before - len(visits):,}")
    dx = [c for c in dict.fromkeys(IPD_ICD_COLS + OPD_ICD_COLS)
          if c in visits.columns]
    print(f"    total {len(visits):,} encounters from "
          f"{visits[ID_COL].nunique():,} patients, {len(dx)} diagnosis columns")
    return visits.reset_index(drop=True), dx, report


def build_masks(visits: pd.DataFrame, dx: list[str], rules: dict):
    """Boolean masks for each anchor type, matching each coding system separately.

    Codes are matched as prefixes because this cohort stores full codes, and each
    recorded value is classified first so that a value valid in both systems is
    not matched against the wrong one.
    """
    norm, kind = {}, {}
    for c in dx:
        v = visits[c].map(lambda x: clean_code(x) if pd.notna(x) else "")
        norm[c] = pd.Series(v, index=visits.index, dtype="string").fillna("")
        kind[c] = norm[c].map(classify)
    norm = pd.DataFrame(norm, index=visits.index)
    kind = pd.DataFrame(kind, index=visits.index)

    icd10_only = {c for c in dx if "ICD10" in c.upper()}
    pre = visits[DATE_COL] < TRANSITION

    tally = []

    def mask_for(codes10, codes9, label):
        hit10 = pd.Series(False, index=visits.index)
        hit9 = pd.Series(False, index=visits.index)
        for c in dx:
            col, k = norm[c], kind[c]
            allow10 = (k == "icd10") | ((k == "ambiguous") & (c in icd10_only))
            allow9 = (k == "icd9") | ((k == "ambiguous")
                                      & (c not in icd10_only) & pre)
            for p in codes10:
                hit10 |= col.str.startswith(p, na=False) & allow10
            for p in codes9:
                hit9 |= col.str.startswith(p, na=False) & allow9
        tally.append({"anchor_type": label,
                      "matched_icd10_only": int((hit10 & ~hit9).sum()),
                      "matched_icd9_only": int((hit9 & ~hit10).sum()),
                      "matched_both": int((hit10 & hit9).sum()),
                      "matched_total": int((hit10 | hit9).sum()),
                      "pre_transition": int(((hit10 | hit9) & pre).sum()),
                      "post_transition": int(((hit10 | hit9) & ~pre).sum())})
        return hit10 | hit9

    m = {
        "abortus": mask_for(rules["abort10"], rules["abort9"], "abortive"),
        "partus": mask_for(rules["partus10"], rules["partus9"], "delivery"),
        "preterm": mask_for(rules["preterm10"], rules["preterm9"], "preterm"),
        "postterm": mask_for(rules["postterm10"], rules["postterm9"],
                             "post-term"),
    }
    t = pd.DataFrame(tally)
    print(f"    records before {TRANSITION.date()} : {int(pre.sum()):>10,}")
    print(f"    records on or after               : {int((~pre).sum()):>10,}")
    print(t.to_string(index=False))
    if t.loc[t["anchor_type"] == "abortive", "matched_total"].iloc[0] == 0 and \
            t.loc[t["anchor_type"] == "delivery", "matched_total"].iloc[0] == 0:
        print("\n[FATAL] no termination codes matched. Check the diagnosis "
              "column names and the recorded code format.")
        for c in dx[:6]:
            ex = norm[c][norm[c] != ""].unique()[:8]
            print(f"      {c}: {list(ex)}")
        sys.exit(1)
    return m, t


def add_demographics(visits: pd.DataFrame, rules: dict) -> tuple:
    """Per-encounter age, extreme-age indicator, and region.

    Mirrors the development pipeline exactly. There, age is the difference in
    calendar years between the encounter date and the date of birth, computed
    per encounter; the pregnancy-level value is then the minimum across the
    encounters assigned to the episode, and the age exclusion is applied to the
    episode afterwards.

    Neither a sex filter nor a patient-level age exclusion is applied, because
    the development pipeline applies neither. A patient-level rule would discard
    valid episodes of women observed across a long window, since a woman aged 50
    at her first encounter is 57 at her last.
    """
    chr_basic = load_table("chr_basic")
    if chr_basic.empty or "BIRTH_DATE" not in chr_basic.columns:
        print("    [WARNING] chr_basic unavailable; age cannot be derived")
        visits["age"] = pd.NA
        visits["age_risk"] = pd.NA
        return visits, 0, 0

    chr_basic["BIRTH_DATE"] = parse_roc_date(chr_basic["BIRTH_DATE"])
    chr_basic = chr_basic.drop_duplicates(ID_COL, keep="last")
    print(f"    birth dates available for {len(chr_basic):,} patients")

    visits = visits.merge(chr_basic[[ID_COL, "BIRTH_DATE"]], on=ID_COL,
                          how="left")
    # Age per encounter, from the encounter date, as in the development cohort.
    visits["age"] = (visits[DATE_COL].dt.year
                     - visits["BIRTH_DATE"].dt.year).astype("Int64")
    visits["age_risk"] = ((visits["age"] < rules["age_risk_low"])
                          | (visits["age"] > rules["age_risk_high"])
                          ).astype("Int64")
    visits["dom"] = pd.to_numeric(visits["hospital_id"], errors="coerce") \
        .map(HOSPITAL_TO_REGION).astype("Int64")
    visits["ref_year"] = visits["fin_g"].dt.year

    # Reported on the same terms as the development cohort's visit-level step:
    # counts are flagged, not acted on. The exclusion happens at the episode.
    age = pd.to_numeric(visits["age"], errors="coerce")
    n_missing = int(age.isna().sum())
    n_oob = int(((age < rules["age_min"]) | (age > rules["age_max"])).sum())
    n_ext = int(visits["age_risk"].fillna(0).astype(int).sum())
    print(f"    age missing                        : {n_missing:>12,}")
    print(f"    age outside {rules['age_min']}-{rules['age_max']} "
          f"(flagged only)     : {n_oob:>12,}")
    print(f"    extreme age (<{rules['age_risk_low']} or "
          f">{rules['age_risk_high']})           : {n_ext:>12,}")
    print("    the age exclusion is applied to the episode in step 13, "
          "as at the pregnancy level of the development cohort")
    return visits.drop(columns=["BIRTH_DATE"], errors="ignore"), \
        visits[ID_COL].nunique(), visits[ID_COL].nunique()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 74)
    print(" STEP 12 -- BUILD THE EXTERNAL VISIT-LEVEL TABLE")
    print("=" * 74)

    print("\n[1] Rules imported from the development pipeline")
    rules = import_rules()

    print("\n[2] Load encounters")
    visits, dx, load_report = load_visits()

    print("\n[3] Match anchor codes, each coding system separately")
    masks, tally = build_masks(visits, dx, rules)

    print("\n[4] Anchor episodes using the shared rule")
    s1 = importlib.import_module(STEP1_MODULE)
    visits, diag = s1.anchor_from_masks(
        visits, rules["gap"],
        abortus_mask=masks["abortus"], partus_mask=masks["partus"],
        preterm_mask=masks["preterm"], postterm_mask=masks["postterm"],
        days_abort=rules["days_abort"], days_preterm=rules["days_preterm"],
        days_postterm=rules["days_postterm"], days_term=rules["days_term"],
        id_col=ID_COL, date_col=DATE_COL)

    print("\n[5] Retain the patients who have at least one episode")
    with_ep = visits.loc[visits["fin_g"].notna(), ID_COL].unique()
    n_before = visits[ID_COL].nunique()
    visits = visits[visits[ID_COL].isin(with_ep)]
    print(f"    patients with an episode : {len(with_ep):,} of {n_before:,}")
    print(f"    their encounters         : {len(visits):,}")
    print("    every encounter of an included patient is retained; there is no")
    print("    look-back or follow-forward window, matching the development")
    print("    cohort, where the phase boundaries alone decide the assignment")

    print("\n[6] Demographics and the age exclusion")
    visits, n_pat_before, n_pat_after = add_demographics(visits, rules)

    print("\n[7] Export")
    core = [ID_COL, DATE_COL, "fin_g", "ref_start", "n_preg", "doa", "dol",
            "dopt", "age", "age_risk", "dom", "hospital_id", "ref_year",
            "source"]
    cols = [c for c in core if c in visits.columns] + dx
    out = visits[cols].copy()
    out_path = OUT_DIR / OUT_NAME
    out.to_csv(out_path, index=False)
    print(f"    wrote {len(out):,} rows x {len(cols)} columns -> {out_path}")
    print(f"    patients {out[ID_COL].nunique():,}, episodes "
          f"{int(out['fin_g'].notna().sum()):,}")

    pd.DataFrame(load_report).to_csv(OUT_DIR / "tmucrd_build_report.csv",
                                     index=False)
    tally.to_csv(OUT_DIR / "tmucrd_anchor_diagnostics.csv", index=False)
    pd.DataFrame([diag]).to_csv(OUT_DIR / "tmucrd_anchor_summary.csv",
                                index=False)

    print(f"\n    Set DATE_COL = \"{DATE_COL}\" in step 13 before running it.")
    print(f"\nOutput written to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()