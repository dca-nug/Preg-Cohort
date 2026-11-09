"""
=============================================================================
 STEP 2 -- DISTRIBUTION OF INTERVALS BETWEEN CONSECUTIVE TERMINATIONS
=============================================================================

Produces the empirical justification for the episode separation threshold,
reported as Supplementary Figure 1.

The rule has to distinguish two things that look identical in claims data: a
termination code reappearing at follow-up visits belonging to the SAME
pregnancy, and a genuinely distinct subsequent pregnancy. The interval
distribution separates them where a trough exists between the initial peak and
the subsequent mode, and shows where it does not.

    first peak (0-40 days)   re-coding of the same pregnancy
    trough                   the empirically supported threshold
    second, flatter mode     the next, genuinely distinct pregnancy

Where no trough is present and the distribution decays smoothly past the
threshold, that is evidence the rule does not split single pregnancies. That is
also a reportable result and is stated as such rather than left ambiguous.

WHY THIS RUNS ON STEP 1's OUTPUT
--------------------------------
The input must be the visit-level table BEFORE the separation rule collapses
anything, and it must still carry the raw termination dates. Step 1 retains
`doa`, `dol`, and `dopt` for exactly this reason. The original notebook dropped
them, leaving `fin_g` as the only date available, and `fin_g` holds anchor dates
only: every interval below the threshold has already been removed from it, so an
analysis based on `fin_g` finds an empty left-hand side and would conclude,
wrongly, that short intervals do not occur.

If those columns are absent, MODE "C" rebuilds them from `combined_date` and the
diagnosis columns using the same code sets as step 1.

CODE SETS ARE IMPORTED
----------------------
The abortive and delivery code lists, the diagnosis columns, and the separation
threshold are imported from step 1 rather than restated here, so this analysis
cannot describe a threshold other than the one the pipeline applied.

CHAIN ABSORPTION
----------------
The separation interval is measured against the immediately preceding
termination record rather than the last accepted anchor, so a chain of records
each closer than the threshold is absorbed into one episode however long the
total span. That is a deliberate property of the pipeline, and this step
quantifies it: how many episodes absorb more than one record, how far those
episodes span, and how many exceed a duration no single pregnancy can have.

INPUT
-----
    pregnancy by visit.csv               from step 1

OUTPUT
------
    Supplementary_Figure_1.png / .pdf    2 x 2 panel, one per termination pair
    interval_summary.csv                 median, IQR, counts below each threshold
    coding_tail_by_preceding_type.png    normalised tail comparison
    threshold_simulation.csv             episode counts under each candidate rule
    chain_absorption.csv                 span and record count per episode
    interval_pairs_raw.csv               pair-level data
    hist_*.png                           single-panel versions
    benchmark_step2.csv                 per-stage timing and memory
    environment_step2.txt               library versions and machine specification

USAGE
-----
    python 02_interval_distribution.py
=============================================================================
"""

import importlib
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipeline_utils import Bench, environment_report, warn_if_no_psutil

# Relative paths in the configuration block below are resolved against the
# folder holding this script, not the shell's working directory. Without this,
# launching from an editor's Run button resolves them against the editor's
# install directory instead, and every path fails.
os.chdir(Path(__file__).resolve().parent)

# =============================================================================
# CONFIGURATION -- EDIT THIS BLOCK
# =============================================================================

# Visit-level file from step 1, before the separation rule collapses anything.
INPUT_FILE = r"./output_step1/pregnancy by visit.csv"

# Output folder
OUTDIR = r"./output_step2"

# Step 1 is the single source of the code sets and of the threshold.
STEP1_MODULE = "01_build_visit_level"

# --- Column names -----------------------------------------------------------
PID_COL = "PSTV01"          # patient identifier

# Three input layouts are supported. Pick one.
#
# MODE "A" : two separate date columns, DOA (abortive) and DOL (delivery)
# MODE "B" : a single DOPT column plus a binary type indicator
# MODE "C" : rebuild doa/dol/dopt from combined_date + the four diagnosis
#            columns (notebook cell 8 logic, verbatim). USE THIS for
#            "pregnancy by visit cleaned.csv" or "pregnancy by visit.csv".
MODE = "A"   # step 1 retains doa/dol; use "C" if they are absent

# Used when MODE = "A"
DOA_COL = "DOA"             # date of abortive termination
DOL_COL = "DOL"             # date of labour

# Used when MODE = "B"
DOPT_COL = "DOPT"
ABORTIVE_FLAG_COL = "c_abortive"   # 1 = abortive, 0 = delivery

# Used when MODE = "A"   # step 1 retains doa/dol; use "C" if they are absent -- identical to notebook cell 8
COMBINED_DATE_COL = "combined_date"
FKP03_COL = "FKP03"          # fallback if combined_date is absent or empty
FKL03_COL = "FKL03"
DX_COLS      = ["FKP14A", "FKL15A", "FKL17A", "FKL24A"]
ABORT_CODES  = ["O00", "O01", "O02", "O03", "O04", "O05", "O06", "O07"]
PARTUS_CODES = ["O80", "O81", "O82", "O83", "O84"]

# --- Analysis parameters ----------------------------------------------------
MAX_DAYS   = 400            # x-axis limit; intervals beyond this are reported
                            # in the caption rather than piled into the last bin
BIN_WIDTH  = 7              # bin width, days
SMOOTH_WIN = 5              # moving-average window for trough detection (bins)
VALLEY_LO  = 14             # search for a trough from this day
VALLEY_HI  = 400            # ... to this day
VALLEY_MIN_DENSITY = 0.02   # a trough is only reported if its smoothed height
                            # is at least this fraction of the peak height.
                            # Prevents spurious troughs in near-empty regions.

# Thresholds reported in the summary table (numbers only, not drawn)
THRESHOLDS = [42, 60, 90, 105, 120, 180, 182, 196, 259, 300]

# Vertical reference lines drawn on the figures
THRESHOLD_LINES = [182]
LINE_COLORS     = ["#8172B2"]

# Sensitivity scheme, by type of the LATER record of each pair (step 1's
# SEPARATION_SENS). Drawn per panel as a dotted line.
SEPARATION_SENS = {"delivery": 259, "abortive": 105}
SENS_COLOR      = "#DD8452"

# Moving average: useful for trough detection, but visually swamped in panels
# where a single bin dominates the y-scale. Off by default for publication.
SHOW_MOVING_AVG = False
LOG_Y           = False      # True = log y-axis, makes small modes visible

PAIR_ORDER = ["abortive -> abortive", "abortive -> delivery",
              "delivery -> abortive", "delivery -> delivery"]

PAIR_LABEL = {"abortive -> abortive": "Abortive \u2192 abortive",
              "abortive -> delivery": "Abortive \u2192 delivery",
              "delivery -> abortive": "Delivery \u2192 abortive",
              "delivery -> delivery": "Delivery \u2192 delivery"}


def _import_step1_constants() -> str:
    """Override the code sets and the threshold with step 1's definitions."""
    global DX_COLS, ABORT_CODES, PARTUS_CODES, THRESHOLD_LINES, SEPARATION_SENS
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        m = importlib.import_module(STEP1_MODULE)
    except Exception as exc:
        return (f"[NOTE] {STEP1_MODULE}.py not importable ({exc}); "
                "the values written in this file are used instead")
    DX_COLS = list(m.DX_COLS)
    ABORT_CODES = list(m.ABORT_CODES)
    PARTUS_CODES = list(m.PARTUS_CODES)
    THRESHOLD_LINES = [int(m.SEPARATION_DAYS)]
    SEPARATION_SENS = dict(getattr(m, "SEPARATION_SENS", SEPARATION_SENS))
    return (f"code sets and the {m.SEPARATION_DAYS}-day threshold imported "
            f"from {STEP1_MODULE}.py")

# =============================================================================


def log(msg):
    print(msg, flush=True)


def load_data(path):
    """Read a CSV or Parquet file, with diagnostics."""
    if not os.path.exists(path):
        sys.exit(f"[FATAL] File not found: {os.path.abspath(path)}")

    ext = os.path.splitext(path)[1].lower()
    log(f"[1] Reading {path} ...")
    if ext in (".parquet", ".pq"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, low_memory=False)
    log(f"    rows = {len(df):,}   columns = {len(df.columns)}")
    return df


def check_columns(df, needed):
    """Verify that the required columns exist; list candidates if they do not."""
    lower = {c.lower(): c for c in df.columns}
    missing = [c for c in needed if c.lower() not in lower]
    if missing:
        log("\n[FATAL] The following columns were not found: " + ", ".join(missing))
        log("\nColumns present in the file:")
        for c in sorted(df.columns):
            log(f"    {c}")
        sys.exit(1)
    return {c: lower[c.lower()] for c in needed}


def build_event_table(df):
    """
    Build a termination-event table: one row per (patient, date, type).

    Several visits may carry the same termination date, so exact duplicates are
    removed first. This is NOT the separation rule -- it only drops identical rows.
    """
    if MODE.upper() == "A":
        cols = check_columns(df, [PID_COL, DOA_COL, DOL_COL])
        pid, doa, dol = cols[PID_COL], cols[DOA_COL], cols[DOL_COL]

        a = df.loc[df[doa].notna(), [pid, doa]].copy()
        a.columns = ["pid", "dopt"]
        a["kind"] = "abortive"

        d = df.loc[df[dol].notna(), [pid, dol]].copy()
        d.columns = ["pid", "dopt"]
        d["kind"] = "delivery"

        ev = pd.concat([a, d], ignore_index=True)
        log(f"[2] Raw termination events: abortive = {len(a):,}   "
            f"delivery = {len(d):,}   total = {len(ev):,}")

    elif MODE.upper() == "C":
        # Rebuild doa/dol/dopt from combined_date + the diagnosis columns.
        # Logic identical to notebook cell 8 -- nothing simplified.
        cols = check_columns(df, [PID_COL] + DX_COLS)
        pid = cols[PID_COL]
        dxc = [cols[c] for c in DX_COLS]

        if COMBINED_DATE_COL in df.columns:
            cd = pd.to_datetime(df[COMBINED_DATE_COL], errors="coerce")
            log(f"    using column '{COMBINED_DATE_COL}'")
        else:
            check_columns(df, [FKP03_COL, FKL03_COL])
            a = pd.to_datetime(df[FKP03_COL], errors="coerce")
            b = pd.to_datetime(df[FKL03_COL], errors="coerce")
            cd = pd.concat([a, b], axis=1).min(axis=1)
            log(f"    combined_date derived as min({FKP03_COL}, {FKL03_COL})")

        abortus_mask = df[dxc].isin(ABORT_CODES).any(axis=1)
        partus_mask  = df[dxc].isin(PARTUS_CODES).any(axis=1)
        log(f"    visits with an abortive code (O00-O07): {int(abortus_mask.sum()):,}")
        log(f"    visits with a delivery code (O80-O84): {int(partus_mask.sum()):,}")
        both = int((abortus_mask & partus_mask).sum())
        if both:
            log(f"    visits carrying BOTH: {both:,} -> treated as abortive "
                f"(consistent with dopt = doa.fillna(dol))")
        if abortus_mask.sum() == 0 and partus_mask.sum() == 0:
            log("\n[FATAL] No codes matched. The diagnosis columns may hold codes "
                "longer than three characters (e.g. 'O030').")
            for c in dxc:
                vals = df[c].dropna().astype(str)
                ex = vals[vals.str.startswith("O")].unique()[:12]
                log(f"    {c}: example values beginning with O -> {list(ex)}")
            sys.exit(1)

        doa = cd.where(abortus_mask)
        dol = cd.where(partus_mask)
        dopt = doa.fillna(dol)

        ev = pd.DataFrame({"pid": df[pid], "dopt": dopt})
        ev["kind"] = np.where(abortus_mask, "abortive", "delivery")
        ev = ev[ev["dopt"].notna()]
        log(f"[2] Raw termination events: total = {len(ev):,}")

    elif MODE.upper() == "B":
        cols = check_columns(df, [PID_COL, DOPT_COL, ABORTIVE_FLAG_COL])
        pid, dopt, flag = cols[PID_COL], cols[DOPT_COL], cols[ABORTIVE_FLAG_COL]

        ev = df.loc[df[dopt].notna(), [pid, dopt, flag]].copy()
        ev.columns = ["pid", "dopt", "flag"]
        ev["kind"] = np.where(pd.to_numeric(ev["flag"], errors="coerce").fillna(0) == 1,
                              "abortive", "delivery")
        ev = ev.drop(columns=["flag"])
        log(f"[2] Raw termination events: total = {len(ev):,}")
    else:
        sys.exit("[FATAL] MODE must be 'A', 'B', or 'C'.")

    before = len(ev)
    ev["dopt"] = pd.to_datetime(ev["dopt"], errors="coerce")
    ev = ev.dropna(subset=["dopt", "pid"])
    if before - len(ev):
        log(f"    dropped for invalid date or identifier: {before - len(ev):,}")

    before = len(ev)
    ev = ev.drop_duplicates(subset=["pid", "dopt", "kind"])
    log(f"    dropped exact duplicates: {before - len(ev):,}")

    # If the same patient has both an abortive and a delivery code on the same
    # date, abortive wins (consistent with dopt = doa if present, else dol).
    before = len(ev)
    ev = ev.sort_values(["pid", "dopt", "kind"])      # 'abortive' sorts before 'delivery'
    ev = ev.drop_duplicates(subset=["pid", "dopt"], keep="first")
    if before - len(ev):
        log(f"    same-date abortive+delivery conflicts "
            f"(abortive retained): {before - len(ev):,}")

    log(f"    final unique events: {len(ev):,} from {ev['pid'].nunique():,} patients")
    return ev.reset_index(drop=True)


def compute_intervals(ev):
    """Intervals in days between consecutive termination events, within patient."""
    ev = ev.sort_values(["pid", "dopt"]).reset_index(drop=True)
    ev["prev_dopt"] = ev.groupby("pid")["dopt"].shift(1)
    ev["prev_kind"] = ev.groupby("pid")["kind"].shift(1)

    iv = ev.dropna(subset=["prev_dopt"]).copy()
    iv["interval_days"] = (iv["dopt"] - iv["prev_dopt"]).dt.days
    iv["pair"] = iv["prev_kind"] + " -> " + iv["kind"]

    log(f"[3] Consecutive pairs formed: {len(iv):,}")
    log(f"    patients with >=2 events: {iv['pid'].nunique():,}")
    neg = (iv["interval_days"] < 0).sum()
    if neg:
        log(f"    [WARNING] negative intervals: {neg:,} -- dropped")
        iv = iv[iv["interval_days"] >= 0]
    return iv


def find_valley(x, counts):
    """
    Locate the lowest point between the first peak and the second mode.

    Returns (day, height) or (None, None) if no credible trough exists. A trough
    is rejected when it sits at the edge of the search window, or when its
    smoothed height falls below VALLEY_MIN_DENSITY of the peak height -- the
    latter guards against spurious minima in near-empty regions.
    """
    if len(counts) < SMOOTH_WIN + 2:
        return None, None
    sm = pd.Series(counts).rolling(SMOOTH_WIN, center=True, min_periods=1).mean().values
    peak = sm.max()
    if peak <= 0:
        return None, None
    mask = (x >= VALLEY_LO) & (x <= VALLEY_HI)
    if mask.sum() < 3:
        return None, None
    xs, ys = x[mask], sm[mask]
    i = int(np.argmin(ys))
    if i == 0 or i == len(ys) - 1:
        return None, None
    if ys[i] < VALLEY_MIN_DENSITY * peak:
        return None, None          # region is essentially empty; no real trough
    return float(xs[i]), float(ys[i])


def summarise(iv, outdir):
    """Summary table: pairs below each threshold, by pair type."""
    rows = []
    for pair, g in iv.groupby("pair"):
        n = len(g)
        row = {"pair": pair, "n_pairs": n,
               "median_days": g["interval_days"].median(),
               "p25": g["interval_days"].quantile(.25),
               "p75": g["interval_days"].quantile(.75),
               f"beyond_{MAX_DAYS}d_n": int((g["interval_days"] > MAX_DAYS).sum())}
        for t in THRESHOLDS:
            k = int((g["interval_days"] < t).sum())
            row[f"<{t}d_n"] = k
            row[f"<{t}d_pct"] = round(k / n * 100, 2) if n else np.nan
        rows.append(row)

    n = len(iv)
    tot = {"pair": "ALL", "n_pairs": n,
           "median_days": iv["interval_days"].median(),
           "p25": iv["interval_days"].quantile(.25),
           "p75": iv["interval_days"].quantile(.75),
           f"beyond_{MAX_DAYS}d_n": int((iv["interval_days"] > MAX_DAYS).sum())}
    for t in THRESHOLDS:
        k = int((iv["interval_days"] < t).sum())
        tot[f"<{t}d_n"] = k
        tot[f"<{t}d_pct"] = round(k / n * 100, 2) if n else np.nan
    rows.append(tot)

    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(outdir, "interval_summary.csv"), index=False)
    return summary


def draw_panel(ax, g, pair, show_legend=False):
    """Draw one histogram panel. Intervals beyond MAX_DAYS are excluded, not clipped."""
    vals = g.loc[g["interval_days"] <= MAX_DAYS, "interval_days"]
    beyond = int((g["interval_days"] > MAX_DAYS).sum())

    bins = np.arange(0, MAX_DAYS + BIN_WIDTH, BIN_WIDTH)
    counts, edges = np.histogram(vals, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2

    vx, _ = find_valley(centers, counts)

    ax.bar(centers, counts, width=BIN_WIDTH * .9, color="#4C72B0",
           edgecolor="none", alpha=.85)
    if SHOW_MOVING_AVG:
        sm = pd.Series(counts).rolling(SMOOTH_WIN, center=True, min_periods=1).mean()
        ax.plot(centers, sm, color="#C44E52", lw=1.6,
                label=f"Moving average ({SMOOTH_WIN} bins)")

    for t, c in zip(THRESHOLD_LINES, LINE_COLORS):
        if t <= MAX_DAYS:
            ax.axvline(t, ls="--", lw=1.4, color=c, label=f"{t}-day threshold")
    later = pair.split(" -> ")[-1] if " -> " in pair else None
    if later in SEPARATION_SENS and SEPARATION_SENS[later] <= MAX_DAYS:
        ax.axvline(SEPARATION_SENS[later], ls=":", lw=1.6, color=SENS_COLOR,
                   label=f"{SEPARATION_SENS[later]}-day threshold (sensitivity)")
    if vx is not None:
        ax.axvline(vx, ls="-", lw=2.0, color="black", alpha=.8,
                   label=f"Empirical trough = {vx:.0f} days")

    ax.set_title(f"{PAIR_LABEL.get(pair, pair)}  (n = {len(g):,})", fontsize=11)
    ax.set_xlim(0, MAX_DAYS)
    if LOG_Y:
        ax.set_yscale("log")
    ax.spines[["top", "right"]].set_visible(False)
    if show_legend:
        ax.legend(fontsize=8, frameon=False)
    return vx, beyond


def plot_individual(iv, pair, outdir):
    """Single-panel figure for one pair type."""
    g = iv[iv["pair"] == pair]
    if len(g) < 50:
        log(f"    skipped '{pair}' (only {len(g)} pairs)")
        return None
    fig, ax = plt.subplots(figsize=(10, 5.5))
    vx, beyond = draw_panel(ax, g, pair, show_legend=True)
    ax.set_xlabel("Interval between consecutive terminations (days)")
    ax.set_ylabel("Number of pairs")
    fig.tight_layout()
    fname = "hist_" + pair.replace(" -> ", "_to_").replace(" ", "") + ".png"
    fig.savefig(os.path.join(outdir, fname), dpi=300)
    plt.close(fig)
    log(f"    saved: {fname}"
        + (f"   trough = {vx:.0f} days" if vx else "   trough = not detected")
        + f"   (beyond {MAX_DAYS} d: {beyond:,})")
    return vx


def plot_supp_figure1(iv, outdir):
    """Publication panel: 2 x 2 grid, one panel per pair type."""
    pairs = [p for p in PAIR_ORDER if (iv["pair"] == p).sum() >= 50]
    if not pairs:
        return {}
    ncol = 2
    nrow = int(np.ceil(len(pairs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(12, 4.2 * nrow), squeeze=False)
    troughs, beyond_total = {}, 0
    for i, pair in enumerate(pairs):
        ax = axes[i // ncol][i % ncol]
        vx, beyond = draw_panel(ax, iv[iv["pair"] == pair], pair, show_legend=(i == 0))
        if vx is not None:
            troughs[pair] = vx
        beyond_total += beyond
        ax.text(-0.08, 1.06, "ABCD"[i], transform=ax.transAxes,
                fontsize=13, fontweight="bold", va="top")
        if i // ncol == nrow - 1:
            ax.set_xlabel("Interval between consecutive terminations (days)")
        if i % ncol == 0:
            ax.set_ylabel("Number of pairs")
    for j in range(len(pairs), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "Supplementary_Figure_1.png"), dpi=300)
    fig.savefig(os.path.join(outdir, "Supplementary_Figure_1.pdf"))
    plt.close(fig)
    log(f"    saved: Supplementary_Figure_1.png / .pdf "
        f"(intervals beyond {MAX_DAYS} d excluded: {beyond_total:,})")
    return troughs


def plot_coding_tail(iv, outdir):
    """Normalised comparison of the coding tail by preceding termination type."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bins = np.arange(0, MAX_DAYS + BIN_WIDTH, BIN_WIDTH)
    colors = {"abortive": "#C44E52", "delivery": "#4C72B0"}
    labels = {"abortive": "After an abortive termination",
              "delivery": "After a delivery"}

    for kind in ["abortive", "delivery"]:
        g = iv[(iv["prev_kind"] == kind) & (iv["interval_days"] <= MAX_DAYS)]
        if len(g) < 50:
            continue
        counts, edges = np.histogram(g["interval_days"], bins=bins)
        centers = (edges[:-1] + edges[1:]) / 2
        dens = counts / counts.sum() * 100
        sm = pd.Series(dens).rolling(SMOOTH_WIN, center=True, min_periods=1).mean()
        ax.plot(centers, sm, lw=2, color=colors[kind],
                label=f"{labels[kind]} (n = {len(g):,})")

    for t, c in zip(THRESHOLD_LINES, LINE_COLORS):
        ax.axvline(t, ls="--", lw=1.4, color=c, label=f"{t}-day threshold")

    ax.set_xlabel("Interval to the next termination (days)")
    ax.set_ylabel("Percentage of pairs (%)")
    ax.set_title("Coding tail by type of preceding termination")
    ax.set_xlim(0, MAX_DAYS)
    ax.legend(fontsize=9, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "coding_tail_by_preceding_type.png"), dpi=300)
    plt.close(fig)
    log("    saved: coding_tail_by_preceding_type.png")


def simulate_thresholds(ev, outdir):
    """
    Number of episodes formed under each threshold rule.

    Compares uniform thresholds with the two-tier rule used for French national
    claims (28 weeks = 196 days after a birth, 42 days after a termination).
    The type used is that of the PRECEDING event, since the question is how long
    the coding tail extends after that event.
    """
    ev = ev.sort_values(["pid", "dopt"]).reset_index(drop=True)
    gap = ev.groupby("pid")["dopt"].diff().dt.days
    prev_kind = ev.groupby("pid")["kind"].shift(1)

    rows = []
    for t in THRESHOLDS:
        new_ep = gap.isna() | (gap >= t)
        rows.append({
            "rule": f"uniform {t} days",
            "n_episodes": int(new_ep.sum()),
            "n_abortive_episodes": int((new_ep & (ev["kind"] == "abortive")).sum()),
            "n_delivery_episodes": int((new_ep & (ev["kind"] == "delivery")).sum()),
        })

    thr = np.where(prev_kind.eq("delivery"), 196, 42)
    new_ep = gap.isna() | (gap >= thr)
    rows.append({
        "rule": "two-tier (196 days after delivery / 42 days after abortive)",
        "n_episodes": int(new_ep.sum()),
        "n_abortive_episodes": int((new_ep & (ev["kind"] == "abortive")).sum()),
        "n_delivery_episodes": int((new_ep & (ev["kind"] == "delivery")).sum()),
    })

    # Sensitivity scheme of step 11: threshold set by the LATER record's type.
    thr = np.where(ev["kind"].eq("abortive"), SEPARATION_SENS["abortive"],
                   SEPARATION_SENS["delivery"])
    new_ep = gap.isna() | (gap >= thr)
    rows.append({
        "rule": (f"by later record ({SEPARATION_SENS['delivery']} days before a "
                 f"delivery / {SEPARATION_SENS['abortive']} days before an abortive)"),
        "n_episodes": int(new_ep.sum()),
        "n_abortive_episodes": int((new_ep & (ev["kind"] == "abortive")).sum()),
        "n_delivery_episodes": int((new_ep & (ev["kind"] == "delivery")).sum()),
    })

    sim = pd.DataFrame(rows)
    main_t = THRESHOLD_LINES[0] if THRESHOLD_LINES else 182
    base = sim.loc[sim["rule"] == f"uniform {main_t} days", "n_episodes"]
    if len(base):
        b = base.iloc[0]
        sim[f"diff_vs_{main_t}d"] = sim["n_episodes"] - b
        sim[f"pct_vs_{main_t}d"] = (sim["n_episodes"] / b * 100 - 100).round(2)
    sim.to_csv(os.path.join(outdir, "threshold_simulation.csv"), index=False)
    return sim


def chain_absorption(ev, gap=182):
    """
    Chained rule versus re-anchoring.

    Notebook cell 8 measures the gap against the PRECEDING ROW, not against the
    last accepted anchor:

        prev      = groupby('PSTV01')['dopt'].shift()
        is_anchor = dopt.notna() & (prev.isna() | (gap >= threshold))

    A chain of events in which every consecutive pair is closer than the threshold
    apart is therefore absorbed entirely into ONE episode, however long the
    total span. With 182 days, terminations on days 0, 100, 200, 300 and 400 yield a single
    anchor on day 0, so one episode spans 400 days.

    This function counts how many events are absorbed and how far the resulting
    episodes span, and compares the episode count against re-anchoring, where
    the gap is measured from the last accepted anchor.
    """
    ev = ev.sort_values(["pid", "dopt"]).reset_index(drop=True)
    pid = ev["pid"].to_numpy()
    day = ev["dopt"].values.astype("datetime64[D]").astype(np.int64)

    n = len(ev)
    anchor_chain = np.zeros(n, dtype=bool)
    anchor_re    = np.zeros(n, dtype=bool)
    chain_id     = np.zeros(n, dtype=np.int64)

    last_row = None
    last_anch = None
    cid = -1
    for i in range(n):
        new_pat = (i == 0) or (pid[i] != pid[i - 1])
        if new_pat:
            anchor_chain[i] = True
            anchor_re[i] = True
            last_row = day[i]
            last_anch = day[i]
            cid += 1
        else:
            if day[i] - last_row >= gap:
                anchor_chain[i] = True
                cid += 1
            last_row = day[i]
            if day[i] - last_anch >= gap:
                anchor_re[i] = True
                last_anch = day[i]
        chain_id[i] = cid

    ev["_chain"] = chain_id
    spans = ev.groupby("_chain").agg(
        n_events=("dopt", "size"),
        span_days=("dopt", lambda s: (s.max() - s.min()).days))
    absorbed = spans[spans["n_events"] > 1]

    log(f"[8] Chain-absorption diagnostics (gap = {gap} days)")
    log(f"    episodes under the chained rule (cell 8) : {int(anchor_chain.sum()):,}")
    log(f"    episodes under re-anchoring              : {int(anchor_re.sum()):,}")
    log(f"    difference                               : "
        f"{int(anchor_re.sum() - anchor_chain.sum()):+,}")
    log(f"    episodes absorbing >1 event              : {len(absorbed):,}")
    if len(absorbed):
        log(f"    span of absorbed episodes   median : {absorbed['span_days'].median():.0f} days")
        log(f"                                p95    : {absorbed['span_days'].quantile(.95):.0f} days")
        log(f"                                maximum: {absorbed['span_days'].max():.0f} days")
        over = int((absorbed["span_days"] > 300).sum())
        log(f"    episodes spanning >300 days              : {over:,}"
            f"   <-- upper bound on wrongly merged pregnancies")
    return spans, int(anchor_chain.sum()), int(anchor_re.sum())


def main() -> None:
    os.makedirs(OUTDIR, exist_ok=True)
    print("=" * 74)
    print(" STEP 2 -- INTERVALS BETWEEN CONSECUTIVE TERMINATIONS")
    print("=" * 74)
    print(environment_report(os.path.join(OUTDIR, "environment_step2.txt")))
    warn_if_no_psutil()
    print("\n" + _import_step1_constants())

    bench = Bench("step 2, interval distribution")

    with bench.stage("Load visit-level table", "pandas") as b:
        df = load_data(INPUT_FILE)
        b["rows_out"] = len(df)

    with bench.stage("Build termination event table", "pandas",
                     rows_in=len(df)) as b:
        ev = build_event_table(df)
        b["rows_out"] = len(ev)

    with bench.stage("Compute consecutive intervals", "pandas",
                     rows_in=len(ev)) as b:
        iv = compute_intervals(ev)
        if len(iv) == 0:
            sys.exit("[FATAL] no consecutive pairs found. Check the date columns.")
        b["rows_out"] = len(iv)

    with bench.stage("Summarise by pair type", "pandas", rows_in=len(iv)) as b:
        summary = summarise(iv, OUTDIR)
        log(summary.to_string(index=False))
        b["rows_out"] = len(summary)

    with bench.stage("Draw figures", "matplotlib", rows_in=len(iv)) as b:
        troughs = plot_supp_figure1(iv, OUTDIR)
        for pair in PAIR_ORDER:
            if (iv["pair"] == pair).sum() >= 50:
                plot_individual(iv, pair, OUTDIR)
        plot_coding_tail(iv, OUTDIR)
        b["rows_out"] = len(troughs)

    with bench.stage("Simulate candidate thresholds", "pandas",
                     rows_in=len(ev)) as b:
        sim = simulate_thresholds(ev, OUTDIR)
        log(sim.to_string(index=False))
        b["rows_out"] = len(sim)

    gap = THRESHOLD_LINES[0] if THRESHOLD_LINES else 182
    with bench.stage("Chain-absorption diagnostics", "python",
                     rows_in=len(ev)) as b:
        spans, n_chain, n_re = chain_absorption(ev, gap=gap)
        spans.to_csv(os.path.join(OUTDIR, "chain_absorption.csv"))
        b["rows_out"] = len(spans)

    log("\n    EMPIRICAL TROUGHS")
    if troughs:
        for k, v in troughs.items():
            log(f"      {k:<28} {v:.0f} days")
    else:
        log(f"      no credible trough between {VALLEY_LO} and {VALLEY_HI} days")
        log(f"      a smooth decay means the {gap}-day rule does not split")
        log("      single pregnancies, which is itself reportable")

    iv[["pid", "prev_dopt", "dopt", "prev_kind", "kind",
        "interval_days", "pair"]].to_csv(
        os.path.join(OUTDIR, "interval_pairs_raw.csv"), index=False)

    bench.finalise(os.path.join(OUTDIR, "benchmark_step2.csv"),
                   rows_out=len(iv))
    log(f"\nOutput written to {os.path.abspath(OUTDIR)}")
    log("  Supplementary_Figure_1.png / .pdf  -- 2x2 panel, one per pair type")
    log("  interval_summary.csv               -- summary by pair type")
    log("  threshold_simulation.csv           -- episode counts under each rule")
    log("  coding_tail_by_preceding_type.png  -- normalised tail comparison")
    log("  chain_absorption.csv               -- span and count per episode")
    log("  interval_pairs_raw.csv             -- pair-level data")


if __name__ == "__main__":
    main()
