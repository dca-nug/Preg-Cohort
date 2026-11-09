# BPJS-Preg-Cohort

Rule-based reconstruction and quality control of pregnancy episodes from Indonesian national health insurance (BPJS Kesehatan) claims that carry no gestational age or last menstrual period markers.

This repository holds the code for the manuscript:

> Nugroho DCA, et al. *Quality control and unit of analysis in pregnancy episodes reconstructed from Indonesian insurance claims without gestational age markers.* (in preparation)

The code builds a pregnancy-level dataset from visit-level claims, removes episodes that fail age, diagnostic-conflict and sequence checks, and measures how each check changes the associations drawn from the same records. The same rules are applied unchanged to the Taipei Medical University Clinical Research Database (TMUCRD).

No data are included. See [Data access](#data-access).

---

## Pipeline

Scripts are numbered in execution order. Each script writes to `output_stepN/`, where `N` is its own number, and reads only from the folders of earlier steps.

| Step | Script | Purpose | Main output |
|---|---|---|---|
| 01 | `01_build_visit_level.py` | Merge primary care, hospital and membership tables. Anchor episodes at abortive (O00–O07) or delivery (O80–O84) codes and back-calculate onset. | `pregnancy by visit.csv` |
| 02 | `02_interval_distribution.py` | Interval between consecutive terminations, by pair of termination types. Simulates episode counts under alternative separation rules. | `Supplementary_Figure_1`, `threshold_simulation.csv` |
| 03 | `03_build_pregnancy_level.py` | Assign diagnoses to before (`b_`), during (`c_`) and after (`a_`) phases. Collapse to one row per episode. Apply the 2015 washout. | `pregnancy by episode.csv` (pre-QC) |
| 03b | `03b_build_pregnancy_level_equalwindow.py` | Auxiliary. Step 03 with the antenatal window closed 140 days after onset for every episode. Used only by the ML-Preg sensitivity analysis. Not part of the main pipeline. | `pregnancy by episode.csv` in `output_step3b/` |
| 04 | `04_apply_pregnancy_qc.py` | Quality control: age, diagnostic conflict, episode sequence. Writes the analytic cohort roster. | `pregnancy by episode cleaned.csv`, `exclusion_report_step4.csv` |
| 05 | `05_build_individual_level.py` | Individual-level dataset, restricted to the analytic cohort. | `pregnancy by individual.csv` |
| 06 | `06_apply_age_exclusion.py` | Visit-level analytic dataset, restricted to the analytic cohort. | `pregnancy by visit cleaned.csv` |
| 07 | `07_table1_baseline.py` | Baseline characteristics at the three levels, with sum checks. | `table1_baseline.csv` |
| 08 | `08_odds_ratios.py` | Crude odds ratios for the abortive outcome, pregnancy level. | `or_long.csv`, `OR by episode.xlsx` |
| 09 | `09_odds_ratios_visit_individual.py` | Crude odds ratios, visit and individual levels. | `or_long_visit_individual.csv` |
| 10 | `10_forest_plots.py` | Forest plots across the three levels. | `1_baseline.png` … `4_delivery_outcomes.png` |
| 11 | `11_sensitivity_gap.py` | Reruns steps 01, 03, 04 and 08 under each separation scheme. | `sensitivity_or_comparison.csv`, `sensitivity_qc_counts.csv` |
| 12 | `12_tmucrd_build_visit_level.py` | TMUCRD visit-level table, anchored with the step 01 function. | `tmucrd_pregnancy_by_visit.csv` |
| 13 | `13_external_validation.py` | TMUCRD episodes, the same QC, odds ratios and direction agreement with BPJS. | `external_concordance_summary.csv`, QC reports |
| 14 | `14_qc_impact.py` | Odds ratios after each QC rule, and the QC yield in both systems. | `qc_switches.csv`, `qc_yield_by_system.csv` |
| 15 | `15_build_survival_visit_level.py` | Separate analysis. Counting-process dataset for time-to-event analysis. Not used in the manuscript above. | `survival_intervals.csv` |

`pipeline_utils.py` holds shared helpers (benchmarking, environment report, the person-level eligibility rule) and must sit in the same folder as the scripts.

### Run order

```
python 01_build_visit_level.py
python 02_interval_distribution.py
python 03_build_pregnancy_level.py
python 04_apply_pregnancy_qc.py
python 05_build_individual_level.py
python 06_apply_age_exclusion.py
python 07_table1_baseline.py
python 08_odds_ratios.py
python 09_odds_ratios_visit_individual.py
python 10_forest_plots.py
python 11_sensitivity_gap.py
python 12_tmucrd_build_visit_level.py
python 13_external_validation.py
python 14_qc_impact.py
```

Step 12 can run at any point before step 13. Step 14 reads the QC reports of step 13 and must run last. Steps 05 and 06 read the cohort roster written by step 04.

---

## Rules

All rules are set in the configuration block at the top of each script. Nothing is set on the command line.

### Episode anchoring (step 01)

| Rule | Value | Basis |
|---|---|---|
| Separation threshold, main analysis | 182 days (26 weeks) | Earliest return of ovulation after delivery in most women (6 weeks) plus the earliest gestation recorded as a delivery (20 weeks). |
| Separation threshold, sensitivity (step 11) | 259 days before a delivery record, 105 days before an abortive record | Empirical troughs of the interval distributions (step 02), rounded to whole weeks. |
| Onset, abortive | 140 days before termination | Upper limit of abortion (<20 weeks). |
| Onset, preterm (O60 at the termination visit) | 259 days | Upper limit of preterm birth (<37 weeks). |
| Onset, term | 280 days | Estimated date of delivery (40 weeks). |
| Onset, post-term (O48 at the termination visit) | 294 days | Lower limit of post-term (≥42 weeks). |

The gap is measured against the preceding termination record, not the last anchor. The episode type is that of its first record. An abortive code takes precedence when a record carries both.

### Quality control (step 04)

Applied in this order:

1. **Age.** Women with any recorded age outside 12–55 years are removed with all their episodes.
2. **Diagnostic conflict.** Abortive episodes with a condition incompatible with a completed abortion inside the during-pregnancy window are removed.
3. **Episode sequence.**
   - A delivery whose onset precedes the termination of a previous delivery is removed.
   - An abortive episode is removed when the next episode is a delivery whose onset precedes the abortive date by more than 14 days.
   - An abortive episode that contains a later delivery record is removed.
   - Overlaps in which the later episode is abortive are counted, not removed. They follow from the 140-day upper-limit duration.

Conflicting episodes are removed, not relabelled. The claims hold no information to decide which code is wrong.

---

## Data access

**BPJS Kesehatan.** The analysis uses the Contextual Maternal and Child Health sample (released 2023; women with a reimbursed delivery in 2018–2022, all encounters 2015–2022). The data cannot be redistributed. Access is requested from BPJS Kesehatan through its sample data portal (https://data.bpjs-kesehatan.go.id), which can be reached only from Indonesian internet addresses. Step 01 expects four Stata files in `DATA_DIR`:

```
FKTP_2023.dta              primary care visits
FKRTL_2023.dta             hospital visits
FKRTL_Sekunder_2023.dta    hospital secondary diagnoses
Kepesertaan_2023.dta       membership
```

**TMUCRD.** Not publicly available. Available from the Taipei Medical University Clinical Data Center on reasonable request. Step 12 expects `ipd_basic.csv`, `opd_basic.csv`, `chr_basic.csv` in `TABLE_DIR`, and `icd_crosswalk.csv`.

---

## Requirements

Python 3.11. Install with

```
pip install -r requirements.txt
```

`requirements-lock.txt` records the exact environment used for the published results. Each step also writes `environment_stepN.txt` with library versions and machine specification, and `benchmark_stepN.csv` with timing and memory per stage.

The full pipeline ran on a Windows workstation with 16 GB of memory. DuckDB handles the tables that do not fit in memory.

---

## Versions

| Version | Used by | Main differences |
|---|---|---|
| 1.x | ML-Preg (prediction of pregnancy complications) | Separation 180 days. No sequence QC. Sensitivity at 90/180/300 days. |
| 2.0.0 | This manuscript | See `CHANGELOG.md`. |

Results from the two versions are not interchangeable. Cite the version that produced the cohort you use.

---

## Citation

See `CITATION.cff`. The archived release has a DOI on Zenodo.

## License

MIT. See `LICENSE`.
