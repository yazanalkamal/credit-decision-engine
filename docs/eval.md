# Evaluation methodology

> 🚧 Phase 1 deliverable, in progress. This document is the contract every model
> candidate is judged against — written before the models exist, on purpose, so
> the metrics cannot be chosen after seeing the results.

## Why not accuracy

The positive class (default) is ~8% of applicants. A model that approves
everyone scores ~92% accuracy and is worthless. Accuracy appears nowhere in this
project.

## Metrics

| Metric | Why it is here |
|---|---|
| **PR-AUC** | Precision/recall trade-off across all thresholds, unaffected by the large true-negative mass that flatters ROC-AUC on imbalanced data. Primary selection metric. |
| **Recall @ 1% FPR** | The realistic strict operating point: how many defaulters are caught while wrongly declining 1% of good applicants. |
| **Recall @ 5% FPR** | The looser operating point, for the business trade-off discussion. |
| **Calibration** | A score used for a threshold policy must mean what it says. Reliability curve + Brier score. |
| ROC-AUC | Reported for comparability with published Home Credit results only. Not a selection metric. |

## Protocol

### Split design

```
application_train (307,511 rows, one row per SK_ID_CURR)
   │
   ├── TEST  20%  ── opened once, at the end, to report. Never for selection.
   │
   └── DEV   80%
         └── StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
              → every comparison, every hyperparameter, the whole bake-off
```

Two levels, two jobs. Model selection happens entirely inside cross-validation on
DEV. TEST exists to produce the final reported numbers once the winner is already
chosen. Evaluating on TEST repeatedly and picking the best would fit it through
human decisions, and the "held out" number would no longer mean anything.

### Why stratified

The positive class is 8.07%. Both the held-out split and every CV fold are
stratified on `TARGET`.

Not because unstratified folds would be degenerate — at this size they are not.
Measured on this dataset, five unstratified folds range from 7.97% to 8.26%
positive (0.29 pp spread), against 0.0001 pp under stratification. Every fold
still contains ~5,000 defaults either way.

The reason is that **PR-AUC's floor is the base rate**. A fold at 8.26% positives
has a structurally higher PR-AUC than one at 7.97%, so unstratified fold-to-fold
variance partly measures the split rather than the model. Candidates in this
bake-off are expected to differ by small margins; measurement noise that can be
removed at zero cost should be.

### Seed policy

`random_state=42` for the held-out split, the CV folds, and every model that
accepts a seed. Fixed and recorded so results reproduce on a clean machine.

### What is fitted where

Anything learned from data is learned on training rows only, and applied to
validation and test rows unchanged — imputer statistics, scaler parameters,
one-hot category lists.

Anything learned from the **target** is fitted **inside** each CV fold, never
across the whole DEV set: target encoding (if used), probability calibration, and
any feature selection that consults `TARGET`. Fitting these outside the loop leaks
into every fold and inflates every number.

Transformations that are row-local — child-table aggregations, sentinel handling,
ratio features — are stateless and give identical results on either side of the
split. They are nonetheless applied per-split, so the ordering stays safe without
requiring a correct judgement about which bucket each transformation falls into.

### Reporting

Cross-validated results are reported as **mean ± standard deviation** across the
five folds, never as a single number. A gap between candidates smaller than the
fold standard deviation is not treated as a difference.

### Known limitation: no out-of-time validation

`application_train` contains **no calendar date**. Every temporal column is
relative to each application's own date (`DAYS_BIRTH`, `DAYS_EMPLOYED`,
`DAYS_REGISTRATION`); the closest fields to a timestamp are
`WEEKDAY_APPR_PROCESS_START` and `HOUR_APPR_PROCESS_START`, which carry no week or
year. Applications therefore cannot be ordered in time.

A production credit model would be validated **out-of-time** — trained on an
earlier period, evaluated on a later one — because that is the real deployment
condition: a model trained on the past, scoring the future. That is impossible on
this dataset, and random cross-validation is not equivalent to it. Random CV
answers "can this model score an unseen applicant from the same period?", which is
a narrower question, and it says nothing about population drift over time.

This limitation is the reason drift monitoring (Phase 5) is part of the project
rather than an optional extra: what cannot be validated offline has to be watched
in production.

Note that no split strategy — random or temporal — detects leakage introduced
during feature construction, because that contamination is carried per row and
appears on both sides of any split. Point-in-time correctness is enforced when
features are built and audited below, not measured here.

## Leakage review

Leakage is information available at training time that will not be available when
`POST /decision` scores a real application. It is audited here rather than measured
in the protocol above, because **no split strategy detects it**: contamination
introduced during feature construction is carried per row and therefore appears on
both sides of any split, random or temporal.

Three kinds are checked separately:

- **Target leakage** — the feature is a consequence of the outcome.
- **Temporal leakage** — the feature is legitimate, but its value comes from after
  the decision point.
- **Train/test contamination** — statistics crossing the split boundary. Prevented
  structurally by the fit-on-train-only rule in the Protocol above.

### Method

Every child table is keyed to the application by a relative date column. Point-in-time
correctness is verified by asserting those columns are bounded at or before the
decision point, since the dataset encodes all dates as days/months *before* the
application (negative values).

### Findings

| Table | Date column | Range | Verdict |
|---|---|---|---|
| `bureau` | `DAYS_CREDIT` | −2,922 … **0** | ✅ Clean — no credit applied for after the decision |
| `bureau` | `DAYS_ENDDATE_FACT` | −42,023 … **0** | ✅ Clean — closed credits ended at or before the decision |
| `bureau` | `DAYS_CREDIT_UPDATE` | −41,947 … **+372** | ⚠️ See below |
| `bureau` | `DAYS_CREDIT_ENDDATE` | −42,060 … **+31,199** | ✅ Not leakage — see below |
| `bureau_balance` | `MONTHS_BALANCE` | −96 … **0** | ✅ Clean |
| `previous_application` | `DAYS_DECISION` | −2,922 … **−1** | ✅ Clean |
| `POS_CASH_balance` | `MONTHS_BALANCE` | −96 … **−1** | ✅ Clean |
| `credit_card_balance` | `MONTHS_BALANCE` | −96 … **−1** | ✅ Clean |
| `installments_payments` | `DAYS_INSTALMENT` | −2,922 … **−1** | ✅ Clean |
| `installments_payments` | `DAYS_ENTRY_PAYMENT` | −4,921 … **−1** | ✅ Clean |

The dataset's own column descriptions corroborate this: bureau fields are documented
as measured *"at the time of application"* and *"before current application"*. Home
Credit snapshotted the relational tables as of each decision date rather than
exporting current state — which is the usual source of temporal leakage in credit
data and is the reason this table exists rather than an assumption.

**`DAYS_CREDIT_ENDDATE` up to +31,199 is not leakage.** It is the *scheduled* end
date of a bureau credit, which is fixed in the contract and therefore known at
application time; 35.1% of rows are positive simply because the credit was still
running. It is, however, an **outlier problem**: 42,690 rows have a scheduled end
more than 30 years out. Handled as an outlier in feature engineering, not excluded
here.

**`DAYS_CREDIT_UPDATE` up to +372 is a genuine point-in-time violation** — the
bureau record was last refreshed *after* the application, so those rows may carry
post-decision information. It affects **17 rows and 17 applicants (0.001%)**. Not
material, but not ignored: rows with `DAYS_CREDIT_UPDATE > 0` are excluded from
bureau aggregations, and the filter stays in the feature code as documentation of
the check.

### Unresolved suspect: social-circle default counts

`DEF_30_CNT_SOCIAL_CIRCLE` and `DEF_60_CNT_SOCIAL_CIRCLE` count how many people in
the applicant's social surroundings defaulted at 30 and 60 DPD. **The observation
window is not specified anywhere in the data dictionary**, so it cannot be
established from the data whether those defaults were observed before or after this
application. If after, they are temporally leaked.

The magnitude is small and the features are weak, so they are retained — but flagged
here as an unverifiable assumption rather than a cleared one. Stating what could not
be resolved is part of the audit.

By contrast, `AMT_REQ_CREDIT_BUREAU_HOUR/DAY/WEEK/MON/QRT/YEAR` sound like the same
risk and are not: the dictionary pins each to an explicit window *before* the
application. Cleared.

### Target leakage

No column in `application_train` describes the outcome of the loan being applied
for. The table contains application-time attributes only, and the competition
organisers removed servicing and collections fields before release. No exclusions
required on this basis.

### `EXT_SOURCE_1/2/3`

Not a leakage issue. These are anonymised third-party scores available at
application time, so they satisfy the point-in-time test. They are an
**explainability** problem, handled by ADR #14 and measured by the `EXT_SOURCE`
ablation below.

### Worked examples — the method applied to three columns

The child tables are cleared by date bounds because point-in-time correctness is a
property of the *row*: one bounded date column governs every field on that row.
`application_train` has no such column — its 122 fields are application-time
attributes by assertion — so columns there are classified individually. Three
worked through in full, one per outcome:

**`DAYS_EMPLOYED`** — *"How many days before the application the person started
current employment."*

| Flavour | Verdict |
|---|---|
| Target leakage | No. Employment tenure is not a consequence of loan repayment. |
| Temporal leakage | No. The description pins it *before the application*, and the value is fixed at that moment. |
| Contamination | No. Row-local; no statistic crosses the split. |

**Clean — but not harmless.** 18.01% of rows hold the sentinel `365243`
(≈1000 years), which is a NULL placeholder for "not employed"; 55,352 of those
55,374 rows are pensioners, and they default at 5.40% against 8.66% for the
employed. Converted to `NaN` plus a `DAYS_EMPLOYED_SENTINEL` indicator in feature
engineering. A data-quality problem, not a leakage one — recorded here because the
distinction is the point.

**`AMT_REQ_CREDIT_BUREAU_QRT`** — *"Number of enquiries to Credit Bureau about the
client 3 month before application (excluding one month before application)."*

| Flavour | Verdict |
|---|---|
| Target leakage | No. An enquiry count is a signal of credit-seeking behaviour, not an outcome. |
| Temporal leakage | No. The window is explicitly bounded *before* the application, and stated in the dictionary. |
| Contamination | No. Row-local. |

**Cleared.** This family is the useful counter-example: it *sounds* like the
riskiest thing in the table — bureau activity, of all things — and it is clean,
because the observation window is documented. The sibling columns
(`_HOUR/_DAY/_WEEK/_MON/_YEAR`) carry the same explicit bounding and are cleared on
the same basis.

**`DEF_30_CNT_SOCIAL_CIRCLE`** — *"How many observation of client's social
surroundings defaulted on 30 DPD (days past due)."*

| Flavour | Verdict |
|---|---|
| Target leakage | No. It describes other people's outcomes, not this applicant's. |
| Temporal leakage | **Unresolved.** No observation window is stated anywhere in the data dictionary. If those defaults were observed after this application, the feature is leaked. |
| Contamination | No. Row-local. |

**Retained with a flagged assumption**, as recorded above. The contrast with
`AMT_REQ_CREDIT_BUREAU_QRT` is exactly the lesson: two features of comparable
plausibility, and the one that clears is the one whose window is documented. Where
the dictionary is silent, the honest verdict is "unverifiable", not "fine".

### Population limitation (not leakage, recorded here for adjacency)

The dataset contains **accepted applicants only** — rejected applicants have no
observed outcome, so the training population is censored relative to the real
applicant population. This is the classic **reject inference** problem. It cannot
be fixed with this dataset and is recorded as a stated limitation in the model card.

## Planned experiments

### The `EXT_SOURCE` ablation (explainability cost)

Exploration found `EXT_SOURCE_2` to be the strongest single predictor in the
dataset — default rate falls monotonically from 18.4% in the bottom decile to
3.0% in the top, a 6.2× spread — while being an anonymous third-party score that
cannot be explained to an applicant, a regulator, or us.

Architecture decision #14 keeps these features and mitigates at the explanation
layer. That decision is explicitly **conditional on a measured cost**, so the
ablation is part of the eval contract rather than an optional extra:

- Train the chosen model family twice — with and without `EXT_SOURCE_1/2/3` —
  everything else held constant, same split, same seed.
- Report the gap on the primary metrics: PR-AUC, recall @ 1% FPR, recall @ 5% FPR.
- **If the gap is small, revisit decision #14 and prefer the fully explainable
  model.** A cleaner explainability story is worth buying when it is cheap.

Stating the rule before seeing the number is the point — same reason this whole
document is written before the models exist.

## Results

- _TBD: per-candidate table, PR curves, calibration plots, chosen operating point._
