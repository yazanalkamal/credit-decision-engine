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

- _TBD: split strategy (stratified held-out), seed policy, CV design._

## Leakage review

- _TBD: features audited and any excluded, with reasoning._

## Results

- _TBD: per-candidate table, PR curves, calibration plots, chosen operating point._
