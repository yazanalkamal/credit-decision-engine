# Notebooks

Two kinds live here, and neither ships:

- **`exploration.ipynb`** — exploration. Looking at the data to make decisions.
  Anything that survives exploration moves into a tested module under
  `src/credit_engine/`.
- **`learning_*.ipynb`** — learning notebooks. Runnable explanations of a
  mechanic (joins, splits, calibration) on real project data, with exercises.
  Companions to the corresponding note in the DataPlan.

Outputs are stripped on commit (see `.pre-commit-config.yaml`), so run them
yourself rather than reading committed output.
