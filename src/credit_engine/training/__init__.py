"""Model training.

Candidates in bake-off order: logistic regression baseline, then LightGBM/
XGBoost, then a PyTorch tabular model. Deterministic seeds throughout; the
winner on the eval harness ships, even if it is the simple one.
"""
