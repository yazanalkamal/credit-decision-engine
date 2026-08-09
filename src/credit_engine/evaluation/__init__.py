"""Evaluation harness — the contract every model candidate is judged against.

Imbalance-appropriate metrics only: PR-AUC, recall at fixed FPR (1% and 5%),
and calibration. Headline accuracy is meaningless at ~8% positive class and is
deliberately absent (ARCHITECTURE.md §2.2).
"""
