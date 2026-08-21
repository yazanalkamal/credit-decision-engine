"""Train/dev/test splitting — the one definition every candidate is scored against.

The splitting *algorithm* is scikit-learn's. What lives here is the **policy**:
which column stratifies, which seed, which fraction, and the rule that the test
set is opened once. Scattering `train_test_split(...)` across training scripts
would make "same split, same seed" (docs/eval.md) a promise kept by memory; here
it is kept by construction.

Splits are returned as **positional indices** into the frame that was split, not
as sub-frames. Indices are small, cheap to compare, and can be applied to
different feature matrices built from the same rows — which is exactly what the
`EXT_SOURCE` ablation needs to hold the split constant while the columns change.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

SEED = 42
TEST_FRACTION = 0.2
N_FOLDS = 5
ID_COLUMN = "SK_ID_CURR"
TARGET_COLUMN = "TARGET"


class TestSetAccessError(RuntimeError):
    """Raised when the held-out test set is requested without a stated reason."""


@dataclass(frozen=True)
class HoldoutSplit:
    """A dev/test partition of one dataset, addressed by positional index.

    `dev` is free to use: cross-validation, tuning, model selection all happen
    inside it. The test indices are deliberately awkward to reach — see
    `unseal_test`.
    """

    dev: np.ndarray
    n_rows: int
    seed: int
    test_fraction: float
    _test: np.ndarray = field(repr=False)

    @property
    def n_dev(self) -> int:
        return int(self.dev.size)

    @property
    def n_test(self) -> int:
        return int(self._test.size)

    def unseal_test(self, *, reason: str) -> np.ndarray:
        """Return the held-out test indices. Requires an explicit stated reason.

        The test set may be scored **once**, to report final numbers for a model
        that has already been chosen on cross-validation. Using it to compare
        candidates turns it into a slow manual training loop and the reported
        number stops meaning anything (docs/eval.md, "Split design").

        The `reason` is not decoration: needing to write one is the friction, and
        it is what shows up in the transcript when this is audited later.
        """
        if not reason or not reason.strip():
            raise TestSetAccessError(
                "The test set requires a stated reason, e.g. "
                'unseal_test(reason="final report, LightGBM chosen on CV"). '
                "If the reason would be 'to see how it does', use cross-validation "
                "on the dev split instead."
            )
        warnings.warn(
            f"Held-out test set opened ({self.n_test:,} rows). Reason: {reason.strip()}",
            UserWarning,
            stacklevel=2,
        )
        return self._test.copy()

    def cv_folds(
        self,
        y: pd.Series | np.ndarray,
        n_folds: int = N_FOLDS,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Stratified CV folds *within* the dev split.

        `y` is the target for the full dataset (the same frame that was split).
        Returned indices are positional into that full dataset, so folds compose
        directly with `dev` and with any feature matrix built from the same rows.
        """
        y_arr = np.asarray(y)
        if y_arr.shape[0] != self.n_rows:
            raise ValueError(
                f"y has {y_arr.shape[0]:,} rows but the split was built over "
                f"{self.n_rows:,}. Pass the target for the full dataset."
            )
        if n_folds < 2:
            raise ValueError(f"n_folds must be at least 2, got {n_folds}")

        y_dev = y_arr[self.dev]
        folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.seed)
        return [
            (self.dev[train_idx], self.dev[val_idx])
            for train_idx, val_idx in folds.split(np.zeros(y_dev.shape[0]), y_dev)
        ]


def make_holdout_split(
    df: pd.DataFrame,
    *,
    test_fraction: float = TEST_FRACTION,
    seed: int = SEED,
    target_column: str = TARGET_COLUMN,
    id_column: str = ID_COLUMN,
) -> HoldoutSplit:
    """Partition `df` into a dev split and a sealed test split.

    Stratified on the target because the positive class is ~8%: unstratified
    folds drift in base rate, and PR-AUC's floor *is* the base rate, so that
    drift would land in the metric as noise (docs/eval.md, "Why stratified").

    The grain is asserted rather than assumed — a duplicated id would put the
    same applicant on both sides of the split, which is train/test contamination
    however carefully everything downstream is written.
    """
    for column in (target_column, id_column):
        if column not in df.columns:
            raise ValueError(f"{column!r} not found in the frame passed to make_holdout_split")

    if not df[id_column].is_unique:
        n_dupes = int(df[id_column].duplicated().sum())
        raise ValueError(
            f"grain broken: {n_dupes:,} duplicate {id_column} values. "
            "One row per application is required before splitting; a duplicated id "
            "would appear in both dev and test."
        )

    target = df[target_column]
    if target.isna().any():
        raise ValueError(f"{target_column!r} contains nulls; cannot stratify")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")

    positions = np.arange(len(df))
    dev, test = train_test_split(
        positions,
        test_size=test_fraction,
        stratify=target.to_numpy(),
        random_state=seed,
        shuffle=True,
    )

    return HoldoutSplit(
        dev=np.sort(dev),
        _test=np.sort(test),
        n_rows=len(df),
        seed=seed,
        test_fraction=test_fraction,
    )
