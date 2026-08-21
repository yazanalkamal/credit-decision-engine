"""Tests for the split policy.

These check the properties the eval contract depends on: the split is
reproducible, it preserves the base rate on both sides, the same applicant never
lands on both sides, and the test set cannot be opened by accident.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from credit_engine.evaluation.split import (
    SEED,
    HoldoutSplit,
    TestSetAccessError,
    make_holdout_split,
)

BASE_RATE = 0.0807  # matches Home Credit; the point is that it is small
N_ROWS = 20_000


@pytest.fixture
def frame() -> pd.DataFrame:
    """A synthetic frame with Home Credit's grain and class imbalance."""
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "SK_ID_CURR": np.arange(100_000, 100_000 + N_ROWS),
            "TARGET": (rng.random(N_ROWS) < BASE_RATE).astype(int),
            "FEATURE": rng.normal(size=N_ROWS),
        }
    )


@pytest.fixture
def split(frame: pd.DataFrame) -> HoldoutSplit:
    return make_holdout_split(frame)


# --------------------------------------------------------------------------
# partition properties
# --------------------------------------------------------------------------


def test_split_covers_every_row_exactly_once(frame: pd.DataFrame, split: HoldoutSplit):
    test_idx = split.unseal_test(reason="test: partition check")
    combined = np.concatenate([split.dev, test_idx])

    assert combined.size == len(frame)
    assert np.array_equal(np.sort(combined), np.arange(len(frame)))


def test_no_applicant_appears_on_both_sides(frame: pd.DataFrame, split: HoldoutSplit):
    test_idx = split.unseal_test(reason="test: contamination check")
    dev_ids = set(frame.SK_ID_CURR.to_numpy()[split.dev])
    test_ids = set(frame.SK_ID_CURR.to_numpy()[test_idx])

    assert dev_ids.isdisjoint(test_ids)


def test_test_fraction_is_honoured(frame: pd.DataFrame, split: HoldoutSplit):
    assert split.n_test == pytest.approx(len(frame) * 0.2, rel=0.01)


# --------------------------------------------------------------------------
# stratification — the reason the split function exists
# --------------------------------------------------------------------------


def test_base_rate_is_preserved_on_both_sides(frame: pd.DataFrame, split: HoldoutSplit):
    test_idx = split.unseal_test(reason="test: stratification check")
    target = frame.TARGET.to_numpy()
    overall = target.mean()

    assert target[split.dev].mean() == pytest.approx(overall, abs=0.002)
    assert target[test_idx].mean() == pytest.approx(overall, abs=0.002)


def test_cv_folds_preserve_base_rate(frame: pd.DataFrame, split: HoldoutSplit):
    target = frame.TARGET.to_numpy()
    dev_rate = target[split.dev].mean()

    for _, val_idx in split.cv_folds(frame.TARGET):
        assert target[val_idx].mean() == pytest.approx(dev_rate, abs=0.005)


# --------------------------------------------------------------------------
# cross-validation folds
# --------------------------------------------------------------------------


def test_cv_folds_partition_the_dev_split(frame: pd.DataFrame, split: HoldoutSplit):
    folds = split.cv_folds(frame.TARGET)

    assert len(folds) == 5
    val_indices = np.concatenate([val for _, val in folds])
    assert np.array_equal(np.sort(val_indices), split.dev)


def test_cv_train_and_val_are_disjoint(frame: pd.DataFrame, split: HoldoutSplit):
    for train_idx, val_idx in split.cv_folds(frame.TARGET):
        assert set(train_idx).isdisjoint(set(val_idx))


def test_cv_folds_never_touch_the_test_split(frame: pd.DataFrame, split: HoldoutSplit):
    test_idx = set(split.unseal_test(reason="test: leakage check"))

    for train_idx, val_idx in split.cv_folds(frame.TARGET):
        assert test_idx.isdisjoint(set(train_idx))
        assert test_idx.isdisjoint(set(val_idx))


def test_cv_folds_reject_a_mismatched_target(frame: pd.DataFrame, split: HoldoutSplit):
    with pytest.raises(ValueError, match="Pass the target for the full dataset"):
        split.cv_folds(frame.TARGET.iloc[:100])


# --------------------------------------------------------------------------
# reproducibility — docs/eval.md promises "same split, same seed"
# --------------------------------------------------------------------------


def test_same_seed_gives_the_same_split(frame: pd.DataFrame):
    a = make_holdout_split(frame, seed=SEED)
    b = make_holdout_split(frame, seed=SEED)

    assert np.array_equal(a.dev, b.dev)
    assert np.array_equal(
        a.unseal_test(reason="test: reproducibility"),
        b.unseal_test(reason="test: reproducibility"),
    )


def test_different_seed_gives_a_different_split(frame: pd.DataFrame):
    a = make_holdout_split(frame, seed=SEED)
    b = make_holdout_split(frame, seed=SEED + 1)

    assert not np.array_equal(a.dev, b.dev)


def test_cv_folds_are_reproducible(frame: pd.DataFrame):
    a = make_holdout_split(frame).cv_folds(frame.TARGET)
    b = make_holdout_split(frame).cv_folds(frame.TARGET)

    for (a_train, a_val), (b_train, b_val) in zip(a, b, strict=True):
        assert np.array_equal(a_train, b_train)
        assert np.array_equal(a_val, b_val)


# --------------------------------------------------------------------------
# the sealed test set
# --------------------------------------------------------------------------


def test_test_set_requires_a_reason(split: HoldoutSplit):
    with pytest.raises(TestSetAccessError, match="requires a stated reason"):
        split.unseal_test(reason="")

    with pytest.raises(TestSetAccessError):
        split.unseal_test(reason="   ")


def test_opening_the_test_set_warns(split: HoldoutSplit):
    with pytest.warns(UserWarning, match="Held-out test set opened"):
        split.unseal_test(reason="final report, baseline chosen on CV")


def test_test_indices_cannot_be_mutated_through_the_split(split: HoldoutSplit):
    first = split.unseal_test(reason="test: mutation check")
    first[0] = -999
    second = split.unseal_test(reason="test: mutation check")

    assert second[0] != -999


def test_repr_does_not_leak_test_indices(split: HoldoutSplit):
    assert "_test" not in repr(split)


# --------------------------------------------------------------------------
# grain and input validation
# --------------------------------------------------------------------------


def test_duplicate_ids_are_rejected(frame: pd.DataFrame):
    duplicated = pd.concat([frame, frame.iloc[:5]], ignore_index=True)

    with pytest.raises(ValueError, match="grain broken"):
        make_holdout_split(duplicated)


def test_missing_columns_are_rejected(frame: pd.DataFrame):
    with pytest.raises(ValueError, match="TARGET"):
        make_holdout_split(frame.drop(columns=["TARGET"]))

    with pytest.raises(ValueError, match="SK_ID_CURR"):
        make_holdout_split(frame.drop(columns=["SK_ID_CURR"]))


def test_null_target_is_rejected(frame: pd.DataFrame):
    frame.loc[0, "TARGET"] = np.nan

    with pytest.raises(ValueError, match="contains nulls"):
        make_holdout_split(frame)


@pytest.mark.parametrize("bad_fraction", [0.0, 1.0, -0.1, 1.5])
def test_invalid_test_fraction_is_rejected(frame: pd.DataFrame, bad_fraction: float):
    with pytest.raises(ValueError, match="test_fraction"):
        make_holdout_split(frame, test_fraction=bad_fraction)


# --------------------------------------------------------------------------
# the real dataset — skipped when the CSVs are absent (they are gitignored)
# --------------------------------------------------------------------------

REAL_DATA = Path(__file__).resolve().parents[2] / "data" / "raw" / "application_train.csv"


@pytest.mark.slow
@pytest.mark.skipif(not REAL_DATA.exists(), reason="Home Credit data not present")
def test_split_on_the_real_application_table():
    df = pd.read_csv(REAL_DATA, usecols=["SK_ID_CURR", "TARGET"])
    split = make_holdout_split(df)

    assert split.n_dev + split.n_test == 307_511

    target = df.TARGET.to_numpy()
    test_idx = split.unseal_test(reason="test: real-data sanity check")
    assert target[split.dev].mean() == pytest.approx(0.0807, abs=0.001)
    assert target[test_idx].mean() == pytest.approx(0.0807, abs=0.001)
