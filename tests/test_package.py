"""Scaffolding-level checks: the package installs and imports cleanly.

These exist so CI fails loudly on a broken packaging/install setup rather than
on a confusing ImportError inside a later, more interesting test.
"""

import importlib

import pytest

import credit_engine

SUBPACKAGES = [
    "credit_engine.features",
    "credit_engine.training",
    "credit_engine.evaluation",
]


def test_version_is_exposed():
    assert credit_engine.__version__ == "0.1.0"


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_imports(name: str):
    assert importlib.import_module(name) is not None
