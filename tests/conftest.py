"""Shared pytest fixtures for the data-system test-suite."""

from __future__ import annotations

import pandas as pd
import pytest

from tests.helpers import FakeProvider, make_raw_frame, make_standardized_frame


@pytest.fixture
def raw_frame() -> pd.DataFrame:
    """A raw, Yahoo-shaped daily frame with 250 business days."""
    return make_raw_frame()


@pytest.fixture
def standardized_frame() -> pd.DataFrame:
    """The standardized version of :func:`raw_frame`."""
    return make_standardized_frame()


@pytest.fixture
def fake_provider() -> FakeProvider:
    """A ``DataProvider`` that serves :func:`make_raw_frame` from memory."""
    return FakeProvider()
