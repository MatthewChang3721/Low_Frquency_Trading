"""Tests for :mod:`data_sys.standardize`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data_sys.errors import StandardizationError
from data_sys.schema import STANDARD_COLUMNS
from data_sys.standardize import standardize_bars
from tests.helpers import make_raw_frame


def test_produces_the_contract(raw_frame: pd.DataFrame) -> None:
    out = standardize_bars(raw_frame, "aapl")

    assert list(out.columns) == STANDARD_COLUMNS
    assert out["symbol"].unique().tolist() == ["AAPL"]
    assert str(out["date"].dtype) == "datetime64[s]"
    assert str(out["volume"].dtype) == "int64"
    for column in ["open", "high", "low", "close", "adjusted_close", "dollar_volume"]:
        assert str(out[column].dtype) == "float64", column
    assert str(out["daily_return"].dtype) == "float64"


def test_row_count_and_ordering_are_preserved(raw_frame: pd.DataFrame) -> None:
    out = standardize_bars(raw_frame, "AAPL")

    assert len(out) == len(raw_frame)
    assert out["date"].is_monotonic_increasing
    assert out["date"].is_unique


def test_derived_features(raw_frame: pd.DataFrame) -> None:
    out = standardize_bars(raw_frame, "AAPL")

    assert pd.isna(out["daily_return"].iloc[0])
    assert out["daily_return"].iloc[1:].notna().all()

    expected_return = out["close"].iloc[1] / out["close"].iloc[0] - 1.0
    assert out["daily_return"].iloc[1] == pytest.approx(expected_return)

    assert np.allclose(
        out["dollar_volume"].to_numpy(),
        (out["close"] * out["volume"]).to_numpy(),
    )


def test_adjusted_close_is_kept_separately(raw_frame: pd.DataFrame) -> None:
    out = standardize_bars(raw_frame, "AAPL")

    assert np.allclose(out["adjusted_close"].to_numpy(), raw_frame["Adj Close"].to_numpy())
    assert not np.allclose(out["close"].to_numpy(), out["adjusted_close"].to_numpy())


def test_unsorted_input_is_sorted(raw_frame: pd.DataFrame) -> None:
    shuffled = raw_frame.sample(frac=1.0, random_state=42)

    out = standardize_bars(shuffled, "AAPL")

    assert out["date"].is_monotonic_increasing
    assert out["date"].tolist() == sorted(raw_frame.index.tolist())


def test_missing_raw_column_raises(raw_frame: pd.DataFrame) -> None:
    with pytest.raises(StandardizationError):
        standardize_bars(raw_frame.drop(columns=["Adj Close"]), "AAPL")


def test_empty_frame_raises() -> None:
    with pytest.raises(StandardizationError):
        standardize_bars(pd.DataFrame(), "AAPL")


def test_invalid_symbol_raises(raw_frame: pd.DataFrame) -> None:
    with pytest.raises(StandardizationError):
        standardize_bars(raw_frame, "aapl;drop table")


def test_volume_keeps_float_when_null_so_quality_can_report_it(raw_frame: pd.DataFrame) -> None:
    frame = raw_frame.copy()
    frame.iloc[2, frame.columns.get_loc("Volume")] = np.nan

    out = standardize_bars(frame, "AAPL")

    assert str(out["volume"].dtype) == "float64"
    assert out["volume"].isna().sum() == 1


def test_date_column_input_without_datetime_index() -> None:
    raw = make_raw_frame(periods=10).reset_index()
    assert "Date" in raw.columns

    out = standardize_bars(raw, "AAPL")

    assert len(out) == 10
    assert str(out["date"].dtype) == "datetime64[s]"
