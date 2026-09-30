"""Tests for :mod:`data_sys.standardize`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data_sys.errors import StandardizationError
from data_sys.schema import STANDARD_COLUMNS
from data_sys.standardize import derive_features, standardize_bars
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

    # `daily_return` is the *adjusted-close* return: unadjusted OHLC is the real
    # traded price, `adjusted_close` is the series that removes split/dividend
    # jumps, so only its returns are comparable across a corporate action.
    assert pd.isna(out["daily_return"].iloc[0])
    assert out["daily_return"].iloc[1:].notna().all()

    expected_return = out["adjusted_close"].iloc[1] / out["adjusted_close"].iloc[0] - 1.0
    assert out["daily_return"].iloc[1] == pytest.approx(expected_return)

    assert np.allclose(
        out["dollar_volume"].to_numpy(),
        (out["close"] * out["volume"]).to_numpy(),
    )


def test_daily_return_is_not_the_unadjusted_return(raw_frame: pd.DataFrame) -> None:
    out = standardize_bars(raw_frame, "AAPL")

    unadjusted = out["close"].pct_change()

    assert not np.allclose(out["daily_return"].iloc[1:], unadjusted.iloc[1:])


def test_first_row_return_is_null_and_the_rest_are_not() -> None:
    """The schema rule that bounds the whole column, checked directly."""
    out = standardize_bars(make_raw_frame(periods=60), "AAPL")

    assert len(out) == 60
    assert out["daily_return"].isna().sum() == 1
    assert pd.isna(out["daily_return"].iloc[0])
    assert out["daily_return"].iloc[1:].notna().all()


def test_single_row_history_only_has_a_null_return() -> None:
    out = standardize_bars(make_raw_frame(periods=1), "AAPL")

    assert len(out) == 1
    assert pd.isna(out["daily_return"].iloc[0])


def test_derive_features_is_idempotent_and_does_not_mutate(
    standardized_frame: pd.DataFrame,
) -> None:
    before = standardized_frame.copy(deep=True)

    result = derive_features(standardized_frame)

    pd.testing.assert_frame_equal(standardized_frame, before)
    assert result is not standardized_frame
    pd.testing.assert_series_equal(result["daily_return"], standardized_frame["daily_return"])
    pd.testing.assert_series_equal(result["dollar_volume"], standardized_frame["dollar_volume"])


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
