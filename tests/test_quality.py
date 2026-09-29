"""Tests for :mod:`data_sys.quality`."""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_sys.quality import validate_bars


def test_good_frame_passes(standardized_frame: pd.DataFrame) -> None:
    report = validate_bars(standardized_frame)

    assert report.ok, report.reasons()
    assert report.failures == []
    assert report.row_count == len(standardized_frame)
    assert report.stats["symbol"] == "AAPL"


def test_empty_frame_fails() -> None:
    report = validate_bars(pd.DataFrame())

    assert report.ok is False
    assert report.reasons()


def test_none_frame_fails() -> None:
    report = validate_bars(None)

    assert report.ok is False


def test_duplicate_symbol_date_fails(standardized_frame: pd.DataFrame) -> None:
    duplicated = pd.concat(
        [standardized_frame, standardized_frame.iloc[[0]]], ignore_index=True
    )

    report = validate_bars(duplicated)

    assert report.ok is False
    assert any("duplicate" in reason for reason in report.reasons()), report.reasons()


def test_non_positive_close_fails(standardized_frame: pd.DataFrame) -> None:
    bad = standardized_frame.copy()
    bad.loc[5, "close"] = -1.0

    report = validate_bars(bad)

    assert report.ok is False
    assert any(failure.column == "close" for failure in report.failures), report.failures
    assert any(failure.index == 5 for failure in report.failures), report.failures


def test_high_below_other_prices_fails(standardized_frame: pd.DataFrame) -> None:
    bad = standardized_frame.copy()
    bad.loc[10, "high"] = 0.5

    report = validate_bars(bad)

    assert report.ok is False
    assert any("high" in reason for reason in report.reasons()), report.reasons()


def test_low_above_other_prices_fails(standardized_frame: pd.DataFrame) -> None:
    bad = standardized_frame.copy()
    bad.loc[11, "low"] = 99_999.0

    report = validate_bars(bad)

    assert report.ok is False
    assert any("low" in reason for reason in report.reasons()), report.reasons()


def test_null_price_fails(standardized_frame: pd.DataFrame) -> None:
    bad = standardized_frame.copy()
    bad.loc[3, "high"] = np.nan

    report = validate_bars(bad)

    assert report.ok is False
    assert any(failure.column == "high" for failure in report.failures), report.failures


def test_first_daily_return_must_be_null(standardized_frame: pd.DataFrame) -> None:
    bad = standardized_frame.copy()
    bad.loc[0, "daily_return"] = 0.5

    report = validate_bars(bad)

    assert report.ok is False
    assert any("daily_return" in reason for reason in report.reasons()), report.reasons()


def test_float_volume_fails_dtype_check(standardized_frame: pd.DataFrame) -> None:
    bad = standardized_frame.copy()
    bad["volume"] = bad["volume"].astype("float64")

    report = validate_bars(bad)

    assert report.ok is False
    assert any(failure.column == "volume" for failure in report.failures), report.failures


def test_lowercase_symbol_fails(standardized_frame: pd.DataFrame) -> None:
    bad = standardized_frame.copy()
    bad["symbol"] = "aapl"

    report = validate_bars(bad)

    assert report.ok is False
    assert any(failure.column == "symbol" for failure in report.failures), report.failures


def test_unsorted_dates_fail(standardized_frame: pd.DataFrame) -> None:
    bad = standardized_frame.sort_values("date", ascending=False).reset_index(drop=True)

    report = validate_bars(bad)

    assert report.ok is False
    assert any("increasing" in reason for reason in report.reasons()), report.reasons()
