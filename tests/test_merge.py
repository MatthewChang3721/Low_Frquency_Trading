"""Tests for :mod:`data_sys.merge` (merging downloads into the published history)."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from data_sys.errors import DataPipelineError
from data_sys.merge import date_set, empty_bars_frame, merge_bars
from data_sys.quality import validate_bars
from data_sys.schema import STANDARD_COLUMNS
from data_sys.standardize import derive_features, standardize_bars
from tests.helpers import SessionFakeProvider


def _bars(
    symbol: str = "AAPL",
    start: str = "2024-01-02",
    end: str = "2024-03-29",
    *,
    seed: int = 1,
    drop: tuple[str, ...] = (),
) -> pd.DataFrame:
    """A standardized, session-aligned history built from a different seed."""
    raw = SessionFakeProvider(start=start, end=end, seed=seed).frame
    bars = standardize_bars(raw, symbol)
    if drop:
        removed = {date.fromisoformat(day) for day in drop}
        bars = bars[~bars["date"].dt.date.isin(removed)].reset_index(drop=True)
        bars = derive_features(bars)
    return bars


def _window(bars: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """The rows of ``bars`` inside ``[start, end]``."""
    dates = bars["date"].dt.date
    mask = (dates >= date.fromisoformat(start)) & (dates <= date.fromisoformat(end))
    return bars.loc[mask].reset_index(drop=True)


def _close_on(bars: pd.DataFrame, day: str) -> float:
    value = bars.loc[bars["date"].dt.date == date.fromisoformat(day), "close"]
    assert len(value) == 1, f"{day} is not a unique row"
    return float(value.iloc[0])


# ---------------------------------------------------------------------------
# first-time publish
# ---------------------------------------------------------------------------
def test_first_publish_keeps_every_downloaded_row() -> None:
    bars = _bars()

    merged, report = merge_bars(None, bars)

    assert list(merged.columns) == STANDARD_COLUMNS
    assert len(merged) == len(bars)
    assert merged["date"].tolist() == bars["date"].tolist()
    assert report.existing_row_count == 0
    assert report.downloaded_row_count == len(bars)
    assert report.merged_row_count == len(bars)
    assert report.new_dates_count == len(bars)
    assert report.replaced_dates_count == 0
    assert report.retained_row_count == 0
    assert report.min_date == date(2024, 1, 2)
    assert report.max_date == date(2024, 3, 28)


def test_merging_two_empty_frames_is_an_error() -> None:
    with pytest.raises(DataPipelineError):
        merge_bars(None, None)

    with pytest.raises(DataPipelineError):
        merge_bars(empty_bars_frame(), pd.DataFrame())


def test_merge_output_passes_the_quality_schema() -> None:
    merged, _ = merge_bars(None, _bars())

    report = validate_bars(merged)

    assert report.ok, report.reasons()


# ---------------------------------------------------------------------------
# overlaps
# ---------------------------------------------------------------------------
def test_downloaded_rows_win_on_overlapping_dates() -> None:
    published = _bars(seed=1)
    refreshed = _window(_bars(seed=99), "2024-03-22", "2024-03-29")

    merged, report = merge_bars(published, refreshed)

    assert report.replaced_dates_count == len(refreshed)
    assert report.new_dates_count == 0
    assert report.retained_row_count == len(published) - len(refreshed)
    assert len(merged) == len(published)
    assert not merged.duplicated(subset=["symbol", "date"]).any()
    assert merged["date"].is_monotonic_increasing

    for day in ("2024-03-22", "2024-03-28"):
        assert _close_on(merged, day) == _close_on(refreshed, day)
        assert _close_on(merged, day) != _close_on(published, day)


def test_published_rows_outside_the_download_are_untouched() -> None:
    published = _bars(seed=1)
    refreshed = _window(_bars(seed=99), "2024-03-25", "2024-03-29")

    merged, _ = merge_bars(published, refreshed)

    early = _window(merged, "2024-01-02", "2024-03-21")
    original = _window(published, "2024-01-02", "2024-03-21")
    assert np.allclose(early["close"].to_numpy(), original["close"].to_numpy())
    assert len(merged) == len(published)
    assert merged["date"].max().date() == date(2024, 3, 28)


def test_new_tail_rows_are_appended_in_order() -> None:
    published = _window(_bars(seed=1), "2024-01-02", "2024-03-15")
    tail = _window(_bars(seed=1), "2024-03-18", "2024-03-29")

    merged, report = merge_bars(published, tail)

    assert report.new_dates_count == len(tail)
    assert report.replaced_dates_count == 0
    assert len(merged) == len(published) + len(tail)
    assert merged["date"].is_monotonic_increasing
    assert merged["date"].max().date() == date(2024, 3, 28)


# ---------------------------------------------------------------------------
# derived columns are re-derived, never inherited
# ---------------------------------------------------------------------------
def _return_on(bars: pd.DataFrame, day: str) -> float:
    value = bars.loc[bars["date"].dt.date == date.fromisoformat(day), "daily_return"]
    assert len(value) == 1
    return float(value.iloc[0])


def test_daily_return_is_recomputed_across_the_overlap_boundary() -> None:
    """The boundary row mixes old and new data, so its return must be re-derived."""
    published = _bars(seed=1)
    refreshed = _window(_bars(seed=99), "2024-03-25", "2024-03-29")
    boundary = "2024-03-25"
    previous = "2024-03-22"

    assert _return_on(published, boundary) != _return_on(refreshed, boundary)

    merged, _ = merge_bars(published, refreshed)

    expected = (
        _close_on_adjusted(merged, boundary) / _close_on_adjusted(merged, previous) - 1.0
    )
    assert _return_on(merged, boundary) == pytest.approx(expected)
    assert _return_on(merged, boundary) != pytest.approx(_return_on(refreshed, boundary))


def test_daily_return_of_the_row_after_the_boundary_is_recomputed() -> None:
    published = _bars(seed=1)
    refreshed = _window(_bars(seed=99), "2024-03-25", "2024-03-29")

    merged, _ = merge_bars(published, refreshed)

    for day, previous in (("2024-03-26", "2024-03-25"), ("2024-03-28", "2024-03-27")):
        expected = (
            _close_on_adjusted(merged, day) / _close_on_adjusted(merged, previous) - 1.0
        )
        assert _return_on(merged, day) == pytest.approx(expected)


def _close_on_adjusted(bars: pd.DataFrame, day: str) -> float:
    value = bars.loc[bars["date"].dt.date == date.fromisoformat(day), "adjusted_close"]
    assert len(value) == 1
    return float(value.iloc[0])


def test_daily_return_is_re_derived_from_adjusted_close_not_the_stale_value() -> None:
    published = _bars(seed=1)
    refreshed = _window(_bars(seed=99), "2024-03-25", "2024-03-29")

    merged, _ = merge_bars(published, refreshed)

    adjusted = merged["adjusted_close"]
    expected = adjusted / adjusted.shift(1) - 1.0
    assert np.allclose(
        merged["daily_return"].iloc[1:].to_numpy(), expected.iloc[1:].to_numpy()
    )
    # the untouched head of the history keeps exactly the returns it had before
    untouched = _window(merged, "2024-01-02", "2024-03-22")
    original = _window(published, "2024-01-02", "2024-03-22")
    assert len(untouched) == len(original)
    assert np.allclose(
        untouched["daily_return"].iloc[1:].to_numpy(),
        original["daily_return"].iloc[1:].to_numpy(),
    )


def test_dollar_volume_is_recomputed_for_every_row() -> None:
    published = _bars(seed=1)
    published.loc[:, "dollar_volume"] = -1.0  # poison the published value
    refreshed = _window(_bars(seed=99), "2024-03-25", "2024-03-29")

    merged, _ = merge_bars(published, refreshed)

    assert np.allclose(
        merged["dollar_volume"].to_numpy(),
        (merged["close"] * merged["volume"]).to_numpy(),
    )
    assert (merged["dollar_volume"] > 0).all()


def test_first_row_return_is_null_after_a_forward_backfill() -> None:
    published = _window(_bars(seed=1), "2024-03-01", "2024-03-29")
    head = _window(_bars(seed=99), "2024-01-02", "2024-02-29")

    merged, report = merge_bars(published, head)

    assert report.new_dates_count == len(head)
    assert merged["date"].min().date() == date(2024, 1, 2)
    assert pd.isna(merged["daily_return"].iloc[0])
    assert merged["daily_return"].iloc[1:].notna().all()
    assert validate_bars(merged).ok


def test_filling_an_internal_gap_recomputes_both_sides() -> None:
    published = _bars(seed=1, drop=("2024-02-15",))
    filler = _window(_bars(seed=99), "2024-02-15", "2024-02-15")

    merged, report = merge_bars(published, filler)

    assert report.new_dates_count == 1
    assert date(2024, 2, 15) in date_set(merged)
    assert _close_on(merged, "2024-02-15") == _close_on(filler, "2024-02-15")

    for day, previous in (("2024-02-15", "2024-02-14"), ("2024-02-16", "2024-02-15")):
        expected = (
            _close_on_adjusted(merged, day) / _close_on_adjusted(merged, previous) - 1.0
        )
        assert _return_on(merged, day) == pytest.approx(expected)


def test_duplicate_rows_inside_the_download_are_collapsed() -> None:
    published = _bars(seed=1)
    refreshed = _window(_bars(seed=99), "2024-03-27", "2024-03-29")
    doubled = pd.concat([refreshed, refreshed], ignore_index=True)

    merged, report = merge_bars(published, doubled)

    assert not merged.duplicated(subset=["symbol", "date"]).any()
    assert len(merged) == len(published)
    assert report.replaced_dates_count == len(refreshed)


def test_an_unsorted_download_is_sorted_into_place() -> None:
    published = _bars(seed=1)
    refreshed = _window(_bars(seed=99), "2024-03-25", "2024-03-29")
    shuffled = refreshed.sample(frac=1.0, random_state=7)

    merged, _ = merge_bars(published, shuffled)

    assert merged["date"].is_monotonic_increasing
    assert merged["date"].is_unique


# ---------------------------------------------------------------------------
# hive-read frames and malformed input
# ---------------------------------------------------------------------------
def test_a_hive_read_published_frame_loses_its_year_column() -> None:
    """``read_partitioned`` re-adds ``year``; the merge must drop it again."""
    published = _bars(seed=1).copy()
    published["year"] = published["date"].dt.year

    merged, _ = merge_bars(published, _window(_bars(seed=99), "2024-03-29", "2024-03-29"))

    assert list(merged.columns) == STANDARD_COLUMNS
    assert "year" not in merged.columns


def test_a_published_frame_missing_a_column_raises_a_clear_error() -> None:
    published = _bars(seed=1).drop(columns=["adjusted_close"])

    with pytest.raises(DataPipelineError, match="adjusted_close"):
        merge_bars(published, _window(_bars(seed=1), "2024-03-29", "2024-03-29"))

    with pytest.raises(DataPipelineError, match="published"):
        merge_bars(published, None)


def test_a_downloaded_frame_missing_a_column_names_the_downloaded_side() -> None:
    broken = _bars(seed=1).drop(columns=["volume"])

    with pytest.raises(DataPipelineError, match="downloaded"):
        merge_bars(_bars(seed=1), broken)


def test_unparseable_dates_in_a_frame_raise() -> None:
    broken = _bars(seed=1)
    broken.loc[0, "date"] = pd.NaT

    with pytest.raises(DataPipelineError, match="unparseable"):
        merge_bars(broken, _window(_bars(seed=1), "2024-03-29", "2024-03-29"))


def test_merge_does_not_mutate_its_inputs() -> None:
    published = _bars(seed=1)
    refreshed = _window(_bars(seed=99), "2024-03-25", "2024-03-29")
    before_published = published.copy(deep=True)
    before_refreshed = refreshed.copy(deep=True)

    merge_bars(published, refreshed)

    pd.testing.assert_frame_equal(published, before_published)
    pd.testing.assert_frame_equal(refreshed, before_refreshed)


def test_date_set_and_empty_frame_helpers() -> None:
    bars = _bars(seed=1)

    assert date_set(None) == set()
    assert date_set(pd.DataFrame()) == set()
    assert date_set(bars) == set(bars["date"].dt.date)
    assert list(empty_bars_frame().columns) == STANDARD_COLUMNS
    assert empty_bars_frame().empty


def test_downloaded_only_merge_recomputes_the_first_return_as_null() -> None:
    """A single-row history cannot have a return for its first row."""
    single = _window(_bars(seed=1), "2024-01-02", "2024-01-02")

    merged, report = merge_bars(None, single)

    assert len(merged) == 1
    assert report.new_dates_count == 1
    assert pd.isna(merged["daily_return"].iloc[0])


