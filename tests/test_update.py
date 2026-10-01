"""End-to-end tests for :mod:`data_sys.update` -- incremental single-symbol updates.

Every provider is an in-memory fake, so no test touches the network.  The fake
slices its master frame by the requested window, which lets the assertions pin
down exactly *which* windows the incremental planner decided to download.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pytest

import data_sys.storage as storage
from data_sys.errors import DataPipelineError, DownloadError
from data_sys.merge import date_set
from data_sys.planner import STATUS_NO_SESSIONS, STATUS_UP_TO_DATE
from data_sys.quality import validate_bars
from data_sys.storage import read_symbol_partition, write_partitioned
from data_sys.update import (
    STATUS_FAILED,
    STATUS_SKIPPED,
    STATUS_SUCCESS,
    build_config,
    update_symbol,
)
from tests.helpers import NEVER_NULL_COLUMNS, SessionFakeProvider, session_dates


def _live_dir(data_root: Path, symbol: str = "AAPL") -> Path:
    return data_root / "standardized" / "market_bars" / f"symbol={symbol}"


def _run_records(data_root: Path) -> list[dict]:
    """Every run record, oldest first, ordered by the run's *own* timestamp.

    ``run_*.json`` file names end in a one-second timestamp plus a **random**
    run-id suffix, so sorting them by name says nothing about which run happened
    first and must never decide "the last run".  Use :func:`_run_record` with a
    known ``run_id`` whenever a test knows which run it is asking about.
    """
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (data_root / "metadata").glob("run_*.json")
    ]
    return sorted(records, key=lambda record: (record["run_at_utc"], record["run_id"]))


def _run_record(data_root: Path, run_id: str) -> dict:
    """The record of ``run_id``, looked up by id rather than guessed by order.

    ``run_at_utc`` has one-second resolution, so two runs of the same test can
    share it; only the run id identifies a record exactly.
    """
    files = sorted(path.name for path in (data_root / "metadata").glob("run_*.json"))
    path = data_root / "metadata" / f"run_{run_id}.json"
    assert path.exists(), f"no record for run {run_id}; metadata holds {files}"
    return json.loads(path.read_text(encoding="utf-8"))


def _published(data_root: Path, symbol: str = "AAPL") -> pd.DataFrame:
    """The published frame for ``symbol``, partition keys included."""
    return read_symbol_partition(data_root / "standardized" / "market_bars", symbol)


def _sessions(start: str, end: str) -> list[date]:
    return session_dates(start, end)


def _no_leftovers(data_root: Path) -> None:
    for name in (".staging", ".trash"):
        directory = data_root / name
        if directory.exists():
            assert not list(directory.rglob("*")), name


# ---------------------------------------------------------------------------
# 1. brand-new symbol
# ---------------------------------------------------------------------------
def test_first_update_downloads_the_whole_requested_range(tmp_path: Path) -> None:
    provider = SessionFakeProvider(start="2024-01-02", end="2024-03-29")

    result = update_symbol(
        symbol="AAPL", start="2024-01-02", end="2024-03-29",
        data_root=tmp_path, provider=provider,
    )

    assert result.status == STATUS_SUCCESS
    assert result.published is True
    assert provider.windows() == [(date(2024, 1, 2), date(2024, 3, 29))]

    expected = _sessions("2024-01-02", "2024-03-29")
    published = _published(tmp_path)
    assert len(published) == len(expected)
    assert sorted(date_set(published)) == expected
    assert result.new_dates_count == len(expected)
    assert result.replaced_dates_count == 0
    assert result.existing_row_count == 0
    assert result.published_row_count == len(expected)

    # Disk-level cross-check through the production read path (PyArrow): the live
    # partition was replaced during this run, so the process must never hand it
    # to DuckDB (see data_sys.summary).
    assert set(published["year"].unique()) == {2024}  # Hive keys rebuilt
    assert not published.duplicated(subset=["symbol", "date"]).any()
    assert not published[list(NEVER_NULL_COLUMNS)].isna().any().any()
    schema = ds.dataset(_live_dir(tmp_path), format="parquet", partitioning="hive").schema
    assert str(schema.field("date").type) == "date32[day]"

    assert result.raw_paths and len(result.raw_paths) == 1
    assert Path(result.raw_paths[0]).exists()
    assert len(list((tmp_path / "raw" / "market_bars" / "AAPL").glob("*.meta.json"))) == 1
    _no_leftovers(tmp_path)


def test_first_update_metadata_records_the_incremental_fields(tmp_path: Path) -> None:
    provider = SessionFakeProvider(start="2024-01-02", end="2024-02-29")

    result = update_symbol(
        symbol="AAPL", start="2024-01-02", end="2024-02-29",
        data_root=tmp_path, provider=provider,
    )

    records = _run_records(tmp_path)
    assert [record["run_id"] for record in records] == [result.run_id]  # one run, one record
    record = records[0]
    expected = len(_sessions("2024-01-02", "2024-02-29"))

    assert record["status"] == "success"
    assert record["published"] is True
    assert record["existing_row_count"] == 0
    assert record["downloaded_row_count"] == expected
    assert record["published_row_count"] == expected
    assert record["row_count"] == expected
    assert record["new_dates_count"] == expected
    assert record["replaced_dates_count"] == 0
    assert record["missing_ranges"] == [["2024-01-02", "2024-02-29"]]
    assert record["refresh_range"] is None
    assert record["raw_paths"] == [record["raw_path"]]
    assert record["plan"]["status"] == "planned"
    assert record["plan"]["expected_sessions"] == expected
    assert [window["reason"] for window in record["plan"]["windows"]] == ["initial_backfill"]
    assert record["min_date"] == "2024-01-02"
    assert record["years"] == [[2024, expected]]


# ---------------------------------------------------------------------------
# 2. incremental tail update
# ---------------------------------------------------------------------------
def test_second_update_only_downloads_the_tail_and_the_refresh_window(tmp_path: Path) -> None:
    first = SessionFakeProvider(start="2024-01-02", end="2024-06-28")
    update_symbol(
        symbol="AAPL", start="2024-01-02", end="2024-06-28",
        data_root=tmp_path, provider=first,
    )
    before = _published(tmp_path)

    second = SessionFakeProvider(start="2024-01-02", end="2024-07-05")
    result = update_symbol(
        symbol="AAPL", start="2024-01-02", end="2024-07-05",
        data_root=tmp_path, provider=second, refresh_overlap_sessions=5,
    )

    # 2024-07-04 is Independence Day, so the tail gap is 01/02/03/05 and the
    # 5-session refresh window (from 06-28) covers it -> exactly one request.
    assert second.windows() == [(date(2024, 6, 28), date(2024, 7, 5))]
    assert result.status == STATUS_SUCCESS
    assert result.existing_row_count == len(before)
    assert result.downloaded_row_count == len(_sessions("2024-06-28", "2024-07-05"))
    assert result.new_dates_count == 4
    assert result.replaced_dates_count == 1
    assert result.missing_ranges == [["2024-07-01", "2024-07-05"]]
    assert result.refresh_range == ["2024-06-28", "2024-07-05"]

    published = _published(tmp_path)
    assert sorted(date_set(published)) == _sessions("2024-01-02", "2024-07-05")
    assert len(published) == len(before) + 4
    assert not published.duplicated(subset=["symbol", "date"]).any()
    assert validate_bars(published).ok


def test_second_update_is_a_no_op_when_the_history_is_complete_and_refresh_is_off(
    tmp_path: Path,
) -> None:
    first = SessionFakeProvider(start="2024-01-02", end="2024-03-29")
    update_symbol(
        symbol="AAPL", start="2024-01-02", end="2024-03-29",
        data_root=tmp_path, provider=first,
    )
    before = _published(tmp_path)

    second = SessionFakeProvider(start="2024-01-02", end="2024-03-29")
    result = update_symbol(
        symbol="AAPL", start="2024-01-02", end="2024-03-29",
        data_root=tmp_path, provider=second, refresh_overlap_sessions=0,
    )

    assert second.calls == []
    assert result.status == STATUS_SUCCESS
    assert result.plan_status == STATUS_UP_TO_DATE
    assert result.published is False
    assert result.downloaded_row_count == 0
    assert result.raw_paths == []

    after = _published(tmp_path)
    pd.testing.assert_frame_equal(after, before)
    # the *second* run's record -- looked up by id, never by file-name order
    record = _run_record(tmp_path, result.run_id)
    assert record["plan"]["status"] == STATUS_UP_TO_DATE
    assert record["downloaded_row_count"] == 0
    assert any("refresh overlap disabled" in note for note in record["notes"])


# ---------------------------------------------------------------------------
# 5. fully covered with refresh enabled
# ---------------------------------------------------------------------------
def test_a_complete_history_only_refreshes_the_last_sessions(tmp_path: Path) -> None:
    provider = SessionFakeProvider(start="2024-01-02", end="2024-03-29")
    update_symbol(
        symbol="AAPL", start="2024-01-02", end="2024-03-29",
        data_root=tmp_path, provider=provider,
    )
    before = _published(tmp_path)

    second = SessionFakeProvider(start="2024-01-02", end="2024-03-29")
    result = update_symbol(
        symbol="AAPL", start="2024-01-02", end="2024-03-29",
        data_root=tmp_path, provider=second, refresh_overlap_sessions=5,
    )

    # 2024-03-29 is Good Friday, so the last 5 sessions end on 03-28.
    assert second.windows() == [(date(2024, 3, 22), date(2024, 3, 28))]
    assert result.new_dates_count == 0
    assert result.replaced_dates_count == 5
    assert result.missing_ranges == []
    assert result.refresh_range == ["2024-03-22", "2024-03-28"]

    after = _published(tmp_path)
    assert len(after) == len(before)
    assert sorted(date_set(after)) == sorted(date_set(before))


# ---------------------------------------------------------------------------
# 6. --no-refresh never triggers a refresh-only download
# ---------------------------------------------------------------------------
def test_no_refresh_never_downloads_when_nothing_is_missing(tmp_path: Path) -> None:
    provider = SessionFakeProvider(start="2024-01-02", end="2024-06-28")
    update_symbol(
        symbol="AAPL", start="2024-01-02", end="2024-06-28",
        data_root=tmp_path, provider=provider,
    )

    second = SessionFakeProvider(start="2024-01-02", end="2024-06-28")
    result = update_symbol(
        symbol="AAPL", start="2024-01-02", end="2024-06-28",
        data_root=tmp_path, provider=second, refresh_overlap_sessions=0,
    )

    assert second.calls == []
    assert result.status == STATUS_SUCCESS
    assert result.published is False


def test_no_refresh_still_downloads_a_real_tail_gap(tmp_path: Path) -> None:
    provider = SessionFakeProvider(start="2024-01-02", end="2024-01-31")
    update_symbol(
        symbol="AAPL", start="2024-01-02", end="2024-01-31",
        data_root=tmp_path, provider=provider,
    )

    second = SessionFakeProvider(start="2024-01-02", end="2024-02-29")
    result = update_symbol(
        symbol="AAPL", start="2024-01-02", end="2024-02-29",
        data_root=tmp_path, provider=second, refresh_overlap_sessions=0,
    )

    assert second.windows() == [(date(2024, 2, 1), date(2024, 2, 29))]
    assert result.missing_ranges == [["2024-02-01", "2024-02-29"]]
    assert result.refresh_range is None
    assert sorted(date_set(_published(tmp_path))) == _sessions("2024-01-02", "2024-02-29")


# ---------------------------------------------------------------------------
# 3/4. holes are detected on the trading calendar
# ---------------------------------------------------------------------------
def _rewrite_published(tmp_path: Path, frame: pd.DataFrame) -> None:
    """Replace the live partition with ``frame`` (used to simulate a hole)."""
    write_partitioned(frame, tmp_path / "standardized" / "market_bars")


def test_an_internal_hole_is_detected_and_filled(tmp_path: Path) -> None:
    provider = SessionFakeProvider(start="2024-01-02", end="2024-03-29")
    update_symbol(
        symbol="AAPL", start="2024-01-02", end="2024-03-29",
        data_root=tmp_path, provider=provider,
    )

    holed = _published(tmp_path)
    missing = {date(2024, 2, 15), date(2024, 2, 16)}
    holed = holed[~holed["date"].dt.date.isin(missing)].reset_index(drop=True)
    _rewrite_published(tmp_path, holed)

    second = SessionFakeProvider(start="2024-01-02", end="2024-03-29")
    result = update_symbol(
        symbol="AAPL", start="2024-01-02", end="2024-03-29",
        data_root=tmp_path, provider=second, refresh_overlap_sessions=0,
    )

    assert second.windows() == [(date(2024, 2, 15), date(2024, 2, 16))]
    assert result.missing_ranges == [["2024-02-15", "2024-02-16"]]
    assert result.new_dates_count == 2
    assert result.replaced_dates_count == 0
    assert sorted(date_set(_published(tmp_path))) == _sessions("2024-01-02", "2024-03-29")


def test_a_hole_before_the_request_is_left_alone(tmp_path: Path) -> None:
    provider = SessionFakeProvider(start="2023-01-03", end="2024-03-29")
    update_symbol(
        symbol="AAPL", start="2023-01-03", end="2024-03-29",
        data_root=tmp_path, provider=provider,
    )
    holed = _published(tmp_path)
    holed = holed[holed["date"].dt.date != date(2023, 6, 15)].reset_index(drop=True)
    _rewrite_published(tmp_path, holed)
    before = _published(tmp_path)

    second = SessionFakeProvider(start="2023-01-03", end="2024-03-29")
    update_symbol(
        symbol="AAPL", start="2024-01-02", end="2024-03-29",
        data_root=tmp_path, provider=second, refresh_overlap_sessions=0,
    )

    assert second.calls == []
    pd.testing.assert_frame_equal(_published(tmp_path), before)


def test_a_request_starting_before_the_local_history_backfills_the_head(tmp_path: Path) -> None:
    provider = SessionFakeProvider(start="2024-01-02", end="2024-12-31")
    update_symbol(
        symbol="AAPL", start="2024-06-03", end="2024-12-31",
        data_root=tmp_path, provider=provider,
    )
    before = _published(tmp_path)

    second = SessionFakeProvider(start="2024-01-02", end="2024-12-31")
    result = update_symbol(
        symbol="AAPL", start="2024-01-02", end="2024-12-31",
        data_root=tmp_path, provider=second, refresh_overlap_sessions=0,
    )

    assert second.windows() == [(date(2024, 1, 2), date(2024, 5, 31))]
    assert result.missing_ranges == [["2024-01-02", "2024-05-31"]]
    after = _published(tmp_path)
    assert len(after) > len(before)
    assert sorted(date_set(after)) == _sessions("2024-01-02", "2024-12-31")
    assert result.min_date == "2024-01-02"
    assert validate_bars(after).ok


def test_the_requested_window_can_be_narrower_than_the_local_history(tmp_path: Path) -> None:
    provider = SessionFakeProvider(start="2024-01-02", end="2024-12-31")
    update_symbol(
        symbol="AAPL", start="2024-01-02", end="2024-12-31",
        data_root=tmp_path, provider=provider,
    )
    before = _published(tmp_path)

    second = SessionFakeProvider(start="2024-01-02", end="2024-12-31")
    result = update_symbol(
        symbol="AAPL", start="2024-01-02", end="2024-03-29",
        data_root=tmp_path, provider=second, refresh_overlap_sessions=0,
    )

    # nothing missing inside the narrow request, so nothing is downloaded and
    # the rows *outside* the request are never deleted
    assert second.calls == []
    assert result.plan_status == STATUS_UP_TO_DATE
    pd.testing.assert_frame_equal(_published(tmp_path), before)


def test_a_request_before_the_calendar_start_is_clamped(tmp_path: Path) -> None:
    provider = SessionFakeProvider(start="2006-10-02", end="2006-12-29")

    result = update_symbol(
        symbol="AAPL", start="2006-01-01", end="2006-12-31",
        data_root=tmp_path, provider=provider, refresh_overlap_sessions=0,
    )

    assert provider.windows() == [(date(2006, 10, 2), date(2006, 12, 31))]
    assert result.status == STATUS_SUCCESS
    record = _run_record(tmp_path, result.run_id)
    assert record["requested_start"] == "2006-01-01"
    assert record["plan"]["effective_start"] == "2006-10-02"
    assert any("clamped" in note for note in record["notes"])


# ---------------------------------------------------------------------------
# 7/8/9. merged values, returns and liquidity, end to end
# ---------------------------------------------------------------------------
def test_overlapping_rows_take_the_new_values_and_returns_are_recomputed(
    tmp_path: Path,
) -> None:
    first = SessionFakeProvider(start="2024-01-02", end="2024-03-25", seed=11)
    update_symbol(
        symbol="AAPL", start="2024-01-02", end="2024-03-25",
        data_root=tmp_path, provider=first,
    )
    before = _published(tmp_path)
    boundary = date(2024, 3, 22)

    second = SessionFakeProvider(start="2024-01-02", end="2024-03-25", seed=77)
    result = update_symbol(
        symbol="AAPL", start="2024-01-02", end="2024-03-25",
        data_root=tmp_path, provider=second, refresh_overlap_sessions=2,
    )

    after = _published(tmp_path)
    fresh = second.frame.loc[
        (pd.to_datetime(second.frame.index) >= pd.Timestamp("2024-03-22"))
        & (pd.to_datetime(second.frame.index) <= pd.Timestamp("2024-03-25"))
    ]

    assert result.new_dates_count == 0
    assert result.replaced_dates_count == len(fresh)
    assert not after.duplicated(subset=["symbol", "date"]).any()
    assert after["date"].is_monotonic_increasing

    # overlapping rows now carry the freshly downloaded prices ...
    for index, day in enumerate(fresh.index):
        row = after.loc[after["date"].dt.date == day.date()]
        assert len(row) == 1
        assert float(row["close"].iloc[0]) == float(fresh["Close"].iloc[index])
        assert float(row["adjusted_close"].iloc[0]) == float(fresh["Adj Close"].iloc[index])

    # ... and the boundary return is re-derived from the merged adjusted series
    merged_adj = after.set_index(after["date"].dt.date)["adjusted_close"]
    expected = merged_adj[boundary] / merged_adj[date(2024, 3, 21)] - 1.0
    actual = after.loc[after["date"].dt.date == boundary, "daily_return"].iloc[0]
    assert float(actual) == pytest.approx(expected)

    old_adj = before.set_index(before["date"].dt.date)["adjusted_close"]
    assert float(actual) != pytest.approx(old_adj[boundary] / old_adj[date(2024, 3, 21)] - 1.0)

    assert validate_bars(after).ok


def test_published_dollar_volume_is_always_close_times_volume(tmp_path: Path) -> None:
    provider = SessionFakeProvider(start="2024-01-02", end="2024-03-29")
    update_symbol(
        symbol="AAPL", start="2024-01-02", end="2024-03-29",
        data_root=tmp_path, provider=provider,
    )

    second = SessionFakeProvider(start="2024-01-02", end="2024-04-30", seed=42)
    update_symbol(
        symbol="AAPL", start="2024-01-02", end="2024-04-30",
        data_root=tmp_path, provider=second,
    )

    published = _published(tmp_path)
    assert np.allclose(
        published["dollar_volume"].to_numpy(),
        (published["close"] * published["volume"]).to_numpy(),
    )
    adjusted = published["adjusted_close"]
    assert np.allclose(
        published["daily_return"].iloc[1:].to_numpy(),
        (adjusted / adjusted.shift(1) - 1.0).iloc[1:].to_numpy(),
    )
    assert published["daily_return"].iloc[1:].notna().all()
    assert len(published.loc[published["daily_return"].isna()]) == 1


# ---------------------------------------------------------------------------
# 10. failures never touch the published dataset
# ---------------------------------------------------------------------------
def _publish_once(tmp_path: Path, *, seed: int = 11) -> pd.DataFrame:
    provider = SessionFakeProvider(start="2024-01-02", end="2024-03-29", seed=seed)
    assert (
        update_symbol(
            symbol="AAPL", start="2024-01-02", end="2024-03-29",
            data_root=tmp_path, provider=provider,
        ).status
        == STATUS_SUCCESS
    )
    return _published(tmp_path)


def test_a_download_failure_keeps_the_published_dataset(tmp_path: Path) -> None:
    before = _publish_once(tmp_path)

    failing = SessionFakeProvider(error=DownloadError("simulated network outage"))
    result = update_symbol(
        symbol="AAPL", start="2024-01-02", end="2024-04-30",
        data_root=tmp_path, provider=failing,
    )

    assert result.status == STATUS_FAILED
    assert "network outage" in (result.failure_reason or "")
    assert result.published is False
    pd.testing.assert_frame_equal(_published(tmp_path), before)

    record = _run_record(tmp_path, result.run_id)
    assert record["status"] == "failed"
    assert record["published"] is False
    assert any("network outage" in reason for reason in record["failure_reasons"])
    _no_leftovers(tmp_path)


def test_a_quality_failure_keeps_the_published_dataset(tmp_path: Path) -> None:
    before = _publish_once(tmp_path)

    # a negative close on a day inside the refresh window
    bad = SessionFakeProvider(
        start="2024-01-02", end="2024-04-30", close_override={"2024-04-01": -5.0}
    )
    result = update_symbol(
        symbol="AAPL", start="2024-01-02", end="2024-04-30",
        data_root=tmp_path, provider=bad,
    )

    assert result.status == STATUS_FAILED
    assert "quality" in (result.failure_reason or "").lower()
    assert result.published is False
    pd.testing.assert_frame_equal(_published(tmp_path), before)

    record = _run_record(tmp_path, result.run_id)
    assert record["status"] == "failed"
    assert record["published"] is False
    assert any("quality" in reason.lower() for reason in record["failure_reasons"])
    _no_leftovers(tmp_path)


def test_a_publish_failure_keeps_the_published_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = _publish_once(tmp_path)

    real_move = storage.shutil.move
    calls = {"count": 0}

    def flaky_move(src, dst):
        calls["count"] += 1
        if calls["count"] == 2:  # 1st = backup live->trash, 2nd = staged->live
            raise OSError("simulated disk failure")
        return real_move(src, dst)

    monkeypatch.setattr(storage.shutil, "move", flaky_move)

    fresh = SessionFakeProvider(start="2024-01-02", end="2024-04-30", seed=99)
    result = update_symbol(
        symbol="AAPL", start="2024-01-02", end="2024-04-30",
        data_root=tmp_path, provider=fresh,
    )

    assert result.status == STATUS_FAILED
    assert "publish" in (result.failure_reason or "").lower()
    assert result.published is False
    pd.testing.assert_frame_equal(_published(tmp_path), before)
    _no_leftovers(tmp_path)


def test_a_failed_publish_confirmation_keeps_the_published_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The read-back confirmation is the commit gate of the publish transaction.

    ``result.published`` may only turn true once the live partition has been
    confirmed, so a confirmation that reports an issue restores the previous
    version instead of leaving the new one live behind a "failed" run.
    """
    before = _publish_once(tmp_path)

    monkeypatch.setattr(
        "data_sys.update._confirm_publish",
        lambda config, expected_rows: ["simulated read-back mismatch"],
    )

    fresh = SessionFakeProvider(start="2024-01-02", end="2024-04-30", seed=99)
    result = update_symbol(
        symbol="AAPL", start="2024-01-02", end="2024-04-30",
        data_root=tmp_path, provider=fresh,
    )

    assert result.status == STATUS_FAILED
    assert "confirmation" in (result.failure_reason or "")
    assert result.published is False
    # value by value: the rollback put the *previous* parquet data back, it did
    # not merely keep the directory around
    pd.testing.assert_frame_equal(_published(tmp_path), before)
    assert result.published_row_count == len(before)  # what is live, not what failed

    record = _run_record(tmp_path, result.run_id)
    assert record["status"] == "failed"
    assert record["published"] is False
    assert any("read-back mismatch" in reason for reason in record["failure_reasons"])
    _no_leftovers(tmp_path)


def test_a_failed_confirmation_of_a_first_publish_leaves_no_partition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A first publish that fails its confirmation must leave *no* partition.

    There is no previous version to restore here, so the rollback has to remove
    the partition this very run swapped in: the symbol has to go back to "never
    published" instead of keeping the rows of a run that is reported as failed.
    """
    monkeypatch.setattr(
        "data_sys.update._confirm_publish",
        lambda config, expected_rows: ["simulated read-back mismatch"],
    )

    provider = SessionFakeProvider(start="2024-01-02", end="2024-03-29")
    result = update_symbol(
        symbol="AAPL", start="2024-01-02", end="2024-03-29",
        data_root=tmp_path, provider=provider,
    )

    assert result.status == STATUS_FAILED
    assert result.published is False
    assert result.published_row_count == 0
    assert "read-back mismatch" in (result.failure_reason or "")

    assert not _live_dir(tmp_path).exists()  # no partition for this symbol
    assert _published(tmp_path).empty
    # nothing else may be left in the published dataset root either
    assert not list((tmp_path / "standardized" / "market_bars").rglob("*"))

    record = _run_record(tmp_path, result.run_id)
    assert record["status"] == "failed"
    assert record["published"] is False
    assert record["published_row_count"] == 0
    assert any("read-back mismatch" in reason for reason in record["failure_reasons"])
    _no_leftovers(tmp_path)


def test_a_staged_verification_failure_keeps_the_published_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DuckDB verifies the *staging* dataset only -- never the published one."""
    from data_sys.duckdb_verify import VerificationReport

    before = _publish_once(tmp_path)

    def failing_verify(target):
        # one update run may only ever point DuckDB at its staging tree: the
        # published partitions are replaced in place (see data_sys.summary)
        assert ".staging" in str(target), f"DuckDB was pointed at {target}"
        return VerificationReport(target=str(target), ok=False, issues=["boom"])

    monkeypatch.setattr("data_sys.update.verify_dataset", failing_verify)

    fresh = SessionFakeProvider(start="2024-01-02", end="2024-04-30", seed=99)
    result = update_symbol(
        symbol="AAPL", start="2024-01-02", end="2024-04-30",
        data_root=tmp_path, provider=fresh,
    )

    assert result.status == STATUS_FAILED
    assert "DuckDB verification" in (result.failure_reason or "")
    pd.testing.assert_frame_equal(_published(tmp_path), before)
    _no_leftovers(tmp_path)


def test_an_unexpected_error_is_captured_as_a_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = _publish_once(tmp_path)

    def boom(*args, **kwargs):
        raise RuntimeError("unexpected")

    monkeypatch.setattr("data_sys.update.validate_bars", boom)

    fresh = SessionFakeProvider(start="2024-01-02", end="2024-04-30", seed=99)
    result = update_symbol(
        symbol="AAPL", start="2024-01-02", end="2024-04-30",
        data_root=tmp_path, provider=fresh,
    )

    assert result.status == STATUS_FAILED
    assert "RuntimeError" in (result.failure_reason or "")
    pd.testing.assert_frame_equal(_published(tmp_path), before)


# ---------------------------------------------------------------------------
# safe skips
# ---------------------------------------------------------------------------
def test_a_request_with_no_session_is_skipped_without_downloading(tmp_path: Path) -> None:
    provider = SessionFakeProvider(start="2024-01-02", end="2024-03-29")

    result = update_symbol(
        symbol="AAPL", start="2024-01-06", end="2024-01-07",
        data_root=tmp_path, provider=provider,
    )

    assert result.status == STATUS_SKIPPED
    assert result.plan_status == STATUS_NO_SESSIONS
    assert result.published is False
    assert provider.calls == []
    assert not _live_dir(tmp_path).exists()

    record = _run_record(tmp_path, result.run_id)
    assert record["status"] == "skipped"
    assert record["plan"]["status"] == STATUS_NO_SESSIONS
    assert any("no XNYS session" in note for note in record["notes"])


def test_a_skipped_request_keeps_the_published_dataset(tmp_path: Path) -> None:
    before = _publish_once(tmp_path)

    provider = SessionFakeProvider(start="2024-01-02", end="2024-03-29")
    result = update_symbol(
        symbol="AAPL", start="2024-07-06", end="2024-07-07",
        data_root=tmp_path, provider=provider,
    )

    assert result.status == STATUS_SKIPPED
    assert provider.calls == []
    pd.testing.assert_frame_equal(_published(tmp_path), before)
    assert result.published_row_count == len(before)


def test_a_request_entirely_outside_the_calendar_is_skipped(tmp_path: Path) -> None:
    provider = SessionFakeProvider(start="2024-01-02", end="2024-03-29")

    result = update_symbol(
        symbol="AAPL", start="1990-01-01", end="1990-12-31",
        data_root=tmp_path, provider=provider,
    )

    assert result.status == STATUS_SKIPPED
    assert provider.calls == []
    assert not _live_dir(tmp_path).exists()


# ---------------------------------------------------------------------------
# multiple windows, raw archival, config plumbing
# ---------------------------------------------------------------------------
def test_every_window_gets_its_own_raw_file_and_sidecar(tmp_path: Path) -> None:
    provider = SessionFakeProvider(start="2024-01-02", end="2024-06-28")
    update_symbol(
        symbol="AAPL", start="2024-01-02", end="2024-06-28",
        data_root=tmp_path, provider=provider,
    )

    # a tail gap plus a refresh window at a distance: two requests, two archives
    second = SessionFakeProvider(start="2024-01-02", end="2024-07-31")
    result = update_symbol(
        symbol="AAPL", start="2024-01-02", end="2024-07-31",
        data_root=tmp_path, provider=second, refresh_overlap_sessions=3,
    )

    assert len(second.windows()) >= 1
    assert len(result.raw_paths) == len(second.windows())
    raw_dir = tmp_path / "raw" / "market_bars" / "AAPL"
    assert len(list(raw_dir.glob("*.parquet"))) == 1 + len(second.windows())
    assert len(list(raw_dir.glob("*.meta.json"))) == 1 + len(second.windows())

    newest = max(raw_dir.glob("*.meta.json"), key=lambda path: path.stat().st_mtime_ns)
    sidecar = json.loads(newest.read_text(encoding="utf-8"))
    assert sidecar["plan_reason"]
    assert sidecar["window_start"] <= sidecar["window_end"]
    assert sidecar["run_id"] == result.run_id


def test_raw_file_names_carry_symbol_dates_and_run_id(tmp_path: Path) -> None:
    provider = SessionFakeProvider(start="2024-01-02", end="2024-03-29")

    result = update_symbol(
        symbol="AAPL", start="2024-01-02", end="2024-03-29",
        data_root=tmp_path, provider=provider,
    )

    name = Path(result.raw_paths[0]).name
    assert name == f"AAPL_2024-01-02_2024-03-29_{result.run_id}.parquet"


def test_a_batch_membership_is_recorded(tmp_path: Path) -> None:
    provider = SessionFakeProvider(start="2024-01-02", end="2024-03-29")

    result = update_symbol(
        symbol="AAPL", start="2024-01-02", end="2024-03-29",
        data_root=tmp_path, provider=provider,
        run_id="20240101T000000Z-abc123-AAPL", batch_run_id="20240101T000000Z-abc123",
    )

    assert result.run_id == "20240101T000000Z-abc123-AAPL"
    record = _run_record(tmp_path, result.run_id)
    assert record["batch_run_id"] == "20240101T000000Z-abc123"
    assert record["run_id"] == result.run_id


def test_the_symbol_is_normalized_before_download(tmp_path: Path) -> None:
    provider = SessionFakeProvider(start="2024-01-02", end="2024-03-29")

    result = update_symbol(
        symbol="  aapl  ", start="2024-01-02", end="2024-03-29",
        data_root=tmp_path, provider=provider,
    )

    assert result.symbol == "AAPL"
    assert provider.calls[0][0] == "AAPL"
    assert _live_dir(tmp_path, "AAPL").exists()


def test_a_dotted_symbol_stays_canonical_outside_the_provider(tmp_path: Path) -> None:
    """The vendor spelling is the provider's business: ``BRK.B`` travels here."""
    provider = SessionFakeProvider(start="2024-01-02", end="2024-03-29")

    result = update_symbol(
        symbol="brk.b", start="2024-01-02", end="2024-03-29",
        data_root=tmp_path, provider=provider,
    )

    assert result.symbol == "BRK.B"
    assert provider.calls[0][0] == "BRK.B"
    assert _live_dir(tmp_path, "BRK.B").exists()
    assert (tmp_path / "raw" / "market_bars" / "BRK.B").is_dir()
    assert Path(result.raw_paths[0]).name.startswith("BRK.B_2024-01-02_2024-03-29_")


def test_build_config_rejects_an_inverted_range() -> None:
    with pytest.raises(DataPipelineError):
        build_config("AAPL", "2024-06-03", "2024-01-02", None)


def test_build_config_defaults_and_derived_paths(tmp_path: Path) -> None:
    cfg = build_config("aapl", None, None, tmp_path)

    assert cfg.symbol == "AAPL"
    assert cfg.start == date(2020, 1, 1)
    assert cfg.end == date.today()
    assert cfg.data_root == tmp_path
    assert cfg.raw_symbol_dir == tmp_path / "raw" / "market_bars" / "AAPL"
    assert cfg.metadata_path("r1") == tmp_path / "metadata" / "run_r1.json"
    assert cfg.batch_metadata_path("b1") == tmp_path / "metadata" / "batch_b1.json"


def test_the_result_status_dict_is_json_serializable(tmp_path: Path) -> None:
    provider = SessionFakeProvider(start="2024-01-02", end="2024-03-29")

    result = update_symbol(
        symbol="AAPL", start="2024-01-02", end="2024-03-29",
        data_root=tmp_path, provider=provider,
    )

    payload = result.to_status_dict()

    assert json.loads(json.dumps(payload)) == payload
    assert result.ok is True
    for key in (
        "symbol",
        "status",
        "existing_row_count",
        "downloaded_row_count",
        "published_row_count",
        "new_dates_count",
        "replaced_dates_count",
        "missing_ranges",
        "refresh_range",
        "raw_paths",
        "failure_reason",
    ):
        assert key in payload, key






