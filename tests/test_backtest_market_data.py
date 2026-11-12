"""Tests for the backtest layer's read-only market-data reader.

Every dataset here is built in ``tmp_path`` and published with the real
:func:`data_sys.storage.write_partitioned`, so the reader is exercised through the
same Hive-partitioned layout the pipeline writes: no committed ``data/`` directory
is read, nothing is written outside ``tmp_path`` and nothing touches the network.

Covered in this order:

* :class:`BacktestMarketDataReader` as the replaceable interface (abstract; a
  minimal in-memory implementation drops in for the parquet one),
* the default and the custom dataset roots of
  :class:`ParquetBacktestMarketDataReader`,
* the read contract: both ends of the window included, ``end_date`` as a hard upper
  bound, column order, ``(symbol, date)`` order, untouched values, no derived
  column,
* empty answers that are still correctly shaped (no ticker, absent symbol, empty
  window, a gap in the middle, a short history),
* every rejection: bad symbols, impossible dates, bad field lists, unusable roots, a
  missing field, an unreadable ``date`` column and a duplicated bar,
* read-only, offline behaviour, plus the module's architectural boundary and the
  package surface.
"""

from __future__ import annotations

import ast
import re
import socket
from collections.abc import Iterable, Sequence
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest

import backtest_sys
import backtest_sys.market_data as market_data
from backtest_sys import (
    BAR_FIELDS,
    BacktestContractError,
    BacktestError,
    BacktestMarketDataError,
    BacktestMarketDataReader,
    ParquetBacktestMarketDataReader,
)
from data_sys.config import PipelineConfig
from data_sys.errors import DataPipelineError
from data_sys.schema import STANDARD_COLUMNS
from data_sys.storage import read_partitioned, write_partitioned
from tests.helpers import make_standardized_frame

#: The symbols every fixture publishes; a read is free to ask for fewer.
SYMBOLS: tuple[str, ...] = ("AAPL", "MSFT", "NVDA")

#: First session of the published fixture data (a Wednesday).
DATASET_START: str = "2023-11-01"

#: Sessions per symbol: enough to cross the 2023/2024 partition boundary.
DATASET_PERIODS: int = 60

#: The module under test, for the boundary tests at the end of this file.
MODULE_PATH: Path = Path(backtest_sys.__file__).parent / "market_data.py"


# ---------------------------------------------------------------------------
# Datasets (tmp_path only) and expectations (plain pandas only)
# ---------------------------------------------------------------------------
def dataset_frame(
    symbols: Sequence[str] = SYMBOLS,
    *,
    periods: int = DATASET_PERIODS,
    start: str = DATASET_START,
) -> pd.DataFrame:
    """A contract-valid standardized frame for ``symbols``, ready to publish."""
    frames = [
        make_standardized_frame(symbol=symbol, periods=periods, start=start) for symbol in symbols
    ]
    return pd.concat(frames, ignore_index=True)


def publish(root: Path, frame: pd.DataFrame | None = None) -> Path:
    """Publish ``frame`` (by default one :func:`dataset_frame`) under ``root``."""
    write_partitioned(dataset_frame() if frame is None else frame, root)
    return root


def stored_frame(root: Path) -> pd.DataFrame:
    """The published dataset, read back by the storage layer itself."""
    return read_partitioned(root)


def stored_dates(root: Path) -> list[date]:
    """Every session the published dataset holds, ascending."""
    return sorted({pd.Timestamp(value).date() for value in stored_frame(root)["date"]})


def window(root: Path, first: int = 0, last: int = -1) -> tuple[date, date]:
    """A ``(start, end)`` pair taken from the dataset's own sessions."""
    sessions = stored_dates(root)
    return sessions[first], sessions[last]


def explicit_frame(
    symbol: str = "AAPL",
    dates: Sequence[str] = ("2024-01-02", "2024-01-03", "2024-01-04"),
    *,
    opens: Sequence[float] = (10.0, 11.0, 12.0),
    closes: Sequence[float] = (20.0, 21.0, 22.0),
    adjusted: Sequence[float] = (31.0, 32.5, 33.75),
) -> pd.DataFrame:
    """A standardized-shaped frame whose three price columns never agree.

    ``open``, ``close`` and ``adjusted_close`` are given *disjoint* value sets on
    purpose: a reader that confused them, rescaled one into another or substituted
    ``adjusted_close`` for a traded price would be caught by the numbers alone.
    """
    sessions = list(dates)
    volumes = [1_000 + 1_000 * index for index in range(len(sessions))]
    return pd.DataFrame(
        {
            "date": pd.to_datetime(sessions),
            "symbol": [symbol] * len(sessions),
            "open": list(opens),
            "high": [40.0] * len(sessions),
            "low": [5.0] * len(sessions),
            "close": list(closes),
            "adjusted_close": list(adjusted),
            "volume": volumes,
            "daily_return": [float("nan"), 0.05, 0.04][: len(sessions)],
            "dollar_volume": [
                close * volume for close, volume in zip(closes, volumes, strict=True)
            ],
        }
    )


def expected_frame(
    root: Path,
    symbols: Sequence[str],
    start: date | str,
    end: date | str,
    fields: Sequence[str],
) -> pd.DataFrame:
    """The answer a read must give, computed independently with plain pandas.

    Deliberately the obvious implementation -- ``date`` *objects* compared with
    ``>=``/``<=`` and only ``isin``/``sort_values``/``reset_index`` -- so agreement
    with the reader is evidence instead of a tautology.
    """
    wanted = set(symbols)
    first = start if isinstance(start, date) else date.fromisoformat(start)
    last = end if isinstance(end, date) else date.fromisoformat(end)

    stored = stored_frame(root)
    sessions = [pd.Timestamp(value).date() for value in stored["date"]]
    keep = [
        symbol in wanted and first <= session <= last
        for symbol, session in zip(stored["symbol"], sessions, strict=True)
    ]
    selected = stored.loc[keep, ["date", "symbol", *fields]]
    return selected.sort_values(["symbol", "date"], kind="stable").reset_index(drop=True)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """A published three-symbol dataset under ``tmp_path``."""
    return publish(tmp_path / "market_bars")


@pytest.fixture
def reader(root: Path) -> ParquetBacktestMarketDataReader:
    """A reader pointed at the fixture dataset."""
    return ParquetBacktestMarketDataReader(root)


# ---------------------------------------------------------------------------
# 1. The interface: abstract, and replaceable by any source
# ---------------------------------------------------------------------------
class InMemoryReader(BacktestMarketDataReader):
    """A minimal source that keeps its bars in a frame and answers for itself.

    Written without a single call into this package, because that is what
    "replaceable" has to mean: a vendor drop or a database implements
    :meth:`read_bars` and nothing above it notices.
    """

    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame

    def read_bars(
        self,
        symbols: tuple[str, ...],
        start_date: date | str,
        end_date: date | str,
        required_fields: Iterable[str],
    ) -> pd.DataFrame:
        fields = tuple(required_fields)
        sessions = pd.to_datetime(self.frame["date"])
        keep = (
            self.frame["symbol"].isin(symbols)
            & (sessions >= pd.Timestamp(start_date))
            & (sessions <= pd.Timestamp(end_date))
        ).to_numpy()
        selected = self.frame.loc[keep, ["date", "symbol", *fields]]
        return selected.sort_values(["symbol", "date"], kind="stable").reset_index(drop=True)


def test_the_reader_interface_is_abstract_with_one_abstract_method() -> None:
    assert BacktestMarketDataReader.__abstractmethods__ == frozenset({"read_bars"})
    assert isinstance(BacktestMarketDataReader, type)
    assert issubclass(BacktestMarketDataReader, object)

    with pytest.raises(TypeError):
        BacktestMarketDataReader()  # type: ignore[abstract]


def test_the_parquet_reader_is_an_implementation_of_the_interface(
    reader: ParquetBacktestMarketDataReader,
) -> None:
    assert isinstance(reader, BacktestMarketDataReader)
    assert ParquetBacktestMarketDataReader.__abstractmethods__ == frozenset()


def test_a_minimal_in_memory_reader_can_replace_the_parquet_reader(root: Path) -> None:
    start, end = window(root, 5, 25)
    fields = ("open", "close")

    def read_with(source: BacktestMarketDataReader) -> pd.DataFrame:
        return source.read_bars(symbols, start, end, fields)

    symbols = ("AAPL", "MSFT")
    from_disk = read_with(ParquetBacktestMarketDataReader(root))
    from_memory = read_with(InMemoryReader(stored_frame(root)))

    assert not from_disk.empty
    pd.testing.assert_frame_equal(from_disk, from_memory)
    assert isinstance(from_memory, pd.DataFrame)


# ---------------------------------------------------------------------------
# 2. The dataset root: the pipeline's default, or a directory of the caller's
# ---------------------------------------------------------------------------
def test_the_default_root_is_the_published_dataset() -> None:
    assert ParquetBacktestMarketDataReader().root == PipelineConfig().standardized_dir
    assert ParquetBacktestMarketDataReader(None).root == PipelineConfig().standardized_dir


def test_a_custom_root_is_accepted_as_a_path_or_a_string(root: Path) -> None:
    assert ParquetBacktestMarketDataReader(root).root == root
    assert ParquetBacktestMarketDataReader(str(root)).root == root
    assert ParquetBacktestMarketDataReader(root).root.is_absolute()


def test_the_repr_names_the_root(root: Path) -> None:
    assert repr(ParquetBacktestMarketDataReader(root)) == (
        f"ParquetBacktestMarketDataReader(market_bars_root={str(root)!r})"
    )


def test_building_a_reader_touches_nothing_on_disk(tmp_path: Path) -> None:
    missing = tmp_path / "not_published_yet"

    reader = ParquetBacktestMarketDataReader(missing)

    assert reader.root == missing
    assert not missing.exists()


# ---------------------------------------------------------------------------
# 3. The answer: columns, field order, row order, a fresh frame
# ---------------------------------------------------------------------------
def test_a_multi_symbol_read_crossing_a_partition_boundary(root: Path) -> None:
    sessions = stored_dates(root)
    start, end = sessions[0], sessions[-1]
    fields = ("open", "close")

    bars = ParquetBacktestMarketDataReader(root).read_bars(SYMBOLS, start, end, fields)

    assert not bars.empty
    assert start.year == 2023 and end.year == 2024
    assert sorted(set(bars["symbol"])) == list(SYMBOLS)
    assert bars["date"].min().date() < date(2024, 1, 1) < bars["date"].max().date()
    pd.testing.assert_frame_equal(bars, expected_frame(root, SYMBOLS, start, end, fields))


def test_the_columns_are_the_labels_then_the_requested_fields(
    reader: ParquetBacktestMarketDataReader, root: Path
) -> None:
    start, end = window(root, 0, 10)

    bars = reader.read_bars(("AAPL",), start, end, ("open", "close", "adjusted_close"))

    assert list(bars.columns) == ["date", "symbol", "open", "close", "adjusted_close"]


def test_the_requested_field_order_is_preserved(
    reader: ParquetBacktestMarketDataReader, root: Path
) -> None:
    start, end = window(root, 0, 10)

    bars = reader.read_bars(("AAPL",), start, end, ("volume", "open", "low"))

    assert list(bars.columns) == ["date", "symbol", "volume", "open", "low"]


def test_any_iterable_of_fields_is_accepted_and_kept_in_order(
    reader: ParquetBacktestMarketDataReader, root: Path
) -> None:
    start, end = window(root, 0, 5)

    from_list = reader.read_bars(("AAPL",), start, end, ["close", "open"])
    from_generator = reader.read_bars(("AAPL",), start, end, (field for field in ("close", "open")))

    assert list(from_list.columns) == ["date", "symbol", "close", "open"]
    pd.testing.assert_frame_equal(from_list, from_generator)


def test_the_rows_are_sorted_by_symbol_then_date_on_a_range_index(
    reader: ParquetBacktestMarketDataReader, root: Path
) -> None:
    start, end = window(root, 0, 20)

    bars = reader.read_bars(SYMBOLS, start, end, ("close",))

    assert isinstance(bars.index, pd.RangeIndex)
    assert list(bars.index) == list(range(len(bars)))
    keys = list(zip(bars["symbol"], bars["date"], strict=True))
    assert keys == sorted(keys)
    for symbol in SYMBOLS:
        symbol_dates = list(bars.loc[bars["symbol"] == symbol, "date"])
        assert symbol_dates == sorted(symbol_dates)


def test_the_hive_partition_key_is_never_returned(
    reader: ParquetBacktestMarketDataReader, root: Path
) -> None:
    start, end = window(root, 0, 5)

    bars = reader.read_bars(("AAPL",), start, end, BAR_FIELDS)

    assert "year" not in bars.columns
    assert list(bars.columns) == ["date", "symbol", *BAR_FIELDS]
    assert "year" in stored_frame(root).columns  # the dataset does publish it


def test_a_mutated_result_never_changes_the_next_read(
    reader: ParquetBacktestMarketDataReader, root: Path
) -> None:
    start, end = window(root, 0, 10)
    fields = ("open", "close")

    first = reader.read_bars(("AAPL",), start, end, fields)
    untouched = first.copy(deep=True)

    first.loc[first.index[0], "open"] = -1.0
    first["injected"] = 0.0
    first.drop(first.index, inplace=True)

    second = reader.read_bars(("AAPL",), start, end, fields)

    pd.testing.assert_frame_equal(second, untouched)
    assert second is not first


# ---------------------------------------------------------------------------
# 4. The price language: three columns, three jobs, no rewriting
# ---------------------------------------------------------------------------
def test_the_three_price_columns_are_handed_through_untouched(tmp_path: Path) -> None:
    source = explicit_frame()
    root = publish(tmp_path / "explicit", source)

    bars = ParquetBacktestMarketDataReader(root).read_bars(
        ("AAPL",), "2024-01-02", "2024-01-04", ("open", "close", "adjusted_close")
    )

    opens = tuple(bars["open"])
    closes = tuple(bars["close"])
    adjusted = tuple(bars["adjusted_close"])
    assert opens == (10.0, 11.0, 12.0)
    assert closes == (20.0, 21.0, 22.0)
    assert adjusted == (31.0, 32.5, 33.75)
    # Three disjoint value sets: nothing was scaled, substituted or mixed up.
    assert not set(opens) & set(closes)
    assert not set(opens) & set(adjusted)
    assert not set(closes) & set(adjusted)


def test_the_whole_bar_comes_back_exactly_as_published(tmp_path: Path) -> None:
    source = explicit_frame()
    root = publish(tmp_path / "explicit", source)

    bars = ParquetBacktestMarketDataReader(root).read_bars(
        SYMBOLS[:1], "2024-01-01", "2024-12-31", BAR_FIELDS
    )

    pd.testing.assert_frame_equal(
        bars,
        source[["date", "symbol", *BAR_FIELDS]].reset_index(drop=True),
        check_dtype=False,
    )


def test_the_reader_returns_no_derived_column(
    reader: ParquetBacktestMarketDataReader, root: Path
) -> None:
    start, end = window(root, 0, 10)

    bars = reader.read_bars(("AAPL",), start, end, BAR_FIELDS)

    assert "daily_return" not in bars.columns
    assert "dollar_volume" not in bars.columns
    assert set(bars.columns) - {"date", "symbol"} == set(BAR_FIELDS)
    assert set(bars.columns) < set(STANDARD_COLUMNS) | {"date", "symbol"}


def test_a_symbol_without_a_partition_is_not_an_error(
    reader: ParquetBacktestMarketDataReader, root: Path
) -> None:
    start, end = window(root, 0, 10)

    bars = reader.read_bars(("ZZZZ",), start, end, ("close",))

    assert list(bars.columns) == ["date", "symbol", "close"]
    assert bars.empty


# ---------------------------------------------------------------------------
# 5. The window: both ends included, end_date a hard upper bound
# ---------------------------------------------------------------------------
def test_both_ends_of_the_window_are_included(
    reader: ParquetBacktestMarketDataReader, root: Path
) -> None:
    sessions = stored_dates(root)
    start, end = sessions[10], sessions[20]

    bars = reader.read_bars(("AAPL",), start, end, ("close",))

    read_sessions = [value.date() for value in bars["date"]]
    assert read_sessions[0] == start
    assert read_sessions[-1] == end
    assert read_sessions == sessions[10:21]
    assert len(bars) == 11


def test_the_end_date_is_a_hard_upper_bound(
    reader: ParquetBacktestMarketDataReader, root: Path
) -> None:
    sessions = stored_dates(root)
    start, end = sessions[0], sessions[5]

    bars = reader.read_bars(("AAPL",), start, end, ("open", "close"))
    later = reader.read_bars(("AAPL",), start, sessions[6], ("open", "close"))

    assert bars["date"].max().date() == end
    assert (bars["date"].dt.date <= end).all()
    assert len(later) == len(bars) + 1
    assert later["date"].max().date() == sessions[6]


def test_a_window_that_runs_backwards_is_refused(
    reader: ParquetBacktestMarketDataReader, root: Path
) -> None:
    start, end = window(root, 0, 10)

    with pytest.raises(BacktestMarketDataError) as excinfo:
        reader.read_bars(("AAPL",), end, start, ("close",))

    assert "is after" in str(excinfo.value)
    assert excinfo.value.problems


def test_a_strict_iso_string_and_a_date_are_the_same_window(
    reader: ParquetBacktestMarketDataReader, root: Path
) -> None:
    start, end = window(root, 3, 13)

    from_dates = reader.read_bars(("MSFT",), start, end, ("close",))
    from_strings = reader.read_bars(("MSFT",), start.isoformat(), end.isoformat(), ("close",))

    assert not from_dates.empty
    pd.testing.assert_frame_equal(from_dates, from_strings)


def test_a_padded_iso_string_names_the_same_day(
    reader: ParquetBacktestMarketDataReader, root: Path
) -> None:
    start, end = window(root, 3, 13)

    padded = reader.read_bars(
        ("MSFT",), f" {start.isoformat()} ", f" {end.isoformat()} ", ("close",)
    )
    plain = reader.read_bars(("MSFT",), start, end, ("close",))

    pd.testing.assert_frame_equal(padded, plain)


def test_a_window_before_the_dataset_returns_the_header_only(
    reader: ParquetBacktestMarketDataReader,
) -> None:
    bars = reader.read_bars(("AAPL",), "2019-01-01", "2019-12-31", ("open", "close"))

    assert list(bars.columns) == ["date", "symbol", "open", "close"]
    assert bars.empty


# ---------------------------------------------------------------------------
# 6. Empty and partial answers: an answer, never an error
# ---------------------------------------------------------------------------
def test_an_empty_symbol_tuple_returns_the_header_only(
    reader: ParquetBacktestMarketDataReader, root: Path
) -> None:
    start, end = window(root, 0, 10)

    bars = reader.read_bars((), start, end, ("open", "close"))

    assert list(bars.columns) == ["date", "symbol", "open", "close"]
    assert bars.empty
    assert isinstance(bars.index, pd.RangeIndex)


def test_an_empty_symbol_tuple_needs_no_dataset_at_all(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def explode(*args: object, **kwargs: object) -> pd.DataFrame:
        raise AssertionError("a read of no symbol must not touch the dataset")

    monkeypatch.setattr(market_data, "read_partitioned", explode)

    bars = ParquetBacktestMarketDataReader(tmp_path / "never_published").read_bars(
        (), "2024-01-01", "2024-01-31", ("open",)
    )

    assert list(bars.columns) == ["date", "symbol", "open"]
    assert bars.empty


def test_a_window_inside_one_symbols_history_returns_the_others(
    tmp_path: Path,
) -> None:
    late = make_standardized_frame(symbol="LATE", periods=5, start="2024-02-01")
    root = publish(
        tmp_path / "partial", pd.concat([dataset_frame(("AAPL",)), late], ignore_index=True)
    )
    reader = ParquetBacktestMarketDataReader(root)

    bars = reader.read_bars(("AAPL", "LATE"), "2023-12-01", "2023-12-15", ("close",))

    assert not bars.empty
    assert set(bars["symbol"]) == {"AAPL"}


def test_a_window_matching_no_row_returns_a_shaped_empty_answer(
    reader: ParquetBacktestMarketDataReader,
) -> None:
    bars = reader.read_bars(SYMBOLS, "2030-01-01", "2030-01-31", ("open", "volume"))

    assert list(bars.columns) == ["date", "symbol", "open", "volume"]
    assert bars.empty


def test_a_gap_in_the_middle_of_a_history_is_returned_as_it_is(tmp_path: Path) -> None:
    source = dataset_frame(("AAPL",)).reset_index(drop=True)
    missing = pd.Timestamp(source.loc[10, "date"])
    holed = source.drop(index=[10]).reset_index(drop=True)
    root = publish(tmp_path / "holed", holed)

    bars = ParquetBacktestMarketDataReader(root).read_bars(
        ("AAPL",),
        pd.Timestamp(source.loc[8, "date"]).date(),
        pd.Timestamp(source.loc[12, "date"]).date(),
        ("close", "open"),
    )

    sessions = [value.date() for value in bars["date"]]
    assert missing.date() not in sessions
    assert len(bars) == 4  # five sessions published, one removed: no row is invented
    assert sessions == sorted(sessions)


def test_a_symbol_whose_history_ends_early_is_read_up_to_its_last_bar(tmp_path: Path) -> None:
    long = dataset_frame(("AAPL",))
    short = make_standardized_frame(symbol="OLD", periods=15, start=DATASET_START)
    root = publish(tmp_path / "delisted", pd.concat([long, short], ignore_index=True))
    reader = ParquetBacktestMarketDataReader(root)

    bars = reader.read_bars(("OLD",), DATASET_START, "2024-12-31", ("close",))

    assert len(bars) == len(short)
    assert bars["date"].min().date() == date.fromisoformat(DATASET_START)
    assert bars["date"].max() == pd.Timestamp(short["date"].max())


# ---------------------------------------------------------------------------
# 7. A dataset that cannot answer honestly is refused, in this layer's words
# ---------------------------------------------------------------------------
def test_a_missing_root_is_refused(tmp_path: Path) -> None:
    reader = ParquetBacktestMarketDataReader(tmp_path / "gone")

    with pytest.raises(BacktestMarketDataError) as excinfo:
        reader.read_bars(("AAPL",), "2024-01-01", "2024-01-31", ("close",))

    assert "does not exist" in str(excinfo.value)


def test_a_root_that_is_a_file_is_refused(tmp_path: Path) -> None:
    target = tmp_path / "market_bars"
    target.write_text("not a dataset", encoding="utf-8")

    with pytest.raises(BacktestMarketDataError) as excinfo:
        ParquetBacktestMarketDataReader(target).read_bars(
            ("AAPL",), "2024-01-01", "2024-01-31", ("close",)
        )

    assert "is not a directory" in str(excinfo.value)


def test_an_empty_directory_is_refused(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()

    with pytest.raises(BacktestMarketDataError) as excinfo:
        ParquetBacktestMarketDataReader(empty).read_bars(
            ("AAPL",), "2024-01-01", "2024-01-31", ("close",)
        )

    assert "no '*.parquet' file" in str(excinfo.value)


def test_a_directory_holding_only_other_files_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "with_notes"
    (root / "symbol=AAPL" / "year=2024").mkdir(parents=True)
    (root / "symbol=AAPL" / "year=2024" / "notes.csv").write_text("date,close\n", encoding="utf-8")
    (root / "README.txt").write_text("published elsewhere\n", encoding="utf-8")

    with pytest.raises(BacktestMarketDataError) as excinfo:
        ParquetBacktestMarketDataReader(root).read_bars(
            ("AAPL",), "2024-01-01", "2024-01-31", ("close",)
        )

    assert "no '*.parquet' file" in str(excinfo.value)


def test_a_requested_field_the_dataset_does_not_publish_is_refused(tmp_path: Path) -> None:
    source = dataset_frame(("AAPL",)).drop(columns=["close"])
    root = publish(tmp_path / "no_close", source)
    reader = ParquetBacktestMarketDataReader(root)

    with pytest.raises(BacktestMarketDataError) as excinfo:
        reader.read_bars(("AAPL",), "2023-12-01", "2023-12-15", ("open", "close"))

    assert "close" in str(excinfo.value)
    assert "does not provide" in str(excinfo.value)

    # The fields the dataset does publish are still readable.
    bars = reader.read_bars(("AAPL",), "2023-12-01", "2023-12-15", ("open",))
    assert not bars.empty


def test_a_duplicated_bar_is_refused_instead_of_chosen_between(tmp_path: Path) -> None:
    source = dataset_frame(("AAPL",)).reset_index(drop=True)
    duplicated = pd.concat([source, source.iloc[[10]]], ignore_index=True)
    root = publish(tmp_path / "duplicated", duplicated)

    with pytest.raises(BacktestMarketDataError) as excinfo:
        ParquetBacktestMarketDataReader(root).read_bars(
            ("AAPL",), "2023-12-01", "2023-12-15", ("close",)
        )

    message = str(excinfo.value)
    assert "more than one bar" in message
    assert pd.Timestamp(source.loc[10, "date"]).date().isoformat() in message


def test_a_duplicate_outside_the_window_is_still_refused(tmp_path: Path) -> None:
    """A doubled bar makes the symbol unreadable, wherever the two copies sit."""
    source = dataset_frame(("AAPL",)).reset_index(drop=True)
    duplicated = pd.concat([source, source.iloc[[0]]], ignore_index=True)
    root = publish(tmp_path / "duplicated", duplicated)

    with pytest.raises(BacktestMarketDataError):
        ParquetBacktestMarketDataReader(root).read_bars(
            ("AAPL",), "2023-12-01", "2023-12-15", ("close",)
        )


def test_a_duplicate_of_a_symbol_nobody_asked_for_is_ignored(tmp_path: Path) -> None:
    duplicated_msft = dataset_frame(("MSFT",))
    duplicated_msft = pd.concat([duplicated_msft, duplicated_msft.iloc[[0]]], ignore_index=True)
    root = publish(
        tmp_path / "duplicated",
        pd.concat([dataset_frame(("AAPL",)), duplicated_msft], ignore_index=True),
    )

    bars = ParquetBacktestMarketDataReader(root).read_bars(
        ("AAPL",), "2023-12-01", "2023-12-15", ("close",)
    )

    assert not bars.empty
    assert set(bars["symbol"]) == {"AAPL"}


def test_a_dataset_without_a_date_column_is_refused(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = stored_frame(root).drop(columns=["date"])
    monkeypatch.setattr(market_data, "read_partitioned", lambda path: source.copy())

    with pytest.raises(BacktestMarketDataError) as excinfo:
        ParquetBacktestMarketDataReader(root).read_bars(
            ("AAPL",), "2024-01-01", "2024-01-31", ("close",)
        )

    assert "'date'" in str(excinfo.value)


def test_a_dataset_without_a_symbol_column_is_refused(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = stored_frame(root).drop(columns=["symbol"])
    monkeypatch.setattr(market_data, "read_partitioned", lambda path: source.copy())

    with pytest.raises(BacktestMarketDataError) as excinfo:
        ParquetBacktestMarketDataReader(root).read_bars(
            ("AAPL",), "2024-01-01", "2024-01-31", ("close",)
        )

    assert "'symbol'" in str(excinfo.value)


def test_a_date_column_that_cannot_be_read_is_refused(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = stored_frame(root)
    source["date"] = "not-a-date"
    monkeypatch.setattr(market_data, "read_partitioned", lambda path: source.copy())

    with pytest.raises(BacktestMarketDataError) as excinfo:
        ParquetBacktestMarketDataReader(root).read_bars(
            ("AAPL",), "2024-01-01", "2024-01-31", ("close",)
        )

    assert "missing or unparseable" in str(excinfo.value)


def test_a_missing_date_value_is_refused(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = stored_frame(root)
    source.loc[source.index[0], "date"] = None
    monkeypatch.setattr(market_data, "read_partitioned", lambda path: source.copy())

    with pytest.raises(BacktestMarketDataError) as excinfo:
        ParquetBacktestMarketDataReader(root).read_bars(
            ("AAPL",), "2024-01-01", "2024-01-31", ("close",)
        )

    assert "1 missing or unparseable value(s)" in str(excinfo.value)


def test_a_storage_failure_is_wrapped_in_this_layers_error(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(path: Path) -> pd.DataFrame:
        raise OSError("the parquet file is corrupt")

    monkeypatch.setattr(market_data, "read_partitioned", broken)

    with pytest.raises(BacktestMarketDataError) as excinfo:
        ParquetBacktestMarketDataReader(root).read_bars(
            ("AAPL",), "2024-01-01", "2024-01-31", ("close",)
        )

    assert "could not be read" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, OSError)


def test_a_source_that_returns_something_else_than_a_frame_is_refused(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(market_data, "read_partitioned", lambda path: {"close": [1.0]})

    with pytest.raises(BacktestMarketDataError) as excinfo:
        ParquetBacktestMarketDataReader(root).read_bars(
            ("AAPL",), "2024-01-01", "2024-01-31", ("close",)
        )

    assert "pandas DataFrame" in str(excinfo.value)


def test_a_source_that_returns_a_frame_without_columns_is_refused(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(market_data, "read_partitioned", lambda path: pd.DataFrame())

    with pytest.raises(BacktestMarketDataError) as excinfo:
        ParquetBacktestMarketDataReader(root).read_bars(
            ("AAPL",), "2024-01-01", "2024-01-31", ("close",)
        )

    assert "no columns at all" in str(excinfo.value)


# ---------------------------------------------------------------------------
# 8. Requests this layer refuses, with every reason collected
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "symbols",
    [
        pytest.param("AAPL", id="bare-string"),
        pytest.param(["AAPL"], id="list"),
        pytest.param({"AAPL"}, id="set"),
        pytest.param(("AAPL", "AAPL"), id="duplicate"),
        pytest.param(("aapl",), id="lower-case"),
        pytest.param(("AAPL ",), id="trailing-space"),
        pytest.param(("AAPL\n",), id="trailing-newline"),
        pytest.param(("AA PL",), id="inner-space"),
        pytest.param(("",), id="blank"),
        pytest.param((42,), id="not-a-string"),
        pytest.param(("CASH",), id="cash"),
        pytest.param(("AAPL", "CASH"), id="cash-among-symbols"),
    ],
)
def test_an_unusable_symbol_set_is_refused(
    reader: ParquetBacktestMarketDataReader, root: Path, symbols: object
) -> None:
    start, end = window(root, 0, 5)

    with pytest.raises(BacktestMarketDataError) as excinfo:
        reader.read_bars(symbols, start, end, ("close",))  # type: ignore[arg-type]

    assert excinfo.value.problems
    assert "symbol" in str(excinfo.value)


def test_exchange_style_tickers_are_accepted(
    reader: ParquetBacktestMarketDataReader, root: Path
) -> None:
    """``BRK.B`` and ``BF-B`` are tickers, not mistakes: the pattern allows them."""
    start, end = window(root, 0, 5)

    bars = reader.read_bars(("BRK.B", "BF-B", "AAPL"), start, end, ("close",))

    assert set(bars["symbol"]) == {"AAPL"}
    assert list(bars.columns) == ["date", "symbol", "close"]


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(datetime(2024, 1, 2, 9, 30), id="datetime"),
        pytest.param(pd.Timestamp("2024-01-02"), id="timestamp"),
        pytest.param(pd.Timestamp("2024-01-02 15:00"), id="timestamp-with-time"),
        pytest.param("2024-01-02T00:00:00", id="iso-with-time"),
        pytest.param("2024-01-02 00:00:00", id="iso-with-space"),
        pytest.param("2024-1-2", id="unpadded"),
        pytest.param("01/02/2024", id="us-order"),
        pytest.param("20240102", id="compact"),
        pytest.param("2024-02-30", id="no-such-day"),
        pytest.param("2024-13-01", id="no-such-month"),
        pytest.param(None, id="none"),
        pytest.param(20240102, id="int"),
        pytest.param(4.5, id="float"),
        pytest.param(b"2024-01-02", id="bytes"),
    ],
)
def test_an_impossible_date_is_refused_in_either_position(
    reader: ParquetBacktestMarketDataReader, root: Path, value: object
) -> None:
    start, end = window(root, 0, 10)

    with pytest.raises(BacktestMarketDataError) as as_start:
        reader.read_bars(("AAPL",), value, end, ("close",))  # type: ignore[arg-type]
    with pytest.raises(BacktestMarketDataError) as as_end:
        reader.read_bars(("AAPL",), start, value, ("close",))  # type: ignore[arg-type]

    assert "start_date" in str(as_start.value)
    assert "end_date" in str(as_end.value)


@pytest.mark.parametrize(
    "fields",
    [
        pytest.param((), id="empty"),
        pytest.param("open", id="bare-string"),
        pytest.param(("open", "open"), id="duplicate"),
        pytest.param(("",), id="blank"),
        pytest.param((" ",), id="space"),
        pytest.param((None,), id="not-a-string"),
        pytest.param(("date",), id="label-date"),
        pytest.param(("symbol",), id="label-symbol"),
        pytest.param(("close", "pe_ratio"), id="unknown"),
        pytest.param(("daily_return",), id="derived-return"),
        pytest.param(("dollar_volume",), id="derived-dollar-volume"),
        pytest.param(None, id="none"),
        pytest.param(42, id="not-iterable"),
    ],
)
def test_an_unusable_field_list_is_refused(
    reader: ParquetBacktestMarketDataReader, root: Path, fields: object
) -> None:
    start, end = window(root, 0, 5)

    with pytest.raises(BacktestMarketDataError) as excinfo:
        reader.read_bars(("AAPL",), start, end, fields)  # type: ignore[arg-type]

    assert excinfo.value.problems


def test_every_problem_of_one_request_is_reported_at_once(
    reader: ParquetBacktestMarketDataReader, root: Path
) -> None:
    start, end = window(root, 0, 10)

    with pytest.raises(BacktestMarketDataError) as excinfo:
        reader.read_bars(
            ("aapl", "aapl"),  # two problems: lower case, then the repeat
            end,
            start,  # a third: the window runs backwards
            ("close", "date", "pe_ratio"),  # three more: label, unknown, ...
        )

    problems = excinfo.value.problems
    assert len(problems) >= 5
    bullets = str(excinfo.value).splitlines()
    assert bullets[0] == "invalid read request"
    assert all(line.startswith("  - ") for line in bullets[1:])
    assert len(bullets) - 1 == len(problems)


@pytest.mark.parametrize(
    "value",
    [
        pytest.param("", id="empty-string"),
        pytest.param("   ", id="blank-string"),
        pytest.param(42, id="int"),
        pytest.param(4.5, id="float"),
        pytest.param(object(), id="object"),
    ],
)
def test_an_unusable_root_is_refused_when_the_reader_is_built(value: object) -> None:
    with pytest.raises(BacktestMarketDataError) as excinfo:
        ParquetBacktestMarketDataReader(value)  # type: ignore[arg-type]

    assert "market_bars_root" in str(excinfo.value)


# ---------------------------------------------------------------------------
# 9. Refusal happens before the disk, so a bad request never depends on data
# ---------------------------------------------------------------------------
def test_an_illegal_request_is_refused_before_the_dataset_is_read(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def explode(*args: object, **kwargs: object) -> pd.DataFrame:
        raise AssertionError("read_partitioned must not run for a refused request")

    monkeypatch.setattr(market_data, "read_partitioned", explode)
    reader = ParquetBacktestMarketDataReader(root)

    with pytest.raises(BacktestMarketDataError):
        reader.read_bars(("aapl",), "2024-01-02", "2024-01-03", ("close",))
    with pytest.raises(BacktestMarketDataError):
        reader.read_bars(("AAPL",), "2024-01-03", "2024-01-02", ("close",))
    with pytest.raises(BacktestMarketDataError):
        reader.read_bars(("AAPL",), "2024-01-02", "2024-01-03", ("pe_ratio",))


def test_an_illegal_request_fails_identically_with_or_without_a_dataset(
    root: Path, tmp_path: Path
) -> None:
    def message(reader: ParquetBacktestMarketDataReader) -> str:
        with pytest.raises(BacktestMarketDataError) as excinfo:
            reader.read_bars(("aapl",), "2024-01-03", "2024-01-02", ("close", "date"))
        return str(excinfo.value)

    published = message(ParquetBacktestMarketDataReader(root))
    unpublished = message(ParquetBacktestMarketDataReader(tmp_path / "gone"))

    assert published == unpublished
    assert "does not exist" not in published


def test_an_unreadable_dataset_is_reported_after_the_request_is_accepted(
    tmp_path: Path,
) -> None:
    with pytest.raises(BacktestMarketDataError) as excinfo:
        ParquetBacktestMarketDataReader(tmp_path / "gone").read_bars(
            ("AAPL",), "2024-01-02", "2024-01-03", ("close",)
        )

    assert "does not exist" in str(excinfo.value)


# ---------------------------------------------------------------------------
# 10. Read-only and offline: a read changes no file and opens no socket
# ---------------------------------------------------------------------------
def files_under(root: Path) -> list[tuple[str, int, int]]:
    """Every file under ``root``: its relative name, size and modification time."""
    return sorted(
        (
            str(path.relative_to(root)),
            path.stat().st_size,
            path.stat().st_mtime_ns,
        )
        for path in root.rglob("*")
        if path.is_file()
    )


def test_a_read_leaves_every_published_file_byte_for_byte_in_place(
    reader: ParquetBacktestMarketDataReader, root: Path
) -> None:
    before = files_under(root)
    start, end = window(root, 0, 20)

    bars = reader.read_bars(SYMBOLS, start, end, BAR_FIELDS)

    assert not bars.empty
    assert files_under(root) == before


def test_a_read_never_opens_a_socket(
    reader: ParquetBacktestMarketDataReader, root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("a market-data read must never reach the network")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)

    start, end = window(root, 0, 10)
    bars = reader.read_bars(SYMBOLS, start, end, ("open", "close"))

    assert not bars.empty


def test_the_reader_uses_the_shared_storage_reader() -> None:
    assert market_data.read_partitioned is read_partitioned


# ---------------------------------------------------------------------------
# 11. The architectural boundary
# ---------------------------------------------------------------------------
#: Imports that would drag in a network client, a query engine or another layer.
FORBIDDEN_IMPORTS = re.compile(
    r"^\s*(?:import|from)\s+(?:yfinance|requests|httpx|urllib|urllib3|socket|duckdb|pyarrow"
    r"|factor_sys|strategy_sys|portfolio_sys"
    r"|data_sys\.(?:duckdb_verify|summary|providers|update|batch|pipeline|readers|standardize))"
    r"\b",
    re.MULTILINE,
)

#: Tokens that would betray a reader that writes, or reads parquet by hand.
FORBIDDEN_TOKENS: tuple[str, ...] = (
    "to_parquet",
    "read_parquet",
    "write_partitioned",
    "write_raw",
    "read_symbol_partition",
    "mkdir(",
    "unlink(",
    "rmtree",
    "write_text",
    "write_bytes",
    "to_csv",
    "ds.dataset",
    "symbol=",
    "hive_partitioning",
)

#: The only imports the reader module is allowed to have (its module boundary).
ALLOWED_IMPORTS: frozenset[str] = frozenset(
    {
        "__future__",
        "abc",
        "backtest_sys.errors",
        "collections.abc",
        "data_sys.config",
        "data_sys.schema",
        "data_sys.storage",
        "dataclasses",
        "datetime",
        "pandas",
        "pathlib",
        "re",
        "typing",
    }
)


def module_code() -> str:
    """The reader module's source with every docstring blanked out.

    Prose may name what the layer deliberately avoids -- ``symbol=<TICKER>`` in the
    layout it reads through, for instance -- so only code is searched for the
    forbidden tokens.
    """
    source = MODULE_PATH.read_text(encoding="utf-8")
    lines = source.splitlines(keepends=True)
    documented = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    for node in ast.walk(ast.parse(source, filename=str(MODULE_PATH))):
        if isinstance(node, documented) and ast.get_docstring(node, clean=False) is not None:
            docstring = node.body[0]
            for index in range(docstring.lineno - 1, docstring.end_lineno):
                lines[index] = ""
    return "".join(lines)


def imported_modules(path: Path) -> set[str]:
    """Every module named by an import statement in ``path``."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_the_reader_module_neither_writes_nor_reads_parquet_itself() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")

    match = FORBIDDEN_IMPORTS.search(source)
    assert match is None, f"market_data.py must not import {match.group(0).strip()!r}"

    code = module_code()
    offenders = [token for token in FORBIDDEN_TOKENS if token in code]
    assert not offenders, f"market_data.py must not write or re-read parquet: {offenders}"


def test_the_reader_module_depends_only_on_its_declared_layers() -> None:
    names = imported_modules(MODULE_PATH)

    assert names <= ALLOWED_IMPORTS, f"unexpected dependency: {sorted(names - ALLOWED_IMPORTS)}"
    assert {
        "pandas",
        "backtest_sys.errors",
        "data_sys.config",
        "data_sys.schema",
        "data_sys.storage",
    } <= names


def test_the_reader_module_does_not_build_on_the_contract_modules_private_helpers() -> None:
    assert "backtest_sys.contracts" not in imported_modules(MODULE_PATH)


def test_the_reader_module_computes_no_price() -> None:
    """No arithmetic: a reader selects rows and columns, it never scales a price.

    Boolean masks are combined with ``&``/``|``, so those operators are allowed --
    ``and``/``or`` cannot combine two numpy masks -- while every true arithmetic
    operator is refused.
    """
    forbidden = (
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.MatMult,
        ast.Invert,
        ast.USub,
    )

    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"), filename=str(MODULE_PATH))
    used = [type(node.op) for node in ast.walk(tree) if isinstance(node, (ast.BinOp, ast.UnaryOp))]

    assert [operator for operator in used if operator in forbidden] == []


def test_the_module_documents_the_three_price_roles() -> None:
    doc = market_data.__doc__ or ""

    assert "真实 open" in doc
    assert "真实 close" in doc
    assert "研究口径" in doc
    assert "execution" in doc
    assert "ledger" in doc
    for word in ("dividend", "split", "delisting"):
        assert word in doc.lower(), word


def test_the_bar_fields_are_the_price_and_size_columns_the_dataset_publishes() -> None:
    assert BAR_FIELDS == ("open", "high", "low", "close", "adjusted_close", "volume")
    assert set(BAR_FIELDS) <= set(STANDARD_COLUMNS)
    assert "daily_return" not in BAR_FIELDS
    assert "dollar_volume" not in BAR_FIELDS


def test_the_reader_api_is_exported_from_the_package_root() -> None:
    expected = {
        "BAR_FIELDS",
        "BacktestMarketDataError",
        "BacktestMarketDataReader",
        "ParquetBacktestMarketDataReader",
    }

    assert expected <= set(backtest_sys.__all__)
    assert backtest_sys.BacktestMarketDataReader is BacktestMarketDataReader
    assert backtest_sys.ParquetBacktestMarketDataReader is ParquetBacktestMarketDataReader
    assert backtest_sys.BacktestMarketDataError is BacktestMarketDataError
    assert backtest_sys.BAR_FIELDS == BAR_FIELDS


def test_a_read_failure_is_a_backtest_error_and_not_a_contract_violation() -> None:
    assert issubclass(BacktestMarketDataError, BacktestError)
    assert not issubclass(BacktestMarketDataError, BacktestContractError)
    assert not issubclass(BacktestMarketDataError, DataPipelineError)
    assert BacktestMarketDataError("invalid read request").problems == []


def test_the_error_collects_problems_the_way_the_contract_error_does() -> None:
    problems = ["first", "second"]

    error = BacktestMarketDataError("invalid read request", problems)
    problems.append("third")

    assert error.problems == ["first", "second"]
    assert str(error) == "invalid read request\n  - first\n  - second"
