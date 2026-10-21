"""Tests for the factor system's local market-data reader (:mod:`factor_sys.market_data`).

Every dataset here is built in ``tmp_path`` and published with the real
:func:`data_sys.storage.write_partitioned`, so the reader is exercised through the
same Hive-partitioned layout the pipeline writes: no committed ``data/`` directory
is read, nothing is written outside ``tmp_path`` and nothing touches the network.

Covered in this order:

* :class:`MarketDataReader` as the replaceable interface (abstract; a minimal
  implementation and a frame-backed one both drop in),
* the default and the custom dataset roots of :class:`ParquetMarketDataReader`,
* the read contract: window with both ends inclusive, ``as_of_date`` as a hard
  upper bound, column order, ``(symbol, date)`` order, untouched values,
* empty answers that are still correctly shaped (absent symbol, short history,
  empty window),
* every rejection: missing/empty roots, a start after ``as_of_date``, unknown,
  repeated, reserved, blank and bare-string fields, impossible dates,
* isolation (a mutated result never changes a later read) plus read-only,
  offline behaviour and the module's architectural boundary.
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

import factor_sys
import factor_sys.market_data as market_data
from data_sys.config import PipelineConfig
from data_sys.errors import DataPipelineError
from data_sys.schema import STANDARD_COLUMNS
from data_sys.storage import read_partitioned, write_partitioned
from factor_sys import (
    FactorContext,
    FactorContractError,
    FactorError,
    MarketDataReader,
    MarketDataReadError,
    ParquetMarketDataReader,
    StaticUniverseProvider,
    UniverseSnapshot,
)
from tests.helpers import make_standardized_frame, write_universe_file

#: The symbols every fixture publishes; a snapshot is free to ask for fewer.
SYMBOLS: tuple[str, ...] = ("AAPL", "MSFT", "NVDA")

#: First session of the published fixture data (a Wednesday).
DATASET_START: str = "2023-11-01"

#: Sessions per symbol: enough to cross the 2023/2024 partition boundary.
DATASET_PERIODS: int = 60

#: The session most single-purpose tests stop at; inside the fixture's range.
AS_OF: date = date(2024, 1, 10)

#: The module under test, for the boundary tests at the end of this file.
MODULE_PATH: Path = Path(factor_sys.__file__).parent / "market_data.py"


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
    """A ``(start, as_of)`` pair taken from the dataset's own sessions."""
    sessions = stored_dates(root)
    return sessions[first], sessions[last]


def snapshot_for(as_of: date | str, symbols: Sequence[str] = SYMBOLS) -> UniverseSnapshot:
    """The snapshot a read is asked for."""
    return UniverseSnapshot(as_of_date=as_of, symbols=tuple(symbols), source="tests")


def as_date(value: date | str) -> date:
    """The plain ``date`` a test's window end point stands for."""
    if isinstance(value, date):
        return value
    return date.fromisoformat(value)


def expected_frame(
    root: Path,
    snapshot: UniverseSnapshot,
    start: date | str,
    as_of: date | str,
    fields: Sequence[str],
) -> pd.DataFrame:
    """The answer a read must give, computed independently with plain pandas.

    Deliberately the obvious implementation -- ``date`` *objects* compared with
    ``>=``/``<=`` and only ``isin``/``sort_values``/``reset_index`` -- so that
    agreement with the reader is evidence instead of a tautology.
    """
    frame = stored_frame(root)
    days = pd.to_datetime(frame["date"]).dt.date
    keep = (
        frame["symbol"].isin(snapshot.symbols) & (days >= as_date(start)) & (days <= as_date(as_of))
    )
    return (
        frame.loc[keep, ["date", "symbol", *fields]]
        .sort_values(["symbol", "date"], kind="stable")
        .reset_index(drop=True)
    )


def returned_dates(frame: pd.DataFrame) -> list[date]:
    """The distinct sessions a returned frame covers, ascending."""
    return sorted({pd.Timestamp(value).date() for value in frame["date"]})


def tree_state(root: Path) -> dict[str, tuple[int, int]]:
    """Size and modification time of every entry under ``root``."""
    return {
        str(path.relative_to(root)): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in sorted(root.rglob("*"))
    }


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


# ---------------------------------------------------------------------------
# Other sources: the interface must not care which one is in use
# ---------------------------------------------------------------------------
class LiteralReader(MarketDataReader):
    """The smallest legal implementation: one method, no file, no helper."""

    def read_history(
        self,
        snapshot: UniverseSnapshot,
        start_date: date | str,
        required_fields: Iterable[str],
    ) -> pd.DataFrame:
        values = {field: [1.0] for field in required_fields}
        frame = pd.DataFrame(
            {
                "date": [pd.Timestamp(snapshot.as_of_date)],
                "symbol": [snapshot.symbols[0]],
                **values,
            }
        )
        return frame[["date", "symbol", *required_fields]]


class InMemoryReader(MarketDataReader):
    """A frame-backed source: the same rules, no dataset on disk."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame

    def read_history(
        self,
        snapshot: UniverseSnapshot,
        start_date: date | str,
        required_fields: Iterable[str],
    ) -> pd.DataFrame:
        request = market_data._prepare_request(snapshot, start_date, required_fields)
        return market_data._history_frame(self.frame, request)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """A published dataset: three symbols across two calendar years."""
    return publish(tmp_path / "standardized" / "market_bars")


@pytest.fixture
def reader(root: Path) -> ParquetMarketDataReader:
    """The v1 reader, pointed at the published fixture dataset."""
    return ParquetMarketDataReader(root)


# ---------------------------------------------------------------------------
# The interface (MarketDataReader)
# ---------------------------------------------------------------------------
def test_the_reader_interface_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        MarketDataReader()  # type: ignore[abstract]

    assert MarketDataReader.__abstractmethods__ == frozenset({"read_history"})


def test_the_parquet_reader_is_a_market_data_reader(reader: ParquetMarketDataReader) -> None:
    assert issubclass(ParquetMarketDataReader, MarketDataReader)
    assert isinstance(reader, MarketDataReader)


def test_a_minimal_reader_can_replace_the_parquet_one() -> None:
    """One method is the whole contract: no disk, no helpers, no base logic."""
    candidate: MarketDataReader = LiteralReader()

    result = candidate.read_history(snapshot_for(AS_OF), "2024-01-01", ("close",))

    assert list(result.columns) == ["date", "symbol", "close"]
    assert result.to_dict("records") == [
        {"date": pd.Timestamp(AS_OF), "symbol": "AAPL", "close": 1.0}
    ]


def test_a_second_source_reuses_the_shared_pipeline(
    root: Path, reader: ParquetMarketDataReader
) -> None:
    """Same bars, two sources, one answer -- the point of the shared half."""
    start, as_of = window(root, 10, 20)
    snapshot = snapshot_for(as_of)
    in_memory = InMemoryReader(stored_frame(root))

    from_disk = reader.read_history(snapshot, start, ("close", "volume"))
    from_memory = in_memory.read_history(snapshot, start, ("close", "volume"))

    pd.testing.assert_frame_equal(from_disk, from_memory)


def test_a_snapshot_from_the_static_provider_is_enough(tmp_path: Path, root: Path) -> None:
    """A reader only ever asks for a snapshot value, whatever produced it."""
    pool_file = write_universe_file(tmp_path / "universe.csv", ("AAPL", "MSFT"))
    start, as_of = window(root)
    snapshot = StaticUniverseProvider(pool_file).snapshot(as_of)

    result = ParquetMarketDataReader(root).read_history(snapshot, start, ("close",))

    assert set(result["symbol"]) == {"AAPL", "MSFT"}
    assert returned_dates(result) == stored_dates(root)


# ---------------------------------------------------------------------------
# Roots: the published dataset by default, any dataset on request
# ---------------------------------------------------------------------------
def test_the_fixture_dataset_really_spans_two_years(root: Path) -> None:
    """The contract tests below are only meaningful while this holds."""
    years = sorted(path.name for path in (root / "symbol=AAPL").glob("year=*"))

    assert years == ["year=2023", "year=2024"]
    assert len(stored_dates(root)) == DATASET_PERIODS
    assert set(stored_frame(root)["symbol"]) == set(SYMBOLS)


def test_the_default_root_is_the_published_dataset_directory() -> None:
    default = ParquetMarketDataReader()

    assert default.root == PipelineConfig().standardized_dir
    assert Path(default.root).parts[-3:] == ("data", "standardized", "market_bars")


def test_a_custom_root_is_read_instead(tmp_path: Path) -> None:
    elsewhere = publish(tmp_path / "elsewhere", dataset_frame(periods=5, start="2024-03-04"))
    custom = ParquetMarketDataReader(elsewhere)

    assert custom.root == elsewhere
    assert custom.root != ParquetMarketDataReader().root

    result = custom.read_history(snapshot_for("2024-03-31"), "2024-01-01", ("close",))

    assert list(result.columns) == ["date", "symbol", "close"]
    assert returned_dates(result) == stored_dates(elsewhere)
    assert set(result["symbol"]) == set(SYMBOLS)


def test_the_root_may_be_a_string(tmp_path: Path) -> None:
    root = publish(tmp_path / "ds", dataset_frame(periods=5, start="2024-03-04"))
    reader = ParquetMarketDataReader(str(root))
    start, as_of = window(root)

    assert reader.root == root
    assert not reader.read_history(snapshot_for(as_of), start, ("close",)).empty


@pytest.mark.parametrize("value", [" ", 42, 3.5, object()])
def test_an_impossible_root_is_refused_when_the_reader_is_built(value: object) -> None:
    with pytest.raises(MarketDataReadError) as excinfo:
        ParquetMarketDataReader(value)  # type: ignore[arg-type]

    assert "market_bars_root" in str(excinfo.value)


def test_a_blank_root_is_refused() -> None:
    with pytest.raises(MarketDataReadError) as excinfo:
        ParquetMarketDataReader("   ")

    assert "blank" in str(excinfo.value)


def test_a_root_that_does_not_exist_yet_is_accepted_until_a_read(tmp_path: Path) -> None:
    """The pipeline may publish *after* a reader was built, so this is not fatal."""
    reader = ParquetMarketDataReader(tmp_path / "published" / "later")

    assert reader.root == tmp_path / "published" / "later"

    with pytest.raises(MarketDataReadError) as excinfo:
        reader.read_history(snapshot_for(AS_OF), "2024-01-01", ("close",))
    assert "does not exist" in str(excinfo.value)


def test_a_reader_only_ever_sees_its_own_root(tmp_path: Path) -> None:
    first = publish(tmp_path / "first", dataset_frame(periods=5, start="2023-11-01"))
    second = publish(tmp_path / "second", dataset_frame(periods=5, start="2024-06-03"))
    snapshot = snapshot_for(date(2024, 12, 31))

    answers = {
        name: ParquetMarketDataReader(path).read_history(snapshot, "2000-01-01", ("close",))
        for name, path in (("first", first), ("second", second))
    }

    assert returned_dates(answers["first"]) == stored_dates(first)
    assert returned_dates(answers["second"]) == stored_dates(second)
    assert not set(returned_dates(answers["first"])) & set(returned_dates(answers["second"]))


# ---------------------------------------------------------------------------
# The read contract: window, order, columns, values
# ---------------------------------------------------------------------------
def test_a_read_returns_the_window_of_every_symbol(
    root: Path, reader: ParquetMarketDataReader
) -> None:
    sessions = stored_dates(root)
    start, as_of = sessions[10], sessions[20]
    snapshot = snapshot_for(as_of)

    result = reader.read_history(snapshot, start, ("close",))

    assert list(result.columns) == ["date", "symbol", "close"]
    assert len(result) == 11 * len(SYMBOLS)
    pd.testing.assert_frame_equal(result, expected_frame(root, snapshot, start, as_of, ("close",)))


def test_the_window_includes_both_ends_and_nothing_else(
    root: Path, reader: ParquetMarketDataReader
) -> None:
    sessions = stored_dates(root)
    start, as_of = sessions[10], sessions[20]

    result = reader.read_history(snapshot_for(as_of), start, ("close",))

    assert returned_dates(result) == sessions[10:21]
    assert sessions[9] not in returned_dates(result)
    assert sessions[21] not in returned_dates(result)


@pytest.mark.parametrize("index", [0, 1, 17, 30, 45, -1])
def test_the_snapshot_date_is_a_hard_upper_bound(
    root: Path, reader: ParquetMarketDataReader, index: int
) -> None:
    """No read may hand a factor a bar it could not have seen when the day closed."""
    sessions = stored_dates(root)
    resolved = index if index >= 0 else len(sessions) + index
    as_of = sessions[resolved]

    result = reader.read_history(snapshot_for(as_of), sessions[0], ("close",))

    assert returned_dates(result) == sessions[: resolved + 1]
    assert bool((result["date"] <= pd.Timestamp(as_of)).all())
    assert not bool((result["date"] > pd.Timestamp(as_of)).any())


def test_the_result_is_sorted_by_symbol_then_date(tmp_path: Path) -> None:
    """A dataset published in any row order still answers in contract order."""
    shuffled = dataset_frame().sample(frac=1.0, random_state=7).reset_index(drop=True)
    root = publish(tmp_path / "shuffled", shuffled)
    stored = stored_frame(root)
    stored_keys = list(zip(stored["symbol"], stored["date"], strict=True))
    assert stored_keys != sorted(stored_keys), "the fixture must not be sorted already"

    result = ParquetMarketDataReader(root).read_history(
        snapshot_for(date(2024, 12, 31)), "2000-01-01", ("close",)
    )

    keys = list(zip(result["symbol"], result["date"], strict=True))
    assert keys == sorted(keys)
    assert sorted(keys) == sorted(stored_keys)


@pytest.mark.parametrize("pool", [("AAPL",), ("MSFT", "NVDA"), SYMBOLS, ("AAPL", "ZZZZ")])
def test_the_pool_decides_which_symbols_come_back(
    root: Path, reader: ParquetMarketDataReader, pool: tuple[str, ...]
) -> None:
    start, as_of = window(root)
    snapshot = snapshot_for(as_of, pool)

    result = reader.read_history(snapshot, start, ("close",))

    assert set(result["symbol"]) == {symbol for symbol in pool if symbol in SYMBOLS}


@pytest.mark.parametrize(
    "fields",
    [
        ("close",),
        ("volume", "close"),
        ("close", "volume", "dollar_volume"),
        tuple(STANDARD_COLUMNS[2:]),
    ],
)
def test_the_result_holds_the_labels_then_the_fields_in_order(
    root: Path, reader: ParquetMarketDataReader, fields: tuple[str, ...]
) -> None:
    start, as_of = window(root, 15, 25)
    snapshot = snapshot_for(as_of)

    result = reader.read_history(snapshot, start, fields)

    assert list(result.columns) == ["date", "symbol", *fields]
    assert "year" not in result.columns
    assert set(result.columns) == {"date", "symbol", *fields}
    pd.testing.assert_frame_equal(result, expected_frame(root, snapshot, start, as_of, fields))


def test_the_requestable_fields_are_the_standardized_bar_columns(
    reader: ParquetMarketDataReader,
) -> None:
    result = reader.read_history(snapshot_for(AS_OF), "2024-01-01", tuple(STANDARD_COLUMNS[2:]))

    assert list(result.columns) == list(STANDARD_COLUMNS)

    with pytest.raises(MarketDataReadError) as excinfo:
        reader.read_history(snapshot_for(AS_OF), "2024-01-01", ("year",))
    assert "unknown field 'year'" in str(excinfo.value)


def test_the_values_and_dtypes_are_passed_through_untouched(
    root: Path, reader: ParquetMarketDataReader
) -> None:
    start, as_of = window(root, 5, 9)
    stored = stored_frame(root)

    result = reader.read_history(snapshot_for(as_of), start, ("close", "volume", "adjusted_close"))

    assert result["close"].dtype == stored["close"].dtype
    assert result["volume"].dtype == stored["volume"].dtype
    assert result["date"].dtype == stored["date"].dtype
    for row in result.itertuples():
        origin = stored[(stored["symbol"] == row.symbol) & (stored["date"] == row.date)]
        assert len(origin) == 1
        assert float(origin["close"].iloc[0]) == row.close
        assert int(origin["volume"].iloc[0]) == row.volume
        assert float(origin["adjusted_close"].iloc[0]) == row.adjusted_close


def test_the_reader_answers_many_requests_in_a_row(
    root: Path, reader: ParquetMarketDataReader
) -> None:
    window_start, window_end = window(root, 0, 30)
    requests = [
        (snapshot_for(window_end), window_start, ("close",)),
        (snapshot_for(window_start), window_start, ("volume",)),
        (snapshot_for(window_end), window_end, ("open", "high", "low")),
        (snapshot_for(window_end, ("NVDA",)), "2000-01-01", ("dollar_volume",)),
    ]

    for snapshot, start, fields in requests:
        result = reader.read_history(snapshot, start, fields)
        expected = expected_frame(root, snapshot, start, snapshot.as_of_date, fields)
        pd.testing.assert_frame_equal(result, expected)


# ---------------------------------------------------------------------------
# An empty answer is still an answer (no eligibility judgement here)
# ---------------------------------------------------------------------------
def test_a_pool_symbol_without_a_partition_is_not_an_error(
    root: Path, reader: ParquetMarketDataReader
) -> None:
    start, as_of = window(root)

    result = reader.read_history(snapshot_for(as_of, ("AAPL", "ZZZZ", "YYYY")), start, ("close",))

    assert not result.empty
    assert set(result["symbol"]) == {"AAPL"}


def test_a_pool_without_any_data_is_an_empty_but_shaped_frame(
    root: Path, reader: ParquetMarketDataReader
) -> None:
    start, as_of = window(root)

    result = reader.read_history(snapshot_for(as_of, ("ZZZ1", "ZZZ2")), start, ("close",))

    assert result.empty
    assert list(result.columns) == ["date", "symbol", "close"]
    assert str(result["date"].dtype) == "datetime64[s]"


def test_a_window_that_no_session_falls_into_is_an_empty_but_shaped_frame(
    root: Path, reader: ParquetMarketDataReader
) -> None:
    stored = stored_frame(root)

    result = reader.read_history(
        snapshot_for(date(2019, 12, 31)), "2019-01-01", ("close", "volume")
    )

    assert result.empty
    assert list(result.columns) == ["date", "symbol", "close", "volume"]
    assert result["close"].dtype == stored["close"].dtype
    assert result["volume"].dtype == stored["volume"].dtype


def test_a_history_shorter_than_the_window_is_not_an_error(
    root: Path, reader: ParquetMarketDataReader
) -> None:
    sessions = stored_dates(root)

    result = reader.read_history(snapshot_for(sessions[0]), "2000-01-01", ("close",))

    assert returned_dates(result) == [sessions[0]]
    assert len(result) == len(SYMBOLS)


def test_a_symbol_with_a_shorter_history_is_neither_filled_nor_dropped(
    tmp_path: Path,
) -> None:
    frame = pd.concat(
        [dataset_frame(("AAPL", "MSFT"), periods=10), dataset_frame(("NVDA",), periods=40)],
        ignore_index=True,
    )
    root = publish(tmp_path / "ragged", frame)
    start, as_of = window(root)

    result = ParquetMarketDataReader(root).read_history(snapshot_for(as_of), start, ("close",))

    assert result["symbol"].value_counts().to_dict() == {"NVDA": 40, "AAPL": 10, "MSFT": 10}
    assert len(result) == 60
    assert "is_eligible" not in result.columns


def test_a_symbol_missing_from_one_year_is_not_filled(tmp_path: Path) -> None:
    """A hole in the middle of the range is data, not something to invent."""
    frame = dataset_frame(("AAPL",))
    stored = stored_frame(publish(tmp_path / "reference", frame))
    sessions = sorted({pd.Timestamp(value).date() for value in stored["date"]})
    holed = stored[stored["date"] != pd.Timestamp(sessions[10])]
    root = publish(tmp_path / "holed", holed)

    result = ParquetMarketDataReader(root).read_history(
        snapshot_for(sessions[-1]), sessions[0], ("close",)
    )

    assert returned_dates(result) == [value for value in sessions if value != sessions[10]]
    assert len(result) == len(sessions) - 1


# ---------------------------------------------------------------------------
# Requests the system refuses to honour
# ---------------------------------------------------------------------------
#: Every spelling of a start date that must be rejected, not rounded.
BAD_START_DATES: tuple[object, ...] = (
    datetime(2024, 1, 5),
    pd.Timestamp("2024-01-05"),
    "2024-01-05T00:00:00",
    "2024-01-05 15:00",
    "2024-1-5",
    "2024/01/05",
    "05-01-2024",
    "2024-01-05Z",
    "20240105",
    "2024-02-30",
    "2024-13-01",
    "not a date",
    "",
    "   ",
    20240105,
    1.5,
    None,
    ["2024-01-05"],
    date,
)

#: Impossible field requests, each with a fragment the error must carry.
BAD_FIELD_REQUESTS: tuple[tuple[object, str], ...] = (
    ("close", "bare string"),
    ((), "must not be empty"),
    ([], "must not be empty"),
    (None, "bare string"),
    (42, "iterable of bar field names"),
    (("close", "close"), "duplicate field 'close'"),
    (("date",), "must not be requested"),
    (("symbol",), "must not be requested"),
    (("close", "date"), "must not be requested"),
    (("pe_ratio",), "unknown field 'pe_ratio'"),
    (("close", "not_a_column"), "unknown field 'not_a_column'"),
    (("Close",), "unknown field 'Close'"),
    (("",), "invalid field name"),
    (("close", "  "), "invalid field name"),
    (("close", 3), "invalid field name"),
    (("close", None), "invalid field name"),
)


@pytest.mark.parametrize("value", BAD_START_DATES)
def test_an_impossible_start_date_is_refused(
    reader: ParquetMarketDataReader, value: object
) -> None:
    with pytest.raises(MarketDataReadError) as excinfo:
        reader.read_history(snapshot_for(AS_OF), value, ("close",))  # type: ignore[arg-type]

    assert "start_date" in str(excinfo.value)
    assert isinstance(excinfo.value, FactorError)


@pytest.mark.parametrize("value", [date(2024, 1, 5), "2024-01-05", " 2024-01-05 "])
def test_a_well_spelled_start_date_is_accepted(
    reader: ParquetMarketDataReader, value: date | str
) -> None:
    result = reader.read_history(snapshot_for(date(2024, 1, 31)), value, ("close",))

    assert list(result.columns) == ["date", "symbol", "close"]
    assert not result.empty


def test_a_start_after_the_snapshot_date_is_refused(reader: ParquetMarketDataReader) -> None:
    with pytest.raises(MarketDataReadError) as excinfo:
        reader.read_history(snapshot_for(date(2024, 1, 5)), "2024-01-06", ("close",))

    message = str(excinfo.value)
    assert "2024-01-06" in message
    assert "2024-01-05" in message
    assert "forward" in message


def test_a_start_equal_to_the_snapshot_date_yields_that_one_session(
    root: Path, reader: ParquetMarketDataReader
) -> None:
    start, as_of = window(root, 20, 20)

    result = reader.read_history(snapshot_for(as_of), start, ("close",))

    assert returned_dates(result) == [as_of]
    assert len(result) == len(SYMBOLS)


@pytest.mark.parametrize("fields,fragment", BAD_FIELD_REQUESTS)
def test_an_impossible_field_request_is_refused(
    reader: ParquetMarketDataReader, fields: object, fragment: str
) -> None:
    with pytest.raises(MarketDataReadError) as excinfo:
        reader.read_history(snapshot_for(AS_OF), "2024-01-01", fields)  # type: ignore[arg-type]

    assert fragment in str(excinfo.value)
    assert isinstance(excinfo.value, FactorError)


def test_every_bad_field_is_reported_in_one_error(reader: ParquetMarketDataReader) -> None:
    with pytest.raises(MarketDataReadError) as excinfo:
        reader.read_history(
            snapshot_for(AS_OF), "2024-01-01", ("close", "pe_ratio", "date", "close", "")
        )

    problems = "\n".join(excinfo.value.problems)
    assert len(excinfo.value.problems) == 4
    for fragment in (
        "invalid field name ''",
        "field 'date' must not be requested",
        "duplicate field 'close'",
        "unknown field 'pe_ratio'",
    ):
        assert fragment in problems


@pytest.mark.parametrize(
    "value", [None, "AAPL", ("AAPL",), {"symbols": ("AAPL",)}, object(), Path("ds")]
)
def test_a_snapshot_that_is_not_a_snapshot_is_refused(
    reader: ParquetMarketDataReader, value: object
) -> None:
    with pytest.raises(MarketDataReadError) as excinfo:
        reader.read_history(value, "2024-01-01", ("close",))  # type: ignore[arg-type]

    assert "UniverseSnapshot" in str(excinfo.value)
    assert type(value).__name__ in str(excinfo.value)


# ---------------------------------------------------------------------------
# Datasets that cannot answer a request
# ---------------------------------------------------------------------------
@pytest.fixture
def unpublished_root(tmp_path: Path) -> Path:
    """A dataset directory that exists but holds no published partition."""
    path = tmp_path / "standardized" / "market_bars"
    path.mkdir(parents=True)
    return path


def test_a_missing_root_is_refused_at_read_time(tmp_path: Path) -> None:
    """Building a reader touches nothing; an empty dataset is a read failure."""
    reader = ParquetMarketDataReader(tmp_path / "never" / "published")

    with pytest.raises(MarketDataReadError) as excinfo:
        reader.read_history(snapshot_for(AS_OF), "2024-01-01", ("close",))

    assert "does not exist" in str(excinfo.value)
    assert "never" in str(excinfo.value)


def test_a_root_that_is_a_file_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "not-a-directory.parquet"
    path.write_bytes(b"")
    reader = ParquetMarketDataReader(path)

    with pytest.raises(MarketDataReadError) as excinfo:
        reader.read_history(snapshot_for(AS_OF), "2024-01-01", ("close",))

    assert "is not a directory" in str(excinfo.value)


def test_an_empty_root_is_refused(unpublished_root: Path) -> None:
    reader = ParquetMarketDataReader(unpublished_root)

    with pytest.raises(MarketDataReadError) as excinfo:
        reader.read_history(snapshot_for(AS_OF), "2024-01-01", ("close",))

    assert "no '*.parquet' file" in str(excinfo.value)


def test_a_root_without_a_published_file_is_refused(unpublished_root: Path) -> None:
    (unpublished_root / "README.md").write_text("not a dataset yet", encoding="utf-8")
    (unpublished_root / "symbol=AAPL" / "year=2024").mkdir(parents=True)

    with pytest.raises(MarketDataReadError) as excinfo:
        ParquetMarketDataReader(unpublished_root).read_history(
            snapshot_for(AS_OF), "2024-01-01", ("close",)
        )

    assert "no '*.parquet' file" in str(excinfo.value)


@pytest.mark.parametrize("dropped", ["open", "volume", "adjusted_close", "dollar_volume"])
def test_a_request_for_a_column_the_dataset_lacks_is_refused(tmp_path: Path, dropped: str) -> None:
    root = publish(tmp_path / "incomplete", dataset_frame().drop(columns=[dropped]))

    with pytest.raises(MarketDataReadError) as excinfo:
        ParquetMarketDataReader(root).read_history(snapshot_for(AS_OF), "2024-01-01", (dropped,))

    assert dropped in str(excinfo.value)
    assert "does not provide" in str(excinfo.value)


def test_a_dataset_without_one_column_still_answers_other_requests(tmp_path: Path) -> None:
    """The refusal is about the requested field, not about a useless dataset."""
    root = publish(tmp_path / "incomplete", dataset_frame().drop(columns=["volume"]))

    result = ParquetMarketDataReader(root).read_history(
        snapshot_for(AS_OF), "2024-01-01", ("close",)
    )

    assert list(result.columns) == ["date", "symbol", "close"]
    assert not result.empty


# ---------------------------------------------------------------------------
# Isolation, and a reader that only reads
# ---------------------------------------------------------------------------
def test_mutating_a_result_never_changes_a_later_read(
    root: Path, reader: ParquetMarketDataReader
) -> None:
    start, as_of = window(root, 0, 29)
    snapshot = snapshot_for(as_of)
    first = reader.read_history(snapshot, start, ("close", "volume"))

    first.loc[:, "close"] = -1.0
    first["injected"] = 0
    first.drop(index=first.index[0], inplace=True)
    first.index = range(1000, 1000 + len(first))

    second = reader.read_history(snapshot, start, ("close", "volume"))

    assert second is not first
    assert bool((second["close"] > 0).all())
    pd.testing.assert_frame_equal(
        second, expected_frame(root, snapshot, start, as_of, ("close", "volume"))
    )


def test_two_reads_of_one_request_agree(root: Path, reader: ParquetMarketDataReader) -> None:
    start, as_of = window(root, 3, 12)
    snapshot = snapshot_for(as_of)

    first = reader.read_history(snapshot, start, ("close",))
    second = reader.read_history(snapshot, start, ("close",))

    assert first is not second
    pd.testing.assert_frame_equal(first, second)


def test_a_frame_backed_source_is_not_corrupted_by_its_own_answer(root: Path) -> None:
    """The shared pipeline hands out a frame of its own, never a window on one."""
    frame = stored_frame(root)
    before = frame.copy(deep=True)
    reader = InMemoryReader(frame)
    result = reader.read_history(snapshot_for(AS_OF), "2024-01-01", ("close",))

    result.loc[:, "close"] = 0.0
    result.drop(result.index, inplace=True)

    pd.testing.assert_frame_equal(frame, before)


def test_a_read_leaves_the_dataset_byte_for_byte_untouched(
    root: Path, reader: ParquetMarketDataReader
) -> None:
    before = tree_state(root)

    reader.read_history(snapshot_for(AS_OF), "2023-11-01", ("close", "volume"))
    reader.read_history(snapshot_for(AS_OF), "2024-01-01", ("open",))
    with pytest.raises(MarketDataReadError):
        reader.read_history(snapshot_for(AS_OF), "2024-06-01", ("close",))

    assert tree_state(root) == before


def test_a_read_never_reaches_for_the_network(
    root: Path, reader: ParquetMarketDataReader, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("the market data reader must never open a socket")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)

    result = reader.read_history(snapshot_for(AS_OF), "2023-11-01", ("close",))

    assert not result.empty


def test_the_reader_refuses_before_it_touches_the_disk(
    root: Path, reader: ParquetMarketDataReader, monkeypatch: pytest.MonkeyPatch
) -> None:
    def explode(*args: object, **kwargs: object) -> pd.DataFrame:
        raise AssertionError("read_partitioned must not run for a refused request")

    monkeypatch.setattr(market_data, "read_partitioned", explode)

    with pytest.raises(MarketDataReadError):
        reader.read_history(snapshot_for(AS_OF), "2024-13-01", ("close",))
    with pytest.raises(MarketDataReadError):
        reader.read_history(snapshot_for(AS_OF), "2024-01-01", ("pe_ratio",))
    with pytest.raises(MarketDataReadError) as excinfo:
        ParquetMarketDataReader(root / "gone").read_history(
            snapshot_for(AS_OF), "2024-01-01", ("close",)
        )
    assert "does not exist" in str(excinfo.value)


def test_the_reader_uses_the_shared_storage_reader() -> None:
    assert market_data.read_partitioned is read_partitioned


# ---------------------------------------------------------------------------
# Architectural boundary and packaging
# ---------------------------------------------------------------------------
#: Imports that would drag a network client, a query engine or the writer in.
FORBIDDEN_IMPORTS = re.compile(
    r"^\s*(?:import|from)\s+(?:yfinance|requests|httpx|urllib|urllib3|socket|duckdb|pyarrow"
    r"|data_sys\.(?:duckdb_verify|summary|providers|update|batch|pipeline|readers))\b",
    re.MULTILINE,
)

#: Tokens that would betray a reader that writes, or re-reads parquet by hand.
FORBIDDEN_TOKENS: tuple[str, ...] = (
    "to_parquet",
    "read_parquet",
    "write_partitioned",
    "write_raw",
    "open(",
    "mkdir(",
    "unlink(",
    "rmtree",
    "write_text",
    "write_bytes",
    "to_csv",
    "ds.dataset",
)

#: The only imports the reader module is allowed to have (its module boundary).
ALLOWED_IMPORTS: frozenset[str] = frozenset(
    {
        "__future__",
        "abc",
        "collections.abc",
        "datetime",
        "pathlib",
        "re",
        "typing",
        "pandas",
        "data_sys.config",
        "data_sys.schema",
        "data_sys.storage",
        "factor_sys.errors",
        "factor_sys.universe",
    }
)


def test_the_reader_module_neither_writes_nor_reads_parquet_itself() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")

    match = FORBIDDEN_IMPORTS.search(source)
    assert match is None, f"market_data.py must not import {match.group(0).strip()!r}"

    offenders = [token for token in FORBIDDEN_TOKENS if token in source]
    assert not offenders, f"market_data.py must not write or re-read parquet: {offenders}"


def test_the_reader_module_depends_only_on_its_declared_layers() -> None:
    names = imported_modules(MODULE_PATH)

    assert names <= ALLOWED_IMPORTS, f"unexpected dependency: {sorted(names - ALLOWED_IMPORTS)}"
    assert {"pandas", "data_sys.storage", "factor_sys.universe"} <= names


def test_the_reader_module_borrows_only_the_snapshot_from_the_universe_module() -> None:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"), filename=str(MODULE_PATH))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "factor_sys.universe":
            imported.update(alias.name for alias in node.names)

    assert imported == {"UniverseSnapshot"}


def test_the_reader_api_is_exported_from_the_package_root() -> None:
    expected = {"MarketDataReadError", "MarketDataReader", "ParquetMarketDataReader"}

    assert expected <= set(factor_sys.__all__)
    assert factor_sys.MarketDataReader is MarketDataReader
    assert factor_sys.ParquetMarketDataReader is ParquetMarketDataReader
    assert factor_sys.MarketDataReadError is MarketDataReadError


def test_a_read_failure_is_a_factor_error_and_not_a_contract_violation() -> None:
    assert issubclass(MarketDataReadError, FactorError)
    assert not issubclass(MarketDataReadError, FactorContractError)
    assert not issubclass(MarketDataReadError, DataPipelineError)


def test_the_answer_can_be_handed_straight_to_a_factor_context(
    root: Path, reader: ParquetMarketDataReader
) -> None:
    """The reader's output is exactly the input the contract expects."""
    start, as_of = window(root, 10, 20)
    snapshot = snapshot_for(as_of)

    bars = reader.read_history(snapshot, start, ("close", "adjusted_close", "volume"))
    context = FactorContext(as_of_date=snapshot.as_of_date, universe=snapshot.symbols, bars=bars)

    assert list(context.bars.columns) == ["date", "symbol", "close", "adjusted_close", "volume"]
    assert len(context.bars) == len(bars)
    assert set(context.bars["symbol"]) == set(SYMBOLS)
