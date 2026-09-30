"""End-to-end tests for :mod:`data_sys.batch_query` -- a whole universe per run.

Every provider is an in-memory fake (``SessionFakeProvider``), so nothing here
touches the network or Yahoo, and the published tree is always read back with
**PyArrow** -- never DuckDB.  DuckDB caches file handles per path, and a batch
replaces ``symbol=*`` partitions while it runs, so pointing it at the published
tree is exactly the pattern that was observed to kill the interpreter on Windows
(see :mod:`data_sys.summary`).

The batch is the only place where "one broken ticker" is allowed to happen, so
these tests pin down both sides of that contract: the failing symbol is reported
without hiding the ones that worked, and a symbol that did publish is never
rolled back because a later symbol failed.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import date
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds
import pytest

from data_sys import batch_query
from data_sys.batch_query import run_batch
from data_sys.errors import DatasetSummaryError, DownloadError
from data_sys.market_calendar import DEFAULT_CALENDAR_NAME
from data_sys.merge import date_set
from data_sys.storage import read_symbol_partition
from data_sys.update import STATUS_FAILED, STATUS_SUCCESS
from tests.helpers import (
    SessionFakeProvider,
    make_provider_factory,
    session_dates,
    write_universe_file,
)

#: Symbols of the fake universe, in the exact order the CSV lists them.
SYMBOLS: list[str] = ["AAPL", "MSFT", "NVDA"]

#: Price columns that must survive an unrelated failure of a peer symbol.
PRICE_COLUMNS: tuple[str, ...] = (
    "open",
    "high",
    "low",
    "close",
    "adjusted_close",
    "volume",
)

#: A window straddling two years, so every symbol ends up with a ``year=2023``
#: *and* a ``year=2024`` partition.
MULTI_YEAR_START = "2023-01-03"
MULTI_YEAR_END = "2024-06-28"

#: A single-year window for the tests that only care about ordering / isolation.
SHORT_START = "2024-01-02"
SHORT_END = "2024-03-29"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _universe(tmp_path: Path) -> Path:
    """A universe CSV listing :data:`SYMBOLS` in order (inactive filler excluded)."""
    return write_universe_file(tmp_path / "config" / "universe_seed.csv", SYMBOLS)


def _bars_root(data_root: Path) -> Path:
    return data_root / "standardized" / "market_bars"


def _published(data_root: Path, symbol: str) -> pd.DataFrame:
    """The published frame of ``symbol`` (Hive partition keys included)."""
    return read_symbol_partition(_bars_root(data_root), symbol)


def _batch_record(data_root: Path, batch_run_id: str) -> dict:
    """The batch summary exactly as it was written to disk."""
    path = data_root / "metadata" / f"batch_{batch_run_id}.json"
    assert path.exists(), f"no batch record at {path}"
    return json.loads(path.read_text(encoding="utf-8"))


def _run_record(data_root: Path, run_id: str) -> dict:
    """The per-symbol record of ``run_id``, looked up by id, never by order."""
    path = data_root / "metadata" / f"run_{run_id}.json"
    assert path.exists(), f"no run record at {path}"
    return json.loads(path.read_text(encoding="utf-8"))


def _recording_factory(
    *,
    started: list[str],
    providers: dict[str, SessionFakeProvider],
    **kwargs: object,
) -> Callable[[str], SessionFakeProvider]:
    """A provider factory that records the order in which symbols were built.

    ``run_batch`` builds the provider of a symbol immediately before updating it,
    so the construction order *is* the processing order -- which is how the tests
    assert that the batch walks the universe sequentially, in file order.
    """
    factory = make_provider_factory(**kwargs)

    def build(symbol: str) -> SessionFakeProvider:
        provider = factory(symbol)
        started.append(symbol)
        providers[symbol] = provider
        return provider

    return build


def _summary_from_parquet(root: Path) -> list[dict[str, object]]:
    """``(symbol, year, n_rows, min_date, max_date)`` read from the files alone.

    Deliberately built with PyArrow only -- one fragment per ``symbol=``/``year=``
    directory -- so it can never agree with the batch record by sharing DuckDB
    with the code under test.
    """
    records: list[dict[str, object]] = []
    for symbol_dir in sorted(root.glob("symbol=*")):
        symbol = symbol_dir.name.partition("=")[2]
        for year_dir in sorted(symbol_dir.glob("year=*")):
            table = ds.dataset(year_dir, format="parquet").to_table()
            days = sorted(table.column("date").to_pylist())
            records.append(
                {
                    "symbol": symbol,
                    "year": int(year_dir.name.partition("=")[2]),
                    "n_rows": table.num_rows,
                    "min_date": days[0].isoformat(),
                    "max_date": days[-1].isoformat(),
                }
            )
    return records


# ---------------------------------------------------------------------------
# 1. every symbol succeeds, in universe file order
# ---------------------------------------------------------------------------
def test_a_batch_updates_every_symbol_in_universe_order(tmp_path: Path) -> None:
    universe = _universe(tmp_path)
    started: list[str] = []
    providers: dict[str, SessionFakeProvider] = {}

    exit_code, meta = run_batch(
        universe_path=universe,
        start=SHORT_START,
        end=SHORT_END,
        refresh_overlap_sessions=0,
        data_root=tmp_path,
        provider_factory=_recording_factory(
            started=started, providers=providers, start=SHORT_START, end=SHORT_END
        ),
        batch_run_id="batch-order",
    )

    expected = session_dates(SHORT_START, SHORT_END)

    # sequential execution, one symbol after the other, in file order
    assert exit_code == batch_query.EXIT_OK
    assert started == SYMBOLS
    assert meta.requested_symbols == SYMBOLS
    assert [status["symbol"] for status in meta.per_symbol_status] == SYMBOLS

    assert meta.total_symbols == len(SYMBOLS)
    assert meta.successful_symbols == len(SYMBOLS)
    assert meta.failed_symbols == 0
    assert meta.skipped_symbols == 0
    assert meta.failed == []
    assert meta.exit_code == batch_query.EXIT_OK

    for symbol in SYMBOLS:
        status = meta.per_symbol_status[SYMBOLS.index(symbol)]
        assert status["status"] == STATUS_SUCCESS
        assert status["published"] is True
        assert status["downloaded_row_count"] == len(expected)
        assert status["published_row_count"] == len(expected)

        published = _published(tmp_path, symbol)
        assert published["symbol"].unique().tolist() == [symbol]
        assert len(published) == len(expected)
        assert sorted(date_set(published)) == expected

        # a real, partitioned parquet dataset on disk, not just a frame in memory
        symbol_dir = _bars_root(tmp_path) / f"symbol={symbol}"
        assert (symbol_dir / "year=2024").is_dir()
        assert list(symbol_dir.rglob("*.parquet"))

        # exactly one request per symbol, for the whole (not yet existing) history
        assert providers[symbol].windows() == [(date(2024, 1, 2), date(2024, 3, 29))]

    # each symbol got its own series: no symbol was fed another symbol's rows
    first_closes = {float(_published(tmp_path, symbol)["close"].iloc[0]) for symbol in SYMBOLS}
    assert len(first_closes) == len(SYMBOLS)


# ---------------------------------------------------------------------------
# 2. one broken ticker is isolated -- and never rolls back the others
# ---------------------------------------------------------------------------
def test_a_failed_download_is_isolated_and_never_rolls_back_the_others(
    tmp_path: Path,
) -> None:
    """The exact batch contract: report the failure, keep every success.

    Run 1 publishes a baseline history for all three symbols; run 2 extends the
    window by a month while the middle symbol's download explodes.  The batch must
    exit non-zero, report MSFT as failed, still publish AAPL and NVDA -- and leave
    the already-published rows of AAPL, MSFT and NVDA exactly as they were.
    """
    universe = _universe(tmp_path)

    run_batch(
        universe_path=universe,
        start=SHORT_START,
        end="2024-02-29",
        refresh_overlap_sessions=0,
        data_root=tmp_path,
        provider_factory=make_provider_factory(start=SHORT_START, end="2024-02-29"),
        batch_run_id="batch-baseline",
    )
    baseline = {symbol: _published(tmp_path, symbol).copy() for symbol in SYMBOLS}
    assert all(len(frame) for frame in baseline.values())

    started: list[str] = []
    providers: dict[str, SessionFakeProvider] = {}
    exit_code, meta = run_batch(
        universe_path=universe,
        start=SHORT_START,
        end=SHORT_END,
        refresh_overlap_sessions=0,
        data_root=tmp_path,
        provider_factory=_recording_factory(
            started=started,
            providers=providers,
            start=SHORT_START,
            end=SHORT_END,
            errors={"MSFT": DownloadError("simulated vendor outage")},
        ),
        batch_run_id="batch-degraded",
    )

    # the batch failed as a whole ...
    assert exit_code == batch_query.EXIT_FAILED
    assert meta.exit_code == batch_query.EXIT_FAILED
    assert meta.failed == ["MSFT"]
    assert meta.total_symbols == len(SYMBOLS)
    assert meta.successful_symbols == len(SYMBOLS) - 1
    assert meta.failed_symbols == 1
    assert meta.skipped_symbols == 0

    # ... after continuing past the broken symbol (NVDA comes after MSFT)
    assert started == SYMBOLS

    # the degraded run only had to fetch the sessions past the baseline, and the
    # request stops at the last *session* of the window (2024-03-29 is Good Friday)
    tail = session_dates("2024-03-01", SHORT_END)
    assert providers["MSFT"].windows() == [(tail[0], tail[-1])]

    statuses = {status["symbol"]: status for status in meta.per_symbol_status}
    assert statuses["MSFT"]["status"] == STATUS_FAILED
    assert statuses["MSFT"]["published"] is False
    assert "simulated vendor outage" in statuses["MSFT"]["failure_reason"]
    for symbol in ("AAPL", "NVDA"):
        assert statuses[symbol]["status"] == STATUS_SUCCESS
        assert statuses[symbol]["published"] is True

    # the failed symbol keeps the version it already had: nothing partial, nothing lost
    assert sorted(date_set(_published(tmp_path, "MSFT"))) == session_dates(
        SHORT_START, "2024-02-29"
    )
    assert _published(tmp_path, "MSFT").equals(baseline["MSFT"])

    # the successful symbols were published *on top of* their old history -- the
    # failure of a peer cannot roll a partition that already committed back
    for symbol in ("AAPL", "NVDA"):
        after = _published(tmp_path, symbol)
        assert sorted(date_set(after)) == session_dates(SHORT_START, SHORT_END)
        assert statuses[symbol]["downloaded_row_count"] == len(tail)

        shared = baseline[symbol].merge(after, on="date", suffixes=("_before", "_after"))
        assert len(shared) == len(baseline[symbol])
        for column in PRICE_COLUMNS:
            assert shared[f"{column}_before"].tolist() == shared[f"{column}_after"].tolist()


# ---------------------------------------------------------------------------
# 3. the batch record describes every symbol and the whole dataset
# ---------------------------------------------------------------------------
def test_batch_metadata_reports_every_symbol_and_the_dataset_summary(
    tmp_path: Path,
) -> None:
    universe = _universe(tmp_path)

    exit_code, meta = run_batch(
        universe_path=universe,
        start=MULTI_YEAR_START,
        end=MULTI_YEAR_END,
        refresh_overlap_sessions=0,
        data_root=tmp_path,
        provider_factory=make_provider_factory(start=MULTI_YEAR_START, end=MULTI_YEAR_END),
        batch_run_id="batch-metadata",
    )

    expected = session_dates(MULTI_YEAR_START, MULTI_YEAR_END)

    assert exit_code == batch_query.EXIT_OK
    assert meta.batch_run_id == "batch-metadata"
    assert meta.universe_path == str(universe)
    assert meta.requested_symbols == SYMBOLS
    assert (meta.requested_start, meta.requested_end) == (MULTI_YEAR_START, MULTI_YEAR_END)
    assert meta.refresh_overlap_sessions == 0
    assert meta.calendar == DEFAULT_CALENDAR_NAME
    assert meta.started_at_utc and meta.finished_at_utc
    assert meta.started_at_utc <= meta.finished_at_utc
    assert meta.notes == []
    assert meta.data_source == "provider_factory"

    # -- one status block per symbol, in file order, with its own numbers --------
    for status in meta.per_symbol_status:
        symbol = status["symbol"]
        assert status["status"] == STATUS_SUCCESS
        assert status["published"] is True
        assert status["run_id"] == f"batch-metadata-{symbol}"
        assert status["batch_run_id"] == "batch-metadata"
        assert status["existing_row_count"] == 0
        assert status["downloaded_row_count"] == len(expected)
        assert status["published_row_count"] == len(expected)
        assert status["new_dates_count"] == len(expected)
        assert status["replaced_dates_count"] == 0
        assert status["min_date"] == expected[0].isoformat()
        assert status["max_date"] == expected[-1].isoformat()
        assert [window["start"] for window in status["windows"]] == [MULTI_YEAR_START]
        assert [window["end"] for window in status["windows"]] == [MULTI_YEAR_END]
        assert any(raw.endswith(".parquet") for raw in status["raw_paths"])

    # -- the dataset summary: one row per (symbol, year) -----------------------
    summary = meta.dataset_summary
    assert [(record["symbol"], record["year"]) for record in summary] == [
        (symbol, year) for symbol in SYMBOLS for year in (2023, 2024)
    ]
    for record in summary:
        year_rows = _published(tmp_path, record["symbol"])
        year_rows = year_rows[year_rows["year"] == record["year"]]
        days = sorted(date_set(year_rows))
        assert record == {
            "symbol": record["symbol"],
            "year": record["year"],
            "n_rows": len(year_rows),
            "min_date": days[0].isoformat(),
            "max_date": days[-1].isoformat(),
        }
    assert sum(record["n_rows"] for record in summary) == len(SYMBOLS) * len(expected)

    # -- the file on disk is the record, and it ties the per-symbol runs together
    written = _batch_record(tmp_path, "batch-metadata")
    assert written == json.loads(json.dumps(meta.to_dict()))
    assert written["batch_run_id"] == "batch-metadata"
    assert written["requested_symbols"] == SYMBOLS
    assert written["per_symbol_status"] == meta.per_symbol_status
    assert written["dataset_summary"] == summary
    assert written["total_symbols"] == len(SYMBOLS)
    assert written["exit_code"] == batch_query.EXIT_OK

    for symbol in SYMBOLS:
        run = _run_record(tmp_path, f"batch-metadata-{symbol}")
        assert run["batch_run_id"] == "batch-metadata"
        assert run["symbol"] == symbol
        assert run["status"] == STATUS_SUCCESS


# ---------------------------------------------------------------------------
# 4. the summary covers several symbols *and* several year partitions
# ---------------------------------------------------------------------------
def test_the_dataset_summary_spans_symbols_and_years_with_pyarrow_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A batch of three symbols over two calendar years, summarized with PyArrow.

    The expectation is rebuilt here from the parquet files themselves (one
    fragment per ``symbol=``/``year=`` directory), so the record can only agree
    with the code under test if both read the same physical layout -- DuckDB is
    never involved on either side.
    """
    universe = _universe(tmp_path)
    roots: list[Path] = []
    real_summarize = batch_query.summarize_dataset

    def spy(root: Path | str) -> pd.DataFrame:
        roots.append(Path(root))
        return real_summarize(root)

    monkeypatch.setattr(batch_query, "summarize_dataset", spy)

    exit_code, meta = run_batch(
        universe_path=universe,
        start=MULTI_YEAR_START,
        end=MULTI_YEAR_END,
        refresh_overlap_sessions=0,
        data_root=tmp_path,
        provider_factory=make_provider_factory(start=MULTI_YEAR_START, end=MULTI_YEAR_END),
        batch_run_id="batch-summary",
    )

    assert exit_code == batch_query.EXIT_OK
    # the summary is produced by the published-tree reader, once, on that tree
    assert roots == [_bars_root(tmp_path)]

    for symbol in SYMBOLS:
        for year in (2023, 2024):
            assert (_bars_root(tmp_path) / f"symbol={symbol}" / f"year={year}").is_dir()

    expected = _summary_from_parquet(_bars_root(tmp_path))
    assert [(record["symbol"], record["year"]) for record in expected] == [
        (symbol, year) for symbol in SYMBOLS for year in (2023, 2024)
    ]
    assert meta.dataset_summary == expected
    assert len(meta.dataset_summary) == len(SYMBOLS) * 2
    assert all(record["n_rows"] > 0 for record in meta.dataset_summary)

    # DuckDB only ever reads a freshly written staging path, so it leaves no
    # database file behind anywhere -- least of all next to the published tree
    assert not list(tmp_path.rglob("*.duckdb"))


# ---------------------------------------------------------------------------
# 5. a summary failure never costs the batch record
# ---------------------------------------------------------------------------
def test_a_summary_failure_still_records_the_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``_attach_dataset_summary`` is best-effort: the run is still a success."""
    universe = _universe(tmp_path)

    def explode(root: Path | str) -> pd.DataFrame:
        raise DatasetSummaryError("the reader is unavailable")

    monkeypatch.setattr(batch_query, "summarize_dataset", explode)

    exit_code, meta = run_batch(
        universe_path=universe,
        start=SHORT_START,
        end=SHORT_END,
        refresh_overlap_sessions=0,
        data_root=tmp_path,
        provider_factory=make_provider_factory(start=SHORT_START, end=SHORT_END),
        batch_run_id="batch-no-summary",
    )

    assert exit_code == batch_query.EXIT_OK
    assert meta.successful_symbols == len(SYMBOLS)
    assert meta.dataset_summary == []
    assert [note.split(":")[0] for note in meta.notes] == ["dataset summary unavailable"]

    written = _batch_record(tmp_path, "batch-no-summary")
    assert written["dataset_summary"] == []
    assert written["notes"] == meta.notes
    assert written["exit_code"] == batch_query.EXIT_OK
