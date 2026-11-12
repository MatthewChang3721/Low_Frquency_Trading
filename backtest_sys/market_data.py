"""Read-only market bars for a replay: the prices a fill and a valuation cite.

The question this module answers
--------------------------------
::

    reader = ParquetBacktestMarketDataReader()

    bars = reader.read_bars(
        symbols=("AAPL", "MSFT"),
        start_date="2024-01-01",
        end_date="2024-01-31",
        required_fields=("open", "close", "adjusted_close"),
    )

    list(bars.columns)   # ['date', 'symbol', 'open', 'close', 'adjusted_close']

*What did these symbols trade at, over this window, in these fields?*  Nothing
else.  The reader selects rows and columns; it prices nothing, calculates nothing
and decides nothing about eligibility.

Why the backtest layer has a reader of its own
----------------------------------------------
:mod:`factor_sys.market_data` is the factor system's entry point and it answers a
*different* question: it is handed a
:class:`~factor_sys.universe.UniverseSnapshot` and returns only the bars of the
symbols that snapshot names.  That is exactly right for a factor, and exactly
wrong for a replay:

* a replay must value and sell what it **already owns**, and a holding outlives
  its membership in any snapshot -- a strategy drops a name, a universe refresh
  retires it, a symbol is delisted after the last session the account held it;
* a replay must read the bars of a symbol that is **not** in today's universe at
  all, because yesterday's fill is still in the ledger;
* a replay must read an explicit ``[start_date, end_date]`` window, not "up to the
  snapshot's point in time", because the window is the run's configuration.

So this module asks :mod:`data_sys.storage` for the published bars and filters
them itself.  It never imports the factor layer, and no layer above it may assume
the account's holdings are a subset of any snapshot.

The price language
------------------
Three columns, three jobs.  They are **not** interchangeable, and this reader
hands each of them through exactly as it was stored -- it never derives one from
another, never rescales one into another and never lets one stand in for another:

``open``
    真实 open -- the *real, unadjusted* open of the session, the price the
    standardized dataset recorded for the first trade of the day.  This is the
    candidate price the **execution layer** fills at in the timeline this project
    follows (decide at a close, fill at the next session's open).
``close``
    真实 close -- the *real, unadjusted* close of the session.  This is the
    candidate price the **ledger layer** uses to value the account at that
    session's close.  Cash and market value are stated in money that actually
    changed hands, so the number a valuation cites has to be the traded one.
``adjusted_close``
    研究口径 -- research, factors and return labels only.  It is a *back-adjusted*
    series, so it is not a price anybody could have traded and not a cash amount
    anybody could have booked: it must never be substituted for ``open`` as a fill
    price, nor for ``close`` as a valuation price, even though it is the right
    series for a factor window (which is why the factor layer reads it).

``high``, ``low`` and ``volume`` are readable for the same reason -- a later
slippage, liquidity or participation model will want them -- and are equally
untouched.

No corporate actions: not now, not faked
---------------------------------------
The published dataset is a daily-bar table: ``date``, ``symbol``, the six bar
fields, and the two derived columns ``daily_return`` and ``dollar_volume``.  There
is **no** dividend table, **no** split table, **no** delisting table and no
per-symbol corporate-action event stream anywhere in this project.

This module therefore does not touch a holding.  It does not infer a split, a cash
dividend or a delisting from ``adjusted_close``, and it does not adjust, restate or
rebase a bar, because a back-adjusted series cannot say *when* a cash flow
happened, how much it was, or in what ratio shares changed -- a time series of
prices is simply not that record.  Faking a dividend out of an adjustment factor
would invent cash that never arrived and a share count that never changed.  Real
corporate-action handling needs a real event source, and it belongs to a later
task, before any ledger books a split or a dividend.

The read contract
-----------------
A request is ``(symbols, start_date, end_date, required_fields)`` and the answer is
always a **long** frame with exactly these columns, in this order::

    ["date", "symbol", *required_fields]

* ``symbols`` is a ``tuple`` of tickers -- an empty tuple is legal and asks for
  nothing.  Tickers are taken exactly as given: no trimming, no upper-casing, no
  repair.  Each must match :data:`data_sys.schema.SYMBOL_PATTERN`, which is the
  project's single ticker rule, and must not repeat.  ``"CASH"`` is refused by name:
  cash is an account field, not a security with a bar to read.  Nothing is required
  of the current universe: a symbol with no partition is simply a symbol with no row.
* ``start_date`` / ``end_date`` are plain :class:`datetime.date` values or strict
  ISO ``"YYYY-MM-DD"`` strings, and the window is ``[start_date, end_date]`` with
  **both ends included**.  ``end_date`` is a hard upper bound: no row dated after it
  is ever returned, which is what keeps a replay from seeing tomorrow.  A
  ``datetime``, a ``Timestamp``, a string carrying a time or a looser spelling
  (``"2024-1-5"``, ``"01/05/2024"``, ``"20240105"``) is refused rather than rounded
  to a day: the caller, not the reader, decides where a window starts.
* ``required_fields`` is a non-empty iterable of names from :data:`BAR_FIELDS` --
  ``open``, ``high``, ``low``, ``close``, ``adjusted_close``, ``volume`` -- and the
  caller's order is kept.  ``date`` and ``symbol`` are the frame's labels and are
  always returned, so requesting them is an error instead of a no-op.
  ``daily_return`` and ``dollar_volume`` are not bar fields: a backtest prices and
  values with what traded, so the reader refuses to hand over a derived series
  that only looks like a price.
* Rows come back sorted by ``(symbol, date)`` with a stable sort, on a
  ``RangeIndex``, as a fresh frame: mutating the result cannot change what the next
  read returns.  The Hive partition key ``year`` never appears.

An empty answer is still an answer
----------------------------------
A symbol with no partition, a window that matches nothing, a history that starts
inside the window and a gap in the middle of a history are **not** errors: the read
returns the rows it could find, with the requested columns, and nothing more.
Whether a symbol may be traded on a session, or valued at a close, is the execution
and ledger layers' decision -- a reader that refused a gap would be making that
decision for them.  Only a request this layer refuses, or a dataset that cannot be
read honestly, raises :class:`~backtest_sys.errors.BacktestMarketDataError`.

Two rows of the same bar are one too many: if the published data holds two bars for
the same ``(symbol, date)`` among the requested symbols, the read fails.  It does
not pick the first, the last or the higher price -- a silent choice is exactly the
kind of unauditable answer a replay must not produce.

Layering
--------
The Hive layout (``symbol=<TICKER>/year=<YYYY>/*.parquet``) has **one**
implementation in this project -- :func:`data_sys.storage.read_partitioned` -- and
this module calls it instead of walking directories, opening parquet files or
naming partitions.  Nothing here writes: no directory is created, no file is
touched, nothing is fetched from a network and DuckDB is never involved.  The only
project modules this one knows are :mod:`data_sys.config` (the default root),
:mod:`data_sys.schema` (the ticker pattern), :mod:`data_sys.storage` (the reader)
and :mod:`backtest_sys.errors` (the error); the factor, strategy and portfolio
layers are not imported, and neither is the data pipeline's writer.

Examples
--------
Building a reader touches nothing on disk; the dataset is only checked when a read
actually needs it::

>>> reader = ParquetBacktestMarketDataReader("some/dir/market_bars")
>>> reader.root.name
'market_bars'
>>> reader.root.is_absolute()
False
>>> BAR_FIELDS[:3]
('open', 'high', 'low')
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import cast

import pandas as pd

from backtest_sys.errors import BacktestMarketDataError
from data_sys.config import PipelineConfig
from data_sys.schema import SYMBOL_PATTERN
from data_sys.storage import read_partitioned

# ---------------------------------------------------------------------------
# The vocabulary of a read
# ---------------------------------------------------------------------------
#: The frame's two labels: always present, never requested, always first.
DATE_COLUMN: str = "date"
SYMBOL_COLUMN: str = "symbol"

#: The bar fields a backtest read may request, in the dataset's own column order.
#:
#: Deliberately *not* :data:`data_sys.schema.STANDARD_COLUMNS`: ``daily_return`` and
#: ``dollar_volume`` are derived series, and a derived series is not a price a fill
#: or a valuation may cite.  A reader that handed one out would invite exactly that
#: confusion, so it refuses to.
BAR_FIELDS: tuple[str, ...] = (
    "open",
    "high",
    "low",
    "close",
    "adjusted_close",
    "volume",
)

#: The one date spelling this layer accepts as a string.
_ISO_DAY: re.Pattern[str] = re.compile(r"^\d{4}-\d{2}-\d{2}$")

#: The project's single ticker rule, compiled once from :mod:`data_sys.schema`.
_SYMBOL: re.Pattern[str] = re.compile(SYMBOL_PATTERN)

#: The name that is an account field, never a ticker: it has no bar to read.
_CASH_SYMBOL: str = "CASH"


# ---------------------------------------------------------------------------
# One validated read
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _ReadRequest:
    """A read request that has passed every check, frozen before any I/O.

    Freezing matters: the window, the symbol set and the column list are decided
    once, so the frame that comes back cannot quietly disagree with the request
    that produced it.
    """

    start: date
    end: date
    symbols: tuple[str, ...]
    fields: tuple[str, ...]

    @property
    def columns(self) -> list[str]:
        """The frame header this request must produce, in order."""
        return [DATE_COLUMN, SYMBOL_COLUMN, *self.fields]


# ---------------------------------------------------------------------------
# Request validation: everything below runs before a byte is read
# ---------------------------------------------------------------------------
def _symbols_or_problems(symbols: object) -> tuple[tuple[str, ...], list[str]]:
    """The requested tickers, plus every reason the symbol set is not a set.

    A ``tuple`` is required rather than merely iterated: ``"AAPL"`` is a string of
    four characters and a ``list`` is a container some caller is still free to
    mutate, so accepting either would mean guessing what was meant.  Order is kept
    as given, because a caller may read the frame block by block, and a repeat is
    refused rather than folded away: two mentions of ``"AAPL"`` are one symbol and
    also a request that cannot be answered unambiguously.
    """
    if not isinstance(symbols, tuple):
        return (), [
            f"symbols must be a tuple of tickers, got {type(symbols).__name__} {symbols!r}: "
            "a bare string is a sequence of characters, and a mutable container is a symbol "
            "set somebody can still change"
        ]

    problems: list[str] = []
    seen: set[str] = set()
    for value in symbols:
        if not isinstance(value, str):
            problems.append(f"invalid symbol {value!r}: a ticker must be a string")
        elif not _SYMBOL.fullmatch(value):
            problems.append(
                f"invalid symbol {value!r}: a ticker must match {SYMBOL_PATTERN} exactly -- "
                "upper case, no surrounding space, no repair"
            )
        elif value == _CASH_SYMBOL:
            problems.append(
                "symbol 'CASH' is not a security: cash is an account field with no bar of its "
                "own, so a market-data read can never price or value it"
            )
        elif value in seen:
            problems.append(f"duplicate symbol {value!r}: one read names each ticker once")
        else:
            seen.add(value)
    return tuple(symbols), problems


def _day_or_problem(value: object, label: str) -> tuple[date | None, str | None]:
    """``value`` as a plain calendar day, or the reason it is not one.

    A ``datetime`` is refused even though it *is* a ``date``: a daily bar has no use
    for a time of day, and silently truncating one would mean the reader, not the
    caller, decided where the window starts.  A string must be strict ISO
    ``"YYYY-MM-DD"`` and must name a real day, so ``"2024-02-30"`` fails here instead
    of becoming a ``NaT`` three layers later.
    """
    if isinstance(value, datetime):
        return None, (
            f"{label} must be a date or a strict ISO 'YYYY-MM-DD' string, got the datetime "
            f"{value!r}: a daily read has no use for a time of day, pass {label}.date() to "
            "name the session"
        )
    if isinstance(value, date):
        return value, None
    if isinstance(value, str):
        text = value.strip()
        if _ISO_DAY.fullmatch(text):
            try:
                return date.fromisoformat(text), None
            except ValueError:
                pass  # right shape, but no such day (e.g. '2024-02-30')
    return None, f"{label} must be a date or a strict ISO 'YYYY-MM-DD' string, got {value!r}"


def _fields_or_problems(required_fields: object) -> tuple[tuple[str, ...], list[str]]:
    """The requested bar fields, in the caller's order, plus every problem.

    Collecting instead of stopping at the first bad name is the project's style --
    one failed call reports every reason, so a caller fixes the list once.  The
    order is preserved because a caller may unpack the frame by position, and
    ``date`` / ``symbol`` are refused rather than ignored: they are always there, so
    asking for them is a misunderstanding worth naming.
    """
    if required_fields is None or isinstance(required_fields, str):
        return (), [
            "required_fields must be a non-empty iterable of bar field names (not a bare "
            f"string), got {required_fields!r}"
        ]
    try:
        candidates = list(required_fields)  # type: ignore[call-overload]
    except TypeError:
        return (), [
            "required_fields must be an iterable of bar field names, got "
            f"{type(required_fields).__name__}"
        ]

    fields: list[str] = []
    problems: list[str] = []
    if not candidates:
        problems.append(f"required_fields must not be empty: request at least one of {BAR_FIELDS}")
    for value in candidates:
        if not isinstance(value, str) or not value.strip():
            problems.append(f"invalid field name {value!r}: expected a non-empty string")
        elif value in (DATE_COLUMN, SYMBOL_COLUMN):
            problems.append(
                f"field {value!r} must not be requested: {DATE_COLUMN!r} and "
                f"{SYMBOL_COLUMN!r} are the frame's labels and are always returned"
            )
        elif value in fields:
            problems.append(f"duplicate field {value!r}")
        elif value not in BAR_FIELDS:
            problems.append(
                f"unknown field {value!r}: a backtest read provides {BAR_FIELDS} and nothing "
                "derived"
            )
        else:
            fields.append(value)
    return tuple(fields), problems


def _prepare_request(
    symbols: object,
    start_date: object,
    end_date: object,
    required_fields: object,
) -> _ReadRequest:
    """Validate one whole request and freeze it, before any dataset is touched.

    Every part is checked and every finding is collected, so an illegal request
    fails the same way -- same error type, all of its bullets -- whether or not a
    dataset has been published yet.  A request that gets past this function is
    answerable by selection alone; nothing that follows can turn a legal request
    into an illegal one.
    """
    problems: list[str] = []

    requested_symbols, symbol_problems = _symbols_or_problems(symbols)
    problems.extend(symbol_problems)

    start, start_problem = _day_or_problem(start_date, "start_date")
    if start_problem is not None:
        problems.append(start_problem)
    end, end_problem = _day_or_problem(end_date, "end_date")
    if end_problem is not None:
        problems.append(end_problem)
    if start is not None and end is not None and start > end:
        problems.append(
            f"start_date {start.isoformat()} is after end_date {end.isoformat()}: the window "
            "[start_date, end_date] would be empty, so no session could be read"
        )

    fields, field_problems = _fields_or_problems(required_fields)
    problems.extend(field_problems)

    if problems:
        raise BacktestMarketDataError("invalid read request", problems)

    # The raises above guarantee both days parsed, so the casts cannot lie.
    return _ReadRequest(
        start=cast(date, start),
        end=cast(date, end),
        symbols=requested_symbols,
        fields=fields,
    )


# ---------------------------------------------------------------------------
# Applying a request to a frame of published bars (shared by every reader)
# ---------------------------------------------------------------------------
def _empty_frame(request: _ReadRequest) -> pd.DataFrame:
    """The header ``request`` must produce, with no row at all.

    Used when the request names no ticker: the answer is knowable without reading
    anything, so no dataset is touched -- a reader that insisted on a published
    directory before saying "you asked for no symbol" would be refusing a question
    it can answer.  The price columns are typed as floats and ``volume`` as an
    integer so the empty answer matches the dtypes a real read returns.
    """
    frame = pd.DataFrame(
        {
            DATE_COLUMN: pd.Series(dtype="datetime64[s]"),
            SYMBOL_COLUMN: pd.Series(dtype="object"),
        }
    )
    for field in request.fields:
        frame[field] = pd.Series(dtype="int64" if field == "volume" else "float64")
    return frame


def _bars_frame(frame: pd.DataFrame, request: _ReadRequest) -> pd.DataFrame:
    """Cut ``frame`` down to ``request``: symbols, window, columns, order.

    This is the half of a read that must never differ between sources: a reader
    only has to hand over the bars it stores -- a long frame holding at least
    ``date``, ``symbol`` and the requested fields -- and everything the backtest
    layer relies on (both ends of the window included, the hard ``end_date`` bound,
    the column order, the stable ``(symbol, date)`` sort and a result that is a
    fresh object) is applied here, once, for everybody.

    Values are never rewritten: rows are selected, columns are selected and the
    result is re-indexed, so what the dataset stored is what the caller receives.
    An empty selection is a normal answer, not a failure.

    Raises
    ------
    BacktestMarketDataError
        If ``frame`` is not a DataFrame, has no columns, does not provide the
        ``date``/``symbol`` labels or a requested field, cannot have its dates read,
        or holds two bars for one ``(symbol, date)`` among the requested symbols --
        each of which means the source cannot answer the request honestly, which is
        different from answering it with no rows.
    """
    if not isinstance(frame, pd.DataFrame):
        raise BacktestMarketDataError(
            f"a market data source must return a pandas DataFrame, got {type(frame).__name__}"
        )

    columns = request.columns
    if not len(frame.columns):
        raise BacktestMarketDataError(
            "the market data source returned a frame with no columns at all; a source must "
            f"expose at least {columns}, even when it holds no bar"
        )

    provided = [str(column) for column in frame.columns]
    missing_labels = [
        column for column in (DATE_COLUMN, SYMBOL_COLUMN) if column not in frame.columns
    ]
    if missing_labels:
        raise BacktestMarketDataError(
            f"the dataset does not provide the label column(s) {missing_labels}: every bar "
            f"needs both {DATE_COLUMN!r} and {SYMBOL_COLUMN!r} to be placed on the timeline; "
            f"it provides {provided}"
        )
    missing_fields = [field for field in request.fields if field not in frame.columns]
    if missing_fields:
        raise BacktestMarketDataError(
            f"the dataset does not provide the requested field(s) {missing_fields}; it "
            f"provides {provided}"
        )

    try:
        dates = pd.to_datetime(frame[DATE_COLUMN], errors="coerce")
    except (TypeError, ValueError) as exc:
        raise BacktestMarketDataError(
            f"the {DATE_COLUMN!r} column cannot be read as dates: {exc}"
        ) from exc
    unreadable = dates.isna().to_numpy()
    if unreadable.any():
        raise BacktestMarketDataError(
            f"the {DATE_COLUMN!r} column has {int(unreadable.sum())} missing or unparseable "
            "value(s): the read window cannot be decided for those rows"
        )

    # A requested symbol that published two bars for one session is a
    # contradiction in the dataset, not in the request: refuse it, and refuse it
    # wherever the duplicate sits, so no read ever has to choose a price.
    in_symbols = frame[SYMBOL_COLUMN].isin(request.symbols).to_numpy()
    pairs = pd.DataFrame(
        {
            SYMBOL_COLUMN: frame.loc[in_symbols, SYMBOL_COLUMN].to_numpy(),
            DATE_COLUMN: dates.loc[in_symbols].to_numpy(),
        }
    )
    repeated = pairs.duplicated(keep="first").to_numpy()
    if repeated.any():
        clashes = sorted(
            {
                f"{row.symbol} {pd.Timestamp(row.date).date().isoformat()}"
                for row in pairs.loc[repeated].itertuples(index=False)
            }
        )
        raise BacktestMarketDataError(
            "the dataset holds more than one bar for the same (symbol, date) among the "
            f"requested symbols, e.g. {clashes[:5]}: a read refuses to choose one of two "
            "prices for one session"
        )

    # Positional boolean masks: they line up with the frame by construction, no
    # matter what index the source used, and a row may only survive both.
    in_window = (
        (dates >= pd.Timestamp(request.start)) & (dates <= pd.Timestamp(request.end))
    ).to_numpy()

    selected = frame.loc[in_symbols & in_window, columns]
    ordered = selected.sort_values([SYMBOL_COLUMN, DATE_COLUMN], kind="stable")
    return ordered.reset_index(drop=True)


# ---------------------------------------------------------------------------
# The interface every backtest market data source implements
# ---------------------------------------------------------------------------
class BacktestMarketDataReader(ABC):
    """The stable read-only market-data interface of the backtest layer.

    An implementation answers one request with a long frame
    ``["date", "symbol", *required_fields]`` covering ``[start_date, end_date]``
    (both ends included, nothing dated later), sorted by ``(symbol, date)``, holding
    the requested fields in the order the caller declared them.  It is handed the
    symbols it must answer for -- not a universe, not a snapshot -- so a reader can
    always serve a holding that has left every index the strategy looks at.

    Why this is an interface rather than a function
    ----------------------------------------------
    A replay should be replayable against other bars: a vendor drop, a database, a
    test fixture that fails on purpose.  Such a source implements this one method
    and nothing else changes -- in particular nothing above it may assume the
    published Hive layout.  A source that happens to hold its bars in a frame
    should build its answer with :func:`_prepare_request` and :func:`_bars_frame`,
    which is how the window, the column order, the duplicate refusal and the sort
    stay identical across sources; a source that does not is free to produce the
    frame itself, as long as it keeps the contract.

    What an implementation must **not** do
    --------------------------------------
    Fill, interpolate, adjust, aggregate, de-duplicate, forward-fill, re-index
    prices or choose between two bars for one session.  A reader selects; every
    judgement about eligibility, liquidity and corporate actions belongs to a
    later layer with the data to make it.
    """

    @abstractmethod
    def read_bars(
        self,
        symbols: tuple[str, ...],
        start_date: date | str,
        end_date: date | str,
        required_fields: Iterable[str],
    ) -> pd.DataFrame:
        """Read the stored bars of ``symbols`` over the inclusive window.

        Parameters
        ----------
        symbols:
            The tickers to read, taken exactly as given -- no trimming, no
            upper-casing, no repair.  Each must match
            :data:`data_sys.schema.SYMBOL_PATTERN` and must not repeat; an empty
            tuple is legal and asks for nothing, which is answered with the header
            and no row.  The symbols need not belong to any universe: the account's
            holdings are the requirement, not an index membership.
        start_date:
            First session of the window, inclusive; a :class:`datetime.date` or a
            strict ISO ``"YYYY-MM-DD"`` string (see :func:`_day_or_problem`).
        end_date:
            Last session of the window, inclusive, and a hard upper bound: no row
            dated after it is ever returned.  ``start_date <= end_date``.
        required_fields:
            The bar fields to return, e.g. ``("open", "close")``.  Each must be a
            member of :data:`BAR_FIELDS`; ``"date"`` and ``"symbol"`` may not be
            requested, as they are always returned, and derived columns
            (``daily_return``, ``dollar_volume``) are refused outright.

        Returns
        -------
        pandas.DataFrame
            The requested rows with columns ``["date", "symbol", *required_fields]``
            in the caller's field order, sorted by ``(symbol, date)`` on a
            ``RangeIndex``.  Empty of rows when nothing matches, but never empty of
            columns, and a fresh object every time.

        Raises
        ------
        backtest_sys.errors.BacktestMarketDataError
            If the request cannot be honoured, or the source cannot answer it
            honestly (missing field, unreadable dates, two bars for one session).
        """


class ParquetBacktestMarketDataReader(BacktestMarketDataReader):
    """The v1 reader: the daily bars the data pipeline has published.

    The dataset is read through :func:`data_sys.storage.read_partitioned`, so the
    Hive layout (``symbol=<TICKER>/year=<YYYY>/*.parquet``) keeps exactly one
    implementation in the project and this reader selects from the returned frame
    instead of walking directories.  Values, rows and dtypes are handed through as
    stored.

    Parameters
    ----------
    market_bars_root:
        Root of the dataset.  Defaults to
        :attr:`data_sys.config.PipelineConfig.standardized_dir`, the project's
        ``data/standardized/market_bars``; tests pass a temporary directory.

    Notes
    -----
    * Building a reader touches nothing on disk, because the pipeline may publish
      the dataset after the reader was created.
    * A root that does not exist, is not a directory or holds no ``*.parquet`` file
      is refused when :meth:`read_bars` is called -- and the request is validated
      first, so an illegal request fails identically whether or not data exists.
    * A request that names no symbol is answered with the header alone, without
      touching the dataset: "read nothing" needs no bar to read it from.
    * Only the ``*.parquet`` existence check and the read itself look at the disk;
      neither writes, and no network or DuckDB is involved.
    """

    def __init__(self, market_bars_root: Path | str | None = None) -> None:
        self._root = _market_bars_root(market_bars_root)

    @property
    def root(self) -> Path:
        """The dataset root this reader reads from."""
        return self._root

    def read_bars(
        self,
        symbols: tuple[str, ...],
        start_date: date | str,
        end_date: date | str,
        required_fields: Iterable[str],
    ) -> pd.DataFrame:
        """Read the requested window out of the published dataset.

        The request is validated before the dataset is touched, so an impossible
        request fails the same way whether or not anything has been published yet.
        """
        request = _prepare_request(symbols, start_date, end_date, required_fields)
        if not request.symbols:
            return _empty_frame(request)
        return _bars_frame(self._read_dataset(), request)

    def _read_dataset(self) -> pd.DataFrame:
        """Every published bar, exactly as the storage layer returns it."""
        root = self._root
        if not root.exists():
            raise BacktestMarketDataError(
                f"market bars root {str(root)!r} does not exist: there is no dataset to read"
            )
        if not root.is_dir():
            raise BacktestMarketDataError(
                f"market bars root {str(root)!r} is not a directory, so it cannot hold the "
                "parquet dataset"
            )
        if not any(root.rglob("*.parquet")):
            raise BacktestMarketDataError(
                f"market bars root {str(root)!r} holds no '*.parquet' file: the standardized "
                "dataset is empty, so there is no bar to read"
            )
        try:
            return read_partitioned(root)
        except Exception as exc:  # noqa: BLE001 - a storage failure must not leak as pyarrow
            raise BacktestMarketDataError(
                f"the parquet dataset at {str(root)!r} could not be read: {exc}"
            ) from exc

    def __repr__(self) -> str:
        return f"{type(self).__name__}(market_bars_root={str(self._root)!r})"


def _market_bars_root(value: Path | str | None) -> Path:
    """The dataset root of a reader: the given path, or the pipeline's default."""
    if value is None:
        return PipelineConfig().standardized_dir
    if isinstance(value, str):
        if not value.strip():
            raise BacktestMarketDataError(
                "market_bars_root must not be blank: pass a directory, or omit it to read the "
                "published dataset"
            )
        return Path(value)
    if isinstance(value, Path):
        return value
    raise BacktestMarketDataError(
        f"market_bars_root must be a path or a string, got {type(value).__name__}"
    )


__all__ = [
    "BAR_FIELDS",
    "BacktestMarketDataReader",
    "ParquetBacktestMarketDataReader",
]
