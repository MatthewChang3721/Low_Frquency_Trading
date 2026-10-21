"""Local market bars for the factor system: one window, one shape, no look-ahead.

What this module is
-------------------
``factor_sys.market_data`` is the factor system's only market-data entry point.
It turns a universe snapshot and a start date into the long frame a factor is
computed on:

``MarketDataReader``
    the interface every source implements:
    ``read_history(snapshot, start_date, required_fields) -> DataFrame``
``ParquetMarketDataReader``
    the v1 implementation: the published Hive dataset under
    ``data/standardized/market_bars``, read through
    :func:`data_sys.storage.read_partitioned`
``MarketDataReadError``
    raised when a request cannot be honoured, or the local dataset cannot
    answer it

The read contract
-----------------
A request is ``(snapshot, start_date, required_fields)`` and the answer is
always a **long** frame with exactly these columns, in this order::

    ["date", "symbol", *required_fields]

* ``date`` and ``symbol`` come first; the requested fields follow in the order
  the caller declared them.  Nothing else is ever returned -- the Hive
  partition key ``year`` stays out of the frame.
* The window is ``[start_date, snapshot.as_of_date]``, **both ends included**.
* ``snapshot.as_of_date`` is a hard upper bound: no row dated after it is ever
  returned.  A factor may only use what was public when ``as_of_date`` closed,
  so the point-in-time guard is applied here as well as in
  :class:`factor_sys.contracts.FactorContext` -- a leak fails at the read, not
  only when the context is built.
* Rows are sorted by ``(symbol, date)`` with a stable sort, so the same request
  always answers in the same order.
* The frame returned is a fresh object: mutating it changes neither the reader
  nor what the next read returns.

The reader selects; it never rewrites
-------------------------------------
Values are handed through untouched, exactly as the storage layer stored them:
nothing is filled, interpolated, adjusted or de-duplicated.  A symbol with no
partition at all, a symbol whose history starts inside the window and a window
that matches nothing are **not** errors -- they are answered with a frame that
has the right columns and no rows.  Deciding whether a symbol is *eligible*
(enough history, no gaps) is the factor's own business -- ``is_eligible``
belongs to :class:`factor_sys.contracts.FactorPanel` -- so a reader never
guesses.

Dates are strict here
---------------------
``start_date`` is a :class:`datetime.date` or a strict ISO ``"YYYY-MM-DD"``
string.  A ``datetime`` is refused even though it *is* a ``date``, and so is any
looser spelling (``"2024-01-05T15:00"``, ``"2024-1-5"``, ``"05/01/2024"``,
``"20240105"``): quietly rounding a timestamp to a day would decide the read
window on the caller's behalf.  Pass ``value.date()`` when the day is what you
mean.  ``UniverseSnapshot`` stays the one place where a ``datetime`` is
accepted, because a snapshot is a label; a window is not.

Layering
--------
This module depends on the storage layer's reader and never on its writer: it
creates no directory, deletes nothing, persists nothing, makes no network call
and never involves DuckDB.  A future source -- another vendor's files, a
database, an in-memory fixture in a test -- implements
:meth:`MarketDataReader.read_history` and is otherwise free; reusing
:func:`_prepare_request` and :func:`_history_frame` is what keeps the contract
above identical across sources.  A reader only ever sees a
:class:`~factor_sys.universe.UniverseSnapshot`, never
:class:`~factor_sys.universe.StaticUniverseProvider`, so replacing either one
leaves the other untouched.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Iterable
from datetime import date, datetime
from pathlib import Path
from typing import Any, NamedTuple

import pandas as pd

from data_sys.config import PipelineConfig
from data_sys.schema import STANDARD_COLUMNS
from data_sys.storage import DATE_COLUMN, SYMBOL_COLUMN, read_partitioned
from factor_sys.errors import MarketDataReadError
from factor_sys.universe import UniverseSnapshot

#: The only string shape a read window accepts: a strict ISO calendar day.
_ISO_DAY = re.compile(r"\d{4}-\d{2}-\d{2}")


class _ReadRequest(NamedTuple):
    """One validated read request: exactly what to cut out of a dataset.

    Freezing the request before any data is touched means every reader applies
    the same window, the same universe and the same column order, whatever the
    source looks like.
    """

    start: date
    as_of: date
    universe: tuple[str, ...]
    fields: tuple[str, ...]

    @property
    def columns(self) -> list[str]:
        """The frame header this request must produce, in order."""
        return [DATE_COLUMN, SYMBOL_COLUMN, *self.fields]


# ---------------------------------------------------------------------------
# Request validation (before any dataset is touched)
# ---------------------------------------------------------------------------
def _prepare_request(
    snapshot: UniverseSnapshot,
    start_date: date | str,
    required_fields: Iterable[str],
) -> _ReadRequest:
    """Validate one read request and freeze it into a :class:`_ReadRequest`.

    The whole request is checked here, before a single byte of data is read, so
    a bad request fails the same way whether or not a dataset exists yet.
    """
    if not isinstance(snapshot, UniverseSnapshot):
        raise MarketDataReadError(
            "snapshot must be a UniverseSnapshot (the universe interface), got "
            f"{type(snapshot).__name__}"
        )

    start = _as_day(start_date, "start_date")
    as_of = snapshot.as_of_date
    if start > as_of:
        raise MarketDataReadError(
            f"start_date {start.isoformat()} is after the snapshot's as_of_date "
            f"{as_of.isoformat()}: the window is empty and a factor must never look "
            "forward in time"
        )

    return _ReadRequest(
        start=start,
        as_of=as_of,
        universe=snapshot.symbols,
        fields=_requested_fields(required_fields),
    )


def _as_day(value: Any, label: str) -> date:
    """A plain :class:`datetime.date` from ``value`` -- strict, never a timestamp.

    A ``datetime`` is rejected even though it *is* a ``date``: a daily reader has
    no honest use for a time of day, and silently truncating one would mean the
    reader, not the caller, decided where the window starts.  Strings must be
    strict ISO ``"YYYY-MM-DD"`` and must name a real calendar day.
    """
    if isinstance(value, datetime):
        raise MarketDataReadError(
            f"{label} must be a date or a strict ISO 'YYYY-MM-DD' string, got the "
            f"datetime {value!r}: a daily reader has no use for a time of day, pass "
            f"{label}.date() to name the session"
        )
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        text = value.strip()
        if _ISO_DAY.fullmatch(text):
            try:
                return date.fromisoformat(text)
            except ValueError:
                pass  # right shape, but no such day (e.g. '2024-02-30')
    raise MarketDataReadError(
        f"{label} must be a date or a strict ISO 'YYYY-MM-DD' string, got {value!r}"
    )


def _requested_fields(required_fields: Iterable[str]) -> tuple[str, ...]:
    """Validate the requested bar fields, keeping the caller's declaration order.

    Every problem is collected, so one bad call reports all of them at once (the
    style of :class:`factor_sys.errors.FactorContractError`).  ``date`` and
    ``symbol`` are labels rather than requests: they are always returned, so
    asking for them is a mistake worth reporting instead of ignoring.
    """
    if required_fields is None or isinstance(required_fields, str):
        raise MarketDataReadError(
            "required_fields must be a non-empty iterable of bar field names (not a "
            f"bare string), got {required_fields!r}"
        )
    try:
        candidates = list(required_fields)
    except TypeError as exc:
        raise MarketDataReadError(
            "required_fields must be an iterable of bar field names, got "
            f"{type(required_fields).__name__}"
        ) from exc

    fields: list[str] = []
    problems: list[str] = []
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
        elif value not in STANDARD_COLUMNS:
            problems.append(
                f"unknown field {value!r}: a standardized bar provides {STANDARD_COLUMNS}"
            )
        else:
            fields.append(value)

    if not candidates:
        problems.append("required_fields must not be empty")
    if problems:
        raise MarketDataReadError("invalid read request", problems)
    return tuple(fields)


# ---------------------------------------------------------------------------
# Applying one request to a source's frame (shared by every reader)
# ---------------------------------------------------------------------------
def _history_frame(frame: pd.DataFrame, request: _ReadRequest) -> pd.DataFrame:
    """Cut ``frame`` down to ``request``: universe, window, columns, order.

    This is the half of a read that must never differ between sources: a reader
    only has to hand over the bars it stores -- a long frame holding at least
    ``date``, ``symbol`` and the requested fields -- and everything the factor
    system relies on (the inclusive window, the hard ``as_of_date`` bound, the
    column order, the stable ``(symbol, date)`` sort and a result that is a
    fresh object) is applied here, once, for everybody.

    Values are never rewritten: rows are selected, columns are selected and the
    result is re-indexed, so what the source stored is what the factor sees.
    An empty selection is a normal answer, not a failure.

    Raises
    ------
    MarketDataReadError
        If ``frame`` is not a DataFrame at all, has no columns, or does not
        provide one of the requested columns -- the source cannot answer this
        request, which is different from answering it with no rows.
    """
    if not isinstance(frame, pd.DataFrame):
        raise MarketDataReadError(
            f"a market data source must return a pandas DataFrame, got {type(frame).__name__}"
        )

    columns = request.columns
    if not len(frame.columns):
        raise MarketDataReadError(
            "the market data source returned a frame with no columns at all; a source "
            f"must expose at least {columns}, even when it holds no bar"
        )
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise MarketDataReadError(
            f"the market data source does not provide column(s) {missing}; it provides "
            f"{[str(column) for column in frame.columns]}"
        )

    try:
        dates = pd.to_datetime(frame[DATE_COLUMN], errors="coerce")
    except (TypeError, ValueError) as exc:
        raise MarketDataReadError(
            f"the {DATE_COLUMN!r} column cannot be read as dates: {exc}"
        ) from exc
    unreadable = dates.isna().to_numpy()
    if unreadable.any():
        raise MarketDataReadError(
            f"the {DATE_COLUMN!r} column has {int(unreadable.sum())} missing or "
            "unparseable value(s): the read window cannot be decided for those rows"
        )

    # Positional boolean masks: they line up with the frame by construction, no
    # matter what index the source used, and a row may only survive all three.
    in_universe = frame[SYMBOL_COLUMN].isin(request.universe).to_numpy()
    in_window = (
        (dates >= pd.Timestamp(request.start)) & (dates <= pd.Timestamp(request.as_of))
    ).to_numpy()

    selected = frame.loc[in_universe & in_window, columns]
    ordered = selected.sort_values([SYMBOL_COLUMN, DATE_COLUMN], kind="stable")
    return ordered.reset_index(drop=True)


# ---------------------------------------------------------------------------
# The interface every market data source implements
# ---------------------------------------------------------------------------
class MarketDataReader(ABC):
    """The stable market-data interface of the factor system.

    An implementation answers one request with a long frame
    ``["date", "symbol", *required_fields]`` that covers
    ``[start_date, snapshot.as_of_date]`` (both ends included, nothing dated
    later), sorted by ``(symbol, date)``, holding the requested fields in the
    order the caller declared them.  A symbol with no data, a history shorter
    than the window and a window that matches nothing are *answers*, not errors:
    eligibility is the factor's decision, not the reader's.  The module
    docstring spells the contract out in full.

    A reader only ever sees a :class:`~factor_sys.universe.UniverseSnapshot`, so
    a source for another vendor's files, a database or a test fixture drops in
    without the rest of the factor system noticing.  Implementations that store
    their bars as a frame should build the answer with
    :func:`_prepare_request` and :func:`_history_frame`, which is how the window,
    the column order and the sort stay identical across sources.
    """

    @abstractmethod
    def read_history(
        self,
        snapshot: UniverseSnapshot,
        start_date: date | str,
        required_fields: Iterable[str],
    ) -> pd.DataFrame:
        """Read the stored history of ``snapshot``'s symbols over the window.

        Parameters
        ----------
        snapshot:
            The universe snapshot to read for.  ``snapshot.symbols`` is the only
            set of symbols that may appear in the result and
            ``snapshot.as_of_date`` is the last date that may appear.
        start_date:
            First date of the window, inclusive; a :class:`datetime.date` or a
            strict ISO ``"YYYY-MM-DD"`` string (see :func:`_as_day`).
        required_fields:
            Bar fields to return, e.g. ``("close", "volume")``.  Each must be a
            member of :data:`data_sys.schema.STANDARD_COLUMNS`; ``"date"`` and
            ``"symbol"`` may not be requested, as they are always returned.

        Returns
        -------
        pandas.DataFrame
            The requested rows with columns
            ``["date", "symbol", *required_fields]``, sorted by
            ``(symbol, date)``; empty of rows when nothing matches, but never
            empty of columns.

        Raises
        ------
        MarketDataReadError
            If the request cannot be honoured, or the source cannot answer it.
        """


class ParquetMarketDataReader(MarketDataReader):
    """The v1 reader: the published daily bars of the data pipeline.

    The dataset is read through :func:`data_sys.storage.read_partitioned`, so the
    Hive layout (``symbol=<TICKER>/year=<YYYY>/*.parquet``) has exactly one
    implementation in the project, and the reader selects from the returned frame
    instead of walking directories itself.  Values, rows and dtypes are passed
    through as stored.

    Parameters
    ----------
    market_bars_root:
        Root of the dataset.  Defaults to
        :attr:`data_sys.config.PipelineConfig.standardized_dir`, the project's
        ``data/standardized/market_bars``; tests pass a temporary directory.

    A root that does not exist, is not a directory, or holds no ``*.parquet`` file
    is refused when :meth:`read_history` is called -- not when the reader is
    built, because the pipeline may publish the dataset after the reader was
    created.  Building a reader touches nothing on disk.
    """

    def __init__(self, market_bars_root: Path | str | None = None) -> None:
        self._root = _market_bars_root(market_bars_root)

    @property
    def root(self) -> Path:
        """The dataset root this reader reads from."""
        return self._root

    def read_history(
        self,
        snapshot: UniverseSnapshot,
        start_date: date | str,
        required_fields: Iterable[str],
    ) -> pd.DataFrame:
        """Read the requested window out of the published dataset.

        The request is validated before the dataset is touched, so an impossible
        request fails identically whether or not data has been published yet.
        """
        request = _prepare_request(snapshot, start_date, required_fields)
        return _history_frame(self._read_dataset(), request)

    def _read_dataset(self) -> pd.DataFrame:
        """Every published bar, exactly as the storage layer returns it."""
        root = self._root
        if not root.exists():
            raise MarketDataReadError(
                f"market bars root {str(root)!r} does not exist: there is no dataset to read"
            )
        if not root.is_dir():
            raise MarketDataReadError(
                f"market bars root {str(root)!r} is not a directory, so it cannot hold the "
                "parquet dataset"
            )
        if not any(root.rglob("*.parquet")):
            raise MarketDataReadError(
                f"market bars root {str(root)!r} holds no '*.parquet' file: the standardized "
                "dataset is empty, so there is no bar to read"
            )
        return read_partitioned(root)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(market_bars_root={str(self._root)!r})"


def _market_bars_root(value: Path | str | None) -> Path:
    """The dataset root of a reader: the given path, or the pipeline's default."""
    if value is None:
        return PipelineConfig().standardized_dir
    if isinstance(value, str):
        if not value.strip():
            raise MarketDataReadError(
                "market_bars_root must not be blank: pass a directory, or omit it to read the "
                "published dataset"
            )
        return Path(value)
    if isinstance(value, Path):
        return value
    raise MarketDataReadError(
        f"market_bars_root must be a path or a string, got {type(value).__name__}"
    )


__all__ = ["MarketDataReader", "ParquetMarketDataReader"]
