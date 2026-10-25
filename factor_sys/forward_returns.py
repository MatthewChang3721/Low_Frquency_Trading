"""Forward returns: the research label that says what happened *after* a decision.

What this module is
-------------------
``factor_sys.forward_returns`` answers the one question a factor must never ask:
*what happened next?*  For a decision date ``t`` and a horizon of ``h`` sessions
it labels every symbol of a panel's universe with the close-to-close total
return::

    forward_return(t) = adjusted_close(t + h sessions) / adjusted_close(t) - 1

One factor, one decision date, one horizon and one universe become one frozen
:class:`ForwardReturnPanel`; :class:`ForwardReturnCalculator` builds it from bars
read through :class:`~factor_sys.market_data.MarketDataReader` and nothing else.

A research label, not a trade
-----------------------------
The return is measured from the close of ``t`` to the close of ``t + h
sessions`` on **adjusted close**, so it is the total return a holder of the stock
earned over the window, dividends and splits included.  It is *not* the return a
strategy would have booked: a backtest enters after the decision (on the next
tradable session's open), pays costs and slippage, and may not fill at all.
That executable return belongs to the backtest and is measured there, never
here; nothing in this module knows what an order is.

Where the future is allowed to enter
------------------------------------
A future price may only be read *after* a
:class:`~factor_sys.contracts.FactorPanel` exists, and only by this module.  So
the calculator

* creates its own :class:`~factor_sys.universe.UniverseSnapshot` whose
  ``as_of_date`` is the computed ``end_date`` -- a snapshot is what bounds a
  read, so the label read is bounded by the label window and by nothing a factor
  can see;
* calls ``MarketDataReader.read_history`` exactly once, for
  ``[panel.as_of_date, end_date]`` and the single field ``adjusted_close``;
* returns a :class:`ForwardReturnPanel`, a separate object that holds labels and
  nothing else.

A forward return never enters a ``FactorContext``, a ``FactorPanel``, a factor
formula, ``FactorRunner`` or the panel store.  This module imports none of them,
and ``raw_value`` is never read: two panels with the same identity, universe and
decision date get exactly the same labels, whatever their factor values are.

Counting a session
------------------
``h`` counts *exchange sessions*, never calendar days: the ``XNYS`` calendar
decides what a session is, so ``horizon_sessions=1`` is the next session after
``t``, and 2024-03-28 (a Thursday) is followed by 2024-04-01 (a Monday) because
Good Friday is not a session.  ``panel.as_of_date`` must itself be a session: a
weekend or an exchange holiday is refused instead of being nudged to a
neighbouring day.

What a missing price means
--------------------------
A symbol whose price on ``t`` or on ``end_date`` is absent, ``NaN``, ``+inf``,
``-inf``, not a number, or not strictly positive gets ``is_available=False`` and
``forward_return=NaN``.  Nothing is forward-filled, back-filled, interpolated or
replaced by a neighbouring session's close: a stale price would be a quietly
wrong label, while a gap is an honest one.  One unusable symbol never affects the
others, and an entirely empty read is a legal answer (every label unavailable),
not an error.

Anything that would make a label *wrong* rather than *missing* -- a reader
answering with something other than a bar frame, a duplicated ``(symbol, date)``,
a date outside the window, a symbol outside the universe -- is refused with
:class:`~factor_sys.errors.ForwardReturnError` instead of being interpreted.

Not implemented here, on purpose: IC/ICIR, quantile portfolios, factor ranks or
scores, multi-factor combinations, backtest fills, trading costs, label storage
or caching, batch scheduling and any CLI.

Example
-------
>>> from datetime import date
>>> import pandas as pd
>>> frame = pd.DataFrame(
...     {
...         "as_of_date": [pd.Timestamp("2024-03-28")] * 2,
...         "end_date": [pd.Timestamp("2024-04-01")] * 2,
...         "symbol": ["AAPL", "MSFT"],
...         "factor_name": ["momentum_12_1"] * 2,
...         "factor_version": ["1.0.0"] * 2,
...         "horizon_sessions": [1, 1],
...         "forward_return": [0.02, float("nan")],
...         "is_available": [True, False],
...     }
... )
>>> labels = ForwardReturnPanel(
...     frame,
...     universe=["AAPL", "MSFT"],
...     as_of_date=date(2024, 3, 28),
...     end_date=date(2024, 4, 1),
...     factor_name="momentum_12_1",
...     factor_version="1.0.0",
...     horizon_sessions=1,
... )
>>> labels.end_date
datetime.date(2024, 4, 1)
>>> labels.frame["is_available"].tolist()
[True, False]
>>> (labels.horizon_sessions, len(labels))
(1, 2)
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from numbers import Integral, Real
from typing import Any

import pandas as pd

from data_sys.errors import CalendarError
from data_sys.market_calendar import (
    DEFAULT_CALENDAR_NAME,
    calendar_bounds,
    get_calendar,
    is_session,
    sessions_in_range,
)
from data_sys.schema import SYMBOL_PATTERN
from factor_sys.contracts import (
    FACTOR_NAME_PATTERN,
    FactorPanel,
    _canonical_universe,
    _describe,
    _duplicates,
    _parse_dates,
    _unique,
)
from factor_sys.errors import FactorContractError, ForwardReturnError
from factor_sys.market_data import MarketDataReader
from factor_sys.universe import UniverseSnapshot

# ---------------------------------------------------------------------------
# Table shape
# ---------------------------------------------------------------------------
#: Fixed column order of a forward-return table (one factor, one session, one
#: horizon, one row per universe symbol).
FORWARD_RETURN_COLUMNS: tuple[str, ...] = (
    "as_of_date",
    "end_date",
    "symbol",
    "factor_name",
    "factor_version",
    "horizon_sessions",
    "forward_return",
    "is_available",
)

#: The only bar field a forward return reads: the total-return adjusted close.
PRICE_FIELD: str = "adjusted_close"

#: Label columns, named once so a typo fails here instead of in a comparison.
_DATE_COLUMN = "as_of_date"
_END_COLUMN = "end_date"
_SYMBOL_COLUMN = "symbol"
_NAME_COLUMN = "factor_name"
_VERSION_COLUMN = "factor_version"
_HORIZON_COLUMN = "horizon_sessions"
_RETURN_COLUMN = "forward_return"
_AVAILABLE_COLUMN = "is_available"

#: Bar columns the reader must hand back (``date``/``symbol`` plus the field).
_BAR_COLUMNS: tuple[str, ...] = ("date", "symbol", PRICE_FIELD)

#: A factor name is the contract's snake_case identifier; reused, not redefined.
_NAME_RE = re.compile(FACTOR_NAME_PATTERN)

#: A symbol is the project's one ticker pattern; reused, not redefined.
_SYMBOL_RE = re.compile(SYMBOL_PATTERN)

#: Rough number of calendar days per trading session, used to bound the search
#: for the sessions *after* a decision date.  Generous on purpose: the extra
#: margin below absorbs a long holiday run, and ``sessions_in_range`` clamps the
#: window to the calendar anyway.
_CALENDAR_DAYS_PER_SESSION = 3

#: Extra calendar days added to that bound, so a search never truncates early.
_SESSION_SEARCH_MARGIN_DAYS = 14


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------
def _is_count(value: object) -> bool:
    """``True`` for a real count: a ``bool`` is a flag, not a count."""
    return isinstance(value, Integral) and not isinstance(value, bool)


def _is_real(value: object) -> bool:
    """``True`` for a real number: a ``bool`` is not a price here."""
    return isinstance(value, Real) and not isinstance(value, bool)


def _plain_date(value: Any, label: str, problems: list[str]) -> date | None:
    """A plain ``date`` -- never a ``datetime``, never a string, never a guess.

    The two dates of a label are sessions the calendar already decided, so a
    timestamp or a loosely spelled string cannot become one here: parsing it
    would mean this contract, not the calendar, chose the window.
    """
    if isinstance(value, datetime):
        problems.append(
            f"{label} must be a datetime.date (a session, not a datetime), got {value!r}"
        )
        return None
    if not isinstance(value, date):
        problems.append(f"{label} must be a datetime.date, got {value!r}")
        return None
    return value


def _horizon_sessions(value: Any, label: str, problems: list[str]) -> int | None:
    """A positive number of sessions, or ``None`` (plus a recorded problem)."""
    if not _is_count(value) or value < 1:
        problems.append(
            f"{label} must be a positive integer number of sessions, got {value!r}: "
            "one session is the next trading day, and zero or fewer is no horizon at all"
        )
        return None
    return int(value)


def _factor_name(value: Any, label: str, problems: list[str]) -> str | None:
    """A factor name in the contract's snake_case dialect, or ``None``."""
    if not isinstance(value, str) or not _NAME_RE.fullmatch(value):
        problems.append(
            f"{label} must be a snake_case identifier ({FACTOR_NAME_PATTERN}), got {value!r}"
        )
        return None
    return value


def _factor_version(value: Any, label: str, problems: list[str]) -> str | None:
    """A non-empty version string, or ``None`` (plus a recorded problem)."""
    if not isinstance(value, str) or not value.strip():
        problems.append(f"{label} must be a non-empty string, got {value!r}")
        return None
    return value


def _date_values(
    values: pd.Series,
    label: str,
    problems: list[str],
) -> tuple[pd.Series | None, list[date]]:
    """``values`` as normalized dates, plus the same dates as plain ``date``s.

    Parsing reuses :func:`factor_sys.contracts._parse_dates`, so a label table
    reads dates exactly the way a factor panel does.  A column that cannot be
    parsed is reported (never raised) and returns ``(None, [])``: the caller goes
    on collecting every other problem of the same table.
    """
    try:
        parsed = _parse_dates(values, label)
    except FactorContractError as exc:
        problems.append(str(exc))
        return None, []
    return parsed, [stamp.date() for stamp in parsed]


# ---------------------------------------------------------------------------
# Row validation (shared by the panel's constructor)
# ---------------------------------------------------------------------------
def _validated_rows(
    frame: Any,
    symbols: tuple[str, ...],
    as_of: date | None,
    end: date | None,
    name: str | None,
    version: str | None,
    horizon: int | None,
    problems: list[str],
) -> pd.DataFrame | None:
    """Validate ``frame`` as one forward-return table; extend ``problems``.

    ``rows`` is the contract's own copy in canonical column order, with
    ``datetime64[s]`` dates, an ``int64`` horizon, ``float64`` returns and a plain
    ``bool`` availability flag.  ``None`` means the frame could not even be
    interpreted (wrong type, wrong columns, no rows), so row-level checks were
    skipped.
    """
    if not isinstance(frame, pd.DataFrame):
        problems.append(f"frame must be a pandas DataFrame, got {type(frame).__name__}")
        return None

    missing = [column for column in FORWARD_RETURN_COLUMNS if column not in frame.columns]
    unexpected = [column for column in frame.columns if column not in FORWARD_RETURN_COLUMNS]
    if missing or unexpected:
        if missing:
            problems.append(f"missing column(s) {missing}")
        if unexpected:
            problems.append(f"unexpected column(s) {[str(column) for column in unexpected]}")
        return None
    if len(frame) == 0:
        problems.append(
            "frame must not be empty: the universe of "
            f"{len(symbols)} symbol(s) needs one row per symbol"
        )
        return None

    rows = frame.loc[:, list(FORWARD_RETURN_COLUMNS)].copy(deep=True).reset_index(drop=True)

    # -- as_of_date / end_date: real days, one each, in the right order ------
    frame_dates: dict[str, date] = {}
    for column in (_DATE_COLUMN, _END_COLUMN):
        parsed, values = _date_values(rows[column], f"ForwardReturnPanel.{column}", problems)
        if parsed is None:
            continue
        distinct = _unique(values)
        if len(distinct) != 1:
            problems.append(
                f"{column} must hold exactly one date, got {len(distinct)}: "
                f"{_describe(day.isoformat() for day in distinct)}"
            )
            continue
        frame_dates[column] = distinct[0]
        rows[column] = parsed

    frame_as_of = frame_dates.get(_DATE_COLUMN)
    frame_end = frame_dates.get(_END_COLUMN)
    if frame_as_of is not None and frame_end is not None and frame_end <= frame_as_of:
        problems.append(
            f"end_date {frame_end.isoformat()} must be after as_of_date "
            f"{frame_as_of.isoformat()}: a forward return is measured to a later session"
        )
    for column, declared, found in (
        (_DATE_COLUMN, as_of, frame_as_of),
        (_END_COLUMN, end, frame_end),
    ):
        if declared is not None and found is not None and found != declared:
            problems.append(
                f"{column} {found.isoformat()} does not match the label's {column} "
                f"{declared.isoformat()}: the rows and the object must agree"
            )

    # -- symbol: one row per universe symbol, in the universe's own order ----
    raw_symbols = rows[_SYMBOL_COLUMN].tolist()
    malformed = _unique(
        value for value in raw_symbols if not isinstance(value, str) or not _SYMBOL_RE.match(value)
    )
    if malformed:
        problems.append(
            f"symbol column has invalid value(s) {_describe(malformed)}: expected "
            f"{SYMBOL_PATTERN} (upper-case letters, digits, '.' and '-')"
        )

    known = set(symbols)
    outside = [
        value for value in _unique(raw_symbols) if isinstance(value, str) and value not in known
    ]
    if outside:
        problems.append(
            f"symbol column contains symbol(s) outside the universe: {_describe(outside)}"
        )

    repeated = _duplicates([value for value in raw_symbols if isinstance(value, str)])
    if repeated:
        problems.append(
            f"symbol column repeats symbol(s) {_describe(repeated)}: a label table holds one "
            "row per symbol, never two"
        )
    present = {value for value in raw_symbols if isinstance(value, str)}
    absent = [symbol for symbol in symbols if symbol not in present]
    if absent:
        problems.append(
            f"frame must cover every symbol of the universe; missing: {_describe(absent)}"
        )
    if len(rows) != len(symbols):
        problems.append(
            f"frame has {len(rows)} row(s) for a universe of {len(symbols)} symbol(s): "
            "each symbol needs exactly one row"
        )
    elif raw_symbols != list(symbols) and not (malformed or outside or repeated or absent):
        problems.append(
            "row order must equal the universe order; expected "
            f"{_describe(symbols)}, got {_describe(raw_symbols)}"
        )

    return _validated_identity_and_values(rows, name, version, horizon, problems)


def _validated_identity_and_values(
    rows: pd.DataFrame,
    name: str | None,
    version: str | None,
    horizon: int | None,
    problems: list[str],
) -> pd.DataFrame:
    """Check the identity columns, the horizon and the label values of ``rows``."""
    names = _unique(rows[_NAME_COLUMN].tolist())
    if len(names) != 1:
        problems.append(
            f"factor_name must hold exactly one value, got {len(names)}: {_describe(names)}"
        )
    elif not isinstance(names[0], str) or not _NAME_RE.fullmatch(names[0]):
        problems.append(f"factor_name {names[0]!r} must be snake_case ({FACTOR_NAME_PATTERN})")
    elif name is not None and names[0] != name:
        problems.append(f"factor_name {names[0]!r} does not match the label's factor_name {name!r}")

    versions = _unique(rows[_VERSION_COLUMN].tolist())
    if len(versions) != 1:
        problems.append(
            "factor_version must hold exactly one value, got "
            f"{len(versions)}: {_describe(versions)}"
        )
    elif not isinstance(versions[0], str) or not versions[0].strip():
        problems.append(f"factor_version must be a non-empty string, got {versions[0]!r}")
    elif version is not None and versions[0] != version:
        problems.append(
            f"factor_version {versions[0]!r} does not match the label's factor_version {version!r}"
        )

    # -- horizon: one positive whole number of sessions ----------------------
    horizons = rows[_HORIZON_COLUMN]
    if not pd.api.types.is_integer_dtype(horizons.dtype):
        problems.append(f"horizon_sessions must be an integer column, got dtype {horizons.dtype}")
    elif bool(horizons.isna().any()):
        problems.append(
            f"horizon_sessions must not be missing ({int(horizons.isna().sum())} row(s) are NA)"
        )
    else:
        distinct_horizons = _unique(int(value) for value in horizons.tolist())
        if len(distinct_horizons) != 1:
            problems.append(
                "horizon_sessions must hold exactly one value, got "
                f"{len(distinct_horizons)}: {_describe(distinct_horizons)}"
            )
        elif distinct_horizons[0] < 1:
            problems.append(
                "horizon_sessions must be a positive number of sessions, got "
                f"{distinct_horizons[0]}"
            )
        else:
            rows[_HORIZON_COLUMN] = pd.Series([distinct_horizons[0]] * len(rows), dtype="int64")
            if horizon is not None and distinct_horizons[0] != horizon:
                problems.append(
                    f"horizon_sessions {distinct_horizons[0]} does not match the label's "
                    f"horizon_sessions {horizon}"
                )

    # -- is_available: a real boolean, never missing -------------------------
    availability_column = rows[_AVAILABLE_COLUMN]
    availability: list[bool] | None = None
    if not pd.api.types.is_bool_dtype(availability_column.dtype):
        problems.append(
            "is_available must be a boolean column (True/False), got dtype "
            f"{availability_column.dtype}"
        )
    elif bool(availability_column.isna().any()):
        problems.append(
            "is_available must not be missing "
            f"({int(availability_column.isna().sum())} row(s) are NA)"
        )
    else:
        availability = [bool(value) for value in availability_column.tolist()]

    # -- forward_return: float, finite when promised, NaN when not ----------
    raw_returns = rows[_RETURN_COLUMN]
    try:
        numeric = pd.to_numeric(raw_returns, errors="coerce")
    except (TypeError, ValueError) as exc:
        problems.append(f"forward_return must be numeric, got dtype {raw_returns.dtype}: {exc}")
        return rows
    unconvertible = raw_returns.notna() & numeric.isna()
    if bool(unconvertible.any()):
        problems.append(
            f"forward_return must be convertible to float; {int(unconvertible.sum())} row(s) "
            f"are not, e.g. {_describe(raw_returns[unconvertible].tolist())}"
        )
        return rows

    values = pd.Series(
        numeric.to_numpy(dtype="float64", na_value=float("nan")),
        dtype="float64",
    )
    rows[_RETURN_COLUMN] = values
    if availability is None:
        return rows

    is_available = pd.Series(availability, dtype=bool)
    finite = pd.Series([math.isfinite(value) for value in values.tolist()], dtype=bool)
    promised = is_available & ~finite
    if bool(promised.any()):
        problems.append(
            f"{int(promised.sum())} row(s) marked is_available=True hold a NaN or infinite "
            f"forward_return ({_describe(rows.loc[promised, _SYMBOL_COLUMN].tolist())}): an "
            "available label is always a finite number"
        )
    denied = ~is_available & ~values.isna()
    if bool(denied.any()):
        problems.append(
            f"{int(denied.sum())} row(s) marked is_available=False hold a forward_return "
            f"({_describe(rows.loc[denied, _SYMBOL_COLUMN].tolist())}): an unavailable label is "
            "always NaN"
        )
    rows[_AVAILABLE_COLUMN] = is_available
    return rows


# ---------------------------------------------------------------------------
# ForwardReturnPanel -- the label table
# ---------------------------------------------------------------------------
@dataclass(frozen=True, init=False, eq=False)
class ForwardReturnPanel:
    """Future returns for one factor, one decision date and one horizon.

    One object holds exactly one ``as_of_date``, one ``end_date``, one factor
    name and version, one horizon and one row per symbol of one universe, in the
    universe's own order.  It is the *only* place a future return lives: it is a
    separate object from :class:`~factor_sys.contracts.FactorPanel`, it has no
    ``raw_value`` column, and nothing in the factor path ever receives one.

    A row is one symbol's label:

    ==================  =====================================================
    ``as_of_date``      the decision session ``t`` (a real ``date``)
    ``end_date``        the session ``h`` sessions later (a real ``date``)
    ``symbol``          ticker, canonical upper case, from the universe
    ``factor_name``     snake_case factor identifier
    ``factor_version``  non-empty version string
    ``horizon_sessions`` positive integer number of sessions
    ``forward_return``  ``adjusted_close(end_date) / adjusted_close(t) - 1``,
                        finite when ``is_available`` and ``NaN`` when not
    ``is_available``    boolean: was a usable price published at both ends?
    ==================  =====================================================

    ``is_available=False`` always means ``forward_return`` is ``NaN``, and
    ``is_available=True`` always means it is a finite number: there is no third
    state, so a consumer may filter on the flag and sum the values without
    checking for infinities.  A missing label is a fact about the data, not an
    error.

    Parameters
    ----------
    frame:
        Long-format labels with exactly the :data:`FORWARD_RETURN_COLUMNS`
        columns (any column order is accepted and normalized).
    universe:
        Symbols the labels cover: non-empty, canonical, unique, and exactly the
        symbols of ``frame`` -- in that same order.
    as_of_date / end_date:
        The two sessions of the label, as plain :class:`datetime.date` values.
        ``end_date`` must be after ``as_of_date``; a ``datetime`` or a string is
        refused, because the calendar, not this contract, decides a session.
    factor_name / factor_version:
        The identity of the factor the labels belong to, copied from the panel;
        they must agree with the ``factor_name``/``factor_version`` columns.
    horizon_sessions:
        Sessions between the two dates, as a positive integer (``bool``, ``0``,
        negatives, floats and strings are refused).

    Raises
    ------
    factor_sys.errors.ForwardReturnError
        If any row, column, identity field or value breaks the contract above;
        every problem found is reported at once.

    Notes
    -----
    Panels are compared by identity (``eq=False``) because comparing two frames
    elementwise is ambiguous; use :meth:`frame` for value comparisons.

    Examples
    --------
    >>> from datetime import date
    >>> import pandas as pd
    >>> frame = pd.DataFrame(
    ...     {
    ...         "as_of_date": [pd.Timestamp("2024-03-28")],
    ...         "end_date": [pd.Timestamp("2024-04-01")],
    ...         "symbol": ["AAPL"],
    ...         "factor_name": ["momentum_12_1"],
    ...         "factor_version": ["1.0.0"],
    ...         "horizon_sessions": [1],
    ...         "forward_return": [0.0125],
    ...         "is_available": [True],
    ...     }
    ... )
    >>> labels = ForwardReturnPanel(
    ...     frame,
    ...     universe=["AAPL"],
    ...     as_of_date=date(2024, 3, 28),
    ...     end_date=date(2024, 4, 1),
    ...     factor_name="momentum_12_1",
    ...     factor_version="1.0.0",
    ...     horizon_sessions=1,
    ... )
    >>> labels.symbols
    ('AAPL',)
    """

    as_of_date: date
    end_date: date
    factor_name: str
    factor_version: str
    horizon_sessions: int
    universe: tuple[str, ...]
    _rows: pd.DataFrame = field(repr=False)

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        universe: Iterable[str],
        as_of_date: date,
        end_date: date,
        factor_name: str,
        factor_version: str,
        horizon_sessions: int,
    ) -> None:
        """Validate ``frame`` as one label table and freeze it."""
        try:
            symbols = _canonical_universe(universe, "ForwardReturnPanel.universe")
        except FactorContractError as exc:
            raise ForwardReturnError(str(exc), exc.problems) from exc

        problems: list[str] = []
        as_of = _plain_date(as_of_date, "ForwardReturnPanel.as_of_date", problems)
        end = _plain_date(end_date, "ForwardReturnPanel.end_date", problems)
        name = _factor_name(factor_name, "ForwardReturnPanel.factor_name", problems)
        version = _factor_version(factor_version, "ForwardReturnPanel.factor_version", problems)
        horizon = _horizon_sessions(
            horizon_sessions, "ForwardReturnPanel.horizon_sessions", problems
        )
        rows = _validated_rows(frame, symbols, as_of, end, name, version, horizon, problems)

        if rows is None or problems:
            raise ForwardReturnError(
                "ForwardReturnPanel(...) is not a valid set of forward-return labels", problems
            )

        object.__setattr__(self, "as_of_date", as_of)
        object.__setattr__(self, "end_date", end)
        object.__setattr__(self, "factor_name", name)
        object.__setattr__(self, "factor_version", version)
        object.__setattr__(self, "horizon_sessions", horizon)
        object.__setattr__(self, "universe", symbols)
        object.__setattr__(self, "_rows", rows)

    @property
    def frame(self) -> pd.DataFrame:
        """A copy of the labels in canonical column order.

        Mutating the returned frame cannot change the panel: every call hands
        out a fresh deep copy.
        """
        return self._rows.copy(deep=True)

    @property
    def symbols(self) -> tuple[str, ...]:
        """The labeled symbols, in row order (always ``universe``)."""
        return tuple(self._rows[_SYMBOL_COLUMN].tolist())

    def __len__(self) -> int:
        """Number of rows: one per symbol of the universe."""
        return int(len(self._rows))

    def __repr__(self) -> str:
        """A one-line summary for logs; the label rows stay out of it."""
        available = int(self._rows[_AVAILABLE_COLUMN].sum())
        return (
            f"ForwardReturnPanel(as_of_date={self.as_of_date.isoformat()!r}, "
            f"end_date={self.end_date.isoformat()!r}, "
            f"factor={self.factor_name}@{self.factor_version}, "
            f"horizon_sessions={self.horizon_sessions}, universe={len(self.universe)}, "
            f"available={available})"
        )


# ---------------------------------------------------------------------------
# ForwardReturnCalculator -- bars in, labels out
# ---------------------------------------------------------------------------
def _calendar_named(calendar_name: Any) -> tuple[str, Any]:
    """Resolve ``calendar_name`` once: ``(name, calendar)`` or a clear failure."""
    if not isinstance(calendar_name, str) or not calendar_name.strip():
        raise ForwardReturnError(
            "calendar_name must be a non-empty trading calendar name such as "
            f"{DEFAULT_CALENDAR_NAME!r}, got {calendar_name!r}"
        )
    name = calendar_name.strip()
    try:
        return name, get_calendar(name)
    except CalendarError as exc:
        raise ForwardReturnError(f"could not load the trading calendar {name!r}: {exc}") from exc


class ForwardReturnCalculator:
    """Label a panel's universe with the return that followed its decision date.

    The calculator is the one object that is allowed to look forward, and it
    only does so after a :class:`~factor_sys.contracts.FactorPanel` already
    exists.  One call is always these steps, in this order:

    1. the panel must be a :class:`~factor_sys.contracts.FactorPanel` and the
       horizon a positive integer (``bool`` is refused, and so is ``0``);
    2. ``panel.as_of_date`` must be a session of the calculator's calendar --
       never nudged to a neighbouring day;
    3. ``end_date`` is the ``h``-th session *after* ``t``
       (:func:`data_sys.market_calendar.sessions_in_range`), so weekends and
       exchange holidays are skipped by the exchange itself;
    4. a :class:`~factor_sys.universe.UniverseSnapshot` is built for exactly the
       panel's universe, with ``as_of_date=end_date``: the snapshot is the upper
       bound of the read, so the label window cannot leak past the horizon;
    5. ``read_history`` is called once, for
       ``[panel.as_of_date, end_date]`` and exactly the field
       ``adjusted_close``;
    6. the answer is validated (it must be a bar frame covering that window and
       that universe, with no duplicate ``(symbol, date)``) and then each symbol
       is labelled on its own;
    7. a frozen :class:`ForwardReturnPanel` in ``panel.universe`` order is
       returned.

    Nothing is written, cached or scheduled: no store, no registry, no runner
    and no file is involved, and the panel is only read (never modified), which
    is what makes the same call always produce the same labels.

    Parameters
    ----------
    market_data_reader:
        Where the bars come from.  Only
        :meth:`~factor_sys.market_data.MarketDataReader.read_history` is called,
        so a Parquet reader, a database reader or an in-memory fixture all work
        without this class changing.
    calendar_name:
        Trading calendar that defines "a session" (``"XNYS"`` by default).  It is
        resolved once, when the calculator is built, so a misspelled name fails
        immediately instead of at the first computation.

    Raises
    ------
    factor_sys.errors.ForwardReturnError
        If ``market_data_reader`` is not a
        :class:`~factor_sys.market_data.MarketDataReader`, or ``calendar_name``
        is blank or unknown.

    Examples
    --------
    >>> class InMemoryReader(MarketDataReader):
    ...     def read_history(self, snapshot, start_date, required_fields):
    ...         return pd.DataFrame(
    ...             {
    ...                 "date": [pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-08")],
    ...                 "symbol": ["AAPL", "AAPL"],
    ...                 "adjusted_close": [100.0, 110.0],
    ...             }
    ...         )
    >>> calculator = ForwardReturnCalculator(InMemoryReader())
    >>> calculator.calendar_name
    'XNYS'
    """

    def __init__(
        self,
        market_data_reader: MarketDataReader,
        *,
        calendar_name: str = DEFAULT_CALENDAR_NAME,
    ) -> None:
        if not isinstance(market_data_reader, MarketDataReader):
            raise ForwardReturnError(
                "market_data_reader must be a MarketDataReader, got "
                f"{type(market_data_reader).__name__}: labels read bars through "
                "read_history(snapshot, start_date, required_fields) and nothing else"
            )
        name, calendar = _calendar_named(calendar_name)

        self._market_data_reader = market_data_reader
        self._calendar_name = name
        self._calendar = calendar

    @property
    def market_data_reader(self) -> MarketDataReader:
        """The reader every labelling call asks for bars."""
        return self._market_data_reader

    @property
    def calendar_name(self) -> str:
        """Name of the trading calendar that defines "a session" here."""
        return self._calendar_name

    @property
    def calendar(self) -> Any:
        """The resolved calendar object (an ``exchange_calendars`` calendar)."""
        return self._calendar

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(calendar_name={self._calendar_name!r}, "
            f"market_data_reader={type(self._market_data_reader).__name__})"
        )

    # -- the computation -----------------------------------------------------
    def compute(self, panel: FactorPanel, horizon_sessions: int) -> ForwardReturnPanel:
        """Label ``panel``'s universe with the return ``horizon_sessions`` later.

        The label of one symbol is

            ``adjusted_close(end_date) / adjusted_close(as_of_date) - 1``

        with ``end_date`` the ``horizon_sessions``-th session after
        ``panel.as_of_date``.  Only the panel's identity, date and universe are
        read: ``raw_value`` and ``is_eligible`` play no part, so two panels that
        share a date and a universe get identical labels whatever their factor
        values are.

        A symbol is labelled only if both ends carry a usable price.  A missing
        bar, ``NaN``, ``+inf``, ``-inf``, a non-numeric value or a price that is
        not positive makes that one label unavailable (``is_available=False``,
        ``forward_return=NaN``) without touching the other symbols and without
        raising: a gap in the data is a missing label, not a broken request.

        Parameters
        ----------
        panel:
            The cross-section produced for one decision date.  It is read
            through its public attributes only and is never modified.
        horizon_sessions:
            Sessions between the decision date and the measurement date, as a
            positive integer (``1`` is the next session; ``bool``, ``0``,
            negative numbers, floats and strings are refused).

        Returns
        -------
        ForwardReturnPanel
            One row per ``panel.universe`` symbol, in the panel's universe order,
            dated ``panel.as_of_date`` and ``end_date``.

        Raises
        ------
        ForwardReturnError
            If ``panel`` is not a ``FactorPanel``, ``horizon_sessions`` is not a
            positive integer, ``panel.as_of_date`` is not a session the calendar
            covers, the calendar does not reach ``horizon_sessions`` sessions
            ahead, or the reader answers with something that is not a trustworthy
            bar window (see :func:`_validated_prices`).
        MarketDataReadError
            Propagated unchanged from the reader: an impossible request or a
            dataset that cannot answer it is the reader's failure to report.
        """
        if not isinstance(panel, FactorPanel):
            raise ForwardReturnError(
                "ForwardReturnCalculator.compute() needs a FactorPanel, got "
                f"{type(panel).__name__}: a forward return labels a validated cross-section "
                "after it exists, never a raw frame or a price series"
            )

        problems: list[str] = []
        horizon = _horizon_sessions(horizon_sessions, "horizon_sessions", problems)
        if horizon is None:
            raise ForwardReturnError(
                "cannot label the cross-section of "
                f"{panel.factor_name!r} for {panel.as_of_date.isoformat()}",
                problems,
            )

        end_date = self._end_date(panel.as_of_date, horizon)
        snapshot = self._label_snapshot(panel, horizon, end_date)

        bars = self._market_data_reader.read_history(
            snapshot=snapshot,
            start_date=panel.as_of_date,
            required_fields=(PRICE_FIELD,),
        )
        prices = _validated_prices(bars, panel.universe, panel.as_of_date, end_date)
        frame = _label_rows(panel, horizon, end_date, prices)

        return ForwardReturnPanel(
            frame,
            universe=panel.universe,
            as_of_date=panel.as_of_date,
            end_date=end_date,
            factor_name=panel.factor_name,
            factor_version=panel.factor_version,
            horizon_sessions=horizon,
        )

    # -- steps that need the calendar ---------------------------------------
    def _end_date(self, as_of: date, horizon: int) -> date:
        """The ``horizon``-th session after ``as_of``, straight from the calendar.

        The decision date must be a session the calendar covers, and it is never
        moved to a neighbouring day: measuring from a day the market was shut
        would quietly relabel the feature's date.  Sessions are then counted
        forward with :func:`data_sys.market_calendar.sessions_in_range`, so a
        weekend, Good Friday or any other exchange holiday costs no session.
        """
        first, last = calendar_bounds(self._calendar)
        if as_of < first or as_of > last:
            raise ForwardReturnError(
                f"{as_of.isoformat()} lies outside the {self._calendar_name} calendar "
                f"({first.isoformat()}..{last.isoformat()}): a forward return is measured "
                "from a decision date the calendar actually covers"
            )
        if not is_session(self._calendar, as_of):
            raise ForwardReturnError(
                f"{as_of.isoformat()} is not a {self._calendar_name} trading session (a "
                "weekend or an exchange holiday): forward returns start on the panel's own "
                "session, and it is never moved to a neighbouring one"
            )

        search_days = horizon * _CALENDAR_DAYS_PER_SESSION + _SESSION_SEARCH_MARGIN_DAYS
        window_end = min(as_of + timedelta(days=search_days), last)
        # ``as_of`` is a session, so the window starts with ``as_of`` itself and
        # the element at index ``horizon`` is the end date.
        sessions = sessions_in_range(self._calendar, as_of, window_end)
        if len(sessions) <= horizon:
            raise ForwardReturnError(
                f"the {self._calendar_name} calendar covers only "
                f"{max(len(sessions) - 1, 0)} session(s) after {as_of.isoformat()} "
                f"(last session {last.isoformat()}), so a horizon of {horizon} session(s) "
                "has no end date to measure to"
            )
        return sessions[horizon]

    def _label_snapshot(self, panel: FactorPanel, horizon: int, end_date: date) -> UniverseSnapshot:
        """The label read's own snapshot: the panel's universe, bounded by ``end_date``.

        A snapshot is what bounds a read, so giving the label read a snapshot
        whose ``as_of_date`` *is* the horizon's end date makes the window exactly
        ``[as_of_date, end_date]``.  This snapshot exists only here and for this
        call: it is never handed to a factor, a context or the runner.
        """
        return UniverseSnapshot(
            as_of_date=end_date,
            symbols=panel.universe,
            source=(
                f"forward-return labels: {panel.factor_name}@{panel.factor_version}, "
                f"{horizon} {self._calendar_name} session(s) after "
                f"{panel.as_of_date.isoformat()}"
            ),
        )


# ---------------------------------------------------------------------------
# The reader's answer, and the labels built from it
# ---------------------------------------------------------------------------
def _validated_prices(
    frame: Any,
    universe: tuple[str, ...],
    as_of: date,
    end_date: date,
) -> pd.DataFrame:
    """The reader's answer, checked as a price window ``[as_of, end_date]``.

    A custom reader is not trusted to be well behaved, and a label built on a
    malformed answer would be silently wrong rather than merely missing.  So the
    answer must be a frame holding ``date``, ``symbol`` and ``adjusted_close``,
    with parseable dates, canonical symbols, no symbol outside the universe, no
    duplicated ``(symbol, date)`` and no date outside the requested window;
    anything else is reported (all at once) instead of interpreted.  Missing
    rows, on the other hand, are fine: an unfinished window is a missing label.

    Returns
    -------
    pandas.DataFrame
        The three label columns, one row per published bar, in the reader's
        order.  Extra columns the reader volunteered are ignored.
    """
    if not isinstance(frame, pd.DataFrame):
        raise ForwardReturnError(
            "the reader's answer cannot be trusted as a price window",
            [
                "MarketDataReader.read_history() must return a pandas DataFrame, got "
                f"{type(frame).__name__}: a price window is selected rows, never an "
                "arbitrary object"
            ],
        )

    missing = [column for column in _BAR_COLUMNS if column not in frame.columns]
    if missing:
        raise ForwardReturnError(
            "the reader's answer cannot be trusted as a price window",
            [
                f"the frame is missing required column(s) {missing}: a forward return is "
                f"read from {list(_BAR_COLUMNS)}"
            ],
        )

    rows = frame.loc[:, list(_BAR_COLUMNS)].copy(deep=True).reset_index(drop=True)
    problems: list[str] = []

    parsed, dates = _date_values(rows["date"], "the reader's 'date' column", problems)
    if parsed is not None:
        rows["date"] = parsed
        far_rows = [day for day in dates if day < as_of or day > end_date]
        if far_rows:
            problems.append(
                f"the reader returned {len(far_rows)} row(s) dated outside "
                f"[{as_of.isoformat()}, {end_date.isoformat()}], e.g. "
                f"{_describe(day.isoformat() for day in sorted(set(far_rows)))}"
            )

    raw_symbols = rows["symbol"].tolist()
    malformed = _unique(
        value for value in raw_symbols if not isinstance(value, str) or not _SYMBOL_RE.match(value)
    )
    if malformed:
        problems.append(
            f"the reader's 'symbol' column has invalid value(s) {_describe(malformed)}: expected "
            f"{SYMBOL_PATTERN} (upper-case letters, digits, '.' and '-')"
        )
    known = set(universe)
    outside = [
        value for value in _unique(raw_symbols) if isinstance(value, str) and value not in known
    ]
    if outside:
        problems.append(f"the reader returned symbol(s) outside the universe: {_describe(outside)}")

    if parsed is not None:
        duplicated = rows.duplicated(subset=["date", "symbol"])
        if bool(duplicated.any()):
            problems.append(
                f"the reader returned {int(duplicated.sum())} duplicate (symbol, date) row(s): "
                f"{_describe(_unique(rows.loc[duplicated, 'symbol'].tolist()))}"
            )

    if problems:
        raise ForwardReturnError(
            "the reader's answer cannot be trusted as a price window", problems
        )
    return rows


def _positive_price(value: Any) -> float | None:
    """``value`` as a usable close, or ``None`` when it cannot be one.

    A price has to be a real, finite, strictly positive number: ``NaN``,
    ``+inf``, ``-inf``, ``0``, a negative value, a string or a missing entry is
    not a price, and a return computed from one would be wrong rather than
    absent.
    """
    if not _is_real(value):
        return None
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        return None
    return number


def _label_rows(
    panel: FactorPanel,
    horizon: int,
    end_date: date,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """The label table of ``panel``: one row per symbol, in the universe's order.

    The two ends of the window are taken *exactly*: a bar dated ``as_of_date``
    and a bar dated ``end_date``.  A symbol missing either one is unavailable --
    the previous session's close is never borrowed, because a stale price would
    turn "no data" into a wrong number.
    """
    start_stamp = pd.Timestamp(panel.as_of_date)
    end_stamp = pd.Timestamp(end_date)
    at_start: dict[str, Any] = {}
    at_end: dict[str, Any] = {}
    for symbol, day, value in zip(
        prices["symbol"].tolist(),
        prices["date"].tolist(),
        prices[PRICE_FIELD].tolist(),
        strict=True,
    ):
        if day == start_stamp:
            at_start[symbol] = value
        elif day == end_stamp:
            at_end[symbol] = value

    returns: list[float] = []
    flags: list[bool] = []
    for symbol in panel.universe:
        start_price = _positive_price(at_start.get(symbol))
        end_price = _positive_price(at_end.get(symbol))
        if start_price is None or end_price is None:
            returns.append(float("nan"))
            flags.append(False)
            continue
        value = end_price / start_price - 1.0
        # A ratio beyond the float range cannot be a label: report it missing
        # rather than as an infinity, so ``is_available`` keeps its promise.
        finite = math.isfinite(value)
        returns.append(value if finite else float("nan"))
        flags.append(finite)

    count = len(panel.universe)
    frame = pd.DataFrame(
        {
            _DATE_COLUMN: pd.Series([start_stamp] * count, dtype="datetime64[s]"),
            _END_COLUMN: pd.Series([end_stamp] * count, dtype="datetime64[s]"),
            _SYMBOL_COLUMN: list(panel.universe),
            _NAME_COLUMN: [panel.factor_name] * count,
            _VERSION_COLUMN: [panel.factor_version] * count,
            _HORIZON_COLUMN: pd.Series([horizon] * count, dtype="int64"),
            _RETURN_COLUMN: pd.Series(returns, dtype="float64"),
            _AVAILABLE_COLUMN: pd.Series(flags, dtype=bool),
        }
    )
    return frame.loc[:, list(FORWARD_RETURN_COLUMNS)]


__all__ = [
    "FORWARD_RETURN_COLUMNS",
    "PRICE_FIELD",
    "ForwardReturnCalculator",
    "ForwardReturnError",
    "ForwardReturnPanel",
]
