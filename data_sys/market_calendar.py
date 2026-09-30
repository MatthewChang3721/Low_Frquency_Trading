"""NYSE trading-calendar helpers, the single definition of "a trading day".

Why this module exists
----------------------
Every incremental-update decision is expressed in *NYSE sessions*, never in
calendar days:

* a weekend or an exchange holiday must never look like a data gap,
* "the last 5 trading days" must mean the last 5 sessions,
* a "contiguous gap" means missing sessions that are adjacent **in the session
  sequence** (so Friday + Monday is one gap, while Thursday + Monday is two).

Everything is offline: ``exchange_calendars`` ships the holiday rules, no
network access is involved.  Inputs may be :class:`datetime.date` or
``YYYY-MM-DD`` strings; every output is a plain :class:`datetime.date`.

Note on ``exchange_calendars`` 4.13.2: passing a *timezone-aware*
``pandas.Timestamp`` to ``sessions_in_range`` raises ``AttributeError`` inside
the library, so this module deliberately sticks to naive ``date``/``Timestamp``
values.
"""

from __future__ import annotations

import bisect
import functools
import logging
from datetime import date, timedelta
from typing import Any

import exchange_calendars as xcals
import pandas as pd

from data_sys.errors import CalendarError
from data_sys.utils import parse_date

logger = logging.getLogger(__name__)

#: New York Stock Exchange (``XNYS`` covers NYSE / Nasdaq / NYSE American hours).
DEFAULT_CALENDAR_NAME = "XNYS"

#: Rough number of calendar days per trading session, used to look back far
#: enough when collecting the N most recent sessions.
_CALENDAR_DAYS_PER_SESSION = 3


@functools.lru_cache(maxsize=8)
def get_calendar(name: str = DEFAULT_CALENDAR_NAME) -> Any:
    """Return (and cache) the ``exchange_calendars`` calendar called ``name``.

    Raises
    ------
    data_sys.errors.CalendarError
        If the calendar name is unknown or the library cannot build it.
    """
    try:
        calendar = xcals.get_calendar(name)
    except Exception as exc:  # noqa: BLE001 - normalize library lookup errors
        raise CalendarError(f"could not load trading calendar {name!r}: {exc}") from exc
    logger.debug("loaded trading calendar %s", name)
    return calendar


def calendar_bounds(calendar: Any) -> tuple[date, date]:
    """First and last session the calendar has data for (inclusive)."""
    return calendar.first_session.date(), calendar.last_session.date()


def _as_date(value: Any, label: str) -> date:
    """Normalize ``value`` to a plain date, or explain why it cannot be."""
    try:
        parsed = parse_date(value)
    except (ValueError, TypeError) as exc:
        raise CalendarError(f"{label} is not a usable date: {value!r} ({exc})") from exc
    if parsed is None:
        raise CalendarError(f"{label} is required, got {value!r}")
    return parsed


def clamp_window(calendar: Any, start: date | str, end: date | str) -> tuple[date, date] | None:
    """Intersect ``[start, end]`` with the calendar's session bounds.

    ``exchange_calendars`` raises ``DateOutOfBounds`` for a start before its
    first session (e.g. ``--start 2006-01-01`` while ``XNYS`` only begins on
    2006-10-02), so the request is *intersected* instead of passed through.

    This is a real intersection, not a shift: a request that lies entirely
    outside the calendar (say 1990) yields ``None``, whereas shifting it would
    silently turn a nonsense request into a download of the first few sessions
    of 2006.
    """
    first, last = calendar_bounds(calendar)
    lo = max(_as_date(start, "start"), first)
    hi = min(_as_date(end, "end"), last)
    if hi < lo:
        return None
    return lo, hi


def is_session(calendar: Any, value: date | str) -> bool:
    """``True`` when ``value`` is a trading day for ``calendar``."""
    day = _as_date(value, "value")
    first, last = calendar_bounds(calendar)
    if day < first or day > last:
        return False
    return bool(calendar.is_session(pd.Timestamp(day)))


def sessions_in_range(calendar: Any, start: date | str, end: date | str) -> list[date]:
    """Sorted NYSE sessions inside ``[start, end]`` (clamped to the calendar).

    Returns an empty list when the window holds no session at all, which is the
    signal for "nothing to download" rather than an error.
    """
    window = clamp_window(calendar, start, end)
    if window is None:
        return []
    lo, hi = window
    index: pd.DatetimeIndex = calendar.sessions_in_range(lo, hi)
    # `date` inputs make `exchange_calendars` return a *naive* DatetimeIndex.
    return [stamp.date() for stamp in index]


def last_sessions(calendar: Any, end: date | str, count: int) -> list[date]:
    """The ``count`` most recent sessions ending at or before ``end``.

    Returns fewer sessions (or none) when the calendar starts later than the
    look-back window.
    """
    if count <= 0:
        return []
    anchor = min(_as_date(end, "end"), calendar_bounds(calendar)[1])
    first = calendar_bounds(calendar)[0]
    if anchor < first:
        return []
    guess = anchor - timedelta(days=count * _CALENDAR_DAYS_PER_SESSION + 14)
    candidates = sessions_in_range(calendar, max(guess, first), anchor)
    return candidates[-count:]


def session_index(sessions: list[date]) -> dict[date, int]:
    """Map each session to its position in the ordered session list."""
    return {session: position for position, session in enumerate(sessions)}


def next_session_after(sessions: list[date], value: date | str) -> date | None:
    """First session strictly after ``value`` (``None`` when there is none).

    Used to decide whether two download windows are *adjacent on the trading
    calendar* and can therefore be fetched with a single request.
    """
    position = bisect.bisect_right(sessions, _as_date(value, "value"))
    return sessions[position] if position < len(sessions) else None


__all__ = [
    "DEFAULT_CALENDAR_NAME",
    "calendar_bounds",
    "clamp_window",
    "get_calendar",
    "is_session",
    "last_sessions",
    "next_session_after",
    "session_index",
    "sessions_in_range",
]
