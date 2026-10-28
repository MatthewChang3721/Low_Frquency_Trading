"""Which sessions a multi-date research run decides on: a pure date rule.

What this module is
-------------------
One question, answered without touching the disk or a single price:

    *given a requested window and a cadence, which sessions does the study
    decide on?*

::

    schedule = decision_schedule(
        "2024-01-01", "2024-12-31", frequency="weekly", anchor_weekday="THU"
    )
    schedule.dates            # (date(2024, 1, 4), date(2024, 1, 11), ...)
    schedule.anchor_weekday   # 'THU'
    schedule.biweekly_anchor  # None
    schedule.calendar_name    # 'XNYS'

``select_decision_dates(...)``
    the same answer as a plain ``tuple[date, ...]``, for a caller that does not
    need to describe the rule it used
``decision_schedule(...)``
    the answer *plus* the rule that produced it (:class:`DecisionSchedule`)
``DecisionSchedule``
    a frozen record of one selection: the requested window, the cadence, the
    anchor (``None`` for ``monthly``), the calendar, the biweekly phase and the
    sessions themselves -- and it re-checks every date against that calendar
    when it is built, so even a hand-built record cannot claim a weekend or a
    holiday as a decision day
``FactorScheduleError``
    a cadence, anchor, window or calendar that cannot be honoured

The three cadences
------------------
``weekly``
    One session per calendar week (ISO weeks, Monday to Sunday): the *anchor
    weekday* if the exchange trades that day, otherwise the closest session
    **before** it.  A Thursday anchor in Thanksgiving week is therefore the
    Wednesday, and a Wednesday anchor in Christmas week is the Tuesday::

        THU anchor, week of 2024-11-25 -> 2024-11-28 is closed -> 2024-11-27
        WED anchor, week of 2024-12-23 -> 2024-12-25 is closed -> 2024-12-24

``biweekly``
    The weekly candidates, keeping every second one starting with the first
    candidate *inside the requested window* -- never "the first week of the
    year", never "this week".  The phase is a pure function of
    ``(start_date, end_date, frequency, anchor_weekday, calendar_name)``, so the
    same request always selects the same days, and the record says which rule
    was used: ``biweekly_anchor`` is
    :data:`BIWEEKLY_ANCHOR` (``"first_selected_week_in_requested_range"``).

``monthly``
    The last session of every calendar month that overlaps the window.  There
    is no anchor to choose, so ``anchor_weekday`` is ``None`` in the result and
    passing one is an error instead of a silently ignored flag.

The window is a closed interval
-------------------------------
``start_date`` and ``end_date`` are inclusive, and a selected day is always
inside them: a candidate that would fall before ``start_date`` (a Monday anchor
requested from a Tuesday, a holiday Thursday whose previous session is before
the window) or after ``end_date`` is **dropped**, never clamped into range --
a study must not decide on a day it did not ask for.  A window can therefore
select nothing at all, which is reported as an empty tuple of dates rather than
as an error.

Sessions, not holidays
----------------------
Every boundary (weekend, exchange holiday, month end, year end) is decided by
:mod:`data_sys.market_calendar`, the project's single source of truth for "a
trading day".  This module keeps no holiday list and never guesses: a date it
returns is a session that calendar actually has, and a window the calendar does
not cover contributes no dates (nothing is fabricated from before the
calendar's first session).  The same test runs again whenever a
:class:`DecisionSchedule` is built by hand, so a record cannot claim a day the
exchange was closed.

What this module is not
-----------------------
* **not a factor, a runner or a store**: it computes no value, reads no bars and
  writes no file -- building a schedule costs one cached calendar lookup and
  nothing else;
* **not a batch runner**: selecting the dates and running them is the caller's
  job (``scripts/run_factor_batch.py``), so the IC and quantile entry points can
  reuse exactly this rule for exactly these dates;
* **not a price calendar**: it never decides *whether* a session had data, only
  which sessions exist.

Examples
--------
>>> schedule = decision_schedule("2024-11-25", "2024-11-29", anchor_weekday="THU")
>>> [day.isoformat() for day in schedule.dates]
['2024-11-27']
>>> schedule.frequency, schedule.anchor_weekday
('weekly', 'THU')
>>> select_decision_dates("2024-12-01", "2024-12-31", frequency="monthly")
(datetime.date(2024, 12, 31),)
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

from data_sys.errors import CalendarError
from data_sys.market_calendar import (
    DEFAULT_CALENDAR_NAME,
    calendar_bounds,
    get_calendar,
    last_sessions,
    sessions_in_range,
)
from factor_sys.errors import FactorScheduleError

#: One session per calendar week (the anchor weekday, or the session before it).
WEEKLY = "weekly"

#: Every second weekly candidate, phased on the first candidate in the window.
BIWEEKLY = "biweekly"

#: The last session of every calendar month overlapping the window.
MONTHLY = "monthly"

#: Every cadence this module selects; anything else is refused by name.
FREQUENCIES: tuple[str, ...] = (WEEKLY, BIWEEKLY, MONTHLY)

#: The cadence a caller gets when it does not ask for one.
DEFAULT_FREQUENCY: str = WEEKLY

#: Every anchor weekday the calendar definition can name, in week order.
ANCHOR_WEEKDAYS: tuple[str, ...] = ("MON", "TUE", "WED", "THU", "FRI")

#: The anchor a weekly/biweekly schedule uses when none is given.
DEFAULT_ANCHOR_WEEKDAY: str = "THU"

#: How the biweekly phase is anchored, recorded verbatim in every manifest.
BIWEEKLY_ANCHOR: str = "first_selected_week_in_requested_range"

#: Offset of each anchor weekday inside an ISO week (Monday is ``0``).
_WEEKDAY_OFFSETS: dict[str, int] = {"MON": 0, "TUE": 1, "WED": 2, "THU": 3, "FRI": 4}

#: The strict shape a date string must have before it is parsed at all.
_ISO_DAY = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")


# ---------------------------------------------------------------------------
# One selection, with the rule that produced it
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DecisionSchedule:
    """One requested cadence resolved into concrete sessions.

    The object is immutable and self-checking: it is the record a manifest (or a
    later IC / quantile run) can quote to prove *which* days a study used and
    *why*, so a hand-built or half-updated instance that contradicts itself is
    refused at construction instead of quietly steering an analysis.

    Parameters
    ----------
    start_date / end_date:
        The requested window, inclusive, already normalized to plain dates.
    frequency:
        One of :data:`FREQUENCIES`.
    anchor_weekday:
        ``"MON"``..``"FRI"`` for ``weekly``/``biweekly``, and ``None`` for
        ``monthly`` -- which has no anchor and never pretends to have one.
    calendar_name:
        The trading calendar the sessions were taken from.
    biweekly_anchor:
        :data:`BIWEEKLY_ANCHOR` for ``biweekly``, ``None`` otherwise: the phase
        of the every-second-week rule is part of the result, not a hidden
        assumption.
    dates:
        The selected sessions: strictly ascending, free of duplicates, all
        inside ``[start_date, end_date]`` and every one a real session of
        ``calendar_name`` -- the last point is checked against the calendar on
        construction, so a hand-built record cannot carry a weekend, a holiday
        or a date the calendar does not cover.

    Raises
    ------
    FactorScheduleError
        Any field that contradicts the rest (a monthly schedule with an anchor,
        dates out of order or out of the window, a date that is not a session of
        ``calendar_name``, a calendar that cannot be loaded).

    Examples
    --------
    >>> schedule = decision_schedule("2024-01-01", "2024-01-12")
    >>> schedule.dates[0], len(schedule)
    (datetime.date(2024, 1, 4), 2)
    >>> "2024-01-11" in schedule
    True
    """

    start_date: date
    end_date: date
    frequency: str
    anchor_weekday: str | None
    calendar_name: str
    biweekly_anchor: str | None
    dates: tuple[date, ...]

    def __post_init__(self) -> None:
        problems: list[str] = []

        for label, value in (("start_date", self.start_date), ("end_date", self.end_date)):
            if isinstance(value, datetime) or not isinstance(value, date):
                problems.append(f"{label} must be a plain date, got {value!r}")
        window_ok = not problems
        if window_ok and self.end_date < self.start_date:
            problems.append(
                f"start_date ({self.start_date.isoformat()}) is after end_date "
                f"({self.end_date.isoformat()})"
            )
            window_ok = False

        if self.frequency not in FREQUENCIES:
            problems.append(
                f"frequency must be one of {_quoted(FREQUENCIES)}, got {self.frequency!r}"
            )
        elif self.frequency == MONTHLY:
            if self.anchor_weekday is not None:
                problems.append(
                    "a monthly schedule takes the last session of each month and cannot carry "
                    f"the anchor {self.anchor_weekday!r}; its anchor_weekday is None"
                )
            if self.biweekly_anchor is not None:
                problems.append(
                    f"a monthly schedule is not phased by {BIWEEKLY_ANCHOR!r}; its "
                    "biweekly_anchor is None"
                )
        else:
            if self.anchor_weekday not in ANCHOR_WEEKDAYS:
                problems.append(
                    f"anchor_weekday must be one of {_quoted(ANCHOR_WEEKDAYS)} for a "
                    f"{self.frequency} schedule, got {self.anchor_weekday!r}"
                )
            expected = BIWEEKLY_ANCHOR if self.frequency == BIWEEKLY else None
            if self.biweekly_anchor != expected:
                problems.append(
                    f"biweekly_anchor must be {expected!r} for a {self.frequency} schedule, "
                    f"got {self.biweekly_anchor!r}"
                )

        calendar_ok = isinstance(self.calendar_name, str) and bool(self.calendar_name.strip())
        if not calendar_ok:
            problems.append(
                f"calendar_name must be a non-empty trading calendar name, got "
                f"{self.calendar_name!r}"
            )

        dates_ok = isinstance(self.dates, tuple)
        if not dates_ok:
            problems.append(f"dates must be a tuple of dates, got {type(self.dates).__name__}")
        else:
            previous: date | None = None
            for day in self.dates:
                if isinstance(day, datetime) or not isinstance(day, date):
                    problems.append(f"every selected date must be a plain date, got {day!r}")
                    dates_ok = False
                    break
                if previous is not None and day <= previous:
                    problems.append(
                        "selected dates must be strictly ascending and free of duplicates, but "
                        f"{day.isoformat()} follows {previous.isoformat()}"
                    )
                    dates_ok = False
                    break
                previous = day

            if dates_ok and window_ok and self.dates:
                if self.dates[0] < self.start_date or self.dates[-1] > self.end_date:
                    problems.append(
                        f"selected dates must lie inside {self.start_date.isoformat()}.."
                        f"{self.end_date.isoformat()}, got "
                        f"{self.dates[0].isoformat()}..{self.dates[-1].isoformat()}"
                    )
                    dates_ok = False

        # A record that calls itself a list of sessions has to prove it: the
        # calendar is asked again, so a weekend, an exchange holiday or a date
        # the calendar never covered cannot be quoted as a decision day.
        if dates_ok and calendar_ok and self.dates:
            problems.extend(self._non_session_problems())

        if problems:
            raise FactorScheduleError(
                "the decision schedule contradicts itself: " + "; ".join(problems)
            )

    def _non_session_problems(self) -> list[str]:
        """Every selected date that is not a session of ``calendar_name``."""
        try:
            trading_calendar = get_calendar(self.calendar_name)
            sessions = set(sessions_in_range(trading_calendar, self.dates[0], self.dates[-1]))
        except (CalendarError, ValueError) as exc:
            return [
                "the selected dates cannot be checked against the trading calendar "
                f"{self.calendar_name!r}: {exc}"
            ]

        missing = [day for day in self.dates if day not in sessions]
        if not missing:
            return []
        shown = ", ".join(_describe_session(day) for day in missing[:5])
        more = f" (and {len(missing) - 5} more)" if len(missing) > 5 else ""
        return [
            f"selected dates must all be trading sessions of {self.calendar_name}, but "
            f"these are not: {shown}{more}"
        ]

    def __len__(self) -> int:
        """How many decisions this schedule selects."""
        return len(self.dates)

    def __contains__(self, day: object) -> bool:
        """``day in schedule``; a plain date or a strict ISO string both work."""
        if isinstance(day, str):
            try:
                day = date.fromisoformat(day.strip())
            except ValueError:
                return False
        return isinstance(day, date) and not isinstance(day, datetime) and day in self.dates

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(frequency={self.frequency!r}, "
            f"anchor_weekday={self.anchor_weekday!r}, "
            f"calendar_name={self.calendar_name!r}, "
            f"start_date={self.start_date.isoformat()!r}, "
            f"end_date={self.end_date.isoformat()!r}, dates={len(self.dates)})"
        )


# ---------------------------------------------------------------------------
# The public selection
# ---------------------------------------------------------------------------
def decision_schedule(
    start_date: date | str,
    end_date: date | str,
    *,
    frequency: str = DEFAULT_FREQUENCY,
    anchor_weekday: str | None = None,
    calendar_name: str | None = None,
) -> DecisionSchedule:
    """Resolve one requested cadence into the sessions a study decides on.

    Parameters
    ----------
    start_date / end_date:
        The closed window, as plain :class:`datetime.date` values or strict
        ISO-8601 ``"YYYY-MM-DD"`` strings.  ``start_date`` after ``end_date`` is
        refused; a window that simply holds no decision day is not.
    frequency:
        ``"weekly"`` (the default), ``"biweekly"`` or ``"monthly"`` -- see the
        module docstring for exactly what each one selects.
    anchor_weekday:
        ``"MON"``..``"FRI"``, the weekday a weekly/biweekly candidate probes.
        ``None`` (the default) means :data:`DEFAULT_ANCHOR_WEEKDAY` (``"THU"``),
        and it is the only accepted value for ``monthly``: a monthly schedule
        has no anchor, so passing one is an error rather than a silently
        ignored flag.
    calendar_name:
        Trading calendar that decides what a session is.  ``None`` means
        :data:`data_sys.market_calendar.DEFAULT_CALENDAR_NAME`.

    Returns
    -------
    DecisionSchedule
        The window, cadence, anchor, calendar, biweekly phase and the selected
        sessions -- immutable, strictly ascending, free of duplicates and all
        inside the window (possibly empty).

    Raises
    ------
    FactorScheduleError
        An unusable window, cadence, anchor or calendar; an anchor given for a
        monthly cadence.

    Examples
    --------
    >>> schedule = decision_schedule(
    ...     "2024-01-01", "2024-01-31", frequency="biweekly", anchor_weekday="THU"
    ... )
    >>> [day.isoformat() for day in schedule.dates]
    ['2024-01-04', '2024-01-18']
    >>> schedule.biweekly_anchor
    'first_selected_week_in_requested_range'
    """
    start = _as_date(start_date, "start_date")
    end = _as_date(end_date, "end_date")
    if end < start:
        raise FactorScheduleError(
            f"start_date ({start.isoformat()}) must not be after end_date "
            f"({end.isoformat()}): the window is a closed interval, so an inverted one is a "
            "typo rather than a request for nothing"
        )

    cadence = _checked_frequency(frequency)
    anchor = _checked_anchor(cadence, anchor_weekday)
    label, trading_calendar = _calendar(calendar_name)

    if cadence == MONTHLY:
        dates = _monthly_sessions(start, end, trading_calendar)
    else:
        candidates = _weekly_sessions(start, end, anchor, trading_calendar)
        dates = candidates if cadence == WEEKLY else candidates[::2]

    return DecisionSchedule(
        start_date=start,
        end_date=end,
        frequency=cadence,
        anchor_weekday=anchor,
        calendar_name=label,
        biweekly_anchor=BIWEEKLY_ANCHOR if cadence == BIWEEKLY else None,
        dates=dates,
    )


def select_decision_dates(
    start_date: date | str,
    end_date: date | str,
    *,
    frequency: str = DEFAULT_FREQUENCY,
    anchor_weekday: str | None = None,
    calendar_name: str | None = None,
) -> tuple[date, ...]:
    """The sessions :func:`decision_schedule` selects, without the bookkeeping.

    Identical arguments and identical refusals; this is a convenience for a
    caller that only needs the dates (an IC or quantile batch, a test, a quick
    check).  Anything that has to *record* the rule -- a manifest -- calls
    :func:`decision_schedule` instead, because the anchor and the biweekly
    phase are part of the answer.
    """
    return decision_schedule(
        start_date,
        end_date,
        frequency=frequency,
        anchor_weekday=anchor_weekday,
        calendar_name=calendar_name,
    ).dates


# ---------------------------------------------------------------------------
# The rules themselves
# ---------------------------------------------------------------------------
def _weekly_sessions(
    start: date, end: date, anchor: str, trading_calendar: Any
) -> tuple[date, ...]:
    """One session per ISO week in the window: the anchor or the session before it.

    A probe the calendar does not cover is skipped outright (a request from
    before the calendar's first session selects nothing until the calendar
    starts), and a resolved session outside ``[start, end]`` is dropped rather
    than clamped: the study only decides on days it asked for.
    """
    offset = _WEEKDAY_OFFSETS[anchor]
    first, last = calendar_bounds(trading_calendar)
    probe = start - timedelta(days=start.weekday()) + timedelta(days=offset)

    selected: list[date] = []
    while probe <= end:
        if first <= probe <= last:
            sessions = last_sessions(trading_calendar, probe, 1)
            if sessions and start <= sessions[0] <= end:
                selected.append(sessions[0])
        probe += timedelta(days=7)
    return tuple(selected)


def _monthly_sessions(start: date, end: date, trading_calendar: Any) -> tuple[date, ...]:
    """The last session of every calendar month overlapping ``[start, end]``.

    Each month is asked for its own sessions through the calendar, so weekends,
    holidays and a month end the exchange does not trade on are decided by the
    calendar -- never by arithmetic on the 28th/30th/31st.
    """
    selected: list[date] = []
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        last_day = calendar.monthrange(year, month)[1]
        sessions = sessions_in_range(
            trading_calendar, date(year, month, 1), date(year, month, last_day)
        )
        if sessions and start <= sessions[-1] <= end:
            selected.append(sessions[-1])
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return tuple(selected)


# ---------------------------------------------------------------------------
# Argument checks -- one readable refusal per rule
# ---------------------------------------------------------------------------
def _as_date(value: Any, label: str) -> date:
    """``value`` as a plain date, or a clear refusal.

    A ``datetime`` is refused even though it *is* a ``date``: a decision day is
    a whole session, and silently dropping a time of day would let this module
    decide which session the caller meant.
    """
    if isinstance(value, datetime):
        raise FactorScheduleError(
            f"{label} must be a date or a strict ISO 'YYYY-MM-DD' string, got the datetime "
            f"{value!r}: a decision day is a whole session, pass {label}.date()"
        )
    if isinstance(value, date):
        return value
    if isinstance(value, str) and _ISO_DAY.fullmatch(value.strip()):
        try:
            return date.fromisoformat(value.strip())
        except ValueError:
            pass  # right shape, but no such day (e.g. '2024-02-30')
    raise FactorScheduleError(
        f"{label} must be a date or a strict ISO 'YYYY-MM-DD' string, got {value!r}"
    )


def _checked_frequency(value: Any) -> str:
    """``value`` as a known cadence, or a refusal that explains all three."""
    if not isinstance(value, str) or value not in FREQUENCIES:
        raise FactorScheduleError(
            f"frequency must be one of {_quoted(FREQUENCIES)}, got {value!r}: 'weekly' anchors "
            "one weekday per week, 'biweekly' keeps every second of those weeks and 'monthly' "
            "takes the last session of each month"
        )
    return value


def _checked_anchor(cadence: str, anchor_weekday: Any) -> str | None:
    """The effective anchor for ``cadence``: a weekday, or ``None`` for monthly."""
    if cadence == MONTHLY:
        if anchor_weekday is not None:
            raise FactorScheduleError(
                "anchor_weekday does not apply to a monthly schedule: it already selects the "
                f"last session of each month, so pass anchor_weekday=None (or omit it) instead "
                f"of {anchor_weekday!r}"
            )
        return None
    if anchor_weekday is None:
        return DEFAULT_ANCHOR_WEEKDAY
    if not isinstance(anchor_weekday, str) or anchor_weekday not in ANCHOR_WEEKDAYS:
        raise FactorScheduleError(
            f"anchor_weekday must be one of {_quoted(ANCHOR_WEEKDAYS)} or None, got "
            f"{anchor_weekday!r}: a weekly schedule probes exactly one weekday per week"
        )
    return anchor_weekday


def _calendar(calendar_name: Any) -> tuple[str, Any]:
    """Resolve the trading calendar by name: ``(name, calendar)`` or a refusal.

    The calendar belongs to the data system, so it is asked for by name instead
    of being rebuilt here; only the failure is translated, so an unknown name is
    one readable message instead of a data-pipeline exception.
    """
    if calendar_name is None:
        name = DEFAULT_CALENDAR_NAME
    elif isinstance(calendar_name, str) and calendar_name.strip():
        name = calendar_name.strip()
    else:
        raise FactorScheduleError(
            f"calendar_name must be a non-empty trading calendar name such as "
            f"{DEFAULT_CALENDAR_NAME!r}, got {calendar_name!r}"
        )
    try:
        return name, get_calendar(name)
    except CalendarError as exc:
        raise FactorScheduleError(f"could not load the trading calendar {name!r}: {exc}") from exc


def _quoted(values: tuple[str, ...]) -> str:
    """``values`` as a comma-separated list of quoted names, for one message."""
    return ", ".join(repr(value) for value in values)


def _describe_session(day: date) -> str:
    """``day`` as ``2024-01-06 (SAT)``: the weekday is the point of the message."""
    return f"{day.isoformat()} ({calendar.day_abbr[day.weekday()].upper()})"


__all__ = [
    "ANCHOR_WEEKDAYS",
    "BIWEEKLY",
    "BIWEEKLY_ANCHOR",
    "DEFAULT_ANCHOR_WEEKDAY",
    "DEFAULT_FREQUENCY",
    "FREQUENCIES",
    "MONTHLY",
    "WEEKLY",
    "DecisionSchedule",
    "FactorScheduleError",
    "decision_schedule",
    "select_decision_dates",
]
