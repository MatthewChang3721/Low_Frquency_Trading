"""When a replay decides and when it trades: the calendar a rebalance is scheduled on.

The one question
----------------
::

    policy = RebalancePolicy()                        # weekly, Thursday close, next open
    schedule = rebalance_schedule("2024-01-01", "2024-12-31", policy=policy)

    schedule.decision_dates[0]           # date(2024, 1, 4) -- the first Thursday close
    schedule.events[0]                   # RebalanceEvent(2024-01-04 -> 2024-01-05)
    schedule.unexecuted_decision_dates   # decisions whose fill is past the window

*Which sessions does this window decide on, and in which session's open does each
decision trade?*  That is all this module answers: it reads no price, builds no
order, weighs no target and replays nothing.

The timeline it describes
-------------------------
The project's default is a rule, not a preference::

    Thursday close   the decision is taken, from data known at that close
    next open        the orders decided on Thursday execute in the next session

Two halves of that sentence are enforced here:

* the decision day is a real session -- if the exchange is closed on the anchor
  weekday, the decision moves to the **previous** session of the same week, so
  Thanksgiving 2024 decides on Wednesday the 27th, never on the closed Thursday
  28th;
* the execution day is a **later** session -- if the session after the decision
  is a holiday, the fill waits for the exchange to open, so Good Friday 2024
  decides on Thursday the 28th and fills on Monday April 1st.

:attr:`RebalancePolicy.execution_delay_sessions` generalises the second half from
one session to ``n``: a Thursday decision with a delay of two fills on Monday.
:attr:`RebalancePolicy.execution_price` names the price convention a fill would
use; this version accepts exactly ``"open"`` and reads no price at all, so the
field *declares* what the execution model will do rather than doing it.

Where the dates come from
-------------------------
Nothing about weeks, months, holidays or biweekly phase is decided here:

* which sessions are decision candidates is
  :func:`factor_sys.factor_schedule.decision_schedule`, the project's single
  public rule for ``weekly`` / ``biweekly`` / ``monthly`` -- including the
  "move back to the previous session" behaviour above;
* which days are sessions at all is :mod:`data_sys.market_calendar`, asked by
  name (:func:`~data_sys.market_calendar.get_calendar` and
  :func:`~data_sys.market_calendar.sessions_in_range`).

This module adds exactly one step of its own: *given a decision session, which
session is ``n`` sessions later, and is it still inside the requested window?*
A second weekday table, holiday list or month-end rule would be a second answer
to a question that already has one.

The window is closed, and nothing is clamped
--------------------------------------------
``start_date`` and ``end_date`` are inclusive.  A decision whose execution falls
after ``end_date`` -- the ordinary case when a window ends on a decision Thursday
-- is neither pulled back into range nor dropped silently: it is listed in
:attr:`RebalanceSchedule.unexecuted_decision_dates`, while
:attr:`RebalanceSchedule.events` holds only the rebalances the window could
actually complete.  A window the calendar does not cover contributes no decision
at all, which is a legal, empty schedule rather than an error.

What this module is not
-----------------------
Not a price reader, not an order generator, not an execution or commission
model, not a ledger, not an engine and not a report.  It knows *when*; what is
bought, at which price, at what cost and what the account was worth afterwards
belong to the tasks that come after it.

Examples
--------
>>> policy = RebalancePolicy()
>>> policy.frequency, policy.anchor_weekday, policy.execution_delay_sessions
('weekly', 'THU', 1)
>>> thanksgiving = rebalance_schedule("2024-11-25", "2024-11-29")
>>> thanksgiving.decision_dates[0], thanksgiving.execution_dates[0]
(datetime.date(2024, 11, 27), datetime.date(2024, 11, 29))
>>> good_friday = rebalance_schedule("2024-03-25", "2024-04-05")
>>> good_friday.events[0]
RebalanceEvent(2024-03-28 -> 2024-04-01)
>>> rebalance_schedule("2024-01-01", "2024-01-04").unexecuted_decision_dates
(datetime.date(2024, 1, 4),)
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from datetime import date
from itertools import pairwise
from typing import Any

from backtest_sys.contracts import DEFAULT_CALENDAR_NAME, _describe, _session_date
from backtest_sys.errors import BacktestContractError, RebalancePolicyError
from data_sys.errors import CalendarError
from data_sys.market_calendar import calendar_bounds, get_calendar, sessions_in_range
from factor_sys.errors import FactorScheduleError
from factor_sys.factor_schedule import (
    ANCHOR_WEEKDAYS,
    DEFAULT_ANCHOR_WEEKDAY,
    DEFAULT_FREQUENCY,
    FREQUENCIES,
    MONTHLY,
    decision_schedule,
)

#: The only price convention this version can name: the open of the execution
#: session.  A string, not a reader -- nothing here opens a bar.
DEFAULT_EXECUTION_PRICE: str = "open"

#: Every price convention a policy may state, spelled exactly like this.
EXECUTION_PRICES: tuple[str, ...] = (DEFAULT_EXECUTION_PRICE,)

#: The delay a policy uses when none is given: decide at one close, fill at the
#: next session's open.
DEFAULT_EXECUTION_DELAY_SESSIONS: int = 1


# ---------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RebalancePolicy:
    """The rule that turns a window into decision sessions and execution sessions.

    A policy is a *plan*: it says which cadence decides, which weekday anchors a
    weekly decision, which calendar defines a session, how many sessions later the
    trade happens and at which price convention.  It says nothing about prices,
    weights, symbols or costs, and it loads no calendar -- whether a given window
    can honour it is answered by :func:`rebalance_schedule`, which is why that
    function, not this object, raises :class:`RebalancePolicyError`.

    Parameters
    ----------
    frequency:
        ``"weekly"`` (the default), ``"biweekly"`` or ``"monthly"`` -- exactly the
        vocabulary of :data:`factor_sys.factor_schedule.FREQUENCIES`, reused rather
        than restated, so a cadence means the same thing here as in a factor study.
    anchor_weekday:
        ``"MON"``..``"FRI"``, the weekday a weekly or biweekly decision probes.
        ``None`` means the project default,
        :data:`factor_sys.factor_schedule.DEFAULT_ANCHOR_WEEKDAY` (``"THU"``), and
        the object **normalizes it**: a default policy carries ``"THU"``, because a
        schedule has to be able to state which day it used.  A ``monthly`` policy
        has no anchor at all, so it must stay ``None`` and passing a weekday is
        refused instead of ignored -- the same refusal
        :func:`~factor_sys.factor_schedule.decision_schedule` makes.
    calendar_name:
        The trading calendar the sessions are looked up on; ``"XNYS"`` by default.
        The name is stored, never loaded: :func:`rebalance_schedule` asks
        :func:`~data_sys.market_calendar.get_calendar` for it when it needs it.  An
        empty name, or one padded with whitespace, is refused, because ``" XNYS"``
        and ``"XNYS"`` would silently be two different calendars.
    execution_delay_sessions:
        How many sessions after the decision the trade executes: a whole number of
        at least ``1``, ``1`` by default (decide at Thursday's close, fill at the
        next session's open).  ``bool``, ``float``, ``str``, ``0`` and negative
        numbers are refused: ``True`` is not one session, ``1.0`` is not a count of
        them, and ``0`` would fill at the decision's own close -- look-ahead rather
        than a delay.
    execution_price:
        The price convention a fill would use.  This version accepts exactly
        ``"open"``; ``"OPEN"``, ``"Open"`` and ``"close"`` are refused rather than
        folded into it, because the string decides which price an execution model
        reads.  No price is read, fetched or validated here.

    Raises
    ------
    backtest_sys.errors.BacktestContractError
        If a field is not the vocabulary above.  Every problem is collected, so one
        error names them all instead of one per run.

    Notes
    -----
    * Frozen and hashable, so a policy can key a cache or be quoted verbatim in a
      run record.
    * The default is the project timeline -- Thursday's close decides with
      Thursday's knowledge, the next session's open executes -- and an anchor
      holiday moves the decision to the previous session rather than the next.
    * ``execution_price`` is deliberately a closed vocabulary rather than free
      text: a typo would otherwise announce a different model.

    Examples
    --------
    >>> policy = RebalancePolicy()
    >>> policy == RebalancePolicy(
    ...     frequency="weekly",
    ...     anchor_weekday="THU",
    ...     calendar_name="XNYS",
    ...     execution_delay_sessions=1,
    ...     execution_price="open",
    ... )
    True
    >>> RebalancePolicy(frequency="monthly").anchor_weekday is None
    True
    >>> RebalancePolicy(anchor_weekday="WED").anchor_weekday
    'WED'
    >>> RebalancePolicy(execution_delay_sessions=2).execution_delay_sessions
    2
    """

    frequency: str = DEFAULT_FREQUENCY
    anchor_weekday: str | None = None
    calendar_name: str = DEFAULT_CALENDAR_NAME
    execution_delay_sessions: int = DEFAULT_EXECUTION_DELAY_SESSIONS
    execution_price: str = DEFAULT_EXECUTION_PRICE

    def __post_init__(self) -> None:
        problems: list[str] = []

        frequency = self.frequency
        known_frequency = isinstance(frequency, str) and frequency in FREQUENCIES
        if not known_frequency:
            problems.append(
                f"frequency must be one of {_quoted(FREQUENCIES)}, got {frequency!r}: 'weekly' "
                "anchors one weekday per week, 'biweekly' keeps every second of those weeks and "
                "'monthly' takes the last session of each month"
            )

        anchor: str | None = None
        requested_anchor = self.anchor_weekday
        if known_frequency and frequency == MONTHLY:
            if requested_anchor is not None:
                problems.append(
                    "anchor_weekday must be None for a monthly policy, got "
                    f"{requested_anchor!r}: monthly already takes the last session of each "
                    "month, so an anchor is refused rather than silently ignored"
                )
        else:
            anchor = DEFAULT_ANCHOR_WEEKDAY if requested_anchor is None else requested_anchor
            if not isinstance(anchor, str) or anchor not in ANCHOR_WEEKDAYS:
                problems.append(
                    f"anchor_weekday must be one of {_quoted(ANCHOR_WEEKDAYS)} or None, got "
                    f"{requested_anchor!r}: the anchor names the weekday a weekly decision "
                    f"probes, and None means {DEFAULT_ANCHOR_WEEKDAY!r}"
                )
                anchor = None

        name = self.calendar_name
        if not isinstance(name, str):
            problems.append(
                f"calendar_name must be a str, got {type(name).__name__}: the calendar is "
                "named here and loaded by the schedule, so the name has to be text"
            )
        elif not name.strip():
            problems.append(
                "calendar_name must not be empty: a schedule is measured on one named calendar "
                f"(the default is {DEFAULT_CALENDAR_NAME!r})"
            )
        elif name != name.strip():
            problems.append(
                f"calendar_name {name!r} has leading or trailing whitespace: the name is "
                "compared verbatim, so it is refused rather than trimmed"
            )

        delay = self.execution_delay_sessions
        if isinstance(delay, bool) or not isinstance(delay, int):
            problems.append(
                "execution_delay_sessions must be a whole number of sessions, got "
                f"{type(delay).__name__}: {delay!r}; True is not one session and 1.0 is not a "
                "count of them"
            )
            delay = None
        elif delay < 1:
            problems.append(
                f"execution_delay_sessions must be at least 1, got {delay!r}: a decision is "
                "taken at one close and can only execute in a later session, so a delay of 0 "
                "would fill before the decision was known"
            )
            delay = None

        price = self.execution_price
        if not isinstance(price, str) or price not in EXECUTION_PRICES:
            problems.append(
                f"execution_price must be {_quoted(EXECUTION_PRICES)}, got {price!r}: this "
                "version fills at the open of the session after the decision, so 'OPEN' and "
                "'Open' are refused rather than folded into it"
            )

        if problems:
            raise BacktestContractError("invalid RebalancePolicy", problems)

        object.__setattr__(self, "anchor_weekday", anchor)
        object.__setattr__(self, "execution_delay_sessions", delay)


# ---------------------------------------------------------------------------
# One decision, one execution
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RebalanceEvent:
    """One calendar fact: *this session's close decides, that session's open fills*.

    An event is deliberately the smallest statement of the timeline.  It holds two
    dates and nothing else: no price, no order, no share count, no strategy, no
    weight, no cost and no fill result.  Those belong to the objects a caller
    builds *from* an event, which is what lets the calendar be audited on its own.

    Parameters
    ----------
    decision_date:
        The session whose close takes the decision: a plain
        :class:`datetime.date` or a bare ISO-8601 ``"YYYY-MM-DD"`` string.  A
        ``datetime`` -- and therefore every ``pandas.Timestamp`` -- is refused even
        at midnight, and so is ``"2024-1-5"`` or ``"05/01/2024"``: a session is a
        whole day and is never guessed at.
    execution_date:
        The session whose open executes the trade, in the same two forms.  It must
        be **strictly later** than :attr:`decision_date`: a decision cannot fill
        before or on the close that took it.

    Raises
    ------
    backtest_sys.errors.BacktestContractError
        If a date is not a plain date or an ISO day, or the execution is not later
        than the decision.  Every problem is collected.

    Notes
    -----
    * An event does **not** load a calendar and does not claim its dates are
      sessions: whether Thursday the 28th actually traded is decided by
      :class:`RebalanceSchedule`, which checks every event against
      ``policy.calendar_name``.  The event only refuses a schedule that points
      backwards.
    * Frozen and hashable, so an event can be a key of a fill map.

    Examples
    --------
    >>> event = RebalanceEvent("2024-01-04", "2024-01-05")
    >>> event.decision_date, event.execution_date
    (datetime.date(2024, 1, 4), datetime.date(2024, 1, 5))
    >>> event
    RebalanceEvent(2024-01-04 -> 2024-01-05)
    >>> event == RebalanceEvent(date(2024, 1, 4), date(2024, 1, 5))
    True
    """

    decision_date: date
    execution_date: date

    def __post_init__(self) -> None:
        problems: list[str] = []

        decision: date | None = None
        execution: date | None = None
        try:
            decision = _session_date(self.decision_date, "RebalanceEvent.decision_date")
        except BacktestContractError as exc:
            problems.append(str(exc))
        try:
            execution = _session_date(self.execution_date, "RebalanceEvent.execution_date")
        except BacktestContractError as exc:
            problems.append(str(exc))

        if decision is not None and execution is not None and execution <= decision:
            problems.append(
                f"execution_date {execution.isoformat()} must be later than decision_date "
                f"{decision.isoformat()}: a decision taken at a close cannot execute before "
                "the next session's open"
            )

        if problems:
            raise BacktestContractError("invalid RebalanceEvent", problems)

        object.__setattr__(self, "decision_date", decision)
        object.__setattr__(self, "execution_date", execution)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"{self.decision_date.isoformat()} -> {self.execution_date.isoformat()})"
        )


# ---------------------------------------------------------------------------
# The whole window
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RebalanceSchedule:
    """One window resolved into decisions and the trades it could actually complete.

    The record answers the question of this module and stops there: *these sessions
    decide, these later sessions execute, and these decisions had to be left
    unexecuted*.  It is immutable and self-checking, so a hand-built or half-edited
    schedule cannot quietly steer a replay onto a day the exchange was closed, onto
    the wrong session or onto a decision the rule never made.

    Parameters
    ----------
    start_date / end_date:
        The requested window, inclusive, as plain dates or strict ISO days.
    policy:
        The :class:`RebalancePolicy` the window was resolved under; the schedule is
        meaningless without the rule that produced it.
    decision_dates:
        Every session inside the window on which the policy decides, strictly
        ascending, free of duplicates.  It must equal exactly what
        :func:`factor_sys.factor_schedule.decision_schedule` selects for the same
        window, cadence, anchor and calendar -- not one day more, not one fewer.
    events:
        The decisions whose execution fits inside the window, strictly ascending by
        ``(decision_date, execution_date)``: at most one event per decision date and
        no event for a decision outside :attr:`decision_dates`.

    What it refuses
    ---------------
    A schedule is checked in two passes, so one error names every defect:

    1. the window must not run backwards, and ``policy`` must be a
       :class:`RebalancePolicy`;
    2. ``decision_dates`` and ``events`` must be tuples -- a list is refused rather
       than copied, because the record claims to be immutable;
    3. ``decision_dates`` must be strictly ascending, duplicate-free and inside the
       window;
    4. every decision date must be a real session of ``policy.calendar_name``;
    5. ``decision_dates`` must be exactly the selection
       :func:`~factor_sys.factor_schedule.decision_schedule` makes for the same
       window, cadence, anchor and calendar;
    6. ``events`` must be strictly ascending by ``(decision_date, execution_date)``;
    7. every event's decision date must appear once in ``decision_dates``;
    8. every event's execution date must be the session **exactly**
       ``policy.execution_delay_sessions`` sessions after its decision date, and
       inside the window;
    9. every decision whose execution is inside the window must have exactly one
       event, and every decision whose execution is outside the window -- or has no
       session left on the calendar -- must have none, which is what
       :attr:`unexecuted_decision_dates` reports.

    Raises
    ------
    backtest_sys.errors.BacktestContractError
        If the record contradicts itself in any of the ways above.
    backtest_sys.errors.RebalancePolicyError
        If ``policy.calendar_name`` cannot be loaded, so the record cannot be
        checked at all.  An unknown calendar is not a malformed record: it is a
        rule the environment cannot satisfy.

    Notes
    -----
    * An **empty** schedule is legal: a window the calendar does not cover selects
      no decision, and a window with no selected decision produces no event.
    * ``events`` is not ``decision_dates`` truncated: a decision near the end of
      the window usually has no event, because its fill would land after the window
      closes.  Clamping that fill back into range would be a trade the replay never
      asked for.
    * Both histories are tuples of dates, never a dataframe or a set: a schedule is
      a small, ordered, immutable record that can be quoted in a report.

    Examples
    --------
    >>> schedule = rebalance_schedule("2024-01-01", "2024-01-12")
    >>> len(schedule.decision_dates), len(schedule.events)
    (2, 2)
    >>> schedule
    RebalanceSchedule(weekly THU on XNYS, 2024-01-01..2024-01-12, 2 decisions, 2 events)
    >>> schedule.execution_dates
    (datetime.date(2024, 1, 5), datetime.date(2024, 1, 12))
    >>> rebalance_schedule("2024-01-01", "2024-01-04").unexecuted_decision_dates
    (datetime.date(2024, 1, 4),)
    """

    start_date: date
    end_date: date
    policy: RebalancePolicy
    decision_dates: tuple[date, ...]
    events: tuple[RebalanceEvent, ...]

    def __post_init__(self) -> None:
        problems: list[str] = []

        window = _checked_window(self.start_date, self.end_date, problems)

        policy = self.policy if isinstance(self.policy, RebalancePolicy) else None
        if policy is None:
            problems.append(
                f"policy must be a RebalancePolicy, got {type(self.policy).__name__}: a "
                "schedule only means something together with the rule that produced it"
            )

        dates = _checked_decision_dates(self.decision_dates, window, problems)
        events = _checked_events(self.events, problems)

        # Two passes on purpose: a record whose own shape is broken is not worth
        # interrogating the calendar about, and the calendar complaints would only
        # bury the defect that caused them.
        if not problems and window is not None and policy is not None:
            problems.extend(_rule_problems(policy, window[0], window[1], dates, events))

        if problems:
            raise BacktestContractError("invalid RebalanceSchedule", problems)

        if window is not None:
            object.__setattr__(self, "start_date", window[0])
            object.__setattr__(self, "end_date", window[1])
        object.__setattr__(self, "decision_dates", dates)
        object.__setattr__(self, "events", events)

    @property
    def execution_dates(self) -> tuple[date, ...]:
        """The session each event fills in, in decision order; empty when none executes."""
        return tuple(event.execution_date for event in self.events)

    @property
    def unexecuted_decision_dates(self) -> tuple[date, ...]:
        """Decisions with no event: their fill lands after the window, or off the calendar."""
        executed = {event.decision_date for event in self.events}
        return tuple(day for day in self.decision_dates if day not in executed)

    def __len__(self) -> int:
        """How many rebalances this window can execute -- the number of events."""
        return len(self.events)

    def __repr__(self) -> str:
        policy = self.policy
        rule = (
            f"{policy.frequency} {policy.anchor_weekday or 'month-end'} on {policy.calendar_name}"
            if isinstance(policy, RebalancePolicy)
            else type(policy).__name__
        )
        decisions = len(self.decision_dates) if isinstance(self.decision_dates, tuple) else "?"
        fills = len(self.events) if isinstance(self.events, tuple) else "?"
        return (
            f"{type(self).__name__}({rule}, {_isoformat(self.start_date)}.."
            f"{_isoformat(self.end_date)}, {decisions} decisions, {fills} events)"
        )


# ---------------------------------------------------------------------------
# Small helpers: one wording per rule
# ---------------------------------------------------------------------------
def _quoted(values: tuple[str, ...]) -> str:
    """``values`` as a comma-separated list of quoted names, for one message."""
    return ", ".join(repr(value) for value in values)


def _isoformat(value: Any) -> str:
    """``value`` as ``YYYY-MM-DD`` when it is a date, otherwise as its repr.

    Used by :meth:`RebalanceSchedule.__repr__`, which may be asked to describe a
    record that never passed validation.
    """
    return value.isoformat() if isinstance(value, date) else repr(value)


def _window_date(value: Any, label: str) -> date:
    """``value`` as a plain date, refused as a policy error rather than a contract one.

    The rule itself -- a plain date or a strict ISO day, never a timestamp -- is the
    contract's, reused so both layers refuse exactly the same inputs.
    """
    try:
        return _session_date(value, label)
    except BacktestContractError as exc:
        raise RebalancePolicyError(str(exc)) from exc


def _checked_window(
    start_value: Any, end_value: Any, problems: list[str]
) -> tuple[date, date] | None:
    """The window as plain dates, or ``None`` (with problems) when it is unusable."""
    start: date | None = None
    end: date | None = None
    try:
        start = _session_date(start_value, "RebalanceSchedule.start_date")
    except BacktestContractError as exc:
        problems.append(str(exc))
    try:
        end = _session_date(end_value, "RebalanceSchedule.end_date")
    except BacktestContractError as exc:
        problems.append(str(exc))

    if start is None or end is None:
        return None
    if end < start:
        problems.append(
            f"start_date ({start.isoformat()}) is after end_date ({end.isoformat()}): the window "
            "is a closed interval, so an inverted one selects nothing by mistake"
        )
        return None
    return start, end


def _checked_decision_dates(
    values: Any, window: tuple[date, date] | None, problems: list[str]
) -> tuple[date, ...]:
    """``values`` as the schedule's decision dates, or ``()`` when they are not dates."""
    if not isinstance(values, tuple):
        problems.append(
            f"decision_dates must be a tuple[date, ...], got {type(values).__name__}: the "
            "record is immutable, so a list is refused rather than copied"
        )
        return ()

    dates: list[date] = []
    for value in values:
        try:
            dates.append(_session_date(value, "RebalanceSchedule.decision_dates entry"))
        except BacktestContractError as exc:
            problems.append(str(exc))
            return ()

    for index in range(1, len(dates)):
        if dates[index] <= dates[index - 1]:
            problems.append(
                "decision_dates must be strictly ascending and free of duplicates, but "
                f"{dates[index].isoformat()} follows {dates[index - 1].isoformat()}"
            )
            break

    if window is not None and dates:
        start, end = window
        outside = [day.isoformat() for day in dates if day < start or day > end]
        if outside:
            problems.append(
                "every decision date must lie inside "
                f"{start.isoformat()}..{end.isoformat()}, but {_describe(outside)} does not"
            )
    return tuple(dates)


def _checked_events(values: Any, problems: list[str]) -> tuple[RebalanceEvent, ...]:
    """``values`` as the schedule's events, or ``()`` when the field is not a tuple."""
    if not isinstance(values, tuple):
        problems.append(
            f"events must be a tuple[RebalanceEvent, ...], got {type(values).__name__}: the "
            "record is immutable, so a list is refused rather than copied"
        )
        return ()

    unknown = [entry for entry in values if not isinstance(entry, RebalanceEvent)]
    if unknown:
        problems.append(f"every events entry must be a RebalanceEvent, got {_describe(unknown)}")
    events = tuple(entry for entry in values if isinstance(entry, RebalanceEvent))

    for previous, current in pairwise(events):
        if (current.decision_date, current.execution_date) <= (
            previous.decision_date,
            previous.execution_date,
        ):
            problems.append(
                "events must be strictly ascending by (decision_date, execution_date), but "
                f"{current.decision_date.isoformat()} -> {current.execution_date.isoformat()} "
                f"follows {previous.decision_date.isoformat()} -> "
                f"{previous.execution_date.isoformat()}"
            )
            break

    counts: dict[date, int] = {}
    for event in events:
        counts[event.decision_date] = counts.get(event.decision_date, 0) + 1
    duplicated = [day.isoformat() for day, count in counts.items() if count > 1]
    if duplicated:
        problems.append(
            "each decision date may appear in at most one event, but "
            f"{_describe(duplicated)} appears more than once"
        )
    return events


def _sessions_from(calendar: Any, start: date) -> list[date]:
    """Every session from ``start`` to the calendar's last one, in order.

    One sorted list serves the whole window, so "which session is ``n`` sessions
    later" is a lookup instead of a calendar call per decision -- and a decision
    after the calendar's last session is simply absent from it.
    """
    return sessions_in_range(calendar, start, calendar_bounds(calendar)[1])


def _execution_session(sessions: list[date], decision: date, delay: int) -> date | None:
    """The session ``delay`` sessions after ``decision``, or ``None`` if there is none.

    ``None`` covers both "the decision is not a session" and "the calendar ends
    before the delay is spent"; the caller has already ruled out the first.
    """
    position = bisect.bisect_left(sessions, decision)
    if position == len(sessions) or sessions[position] != decision:
        return None
    target = position + delay
    return sessions[target] if target < len(sessions) else None


def _rule_problems(
    policy: RebalancePolicy,
    start: date,
    end: date,
    dates: tuple[date, ...],
    events: tuple[RebalanceEvent, ...],
) -> list[str]:
    """What the calendar and the factor rule say about an otherwise well-shaped record."""
    problems: list[str] = []

    try:
        calendar = get_calendar(policy.calendar_name)
    except CalendarError as exc:
        raise RebalancePolicyError(
            f"could not load the trading calendar {policy.calendar_name!r} needed to check "
            f"{start.isoformat()}..{end.isoformat()}: {exc}"
        ) from exc

    sessions = _sessions_from(calendar, start)
    session_set = set(sessions)
    missing = [day.isoformat() for day in dates if day not in session_set]
    if missing:
        problems.append(
            f"every decision date must be a trading session of {policy.calendar_name}, but "
            f"{_describe(missing)} is not"
        )

    try:
        selected = decision_schedule(
            start,
            end,
            frequency=policy.frequency,
            anchor_weekday=policy.anchor_weekday,
            calendar_name=policy.calendar_name,
        ).dates
    except FactorScheduleError as exc:
        raise RebalancePolicyError(
            f"the {policy.frequency!r} rule on {policy.calendar_name!r} cannot be resolved for "
            f"{start.isoformat()}..{end.isoformat()}: {exc}"
        ) from exc

    # A date that is not a session has already been named above; calling it missing
    # from the rule as well would report one defect twice.
    if missing:
        return problems
    if dates != selected:
        listed = set(dates)
        chosen = set(selected)
        problems.append(
            "decision_dates must be exactly the sessions decision_schedule("
            f"{start.isoformat()}, {end.isoformat()}, frequency={policy.frequency!r}, "
            f"anchor_weekday={policy.anchor_weekday!r}, calendar_name={policy.calendar_name!r}) "
            f"selects, but {_describe([day.isoformat() for day in selected if day not in listed])}"
            f" is missing and {_describe([day.isoformat() for day in dates if day not in chosen])}"
            " does not belong"
        )
        return problems

    listed_dates = set(dates)
    foreign = [
        event.decision_date.isoformat()
        for event in events
        if event.decision_date not in listed_dates
    ]
    if foreign:
        problems.append(
            "every event must quote a decision date listed in decision_dates, but "
            f"{_describe(foreign)} is not one of them"
        )

    delay = policy.execution_delay_sessions
    executions = {day: _execution_session(sessions, day, delay) for day in dates}
    recorded = {event.decision_date: event.execution_date for event in events}
    for day in dates:
        execution = executions[day]
        recorded_execution = recorded.get(day)
        if execution is not None and execution <= end:
            if recorded_execution is None:
                problems.append(
                    f"decision {day.isoformat()} needs an event: {delay} session(s) later is "
                    f"{execution.isoformat()}, which is inside {start.isoformat()}.."
                    f"{end.isoformat()}"
                )
            elif recorded_execution != execution:
                problems.append(
                    f"event {day.isoformat()} -> {recorded_execution.isoformat()} does not "
                    f"follow the policy: {delay} session(s) after {day.isoformat()} is "
                    f"{execution.isoformat()}"
                )
        elif recorded_execution is not None:
            reach = (
                f"falls on {execution.isoformat()}, after the window ends"
                if execution is not None
                else "has no session left on the calendar"
            )
            problems.append(
                f"decision {day.isoformat()} must not carry an event: {delay} session(s) later "
                f"{reach}, so the decision stays unexecuted"
            )
    return problems


# ---------------------------------------------------------------------------
# The public entry point
# ---------------------------------------------------------------------------
def rebalance_schedule(
    start_date: date | str,
    end_date: date | str,
    *,
    policy: RebalancePolicy | None = None,
) -> RebalanceSchedule:
    """Resolve a closed window into decision sessions and the trades it can complete.

    Steps, in order: the window is parsed into plain dates; the decision sessions
    come from :func:`factor_sys.factor_schedule.decision_schedule`; the trading
    calendar named by the policy is loaded once; and every decision is paired with
    the session ``execution_delay_sessions`` sessions after it, but only when that
    session is still inside the window.  The result is a
    :class:`RebalanceSchedule` that has already re-checked itself, so a caller can
    trust the record without repeating any of these steps.

    Parameters
    ----------
    start_date / end_date:
        The window, inclusive, as plain :class:`datetime.date` objects or strict
        ISO-8601 ``"YYYY-MM-DD"`` strings.  Nothing is snapped to a session: the
        selected decisions are the rule's own, and a window that ends mid-week has
        simply fewer of them.
    policy:
        The :class:`RebalancePolicy` to resolve; ``None`` builds a fresh
        :class:`RebalancePolicy` -- weekly on Thursday's close, executed at the next
        session's open -- per call, so no state is shared between calls.

    Returns
    -------
    RebalanceSchedule
        ``decision_dates`` for the whole window, ``events`` for the decisions whose
        execution fits in it, and
        :attr:`~RebalanceSchedule.unexecuted_decision_dates` for the rest.  An empty
        schedule is a normal answer for a window the calendar does not cover.

    Raises
    ------
    backtest_sys.errors.RebalancePolicyError
        When the window is not a closed interval of plain dates, when the calendar
        cannot be loaded, or when the rule cannot be resolved on it.  Failures from
        :mod:`data_sys.market_calendar` and from
        :func:`factor_sys.factor_schedule.decision_schedule` are wrapped here with
        their ``__cause__`` kept, so a caller of the backtest layer never has to
        catch a factor-layer or calendar-layer error.

    Notes
    -----
    * Nothing is read, fetched or written: no price, no position, no order, no disk
      and no network.  The only external input is the calendar definition.
    * A decision whose fill would land after ``end_date`` is **not** executed late
      and **not** clamped into the window: it stays listed as unexecuted, because a
      replay must not trade on a date its owner never asked about.

    Examples
    --------
    >>> schedule = rebalance_schedule("2024-01-01", "2024-01-04")
    >>> schedule.decision_dates
    (datetime.date(2024, 1, 4),)
    >>> schedule.events
    ()
    >>> schedule.unexecuted_decision_dates
    (datetime.date(2024, 1, 4),)
    >>> slowly = rebalance_schedule(
    ...     "2024-01-01", "2024-01-12", policy=RebalancePolicy(execution_delay_sessions=2)
    ... )
    >>> slowly.events[0]
    RebalanceEvent(2024-01-04 -> 2024-01-08)
    >>> monthly = rebalance_schedule(
    ...     "2024-01-01", "2024-02-29", policy=RebalancePolicy(frequency="monthly")
    ... )
    >>> monthly.events[0]
    RebalanceEvent(2024-01-31 -> 2024-02-01)
    """
    start = _window_date(start_date, "rebalance_schedule start_date")
    end = _window_date(end_date, "rebalance_schedule end_date")
    if end < start:
        raise RebalancePolicyError(
            f"start_date ({start.isoformat()}) is after end_date ({end.isoformat()}): a "
            "rebalance calendar needs a closed interval, so an inverted one is refused"
        )

    rule = RebalancePolicy() if policy is None else policy
    if not isinstance(rule, RebalancePolicy):
        raise RebalancePolicyError(
            f"policy must be a RebalancePolicy or None, got {type(policy).__name__}: a schedule "
            "is generated from exactly one rule and cannot blend several"
        )

    try:
        calendar = get_calendar(rule.calendar_name)
    except CalendarError as exc:
        raise RebalancePolicyError(
            f"could not load the trading calendar {rule.calendar_name!r} for "
            f"{start.isoformat()}..{end.isoformat()}: {exc}"
        ) from exc

    try:
        dates = decision_schedule(
            start,
            end,
            frequency=rule.frequency,
            anchor_weekday=rule.anchor_weekday,
            calendar_name=rule.calendar_name,
        ).dates
    except FactorScheduleError as exc:
        raise RebalancePolicyError(
            f"the {rule.frequency!r} rule on {rule.calendar_name!r} cannot be resolved for "
            f"{start.isoformat()}..{end.isoformat()}: {exc}"
        ) from exc

    sessions = _sessions_from(calendar, start)
    events: list[RebalanceEvent] = []
    for day in dates:
        execution = _execution_session(sessions, day, rule.execution_delay_sessions)
        if execution is not None and execution <= end:
            events.append(RebalanceEvent(day, execution))

    return RebalanceSchedule(
        start_date=start,
        end_date=end,
        policy=rule,
        decision_dates=dates,
        events=tuple(events),
    )


__all__ = [
    "DEFAULT_EXECUTION_DELAY_SESSIONS",
    "DEFAULT_EXECUTION_PRICE",
    "EXECUTION_PRICES",
    "RebalanceEvent",
    "RebalancePolicy",
    "RebalancePolicyError",
    "RebalanceSchedule",
    "rebalance_schedule",
]
