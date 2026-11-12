"""Tests for :mod:`backtest_sys.rebalance_policy` -- the rebalance calendar.

Everything here is in memory and offline: no bar, no price, no factor store, no
DuckDB connection, no file and no network.  The only real dependency is the
project's own XNYS trading calendar, so every expectation below is a hand-written
date list for a window whose holidays are well known -- New Year, MLK Day, Good
Friday, Thanksgiving, Christmas and the 2024/2025 boundary.

Covered in this order:

* ``RebalancePolicy``: the default (Thursday close -> next open), the normalized
  anchor, the monthly policy that has none, and every refusal -- cadence, anchor,
  calendar name, delay and price convention -- collected in one error,
* ``RebalanceEvent``: two plain dates, the ISO form, the refusal of a timestamp or
  a loose string, and the refusal of an execution that is not *after* its decision,
* ``rebalance_schedule`` in its default shape: Thursday decisions, Friday fills,
  Thanksgiving moving the decision back to Wednesday, Good Friday pushing the fill
  to Monday, a weekend, Christmas, and a different anchor,
* ``biweekly`` and ``monthly``: both must agree with
  :func:`factor_sys.factor_schedule.decision_schedule` day for day,
* the delay: two and three sessions, the weekend they skip, and the decision left
  unexecuted because its fill would leave the window,
* the window edges: a window ending on the decision day, a single-day window, a
  window before and after the calendar's coverage, an inverted window and a date
  that is not a date,
* the refusals that must surface as ``RebalancePolicyError`` with the original
  cause kept, so a caller never needs a calendar or factor-layer error,
* ``RebalanceSchedule`` self-checking: a forged decision, a missing or extra one,
  the wrong order, a duplicate, a list instead of a tuple, an event on the wrong
  session, an event past the window, a missing event, two events for one decision,
  and an event for a day that never decided,
* immutability, hashing, the derived properties and the relation between the three
  histories,
* the module boundary: no dataframe, no store, no runner, no price, no file.
"""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, fields
from datetime import date, datetime
from itertools import pairwise
from pathlib import Path

import pandas as pd
import pytest

import backtest_sys
import backtest_sys.rebalance_policy as policy_module
import factor_sys.factor_schedule as factor_schedule_module
from backtest_sys import (
    RebalanceEvent,
    RebalancePolicy,
    RebalancePolicyError,
    RebalanceSchedule,
    rebalance_schedule,
)
from backtest_sys.errors import BacktestContractError, BacktestError
from data_sys.errors import CalendarError
from data_sys.market_calendar import calendar_bounds, get_calendar, is_session, sessions_in_range
from factor_sys.factor_schedule import (
    ANCHOR_WEEKDAYS,
    DEFAULT_ANCHOR_WEEKDAY,
    DEFAULT_FREQUENCY,
    FREQUENCIES,
    decision_schedule,
)

#: The module under test, as a file (for the boundary tests).
SOURCE: Path = Path(policy_module.__file__).resolve()

#: The default week asserted most often below: Thursday the 4th decides, Friday the
#: 5th fills.
FIRST_DECISION: date = date(2024, 1, 4)
FIRST_EXECUTION: date = date(2024, 1, 5)


def days(*values: str) -> tuple[date, ...]:
    """The ISO dates in ``values`` as a tuple of plain dates."""
    return tuple(date.fromisoformat(value) for value in values)


def month_ends_2024() -> tuple[date, ...]:
    """The last XNYS session of each month of 2024, written out.

    March ends on Good Friday (the 29th), June on a Friday (the 29th is a Saturday's
    eve -- the 28th is the last session), August on a Friday (the 31st is a
    Saturday) and November on the Friday after Thanksgiving, so a rule that trusted
    the 28th/30th/31st would miss four of these twelve.
    """
    return days(
        "2024-01-31",
        "2024-02-29",
        "2024-03-28",
        "2024-04-30",
        "2024-05-31",
        "2024-06-28",
        "2024-07-31",
        "2024-08-30",
        "2024-09-30",
        "2024-10-31",
        "2024-11-29",
        "2024-12-31",
    )


def later_sessions(decision_dates: tuple[date, ...], delay: int = 1) -> tuple[date | None, ...]:
    """The ``delay``-th session after each decision, straight from ``data_sys``.

    An independent oracle: it asks the trading calendar, not the module under test,
    so a bug in "which session comes next" cannot hide by being checked against
    itself.  Every ``decision_dates`` entry must already be a session.
    """
    calendar = get_calendar()
    last = calendar_bounds(calendar)[1]
    executions: list[date | None] = []
    for decision in decision_dates:
        sessions = sessions_in_range(calendar, decision, last)
        executions.append(sessions[delay] if len(sessions) > delay else None)
    return tuple(executions)


# ---------------------------------------------------------------------------
# 1. RebalancePolicy: the project timeline, spelled out
# ---------------------------------------------------------------------------
def test_a_default_policy_is_the_projects_weekly_thursday_timeline() -> None:
    policy = RebalancePolicy()

    assert policy.frequency == "weekly"
    assert policy.anchor_weekday == "THU"
    assert policy.calendar_name == "XNYS"
    assert policy.execution_delay_sessions == 1
    assert policy.execution_price == "open"
    assert policy == RebalancePolicy(
        frequency="weekly",
        anchor_weekday="THU",
        calendar_name="XNYS",
        execution_delay_sessions=1,
        execution_price="open",
    )


def test_a_policy_with_no_anchor_is_normalized_to_the_project_default() -> None:
    assert RebalancePolicy(anchor_weekday=None) == RebalancePolicy()
    assert RebalancePolicy().anchor_weekday == DEFAULT_ANCHOR_WEEKDAY == "THU"


def test_a_monthly_policy_carries_no_anchor() -> None:
    assert RebalancePolicy(frequency="monthly").anchor_weekday is None
    assert RebalancePolicy(frequency="monthly").frequency == "monthly"


def test_a_policy_declares_exactly_the_five_documented_fields() -> None:
    assert [field.name for field in fields(RebalancePolicy)] == [
        "frequency",
        "anchor_weekday",
        "calendar_name",
        "execution_delay_sessions",
        "execution_price",
    ]


def test_the_policy_vocabulary_is_the_factor_systems_own() -> None:
    assert policy_module.FREQUENCIES is FREQUENCIES
    assert policy_module.ANCHOR_WEEKDAYS is ANCHOR_WEEKDAYS
    assert policy_module.DEFAULT_FREQUENCY is DEFAULT_FREQUENCY
    assert policy_module.DEFAULT_ANCHOR_WEEKDAY is DEFAULT_ANCHOR_WEEKDAY
    assert FREQUENCIES == ("weekly", "biweekly", "monthly")
    assert ANCHOR_WEEKDAYS == ("MON", "TUE", "WED", "THU", "FRI")
    assert RebalancePolicy().frequency == DEFAULT_FREQUENCY


def test_a_policy_is_frozen_and_hashable() -> None:
    policy = RebalancePolicy()

    with pytest.raises(FrozenInstanceError):
        policy.frequency = "monthly"  # type: ignore[misc]

    assert hash(policy) == hash(RebalancePolicy())
    assert len({RebalancePolicy(), RebalancePolicy(anchor_weekday="WED")}) == 2


# ---------------------------------------------------------------------------
# 2. RebalancePolicy refuses a rule it cannot execute
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "frequency", ["daily", "Weekly", "week", "month", "", "quarterly", 1, None]
)
def test_a_policy_refuses_a_cadence_that_is_not_one_of_the_three(frequency: object) -> None:
    with pytest.raises(BacktestContractError) as caught:
        RebalancePolicy(frequency=frequency)  # type: ignore[arg-type]

    message = str(caught.value)
    assert "frequency" in message
    assert "'weekly'" in message


@pytest.mark.parametrize("anchor", ["SAT", "SUN", "thu", "Thursday", "THURSDAY", "", 4])
def test_a_policy_refuses_an_anchor_outside_the_week(anchor: object) -> None:
    with pytest.raises(BacktestContractError) as caught:
        RebalancePolicy(frequency="weekly", anchor_weekday=anchor)  # type: ignore[arg-type]

    message = str(caught.value)
    assert "anchor_weekday" in message
    assert "'THU'" in message


def test_a_monthly_policy_refuses_an_anchor_rather_than_ignoring_it() -> None:
    with pytest.raises(BacktestContractError) as caught:
        RebalancePolicy(frequency="monthly", anchor_weekday="THU")

    message = str(caught.value)
    assert "anchor_weekday must be None for a monthly policy" in message
    assert "'THU'" in message


@pytest.mark.parametrize("name", ["", "   ", " XNYS", "XNYS ", 42, None])
def test_a_policy_refuses_a_calendar_name_that_is_not_a_bare_name(name: object) -> None:
    with pytest.raises(BacktestContractError) as caught:
        RebalancePolicy(calendar_name=name)  # type: ignore[arg-type]

    assert "calendar_name" in str(caught.value)


@pytest.mark.parametrize("delay", [0, -1, -3, 1.0, 2.5, "1", True, None])
def test_a_policy_refuses_a_delay_that_is_not_a_whole_number_of_sessions(delay: object) -> None:
    with pytest.raises(BacktestContractError) as caught:
        RebalancePolicy(execution_delay_sessions=delay)  # type: ignore[arg-type]

    message = str(caught.value)
    assert "execution_delay_sessions must be" in message


@pytest.mark.parametrize("price", ["OPEN", "Open", "close", "vwap", "", 1, None])
def test_a_policy_refuses_a_price_convention_this_version_cannot_honour(price: object) -> None:
    with pytest.raises(BacktestContractError) as caught:
        RebalancePolicy(execution_price=price)  # type: ignore[arg-type]

    message = str(caught.value)
    assert "execution_price" in message
    assert "'open'" in message


def test_a_policy_collects_every_problem_in_one_error() -> None:
    with pytest.raises(BacktestContractError) as caught:
        RebalancePolicy(
            frequency="daily",
            calendar_name=" ",
            execution_delay_sessions=0,
            execution_price="close",
        )

    message = str(caught.value)
    assert message.startswith("invalid RebalancePolicy")
    for field in ("frequency", "calendar_name", "execution_delay_sessions", "execution_price"):
        assert field in message, field
    assert message.count("\n  - ") == 4


def test_a_policy_accepts_a_longer_delay() -> None:
    assert RebalancePolicy(execution_delay_sessions=2).execution_delay_sessions == 2
    assert RebalancePolicy(execution_delay_sessions=5).execution_delay_sessions == 5


# ---------------------------------------------------------------------------
# 3. RebalanceEvent: two days, nothing else
# ---------------------------------------------------------------------------
def test_an_event_accepts_a_plain_date_or_a_bare_iso_day() -> None:
    plain = RebalanceEvent(FIRST_DECISION, FIRST_EXECUTION)
    written = RebalanceEvent("2024-01-04", "2024-01-05")

    assert plain == written
    assert isinstance(plain.decision_date, date)
    assert isinstance(plain.execution_date, date)
    assert (plain.decision_date, plain.execution_date) == (FIRST_DECISION, FIRST_EXECUTION)


def test_an_event_prints_its_two_days_as_iso() -> None:
    event = RebalanceEvent(FIRST_DECISION, FIRST_EXECUTION)

    assert repr(event) == "RebalanceEvent(2024-01-04 -> 2024-01-05)"
    assert str(event) == "RebalanceEvent(2024-01-04 -> 2024-01-05)"


@pytest.mark.parametrize(
    "value",
    [
        datetime(2024, 1, 4),
        datetime(2024, 1, 4, 16, 0),
        pd.Timestamp("2024-01-04"),
        pd.NaT,
        "2024-1-4",
        "2024-01-04T00:00",
        "04/01/2024",
        "20240104",
        20240104,
        None,
    ],
)
def test_an_event_refuses_anything_that_is_not_the_session_itself(value: object) -> None:
    with pytest.raises(BacktestContractError) as caught:
        RebalanceEvent(value, FIRST_EXECUTION)  # type: ignore[arg-type]

    message = str(caught.value)
    assert "decision_date" in message
    assert "RebalanceEvent" in message


def test_an_event_reads_a_padded_iso_day_exactly_like_the_contract_does() -> None:
    assert RebalanceEvent(" 2024-01-04 ", FIRST_EXECUTION) == RebalanceEvent(
        FIRST_DECISION, FIRST_EXECUTION
    )


def test_an_event_refuses_an_execution_that_is_not_later_than_the_decision() -> None:
    with pytest.raises(BacktestContractError) as caught:
        RebalanceEvent(FIRST_DECISION, FIRST_DECISION)

    message = str(caught.value)
    assert "execution_date" in message
    assert "later" in message


def test_an_event_refuses_an_execution_before_the_decision() -> None:
    with pytest.raises(BacktestContractError):
        RebalanceEvent(FIRST_EXECUTION, FIRST_DECISION)


def test_an_event_is_frozen_hashable_and_holds_no_price_or_quantity() -> None:
    event = RebalanceEvent(FIRST_DECISION, FIRST_EXECUTION)

    with pytest.raises(FrozenInstanceError):
        event.execution_date = date(2024, 1, 8)  # type: ignore[misc]

    assert hash(event) == hash(RebalanceEvent("2024-01-04", "2024-01-05"))
    assert len({event, RebalanceEvent(FIRST_DECISION, FIRST_EXECUTION)}) == 1
    assert [field.name for field in fields(RebalanceEvent)] == ["decision_date", "execution_date"]


# ---------------------------------------------------------------------------
# 4. The default schedule: Thursday decides, the next session fills
# ---------------------------------------------------------------------------
def test_the_default_schedule_fills_every_decision_in_the_next_session() -> None:
    schedule = rebalance_schedule("2024-01-01", "2024-12-31")

    assert len(schedule.decision_dates) == 52
    assert len(schedule.events) == 52
    assert schedule.unexecuted_decision_dates == ()
    assert schedule.execution_dates == later_sessions(schedule.decision_dates)


def test_the_first_week_of_2024_decides_on_thursday_and_fills_on_friday() -> None:
    schedule = rebalance_schedule("2024-01-01", "2024-01-31")

    assert schedule.decision_dates[:3] == days("2024-01-04", "2024-01-11", "2024-01-18")
    assert schedule.events[:3] == (
        RebalanceEvent("2024-01-04", "2024-01-05"),
        RebalanceEvent("2024-01-11", "2024-01-12"),
        RebalanceEvent("2024-01-18", "2024-01-19"),
    )
    assert schedule.events[0].execution_date == FIRST_EXECUTION


def test_every_decision_is_a_session_and_every_fill_comes_after_it() -> None:
    calendar = get_calendar()
    schedule = rebalance_schedule("2024-01-01", "2024-12-31")

    for event in schedule.events:
        assert is_session(calendar, event.decision_date)
        assert is_session(calendar, event.execution_date)
        assert event.execution_date > event.decision_date
        # Thursday, or Wednesday when the exchange is closed on Thursday.
        assert event.decision_date.isoweekday() in (3, 4)


def test_thanksgiving_moves_the_decision_to_the_wednesday() -> None:
    calendar = get_calendar()
    schedule = rebalance_schedule("2024-11-25", "2024-11-29")

    assert not is_session(calendar, date(2024, 11, 28))
    assert schedule.decision_dates == days("2024-11-27")
    assert schedule.events == (RebalanceEvent("2024-11-27", "2024-11-29"),)


def test_good_friday_pushes_the_fill_to_monday() -> None:
    calendar = get_calendar()
    schedule = rebalance_schedule("2024-03-25", "2024-04-05")

    assert not is_session(calendar, date(2024, 3, 29))
    assert schedule.events[0] == RebalanceEvent("2024-03-28", "2024-04-01")
    assert schedule.execution_dates[0] == date(2024, 4, 1)


def test_a_holiday_between_the_decision_and_the_fill_is_skipped() -> None:
    calendar = get_calendar()
    schedule = rebalance_schedule("2024-12-23", "2024-12-31")

    assert not is_session(calendar, date(2024, 12, 25))
    assert schedule.events == (RebalanceEvent("2024-12-26", "2024-12-27"),)


def test_a_weekend_is_skipped_by_the_delay() -> None:
    calendar = get_calendar()
    policy = RebalancePolicy(execution_delay_sessions=3)
    schedule = rebalance_schedule("2024-01-04", "2024-01-09", policy=policy)

    assert not is_session(calendar, date(2024, 1, 6))
    assert not is_session(calendar, date(2024, 1, 7))
    assert schedule.events == (RebalanceEvent("2024-01-04", "2024-01-09"),)


@pytest.mark.parametrize(
    ("anchor", "expected"),
    [
        ("MON", days("2024-01-08", "2024-01-12", "2024-01-22")),
        ("TUE", days("2024-01-02", "2024-01-09", "2024-01-16")),
        ("WED", days("2024-01-03", "2024-01-10", "2024-01-17")),
        ("THU", days("2024-01-04", "2024-01-11", "2024-01-18")),
        ("FRI", days("2024-01-05", "2024-01-12", "2024-01-19")),
    ],
)
def test_the_anchor_moves_the_decision_and_the_fill_follows(
    anchor: str, expected: tuple[date, ...]
) -> None:
    policy = RebalancePolicy(anchor_weekday=anchor)
    schedule = rebalance_schedule("2024-01-01", "2024-01-24", policy=policy)

    assert schedule.decision_dates[:3] == expected
    assert schedule.execution_dates == later_sessions(
        schedule.decision_dates[: len(schedule.events)]
    )


# ---------------------------------------------------------------------------
# 5. biweekly and monthly: the factor schedule's own answer, not a second one
# ---------------------------------------------------------------------------
def test_biweekly_reuses_the_factor_schedules_phase() -> None:
    policy = RebalancePolicy(frequency="biweekly")
    schedule = rebalance_schedule("2024-01-01", "2024-12-31", policy=policy)

    assert (
        schedule.decision_dates
        == decision_schedule("2024-01-01", "2024-12-31", frequency="biweekly").dates
    )
    assert len(schedule.decision_dates) == 26


def test_biweekly_keeps_every_second_week_starting_from_the_first() -> None:
    policy = RebalancePolicy(frequency="biweekly")
    schedule = rebalance_schedule("2024-01-01", "2024-06-30", policy=policy)

    assert schedule.decision_dates[:4] == days(
        "2024-01-04", "2024-01-18", "2024-02-01", "2024-02-15"
    )
    assert len(schedule.decision_dates) == 13
    for earlier, later in pairwise(schedule.decision_dates):
        assert (later - earlier).days == 14
    assert schedule.execution_dates == later_sessions(schedule.decision_dates)


def test_biweekly_with_another_anchor_still_matches_the_factor_schedule() -> None:
    policy = RebalancePolicy(frequency="biweekly", anchor_weekday="WED")
    schedule = rebalance_schedule("2024-01-01", "2024-02-29", policy=policy)

    assert schedule.decision_dates[:3] == days("2024-01-03", "2024-01-17", "2024-01-31")
    assert (
        schedule.decision_dates
        == decision_schedule(
            "2024-01-01", "2024-02-29", frequency="biweekly", anchor_weekday="WED"
        ).dates
    )


def test_monthly_decides_on_the_last_session_of_each_month() -> None:
    policy = RebalancePolicy(frequency="monthly")
    schedule = rebalance_schedule("2024-01-01", "2024-12-31", policy=policy)

    assert schedule.decision_dates == month_ends_2024()
    assert (
        schedule.decision_dates
        == decision_schedule("2024-01-01", "2024-12-31", frequency="monthly").dates
    )
    assert len(schedule.events) == 11
    assert schedule.events[0] == RebalanceEvent("2024-01-31", "2024-02-01")
    assert schedule.events[2] == RebalanceEvent("2024-03-28", "2024-04-01")
    assert schedule.events[10] == RebalanceEvent("2024-11-29", "2024-12-02")


def test_the_last_month_of_a_year_decides_but_does_not_fill_inside_it() -> None:
    policy = RebalancePolicy(frequency="monthly")
    schedule = rebalance_schedule("2024-01-01", "2024-12-31", policy=policy)

    assert schedule.unexecuted_decision_dates == days("2024-12-31")


def test_a_monthly_window_crosses_the_year_boundary() -> None:
    policy = RebalancePolicy(frequency="monthly")
    schedule = rebalance_schedule("2024-12-01", "2025-01-31", policy=policy)

    assert schedule.decision_dates == days("2024-12-31", "2025-01-31")
    assert schedule.events == (RebalanceEvent("2024-12-31", "2025-01-02"),)
    assert schedule.unexecuted_decision_dates == days("2025-01-31")


def test_a_monthly_schedule_refuses_an_anchor_at_construction() -> None:
    with pytest.raises(BacktestContractError):
        RebalancePolicy(frequency="monthly", anchor_weekday="THU")


# ---------------------------------------------------------------------------
# 6. The delay: n sessions later, and no further
# ---------------------------------------------------------------------------
def test_two_sessions_of_delay_fill_on_the_following_monday() -> None:
    policy = RebalancePolicy(execution_delay_sessions=2)
    schedule = rebalance_schedule("2024-01-01", "2024-01-08", policy=policy)

    assert schedule.events == (RebalanceEvent("2024-01-04", "2024-01-08"),)
    assert schedule.unexecuted_decision_dates == ()


def test_three_sessions_of_delay_skip_the_weekend() -> None:
    policy = RebalancePolicy(execution_delay_sessions=3)
    schedule = rebalance_schedule("2024-01-01", "2024-01-31", policy=policy)

    assert schedule.events[0] == RebalanceEvent("2024-01-04", "2024-01-09")
    assert schedule.events[1] == RebalanceEvent("2024-01-11", "2024-01-17")
    assert schedule.execution_dates == later_sessions(schedule.decision_dates, delay=3)


def test_a_delay_leaves_the_decisions_whose_fill_leaves_the_window_unexecuted() -> None:
    policy = RebalancePolicy(execution_delay_sessions=2)
    schedule = rebalance_schedule("2024-01-01", "2024-01-12", policy=policy)

    assert schedule.decision_dates == days("2024-01-04", "2024-01-11")
    assert schedule.events == (RebalanceEvent("2024-01-04", "2024-01-08"),)
    assert schedule.unexecuted_decision_dates == days("2024-01-11")


# ---------------------------------------------------------------------------
# 7. Window edges: the interval is closed and nothing is clamped
# ---------------------------------------------------------------------------
def test_a_window_that_ends_on_the_decision_weekday_keeps_it_unexecuted() -> None:
    schedule = rebalance_schedule("2024-01-01", "2024-01-04")

    assert schedule.decision_dates == days("2024-01-04")
    assert schedule.events == ()
    assert schedule.execution_dates == ()
    assert schedule.unexecuted_decision_dates == days("2024-01-04")
    assert len(schedule) == 0


def test_a_window_that_ends_on_the_fill_day_completes_the_rebalance() -> None:
    schedule = rebalance_schedule("2024-01-01", "2024-01-05")

    assert schedule.events == (RebalanceEvent("2024-01-04", "2024-01-05"),)
    assert schedule.unexecuted_decision_dates == ()


def test_a_single_day_window_on_a_decision_day_is_still_a_decision() -> None:
    schedule = rebalance_schedule("2024-01-04", "2024-01-04")

    assert schedule.decision_dates == days("2024-01-04")
    assert schedule.events == ()
    assert schedule.unexecuted_decision_dates == days("2024-01-04")


def test_the_window_is_inclusive_at_both_ends() -> None:
    schedule = rebalance_schedule("2024-01-04", "2024-01-05")

    assert schedule.start_date == FIRST_DECISION
    assert schedule.end_date == FIRST_EXECUTION
    assert schedule.events == (RebalanceEvent(FIRST_DECISION, FIRST_EXECUTION),)


@pytest.mark.parametrize(
    ("start", "end"), [("1990-01-01", "1990-12-31"), ("2030-01-01", "2030-12-31")]
)
def test_a_window_the_calendar_does_not_cover_is_an_empty_schedule(start: str, end: str) -> None:
    schedule = rebalance_schedule(start, end)

    assert schedule.decision_dates == ()
    assert schedule.events == ()
    assert schedule.execution_dates == ()
    assert schedule.unexecuted_decision_dates == ()
    assert len(schedule) == 0
    assert schedule.policy == RebalancePolicy()


def test_a_cross_year_window_skips_the_new_year_holiday() -> None:
    calendar = get_calendar()
    schedule = rebalance_schedule("2024-12-30", "2025-01-03")

    assert not is_session(calendar, date(2025, 1, 1))
    assert schedule.decision_dates == days("2025-01-02")
    assert schedule.events == (RebalanceEvent("2025-01-02", "2025-01-03"),)


def test_the_window_accepts_a_plain_date_or_a_bare_iso_day() -> None:
    from_dates = rebalance_schedule(FIRST_DECISION, date(2024, 1, 31))
    from_strings = rebalance_schedule("2024-01-04", "2024-01-31")

    assert from_dates == from_strings
    assert isinstance(from_dates.start_date, date)
    assert isinstance(from_dates.end_date, date)


def test_an_inverted_window_is_refused() -> None:
    with pytest.raises(RebalancePolicyError) as caught:
        rebalance_schedule("2024-01-31", "2024-01-02")

    message = str(caught.value)
    assert "2024-01-31" in message
    assert "2024-01-02" in message
    assert "after" in message


@pytest.mark.parametrize(
    "value",
    [
        datetime(2024, 1, 4),
        pd.Timestamp("2024-01-04"),
        "2024-1-4",
        "2024-01-04T00:00",
        20240104,
        None,
    ],
)
def test_a_window_edge_that_is_not_a_date_is_refused_as_a_policy_error(value: object) -> None:
    with pytest.raises(RebalancePolicyError) as caught:
        rebalance_schedule(value, "2024-02-01")  # type: ignore[arg-type]

    assert "start_date" in str(caught.value)
    assert isinstance(caught.value.__cause__, BacktestContractError)


# ---------------------------------------------------------------------------
# 8. The refusals a caller sees: one error type, the original cause kept
# ---------------------------------------------------------------------------
def test_an_unknown_calendar_is_a_policy_error_that_keeps_its_cause() -> None:
    policy = RebalancePolicy(calendar_name="NOT_A_CALENDAR")

    with pytest.raises(RebalancePolicyError) as caught:
        rebalance_schedule("2024-01-01", "2024-02-01", policy=policy)

    message = str(caught.value)
    assert "NOT_A_CALENDAR" in message
    assert "2024-01-01..2024-02-01" in message
    assert isinstance(caught.value.__cause__, CalendarError)


def test_a_policy_that_is_not_a_policy_is_refused() -> None:
    with pytest.raises(RebalancePolicyError) as caught:
        rebalance_schedule("2024-01-01", "2024-02-01", policy="weekly")  # type: ignore[arg-type]

    assert "policy" in str(caught.value)


def test_every_refusal_is_a_backtest_error() -> None:
    with pytest.raises(BacktestError):
        rebalance_schedule("2024-01-01", "2024-02-01", policy=RebalancePolicy(calendar_name="nope"))


def test_the_schedule_carries_the_rule_that_built_it() -> None:
    policy = RebalancePolicy(anchor_weekday="MON", execution_delay_sessions=2)
    schedule = rebalance_schedule("2024-01-01", "2024-01-31", policy=policy)

    assert schedule.policy is policy
    assert schedule.policy == policy


def test_each_call_builds_its_own_default_policy() -> None:
    first = rebalance_schedule("2024-01-01", "2024-01-31")
    second = rebalance_schedule("2024-01-01", "2024-01-31")

    assert first == second
    assert first is not second
    assert first.policy is not second.policy
    assert first.policy == second.policy == RebalancePolicy()


# ---------------------------------------------------------------------------
# 9. RebalanceSchedule checks itself: a forged record is refused, not trusted
# ---------------------------------------------------------------------------
def test_a_hand_built_schedule_is_accepted_when_it_says_the_same_thing() -> None:
    generated = rebalance_schedule("2024-01-01", "2024-01-31")
    rebuilt = RebalanceSchedule(
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 31),
        policy=RebalancePolicy(),
        decision_dates=generated.decision_dates,
        events=generated.events,
    )

    assert rebuilt == generated


def test_a_schedule_refuses_a_decision_that_is_not_a_session() -> None:
    with pytest.raises(BacktestContractError) as caught:
        RebalanceSchedule(
            start_date="2024-11-25",
            end_date="2024-11-29",
            policy=RebalancePolicy(),
            decision_dates=days("2024-11-28"),  # Thanksgiving: the exchange is closed
            events=(RebalanceEvent("2024-11-28", "2024-11-29"),),
        )

    message = str(caught.value)
    assert "must be a trading session of XNYS" in message
    assert "2024-11-28" in message


def test_a_schedule_refuses_a_missing_decision() -> None:
    with pytest.raises(BacktestContractError) as caught:
        RebalanceSchedule(
            start_date="2024-01-01",
            end_date="2024-01-12",
            policy=RebalancePolicy(),
            decision_dates=days("2024-01-04"),
            events=(RebalanceEvent("2024-01-04", "2024-01-05"),),
        )

    message = str(caught.value)
    assert "is missing" in message
    assert "2024-01-11" in message


def test_a_schedule_refuses_a_decision_the_rule_did_not_select() -> None:
    with pytest.raises(BacktestContractError) as caught:
        RebalanceSchedule(
            start_date="2024-01-01",
            end_date="2024-01-12",
            policy=RebalancePolicy(),
            decision_dates=days("2024-01-04", "2024-01-05"),
            events=(
                RebalanceEvent("2024-01-04", "2024-01-05"),
                RebalanceEvent("2024-01-05", "2024-01-08"),
            ),
        )

    message = str(caught.value)
    assert "does not belong" in message
    assert "'2024-01-05'" in message
    assert "2024-01-11" in message


@pytest.mark.parametrize(
    ("dates", "worded"),
    [
        (("2024-01-11", "2024-01-04"), "2024-01-04 follows 2024-01-11"),
        (("2024-01-04", "2024-01-04"), "2024-01-04 follows 2024-01-04"),
    ],
)
def test_a_schedule_refuses_unsorted_or_repeated_decisions(
    dates: tuple[str, str], worded: str
) -> None:
    with pytest.raises(BacktestContractError) as caught:
        RebalanceSchedule(
            start_date="2024-01-01",
            end_date="2024-01-12",
            policy=RebalancePolicy(),
            decision_dates=dates,
            events=(),
        )

    message = str(caught.value)
    assert "strictly ascending" in message
    assert worded in message


def test_a_schedule_refuses_a_decision_outside_its_own_window() -> None:
    with pytest.raises(BacktestContractError) as caught:
        RebalanceSchedule(
            start_date="2024-01-08",
            end_date="2024-01-31",
            policy=RebalancePolicy(),
            decision_dates=days("2024-01-04"),
            events=(),
        )

    message = str(caught.value)
    assert "must lie inside" in message
    assert "2024-01-04" in message


def test_a_schedule_refuses_a_list_where_an_immutable_tuple_belongs() -> None:
    with pytest.raises(BacktestContractError) as caught:
        RebalanceSchedule(
            start_date="2024-01-01",
            end_date="2024-01-31",
            policy=RebalancePolicy(),
            decision_dates=days("2024-01-04"),  # type: ignore[arg-type]
            events=list(rebalance_schedule("2024-01-01", "2024-01-31").events),  # type: ignore[arg-type]
        )

    message = str(caught.value)
    assert "events must be a tuple" in message


def test_a_schedule_refuses_a_decision_date_field_that_is_not_a_tuple() -> None:
    with pytest.raises(BacktestContractError) as caught:
        RebalanceSchedule(
            start_date="2024-01-01",
            end_date="2024-01-31",
            policy=RebalancePolicy(),
            decision_dates=list(days("2024-01-04")),  # type: ignore[arg-type]
            events=(),
        )

    assert "decision_dates must be a tuple" in str(caught.value)


def test_a_schedule_refuses_an_event_on_the_wrong_session() -> None:
    with pytest.raises(BacktestContractError) as caught:
        RebalanceSchedule(
            start_date="2024-01-01",
            end_date="2024-01-12",
            policy=RebalancePolicy(),
            decision_dates=days("2024-01-04", "2024-01-11"),
            events=(
                RebalanceEvent("2024-01-04", "2024-01-08"),
                RebalanceEvent("2024-01-11", "2024-01-12"),
            ),
        )

    message = str(caught.value)
    assert "does not follow the policy" in message
    assert "1 session(s) after 2024-01-04 is 2024-01-05" in message


def test_a_schedule_refuses_a_fill_that_is_not_a_session() -> None:
    with pytest.raises(BacktestContractError) as caught:
        RebalanceSchedule(
            start_date="2024-01-01",
            end_date="2024-01-12",
            policy=RebalancePolicy(),
            decision_dates=days("2024-01-04", "2024-01-11"),
            events=(
                RebalanceEvent("2024-01-04", "2024-01-06"),  # a Saturday
                RebalanceEvent("2024-01-11", "2024-01-12"),
            ),
        )

    assert "does not follow the policy" in str(caught.value)


def test_a_schedule_refuses_a_decision_that_has_no_event_but_needs_one() -> None:
    with pytest.raises(BacktestContractError) as caught:
        RebalanceSchedule(
            start_date="2024-01-01",
            end_date="2024-01-12",
            policy=RebalancePolicy(),
            decision_dates=days("2024-01-04", "2024-01-11"),
            events=(),
        )

    message = str(caught.value)
    assert "decision 2024-01-04 needs an event" in message
    assert "decision 2024-01-11 needs an event" in message
    assert "which is inside 2024-01-01..2024-01-12" in message


def test_a_schedule_refuses_an_event_whose_fill_leaves_the_window() -> None:
    with pytest.raises(BacktestContractError) as caught:
        RebalanceSchedule(
            start_date="2024-01-01",
            end_date="2024-01-04",
            policy=RebalancePolicy(),
            decision_dates=days("2024-01-04"),
            events=(RebalanceEvent("2024-01-04", "2024-01-05"),),
        )

    message = str(caught.value)
    assert "must not carry an event" in message
    assert "after the window ends" in message


def test_a_schedule_refuses_two_events_for_one_decision() -> None:
    with pytest.raises(BacktestContractError) as caught:
        RebalanceSchedule(
            start_date="2024-01-01",
            end_date="2024-01-12",
            policy=RebalancePolicy(),
            decision_dates=days("2024-01-04", "2024-01-11"),
            events=(
                RebalanceEvent("2024-01-04", "2024-01-05"),
                RebalanceEvent("2024-01-04", "2024-01-05"),
                RebalanceEvent("2024-01-11", "2024-01-12"),
            ),
        )

    message = str(caught.value)
    assert "strictly ascending" in message
    assert "appears more than once" in message


def test_a_schedule_refuses_an_event_for_a_day_that_never_decided() -> None:
    with pytest.raises(BacktestContractError) as caught:
        RebalanceSchedule(
            start_date="2024-01-01",
            end_date="2024-01-12",
            policy=RebalancePolicy(),
            decision_dates=days("2024-01-04", "2024-01-11"),
            events=(
                RebalanceEvent("2024-01-04", "2024-01-05"),
                RebalanceEvent("2024-01-05", "2024-01-08"),
                RebalanceEvent("2024-01-11", "2024-01-12"),
            ),
        )

    message = str(caught.value)
    assert "every event must quote a decision date listed in decision_dates" in message
    assert "2024-01-05" in message


def test_a_schedule_refuses_a_policy_that_is_not_a_policy() -> None:
    with pytest.raises(BacktestContractError) as caught:
        RebalanceSchedule(
            start_date="2024-01-01",
            end_date="2024-01-31",
            policy="weekly",  # type: ignore[arg-type]
            decision_dates=(),
            events=(),
        )

    assert "policy must be a RebalancePolicy" in str(caught.value)


def test_a_schedule_refuses_an_inverted_window() -> None:
    with pytest.raises(BacktestContractError) as caught:
        RebalanceSchedule(
            start_date="2024-01-31",
            end_date="2024-01-02",
            policy=RebalancePolicy(),
            decision_dates=(),
            events=(),
        )

    assert "is after end_date" in str(caught.value)


def test_an_empty_schedule_is_a_legal_record() -> None:
    generated = rebalance_schedule("1990-01-01", "1990-12-31")
    rebuilt = RebalanceSchedule(
        start_date="1990-01-01",
        end_date="1990-12-31",
        policy=RebalancePolicy(),
        decision_dates=(),
        events=(),
    )

    assert rebuilt == generated
    assert rebuilt.decision_dates == ()


# ---------------------------------------------------------------------------
# 10. The record is immutable and its properties are derived, never stored
# ---------------------------------------------------------------------------
def test_a_schedule_is_frozen() -> None:
    schedule = rebalance_schedule("2024-01-01", "2024-01-31")

    for field in fields(RebalanceSchedule):
        with pytest.raises(FrozenInstanceError):
            setattr(schedule, field.name, None)


def test_a_schedule_declares_exactly_the_five_documented_fields() -> None:
    assert [field.name for field in fields(RebalanceSchedule)] == [
        "start_date",
        "end_date",
        "policy",
        "decision_dates",
        "events",
    ]


def test_a_schedule_is_hashable_and_prints_its_rule() -> None:
    schedule = rebalance_schedule("2024-01-01", "2024-01-31")

    assert hash(schedule) == hash(rebalance_schedule("2024-01-01", "2024-01-31"))
    assert repr(schedule) == (
        "RebalanceSchedule(weekly THU on XNYS, 2024-01-01..2024-01-31, 4 decisions, 4 events)"
    )

    monthly = rebalance_schedule(
        "2024-01-01", "2024-12-31", policy=RebalancePolicy(frequency="monthly")
    )
    assert "monthly month-end on XNYS" in repr(monthly)


def test_the_execution_dates_are_the_events_own_dates() -> None:
    schedule = rebalance_schedule("2024-01-01", "2024-12-31")

    assert schedule.execution_dates == tuple(event.execution_date for event in schedule.events)
    assert schedule.execution_dates == later_sessions(schedule.decision_dates)
    assert len(schedule) == len(schedule.events)


def test_the_unexecuted_decisions_are_the_ones_without_an_event() -> None:
    schedule = rebalance_schedule("2024-01-01", "2024-01-04")

    assert schedule.unexecuted_decision_dates == days("2024-01-04")
    assert tuple(event.decision_date for event in schedule.events) == ()


@pytest.mark.parametrize(
    "policy",
    [
        RebalancePolicy(),
        RebalancePolicy(frequency="biweekly"),
        RebalancePolicy(frequency="monthly"),
        RebalancePolicy(anchor_weekday="WED"),
        RebalancePolicy(execution_delay_sessions=4),
    ],
)
def test_the_three_histories_partition_the_decisions_of_the_year(
    policy: RebalancePolicy,
) -> None:
    schedule = rebalance_schedule("2024-01-01", "2024-12-31", policy=policy)

    executed = {event.decision_date for event in schedule.events}
    unexecuted = set(schedule.unexecuted_decision_dates)

    assert not executed & unexecuted
    assert tuple(sorted(executed | unexecuted)) == schedule.decision_dates
    assert len(executed) == len(schedule.events) == len(schedule)
    assert len(schedule.execution_dates) == len(schedule.events)
    for event in schedule.events:
        assert event.execution_date > event.decision_date
        assert schedule.start_date <= event.decision_date <= schedule.end_date
        assert schedule.start_date <= event.execution_date <= schedule.end_date


# ---------------------------------------------------------------------------
# 11. The module boundary: a calendar, not a data pipeline
# ---------------------------------------------------------------------------
ALLOWED_IMPORTS: frozenset[str] = frozenset(
    {
        "__future__",
        "backtest_sys.contracts",
        "backtest_sys.errors",
        "bisect",
        "data_sys.errors",
        "data_sys.market_calendar",
        "dataclasses",
        "datetime",
        "factor_sys.errors",
        "factor_sys.factor_schedule",
        "itertools",
        "typing",
    }
)

FORBIDDEN_NAMES: frozenset[str] = frozenset(
    {
        "duckdb",
        "json",
        "numpy",
        "open",
        "os",
        "pandas",
        "print",
        "requests",
        "socket",
        "subprocess",
        "yaml",
    }
)


def imported_modules(source: Path) -> set[str]:
    """Every module ``source`` imports, by dotted name."""
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            names.add(node.module)
    return names


def test_the_rule_module_imports_only_its_allowed_layers() -> None:
    assert imported_modules(SOURCE) == ALLOWED_IMPORTS


def test_the_rule_module_reads_no_price_and_opens_no_file() -> None:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }

    assert not imported & FORBIDDEN_NAMES
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in FORBIDDEN_NAMES, node.func.id


def test_the_rule_module_carries_no_price_or_dataframe_type() -> None:
    text = SOURCE.read_text(encoding="utf-8")

    for forbidden in ("DataFrame", "Series", "read_parquet", "read_csv", "sqlite3"):
        assert forbidden not in text, forbidden


def test_the_rule_module_exports_exactly_its_public_names() -> None:
    assert policy_module.__all__ == [
        "DEFAULT_EXECUTION_DELAY_SESSIONS",
        "DEFAULT_EXECUTION_PRICE",
        "EXECUTION_PRICES",
        "RebalanceEvent",
        "RebalancePolicy",
        "RebalancePolicyError",
        "RebalanceSchedule",
        "rebalance_schedule",
    ]


def test_the_package_reexports_the_very_same_rule_objects() -> None:
    assert backtest_sys.RebalancePolicy is policy_module.RebalancePolicy
    assert backtest_sys.RebalanceEvent is policy_module.RebalanceEvent
    assert backtest_sys.RebalanceSchedule is policy_module.RebalanceSchedule
    assert backtest_sys.rebalance_schedule is policy_module.rebalance_schedule
    assert backtest_sys.RebalancePolicyError is policy_module.RebalancePolicyError


def test_the_rule_module_borrows_the_factor_vocabulary_instead_of_copying_it() -> None:
    assert policy_module.MONTHLY is factor_schedule_module.MONTHLY
    assert "MONTHLY" not in policy_module.__all__
