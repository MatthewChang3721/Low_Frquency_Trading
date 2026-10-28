"""Tests for :mod:`factor_sys.factor_schedule` -- the pure decision-date rule.

Nothing here touches a factor, a store, a bar or the network: the only real
dependency is the project's own trading calendar, so every expectation below is
a hand-written date list for a window whose holidays are well known (New Year,
MLK Day, Good Friday, Thanksgiving, Christmas).

Covered in this order:

* ``weekly``: the anchor weekday, the previous session when the anchor is a
  holiday, a different anchor producing a different series, and a window whose
  first probe falls outside it,
* ``biweekly``: every second candidate week, the phase anchored on the first
  candidate inside the window, and determinism for identical requests,
* ``monthly``: the last session of each month, month ends that are not trading
  days, the year boundary, and the explicit refusal of an anchor,
* the invariants every selection must keep: ascending, unique, all sessions, all
  inside the closed window -- and an empty selection when no day qualifies,
* the refusals: a bad window, date, cadence, anchor or calendar, plus a
  hand-built :class:`~factor_sys.factor_schedule.DecisionSchedule` that
  contradicts itself -- including one that smuggles in a weekend or an exchange
  holiday, which the record re-checks against its own calendar,
* the module boundary: no dataframe, no store, no runner, no formula, no script,
  and no file is ever opened.
"""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
from datetime import date, datetime
from pathlib import Path

import pytest

import factor_sys.factor_schedule as schedule_module
from data_sys.market_calendar import get_calendar, is_session, sessions_in_range
from factor_sys import (
    ANCHOR_WEEKDAYS,
    BIWEEKLY,
    BIWEEKLY_ANCHOR,
    DEFAULT_ANCHOR_WEEKDAY,
    DEFAULT_FREQUENCY,
    FREQUENCIES,
    MONTHLY,
    WEEKLY,
    DecisionSchedule,
    FactorScheduleError,
    decision_schedule,
    select_decision_dates,
)

#: The module under test, as a file (for the boundary tests).
SOURCE: Path = Path(schedule_module.__file__).resolve()

#: Every window asserted below lies inside 2024, so the calendar's own 2006 start
#: is never touched and the known holidays of that year are enough.


def days(*values: str) -> tuple[date, ...]:
    """The ISO dates in ``values`` as a tuple of plain dates."""
    return tuple(date.fromisoformat(value) for value in values)


def month_ends_2024() -> tuple[date, ...]:
    """The last XNYS session of each month of 2024, written out.

    March ends on Good Friday (the 29th), June on a Friday (the 30th is a
    Sunday), August on a Friday (the 31st is a Saturday) and November on a
    Friday (the 30th is a Saturday), so a rule that trusted the 28th/30th/31st
    would fail four of these twelve.
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


# ---------------------------------------------------------------------------
# 1. weekly: the anchor, or the session before it
# ---------------------------------------------------------------------------
def test_a_weekly_thu_schedule_selects_every_thursday_of_the_window() -> None:
    """The default cadence and the default anchor are Thursday."""
    schedule = decision_schedule("2024-01-01", "2024-01-31")

    assert schedule.dates == days("2024-01-04", "2024-01-11", "2024-01-18", "2024-01-25")
    assert schedule.frequency == WEEKLY == DEFAULT_FREQUENCY
    assert schedule.anchor_weekday == "THU" == DEFAULT_ANCHOR_WEEKDAY
    assert schedule.calendar_name == "XNYS"
    assert schedule.biweekly_anchor is None
    assert (schedule.start_date, schedule.end_date) == (date(2024, 1, 1), date(2024, 1, 31))


def test_a_wed_anchor_selects_a_different_series_than_a_thu_anchor() -> None:
    """The anchor chooses the weekday; it never reuses another anchor's days."""
    wednesday = select_decision_dates("2024-01-01", "2024-01-31", anchor_weekday="WED")
    thursday = select_decision_dates("2024-01-01", "2024-01-31", anchor_weekday="THU")

    assert wednesday == days("2024-01-03", "2024-01-10", "2024-01-17", "2024-01-24", "2024-01-31")
    assert thursday == days("2024-01-04", "2024-01-11", "2024-01-18", "2024-01-25")
    assert not set(wednesday) & set(thursday)
    assert {day.weekday() for day in wednesday} == {2}
    assert {day.weekday() for day in thursday} == {3}


def test_a_thursday_anchor_moves_back_over_thanksgiving() -> None:
    """2024-11-28 is closed, so that week's decision day is the Wednesday."""
    schedule = decision_schedule("2024-11-25", "2024-11-29", anchor_weekday="THU")

    assert schedule.dates == days("2024-11-27")


def test_a_wednesday_anchor_moves_back_over_christmas() -> None:
    """2024-12-25 is closed, so that week's decision day is the Tuesday."""
    assert select_decision_dates("2024-12-23", "2024-12-27", anchor_weekday="WED") == days(
        "2024-12-24"
    )


def test_a_monday_anchor_crosses_a_holiday_and_a_weekend() -> None:
    """MLK Day moves that week to the Friday before, inside the same window."""
    schedule = decision_schedule("2024-01-01", "2024-01-19", anchor_weekday="MON")

    # 2024-01-01 is New Year (and before the window's first selected day), and
    # 2024-01-15 is MLK Day, whose previous session is Friday 2024-01-12.
    assert schedule.dates == days("2024-01-08", "2024-01-12")


def test_a_candidate_before_the_window_is_dropped_not_clamped() -> None:
    """A Thursday window that starts on a Friday selects the *next* Thursday."""
    assert select_decision_dates("2024-01-05", "2024-01-12", anchor_weekday="THU") == days(
        "2024-01-11"
    )


def test_a_holiday_shift_before_the_window_start_is_dropped() -> None:
    """Thanksgiving week asked from Thanksgiving itself selects nothing."""
    assert select_decision_dates("2024-11-28", "2024-11-29", anchor_weekday="THU") == ()


# ---------------------------------------------------------------------------
# 2. biweekly: every second candidate week, phased on the window
# ---------------------------------------------------------------------------
def test_biweekly_keeps_every_second_weekly_candidate() -> None:
    """Four Thursdays become two decision days, starting with the first."""
    weekly = select_decision_dates("2024-01-01", "2024-01-31", anchor_weekday="THU")
    schedule = decision_schedule(
        "2024-01-01", "2024-01-31", frequency=BIWEEKLY, anchor_weekday="THU"
    )

    assert weekly == days("2024-01-04", "2024-01-11", "2024-01-18", "2024-01-25")
    assert schedule.dates == weekly[::2] == days("2024-01-04", "2024-01-18")
    assert schedule.biweekly_anchor == BIWEEKLY_ANCHOR


def test_the_biweekly_phase_starts_at_the_first_candidate_inside_the_window() -> None:
    """A window that opens on a Friday starts its fortnight with the next Thursday."""
    schedule = decision_schedule(
        "2024-01-05", "2024-01-31", frequency=BIWEEKLY, anchor_weekday="THU"
    )

    assert schedule.dates == days("2024-01-11", "2024-01-25")


def test_the_same_biweekly_request_always_selects_the_same_days() -> None:
    """The phase is a pure function of the request: no clock, no random state."""
    first = decision_schedule("2024-01-01", "2024-06-30", frequency=BIWEEKLY)
    second = decision_schedule("2024-01-01", "2024-06-30", frequency=BIWEEKLY)

    assert first == second
    assert first.dates == decision_schedule("2024-01-01", "2024-06-30", frequency=WEEKLY).dates[::2]


def test_only_a_biweekly_schedule_records_a_phase() -> None:
    """``biweekly_anchor`` is the phase for biweekly and ``None`` otherwise."""
    assert decision_schedule("2024-01-01", "2024-01-31", frequency=WEEKLY).biweekly_anchor is None
    assert decision_schedule("2024-01-01", "2024-01-31", frequency=MONTHLY).biweekly_anchor is None
    assert (
        decision_schedule("2024-01-01", "2024-01-31", frequency=BIWEEKLY).biweekly_anchor
        == BIWEEKLY_ANCHOR
        == "first_selected_week_in_requested_range"
    )


# ---------------------------------------------------------------------------
# 3. monthly: the last session of each month
# ---------------------------------------------------------------------------
def test_monthly_takes_the_last_session_of_each_month_of_a_year() -> None:
    """Twelve months, twelve month ends, four of them moved off the month's end."""
    schedule = decision_schedule("2024-01-01", "2024-12-31", frequency=MONTHLY)

    assert schedule.dates == month_ends_2024()
    assert schedule.frequency == MONTHLY
    assert schedule.anchor_weekday is None  # a monthly schedule has no anchor
    assert schedule.biweekly_anchor is None
    assert all(day.weekday() < 5 for day in schedule.dates)


def test_monthly_crosses_the_year_boundary_and_a_weekend_month_end() -> None:
    """2023-12-31 is a Sunday and 2024-01-31 a Wednesday: both months end early."""
    schedule = decision_schedule("2023-12-01", "2024-01-31", frequency=MONTHLY)

    assert schedule.dates == days("2023-12-29", "2024-01-31")


def test_a_month_whose_month_end_is_outside_the_window_contributes_nothing() -> None:
    """November 2024's last session is the 29th, so a 30th-only window is empty."""
    assert select_decision_dates("2024-11-30", "2024-11-30", frequency=MONTHLY) == ()
    assert select_decision_dates("2024-11-01", "2024-11-29", frequency=MONTHLY) == days(
        "2024-11-29"
    )


def test_monthly_refuses_an_explicit_anchor() -> None:
    """An anchor a monthly schedule cannot use is an error, never a silent ignore."""
    with pytest.raises(FactorScheduleError, match="does not apply to a monthly schedule"):
        decision_schedule("2024-01-01", "2024-12-31", frequency=MONTHLY, anchor_weekday="THU")


# ---------------------------------------------------------------------------
# 4. The invariants every selection keeps
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("frequency", FREQUENCIES)
@pytest.mark.parametrize("anchor", ANCHOR_WEEKDAYS)
def test_every_selected_date_is_a_session_inside_the_window(frequency: str, anchor: str) -> None:
    """Ascending, unique, never outside the window, always a real session."""
    start, end = date(2024, 1, 1), date(2024, 12, 31)
    schedule = decision_schedule(
        start,
        end,
        frequency=frequency,
        anchor_weekday=None if frequency == MONTHLY else anchor,
    )

    assert schedule.dates == tuple(sorted(set(schedule.dates)))
    assert len(schedule.dates) == len(set(schedule.dates))
    assert all(type(day) is date for day in schedule.dates)
    assert all(start <= day <= end for day in schedule.dates)
    calendar = get_calendar("XNYS")
    assert all(is_session(calendar, day) for day in schedule.dates)
    # Every selected day is one the calendar itself lists for that window.
    assert set(schedule.dates) <= set(sessions_in_range(calendar, start, end))


def test_a_window_with_no_candidate_selects_nothing_without_failing() -> None:
    """A Thursday anchor over a Monday-Wednesday window is simply empty."""
    schedule = decision_schedule("2024-01-01", "2024-01-03")

    assert schedule.dates == ()
    assert len(schedule) == 0
    assert "2024-01-04" not in schedule


def test_a_window_the_calendar_does_not_cover_selects_nothing() -> None:
    """Nothing is fabricated from before the calendar's first session."""
    assert select_decision_dates("1990-01-01", "1990-12-31", frequency=MONTHLY) == ()
    assert select_decision_dates("1990-01-01", "1990-12-31") == ()


def test_a_window_far_beyond_the_calendar_selects_nothing_either() -> None:
    """The same holds after the calendar's last session, without raising."""
    assert select_decision_dates("2099-01-01", "2099-12-31", frequency=MONTHLY) == ()


def test_the_anchor_defaults_are_documented_values() -> None:
    """The vocabulary itself is part of the public API."""
    assert FREQUENCIES == ("weekly", "biweekly", "monthly")
    assert ANCHOR_WEEKDAYS == ("MON", "TUE", "WED", "THU", "FRI")
    assert (DEFAULT_FREQUENCY, DEFAULT_ANCHOR_WEEKDAY) == ("weekly", "THU")


def test_a_schedule_is_immutable_and_describes_itself() -> None:
    """The record cannot be edited after the fact, and its repr is readable."""
    schedule = decision_schedule("2024-01-01", "2024-01-31")

    with pytest.raises(FrozenInstanceError):
        schedule.frequency = MONTHLY  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        schedule.dates = ()  # type: ignore[misc]

    assert repr(schedule) == (
        "DecisionSchedule(frequency='weekly', anchor_weekday='THU', calendar_name='XNYS', "
        "start_date='2024-01-01', end_date='2024-01-31', dates=4)"
    )
    assert "2024-01-11" in schedule
    assert date(2024, 1, 11) in schedule
    assert "2024-01-12" not in schedule
    assert "not-a-date" not in schedule


def test_select_decision_dates_is_the_dates_of_the_schedule() -> None:
    """The convenience function and the record can never disagree."""
    schedule = decision_schedule("2024-01-01", "2024-03-31", frequency=BIWEEKLY)

    assert select_decision_dates("2024-01-01", "2024-03-31", frequency=BIWEEKLY) == schedule.dates
    assert isinstance(schedule.dates, tuple)


# ---------------------------------------------------------------------------
# 5. The refusals: one readable error per rule
# ---------------------------------------------------------------------------
def test_an_inverted_window_is_refused() -> None:
    """``start > end`` is a typo, not a request for an empty schedule."""
    with pytest.raises(FactorScheduleError, match="must not be after end_date"):
        decision_schedule("2024-01-31", "2024-01-01")


def test_a_single_day_window_is_allowed() -> None:
    """The window is closed: a session asked for as start = end is selected."""
    assert select_decision_dates("2024-01-04", "2024-01-04") == days("2024-01-04")


@pytest.mark.parametrize(
    "value",
    [
        "2024-1-5",
        "31/12/2024",
        "2024-02-30",
        "",
        "   ",
        None,
        20240105,
        b"2024-01-05",
        date(2024, 1, 5).isoformat() + "T09:30:00",
    ],
)
def test_a_bound_that_is_not_a_strict_iso_day_is_refused(value: object) -> None:
    """Only a plain date or a strict ``YYYY-MM-DD`` string names a window."""
    with pytest.raises(FactorScheduleError, match="must be a date or a strict ISO"):
        decision_schedule(value, "2024-01-31")


def test_a_datetime_bound_is_refused_rather_than_silently_truncated() -> None:
    """A time of day would let this module choose the session, so it refuses one."""
    with pytest.raises(FactorScheduleError, match="a decision day is a whole session"):
        decision_schedule(datetime(2024, 1, 5, 16, 0), "2024-01-31")


@pytest.mark.parametrize("frequency", ["daily", "WEEKLY", "Weekly", "", None, 3, True])
def test_an_unknown_frequency_is_refused(frequency: object) -> None:
    """The three cadences are a closed vocabulary, spelled exactly."""
    with pytest.raises(FactorScheduleError, match="frequency must be one of"):
        decision_schedule("2024-01-01", "2024-01-31", frequency=frequency)  # type: ignore[arg-type]


@pytest.mark.parametrize("anchor", ["MONDAY", "thu", "Thu", "SAT", "", 0, True])
def test_an_unknown_anchor_weekday_is_refused(anchor: object) -> None:
    """Only ``MON``..``FRI`` can anchor a weekly schedule."""
    with pytest.raises(FactorScheduleError, match="anchor_weekday must be one of"):
        decision_schedule(
            "2024-01-01",
            "2024-01-31",
            anchor_weekday=anchor,  # type: ignore[arg-type]
        )


def test_an_unknown_calendar_is_refused() -> None:
    """The calendar name is asked of the data system, not guessed."""
    with pytest.raises(FactorScheduleError, match="could not load the trading calendar"):
        decision_schedule("2024-01-01", "2024-01-31", calendar_name="NOPE")


def test_a_blank_calendar_name_is_refused() -> None:
    """A whitespace-only name is not a calendar."""
    with pytest.raises(FactorScheduleError, match="must be a non-empty trading calendar name"):
        decision_schedule("2024-01-01", "2024-01-31", calendar_name="   ")


def test_a_calendar_name_is_normalized_into_the_schedule() -> None:
    """Surrounding spaces are not part of the name that gets recorded."""
    schedule = decision_schedule("2024-01-01", "2024-01-31", calendar_name=" XNYS ")

    assert schedule.calendar_name == "XNYS"
    assert schedule.dates == days("2024-01-04", "2024-01-11", "2024-01-18", "2024-01-25")


# ---------------------------------------------------------------------------
# 6. A hand-built schedule must not contradict itself
# ---------------------------------------------------------------------------
def weekly_fields(**overrides: object) -> dict[str, object]:
    """The fields of one valid weekly schedule, with any of them replaced."""
    fields: dict[str, object] = {
        "start_date": date(2024, 1, 1),
        "end_date": date(2024, 1, 31),
        "frequency": WEEKLY,
        "anchor_weekday": "THU",
        "calendar_name": "XNYS",
        "biweekly_anchor": None,
        "dates": days("2024-01-04", "2024-01-11"),
    }
    fields.update(overrides)
    return fields


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"end_date": date(2023, 12, 31)}, "is after end_date"),
        ({"frequency": "daily"}, "frequency must be one of"),
        ({"anchor_weekday": "THU", "frequency": MONTHLY}, "cannot carry the anchor"),
        (
            {"frequency": MONTHLY, "anchor_weekday": None, "biweekly_anchor": BIWEEKLY_ANCHOR},
            "is not phased",
        ),
        ({"anchor_weekday": None}, "anchor_weekday must be one of"),
        ({"frequency": BIWEEKLY}, "biweekly_anchor must be"),
        ({"biweekly_anchor": BIWEEKLY_ANCHOR}, "biweekly_anchor must be None"),
        ({"calendar_name": "  "}, "must be a non-empty trading calendar name"),
        ({"dates": days("2024-01-11", "2024-01-04")}, "strictly ascending"),
        ({"dates": days("2024-01-04", "2024-01-04")}, "strictly ascending"),
        ({"dates": days("2023-12-28")}, "must lie inside"),
        ({"dates": (datetime(2024, 1, 4, 16),)}, "must be a plain date"),
        ({"dates": [date(2024, 1, 4)]}, "dates must be a tuple"),
        ({"start_date": datetime(2024, 1, 1, 16)}, "must be a plain date"),
    ],
)
def test_an_inconsistent_schedule_is_refused(overrides: dict[str, object], message: str) -> None:
    """The record checks itself, so no analysis can quote an impossible rule."""
    with pytest.raises(FactorScheduleError, match=message):
        DecisionSchedule(**weekly_fields(**overrides))  # type: ignore[arg-type]


def test_the_valid_weekly_fields_build_a_schedule() -> None:
    """The same fields, unmodified, are a valid record."""
    schedule = DecisionSchedule(**weekly_fields())  # type: ignore[arg-type]

    assert schedule.dates == days("2024-01-04", "2024-01-11")
    assert len(schedule) == 2


def test_a_hand_built_schedule_cannot_claim_a_weekend() -> None:
    """2024-01-06 is inside the window, but no exchange trades on a Saturday."""
    with pytest.raises(FactorScheduleError) as excinfo:
        DecisionSchedule(**weekly_fields(dates=days("2024-01-06")))  # type: ignore[arg-type]

    message = str(excinfo.value)
    assert "must all be trading sessions of XNYS" in message
    assert "these are not: 2024-01-06 (SAT)" in message


def test_a_hand_built_schedule_cannot_claim_an_exchange_holiday() -> None:
    """New Year's Day 2024 is a Monday: a weekday is not enough, the calendar decides."""
    with pytest.raises(FactorScheduleError) as excinfo:
        DecisionSchedule(**weekly_fields(dates=days("2024-01-01")))  # type: ignore[arg-type]

    assert "2024-01-01 (MON)" in str(excinfo.value)


def test_every_offending_date_is_named_once_and_long_lists_are_truncated() -> None:
    """The refusal names the dates it found, five of them plus a count of the rest."""
    six_saturdays = days(
        "2024-02-03", "2024-02-10", "2024-02-17", "2024-02-24", "2024-03-02", "2024-03-09"
    )
    with pytest.raises(FactorScheduleError) as excinfo:
        DecisionSchedule(
            **weekly_fields(
                start_date=date(2024, 1, 1),
                end_date=date(2024, 3, 31),
                dates=six_saturdays,
            )
        )  # type: ignore[arg-type]

    message = str(excinfo.value)
    assert "2024-02-03 (SAT)" in message
    assert "2024-03-02 (SAT)" in message
    assert "2024-03-09" not in message  # the sixth is behind the counter
    assert "(and 1 more)" in message


def test_a_hand_built_schedule_cannot_use_a_date_the_calendar_never_covered() -> None:
    """Nothing is fabricated from before the calendar's first session."""
    with pytest.raises(FactorScheduleError, match="must all be trading sessions"):
        DecisionSchedule(
            **weekly_fields(
                start_date=date(1990, 1, 1),
                end_date=date(1990, 12, 31),
                dates=days("1990-01-04"),
            )
        )  # type: ignore[arg-type]


def test_a_hand_built_schedule_with_an_unknown_calendar_is_refused() -> None:
    """A record whose dates cannot be verified against its own calendar proves nothing."""
    with pytest.raises(
        FactorScheduleError,
        match="cannot be checked against the trading calendar 'NOPE'",
    ):
        DecisionSchedule(**weekly_fields(calendar_name="NOPE"))  # type: ignore[arg-type]


def test_a_hand_built_schedule_that_selected_nothing_needs_no_calendar_verdict() -> None:
    """An empty selection has no date to verify, so it stays valid."""
    schedule = DecisionSchedule(**weekly_fields(dates=()))  # type: ignore[arg-type]

    assert schedule.dates == ()
    assert len(schedule) == 0


# ---------------------------------------------------------------------------
# 7. The module boundary: a date rule, nothing else
# ---------------------------------------------------------------------------
def schedule_tree() -> ast.Module:
    """The parsed source of the module under test."""
    return ast.parse(SOURCE.read_text(encoding="utf-8"))


def imported_modules(tree: ast.Module) -> set[str]:
    """Every module the module imports, by dotted name."""
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
    return modules


def referenced_names(tree: ast.Module) -> set[str]:
    """Every bare name and attribute the code mentions (strings excluded)."""
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    names |= {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    return names


def test_the_module_imports_only_dates_and_the_project_calendar() -> None:
    """Letting a dataframe, a store or a script in would break the purity claim."""
    modules = imported_modules(schedule_tree())
    forbidden = (
        "pandas",
        "numpy",
        "pyarrow",
        "duckdb",
        "factors",
        "scripts",
        "subprocess",
        "socket",
        "urllib",
        "http",
        "requests",
        "factor_sys.runner",
        "factor_sys.factor_store",
        "factor_sys.market_data",
        "factor_sys.quick_analysis",
        "factor_sys.forward_returns",
        "factor_sys.ic_analysis",
        "factor_sys.quantile_analysis",
        "factor_sys.registry",
        "factor_sys.universe",
        "data_sys.storage",
        "data_sys.update",
    )
    for module in sorted(modules):
        assert not any(
            module == banned or module.startswith(f"{banned}.") for banned in forbidden
        ), f"the schedule rule must not import {module!r}"

    allowed = {
        "__future__",
        "calendar",
        "re",
        "dataclasses",
        "datetime",
        "typing",
        "data_sys.errors",
        "data_sys.market_calendar",
        "factor_sys.errors",
    }
    assert modules <= allowed


def test_the_module_asks_the_project_calendar_instead_of_reimplementing_it() -> None:
    """Weeks, holidays and month ends all come from ``data_sys.market_calendar``."""
    names = referenced_names(schedule_tree())

    assert {"get_calendar", "sessions_in_range", "last_sessions", "calendar_bounds"} <= names
    # No holiday table of its own, and no hand-rolled weekday arithmetic on it.
    assert "holidays" not in names
    assert "business_days" not in names


def test_the_module_opens_no_file_and_holds_no_handle() -> None:
    """Selecting dates must cost a calendar lookup and nothing else."""
    names = referenced_names(schedule_tree())

    for forbidden in (
        "open",
        "Path",
        "read_text",
        "read_bytes",
        "write_text",
        "write_bytes",
        "mkdir",
        "unlink",
        "mkstemp",
        "replace",
        "exists",
        "glob",
        "rglob",
    ):
        assert forbidden not in names, f"the date rule must not use {forbidden!r}"

    assert not {"os", "pathlib", "json", "tempfile", "shutil"} & imported_modules(schedule_tree())
