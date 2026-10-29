"""Tests for :mod:`data_sys.market_calendar` (fully offline)."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from data_sys.errors import CalendarError
from data_sys.market_calendar import (
    DEFAULT_CALENDAR_NAME,
    calendar_bounds,
    clamp_window,
    get_calendar,
    is_session,
    last_sessions,
    next_session_after,
    session_index,
    sessions_in_range,
)

# 2024-01-01 New Year, 2024-07-04 Independence Day, 2024-12-25 Christmas,
# 2024-11-28 Thanksgiving, 2024-03-29 Good Friday, 2024-01-15 MLK Day.
HOLIDAYS_2024 = ("2024-01-01", "2024-07-04", "2024-12-25", "2024-11-28", "2024-03-29")


@pytest.fixture(scope="module")
def calendar():
    return get_calendar(DEFAULT_CALENDAR_NAME)


def test_calendar_is_cached_and_named(calendar) -> None:
    assert calendar.name == "XNYS"
    assert get_calendar("XNYS") is calendar


def test_unknown_calendar_raises_a_domain_error() -> None:
    with pytest.raises(CalendarError):
        get_calendar("NOT-A-CALENDAR")


def test_calendar_bounds(calendar) -> None:
    """The bounds are the calendar's own, not a date this file pinned.

    ``exchange_calendars`` has moved the XNYS first session between releases
    (2006-10-02 in some, 2006-10-30 in others), so asserting a literal date
    tests the installed version instead of the code under test.
    """
    first, last = calendar_bounds(calendar)

    assert type(first) is date
    assert type(last) is date
    assert first < last
    # both bounds really are sessions, and nothing precedes the first one
    assert sessions_in_range(calendar, first, first) == [first]
    assert sessions_in_range(calendar, last, last) == [last]
    assert is_session(calendar, first) is True
    assert is_session(calendar, first - timedelta(days=1)) is False


@pytest.mark.parametrize("day", HOLIDAYS_2024)
def test_exchange_holidays_are_not_sessions(calendar, day: str) -> None:
    assert sessions_in_range(calendar, day, day) == []
    assert is_session(calendar, date.fromisoformat(day)) is False


@pytest.mark.parametrize("day", ("2024-01-06", "2024-01-07", "2025-03-08"))
def test_weekends_return_no_sessions(calendar, day: str) -> None:
    assert sessions_in_range(calendar, day, day) == []


def test_weekend_window_returns_no_sessions_without_raising(calendar) -> None:
    assert sessions_in_range(calendar, "2024-01-06", "2024-01-07") == []


def test_inverted_window_returns_no_sessions_without_raising(calendar) -> None:
    assert sessions_in_range(calendar, "2024-02-01", "2024-01-01") == []


def test_sessions_are_plain_dates(calendar) -> None:
    sessions = sessions_in_range(calendar, "2024-01-02", "2024-01-05")

    assert sessions == [date(2024, 1, day) for day in (2, 3, 4, 5)]
    assert all(type(session) is date for session in sessions)


def test_holidays_are_skipped_inside_a_week(calendar) -> None:
    # 2024-01-15 is MLK Day
    assert sessions_in_range(calendar, "2024-01-15", "2024-01-19") == [
        date(2024, 1, 16),
        date(2024, 1, 17),
        date(2024, 1, 18),
        date(2024, 1, 19),
    ]


def test_a_start_before_the_first_session_is_clamped_not_rejected(calendar) -> None:
    # The request starts before the calendar does and ends inside it; the window
    # is intersected with the calendar, never rejected and never shifted.
    first_session, _ = calendar_bounds(calendar)
    requested_start = first_session - timedelta(days=30)
    requested_end = first_session + timedelta(days=60)

    sessions = sessions_in_range(calendar, requested_start, requested_end)

    assert sessions[0] == first_session
    assert sessions == sorted(sessions)
    assert sessions[-1] <= requested_end
    assert clamp_window(calendar, requested_start, requested_end) == (
        first_session,
        requested_end,
    )
    # a window that ends before the calendar begins holds no session at all
    assert sessions_in_range(calendar, requested_start, requested_start) == []
    assert clamp_window(calendar, requested_start, requested_start) is None


def test_clamp_window_of_a_fully_out_of_range_request_is_none(calendar) -> None:
    # Well before the calendar starts, whatever that start happens to be.
    first_session, _ = calendar_bounds(calendar)
    long_before = first_session - timedelta(days=3650)
    still_before = first_session - timedelta(days=1200)

    assert clamp_window(calendar, long_before, still_before) is None
    assert sessions_in_range(calendar, long_before, still_before) == []
    assert last_sessions(calendar, still_before, 5) == []


def test_clamped_request_still_clamps_an_end_after_the_last_session(calendar) -> None:
    _, last = calendar_bounds(calendar)

    window = clamp_window(calendar, date(2026, 1, 1), date(2099, 1, 1))

    assert window is not None
    assert window[1] == last


def test_last_sessions_ends_on_or_before_the_anchor(calendar) -> None:
    sessions = last_sessions(calendar, date(2026, 9, 30), 5)

    assert sessions == [
        date(2026, 9, 24),
        date(2026, 9, 25),
        date(2026, 9, 28),
        date(2026, 9, 29),
        date(2026, 9, 30),
    ]


def test_last_sessions_skips_a_weekend_going_backwards(calendar) -> None:
    # 2024-01-08 is a Monday, so looking back 5 sessions must cross the weekend.
    sessions = last_sessions(calendar, date(2024, 1, 8), 5)

    assert sessions == [
        date(2024, 1, 2),
        date(2024, 1, 3),
        date(2024, 1, 4),
        date(2024, 1, 5),
        date(2024, 1, 8),
    ]


def test_last_sessions_with_zero_or_negative_count_is_empty(calendar) -> None:
    assert last_sessions(calendar, date(2024, 1, 8), 0) == []
    assert last_sessions(calendar, date(2024, 1, 8), -3) == []


def test_last_sessions_is_capped_by_the_calendar_start(calendar) -> None:
    # Anchored on the calendar's real opening sessions, so the look-back is
    # capped by whatever the installed calendar actually starts at.
    first_session, _ = calendar_bounds(calendar)
    opening = sessions_in_range(calendar, first_session, first_session + timedelta(days=14))
    second_session = next_session_after(opening, first_session)
    assert second_session is not None

    # ten sessions are asked for, but the calendar holds two of them by then
    assert last_sessions(calendar, first_session, 10) == [first_session]
    assert last_sessions(calendar, second_session, 10) == [first_session, second_session]


def test_next_session_after_crosses_the_weekend(calendar) -> None:
    assert next_session_after([], date(2024, 1, 5)) is None

    sessions = sessions_in_range(calendar, "2024-01-04", "2024-01-09")
    assert next_session_after(sessions, date(2024, 1, 5)) == date(2024, 1, 8)
    assert next_session_after(sessions, sessions[-1]) is None


def test_session_index_and_is_session(calendar) -> None:
    sessions = sessions_in_range(calendar, "2024-01-02", "2024-01-05")

    assert session_index(sessions) == {
        date(2024, 1, 2): 0,
        date(2024, 1, 3): 1,
        date(2024, 1, 4): 2,
        date(2024, 1, 5): 3,
    }
    assert is_session(calendar, date(2024, 1, 2)) is True
    assert is_session(calendar, date(1990, 1, 2)) is False


def test_a_long_window_has_no_duplicate_sessions(calendar) -> None:
    sessions = sessions_in_range(calendar, "2020-01-01", "2021-12-31")

    assert len(sessions) == len(set(sessions))
    assert sessions == sorted(sessions)
