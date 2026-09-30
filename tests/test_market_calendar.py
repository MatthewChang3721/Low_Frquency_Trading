"""Tests for :mod:`data_sys.market_calendar` (fully offline)."""

from __future__ import annotations

from datetime import date

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
    first, last = calendar_bounds(calendar)

    assert first == date(2006, 10, 2)
    assert last > date(2026, 1, 1)


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
    sessions = sessions_in_range(calendar, "2006-01-01", "2006-12-31")

    assert sessions[0] == date(2006, 10, 2)
    assert sessions[-1] == date(2006, 12, 29)
    assert clamp_window(calendar, date(2006, 1, 1), date(2006, 12, 31)) == (
        date(2006, 10, 2),
        date(2006, 12, 31),
    )


def test_clamp_window_of_a_fully_out_of_range_request_is_none(calendar) -> None:
    assert clamp_window(calendar, date(1990, 1, 1), date(1990, 12, 31)) is None
    assert sessions_in_range(calendar, "1990-01-01", "1990-12-31") == []
    assert last_sessions(calendar, date(1990, 6, 1), 5) == []


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
    sessions = last_sessions(calendar, date(2006, 10, 3), 10)

    assert sessions[0] == date(2006, 10, 2)
    assert len(sessions) == 2


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
