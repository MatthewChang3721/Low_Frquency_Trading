"""Tests for :mod:`data_sys.planner` -- the incremental download planning rules.

Offline by construction: the planner only needs an ``exchange_calendars``
calendar, which ships with the holiday rules.
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from data_sys.errors import DataPipelineError
from data_sys.market_calendar import get_calendar, last_sessions, sessions_in_range
from data_sys.planner import (
    REASON_HEAD,
    REASON_INITIAL,
    REASON_INTERNAL,
    REASON_OVERLAP,
    REASON_TAIL,
    STATUS_NO_SESSIONS,
    STATUS_PLANNED,
    STATUS_UP_TO_DATE,
    DownloadWindow,
    merge_windows,
    plan_download,
)


@pytest.fixture(scope="module")
def calendar():
    return get_calendar("XNYS")


def _published(start: str, end: str, *, drop: tuple[str, ...] = ()) -> list[date]:
    """Sessions in ``[start, end]`` minus the ISO dates in ``drop``."""
    removed = {date.fromisoformat(day) for day in drop}
    return [
        session
        for session in sessions_in_range(get_calendar("XNYS"), start, end)
        if session not in removed
    ]


# ---------------------------------------------------------------------------
# 1. brand-new symbol
# ---------------------------------------------------------------------------
def test_new_symbol_downloads_the_whole_requested_window(calendar) -> None:
    plan = plan_download("aapl", "2024-01-02", "2024-03-29", set(), calendar=calendar)

    assert plan.symbol == "AAPL"
    assert plan.status == STATUS_PLANNED
    assert plan.is_initial is True
    assert plan.has_work is True
    assert plan.windows == (
        DownloadWindow(date(2024, 1, 2), date(2024, 3, 29), (REASON_INITIAL,)),
    )
    assert plan.expected_sessions == len(sessions_in_range(calendar, "2024-01-02", "2024-03-29"))
    assert plan.missing_sessions == tuple(_published("2024-01-02", "2024-03-29"))
    assert plan.missing_ranges == ((date(2024, 1, 2), date(2024, 3, 29)),)
    assert plan.refresh_range is None
    assert plan.existing_dates_count == 0


def test_new_symbol_window_uses_the_request_not_the_session_span(calendar) -> None:
    """The window is the requested range, so an end on a holiday is honoured."""
    plan = plan_download("AAPL", "2022-01-01", "2022-01-31", None, calendar=calendar)

    assert plan.windows[0].end == date(2022, 1, 31)
    # 2022-01-17 is MLK Day, so fewer sessions than calendar days
    assert plan.expected_sessions == len(sessions_in_range(calendar, "2022-01-01", "2022-01-31"))


def test_new_symbol_does_not_add_a_refresh_window(calendar) -> None:
    plan = plan_download(
        "AAPL", "2024-01-02", "2024-03-29", set(), calendar=calendar
    )

    assert len(plan.windows) == 1
    assert plan.overlap_sessions == ()
    assert REASON_OVERLAP not in plan.windows[0].reasons
    assert any("no published data" in note for note in plan.notes)


# ---------------------------------------------------------------------------
# 2. tail gap
# ---------------------------------------------------------------------------
def test_tail_gap_is_merged_with_the_overlap_window(calendar) -> None:
    """The documented example: covered through 09-28, request ends 09-30."""
    existing = _published("2026-01-02", "2026-09-28")

    plan = plan_download(
        "AAPL", "2026-01-02", "2026-09-30", existing, calendar=calendar,
        refresh_overlap_sessions=5,
    )

    assert plan.status == STATUS_PLANNED
    assert plan.tail_gap == (date(2026, 9, 29), date(2026, 9, 30))
    assert plan.missing_sessions == plan.tail_gap
    assert plan.missing_ranges == ((date(2026, 9, 29), date(2026, 9, 30)),)
    assert plan.refresh_range == (date(2026, 9, 24), date(2026, 9, 30))
    # one request, not two: the tail gap sits inside the refresh window
    assert len(plan.windows) == 1
    assert plan.windows[0].start == date(2026, 9, 24)
    assert plan.windows[0].end == date(2026, 9, 30)
    assert set(plan.windows[0].reasons) == {REASON_OVERLAP, REASON_TAIL}
    assert plan.windows[0].reason == "overlap_refresh+tail_gap"


def test_tail_gap_is_reported_separately_from_the_refresh_range(calendar) -> None:
    existing = _published("2024-01-02", "2024-01-31")

    plan = plan_download(
        "AAPL", "2024-01-02", "2024-02-29", existing, calendar=calendar,
        refresh_overlap_sessions=3,
    )

    assert plan.missing_ranges == ((date(2024, 2, 1), date(2024, 2, 29)),)
    assert plan.refresh_range == (date(2024, 2, 27), date(2024, 2, 29))
    assert plan.tail_gap[0] == date(2024, 2, 1)
    assert plan.notes == ()


# ---------------------------------------------------------------------------
# 3. internal gap
# ---------------------------------------------------------------------------
def test_internal_gap_becomes_its_own_window(calendar) -> None:
    existing = _published("2024-01-02", "2024-03-29", drop=("2024-03-05", "2024-03-06"))

    plan = plan_download(
        "AAPL", "2024-01-02", "2024-03-29", existing, calendar=calendar,
        refresh_overlap_sessions=5,
    )

    assert plan.internal_gap == (date(2024, 3, 5), date(2024, 3, 6))
    assert plan.head_gap == ()
    assert plan.tail_gap == ()
    assert plan.missing_ranges == ((date(2024, 3, 5), date(2024, 3, 6)),)

    # 2024-03-29 is Good Friday, so the last 5 sessions end on 2024-03-28.
    windows = {window.start: window for window in plan.windows}
    assert set(windows) == {date(2024, 3, 5), date(2024, 3, 22)}
    assert windows[date(2024, 3, 5)].reasons == (REASON_INTERNAL,)
    assert windows[date(2024, 3, 5)].end == date(2024, 3, 6)
    assert windows[date(2024, 3, 22)].reasons == (REASON_OVERLAP,)
    assert windows[date(2024, 3, 22)].end == date(2024, 3, 28)


def test_two_internal_gaps_become_two_windows(calendar) -> None:
    existing = _published(
        "2024-01-02", "2024-03-29", drop=("2024-02-01", "2024-03-11", "2024-03-12")
    )

    plan = plan_download(
        "AAPL", "2024-01-02", "2024-03-29", existing, calendar=calendar,
        refresh_overlap_sessions=0,
    )

    assert plan.missing_ranges == (
        (date(2024, 2, 1), date(2024, 2, 1)),
        (date(2024, 3, 11), date(2024, 3, 12)),
    )
    assert [window.start for window in plan.windows] == [date(2024, 2, 1), date(2024, 3, 11)]
    assert all(window.reasons == (REASON_INTERNAL,) for window in plan.windows)


def test_a_weekend_is_not_an_internal_gap(calendar) -> None:
    """Friday + Monday present and the weekend absent -> nothing to download."""
    existing = [date(2024, 1, 4), date(2024, 1, 5), date(2024, 1, 8), date(2024, 1, 9)]

    plan = plan_download(
        "AAPL", "2024-01-04", "2024-01-09", existing, calendar=calendar,
        refresh_overlap_sessions=0,
    )

    assert plan.missing_sessions == ()
    assert plan.windows == ()
    assert plan.status == STATUS_UP_TO_DATE


def test_friday_and_monday_missing_are_a_single_gap(calendar) -> None:
    """Adjacency is measured on the session sequence, so the weekend is bridged."""
    existing = _published("2024-01-02", "2024-01-31", drop=("2024-01-05", "2024-01-08"))

    plan = plan_download(
        "AAPL", "2024-01-02", "2024-01-31", existing, calendar=calendar,
        refresh_overlap_sessions=0,
    )

    assert plan.internal_gap == (date(2024, 1, 5), date(2024, 1, 8))
    assert plan.missing_ranges == ((date(2024, 1, 5), date(2024, 1, 8)),)
    assert len(plan.windows) == 1


# ---------------------------------------------------------------------------
# 4. forward backfill: the request starts before the local history
# ---------------------------------------------------------------------------
def test_head_gap_backfills_the_history_before_the_local_minimum(calendar) -> None:
    existing = _published("2024-06-03", "2024-12-31")

    plan = plan_download(
        "AAPL", "2024-01-02", "2024-12-31", existing, calendar=calendar,
        refresh_overlap_sessions=5,
    )

    assert plan.existing_min_date == date(2024, 6, 3)
    assert plan.head_gap[0] == date(2024, 1, 2)
    assert plan.head_gap[-1] == date(2024, 5, 31)
    assert plan.internal_gap == ()
    assert plan.tail_gap == ()
    assert plan.missing_ranges == ((date(2024, 1, 2), date(2024, 5, 31)),)

    head = plan.windows[0]
    assert head.reasons == (REASON_HEAD,)
    assert head.start == date(2024, 1, 2)
    assert head.end == date(2024, 5, 31)
    # the refresh window at the far end is a second, non-adjacent request
    assert len(plan.windows) == 2
    assert plan.windows[1].reasons == (REASON_OVERLAP,)
    assert plan.refresh_range == (date(2024, 12, 24), date(2024, 12, 31))


def test_head_gap_window_starts_on_a_session_when_the_request_starts_on_a_holiday(
    calendar,
) -> None:
    existing = _published("2024-06-03", "2024-12-31")

    plan = plan_download(
        "AAPL", "2024-01-01", "2024-12-31", existing, calendar=calendar,
        refresh_overlap_sessions=0,
    )

    assert plan.effective_start == date(2024, 1, 1)
    assert plan.windows[0].start == date(2024, 1, 2)  # 2024-01-01 is a holiday


# ---------------------------------------------------------------------------
# 5. fully covered: only the last N sessions are refreshed
# ---------------------------------------------------------------------------
def test_fully_covered_history_only_refreshes_the_last_n_sessions(calendar) -> None:
    existing = _published("2024-01-02", "2024-06-28")

    plan = plan_download(
        "AAPL", "2024-01-02", "2024-06-28", existing, calendar=calendar,
        refresh_overlap_sessions=5,
    )

    expected = last_sessions(calendar, date(2024, 6, 28), 5)
    assert plan.missing_sessions == ()
    assert plan.missing_ranges == ()
    assert plan.status == STATUS_PLANNED
    assert plan.overlap_sessions == tuple(expected)
    assert plan.refresh_range == (expected[0], expected[-1])
    assert plan.windows == (DownloadWindow(expected[0], expected[-1], (REASON_OVERLAP,)),)
    assert any("nothing missing" in note for note in plan.notes)


def test_the_refresh_window_never_reaches_before_the_requested_start(calendar) -> None:
    existing = [date(2024, 1, 2), date(2024, 1, 3)]

    plan = plan_download(
        "AAPL", "2024-01-02", "2024-01-03", existing, calendar=calendar,
        refresh_overlap_sessions=5,
    )

    assert plan.overlap_sessions == (date(2024, 1, 2), date(2024, 1, 3))
    assert plan.windows == (
        DownloadWindow(date(2024, 1, 2), date(2024, 1, 3), (REASON_OVERLAP,)),
    )


# ---------------------------------------------------------------------------
# 6. refresh disabled
# ---------------------------------------------------------------------------
def test_refresh_disabled_means_no_download_at_all(calendar) -> None:
    existing = _published("2024-01-02", "2024-06-28")

    plan = plan_download(
        "AAPL", "2024-01-02", "2024-06-28", existing, calendar=calendar,
        refresh_overlap_sessions=0,
    )

    assert plan.status == STATUS_UP_TO_DATE
    assert plan.windows == ()
    assert plan.has_work is False
    assert plan.refresh_range is None
    assert plan.overlap_sessions == ()
    assert any("refresh overlap disabled" in note for note in plan.notes)


def test_a_negative_refresh_width_behaves_like_zero(calendar) -> None:
    existing = _published("2024-01-02", "2024-06-28")

    plan = plan_download(
        "AAPL", "2024-01-02", "2024-06-28", existing, calendar=calendar,
        refresh_overlap_sessions=-4,
    )

    assert plan.refresh_overlap_sessions == 0
    assert plan.windows == ()


def test_refresh_disabled_still_fills_real_gaps(calendar) -> None:
    existing = _published("2024-01-02", "2024-06-28", drop=("2024-03-05",))

    plan = plan_download(
        "AAPL", "2024-01-02", "2024-06-28", existing, calendar=calendar,
        refresh_overlap_sessions=0,
    )

    assert plan.windows == (
        DownloadWindow(date(2024, 3, 5), date(2024, 3, 5), (REASON_INTERNAL,)),
    )


# ---------------------------------------------------------------------------
# 7. no session in the requested range -> safe skip
# ---------------------------------------------------------------------------
def test_a_weekend_only_request_is_a_safe_skip(calendar) -> None:
    plan = plan_download("AAPL", "2024-01-06", "2024-01-07", None, calendar=calendar)

    assert plan.status == STATUS_NO_SESSIONS
    assert plan.is_no_sessions is True
    assert plan.windows == ()
    assert plan.expected_sessions == 0
    assert plan.effective_start == date(2024, 1, 6)
    assert any("no XNYS session" in note for note in plan.notes)


def test_a_single_holiday_request_is_a_safe_skip(calendar) -> None:
    plan = plan_download("AAPL", "2024-07-04", "2024-07-04", None, calendar=calendar)

    assert plan.status == STATUS_NO_SESSIONS
    assert plan.windows == ()


def test_a_safe_skip_leaves_existing_history_alone(calendar) -> None:
    existing = _published("2024-01-02", "2024-06-28")

    plan = plan_download(
        "AAPL", "2024-07-06", "2024-07-07", existing, calendar=calendar,
        refresh_overlap_sessions=5,
    )

    assert plan.status == STATUS_NO_SESSIONS
    assert plan.windows == ()
    assert plan.missing_ranges == ()


def test_an_inverted_request_is_a_safe_skip(calendar) -> None:
    plan = plan_download("AAPL", "2024-06-03", "2024-01-02", None, calendar=calendar)

    assert plan.status == STATUS_NO_SESSIONS
    assert plan.windows == ()
    assert any("is after requested end" in note for note in plan.notes)


def test_a_request_entirely_before_the_calendar_is_a_safe_skip(calendar) -> None:
    plan = plan_download("AAPL", "1990-01-01", "1990-12-31", None, calendar=calendar)

    assert plan.status == STATUS_NO_SESSIONS
    assert plan.windows == ()
    assert any("outside the XNYS calendar bounds" in note for note in plan.notes)


# ---------------------------------------------------------------------------
# clamping, inputs and serialization
# ---------------------------------------------------------------------------
def test_a_request_before_the_calendar_start_is_clamped_and_recorded(calendar) -> None:
    plan = plan_download("AAPL", "2006-01-01", "2006-12-31", None, calendar=calendar)

    assert plan.requested_start == date(2006, 1, 1)
    assert plan.effective_start == date(2006, 10, 2)
    assert plan.windows[0].start == date(2006, 10, 2)
    assert plan.windows[0].end == date(2006, 12, 31)
    assert any("clamped to the first XNYS session" in note for note in plan.notes)


def test_existing_dates_may_be_strings_dates_or_duplicates(calendar) -> None:
    plan = plan_download(
        "AAPL",
        "2024-01-02",
        "2024-01-05",
        ["2024-01-02", date(2024, 1, 3), "2024-01-03", date(2024, 1, 4)],
        calendar=calendar,
        refresh_overlap_sessions=0,
    )

    assert plan.existing_dates_count == 3
    assert plan.missing_sessions == (date(2024, 1, 5),)
    assert plan.existing_min_date == date(2024, 1, 2)
    assert plan.existing_max_date == date(2024, 1, 4)


def test_dates_outside_the_request_are_ignored_when_looking_for_gaps(calendar) -> None:
    """A hole outside the requested window is not this run's business."""
    existing = _published("2022-01-03", "2024-12-31", drop=("2022-06-15",))

    plan = plan_download(
        "AAPL", "2024-01-02", "2024-12-31", existing, calendar=calendar,
        refresh_overlap_sessions=0,
    )

    assert plan.missing_sessions == ()
    assert plan.windows == ()
    assert plan.existing_min_date == date(2022, 1, 3)


def test_a_missing_start_or_end_is_a_programming_error(calendar) -> None:
    with pytest.raises(DataPipelineError):
        plan_download("AAPL", None, "2024-01-05", None, calendar=calendar)


def test_the_plan_is_json_serializable(calendar) -> None:
    existing = _published("2024-01-02", "2024-03-29", drop=("2024-03-05",))

    plan = plan_download("AAPL", "2024-01-02", "2024-04-30", existing, calendar=calendar)

    payload = plan.as_metadata()

    assert json.loads(json.dumps(payload)) == payload
    assert payload["status"] == STATUS_PLANNED
    assert payload["missing_ranges"] == [
        ["2024-03-05", "2024-03-05"],
        ["2024-04-01", "2024-04-30"],
    ]
    assert payload["refresh_range"] == ["2024-04-24", "2024-04-30"]
    assert [window["reason"] for window in payload["windows"]] == [
        "internal_gap",
        "tail_gap+overlap_refresh",
    ]
    assert payload["expected_sessions"] > 0
    assert payload["calendar"] == "XNYS"


# ---------------------------------------------------------------------------
# merge_windows in isolation
# ---------------------------------------------------------------------------
def test_merge_windows_collapses_overlaps_and_keeps_gap_reasons(calendar) -> None:
    sessions = sessions_in_range(calendar, "2024-01-02", "2024-01-12")

    merged = merge_windows(
        [
            DownloadWindow(date(2024, 1, 2), date(2024, 1, 5), (REASON_HEAD,)),
            DownloadWindow(date(2024, 1, 8), date(2024, 1, 9), (REASON_INTERNAL,)),
            DownloadWindow(date(2024, 1, 4), date(2024, 1, 8), (REASON_OVERLAP,)),
        ],
        sessions,
    )

    # Reasons are appended in merge order, which follows the window sort order
    # (start date), so the reason label is deterministic.
    assert merged == [
        DownloadWindow(
            date(2024, 1, 2),
            date(2024, 1, 9),
            (REASON_HEAD, REASON_OVERLAP, REASON_INTERNAL),
        )
    ]
    assert set(merged[0].reasons) == {REASON_HEAD, REASON_OVERLAP, REASON_INTERNAL}


def test_merge_windows_keeps_windows_separated_by_a_present_session(calendar) -> None:
    sessions = sessions_in_range(calendar, "2024-01-02", "2024-01-12")

    merged = merge_windows(
        [
            DownloadWindow(date(2024, 1, 2), date(2024, 1, 2), (REASON_INTERNAL,)),
            DownloadWindow(date(2024, 1, 4), date(2024, 1, 4), (REASON_INTERNAL,)),
        ],
        sessions,
    )

    assert len(merged) == 2
    assert merged[0].start == date(2024, 1, 2)
    assert merged[1].start == date(2024, 1, 4)


def test_merge_windows_of_nothing_is_nothing(calendar) -> None:
    assert merge_windows([], sessions_in_range(calendar, "2024-01-02", "2024-01-05")) == []




