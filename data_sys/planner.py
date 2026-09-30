"""Download planning for incremental market-data updates.

This module answers exactly one question and nothing else:

    given a requested ``[start, end]`` window, the sessions already published
    locally, an NYSE calendar and an overlap-refresh width, **which ranges must
    be downloaded, and why?**

It performs no I/O when a calendar is supplied, so it is fully unit-testable
offline.  The rules it encodes:

===============================  ===============================================
situation                        planned download
===============================  ===============================================
no published data at all         the full requested window (``initial_backfill``)
missing newer sessions           tail gap (possibly split) + overlap refresh
missing sessions in the middle   one window per contiguous internal gap
request starts before local data forward gap (``head_backfill``)
request already fully covered    refresh of the last N sessions only
``refresh_overlap_sessions = 0``  nothing extra for refresh purposes
no session in the requested range no window at all (``no_sessions``)
===============================  ===============================================

"Contiguous" is measured on the *trading calendar*: two missing sessions belong
to the same gap when they are adjacent in the session sequence, so a missing
Friday and the following Monday form one gap, while a missing Thursday and
Monday do not.

Windows that overlap or that touch on the calendar are merged, so a tail gap
plus the overlap-refresh window covering it collapse into a single request.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from typing import Any

from data_sys.config import DEFAULT_REFRESH_OVERLAP_SESSIONS
from data_sys.errors import DataPipelineError
from data_sys.market_calendar import (
    DEFAULT_CALENDAR_NAME,
    clamp_window,
    get_calendar,
    last_sessions,
    next_session_after,
    sessions_in_range,
)
from data_sys.utils import parse_date

logger = logging.getLogger(__name__)

# -- why a window is downloaded ---------------------------------------------
REASON_INITIAL = "initial_backfill"
REASON_HEAD = "head_backfill"
REASON_INTERNAL = "internal_gap"
REASON_TAIL = "tail_gap"
REASON_OVERLAP = "overlap_refresh"

# -- planner outcome ---------------------------------------------------------
#: At least one window must be downloaded.
STATUS_PLANNED = "planned"
#: The published data already covers the request and no refresh was requested.
STATUS_UP_TO_DATE = "up_to_date"
#: The requested range contains no NYSE session -> safe skip, never a failure.
STATUS_NO_SESSIONS = "no_sessions"


@dataclass(frozen=True)
class DownloadWindow:
    """One ``[start, end]`` range to fetch from the provider."""

    start: date
    end: date
    reasons: tuple[str, ...] = ()

    @property
    def reason(self) -> str:
        """Label such as ``tail_gap+overlap_refresh``."""
        return "+".join(self.reasons)

    def as_iso_range(self) -> list[str]:
        """``[start, end]`` as ISO strings, ready for JSON."""
        return [self.start.isoformat(), self.end.isoformat()]

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly description of this window."""
        return {
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "reason": self.reason,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class UpdatePlan:
    """The complete, testable answer to "what must be downloaded for symbol X"."""

    symbol: str
    requested_start: date
    requested_end: date
    status: str = STATUS_PLANNED
    calendar_name: str = DEFAULT_CALENDAR_NAME
    refresh_overlap_sessions: int = 0

    existing_dates_count: int = 0
    existing_min_date: date | None = None
    existing_max_date: date | None = None

    effective_start: date | None = None
    effective_end: date | None = None
    expected_sessions: int = 0

    missing_sessions: tuple[date, ...] = ()
    head_gap: tuple[date, ...] = ()
    internal_gap: tuple[date, ...] = ()
    tail_gap: tuple[date, ...] = ()
    overlap_sessions: tuple[date, ...] = ()

    missing_ranges: tuple[tuple[date, date], ...] = ()
    refresh_range: tuple[date, date] | None = None
    windows: tuple[DownloadWindow, ...] = ()
    notes: tuple[str, ...] = ()

    # -- convenience ---------------------------------------------------------
    @property
    def has_work(self) -> bool:
        """``True`` when at least one download window must be fetched."""
        return bool(self.windows)

    @property
    def is_initial(self) -> bool:
        """``True`` for a symbol with no published data yet."""
        return self.existing_dates_count == 0

    @property
    def is_no_sessions(self) -> bool:
        """``True`` when the requested range holds no NYSE session."""
        return self.status == STATUS_NO_SESSIONS

    def as_metadata(self) -> dict[str, Any]:
        """JSON-friendly summary stored in the run metadata."""
        return {
            "status": self.status,
            "calendar": self.calendar_name,
            "refresh_overlap_sessions": self.refresh_overlap_sessions,
            "effective_start": self._iso(self.effective_start),
            "effective_end": self._iso(self.effective_end),
            "expected_sessions": self.expected_sessions,
            "existing_dates_count": self.existing_dates_count,
            "existing_min_date": self._iso(self.existing_min_date),
            "existing_max_date": self._iso(self.existing_max_date),
            "missing_sessions_count": len(self.missing_sessions),
            "head_gap_count": len(self.head_gap),
            "internal_gap_count": len(self.internal_gap),
            "tail_gap_count": len(self.tail_gap),
            "overlap_sessions_count": len(self.overlap_sessions),
            "missing_ranges": [list(pair) for pair in self.missing_ranges_iso()],
            "refresh_range": self.refresh_range_iso(),
            "windows": [window.to_dict() for window in self.windows],
            "notes": list(self.notes),
        }

    def missing_ranges_iso(self) -> list[list[str]]:
        """Missing (gap) ranges as ``[[start, end], ...]`` ISO pairs."""
        return [[lo.isoformat(), hi.isoformat()] for lo, hi in self.missing_ranges]

    def refresh_range_iso(self) -> list[str] | None:
        """Overlap-refresh range as ``[start, end]``, or ``None``."""
        if self.refresh_range is None:
            return None
        return [self.refresh_range[0].isoformat(), self.refresh_range[1].isoformat()]

    @staticmethod
    def _iso(value: date | None) -> str | None:
        return None if value is None else value.isoformat()


# ---------------------------------------------------------------------------
# Planning helpers
# ---------------------------------------------------------------------------
def _split_runs(sessions: list[date], selected: set[date]) -> list[list[date]]:
    """Group ``selected`` into runs of sessions adjacent *in the session list*.

    Walking the ordered session list and cutting whenever a session is absent
    gives trading-calendar adjacency for free: a missing Friday followed by a
    missing Monday stays in the same run because the weekend is not in the list.
    """
    runs: list[list[date]] = []
    current: list[date] = []
    for session in sessions:
        if session in selected:
            current.append(session)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    return runs


def merge_windows(
    windows: Iterable[DownloadWindow], sessions: list[date]
) -> list[DownloadWindow]:
    """Merge overlapping or calendar-adjacent windows into single requests.

    Two windows are merged when they overlap, or when the second starts on the
    session immediately after the first one ends: in both cases one provider
    call fetches the union, so issuing two would be a wasted request.
    """
    ordered = sorted(windows, key=lambda window: (window.start, window.end))
    merged: list[DownloadWindow] = []
    for window in ordered:
        if not merged:
            merged.append(window)
            continue
        previous = merged[-1]
        following = next_session_after(sessions, previous.end)
        touches = window.start <= previous.end or (
            following is not None and window.start <= following
        )
        if touches:
            reasons = previous.reasons + tuple(
                reason for reason in window.reasons if reason not in previous.reasons
            )
            merged[-1] = DownloadWindow(
                start=previous.start,
                end=max(previous.end, window.end),
                reasons=reasons,
            )
        else:
            merged.append(window)
    return merged


# ---------------------------------------------------------------------------
# Public planning entry point
# ---------------------------------------------------------------------------
def plan_download(
    symbol: str,
    start: date | str,
    end: date | str,
    existing_dates: Iterable[date | str] | None = None,
    *,
    calendar: Any | None = None,
    calendar_name: str = DEFAULT_CALENDAR_NAME,
    refresh_overlap_sessions: int = DEFAULT_REFRESH_OVERLAP_SESSIONS,
) -> UpdatePlan:
    """Plan the downloads needed to bring ``symbol`` up to date.

    Parameters
    ----------
    symbol:
        Ticker the plan belongs to (upper-cased in the result).
    start, end:
        Inclusive requested window, as ``date`` or ``YYYY-MM-DD`` strings.
    existing_dates:
        Dates already published for ``symbol`` (any iterable of dates/strings).
        ``None`` or empty means "no published data yet".
    calendar:
        Pre-built ``exchange_calendars`` calendar; loaded from ``calendar_name``
        when omitted.
    calendar_name:
        Name of the calendar to load (default ``XNYS``).
    refresh_overlap_sessions:
        Number of most recent sessions to re-download even when nothing is
        missing.  ``0`` disables refresh-only downloads.

    Returns
    -------
    UpdatePlan
        Never raises for an empty or inverted window: such a request comes back
        as :data:`STATUS_NO_SESSIONS` so the caller can skip it safely.
    """
    calendar = calendar if calendar is not None else get_calendar(calendar_name)
    overlap = max(0, int(refresh_overlap_sessions))

    requested_start = parse_date(start)
    requested_end = parse_date(end)
    if requested_start is None or requested_end is None:
        raise DataPipelineError("both a start and an end date are required to plan a download")

    present: set[date] = set()
    for value in existing_dates or ():
        parsed = parse_date(value)
        if parsed is not None:
            present.add(parsed)

    base: dict[str, Any] = {
        "symbol": str(symbol).strip().upper(),
        "requested_start": requested_start,
        "requested_end": requested_end,
        "calendar_name": calendar_name,
        "refresh_overlap_sessions": overlap,
        "existing_dates_count": len(present),
        "existing_min_date": min(present) if present else None,
        "existing_max_date": max(present) if present else None,
    }

    if requested_start > requested_end:
        return UpdatePlan(
            **base,
            status=STATUS_NO_SESSIONS,
            notes=[
                f"requested start {requested_start.isoformat()} is after "
                f"requested end {requested_end.isoformat()}"
            ],
        )

    window = clamp_window(calendar, requested_start, requested_end)
    if window is None:
        first, last = calendar.first_session.date(), calendar.last_session.date()
        return UpdatePlan(
            **base,
            status=STATUS_NO_SESSIONS,
            notes=[
                f"requested window {requested_start.isoformat()}.."
                f"{requested_end.isoformat()} is outside the {calendar_name} "
                f"calendar bounds ({first.isoformat()}..{last.isoformat()})"
            ],
        )

    effective_start, effective_end = window
    notes: list[str] = []
    if effective_start != requested_start:
        notes.append(
            f"requested start {requested_start.isoformat()} clamped to the first "
            f"{calendar_name} session ({effective_start.isoformat()})"
        )

    sessions = sessions_in_range(calendar, effective_start, effective_end)
    if not sessions:
        notes.append(
            f"no {calendar_name} session between {effective_start.isoformat()} "
            f"and {effective_end.isoformat()}"
        )
        return UpdatePlan(
            **base,
            status=STATUS_NO_SESSIONS,
            effective_start=effective_start,
            effective_end=effective_end,
            notes=tuple(notes),
        )

    base["effective_start"] = effective_start
    base["effective_end"] = effective_end
    base["expected_sessions"] = len(sessions)

    # -- brand-new symbol: a single request for the whole requested window ----
    if not present:
        notes.append("no published data for this symbol: full backfill requested")
        return UpdatePlan(
            **base,
            status=STATUS_PLANNED,
            missing_sessions=tuple(sessions),
            missing_ranges=((effective_start, effective_end),),
            windows=(DownloadWindow(effective_start, effective_end, (REASON_INITIAL,)),),
            notes=tuple(notes),
        )

    # -- existing symbol: locate the holes ------------------------------------
    existing_min = min(present)
    existing_max = max(present)
    missing = [session for session in sessions if session not in present]
    head = [session for session in missing if session < existing_min]
    tail = [session for session in missing if session > existing_max]
    internal = [session for session in missing if existing_min < session < existing_max]

    # `head` and `tail` are contiguous by construction (nothing is published
    # outside [existing_min, existing_max]); `internal` may hold several runs.
    gap_runs: list[tuple[str, list[date]]] = []
    for group in _split_runs(sessions, set(head)):
        gap_runs.append((REASON_HEAD, group))
    for group in _split_runs(sessions, set(internal)):
        gap_runs.append((REASON_INTERNAL, group))
    for group in _split_runs(sessions, set(tail)):
        gap_runs.append((REASON_TAIL, group))

    gap_windows = [
        DownloadWindow(group[0], group[-1], (reason,)) for reason, group in gap_runs
    ]
    missing_ranges = tuple((group[0], group[-1]) for _, group in gap_runs)

    # -- overlap refresh: re-fetch recent sessions Yahoo may have revised -----
    overlap_sessions: list[date] = []
    if overlap:
        overlap_sessions = [
            session
            for session in last_sessions(calendar, effective_end, overlap)
            if effective_start <= session <= effective_end
        ]
    else:
        notes.append("refresh overlap disabled (--no-refresh / 0 sessions)")

    refresh_range: tuple[date, date] | None = None
    if overlap_sessions:
        refresh_range = (overlap_sessions[0], overlap_sessions[-1])
        gap_windows.append(
            DownloadWindow(overlap_sessions[0], overlap_sessions[-1], (REASON_OVERLAP,))
        )

    windows = merge_windows(gap_windows, sessions)
    status = STATUS_PLANNED if windows else STATUS_UP_TO_DATE
    if not missing and overlap_sessions:
        notes.append(
            f"nothing missing: only the last {len(overlap_sessions)} session(s) "
            "are refreshed"
        )
    if status == STATUS_UP_TO_DATE:
        notes.append("published data already covers the requested window")

    return UpdatePlan(
        **base,
        status=status,
        missing_sessions=tuple(missing),
        head_gap=tuple(head),
        internal_gap=tuple(internal),
        tail_gap=tuple(tail),
        overlap_sessions=tuple(overlap_sessions),
        missing_ranges=missing_ranges,
        refresh_range=refresh_range,
        windows=tuple(windows),
        notes=tuple(notes),
    )


__all__ = [
    "DownloadWindow",
    "UpdatePlan",
    "REASON_HEAD",
    "REASON_INITIAL",
    "REASON_INTERNAL",
    "REASON_OVERLAP",
    "REASON_TAIL",
    "STATUS_NO_SESSIONS",
    "STATUS_PLANNED",
    "STATUS_UP_TO_DATE",
    "merge_windows",
    "plan_download",
]
