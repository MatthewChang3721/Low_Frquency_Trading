"""Run one factor over a schedule of decision dates: many sessions, one manifest.

What this script is
-------------------
The multi-date face of the single-day entry point.  It owns no formula, no
storage rule, no quality rule and no cache decision -- it asks
:func:`factor_sys.factor_schedule.decision_schedule` *which* sessions the
requested cadence selects, then calls ``scripts.run_factor.run_factor`` once per
session and writes down what happened::

     1  default_factor_registry().create(name)   the factor, and its version
     2  decision_schedule(...)                   which sessions this run decides on
     3  PipelineConfig(data_root=...).data_root  where the manifest is written
     4  run_factor(name, day, ...) per date      hit / miss / forced / failed, one
                                                 date at a time, exactly as a
                                                 single run does
     5  write_manifest(...)                      one JSON record of the whole batch

Step 4 is the *same function* a human runs, with the same flags.  Nothing here
reads bars, computes a value, decides a cache hit, or writes a panel: the batch
adds exactly three things -- the loop, per-date failure isolation and the
manifest.

The cadences
------------
``--frequency weekly`` (the default) and ``--anchor-weekday THU`` (the default)
select one session per calendar week: that Thursday when the exchange trades,
otherwise the closest session before it.  A Thursday anchor is the classic
signal-timing choice -- the signal is formed on Thursday's close, so an order
could be worked on Friday's open -- but this tool places no order, holds no
position and runs no backtest; it only computes factor cross-sections.

``--frequency biweekly`` keeps every second of those weekly candidates, phased on
the first candidate inside the requested window, and records that phase in the
manifest.  ``--frequency monthly`` takes the last session of each month and has
no anchor, so ``--anchor-weekday`` together with ``monthly`` is refused (exit
code ``2``) instead of being silently ignored.  The exact rules live in
:mod:`factor_sys.factor_schedule`.

Usage (project root, virtual environment active)::

    python scripts/run_factor_batch.py --factor momentum_12_1 `
        --start 2024-01-01 --end 2024-12-31 --frequency weekly --anchor-weekday THU

    python scripts/run_factor_batch.py --factor momentum_12_1 `
        --start 2024-01-01 --end 2024-12-31 --frequency monthly

    # recompute every selected date, replacing each stored panel atomically:
    python scripts/run_factor_batch.py --factor momentum_12_1 `
        --start 2024-01-01 --end 2024-12-31 --force

``--data-root``, ``--universe-path`` and ``--calendar`` mean exactly what they
mean for a single run and are passed through unchanged.

The cache is the same key
-------------------------
An existing panel is identified by ``(factor_name, factor_version,
as_of_date)`` alone: the cadence and the anchor only decide *which* dates this
batch visits, so a date already stored by any earlier run is read back
(``hit``) instead of being computed twice or stored twice.  Only a missing key
is computed (``miss``), and ``--force`` recomputes every selected date
(``forced``).  The frequency is never part of the key, the path or the panel.

One date failing never stops the others
---------------------------------------
A damaged stored panel, a missing dataset, a factor that raised: the date is
recorded in ``failures`` and the loop continues with the next session.  Nothing
is rolled back, deleted or overwritten because a neighbouring date failed, and a
quality finding is **not** a failure (it is printed and still exits ``0``).
An unknown factor or an unknown calendar fails before the loop, because
retrying it once per date would only produce the same message N times.

What is printed
---------------
One status line per selected date in schedule order, then one line per failure
with its whole message, then a stable block of ``key: value`` summary lines
ending with ``manifest_path``::

    2024-01-04  miss    clean
    2024-01-11  hit     clean
    2024-01-18  failed  FactorStoreError
    failure: 2024-01-18 FactorStoreError: ...
    factor_name: momentum_12_1
    ...
    selected_dates: 3
    computed_count: 1
    cached_count: 1
    forced_count: 0
    failed_count: 1
    manifest_path: data/factors/batch_runs/20240102T030405Z_9f8e7d.json

Exit codes
----------
``0``
    every selected date succeeded and the manifest was written.  A selection
    with no date at all is also ``0``: the request was honoured, and the manifest
    records that nothing was due.
``1``
    at least one date failed, or the manifest could not be written; the message
    goes to stderr as one ``error: ...`` line, never as a traceback.
``2``
    argparse refused the command line: a missing or unknown flag, an unknown
    ``--frequency`` / ``--anchor-weekday``, or ``--anchor-weekday`` combined with
    ``--frequency monthly``.

The manifest
------------
Every run writes a new ``<data_root>/factors/batch_runs/<batch_run_id>.json``
(temp file plus one :func:`os.replace`, so an existing manifest is never
half-written).  It records the exact window, cadence, anchor, calendar, the full
``selected_dates`` list, one result per date (status, quality status, panel
path relative to ``data_root``) and every failure with its error type and
message.  A manifest is a *record of one run*, not a cache key: running the same
request twice intentionally produces two independent, comparable manifests.

Future research must consume either this manifest or the same
``decision_schedule`` call.  Pooling every stored panel into one IC or quantile
series would mix weekly, biweekly and monthly runs -- different overlapping
decision dates -- into an average that describes no strategy at all.

What this script is not
-----------------------
It computes no forward return, no IC, no quantile spread, no strategy return and
no chart: it only fills in (or reads back) panels for a list of dates.  Labels
and analysis are the next, separate steps.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime
from pathlib import Path
from typing import Any

# Running this file directly puts ``scripts/`` (not the project root) on
# ``sys.path``; add the root so ``python scripts/run_factor_batch.py`` works from
# a checkout, with or without the package installed in the environment.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from data_sys.config import PipelineConfig  # noqa: E402
from data_sys.market_calendar import DEFAULT_CALENDAR_NAME  # noqa: E402
from data_sys.utils import new_run_id, utc_now_iso  # noqa: E402
from factor_sys import (  # noqa: E402
    ANCHOR_WEEKDAYS,
    BIWEEKLY,
    BIWEEKLY_ANCHOR,
    DEFAULT_ANCHOR_WEEKDAY,
    DEFAULT_FREQUENCY,
    FREQUENCIES,
    MONTHLY,
    FactorError,
    decision_schedule,
    default_factor_registry,
)
from scripts.run_factor import FactorRunOutcome, run_factor  # noqa: E402

#: Printed for a field that does not apply (a monthly anchor, a biweekly phase).
MISSING_TEXT = "n/a"

#: Version of the manifest layout this script writes.
MANIFEST_FORMAT_VERSION = 1

#: Directory of batch manifests, relative to the run's data root.
MANIFEST_DIRECTORY: tuple[str, ...] = ("factors", "batch_runs")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class BatchRunError(FactorError):
    """A batch request this entry point refuses, or a manifest it cannot write.

    Only the batch's *own* responsibilities land here -- a blank ``--data-root``,
    a path of the wrong type, a manifest payload that contradicts itself, a
    manifest file that cannot be written.  Everything about one date stays with
    the layer that ran it (a damaged panel is a
    :class:`~factor_sys.errors.FactorStoreError`, an unknown factor a
    :class:`~factor_sys.errors.FactorRegistryError`) and is recorded per date.
    """


# ---------------------------------------------------------------------------
# What one batch produced
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DateFailure:
    """One selected date the batch could not deliver, and why.

    The message is the domain error's own text, verbatim: it is the evidence a
    later reader needs to decide whether the date was skipped, misconfigured or
    damaged, so it is quoted rather than summarized.  ``error_type`` is the
    exception class name, kept separately so a caller can group failures without
    parsing English.
    """

    as_of_date: date
    error_type: str
    message: str

    def __post_init__(self) -> None:
        if isinstance(self.as_of_date, datetime) or not isinstance(self.as_of_date, date):
            raise BatchRunError(
                f"a failure needs a plain date as its as_of_date, got {self.as_of_date!r}"
            )
        if not isinstance(self.error_type, str) or not self.error_type.strip():
            raise BatchRunError(f"a failure needs a non-empty error_type, got {self.error_type!r}")
        if not isinstance(self.message, str):
            raise BatchRunError(f"a failure message must be a string, got {self.message!r}")

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(as_of_date={self.as_of_date.isoformat()!r}, "
            f"error_type={self.error_type!r})"
        )


@dataclass(frozen=True)
class FactorBatchOutcome:
    """What one batch run did: the schedule it visited and what each date became.

    Immutable and self-checking, like every other result object in the project:
    the counts are *derived* from the per-date outcomes (never stored twice, so
    they cannot disagree), the dates are in schedule order, and an object whose
    fields contradict each other -- an outcome for a date that was not selected,
    two records for one date, a selected date with no record at all, mixed
    factor versions -- is refused at construction instead of describing a batch
    that never happened.

    Parameters
    ----------
    factor_name / factor_version:
        The factor, as the registry resolved it.
    start_date / end_date:
        The requested window (inclusive), as normalized dates.
    frequency:
        ``"weekly"``, ``"biweekly"`` or ``"monthly"``.
    anchor_weekday:
        The anchor actually used, ``None`` for ``monthly``.
    calendar_name:
        The trading calendar the decisions were checked against.
    biweekly_anchor:
        :data:`factor_sys.factor_schedule.BIWEEKLY_ANCHOR` for a biweekly
        schedule, ``None`` otherwise.
    selected_dates:
        Every date the schedule selected, in order -- including the ones that
        failed, so the record is complete.  Each of them must appear exactly
        once across ``outcomes`` and ``failures``: no gaps, no double records.
    outcomes:
        The successful dates, each the untouched
        :class:`scripts.run_factor.FactorRunOutcome` of that date (statistics,
        quality report, cache status and stored path included).
    failures:
        The dates that failed, with their error type and message, in schedule
        order.
    manifest_path:
        Where this run's manifest lives (or was meant to live).
    manifest_error:
        ``None`` when the manifest was written; the message when it could not
        be, because a failed record must not hide the work that did succeed.

    Examples
    --------
    >>> outcome = FactorBatchOutcome(
    ...     factor_name="momentum_12_1",
    ...     factor_version="1.0.0",
    ...     start_date=date(2024, 1, 1),
    ...     end_date=date(2024, 1, 12),
    ...     frequency="weekly",
    ...     anchor_weekday="THU",
    ...     calendar_name="XNYS",
    ...     biweekly_anchor=None,
    ...     selected_dates=(),
    ...     outcomes=(),
    ...     failures=(),
    ...     manifest_path=Path("data/factors/batch_runs/x.json"),
    ... )
    >>> (outcome.selection_size, outcome.failed_count, outcome.manifest_written)
    (0, 0, True)
    """

    factor_name: str
    factor_version: str
    start_date: date
    end_date: date
    frequency: str
    anchor_weekday: str | None
    calendar_name: str
    biweekly_anchor: str | None
    selected_dates: tuple[date, ...]
    outcomes: tuple[FactorRunOutcome, ...]
    failures: tuple[DateFailure, ...]
    manifest_path: Path
    manifest_error: str | None = None

    def __post_init__(self) -> None:
        problems: list[str] = []

        for label, value in (("start_date", self.start_date), ("end_date", self.end_date)):
            if isinstance(value, datetime) or not isinstance(value, date):
                problems.append(f"{label} must be a plain date, got {value!r}")
        window_ok = not problems
        if window_ok and self.end_date < self.start_date:
            problems.append(
                f"start_date ({self.start_date.isoformat()}) must not be after end_date "
                f"({self.end_date.isoformat()})"
            )
            window_ok = False

        for label, value in (
            ("factor_name", self.factor_name),
            ("factor_version", self.factor_version),
            ("calendar_name", self.calendar_name),
        ):
            if not isinstance(value, str) or not value.strip():
                problems.append(f"{label} must be a non-empty string, got {value!r}")

        if self.frequency not in FREQUENCIES:
            problems.append(
                f"frequency must be one of {_quoted(FREQUENCIES)}, got {self.frequency!r}"
            )
        elif self.frequency == MONTHLY:
            if self.anchor_weekday is not None:
                problems.append(
                    "a monthly batch takes the last session of each month and carries no "
                    f"anchor, got anchor_weekday={self.anchor_weekday!r}"
                )
            if self.biweekly_anchor is not None:
                problems.append(
                    f"a monthly batch is not phased, got biweekly_anchor={self.biweekly_anchor!r}"
                )
        else:
            if self.anchor_weekday not in ANCHOR_WEEKDAYS:
                problems.append(
                    f"anchor_weekday must be one of {_quoted(ANCHOR_WEEKDAYS)} for a "
                    f"{self.frequency} batch, got {self.anchor_weekday!r}"
                )
            expected = BIWEEKLY_ANCHOR if self.frequency == BIWEEKLY else None
            if self.biweekly_anchor != expected:
                problems.append(
                    f"biweekly_anchor must be {expected!r} for a {self.frequency} batch, "
                    f"got {self.biweekly_anchor!r}"
                )

        if not isinstance(self.manifest_path, Path):
            problems.append(
                f"manifest_path must be a pathlib.Path, got {type(self.manifest_path).__name__}"
            )
        if self.manifest_error is not None and not isinstance(self.manifest_error, str):
            problems.append(f"manifest_error must be a string or None, got {self.manifest_error!r}")

        selection_problems, selected = _selection_problems(
            self.selected_dates, self.start_date, self.end_date
        )
        problems.extend(selection_problems)
        selection_ok = window_ok and not selection_problems

        success_dates: list[date] = []
        if not isinstance(self.outcomes, tuple):
            problems.append(
                "outcomes must be a tuple of FactorRunOutcome, got "
                f"{type(self.outcomes).__name__}: an empty batch is (), never a list or None"
            )
        else:
            for item in self.outcomes:
                if not isinstance(item, FactorRunOutcome):
                    problems.append(
                        f"every outcome must be a FactorRunOutcome, got {type(item).__name__}"
                    )
                    continue
                statistics = item.statistics
                day = statistics.as_of_date
                if (statistics.factor_name, statistics.factor_version) != (
                    self.factor_name,
                    self.factor_version,
                ):
                    problems.append(
                        f"the outcome for {day.isoformat()} describes "
                        f"{statistics.factor_name!r} v{statistics.factor_version}, not "
                        f"{self.factor_name!r} v{self.factor_version}"
                    )
                if selection_ok and day not in selected:
                    problems.append(
                        f"the outcome for {day.isoformat()} is not one of the selected dates"
                    )
                if success_dates and day <= success_dates[-1]:
                    problems.append(
                        "outcomes must be in strictly ascending date order, but "
                        f"{day.isoformat()} follows {success_dates[-1].isoformat()}"
                    )
                success_dates.append(day)

        failure_dates: list[date] = []
        if not isinstance(self.failures, tuple):
            problems.append(
                f"failures must be a tuple of DateFailure, got {type(self.failures).__name__}"
            )
        else:
            for failure in self.failures:
                if not isinstance(failure, DateFailure):
                    problems.append(
                        f"every failure must be a DateFailure, got {type(failure).__name__}"
                    )
                    continue
                day = failure.as_of_date
                if selection_ok and day not in selected:
                    problems.append(
                        f"the failure for {day.isoformat()} is not one of the selected dates"
                    )
                if day in success_dates:
                    problems.append(
                        f"{day.isoformat()} is recorded both as an outcome and as a failure"
                    )
                if failure_dates and day <= failure_dates[-1]:
                    problems.append(
                        "failures must be in strictly ascending date order, but "
                        f"{day.isoformat()} follows {failure_dates[-1].isoformat()}"
                    )
                failure_dates.append(day)

        # The two halves must tile the selection exactly: a date that is neither
        # a success nor a failure is a gap in the record -- and a summary that
        # walks the selected dates would have nothing to print for it.
        if selection_ok and isinstance(self.outcomes, tuple) and isinstance(self.failures, tuple):
            covered = set(success_dates) | set(failure_dates)
            uncovered = [day for day in selected if day not in covered]
            if uncovered:
                shown = ", ".join(day.isoformat() for day in uncovered[:5])
                more = f" (and {len(uncovered) - 5} more)" if len(uncovered) > 5 else ""
                problems.append(
                    "every selected date needs exactly one outcome or one failure, but "
                    f"nothing was recorded for {shown}{more}"
                )

        if problems:
            raise BatchRunError(
                "the batch outcome does not describe one run: " + "; ".join(problems)
            )

    # -- the counts, derived from the statuses the single-day runs answered ------
    @property
    def selection_size(self) -> int:
        """How many dates the schedule selected (successes plus failures)."""
        return len(self.selected_dates)

    @property
    def succeeded_count(self) -> int:
        """How many selected dates produced (or read back) a panel."""
        return len(self.outcomes)

    @property
    def failed_count(self) -> int:
        """How many selected dates failed."""
        return len(self.failures)

    @property
    def computed_count(self) -> int:
        """How many dates were computed for the first time (``cache_status == "miss"``)."""
        return sum(1 for item in self.outcomes if item.cache_status == "miss")

    @property
    def cached_count(self) -> int:
        """How many dates were read back from the store (``cache_status == "hit"``)."""
        return sum(1 for item in self.outcomes if item.cache_status == "hit")

    @property
    def forced_count(self) -> int:
        """How many dates were recomputed on request (``cache_status == "forced"``)."""
        return sum(1 for item in self.outcomes if item.cache_status == "forced")

    @property
    def manifest_written(self) -> bool:
        """``True`` when this run's manifest reached the disk."""
        return self.manifest_error is None

    def __len__(self) -> int:
        """How many dates the schedule selected."""
        return len(self.selected_dates)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(factor_name={self.factor_name!r}, "
            f"factor_version={self.factor_version!r}, frequency={self.frequency!r}, "
            f"anchor_weekday={self.anchor_weekday!r}, dates={self.selection_size}, "
            f"computed={self.computed_count}, cached={self.cached_count}, "
            f"forced={self.forced_count}, failed={self.failed_count})"
        )


# ---------------------------------------------------------------------------
# The composition: one schedule, one run per date, one manifest
# ---------------------------------------------------------------------------
def run_factor_batch(
    factor_name: str,
    start_date: date | str,
    end_date: date | str,
    *,
    frequency: str = DEFAULT_FREQUENCY,
    anchor_weekday: str | None = None,
    data_root: Path | str | None = None,
    universe_path: Path | str | None = None,
    calendar_name: str | None = None,
    force: bool = False,
) -> FactorBatchOutcome:
    """Run one factor for every session a cadence selects, and record the run.

    The steps of the module docstring run in order, each through the public
    interface of the module that owns it: the factor is created by name, the
    dates come from :func:`factor_sys.factor_schedule.decision_schedule`, and
    each date is handed to :func:`scripts.run_factor.run_factor` -- the same
    function a single run uses, so a panel, a quality report and a cache status
    are always the ones that entry point really produced.

    A failing date is recorded and the loop continues: one damaged panel or one
    missing window never stops the dates after it, and nothing already written
    is rolled back.  An unknown factor or an unknown calendar is refused before
    the first date, because it would fail identically N times.

    Parameters
    ----------
    factor_name:
        Registered factor name, resolved through
        :func:`factor_sys.default_factor_registry` before any date is visited.
    start_date / end_date:
        The closed window, as dates or strict ISO ``"YYYY-MM-DD"`` strings.
    frequency:
        ``"weekly"`` (the default), ``"biweekly"`` or ``"monthly"``.
    anchor_weekday:
        ``"MON"``..``"FRI"`` for weekly/biweekly (default ``"THU"``); ``None``
        for monthly, which has no anchor and refuses one.
    data_root:
        Root the whole batch is derived from: bars are read from
        ``<data_root>/standardized/market_bars``, panels are stored in
        ``<data_root>/factors/panels`` and the manifest is written to
        ``<data_root>/factors/batch_runs``.  ``None`` means
        :attr:`data_sys.config.PipelineConfig.data_root`.
    universe_path:
        The static stock-pool CSV, passed to every date unchanged.
    calendar_name:
        Trading calendar that defines a session, passed to every date (the
        resolved name, so the schedule and the runs cannot disagree).
    force:
        Recompute every selected date, replacing each stored panel atomically.

    Returns
    -------
    FactorBatchOutcome
        The schedule that was used, one untouched
        :class:`scripts.run_factor.FactorRunOutcome` per successful date, the
        failures with their error type and message, the derived counts and the
        manifest's path (plus ``manifest_error`` when the manifest could not be
        written).

    Raises
    ------
    BatchRunError
        A blank or unusable ``data_root``, or a batch request whose parts
        contradict each other.
    factor_sys.FactorError
        An unknown factor (:class:`~factor_sys.errors.FactorRegistryError`), an
        unusable window, cadence, anchor or calendar
        (:class:`~factor_sys.errors.FactorScheduleError`).  Failures *of a
        date* are never raised; they come back in ``failures``.
    """
    root = _data_root(data_root)

    # Early, before any date: a typo in the factor name is one message, not N.
    spec = default_factor_registry().create(factor_name).spec

    schedule = decision_schedule(
        start_date,
        end_date,
        frequency=frequency,
        anchor_weekday=anchor_weekday,
        calendar_name=calendar_name,
    )

    outcomes: list[FactorRunOutcome] = []
    failures: list[DateFailure] = []
    for day in schedule.dates:
        try:
            outcomes.append(
                run_factor(
                    spec.name,
                    day,
                    data_root=root,
                    universe_path=universe_path,
                    calendar_name=schedule.calendar_name,
                    force=force,
                )
            )
        except FactorError as exc:  # one date, not the batch
            failures.append(DateFailure(day, type(exc).__name__, str(exc)))

    batch_run_id = new_run_id()
    manifest_path = _manifest_path(root, batch_run_id)
    outcome = FactorBatchOutcome(
        factor_name=spec.name,
        factor_version=spec.version,
        start_date=schedule.start_date,
        end_date=schedule.end_date,
        frequency=schedule.frequency,
        anchor_weekday=schedule.anchor_weekday,
        calendar_name=schedule.calendar_name,
        biweekly_anchor=schedule.biweekly_anchor,
        selected_dates=schedule.dates,
        outcomes=tuple(outcomes),
        failures=tuple(failures),
        manifest_path=manifest_path,
    )

    payload = manifest_payload(
        outcome,
        batch_run_id=batch_run_id,
        created_at_utc=utc_now_iso(),
        data_root=root,
    )
    try:
        write_manifest(payload, manifest_path)
    except BatchRunError as exc:
        # The work still happened: report it, and let the CLI fail the command.
        return replace(outcome, manifest_error=str(exc))
    return outcome


# ---------------------------------------------------------------------------
# The manifest: one JSON record per run, written atomically
# ---------------------------------------------------------------------------
def write_manifest(manifest: Mapping[str, Any], path: Path) -> Path:
    """Write ``manifest`` to ``path`` through a temporary file and one replace.

    The temporary file lives in ``path``'s own directory, so :func:`os.replace`
    is a rename within one filesystem and therefore atomic: a reader sees the
    previous file or the complete new one, never half of either.  Any failure
    removes the temporary file, leaves the previous manifest exactly as it was
    and raises :class:`BatchRunError`.

    Parameters
    ----------
    manifest:
        The JSON-safe payload (dicts, lists, strings, numbers, ``None``).
    path:
        Where the manifest belongs.  Its directory is created if needed.

    Returns
    -------
    Path
        ``path`` itself, so a caller can keep using the return value.
    """
    parent = path.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise BatchRunError(
            f"cannot create the batch-run directory {str(parent)!r}: {exc}"
        ) from exc

    try:
        handle, temp_name = tempfile.mkstemp(
            dir=str(parent), prefix=f".{path.name}.", suffix=".tmp"
        )
        os.close(handle)
    except OSError as exc:
        raise BatchRunError(f"cannot create a temporary file next to {str(path)!r}: {exc}") from exc

    temp = Path(temp_name)
    try:
        with temp.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(manifest, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # Confirm the temporary file is there and complete before it may replace
        # anything: an empty manifest is worse than an absent one.
        if not temp.is_file():
            raise BatchRunError(f"the temporary manifest {str(temp)!r} was not created")
        if temp.stat().st_size == 0:
            raise BatchRunError(f"the temporary manifest {str(temp)!r} is empty")
        os.replace(str(temp), str(path))
    except Exception as exc:  # noqa: BLE001 - keep the old file, then normalize the error
        _remove_quietly(temp)
        if isinstance(exc, BatchRunError):
            raise
        raise BatchRunError(f"could not write the batch manifest to {str(path)!r}: {exc}") from exc
    return path


def manifest_payload(
    outcome: FactorBatchOutcome,
    *,
    batch_run_id: str,
    created_at_utc: str,
    data_root: Path,
) -> dict[str, Any]:
    """The JSON-safe record of one batch run.

    Everything a later reader needs is here in the order it happened: the window
    that was requested, the cadence and anchor that turned it into dates, the
    complete ``selected_dates`` list, one result per successful date and one
    failure per failed date.  ``panel_path`` is stored relative to
    ``data_root`` when the panel lives there, so a manifest stays meaningful
    after the tree is moved to another machine.
    """
    return {
        "format_version": MANIFEST_FORMAT_VERSION,
        "batch_run_id": batch_run_id,
        "created_at_utc": created_at_utc,
        "factor_name": outcome.factor_name,
        "factor_version": outcome.factor_version,
        "requested_start_date": outcome.start_date.isoformat(),
        "requested_end_date": outcome.end_date.isoformat(),
        "frequency": outcome.frequency,
        "anchor_weekday": outcome.anchor_weekday,
        "biweekly_anchor": outcome.biweekly_anchor,
        "calendar_name": outcome.calendar_name,
        "selected_dates": [day.isoformat() for day in outcome.selected_dates],
        "results": [
            {
                "as_of_date": item.statistics.as_of_date.isoformat(),
                "status": item.cache_status,
                "quality_status": item.quality_status,
                "panel_path": _portable(item.stored_panel_path, data_root),
            }
            for item in outcome.outcomes
        ],
        "failures": [
            {
                "as_of_date": failure.as_of_date.isoformat(),
                "error_type": failure.error_type,
                "message": failure.message,
            }
            for failure in outcome.failures
        ],
    }


def _portable(path: Path, data_root: Path) -> str:
    """``path`` relative to ``data_root`` when it lives there, otherwise verbatim.

    A portable manifest must not embed ``C:\\Users\\...``: the same run on another
    checkout would look like it had written the panel somewhere else.
    """
    try:
        return path.relative_to(data_root).as_posix()
    except ValueError:
        return path.as_posix()


def _manifest_path(data_root: Path, batch_run_id: str) -> Path:
    """Where the manifest of ``batch_run_id`` belongs under ``data_root``."""
    return data_root.joinpath(*MANIFEST_DIRECTORY) / f"{batch_run_id}.json"


def _data_root(value: Path | str | None) -> Path:
    """The root every path of this batch is derived from, or a clear refusal.

    The batch owns exactly one path of its own -- the manifest directory -- so it
    validates that root once, before the first date, instead of letting a blank
    value become the current directory or fail once per date.
    """
    if value is None:
        return PipelineConfig().data_root
    if isinstance(value, str) and not value.strip():
        raise BatchRunError(
            "--data-root must not be blank: pass a directory, or omit it to read and "
            f"write the project's default {PipelineConfig().data_root}"
        )
    if not isinstance(value, (str, Path)):
        raise BatchRunError(f"data_root must be a path or a string, got {type(value).__name__}")
    return PipelineConfig(data_root=Path(value)).data_root


def _remove_quietly(path: Path) -> None:
    """Delete ``path`` if it exists; a leftover temporary file is never the story."""
    try:
        path.unlink()
    except OSError:
        pass  # noqa: BLE001 - a failed cleanup must not mask the real error


def _selection_problems(
    selected: Any, start: date, end: date
) -> tuple[list[str], tuple[date, ...]]:
    """Check one selection: ``(problems, dates)``, the dates being usable as-is."""
    if not isinstance(selected, tuple):
        return (
            [f"selected_dates must be a tuple of dates, got {type(selected).__name__}"],
            (),
        )

    problems: list[str] = []
    previous: date | None = None
    for day in selected:
        if isinstance(day, datetime) or not isinstance(day, date):
            return [f"every selected date must be a plain date, got {day!r}"], ()
        if previous is not None and day <= previous:
            return (
                [
                    "selected dates must be strictly ascending and free of duplicates, but "
                    f"{day.isoformat()} follows {previous.isoformat()}"
                ],
                (),
            )
        previous = day

    dated = (
        isinstance(start, date)
        and not isinstance(start, datetime)
        and isinstance(end, date)
        and not isinstance(end, datetime)
    )
    if selected and dated and (selected[0] < start or selected[-1] > end):
        problems.append(
            f"selected dates must lie inside {start.isoformat()}..{end.isoformat()}, got "
            f"{selected[0].isoformat()}..{selected[-1].isoformat()}"
        )
    return problems, selected


def _quoted(values: tuple[str, ...]) -> str:
    """``values`` as a comma-separated list of quoted names, for one message."""
    return ", ".join(repr(value) for value in values)


# ---------------------------------------------------------------------------
# The summary -- the only rendering of a batch
# ---------------------------------------------------------------------------
def format_summary(outcome: FactorBatchOutcome) -> str:
    """Render ``outcome`` as status lines, failure lines and a summary block.

    Three kinds of line, in this order and nowhere else:

    * one status line per selected date, in schedule order --
      ``2024-01-04  miss    clean``: the date, the ``cache_status`` the store
      answered (or ``failed``) and, after it, the quality status for a success
      or the error type for a failure;
    * one ``failure:`` line per failed date, carrying the domain error's own
      message collapsed onto a single line (the multi-line original stays in the
      manifest);
    * the summary block, one ``key: value`` per line, always in the same order
      and always ending with ``manifest_path``.

    The text is a reporting surface, never a storage format: every count is
    derived from the outcomes themselves, and a field that does not apply (the
    anchor of a monthly schedule, the phase of a non-biweekly one) is printed as
    :data:`MISSING_TEXT` rather than invented.  Every selected date has exactly
    one outcome or one failure -- :class:`FactorBatchOutcome` guarantees it -- so
    every date gets a status line and no line is ever missing.
    """
    lines: list[str] = []

    successes = {item.statistics.as_of_date: item for item in outcome.outcomes}
    failures = {failure.as_of_date: failure for failure in outcome.failures}
    for day in outcome.selected_dates:
        item = successes.get(day)
        if item is not None:
            lines.append(f"{day.isoformat()}  {item.cache_status:<7} {item.quality_status}")
            continue
        failure = failures[day]
        lines.append(f"{day.isoformat()}  {'failed':<7} {failure.error_type}")

    for failure in outcome.failures:
        lines.append(
            f"failure: {failure.as_of_date.isoformat()} {failure.error_type}: "
            f"{_one_line(failure.message)}"
        )

    lines.extend(
        [
            f"factor_name: {outcome.factor_name}",
            f"factor_version: {outcome.factor_version}",
            f"start_date: {outcome.start_date.isoformat()}",
            f"end_date: {outcome.end_date.isoformat()}",
            f"frequency: {outcome.frequency}",
            f"anchor_weekday: {outcome.anchor_weekday or MISSING_TEXT}",
            f"biweekly_anchor: {outcome.biweekly_anchor or MISSING_TEXT}",
            f"calendar_name: {outcome.calendar_name}",
            f"selected_dates: {outcome.selection_size}",
            f"computed_count: {outcome.computed_count}",
            f"cached_count: {outcome.cached_count}",
            f"forced_count: {outcome.forced_count}",
            f"failed_count: {outcome.failed_count}",
            f"manifest_path: {outcome.manifest_path}",
        ]
    )
    return "\n".join(lines)


def _one_line(text: str) -> str:
    """``text`` with every run of whitespace collapsed to one space."""
    return " ".join(text.split())


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """Build the parser, naming the defaults the run will actually use."""
    default_data_root = PipelineConfig().data_root
    parser = argparse.ArgumentParser(
        prog="run_factor_batch.py",
        description=(
            "Compute one factor for every session a weekly/biweekly/monthly schedule "
            "selects, reusing stored panels, and record the exact dates in a manifest."
        ),
    )
    parser.add_argument(
        "--factor",
        required=True,
        metavar="FACTOR_NAME",
        help="registered factor name, e.g. momentum_12_1 (resolved through the registry)",
    )
    parser.add_argument(
        "--start",
        dest="start_date",
        required=True,
        metavar="YYYY-MM-DD",
        help="first day of the closed window (inclusive)",
    )
    parser.add_argument(
        "--end",
        dest="end_date",
        required=True,
        metavar="YYYY-MM-DD",
        help="last day of the closed window (inclusive)",
    )
    parser.add_argument(
        "--frequency",
        choices=FREQUENCIES,
        default=DEFAULT_FREQUENCY,
        help=(
            "date cadence: weekly (one session per week: the anchor, or the session "
            "before it when the exchange is closed), biweekly (every second such "
            "week) or monthly (the last session of each month) "
            f"(default: {DEFAULT_FREQUENCY})"
        ),
    )
    parser.add_argument(
        "--anchor-weekday",
        dest="anchor_weekday",
        choices=ANCHOR_WEEKDAYS,
        default=None,
        help=(
            "weekday a weekly/biweekly schedule probes "
            f"(default: {DEFAULT_ANCHOR_WEEKDAY}); it does not apply to --frequency monthly"
        ),
    )
    parser.add_argument(
        "--data-root",
        dest="data_root",
        metavar="PATH",
        default=None,
        help=(
            "root holding standardized/market_bars, receiving factors/panels and "
            f"factors/batch_runs (default: {default_data_root})"
        ),
    )
    parser.add_argument(
        "--universe-path",
        dest="universe_path",
        metavar="PATH",
        default=None,
        help="static stock-pool CSV (default: the committed config/universe_seed.csv)",
    )
    parser.add_argument(
        "--calendar",
        metavar="NAME",
        default=None,
        help=f"trading calendar that defines a session (default: {DEFAULT_CALENDAR_NAME})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "recompute every selected date even when its panel is already stored, "
            "replacing each file atomically (default: read stored panels back)"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse ``argv``, run the batch, print its report and return the exit code.

    Parameters
    ----------
    argv:
        Command-line arguments without the program name (``None`` means
        ``sys.argv[1:]``).  Passing a list lets a test drive the command exactly
        as a shell would, without starting a second process.

    Returns
    -------
    int
        ``0`` when every selected date succeeded and the manifest was written
        (an empty selection included); ``1`` when at least one date failed or the
        manifest could not be written.  Domain errors are printed to stderr as
        one ``error: ...`` line, never as a traceback, and no date is rolled back
        because another one failed.  ``argparse`` keeps its own exit code ``2``.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.frequency == MONTHLY and args.anchor_weekday is not None:
        parser.error(
            "--anchor-weekday does not apply to --frequency monthly: a monthly schedule "
            "already selects the last session of each month, so omit the flag"
        )

    try:
        outcome = run_factor_batch(
            args.factor,
            args.start_date,
            args.end_date,
            frequency=args.frequency,
            anchor_weekday=args.anchor_weekday,
            data_root=args.data_root,
            universe_path=args.universe_path,
            calendar_name=args.calendar,
            force=args.force,
        )
    except FactorError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(format_summary(outcome))

    if outcome.manifest_error is not None:
        print(f"error: {outcome.manifest_error}", file=sys.stderr)
        return 1
    return 1 if outcome.failed_count else 0


__all__ = [
    "MANIFEST_DIRECTORY",
    "MANIFEST_FORMAT_VERSION",
    "MISSING_TEXT",
    "BatchRunError",
    "DateFailure",
    "FactorBatchOutcome",
    "build_parser",
    "format_summary",
    "main",
    "manifest_payload",
    "run_factor_batch",
    "write_manifest",
]


if __name__ == "__main__":
    raise SystemExit(main())
