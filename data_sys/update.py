"""The one place a symbol's published history is updated.

Both CLIs are thin wrappers around :func:`update_symbol`:

* ``python -m data_sys.data_query`` updates a single ticker,
* ``python -m data_sys.batch_query`` walks a validated universe and calls the
  same function once per symbol.

Sharing this function is what keeps the two entry points from drifting apart --
there is exactly one implementation of "plan, download, merge, recompute,
validate, publish", and no CLI owns any of it.

Flow for one symbol
-------------------
1. read the currently published partition (empty for a brand-new symbol),
2. :func:`data_sys.planner.plan_download` decides which windows to fetch,
3. each window is downloaded and archived untouched under ``data/raw/``,
4. every downloaded window is standardized and merged into the published frame,
   with this run's rows winning on overlapping dates,
5. ``daily_return`` / ``dollar_volume`` are recomputed over the merged history,
6. the merged history is validated with pandera, staged, DuckDB-verified and
   only then atomically published,
7. a per-symbol ``run_<run_id>.json`` records everything that happened.

Any failure leaves the published partition exactly as it was: the swap in step 6
is the first and only mutation of the live dataset.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from data_sys.config import (
    DEFAULT_REFRESH_OVERLAP_SESSIONS,
    DEFAULT_START,
    DEFAULT_SYMBOL,
    PipelineConfig,
)
from data_sys.duckdb_verify import verify_dataset
from data_sys.errors import (
    DataPipelineError,
    DuckDBVerificationError,
    QualityCheckError,
)
from data_sys.market_calendar import (
    DEFAULT_CALENDAR_NAME,
    get_calendar,
    sessions_in_range,
)
from data_sys.merge import date_set, merge_bars
from data_sys.metadata import (
    RunMetadata,
    collect_package_versions,
    configure_logging,
    write_metadata,
)
from data_sys.planner import (
    STATUS_NO_SESSIONS,
    UpdatePlan,
    plan_download,
)
from data_sys.providers.base import DataProvider
from data_sys.providers.yahoo import YahooFinanceProvider
from data_sys.quality import validate_bars
from data_sys.standardize import standardize_bars
from data_sys.storage import (
    atomic_publish,
    read_symbol_partition,
    write_partitioned,
    write_raw,
)
from data_sys.utils import new_run_id, parse_date, utc_now_iso

logger = logging.getLogger("data_sys.update")

#: Per-symbol outcome values.  ``skipped`` is a *safe* outcome: nothing was
#: downloaded and nothing was overwritten.
STATUS_SUCCESS = "success"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"


@dataclass
class SymbolUpdateResult:
    """Outcome of one incremental update.

    Used twice: as the return value of :func:`update_symbol` and as the
    ``per_symbol_status`` entry inside a batch metadata file.
    """

    symbol: str
    status: str = STATUS_SUCCESS
    run_id: str = ""
    batch_run_id: str | None = None

    plan_status: str = ""
    existing_row_count: int = 0
    downloaded_row_count: int = 0
    #: Rows in the published dataset *after* the run.  Equals
    #: ``existing_row_count`` when nothing was published, so pair it with
    #: ``published`` rather than reading it as "rows written".
    published_row_count: int = 0
    new_dates_count: int = 0
    replaced_dates_count: int = 0
    missing_ranges: list[list[str]] = field(default_factory=list)
    refresh_range: list[str] | None = None
    raw_paths: list[str] = field(default_factory=list)
    windows: list[dict[str, Any]] = field(default_factory=list)
    failure_reason: str | None = None
    #: ``True`` only when the staged dataset replaced the published partition.
    published: bool = False
    min_date: str | None = None
    max_date: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """``True`` when the symbol neither failed nor needs attention."""
        return self.status in (STATUS_SUCCESS, STATUS_SKIPPED)

    def to_status_dict(self) -> dict[str, Any]:
        """JSON-friendly status record (a key of ``per_symbol_status``)."""
        return {
            "symbol": self.symbol,
            "status": self.status,
            "run_id": self.run_id,
            "batch_run_id": self.batch_run_id,
            "plan_status": self.plan_status,
            "existing_row_count": int(self.existing_row_count),
            "downloaded_row_count": int(self.downloaded_row_count),
            "published_row_count": int(self.published_row_count),
            "new_dates_count": int(self.new_dates_count),
            "replaced_dates_count": int(self.replaced_dates_count),
            "missing_ranges": [list(pair) for pair in self.missing_ranges],
            "refresh_range": None if self.refresh_range is None else list(self.refresh_range),
            "raw_paths": list(self.raw_paths),
            "failure_reason": self.failure_reason,
            "published": self.published,
            "min_date": self.min_date,
            "max_date": self.max_date,
            "windows": list(self.windows),
            "notes": list(self.notes),
        }


def _iso(value: Any) -> str | None:
    return None if value is None else str(value)


def build_config(
    symbol: str,
    start: date | str | None,
    end: date | str | None,
    data_root: Path | str | None = None,
) -> PipelineConfig:
    """Build the frozen per-symbol config, normalizing CLI input.

    Raises
    ------
    data_sys.errors.DataPipelineError
        If the dates cannot be parsed or ``end`` precedes ``start``.
    """
    kwargs: dict[str, Any] = {
        "symbol": str(symbol).strip().upper(),
        "start": parse_date(start) or DEFAULT_START,
        "end": parse_date(end) or date.today(),
    }
    if data_root is not None:
        kwargs["data_root"] = Path(data_root)
    try:
        return PipelineConfig(**kwargs)
    except ValueError as exc:
        raise DataPipelineError(str(exc)) from exc


def _dataset_stats(frame: pd.DataFrame | None) -> tuple[str | None, str | None, list[list[int]]]:
    """``(min_date, max_date, [[year, rows], ...])`` of a published frame."""
    if frame is None or len(frame) == 0 or "date" not in frame.columns:
        return None, None, []
    dates = pd.to_datetime(frame["date"])
    counts = dates.dt.year.value_counts().sort_index()
    years = [[int(year), int(count)] for year, count in counts.items()]
    return dates.min().date().isoformat(), dates.max().date().isoformat(), years


def _record_plan(meta: RunMetadata, result: SymbolUpdateResult, plan: UpdatePlan) -> None:
    """Copy a plan into both the run metadata and the caller-facing result."""
    meta.plan = plan.as_metadata()
    meta.missing_ranges = plan.missing_ranges_iso()
    meta.refresh_range = plan.refresh_range_iso()
    meta.notes = list(plan.notes)

    result.plan_status = plan.status
    result.missing_ranges = plan.missing_ranges_iso()
    result.refresh_range = plan.refresh_range_iso()
    result.windows = [window.to_dict() for window in plan.windows]
    result.notes = list(plan.notes)


def _confirm_publish(config: PipelineConfig, expected_rows: int) -> list[str]:
    """Cross-check the live partition against what was just staged.

    This is the *confirmation* step of the publish transaction: it is handed to
    :func:`data_sys.storage.atomic_publish` as its ``confirm`` callback, so it runs
    after the swap and before the backup is dropped, and any issue it reports
    makes the transaction restore the previous version instead of committing.

    The DuckDB verification of the *staged* dataset already proved the merged
    history is complete and consistent, and it ran before anything was swapped,
    which is exactly what the architecture document demands.  This second check
    is a *reader-side* confirmation that the swap landed: it reads the live
    partition back through the same reader the next update will use.

    It is deliberately not a DuckDB check.  DuckDB keeps file handles keyed by
    path, and pointing it at a partition whose parquet files have just been
    replaced was observed to wedge the interpreter on Windows; a path is only
    ever handed to DuckDB here while it is freshly written (the staging area).
    """
    issues: list[str] = []
    frame = read_symbol_partition(config.standardized_dir, config.symbol)
    if len(frame) != expected_rows:
        issues.append(
            f"expected {expected_rows} row(s) in {config.live_symbol_dir.name}, "
            f"found {len(frame)}"
        )
    if frame.empty:
        return issues

    duplicates = int(frame.duplicated(subset=["symbol", "date"]).sum())
    if duplicates:
        issues.append(f"{duplicates} duplicate (symbol, date) row(s)")
    if not pd.to_datetime(frame["date"]).is_monotonic_increasing:
        issues.append("published dates are not sorted ascending")
    symbols = sorted({str(value) for value in frame["symbol"].unique()})
    if symbols != [config.symbol]:
        issues.append(f"unexpected symbol value(s) {symbols}, expected ['{config.symbol}']")
    return issues


def update_symbol(
    symbol: str = DEFAULT_SYMBOL,
    start: date | str | None = DEFAULT_START,
    end: date | str | None = None,
    *,
    data_root: Path | str | None = None,
    provider: DataProvider | None = None,
    refresh_overlap_sessions: int = DEFAULT_REFRESH_OVERLAP_SESSIONS,
    calendar_name: str = DEFAULT_CALENDAR_NAME,
    calendar: Any | None = None,
    run_id: str | None = None,
    batch_run_id: str | None = None,
) -> SymbolUpdateResult:
    """Bring one symbol's published history up to date; never raise.

    Parameters
    ----------
    symbol, start, end:
        Ticker and inclusive request window (defaults to ``AAPL`` and
        ``2020-01-01 .. today``).
    data_root:
        Override the ``data/`` root (tests and multi-environment runs).
    provider:
        ``DataProvider`` to use; defaults to :class:`YahooFinanceProvider`.
    refresh_overlap_sessions:
        Size of the recent-session overlap window (``0`` disables it).
    calendar_name, calendar:
        Trading calendar to plan against.
    run_id:
        Reuse an existing run id (batch members derive theirs from the batch).
    batch_run_id:
        Recorded in the run metadata when this run belongs to a batch.

    Returns
    -------
    SymbolUpdateResult
        ``status`` is ``success``, ``skipped`` (no session in the requested
        range) or ``failed``.  A failure never modifies the published dataset.
    """
    configure_logging()
    run_id = run_id or new_run_id()
    cfg = build_config(symbol, start, end, data_root)
    provider = provider or YahooFinanceProvider()
    source = getattr(provider, "name", type(provider).__name__)
    active_calendar = calendar if calendar is not None else get_calendar(calendar_name)

    result = SymbolUpdateResult(symbol=cfg.symbol, run_id=run_id, batch_run_id=batch_run_id)
    meta = RunMetadata(
        run_id=run_id,
        data_source=str(source),
        symbol=cfg.symbol,
        requested_start=cfg.start.isoformat(),
        requested_end=cfg.end.isoformat(),
        output_path=str(cfg.standardized_dir),
        run_at_utc=utc_now_iso(),
        package_versions=collect_package_versions(),
        batch_run_id=batch_run_id,
    )

    staging_root = cfg.staging_dir / run_id
    logger.info(
        "run %s starting: symbol=%s range=%s..%s refresh=%d source=%s",
        run_id,
        cfg.symbol,
        cfg.start,
        cfg.end,
        refresh_overlap_sessions,
        source,
    )

    try:
        published = read_symbol_partition(cfg.standardized_dir, cfg.symbol)
        existing_count = 0 if published is None else int(len(published))
        result.existing_row_count = existing_count
        meta.existing_row_count = existing_count

        plan = plan_download(
            cfg.symbol,
            cfg.start,
            cfg.end,
            date_set(published),
            calendar=active_calendar,
            calendar_name=calendar_name,
            refresh_overlap_sessions=refresh_overlap_sessions,
        )
        _record_plan(meta, result, plan)
        logger.info(
            "run %s: plan status=%s windows=%s missing=%d refresh=%s",
            run_id,
            plan.status,
            [window.reason for window in plan.windows],
            len(plan.missing_sessions),
            result.refresh_range,
        )

        # -- nothing to do: keep the published partition untouched -----------
        if plan.status == STATUS_NO_SESSIONS:
            result.status = STATUS_SKIPPED
            result.published_row_count = existing_count
            meta.status = STATUS_SKIPPED
            meta.row_count = existing_count
            meta.published_row_count = existing_count
            meta.min_date, meta.max_date, meta.years = _dataset_stats(published)
            for note in plan.notes:
                logger.warning("run %s: skip reason: %s", run_id, note)
            logger.info("run %s: SKIPPED (no trading session in the requested range)", run_id)
            return result

        if not plan.has_work:
            result.published_row_count = existing_count
            meta.status = STATUS_SUCCESS
            meta.row_count = existing_count
            meta.published_row_count = existing_count
            meta.min_date, meta.max_date, meta.years = _dataset_stats(published)
            logger.info(
                "run %s: SUCCESS (already up to date, %d row(s) kept)",
                run_id,
                existing_count,
            )
            return result

        # -- download every planned window, archiving each raw response ------
        slices: list[pd.DataFrame] = []
        for window in plan.windows:
            raw = provider.fetch_bars(cfg.symbol, window.start, window.end)
            stem = (
                f"{cfg.symbol}_{window.start.isoformat()}_"
                f"{window.end.isoformat()}_{run_id}"
            )
            sidecar = dict(raw.meta)
            sidecar.update(
                {
                    "run_id": run_id,
                    "batch_run_id": batch_run_id,
                    "plan_reason": window.reason,
                    "window_start": window.start.isoformat(),
                    "window_end": window.end.isoformat(),
                }
            )
            raw_path = write_raw(raw.frame, cfg.raw_symbol_dir, stem, sidecar)
            meta.raw_paths.append(str(raw_path))
            result.raw_paths.append(str(raw_path))

            standardized = standardize_bars(raw.frame, cfg.symbol)
            slices.append(standardized)
            logger.info(
                "run %s: window %s (%s) -> %d rows, raw %s",
                run_id,
                window.as_iso_range(),
                window.reason,
                len(standardized),
                raw_path.name,
            )

            expected = len(sessions_in_range(active_calendar, window.start, window.end))
            if len(standardized) < expected:
                note = (
                    f"window {window.start.isoformat()}..{window.end.isoformat()}: "
                    f"provider returned {len(standardized)} of {expected} expected "
                    "session(s); no data is invented for the rest"
                )
                logger.warning("run %s: %s", run_id, note)
                result.notes.append(note)
                meta.notes.append(note)

        downloaded = pd.concat(slices, ignore_index=True)
        # Merged windows never overlap, but a defensive de-duplication keeps the
        # "one row per (symbol, date)" invariant true even if a provider returns
        # a row outside the range it was asked for.
        downloaded = (
            downloaded.drop_duplicates(subset=["symbol", "date"], keep="last")
            .sort_values(["symbol", "date"], kind="stable")
            .reset_index(drop=True)
        )
        meta.raw_path = meta.raw_paths[0] if meta.raw_paths else None

        merged, report = merge_bars(published, downloaded)
        result.downloaded_row_count = report.downloaded_row_count
        result.new_dates_count = report.new_dates_count
        result.replaced_dates_count = report.replaced_dates_count
        result.min_date = None if report.min_date is None else report.min_date.isoformat()
        result.max_date = None if report.max_date is None else report.max_date.isoformat()
        meta.downloaded_row_count = report.downloaded_row_count
        meta.new_dates_count = report.new_dates_count
        meta.replaced_dates_count = report.replaced_dates_count

        quality = validate_bars(merged)
        if not quality.ok:
            raise QualityCheckError(
                "merged dataset failed quality checks", quality.reasons()
            )
        logger.info("run %s: merged history passed quality checks", run_id)

        # -- stage, verify, publish ------------------------------------------
        write_partitioned(merged, staging_root)
        staged = verify_dataset(staging_root)
        if not staged.ok:
            raise DuckDBVerificationError(
                "staged dataset failed DuckDB verification: " + "; ".join(staged.issues)
            )
        logger.info(
            "run %s: staged dataset verified by DuckDB (%d rows, %s..%s)",
            run_id,
            staged.n_rows,
            staged.min_date,
            staged.max_date,
        )

        # The swap and its confirmation are one transaction: the backup only
        # disappears once _confirm_publish has read the *new* live partition back
        # (PyArrow, never DuckDB) and found no issue.  A failed confirmation rolls
        # the previous version back in, and the run is reported as published=False.
        atomic_publish(
            staging_root,
            cfg.standardized_dir,
            cfg.trash_dir,
            run_id,
            confirm=lambda: _confirm_publish(cfg, staged.n_rows),
        )
        result.published = True
        meta.published = True

        result.status = STATUS_SUCCESS
        result.published_row_count = staged.n_rows
        result.min_date = _iso(staged.min_date)
        result.max_date = _iso(staged.max_date)
        meta.status = STATUS_SUCCESS
        meta.row_count = staged.n_rows
        meta.published_row_count = staged.n_rows
        meta.min_date = result.min_date
        meta.max_date = result.max_date
        meta.years = staged.years
        logger.info(
            "run %s: SUCCESS rows=%d range=%s..%s (new dates=%d, replaced=%d)",
            run_id,
            staged.n_rows,
            meta.min_date,
            meta.max_date,
            report.new_dates_count,
            report.replaced_dates_count,
        )
    except DataPipelineError as exc:
        result.status = STATUS_FAILED
        result.failure_reason = str(exc)
        result.published_row_count = result.existing_row_count
        meta.status = STATUS_FAILED
        meta.failure_reasons = [str(exc)]
        meta.published_row_count = result.existing_row_count
        logger.error("run %s: FAILED (%s): %s", run_id, type(exc).__name__, exc)
    except Exception as exc:  # noqa: BLE001 - never leak an uncaught exception
        result.status = STATUS_FAILED
        result.failure_reason = f"{type(exc).__name__}: {exc}"
        result.published_row_count = result.existing_row_count
        meta.status = STATUS_FAILED
        meta.failure_reasons = [result.failure_reason]
        meta.published_row_count = result.existing_row_count
        logger.exception("run %s: UNEXPECTED FAILURE", run_id)
    finally:
        if staging_root.exists():
            shutil.rmtree(staging_root, ignore_errors=True)
        try:
            path = write_metadata(meta, cfg.metadata_dir)
            logger.info("run %s: metadata -> %s", run_id, path)
        except Exception:  # noqa: BLE001
            logger.exception("run %s: could not write metadata", run_id)

    return result


__all__ = [
    "STATUS_FAILED",
    "STATUS_SKIPPED",
    "STATUS_SUCCESS",
    "SymbolUpdateResult",
    "build_config",
    "update_symbol",
]


