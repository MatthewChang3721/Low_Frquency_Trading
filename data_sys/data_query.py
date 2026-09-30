"""Single-symbol incremental daily-bar update (CLI).

The heavy lifting lives in :func:`data_sys.update.update_symbol`, which is shared
verbatim with the universe batch entry point ``python -m data_sys.batch_query``:

1. read the published partition and plan the missing sessions
   (:mod:`data_sys.planner`, NYSE-calendar aware),
2. download only those windows and archive each raw response untouched,
3. merge the new rows into the published history (new rows win on overlaps),
4. recompute ``daily_return`` / ``dollar_volume`` over the merged history,
5. validate with pandera, stage, DuckDB-verify, atomically publish,
6. record ``data/metadata/run_<run_id>.json``.

The published dataset is only ever replaced by step 5, so a failure anywhere
earlier leaves the previous data exactly as it was.

Usage::

    python -m data_sys.data_query --symbol AAPL --start 2006-01-01 --end 2026-09-30
    python -m data_sys.data_query --symbol AAPL --start 2020-01-01 --no-refresh
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import typer

from data_sys.config import (
    DEFAULT_REFRESH_OVERLAP_SESSIONS,
    DEFAULT_START,
    DEFAULT_SYMBOL,
)
from data_sys.market_calendar import DEFAULT_CALENDAR_NAME
from data_sys.providers.base import DataProvider
from data_sys.update import (
    STATUS_SKIPPED,
    STATUS_SUCCESS,
    SymbolUpdateResult,
    update_symbol,
)

logger = logging.getLogger("data_sys.pipeline")

EXIT_OK = 0
EXIT_FAILED = 1

_SUMMARY_FIELDS = (
    "existing_row_count",
    "downloaded_row_count",
    "published_row_count",
    "new_dates_count",
    "replaced_dates_count",
)


def summarize(result: SymbolUpdateResult) -> str:
    """Render one symbol's outcome as a log-friendly block."""
    lines = [
        f"symbol            : {result.symbol}",
        f"status            : {result.status} (plan: {result.plan_status})",
    ]
    lines.extend(f"{name:<18}: {getattr(result, name)}" for name in _SUMMARY_FIELDS)
    lines.extend(
        [
            f"missing ranges    : {result.missing_ranges}",
            f"refresh range     : {result.refresh_range}",
            f"date range        : {result.min_date} .. {result.max_date}",
            f"raw files         : {len(result.raw_paths)}",
            f"published         : {result.published}",
        ]
    )
    if result.failure_reason:
        lines.append(f"failure           : {result.failure_reason}")
    lines.extend(f"  - {note}" for note in result.notes)
    return "\n".join(lines)


def run(
    symbol: str = DEFAULT_SYMBOL,
    start: date | str = DEFAULT_START,
    end: date | str | None = None,
    data_root: Path | str | None = None,
    provider: DataProvider | None = None,
    refresh_overlap_sessions: int = DEFAULT_REFRESH_OVERLAP_SESSIONS,
    calendar_name: str = DEFAULT_CALENDAR_NAME,
) -> int:
    """Run one incremental update; return a process exit code."""
    result = update_symbol(
        symbol=symbol,
        start=start,
        end=end,
        data_root=data_root,
        provider=provider,
        refresh_overlap_sessions=refresh_overlap_sessions,
        calendar_name=calendar_name,
    )
    logger.info("single-symbol update finished:\n%s", summarize(result))
    if result.status == STATUS_SKIPPED:
        return EXIT_OK
    return EXIT_OK if result.status == STATUS_SUCCESS else EXIT_FAILED


app = typer.Typer(
    add_completion=False,
    help="Incremental daily-bar update for one symbol (Yahoo Finance -> Parquet -> DuckDB).",
)


@app.command()
def main(
    symbol: str = typer.Option(
        DEFAULT_SYMBOL, "--symbol", "-s", help="Ticker symbol, e.g. AAPL."
    ),
    start: str = typer.Option(
        DEFAULT_START.isoformat(), "--start", help="Inclusive start date (YYYY-MM-DD)."
    ),
    end: str | None = typer.Option(
        None, "--end", help="Inclusive end date (YYYY-MM-DD). Defaults to today."
    ),
    refresh_overlap_sessions: int = typer.Option(
        DEFAULT_REFRESH_OVERLAP_SESSIONS,
        "--refresh-overlap-sessions",
        help="Recent NYSE sessions re-downloaded to pick up vendor corrections.",
    ),
    no_refresh: bool = typer.Option(
        False,
        "--no-refresh",
        help="Disable the overlap refresh (same as --refresh-overlap-sessions 0).",
    ),
    data_root: str | None = typer.Option(
        None, "--data-root", help="Override the data root directory."
    ),
    calendar: str = typer.Option(
        DEFAULT_CALENDAR_NAME, "--calendar", help="Exchange calendar used for planning."
    ),
) -> None:
    """Download the missing daily bars for ``symbol`` and publish them."""
    exit_code = run(
        symbol=symbol,
        start=start,
        end=end,
        data_root=data_root,
        refresh_overlap_sessions=0 if no_refresh else refresh_overlap_sessions,
        calendar_name=calendar,
    )
    raise typer.Exit(code=exit_code)


if __name__ == "__main__":
    app()

