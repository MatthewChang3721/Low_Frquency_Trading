"""Universe batch incremental update (CLI).

``python -m data_sys.batch_query`` walks a validated stock universe and applies
the *same* :func:`data_sys.update.update_symbol` used by the single-symbol entry
point, once per ticker.  Nothing about planning, merging, validation or
publishing is duplicated here -- this module only adds:

* universe selection (``is_active`` by default, file order preserved),
* strictly sequential execution (no concurrency in this stage),
* per-symbol failure isolation: one broken ticker never stops the batch and
  never rolls back a ticker that already succeeded,
* the batch summary ``data/metadata/batch_<batch_run_id>.json``.

Exit code is ``0`` when every ticker ended in ``success`` or ``skipped``, and
``1`` as soon as one ticker failed -- everything already published stays
published either way.

Usage::

    # one line, copy-paste ready (PowerShell / CMD / Bash)
    python -m data_sys.batch_query -u config/universe_seed.csv --start 2020-01-01 --end 2026-09-30

    # same command with the overlap refresh disabled
    python -m data_sys.batch_query -u config/universe_seed.csv --start 2020-01-01 --no-refresh
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import typer

from data_sys.config import (
    DEFAULT_REFRESH_OVERLAP_SESSIONS,
    DEFAULT_START,
    PipelineConfig,
)
from data_sys.errors import DataPipelineError, UniverseError
from data_sys.market_calendar import DEFAULT_CALENDAR_NAME
from data_sys.metadata import (
    collect_package_versions,
    configure_logging,
    write_json_file,
)
from data_sys.providers.base import DataProvider
from data_sys.providers.yahoo import YahooFinanceProvider
from data_sys.summary import summarize_dataset, summary_records
from data_sys.universe import DEFAULT_UNIVERSE_PATH, Universe, read_universe
from data_sys.update import (
    STATUS_FAILED,
    STATUS_SKIPPED,
    STATUS_SUCCESS,
    SymbolUpdateResult,
    build_config,
    update_symbol,
)
from data_sys.utils import jsonable, new_run_id, utc_now_iso

logger = logging.getLogger("data_sys.batch")

EXIT_OK = 0
EXIT_FAILED = 1

#: Provider factory used by ``run_batch`` when a per-symbol provider is needed.
ProviderFactory = Callable[[str], DataProvider]

@dataclass
class BatchMetadata:
    """Summary record of one universe batch run.

    Written to ``data/metadata/batch_<batch_run_id>.json``.  Per-symbol detail is
    also kept individually in ``run_<batch_run_id>-<SYMBOL>.json``, so this file
    is the index that ties a batch together.
    """

    batch_run_id: str
    universe_path: str
    requested_start: str
    requested_end: str
    refresh_overlap_sessions: int
    calendar: str = DEFAULT_CALENDAR_NAME
    data_source: str = ""
    started_at_utc: str = ""
    finished_at_utc: str = ""
    total_symbols: int = 0
    successful_symbols: int = 0
    failed_symbols: int = 0
    skipped_symbols: int = 0
    exit_code: int = EXIT_OK
    #: Symbols selected from the universe, in the order they were processed.
    requested_symbols: list[str] = field(default_factory=list)
    #: One status record per symbol (see ``data_sys.update.SymbolUpdateResult``).
    per_symbol_status: list[dict[str, Any]] = field(default_factory=list)
    #: ``(symbol, year)`` row counts / date ranges, read back with PyArrow.
    dataset_summary: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    package_versions: dict[str, Any] = field(default_factory=dict)

    @property
    def failed(self) -> list[str]:
        """Symbols whose status is ``failed``."""
        return [
            status["symbol"]
            for status in self.per_symbol_status
            if status["status"] == STATUS_FAILED
        ]

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly view of the whole record."""
        return jsonable(asdict(self))

    def path_for(self, config: PipelineConfig) -> Path:
        """Where this record belongs inside ``config``'s metadata directory."""
        return config.batch_metadata_path(self.batch_run_id)


def _select_entries(
    universe: Universe,
    *,
    include_inactive: bool,
    symbols: Sequence[str] | None,
) -> list[Any]:
    """Pick the universe entries to process, preserving the file's order.

    Raises
    ------
    data_sys.errors.UniverseError
        If a requested symbol is not in the file, or the selection is empty.
    """
    wanted: set[str] | None = None
    if symbols:
        wanted = {str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()}
        if not wanted:
            raise UniverseError("no symbol given to --symbols")
        unknown = sorted(wanted - set(universe.symbols))
        if unknown:
            raise UniverseError(
                f"symbols {unknown} are not present in universe '{universe.source}'",
                [f"unknown symbol {symbol!r}" for symbol in unknown],
            )

    entries = [
        entry for entry in universe.entries if include_inactive or entry.is_active
    ]
    if wanted is not None:
        entries = [entry for entry in entries if entry.symbol in wanted]

    if not entries:
        raise UniverseError(
            f"no symbol selected from universe '{universe.source}'",
            [
                "every entry is is_active=false"
                if not include_inactive
                else "the selection filtered out every entry"
            ],
        )
    return entries


def _tally(results: Sequence[SymbolUpdateResult]) -> tuple[int, int, int]:
    """``(successful, failed, skipped)`` counts for a batch."""
    successful = sum(1 for result in results if result.status == STATUS_SUCCESS)
    failed = sum(1 for result in results if result.status == STATUS_FAILED)
    skipped = sum(1 for result in results if result.status == STATUS_SKIPPED)
    return successful, failed, skipped


def _split_symbols(value: str | None) -> list[str] | None:
    """Parse a comma-separated ``--symbols`` option."""
    if not value:
        return None
    return [token for token in (piece.strip() for piece in value.split(",")) if token]


def run_batch(
    universe_path: Path | str | None = None,
    start: date | str | None = DEFAULT_START,
    end: date | str | None = None,
    refresh_overlap_sessions: int = DEFAULT_REFRESH_OVERLAP_SESSIONS,
    *,
    data_root: Path | str | None = None,
    provider: DataProvider | None = None,
    provider_factory: ProviderFactory | None = None,
    calendar_name: str = DEFAULT_CALENDAR_NAME,
    include_inactive: bool = False,
    symbols: Sequence[str] | None = None,
    batch_run_id: str | None = None,
) -> tuple[int, BatchMetadata]:
    """Update every selected symbol of a universe, sequentially.

    Parameters
    ----------
    universe_path:
        Validated universe CSV; defaults to the committed seed universe.
    start, end:
        Inclusive request window applied to every symbol.
    refresh_overlap_sessions:
        Overlap-refresh width passed straight through to
        :func:`data_sys.update.update_symbol`.
    data_root:
        Override the ``data/`` root.
    provider:
        One provider instance reused for every symbol.
    provider_factory:
        Called per symbol when a symbol-specific provider is wanted; takes
        precedence over ``provider``.
    calendar_name:
        Exchange calendar used for planning.
    include_inactive:
        Also process ``is_active=false`` entries (off by default).
    symbols:
        Optional subset of tickers to process; unknown tickers are an error.
    batch_run_id:
        Reuse an existing batch id (tests); otherwise generated.

    Returns
    -------
    tuple[int, BatchMetadata]
        The process exit code and the record that was written to
        ``data/metadata/batch_<batch_run_id>.json``.  ``UniverseError`` is raised
        (nothing is processed) when the universe itself is unusable.
    """
    configure_logging()
    batch_run_id = batch_run_id or new_run_id()

    universe = read_universe(universe_path)
    entries = _select_entries(
        universe, include_inactive=include_inactive, symbols=symbols
    )

    # A PipelineConfig is symbol-aware only for the paths that mention the
    # symbol; the metadata/standardized *roots* are shared, so building it once
    # from the first entry keeps config.py as the single source of layout truth.
    base_config = build_config(entries[0].symbol, start, end, data_root)
    requested_start = base_config.start.isoformat()
    requested_end = base_config.end.isoformat()

    meta = BatchMetadata(
        batch_run_id=batch_run_id,
        universe_path=universe.source,
        requested_start=requested_start,
        requested_end=requested_end,
        refresh_overlap_sessions=max(0, int(refresh_overlap_sessions)),
        calendar=calendar_name,
        data_source=getattr(provider, "name", type(provider).__name__)
        if provider is not None
        else ("provider_factory" if provider_factory is not None else YahooFinanceProvider.name),
        started_at_utc=utc_now_iso(),
        total_symbols=len(entries),
        requested_symbols=[entry.symbol for entry in entries],
        package_versions=collect_package_versions(),
    )

    logger.info(
        "batch %s starting: %d symbol(s) from %s, range=%s..%s, refresh=%d",
        batch_run_id,
        len(entries),
        universe.source,
        requested_start,
        requested_end,
        meta.refresh_overlap_sessions,
    )

    results: list[SymbolUpdateResult] = []
    for position, entry in enumerate(entries, start=1):
        symbol_provider = (
            provider_factory(entry.symbol)
            if provider_factory is not None
            else (provider or YahooFinanceProvider())
        )
        logger.info(
            "batch %s: [%d/%d] updating %s", batch_run_id, position, len(entries), entry.symbol
        )
        result = update_symbol(
            symbol=entry.symbol,
            start=start,
            end=end,
            data_root=data_root,
            provider=symbol_provider,
            refresh_overlap_sessions=refresh_overlap_sessions,
            calendar_name=calendar_name,
            run_id=f"{batch_run_id}-{entry.symbol}",
            batch_run_id=batch_run_id,
        )
        results.append(result)
        logger.info(
            "batch %s: [%d/%d] %s -> %s (downloaded=%d, published=%d)",
            batch_run_id,
            position,
            len(entries),
            entry.symbol,
            result.status,
            result.downloaded_row_count,
            result.published_row_count,
        )

    successful, failed, skipped = _tally(results)
    meta.successful_symbols = successful
    meta.failed_symbols = failed
    meta.skipped_symbols = skipped
    meta.per_symbol_status = [result.to_status_dict() for result in results]
    meta.exit_code = EXIT_FAILED if failed else EXIT_OK
    meta.finished_at_utc = utc_now_iso()

    _attach_dataset_summary(meta, base_config)

    path = write_json_file(meta.to_dict(), meta.path_for(base_config))
    logger.info(
        "batch %s finished: %s (success=%d, failed=%d, skipped=%d) -> %s",
        batch_run_id,
        "OK" if meta.exit_code == EXIT_OK else "FAILED",
        successful,
        failed,
        skipped,
        path,
    )
    return meta.exit_code, meta


def _attach_dataset_summary(meta: BatchMetadata, config: PipelineConfig) -> None:
    """Add the ``(symbol, year)`` summary of the published dataset.

    The published tree is read with **PyArrow** (:mod:`data_sys.summary`), never
    with DuckDB: a batch replaces ``symbol=*`` partitions while it runs, and
    DuckDB caches file handles per path, so re-reading a just-replaced path can
    kill the interpreter natively on Windows (see :mod:`data_sys.summary`).

    Best-effort: the batch result must still be recorded when the summary cannot
    be produced, so a failure here is noted rather than raised.
    """
    try:
        summary = summarize_dataset(config.standardized_dir)
    except Exception as exc:  # noqa: BLE001 - never lose the batch record
        logger.exception("batch %s: could not summarize the dataset", meta.batch_run_id)
        meta.notes.append(f"dataset summary unavailable: {type(exc).__name__}: {exc}")
        return
    meta.dataset_summary = summary_records(summary)


def format_batch_summary(meta: BatchMetadata) -> str:
    """Render a batch record as a compact, log-friendly block."""
    lines = [
        f"batch run id      : {meta.batch_run_id}",
        f"universe          : {meta.universe_path}",
        f"requested range   : {meta.requested_start} .. {meta.requested_end}",
        f"refresh overlap   : {meta.refresh_overlap_sessions} session(s)",
        f"symbols           : {meta.total_symbols} total, "
        f"{meta.successful_symbols} success, {meta.failed_symbols} failed, "
        f"{meta.skipped_symbols} skipped",
        f"started / finished: {meta.started_at_utc} / {meta.finished_at_utc}",
        f"exit code         : {meta.exit_code}",
    ]
    if meta.failed:
        lines.append(f"failed symbols    : {', '.join(meta.failed)}")
    lines.extend(f"  - {note}" for note in meta.notes)
    return "\n".join(lines)


app = typer.Typer(
    add_completion=False,
    help="Incremental daily-bar update for a validated stock universe.",
)


@app.command()
def main(
    universe: str = typer.Option(
        str(DEFAULT_UNIVERSE_PATH),
        "--universe",
        "-u",
        help="Validated universe CSV (default: config/universe_seed.csv).",
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
    include_inactive: bool = typer.Option(
        False, "--include-inactive", help="Also process is_active=false entries."
    ),
    symbols: str | None = typer.Option(
        None, "--symbols", help="Comma-separated subset of the universe to process."
    ),
    data_root: str | None = typer.Option(
        None, "--data-root", help="Override the data root directory."
    ),
    calendar: str = typer.Option(
        DEFAULT_CALENDAR_NAME, "--calendar", help="Exchange calendar used for planning."
    ),
) -> None:
    """Update every active symbol of ``--universe``, one after the other."""
    try:
        exit_code, meta = run_batch(
            universe_path=universe,
            start=start,
            end=end,
            refresh_overlap_sessions=0 if no_refresh else refresh_overlap_sessions,
            data_root=data_root,
            calendar_name=calendar,
            include_inactive=include_inactive,
            symbols=_split_symbols(symbols),
        )
    except DataPipelineError as exc:
        logger.error("batch aborted before any download: %s", exc)
        raise typer.Exit(code=EXIT_FAILED) from exc

    logger.info("batch finished:\n%s", format_batch_summary(meta))
    raise typer.Exit(code=exit_code)


if __name__ == "__main__":
    app()



