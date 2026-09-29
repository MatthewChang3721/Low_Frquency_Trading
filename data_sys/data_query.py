"""End-to-end market daily-bar MVP (the single pipeline entry point).

Flow
----
1. download raw daily bars from Yahoo Finance (``yfinance``),
2. persist the untouched raw frame under ``data/raw/``,
3. standardize to the daily-bar contract (:mod:`data_sys.schema`),
4. validate with pandera (:mod:`data_sys.quality`),
5. write a Hive-partitioned dataset into ``data/.staging/<run_id>/``,
6. re-validate + DuckDB-verify the staged dataset,
7. atomically publish into ``data/standardized/``,
8. DuckDB-verify the published dataset,
9. write a per-run metadata JSON under ``data/metadata/``.

Any failure aborts before (or rolls back during) the publish, so the live
dataset is never left half-written.

Usage::

    python -m data_sys.data_query --symbol AAPL --start 2020-01-01
"""

from __future__ import annotations

import logging
import shutil
from datetime import date
from pathlib import Path

import typer

from data_sys.config import DEFAULT_START, DEFAULT_SYMBOL, PipelineConfig
from data_sys.duckdb_verify import format_report, verify_dataset
from data_sys.errors import DataPipelineError, DuckDBVerificationError, QualityCheckError
from data_sys.metadata import (
    RunMetadata,
    collect_package_versions,
    configure_logging,
    write_metadata,
)
from data_sys.providers.base import DataProvider
from data_sys.providers.yahoo import YahooFinanceProvider
from data_sys.quality import validate_bars
from data_sys.standardize import standardize_bars
from data_sys.storage import atomic_publish, read_partitioned, write_partitioned, write_raw
from data_sys.utils import new_run_id, parse_date, utc_now_iso

logger = logging.getLogger("data_sys.pipeline")

EXIT_OK = 0
EXIT_FAILED = 1


def _iso(value: object) -> str | None:
    return None if value is None else str(value)


def _build_config(symbol: str, start, end, data_root) -> PipelineConfig:
    kwargs = {
        "symbol": str(symbol).strip().upper(),
        "start": parse_date(start) or DEFAULT_START,
        "end": parse_date(end) or date.today(),
    }
    if data_root is not None:
        kwargs["data_root"] = Path(data_root)
    return PipelineConfig(**kwargs)


def run(
    symbol: str = DEFAULT_SYMBOL,
    start: date | str = DEFAULT_START,
    end: date | str | None = None,
    data_root: Path | str | None = None,
    provider: DataProvider | None = None,
) -> int:
    """Execute the full pipeline for one symbol; return a process exit code."""
    configure_logging()
    run_id = new_run_id()
    cfg = _build_config(symbol, start, end, data_root)

    provider = provider or YahooFinanceProvider()
    source = getattr(provider, "name", type(provider).__name__)

    meta = RunMetadata(
        run_id=run_id,
        data_source=str(source),
        symbol=cfg.symbol,
        requested_start=cfg.start.isoformat(),
        requested_end=cfg.end.isoformat(),
        output_path=str(cfg.standardized_dir),
        run_at_utc=utc_now_iso(),
        package_versions=collect_package_versions(),
    )

    staging_root = cfg.staging_dir / run_id
    logger.info(
        "run %s starting: symbol=%s range=%s..%s source=%s",
        run_id, cfg.symbol, cfg.start, cfg.end, source,
    )

    try:
        raw = provider.fetch_bars(cfg.symbol, cfg.start, cfg.end)
        stem = f"{cfg.symbol}_{cfg.start.isoformat()}_{cfg.end.isoformat()}_{run_id}"
        raw_path = write_raw(raw.frame, cfg.raw_dir / cfg.symbol, stem, raw.meta)
        meta.raw_path = str(raw_path)
        logger.info("run %s: raw rows=%d -> %s", run_id, len(raw.frame), raw_path)

        standardized = standardize_bars(raw.frame, cfg.symbol)
        logger.info("run %s: standardized rows=%d", run_id, len(standardized))

        report = validate_bars(standardized)
        if not report.ok:
            raise QualityCheckError(
                "standardized data failed quality checks", report.reasons()
            )
        logger.info("run %s: in-memory quality checks passed", run_id)

        write_partitioned(standardized, staging_root)
        logger.info("run %s: staged dataset written to %s", run_id, staging_root)

        staged_report = validate_bars(read_partitioned(staging_root))
        if not staged_report.ok:
            raise QualityCheckError(
                "staged dataset failed quality checks", staged_report.reasons()
            )

        staged_verification = verify_dataset(staging_root)
        if not staged_verification.ok:
            raise DuckDBVerificationError(
                "staged dataset failed DuckDB verification: "
                + "; ".join(staged_verification.issues)
            )
        logger.info(
            "run %s: staged dataset verified (%d rows)",
            run_id, staged_verification.n_rows,
        )

        published = atomic_publish(
            staging_root, cfg.standardized_dir, cfg.trash_dir, run_id
        )
        logger.info("run %s: published %s", run_id, [path.name for path in published])

        verification = verify_dataset(cfg.standardized_dir)
        if not verification.ok:
            raise DuckDBVerificationError(
                "published dataset failed DuckDB verification: "
                + "; ".join(verification.issues)
            )
        logger.info(
            "run %s: published dataset verified\n%s", run_id, format_report(verification)
        )

        meta.status = "success"
        meta.row_count = verification.n_rows
        meta.min_date = _iso(verification.min_date)
        meta.max_date = _iso(verification.max_date)
        meta.years = verification.years
        logger.info(
            "run %s: SUCCESS rows=%d range=%s..%s output=%s",
            run_id, meta.row_count, meta.min_date, meta.max_date, cfg.standardized_dir,
        )
        return EXIT_OK

    except DataPipelineError as exc:
        meta.status = "failed"
        meta.failure_reasons = [str(exc)]
        logger.error("run %s: FAILED (%s): %s", run_id, type(exc).__name__, exc)
        return EXIT_FAILED
    except Exception as exc:  # noqa: BLE001
        meta.status = "failed"
        meta.failure_reasons = [f"{type(exc).__name__}: {exc}"]
        logger.exception("run %s: UNEXPECTED FAILURE", run_id)
        return EXIT_FAILED
    finally:
        if staging_root.exists():
            shutil.rmtree(staging_root, ignore_errors=True)
        try:
            path = write_metadata(meta, cfg.metadata_dir)
            logger.info("run %s: metadata -> %s", run_id, path)
        except Exception:  # noqa: BLE001
            logger.exception("run %s: could not write metadata", run_id)


app = typer.Typer(
    add_completion=False,
    help="Market daily-bar MVP pipeline (Yahoo Finance -> Parquet -> DuckDB).",
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
    data_root: str | None = typer.Option(
        None, "--data-root", help="Override the data root directory."
    ),
) -> None:
    """Download, standardize, validate and publish daily bars for ``symbol``."""
    exit_code = run(symbol=symbol, start=start, end=end, data_root=data_root)
    raise typer.Exit(code=exit_code)


if __name__ == "__main__":
    app()

