"""DuckDB verification of a persisted Parquet dataset.

This is an *independent* second pair of eyes: pandera validates the in-memory
contract, DuckDB validates what actually landed on disk (types, row count,
date range, duplicates, nulls, year coverage).

The connection is always closed; no on-disk ``.duckdb`` file is created.
"""

from __future__ import annotations

import glob as _glob
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from data_sys.errors import DuckDBVerificationError

logger = logging.getLogger(__name__)

#: Columns that must never be null in a published dataset.
REQUIRED_NON_NULL_COLUMNS: list[str] = [
    "open",
    "high",
    "low",
    "close",
    "adjusted_close",
    "volume",
    "dollar_volume",
]


@dataclass
class VerificationReport:
    """Outcome of verifying one persisted dataset."""

    target: str
    n_rows: int = 0
    min_date: Any = None
    max_date: Any = None
    duplicate_rows: int = 0
    null_counts: dict[str, int] = field(default_factory=dict)
    years: list[tuple[int, int]] = field(default_factory=list)
    column_types: dict[str, str] = field(default_factory=dict)
    sample_head: pd.DataFrame = field(default_factory=pd.DataFrame)
    sample_tail: pd.DataFrame = field(default_factory=pd.DataFrame)
    ok: bool = True
    issues: list[str] = field(default_factory=list)


def to_glob(target: Path | str) -> str:
    """Normalize a dataset directory (or an existing glob) into a parquet glob."""
    if isinstance(target, str) and any(token in target for token in "*?["):
        return target.replace("\\", "/")
    path = Path(target)
    if path.is_file():
        return path.as_posix()
    return (path / "**" / "*.parquet").as_posix()


def find_parquet_files(pattern: str) -> list[str]:
    """Return the parquet files matched by ``pattern`` (sorted)."""
    return sorted(_glob.glob(pattern, recursive=True))


def verify_dataset(target: Path | str) -> VerificationReport:
    """Verify the parquet dataset located at ``target``.

    Returns a :class:`VerificationReport` describing every check.  A failed read
    raises :class:`~data_sys.errors.DuckDBVerificationError`.
    """
    pattern = to_glob(target)
    report = VerificationReport(target=pattern)

    if not find_parquet_files(pattern):
        report.ok = False
        report.issues.append(f"no parquet files matched '{pattern}'")
        return report

    try:
        connection = duckdb.connect()
    except Exception as exc:  # noqa: BLE001
        raise DuckDBVerificationError(f"could not open a DuckDB connection: {exc}") from exc

    try:
        try:
            connection.execute(
                "CREATE OR REPLACE VIEW bars AS "
                f"SELECT * FROM read_parquet('{pattern}', hive_partitioning=true)"
            )
        except Exception as exc:  # noqa: BLE001
            raise DuckDBVerificationError(f"DuckDB could not read '{pattern}': {exc}") from exc

        described = connection.execute("DESCRIBE SELECT * FROM bars").fetchall()
        columns = [str(row[0]) for row in described]
        report.column_types = {str(row[0]): str(row[1]) for row in described}

        report.n_rows = int(connection.execute("SELECT count(*) FROM bars").fetchone()[0])
        if report.n_rows == 0:
            report.ok = False
            report.issues.append("dataset contains 0 rows")
            return report

        if "date" not in columns:
            report.ok = False
            report.issues.append("dataset has no 'date' column")
            return report

        report.min_date, report.max_date = connection.execute(
            "SELECT min(date), max(date) FROM bars"
        ).fetchone()
        if report.min_date is None or report.max_date is None:
            report.ok = False
            report.issues.append("date range is empty")

        _check_duplicates(connection, columns, report)
        _check_nulls(connection, columns, report)
        _collect_years(connection, columns, report)

        report.sample_head = connection.execute(
            "SELECT * FROM bars ORDER BY date LIMIT 3"
        ).fetch_df()
        report.sample_tail = connection.execute(
            "SELECT * FROM bars ORDER BY date DESC LIMIT 3"
        ).fetch_df()

        report.ok = not report.issues
        return report
    finally:
        connection.close()


def _check_duplicates(
    connection: duckdb.DuckDBPyConnection, columns: list[str], report: VerificationReport
) -> None:
    if "symbol" not in columns:
        return
    report.duplicate_rows = int(
        connection.execute(
            "SELECT count(*) - count(DISTINCT (symbol, date)) FROM bars"
        ).fetchone()[0]
    )
    if report.duplicate_rows:
        report.issues.append(f"{report.duplicate_rows} duplicate (symbol, date) row(s)")


def _check_nulls(
    connection: duckdb.DuckDBPyConnection, columns: list[str], report: VerificationReport
) -> None:
    for column in REQUIRED_NON_NULL_COLUMNS:
        if column not in columns:
            report.null_counts[column] = -1
            report.issues.append(f"missing column '{column}'")
            continue
        nulls = int(
            connection.execute(
                f'SELECT count(*) FROM bars WHERE "{column}" IS NULL'
            ).fetchone()[0]
        )
        report.null_counts[column] = nulls
        if nulls:
            report.issues.append(f"column '{column}' has {nulls} null value(s)")


def _collect_years(
    connection: duckdb.DuckDBPyConnection, columns: list[str], report: VerificationReport
) -> None:
    if "year" in columns:
        rows = connection.execute(
            "SELECT year, count(*) FROM bars GROUP BY year ORDER BY year"
        ).fetchall()
    else:
        rows = connection.execute(
            "SELECT year(date) AS year, count(*) FROM bars GROUP BY 1 ORDER BY 1"
        ).fetchall()
    report.years = [(int(year), int(count)) for year, count in rows]


def format_report(report: VerificationReport) -> str:
    """Render a compact, log-friendly summary of ``report``."""
    lines = [
        f"target        : {report.target}",
        f"rows          : {report.n_rows}",
        f"date range    : {report.min_date} .. {report.max_date}",
        f"duplicates    : {report.duplicate_rows}",
        f"year coverage : {report.years}",
        f"null counts   : {report.null_counts}",
        f"column types  : {report.column_types}",
        f"status        : {'OK' if report.ok else 'FAILED'}",
    ]
    lines.extend(f"  ! {issue}" for issue in report.issues)
    return "\n".join(lines)


__all__ = [
    "VerificationReport",
    "verify_dataset",
    "format_report",
    "to_glob",
    "find_parquet_files",
    "REQUIRED_NON_NULL_COLUMNS",
]
