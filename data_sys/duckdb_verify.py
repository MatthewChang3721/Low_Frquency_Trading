"""DuckDB verification of a freshly written Parquet dataset.

This is an *independent* second pair of eyes: pandera validates the in-memory
contract, DuckDB validates what actually landed on disk (types, row count,
date range, duplicates, nulls, year coverage).

Scope: only ever point this module at a **freshly written, never-replaced**
tree -- in practice the staging dataset ``data/.staging/<run_id>/`` of the
updating process.  DuckDB caches file handles per path internally and redefining
a view does *not* invalidate that cache, so re-reading a path whose parquet files
were replaced underneath it was observed to kill the interpreter with a native
``access violation`` on Windows (and, with the shared connection, to wedge it)
instead of raising a Python exception.  Reads of the *published* dataset
therefore go through PyArrow instead (:mod:`data_sys.summary` and
:func:`data_sys.storage.read_symbol_partition`).

DuckDB is queried through one long-lived in-memory connection: nothing is ever
written to disk.  Creating a fresh in-memory database per check was observed to
crash the interpreter even faster (after a few hundred checks within a single
test session).
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

#: One DuckDB connection for the whole process.  DuckDB is built for long-lived
#: in-process analytics; reconnecting for every check was measured to be not
#: only wasteful but unstable (see the module docstring).
_CONNECTION: duckdb.DuckDBPyConnection | None = None


def connection() -> duckdb.DuckDBPyConnection:
    """Return the process-wide in-memory DuckDB connection.

    Raises
    ------
    data_sys.errors.DuckDBVerificationError
        If DuckDB cannot create the connection.
    """
    global _CONNECTION
    if _CONNECTION is None:
        try:
            _CONNECTION = duckdb.connect()
        except Exception as exc:  # noqa: BLE001
            raise DuckDBVerificationError(
                f"could not open a DuckDB connection: {exc}"
            ) from exc
    return _CONNECTION


def define_view(pattern: str, name: str = "bars") -> None:
    """Point the ``bars`` view at the parquet files matched by ``pattern``.

    The view is *replaced* on every call, so each check queries exactly the
    dataset it asked for.

    .. warning::
       Replacing the view does **not** reset DuckDB's process-wide file-handle
       cache: it rebinds the query, it does not forget the handles of paths that
       were read before.  A path whose parquet files were replaced since the
       last read therefore stays unsafe to re-read (see the module docstring).
    """
    target = connection()
    try:
        target.execute(
            f"CREATE OR REPLACE VIEW {name} AS "
            f"SELECT * FROM read_parquet('{pattern}', hive_partitioning=true)"
        )
    except Exception as exc:  # noqa: BLE001
        raise DuckDBVerificationError(f"DuckDB could not read '{pattern}': {exc}") from exc


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

    ``target`` must be a freshly written tree the process has not read before --
    in practice the current staging dataset.  Never pass the published dataset
    of an updating process: those partitions are replaced in place and
    re-reading them through DuckDB is unsafe (see the module docstring).
    """
    pattern = to_glob(target)
    report = VerificationReport(target=pattern)

    if not find_parquet_files(pattern):
        report.ok = False
        report.issues.append(f"no parquet files matched '{pattern}'")
        return report

    define_view(pattern)
    connection_ = connection()

    described = connection_.execute("DESCRIBE SELECT * FROM bars").fetchall()
    columns = [str(row[0]) for row in described]
    report.column_types = {str(row[0]): str(row[1]) for row in described}

    report.n_rows = int(connection_.execute("SELECT count(*) FROM bars").fetchone()[0])
    if report.n_rows == 0:
        report.ok = False
        report.issues.append("dataset contains 0 rows")
        return report

    if "date" not in columns:
        report.ok = False
        report.issues.append("dataset has no 'date' column")
        return report

    report.min_date, report.max_date = connection_.execute(
        "SELECT min(date), max(date) FROM bars"
    ).fetchone()
    if report.min_date is None or report.max_date is None:
        report.ok = False
        report.issues.append("date range is empty")

    _check_duplicates(connection_, columns, report)
    _check_nulls(connection_, columns, report)
    _collect_years(connection_, columns, report)

    report.sample_head = connection_.execute(
        "SELECT * FROM bars ORDER BY date LIMIT 3"
    ).fetch_df()
    report.sample_tail = connection_.execute(
        "SELECT * FROM bars ORDER BY date DESC LIMIT 3"
    ).fetch_df()

    report.ok = not report.issues
    return report


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


#: Columns of a dataset summary.  Mirrors ``data_sys.summary.SUMMARY_COLUMNS``
#: (kept as a local literal so this verifier stays independent of the reader
#: layer); the names *and their order* are the contract both summarizers share.
SUMMARY_COLUMNS: list[str] = ["symbol", "year", "n_rows", "min_date", "max_date"]


def summarize_dataset(target: Path | str) -> pd.DataFrame:
    """Per-``(symbol, year)`` row counts and date ranges, read through DuckDB.

    Ad-hoc analysis only, and only of a freshly written tree: production code
    summarizes the *published* dataset with
    :func:`data_sys.summary.summarize_dataset` (PyArrow), because the published
    paths are replaced in place and must not be re-read through DuckDB (see the
    module docstring).

    The physical layout is a Hive tree of ``symbol=*/year=*`` directories; DuckDB
    reads all of it as a single logical table via one glob, so no symbol or year
    is special-cased and nothing is reconstructed in pandas.

    Returns
    -------
    pandas.DataFrame
        Columns :data:`SUMMARY_COLUMNS`, one row per symbol/year partition,
        ordered by ``symbol`` then ``year``.  Empty (but correctly shaped) when
        the glob matches no parquet file.

    Raises
    ------
    data_sys.errors.DuckDBVerificationError
        If DuckDB cannot read the dataset.
    """
    pattern = to_glob(target)
    if not find_parquet_files(pattern):
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    define_view(pattern)
    connection_ = connection()

    described = connection_.execute("DESCRIBE SELECT * FROM bars").fetchall()
    columns = [str(row[0]) for row in described]
    if "date" not in columns:
        raise DuckDBVerificationError(f"'{pattern}' has no 'date' column to summarize")

    symbol_expr = "symbol" if "symbol" in columns else "CAST(NULL AS VARCHAR)"
    year_expr = "year" if "year" in columns else "year(date)"
    frame = connection_.execute(
        f"SELECT {symbol_expr} AS symbol, {year_expr} AS year, "
        "count(*) AS n_rows, min(date) AS min_date, max(date) AS max_date "
        "FROM bars GROUP BY 1, 2 ORDER BY 1, 2"
    ).fetch_df()

    return frame.loc[:, SUMMARY_COLUMNS]


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
    "connection",
    "define_view",
    "verify_dataset",
    "format_report",
    "summarize_dataset",
    "to_glob",
    "find_parquet_files",
    "REQUIRED_NON_NULL_COLUMNS",
    "SUMMARY_COLUMNS",
]
