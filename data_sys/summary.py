"""Dataset-wide summary of the *published* Parquet tree -- read with PyArrow.

Why PyArrow (and never DuckDB)?
-------------------------------
``data/standardized/market_bars/`` is the live dataset: an incremental update
*replaces* the files of a ``symbol=*`` partition while the process is still
running.  DuckDB caches file handles per path internally, so handing it a path
whose parquet files were rewritten underneath it is not safe on Windows -- the
interpreter was observed to die with a native ``access violation`` (and, with a
shared connection, to hang) instead of raising a Python exception.  Redefining
the view does **not** reset that cache.

The published tree is therefore read with PyArrow only (this module plus
:mod:`data_sys.storage`), while DuckDB stays confined to verifying the freshly
written, never-replaced staging dataset (:mod:`data_sys.duckdb_verify`).  A
*separate* research process that only queries the published parquet files may of
course use DuckDB -- the rule is "do not mix DuckDB reads with replacements of
the same paths inside one process".

The physical layout is untouched: ``symbol=<SYMBOL>/year=<YEAR>/part-*.parquet``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds

from data_sys.errors import DatasetSummaryError
from data_sys.storage import DATE_COLUMN, SYMBOL_COLUMN, YEAR_COLUMN

logger = logging.getLogger(__name__)

#: Columns of a dataset summary, one row per ``(symbol, year)`` partition.
SUMMARY_COLUMNS: list[str] = [SYMBOL_COLUMN, YEAR_COLUMN, "n_rows", "min_date", "max_date"]

ROWS_COLUMN = "n_rows"
MIN_DATE_COLUMN = "min_date"
MAX_DATE_COLUMN = "max_date"


def open_dataset(root: Path | str) -> ds.Dataset | None:
    """Open a published dataset for reading, or ``None`` when it holds no data.

    Hive partitioning is always enabled: ``symbol`` / ``year`` are partition
    keys, so they are *not* stored inside the parquet files and only exist once
    the reader rebuilds them from the directory names.
    """
    root = Path(root)
    if not root.exists() or not any(root.rglob("*.parquet")):
        return None
    try:
        return ds.dataset(root, format="parquet", partitioning="hive")
    except Exception as exc:  # noqa: BLE001 - normalize the reader error
        raise DatasetSummaryError(f"could not open '{root}' for reading: {exc}") from exc


def empty_summary() -> pd.DataFrame:
    """A correctly shaped, empty summary frame."""
    return pd.DataFrame(
        {
            SYMBOL_COLUMN: pd.Series(dtype="object"),
            YEAR_COLUMN: pd.Series(dtype="Int64"),
            ROWS_COLUMN: pd.Series(dtype="int64"),
            MIN_DATE_COLUMN: pd.Series(dtype="datetime64[ns]"),
            MAX_DATE_COLUMN: pd.Series(dtype="datetime64[ns]"),
        }
    ).loc[:, SUMMARY_COLUMNS]


def _is_missing(value: Any) -> bool:
    """``True`` for ``None``, ``NaN`` and ``pd.NA`` (but never for ``0`` / ``""``)."""
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):  # pragma: no cover - non-scalar input
        return False


def _as_symbol(value: Any) -> str | None:
    return None if _is_missing(value) else str(value)


def _as_year(value: Any) -> int | None:
    return None if _is_missing(value) else int(value)


def _as_iso_date(value: Any) -> str | None:
    return None if _is_missing(value) else pd.Timestamp(value).date().isoformat()


def _ordering_key(item: tuple[tuple[str | None, int | None], list[Any]]) -> tuple[str, int, int]:
    """Sort ``(symbol, year)`` keys, missing values first (DuckDB's ``NULLS FIRST``)."""
    symbol, year = item[0]
    return (
        "" if symbol is None else symbol,
        0 if year is None else 1,
        int(year) if year is not None else 0,
    )


def _fragment_summary(
    fragment: ds.Fragment, names: set[str]
) -> list[tuple[str | None, int | None, int, Any, Any]]:
    """``(symbol, year, n_rows, min_date, max_date)`` for one parquet file.

    A fragment *is* one file and a file lives in exactly one partition
    directory, so its Hive keys are constant and can be read straight off the
    partition expression.  Only the ``date`` column is materialized, which keeps
    a summary of a multi-year dataset cheap.
    """
    keys = ds.get_partition_keys(fragment.partition_expression)
    if keys:
        dates = fragment.to_table(columns=[DATE_COLUMN])[DATE_COLUMN]
        if len(dates) == 0:
            return []
        return [
            (
                _as_symbol(keys.get(SYMBOL_COLUMN)),
                _as_year(keys.get(YEAR_COLUMN)),
                len(dates),
                pc.min(dates).as_py(),
                pc.max(dates).as_py(),
            )
        ]

    # Not a Hive partition: derive the keys from the data itself, so an ad-hoc
    # tree without ``symbol=`` / ``year=`` directories still summarizes.
    columns = [name for name in (SYMBOL_COLUMN, DATE_COLUMN) if name in names]
    table = fragment.to_table(columns=columns)
    if table.num_rows == 0:
        return []

    frame = table.to_pandas()
    if SYMBOL_COLUMN not in frame.columns:
        frame[SYMBOL_COLUMN] = None
    frame[YEAR_COLUMN] = pd.to_datetime(frame[DATE_COLUMN]).dt.year
    grouped = frame.groupby([SYMBOL_COLUMN, YEAR_COLUMN], dropna=False)[DATE_COLUMN].agg(
        ["count", "min", "max"]
    )
    return [
        (_as_symbol(symbol), _as_year(year), int(row["count"]), row["min"], row["max"])
        for (symbol, year), row in grouped.iterrows()
    ]


def summarize_dataset(root: Path | str) -> pd.DataFrame:
    """Per-``(symbol, year)`` row counts and date ranges of a published dataset.

    Same columns and same ordering as ``duckdb_verify.summarize_dataset``, but
    read through PyArrow, so it is safe to call on the live dataset of an
    updating process (see the module docstring).

    Parameters
    ----------
    root:
        Root of the Hive dataset, i.e. ``data/standardized/market_bars``.

    Returns
    -------
    pandas.DataFrame
        Columns :data:`SUMMARY_COLUMNS`, one row per ``(symbol, year)`` ordered
        by ``symbol`` then ``year``.  Empty (but correctly shaped) when the tree
        holds no parquet file.

    Raises
    ------
    data_sys.errors.DatasetSummaryError
        If the dataset cannot be opened, or has no ``date`` column.
    """
    dataset = open_dataset(root)
    if dataset is None:
        logger.debug("no parquet files under %s, returning an empty summary", root)
        return empty_summary()

    names = set(dataset.schema.names)
    if DATE_COLUMN not in names:
        raise DatasetSummaryError(
            f"'{root}' has no '{DATE_COLUMN}' column to summarize (found {sorted(names)})"
        )

    aggregates: dict[tuple[str | None, int | None], list[Any]] = {}
    for fragment in dataset.get_fragments():
        for symbol, year, n_rows, min_date, max_date in _fragment_summary(fragment, names):
            total, low, high = aggregates.setdefault((symbol, year), [0, None, None])
            aggregates[(symbol, year)] = [
                total + n_rows,
                min_date if low is None else min(low, min_date),
                max_date if high is None else max(high, max_date),
            ]

    if not aggregates:
        return empty_summary()

    records = [
        {
            SYMBOL_COLUMN: symbol,
            YEAR_COLUMN: year,
            ROWS_COLUMN: int(values[0]),
            MIN_DATE_COLUMN: values[1],
            MAX_DATE_COLUMN: values[2],
        }
        for (symbol, year), values in sorted(aggregates.items(), key=_ordering_key)
    ]
    frame = pd.DataFrame.from_records(records, columns=SUMMARY_COLUMNS)
    frame[SYMBOL_COLUMN] = frame[SYMBOL_COLUMN].astype("object")
    frame[YEAR_COLUMN] = frame[YEAR_COLUMN].astype("Int64")
    frame[ROWS_COLUMN] = frame[ROWS_COLUMN].astype("int64")
    frame[MIN_DATE_COLUMN] = pd.to_datetime(frame[MIN_DATE_COLUMN])
    frame[MAX_DATE_COLUMN] = pd.to_datetime(frame[MAX_DATE_COLUMN])
    logger.debug("summarized %d partition(s) under %s", len(frame), root)
    return frame.loc[:, SUMMARY_COLUMNS].reset_index(drop=True)


def summary_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """JSON-safe ``records`` view of a :func:`summarize_dataset` frame.

    ``DataFrame.to_dict(orient="records")`` yields numpy scalars, which
    :func:`data_sys.utils.jsonable` does not recognize and would serialize as
    *strings*; this helper emits plain ``int`` and ISO-8601 date values instead,
    so the batch metadata keeps real numbers.

    Raises
    ------
    data_sys.errors.DatasetSummaryError
        If ``frame`` is not a summary frame (missing one of
        :data:`SUMMARY_COLUMNS`).
    """
    missing = [column for column in SUMMARY_COLUMNS if column not in frame.columns]
    if missing:
        raise DatasetSummaryError(f"summary frame is missing column(s): {missing}")

    return [
        {
            SYMBOL_COLUMN: _as_symbol(row[SYMBOL_COLUMN]),
            YEAR_COLUMN: _as_year(row[YEAR_COLUMN]),
            ROWS_COLUMN: None if _is_missing(row[ROWS_COLUMN]) else int(row[ROWS_COLUMN]),
            MIN_DATE_COLUMN: _as_iso_date(row[MIN_DATE_COLUMN]),
            MAX_DATE_COLUMN: _as_iso_date(row[MAX_DATE_COLUMN]),
        }
        for row in frame.loc[:, SUMMARY_COLUMNS].to_dict(orient="records")
    ]


__all__ = [
    "SUMMARY_COLUMNS",
    "empty_summary",
    "open_dataset",
    "summarize_dataset",
    "summary_records",
]

