"""Unit tests for :mod:`data_sys.summary` -- the PyArrow dataset summary.

The summary is the production replacement for the DuckDB one: it reads the
*published* tree, whose partitions are replaced in place while an update runs, so
these tests pin down both the numbers it reports and the fact that it needs no
DuckDB to do it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest

from data_sys import batch_query, summary
from data_sys.duckdb_verify import summarize_dataset as duckdb_summarize_dataset
from data_sys.errors import DatasetSummaryError
from data_sys.storage import write_partitioned
from data_sys.summary import SUMMARY_COLUMNS, summarize_dataset, summary_records
from tests.helpers import make_standardized_frame


def _combined_dataset() -> pd.DataFrame:
    """A tiny three-partition dataset: AAPL 2020+2021, MSFT 2020."""
    return pd.concat(
        [
            make_standardized_frame("AAPL", periods=400, start="2020-01-02"),
            make_standardized_frame("MSFT", periods=30, start="2020-01-02"),
        ],
        ignore_index=True,
    )


def test_summary_reports_one_row_per_symbol_and_year(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    combined = _combined_dataset()
    write_partitioned(combined, root)

    summary_frame = summarize_dataset(root)

    assert list(summary_frame.columns) == SUMMARY_COLUMNS
    assert list(zip(summary_frame["symbol"], summary_frame["year"], strict=True)) == [
        ("AAPL", 2020),
        ("AAPL", 2021),
        ("MSFT", 2020),
    ]
    assert sum(summary_frame["n_rows"]) == len(combined)

    # counts and date ranges agree with the frames that were written
    for symbol, periods in (("AAPL", 400), ("MSFT", 30)):
        source = make_standardized_frame(symbol, periods=periods, start="2020-01-02")
        for year, group in source.groupby(source["date"].dt.year):
            row = summary_frame[
                (summary_frame["symbol"] == symbol) & (summary_frame["year"] == year)
            ].iloc[0]
            assert row["n_rows"] == len(group)
            assert row["min_date"] == pd.Timestamp(group["date"].min())
            assert row["max_date"] == pd.Timestamp(group["date"].max())


def test_the_pyarrow_summary_matches_the_duckdb_summary(tmp_path: Path) -> None:
    """Both summarizers agree on a fresh dataset (the only case DuckDB may read)."""
    root = tmp_path / "ds"
    write_partitioned(_combined_dataset(), root)

    pyarrow_summary = summarize_dataset(root)
    duckdb_summary = duckdb_summarize_dataset(root)

    assert list(pyarrow_summary.columns) == list(duckdb_summary.columns) == SUMMARY_COLUMNS
    assert pyarrow_summary["symbol"].tolist() == duckdb_summary["symbol"].tolist()
    assert pyarrow_summary["year"].astype("int64").tolist() == (
        duckdb_summary["year"].astype("int64").tolist()
    )
    assert pyarrow_summary["n_rows"].tolist() == duckdb_summary["n_rows"].tolist()
    for column in ("min_date", "max_date"):
        assert [pd.Timestamp(value).date() for value in pyarrow_summary[column]] == [
            pd.Timestamp(value).date() for value in duckdb_summary[column]
        ]


def test_summary_of_a_missing_or_empty_tree_is_empty(tmp_path: Path) -> None:
    empty = tmp_path / "nothing-here"
    empty.mkdir()

    for root in (tmp_path / "does-not-exist", empty):
        summary_frame = summarize_dataset(root)
        assert list(summary_frame.columns) == SUMMARY_COLUMNS
        assert summary_frame.empty
        assert summary_records(summary_frame) == []


def test_summary_derives_keys_for_an_unpartitioned_file(tmp_path: Path) -> None:
    """An ad-hoc tree without ``symbol=`` / ``year=`` directories still summarizes."""
    root = tmp_path / "flat"
    root.mkdir()
    frame = make_standardized_frame("AAPL", periods=200, start="2020-01-02")
    frame.to_parquet(root / "bars.parquet", engine="pyarrow")

    summary_frame = summarize_dataset(root)

    keys = list(zip(summary_frame["symbol"], summary_frame["year"], strict=True))
    assert keys == [("AAPL", 2020)]
    assert summary_frame["n_rows"].tolist() == [len(frame)]


def test_summary_leaves_the_symbol_null_without_a_symbol_column(tmp_path: Path) -> None:
    root = tmp_path / "flat"
    root.mkdir()
    frame = make_standardized_frame("AAPL", periods=200, start="2020-01-02")
    frame.drop(columns=["symbol"]).to_parquet(root / "bars.parquet", engine="pyarrow")

    summary_frame = summarize_dataset(root)

    assert summary_frame["symbol"].tolist() == [None]
    assert summary_frame["year"].tolist() == [2020]


def test_summary_requires_a_date_column(tmp_path: Path) -> None:
    root = tmp_path / "no-dates"
    root.mkdir()
    pd.DataFrame({"close": [1.0, 2.0]}).to_parquet(root / "bars.parquet", engine="pyarrow")

    with pytest.raises(DatasetSummaryError, match="date"):
        summarize_dataset(root)


def test_summary_tolerates_a_partition_replaced_in_place(tmp_path: Path) -> None:
    """Every update rewrites the same partition paths -- and they stay readable.

    This is precisely the pattern that is unsafe for DuckDB (it caches file
    handles per path); PyArrow has no such cache.
    """
    root = tmp_path / "ds"
    frame = make_standardized_frame("AAPL", periods=120, start="2020-01-02")

    for _ in range(10):
        write_partitioned(frame, root)
        summary_frame = summarize_dataset(root)
        assert summary_frame["n_rows"].tolist() == [len(frame)]
        assert summary_frame["symbol"].tolist() == ["AAPL"]


def test_summary_records_are_json_safe(tmp_path: Path) -> None:
    """The batch metadata keeps real numbers and ISO dates, not numpy/strings."""
    root = tmp_path / "ds"
    write_partitioned(_combined_dataset(), root)

    records = summary_records(summarize_dataset(root))

    assert json.loads(json.dumps(records)) == records
    assert records[0] == {
        "symbol": "AAPL",
        "year": 2020,
        "n_rows": 261,
        "min_date": "2020-01-02",
        "max_date": "2020-12-31",
    }
    assert type(records[0]["year"]) is int
    assert type(records[0]["n_rows"]) is int


def test_summary_records_reject_a_foreign_frame() -> None:
    with pytest.raises(DatasetSummaryError, match="missing column"):
        summary_records(pd.DataFrame({"symbol": ["AAPL"]}))


def test_the_published_reader_path_never_imports_duckdb() -> None:
    """Architectural guard for the read boundary.

    The published partitions are replaced in place while an update runs and
    DuckDB caches file handles per path, so neither the summary reader nor the
    batch CLI may pull DuckDB in.  (``duckdb_verify`` itself is allowed to import
    it, but is only ever pointed at the staging tree.)
    """
    pattern = r"^\s*(?:import|from)\s+\S*duckdb"
    for module in (summary, batch_query):
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert not re.search(pattern, source, flags=re.MULTILINE), module.__name__
