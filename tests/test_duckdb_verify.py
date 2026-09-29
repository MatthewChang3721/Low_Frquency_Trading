"""Tests for :mod:`data_sys.duckdb_verify`."""

from __future__ import annotations

from pathlib import Path

import pytest

from data_sys.duckdb_verify import (
    find_parquet_files,
    format_report,
    to_glob,
    verify_dataset,
)
from data_sys.errors import DuckDBVerificationError
from data_sys.storage import write_partitioned
from tests.helpers import make_standardized_frame


def test_verify_partitioned_dataset(tmp_path: Path) -> None:
    frame = make_standardized_frame(periods=300)

    root = tmp_path / "ds"
    write_partitioned(frame, root)
    report = verify_dataset(root)

    assert report.ok, report.issues
    assert report.n_rows == len(frame)
    assert report.duplicate_rows == 0
    assert report.null_counts["volume"] == 0
    assert set(report.null_counts) == {
        "open",
        "high",
        "low",
        "close",
        "adjusted_close",
        "volume",
        "dollar_volume",
    }

    assert str(report.min_date) == str(frame["date"].min().date())
    assert str(report.max_date) == str(frame["date"].max().date())

    # stabilized physical types
    assert report.column_types["date"] == "DATE"
    assert report.column_types["symbol"] == "VARCHAR"
    assert report.column_types["close"] == "DOUBLE"

    assert report.years and report.years[0][0] == 2020
    assert sum(count for _, count in report.years) == len(frame)

    assert len(report.sample_head) == 3
    assert len(report.sample_tail) == 3
    assert "OK" in format_report(report)


def test_verify_single_parquet_file(tmp_path: Path) -> None:
    frame = make_standardized_frame(periods=30)

    root = tmp_path / "ds"
    write_partitioned(frame, root)
    files = find_parquet_files(to_glob(root))
    assert len(files) == 1

    report = verify_dataset(files[0])

    assert report.ok, report.issues
    assert report.n_rows == len(frame)


def test_verify_missing_dataset_reports_not_ok(tmp_path: Path) -> None:
    report = verify_dataset(tmp_path / "does-not-exist")

    assert report.ok is False
    assert report.issues
    assert "FAILED" in format_report(report)


def test_verify_corrupt_parquet_raises(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.parquet"
    corrupt.write_text("this is not parquet", encoding="utf-8")

    with pytest.raises(DuckDBVerificationError):
        verify_dataset(corrupt)


def test_to_glob_variants(tmp_path: Path) -> None:
    directory_glob = to_glob(tmp_path)
    assert directory_glob.endswith("**/*.parquet")
    assert "\\" not in directory_glob

    explicit = str(tmp_path / "a" / "**" / "*.parquet")
    assert to_glob(explicit) == explicit.replace("\\", "/")

    single = tmp_path / "one.parquet"
    single.write_bytes(b"")
    assert to_glob(single) == single.as_posix()
