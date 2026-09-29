"""End-to-end tests for the :mod:`data_sys.data_query` orchestrator.

The provider is always an in-memory fake, so these tests exercise the whole
download -> raw -> standardize -> validate -> stage -> verify -> publish ->
metadata chain without any network access.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from data_sys.data_query import EXIT_FAILED, EXIT_OK, run
from data_sys.duckdb_verify import verify_dataset
from data_sys.errors import DownloadError
from data_sys.storage import read_partitioned
from tests.helpers import FakeProvider, make_raw_frame


def _live_dir(data_root: Path) -> Path:
    return data_root / "standardized" / "market_bars"


def _metadata_records(data_root: Path) -> list[dict]:
    files = sorted((data_root / "metadata").glob("run_*.json"))
    return [json.loads(path.read_text(encoding="utf-8")) for path in files]


def test_pipeline_end_to_end(tmp_path: Path) -> None:
    provider = FakeProvider(frame=make_raw_frame(periods=400, start="2020-01-02"))

    exit_code = run(
        symbol="AAPL",
        start="2020-01-02",
        end="2022-01-01",
        data_root=tmp_path,
        provider=provider,
    )

    assert exit_code == EXIT_OK
    assert provider.calls == [
        ("AAPL", pd.Timestamp("2020-01-02").date(), pd.Timestamp("2022-01-01").date())
    ]

    live = _live_dir(tmp_path)
    report = verify_dataset(live)
    assert report.ok, report.issues
    assert report.n_rows == 400
    assert report.column_types["date"] == "DATE"
    assert report.column_types["symbol"] == "VARCHAR"

    frame = read_partitioned(live)
    assert list(frame["symbol"].unique()) == ["AAPL"]

    # raw layer keeps the untouched vendor frame plus its sidecar
    raw_dir = tmp_path / "raw" / "market_bars" / "AAPL"
    assert len(list(raw_dir.glob("*.parquet"))) == 1
    assert len(list(raw_dir.glob("*.meta.json"))) == 1

    # run metadata
    records = _metadata_records(tmp_path)
    assert len(records) == 1
    record = records[0]
    assert record["status"] == "success"
    assert record["row_count"] == 400
    assert record["symbol"] == "AAPL"
    assert record["data_source"] == "fake"
    assert record["min_date"] == "2020-01-02"
    assert record["raw_path"]
    assert record["package_versions"]["pandas"]

    # staging and trash are cleaned up
    for leftover in (".staging", ".trash"):
        directory = tmp_path / leftover
        if directory.exists():
            assert not list(directory.rglob("*")), leftover


def test_pipeline_failure_preserves_the_previous_dataset(tmp_path: Path) -> None:
    good = FakeProvider(frame=make_raw_frame(periods=200, start="2020-01-02"))
    assert (
        run(
            symbol="AAPL",
            start="2020-01-02",
            end="2021-01-01",
            data_root=tmp_path,
            provider=good,
        )
        == EXIT_OK
    )

    live = _live_dir(tmp_path)
    before = read_partitioned(live)
    assert len(before) == 200

    failing = FakeProvider(error=DownloadError("simulated network outage"))
    assert (
        run(
            symbol="AAPL",
            start="2020-01-02",
            end="2021-01-01",
            data_root=tmp_path,
            provider=failing,
        )
        == EXIT_FAILED
    )

    after = read_partitioned(live)
    assert len(after) == 200
    assert after["close"].tolist() == before["close"].tolist()

    records = _metadata_records(tmp_path)
    assert len(records) == 2
    failed = [record for record in records if record["status"] == "failed"]
    assert len(failed) == 1
    assert failed[0]["failure_reasons"]
    assert "network outage" in failed[0]["failure_reasons"][0]


def test_pipeline_rejects_data_that_fails_quality_checks(tmp_path: Path) -> None:
    raw = make_raw_frame(periods=50)
    raw.loc[raw.index[3], "Close"] = -5.0  # non-positive close
    provider = FakeProvider(frame=raw)

    assert (
        run(
            symbol="AAPL",
            start="2020-01-02",
            end="2021-01-01",
            data_root=tmp_path,
            provider=provider,
        )
        == EXIT_FAILED
    )

    assert not _live_dir(tmp_path).exists()
    records = _metadata_records(tmp_path)
    assert records[0]["status"] == "failed"
    assert "quality" in records[0]["failure_reasons"][0].lower()


def test_pipeline_rejects_empty_download(tmp_path: Path) -> None:
    provider = FakeProvider(frame=pd.DataFrame())

    assert (
        run(
            symbol="AAPL",
            start="2020-01-02",
            end="2021-01-01",
            data_root=tmp_path,
            provider=provider,
        )
        == EXIT_FAILED
    )

    assert not _live_dir(tmp_path).exists()
    assert _metadata_records(tmp_path)[0]["status"] == "failed"


def test_pipeline_normalizes_the_symbol(tmp_path: Path) -> None:
    provider = FakeProvider(frame=make_raw_frame(periods=30))

    assert (
        run(
            symbol="  aapl  ",
            start="2020-01-02",
            end="2021-01-01",
            data_root=tmp_path,
            provider=provider,
        )
        == EXIT_OK
    )

    assert provider.calls[0][0] == "AAPL"
    assert list(read_partitioned(_live_dir(tmp_path))["symbol"].unique()) == ["AAPL"]
