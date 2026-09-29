"""Tests for :mod:`data_sys.storage` (partitioning, round trip, atomic publish)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import data_sys.storage as storage
from data_sys.errors import PublishError
from data_sys.storage import (
    atomic_publish,
    partition_dirs,
    read_partitioned,
    write_partitioned,
    write_raw,
)
from tests.helpers import make_raw_frame, make_standardized_frame


def test_write_partitioned_creates_hive_layout(tmp_path: Path) -> None:
    frame = make_standardized_frame(periods=600)

    root = tmp_path / "ds"
    write_partitioned(frame, root)

    years = sorted(path.name for path in root.glob("symbol=AAPL/year=*"))
    assert years == ["year=2020", "year=2021", "year=2022"]
    assert len(list(root.rglob("*.parquet"))) == 3


def test_read_partitioned_round_trip(tmp_path: Path) -> None:
    frame = make_standardized_frame(periods=300)

    root = tmp_path / "ds"
    write_partitioned(frame, root)
    back = read_partitioned(root)

    assert len(back) == len(frame)
    assert set(back["symbol"]) == {"AAPL"}
    assert str(back["date"].dtype) == "datetime64[s]"
    assert back["date"].min() == frame["date"].min()
    assert back["date"].max() == frame["date"].max()

    merged = back.sort_values("date").reset_index(drop=True)
    expected = frame.sort_values("date").reset_index(drop=True)
    assert np.allclose(merged["close"].to_numpy(), expected["close"].to_numpy())
    assert np.allclose(merged["volume"].to_numpy(), expected["volume"].to_numpy())


def test_read_partitioned_on_missing_directory_returns_empty(tmp_path: Path) -> None:
    assert read_partitioned(tmp_path / "nope").empty


def test_write_partitioned_rejects_empty_frame(tmp_path: Path) -> None:
    with pytest.raises(PublishError):
        write_partitioned(pd.DataFrame(), tmp_path / "ds")


def test_write_raw_creates_parquet_and_sidecar(tmp_path: Path) -> None:
    frame = make_raw_frame(periods=10)

    path = write_raw(frame, tmp_path / "raw", "AAPL_test", {"provider": "fake"})

    assert path.exists()
    sidecar = path.with_name("AAPL_test.meta.json")
    assert sidecar.exists()
    assert "fake" in sidecar.read_text(encoding="utf-8")
    assert len(pd.read_parquet(path)) == 10


def test_partition_dirs_lists_symbols(tmp_path: Path) -> None:
    frame = make_standardized_frame(periods=30)
    root = tmp_path / "ds"
    write_partitioned(frame, root)

    assert [path.name for path in partition_dirs(root)] == ["symbol=AAPL"]
    assert partition_dirs(tmp_path / "missing") == []


def test_atomic_publish_swaps_live_data(tmp_path: Path) -> None:
    live = tmp_path / "live"
    trash = tmp_path / "trash"

    first = make_standardized_frame(periods=100)
    stage1 = tmp_path / "stage1"
    write_partitioned(first, stage1)
    atomic_publish(stage1, live, trash, "run1")
    assert len(read_partitioned(live)) == 100
    assert not (trash / "run1").exists()

    second = make_standardized_frame(periods=400)
    stage2 = tmp_path / "stage2"
    write_partitioned(second, stage2)
    atomic_publish(stage2, live, trash, "run2")

    assert len(read_partitioned(live)) == 400
    assert not (trash / "run2").exists()  # backup cleaned up after a success


def test_atomic_publish_restores_previous_version_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    live = tmp_path / "live"
    trash = tmp_path / "trash"

    original = make_standardized_frame(periods=100)
    stage1 = tmp_path / "stage1"
    write_partitioned(original, stage1)
    atomic_publish(stage1, live, trash, "run1")

    replacement = make_standardized_frame(periods=400)
    stage2 = tmp_path / "stage2"
    write_partitioned(replacement, stage2)

    real_move = storage.shutil.move
    calls = {"count": 0}

    def flaky_move(src, dst):
        calls["count"] += 1
        if calls["count"] == 2:  # 1st = backup live->trash, 2nd = staged->live
            raise OSError("simulated disk failure")
        return real_move(src, dst)

    monkeypatch.setattr(storage.shutil, "move", flaky_move)

    with pytest.raises(PublishError):
        atomic_publish(stage2, live, trash, "run2")

    restored = read_partitioned(live)
    assert len(restored) == 100  # the previous version survived the failed run


def test_atomic_publish_without_staged_partitions_raises(tmp_path: Path) -> None:
    with pytest.raises(PublishError):
        atomic_publish(tmp_path / "empty", tmp_path / "live", tmp_path / "trash", "run1")
