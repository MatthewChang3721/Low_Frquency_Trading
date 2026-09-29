"""Parquet storage: Hive-partitioned writes, reads and atomic publish.

Published layout (Hive)::

    <root>/symbol=AAPL/year=2020/part-0.parquet

``symbol`` and ``year`` are partition keys: they are not duplicated inside the
parquet files and are reconstructed by readers that enable Hive partitioning
(DuckDB ``hive_partitioning=true``, ``pyarrow.dataset`` with
``partitioning="hive"``, ``pandas.read_parquet`` on the directory).

Atomic publish
--------------
1. write the new data into ``data/.staging/<run_id>/symbol=.../``,
2. re-validate and DuckDB-verify the staged data,
3. move the current live partition into ``data/.trash/<run_id>/``,
4. move the staged partition into the live location,
5. delete the trash copy.

If step 4 fails the previous version is restored from trash, so a failed run
never leaves the live dataset missing or half-written.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

from data_sys.errors import PublishError
from data_sys.utils import jsonable

logger = logging.getLogger(__name__)

PARTITION_KEYS: list[str] = ["symbol", "year"]
PARTITION_SCHEMA: pa.Schema = pa.schema([("symbol", pa.string()), ("year", pa.int32())])
DATE_COLUMN = "date"
SYMBOL_COLUMN = "symbol"


def to_arrow_table(frame: pd.DataFrame) -> pa.Table:
    """Convert a standardized frame into an Arrow table.

    ``date`` is explicitly cast to ``date32`` so DuckDB reads it as ``DATE``
    (a pandas ``datetime64`` column would otherwise become ``TIMESTAMP``).
    """
    table = pa.Table.from_pandas(frame, preserve_index=False)
    if DATE_COLUMN in frame.columns:
        values = [None if pd.isna(value) else value.date() for value in frame[DATE_COLUMN]]
        date_array = pa.array(values, type=pa.date32())
        table = table.set_column(table.schema.get_field_index(DATE_COLUMN), DATE_COLUMN, date_array)
    return table


def write_partitioned(frame: pd.DataFrame, dest_root: Path | str) -> Path:
    """Write ``frame`` as a Hive-partitioned parquet dataset under ``dest_root``."""
    if frame is None or len(frame) == 0:
        raise PublishError("refusing to write an empty dataset")
    if DATE_COLUMN not in frame.columns or SYMBOL_COLUMN not in frame.columns:
        raise PublishError("frame must contain 'date' and 'symbol' columns")

    dest_root = Path(dest_root)
    if dest_root.exists():
        shutil.rmtree(dest_root)
    dest_root.mkdir(parents=True, exist_ok=True)

    table = to_arrow_table(frame)
    if "year" in table.column_names:
        table = table.drop(["year"])
    year_array = pa.array([int(value.year) for value in frame[DATE_COLUMN]], type=pa.int32())
    table = table.append_column("year", year_array)

    ds.write_dataset(
        table,
        dest_root,
        format="parquet",
        partitioning=ds.partitioning(PARTITION_SCHEMA, flavor="hive"),
    )
    logger.debug("wrote %d row(s) to %s", len(frame), dest_root)
    return dest_root


def read_partitioned(root: Path | str) -> pd.DataFrame:
    """Read a Hive-partitioned dataset written by :func:`write_partitioned`."""
    root = Path(root)
    if not root.exists() or not any(root.rglob("*.parquet")):
        return pd.DataFrame()

    frame = ds.dataset(root, format="parquet", partitioning="hive").to_table().to_pandas()
    if DATE_COLUMN in frame.columns:
        frame[DATE_COLUMN] = (
            pd.to_datetime(frame[DATE_COLUMN]).dt.normalize().astype("datetime64[s]")
        )
    return frame


def write_raw(frame: pd.DataFrame, dest_dir: Path | str, stem: str, meta: dict) -> Path:
    """Persist the untouched provider frame plus a JSON sidecar with fetch metadata."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    path = dest_dir / f"{stem}.parquet"
    frame.to_parquet(path, engine="pyarrow")

    sidecar = dest_dir / f"{stem}.meta.json"
    sidecar.write_text(json.dumps(jsonable(meta), indent=2, ensure_ascii=False), encoding="utf-8")

    logger.debug("raw data written to %s", path)
    return path


def partition_dirs(root: Path | str) -> list[Path]:
    """Return the ``symbol=*`` partition directories found under ``root``."""
    root = Path(root)
    if not root.exists():
        return []
    return sorted(path for path in root.glob("symbol=*") if path.is_dir())


def atomic_publish(
    staging_root: Path | str,
    live_root: Path | str,
    trash_root: Path | str,
    run_id: str,
) -> list[Path]:
    """Swap staged partitions into the live dataset, restoring on failure."""
    staging_root = Path(staging_root)
    live_root = Path(live_root)
    trash_root = Path(trash_root)

    staged = partition_dirs(staging_root)
    if not staged:
        raise PublishError(f"no 'symbol=*' partitions found under {staging_root}")

    live_root.mkdir(parents=True, exist_ok=True)
    trash_run = trash_root / run_id
    swapped: list[tuple[Path, Path | None]] = []

    try:
        for staged_dir in staged:
            live_target = live_root / staged_dir.name
            backup: Path | None = None

            if live_target.exists():
                trash_run.mkdir(parents=True, exist_ok=True)
                backup = trash_run / live_target.name
                if backup.exists():
                    shutil.rmtree(backup)
                shutil.move(str(live_target), str(backup))
                logger.info("backed up %s -> %s", live_target.name, backup)

            # Record the swap *before* attempting it: if the publish move fails
            # the backup must still be restored, even for the first partition.
            swapped.append((live_target, backup))
            shutil.move(str(staged_dir), str(live_target))
            logger.info("published %s", live_target)
    except Exception as exc:  # noqa: BLE001 - roll back, then normalize the error
        logger.error("atomic publish failed, rolling back: %s", exc)
        for live_target, backup in reversed(swapped):
            try:
                if live_target.exists():
                    shutil.rmtree(live_target)
                if backup is not None and backup.exists():
                    shutil.move(str(backup), str(live_target))
                    logger.info("restored %s from trash", live_target.name)
            except Exception:  # noqa: BLE001
                logger.exception("rollback failed for %s", live_target)
        raise PublishError(f"atomic publish failed for run {run_id}: {exc}") from exc

    if trash_run.exists():
        shutil.rmtree(trash_run, ignore_errors=True)

    return [live_root / path.name for path in staged]


__all__ = [
    "PARTITION_KEYS",
    "PARTITION_SCHEMA",
    "atomic_publish",
    "partition_dirs",
    "read_partitioned",
    "to_arrow_table",
    "write_partitioned",
    "write_raw",
]
