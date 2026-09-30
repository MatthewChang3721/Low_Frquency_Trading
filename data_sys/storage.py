"""Parquet storage: Hive-partitioned writes, reads and atomic publish.

Published layout (Hive)::

    <root>/symbol=AAPL/year=2020/part-0.parquet

``symbol`` and ``year`` are partition keys: they are not duplicated inside the
parquet files and are reconstructed by readers that enable Hive partitioning
(DuckDB ``hive_partitioning=true``, ``pyarrow.dataset`` with
``partitioning="hive"``, ``pandas.read_parquet`` on the directory).

Atomic publish (a two-phase transaction)
----------------------------------------
1. write the new data into ``data/.staging/<run_id>/symbol=.../``,
2. re-validate and DuckDB-verify the staged data,
3. **apply**: back the live partition up into ``data/.trash/<run_id>/`` and move
   the staged partition into the live location, journaling every swap,
4. **confirm**: the caller's ``confirm`` callback reads the live partition back
   and reports the issues it found,
5. **commit**: the backup of step 3 is deleted -- but only once every swap has
   applied *and* the confirmation reported no issue,
6. **roll back** instead, whenever step 3 or 4 fails: every journaled backup goes
   back to its live location and the transaction removes the backup area it
   created itself (never a backup it could not restore) before raising
   :class:`~data_sys.errors.PublishError`.

A failed run therefore leaves either the complete previous version or the
complete new one, never a partition that is missing or half-written.
"""

from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import NamedTuple

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
YEAR_COLUMN = "year"


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
    if YEAR_COLUMN in table.column_names:
        table = table.drop([YEAR_COLUMN])
    year_array = pa.array([int(value.year) for value in frame[DATE_COLUMN]], type=pa.int32())
    table = table.append_column(YEAR_COLUMN, year_array)

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


def read_symbol_partition(
    root: Path | str, symbol: str, *, sort: bool = True
) -> pd.DataFrame:
    """Read one symbol's partitions of a Hive dataset as a single frame.

    Why not simply ``read_partitioned(root / f"symbol={symbol}")``?  Hive
    partitioning recovers partition keys from the path *relative to the dataset
    root*, so pointing a reader at the symbol directory recovers ``year`` but
    silently drops ``symbol``.  This helper re-attaches the symbol, which the
    downstream merge and the pandera contract both require.

    The read deliberately goes through ``pyarrow`` (like
    :func:`read_partitioned`) and *not* through DuckDB: the incremental update
    rereads the live partition right after replacing it, and repeatedly handing
    DuckDB a path whose parquet files have just been rewritten was observed to
    crash the interpreter natively on Windows.  DuckDB stays in the verification
    role, where it looks at freshly written datasets.

    Parameters
    ----------
    root:
        Root of the Hive dataset, i.e. ``data/standardized/market_bars``.
    symbol:
        Ticker whose partition should be read.
    sort:
        Sort by ``(symbol, date)`` so callers get a deterministic order.

    Returns
    -------
    pandas.DataFrame
        The symbol's rows, or an empty frame when the partition does not exist
        yet (a brand-new symbol is not an error).
    """
    symbol = str(symbol).strip().upper()
    partition = Path(root) / f"symbol={symbol}"
    if not partition.exists() or not any(partition.rglob("*.parquet")):
        return pd.DataFrame()

    frame = read_partitioned(partition)
    if frame.empty:
        return frame

    if SYMBOL_COLUMN not in frame.columns:
        frame[SYMBOL_COLUMN] = symbol
    if sort and DATE_COLUMN in frame.columns:
        frame = frame.sort_values(
            [SYMBOL_COLUMN, DATE_COLUMN], kind="stable"
        ).reset_index(drop=True)
    logger.debug("read %d published row(s) for %s", len(frame), symbol)
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


class _Swap(NamedTuple):
    """One partition moved by :func:`atomic_publish` -- an entry of its journal."""

    live_target: Path
    backup: Path | None  # ``None`` while the symbol has no live partition yet


def _roll_back(swapped: Sequence[_Swap], trash_run: Path) -> list[str]:
    """Undo ``swapped`` in reverse order; return the partitions left unrestored.

    The transaction also removes the backup area it created itself, but only once
    that area is *empty* (``rmdir``, never ``rmtree``): a backup which could not
    be restored must survive, so the previous version is never destroyed by the
    rollback itself.
    """
    unrestored: list[str] = []
    for entry in reversed(swapped):
        try:
            if entry.live_target.exists():
                shutil.rmtree(entry.live_target)
            if entry.backup is not None and entry.backup.exists():
                shutil.move(str(entry.backup), str(entry.live_target))
                logger.info("restored %s from trash", entry.live_target.name)
        except Exception:  # noqa: BLE001 - keep restoring the other partitions
            logger.exception("rollback failed for %s", entry.live_target)
            unrestored.append(entry.live_target.name)

    if not unrestored:
        try:
            trash_run.rmdir()
            logger.debug("removed the empty backup area %s", trash_run)
        except FileNotFoundError:
            pass  # the first publish of a symbol needs no backup at all
        except OSError:
            logger.warning("%s still holds entries after the rollback", trash_run)
    return unrestored


def atomic_publish(
    staging_root: Path | str,
    live_root: Path | str,
    trash_root: Path | str,
    run_id: str,
    *,
    confirm: Callable[[], Sequence[str]] | None = None,
) -> list[Path]:
    """Swap staged partitions into the live dataset as a single transaction.

    ``confirm`` runs after every swap has been applied and before the backup is
    dropped; it returns the issues it found, an empty sequence meaning "the swap
    landed".  Any problem -- a failing move *or* a failing confirmation -- rolls
    the transaction back and raises :class:`~data_sys.errors.PublishError`, so the
    live dataset is either the complete new version or the complete previous one.

    Raises
    ------
    data_sys.errors.PublishError
        If ``staging_root`` holds no ``symbol=*`` partition, or if the
        transaction failed (every restored partition is named in the message,
        plus any partition a failed rollback left in ``trash_root``).
    """
    staging_root = Path(staging_root)
    live_root = Path(live_root)
    trash_root = Path(trash_root)

    staged = partition_dirs(staging_root)
    if not staged:
        raise PublishError(f"no 'symbol=*' partitions found under {staging_root}")

    live_root.mkdir(parents=True, exist_ok=True)
    trash_run = trash_root / run_id
    swapped: list[_Swap] = []

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

            # Journal the swap *before* attempting it: if the publish move fails
            # the backup must still be restored, even for the first partition.
            swapped.append(_Swap(live_target, backup))
            shutil.move(str(staged_dir), str(live_target))
            logger.info("published %s", live_target)

        issues = list(confirm()) if confirm is not None else []
        if issues:
            raise PublishError("the published data failed its confirmation: " + "; ".join(issues))
    except Exception as exc:  # noqa: BLE001 - roll back, then normalize the error
        logger.error("atomic publish failed, rolling back: %s", exc)
        unrestored = _roll_back(swapped, trash_run)
        detail = f"; not restored: {unrestored}" if unrestored else ""
        raise PublishError(f"atomic publish failed for run {run_id}: {exc}{detail}") from exc

    # commit: every swap is in place and the confirmation passed, so the backup
    # of this run is no longer needed
    if trash_run.exists():
        shutil.rmtree(trash_run, ignore_errors=True)

    return [live_root / path.name for path in staged]


__all__ = [
    "PARTITION_KEYS",
    "PARTITION_SCHEMA",
    "DATE_COLUMN",
    "SYMBOL_COLUMN",
    "YEAR_COLUMN",
    "atomic_publish",
    "partition_dirs",
    "read_partitioned",
    "read_symbol_partition",
    "to_arrow_table",
    "write_partitioned",
    "write_raw",
]
