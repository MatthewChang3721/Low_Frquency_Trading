"""Tests for :mod:`factor_sys.factor_store`: one panel, one file, one atomic replace.

Every test here writes to ``tmp_path``.  Nothing reads or writes the project's
real ``data/`` directory and nothing reaches the network.
"""

from __future__ import annotations

import ast
import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import factor_sys
import factor_sys.factor_store as store_module
from data_sys.config import PipelineConfig
from factor_sys import (
    FactorContext,
    FactorPanel,
    FactorStore,
    FactorStoreError,
    ParquetFactorStore,
)
from factor_sys.contracts import FACTOR_PANEL_COLUMNS
from factor_sys.errors import FactorContractError
from factors import Momentum12_1

SYMBOLS: tuple[str, ...] = ("AAPL", "MSFT", "NVDA")
NAME: str = "momentum_12_1"
VERSION: str = "1.0.0"
AS_OF: date = date(2024, 12, 31)
SESSIONS: int = 253


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def make_panel(
    *,
    name: str = NAME,
    version: str = VERSION,
    as_of: date = AS_OF,
    symbols: tuple[str, ...] = SYMBOLS,
    values: tuple[float, ...] = (0.10, float("nan"), 0.30),
    eligible: tuple[bool, ...] = (True, False, True),
    row_order: tuple[str, ...] | None = None,
) -> FactorPanel:
    """One valid panel built directly: no bars, no formula, no calendar."""
    order = row_order if row_order is not None else symbols
    value_of = dict(zip(symbols, values, strict=True))
    eligible_of = dict(zip(symbols, eligible, strict=True))
    frame = pd.DataFrame(
        {
            "as_of_date": [pd.Timestamp(as_of)] * len(order),
            "symbol": list(order),
            "factor_name": [name] * len(order),
            "factor_version": [version] * len(order),
            "raw_value": [value_of[symbol] for symbol in order],
            "is_eligible": [eligible_of[symbol] for symbol in order],
        }
    )
    return FactorPanel(frame, universe=symbols)


def momentum_panel(as_of: date = AS_OF) -> FactorPanel:
    """A real ``Momentum12_1`` panel from an in-memory 253-session history."""
    sessions = pd.bdate_range(end=pd.Timestamp(as_of), periods=SESSIONS)
    prices = {
        "AAPL": [100.0] * SESSIONS,  # +30% over 12 months, skipping the last month
        "MSFT": [50.0] * SESSIONS,  # +0%
        "NVDA": [200.0] * SESSIONS,  # -25%
    }
    prices["AAPL"][-22] = 130.0
    prices["NVDA"][-22] = 150.0

    bars = pd.DataFrame(
        [
            {"date": session, "symbol": symbol, "adjusted_close": value}
            for symbol, series in prices.items()
            for session, value in zip(sessions, series, strict=True)
        ]
    )
    context = FactorContext(as_of_date=as_of, universe=SYMBOLS, bars=bars)
    return Momentum12_1().compute(context)


# ---------------------------------------------------------------------------
# Fixtures and small helpers
# ---------------------------------------------------------------------------
@pytest.fixture
def root(tmp_path: Path) -> Path:
    """A store root that does not exist yet."""
    return tmp_path / "store"


@pytest.fixture
def store(root: Path) -> ParquetFactorStore:
    """The Parquet store under test, rooted in the test's temporary directory."""
    return ParquetFactorStore(root)


def stored_files(root: Path) -> list[Path]:
    """Every stored panel file, sorted."""
    return sorted(root.rglob(f"*{store_module.PANEL_SUFFIX}"))


def temporary_files(root: Path) -> list[Path]:
    """Every temporary file a write left behind (none is the expected answer)."""
    return sorted(root.rglob("*.tmp"))


def metadata_names() -> dict[str, str]:
    """``entry -> full schema-metadata key`` for the documented entries."""
    return {entry: f"{store_module.METADATA_PREFIX}{entry}" for entry in store_module.METADATA_KEYS}


def stored_metadata(path: Path) -> dict[str, str]:
    """The factor-store schema metadata of one file, decoded."""
    raw = pq.read_table(str(path)).schema.metadata or {}
    return {
        key.decode(): value.decode()
        for key, value in raw.items()
        if key.startswith(store_module.METADATA_PREFIX.encode())
    }


def with_metadata(table: pa.Table, **overrides: str | None) -> pa.Table:
    """``table`` with its store metadata edited; ``None`` removes one entry."""
    metadata = dict(table.schema.metadata or {})
    for entry, value in overrides.items():
        key = f"{store_module.METADATA_PREFIX}{entry}".encode()
        if value is None:
            metadata.pop(key, None)
        else:
            metadata[key] = value.encode()
    return table.replace_schema_metadata(metadata)


def rewrite(path: Path, table: pa.Table) -> None:
    """Replace a stored file with a hand-made table (a damaged file, on purpose)."""
    pq.write_table(table, str(path), compression="snappy")


# ---------------------------------------------------------------------------
# The interface and the roots it accepts
# ---------------------------------------------------------------------------
def test_the_store_interface_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        FactorStore()  # type: ignore[abstract]


def test_the_parquet_store_is_a_factor_store(store: ParquetFactorStore) -> None:
    assert isinstance(store, FactorStore)


def test_a_minimal_store_can_replace_the_parquet_one() -> None:
    """The interface is enough: a backend does not have to be file-based."""

    class InMemoryStore(FactorStore):
        def __init__(self) -> None:
            self.panels: dict[tuple[str, str, date], FactorPanel] = {}

        def write(self, panel: FactorPanel) -> Path:
            self.panels[(panel.factor_name, panel.factor_version, panel.as_of_date)] = panel
            return Path("memory://panel")

        def read(
            self, factor_name: str, factor_version: str, as_of_date: date | str
        ) -> FactorPanel:
            day = as_of_date if isinstance(as_of_date, date) else date.fromisoformat(as_of_date)
            return self.panels[(factor_name, factor_version, day)]

        def exists(self, factor_name: str, factor_version: str, as_of_date: date | str) -> bool:
            day = as_of_date if isinstance(as_of_date, date) else date.fromisoformat(as_of_date)
            return (factor_name, factor_version, day) in self.panels

        def path_for(self, factor_name: str, factor_version: str, as_of_date: date | str) -> Path:
            return Path(f"memory://{factor_name}/{factor_version}/{as_of_date}")

        def read_range(
            self,
            factor_name: str,
            factor_version: str,
            start_date: date | str,
            end_date: date | str,
        ) -> tuple[FactorPanel, ...]:
            start = start_date if isinstance(start_date, date) else date.fromisoformat(start_date)
            end = end_date if isinstance(end_date, date) else date.fromisoformat(end_date)
            ordered = sorted(self.panels.items(), key=lambda item: item[0][2])
            return tuple(
                panel
                for (name, version, day), panel in ordered
                if (name, version) == (factor_name, factor_version) and start <= day <= end
            )

    store = InMemoryStore()
    panel = make_panel()
    assert store.write(panel) == Path("memory://panel")
    assert store.exists(NAME, VERSION, "2024-12-31") is True
    assert store.path_for(NAME, VERSION, AS_OF) == Path(f"memory://{NAME}/{VERSION}/{AS_OF}")
    assert store.read(NAME, VERSION, AS_OF).universe == SYMBOLS
    assert store.read_range(NAME, VERSION, "2024-12-01", "2024-12-31") == (panel,)


def test_the_default_root_is_the_project_factor_panel_directory() -> None:
    expected = PipelineConfig().data_root / "factors" / "panels"
    assert ParquetFactorStore().root == expected


def test_building_a_store_touches_nothing_on_disk(tmp_path: Path) -> None:
    root = tmp_path / "not-created-yet"
    ParquetFactorStore(root)
    assert not root.exists()


def test_a_custom_root_is_used(tmp_path: Path, store: ParquetFactorStore, root: Path) -> None:
    path = store.write(make_panel())
    assert path.is_relative_to(root)
    assert stored_files(root) == [path]


def test_a_root_may_be_a_string(tmp_path: Path) -> None:
    root = tmp_path / "store"
    store = ParquetFactorStore(str(root))
    path = store.write(make_panel())
    assert path.is_relative_to(root)


def test_the_repr_names_the_root(root: Path) -> None:
    assert repr(ParquetFactorStore(root)) == f"ParquetFactorStore(root={root!r})"


@pytest.mark.parametrize("value", ["", "   ", "\n\t"])
def test_a_blank_root_is_refused(value: str) -> None:
    with pytest.raises(FactorStoreError, match="must not be blank"):
        ParquetFactorStore(value)


@pytest.mark.parametrize("value", [123, 3.5, b"root", ["root"], object()])
def test_a_root_of_the_wrong_type_is_refused(value: Any) -> None:
    with pytest.raises(FactorStoreError, match="path or a string"):
        ParquetFactorStore(value)  # type: ignore[arg-type]


def test_a_root_that_is_a_file_is_refused(tmp_path: Path) -> None:
    target = tmp_path / "not-a-directory"
    target.write_text("x")
    with pytest.raises(FactorStoreError, match="is not a directory"):
        ParquetFactorStore(target)


def test_a_root_that_becomes_a_file_is_refused_on_exists(tmp_path: Path) -> None:
    root = tmp_path / "store"
    store = ParquetFactorStore(root)
    root.write_text("x")
    with pytest.raises(FactorStoreError, match="is not a directory"):
        store.exists(NAME, VERSION, AS_OF)


# ---------------------------------------------------------------------------
# The directory layout and the unique key
# ---------------------------------------------------------------------------
def test_the_panel_is_stored_under_its_hive_path(store: ParquetFactorStore, root: Path) -> None:
    path = store.write(make_panel())
    assert path == (
        root
        / "factor_name=momentum_12_1"
        / "factor_version=1.0.0"
        / "year=2024"
        / "as_of_date=2024-12-31.parquet"
    )
    assert path.is_file()


def test_the_year_and_file_name_come_from_the_panel(store: ParquetFactorStore) -> None:
    path = store.write(make_panel(as_of=date(2023, 6, 15)))
    assert path.parent.name == "year=2023"
    assert path.name == "as_of_date=2023-06-15.parquet"


def test_one_file_holds_one_panel(store: ParquetFactorStore, root: Path) -> None:
    store.write(make_panel())
    files = stored_files(root)
    assert len(files) == 1
    table = pq.read_table(str(files[0]))
    assert table.column_names == list(FACTOR_PANEL_COLUMNS)
    assert table.num_rows == len(SYMBOLS)


def test_write_creates_a_missing_root(tmp_path: Path) -> None:
    root = tmp_path / "a" / "b" / "c"
    store = ParquetFactorStore(root)
    path = store.write(make_panel())
    assert root.is_dir()
    assert path.is_file()


# ---------------------------------------------------------------------------
# Reading a panel back
# ---------------------------------------------------------------------------
def test_a_panel_round_trips_through_the_store(store: ParquetFactorStore) -> None:
    panel = make_panel()
    path = store.write(panel)

    loaded = store.read(NAME, VERSION, AS_OF)

    assert isinstance(loaded, FactorPanel)
    assert loaded is not panel
    assert path.is_file()
    assert loaded.as_of_date == AS_OF
    assert loaded.universe == SYMBOLS
    assert loaded.factor_name == NAME
    assert loaded.factor_version == VERSION
    assert loaded.symbols == panel.symbols
    assert len(loaded) == len(panel)


def test_the_frame_is_restored_exactly(store: ParquetFactorStore) -> None:
    panel = make_panel()
    store.write(panel)

    loaded = store.read(NAME, VERSION, AS_OF)

    pd.testing.assert_frame_equal(loaded.frame, panel.frame)
    assert list(loaded.frame.columns) == list(FACTOR_PANEL_COLUMNS)
    assert loaded.frame["as_of_date"].dtype == panel.frame["as_of_date"].dtype
    assert loaded.frame["raw_value"].dtype == panel.frame["raw_value"].dtype
    assert loaded.frame["is_eligible"].dtype == panel.frame["is_eligible"].dtype


def test_missing_values_and_ineligible_rows_survive(store: ParquetFactorStore) -> None:
    panel = make_panel(values=(0.10, float("nan"), 0.30), eligible=(True, False, True))
    store.write(panel)

    loaded = store.read(NAME, VERSION, AS_OF)

    assert loaded.frame["raw_value"].isna().sum() == 1
    assert loaded.frame["is_eligible"].tolist() == [True, False, True]
    assert loaded.frame.loc[1, "symbol"] == "MSFT"


def test_the_universe_keeps_its_original_order(store: ParquetFactorStore) -> None:
    """Rows may be in any order; the universe metadata is what fixes the order."""
    panel = make_panel(row_order=("NVDA", "AAPL", "MSFT"))
    store.write(panel)

    loaded = store.read(NAME, VERSION, AS_OF)

    assert loaded.universe == SYMBOLS
    assert loaded.symbols == ("NVDA", "AAPL", "MSFT")
    pd.testing.assert_frame_equal(loaded.frame, panel.frame)


def test_a_single_symbol_panel_round_trips(store: ParquetFactorStore) -> None:
    panel = make_panel(symbols=("AAPL",), values=(0.5,), eligible=(True,))
    store.write(panel)

    loaded = store.read(NAME, VERSION, AS_OF)

    assert loaded.universe == ("AAPL",)
    pd.testing.assert_frame_equal(loaded.frame, panel.frame)


@pytest.mark.parametrize("requested", [AS_OF, "2024-12-31"])
def test_the_session_may_be_a_date_or_an_iso_string(
    store: ParquetFactorStore, requested: date | str
) -> None:
    store.write(make_panel())
    assert store.read(NAME, VERSION, requested).as_of_date == AS_OF


def test_a_read_returns_a_new_panel_each_time(store: ParquetFactorStore) -> None:
    store.write(make_panel())
    first = store.read(NAME, VERSION, AS_OF)
    second = store.read(NAME, VERSION, AS_OF)
    assert first is not second
    assert first.frame is not second.frame


def test_mutating_a_read_frame_does_not_change_the_store(store: ParquetFactorStore) -> None:
    panel = make_panel()
    store.write(panel)

    first = store.read(NAME, VERSION, AS_OF)
    mutated = first.frame
    mutated.loc[0, "raw_value"] = 999.0
    mutated.loc[0, "symbol"] = "ZZZZ"

    loaded = store.read(NAME, VERSION, AS_OF)
    pd.testing.assert_frame_equal(loaded.frame, panel.frame)


# ---------------------------------------------------------------------------
# Overwriting one key
# ---------------------------------------------------------------------------
def test_writing_the_same_key_again_replaces_the_panel(
    store: ParquetFactorStore, root: Path
) -> None:
    store.write(make_panel(values=(0.1, 0.2, 0.3), eligible=(True, True, True)))
    second = make_panel(values=(9.0, 8.0, 7.0), eligible=(True, True, True))
    path = store.write(second)

    loaded = store.read(NAME, VERSION, AS_OF)

    assert stored_files(root) == [path]
    assert loaded.frame["raw_value"].tolist() == [9.0, 8.0, 7.0]


def test_an_overwrite_keeps_the_same_path(store: ParquetFactorStore) -> None:
    first = store.write(make_panel(values=(0.1, 0.2, 0.3)))
    second = store.write(make_panel(values=(0.4, 0.5, 0.6)))
    assert first == second


def test_an_overwrite_leaves_no_temporary_file(store: ParquetFactorStore, root: Path) -> None:
    store.write(make_panel(values=(0.1, 0.2, 0.3)))
    store.write(make_panel(values=(0.4, 0.5, 0.6)))
    assert temporary_files(root) == []
    assert len(stored_files(root)) == 1


# ---------------------------------------------------------------------------
# Different keys never collide
# ---------------------------------------------------------------------------
def test_different_dates_are_different_panels(store: ParquetFactorStore, root: Path) -> None:
    first = date(2024, 12, 30)
    second = date(2024, 12, 31)
    store.write(make_panel(as_of=first, values=(0.1, 0.1, 0.1)))
    store.write(make_panel(as_of=second, values=(0.2, 0.2, 0.2)))

    assert len(stored_files(root)) == 2
    assert store.read(NAME, VERSION, first).as_of_date == first
    assert store.read(NAME, VERSION, second).as_of_date == second
    assert store.read(NAME, VERSION, first).frame["raw_value"].tolist() == [0.1, 0.1, 0.1]


def test_different_versions_are_different_panels(store: ParquetFactorStore, root: Path) -> None:
    store.write(make_panel(version="1.0.0", values=(0.1, 0.1, 0.1)))
    store.write(make_panel(version="1.1.0", values=(0.2, 0.2, 0.2)))

    assert len(stored_files(root)) == 2
    assert store.read(NAME, "1.0.0", AS_OF).factor_version == "1.0.0"
    assert store.read(NAME, "1.1.0", AS_OF).factor_version == "1.1.0"


def test_different_factor_names_are_different_panels(store: ParquetFactorStore, root: Path) -> None:
    store.write(make_panel(name="momentum_12_1", values=(0.1, 0.1, 0.1)))
    store.write(make_panel(name="value_quality", values=(0.2, 0.2, 0.2)))

    assert len(stored_files(root)) == 2
    assert store.read("value_quality", VERSION, AS_OF).factor_name == "value_quality"


def test_the_unique_key_is_name_version_and_date(store: ParquetFactorStore, root: Path) -> None:
    dates = (date(2024, 12, 30), date(2024, 12, 31))
    for name in ("momentum_12_1", "value_quality"):
        for version in ("1.0.0", "1.1.0"):
            for day in dates:
                store.write(make_panel(name=name, version=version, as_of=day))

    assert len(stored_files(root)) == 8
    for name in ("momentum_12_1", "value_quality"):
        for version in ("1.0.0", "1.1.0"):
            for day in dates:
                assert store.read(name, version, day).as_of_date == day


# ---------------------------------------------------------------------------
# exists()
# ---------------------------------------------------------------------------
def test_exists_is_false_before_and_true_after_a_write(store: ParquetFactorStore) -> None:
    assert store.exists(NAME, VERSION, AS_OF) is False
    store.write(make_panel())
    assert store.exists(NAME, VERSION, AS_OF) is True
    assert store.exists(NAME, VERSION, "2024-12-31") is True


def test_exists_is_false_for_another_key(store: ParquetFactorStore) -> None:
    store.write(make_panel())
    assert store.exists(NAME, "1.1.0", AS_OF) is False
    assert store.exists("value_quality", VERSION, AS_OF) is False
    assert store.exists(NAME, VERSION, date(2024, 12, 30)) is False


def test_exists_is_false_when_the_root_does_not_exist(tmp_path: Path) -> None:
    store = ParquetFactorStore(tmp_path / "missing")
    assert store.exists(NAME, VERSION, AS_OF) is False


def test_exists_does_not_read_the_file(store: ParquetFactorStore) -> None:
    """A stored file is enough for ``exists``; its content is the reader's business."""
    path = store.write(make_panel())
    path.write_bytes(b"not parquet")
    assert store.exists(NAME, VERSION, AS_OF) is True


# ---------------------------------------------------------------------------
# path_for(): where a key lives, without touching anything
# ---------------------------------------------------------------------------
def test_path_for_is_the_path_write_returns(store: ParquetFactorStore) -> None:
    """One key, one file: the address a caller is told is the file that was written."""
    written = store.write(make_panel())

    assert store.path_for(NAME, VERSION, AS_OF) == written
    assert store.path_for(NAME, VERSION, "2024-12-31") == written


def test_path_for_works_for_a_key_that_was_never_stored(tmp_path: Path) -> None:
    """The path is computed, not looked up: nothing has to exist for an answer."""
    root = tmp_path / "store"
    store = ParquetFactorStore(root)

    path = store.path_for(NAME, VERSION, AS_OF)

    assert path == (
        root
        / f"factor_name={NAME}"
        / f"factor_version={VERSION}"
        / "year=2024"
        / "as_of_date=2024-12-31.parquet"
    )
    assert not path.exists()
    assert not root.exists()  # no directory, no file, no temporary residue


def test_path_for_creates_nothing_under_an_existing_root(store: ParquetFactorStore) -> None:
    """Asking for an address never becomes a write, not even a directory."""
    store.root.mkdir()

    store.path_for(NAME, VERSION, AS_OF)

    assert list(store.root.iterdir()) == []


def test_path_for_leaves_a_stored_panel_untouched(store: ParquetFactorStore) -> None:
    """Reporting the location of a panel does not read, rewrite or restamp it."""
    written = store.write(make_panel())
    before = (written.stat().st_mtime_ns, written.read_bytes())

    assert store.path_for(NAME, VERSION, AS_OF) == written

    assert (written.stat().st_mtime_ns, written.read_bytes()) == before


def test_path_for_agrees_with_exists_and_read(store: ParquetFactorStore) -> None:
    """All three name the same key: one address, one file, one panel."""
    written = store.write(make_panel())

    assert store.path_for(NAME, VERSION, AS_OF) == written
    assert store.exists(NAME, VERSION, AS_OF) is True
    assert store.read(NAME, VERSION, AS_OF).as_of_date == AS_OF
    assert store.path_for(NAME, VERSION, date(2024, 12, 30)) != written
    assert store.exists(NAME, VERSION, date(2024, 12, 30)) is False


@pytest.mark.parametrize("value", [None, 20241231, 20241231.0, True, object()])
def test_path_for_refuses_an_illegal_session(store: ParquetFactorStore, value: Any) -> None:
    """The key rules are the same ones ``exists`` and ``read`` apply."""
    with pytest.raises(FactorStoreError, match="as_of_date"):
        store.path_for(NAME, VERSION, value)
    with pytest.raises(FactorStoreError, match="as_of_date"):
        store.exists(NAME, VERSION, value)


@pytest.mark.parametrize("value", ["2024-02-30", "2024-13-01", "31/12/2024", "  ", ""])
def test_path_for_refuses_a_session_that_is_not_a_day(
    store: ParquetFactorStore, value: Any
) -> None:
    with pytest.raises(FactorStoreError, match="as_of_date"):
        store.path_for(NAME, VERSION, value)


def test_path_for_refuses_a_datetime(store: ParquetFactorStore) -> None:
    """A panel is stored per session: a time of day is not part of the key."""
    with pytest.raises(FactorStoreError, match="datetime"):
        store.path_for(NAME, VERSION, datetime(2024, 12, 31, 9, 30))


@pytest.mark.parametrize("name", ["Not A Name", "", "momentum-12-1", None, 42])
def test_path_for_refuses_an_illegal_name(store: ParquetFactorStore, name: Any) -> None:
    with pytest.raises(FactorStoreError, match="factor_name"):
        store.path_for(name, VERSION, AS_OF)


@pytest.mark.parametrize("version", ["", "  ", None, 100, ".", ".."])
def test_path_for_refuses_an_illegal_version(store: ParquetFactorStore, version: Any) -> None:
    with pytest.raises(FactorStoreError, match="factor_version"):
        store.path_for(NAME, version, AS_OF)


@pytest.mark.parametrize(
    "version",
    [
        "../escape",
        "..\\escape",
        "..\\..\\escape",
        "a/b",
        "a\\b",
        "C:/evil",
        "/absolute",
        "..%2F..",
        "with space",
        "a\x00b",
        "version..",
        "version ",
        "CON",
        "nul.txt",
    ],
)
def test_path_for_encodes_a_dangerous_version_inside_the_root(tmp_path: Path, version: str) -> None:
    """The store's own encoding is used, so no version can name a path outside it."""
    root = tmp_path / "store"
    store = ParquetFactorStore(root)

    path = store.path_for(NAME, version, AS_OF)

    assert path.resolve().is_relative_to(root.resolve())
    assert path.name == "as_of_date=2024-12-31.parquet"
    version_dir = path.parents[1]
    assert version_dir.name.startswith("factor_version=")
    segment = version_dir.name.removeprefix("factor_version=")
    assert "/" not in segment
    assert "\\" not in segment
    assert not path.exists()
    assert not root.exists()


def test_path_for_is_the_path_a_dangerous_version_is_written_to(tmp_path: Path) -> None:
    """A hit reports exactly where the write of the same key put the panel."""
    store = ParquetFactorStore(tmp_path / "store")
    version = "..\\escape"

    written = store.write(make_panel(version=version))

    assert store.path_for(NAME, version, AS_OF) == written
    assert written.is_file()


# ---------------------------------------------------------------------------
# A missing panel
# ---------------------------------------------------------------------------
def test_reading_a_missing_panel_is_a_store_error(store: ParquetFactorStore) -> None:
    with pytest.raises(FactorStoreError) as excinfo:
        store.read(NAME, VERSION, AS_OF)
    message = str(excinfo.value)
    assert NAME in message
    assert VERSION in message
    assert "2024-12-31" in message
    assert "does not exist" in message
    assert not isinstance(excinfo.value, FileNotFoundError)


def test_reading_from_a_root_that_does_not_exist_is_a_store_error(tmp_path: Path) -> None:
    store = ParquetFactorStore(tmp_path / "missing")
    with pytest.raises(FactorStoreError, match="does not exist"):
        store.read(NAME, VERSION, AS_OF)


def test_reading_a_key_that_was_never_written_names_the_key(
    store: ParquetFactorStore,
) -> None:
    store.write(make_panel(as_of=date(2024, 12, 30)))
    with pytest.raises(FactorStoreError, match="2024-12-31"):
        store.read(NAME, VERSION, AS_OF)


# ---------------------------------------------------------------------------
# Illegal requests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("value", [None, pd.DataFrame(), {}, "panel", 42])
def test_writing_something_that_is_not_a_panel_is_refused(
    store: ParquetFactorStore, value: Any
) -> None:
    with pytest.raises(FactorStoreError, match="FactorPanel"):
        store.write(value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "name",
    ["Momentum12_1", "momentum 12", "momentum-12", "", "momentum/12", "momentum_12_1\n", None, 12],
)
def test_an_illegal_factor_name_is_refused(store: ParquetFactorStore, name: Any) -> None:
    with pytest.raises(FactorStoreError, match="factor_name"):
        store.read(name, VERSION, AS_OF)
    with pytest.raises(FactorStoreError, match="factor_name"):
        store.exists(name, VERSION, AS_OF)


@pytest.mark.parametrize("version", ["", "   ", None, 1, 3.5, b"1.0.0"])
def test_an_illegal_version_is_refused(store: ParquetFactorStore, version: Any) -> None:
    with pytest.raises(FactorStoreError, match="factor_version"):
        store.read(NAME, version, AS_OF)
    with pytest.raises(FactorStoreError, match="factor_version"):
        store.exists(NAME, version, AS_OF)


@pytest.mark.parametrize("version", [".", ".."])
def test_a_dotted_version_is_refused(store: ParquetFactorStore, version: str) -> None:
    with pytest.raises(FactorStoreError, match="cannot name a directory"):
        store.read(NAME, version, AS_OF)
    with pytest.raises(FactorStoreError, match="cannot name a directory"):
        ParquetFactorStore(store.root).write(make_panel(version=version))


@pytest.mark.parametrize(
    "value",
    [
        None,
        123,
        "2024-1-5",
        "20240105",
        "05/01/2024",
        "2024-02-30",
        "2024-12-31T00:00:00",
        datetime(2024, 12, 31, 16, 0),
    ],
)
def test_an_illegal_session_is_refused(store: ParquetFactorStore, value: Any) -> None:
    with pytest.raises(FactorStoreError, match="as_of_date"):
        store.read(NAME, VERSION, value)
    with pytest.raises(FactorStoreError, match="as_of_date"):
        store.exists(NAME, VERSION, value)


def test_an_illegal_request_fails_before_anything_is_touched(
    tmp_path: Path,
) -> None:
    root = tmp_path / "store"
    store = ParquetFactorStore(root)
    with pytest.raises(FactorStoreError, match="factor_name"):
        store.read("Not A Name", VERSION, AS_OF)
    assert not root.exists()


@pytest.mark.parametrize(
    "version",
    [
        "../escape",
        "..\\escape",
        "..\\..\\escape",
        "a/b",
        "a\\b",
        "C:/evil",
        "/absolute",
        "..%2F..",
        "with space",
        "a\x00b",
        "version..",
        "version ",
        "CON",
        "nul.txt",
    ],
)
def test_a_dangerous_version_cannot_escape_the_root(tmp_path: Path, version: str) -> None:
    root = tmp_path / "store"
    store = ParquetFactorStore(root)

    path = store.write(make_panel(version=version))

    assert path.resolve().is_relative_to(root.resolve())
    version_dir = path.parents[1]
    assert version_dir.name.startswith("factor_version=")
    segment = version_dir.name.removeprefix("factor_version=")
    assert "/" not in segment
    assert "\\" not in segment
    assert set(tmp_path.iterdir()) == {root}
    assert store.exists(NAME, version, AS_OF) is True
    loaded = store.read(NAME, version, AS_OF)
    assert loaded.factor_version == version
    pd.testing.assert_frame_equal(loaded.frame, make_panel(version=version).frame)


def test_the_version_encoding_never_merges_two_keys(store: ParquetFactorStore, root: Path) -> None:
    store.write(make_panel(version="1.0.0", values=(0.1, 0.1, 0.1)))
    store.write(make_panel(version="1%2E0%2E0", values=(0.2, 0.2, 0.2)))

    assert len(stored_files(root)) == 2
    assert store.read(NAME, "1.0.0", AS_OF).factor_version == "1.0.0"
    assert store.read(NAME, "1%2E0%2E0", AS_OF).factor_version == "1%2E0%2E0"


def test_writing_under_a_root_that_is_a_file_fails_clearly(tmp_path: Path) -> None:
    root = tmp_path / "store"
    store = ParquetFactorStore(root)
    root.write_text("x")
    with pytest.raises(FactorStoreError):
        store.write(make_panel())
    assert root.is_file()
    assert stored_files(tmp_path) == []


# ---------------------------------------------------------------------------
# What the file carries
# ---------------------------------------------------------------------------
def test_the_file_carries_the_documented_metadata(store: ParquetFactorStore) -> None:
    path = store.write(make_panel())

    metadata = stored_metadata(path)

    assert set(metadata) == set(metadata_names().values())
    assert metadata[metadata_names()["format_version"]] == store_module.FORMAT_VERSION
    assert metadata[metadata_names()["factor_name"]] == NAME
    assert metadata[metadata_names()["factor_version"]] == VERSION
    assert metadata[metadata_names()["as_of_date"]] == "2024-12-31"
    assert json.loads(metadata[metadata_names()["universe"]]) == list(SYMBOLS)


def test_the_metadata_records_a_utc_write_time(store: ParquetFactorStore) -> None:
    path = store.write(make_panel())
    stored_at = datetime.fromisoformat(stored_metadata(path)[metadata_names()["stored_at_utc"]])
    assert stored_at.tzinfo is not None
    assert stored_at.utcoffset() == timedelta(0)
    assert stored_at <= datetime.now(stored_at.tzinfo)


def test_the_universe_order_is_written_into_the_metadata(
    store: ParquetFactorStore,
) -> None:
    panel = make_panel(row_order=("NVDA", "AAPL", "MSFT"))
    path = store.write(panel)
    assert json.loads(stored_metadata(path)[metadata_names()["universe"]]) == list(SYMBOLS)


def test_the_table_holds_only_the_panel_columns(store: ParquetFactorStore) -> None:
    path = store.write(make_panel())
    assert pq.read_schema(str(path)).names == list(FACTOR_PANEL_COLUMNS)


def test_a_rewrite_replaces_the_panel_and_its_metadata(store: ParquetFactorStore) -> None:
    """``stored_at_utc`` is provenance, not identity: it is not part of the key."""
    path = store.write(make_panel())
    first = stored_metadata(path)[metadata_names()["stored_at_utc"]]
    store.write(make_panel(values=(0.4, 0.5, 0.6)))
    second = stored_metadata(path)[metadata_names()["stored_at_utc"]]
    assert first <= second
    assert store.read(NAME, VERSION, AS_OF).frame["raw_value"].tolist() == [0.4, 0.5, 0.6]


# ---------------------------------------------------------------------------
# Damaged files are refused, never guessed at
# ---------------------------------------------------------------------------
def test_a_file_without_metadata_is_refused(store: ParquetFactorStore) -> None:
    path = store.write(make_panel())
    rewrite(path, pq.read_table(str(path)).replace_schema_metadata(None))
    with pytest.raises(FactorStoreError, match="no parquet schema metadata"):
        store.read(NAME, VERSION, AS_OF)


@pytest.mark.parametrize("entry", store_module.METADATA_KEYS)
def test_a_file_missing_one_metadata_entry_is_refused(
    store: ParquetFactorStore, entry: str
) -> None:
    path = store.write(make_panel())
    rewrite(path, with_metadata(pq.read_table(str(path)), **{entry: None}))
    with pytest.raises(FactorStoreError, match="missing the metadata entries"):
        store.read(NAME, VERSION, AS_OF)


def test_an_unknown_format_version_is_refused(store: ParquetFactorStore) -> None:
    path = store.write(make_panel())
    rewrite(path, with_metadata(pq.read_table(str(path)), format_version="999"))
    with pytest.raises(FactorStoreError, match="format '999'"):
        store.read(NAME, VERSION, AS_OF)


def test_a_metadata_name_that_disagrees_with_the_request_is_refused(
    store: ParquetFactorStore,
) -> None:
    path = store.write(make_panel())
    rewrite(path, with_metadata(pq.read_table(str(path)), factor_name="value_quality"))
    with pytest.raises(FactorStoreError, match="belongs to another key"):
        store.read(NAME, VERSION, AS_OF)


def test_a_metadata_version_that_disagrees_with_the_request_is_refused(
    store: ParquetFactorStore,
) -> None:
    path = store.write(make_panel())
    rewrite(path, with_metadata(pq.read_table(str(path)), factor_version="2.0.0"))
    with pytest.raises(FactorStoreError, match="belongs to another key"):
        store.read(NAME, VERSION, AS_OF)


def test_a_metadata_date_that_disagrees_with_the_request_is_refused(
    store: ParquetFactorStore,
) -> None:
    path = store.write(make_panel())
    rewrite(path, with_metadata(pq.read_table(str(path)), as_of_date="2024-12-30"))
    with pytest.raises(FactorStoreError, match="belongs to another key"):
        store.read(NAME, VERSION, AS_OF)


def test_a_metadata_date_that_is_not_a_day_is_refused(store: ParquetFactorStore) -> None:
    path = store.write(make_panel())
    rewrite(path, with_metadata(pq.read_table(str(path)), as_of_date="31/12/2024"))
    with pytest.raises(FactorStoreError, match="as_of_date"):
        store.read(NAME, VERSION, AS_OF)


@pytest.mark.parametrize(
    "universe",
    ["not json", "{}", "[]", "[1, 2]", '["AAPL", "AAPL", "MSFT"]', '["AAPL", ""]'],
)
def test_a_malformed_universe_in_the_metadata_is_refused(
    store: ParquetFactorStore, universe: str
) -> None:
    path = store.write(make_panel())
    rewrite(path, with_metadata(pq.read_table(str(path)), universe=universe))
    with pytest.raises(FactorStoreError, match="universe"):
        store.read(NAME, VERSION, AS_OF)


def test_a_table_whose_name_disagrees_with_the_metadata_is_refused(
    store: ParquetFactorStore,
) -> None:
    path = store.write(make_panel())
    table = pq.read_table(str(path))
    frame = table.to_pandas()
    frame["factor_name"] = "value_quality"
    rewrite(
        path,
        pa.Table.from_pandas(frame, preserve_index=False).replace_schema_metadata(
            table.schema.metadata
        ),
    )
    with pytest.raises(FactorStoreError, match="factor_name"):
        store.read(NAME, VERSION, AS_OF)


def test_a_table_whose_version_disagrees_with_the_metadata_is_refused(
    store: ParquetFactorStore,
) -> None:
    path = store.write(make_panel())
    table = pq.read_table(str(path))
    frame = table.to_pandas()
    frame["factor_version"] = "9.9.9"
    rewrite(
        path,
        pa.Table.from_pandas(frame, preserve_index=False).replace_schema_metadata(
            table.schema.metadata
        ),
    )
    with pytest.raises(FactorStoreError, match="factor_version"):
        store.read(NAME, VERSION, AS_OF)


def test_a_table_whose_date_disagrees_with_the_metadata_is_refused(
    store: ParquetFactorStore,
) -> None:
    path = store.write(make_panel())
    table = pq.read_table(str(path))
    frame = table.to_pandas()
    frame["as_of_date"] = pd.Timestamp("2024-12-30")
    rewrite(
        path,
        pa.Table.from_pandas(frame, preserve_index=False).replace_schema_metadata(
            table.schema.metadata
        ),
    )
    with pytest.raises(FactorStoreError, match="dated"):
        store.read(NAME, VERSION, AS_OF)


def test_a_table_with_missing_dates_is_refused(store: ParquetFactorStore) -> None:
    path = store.write(make_panel())
    table = pq.read_table(str(path))
    frame = table.to_pandas()
    frame.loc[0, "as_of_date"] = pd.NaT
    rewrite(
        path,
        pa.Table.from_pandas(frame, preserve_index=False).replace_schema_metadata(
            table.schema.metadata
        ),
    )
    with pytest.raises(FactorStoreError, match="as_of_date"):
        store.read(NAME, VERSION, AS_OF)


def test_bytes_that_are_not_parquet_are_refused(store: ParquetFactorStore) -> None:
    path = store.write(make_panel())
    path.write_bytes(b"not a parquet file")
    with pytest.raises(FactorStoreError, match="cannot be read as parquet") as excinfo:
        store.read(NAME, VERSION, AS_OF)
    assert excinfo.value.__cause__ is not None


def test_an_empty_file_is_refused(store: ParquetFactorStore) -> None:
    path = store.write(make_panel())
    path.write_bytes(b"")
    with pytest.raises(FactorStoreError, match="cannot be read as parquet"):
        store.read(NAME, VERSION, AS_OF)


def test_a_table_with_an_extra_column_is_refused(store: ParquetFactorStore) -> None:
    path = store.write(make_panel())
    table = pq.read_table(str(path)).append_column("extra", pa.array([1, 2, 3]))
    rewrite(path, table)
    with pytest.raises(FactorStoreError, match="panel columns"):
        store.read(NAME, VERSION, AS_OF)


def test_a_table_with_a_missing_row_is_refused(store: ParquetFactorStore) -> None:
    path = store.write(make_panel())
    table = pq.read_table(str(path))
    rewrite(path, table.slice(0, len(SYMBOLS) - 1))
    with pytest.raises(FactorStoreError, match="lists 3 universe symbol"):
        store.read(NAME, VERSION, AS_OF)


def test_a_table_that_cannot_be_a_panel_keeps_the_original_error(
    store: ParquetFactorStore,
) -> None:
    """A table can pass the metadata checks and still fail the panel contract."""
    path = store.write(make_panel())
    table = pq.read_table(str(path))
    frame = table.to_pandas()
    frame.loc[frame.index[-1], "symbol"] = "TSLA"
    rewrite(
        path,
        pa.Table.from_pandas(frame, preserve_index=False).replace_schema_metadata(
            table.schema.metadata
        ),
    )
    with pytest.raises(FactorStoreError, match="does not restore as a FactorPanel") as excinfo:
        store.read(NAME, VERSION, AS_OF)
    assert isinstance(excinfo.value.__cause__, FactorContractError)


# ---------------------------------------------------------------------------
# A failed write never loses the previous panel
# ---------------------------------------------------------------------------
def test_a_failed_write_keeps_the_previous_panel(
    store: ParquetFactorStore, root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store.write(make_panel(values=(0.1, 0.2, 0.3)))

    def refuse(*args: Any, **kwargs: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(store_module.pq, "write_table", refuse)
    with pytest.raises(FactorStoreError, match="could not write") as excinfo:
        store.write(make_panel(values=(9.0, 9.0, 9.0)))
    assert isinstance(excinfo.value.__cause__, OSError)

    assert temporary_files(root) == []
    loaded = store.read(NAME, VERSION, AS_OF)
    assert loaded.frame["raw_value"].tolist() == [0.1, 0.2, 0.3]


def test_a_failed_replace_keeps_the_previous_panel(
    store: ParquetFactorStore, root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store.write(make_panel(values=(0.1, 0.2, 0.3)))

    def refuse(source: str, destination: str) -> None:
        raise OSError("the file is locked")

    monkeypatch.setattr(store_module.os, "replace", refuse)
    with pytest.raises(FactorStoreError, match="could not write"):
        store.write(make_panel(values=(9.0, 9.0, 9.0)))

    assert temporary_files(root) == []
    assert len(stored_files(root)) == 1
    assert store.read(NAME, VERSION, AS_OF).frame["raw_value"].tolist() == [0.1, 0.2, 0.3]


def test_a_failed_first_write_leaves_no_panel(
    store: ParquetFactorStore, root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(*args: Any, **kwargs: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(store_module.pq, "write_table", refuse)
    with pytest.raises(FactorStoreError):
        store.write(make_panel())

    assert stored_files(root) == []
    assert store.exists(NAME, VERSION, AS_OF) is False


def test_a_temporary_file_that_is_not_parquet_is_refused(
    store: ParquetFactorStore, root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store.write(make_panel(values=(0.1, 0.2, 0.3)))

    def write_garbage(table: pa.Table, path: str, **kwargs: Any) -> None:
        Path(path).write_bytes(b"not parquet")

    monkeypatch.setattr(store_module.pq, "write_table", write_garbage)
    with pytest.raises(FactorStoreError, match="could not write"):
        store.write(make_panel(values=(9.0, 9.0, 9.0)))

    assert temporary_files(root) == []
    assert store.read(NAME, VERSION, AS_OF).frame["raw_value"].tolist() == [0.1, 0.2, 0.3]


# ---------------------------------------------------------------------------
# Module boundary: the store is a leaf, not a second pipeline
# ---------------------------------------------------------------------------
STORE_PATH = Path(store_module.__file__)
ALLOWED_MODULES: frozenset[str] = frozenset(
    {
        "__future__",
        "abc",
        "datetime",
        "json",
        "os",
        "pathlib",
        "re",
        "tempfile",
        "typing",
        "urllib.parse",
        "pandas",
        "pyarrow",
        "pyarrow.parquet",
        "data_sys.config",
        "factor_sys.contracts",
        "factor_sys.errors",
    }
)
FORBIDDEN_MODULES: tuple[str, ...] = (
    "data_sys.storage",
    "data_sys.update",
    "data_sys.batch_query",
    "data_sys.providers",
    "data_sys.data_query",
    "data_sys.summary",
    "duckdb",
    "yfinance",
    "httpx",
    "requests",
    "socket",
    "urllib.request",
    "factors",
    "factor_sys.runner",
    "factor_sys.market_data",
    "factor_sys.universe",
)


def store_tree() -> ast.Module:
    """The parsed source of the module under test."""
    return ast.parse(STORE_PATH.read_text(encoding="utf-8"), filename=str(STORE_PATH))


def imported_modules(tree: ast.Module) -> set[str]:
    """Every module the source imports, however it spells the import."""
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def code_strings(tree: ast.Module) -> list[str]:
    """Every string literal that is code, not a docstring."""
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                docstrings.add(id(body[0].value))
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
    ]


def test_the_store_imports_only_its_own_layer() -> None:
    imported = imported_modules(store_tree())
    assert imported <= ALLOWED_MODULES
    assert not [module for module in imported if module.startswith(FORBIDDEN_MODULES)]


def test_the_store_never_touches_the_network_or_duckdb() -> None:
    names = {
        node.id
        for node in ast.walk(store_tree())
        if isinstance(node, ast.Name) and isinstance(node.id, str)
    }
    names.update(
        node.attr
        for node in ast.walk(store_tree())
        if isinstance(node, ast.Attribute) and isinstance(node.attr, str)
    )
    assert not names & {"urlopen", "connect", "socket", "duckdb", "read_html"}
    assert not [name for name in names if name.lower().startswith("download")]


def test_the_store_hard_codes_no_data_directory() -> None:
    suspicious = [
        text
        for text in code_strings(store_tree())
        if text == "data" or text.startswith(("data/", "data\\"))
    ]
    assert suspicious == []


def test_the_default_root_comes_from_the_pipeline_config() -> None:
    assert "data_sys.config" in imported_modules(store_tree())
    strings = code_strings(store_tree())
    assert "factors" in strings
    assert "panels" in strings


def test_the_store_does_not_compute_or_read_market_data() -> None:
    names = {node.id for node in ast.walk(store_tree()) if isinstance(node, ast.Name)}
    names.update(
        node.attr
        for node in ast.walk(store_tree())
        if isinstance(node, ast.Attribute) and isinstance(node.attr, str)
    )
    assert not names & {"FactorRunner", "factor_runner", "MarketDataReader", "Momentum12_1"}


def test_the_package_root_exports_the_store() -> None:
    assert factor_sys.FactorStore is store_module.FactorStore
    assert factor_sys.ParquetFactorStore is store_module.ParquetFactorStore
    assert factor_sys.FactorStoreError is store_module.FactorStoreError
    for name in ("FactorStore", "ParquetFactorStore", "FactorStoreError"):
        assert name in factor_sys.__all__


def test_the_store_error_belongs_to_the_factor_system() -> None:
    assert issubclass(FactorStoreError, factor_sys.FactorError)


# ---------------------------------------------------------------------------
# End to end: a real factor panel, stored and read back for downstream use
# ---------------------------------------------------------------------------
def test_a_momentum_panel_round_trips_end_to_end(store: ParquetFactorStore, root: Path) -> None:
    panel = momentum_panel()

    path = store.write(panel)
    loaded = store.read("momentum_12_1", "1.0.0", "2024-12-31")

    assert path == (
        root
        / "factor_name=momentum_12_1"
        / "factor_version=1.0.0"
        / "year=2024"
        / "as_of_date=2024-12-31.parquet"
    )
    assert isinstance(loaded, FactorPanel)
    assert loaded.as_of_date == panel.as_of_date
    assert loaded.factor_name == panel.factor_name
    assert loaded.factor_version == panel.factor_version
    assert loaded.universe == panel.universe
    pd.testing.assert_frame_equal(loaded.frame, panel.frame)

    values = dict(zip(loaded.symbols, loaded.frame["raw_value"], strict=True))
    assert values["AAPL"] == pytest.approx(0.30)
    assert values["MSFT"] == pytest.approx(0.0)
    assert values["NVDA"] == pytest.approx(-0.25)
    assert loaded.frame["is_eligible"].all()


def test_a_stored_panel_is_a_valid_panel_for_downstream_use(
    store: ParquetFactorStore,
) -> None:
    store.write(momentum_panel())

    loaded = store.read("momentum_12_1", "1.0.0", AS_OF)

    # A downstream consumer (a transform, an IC analysis, a strategy) may rely on
    # every guarantee of the contract without knowing that the panel came from disk.
    again = FactorPanel(loaded.frame, universe=loaded.universe, as_of_date=loaded.as_of_date)
    pd.testing.assert_frame_equal(again.frame, loaded.frame)

    eligible = loaded.frame.loc[loaded.frame["is_eligible"], "raw_value"]
    assert eligible.notna().all()
    assert len(eligible) == len(SYMBOLS)
    assert eligible.rank(ascending=False).tolist() == [1.0, 2.0, 3.0]


def test_two_sessions_of_one_factor_are_separate_panels(
    store: ParquetFactorStore, root: Path
) -> None:
    first = date(2024, 12, 30)
    second = date(2024, 12, 31)
    store.write(momentum_panel(first))
    store.write(momentum_panel(second))

    assert len(stored_files(root)) == 2
    assert store.read("momentum_12_1", "1.0.0", first).as_of_date == first
    assert store.read("momentum_12_1", "1.0.0", second).as_of_date == second


def test_a_small_batch_of_sessions_can_be_written_and_read_back(
    store: ParquetFactorStore, root: Path
) -> None:
    sessions = [date(2024, 12, day) for day in (23, 24, 26, 27, 30, 31)]

    for session in sessions:
        store.write(momentum_panel(session))

    assert len(stored_files(root)) == len(sessions)
    for session in sessions:
        assert store.exists("momentum_12_1", "1.0.0", session) is True
        assert store.read("momentum_12_1", "1.0.0", session).as_of_date == session


def test_the_whole_store_tree_is_hive_partitioned(store: ParquetFactorStore, root: Path) -> None:
    """A later analysis step can open the tree with PyArrow's Hive partitioning."""
    store.write(make_panel(as_of=date(2024, 12, 30)))
    store.write(make_panel(as_of=date(2024, 12, 31)))
    store.write(make_panel(as_of=date(2023, 6, 15)))

    import pyarrow.dataset as ds

    dataset = ds.dataset(root, format="parquet", partitioning="hive")
    table = dataset.to_table()

    assert table.num_rows == len(SYMBOLS) * 3
    assert set(table.column("year").to_pylist()) == {2023, 2024}
    assert {value.date() for value in table.column("as_of_date").to_pylist()} == {
        date(2023, 6, 15),
        date(2024, 12, 30),
        date(2024, 12, 31),
    }
