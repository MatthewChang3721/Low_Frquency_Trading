"""Tests for reading a *range* of stored factor panels: history that already exists.

``ParquetFactorStore.read_range`` answers "which panels did I already store
between these two days", so these tests only write and read panels in
``tmp_path`` -- no factor is computed, no calendar is consulted and nothing
reads the project's real ``data/`` directory or the network.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
import pytest

import factor_sys
import factor_sys.factor_store as store_module
from factor_sys import FactorPanel, FactorStore, FactorStoreError, ParquetFactorStore

SYMBOLS: tuple[str, ...] = ("AAPL", "MSFT", "NVDA")
NAME: str = "momentum_12_1"
VERSION: str = "1.0.0"


# ---------------------------------------------------------------------------
# Builders and helpers
# ---------------------------------------------------------------------------
def make_panel(
    *,
    name: str = NAME,
    version: str = VERSION,
    as_of: date = date(2024, 12, 31),
) -> FactorPanel:
    """One valid panel built directly: no bars, no formula, no calendar."""
    frame = pd.DataFrame(
        {
            "as_of_date": [pd.Timestamp(as_of)] * len(SYMBOLS),
            "symbol": list(SYMBOLS),
            "factor_name": [name] * len(SYMBOLS),
            "factor_version": [version] * len(SYMBOLS),
            "raw_value": [0.10, float("nan"), 0.30],
            "is_eligible": [True, False, True],
        }
    )
    return FactorPanel(frame, universe=SYMBOLS)


def stored_panels(
    store: ParquetFactorStore, sessions: list[date] | tuple[date, ...], **kwargs: Any
) -> list[date]:
    """Store one panel per session and return the sessions, for readability."""
    for session in sessions:
        store.write(make_panel(as_of=session, **kwargs))
    return list(sessions)


def panel_files(root: Path) -> list[Path]:
    """Every stored panel file, sorted."""
    return sorted(root.rglob(f"*{store_module.PANEL_SUFFIX}"))


def snapshot(root: Path) -> dict[str, str]:
    """Every file and directory under ``root``, with a digest of each file."""
    state: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        state[relative] = "dir" if path.is_dir() else hashlib.sha256(path.read_bytes()).hexdigest()
    return state


def edit_metadata(path: Path, **overrides: str) -> None:
    """Rewrite one stored file with different store metadata (damage, on purpose)."""
    table = pq.read_table(str(path))
    metadata = dict(table.schema.metadata or {})
    for entry, value in overrides.items():
        metadata[f"{store_module.METADATA_PREFIX}{entry}".encode()] = value.encode()
    pq.write_table(table.replace_schema_metadata(metadata), str(path), compression="snappy")


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """A store root that does not exist yet."""
    return tmp_path / "panels"


@pytest.fixture
def store(root: Path) -> ParquetFactorStore:
    """The Parquet store under test, rooted in the test's temporary directory."""
    return ParquetFactorStore(root)


# ---------------------------------------------------------------------------
# The interface
# ---------------------------------------------------------------------------
def test_a_range_read_belongs_to_the_store_interface() -> None:
    assert "read_range" in FactorStore.__abstractmethods__
    parameters = list(inspect.signature(FactorStore.read_range).parameters)
    assert parameters == [
        "self",
        "factor_name",
        "factor_version",
        "start_date",
        "end_date",
    ]


def test_a_minimal_in_memory_store_can_implement_the_range_read() -> None:
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

        def read_range(
            self,
            factor_name: str,
            factor_version: str,
            start_date: date | str,
            end_date: date | str,
        ) -> tuple[FactorPanel, ...]:
            start = start_date if isinstance(start_date, date) else date.fromisoformat(start_date)
            end = end_date if isinstance(end_date, date) else date.fromisoformat(end_date)
            return tuple(
                panel
                for (name, version, day), panel in sorted(
                    self.panels.items(), key=lambda item: item[0][2]
                )
                if (name, version) == (factor_name, factor_version) and start <= day <= end
            )

        def exists(self, factor_name: str, factor_version: str, as_of_date: date | str) -> bool:
            day = as_of_date if isinstance(as_of_date, date) else date.fromisoformat(as_of_date)
            return (factor_name, factor_version, day) in self.panels

        def path_for(self, factor_name: str, factor_version: str, as_of_date: date | str) -> Path:
            return Path(f"memory://{factor_name}/{factor_version}/{as_of_date}")

    memory = InMemoryStore()
    for session in (date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)):
        memory.write(make_panel(as_of=session))

    loaded = memory.read_range(NAME, VERSION, "2024-01-02", "2024-01-03")
    assert [panel.as_of_date for panel in loaded] == [date(2024, 1, 2), date(2024, 1, 3)]
    assert memory.path_for(NAME, VERSION, "2024-01-02") == Path(
        f"memory://{NAME}/{VERSION}/2024-01-02"
    )


# ---------------------------------------------------------------------------
# What a range returns
# ---------------------------------------------------------------------------
def test_a_range_reads_a_history_across_two_years(store: ParquetFactorStore, root: Path) -> None:
    sessions = stored_panels(
        store,
        [
            date(2023, 12, 28),
            date(2023, 12, 29),
            date(2024, 1, 2),
            date(2024, 6, 28),
            date(2024, 12, 31),
            date(2025, 1, 2),
        ],
    )

    panels = store.read_range(NAME, VERSION, "2023-01-01", "2024-12-31")

    assert [panel.as_of_date for panel in panels] == sessions[0:5]
    assert len(panel_files(root)) == len(sessions)


def test_both_ends_of_the_range_are_included(store: ParquetFactorStore) -> None:
    first, middle, last = date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)
    stored_panels(store, [first, middle, last])

    assert [panel.as_of_date for panel in store.read_range(NAME, VERSION, first, last)] == [
        first,
        middle,
        last,
    ]
    assert [panel.as_of_date for panel in store.read_range(NAME, VERSION, middle, middle)] == [
        middle
    ]
    assert store.read_range(NAME, VERSION, last, last)[0].as_of_date == last


def test_the_answer_is_a_tuple_of_independent_panels(store: ParquetFactorStore) -> None:
    sessions = (date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4))
    stored_panels(store, sessions)

    panels = store.read_range(NAME, VERSION, "2024-01-01", "2024-12-31")

    assert isinstance(panels, tuple)
    assert len(panels) == len(sessions)
    assert len({id(panel) for panel in panels}) == len(sessions)
    for panel, session in zip(panels, sessions, strict=True):
        assert isinstance(panel, FactorPanel)
        assert panel.as_of_date == session
        assert panel.factor_name == NAME
        assert panel.factor_version == VERSION
        assert panel.universe == SYMBOLS
        assert list(panel.frame["symbol"]) == list(SYMBOLS)
        pd.testing.assert_frame_equal(panel.frame, make_panel(as_of=session).frame)

    # Each element is a fresh panel whose ``frame`` is a copy: touching it
    # reaches neither the store, nor another element, nor a later read.
    scratch = panels[0].frame
    scratch.loc[:, "raw_value"] = 999.0
    assert panels[0].frame["raw_value"].iloc[0] == pytest.approx(0.10)
    again = store.read(NAME, VERSION, sessions[0])
    pd.testing.assert_frame_equal(again.frame, make_panel(as_of=sessions[0]).frame)

    # A later read builds new panels again, never the ones handed out before.
    second_read = store.read_range(NAME, VERSION, "2024-01-01", "2024-12-31")
    assert len({id(panel) for panel in second_read} & {id(panel) for panel in panels}) == 0


def test_a_date_hole_returns_only_the_stored_days(store: ParquetFactorStore) -> None:
    stored = stored_panels(
        store,
        [date(2024, 1, 2), date(2024, 1, 5), date(2024, 1, 9)],
    )

    panels = store.read_range(NAME, VERSION, "2024-01-01", "2024-01-31")

    assert [panel.as_of_date for panel in panels] == stored
    # Business days inside the window without a stored panel are simply absent.
    dates = {panel.as_of_date for panel in panels}
    assert date(2024, 1, 3) not in dates
    assert date(2024, 1, 8) not in dates
    assert len(panels) == 3


def test_a_legal_range_that_holds_nothing_is_an_empty_tuple(
    store: ParquetFactorStore, root: Path
) -> None:
    assert store.read_range(NAME, VERSION, "2024-01-01", "2024-12-31") == ()
    assert not root.exists()

    stored_panels(store, [date(2024, 1, 2)])

    assert store.read_range(NAME, VERSION, "2020-01-01", "2020-12-31") == ()
    assert store.read_range("value_rank", VERSION, "2024-01-01", "2024-12-31") == ()
    assert store.read_range(NAME, "9.9.9", "2024-01-01", "2024-12-31") == ()


def test_the_order_comes_from_the_dates_not_from_the_directory(store: ParquetFactorStore) -> None:
    # Written out of order on purpose: 2024, then 2023, then 2025.
    stored_panels(store, [date(2024, 5, 1), date(2023, 5, 1), date(2025, 5, 1)])

    panels = store.read_range(NAME, VERSION, "2023-01-01", "2025-12-31")

    assert [panel.as_of_date for panel in panels] == [
        date(2023, 5, 1),
        date(2024, 5, 1),
        date(2025, 5, 1),
    ]


# ---------------------------------------------------------------------------
# Request validation: the same rules as the single-panel key
# ---------------------------------------------------------------------------
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
        pd.Timestamp("2024-12-31"),
    ],
)
def test_an_illegal_bound_is_reported_as_the_bound_it_is(
    store: ParquetFactorStore, value: Any
) -> None:
    with pytest.raises(FactorStoreError, match="start_date"):
        store.read_range(NAME, VERSION, value, "2024-12-31")
    with pytest.raises(FactorStoreError, match="end_date"):
        store.read_range(NAME, VERSION, "2024-01-01", value)


def test_both_bounds_accept_dates_and_strict_iso_strings(store: ParquetFactorStore) -> None:
    session = date(2024, 1, 2)
    stored_panels(store, [session])

    assert store.read_range(NAME, VERSION, session, session)[0].as_of_date == session
    assert store.read_range(NAME, VERSION, "2024-01-02", "2024-01-02")[0].as_of_date == session


def test_a_backwards_range_is_refused(store: ParquetFactorStore, root: Path) -> None:
    with pytest.raises(FactorStoreError, match="after end_date"):
        store.read_range(NAME, VERSION, "2024-12-31", "2024-01-01")
    assert not root.exists()


@pytest.mark.parametrize(
    "name",
    ["Momentum12_1", "momentum 12", "momentum-12", "", "momentum/12", None, 12],
)
def test_an_illegal_factor_name_is_refused_for_a_range(
    store: ParquetFactorStore, name: Any
) -> None:
    with pytest.raises(FactorStoreError, match="factor_name"):
        store.read_range(name, VERSION, "2024-01-01", "2024-12-31")


@pytest.mark.parametrize("version", ["", "   ", None, 1, 3.5, b"1.0.0", ".", ".."])
def test_an_illegal_factor_version_is_refused_for_a_range(
    store: ParquetFactorStore, version: Any
) -> None:
    with pytest.raises(FactorStoreError, match="factor_version|cannot name a directory"):
        store.read_range(NAME, version, "2024-01-01", "2024-12-31")


@pytest.mark.parametrize("requested", ["2024-01-02", date(2024, 1, 2)])
def test_a_range_can_be_asked_for_with_a_date_or_a_string(
    store: ParquetFactorStore, requested: date | str
) -> None:
    stored_panels(store, [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)])

    panels = store.read_range(NAME, VERSION, requested, requested)

    assert [panel.as_of_date for panel in panels] == [date(2024, 1, 2)]


# ---------------------------------------------------------------------------
# One factor, one version: nothing else may leak into the answer
# ---------------------------------------------------------------------------
def test_another_name_or_version_is_never_mixed_in(store: ParquetFactorStore, root: Path) -> None:
    sessions = [date(2024, 1, 2), date(2024, 1, 3)]
    stored_panels(store, sessions)
    stored_panels(store, sessions, name="value_rank")
    stored_panels(store, sessions, version="1.1.0")

    assert len(panel_files(root)) == len(sessions) * 3

    ours = store.read_range(NAME, VERSION, "2024-01-01", "2024-12-31")
    other_name = store.read_range("value_rank", VERSION, "2024-01-01", "2024-12-31")
    other_version = store.read_range(NAME, "1.1.0", "2024-01-01", "2024-12-31")

    assert [(panel.factor_name, panel.factor_version) for panel in ours] == [
        (NAME, VERSION),
        (NAME, VERSION),
    ]
    assert [(panel.factor_name, panel.factor_version) for panel in other_name] == [
        ("value_rank", VERSION),
        ("value_rank", VERSION),
    ]
    assert [(panel.factor_name, panel.factor_version) for panel in other_version] == [
        (NAME, "1.1.0"),
        (NAME, "1.1.0"),
    ]
    assert [panel.as_of_date for panel in ours] == sessions


@pytest.mark.parametrize(
    "version",
    ["1.0.0+build", "a/b", "with space", "CON", "version..", "1%2E0%2E0"],
)
def test_a_special_version_still_reads_its_own_range(tmp_path: Path, version: str) -> None:
    root = tmp_path / "panels"
    store = ParquetFactorStore(root)
    sessions = [date(2024, 1, 2), date(2024, 1, 3)]
    stored_panels(store, sessions, version=version)

    panels = store.read_range(NAME, version, "2024-01-01", "2024-01-31")

    assert [panel.as_of_date for panel in panels] == sessions
    assert {panel.factor_version for panel in panels} == {version}
    for panel in panels:
        assert panel.universe == SYMBOLS
    for path in panel_files(root):
        assert path.resolve().is_relative_to(root.resolve())


# ---------------------------------------------------------------------------
# A damaged history fails the read instead of shrinking it
# ---------------------------------------------------------------------------
def test_a_corrupt_panel_fails_the_whole_range(store: ParquetFactorStore, root: Path) -> None:
    stored_panels(store, [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)])
    broken = panel_files(root)[1]
    broken.write_bytes(b"not parquet at all")

    with pytest.raises(FactorStoreError, match="cannot be read as parquet") as caught:
        store.read_range(NAME, VERSION, "2024-01-01", "2024-12-31")

    assert broken.name in str(caught.value)


def test_an_empty_panel_file_fails_the_whole_range(store: ParquetFactorStore, root: Path) -> None:
    stored_panels(store, [date(2024, 1, 2), date(2024, 1, 3)])
    panel_files(root)[0].write_bytes(b"")

    with pytest.raises(FactorStoreError):
        store.read_range(NAME, VERSION, "2024-01-01", "2024-12-31")


@pytest.mark.parametrize(
    ("entry", "value"),
    [
        ("factor_name", "value_rank"),
        ("factor_version", "9.9.9"),
        ("as_of_date", "2024-01-04"),
    ],
)
def test_metadata_that_names_another_key_fails_the_range(
    store: ParquetFactorStore, root: Path, entry: str, value: str
) -> None:
    stored_panels(store, [date(2024, 1, 2), date(2024, 1, 3)])
    damaged = panel_files(root)[1]
    edit_metadata(damaged, **{entry: value})

    with pytest.raises(FactorStoreError, match="belongs to another key") as caught:
        store.read_range(NAME, VERSION, "2024-01-01", "2024-12-31")

    assert damaged.name in str(caught.value)


def test_a_file_without_store_metadata_fails_the_range(
    store: ParquetFactorStore, root: Path
) -> None:
    stored_panels(store, [date(2024, 1, 2)])
    damaged = panel_files(root)[0]
    table = pq.read_table(str(damaged)).replace_schema_metadata(None)
    pq.write_table(table, str(damaged), compression="snappy")

    with pytest.raises(FactorStoreError, match="no parquet schema metadata"):
        store.read_range(NAME, VERSION, "2024-01-01", "2024-12-31")


@pytest.mark.parametrize(
    "file_name",
    ["momentum.parquet", "as_of_date=20240203.parquet", "as_of_date=2024-1-3.parquet"],
)
def test_a_parquet_file_that_is_not_named_as_a_panel_fails_the_range(
    store: ParquetFactorStore, root: Path, file_name: str
) -> None:
    stored_panels(store, [date(2024, 1, 2)])
    source = panel_files(root)[0]
    (source.parent / file_name).write_bytes(source.read_bytes())

    with pytest.raises(FactorStoreError, match="is not a stored factor panel file"):
        store.read_range(NAME, VERSION, "2024-01-01", "2024-12-31")


def test_a_file_name_with_an_impossible_day_fails_the_range(
    store: ParquetFactorStore, root: Path
) -> None:
    stored_panels(store, [date(2024, 1, 2)])
    source = panel_files(root)[0]
    (source.parent / "as_of_date=2024-02-30.parquet").write_bytes(source.read_bytes())

    with pytest.raises(FactorStoreError, match="naming a real day"):
        store.read_range(NAME, VERSION, "2024-01-01", "2024-12-31")


def test_a_year_partition_that_disagrees_with_its_panel_fails_the_range(
    store: ParquetFactorStore, root: Path
) -> None:
    stored_panels(store, [date(2024, 1, 2)])
    written = panel_files(root)[0]
    wrong_year = written.parents[1] / "year=2023"
    wrong_year.mkdir()
    written.replace(wrong_year / written.name)

    with pytest.raises(FactorStoreError, match="partition and the panel disagree"):
        store.read_range(NAME, VERSION, "2024-01-01", "2024-12-31")


def test_a_panel_file_outside_a_year_partition_fails_the_range(
    store: ParquetFactorStore, root: Path
) -> None:
    stored_panels(store, [date(2024, 1, 2)])
    written = panel_files(root)[0]
    (written.parents[1] / written.name).write_bytes(written.read_bytes())

    with pytest.raises(FactorStoreError, match="outside a 'year=YYYY' partition"):
        store.read_range(NAME, VERSION, "2024-01-01", "2024-12-31")


def test_a_directory_where_a_panel_belongs_fails_the_range(
    store: ParquetFactorStore, root: Path
) -> None:
    stored_panels(store, [date(2024, 1, 2)])
    written = panel_files(root)[0]
    (written.parent / "as_of_date=2024-01-03.parquet").mkdir()

    with pytest.raises(FactorStoreError, match="is a directory"):
        store.read_range(NAME, VERSION, "2024-01-01", "2024-12-31")


def test_a_directory_that_is_not_a_year_partition_fails_the_range(
    store: ParquetFactorStore, root: Path
) -> None:
    stored_panels(store, [date(2024, 1, 2)])
    written = panel_files(root)[0]
    (written.parents[1] / "scratch").mkdir()

    with pytest.raises(FactorStoreError, match="is not a factor-store year partition"):
        store.read_range(NAME, VERSION, "2024-01-01", "2024-12-31")


def test_a_panel_outside_the_range_is_named_but_never_read(
    store: ParquetFactorStore, root: Path
) -> None:
    stored_panels(store, [date(2023, 1, 3), date(2024, 1, 3)])
    old = panel_files(root)[0]
    assert old.parent.name == "year=2023"
    old.write_bytes(b"not parquet at all")

    panels = store.read_range(NAME, VERSION, "2024-01-01", "2024-12-31")

    # The whole version directory is checked for names and partitions, but a
    # panel from outside the window is not read: an old damaged file cannot
    # block a newer range, and it is not silently dropped from its own range.
    assert [panel.as_of_date for panel in panels] == [date(2024, 1, 3)]
    with pytest.raises(FactorStoreError, match="cannot be read as parquet"):
        store.read_range(NAME, VERSION, "2023-01-01", "2023-12-31")


def test_a_temporary_file_left_by_an_interrupted_write_is_ignored(
    store: ParquetFactorStore, root: Path
) -> None:
    sessions = [date(2024, 1, 2), date(2024, 1, 3)]
    stored_panels(store, sessions)
    written = panel_files(root)[0]
    (written.parent / f".{written.name}.deadbeef.tmp").write_bytes(b"half a panel")
    (written.parent / "notes.txt").write_text("not a panel")

    panels = store.read_range(NAME, VERSION, "2024-01-01", "2024-01-31")

    assert [panel.as_of_date for panel in panels] == sessions


# ---------------------------------------------------------------------------
# A read is a read: the tree is never touched
# ---------------------------------------------------------------------------
def test_a_range_read_writes_changes_and_deletes_nothing(
    store: ParquetFactorStore, root: Path
) -> None:
    stored_panels(store, [date(2024, 1, 2), date(2023, 12, 29), date(2024, 1, 3)])
    before = snapshot(root)

    store.read_range(NAME, VERSION, "2023-01-01", "2024-12-31")
    store.read_range(NAME, VERSION, "2020-01-01", "2020-12-31")

    assert snapshot(root) == before


def test_a_failed_range_read_leaves_the_tree_untouched(
    store: ParquetFactorStore, root: Path
) -> None:
    stored_panels(store, [date(2024, 1, 2), date(2024, 1, 3)])
    panel_files(root)[0].write_bytes(b"not parquet at all")
    before = snapshot(root)

    with pytest.raises(FactorStoreError):
        store.read_range(NAME, VERSION, "2024-01-01", "2024-12-31")

    assert snapshot(root) == before


def test_a_root_that_became_a_file_is_refused_by_a_range_read(
    store: ParquetFactorStore, root: Path
) -> None:
    root.write_text("x")

    with pytest.raises(FactorStoreError, match="is not a directory"):
        store.read_range(NAME, VERSION, "2024-01-01", "2024-12-31")


# ---------------------------------------------------------------------------
# Module boundary: a range read is still only a read
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
    "factor_sys.registry",
    "factor_sys.quality",
    "factor_sys.universe",
)
FORBIDDEN_NAMES: frozenset[str] = frozenset(
    {
        "FactorRunner",
        "factor_runner",
        "FactorRegistry",
        "default_factor_registry",
        "FactorQualityChecker",
        "DefaultFactorQualityChecker",
        "MarketDataReader",
        "ParquetMarketDataReader",
        "Momentum12_1",
        "duckdb",
        "urlopen",
        "connect",
        "socket",
        "read_html",
    }
)
FORBIDDEN_IN_A_READ: frozenset[str] = frozenset(
    {
        "write",
        "_atomic_write",
        "write_table",
        "write_bytes",
        "write_text",
        "unlink",
        "mkdir",
        "rmdir",
        "replace",
        "rename",
    }
)


def store_tree() -> ast.Module:
    """The parsed source of the store module."""
    return ast.parse(STORE_PATH.read_text(encoding="utf-8"), filename=str(STORE_PATH))


def read_range_function(tree: ast.Module) -> ast.FunctionDef:
    """The ``read_range`` implementation of the Parquet store."""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "read_range":
            return node
    raise AssertionError("read_range is missing from the factor store")


def imported_modules(tree: ast.Module) -> set[str]:
    """Every module the source imports, however it spells the import."""
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def referenced_names(node: ast.AST) -> set[str]:
    """Every plain name and attribute the code mentions."""
    names = {child.id for child in ast.walk(node) if isinstance(child, ast.Name)}
    names.update(child.attr for child in ast.walk(node) if isinstance(child, ast.Attribute))
    return names


def test_the_range_reader_imports_only_its_own_layer() -> None:
    imported = imported_modules(store_tree())
    assert imported <= ALLOWED_MODULES
    assert not [module for module in imported if module.startswith(FORBIDDEN_MODULES)]


def test_the_range_reader_neither_computes_nor_reads_market_data() -> None:
    assert not referenced_names(read_range_function(store_tree())) & FORBIDDEN_NAMES


def test_the_range_reader_only_reads_files() -> None:
    names = referenced_names(read_range_function(store_tree()))
    assert not names & FORBIDDEN_IN_A_READ


# ---------------------------------------------------------------------------
# The package root and the single-panel operations do not regress
# ---------------------------------------------------------------------------
def test_the_package_root_and_the_single_panel_operations_are_unchanged(
    store: ParquetFactorStore,
) -> None:
    assert factor_sys.FactorStore is store_module.FactorStore
    assert factor_sys.ParquetFactorStore is store_module.ParquetFactorStore
    assert factor_sys.FactorStoreError is store_module.FactorStoreError
    for name in ("FactorStore", "ParquetFactorStore", "FactorStoreError"):
        assert name in factor_sys.__all__

    session = date(2024, 1, 2)
    written = store.write(make_panel(as_of=session))
    assert written.is_file()
    assert store.exists(NAME, VERSION, session) is True
    assert store.exists(NAME, VERSION, date(2024, 1, 3)) is False

    loaded = store.read(NAME, VERSION, session)
    pd.testing.assert_frame_equal(loaded.frame, make_panel(as_of=session).frame)

    with pytest.raises(FactorStoreError, match="does not exist"):
        store.read(NAME, VERSION, date(2024, 1, 3))

    assert store.read_range(NAME, VERSION, session, session)[0].as_of_date == session
