"""Tests for the static stock-universe interface (:mod:`factor_sys.universe`).

Everything runs in ``tmp_path`` or in memory: no ``data/`` directory is read and
nothing hits the network.  The only committed file that *is* touched --
``config/universe_seed.csv`` -- is version-controlled configuration (the same way
``tests/test_universe.py`` reads it) and only to prove that the default provider
points at it.

Covered in this order:

* :class:`StaticUniverseProvider` returns the ``is_active=true`` symbols only,
* the pool is static in time and frozen at construction time,
* custom files, broken files, empty pools and bad paths,
* :class:`UniverseSnapshot` validation, normalisation and immutability,
* :class:`UniverseProvider` as the replaceable interface,
* the architectural boundary: no market reads, no storage layer, no second
  ticker rule.
"""

from __future__ import annotations

import re
from dataclasses import FrozenInstanceError
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest

import factor_sys
from data_sys.errors import UniverseError
from data_sys.schema import SYMBOL_PATTERN
from data_sys.universe import DEFAULT_UNIVERSE_PATH, read_universe
from factor_sys import (
    FactorContractError,
    StaticUniverseProvider,
    UniverseProvider,
    UniverseSnapshot,
)
from tests.helpers import write_universe_file

#: The active pool every fixture below writes (12 symbols, well above the
#: data system's minimum universe size).
ACTIVE: tuple[str, ...] = (
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "META",
    "TSLA",
    "AVGO",
    "COST",
    "BRK.B",
    "JPM",
    "V",
)

#: Symbols written with ``is_active=false``: they must never reach a snapshot.
INACTIVE: tuple[str, ...] = ("DEAD", "OLD1", "OTC1")

#: A completely different pool, used to prove a file is read instead of assumed.
OTHER: tuple[str, ...] = tuple(f"ZZ{index:02d}" for index in range(12))

#: The session most tests ask for.
AS_OF: date = date(2024, 1, 5)


# ---------------------------------------------------------------------------
# Helpers and fixtures (temporary CSVs only)
# ---------------------------------------------------------------------------
def as_date(value: object) -> date:
    """The plain ``date`` a provider must echo back for ``value``."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    assert isinstance(value, str), value
    return date.fromisoformat(value)


def inactive_only_file(path: Path) -> Path:
    """A contract-valid universe file whose every row is ``is_active=false``."""
    return write_universe_file(path, [], inactive=ACTIVE)


class SnapshotOnlyProvider(UniverseProvider):
    """A minimal legal implementation: ``snapshot`` is the whole interface."""

    def snapshot(self, as_of_date: date | str) -> UniverseSnapshot:
        return UniverseSnapshot(as_of_date=as_of_date, symbols=("AAPL",), source="memory")


class IncompleteProvider(UniverseProvider):
    """Misses ``snapshot``, so it must stay abstract and uninstantiable."""


class PseudoHistoricalProvider(UniverseProvider):
    """Stand-in for the future point-in-time provider: one symbol per year.

    It exists only to show that the interface -- not the static implementation --
    is what a caller depends on.
    """

    def snapshot(self, as_of_date: date | str) -> UniverseSnapshot:
        year = as_date(as_of_date).year
        return UniverseSnapshot(
            as_of_date=as_of_date,
            symbols=(f"Y{year}",),
            source="pseudo-history",
        )


def pool_for(provider: UniverseProvider, as_of_date: date | str) -> tuple[str, ...]:
    """Ask any provider for its pool: the only call site the system needs."""
    return provider.snapshot(as_of_date).symbols


@pytest.fixture
def pool_file(tmp_path: Path) -> Path:
    """A temporary universe file: 12 active symbols plus 3 inactive ones."""
    return write_universe_file(tmp_path / "universe.csv", ACTIVE, inactive=INACTIVE)


@pytest.fixture
def provider(pool_file: Path) -> StaticUniverseProvider:
    """A static provider built from :func:`pool_file`."""
    return StaticUniverseProvider(pool_file)


# ---------------------------------------------------------------------------
# StaticUniverseProvider -- the active pool
# ---------------------------------------------------------------------------
def test_the_provider_returns_the_active_symbols(provider: StaticUniverseProvider) -> None:
    snapshot = provider.snapshot(AS_OF)

    assert snapshot.symbols == tuple(sorted(ACTIVE))
    assert provider.symbols == tuple(sorted(ACTIVE))
    assert len(provider) == len(ACTIVE) == len(snapshot)


@pytest.mark.parametrize("symbol", INACTIVE)
def test_inactive_symbols_never_reach_a_snapshot(
    provider: StaticUniverseProvider, symbol: str
) -> None:
    snapshot = provider.snapshot(AS_OF)

    assert symbol not in snapshot.symbols
    assert symbol not in provider.symbols
    assert symbol not in snapshot


def test_the_pool_is_ordered_deterministically(provider: StaticUniverseProvider) -> None:
    """The same file always yields the same order, so runs stay comparable."""
    first = provider.snapshot(AS_OF).symbols
    second = provider.snapshot("2010-06-30").symbols

    assert first == tuple(sorted(first))
    assert first == second


def test_the_file_is_only_read_through_the_data_system_reader(pool_file: Path) -> None:
    """The provider agrees with ``read_universe`` and records its path."""
    provider = StaticUniverseProvider(pool_file)

    assert provider.symbols == tuple(read_universe(pool_file).active_symbols)
    assert provider.path == pool_file
    assert provider.source == str(pool_file)
    assert provider.snapshot(AS_OF).source == str(pool_file)


def test_lower_case_symbols_in_the_file_are_canonicalized(tmp_path: Path) -> None:
    """Normalisation is inherited from the data system, not re-implemented."""
    path = write_universe_file(tmp_path / "lower.csv", [s.lower() for s in ACTIVE])

    provider = StaticUniverseProvider(path)

    assert provider.symbols == tuple(sorted(ACTIVE))


def test_the_default_provider_uses_the_committed_seed_file() -> None:
    """Without a path the provider reads the version-controlled seed universe."""
    provider = StaticUniverseProvider()

    assert provider.path == DEFAULT_UNIVERSE_PATH
    assert provider.source == str(DEFAULT_UNIVERSE_PATH)
    assert provider.snapshot(AS_OF).source == str(DEFAULT_UNIVERSE_PATH)
    assert provider.symbols == tuple(read_universe().active_symbols)
    assert "AAPL" in provider.snapshot(AS_OF)


def test_a_custom_universe_path_is_used(tmp_path: Path) -> None:
    """A second file is honoured end to end: pool *and* source change."""
    path = write_universe_file(tmp_path / "custom.csv", OTHER)
    custom = StaticUniverseProvider(path)

    assert custom.symbols == tuple(sorted(OTHER))
    assert custom.snapshot(AS_OF).symbols == tuple(sorted(OTHER))
    assert custom.snapshot(AS_OF).source == str(path)
    assert custom.source != StaticUniverseProvider().source
    assert set(custom.symbols).isdisjoint(ACTIVE)


# ---------------------------------------------------------------------------
# The pool is static in time and frozen at construction
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "as_of_date",
    ["2001-03-05", date(2010, 6, 30), datetime(2024, 1, 5, 21, 30), "2099-12-31"],
)
def test_the_same_symbols_are_returned_for_every_date(
    provider: StaticUniverseProvider, as_of_date: object
) -> None:
    """Dates far outside any listing window still get today's pool."""
    snapshot = provider.snapshot(as_of_date)  # type: ignore[arg-type]

    assert snapshot.symbols == provider.symbols == tuple(sorted(ACTIVE))
    assert snapshot.source == provider.source
    assert snapshot.as_of_date == as_date(as_of_date)


def test_two_dates_differ_only_in_the_date(provider: StaticUniverseProvider) -> None:
    early = provider.snapshot("2001-03-05")
    late = provider.snapshot(AS_OF)

    assert early.as_of_date != late.as_of_date
    assert early.symbols == late.symbols
    assert early.source == late.source
    assert early != late


def test_every_snapshot_is_a_new_object(provider: StaticUniverseProvider) -> None:
    first = provider.snapshot(AS_OF)
    second = provider.snapshot(AS_OF)

    assert first is not second
    assert first == second


def test_the_pool_is_frozen_when_the_provider_is_created(pool_file: Path) -> None:
    """An existing provider is unaffected by a later edit of its file."""
    provider = StaticUniverseProvider(pool_file)
    before = provider.snapshot(AS_OF)

    write_universe_file(pool_file, OTHER, inactive=INACTIVE)

    assert provider.snapshot(AS_OF).symbols == before.symbols == tuple(sorted(ACTIVE))
    assert provider.symbols == tuple(sorted(ACTIVE))
    assert provider.source == before.source == str(pool_file)

    # ... while a provider created afterwards reflects the new file.
    rebuilt = StaticUniverseProvider(pool_file)
    assert rebuilt.symbols == tuple(sorted(OTHER))


def test_the_file_is_not_read_again_on_every_snapshot(pool_file: Path) -> None:
    """Once built, the provider needs no file access at all."""
    provider = StaticUniverseProvider(pool_file)
    pool_file.unlink()

    first = provider.snapshot("2024-01-05")
    second = provider.snapshot("2024-01-05")

    assert first.symbols == second.symbols == tuple(sorted(ACTIVE))


def test_the_provider_repr_names_the_file_and_the_pool_size(pool_file: Path) -> None:
    text = repr(StaticUniverseProvider(pool_file))

    assert "StaticUniverseProvider" in text
    assert "universe.csv" in text
    assert str(len(ACTIVE)) in text


# ---------------------------------------------------------------------------
# Broken configuration is reported as a factor contract error
# ---------------------------------------------------------------------------
def test_a_missing_file_is_a_factor_contract_error(tmp_path: Path) -> None:
    missing = tmp_path / "nope.csv"

    with pytest.raises(FactorContractError) as excinfo:
        StaticUniverseProvider(missing)

    assert str(missing) in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, UniverseError)


def test_a_file_violating_the_universe_contract_is_a_factor_contract_error(
    tmp_path: Path,
) -> None:
    broken = tmp_path / "broken.csv"
    pd.DataFrame({"symbol": list(ACTIVE), "name": list(ACTIVE)}).to_csv(broken, index=False)

    with pytest.raises(FactorContractError) as excinfo:
        StaticUniverseProvider(broken)

    assert str(broken) in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, UniverseError)


def test_a_file_without_any_active_symbol_is_rejected(tmp_path: Path) -> None:
    path = inactive_only_file(tmp_path / "inactive.csv")

    with pytest.raises(FactorContractError) as excinfo:
        StaticUniverseProvider(path)

    message = str(excinfo.value)
    assert "is_active=true" in message
    assert str(path) in message


@pytest.mark.parametrize("path", [123, 4.5, ["universe.csv"]])
def test_a_path_of_the_wrong_type_is_rejected(path: object) -> None:
    with pytest.raises(FactorContractError) as excinfo:
        StaticUniverseProvider(path)  # type: ignore[arg-type]

    assert "universe path" in str(excinfo.value)


# ---------------------------------------------------------------------------
# UniverseSnapshot -- validation and immutability
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("as_of_date", [AS_OF, "2024-01-05", datetime(2024, 1, 5, 21, 30)])
def test_a_snapshot_normalizes_its_fields(as_of_date: object) -> None:
    snapshot = UniverseSnapshot(
        as_of_date=as_of_date,  # type: ignore[arg-type]
        symbols=["MSFT", "AAPL"],
        source="  config/universe_seed.csv  ",
    )

    assert snapshot.as_of_date == AS_OF
    assert isinstance(snapshot.as_of_date, date)
    assert not isinstance(snapshot.as_of_date, datetime)
    assert snapshot.symbols == ("MSFT", "AAPL")
    assert snapshot.source == "config/universe_seed.csv"


def test_the_symbol_pool_is_frozen_against_the_caller() -> None:
    symbols = ["AAPL", "MSFT"]
    snapshot = UniverseSnapshot(as_of_date=AS_OF, symbols=symbols, source="memory")

    symbols.append("NVDA")

    assert snapshot.symbols == ("AAPL", "MSFT")
    assert isinstance(snapshot.symbols, tuple)
    assert not hasattr(snapshot.symbols, "append")
    assert len(snapshot) == 2
    assert "AAPL" in snapshot
    assert "NVDA" not in snapshot


def test_a_snapshot_cannot_be_modified() -> None:
    snapshot = UniverseSnapshot(as_of_date=AS_OF, symbols=("AAPL",), source="memory")

    with pytest.raises(FrozenInstanceError):
        snapshot.symbols = ("MSFT",)  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        snapshot.as_of_date = date(2020, 1, 1)  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        snapshot.source = "other"  # type: ignore[misc]


@pytest.mark.parametrize(
    "symbols",
    [
        (),
        [],
        ("AAPL", "AAPL"),
        ("aapl",),
        ("AA PL",),
        ("A$P",),
        ("a.a",),
        ("",),
        "AAPL",
        42,
        None,
    ],
)
def test_a_snapshot_rejects_an_invalid_symbol_pool(symbols: object) -> None:
    with pytest.raises(FactorContractError) as excinfo:
        UniverseSnapshot(as_of_date=AS_OF, symbols=symbols, source="memory")  # type: ignore[arg-type]

    assert "symbols" in str(excinfo.value)


@pytest.mark.parametrize(
    "as_of_date", ["05/01/2024", "2024-13-40", "", "   ", None, 20240105, object()]
)
def test_a_snapshot_rejects_an_invalid_date(as_of_date: object) -> None:
    with pytest.raises(FactorContractError) as excinfo:
        UniverseSnapshot(as_of_date=as_of_date, symbols=("AAPL",), source="memory")  # type: ignore[arg-type]

    assert "as_of_date" in str(excinfo.value)


@pytest.mark.parametrize("source", ["", "   ", None, 42, ["universe.csv"]])
def test_a_snapshot_rejects_an_empty_source(source: object) -> None:
    with pytest.raises(FactorContractError) as excinfo:
        UniverseSnapshot(as_of_date=AS_OF, symbols=("AAPL",), source=source)  # type: ignore[arg-type]

    assert "source" in str(excinfo.value)


def test_a_rejected_symbol_pool_reports_every_problem() -> None:
    """Like every other contract object, validation collects instead of stopping."""
    with pytest.raises(FactorContractError) as excinfo:
        UniverseSnapshot(as_of_date=AS_OF, symbols=("aapl", "", "A$P"), source="memory")

    problems = excinfo.value.problems
    assert len(problems) == 3
    assert any("aapl" in problem for problem in problems)
    assert any("A$P" in problem for problem in problems)


# ---------------------------------------------------------------------------
# UniverseProvider -- the replaceable interface
# ---------------------------------------------------------------------------
def test_the_universe_provider_is_abstract() -> None:
    assert UniverseProvider.__abstractmethods__ == {"snapshot"}

    with pytest.raises(TypeError):
        UniverseProvider()  # type: ignore[abstract]


def test_an_incomplete_subclass_stays_abstract() -> None:
    with pytest.raises(TypeError):
        IncompleteProvider()  # type: ignore[abstract]


def test_a_snapshot_only_subclass_is_usable() -> None:
    provider = SnapshotOnlyProvider()

    assert isinstance(provider, UniverseProvider)
    assert isinstance(provider.snapshot(AS_OF), UniverseSnapshot)
    assert provider.snapshot(AS_OF).symbols == ("AAPL",)


def test_the_static_provider_implements_the_interface(provider: StaticUniverseProvider) -> None:
    assert issubclass(StaticUniverseProvider, UniverseProvider)
    assert isinstance(provider, UniverseProvider)


def test_a_point_in_time_provider_can_replace_the_static_one(
    provider: StaticUniverseProvider,
) -> None:
    """One call site, two implementations -- the point of the interface."""
    historical = PseudoHistoricalProvider()

    # the static provider answers the same way whatever the date
    assert pool_for(provider, "2015-06-30") == pool_for(provider, "2016-06-30")
    # a date-aware implementation answers differently...
    assert pool_for(historical, "2015-06-30") == ("Y2015",)
    assert pool_for(historical, "2015-06-30") != pool_for(historical, "2016-06-30")
    # ...and both are usable wherever a UniverseProvider is expected
    assert isinstance(historical, UniverseProvider)


def test_the_universe_api_is_exported_from_the_package_root() -> None:
    expected = {"StaticUniverseProvider", "UniverseProvider", "UniverseSnapshot"}

    assert expected <= set(factor_sys.__all__)
    assert factor_sys.UniverseSnapshot is UniverseSnapshot
    assert factor_sys.UniverseProvider is UniverseProvider
    assert factor_sys.StaticUniverseProvider is StaticUniverseProvider


# ---------------------------------------------------------------------------
# Architectural boundary
# ---------------------------------------------------------------------------
#: Source lines that would couple the universe module to storage or market data.
FORBIDDEN_IMPORTS = re.compile(
    r"^\s*(?:import|from)\s+(?:pandas|pyarrow|duckdb|yfinance|httpx|requests|re\b"
    r"|data_sys\.(?:storage|providers|readers|api|data_query|batch_query))\b",
    re.MULTILINE,
)

#: Tokens that would betray a hand-rolled file/format reader.
FORBIDDEN_TOKENS: tuple[str, ...] = (
    "read_csv(",
    "read_parquet",
    "write_parquet",
    "to_parquet",
    "re.compile(",
    "[A-Z0-9",
)


def test_the_universe_module_reads_no_market_data_and_no_storage() -> None:
    """The provider reads configuration; market bars and storage stay elsewhere."""
    module = Path(factor_sys.__file__).parent / "universe.py"
    source = module.read_text(encoding="utf-8")

    match = FORBIDDEN_IMPORTS.search(source)
    assert match is None, f"universe.py must not import {match.group(0).strip()!r}"

    offenders = [token for token in FORBIDDEN_TOKENS if token in source]
    assert not offenders, f"universe.py must not carry its own reader/regex: {offenders}"


def test_the_universe_module_does_not_redeclare_the_ticker_rule() -> None:
    """The ticker rule is imported, never copied: no pattern is defined here."""
    module = Path(factor_sys.__file__).parent / "universe.py"
    source = module.read_text(encoding="utf-8")

    assert re.search(r"^\s*_?\w*PATTERN\s*[:=]", source, re.MULTILINE) is None


def test_the_shared_ticker_rule_is_the_one_that_is_enforced() -> None:
    """A rejection quotes ``data_sys.schema.SYMBOL_PATTERN``, not a local copy."""
    with pytest.raises(FactorContractError) as excinfo:
        UniverseSnapshot(as_of_date=AS_OF, symbols=("AA PL",), source="memory")

    assert SYMBOL_PATTERN in str(excinfo.value)
