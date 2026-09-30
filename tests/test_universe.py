"""Tests for the security master and the committed seed universe.

Two groups:

* **seed-file tests** - assert the properties the committed
  ``config/universe_seed.csv`` must always have.  These fail if someone edits the
  file into an invalid state, which is exactly the point of version-controlling it.
* **reader tests** - feed deliberately broken files through
  :func:`data_sys.universe.read_universe` and check that every problem is reported.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from data_sys.errors import UniverseError
from data_sys.schema import SYMBOL_PATTERN
from data_sys.universe import (
    ALLOWED_EXCHANGES,
    ALLOWED_SECURITY_TYPES,
    DEFAULT_UNIVERSE_PATH,
    MAX_SYMBOL_LENGTH,
    MIN_UNIVERSE_SIZE,
    REQUIRED_COLUMNS,
    Universe,
    load_universe_symbols,
    read_universe,
    validate_universe,
)

#: Realistic tickers reused by the in-memory fixtures.
SYMBOL_POOL: list[str] = [
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
    "MA",
    "UNH",
    "JNJ",
    "LLY",
    "XOM",
    "PG",
    "HD",
    "WMT",
]


def write_universe(path: Path, records: list[dict[str, str]]) -> Path:
    """Write ``records`` to ``path`` as a universe CSV and return the path."""
    pd.DataFrame(records).to_csv(path, index=False)
    return path


def valid_records(count: int = MIN_UNIVERSE_SIZE) -> list[dict[str, str]]:
    """A valid universe of ``count`` rows (defaults to exactly the minimum)."""
    return [
        {
            "symbol": symbol,
            "name": f"{symbol} Holdings Inc.",
            "exchange": "NASDAQ",
            "security_type": "common_stock",
            "is_active": "true",
        }
        for symbol in SYMBOL_POOL[:count]
    ]


@pytest.fixture
def universe_csv(tmp_path: Path) -> Path:
    """A temporary valid universe file with exactly ``MIN_UNIVERSE_SIZE`` rows."""
    return write_universe(tmp_path / "universe.csv", valid_records())


@pytest.fixture
def seed_frame() -> pd.DataFrame:
    """The committed seed file read as raw text (no type coercion)."""
    return pd.read_csv(DEFAULT_UNIVERSE_PATH, dtype=str, keep_default_na=False)


# ---------------------------------------------------------------------------
# The committed seed file
# ---------------------------------------------------------------------------
def test_seed_file_is_committed_at_the_expected_path() -> None:
    assert DEFAULT_UNIVERSE_PATH.name == "universe_seed.csv"
    assert DEFAULT_UNIVERSE_PATH.parent.name == "config"
    assert DEFAULT_UNIVERSE_PATH.exists(), (
        f"the seed universe must be committed: {DEFAULT_UNIVERSE_PATH}"
    )


def test_min_universe_size_is_ten() -> None:
    """The requirement is 'at least 10 stocks'; guard the constant itself."""
    assert MIN_UNIVERSE_SIZE == 10


def test_seed_has_all_required_fields(seed_frame: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in seed_frame.columns]
    assert not missing, f"seed file is missing {missing}"
    assert len(seed_frame) >= MIN_UNIVERSE_SIZE
    assert len(seed_frame) >= 10


def test_seed_has_no_blank_required_values(seed_frame: pd.DataFrame) -> None:
    for column in REQUIRED_COLUMNS:
        blanks = seed_frame.index[seed_frame[column].astype(str).str.strip() == ""]
        assert blanks.empty, f"blank {column!r} at data row(s) {[int(i) + 1 for i in blanks]}"


def test_seed_symbols_are_upper_case(seed_frame: pd.DataFrame) -> None:
    offenders = seed_frame.loc[seed_frame["symbol"] != seed_frame["symbol"].str.upper()]
    assert offenders.empty, f"symbols must be upper-case: {offenders['symbol'].tolist()}"


def test_seed_symbols_are_unique(seed_frame: pd.DataFrame) -> None:
    duplicated = seed_frame["symbol"][seed_frame["symbol"].duplicated(keep=False)]
    assert duplicated.empty, f"duplicate symbols: {sorted(duplicated.unique())}"


def test_seed_symbols_match_the_shared_ticker_pattern(seed_frame: pd.DataFrame) -> None:
    pattern = re.compile(SYMBOL_PATTERN)
    offenders = [
        symbol for symbol in seed_frame["symbol"] if not pattern.match(symbol)
    ]
    assert not offenders, f"symbols violate {SYMBOL_PATTERN}: {offenders}"


def test_seed_security_types_are_all_common_stock(seed_frame: pd.DataFrame) -> None:
    found = set(seed_frame["security_type"])
    assert found == {"common_stock"}
    assert found <= ALLOWED_SECURITY_TYPES


def test_seed_exchanges_are_known_us_venues(seed_frame: pd.DataFrame) -> None:
    found = set(seed_frame["exchange"])
    assert found <= ALLOWED_EXCHANGES, f"unknown exchanges: {sorted(found - ALLOWED_EXCHANGES)}"
    assert found, "the seed file must record at least one exchange"


def test_seed_is_active_values_are_booleans(seed_frame: pd.DataFrame) -> None:
    readable = seed_frame["is_active"].astype(str).str.strip().str.lower()
    assert set(readable) <= {"true", "false", "yes", "no", "1", "0"}


def test_seed_contains_a_multi_class_ticker(seed_frame: pd.DataFrame) -> None:
    """``BRK.B`` proves dot-separated share classes survive the symbol pattern."""
    assert "BRK.B" in set(seed_frame["symbol"])


# ---------------------------------------------------------------------------
# Public API against the committed seed file
# ---------------------------------------------------------------------------
def test_read_universe_returns_every_seed_entry() -> None:
    universe = read_universe()

    assert isinstance(universe, Universe)
    assert len(universe) >= MIN_UNIVERSE_SIZE
    assert universe.source == str(DEFAULT_UNIVERSE_PATH)
    assert len(universe.symbols) == len(universe)


def test_read_universe_normalizes_entries() -> None:
    for entry in read_universe():
        assert entry.symbol == entry.symbol.upper()
        assert entry.name == entry.name.strip()
        assert entry.exchange == entry.exchange.upper()
        assert entry.security_type in ALLOWED_SECURITY_TYPES
        assert isinstance(entry.is_active, bool)


def test_load_universe_symbols_is_sorted_unique_and_upper_case() -> None:
    """The acceptance criterion: normalized, de-duplicated symbols."""
    symbols = load_universe_symbols()

    assert isinstance(symbols, list)
    assert len(symbols) >= 10
    assert symbols == sorted(symbols)
    assert len(symbols) == len(set(symbols))
    assert all(symbol == symbol.upper() for symbol in symbols)


def test_universe_symbols_and_active_symbols_agree_when_everything_is_active() -> None:
    universe = read_universe()

    assert universe.symbols == universe.active_symbols
    assert load_universe_symbols(active_only=False) == universe.symbols


def test_by_symbol_lookup_exposes_the_security_master() -> None:
    entry = read_universe().by_symbol()["AAPL"]

    assert entry.name
    assert entry.exchange in ALLOWED_EXCHANGES
    assert entry.security_type == "common_stock"


def test_to_frame_round_trips_the_contract_columns() -> None:
    universe = read_universe()
    frame = universe.to_frame()

    assert list(frame.columns) == list(REQUIRED_COLUMNS)
    assert len(frame) == len(universe)
    assert frame["is_active"].dtype == bool
    assert set(frame["symbol"]) == set(universe.symbols)


def test_empty_universe_to_frame_is_still_typed() -> None:
    frame = Universe(entries=()).to_frame()

    assert list(frame.columns) == list(REQUIRED_COLUMNS)
    assert len(frame) == 0


# ---------------------------------------------------------------------------
# Malformed files must fail loudly
# ---------------------------------------------------------------------------
def write_raw_universe(path: Path, text: str) -> Path:
    """Write ``text`` verbatim, for cases a DataFrame cannot express."""
    path.write_text(text, encoding="utf-8")
    return path


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(UniverseError, match="not found"):
        read_universe(tmp_path / "does_not_exist.csv")


def test_directory_instead_of_file_raises(tmp_path: Path) -> None:
    target = tmp_path / "config"
    target.mkdir()

    with pytest.raises(UniverseError, match="directory"):
        read_universe(target)


def test_completely_empty_file_raises(tmp_path: Path) -> None:
    target = write_raw_universe(tmp_path / "empty.csv", "")

    with pytest.raises(UniverseError, match="empty"):
        read_universe(target)


def test_header_only_file_raises(tmp_path: Path) -> None:
    target = write_raw_universe(tmp_path / "header_only.csv", ",".join(REQUIRED_COLUMNS) + "\n")

    with pytest.raises(UniverseError) as exc:
        read_universe(target)

    assert "no data rows" in str(exc.value)


@pytest.mark.parametrize("column", REQUIRED_COLUMNS)
def test_missing_required_column_raises(tmp_path: Path, column: str) -> None:
    records = valid_records()
    for record in records:
        record.pop(column)
    target = write_universe(tmp_path / "missing_column.csv", records)

    with pytest.raises(UniverseError) as exc:
        read_universe(target)

    assert any("missing required column" in problem for problem in exc.value.problems)
    assert any(column in problem for problem in exc.value.problems)


def test_missing_columns_short_circuit_value_checks() -> None:
    """Without the required headers, value messages would only be noise."""
    frame = pd.DataFrame([{"symbol": "aapl", "name": ""}])

    problems = validate_universe(frame)

    assert len(problems) == 1
    assert "missing required column" in problems[0]


def test_duplicate_header_raises(tmp_path: Path) -> None:
    target = write_raw_universe(
        tmp_path / "dup_header.csv",
        "symbol,name,exchange,symbol,is_active\nAAPL,Apple,NASDAQ,AAPL,true\n",
    )

    with pytest.raises(UniverseError) as exc:
        read_universe(target)

    assert any("duplicate header" in problem for problem in exc.value.problems)


def test_duplicate_symbol_raises(tmp_path: Path) -> None:
    records = valid_records()
    records[1]["symbol"] = records[0]["symbol"]
    target = write_universe(tmp_path / "duplicate.csv", records)

    with pytest.raises(UniverseError) as exc:
        read_universe(target)

    problems = exc.value.problems
    assert any("duplicate symbol 'AAPL'" in problem for problem in problems)
    assert any("data row 1" in problem and "data row 2" in problem for problem in problems)


def test_case_insensitive_duplicate_raises(tmp_path: Path) -> None:
    """Uniqueness is checked *after* normalization, so 'aapl' collides with 'AAPL'."""
    records = valid_records()
    records[1]["symbol"] = records[0]["symbol"].lower()
    target = write_universe(tmp_path / "duplicate_case.csv", records)

    with pytest.raises(UniverseError) as exc:
        read_universe(target)

    assert any("duplicate symbol 'AAPL'" in problem for problem in exc.value.problems)


def test_duplicate_reporting_ignores_already_reported_blanks(tmp_path: Path) -> None:
    records = valid_records()
    records[0]["symbol"] = ""
    records[1]["symbol"] = ""
    target = write_universe(tmp_path / "blank.csv", records)

    with pytest.raises(UniverseError) as exc:
        read_universe(target)

    problems = exc.value.problems
    assert sum("'symbol' must not be empty" in problem for problem in problems) == 2
    assert not any("duplicate symbol" in problem for problem in problems)


@pytest.mark.parametrize("bad_symbol", ["AA PL", "AA/PL", "AAPL!", "AAA$", "A:A"])
def test_invalid_symbol_format_raises(tmp_path: Path, bad_symbol: str) -> None:
    records = valid_records()
    records[0]["symbol"] = bad_symbol
    target = write_universe(tmp_path / "bad_symbol.csv", records)

    with pytest.raises(UniverseError) as exc:
        read_universe(target)

    assert any("does not match" in problem for problem in exc.value.problems)


@pytest.mark.parametrize("bad_symbol", ["", "   "])
def test_blank_symbol_raises(tmp_path: Path, bad_symbol: str) -> None:
    records = valid_records()
    records[0]["symbol"] = bad_symbol
    target = write_universe(tmp_path / "empty_symbol.csv", records)

    with pytest.raises(UniverseError) as exc:
        read_universe(target)

    assert any("'symbol' must not be empty" in problem for problem in exc.value.problems)


def test_over_long_symbol_raises(tmp_path: Path) -> None:
    records = valid_records()
    records[0]["symbol"] = "A" * (MAX_SYMBOL_LENGTH + 1)
    target = write_universe(tmp_path / "long_symbol.csv", records)

    with pytest.raises(UniverseError) as exc:
        read_universe(target)

    assert any("longer than" in problem for problem in exc.value.problems)


def test_blank_name_raises(tmp_path: Path) -> None:
    records = valid_records()
    records[0]["name"] = "  "
    target = write_universe(tmp_path / "blank_name.csv", records)

    with pytest.raises(UniverseError) as exc:
        read_universe(target)

    assert any("'name' must not be empty" in problem for problem in exc.value.problems)


@pytest.mark.parametrize("bad_exchange", ["NASDQ", "LSE", "TSE", ""])
def test_unknown_or_blank_exchange_raises(tmp_path: Path, bad_exchange: str) -> None:
    records = valid_records()
    records[0]["exchange"] = bad_exchange
    target = write_universe(tmp_path / "bad_exchange.csv", records)

    with pytest.raises(UniverseError) as exc:
        read_universe(target)

    problems = exc.value.problems
    assert any(
        "unknown exchange" in problem or "'exchange' must not be empty" in problem
        for problem in problems
    )


@pytest.mark.parametrize("bad_type", ["etf", "preferred_stock", "adr", "index", ""])
def test_unsupported_security_type_raises(tmp_path: Path, bad_type: str) -> None:
    records = valid_records()
    records[0]["security_type"] = bad_type
    target = write_universe(tmp_path / "bad_type.csv", records)

    with pytest.raises(UniverseError) as exc:
        read_universe(target)

    problems = exc.value.problems
    assert any(
        "not supported yet" in problem or "'security_type' must not be empty" in problem
        for problem in problems
    )


@pytest.mark.parametrize("bad_flag", ["maybe", "2", "", "tru"])
def test_invalid_is_active_raises(tmp_path: Path, bad_flag: str) -> None:
    records = valid_records()
    records[0]["is_active"] = bad_flag
    target = write_universe(tmp_path / "bad_flag.csv", records)

    with pytest.raises(UniverseError) as exc:
        read_universe(target)

    assert any("must be a boolean" in problem for problem in exc.value.problems)


def test_too_few_stocks_raises(tmp_path: Path) -> None:
    target = write_universe(tmp_path / "small.csv", valid_records(MIN_UNIVERSE_SIZE - 1))

    with pytest.raises(UniverseError) as exc:
        read_universe(target)

    assert any(
        f"at least {MIN_UNIVERSE_SIZE} are required" in problem for problem in exc.value.problems
    )


def test_single_security_message_is_grammatical(tmp_path: Path) -> None:
    target = write_universe(tmp_path / "one.csv", valid_records(1))

    with pytest.raises(UniverseError) as exc:
        read_universe(target)

    assert any("1 security;" in problem for problem in exc.value.problems)


def test_all_problems_are_reported_in_one_exception(tmp_path: Path) -> None:
    records = valid_records()
    records[0]["symbol"] = "bad symbol"
    records[1]["security_type"] = "etf"
    records[2]["is_active"] = "maybe"
    target = write_universe(tmp_path / "many_problems.csv", records)

    with pytest.raises(UniverseError) as exc:
        read_universe(target)

    problems = exc.value.problems
    assert any("does not match" in problem for problem in problems)
    assert any("not supported yet" in problem for problem in problems)
    assert any("must be a boolean" in problem for problem in problems)
    assert len(problems) >= 3


def test_error_message_names_the_file_and_lists_the_problems(tmp_path: Path) -> None:
    target = write_universe(tmp_path / "broken.csv", valid_records(1))

    with pytest.raises(UniverseError) as exc:
        read_universe(target)

    assert str(target) in str(exc.value)
    assert "\n  - " in str(exc.value)


# ---------------------------------------------------------------------------
# Normalization and tolerated-but-harmless variations
# ---------------------------------------------------------------------------
def test_symbols_exchanges_and_types_are_case_folded(tmp_path: Path) -> None:
    records = valid_records()
    for record in records:
        record["symbol"] = record["symbol"].lower()
    records[0]["exchange"] = "nasdaq"
    records[0]["security_type"] = "Common_Stock"
    target = write_universe(tmp_path / "case_folded.csv", records)

    universe = read_universe(target, min_size=MIN_UNIVERSE_SIZE)

    assert len(universe) == MIN_UNIVERSE_SIZE
    assert all(entry.symbol == entry.symbol.upper() for entry in universe)
    assert universe.by_symbol()["AAPL"].exchange == "NASDAQ"
    assert universe.by_symbol()["AAPL"].security_type == "common_stock"


def test_surrounding_whitespace_is_stripped(tmp_path: Path) -> None:
    records = valid_records()
    records[0]["symbol"] = "  aapl  "
    records[0]["name"] = "  Apple Inc.  "
    records[0]["exchange"] = " nasdaq "
    records[0]["is_active"] = " true "
    target = write_universe(tmp_path / "whitespace.csv", records)

    entry = read_universe(target, min_size=MIN_UNIVERSE_SIZE).by_symbol()["AAPL"]

    assert entry.name == "Apple Inc."
    assert entry.exchange == "NASDAQ"
    assert entry.is_active is True


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("true", True),
        ("TRUE", True),
        ("yes", True),
        ("Y", True),
        ("1", True),
        ("false", False),
        ("FALSE", False),
        ("no", False),
        ("N", False),
        ("0", False),
    ],
)
def test_boolean_aliases_are_accepted(tmp_path: Path, raw: str, expected: bool) -> None:
    records = valid_records()
    records[0]["is_active"] = raw
    target = write_universe(tmp_path / "flag.csv", records)

    entry = read_universe(target, min_size=MIN_UNIVERSE_SIZE).by_symbol()[records[0]["symbol"]]

    assert entry.is_active is expected


def test_extra_columns_are_ignored(tmp_path: Path) -> None:
    """Forward compatibility: the contract is a *minimum* column set."""
    records = valid_records()
    for index, record in enumerate(records):
        record["sector"] = "Technology"
        record["note"] = f"row {index}"
    target = write_universe(tmp_path / "extra.csv", records)

    universe = read_universe(target, min_size=MIN_UNIVERSE_SIZE)

    assert len(universe) == MIN_UNIVERSE_SIZE
    assert "sector" not in universe.to_frame().columns


def test_blank_lines_are_skipped(tmp_path: Path) -> None:
    header = ",".join(REQUIRED_COLUMNS)
    body = "\n".join(
        f"{record['symbol']},{record['name']},{record['exchange']},"
        f"{record['security_type']},{record['is_active']}"
        for record in valid_records(3)
    )
    target = write_raw_universe(tmp_path / "blanks.csv", f"{header}\n\n{body}\n\n")

    universe = read_universe(target, min_size=3)

    assert len(universe) == 3


def test_active_only_excludes_flagged_rows(tmp_path: Path) -> None:
    records = valid_records()
    retired = records[0]["symbol"]
    records[0]["is_active"] = "false"
    target = write_universe(tmp_path / "mixed.csv", records)

    everything = load_universe_symbols(target, active_only=False)
    active = load_universe_symbols(target, active_only=True)

    assert len(everything) == MIN_UNIVERSE_SIZE
    assert len(active) == MIN_UNIVERSE_SIZE - 1
    assert retired in everything
    assert retired not in active


def test_min_size_is_enforced_by_default_and_can_be_relaxed(tmp_path: Path) -> None:
    target = write_universe(tmp_path / "two.csv", valid_records(2))

    with pytest.raises(UniverseError):
        read_universe(target)

    assert len(read_universe(target, min_size=2)) == 2


def test_allowed_exchanges_can_be_widened_per_call(tmp_path: Path) -> None:
    """A future OTC listing must not require editing the module constants."""
    records = valid_records()
    records[0]["exchange"] = "XETRA"
    target = write_universe(tmp_path / "xetra.csv", records)

    with pytest.raises(UniverseError):
        read_universe(target, min_size=MIN_UNIVERSE_SIZE)

    universe = read_universe(
        target, min_size=MIN_UNIVERSE_SIZE, allowed_exchanges=[*ALLOWED_EXCHANGES, "XETRA"]
    )

    assert universe.by_symbol()[records[0]["symbol"]].exchange == "XETRA"


# ---------------------------------------------------------------------------
# validate_universe: the non-raising half of the API
# ---------------------------------------------------------------------------
def test_validate_universe_accepts_a_valid_frame(universe_csv: Path) -> None:
    frame = pd.read_csv(universe_csv, dtype=str, keep_default_na=False)

    assert validate_universe(frame) == []


def test_validate_universe_collects_problems_without_raising() -> None:
    frame = pd.DataFrame(
        [
            {
                "symbol": "aapl",
                "name": "",
                "exchange": "LSE",
                "security_type": "etf",
                "is_active": "?",
            }
        ]
    )

    problems = validate_universe(frame, min_size=1)

    assert len(problems) >= 4
    assert any("unknown exchange" in problem for problem in problems)
    assert any("not supported yet" in problem for problem in problems)
    assert any("must be a boolean" in problem for problem in problems)
    assert not any("duplicate symbol" in problem for problem in problems)


def test_validate_universe_handles_a_missing_frame() -> None:
    assert validate_universe(None) == ["universe frame is missing (None)"]
