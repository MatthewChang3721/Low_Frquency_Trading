"""End-to-end tests for the minimal factor entry point (``scripts/run_factor.py``).

Every test here runs against **real files in ``tmp_path``**: a Hive-partitioned
Parquet dataset written through :func:`data_sys.storage.write_partitioned`, a real
universe CSV, the real registry, the real runner, the real store and the real
analyzer.  Nothing reads the project's ``data/``, nothing touches the network and
nothing is stubbed except the socket, which one test patches precisely to prove
it is never used.

Covered in this order:

* the success path: name -> registry -> universe -> bars -> runner -> panel ->
  stored Parquet -> statistics, with the moments recomputed independently in
  plain pandas and the panel read back through ``ParquetFactorStore.read``,
* the printed contract: the sixteen stable fields (``cache_status`` last), their
  values, and the absence of any issue line for a clean panel,
* partial history: the command still succeeds, the panel is still stored, and the
  report is printed instead of raised,
* the four failure modes that must never look like success: an unknown factor, a
  date that is not a session (or not a date at all), a missing or empty dataset
  and a dataset without the requested field -- each one exits non-zero and writes
  nothing,
* ``--data-root`` and ``--universe-path``: the flags really decide which series is
  read and which pool is used, and the default ``data/`` tree is untouched,
* the cache: the first run of a key is a ``miss``, the second one is a ``hit``
  that reads bars not at all, runs no factor and rewrites no file, ``--force`` is
  a ``forced`` recomputation that replaces that one file, a damaged stored panel
  fails instead of being recomputed over, and neither a weekend nor an unknown
  calendar can ever be a hit,
* the documented command in a real process, with the network patched to fail,
* the module boundary: no formula import, no download, no storage or quality rule
  re-implemented, no private store member, and no date-range or horizon flag.
"""

from __future__ import annotations

import ast
import socket
import subprocess
import sys
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import NamedTuple

import pandas as pd
import pytest

import scripts.run_factor as run_factor_module
from data_sys.config import PipelineConfig
from data_sys.standardize import standardize_bars
from data_sys.storage import read_partitioned, write_partitioned
from factor_sys import (
    DefaultFactorQualityChecker,
    FactorContractError,
    FactorPanelStatistics,
    FactorQualityIssue,
    FactorQualityReport,
    FactorRegistryError,
    FactorRunError,
    FactorRunner,
    FactorStoreError,
    MarketDataReadError,
    ParquetFactorStore,
    ParquetMarketDataReader,
    QuickAnalyzer,
    StaticUniverseProvider,
)
from scripts.run_factor import (
    CACHE_STATUSES,
    FactorRunOutcome,
    RunFactorError,
    build_parser,
    format_summary,
    main,
    run_factor,
)
from tests.helpers import make_session_prices, session_dates, write_universe_file

#: The command under test, as a file (for the boundary and subprocess tests).
SCRIPT: Path = Path(run_factor_module.__file__).resolve()

#: The project root the documented command is run from.
PROJECT_ROOT: Path = SCRIPT.parents[1]

#: A real NYSE session (the last one of 2024) that every run decides on.
AS_OF: date = date(2024, 12, 31)

#: The only factor the project ships, and the version it declares in its spec.
FACTOR_NAME: str = "momentum_12_1"
FACTOR_VERSION: str = "1.0.0"

#: The pool the fixtures publish and the tests ask for.
SYMBOLS: tuple[str, ...] = ("AAPL", "MSFT", "NVDA", "AMZN")

#: First session of the published history: far enough back for a full window.
HISTORY_START: str = "2023-06-01"

#: ``lookback_sessions + 1``: the observations a complete 12-1 window holds.
WINDOW: int = 253

#: Sessions skipped between the end anchor and the decision date.
SKIP: int = 21

#: Fields the printed summary must always carry, in this order.
SUMMARY_FIELDS: tuple[str, ...] = (
    "factor_name",
    "factor_version",
    "as_of_date",
    "universe_size",
    "eligible_count",
    "eligible_coverage",
    "valid_count",
    "missing_count",
    "missing_rate",
    "mean",
    "std",
    "minimum",
    "maximum",
    "stored_panel_path",
    "quality_status",
    "cache_status",
)

#: The flag destinations the command accepts -- and nothing else.
EXPECTED_DESTS: frozenset[str] = frozenset(
    {"factor", "as_of", "data_root", "universe_path", "calendar", "force", "help"}
)


# ---------------------------------------------------------------------------
# Fixtures: real Parquet on disk, written by the data system's own writer
# ---------------------------------------------------------------------------
class Market(NamedTuple):
    """One published dataset plus the universe file that goes with it."""

    root: Path
    universe: Path
    frame: pd.DataFrame


def publish(
    root: Path,
    symbols: tuple[str, ...] = SYMBOLS,
    *,
    start: str = HISTORY_START,
    end: str = AS_OF.isoformat(),
    seed: int = 20200102,
) -> pd.DataFrame:
    """Publish contract-valid bars for ``symbols`` under ``root`` and return them."""
    sessions = session_dates(start, end)
    frames = [
        standardize_bars(make_session_prices(sessions, seed=seed + position), symbol)
        for position, symbol in enumerate(symbols)
    ]
    frame = pd.concat(frames, ignore_index=True)
    write_partitioned(frame, root / "standardized" / "market_bars")
    return frame


def published_frame(root: Path) -> pd.DataFrame:
    """The standardized dataset currently published under ``root``, read back."""
    return read_partitioned(root / "standardized" / "market_bars")


@pytest.fixture
def market(tmp_path: Path) -> Market:
    """A complete history for :data:`SYMBOLS` plus a matching universe file."""
    root = tmp_path / "data"
    frame = publish(root)
    universe = write_universe_file(tmp_path / "universe.csv", SYMBOLS)
    return Market(root=root, universe=universe, frame=frame)


# ---------------------------------------------------------------------------
# Expectations: the formula and the moments, computed independently
# ---------------------------------------------------------------------------
def symbol_rows(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """``symbol``'s bars, oldest first."""
    rows = frame.loc[frame["symbol"] == symbol]
    return rows.sort_values("date", kind="stable").reset_index(drop=True)


def expected_value(frame: pd.DataFrame, symbol: str, as_of: date = AS_OF) -> float:
    """The 12-1 return of one symbol, from the stored prices alone.

    Deliberately written with plain positions over the visible window, so
    agreement with the factor is evidence instead of a restatement of it.
    """
    rows = symbol_rows(frame, symbol)
    window = rows.loc[rows["date"] <= pd.Timestamp(as_of)].tail(WINDOW)
    assert len(window) == WINDOW, f"{symbol} does not have a full window"
    start = float(window["adjusted_close"].iloc[0])
    end = float(window["adjusted_close"].iloc[-SKIP - 1])
    return end / start - 1.0


def expected_values(frame: pd.DataFrame, symbols: tuple[str, ...] = SYMBOLS) -> list[float]:
    """The 12-1 return of every symbol, in ``symbols`` order."""
    return [expected_value(frame, symbol) for symbol in symbols]


def mean_of(values: list[float]) -> float:
    """The arithmetic mean, written out."""
    return sum(values) / len(values)


def spread_of(values: list[float]) -> float:
    """The population standard deviation (``ddof=0``), written out."""
    centre = mean_of(values)
    return (sum((value - centre) ** 2 for value in values) / len(values)) ** 0.5


def panel_files(root: Path) -> list[Path]:
    """Every stored factor panel file under ``root``, sorted."""
    panels = root / "factors" / "panels"
    return sorted(panels.rglob("*.parquet")) if panels.exists() else []


def temp_files(root: Path) -> list[Path]:
    """Every leftover temporary file of the store's atomic write."""
    panels = root / "factors" / "panels"
    return sorted(panels.rglob("*.tmp")) if panels.exists() else []


def stored_panel_path(root: Path, as_of: date = AS_OF) -> Path:
    """The one file the store must use for ``(momentum_12_1, 1.0.0, as_of)``."""
    return (
        root
        / "factors"
        / "panels"
        / f"factor_name={FACTOR_NAME}"
        / f"factor_version={FACTOR_VERSION}"
        / f"year={as_of.year:04d}"
        / f"as_of_date={as_of.isoformat()}.parquet"
    )


def summary_fields(text: str) -> dict[str, str]:
    """Parse the ``key: value`` summary into a dict (issue lines excluded)."""
    fields: dict[str, str] = {}
    for line in text.splitlines():
        if line.startswith("quality_issue_"):
            continue
        key, separator, value = line.partition(": ")
        assert separator, f"summary line {line!r} is not 'key: value'"
        fields[key] = value
    return fields


def tree_state(root: Path) -> dict[str, tuple[int, int]]:
    """Size and modification time of every file under ``root`` (empty if absent)."""
    if not root.exists():
        return {}
    return {
        str(path.relative_to(root)): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


# ---------------------------------------------------------------------------
# 1. The success path: real data in, real Parquet out
# ---------------------------------------------------------------------------
def test_momentum_end_to_end_computes_the_panel_and_stores_it(market: Market) -> None:
    """One command: registry -> bars -> panel -> Parquet -> statistics."""
    outcome = run_factor(
        FACTOR_NAME,
        AS_OF.isoformat(),
        data_root=market.root,
        universe_path=market.universe,
    )

    assert isinstance(outcome, FactorRunOutcome)
    panel = outcome.panel
    pool = StaticUniverseProvider(market.universe).symbols

    assert (panel.factor_name, panel.factor_version) == (FACTOR_NAME, FACTOR_VERSION)
    assert panel.as_of_date == AS_OF
    assert panel.universe == pool
    assert panel.symbols == pool

    rows = panel.frame
    assert rows["symbol"].tolist() == list(pool)
    assert rows["is_eligible"].tolist() == [True] * len(pool)
    by_symbol = {symbol: expected_value(market.frame, symbol) for symbol in pool}
    assert rows["raw_value"].tolist() == pytest.approx(
        [by_symbol[symbol] for symbol in pool], rel=1e-12
    )

    statistics = outcome.statistics
    assert isinstance(statistics, FactorPanelStatistics)
    assert (statistics.factor_name, statistics.factor_version) == (FACTOR_NAME, FACTOR_VERSION)
    assert statistics.as_of_date == AS_OF
    assert statistics.universe_size == len(pool)
    assert statistics.eligible_count == len(pool)
    assert statistics.valid_count == len(pool)
    assert statistics.missing_count == 0
    assert statistics.eligible_coverage == 1.0
    assert statistics.missing_rate == 0.0

    values = rows["raw_value"].tolist()
    assert statistics.mean == pytest.approx(mean_of(values), rel=1e-12)
    assert statistics.std == pytest.approx(spread_of(values), rel=1e-12)
    assert statistics.minimum == min(values)
    assert statistics.maximum == max(values)

    assert outcome.quality_report is None
    assert outcome.quality_status == "clean"
    assert outcome.issues == ()

    assert outcome.stored_panel_path == stored_panel_path(market.root)
    assert outcome.stored_panel_path.is_file()
    assert panel_files(market.root) == [outcome.stored_panel_path]
    assert temp_files(market.root) == []
    assert outcome.cache_status == "miss"


def test_the_stored_panel_reads_back_identical(market: Market) -> None:
    """The file *is* the panel: the store rebuilds exactly what was computed."""
    outcome = run_factor(
        FACTOR_NAME,
        AS_OF,
        data_root=market.root,
        universe_path=market.universe,
    )

    store = ParquetFactorStore(market.root / "factors" / "panels")
    loaded = store.read(FACTOR_NAME, FACTOR_VERSION, AS_OF)

    pd.testing.assert_frame_equal(loaded.frame, outcome.panel.frame)
    assert loaded.universe == outcome.panel.universe
    assert loaded.as_of_date == outcome.panel.as_of_date
    assert (loaded.factor_name, loaded.factor_version) == (FACTOR_NAME, FACTOR_VERSION)

    again = store.read(FACTOR_NAME, FACTOR_VERSION, AS_OF.isoformat())
    assert again is not loaded
    pd.testing.assert_frame_equal(again.frame, loaded.frame)


def test_an_iso_string_and_a_date_are_the_same_run(market: Market) -> None:
    """The date is a label: both spellings reach the same key and the same rows."""
    from_text = run_factor(
        FACTOR_NAME,
        AS_OF.isoformat(),
        data_root=market.root,
        universe_path=market.universe,
    )
    from_date = run_factor(
        FACTOR_NAME,
        AS_OF,
        data_root=market.root,
        universe_path=market.universe,
    )

    assert from_text.cache_status == "miss"
    assert from_date.cache_status == "hit"  # one key, so the second spelling hits it
    assert from_text.stored_panel_path == from_date.stored_panel_path
    pd.testing.assert_frame_equal(from_text.panel.frame, from_date.panel.frame)
    assert from_text.statistics == from_date.statistics
    assert len(panel_files(market.root)) == 1


# ---------------------------------------------------------------------------
# 2. The printed contract
# ---------------------------------------------------------------------------
def test_the_command_prints_the_contract_fields(
    market: Market, capsys: pytest.CaptureFixture
) -> None:
    """Exit code 0, sixteen stable fields, and the numbers the run produced."""
    code = main(
        [
            "--factor",
            FACTOR_NAME,
            "--as-of",
            AS_OF.isoformat(),
            "--data-root",
            str(market.root),
            "--universe-path",
            str(market.universe),
        ]
    )
    captured = capsys.readouterr()
    assert code == 0
    assert captured.err == ""

    lines = captured.out.splitlines()
    assert [line.split(": ", 1)[0] for line in lines] == list(SUMMARY_FIELDS)

    fields = summary_fields(captured.out)
    assert fields["factor_name"] == FACTOR_NAME
    assert fields["factor_version"] == FACTOR_VERSION
    assert fields["as_of_date"] == AS_OF.isoformat()
    assert fields["universe_size"] == str(len(SYMBOLS))
    assert fields["eligible_count"] == str(len(SYMBOLS))
    assert fields["eligible_coverage"] == "1.000000"
    assert fields["valid_count"] == str(len(SYMBOLS))
    assert fields["missing_count"] == "0"
    assert fields["missing_rate"] == "0.000000"
    assert fields["stored_panel_path"] == str(stored_panel_path(market.root))
    assert fields["quality_status"] == "clean"
    assert fields["cache_status"] == "miss"

    values = expected_values(market.frame, tuple(sorted(SYMBOLS)))
    assert fields["mean"] == f"{mean_of(values):.6f}"
    assert fields["std"] == f"{spread_of(values):.6f}"
    assert fields["minimum"] == f"{min(values):.6f}"
    assert fields["maximum"] == f"{max(values):.6f}"


def test_a_clean_run_prints_no_issue_lines(market: Market) -> None:
    """Nothing is invented to fill a gap the checker did not find."""
    outcome = run_factor(
        FACTOR_NAME,
        AS_OF,
        data_root=market.root,
        universe_path=market.universe,
    )
    text = format_summary(outcome)

    assert outcome.quality_report is None
    assert "quality_issue" not in text
    assert text.splitlines()[-2:] == ["quality_status: clean", "cache_status: miss"]
    assert len(text.splitlines()) == len(SUMMARY_FIELDS)


# ---------------------------------------------------------------------------
# 3. History the factor cannot use: reported, never fatal
# ---------------------------------------------------------------------------
def mixed_history(
    tmp_path: Path,
    *,
    short: str = "AMZN",
    pool: tuple[str, ...] = SYMBOLS,
) -> Market:
    """A dataset where ``short`` has too little history and the rest is complete."""
    root = tmp_path / "data"
    full = session_dates(HISTORY_START, AS_OF.isoformat())
    recent = session_dates("2024-09-03", AS_OF.isoformat())
    frames = [
        standardize_bars(
            make_session_prices(recent if symbol == short else full, seed=20200102 + position),
            symbol,
        )
        for position, symbol in enumerate(pool)
    ]
    frame = pd.concat(frames, ignore_index=True)
    write_partitioned(frame, root / "standardized" / "market_bars")
    universe = write_universe_file(tmp_path / "universe.csv", pool)
    return Market(root=root, universe=universe, frame=frame)


def test_partial_history_is_reported_and_still_stored(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """A warning is a finding, not a failure: exit 0 and the panel is on disk."""
    market = mixed_history(tmp_path)
    code = main(
        [
            "--factor",
            FACTOR_NAME,
            "--as-of",
            AS_OF.isoformat(),
            "--data-root",
            str(market.root),
            "--universe-path",
            str(market.universe),
        ]
    )
    captured = capsys.readouterr()
    assert code == 0
    assert captured.err == ""

    fields = summary_fields(captured.out)
    assert fields["quality_status"] == "reported"
    assert fields["universe_size"] == "4"
    assert fields["eligible_count"] == "3"
    assert fields["valid_count"] == "3"
    assert fields["missing_count"] == "1"
    assert fields["missing_rate"] == "0.250000"
    assert fields["eligible_coverage"] == "0.750000"

    assert "quality_issue_code: partial_eligible_coverage" in captured.out
    assert "quality_issue_severity: warning" in captured.out
    assert "quality_issue_affected_symbols: AMZN" in captured.out
    assert "quality_issue_message: 3 of 4 symbol(s) are eligible" in captured.out

    usable = ("AAPL", "MSFT", "NVDA")
    values = expected_values(market.frame, usable)
    assert fields["mean"] == f"{mean_of(values):.6f}"
    assert fields["minimum"] == f"{min(values):.6f}"
    assert fields["maximum"] == f"{max(values):.6f}"

    loaded = ParquetFactorStore(market.root / "factors" / "panels").read(
        FACTOR_NAME, FACTOR_VERSION, AS_OF
    )
    frame = loaded.frame
    assert frame["symbol"].tolist() == ["AAPL", "AMZN", "MSFT", "NVDA"]
    short_row = frame.loc[frame["symbol"] == "AMZN"].iloc[0]
    assert bool(short_row["is_eligible"]) is False
    assert pd.isna(short_row["raw_value"])
    assert frame.loc[frame["symbol"] != "AMZN", "is_eligible"].all()
    assert panel_files(market.root) == [stored_panel_path(market.root)]


def test_a_cross_section_with_no_eligible_symbol_is_stored_and_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """Nothing usable is still an answer: no panel is invented, but none is lost."""
    root = tmp_path / "data"
    publish(root)
    pool = ("AMD", "INTC")
    universe = write_universe_file(tmp_path / "universe.csv", pool)

    code = main(
        [
            "--factor",
            FACTOR_NAME,
            "--as-of",
            AS_OF.isoformat(),
            "--data-root",
            str(root),
            "--universe-path",
            str(universe),
        ]
    )
    captured = capsys.readouterr()
    assert code == 0

    fields = summary_fields(captured.out)
    assert fields["quality_status"] == "reported"
    assert fields["eligible_count"] == "0"
    assert fields["eligible_coverage"] == "0.000000"
    assert fields["valid_count"] == "0"
    assert fields["missing_count"] == "2"
    assert fields["missing_rate"] == "1.000000"
    assert (fields["mean"], fields["std"], fields["minimum"], fields["maximum"]) == (
        "n/a",
        "n/a",
        "n/a",
        "n/a",
    )

    assert "quality_issue_code: zero_eligible_coverage" in captured.out
    assert "quality_issue_severity: error" in captured.out
    assert "quality_issue_affected_symbols: AMD, INTC" in captured.out

    loaded = ParquetFactorStore(root / "factors" / "panels").read(
        FACTOR_NAME, FACTOR_VERSION, AS_OF
    )
    assert loaded.symbols == ("AMD", "INTC")
    assert loaded.frame["is_eligible"].tolist() == [False, False]
    assert panel_files(root) == [stored_panel_path(root)]


# ---------------------------------------------------------------------------
# 4. Failures: exit code 1, one readable line on stderr, nothing written
# ---------------------------------------------------------------------------
def run_cli(args: list[str], capsys: pytest.CaptureFixture) -> tuple[int, str, str]:
    """Drive :func:`main` like a shell would and return ``(code, stdout, stderr)``."""
    code = main(args)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def arguments(market: Market, *, force: bool = False, **overrides: str) -> list[str]:
    """The full command line for ``market``, with any flag replaced."""
    flags = {
        "--factor": FACTOR_NAME,
        "--as-of": AS_OF.isoformat(),
        "--data-root": str(market.root),
        "--universe-path": str(market.universe),
    }
    flags.update(overrides)
    args: list[str] = []
    for flag, value in flags.items():
        args.extend([flag, value])
    if force:
        args.append("--force")
    return args


def file_state(path: Path) -> tuple[int, bytes]:
    """Modification time and exact bytes of one file (a write is visible here)."""
    return path.stat().st_mtime_ns, path.read_bytes()


def test_an_unknown_factor_fails_before_anything_is_written(
    market: Market, capsys: pytest.CaptureFixture
) -> None:
    """The registry answers first, so a typo never reaches the disk."""
    code, out, err = run_cli(arguments(market, **{"--factor": "value_quality"}), capsys)

    assert code == 1
    assert out == ""
    assert err.startswith("error: ")
    assert "value_quality" in err
    assert FACTOR_NAME in err  # the available factors are named
    assert not (market.root / "factors").exists()
    assert panel_files(market.root) == []

    with pytest.raises(FactorRegistryError):
        run_factor(
            "value_quality",
            AS_OF,
            data_root=market.root,
            universe_path=market.universe,
        )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2024-12-28", "not a XNYS trading session"),
        ("2024-12-25", "not a XNYS trading session"),
        ("2024-13-01", "as_of_date must be a date"),
        ("2024-02-30", "as_of_date must be a date"),
        ("31/12/2024", "as_of_date must be a date"),
    ],
)
def test_a_date_that_is_not_a_session_fails_without_writing(
    market: Market,
    capsys: pytest.CaptureFixture,
    value: str,
    expected: str,
) -> None:
    """A weekend, a holiday and a malformed date are all hard failures."""
    code, out, err = run_cli(arguments(market, **{"--as-of": value}), capsys)

    assert code == 1
    assert out == ""
    assert err.startswith("error: ")
    assert expected in err
    assert not (market.root / "factors").exists()

    with pytest.raises(FactorRunError):
        run_factor(
            FACTOR_NAME,
            value,
            data_root=market.root,
            universe_path=market.universe,
        )


def test_a_missing_market_bars_root_fails_without_writing(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """A data root with no standardized dataset is an error, not an empty panel."""
    root = tmp_path / "data"
    root.mkdir()
    market = Market(
        root=root,
        universe=write_universe_file(tmp_path / "universe.csv", SYMBOLS),
        frame=pd.DataFrame(),
    )

    code, out, err = run_cli(arguments(market), capsys)

    assert code == 1
    assert out == ""
    assert "does not exist" in err
    assert "market_bars" in err
    assert panel_files(root) == []

    with pytest.raises(MarketDataReadError, match="does not exist"):
        run_factor(FACTOR_NAME, AS_OF, data_root=root, universe_path=market.universe)


def test_an_empty_market_bars_root_fails_without_writing(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """A directory with no Parquet file cannot answer a read."""
    root = tmp_path / "data"
    (root / "standardized" / "market_bars").mkdir(parents=True)
    market = Market(
        root=root,
        universe=write_universe_file(tmp_path / "universe.csv", SYMBOLS),
        frame=pd.DataFrame(),
    )

    code, out, err = run_cli(arguments(market), capsys)

    assert code == 1
    assert out == ""
    assert "*.parquet" in err
    assert panel_files(root) == []


def test_a_dataset_without_the_requested_field_fails_without_writing(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """A published dataset missing ``adjusted_close`` is a loud read failure."""
    root = tmp_path / "data"
    frame = publish(root)
    write_partitioned(
        frame.drop(columns=["adjusted_close"]),
        root / "standardized" / "market_bars",
    )
    market = Market(
        root=root,
        universe=write_universe_file(tmp_path / "universe.csv", SYMBOLS),
        frame=frame,
    )

    code, out, err = run_cli(arguments(market), capsys)

    assert code == 1
    assert out == ""
    assert "adjusted_close" in err
    assert panel_files(root) == []


def test_an_unreadable_universe_fails_without_writing(
    market: Market, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """A missing pool file is a contract error, never a run over a guess."""
    missing = tmp_path / "nowhere" / "universe.csv"

    code, out, err = run_cli(arguments(market, **{"--universe-path": str(missing)}), capsys)

    assert code == 1
    assert out == ""
    assert "could not load the static universe" in err
    assert panel_files(market.root) == []

    with pytest.raises(FactorContractError):
        run_factor(
            FACTOR_NAME,
            AS_OF,
            data_root=market.root,
            universe_path=missing,
        )


@pytest.mark.parametrize(
    "args",
    [
        ["--as-of", "2024-12-31"],
        ["--factor", "momentum_12_1"],
        ["--factor", "momentum_12_1", "--as-of", "2024-12-31", "--start", "2024-01-01"],
        ["--factor", "momentum_12_1", "--as-of", "2024-12-31", "--cache"],
        ["--force"],
    ],
)
def test_argparse_keeps_its_own_exit_code(args: list[str]) -> None:
    """A missing or unknown flag is argparse's exit code 2, not ours."""
    with pytest.raises(SystemExit) as exit_info:
        main(args)
    assert exit_info.value.code == 2


def test_a_blank_data_root_is_refused(market: Market, capsys: pytest.CaptureFixture) -> None:
    """A blank path would silently mean the current directory: refuse it."""
    code, out, err = run_cli(arguments(market, **{"--data-root": "  "}), capsys)

    assert code == 1
    assert out == ""
    assert "--data-root must not be blank" in err
    assert not (market.root / "factors").exists()

    with pytest.raises(RunFactorError, match="must not be blank"):
        run_factor(
            FACTOR_NAME,
            AS_OF,
            data_root="  ",
            universe_path=market.universe,
        )


# ---------------------------------------------------------------------------
# 5. The flags: one root for both ends, one pool for the run
# ---------------------------------------------------------------------------
def test_data_root_decides_where_bars_are_read_and_panels_are_written(
    tmp_path: Path,
) -> None:
    """The root is used for bars *and* for panels; the default tree is untouched."""
    chosen_root = tmp_path / "chosen"
    decoy_root = tmp_path / "decoy"
    chosen_frame = publish(chosen_root)
    decoy_frame = publish(decoy_root, seed=777)
    universe = write_universe_file(tmp_path / "universe.csv", SYMBOLS)
    pool = tuple(sorted(SYMBOLS))

    default_store = PipelineConfig().data_root / "factors" / "panels"
    before = tree_state(default_store)

    outcome = run_factor(
        FACTOR_NAME,
        AS_OF,
        data_root=chosen_root,
        universe_path=universe,
    )

    assert outcome.stored_panel_path == stored_panel_path(chosen_root)
    assert outcome.stored_panel_path.is_relative_to(chosen_root)
    assert panel_files(chosen_root) == [outcome.stored_panel_path]
    assert panel_files(decoy_root) == []

    values = outcome.panel.frame["raw_value"].tolist()
    assert values == pytest.approx(expected_values(chosen_frame, pool), rel=1e-12)
    assert values != pytest.approx(expected_values(decoy_frame, pool), rel=1e-12)

    assert tree_state(default_store) == before


def test_universe_path_replaces_the_static_pool(tmp_path: Path) -> None:
    """The flag's file is the pool -- not the committed seed, not another file."""
    root = tmp_path / "data"
    publish(root)

    first_universe = write_universe_file(tmp_path / "first.csv", ("NVDA", "AAPL"))
    first = run_factor(
        FACTOR_NAME,
        AS_OF,
        data_root=root,
        universe_path=first_universe,
    )
    assert first.panel.universe == ("AAPL", "NVDA")
    assert first.panel.frame["symbol"].tolist() == ["AAPL", "NVDA"]
    assert first.statistics.universe_size == 2

    second_universe = write_universe_file(tmp_path / "second.csv", ("MSFT",))
    second = run_factor(
        FACTOR_NAME,
        AS_OF,
        data_root=root,
        universe_path=second_universe,
        force=True,
    )
    assert second.cache_status == "forced"
    assert second.panel.universe == ("MSFT",)
    assert second.statistics.universe_size == 1

    # Same factor, same version, same session: one key, therefore one file.
    assert len(panel_files(root)) == 1
    stored = ParquetFactorStore(root / "factors" / "panels").read(
        FACTOR_NAME, FACTOR_VERSION, AS_OF
    )
    assert stored.universe == ("MSFT",)


def test_running_the_same_key_twice_reads_the_panel_back(tmp_path: Path) -> None:
    """A second run is a cache hit: same panel, same file, no write at all."""
    root = tmp_path / "data"
    publish(root)
    universe = write_universe_file(tmp_path / "universe.csv", SYMBOLS)

    first = run_factor(FACTOR_NAME, AS_OF, data_root=root, universe_path=universe)
    stored = first.stored_panel_path
    before = file_state(stored)

    second = run_factor(FACTOR_NAME, AS_OF, data_root=root, universe_path=universe)

    assert (first.cache_status, second.cache_status) == ("miss", "hit")
    assert second.stored_panel_path == stored
    assert file_state(stored) == before  # identical bytes, identical mtime
    assert panel_files(root) == [stored]
    assert temp_files(root) == []
    assert second.statistics == first.statistics
    pd.testing.assert_frame_equal(second.panel.frame, first.panel.frame)

    version_dir = root / "factors" / "panels" / f"factor_name={FACTOR_NAME}"
    assert len(list(version_dir.rglob("*.parquet"))) == 1


def test_the_run_never_opens_a_socket(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The entry point reads local files only: every network door is nailed shut."""
    root = tmp_path / "data"
    publish(root)
    universe = write_universe_file(tmp_path / "universe.csv", SYMBOLS)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("the factor entry point must never use the network")

    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    try:
        import yfinance
    except ImportError:  # pragma: no cover - the dependency is declared
        pass
    else:
        monkeypatch.setattr(yfinance, "download", forbidden)

    outcome = run_factor(FACTOR_NAME, AS_OF, data_root=root, universe_path=universe)

    assert outcome.stored_panel_path == stored_panel_path(root)
    assert outcome.stored_panel_path.is_file()
    assert outcome.cache_status == "miss"


# ---------------------------------------------------------------------------
# 6. The cache: a stored key is read back, not recomputed
# ---------------------------------------------------------------------------
def test_a_hit_reads_no_bars_runs_no_factor_and_writes_nothing(
    market: Market, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stored panel is the answer: nothing under it may be touched again."""
    first = run_factor(
        FACTOR_NAME,
        AS_OF,
        data_root=market.root,
        universe_path=market.universe,
    )

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("a cache hit must not read bars, run a factor or write a panel")

    monkeypatch.setattr(ParquetMarketDataReader, "read_history", forbidden)
    monkeypatch.setattr(FactorRunner, "run", forbidden)
    monkeypatch.setattr(ParquetFactorStore, "write", forbidden)

    second = run_factor(
        FACTOR_NAME,
        AS_OF,
        data_root=market.root,
        universe_path=market.universe,
    )

    assert second.cache_status == "hit"
    assert second.stored_panel_path == first.stored_panel_path
    assert second.stored_panel_path.is_file()
    pd.testing.assert_frame_equal(second.panel.frame, first.panel.frame)
    assert second.statistics == first.statistics
    assert second.quality_report is None


def test_a_hit_still_checks_quality_and_summarizes_the_panel(
    market: Market, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A hit re-derives its description: no report or statistic is read from disk."""
    first = run_factor(
        FACTOR_NAME,
        AS_OF,
        data_root=market.root,
        universe_path=market.universe,
    )

    checked: list[object] = []
    summarized: list[object] = []
    original_check = DefaultFactorQualityChecker.inspect
    original_summary = QuickAnalyzer.summarize

    def spy_check(self: object, panel: object) -> object:
        checked.append(panel)
        return original_check(self, panel)  # type: ignore[arg-type]

    def spy_summarize(self: object, panel: object) -> object:
        summarized.append(panel)
        return original_summary(self, panel)  # type: ignore[arg-type]

    monkeypatch.setattr(DefaultFactorQualityChecker, "inspect", spy_check)
    monkeypatch.setattr(QuickAnalyzer, "summarize", spy_summarize)

    second = run_factor(
        FACTOR_NAME,
        AS_OF,
        data_root=market.root,
        universe_path=market.universe,
    )

    assert second.cache_status == "hit"
    assert checked == [second.panel]
    assert summarized == [second.panel]
    assert second.quality_report is None
    assert second.statistics == first.statistics


def test_a_hit_reports_partial_coverage_from_the_panel_itself(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """The quality report of a hit is the current rule set applied to that panel."""
    market = mixed_history(tmp_path)
    first = run_factor(
        FACTOR_NAME,
        AS_OF,
        data_root=market.root,
        universe_path=market.universe,
    )
    second = run_factor(
        FACTOR_NAME,
        AS_OF,
        data_root=market.root,
        universe_path=market.universe,
    )

    assert (first.cache_status, second.cache_status) == ("miss", "hit")
    assert second.quality_status == "reported"
    assert second.issues == first.issues
    assert second.statistics == first.statistics

    code, out, err = run_cli(arguments(market), capsys)
    assert (code, err) == (0, "")
    fields = summary_fields(out)
    assert (fields["cache_status"], fields["quality_status"]) == ("hit", "reported")
    assert "quality_issue_code: partial_eligible_coverage" in out


def test_force_recomputes_and_replaces_the_one_file(
    market: Market, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--force`` reaches every step again, and still leaves exactly one panel."""
    first = run_factor(
        FACTOR_NAME,
        AS_OF,
        data_root=market.root,
        universe_path=market.universe,
    )
    stored = first.stored_panel_path
    before = file_state(stored)

    calls: dict[str, int] = {"bars": 0, "run": 0, "write": 0}
    original_history = ParquetMarketDataReader.read_history
    original_run = FactorRunner.run
    original_write = ParquetFactorStore.write

    def counting_history(self: object, *args: object, **kwargs: object) -> object:
        calls["bars"] += 1
        return original_history(self, *args, **kwargs)  # type: ignore[arg-type]

    def counting_run(self: object, *args: object, **kwargs: object) -> object:
        calls["run"] += 1
        return original_run(self, *args, **kwargs)  # type: ignore[arg-type]

    def counting_write(self: object, *args: object, **kwargs: object) -> object:
        calls["write"] += 1
        return original_write(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(ParquetMarketDataReader, "read_history", counting_history)
    monkeypatch.setattr(FactorRunner, "run", counting_run)
    monkeypatch.setattr(ParquetFactorStore, "write", counting_write)

    forced = run_factor(
        FACTOR_NAME,
        AS_OF,
        data_root=market.root,
        universe_path=market.universe,
        force=True,
    )

    assert forced.cache_status == "forced"
    assert calls == {"bars": 1, "run": 1, "write": 1}
    assert forced.stored_panel_path == stored
    assert panel_files(market.root) == [stored]
    assert temp_files(market.root) == []
    assert forced.statistics == first.statistics

    after = file_state(stored)
    assert after[0] >= before[0]  # the same file, written again
    # A deterministic panel serializes to the same bytes; the write counter above
    # is what proves the store was asked to replace it.


def test_force_without_a_stored_panel_computes_and_says_forced(market: Market) -> None:
    """``--force`` on an empty store is an ordinary first run, reported as forced."""
    assert not (market.root / "factors").exists()

    outcome = run_factor(
        FACTOR_NAME,
        AS_OF,
        data_root=market.root,
        universe_path=market.universe,
        force=True,
    )

    assert outcome.cache_status == "forced"
    assert outcome.stored_panel_path == stored_panel_path(market.root)
    assert outcome.stored_panel_path.is_file()
    assert panel_files(market.root) == [outcome.stored_panel_path]
    assert temp_files(market.root) == []


def test_a_hit_ignores_a_changed_pool_until_it_is_forced(tmp_path: Path) -> None:
    """The key is a name, a version and a date -- the pool is not part of it."""
    root = tmp_path / "data"
    publish(root)
    first_universe = write_universe_file(tmp_path / "first.csv", ("AAPL", "NVDA"))
    second_universe = write_universe_file(tmp_path / "second.csv", ("MSFT",))

    first = run_factor(FACTOR_NAME, AS_OF, data_root=root, universe_path=first_universe)
    cached = run_factor(FACTOR_NAME, AS_OF, data_root=root, universe_path=second_universe)

    assert cached.cache_status == "hit"
    assert cached.panel.universe == first.panel.universe == ("AAPL", "NVDA")

    forced = run_factor(
        FACTOR_NAME,
        AS_OF,
        data_root=root,
        universe_path=second_universe,
        force=True,
    )

    assert forced.cache_status == "forced"
    assert forced.panel.universe == ("MSFT",)


def test_a_hit_keeps_the_stored_values_when_the_bars_change(tmp_path: Path) -> None:
    """Republished bars are invisible to a hit: that is what ``--force`` is for."""
    root = tmp_path / "data"
    original = publish(root)
    universe = write_universe_file(tmp_path / "universe.csv", SYMBOLS)
    first = run_factor(FACTOR_NAME, AS_OF, data_root=root, universe_path=universe)

    republished = publish(root, seed=909090)
    assert republished["adjusted_close"].tolist() != original["adjusted_close"].tolist()

    cached = run_factor(FACTOR_NAME, AS_OF, data_root=root, universe_path=universe)
    assert cached.cache_status == "hit"
    assert cached.panel.frame["raw_value"].tolist() == first.panel.frame["raw_value"].tolist()

    forced = run_factor(FACTOR_NAME, AS_OF, data_root=root, universe_path=universe, force=True)
    pool = tuple(sorted(SYMBOLS))
    assert forced.cache_status == "forced"
    assert forced.panel.frame["raw_value"].tolist() == pytest.approx(
        expected_values(republished, pool), rel=1e-12
    )
    assert forced.panel.frame["raw_value"].tolist() != first.panel.frame["raw_value"].tolist()


def test_a_damaged_stored_panel_fails_and_is_not_overwritten(
    tmp_path: Path, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Damaged evidence is reported, never quietly replaced by a fresh run."""
    root = tmp_path / "data"
    publish(root)
    universe = write_universe_file(tmp_path / "universe.csv", SYMBOLS)
    first = run_factor(FACTOR_NAME, AS_OF, data_root=root, universe_path=universe)
    stored = first.stored_panel_path
    stored.write_bytes(b"not a parquet file")
    before = file_state(stored)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("a damaged panel must not be recomputed or overwritten")

    monkeypatch.setattr(ParquetMarketDataReader, "read_history", forbidden)
    monkeypatch.setattr(FactorRunner, "run", forbidden)
    monkeypatch.setattr(ParquetFactorStore, "write", forbidden)

    with pytest.raises(FactorStoreError, match="cannot be read as parquet"):
        run_factor(FACTOR_NAME, AS_OF, data_root=root, universe_path=universe)

    market = Market(root=root, universe=universe, frame=pd.DataFrame())
    code, out, err = run_cli(arguments(market), capsys)

    assert (code, out) == (1, "")
    assert err.startswith("error: ")
    assert "cannot be read as parquet" in err
    assert stored.name in err
    assert file_state(stored) == before
    assert panel_files(root) == [stored]


def test_a_weekend_is_never_a_hit_even_with_a_file_at_that_key(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """A file whose name carries a non-session date is not a cache entry."""
    root = tmp_path / "data"
    publish(root)
    universe = write_universe_file(tmp_path / "universe.csv", SYMBOLS)
    market = Market(root=root, universe=universe, frame=pd.DataFrame())
    session = run_factor(FACTOR_NAME, AS_OF, data_root=root, universe_path=universe)
    weekend = date(2024, 12, 28)  # a Saturday

    planted = ParquetFactorStore(root / "factors" / "panels").path_for(
        FACTOR_NAME, FACTOR_VERSION, weekend
    )
    planted.parent.mkdir(parents=True, exist_ok=True)
    planted.write_bytes(session.stored_panel_path.read_bytes())
    before = file_state(planted)

    with pytest.raises(FactorRunError, match="not a XNYS trading session"):
        run_factor(FACTOR_NAME, weekend, data_root=root, universe_path=universe)

    code, out, err = run_cli(arguments(market, **{"--as-of": weekend.isoformat()}), capsys)

    assert (code, out) == (1, "")
    assert "not a XNYS trading session" in err
    assert file_state(planted) == before


def test_an_unknown_calendar_fails_before_the_store(
    market: Market, capsys: pytest.CaptureFixture
) -> None:
    """A calendar that cannot be loaded is a domain error, not a cache miss."""
    code, out, err = run_cli(arguments(market, **{"--calendar": "NOPE"}), capsys)

    assert (code, out) == (1, "")
    assert "could not load the trading calendar" in err

    with pytest.raises(FactorRunError, match="could not load the trading calendar"):
        run_factor(
            FACTOR_NAME,
            AS_OF,
            data_root=market.root,
            universe_path=market.universe,
            calendar_name="NOPE",
        )

    assert not (market.root / "factors").exists()


def test_the_command_reports_miss_hit_and_forced(
    market: Market, capsys: pytest.CaptureFixture
) -> None:
    """The last fixed line of every summary is the one the store answered."""
    code, out, err = run_cli(arguments(market), capsys)
    assert (code, err) == (0, "")
    assert summary_fields(out)["cache_status"] == "miss"
    assert out.splitlines()[-1] == "cache_status: miss"

    code, out, err = run_cli(arguments(market), capsys)
    assert (code, err) == (0, "")
    assert summary_fields(out)["cache_status"] == "hit"
    assert out.splitlines()[-1] == "cache_status: hit"

    code, out, err = run_cli(arguments(market, force=True), capsys)
    assert (code, err) == (0, "")
    assert summary_fields(out)["cache_status"] == "forced"
    assert out.splitlines()[-1] == "cache_status: forced"


def test_the_cache_never_opens_a_socket(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Miss, hit and forced all read local files only."""
    root = tmp_path / "data"
    publish(root)
    universe = write_universe_file(tmp_path / "universe.csv", SYMBOLS)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("the factor entry point must never use the network")

    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    try:
        import yfinance
    except ImportError:  # pragma: no cover - the dependency is declared
        pass
    else:
        monkeypatch.setattr(yfinance, "download", forbidden)

    statuses = [
        run_factor(FACTOR_NAME, AS_OF, data_root=root, universe_path=universe).cache_status,
        run_factor(FACTOR_NAME, AS_OF, data_root=root, universe_path=universe).cache_status,
        run_factor(
            FACTOR_NAME, AS_OF, data_root=root, universe_path=universe, force=True
        ).cache_status,
    ]

    assert statuses == ["miss", "hit", "forced"]
    assert set(statuses) == set(CACHE_STATUSES)


# ---------------------------------------------------------------------------
# 7. The command as a process, and the module boundary
# ---------------------------------------------------------------------------
def test_the_documented_command_runs_in_a_real_process(tmp_path: Path) -> None:
    """``python scripts/run_factor.py ...`` works from the project root."""
    root = tmp_path / "data"
    publish(root)
    universe = write_universe_file(tmp_path / "universe.csv", SYMBOLS)

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--factor",
            FACTOR_NAME,
            "--as-of",
            AS_OF.isoformat(),
            "--data-root",
            str(root),
            "--universe-path",
            str(universe),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=300,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    fields = summary_fields(completed.stdout)
    assert fields["factor_name"] == FACTOR_NAME
    assert fields["factor_version"] == FACTOR_VERSION
    assert fields["as_of_date"] == AS_OF.isoformat()
    assert fields["quality_status"] == "clean"
    assert fields["cache_status"] == "miss"
    assert Path(fields["stored_panel_path"]) == stored_panel_path(root)


def script_tree() -> ast.Module:
    """The parsed source of the command under test."""
    return ast.parse(SCRIPT.read_text(encoding="utf-8"))


def imported_modules(tree: ast.Module) -> set[str]:
    """Every module the script imports, by dotted name."""
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
    return modules


def referenced_names(tree: ast.Module) -> set[str]:
    """Every bare name and attribute the script's code mentions."""
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    names |= {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    return names


def string_constants(tree: ast.Module) -> set[str]:
    """Every string literal in the script (docstrings included)."""
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }


def test_the_script_imports_no_formula_and_no_network_module() -> None:
    """Composition only: no factor module, no downloader, no updater, no HTTP."""
    modules = imported_modules(script_tree())
    forbidden = (
        "factors",
        "yfinance",
        "httpx",
        "requests",
        "urllib",
        "http",
        "socket",
        "subprocess",
        "data_sys.providers",
        "data_sys.update",
        "data_sys.data_query",
        "data_sys.batch_query",
        "data_sys.planner",
        "data_sys.merge",
    )
    for module in sorted(modules):
        assert not any(
            module == banned or module.startswith(f"{banned}.") for banned in forbidden
        ), f"the entry point must not import {module!r}"

    assert {"factor_sys", "data_sys.config", "data_sys.market_calendar"} <= modules


def test_the_script_composes_the_public_interfaces() -> None:
    """Each layer is reached through its documented entry point, in one place."""
    names = referenced_names(script_tree())
    required = {
        "default_factor_registry",
        "StaticUniverseProvider",
        "ParquetMarketDataReader",
        "FactorRunner",
        "ParquetFactorStore",
        "QuickAnalyzer",
        "DefaultFactorQualityChecker",
        "exists",
        "read",
        "write",
        "path_for",
    }
    assert required <= names
    assert "FactorContext" not in names  # the runner builds it, never the CLI


def test_the_script_never_reproduces_the_store_layout() -> None:
    """The cache is keyed through the store's public API, not by rebuilding its paths."""
    tree = script_tree()

    private_members = {
        "_key",
        "_version_segment",
        "_panel_at",
        "_root",
        "_root_directory",
        "_atomic_write",
        "_scanned_keys",
    }
    assert private_members.isdisjoint(referenced_names(tree))

    # A reconstructed Hive path would have to spell these fragments out.
    constants = string_constants(tree)
    for fragment in ("factor_name=", "factor_version=", "year=", "as_of_date=", ".parquet"):
        assert fragment not in constants


def test_the_script_does_not_restate_the_formula_or_the_rules() -> None:
    """No price arithmetic, no quality code and no storage rule re-implemented."""
    tree = script_tree()

    # Only path joining ("/" on a Path), string concatenation and type unions are
    # allowed: a formula would need at least one of -, *, **, // or %.
    operators = {type(node.op) for node in ast.walk(tree) if isinstance(node, ast.BinOp)}
    assert operators <= {ast.Add, ast.BitOr, ast.Div}
    assert not operators & {ast.Sub, ast.Mult, ast.Pow, ast.FloorDiv, ast.Mod}

    # And no arithmetic operand is a number: nothing here counts or divides values.
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp):
            for operand in (node.left, node.right):
                if isinstance(operand, ast.Constant):
                    assert not isinstance(operand.value, (int, float))

    constants = string_constants(tree)
    assert "adjusted_close" not in constants
    assert "partial_eligible_coverage" not in constants
    assert "zero_eligible_coverage" not in constants

    # The research layers above this entry point need a date series, so they are
    # not reachable from a command that runs exactly one session.
    names = referenced_names(tree)
    assert {
        "ForwardReturnCalculator",
        "ICAnalyzer",
        "QuantileAnalyzer",
        "ICReport",
        "QuantileReport",
        "Backtest",
    }.isdisjoint(names)

    for banned in ("pandas", "numpy", "pyarrow", "duckdb"):
        assert banned not in imported_modules(tree)


def test_the_command_accepts_only_its_declared_flags() -> None:
    """One factor, one date, one root, one pool, one calendar, one force switch."""
    parser = build_parser()
    dests = {action.dest for action in parser._actions}

    assert dests == EXPECTED_DESTS
    assert {"start", "end", "horizon", "cache"}.isdisjoint(dests)
    assert (
        parser.parse_args(["--factor", FACTOR_NAME, "--as-of", AS_OF.isoformat(), "--force"]).force
        is True
    )


def test_the_outcome_refuses_parts_from_different_runs(market: Market) -> None:
    """A hand-built outcome cannot claim parts that disagree with each other."""
    outcome = run_factor(
        FACTOR_NAME,
        AS_OF,
        data_root=market.root,
        universe_path=market.universe,
    )
    other_day = replace(outcome.statistics, as_of_date=date(2024, 12, 30))

    with pytest.raises(RunFactorError, match="does not describe one run"):
        FactorRunOutcome(
            panel=outcome.panel,
            statistics=other_day,
            quality_report=None,
            stored_panel_path=outcome.stored_panel_path,
            cache_status=outcome.cache_status,
        )

    with pytest.raises(RunFactorError, match="panel must be a FactorPanel"):
        FactorRunOutcome(
            panel="not a panel",  # type: ignore[arg-type]
            statistics=outcome.statistics,
            quality_report=None,
            stored_panel_path=outcome.stored_panel_path,
            cache_status=outcome.cache_status,
        )

    with pytest.raises(RunFactorError, match="cache_status must be one of"):
        FactorRunOutcome(
            panel=outcome.panel,
            statistics=outcome.statistics,
            quality_report=None,
            stored_panel_path=outcome.stored_panel_path,
            cache_status="stale",
        )

    report = FactorQualityReport(
        as_of_date=AS_OF,
        factor_name=FACTOR_NAME,
        factor_version=FACTOR_VERSION,
        universe_size=len(outcome.panel.universe),
        eligible_count=len(outcome.panel.universe),
        eligible_coverage=1.0,
        issues=(
            FactorQualityIssue(
                code="partial_eligible_coverage",
                severity="warning",
                message="a finding about this very panel",
                affected_symbols=(),
            ),
        ),
    )
    accepted = FactorRunOutcome(
        panel=outcome.panel,
        statistics=outcome.statistics,
        quality_report=report,
        stored_panel_path=outcome.stored_panel_path,
        cache_status="forced",
    )
    assert accepted.quality_status == "reported"
    assert accepted.issues == report.issues
    assert accepted.cache_status == "forced"

    stale = replace(report, factor_version="9.9.9")
    with pytest.raises(RunFactorError, match="does not describe one run"):
        FactorRunOutcome(
            panel=outcome.panel,
            statistics=outcome.statistics,
            quality_report=stale,
            stored_panel_path=outcome.stored_panel_path,
            cache_status=outcome.cache_status,
        )
