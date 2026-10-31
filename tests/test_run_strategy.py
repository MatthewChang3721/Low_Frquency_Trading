"""End-to-end tests for the minimal single-date strategy entry point.

Every test here runs against **real files in ``tmp_path``**: a real static universe
CSV and a real Hive-partitioned Parquet factor store written through
:class:`~factor_sys.factor_store.ParquetFactorStore`, then the real registry, the
real rule, the real runner and the real formatter.  Nothing reads the project's
``data/``, nothing touches the network and nothing is stubbed except where a test
proves a layer is *not* reached (a panel read, a panel write, a socket).

Covered in this order:

* the success path: arguments -> ``FactorRequirement`` -> registry -> runner ->
  result, with the ranking recomputed independently in plain pandas, the selected
  rows printed in rank order and the complete strategy/factor provenance attached,
* the rule's own selection semantics through the command: descending score,
  ascending-symbol tie-break, ``top_n`` larger than the pool and a cross-section
  with nothing rankable (a successful empty selection, not a crash),
* exact-version reading: two versions of one factor name stored side by side, only
  the requested one read, never a fallback,
* the failure modes that must never look like success: a missing panel, a
  non-positive or unparseable ``top_n``, a missing or unknown flag, a malformed
  date and a blank path -- each with its documented exit code and one ``error:``
  line, and none of them touching the store,
* ``--data-root`` and ``--universe-path``: the flags really decide which store and
  which pool the run uses,
* the printed contract: the exact stable text, six decimals, selected rows only,
  no frame index, and every exact ``name/version`` requirement,
* ``StrategyRunOutcome``: frozen, self-consistent, and isolated from the frames it
  hands out,
* the offline boundary, asserted both at runtime (no socket, no factor runner, no
  panel write, no ``raw/`` or ``.staging/`` directory) and on the module's AST (no
  Parquet reading, no data provider, no private member, no portfolio or trading
  logic),
* the documented command in a real process, started from the project root.
"""

from __future__ import annotations

import ast
import inspect
import socket
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import FrozenInstanceError, fields, replace
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

import scripts.run_strategy as run_strategy_module
from data_sys.config import PipelineConfig
from factor_sys import FactorPanel, FactorRunner, ParquetFactorStore, UniverseSnapshot
from scripts.run_strategy import (
    DEFAULT_STRATEGY_VERSION,
    EMPTY_SELECTION_TEXT,
    RunStrategyError,
    Selection,
    StrategyRunOutcome,
    build_parser,
    format_summary,
    main,
    run_strategy,
)
from strategy_sys import (
    STRATEGY_RESULT_COLUMNS,
    FactorRequirement,
    StrategyRegistry,
    StrategyResult,
    StrategyRunner,
    StrategySpec,
)
from strategy_sys.strategies import TopNByFactorStrategy
from tests.helpers import write_universe_file

#: The command under test, as a file (for the boundary and subprocess tests).
SCRIPT: Path = Path(run_strategy_module.__file__).resolve()

#: The project root the documented command is run from.
PROJECT_ROOT: Path = SCRIPT.parents[1]

#: A real NYSE session; the panels the fixtures store are keyed by it.
AS_OF: date = date(2024, 1, 5)

#: Exact factor identities: the one the fixtures store, and a second version.
MOMENTUM = FactorRequirement(factor_name="momentum_12_1", factor_version="1.0.0")
MOMENTUM_V2 = FactorRequirement(factor_name="momentum_12_1", factor_version="2.0.0")

#: The pool, in the order the universe reader publishes it (sorted, upper case).
SYMBOLS: tuple[str, ...] = (
    "AAPL",
    "ADBE",
    "AMD",
    "AMZN",
    "CRM",
    "GOOGL",
    "INTC",
    "META",
    "MSFT",
    "NVDA",
    "ORCL",
    "TSLA",
)

#: One score per symbol of the pool: no ties, so the ranking is unambiguous.
SCORES: dict[str, float] = {
    "AAPL": 0.05,
    "ADBE": 0.58,
    "AMD": 0.44,
    "AMZN": 0.55,
    "CRM": 0.48,
    "GOOGL": 0.62,
    "INTC": 0.12,
    "META": 0.70,
    "MSFT": 0.80,
    "NVDA": 0.95,
    "ORCL": 0.36,
    "TSLA": 0.30,
}

#: The three symbols :data:`SCORES` ranks highest, in rank order.
TOP_THREE: tuple[str, ...] = ("NVDA", "MSFT", "META")

#: Fields the printed summary must always carry before ``selection:``, in order.
SUMMARY_FIELDS: tuple[str, ...] = (
    "strategy_name",
    "strategy_version",
    "as_of_date",
    "universe_size",
    "selected_count",
)

#: The flag destinations the command accepts -- and nothing else.
EXPECTED_DESTS: frozenset[str] = frozenset(
    {
        "factor",
        "factor_version",
        "top_n",
        "as_of",
        "strategy_version",
        "data_root",
        "universe_path",
        "help",
    }
)


# ---------------------------------------------------------------------------
# Fixtures: real files, no project data, no network
# ---------------------------------------------------------------------------
def write_universe(
    root: Path, symbols: Sequence[str] = SYMBOLS, name: str = "universe.csv"
) -> Path:
    """Write a real universe CSV (the contract needs at least ten rows)."""
    return write_universe_file(root / name, symbols)


def panels_root(data_root: Path) -> ParquetFactorStore:
    """The factor store a run against ``data_root`` would read."""
    return ParquetFactorStore(data_root / "factors" / "panels")


def panel_files(data_root: Path) -> dict[Path, int]:
    """Every stored panel file of ``data_root`` and its size."""
    root = data_root / "factors" / "panels"
    if not root.exists():
        return {}
    return {path: path.stat().st_size for path in sorted(root.rglob("*")) if path.is_file()}


def panel_bytes(data_root: Path) -> dict[Path, bytes]:
    """Every stored panel file of ``data_root`` with its exact content."""
    return {path: path.read_bytes() for path in panel_files(data_root)}


def write_panel(
    data_root: Path,
    requirement: FactorRequirement = MOMENTUM,
    scores: Mapping[str, float] | None = None,
    symbols: Sequence[str] = SYMBOLS,
    eligible: Mapping[str, bool] | None = None,
    as_of: date = AS_OF,
) -> Path:
    """Store one real panel for ``requirement`` under ``data_root``'s factor store."""
    values = SCORES if scores is None else scores
    flags: Mapping[str, bool] = {} if eligible is None else eligible
    frame = pd.DataFrame(
        {
            "as_of_date": [pd.Timestamp(as_of)] * len(symbols),
            "symbol": list(symbols),
            "factor_name": [requirement.factor_name] * len(symbols),
            "factor_version": [requirement.factor_version] * len(symbols),
            "raw_value": [values[symbol] for symbol in symbols],
            "is_eligible": [flags.get(symbol, True) for symbol in symbols],
        }
    )
    panel = FactorPanel(frame, universe=list(symbols), as_of_date=as_of)
    return panels_root(data_root).write(panel)


def scenario(
    root: Path,
    scores: Mapping[str, float] | None = None,
    symbols: Sequence[str] = SYMBOLS,
    eligible: Mapping[str, bool] | None = None,
    requirement: FactorRequirement = MOMENTUM,
) -> tuple[Path, Path]:
    """A universe and one stored panel: ``(data_root, universe_path)``."""
    data_root = root / "data"
    universe = write_universe(root, symbols)
    write_panel(data_root, requirement, scores, symbols, eligible)
    return data_root, universe


def command(
    data_root: Path | None = None,
    universe_path: Path | None = None,
    *,
    factor: str = "momentum_12_1",
    factor_version: str = "1.0.0",
    top_n: str = "3",
    as_of: str = AS_OF.isoformat(),
) -> list[str]:
    """The documented command line, with only the parts under test changed."""
    argv = [
        "--factor",
        factor,
        "--factor-version",
        factor_version,
        "--top-n",
        top_n,
        "--as-of",
        as_of,
    ]
    if data_root is not None:
        argv += ["--data-root", str(data_root)]
    if universe_path is not None:
        argv += ["--universe-path", str(universe_path)]
    return argv


def run_cli(argv: Sequence[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, str, str]:
    """Run ``main`` in this process and return ``(exit_code, stdout, stderr)``."""
    code = main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def ranking(scores: Mapping[str, float], symbols: Sequence[str] = SYMBOLS) -> list[str]:
    """The rule's documented order, recomputed independently of the strategy."""
    return sorted(symbols, key=lambda symbol: (-scores[symbol], symbol))


def forbid_panel_access(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make any later panel read or write fail the test loudly."""

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("an invalid request must not reach the factor store")

    monkeypatch.setattr(ParquetFactorStore, "read", forbidden)
    monkeypatch.setattr(ParquetFactorStore, "write", forbidden)


def forbid_the_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make any later strategy run fail the test loudly."""

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("an invalid request must not reach the strategy runner")

    monkeypatch.setattr(StrategyRunner, "run", forbidden)


# ---------------------------------------------------------------------------
# 1. The success path
# ---------------------------------------------------------------------------
def test_end_to_end_runs_the_registry_built_strategy_through_the_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One command: args -> registry -> runner -> result -> printed selection."""
    data_root, universe = scenario(tmp_path)

    runs: list[tuple[object, object, object]] = []
    created: list[object] = []
    original_run = StrategyRunner.run
    original_create = StrategyRegistry.create

    def spy_run(self: StrategyRunner, strategy: object, as_of_date: object) -> object:
        result = original_run(self, strategy, as_of_date)
        runs.append((strategy, as_of_date, result))
        return result

    def spy_create(self: StrategyRegistry, spec: StrategySpec) -> object:
        strategy = original_create(self, spec)
        created.append(strategy)
        return strategy

    monkeypatch.setattr(StrategyRunner, "run", spy_run)
    monkeypatch.setattr(StrategyRegistry, "create", spy_create)

    outcome = run_strategy(
        factor_name="momentum_12_1",
        factor_version="1.0.0",
        top_n=3,
        as_of_date=AS_OF.isoformat(),
        data_root=str(data_root),
        universe_path=universe,
    )

    # The run went through the runner once, with the instance the registry built,
    # and the date was handed over exactly as the caller gave it.
    assert len(runs) == 1
    strategy, as_of_date, result = runs[0]
    assert isinstance(strategy, TopNByFactorStrategy)
    assert created == [strategy]
    assert as_of_date == AS_OF.isoformat()
    assert outcome.result is result

    # The complete provenance travels with the selection.
    assert outcome.strategy_name == "top_3_by_momentum_12_1"
    assert outcome.strategy_version == "1.0.0"
    assert outcome.strategy_spec == StrategySpec(
        name="top_3_by_momentum_12_1",
        version="1.0.0",
        required_factors=(MOMENTUM,),
    )
    assert outcome.as_of_date == AS_OF
    assert outcome.universe_size == len(SYMBOLS)
    assert outcome.selected_count == 3
    assert outcome.factor_requirements == (MOMENTUM,)
    assert outcome.result.factor_requirements == (MOMENTUM,)
    assert outcome.data_root == data_root

    # The selection itself: rank order, exact scores, and the rows the rule flagged.
    # ``selected_symbols`` keeps the result's row order (the universe's), so the
    # ranked view is ``selections``.
    assert outcome.selected_symbols == ("META", "MSFT", "NVDA")
    assert [(row.rank, row.symbol) for row in outcome.selections] == [
        (1, "NVDA"),
        (2, "MSFT"),
        (3, "META"),
    ]
    assert [row.score for row in outcome.selections] == pytest.approx([0.95, 0.80, 0.70])
    assert all(isinstance(row, Selection) for row in outcome.selections)

    frame = outcome.result.frame
    assert tuple(frame.columns) == STRATEGY_RESULT_COLUMNS
    assert list(frame["symbol"]) == list(SYMBOLS)  # universe order is untouched
    assert int(frame["is_selected"].sum()) == 3


def test_every_row_is_ranked_from_the_stored_panel(tmp_path: Path) -> None:
    """Scores and ranks are checked against a plain-``sorted`` reference."""
    data_root, universe = scenario(tmp_path)

    outcome = run_strategy(
        factor_name="momentum_12_1",
        factor_version="1.0.0",
        top_n=3,
        as_of_date=AS_OF,
        data_root=data_root,
        universe_path=universe,
    )

    frame = outcome.result.frame.set_index("symbol")
    expected = {symbol: rank for rank, symbol in enumerate(ranking(SCORES), start=1)}
    assert {symbol: int(frame.loc[symbol, "rank"]) for symbol in SYMBOLS} == expected
    assert {symbol: float(frame.loc[symbol, "score"]) for symbol in SYMBOLS} == pytest.approx(
        SCORES
    )
    assert set(frame.index[frame["is_selected"]]) == set(TOP_THREE)


def test_the_registered_factory_builds_a_new_strategy_per_run(tmp_path: Path) -> None:
    """Two runs never share the one object the registry hands out."""
    data_root, universe = scenario(tmp_path)

    first = run_strategy(
        factor_name="momentum_12_1",
        factor_version="1.0.0",
        top_n=3,
        as_of_date=AS_OF,
        data_root=data_root,
        universe_path=universe,
    )
    second = run_strategy(
        factor_name="momentum_12_1",
        factor_version="1.0.0",
        top_n=3,
        as_of_date=AS_OF,
        data_root=data_root,
        universe_path=universe,
    )

    assert first.result is not second.result
    assert first.result.strategy_spec == second.result.strategy_spec
    assert first.selected_symbols == second.selected_symbols


def test_a_strategy_version_of_its_own_travels_into_the_result(tmp_path: Path) -> None:
    """``--strategy-version`` becomes part of the identity, not just a label."""
    data_root, universe = scenario(tmp_path)

    outcome = run_strategy(
        factor_name="momentum_12_1",
        factor_version="1.0.0",
        top_n=3,
        as_of_date=AS_OF,
        strategy_version="2.1.0",
        data_root=data_root,
        universe_path=universe,
    )

    assert outcome.strategy_version == "2.1.0"
    assert outcome.result.strategy_spec.version == "2.1.0"
    assert set(outcome.result.frame["strategy_version"]) == {"2.1.0"}


# ---------------------------------------------------------------------------
# 2. The rule's selection semantics, seen through the command
# ---------------------------------------------------------------------------
def test_equal_scores_break_the_tie_by_symbol(tmp_path: Path) -> None:
    """A plateau is ordered by symbol, so two runs always print the same table."""
    plateau = {symbol: 0.10 for symbol in SYMBOLS}
    plateau["META"] = 0.50
    plateau["MSFT"] = 0.50
    data_root, universe = scenario(tmp_path, scores=plateau)

    outcome = run_strategy(
        factor_name="momentum_12_1",
        factor_version="1.0.0",
        top_n=2,
        as_of_date=AS_OF,
        data_root=data_root,
        universe_path=universe,
    )

    assert [(row.rank, row.symbol) for row in outcome.selections] == [(1, "META"), (2, "MSFT")]
    assert [row.score for row in outcome.selections] == pytest.approx([0.50, 0.50])


def test_a_selection_larger_than_the_pool_selects_every_rankable_symbol(
    tmp_path: Path,
) -> None:
    """``--top-n`` above the pool size is not an error: everything rankable is in."""
    data_root, universe = scenario(tmp_path)

    outcome = run_strategy(
        factor_name="momentum_12_1",
        factor_version="1.0.0",
        top_n=50,
        as_of_date=AS_OF,
        data_root=data_root,
        universe_path=universe,
    )

    expected = ranking(SCORES)
    assert outcome.selected_count == len(SYMBOLS)
    assert [row.symbol for row in outcome.selections] == expected
    assert [row.rank for row in outcome.selections] == list(range(1, len(SYMBOLS) + 1))
    assert outcome.selected_symbols == SYMBOLS  # the frame keeps universe order
    assert int(outcome.result.frame["is_selected"].sum()) == len(SYMBOLS)


def test_nothing_rankable_is_a_successful_empty_selection(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An all-ineligible cross-section selects nothing and the command still succeeds."""
    ineligible = {symbol: False for symbol in SYMBOLS}
    data_root, universe = scenario(tmp_path, eligible=ineligible)

    outcome = run_strategy(
        factor_name="momentum_12_1",
        factor_version="1.0.0",
        top_n=5,
        as_of_date=AS_OF,
        data_root=data_root,
        universe_path=universe,
    )

    assert outcome.selected_count == 0
    assert outcome.selections == ()
    assert outcome.selected_symbols == ()
    frame = outcome.result.frame
    assert not bool(frame["is_selected"].any())
    assert frame["rank"].isna().all()
    assert len(frame) == len(SYMBOLS)  # every symbol keeps its row

    text = format_summary(outcome)
    assert text.splitlines()[-2:] == ["rank  symbol  score", EMPTY_SELECTION_TEXT]

    code, out, err = run_cli(command(data_root, universe, top_n="5"), capsys)
    assert code == 0
    assert err == ""
    assert "selected_count: 0" in out.splitlines()
    assert out.splitlines()[-2:] == ["rank  symbol  score", EMPTY_SELECTION_TEXT]


# ---------------------------------------------------------------------------
# 3. One exact factor version, and never a fallback
# ---------------------------------------------------------------------------
def test_only_the_requested_factor_version_is_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two versions stored side by side: the flag decides, and nothing else is read."""
    data_root, universe = scenario(tmp_path)
    inverted = {symbol: -value for symbol, value in SCORES.items()}
    write_panel(data_root, requirement=MOMENTUM_V2, scores=inverted)
    before = panel_bytes(data_root)

    reads: list[tuple[str, str, object]] = []
    original_read = ParquetFactorStore.read

    def spy_read(
        self: ParquetFactorStore, factor_name: str, factor_version: str, as_of_date: object
    ) -> FactorPanel:
        reads.append((factor_name, factor_version, as_of_date))
        return original_read(self, factor_name, factor_version, as_of_date)

    monkeypatch.setattr(ParquetFactorStore, "read", spy_read)

    latest = run_strategy(
        factor_name="momentum_12_1",
        factor_version="1.0.0",
        top_n=2,
        as_of_date=AS_OF,
        data_root=data_root,
        universe_path=universe,
    )

    assert reads == [("momentum_12_1", "1.0.0", AS_OF)]
    assert latest.factor_requirements == (MOMENTUM,)
    assert latest.result.factor_requirements == (MOMENTUM,)
    assert [row.symbol for row in latest.selections] == ["NVDA", "MSFT"]
    assert (latest.result.frame["strategy_version"] == "1.0.0").all()

    older = run_strategy(
        factor_name="momentum_12_1",
        factor_version="2.0.0",
        top_n=2,
        as_of_date=AS_OF,
        data_root=data_root,
        universe_path=universe,
    )

    assert reads[-1] == ("momentum_12_1", "2.0.0", AS_OF)
    assert older.factor_requirements == (MOMENTUM_V2,)
    assert older.result.factor_requirements == (MOMENTUM_V2,)
    assert [row.symbol for row in older.selections] == ["AAPL", "INTC"]
    assert older.strategy_name == "top_2_by_momentum_12_1"
    assert older.result.strategy_spec.required_factors == (MOMENTUM_V2,)

    # Reading never rewrites or adds a panel.
    assert panel_bytes(data_root) == before


def test_a_missing_panel_fails_with_one_error_line_and_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """No stored panel: exit ``1``, one ``error:`` line, and no new directory."""
    universe = write_universe(tmp_path)
    data_root = tmp_path / "data"

    code, out, err = run_cli(command(data_root, universe), capsys)

    assert code == 1
    assert out == ""
    lines = err.splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("error: ")
    assert "momentum_12_1" in lines[0]
    assert "1.0.0" in lines[0]
    assert AS_OF.isoformat() in lines[0]
    assert "Traceback" not in err
    assert not data_root.exists()


def test_a_panel_for_another_session_is_not_reused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The date is passed through unchanged, so a neighbouring panel is not a hit."""
    data_root, universe = scenario(tmp_path)

    code, out, err = run_cli(command(data_root, universe, as_of="2024-01-08"), capsys)

    assert code == 1
    assert out == ""
    assert err.startswith("error: ")
    assert "2024-01-08" in err
    stored = sorted(path.name for path in panel_files(data_root))
    assert stored == [f"as_of_date={AS_OF.isoformat()}.parquet"]


def test_a_weekend_panel_is_not_reused_for_the_next_session(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Nothing silently moves the requested date to a session that has a panel."""
    data_root, universe = scenario(tmp_path)

    code, _, err = run_cli(command(data_root, universe, as_of="2024-01-06"), capsys)

    assert code == 1
    assert "2024-01-06" in err


# ---------------------------------------------------------------------------
# 4. Invalid requests: clear exit codes, no storage touched
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("top_n", ["0", "-3"])
def test_a_non_positive_top_n_is_a_domain_error(
    top_n: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The rule refuses the size itself; the store is never opened."""
    universe = write_universe(tmp_path)
    data_root = tmp_path / "data"
    forbid_panel_access(monkeypatch)
    forbid_the_runner(monkeypatch)

    code, out, err = run_cli(command(data_root, universe, top_n=top_n), capsys)

    assert code == 1
    assert out == ""
    lines = err.splitlines()
    assert lines[0].startswith("error: ")
    assert "top_n" in err
    # The rule collects its problems in the project's bullet style: the detail is
    # kept, but there is never a traceback.
    assert all(line.startswith("  - ") for line in lines[1:])
    assert "Traceback" not in err
    assert not data_root.exists()


@pytest.mark.parametrize("top_n", ["abc", "1.5", "3.0", ""])
def test_an_unparseable_top_n_is_an_argparse_error(
    top_n: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--top-n`` is an ``int`` flag, so argparse owns its own exit code ``2``."""
    with pytest.raises(SystemExit) as excinfo:
        main(command(top_n=top_n))

    assert excinfo.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "usage:" in captured.err


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["--factor", "momentum_12_1"],
        ["--factor", "momentum_12_1", "--factor-version", "1.0.0"],
        ["--factor", "momentum_12_1", "--factor-version", "1.0.0", "--top-n", "3"],
        [
            "--factor",
            "momentum_12_1",
            "--factor-version",
            "1.0.0",
            "--top-n",
            "3",
            "--as-of",
            AS_OF.isoformat(),
            "--frequency",
            "weekly",
        ],
        [
            "--factor",
            "momentum_12_1",
            "--factor-version",
            "1.0.0",
            "--top-n",
            "3",
            "--as-of",
            AS_OF.isoformat(),
            "--force",
        ],
    ],
)
def test_a_missing_or_unknown_flag_is_an_argparse_error(
    argv: Sequence[str], capsys: pytest.CaptureFixture[str]
) -> None:
    """Nothing beyond the declared flags is accepted, and no run starts."""
    with pytest.raises(SystemExit) as excinfo:
        main(list(argv))

    assert excinfo.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "usage:" in captured.err


@pytest.mark.parametrize("value", ["2024-02-30", "31/12/2024", "2024-1-5", "", "2024-01-05T00:00"])
def test_a_malformed_as_of_date_is_a_domain_error(
    value: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only a real ISO day is a decision date; nothing is adjusted or recomputed."""
    data_root, universe = scenario(tmp_path)
    before = panel_bytes(data_root)
    forbid_panel_access(monkeypatch)

    code, out, err = run_cli(command(data_root, universe, as_of=value), capsys)

    assert code == 1
    assert out == ""
    lines = err.splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("error: ")
    assert "as_of_date" in lines[0]
    assert panel_bytes(data_root) == before


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ({"factor": ""}, "factor"),
        ({"factor_version": ""}, "version"),
        ({"data_root": "   "}, "data-root"),
    ],
)
def test_a_blank_argument_is_a_domain_error(
    values: Mapping[str, str],
    expected: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A blank value is refused by the layer that owns it, never silently defaulted."""
    universe = write_universe(tmp_path)
    forbid_panel_access(monkeypatch)
    forbid_the_runner(monkeypatch)

    argv = [
        "--factor",
        values.get("factor", "momentum_12_1"),
        "--factor-version",
        values.get("factor_version", "1.0.0"),
        "--top-n",
        "3",
        "--as-of",
        AS_OF.isoformat(),
        "--universe-path",
        str(universe),
    ]
    if "data_root" in values:
        argv += ["--data-root", values["data_root"]]

    code, out, err = run_cli(argv, capsys)

    assert code == 1
    assert out == ""
    lines = err.splitlines()
    assert lines[0].startswith("error: ")
    assert expected in err
    assert all(line.startswith("  - ") for line in lines[1:])
    assert "Traceback" not in err


# ---------------------------------------------------------------------------
# 5. Custom paths decide the run
# ---------------------------------------------------------------------------
def test_a_data_root_of_another_type_is_refused() -> None:
    """A path is a path: an int is refused before any layer is reached."""
    with pytest.raises(RunStrategyError) as excinfo:
        run_strategy(
            factor_name="momentum_12_1",
            factor_version="1.0.0",
            top_n=3,
            as_of_date=AS_OF,
            data_root=123,  # type: ignore[arg-type]
        )

    assert "data_root must be a path or a string" in str(excinfo.value)


def test_custom_data_root_and_universe_path_decide_the_run(tmp_path: Path) -> None:
    """``--data-root`` picks the store, ``--universe-path`` picks the pool."""
    narrow_symbols = SYMBOLS[:10]
    broad_universe = write_universe(tmp_path, name="broad.csv")
    narrow_universe = write_universe(tmp_path, narrow_symbols, name="narrow.csv")

    first = tmp_path / "first"
    second = tmp_path / "second"
    write_panel(first)
    write_panel(
        second,
        symbols=narrow_symbols,
        scores={symbol: -SCORES[symbol] for symbol in narrow_symbols},
    )

    broad = run_strategy(
        factor_name="momentum_12_1",
        factor_version="1.0.0",
        top_n=3,
        as_of_date=AS_OF,
        data_root=first,
        universe_path=broad_universe,
    )
    narrow = run_strategy(
        factor_name="momentum_12_1",
        factor_version="1.0.0",
        top_n=2,
        as_of_date=AS_OF,
        data_root=second,
        universe_path=narrow_universe,
    )

    assert broad.data_root == first
    assert narrow.data_root == second
    assert broad.universe_size == len(SYMBOLS)
    assert narrow.universe_size == len(narrow_symbols)
    assert broad.selected_symbols == ("META", "MSFT", "NVDA")
    assert narrow.selected_symbols == ("AAPL", "INTC")
    assert len(panel_files(first)) == 1
    assert len(panel_files(second)) == 1


def test_the_command_reads_from_the_custom_root_and_pool(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The flags work through ``main`` too: one panel, one pool, one summary."""
    narrow_symbols = SYMBOLS[:10]
    data_root = tmp_path / "elsewhere"
    universe = write_universe(tmp_path, narrow_symbols, name="pool.csv")
    write_panel(data_root, symbols=narrow_symbols)

    code, out, err = run_cli(command(data_root, universe, top_n="2"), capsys)

    assert code == 0
    assert err == ""
    assert "universe_size: 10" in out.splitlines()
    assert out.splitlines()[-2:] == [
        "1     NVDA    0.950000",
        "2     MSFT    0.800000",
    ]


# ---------------------------------------------------------------------------
# 6. The printed contract
# ---------------------------------------------------------------------------
def test_the_summary_is_the_documented_stable_text(tmp_path: Path) -> None:
    """The whole text, exactly: a caller may parse it or diff it."""
    data_root, universe = scenario(tmp_path)

    outcome = run_strategy(
        factor_name="momentum_12_1",
        factor_version="1.0.0",
        top_n=3,
        as_of_date=AS_OF,
        data_root=data_root,
        universe_path=universe,
    )

    assert format_summary(outcome) == "\n".join(
        [
            "strategy_name: top_3_by_momentum_12_1",
            "strategy_version: 1.0.0",
            "as_of_date: 2024-01-05",
            "universe_size: 12",
            "selected_count: 3",
            "required_factor: momentum_12_1/1.0.0",
            "selection:",
            "rank  symbol  score",
            "1     NVDA    0.950000",
            "2     MSFT    0.800000",
            "3     META    0.700000",
        ]
    )


def test_the_metadata_block_is_always_in_the_documented_order(tmp_path: Path) -> None:
    """Five fixed fields, then one line per exact requirement, then the table."""
    data_root, universe = scenario(tmp_path)

    outcome = run_strategy(
        factor_name="momentum_12_1",
        factor_version="1.0.0",
        top_n=3,
        as_of_date=AS_OF,
        data_root=data_root,
        universe_path=universe,
    )

    lines = format_summary(outcome).splitlines()
    assert [line.split(":")[0] for line in lines[:5]] == list(SUMMARY_FIELDS)
    assert lines[5] == "required_factor: momentum_12_1/1.0.0"
    assert lines[6] == "selection:"
    assert lines[7] == "rank  symbol  score"
    assert len(lines) == 8 + 3


def test_the_summary_lists_only_selected_symbols_with_six_decimals(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Two selected rows, six decimals each, and no other symbol anywhere."""
    data_root, universe = scenario(tmp_path)

    code, out, err = run_cli(command(data_root, universe, top_n="2"), capsys)

    assert code == 0
    assert err == ""
    lines = out.splitlines()
    table = lines[lines.index("rank  symbol  score") + 1 :]
    assert len(table) == 2
    assert table[0].split() == ["1", "NVDA", "0.950000"]
    assert table[1].split() == ["2", "MSFT", "0.800000"]
    for line in table:
        assert len(line.split()[2].split(".")[1]) == 6

    for symbol in SYMBOLS:
        if symbol not in {"NVDA", "MSFT"}:
            assert symbol not in out
    assert "dtype" not in out
    assert "Name:" not in out
    assert "nan" not in out.lower()
    assert out.endswith("\n")


# ---------------------------------------------------------------------------
# 7. The outcome object
# ---------------------------------------------------------------------------
def make_result(
    *,
    requirements: Sequence[FactorRequirement] = (MOMENTUM,),
    name: str = "top_3_by_momentum_12_1",
    version: str = "1.0.0",
    symbols: Sequence[str] = SYMBOLS[:3],
    scores: Sequence[float] = (0.50, 0.40, 0.30),
    ranks: Sequence[int | None] = (1, 2, 3),
    selected: Sequence[bool] = (True, True, True),
) -> StrategyResult:
    """A hand-built, contract-valid result -- for the outcome's own checks."""
    spec = StrategySpec(name=name, version=version, required_factors=tuple(requirements))
    universe = UniverseSnapshot(
        as_of_date=AS_OF, symbols=tuple(symbols), source="config/universe_seed.csv"
    )
    frame = pd.DataFrame(
        {
            "as_of_date": [pd.Timestamp(AS_OF)] * len(symbols),
            "strategy_name": [spec.name] * len(symbols),
            "strategy_version": [spec.version] * len(symbols),
            "symbol": list(symbols),
            "score": pd.Series(scores, dtype="float64"),
            "rank": pd.Series(ranks, dtype="Int64"),
            "is_selected": pd.Series(selected, dtype="bool"),
        }
    )
    return StrategyResult(frame, universe=universe, strategy_spec=spec, as_of_date=AS_OF)


def make_outcome(
    *,
    result: StrategyResult | None = None,
    selected_count: int = 3,
    factor_requirements: tuple[FactorRequirement, ...] = (MOMENTUM,),
    data_root: Path = Path("data"),
) -> StrategyRunOutcome:
    """An outcome built by hand, for the tests that must refuse one."""
    return StrategyRunOutcome(
        result=make_result() if result is None else result,
        selected_count=selected_count,
        factor_requirements=factor_requirements,
        data_root=data_root,
    )


def test_every_exact_factor_requirement_is_reported() -> None:
    """A rule with two factors prints both exact ``name/version`` pairs."""
    second = FactorRequirement("book_to_price", "2.0.0")
    result = make_result(
        requirements=(MOMENTUM, second),
        name="blend_of_two",
        scores=(0.50, float("nan"), 0.30),
        ranks=(1, None, 2),
        selected=(True, False, True),
    )
    outcome = make_outcome(result=result, selected_count=2, factor_requirements=(MOMENTUM, second))

    lines = format_summary(outcome).splitlines()
    assert [line for line in lines if line.startswith("required_factor:")] == [
        "required_factor: momentum_12_1/1.0.0",
        "required_factor: book_to_price/2.0.0",
    ]
    assert outcome.selections == (Selection(1, "AAPL", 0.50), Selection(2, "AMD", 0.30))


def test_the_outcome_holds_no_duplicated_frame(tmp_path: Path) -> None:
    """Four fields, none of them a DataFrame: the frame stays in the result."""
    data_root, universe = scenario(tmp_path)

    outcome = run_strategy(
        factor_name="momentum_12_1",
        factor_version="1.0.0",
        top_n=3,
        as_of_date=AS_OF,
        data_root=data_root,
        universe_path=universe,
    )

    names = {field.name for field in fields(StrategyRunOutcome)}
    assert names == {"result", "selected_count", "factor_requirements", "data_root"}
    assert not any(isinstance(getattr(outcome, name), pd.DataFrame) for name in names)


def test_the_outcome_is_frozen_and_checked(tmp_path: Path) -> None:
    """No attribute can be changed, and an inconsistent copy is refused."""
    data_root, universe = scenario(tmp_path)

    outcome = run_strategy(
        factor_name="momentum_12_1",
        factor_version="1.0.0",
        top_n=3,
        as_of_date=AS_OF,
        data_root=data_root,
        universe_path=universe,
    )

    with pytest.raises(FrozenInstanceError):
        outcome.selected_count = 0
    with pytest.raises(FrozenInstanceError):
        outcome.result = None

    assert replace(outcome, selected_count=outcome.selected_count) == outcome
    with pytest.raises(RunStrategyError):
        replace(outcome, selected_count=2)


def test_the_outcome_repr_names_the_run(tmp_path: Path) -> None:
    """The repr is a label for a log line, not a dump of the frame."""
    data_root, universe = scenario(tmp_path)

    outcome = run_strategy(
        factor_name="momentum_12_1",
        factor_version="1.0.0",
        top_n=3,
        as_of_date=AS_OF,
        data_root=data_root,
        universe_path=universe,
    )

    text = repr(outcome)
    assert text.startswith("StrategyRunOutcome(")
    assert "top_3_by_momentum_12_1" in text
    assert "'1.0.0'" in text
    assert "'2024-01-05'" in text
    assert "selected_count=3" in text
    assert "AAPL" not in text


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"result": pd.DataFrame()}, "result must be a StrategyResult"),
        ({"data_root": "data"}, "data_root must be a Path"),
        ({"factor_requirements": [MOMENTUM]}, "factor_requirements must be a tuple"),
        ({"factor_requirements": (MOMENTUM_V2,)}, "factor_requirements are"),
        ({"factor_requirements": ("momentum_12_1/1.0.0",)}, "FactorRequirement values"),
        ({"selected_count": 2}, "selected_count is 2"),
        ({"selected_count": True}, "selected_count must be an int"),
        ({"selected_count": 2.0}, "selected_count must be an int"),
    ],
)
def test_the_outcome_refuses_parts_that_do_not_describe_one_run(
    overrides: Mapping[str, object], expected: str
) -> None:
    """A hand-built outcome that contradicts itself never reaches the printer."""
    values: dict[str, object] = {
        "result": make_result(),
        "selected_count": 3,
        "factor_requirements": (MOMENTUM,),
        "data_root": Path("data"),
    }
    values.update(overrides)

    with pytest.raises(RunStrategyError) as excinfo:
        StrategyRunOutcome(**values)  # type: ignore[arg-type]

    assert expected in str(excinfo.value)


def test_a_frame_handed_out_by_the_outcome_cannot_change_it(tmp_path: Path) -> None:
    """``result.frame`` is a copy, so printing and inspection stay consistent."""
    data_root, universe = scenario(tmp_path)

    outcome = run_strategy(
        factor_name="momentum_12_1",
        factor_version="1.0.0",
        top_n=3,
        as_of_date=AS_OF,
        data_root=data_root,
        universe_path=universe,
    )
    before = format_summary(outcome)

    tampered = outcome.result.frame
    tampered.loc[0, "score"] = 999.0
    tampered.loc[0, "is_selected"] = False
    tampered.loc[0, "rank"] = 99

    assert format_summary(outcome) == before
    assert outcome.selected_count == 3
    assert [row.symbol for row in outcome.selections] == ["NVDA", "MSFT", "META"]
    assert outcome.result.frame is not tampered


# ---------------------------------------------------------------------------
# 8. The command line and the public API
# ---------------------------------------------------------------------------
def test_the_parser_accepts_only_its_declared_flags() -> None:
    """Seven flags: which rule, which factor version, which date, where, which pool."""
    parser = build_parser()
    assert {action.dest for action in parser._actions} == EXPECTED_DESTS

    args = parser.parse_args(command())
    assert args.factor == "momentum_12_1"
    assert args.factor_version == "1.0.0"
    assert args.top_n == 3
    assert args.as_of == AS_OF.isoformat()
    assert args.strategy_version == DEFAULT_STRATEGY_VERSION
    assert args.data_root is None
    assert args.universe_path is None


def test_the_parser_names_the_defaults_it_will_use() -> None:
    """The help text never claims a different root, pool or version than the run uses."""
    text = build_parser().format_help()
    assert str(PipelineConfig().data_root) in text
    assert DEFAULT_STRATEGY_VERSION in text
    assert "config/universe_seed.csv" in text


def test_the_run_function_is_callable_without_argv() -> None:
    """A notebook can call it directly: keyword-only, with the documented defaults."""
    signature = inspect.signature(run_strategy)
    assert list(signature.parameters) == [
        "factor_name",
        "factor_version",
        "top_n",
        "as_of_date",
        "strategy_version",
        "data_root",
        "universe_path",
    ]
    assert {parameter.kind for parameter in signature.parameters.values()} == {
        inspect.Parameter.KEYWORD_ONLY
    }
    assert signature.parameters["strategy_version"].default == DEFAULT_STRATEGY_VERSION
    assert signature.parameters["data_root"].default is None
    assert signature.parameters["universe_path"].default is None


def test_the_module_exports_its_documented_api() -> None:
    """Everything a caller may import is named in ``__all__``."""
    assert set(run_strategy_module.__all__) == {
        "DEFAULT_STRATEGY_VERSION",
        "EMPTY_SELECTION_TEXT",
        "RunStrategyError",
        "Selection",
        "StrategyRunOutcome",
        "build_parser",
        "format_summary",
        "main",
        "run_strategy",
    }
    assert issubclass(RunStrategyError, run_strategy_module.StrategyError)


# ---------------------------------------------------------------------------
# 9a. The offline boundary, at runtime
# ---------------------------------------------------------------------------
def test_a_run_never_downloads_computes_or_writes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A successful run touches the network, the factor layer and the disk never."""
    data_root, universe = scenario(tmp_path)
    before = panel_bytes(data_root)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("the strategy command must only read stored panels")

    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(ParquetFactorStore, "write", forbidden)
    monkeypatch.setattr(FactorRunner, "run", forbidden)

    code, out, err = run_cli(command(data_root, universe), capsys)

    assert code == 0
    assert err == ""
    assert "selected_count: 3" in out.splitlines()
    assert panel_bytes(data_root) == before

    files = sorted(path.relative_to(data_root).as_posix() for path in data_root.rglob("*"))
    assert files == [
        "factors",
        "factors/panels",
        "factors/panels/factor_name=momentum_12_1",
        "factors/panels/factor_name=momentum_12_1/factor_version=1.0.0",
        "factors/panels/factor_name=momentum_12_1/factor_version=1.0.0/year=2024",
        "factors/panels/factor_name=momentum_12_1/factor_version=1.0.0/year=2024/"
        "as_of_date=2024-01-05.parquet",
    ]
    for absent in ("raw", "standardized", ".staging", ".trash", "metadata"):
        assert not (data_root / absent).exists()


# ---------------------------------------------------------------------------
# 9b. The module boundary, asserted on the AST
# ---------------------------------------------------------------------------
def script_source() -> str:
    """The command's own source."""
    return SCRIPT.read_text(encoding="utf-8")


def script_tree() -> ast.Module:
    """The command's parsed AST."""
    return ast.parse(script_source())


def imported_modules(tree: ast.Module) -> set[str]:
    """Every module the script imports (dotted names, no aliases)."""
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            modules.add(node.module)
    return modules


def imported_names(tree: ast.Module) -> set[str]:
    """Every name the script imports, so a private member would be visible."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.update(alias.asname or alias.name for alias in node.names)
    return names


def docstring_constants(tree: ast.Module) -> set[int]:
    """The ids of every docstring constant: prose may *describe* what the code avoids."""
    found: set[int] = set()
    holders = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    for node in ast.walk(tree):
        if not isinstance(node, holders) or not node.body:
            continue
        first = node.body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            found.add(id(first.value))
    return found


def code_constants(tree: ast.Module) -> set[str]:
    """Every string literal that is *code*: docstrings are excluded."""
    docstrings = docstring_constants(tree)
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
    }


def code_names(tree: ast.Module) -> set[str]:
    """Every bare name and attribute the code mentions."""
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    names |= {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    return names


def test_the_script_imports_only_the_public_layers() -> None:
    """stdlib plumbing plus the documented entry points of three layers, nothing else."""
    modules = imported_modules(script_tree())
    allowed = (
        "__future__",
        "argparse",
        "collections",
        "dataclasses",
        "datetime",
        "pathlib",
        "sys",
        "typing",
        "data_sys.config",
        "factor_sys",
        "strategy_sys",
        "strategy_sys.strategies",
    )
    for module in sorted(modules):
        assert any(module == prefix or module.startswith(f"{prefix}.") for prefix in allowed), (
            f"the entry point must not import {module!r}"
        )
    assert {
        "data_sys.config",
        "factor_sys",
        "strategy_sys",
        "strategy_sys.strategies",
    } <= modules


def test_the_script_never_imports_network_storage_or_formula_modules() -> None:
    """No Parquet engine, no vendor, no data pipeline, no formula package."""
    modules = imported_modules(script_tree())
    forbidden = (
        "pandas",
        "numpy",
        "pyarrow",
        "duckdb",
        "yfinance",
        "requests",
        "httpx",
        "urllib",
        "http",
        "socket",
        "subprocess",
        "factors",
        "data_sys.providers",
        "data_sys.update",
        "data_sys.data_query",
        "data_sys.batch_query",
        "data_sys.planner",
        "data_sys.merge",
        "data_sys.storage",
        "data_sys.duckdb_verify",
        "data_sys.standardize",
        "factor_sys.factor_runner",
        "factor_sys.market_data_reader",
    )
    for module in sorted(modules):
        assert not any(
            module == banned or module.startswith(f"{banned}.") for banned in forbidden
        ), f"the entry point must not import {module!r}"


def test_the_script_imports_no_private_member() -> None:
    """Only public names cross the boundary, so no module's internals are reached."""
    private = {name for name in imported_names(script_tree()) if name.startswith("_")}
    assert private == set()


def test_the_script_composes_the_public_interfaces() -> None:
    """Registry, runner, store and pool are each used through one documented call."""
    names = code_names(script_tree())
    required = {
        "FactorRequirement",
        "TopNByFactorStrategy",
        "StrategyRegistry",
        "StrategyRunner",
        "StaticUniverseProvider",
        "ParquetFactorStore",
        "register",
        "create",
        "run",
        "factor_requirements",
        "selected_count",
    }
    assert required <= names
    # The runner builds the context and the registry decides the instance: a CLI
    # that assembled either itself would be a second implementation of the layer.
    assert "FactorContext" not in names
    assert "StrategyContext" not in names
    assert "compute" not in names


def test_the_script_never_reads_parquet_or_rebuilds_the_store_layout() -> None:
    """Panels are reached through ``ParquetFactorStore``, never through a path."""
    tree = script_tree()
    text = "\n".join(sorted(code_constants(tree)))
    for fragment in ("factor_name=", "factor_version=", "year=", ".parquet"):
        assert fragment not in text

    private_members = {
        "_root",
        "_factories",
        "_issued",
        "_universe_provider",
        "_factor_store",
        "_rows",
        "_claim",
        "_built",
        "_key",
        "_panel_at",
        "_atomic_write",
        "_root_directory",
        "_spec_of",
    }
    assert private_members.isdisjoint(code_names(tree))


def test_the_script_does_not_restate_the_ranking_rule() -> None:
    """Scores, ranks and the selection come from the strategy, not from this file."""
    tree = script_tree()

    # Only path joining (``/`` on a path), string concatenation and type unions are
    # allowed: a scoring rule would need at least one of -, *, **, // or %.
    operators = {type(node.op) for node in ast.walk(tree) if isinstance(node, ast.BinOp)}
    assert operators <= {ast.Add, ast.BitOr, ast.Div}
    assert not operators & {ast.Sub, ast.Mult, ast.Pow, ast.FloorDiv, ast.Mod}
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp):
            for operand in (node.left, node.right):
                if isinstance(operand, ast.Constant):
                    assert not isinstance(operand.value, (int, float))

    text = "\n".join(sorted(code_constants(tree)))
    for column in ("raw_value", "is_eligible", "score="):
        assert column not in text
    assert "compute" not in code_names(tree)


def test_the_script_has_no_portfolio_trading_or_backtest_logic() -> None:
    """Weights, orders, fills, cash and returns belong to later layers."""
    names = code_names(script_tree())
    forbidden = {
        "weight",
        "weights",
        "target_weight",
        "portfolio",
        "order",
        "orders",
        "fill",
        "fills",
        "cash",
        "turnover",
        "position",
        "positions",
        "cost",
        "costs",
        "slippage",
        "Backtest",
        "backtest",
        "rebalance",
        "schedule",
        "frequency",
    }
    assert forbidden.isdisjoint(names)


# ---------------------------------------------------------------------------
# 10. The documented command, in a real process
# ---------------------------------------------------------------------------
def test_the_documented_command_runs_in_a_real_process(tmp_path: Path) -> None:
    """``python scripts/run_strategy.py ...`` works from the project root."""
    data_root, universe = scenario(tmp_path)

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--factor",
            "momentum_12_1",
            "--factor-version",
            "1.0.0",
            "--top-n",
            "3",
            "--as-of",
            AS_OF.isoformat(),
            "--data-root",
            str(data_root),
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
    lines = completed.stdout.splitlines()
    assert lines[0] == "strategy_name: top_3_by_momentum_12_1"
    assert lines[2] == "as_of_date: 2024-01-05"
    assert lines[4] == "selected_count: 3"
    assert lines[5] == "required_factor: momentum_12_1/1.0.0"
    assert lines[-3:] == [
        "1     NVDA    0.950000",
        "2     MSFT    0.800000",
        "3     META    0.700000",
    ]


def test_the_documented_command_fails_with_one_line_in_a_real_process(
    tmp_path: Path,
) -> None:
    """A missing panel is exit ``1`` and one ``error:`` line, in a real process too."""
    universe = write_universe(tmp_path)

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--factor",
            "momentum_12_1",
            "--factor-version",
            "1.0.0",
            "--top-n",
            "3",
            "--as-of",
            AS_OF.isoformat(),
            "--data-root",
            str(tmp_path / "data"),
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

    assert completed.returncode == 1
    assert completed.stdout == ""
    lines = completed.stderr.splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("error: ")
    assert "Traceback" not in completed.stderr
