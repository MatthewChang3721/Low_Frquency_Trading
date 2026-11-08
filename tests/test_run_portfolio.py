"""End-to-end tests for the minimal single-date portfolio entry point.

Every test here runs against **real files in ``tmp_path``**: a real static universe CSV
and a real Hive-partitioned Parquet factor store written through
:class:`~factor_sys.factor_store.ParquetFactorStore`, then the real
:func:`scripts.run_strategy.run_strategy`, the real registry, the real
``CappedScoreWeightedConstructor``, the real transition builder and the real formatter.
Nothing reads the project's ``data/``, nothing touches the network, and nothing is
stubbed except where a test proves a layer is *not* reached (a panel read, a panel
write, a factor runner, a socket).

Covered in this order:

* the success paths: an all-cash account and a stated account, each through the whole
  chain ``run_strategy -> StrategyResult -> PortfolioState -> PortfolioConstraints ->
  PortfolioContext -> registry -> constructor -> TargetPortfolio -> transition``, with
  the weights recomputed independently in plain Python,
* the entry/exit/increase/decrease rows of a real rebalance, the union that keeps the
  names only the current state holds, the promised row order and the one-sided
  turnover,
* provenance: the constructor the registry created, the exact factor version, the
  strategy version and the construction spec all travelling into the target and the
  transition,
* the constructor's real semantics through the command: score-proportional split, the
  single-name cap with redistribution, an infeasible cap, a non-positive selected
  score, and an empty selection (an all-cash target, not a failure),
* the two current-state modes: every flag combination, the dates a state may carry, a
  manual all-cash account, and every malformed ``SYMBOL=WEIGHT`` -- each with its
  documented type and exit code,
* the printed contract: the exact stable text, six decimals, no cash row in the target
  table, the complete transition table, the five status words and ``none``,
* the command line: ``main`` success and failure paths, argparse's own ``2``, and the
  documented command in a real process started from the project root,
* the offline and read-only boundary, asserted both at runtime (no socket, no factor
  runner, no panel write, no state or order directory) and on the module's AST (no file
  opening, no Parquet, no vendor, no private member, no weighting, order or backtest
  logic).
"""

from __future__ import annotations

import ast
import inspect
import socket
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import FrozenInstanceError, replace
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

import scripts.run_portfolio as run_portfolio_module
from data_sys.config import PipelineConfig
from factor_sys import FactorError, FactorPanel, FactorRunner, ParquetFactorStore
from portfolio_sys import (
    WEIGHT_TOLERANCE,
    PortfolioConstraints,
    PortfolioConstructionSpec,
    PortfolioConstructorRegistry,
    PortfolioContractError,
    PortfolioPosition,
    PortfolioState,
    PortfolioTransition,
    TargetPortfolio,
    build_portfolio_transition,
)
from portfolio_sys.constructors import CappedScoreWeightedConstructor
from portfolio_sys.errors import PortfolioConstructionError, PortfolioError
from scripts.run_portfolio import (
    EMPTY_TABLE_TEXT,
    PortfolioCommandError,
    PortfolioRunOutcome,
    build_parser,
    check_current_state_arguments,
    format_summary,
    main,
    parse_current_position,
    parse_current_positions,
    run_portfolio,
)
from scripts.run_strategy import DEFAULT_STRATEGY_VERSION, StrategyRunOutcome
from strategy_sys import FactorRequirement, StrategyError
from tests.helpers import write_universe_file

#: The command under test, as a file (for the boundary and subprocess tests).
SCRIPT: Path = Path(run_portfolio_module.__file__).resolve()

#: The project root the documented command is run from.
PROJECT_ROOT: Path = SCRIPT.parents[1]

#: A real NYSE session; the panels the fixtures store are keyed by it.
AS_OF: date = date(2024, 1, 5)

#: An earlier session: the date of the stated account in the rebalance tests.
PRIOR: date = date(2023, 12, 29)

#: A session *after* the decision: a stated account may never be dated this late.
LATER: date = date(2024, 2, 1)

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

#: The three symbols :data:`SCORES` ranks highest, in the *universe* order the
#: constructor publishes them in.
TOP_THREE: tuple[str, ...] = ("META", "MSFT", "NVDA")

#: The cash target and single-name cap the documented command uses.
CASH_WEIGHT = 0.10
MAX_SINGLE_WEIGHT = 0.50

#: The share of the account the documented command invests.
INVESTED = 0.90

#: Fields the printed report always carries, in order, before ``target:``.
SUMMARY_FIELDS: tuple[str, ...] = (
    "as_of_date",
    "current_state_as_of_date",
    "strategy_name",
    "strategy_version",
    "construction_name",
    "construction_version",
    "target_cash_weight",
    "max_single_weight",
    "selected_count",
    "target_position_count",
    "buy_weight",
    "sell_weight",
    "cash_weight_change",
    "turnover",
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
        "target_cash_weight",
        "max_single_weight",
        "initial_cash",
        "current_as_of",
        "current_position",
        "current_cash_weight",
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


def panel_signature(data_root: Path) -> dict[Path, tuple[bytes, int]]:
    """Every stored panel file of ``data_root``: its exact content and modification time."""
    root = data_root / "factors" / "panels"
    if not root.exists():
        return {}
    return {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


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
    cash: str = "0.10",
    cap: str = "0.50",
    extra: Sequence[str] = (),
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
        "--target-cash-weight",
        cash,
        "--max-single-weight",
        cap,
    ]
    if data_root is not None:
        argv += ["--data-root", str(data_root)]
    if universe_path is not None:
        argv += ["--universe-path", str(universe_path)]
    return argv + list(extra)


def run_cli(argv: Sequence[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, str, str]:
    """Run ``main`` in this process and return ``(exit_code, stdout, stderr)``."""
    code = main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def run_api(data_root: Path, universe: Path, **overrides: object) -> PortfolioRunOutcome:
    """Call the public entry function with the documented defaults, allowing overrides."""
    arguments: dict[str, object] = {
        "factor_name": "momentum_12_1",
        "factor_version": "1.0.0",
        "top_n": 3,
        "as_of_date": AS_OF,
        "target_cash_weight": CASH_WEIGHT,
        "max_single_weight": MAX_SINGLE_WEIGHT,
        "initial_cash": True,
        "data_root": data_root,
        "universe_path": universe,
    }
    arguments.update(overrides)
    return run_portfolio(**arguments)  # type: ignore[arg-type]


def ranked(
    scores: Mapping[str, float], count: int = 3, symbols: Sequence[str] = SYMBOLS
) -> tuple[str, ...]:
    """The rule's selection, recomputed independently, in universe order."""
    best = sorted(symbols, key=lambda symbol: (-scores[symbol], symbol))[:count]
    return tuple(symbol for symbol in symbols if symbol in best)


def expected_target_weights(
    scores: Mapping[str, float],
    selected: Sequence[str],
    cash: float = CASH_WEIGHT,
    cap: float = MAX_SINGLE_WEIGHT,
) -> dict[str, float]:
    """The capped score-proportional split, recomputed independently of the layer.

    Water-filling: split the investable share by score, pin every symbol above the cap
    and re-split what is left among the rest, until nothing is above the cap.
    """
    investable = 1.0 - cash
    unpinned = {symbol: scores[symbol] for symbol in selected}
    weights: dict[str, float] = {}
    left = investable

    while unpinned:
        total = sum(unpinned.values())
        breached = [symbol for symbol in unpinned if left * unpinned[symbol] / total > cap]
        if not breached:
            weights.update({symbol: left * score / total for symbol, score in unpinned.items()})
            break
        for symbol in breached:
            weights[symbol] = cap
            left -= cap
            del unpinned[symbol]

    return {symbol: weights[symbol] for symbol in selected}


def metadata(text: str) -> list[tuple[str, str]]:
    """Every ``key: value`` line of a report before the first table, in print order."""
    pairs: list[tuple[str, str]] = []
    for line in text.splitlines():
        if line in ("target:", "transition:", "cash:"):
            break
        if ": " in line:
            key, _, value = line.partition(": ")
            pairs.append((key, value))
    return pairs


def field(text: str, name: str) -> str:
    """The single value printed for ``name`` in the metadata block."""
    values = [value for key, value in metadata(text) if key == name]
    assert len(values) == 1, f"{name!r} appears {len(values)} time(s) in\n{text}"
    return values[0]


def table(text: str, name: str) -> list[list[str]]:
    """The rows of one named table, its header included; ``[["none"]]`` when empty."""
    lines = text.splitlines()
    start = lines.index(f"{name}:")
    rows: list[list[str]] = []
    for line in lines[start + 1 :]:
        if line.endswith(":") and " " not in line:
            break
        rows.append(line.split())
    return rows


def rows(text: str, name: str) -> dict[str, list[str]]:
    """The data rows of one named table, keyed by symbol (``none`` becomes empty)."""
    body = table(text, name)[1:]
    return {row[0]: row for row in body if row != [EMPTY_TABLE_TEXT]}


def statuses(text: str) -> dict[str, str]:
    """The ``status`` column of the transition table, keyed by symbol."""
    return {symbol: row[-1] for symbol, row in rows(text, "transition").items()}


def cash_block(text: str) -> dict[str, str]:
    """The three lines of the ``cash:`` block, keyed by name."""
    lines = text.splitlines()
    start = lines.index("cash:")
    block: dict[str, str] = {}
    for line in lines[start + 1 :]:
        key, _, value = line.partition(": ")
        block[key] = value
    return block


def forbid_panel_access(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make any panel read or write fail the test loudly."""

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("an argument error must not reach the factor store")

    monkeypatch.setattr(ParquetFactorStore, "read", forbidden)
    monkeypatch.setattr(ParquetFactorStore, "write", forbidden)


def manual_flags(
    holdings: Mapping[str, float] | None = None,
    cash: float = 0.10,
    as_of: date = PRIOR,
) -> list[str]:
    """The flags of the manual current-state mode, in the documented form."""
    argv = ["--current-as-of", as_of.isoformat()]
    for symbol, weight in (holdings or {}).items():
        argv += ["--current-position", f"{symbol}={weight}"]
    return argv + ["--current-cash-weight", str(cash)]


def stated_positions(holdings: Mapping[str, float]) -> tuple[PortfolioPosition, ...]:
    """The same holdings as the public API takes them: validated positions."""
    return tuple(PortfolioPosition(symbol, weight) for symbol, weight in holdings.items())


#: The account the rebalance tests state: two names the target keeps and one it drops.
REBALANCE_HOLDINGS: dict[str, float] = {"NVDA": 0.30, "MSFT": 0.30, "AAPL": 0.30}

#: The cash share of that account, which is exactly the target's own cash target.
REBALANCE_CASH = 0.10


# ---------------------------------------------------------------------------
# 1. The first purchase: an all-cash account, end to end
# ---------------------------------------------------------------------------
def test_an_all_cash_start_builds_a_target_and_a_transition(
    tmp_path: Path,
) -> None:
    """``--initial-cash``: one command, one target, one difference."""
    data_root, universe = scenario(tmp_path)

    outcome = run_api(data_root, universe)

    state = outcome.current_state
    target = outcome.target_portfolio
    transition = outcome.transition

    assert state.as_of_date == AS_OF
    assert state.positions == ()
    assert state.cash_weight == 1.0

    assert target.as_of_date == AS_OF
    assert target.cash_weight == pytest.approx(CASH_WEIGHT)
    assert target.target_symbols == TOP_THREE
    assert transition.symbols == TOP_THREE
    assert transition.cash_weight_change == pytest.approx(-INVESTED)
    assert transition.buy_weight == pytest.approx(INVESTED)
    assert transition.sell_weight == pytest.approx(0.0)
    assert transition.turnover == pytest.approx(INVESTED)


def test_the_weights_follow_the_selected_scores_within_the_cap(tmp_path: Path) -> None:
    """The split is score-proportional; it is recomputed here in plain Python."""
    data_root, universe = scenario(tmp_path)

    outcome = run_api(data_root, universe)

    expected = expected_target_weights(SCORES, TOP_THREE)
    actual = outcome.target_portfolio.target_weights_by_symbol

    assert set(actual) == set(expected)
    for symbol, weight in expected.items():
        assert actual[symbol] == pytest.approx(weight, abs=1e-12)
    assert outcome.target_portfolio.invested_weight == pytest.approx(INVESTED, abs=1e-12)
    assert sum(actual.values()) == pytest.approx(INVESTED, abs=1e-12)


def test_the_selection_is_the_strategys_own_top_n(tmp_path: Path) -> None:
    """Which symbols and how many come from the strategy, never from this layer."""
    data_root, universe = scenario(tmp_path)

    outcome = run_api(data_root, universe)

    assert outcome.selected_count == 3
    assert outcome.strategy_result.selected_symbols == TOP_THREE
    assert outcome.target_portfolio.target_symbols == TOP_THREE
    assert len(outcome.target_portfolio.positions) == 3


def test_every_row_of_a_first_purchase_is_an_entry(tmp_path: Path) -> None:
    """Nothing was held, so every target name is a new holding."""
    data_root, universe = scenario(tmp_path)

    outcome = run_api(data_root, universe)

    for change in outcome.transition.changes:
        assert change.is_entry
        assert change.current_weight == 0.0
        assert change.target_weight > 0.0
        assert change.weight_change == pytest.approx(change.target_weight)


def test_a_first_purchase_writes_nothing(tmp_path: Path) -> None:
    """The panel is read, never repaired, and no state file appears anywhere."""
    data_root, universe = scenario(tmp_path)
    before = panel_signature(data_root)

    run_api(data_root, universe)

    assert panel_signature(data_root) == before
    for absent in ("raw", "standardized", ".staging", ".trash", "metadata"):
        assert not (data_root / absent).exists()
    for absent in ("state", "orders", "reports"):
        assert not (tmp_path / absent).exists()


# ---------------------------------------------------------------------------
# 2. A rebalance: a stated account against the new target
# ---------------------------------------------------------------------------
def rebalance_api(
    data_root: Path,
    universe: Path,
    holdings: Mapping[str, float] | None = None,
    cash: float = REBALANCE_CASH,
    as_of: date = PRIOR,
    **overrides: object,
) -> PortfolioRunOutcome:
    """Run the command with a stated account instead of an all-cash start."""
    return run_api(
        data_root,
        universe,
        initial_cash=False,
        current_as_of_date=as_of,
        current_positions=stated_positions(REBALANCE_HOLDINGS if holdings is None else holdings),
        current_cash_weight=cash,
        **overrides,
    )


def test_a_rebalance_reports_entry_exit_increase_and_decrease(tmp_path: Path) -> None:
    """The four moving kinds of row, from one stated account and one target."""
    data_root, universe = scenario(tmp_path)

    outcome = rebalance_api(data_root, universe)
    changes = {change.symbol: change for change in outcome.transition.changes}

    assert changes["META"].is_entry
    assert changes["AAPL"].is_exit
    assert changes["NVDA"].is_buy and not changes["NVDA"].is_entry
    assert changes["MSFT"].is_sell and not changes["MSFT"].is_exit

    assert changes["META"].current_weight == 0.0
    assert changes["AAPL"].target_weight == 0.0
    assert changes["NVDA"].weight_change == pytest.approx(changes["NVDA"].target_weight - 0.30)
    assert changes["MSFT"].weight_change == pytest.approx(changes["MSFT"].target_weight - 0.30)
    assert changes["AAPL"].weight_change == pytest.approx(-0.30)


def test_the_exited_name_keeps_its_row(tmp_path: Path) -> None:
    """A name only the current state holds is reported, never dropped."""
    data_root, universe = scenario(tmp_path)

    outcome = rebalance_api(data_root, universe)

    assert "AAPL" in outcome.transition.symbols
    assert "AAPL" not in outcome.target_portfolio.target_symbols
    assert outcome.transition.exiting_symbols == ("AAPL",)
    assert outcome.target_portfolio.target_symbols == TOP_THREE


def test_the_target_names_come_first_and_the_current_only_names_follow(
    tmp_path: Path,
) -> None:
    """The documented order: the target's own order, then the state's own order."""
    data_root, universe = scenario(tmp_path)

    outcome = rebalance_api(data_root, universe)

    assert outcome.target_portfolio.target_symbols == TOP_THREE
    assert outcome.transition.symbols == (*TOP_THREE, "AAPL")
    assert outcome.current_state.symbols == ("NVDA", "MSFT", "AAPL")


def test_turnover_is_the_one_sided_volume_of_the_run(tmp_path: Path) -> None:
    """``0.5 * (sum(|change|) + |cash change|)``, recomputed from the rows."""
    data_root, universe = scenario(tmp_path)

    transition = rebalance_api(data_root, universe).transition
    moved = sum(abs(change.weight_change) for change in transition.changes)

    assert transition.turnover == pytest.approx(0.5 * (moved + abs(transition.cash_weight_change)))
    assert transition.turnover == pytest.approx(max(transition.buy_weight, transition.sell_weight))
    assert transition.buy_weight == pytest.approx(transition.sell_weight)
    assert transition.sell_weight > 0.0


def test_the_cash_change_is_reported_outside_the_rows(tmp_path: Path) -> None:
    """Cash is a field of a portfolio, so it never becomes a row of the table."""
    data_root, universe = scenario(tmp_path)

    transition = rebalance_api(data_root, universe).transition

    assert transition.cash_weight_change == pytest.approx(0.0)
    assert "CASH" not in transition.symbols
    assert all(change.symbol != "CASH" for change in transition.changes)


def test_a_holding_that_keeps_its_weight_is_unchanged(tmp_path: Path) -> None:
    """Equal scores split the account into thirds, so one holding can stay put."""
    scores = dict.fromkeys(SYMBOLS, 0.0)
    scores.update({"META": 1.0, "MSFT": 1.0, "NVDA": 1.0})
    data_root, universe = scenario(tmp_path, scores=scores)

    outcome = rebalance_api(data_root, universe, holdings={"MSFT": 0.30}, cash=0.70)
    changes = {change.symbol: change for change in outcome.transition.changes}

    assert changes["MSFT"].is_unchanged
    assert changes["MSFT"].weight_change == pytest.approx(0.0, abs=WEIGHT_TOLERANCE)
    assert changes["META"].is_entry and changes["NVDA"].is_entry
    assert outcome.target_portfolio.target_weights_by_symbol["MSFT"] == pytest.approx(0.30)
    assert outcome.transition.turnover == pytest.approx(0.60)


def test_the_stated_state_is_the_one_the_run_was_given(tmp_path: Path) -> None:
    """The outcome echoes the state exactly: date, holdings, cash and order."""
    data_root, universe = scenario(tmp_path)

    outcome = rebalance_api(data_root, universe)
    state = outcome.current_state

    assert isinstance(state, PortfolioState)
    assert state.as_of_date == PRIOR
    assert state.symbols == ("NVDA", "MSFT", "AAPL")
    assert state.weights_by_symbol == REBALANCE_HOLDINGS
    assert state.cash_weight == pytest.approx(REBALANCE_CASH)
    assert state is not outcome.target_portfolio


# ---------------------------------------------------------------------------
# 3. Provenance: which rule, which version, which factor version
# ---------------------------------------------------------------------------
def test_the_target_comes_from_the_instance_the_registry_created(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``create`` decides the instance, and that instance constructs the weights."""
    data_root, universe = scenario(tmp_path)

    created: list[object] = []
    constructed: list[object] = []
    original_create = PortfolioConstructorRegistry.create
    original_construct = CappedScoreWeightedConstructor.construct

    def spy_create(self: PortfolioConstructorRegistry, spec: object) -> object:
        constructor = original_create(self, spec)  # type: ignore[arg-type]
        created.append(constructor)
        return constructor

    def spy_construct(self: CappedScoreWeightedConstructor, context: object) -> object:
        target = original_construct(self, context)  # type: ignore[arg-type]
        constructed.append(target)
        return target

    monkeypatch.setattr(PortfolioConstructorRegistry, "create", spy_create)
    monkeypatch.setattr(CappedScoreWeightedConstructor, "construct", spy_construct)

    outcome = run_api(data_root, universe)

    assert len(created) == 1
    assert type(created[0]) is CappedScoreWeightedConstructor
    assert constructed == [outcome.target_portfolio]
    assert created[0].spec == outcome.construction_spec  # type: ignore[attr-defined]


def test_each_run_gets_its_own_registry_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One registration per command, on a table that is never shared between runs."""
    data_root, universe = scenario(tmp_path)

    tables: list[object] = []
    original_register = PortfolioConstructorRegistry.register

    def spy_register(self: PortfolioConstructorRegistry, factory: object) -> object:
        tables.append(self)
        return original_register(self, factory)  # type: ignore[arg-type]

    monkeypatch.setattr(PortfolioConstructorRegistry, "register", spy_register)

    run_api(data_root, universe)
    run_api(data_root, universe)

    assert len(tables) == 2
    assert tables[0] is not tables[1]


def test_the_exact_factor_version_travels_into_the_target_and_the_transition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second version of one factor name is another panel and another selection."""
    data_root, universe = scenario(tmp_path)
    inverted = {symbol: 1.0 - SCORES[symbol] for symbol in SYMBOLS}
    write_panel(data_root, MOMENTUM_V2, inverted)

    reads: list[tuple[str, str, object]] = []
    original_read = ParquetFactorStore.read

    def spy_read(
        self: ParquetFactorStore, factor_name: str, factor_version: str, as_of_date: object
    ) -> FactorPanel:
        reads.append((factor_name, factor_version, as_of_date))
        return original_read(  # type: ignore[arg-type]
            self, factor_name, factor_version, as_of_date
        )

    monkeypatch.setattr(ParquetFactorStore, "read", spy_read)

    outcome = run_api(data_root, universe, factor_version="2.0.0")

    assert reads == [("momentum_12_1", "2.0.0", AS_OF)]
    assert outcome.target_portfolio.factor_requirements == (MOMENTUM_V2,)
    assert outcome.transition.factor_requirements == (MOMENTUM_V2,)
    assert outcome.strategy_outcome.factor_requirements == (MOMENTUM_V2,)
    assert outcome.target_portfolio.target_symbols == ranked(inverted)


def test_the_strategy_version_is_the_one_that_was_asked_for(tmp_path: Path) -> None:
    """The version on the command line is the version in the target's provenance."""
    data_root, universe = scenario(tmp_path)

    outcome = run_api(data_root, universe, strategy_version="9.9.9")

    assert outcome.target_portfolio.strategy_spec.version == "9.9.9"
    assert outcome.transition.strategy_spec.version == "9.9.9"
    assert outcome.target_portfolio.strategy_spec.name == "top_3_by_momentum_12_1"
    assert outcome.target_portfolio.factor_requirements == (MOMENTUM,)


def test_the_constraints_are_the_ones_the_command_was_given(tmp_path: Path) -> None:
    """Cash target and cap travel with the target; nothing is normalised."""
    data_root, universe = scenario(tmp_path)

    outcome = run_api(data_root, universe, target_cash_weight=0.25, max_single_weight=0.40)

    expected = PortfolioConstraints(cash_weight=0.25, max_single_weight=0.40)
    assert outcome.constraints == expected
    assert outcome.target_portfolio.constraints == expected
    assert outcome.transition.constraints is outcome.target_portfolio.constraints
    assert outcome.target_portfolio.invested_weight == pytest.approx(0.75, abs=1e-12)


def test_the_construction_spec_is_the_registered_one(tmp_path: Path) -> None:
    """The target says which rule, at which version, set its weights."""
    data_root, universe = scenario(tmp_path)

    outcome = run_api(data_root, universe)

    spec = outcome.construction_spec
    assert spec == CappedScoreWeightedConstructor().spec
    assert spec.name == "capped_score_weighted"
    assert spec.version == "1.0.0"
    assert outcome.transition.construction_spec is outcome.target_portfolio.construction_spec


def test_the_outcome_points_at_the_objects_of_one_run(tmp_path: Path) -> None:
    """No part is copied: the target is built from *this* result and diffed once."""
    data_root, universe = scenario(tmp_path)

    outcome = run_api(data_root, universe)

    assert outcome.target_portfolio.source_result is outcome.strategy_outcome.result
    assert outcome.strategy_result is outcome.strategy_outcome.result
    assert outcome.transition.target_portfolio is outcome.target_portfolio
    assert outcome.target_portfolio.as_of_date == outcome.as_of_date
    assert outcome.transition.as_of_date == AS_OF


def test_there_is_no_default_registry_and_no_auto_registration() -> None:
    """The script builds its table per call; the library still has none."""
    assert not hasattr(run_portfolio_module, "default_portfolio_registry")
    assert "default_portfolio_registry" not in run_portfolio_module.__all__
    assert not hasattr(PortfolioConstructorRegistry, "default")


# ---------------------------------------------------------------------------
# 4. The constructor's real semantics, through the command
# ---------------------------------------------------------------------------
def test_the_investable_share_is_split_in_proportion_to_the_scores(tmp_path: Path) -> None:
    """With no cash and no cap in the way, the weights *are* the score proportions."""
    data_root, universe = scenario(tmp_path)

    outcome = run_api(data_root, universe, top_n=2, target_cash_weight=0.0, max_single_weight=1.0)
    weights = outcome.target_portfolio.target_weights_by_symbol
    selected = outcome.target_portfolio.target_symbols

    assert selected == ("MSFT", "NVDA")
    total = SCORES["MSFT"] + SCORES["NVDA"]
    assert weights["MSFT"] == pytest.approx(SCORES["MSFT"] / total, abs=1e-12)
    assert weights["NVDA"] == pytest.approx(SCORES["NVDA"] / total, abs=1e-12)
    assert outcome.target_portfolio.cash_weight == 0.0
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-12)


def test_a_dominant_score_is_pinned_at_the_cap_and_the_rest_is_resplit(
    tmp_path: Path,
) -> None:
    """One name would take 81% of the account, so the cap pins it and the rest splits."""
    scores = dict.fromkeys(SYMBOLS, 0.0)
    scores.update({"META": 0.5, "MSFT": 0.5, "NVDA": 9.0})
    data_root, universe = scenario(tmp_path, scores=scores)

    outcome = run_api(data_root, universe, max_single_weight=0.40)
    weights = outcome.target_portfolio.target_weights_by_symbol

    assert outcome.target_portfolio.target_symbols == ("META", "MSFT", "NVDA")
    assert weights["NVDA"] == pytest.approx(0.40, abs=1e-12)
    assert weights["META"] == pytest.approx(0.25, abs=1e-12)
    assert weights["MSFT"] == pytest.approx(0.25, abs=1e-12)
    assert max(weights.values()) <= 0.40 + WEIGHT_TOLERANCE
    assert sum(weights.values()) == pytest.approx(INVESTED, abs=1e-12)


def test_a_cap_that_cannot_cover_the_investable_share_is_refused(tmp_path: Path) -> None:
    """Three names at a 25% cap cannot carry 90% of the account."""
    data_root, universe = scenario(tmp_path)

    with pytest.raises(PortfolioConstructionError) as excinfo:
        run_api(data_root, universe, max_single_weight=0.25)

    message = str(excinfo.value)
    assert "max_single_weight" in message
    assert "0.75" in message
    assert "0.9" in message


def test_a_selected_score_of_zero_cannot_become_a_holding(tmp_path: Path) -> None:
    """A zero score is a legal panel value and an illegal long-only share."""
    scores = dict.fromkeys(SYMBOLS, 0.0)
    data_root, universe = scenario(tmp_path, scores=scores)

    with pytest.raises(PortfolioConstructionError) as excinfo:
        run_api(data_root, universe, top_n=2)

    assert "score" in str(excinfo.value)


def test_an_empty_selection_is_an_all_cash_target(tmp_path: Path) -> None:
    """Nothing selected is an all-cash target, and that is a success."""
    data_root, universe = scenario(tmp_path, eligible=dict.fromkeys(SYMBOLS, False))

    outcome = run_api(data_root, universe)

    assert outcome.selected_count == 0
    assert outcome.target_portfolio.positions == ()
    assert outcome.target_portfolio.cash_weight == 1.0
    assert outcome.target_portfolio.invested_weight == pytest.approx(0.0)


def test_an_empty_selection_from_an_all_cash_account_is_an_empty_transition(
    tmp_path: Path,
) -> None:
    """Cash to cash: no row, no turnover, and the report prints ``none``."""
    data_root, universe = scenario(tmp_path, eligible=dict.fromkeys(SYMBOLS, False))

    outcome = run_api(data_root, universe)
    transition = outcome.transition

    assert transition.changes == ()
    assert transition.symbols == ()
    assert transition.turnover == 0.0
    assert transition.cash_weight_change == 0.0


def test_an_empty_selection_liquidates_a_stated_account(tmp_path: Path) -> None:
    """Cash to all-cash: every holding is an exit row and the turnover is their weight."""
    data_root, universe = scenario(tmp_path, eligible=dict.fromkeys(SYMBOLS, False))

    outcome = rebalance_api(data_root, universe)
    transition = outcome.transition

    assert transition.exiting_symbols == ("NVDA", "MSFT", "AAPL")
    assert set(transition.symbols) == {"NVDA", "MSFT", "AAPL"}
    assert outcome.target_portfolio.cash_weight == 1.0
    assert transition.cash_weight_change == pytest.approx(INVESTED)
    assert transition.turnover == pytest.approx(INVESTED)


def test_a_larger_top_n_weights_every_selected_name(tmp_path: Path) -> None:
    """``top_n`` bounds the candidates; the constructor weights exactly those."""
    data_root, universe = scenario(tmp_path)

    outcome = run_api(data_root, universe, top_n=10, max_single_weight=0.20)

    assert outcome.selected_count == 10
    assert len(outcome.target_portfolio.positions) == 10
    assert outcome.target_portfolio.invested_weight == pytest.approx(INVESTED, abs=1e-12)


# ---------------------------------------------------------------------------
# 5. The two current-state modes
# ---------------------------------------------------------------------------
def test_initial_cash_is_a_complete_state_on_its_own(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Mode A needs no other flag: cash is 100% and the date is the decision date."""
    data_root, universe = scenario(tmp_path)

    code, out, err = run_cli(command(data_root, universe, extra=["--initial-cash"]), capsys)

    assert code == 0
    assert err == ""
    assert field(out, "current_state_as_of_date") == AS_OF.isoformat()
    assert cash_block(out)["current_weight"] == "1.000000"


@pytest.mark.parametrize(
    "extra",
    [
        ["--current-as-of", PRIOR.isoformat()],
        ["--current-position", "AAPL=0.60"],
        ["--current-cash-weight", "1.0"],
    ],
)
def test_initial_cash_cannot_be_combined_with_a_current_flag(
    extra: Sequence[str],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mode A and mode B are exclusive, and the refusal comes before any read."""
    data_root, universe = scenario(tmp_path)
    forbid_panel_access(monkeypatch)

    code, out, err = run_cli(command(data_root, universe, extra=["--initial-cash", *extra]), capsys)

    assert code == 1
    assert out == ""
    assert err.startswith("error: ")
    assert "--initial-cash" in err
    assert extra[0] in err
    assert "Traceback" not in err


def test_the_conflict_is_a_command_error_at_the_api_level(tmp_path: Path) -> None:
    """The public function refuses the same combination, with the same type."""
    data_root, universe = scenario(tmp_path)

    with pytest.raises(PortfolioCommandError) as excinfo:
        run_api(data_root, universe, current_cash_weight=1.0)

    message = str(excinfo.value)
    assert "--initial-cash" in message
    assert "--current-cash-weight" in message
    assert issubclass(PortfolioCommandError, PortfolioError)


@pytest.mark.parametrize(
    ("extra", "missing"),
    [
        (
            ["--current-position", "AAPL=0.60", "--current-cash-weight", "0.40"],
            "--current-as-of",
        ),
        (
            ["--current-as-of", PRIOR.isoformat(), "--current-position", "AAPL=0.60"],
            "--current-cash-weight",
        ),
    ],
)
def test_the_manual_mode_needs_its_date_and_its_cash(
    extra: Sequence[str],
    missing: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without ``--initial-cash``, both halves of the manual mode are required."""
    data_root, universe = scenario(tmp_path)
    forbid_panel_access(monkeypatch)

    code, out, err = run_cli(command(data_root, universe, extra=extra), capsys)

    assert code == 1
    assert out == ""
    assert err.startswith("error: ")
    assert missing in err
    assert "--initial-cash" in err
    assert "Traceback" not in err


def test_a_state_dated_before_the_decision_is_accepted(tmp_path: Path) -> None:
    """Yesterday's account is a legal input to today's decision."""
    data_root, universe = scenario(tmp_path)

    outcome = rebalance_api(data_root, universe)

    assert outcome.current_state.as_of_date == PRIOR
    assert outcome.as_of_date == AS_OF


def test_a_state_dated_on_the_decision_date_is_accepted(tmp_path: Path) -> None:
    """The same session is legal too: the state is not required to be older."""
    data_root, universe = scenario(tmp_path)

    outcome = rebalance_api(data_root, universe, as_of=AS_OF)

    assert outcome.current_state.as_of_date == AS_OF
    assert outcome.transition.as_of_date == AS_OF


def test_a_state_dated_after_the_decision_is_refused(tmp_path: Path) -> None:
    """A state that only becomes known later is look-ahead, and is refused."""
    data_root, universe = scenario(tmp_path)

    with pytest.raises(PortfolioContractError) as excinfo:
        rebalance_api(data_root, universe, as_of=LATER)

    message = str(excinfo.value)
    assert "look-ahead" in message
    assert LATER.isoformat() in message


def test_the_manual_mode_accepts_an_all_cash_account(tmp_path: Path) -> None:
    """A stated account may hold no positions at all: cash 100%."""
    data_root, universe = scenario(tmp_path)

    outcome = run_api(
        data_root,
        universe,
        initial_cash=False,
        current_as_of_date=PRIOR,
        current_positions=(),
        current_cash_weight=1.0,
    )

    assert outcome.current_state.positions == ()
    assert outcome.current_state.cash_weight == 1.0
    assert outcome.transition.cash_weight_change == pytest.approx(-INVESTED)
    assert outcome.transition.entering_symbols == TOP_THREE


def test_several_positions_are_accepted(tmp_path: Path) -> None:
    """Every holding is carried, in the order it was given."""
    data_root, universe = scenario(tmp_path)

    outcome = rebalance_api(data_root, universe)

    assert len(outcome.current_state.positions) == 3
    assert outcome.current_state.symbols == ("NVDA", "MSFT", "AAPL")


@pytest.mark.parametrize(
    ("text", "detail"),
    [
        ("AAPL", "no '='"),
        ("AAPL=0.60=0.10", "'=' characters"),
        ("=0.60", "symbol is empty"),
        ("AAPL=", "weight is empty"),
        ("AAPL=abc", "not a number"),
        ("", "no '='"),
    ],
)
def test_a_malformed_position_flag_is_refused(
    text: str,
    detail: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``SYMBOL=WEIGHT`` is split on one ``=``, and nothing is guessed."""
    data_root, universe = scenario(tmp_path)

    with pytest.raises(PortfolioCommandError) as excinfo:
        parse_current_position(text)

    assert detail in str(excinfo.value)

    argv = command(
        data_root,
        universe,
        extra=[
            "--current-as-of",
            PRIOR.isoformat(),
            "--current-position",
            text,
            "--current-cash-weight",
            "0.40",
        ],
    )
    code, out, err = run_cli(argv, capsys)

    assert code == 1
    assert out == ""
    assert err.startswith("error: ")
    assert "Traceback" not in err


def test_a_position_value_of_another_type_is_refused() -> None:
    """Only text can be a ``SYMBOL=WEIGHT`` argument."""
    with pytest.raises(PortfolioCommandError):
        parse_current_position(0.6)  # type: ignore[arg-type]

    assert parse_current_positions(["AAPL=0.60", "MSFT=0.40"]) == (
        PortfolioPosition("AAPL", 0.60),
        PortfolioPosition("MSFT", 0.40),
    )
    assert parse_current_positions([]) == ()


def test_a_duplicate_symbol_is_refused(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Two rows for one name are one name, so the state refuses them."""
    data_root, universe = scenario(tmp_path)

    with pytest.raises(PortfolioContractError) as excinfo:
        run_api(
            data_root,
            universe,
            initial_cash=False,
            current_as_of_date=PRIOR,
            current_positions=(
                PortfolioPosition("AAPL", 0.30),
                PortfolioPosition("AAPL", 0.30),
            ),
            current_cash_weight=0.40,
        )

    assert "more than once" in str(excinfo.value)

    code, _, err = run_cli(
        command(
            data_root,
            universe,
            extra=manual_flags({"AAPL": 0.30}, cash=0.40) + ["--current-position", "AAPL=0.30"],
        ),
        capsys,
    )

    assert code == 1
    assert err.startswith("error: ")
    assert "Traceback" not in err


def test_positions_that_do_not_add_up_are_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A portfolio is complete: stocks plus cash must be 100%."""
    data_root, universe = scenario(tmp_path)

    with pytest.raises(PortfolioContractError):
        run_api(
            data_root,
            universe,
            initial_cash=False,
            current_as_of_date=PRIOR,
            current_positions=(PortfolioPosition("AAPL", 0.60),),
            current_cash_weight=0.10,
        )

    code, _, err = run_cli(
        command(data_root, universe, extra=manual_flags({"AAPL": 0.60}, cash=0.10)), capsys
    )

    assert code == 1
    assert err.startswith("error: ")
    assert "Traceback" not in err


def test_cash_can_never_be_a_position(tmp_path: Path) -> None:
    """Cash is a field, so a ``CASH`` holding is not a holding."""
    data_root, universe = scenario(tmp_path)

    with pytest.raises(PortfolioContractError):
        run_api(
            data_root,
            universe,
            initial_cash=False,
            current_as_of_date=PRIOR,
            current_positions=(PortfolioPosition("CASH", 0.30),),
            current_cash_weight=0.70,
        )


@pytest.mark.parametrize("weight", ["nan", "inf", "-0.10", "0", "1.5"])
def test_a_position_weight_that_is_not_a_share_is_refused(weight: str, tmp_path: Path) -> None:
    """Long only, finite and strictly positive: the contract decides, not this command."""
    data_root, universe = scenario(tmp_path)

    with pytest.raises(PortfolioContractError):
        run_api(
            data_root,
            universe,
            initial_cash=False,
            current_as_of_date=PRIOR,
            current_positions=parse_current_positions([f"AAPL={weight}"]),
            current_cash_weight=0.0,
        )


def test_the_symbol_is_never_trimmed_or_upper_cased(tmp_path: Path) -> None:
    """``aapl=0.6`` is not ``AAPL``: the contract refuses it rather than repairing it."""
    data_root, universe = scenario(tmp_path)

    with pytest.raises(PortfolioContractError):
        run_api(
            data_root,
            universe,
            initial_cash=False,
            current_as_of_date=PRIOR,
            current_positions=parse_current_positions(["aapl=0.6"]),
            current_cash_weight=0.4,
        )


def test_the_mode_and_the_constraints_are_checked_before_the_panel_is_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Neither a conflict, a missing field nor a bad weight is a reason to open a store."""
    data_root, universe = scenario(tmp_path)
    forbid_panel_access(monkeypatch)

    with pytest.raises(PortfolioCommandError):
        check_current_state_arguments(
            initial_cash=True,
            current_as_of_date=PRIOR,
            current_positions=None,
            current_cash_weight=None,
        )
    with pytest.raises(PortfolioCommandError):
        check_current_state_arguments(
            initial_cash=False,
            current_as_of_date=None,
            current_positions=None,
            current_cash_weight=None,
        )
    # Both legal shapes pass without saying anything.
    check_current_state_arguments(
        initial_cash=True,
        current_as_of_date=None,
        current_positions=None,
        current_cash_weight=None,
    )
    check_current_state_arguments(
        initial_cash=False,
        current_as_of_date=PRIOR,
        current_positions=(),
        current_cash_weight=1.0,
    )

    with pytest.raises(PortfolioContractError):
        run_api(data_root, universe, target_cash_weight=1.5)
    with pytest.raises(PortfolioContractError):
        run_api(data_root, universe, max_single_weight=0.0)


# ---------------------------------------------------------------------------
# 6. The printed report
# ---------------------------------------------------------------------------
#: The exact report of the documented first purchase; every field, in order.
INITIAL_CASH_REPORT: tuple[str, ...] = (
    "as_of_date: 2024-01-05",
    "current_state_as_of_date: 2024-01-05",
    "strategy_name: top_3_by_momentum_12_1",
    "strategy_version: 1.0.0",
    "required_factor: momentum_12_1/1.0.0",
    "construction_name: capped_score_weighted",
    "construction_version: 1.0.0",
    "target_cash_weight: 0.100000",
    "max_single_weight: 0.500000",
    "selected_count: 3",
    "target_position_count: 3",
    "buy_weight: 0.900000",
    "sell_weight: 0.000000",
    "cash_weight_change: -0.900000",
    "turnover: 0.900000",
    "target:",
    "symbol  target_weight",
    "META    0.257143",
    "MSFT    0.293878",
    "NVDA    0.348980",
    "transition:",
    "symbol  current_weight  target_weight  weight_change  status",
    "META    0.000000        0.257143       0.257143       entry",
    "MSFT    0.000000        0.293878       0.293878       entry",
    "NVDA    0.000000        0.348980       0.348980       entry",
    "cash:",
    "current_weight: 1.000000",
    "target_weight: 0.100000",
    "weight_change: -0.900000",
)


def test_the_report_is_the_documented_stable_text(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """One stable block: metadata, ``target:``, ``transition:``, ``cash:``."""
    data_root, universe = scenario(tmp_path)

    code, out, err = run_cli(command(data_root, universe, extra=["--initial-cash"]), capsys)

    assert code == 0
    assert err == ""
    assert out == "\n".join(INITIAL_CASH_REPORT) + "\n"


def test_the_metadata_block_is_always_in_the_documented_order(tmp_path: Path) -> None:
    """The same first lines, in the same order, every time."""
    data_root, universe = scenario(tmp_path)

    text = format_summary(run_api(data_root, universe))

    assert [key for key, _ in metadata(text)] == [
        *SUMMARY_FIELDS[:4],
        "required_factor",
        *SUMMARY_FIELDS[4:],
    ]


def test_every_weight_is_printed_with_six_decimals(tmp_path: Path) -> None:
    """Weights are shares printed at six decimals -- never a raw float."""
    data_root, universe = scenario(tmp_path)

    text = format_summary(rebalance_api(data_root, universe))

    for line in table(text, "target")[1:]:
        assert line[1] == f"{float(line[1]):.6f}"
    for line in table(text, "transition")[1:]:
        if line == [EMPTY_TABLE_TEXT]:
            continue
        for value in line[1:4]:
            assert value == f"{float(value):.6f}"
    for value in cash_block(text).values():
        assert value == f"{float(value):.6f}"


def test_the_target_table_lists_only_the_targets_positions(tmp_path: Path) -> None:
    """No cash row, no score column: the table is the target portfolio."""
    data_root, universe = scenario(tmp_path)

    text = format_summary(run_api(data_root, universe))
    header, *body = table(text, "target")

    assert header == ["symbol", "target_weight"]
    assert [row[0] for row in body] == list(TOP_THREE)
    assert "CASH" not in {row[0] for row in body}


def test_the_transition_table_lists_the_whole_union(tmp_path: Path) -> None:
    """One row per symbol of both portfolios, target names first."""
    data_root, universe = scenario(tmp_path)

    text = format_summary(rebalance_api(data_root, universe))
    header, *body = table(text, "transition")

    assert header == ["symbol", "current_weight", "target_weight", "weight_change", "status"]
    assert [row[0] for row in body] == [*TOP_THREE, "AAPL"]


def test_the_status_column_uses_the_documented_words(tmp_path: Path) -> None:
    """``entry``, ``exit``, ``increase``, ``decrease``, ``unchanged`` -- nothing else."""
    scores = dict.fromkeys(SYMBOLS, 0.0)
    scores.update({"META": 1.0, "MSFT": 1.0, "NVDA": 1.0})
    data_root, universe = scenario(tmp_path, scores=scores)

    text = format_summary(rebalance_api(data_root, universe, holdings={"MSFT": 0.30}, cash=0.70))

    assert statuses(text) == {"META": "entry", "MSFT": "unchanged", "NVDA": "entry"}
    assert set(statuses(text).values()) <= {
        "entry",
        "exit",
        "increase",
        "decrease",
        "unchanged",
    }


def test_the_status_words_of_a_full_rebalance(tmp_path: Path) -> None:
    """Increase, decrease, entry and exit on one report."""
    data_root, universe = scenario(tmp_path)

    text = format_summary(rebalance_api(data_root, universe))

    assert statuses(text) == {
        "META": "entry",
        "MSFT": "decrease",
        "NVDA": "increase",
        "AAPL": "exit",
    }


def test_an_empty_target_and_an_empty_transition_print_none(tmp_path: Path) -> None:
    """All cash to all cash: the headers stay, the rows are ``none``."""
    data_root, universe = scenario(tmp_path, eligible=dict.fromkeys(SYMBOLS, False))

    text = format_summary(run_api(data_root, universe))

    assert table(text, "target") == [["symbol", "target_weight"], [EMPTY_TABLE_TEXT]]
    assert table(text, "transition") == [
        ["symbol", "current_weight", "target_weight", "weight_change", "status"],
        [EMPTY_TABLE_TEXT],
    ]
    assert field(text, "selected_count") == "0"
    assert field(text, "turnover") == "0.000000"
    assert cash_block(text) == {
        "current_weight": "1.000000",
        "target_weight": "1.000000",
        "weight_change": "0.000000",
    }


def test_the_cash_block_completes_the_report(tmp_path: Path) -> None:
    """The last four lines are the cash side of the difference."""
    data_root, universe = scenario(tmp_path)

    text = format_summary(rebalance_api(data_root, universe))

    assert text.splitlines()[-4:] == [
        "cash:",
        "current_weight: 0.100000",
        "target_weight: 0.100000",
        "weight_change: 0.000000",
    ]


def test_the_report_has_no_frame_index_dtype_or_nan(tmp_path: Path) -> None:
    """A DataFrame rendering would leak an index, a dtype or ``NaN``; this is text."""
    data_root, universe = scenario(tmp_path, eligible=dict.fromkeys(SYMBOLS, False))

    for outcome in (run_api(data_root, universe), rebalance_api(data_root, universe)):
        text = format_summary(outcome)
        for absent in ("NaN", "nan", "dtype", "None", "float64", "object"):
            assert absent not in text
        assert not any(
            line.lstrip()[:1].isdigit() and "  " in line.split(":")[0]
            for line in text.splitlines()[: len(SUMMARY_FIELDS)]
        )


def test_no_line_of_the_report_ends_in_whitespace(tmp_path: Path) -> None:
    """A table is alignment, not padding: every line is stripped."""
    data_root, universe = scenario(tmp_path)

    text = format_summary(rebalance_api(data_root, universe))

    assert all(line == line.rstrip() for line in text.splitlines())


def test_one_required_factor_line_per_factor(tmp_path: Path) -> None:
    """The identity of every exact factor read is on its own line."""
    data_root, universe = scenario(tmp_path)

    outcome = run_api(data_root, universe)
    text = format_summary(outcome)
    lines = [value for key, value in metadata(text) if key == "required_factor"]

    assert lines == [
        str(requirement) for requirement in outcome.target_portfolio.factor_requirements
    ]
    assert lines == ["momentum_12_1/1.0.0"]


def test_the_report_names_the_construction_that_set_the_weights(tmp_path: Path) -> None:
    """Rule and version are always printed, so a target can be reproduced."""
    data_root, universe = scenario(tmp_path)

    text = format_summary(run_api(data_root, universe))

    assert field(text, "construction_name") == "capped_score_weighted"
    assert field(text, "construction_version") == "1.0.0"
    assert field(text, "target_cash_weight") == "0.100000"
    assert field(text, "max_single_weight") == "0.500000"
    assert field(text, "target_position_count") == "3"
    assert field(text, "strategy_name") == "top_3_by_momentum_12_1"


# ---------------------------------------------------------------------------
# 7. The command line and the public API
# ---------------------------------------------------------------------------
def test_main_returns_zero_and_prints_the_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``main`` is the shell's view of :func:`run_portfolio`: same data, as text."""
    data_root, universe = scenario(tmp_path)

    code, out, err = run_cli(command(data_root, universe, extra=["--initial-cash"]), capsys)

    assert code == 0
    assert err == ""
    assert out.splitlines() == list(INITIAL_CASH_REPORT)


def test_the_documented_manual_command_prints_the_rebalance(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Mode B on the command line: the state flags produce the same rows as the API."""
    data_root, universe = scenario(tmp_path)

    argv = command(data_root, universe, extra=manual_flags(REBALANCE_HOLDINGS, cash=REBALANCE_CASH))
    code, out, err = run_cli(argv, capsys)

    assert code == 0
    assert err == ""
    assert field(out, "current_state_as_of_date") == PRIOR.isoformat()
    assert statuses(out) == {
        "META": "entry",
        "MSFT": "decrease",
        "NVDA": "increase",
        "AAPL": "exit",
    }


@pytest.mark.parametrize(
    ("extra", "detail"),
    [
        (["--initial-cash", "--current-cash-weight", "1.0"], "--initial-cash"),
        (
            ["--current-position", "AAPL=0.60", "--current-cash-weight", "0.40"],
            "--current-as-of",
        ),
        (
            [
                "--current-as-of",
                PRIOR.isoformat(),
                "--current-cash-weight",
                "0.40",
                "--current-position",
                "AAPL",
            ],
            "SYMBOL=WEIGHT",
        ),
        (
            ["--current-as-of", LATER.isoformat(), "--current-cash-weight", "1.0"],
            "look-ahead",
        ),
        (
            [
                "--current-as-of",
                PRIOR.isoformat(),
                "--current-cash-weight",
                "0.10",
                "--current-position",
                "AAPL=0.60",
            ],
            "PortfolioState",
        ),
    ],
)
def test_a_failing_command_prints_one_error_line(
    extra: Sequence[str],
    detail: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Every command failure is exit ``1``, one ``error:`` line and no traceback."""
    data_root, universe = scenario(tmp_path)

    code, out, err = run_cli(command(data_root, universe, extra=extra), capsys)

    assert code == 1
    assert out == ""
    assert err.startswith("error: ")
    assert detail in err
    assert "Traceback" not in err


def test_a_cap_that_cannot_be_honoured_fails_on_the_command_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A construction error is exit ``1`` too, with its own arithmetic kept."""
    data_root, universe = scenario(tmp_path)

    code, out, err = run_cli(
        command(data_root, universe, cap="0.25", extra=["--initial-cash"]), capsys
    )

    assert code == 1
    assert out == ""
    assert err.startswith("error: ")
    assert "cap" in err or "max_single_weight" in err
    assert "Traceback" not in err


def test_a_selected_score_of_zero_fails_on_the_command_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The rule refuses the selection, and the command reports it in one line."""
    scores = dict.fromkeys(SYMBOLS, 0.0)
    data_root, universe = scenario(tmp_path, scores=scores)

    code, out, err = run_cli(command(data_root, universe, extra=["--initial-cash"]), capsys)

    assert code == 1
    assert out == ""
    assert err.startswith("error: ")
    assert "score" in err
    assert "Traceback" not in err


def test_a_missing_panel_is_a_domain_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No panel on disk means no construction: one line, no recompute, no download."""
    universe = write_universe(tmp_path)
    data_root = tmp_path / "data"

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("a missing panel must not be written")

    monkeypatch.setattr(ParquetFactorStore, "write", forbidden)

    code, out, err = run_cli(command(data_root, universe, extra=["--initial-cash"]), capsys)

    assert code == 1
    assert out == ""
    assert err.startswith("error: ")
    assert "momentum_12_1" in err
    assert "Traceback" not in err
    assert not data_root.exists()


def test_a_blank_data_root_is_a_domain_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A whitespace root is refused before any panel is looked for."""
    universe = write_universe(tmp_path)

    code, out, err = run_cli(
        command(None, universe, extra=["--initial-cash", "--data-root", "   "]), capsys
    )

    assert code == 1
    assert out == ""
    assert err.startswith("error: ")
    assert "data-root" in err or "data_root" in err
    assert "Traceback" not in err


def test_the_parser_accepts_only_its_declared_flags() -> None:
    """Thirteen flags: the strategy's seven, two constraints and four state values."""
    parser = build_parser()
    assert {action.dest for action in parser._actions} == EXPECTED_DESTS

    args = parser.parse_args(command(extra=["--initial-cash"]))
    assert args.factor == "momentum_12_1"
    assert args.factor_version == "1.0.0"
    assert args.top_n == 3
    assert args.as_of == AS_OF.isoformat()
    assert args.strategy_version == DEFAULT_STRATEGY_VERSION
    assert args.data_root is None
    assert args.universe_path is None
    assert args.target_cash_weight == CASH_WEIGHT
    assert args.max_single_weight == MAX_SINGLE_WEIGHT
    assert args.initial_cash is True
    assert args.current_as_of is None
    assert args.current_position is None
    assert args.current_cash_weight is None


def test_the_parser_collects_repeated_positions_in_order() -> None:
    """``--current-position`` is a repeatable flag, never a comma-separated list."""
    args = build_parser().parse_args(
        command(extra=manual_flags({"AAPL": 0.6, "MSFT": 0.3}, cash=0.1))
    )

    assert args.current_position == ["AAPL=0.6", "MSFT=0.3"]
    assert args.current_as_of == PRIOR.isoformat()
    assert args.current_cash_weight == 0.1


def test_the_parser_names_the_defaults_it_will_use() -> None:
    """The help text never claims a different root, pool or version than the run uses."""
    text = build_parser().format_help()
    assert str(PipelineConfig().data_root) in text
    assert DEFAULT_STRATEGY_VERSION in text
    assert "config/universe_seed.csv" in text
    assert "--initial-cash" in text
    assert "--constructor" not in text


def test_the_run_function_is_callable_without_argv() -> None:
    """A notebook can call it directly: keyword-only, with the documented defaults."""
    signature = inspect.signature(run_portfolio)
    assert list(signature.parameters) == [
        "factor_name",
        "factor_version",
        "top_n",
        "as_of_date",
        "target_cash_weight",
        "max_single_weight",
        "initial_cash",
        "current_as_of_date",
        "current_positions",
        "current_cash_weight",
        "strategy_version",
        "data_root",
        "universe_path",
    ]
    assert {parameter.kind for parameter in signature.parameters.values()} == {
        inspect.Parameter.KEYWORD_ONLY
    }
    assert signature.parameters["initial_cash"].default is False
    assert signature.parameters["current_as_of_date"].default is None
    assert signature.parameters["current_positions"].default is None
    assert signature.parameters["current_cash_weight"].default is None
    assert signature.parameters["strategy_version"].default == DEFAULT_STRATEGY_VERSION
    assert signature.parameters["data_root"].default is None
    assert signature.parameters["universe_path"].default is None


def test_the_module_exports_its_documented_api() -> None:
    """Everything a caller may import is named in ``__all__``."""
    assert set(run_portfolio_module.__all__) == {
        "DEFAULT_STRATEGY_VERSION",
        "EMPTY_TABLE_TEXT",
        "PortfolioCommandError",
        "PortfolioRunOutcome",
        "WEIGHT_DECIMALS",
        "build_parser",
        "check_current_state_arguments",
        "format_summary",
        "main",
        "parse_current_position",
        "parse_current_positions",
        "run_portfolio",
    }
    assert issubclass(PortfolioCommandError, PortfolioError)


def test_the_documented_command_runs_in_a_real_process(tmp_path: Path) -> None:
    """``python scripts/run_portfolio.py ...`` works from the project root."""
    data_root, universe = scenario(tmp_path)

    completed = subprocess.run(
        [sys.executable, str(SCRIPT), *command(data_root, universe, extra=["--initial-cash"])],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=300,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    assert completed.stdout.splitlines() == list(INITIAL_CASH_REPORT)


def test_the_documented_manual_command_runs_in_a_real_process(tmp_path: Path) -> None:
    """The rebalance form works in a real process too, from the project root."""
    data_root, universe = scenario(tmp_path)

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            *command(
                data_root,
                universe,
                extra=manual_flags(REBALANCE_HOLDINGS, cash=REBALANCE_CASH),
            ),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=300,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert f"current_state_as_of_date: {PRIOR.isoformat()}" in completed.stdout
    assert "AAPL    0.300000        0.000000       -0.300000      exit" in completed.stdout


def test_the_documented_command_fails_with_one_line_in_a_real_process(
    tmp_path: Path,
) -> None:
    """A missing panel is exit ``1`` and one ``error:`` line, in a real process too."""
    universe = write_universe(tmp_path)

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            *command(tmp_path / "data", universe, extra=["--initial-cash"]),
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
    assert completed.stderr.startswith("error: ")
    assert "Traceback" not in completed.stderr


def test_an_unknown_flag_exits_two_in_a_real_process(tmp_path: Path) -> None:
    """argparse keeps its own exit code in a real process as well."""
    data_root, universe = scenario(tmp_path)

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            *command(data_root, universe, extra=["--initial-cash", "--frequency", "weekly"]),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=300,
        check=False,
    )

    assert completed.returncode == 2
    assert completed.stdout == ""
    assert "usage:" in completed.stderr


# ---------------------------------------------------------------------------
# 8. Offline, read-only and self-consistent
# ---------------------------------------------------------------------------
def test_a_run_never_downloads_computes_or_writes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A successful run touches the network, the factor layer and the disk never."""
    data_root, universe = scenario(tmp_path)
    before = panel_signature(data_root)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("the portfolio command must only read stored panels")

    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(ParquetFactorStore, "write", forbidden)
    monkeypatch.setattr(FactorRunner, "run", forbidden)

    code, out, err = run_cli(command(data_root, universe, extra=["--initial-cash"]), capsys)

    assert code == 0
    assert err == ""
    assert field(out, "selected_count") == "3"
    assert panel_signature(data_root) == before

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
    for absent in ("state", "states", "orders", "fills", "reports", "backtest"):
        assert not (tmp_path / absent).exists()
        assert not (data_root / absent).exists()


def test_a_rebalance_run_writes_nothing_either(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The stated account is an argument, so there is no state file to save."""
    data_root, universe = scenario(tmp_path)
    before = panel_signature(data_root)
    listing = sorted(path.name for path in tmp_path.iterdir())

    code, _, err = run_cli(
        command(data_root, universe, extra=manual_flags(REBALANCE_HOLDINGS, cash=REBALANCE_CASH)),
        capsys,
    )

    assert code == 0
    assert err == ""
    assert panel_signature(data_root) == before
    assert sorted(path.name for path in tmp_path.iterdir()) == listing


def test_the_layer_errors_keep_their_own_types(tmp_path: Path) -> None:
    """Each layer's refusal keeps its own type, so a caller can branch on it."""
    universe = write_universe(tmp_path)
    empty_root = tmp_path / "data"

    with pytest.raises(FactorError):
        run_api(empty_root, universe)
    with pytest.raises(StrategyError):
        run_portfolio(
            factor_name="momentum_12_1",
            factor_version="1.0.0",
            top_n=3,
            as_of_date=AS_OF,
            target_cash_weight=CASH_WEIGHT,
            max_single_weight=MAX_SINGLE_WEIGHT,
            initial_cash=True,
            data_root="",
            universe_path=universe,
        )

    data_root, universe = scenario(tmp_path / "real")
    outcome = run_api(data_root, universe)

    assert isinstance(outcome.strategy_outcome, StrategyRunOutcome)
    assert isinstance(outcome.target_portfolio, TargetPortfolio)
    assert isinstance(outcome.transition, PortfolioTransition)
    assert isinstance(outcome.current_state, PortfolioState)
    assert isinstance(outcome.constraints, PortfolioConstraints)
    assert isinstance(outcome.construction_spec, PortfolioConstructionSpec)


def test_the_outcome_is_frozen(tmp_path: Path) -> None:
    """A result object cannot be edited into describing another run."""
    data_root, universe = scenario(tmp_path)

    outcome = run_api(data_root, universe)

    with pytest.raises(FrozenInstanceError):
        outcome.target_portfolio = None  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        outcome.transition = None  # type: ignore[misc]


def test_the_outcome_refuses_parts_that_do_not_describe_one_run(tmp_path: Path) -> None:
    """A target of another run, or a transition of another target, is refused."""
    data_root, universe = scenario(tmp_path)
    first = run_api(data_root, universe)
    second = run_api(data_root, universe)

    with pytest.raises(PortfolioContractError):
        replace(first, target_portfolio=second.target_portfolio)
    with pytest.raises(PortfolioContractError):
        replace(first, transition=second.transition)
    with pytest.raises(PortfolioContractError):
        replace(first, strategy_outcome=object())  # type: ignore[arg-type]
    with pytest.raises(PortfolioContractError):
        replace(first, target_portfolio=first.strategy_result)  # type: ignore[arg-type]
    with pytest.raises(PortfolioContractError):
        replace(first, transition=first.target_portfolio)  # type: ignore[arg-type]


def test_the_outcome_accepts_the_same_run_described_again(tmp_path: Path) -> None:
    """Re-describing one run is legal: the identity checks are about the same objects."""
    data_root, universe = scenario(tmp_path)

    outcome = run_api(data_root, universe)
    rebuilt = build_portfolio_transition(
        current_state=outcome.current_state,
        target_portfolio=outcome.target_portfolio,
    )

    again = replace(outcome, transition=rebuilt)

    assert again.transition is rebuilt
    assert format_summary(again) == format_summary(outcome)


def test_the_outcome_refuses_a_transition_taken_against_another_target(
    tmp_path: Path,
) -> None:
    """A difference belongs to one rule under one set of constraints, and says so."""
    data_root, universe = scenario(tmp_path)

    outcome = run_api(data_root, universe)
    target = outcome.target_portfolio

    other_constraints = PortfolioConstraints(cash_weight=0.20, max_single_weight=0.50)
    scale = 0.80 / target.invested_weight
    other_target = TargetPortfolio(
        as_of_date=target.as_of_date,
        construction_spec=PortfolioConstructionSpec(
            name=target.construction_spec.name, version="9.9.9"
        ),
        source_result=target.source_result,
        constraints=other_constraints,
        positions=tuple(
            PortfolioPosition(position.symbol, position.weight * scale)
            for position in target.positions
        ),
        cash_weight=0.20,
    )
    other_transition = build_portfolio_transition(
        current_state=outcome.current_state,
        target_portfolio=other_target,
    )

    with pytest.raises(PortfolioContractError) as excinfo:
        PortfolioRunOutcome(
            strategy_outcome=outcome.strategy_outcome,
            target_portfolio=target,
            transition=other_transition,
        )

    message = str(excinfo.value)
    assert "construction_spec" in message
    assert "constraints" in message


def test_the_outcome_repr_names_the_run(tmp_path: Path) -> None:
    """The repr is a one-line identity of the construction, not a dump of weights."""
    data_root, universe = scenario(tmp_path)

    text = repr(run_api(data_root, universe))

    assert text.startswith("PortfolioRunOutcome(")
    assert f"as_of_date={AS_OF.isoformat()!r}" in text
    assert "construction='capped_score_weighted'" in text
    assert "selected_count=3" in text
    assert "target_position_count=3" in text
    assert "turnover=0.9" in text
    assert "\n" not in text


# ---------------------------------------------------------------------------
# 9. The module boundary, asserted on the AST
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
    """stdlib plumbing plus the documented entry points of four layers, nothing else."""
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
        "portfolio_sys",
        "portfolio_sys.constructors",
        "scripts.run_strategy",
        "strategy_sys",
    )
    for module in sorted(modules):
        assert any(module == prefix or module.startswith(f"{prefix}.") for prefix in allowed), (
            f"the entry point must not import {module!r}"
        )
    assert {
        "data_sys.config",
        "factor_sys",
        "portfolio_sys",
        "portfolio_sys.constructors",
        "scripts.run_strategy",
        "strategy_sys",
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
        "os",
        "json",
        "csv",
        "shutil",
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
        "portfolio_sys.contracts",
        "portfolio_sys.errors",
        "portfolio_sys.registry",
        "portfolio_sys.transition",
        "portfolio_sys.constructors.capped_score_weighted",
        "strategy_sys.strategies",
        "strategy_sys.strategy_runner",
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
    """The registry, the constructor and the transition are each used by one call."""
    names = code_names(script_tree())
    required = {
        "run_strategy",
        "PortfolioConstraints",
        "PortfolioState",
        "PortfolioContext",
        "PortfolioConstructorRegistry",
        "CappedScoreWeightedConstructor",
        "register",
        "create",
        "construct",
        "build_portfolio_transition",
        "TargetPortfolio",
        "PortfolioTransition",
        "PortfolioRunOutcome",
    }
    assert required <= names
    # The strategy layer is *used*, never rebuilt: no registry, runner, store, pool or
    # context of its own appears here.
    for forbidden in (
        "StrategyRegistry",
        "StrategyRunner",
        "StrategySpec",
        "StrategyContext",
        "FactorContext",
        "FactorRequirement",
        "TopNByFactorStrategy",
        "StaticUniverseProvider",
        "ParquetFactorStore",
        "compute",
    ):
        assert forbidden not in names


def test_the_script_never_opens_reads_or_writes_a_file() -> None:
    """Paths only name a process argument: nothing is opened, written or removed."""
    names = code_names(script_tree())
    forbidden = {
        "open",
        "write",
        "writelines",
        "write_text",
        "write_bytes",
        "read_text",
        "read_bytes",
        "read_csv",
        "read_parquet",
        "to_csv",
        "to_parquet",
        "mkdir",
        "unlink",
        "rename",
        "rmdir",
        "dump",
        "dumps",
        "save",
        "load",
    }
    assert forbidden.isdisjoint(names)

    text = "\n".join(sorted(code_constants(script_tree())))
    for fragment in ("factor_name=", "factor_version=", "year=", ".parquet"):
        assert fragment not in text


def test_the_script_has_no_execution_or_backtest_logic() -> None:
    """Orders, fills, prices, costs and frequencies belong to later layers."""
    names = code_names(script_tree())
    forbidden = {
        "order",
        "orders",
        "fill",
        "fills",
        "shares",
        "quantity",
        "notional",
        "price",
        "prices",
        "commission",
        "commission_rate",
        "slippage",
        "cost",
        "costs",
        "backtest",
        "Backtest",
        "rebalance",
        "rebalance_calendar",
        "schedule",
        "frequency",
        "anchor",
        "calendar",
        "download",
        "download_bars",
        "fetch_bars",
        "provider",
        "yfinance",
    }
    assert forbidden.isdisjoint(names)

    text = "\n".join(sorted(code_constants(script_tree())))
    for fragment in ("--frequency", "--force", "--save-state", "--load-state"):
        assert fragment not in text


def test_the_script_does_not_restate_the_weighting_rule() -> None:
    """No weighting arithmetic: the constructor's shares are read, never recomputed."""
    tree = script_tree()

    arithmetic = {type(node.op).__name__ for node in ast.walk(tree) if isinstance(node, ast.BinOp)}
    assert arithmetic.isdisjoint({"Div", "Pow", "Mult", "FloorDiv", "Mod"})

    text = "\n".join(sorted(code_constants(tree)))
    for column in ("is_selected", "raw_value", "is_eligible"):
        assert column not in text

    names = code_names(tree)
    for forbidden in ("cap", "investable", "capped", "proportional", "normalise", "normalize"):
        assert forbidden not in names
    # No frame is touched either: a selection is read through the layer's objects.
    for forbidden in ("frame", "itertuples", "loc", "iloc", "sort_values", "astype"):
        assert forbidden not in names


def test_the_script_never_reaches_into_the_library_internals() -> None:
    """No private attribute of any layer is touched, and no private helper is defined."""
    tree = script_tree()
    private_members = {
        "_find",
        "_factories",
        "_issued",
        "_spec_of",
        "_claim",
        "_built",
        "_key",
        "_rows",
        "_expected_symbols",
        "_capped_weights",
        "_selected_scores",
        "_score_problems",
        "_feasibility_problems",
        "_starved_problems",
        "_panel_at",
        "_atomic_write",
        "_root_directory",
        "_selected_column",
    }
    assert private_members.isdisjoint(code_names(tree))

    imported = {name for name in imported_names(tree) if name.startswith("_")}
    assert imported == set()


def test_the_script_defines_no_cli_only_shortcut() -> None:
    """The command adds no default registry, no auto-registration and no rule table."""
    tree = script_tree()
    names = code_names(tree)
    for forbidden in ("default_portfolio_registry", "auto_register", "CONSTRUCTORS", "RULES"):
        assert forbidden not in names

    defined = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    assert defined == {
        "PortfolioCommandError",
        "PortfolioRunOutcome",
        "_joined",
        "check_current_state_arguments",
        "parse_current_position",
        "parse_current_positions",
        "_current_state",
        "run_portfolio",
        "_format_weight",
        "_transition_status",
        "_table",
        "format_summary",
        "build_parser",
        "_parser_for",
        "main",
    }
