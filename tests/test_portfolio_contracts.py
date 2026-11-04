"""Tests for the portfolio data contract (:mod:`portfolio_sys.contracts`).

Everything here is in memory and offline: the strategy result is built from a
``UniverseSnapshot`` plus a small frame, the current state is a tuple of
positions, and no ``data/`` directory, factor store, trading calendar, network
call or DuckDB connection is involved.  That is exactly the shape a future
constructor will be handed, so these tests pin the shared vocabulary down before
any weighting rule exists.

Covered in this order:

* the abstract :class:`PortfolioConstructor` interface, and a minimal fake,
* ``PortfolioPosition``: one holding, its symbol rule and its weight rule,
* ``PortfolioState``: what the account holds right now,
* ``PortfolioConstraints``: the limits a construction works within,
* ``PortfolioConstructionSpec``: which rule built the weights,
* ``PortfolioContext``: everything one construction may read,
* ``TargetPortfolio``: what the account should hold after the decision,
* collected problems: one error names them all,
* errors, package surface and the architectural boundary.
"""

from __future__ import annotations

import ast
import sys
from dataclasses import FrozenInstanceError, fields
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import portfolio_sys
import portfolio_sys.contracts as contracts_module
from data_sys.errors import DataPipelineError
from factor_sys import FactorError, UniverseSnapshot
from portfolio_sys import (
    CONSTRUCTION_NAME_PATTERN,
    PortfolioConstraints,
    PortfolioConstructionSpec,
    PortfolioConstructor,
    PortfolioContext,
    PortfolioContractError,
    PortfolioError,
    PortfolioPosition,
    PortfolioState,
    TargetPortfolio,
)
from strategy_sys import (
    STRATEGY_NAME_PATTERN,
    FactorRequirement,
    StrategyError,
    StrategyResult,
    StrategySpec,
)

#: The in-memory universe every fixture below works with.
SYMBOLS: tuple[str, ...] = ("AAPL", "MSFT", "NVDA", "TSLA")

#: The decision date of every fixture; the state a rebalance starts from is
#: dated one session earlier, and ``LATER_SESSION`` must never be accepted as a
#: state for ``AS_OF``.
AS_OF: date = date(2024, 1, 5)
PREVIOUS_SESSION: date = date(2024, 1, 4)
LATER_SESSION: date = date(2024, 1, 8)

#: Two exact factor identities: one the fixture spec reads, one it does not.
MOMENTUM = FactorRequirement(factor_name="momentum_12_1", factor_version="1.0.0")
VALUE = FactorRequirement(factor_name="book_to_price", factor_version="2.0.0")

#: The spec the fixture result is produced under.
SPEC = StrategySpec(name="top_n_momentum", version="1.0.0", required_factors=(MOMENTUM,))

#: The same rule computed from another version of the same factor.
SPEC_ON_FACTOR_V2 = StrategySpec(
    name="top_n_momentum",
    version="1.0.0",
    required_factors=(FactorRequirement(factor_name="momentum_12_1", factor_version="2.0.0"),),
)

#: Which symbols the fixture result selects, in universe order.
SELECTED: tuple[str, ...] = ("MSFT", "NVDA")

#: The construction identity the fixtures build targets with.
CONSTRUCTION = PortfolioConstructionSpec(name="equal_weight", version="1.0.0")


# ---------------------------------------------------------------------------
# In-memory fixtures (no disk, no network, no store)
# ---------------------------------------------------------------------------
def make_universe(
    symbols: tuple[str, ...] = SYMBOLS,
    *,
    as_of_date: object = AS_OF,
) -> UniverseSnapshot:
    """A :class:`UniverseSnapshot` for the fixtures."""
    return UniverseSnapshot(  # type: ignore[arg-type]
        as_of_date=as_of_date,
        symbols=tuple(symbols),
        source="config/universe_seed.csv",
    )


def make_result(
    *,
    symbols: tuple[str, ...] = SYMBOLS,
    selected: tuple[str, ...] = SELECTED,
    as_of_date: object = AS_OF,
    spec: StrategySpec = SPEC,
) -> StrategyResult:
    """A strategy result whose ``is_selected`` flags are exactly ``selected``."""
    frame = pd.DataFrame(
        {
            "as_of_date": [pd.Timestamp(as_of_date)] * len(symbols),
            "strategy_name": [spec.name] * len(symbols),
            "strategy_version": [spec.version] * len(symbols),
            "symbol": list(symbols),
            "score": [round(1.0 - 0.1 * position, 2) for position in range(len(symbols))],
            "rank": list(range(1, len(symbols) + 1)),
            "is_selected": [symbol in selected for symbol in symbols],
        }
    )
    return StrategyResult(
        frame,
        universe=make_universe(symbols, as_of_date=as_of_date),
        strategy_spec=spec,
        as_of_date=as_of_date,  # type: ignore[arg-type]
    )


def make_positions(**weights: float) -> tuple[PortfolioPosition, ...]:
    """A holding tuple, e.g. ``make_positions(AAPL=0.6, MSFT=0.4)``."""
    return tuple(PortfolioPosition(symbol, weight) for symbol, weight in weights.items())


def make_state(
    positions: object = None,
    *,
    cash_weight: object = 0.0,
    as_of_date: object = PREVIOUS_SESSION,
) -> PortfolioState:
    """The account before the decision: by default entirely in ``AAPL``."""
    holdings = make_positions(AAPL=1.0) if positions is None else positions
    return PortfolioState(  # type: ignore[arg-type]
        as_of_date=as_of_date,
        positions=holdings,
        cash_weight=cash_weight,
    )


def make_constraints(
    *,
    cash_weight: object = 0.0,
    max_single_weight: object = 1.0,
) -> PortfolioConstraints:
    """The base constraints: fully invested and uncapped, unless told otherwise."""
    return PortfolioConstraints(  # type: ignore[arg-type]
        cash_weight=cash_weight,
        max_single_weight=max_single_weight,
    )


def make_context(
    *,
    as_of_date: object = AS_OF,
    strategy_result: object = None,
    current_state: object = None,
    constraints: object = None,
) -> PortfolioContext:
    """A complete construction input for the fixture result."""
    return PortfolioContext(  # type: ignore[arg-type]
        as_of_date=as_of_date,
        strategy_result=make_result() if strategy_result is None else strategy_result,
        current_state=make_state() if current_state is None else current_state,
        constraints=make_constraints() if constraints is None else constraints,
    )


def make_target(
    weights: dict[str, float] | None = None,
    *,
    cash_weight: object = 0.0,
    as_of_date: object = AS_OF,
    construction_spec: object = CONSTRUCTION,
    source_result: object = None,
    constraints: object = None,
) -> TargetPortfolio:
    """An equally weighted target over ``MSFT``/``NVDA`` unless told otherwise."""
    holdings = (
        make_positions(MSFT=0.5, NVDA=0.5)
        if weights is None
        else tuple(PortfolioPosition(symbol, weight) for symbol, weight in weights.items())
    )
    return TargetPortfolio(  # type: ignore[arg-type]
        as_of_date=as_of_date,
        construction_spec=construction_spec,
        source_result=make_result() if source_result is None else source_result,
        constraints=make_constraints() if constraints is None else constraints,
        positions=holdings,
        cash_weight=cash_weight,
    )


class EqualSplitConstructor(PortfolioConstructor):
    """A minimal in-memory constructor: split the selection evenly, keep the cash.

    It exists only to prove the interface can be implemented without any storage,
    price or calendar -- *not* as the equal-weight rule of a later task.
    """

    spec = PortfolioConstructionSpec(name="fake_equal_split", version="1.0.0")

    def __init__(self, cash_weight: float = 0.0) -> None:
        self.cash_weight = cash_weight

    def construct(self, context: PortfolioContext) -> TargetPortfolio:
        symbols = context.selected_symbols
        share = (1.0 - self.cash_weight) / len(symbols)
        return TargetPortfolio(
            as_of_date=context.as_of_date,
            construction_spec=self.spec,
            source_result=context.strategy_result,
            constraints=context.constraints,
            positions=tuple(PortfolioPosition(symbol, share) for symbol in symbols),
            cash_weight=self.cash_weight,
        )


# ---------------------------------------------------------------------------
# PortfolioConstructor -- the interface
# ---------------------------------------------------------------------------
def test_portfolio_constructor_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        PortfolioConstructor()  # type: ignore[abstract]


def test_a_constructor_must_declare_both_a_spec_and_a_construct() -> None:
    """Either half on its own leaves the class abstract."""

    class OnlySpec(PortfolioConstructor):
        spec = CONSTRUCTION

    class OnlyConstruct(PortfolioConstructor):
        def construct(self, context: PortfolioContext) -> TargetPortfolio:
            return make_target()

    for incomplete in (OnlySpec, OnlyConstruct):
        with pytest.raises(TypeError):
            incomplete()  # type: ignore[abstract]


def test_a_constructor_may_declare_its_spec_as_a_class_attribute_or_a_property() -> None:
    class AsProperty(PortfolioConstructor):
        def __init__(self) -> None:
            self._spec = CONSTRUCTION

        @property
        def spec(self) -> PortfolioConstructionSpec:
            return self._spec

        def construct(self, context: PortfolioContext) -> TargetPortfolio:
            return make_target()

    assert AsProperty().spec is CONSTRUCTION
    assert EqualSplitConstructor().spec == PortfolioConstructionSpec("fake_equal_split", "1.0.0")


def test_the_interface_declares_exactly_a_spec_and_a_construct() -> None:
    assert PortfolioConstructor.__abstractmethods__ == {"spec", "construct"}
    assert isinstance(PortfolioConstructor.spec, property)


def test_a_minimal_constructor_builds_a_valid_target() -> None:
    context = make_context()

    target = EqualSplitConstructor().construct(context)

    assert isinstance(target, TargetPortfolio)
    assert target.as_of_date == AS_OF
    assert target.construction_spec == EqualSplitConstructor.spec
    assert target.source_result is context.strategy_result
    assert target.constraints is context.constraints
    assert target.target_symbols == context.selected_symbols
    assert target.target_weights_by_symbol == {"MSFT": 0.5, "NVDA": 0.5}
    assert target.invested_weight == 1.0
    assert target.cash_weight == 0.0


def test_a_constructor_keeps_cash_out_of_the_holdings() -> None:
    target = EqualSplitConstructor(cash_weight=0.2).construct(make_context())

    assert target.cash_weight == 0.2
    assert target.target_weights_by_symbol == {"MSFT": 0.4, "NVDA": 0.4}
    assert "CASH" not in target.target_symbols


def test_a_constructor_is_a_pure_function_of_its_context() -> None:
    constructor = EqualSplitConstructor()
    context = make_context()
    before = context.strategy_result.frame

    first = constructor.construct(context)
    second = constructor.construct(context)

    assert first.target_weights_by_symbol == second.target_weights_by_symbol
    assert context.strategy_result.frame.equals(before)
    assert context.selected_symbols == SELECTED


def test_the_constructor_repr_names_its_rule() -> None:
    assert "fake_equal_split" in repr(EqualSplitConstructor())


# ---------------------------------------------------------------------------
# PortfolioPosition -- one non-cash holding
# ---------------------------------------------------------------------------
def test_a_position_holds_one_symbol_and_one_share() -> None:
    position = PortfolioPosition(symbol="NVDA", weight=0.09)

    assert position.symbol == "NVDA"
    assert position.weight == 0.09
    assert isinstance(position.weight, float)


def test_a_position_stores_its_weight_as_a_float() -> None:
    """``1`` and ``1.0`` are the same share of the account."""
    position = PortfolioPosition("AAPL", 1)

    assert position.weight == 1.0
    assert isinstance(position.weight, float)
    assert position == PortfolioPosition("AAPL", 1.0)


def test_positions_are_compared_by_value_and_are_hashable() -> None:
    assert PortfolioPosition("AAPL", 0.5) == PortfolioPosition("AAPL", 0.5)
    assert PortfolioPosition("AAPL", 0.5) != PortfolioPosition("AAPL", 0.6)
    assert PortfolioPosition("AAPL", 0.5) != PortfolioPosition("MSFT", 0.5)
    assert len({PortfolioPosition("AAPL", 0.5), PortfolioPosition("AAPL", 0.5)}) == 1


def test_a_position_is_frozen() -> None:
    position = PortfolioPosition("AAPL", 0.5)

    with pytest.raises(FrozenInstanceError):
        position.weight = 0.6  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        position.symbol = "MSFT"  # type: ignore[misc]


@pytest.mark.parametrize("symbol", ["AAPL", "BRK.B", "RDS-A", "F", "ZZZZ", "DELISTED"])
def test_a_position_accepts_the_projects_ticker_syntax(symbol: str) -> None:
    assert PortfolioPosition(symbol, 0.1).symbol == symbol


@pytest.mark.parametrize(
    "symbol",
    [
        "aapl",
        "Aapl",
        "",
        "  ",
        "AAPL ",
        " A",
        "A PL",
        "AAPL\n",
        "AAPL.N\n",
        "CASH",
        None,
        123,
        1.5,
        ["AAPL"],
        ("AAPL",),
    ],
)
def test_a_position_rejects_an_unusable_symbol(symbol: object) -> None:
    with pytest.raises(PortfolioContractError) as excinfo:
        PortfolioPosition(symbol, 0.1)  # type: ignore[arg-type]

    assert "symbol" in str(excinfo.value)


def test_a_position_explains_that_cash_is_a_field_not_a_ticker() -> None:
    with pytest.raises(PortfolioContractError) as excinfo:
        PortfolioPosition("CASH", 0.1)

    assert "cash_weight" in str(excinfo.value)


@pytest.mark.parametrize(
    "weight",
    [
        0,
        0.0,
        -0.5,
        -1,
        float("nan"),
        float("inf"),
        float("-inf"),
        True,
        False,
        "0.5",
        "1",
        None,
        [0.5],
        (0.5,),
    ],
)
def test_a_position_rejects_an_unusable_weight(weight: object) -> None:
    with pytest.raises(PortfolioContractError) as excinfo:
        PortfolioPosition("AAPL", weight)  # type: ignore[arg-type]

    assert "weight" in str(excinfo.value)


def test_a_position_collects_its_symbol_and_weight_problems_at_once() -> None:
    with pytest.raises(PortfolioContractError) as excinfo:
        PortfolioPosition("aapl", -0.5)

    problems = excinfo.value.problems
    assert len(problems) == 2
    assert problems[0].startswith("symbol")
    assert problems[1].startswith("weight")


def test_a_position_does_not_cap_its_own_weight() -> None:
    """A weight above 1 is a portfolio-level question, not a position's."""
    assert PortfolioPosition("AAPL", 2.0).weight == 2.0


def test_a_position_does_not_check_membership_of_any_universe() -> None:
    """One holding knows only itself; membership is the portfolio's question."""
    assert PortfolioPosition("NOTINUNIVERSE", 0.1).symbol == "NOTINUNIVERSE"


# ---------------------------------------------------------------------------
# PortfolioState -- what the account actually holds now
# ---------------------------------------------------------------------------
def test_a_state_holds_several_stocks_and_some_cash() -> None:
    state = PortfolioState(
        as_of_date=PREVIOUS_SESSION,
        positions=make_positions(AAPL=0.35, MSFT=0.25, NVDA=0.3),
        cash_weight=0.1,
    )

    assert state.as_of_date == PREVIOUS_SESSION
    assert state.symbols == ("AAPL", "MSFT", "NVDA")
    assert state.weights_by_symbol == {"AAPL": 0.35, "MSFT": 0.25, "NVDA": 0.3}
    assert state.invested_weight == pytest.approx(0.9)
    assert state.cash_weight == 0.1


def test_a_state_accepts_the_all_cash_starting_point() -> None:
    state = PortfolioState(PREVIOUS_SESSION, (), 1.0)

    assert state.positions == ()
    assert state.symbols == ()
    assert state.weights_by_symbol == {}
    assert state.invested_weight == 0.0
    assert state.cash_weight == 1.0


def test_a_state_accepts_weights_that_only_round_to_one() -> None:
    """Three equal thirds sum to 0.9999999999999999: a correct portfolio."""
    third = 1.0 / 3.0
    positions = make_positions(AAPL=third, MSFT=third, NVDA=third)

    state = PortfolioState(PREVIOUS_SESSION, positions, 0.0)

    assert state.invested_weight == pytest.approx(1.0)


def test_a_state_normalizes_an_iso_date_string() -> None:
    state = PortfolioState("2024-01-04", make_positions(AAPL=1.0), 0.0)

    assert state.as_of_date == PREVIOUS_SESSION
    assert type(state.as_of_date) is date


def test_a_state_is_frozen() -> None:
    state = make_state()

    with pytest.raises(FrozenInstanceError):
        state.cash_weight = 0.5  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        state.as_of_date = AS_OF  # type: ignore[misc]


def test_a_state_rejects_an_empty_portfolio_with_cash_left_over() -> None:
    with pytest.raises(PortfolioContractError) as excinfo:
        PortfolioState(PREVIOUS_SESSION, (), 0.5)

    assert "entirely in cash" in str(excinfo.value)


def test_a_state_rejects_a_total_that_is_not_one() -> None:
    with pytest.raises(PortfolioContractError) as excinfo:
        PortfolioState(PREVIOUS_SESSION, make_positions(AAPL=0.4, MSFT=0.4), 0.0)

    assert "add up to 1.0" in str(excinfo.value)


def test_a_state_rejects_leverage() -> None:
    with pytest.raises(PortfolioContractError) as excinfo:
        PortfolioState(PREVIOUS_SESSION, make_positions(AAPL=0.8, MSFT=0.8), 0.0)

    assert "add up to 1.0" in str(excinfo.value)


def test_a_state_rejects_a_duplicate_symbol() -> None:
    positions = (PortfolioPosition("AAPL", 0.5), PortfolioPosition("AAPL", 0.5))

    with pytest.raises(PortfolioContractError) as excinfo:
        PortfolioState(PREVIOUS_SESSION, positions, 0.0)

    assert "more than once" in str(excinfo.value)


@pytest.mark.parametrize(
    "positions",
    [
        [PortfolioPosition("AAPL", 1.0)],
        {PortfolioPosition("AAPL", 1.0)},
        PortfolioPosition("AAPL", 1.0),
        {"AAPL": 1.0},
        "AAPL",
        None,
    ],
)
def test_a_state_rejects_anything_but_a_tuple_of_positions(positions: object) -> None:
    with pytest.raises(PortfolioContractError) as excinfo:
        PortfolioState(PREVIOUS_SESSION, positions, 1.0)  # type: ignore[arg-type]

    assert "PortfolioState.positions" in str(excinfo.value)


def test_a_state_rejects_a_generator_of_positions() -> None:
    with pytest.raises(PortfolioContractError) as excinfo:
        PortfolioState(PREVIOUS_SESSION, (p for p in ()), 1.0)  # type: ignore[arg-type]

    assert "generator" in str(excinfo.value)


def test_a_state_rejects_a_bare_symbol_among_its_positions() -> None:
    with pytest.raises(PortfolioContractError) as excinfo:
        PortfolioState(PREVIOUS_SESSION, ("AAPL",), 0.0)  # type: ignore[arg-type]

    assert "must be a PortfolioPosition" in str(excinfo.value)


@pytest.mark.parametrize(
    "as_of_date",
    [
        datetime(2024, 1, 4),
        pd.Timestamp("2024-01-04"),
        pd.NaT,
        "2024-01-04T00:00",
        "01/04/2024",
        "2024-13-01",
        "2024-02-30",
        "",
        None,
        20240104,
    ],
)
def test_a_state_rejects_an_unusable_date(as_of_date: object) -> None:
    with pytest.raises(PortfolioContractError) as excinfo:
        PortfolioState(as_of_date, make_positions(AAPL=1.0), 0.0)  # type: ignore[arg-type]

    assert "as_of_date" in str(excinfo.value)


def test_a_state_explains_why_a_timestamp_is_not_a_date() -> None:
    """A midnight ``datetime`` is a moment, not the day a state was known."""
    with pytest.raises(PortfolioContractError) as excinfo:
        PortfolioState(datetime(2024, 1, 4), (), 1.0)

    assert "time of day" in str(excinfo.value)


@pytest.mark.parametrize(
    "cash_weight",
    [-0.1, 1.5, float("nan"), float("inf"), float("-inf"), True, "0.1", None],
)
def test_a_state_rejects_an_unusable_cash_weight(cash_weight: object) -> None:
    with pytest.raises(PortfolioContractError) as excinfo:
        PortfolioState(PREVIOUS_SESSION, make_positions(AAPL=0.5), cash_weight)  # type: ignore[arg-type]

    assert "cash_weight" in str(excinfo.value)


def test_a_state_can_hold_a_symbol_outside_any_universe() -> None:
    """A delisted or long-since-dropped name may still be held."""
    state = PortfolioState(PREVIOUS_SESSION, make_positions(DELISTED=1.0), 0.0)

    assert state.symbols == ("DELISTED",)
    assert state.weights_by_symbol == {"DELISTED": 1.0}


def test_a_states_weights_by_symbol_is_a_new_dictionary() -> None:
    state = PortfolioState(PREVIOUS_SESSION, make_positions(AAPL=0.6, MSFT=0.4), 0.0)

    first = state.weights_by_symbol
    first["AAPL"] = 0.0
    first["NVDA"] = 5.0

    assert state.weights_by_symbol == {"AAPL": 0.6, "MSFT": 0.4}
    assert state.weights_by_symbol is not state.weights_by_symbol


def test_a_state_keeps_the_order_of_its_positions() -> None:
    state = PortfolioState(PREVIOUS_SESSION, make_positions(TSLA=0.5, AAPL=0.5), 0.0)

    assert state.symbols == ("TSLA", "AAPL")


def test_states_are_compared_by_value() -> None:
    assert make_state() == make_state()
    assert make_state() != make_state(positions=make_positions(MSFT=1.0))
    assert make_state() != make_state(cash_weight=1.0, positions=())


# ---------------------------------------------------------------------------
# PortfolioConstraints -- the limits a construction works within
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("cash_weight", "max_single_weight"),
    [(0.0, 1.0), (1.0, 1.0), (0.0, 0.01), (0.10, 0.12), (0.5, 0.5), (0.25, 1)],
)
def test_constraints_accept_their_boundaries(cash_weight: float, max_single_weight: float) -> None:
    constraints = PortfolioConstraints(cash_weight=cash_weight, max_single_weight=max_single_weight)

    assert constraints.cash_weight == float(cash_weight)
    assert constraints.max_single_weight == float(max_single_weight)
    assert constraints.invested_weight == pytest.approx(1.0 - float(cash_weight))


def test_constraints_are_frozen() -> None:
    constraints = make_constraints(cash_weight=0.1, max_single_weight=0.2)

    with pytest.raises(FrozenInstanceError):
        constraints.cash_weight = 0.0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        constraints.max_single_weight = 1.0  # type: ignore[misc]


@pytest.mark.parametrize(
    "cash_weight", [-0.1, 1.1, float("nan"), float("inf"), float("-inf"), True, "0.1", None]
)
def test_constraints_reject_an_unusable_cash_weight(cash_weight: object) -> None:
    with pytest.raises(PortfolioContractError) as excinfo:
        PortfolioConstraints(cash_weight=cash_weight, max_single_weight=1.0)  # type: ignore[arg-type]

    assert "cash_weight" in str(excinfo.value)


@pytest.mark.parametrize(
    "max_single_weight",
    [0, 0.0, -0.5, 1.0001, 2, float("nan"), float("inf"), True, "0.12", None],
)
def test_constraints_reject_an_unusable_max_single_weight(max_single_weight: object) -> None:
    with pytest.raises(PortfolioContractError) as excinfo:
        PortfolioConstraints(cash_weight=0.0, max_single_weight=max_single_weight)  # type: ignore[arg-type]

    assert "max_single_weight" in str(excinfo.value)


def test_constraints_collect_both_problems_at_once() -> None:
    with pytest.raises(PortfolioContractError) as excinfo:
        PortfolioConstraints(cash_weight=-0.1, max_single_weight=0.0)

    problems = excinfo.value.problems
    assert len(problems) == 2
    assert problems[0].startswith("cash_weight")
    assert problems[1].startswith("max_single_weight")


def test_constraints_carry_no_long_only_switch() -> None:
    """Long-only is a property of the layer, not an option that can be turned off."""
    assert {record.name for record in fields(PortfolioConstraints)} == {
        "cash_weight",
        "max_single_weight",
    }
    assert not hasattr(PortfolioConstraints(0.0, 1.0), "long_only")


def test_constraints_do_not_decide_whether_a_selection_is_feasible() -> None:
    """10% cash and a 12% cap over two names is the constructor's question."""
    constraints = PortfolioConstraints(cash_weight=0.10, max_single_weight=0.12)

    context = make_context(constraints=constraints)

    assert context.constraints is constraints
    assert not isinstance(constraints, PortfolioConstructor)


# ---------------------------------------------------------------------------
# PortfolioConstructionSpec -- which rule built the weights
# ---------------------------------------------------------------------------
def test_a_construction_spec_names_a_rule_and_a_version() -> None:
    spec = PortfolioConstructionSpec(name="score_weighted", version="2.1.0")

    assert spec.name == "score_weighted"
    assert spec.version == "2.1.0"


@pytest.mark.parametrize(
    "name", ["equal_weight", "score_weighted", "volatility_weighted", "v1", "a1_b2"]
)
def test_a_construction_spec_accepts_a_snake_case_name(name: str) -> None:
    assert PortfolioConstructionSpec(name=name, version="1.0.0").name == name


@pytest.mark.parametrize(
    "name",
    [
        "",
        " ",
        "Equal-Weight",
        "Equal_Weight",
        " equal_weight",
        "equal weight",
        "equal-weight",
        "equal_weight\n",
        None,
        5,
    ],
)
def test_a_construction_spec_rejects_an_unusable_name(name: object) -> None:
    with pytest.raises(PortfolioContractError) as excinfo:
        PortfolioConstructionSpec(name=name, version="1.0.0")  # type: ignore[arg-type]

    assert "name" in str(excinfo.value)


@pytest.mark.parametrize("version", ["", "   ", None, 1.0, 2])
def test_a_construction_spec_rejects_an_unusable_version(version: object) -> None:
    with pytest.raises(PortfolioContractError) as excinfo:
        PortfolioConstructionSpec(name="equal_weight", version=version)  # type: ignore[arg-type]

    assert "version" in str(excinfo.value)


def test_a_construction_spec_is_frozen_and_hashable() -> None:
    spec = CONSTRUCTION

    with pytest.raises(FrozenInstanceError):
        spec.name = "score_weighted"  # type: ignore[misc]

    assert {spec, PortfolioConstructionSpec("equal_weight", "1.0.0")} == {spec}
    assert {spec: "cached"}[PortfolioConstructionSpec("equal_weight", "1.0.0")] == "cached"


def test_two_versions_of_one_rule_are_two_rules() -> None:
    assert CONSTRUCTION != PortfolioConstructionSpec("equal_weight", "2.0.0")
    assert CONSTRUCTION == PortfolioConstructionSpec("equal_weight", "1.0.0")


def test_a_construction_spec_carries_no_factor_version_and_no_parameters() -> None:
    """Factor versions come from the strategy result; numbers from the constraints."""
    assert {record.name for record in fields(PortfolioConstructionSpec)} == {"name", "version"}


def test_constructor_naming_reuses_the_strategy_pattern() -> None:
    assert CONSTRUCTION_NAME_PATTERN == STRATEGY_NAME_PATTERN


# ---------------------------------------------------------------------------
# PortfolioContext -- everything one construction may read
# ---------------------------------------------------------------------------
def test_a_context_accepts_one_complete_decision() -> None:
    result = make_result()
    state = make_state()
    constraints = make_constraints(cash_weight=0.1, max_single_weight=0.2)

    context = PortfolioContext(AS_OF, result, state, constraints)

    assert context.as_of_date == AS_OF
    assert context.strategy_result is result
    assert context.current_state is state
    assert context.constraints is constraints


def test_a_context_normalizes_an_iso_date_string() -> None:
    context = make_context(as_of_date="2024-01-05")

    assert context.as_of_date == AS_OF
    assert type(context.as_of_date) is date


def test_a_context_is_frozen() -> None:
    context = make_context()

    with pytest.raises(FrozenInstanceError):
        context.as_of_date = LATER_SESSION  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        context.current_state = make_state()  # type: ignore[misc]


def test_a_context_reads_its_provenance_from_the_result() -> None:
    result = make_result()

    context = PortfolioContext(AS_OF, result, make_state(), make_constraints())

    assert context.selected_symbols == SELECTED == ("MSFT", "NVDA")
    assert context.strategy_spec is result.strategy_spec is SPEC
    assert context.factor_requirements is result.factor_requirements
    assert context.universe is result.universe
    assert [str(requirement) for requirement in context.factor_requirements] == [
        "momentum_12_1/1.0.0"
    ]


def test_a_context_carries_the_exact_factor_version_of_its_result() -> None:
    """Same strategy name and version, another factor version: still traceable."""
    result = make_result(spec=SPEC_ON_FACTOR_V2)

    context = PortfolioContext(AS_OF, result, make_state(), make_constraints())

    assert context.strategy_spec is SPEC_ON_FACTOR_V2
    assert context.strategy_spec.name == SPEC.name
    assert [str(requirement) for requirement in context.factor_requirements] == [
        "momentum_12_1/2.0.0"
    ]


def test_a_context_accepts_a_state_from_an_earlier_session() -> None:
    """The last rebalance usually happened before the decision."""
    context = make_context(current_state=make_state(as_of_date=PREVIOUS_SESSION))

    assert context.current_state.as_of_date < context.as_of_date


def test_a_context_accepts_a_state_dated_the_same_day() -> None:
    context = make_context(current_state=make_state(as_of_date=AS_OF))

    assert context.current_state.as_of_date == context.as_of_date


def test_a_context_rejects_a_result_from_another_session() -> None:
    with pytest.raises(PortfolioContractError) as excinfo:
        make_context(strategy_result=make_result(as_of_date=LATER_SESSION))

    assert "never another day" in str(excinfo.value)


def test_a_context_rejects_a_state_from_the_future() -> None:
    with pytest.raises(PortfolioContractError) as excinfo:
        make_context(current_state=make_state(as_of_date=LATER_SESSION))

    assert "look-ahead" in str(excinfo.value)


@pytest.mark.parametrize(
    "strategy_result", [pd.DataFrame(), ["NVDA"], None, object(), pd.Series([1.0])]
)
def test_a_context_rejects_a_result_that_is_not_a_strategy_result(strategy_result: object) -> None:
    with pytest.raises(PortfolioContractError) as excinfo:
        PortfolioContext(AS_OF, strategy_result, make_state(), make_constraints())  # type: ignore[arg-type]

    assert "strategy_result" in str(excinfo.value)


@pytest.mark.parametrize("current_state", [{"AAPL": 1.0}, None, ("AAPL",), object()])
def test_a_context_rejects_a_state_that_is_not_a_portfolio_state(current_state: object) -> None:
    with pytest.raises(PortfolioContractError) as excinfo:
        PortfolioContext(AS_OF, make_result(), current_state, make_constraints())  # type: ignore[arg-type]

    assert "current_state" in str(excinfo.value)


@pytest.mark.parametrize("constraints", [(0.0, 1.0), None, {"cash_weight": 0.0}, object()])
def test_a_context_rejects_constraints_that_are_not_constraints(constraints: object) -> None:
    with pytest.raises(PortfolioContractError) as excinfo:
        PortfolioContext(AS_OF, make_result(), make_state(), constraints)  # type: ignore[arg-type]

    assert "constraints" in str(excinfo.value)


def test_a_context_collects_every_wrong_field_at_once() -> None:
    with pytest.raises(PortfolioContractError) as excinfo:
        PortfolioContext(AS_OF, "not a result", "not a state", "not constraints")  # type: ignore[arg-type]

    problems = excinfo.value.problems
    assert len(problems) == 3
    joined = "\n".join(problems)
    assert "strategy_result" in joined
    assert "current_state" in joined
    assert "constraints" in joined


def test_a_context_collects_a_date_conflict_and_a_wrong_field() -> None:
    with pytest.raises(PortfolioContractError) as excinfo:
        PortfolioContext(  # type: ignore[arg-type]
            AS_OF, make_result(as_of_date=LATER_SESSION), "not a state", make_constraints()
        )

    problems = excinfo.value.problems
    assert len(problems) == 2
    assert "current_state" in problems[0]
    assert "never another day" in problems[1]


@pytest.mark.parametrize(
    "as_of_date",
    [datetime(2024, 1, 5), pd.Timestamp("2024-01-05"), pd.NaT, "2024-01-05T00:00", None, ""],
)
def test_a_context_rejects_an_unusable_date(as_of_date: object) -> None:
    with pytest.raises(PortfolioContractError) as excinfo:
        make_context(as_of_date=as_of_date)

    assert "as_of_date" in str(excinfo.value)


def test_a_context_shares_the_objects_it_was_given() -> None:
    """Nothing is copied: the context cannot drift away from its result."""
    result = make_result()
    context = make_context(strategy_result=result)

    assert context.strategy_result is result
    assert context.current_state.symbols == ("AAPL",)


def test_a_context_is_an_input_not_a_portfolio() -> None:
    context = make_context()

    assert not isinstance(context, (PortfolioState, TargetPortfolio))
    assert not hasattr(context, "positions")


# ---------------------------------------------------------------------------
# TargetPortfolio -- what the account should hold after the decision
# ---------------------------------------------------------------------------
def test_a_target_holds_equal_weights() -> None:
    target = make_target()

    assert target.as_of_date == AS_OF
    assert target.construction_spec is CONSTRUCTION
    assert target.target_symbols == ("MSFT", "NVDA")
    assert target.target_weights_by_symbol == {"MSFT": 0.5, "NVDA": 0.5}
    assert target.invested_weight == 1.0
    assert target.cash_weight == 0.0


def test_a_target_holds_some_cash_too() -> None:
    target = make_target({"NVDA": 0.45, "MSFT": 0.45}, cash_weight=0.1)

    assert target.cash_weight == 0.1
    assert target.invested_weight == pytest.approx(0.9)


def test_a_target_accepts_the_all_cash_answer() -> None:
    target = make_target({}, cash_weight=1.0)

    assert target.positions == ()
    assert target.target_symbols == ()
    assert target.target_weights_by_symbol == {}
    assert target.invested_weight == 0.0
    assert target.cash_weight == 1.0


def test_a_target_rejects_an_empty_portfolio_with_cash_left_over() -> None:
    with pytest.raises(PortfolioContractError) as excinfo:
        make_target({}, cash_weight=0.5)

    assert "entirely in cash" in str(excinfo.value)


def test_a_target_accepts_a_subset_of_the_selection() -> None:
    """A constructor may drop a candidate the constraints leave no room for."""
    target = make_target({"NVDA": 1.0})

    assert target.target_symbols == ("NVDA",)
    assert target.source_result.selected_symbols == SELECTED


def test_a_target_rejects_a_symbol_the_strategy_did_not_select() -> None:
    with pytest.raises(PortfolioContractError) as excinfo:
        make_target({"NVDA": 0.5, "AAPL": 0.5})

    message = str(excinfo.value)
    assert "did not select" in message
    assert "AAPL" in message


def test_a_target_rejects_a_symbol_outside_the_universe() -> None:
    with pytest.raises(PortfolioContractError) as excinfo:
        make_target({"GOOGL": 1.0})

    assert "GOOGL" in str(excinfo.value)


def test_a_target_names_a_bounded_number_of_offenders() -> None:
    """Four unselected symbols: the message stays readable instead of listing all."""
    weights = {"GOOG": 0.2, "AMZN": 0.2, "TSLA": 0.2, "IBM": 0.2}

    with pytest.raises(PortfolioContractError) as excinfo:
        make_target(weights, cash_weight=0.2)

    assert "+1 more" in str(excinfo.value)


def test_a_target_rejects_a_duplicate_symbol() -> None:
    positions = (PortfolioPosition("NVDA", 0.5), PortfolioPosition("NVDA", 0.5))

    with pytest.raises(PortfolioContractError) as excinfo:
        TargetPortfolio(AS_OF, CONSTRUCTION, make_result(), make_constraints(), positions, 0.0)

    assert "more than once" in str(excinfo.value)


def test_a_target_rejects_a_total_that_is_not_one() -> None:
    with pytest.raises(PortfolioContractError) as excinfo:
        make_target({"NVDA": 0.4, "MSFT": 0.4})

    assert "add up to 1.0" in str(excinfo.value)


def test_a_target_rejects_leverage() -> None:
    with pytest.raises(PortfolioContractError) as excinfo:
        make_target({"NVDA": 0.7, "MSFT": 0.7})

    assert "add up to 1.0" in str(excinfo.value)


def test_a_target_accepts_weights_that_only_round_to_one() -> None:
    with pytest.raises(PortfolioContractError):
        make_target({"NVDA": 0.9, "MSFT": 0.1 + 1e-9})

    target = make_target({"NVDA": 0.9, "MSFT": 0.1 + 1e-13})

    assert target.invested_weight == pytest.approx(1.0)


def test_a_target_is_frozen() -> None:
    target = make_target()

    with pytest.raises(FrozenInstanceError):
        target.cash_weight = 1.0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        target.positions = ()  # type: ignore[misc]


def test_a_target_reads_its_provenance_from_the_result() -> None:
    result = make_result()

    target = make_target(source_result=result)

    assert target.source_result is result
    assert target.strategy_spec is result.strategy_spec is SPEC
    assert target.factor_requirements is result.factor_requirements
    assert target.universe is result.universe
    assert [str(requirement) for requirement in target.factor_requirements] == [
        "momentum_12_1/1.0.0"
    ]


def test_a_target_keeps_the_exact_factor_version_of_its_result() -> None:
    target = make_target(source_result=make_result(spec=SPEC_ON_FACTOR_V2))

    assert target.strategy_spec.name == SPEC.name
    assert [str(requirement) for requirement in target.factor_requirements] == [
        "momentum_12_1/2.0.0"
    ]


def test_a_target_keeps_the_order_the_constructor_gave() -> None:
    target = make_target({"NVDA": 0.6, "MSFT": 0.4})

    assert target.target_symbols == ("NVDA", "MSFT")
    assert list(target.target_weights_by_symbol) == ["NVDA", "MSFT"]


def test_a_targets_weights_by_symbol_is_a_new_dictionary() -> None:
    target = make_target()

    first = target.target_weights_by_symbol
    first["NVDA"] = 0.0
    first["AAPL"] = 9.0

    assert target.target_weights_by_symbol == {"MSFT": 0.5, "NVDA": 0.5}
    assert target.target_weights_by_symbol is not target.target_weights_by_symbol


def test_a_target_rejects_a_date_that_is_not_its_results() -> None:
    with pytest.raises(PortfolioContractError) as excinfo:
        make_target(as_of_date=LATER_SESSION)

    assert "decision date" in str(excinfo.value)


@pytest.mark.parametrize(
    "as_of_date",
    [
        datetime(2024, 1, 5),
        pd.Timestamp("2024-01-05"),
        pd.NaT,
        "2024-01-05T00:00",
        "01/05/2024",
        None,
        20240105,
    ],
)
def test_a_target_rejects_an_unusable_date(as_of_date: object) -> None:
    with pytest.raises(PortfolioContractError) as excinfo:
        make_target(as_of_date=as_of_date)

    assert "as_of_date" in str(excinfo.value)


@pytest.mark.parametrize("source_result", [pd.DataFrame(), ["NVDA"], None, object()])
def test_a_target_rejects_a_result_that_is_not_a_strategy_result(source_result: object) -> None:
    with pytest.raises(PortfolioContractError) as excinfo:
        TargetPortfolio(  # type: ignore[arg-type]
            AS_OF, CONSTRUCTION, source_result, make_constraints(), (), 1.0
        )

    assert "source_result" in str(excinfo.value)


@pytest.mark.parametrize(
    "construction_spec", [("equal_weight", "1.0.0"), None, {"name": "equal_weight"}, object()]
)
def test_a_target_rejects_a_spec_that_is_not_a_construction_spec(
    construction_spec: object,
) -> None:
    with pytest.raises(PortfolioContractError) as excinfo:
        TargetPortfolio(  # type: ignore[arg-type]
            AS_OF, construction_spec, make_result(), make_constraints(), (), 1.0
        )

    assert "construction_spec" in str(excinfo.value)


@pytest.mark.parametrize("constraints", [(0.0, 1.0), None, {"cash_weight": 0.0}, object()])
def test_a_target_rejects_constraints_that_are_not_constraints(constraints: object) -> None:
    with pytest.raises(PortfolioContractError) as excinfo:
        TargetPortfolio(  # type: ignore[arg-type]
            AS_OF, CONSTRUCTION, make_result(), constraints, (), 1.0
        )

    assert "constraints" in str(excinfo.value)


@pytest.mark.parametrize(
    "positions",
    [
        [PortfolioPosition("NVDA", 1.0)],
        {PortfolioPosition("NVDA", 1.0)},
        PortfolioPosition("NVDA", 1.0),
        {"NVDA": 1.0},
        None,
    ],
)
def test_a_target_rejects_anything_but_a_tuple_of_positions(positions: object) -> None:
    with pytest.raises(PortfolioContractError) as excinfo:
        TargetPortfolio(  # type: ignore[arg-type]
            AS_OF, CONSTRUCTION, make_result(), make_constraints(), positions, 1.0
        )

    assert "TargetPortfolio.positions" in str(excinfo.value)


def test_a_target_rejects_a_generator_of_positions() -> None:
    with pytest.raises(PortfolioContractError) as excinfo:
        TargetPortfolio(  # type: ignore[arg-type]
            AS_OF, CONSTRUCTION, make_result(), make_constraints(), (p for p in ()), 1.0
        )

    assert "generator" in str(excinfo.value)


@pytest.mark.parametrize(
    "cash_weight",
    [-0.1, 1.5, float("nan"), float("inf"), float("-inf"), True, "0.1", None],
)
def test_a_target_rejects_an_unusable_cash_weight(cash_weight: object) -> None:
    with pytest.raises(PortfolioContractError) as excinfo:
        make_target({"NVDA": 0.5, "MSFT": 0.5}, cash_weight=cash_weight)

    assert "cash_weight" in str(excinfo.value)


@pytest.mark.parametrize("weight", [0, 0.0, -0.5, float("nan"), float("inf"), float("-inf"), True])
def test_a_target_cannot_hold_a_zero_negative_or_unbounded_weight(weight: object) -> None:
    with pytest.raises(PortfolioContractError) as excinfo:
        make_target({"NVDA": weight, "MSFT": 1.0})  # type: ignore[dict-item]

    assert "weight" in str(excinfo.value)


def test_a_target_records_its_constraints_without_re_checking_them() -> None:
    """10% cash and a 12% cap are the constructor's problem, not the target's."""
    constraints = make_constraints(cash_weight=0.1, max_single_weight=0.12)

    target = make_target({"NVDA": 0.9, "MSFT": 0.1}, constraints=constraints)

    assert target.constraints is constraints
    assert target.target_weights_by_symbol["NVDA"] == 0.9


def test_a_target_collects_every_problem_at_once() -> None:
    positions = (PortfolioPosition("AAPL", 0.3),)

    with pytest.raises(PortfolioContractError) as excinfo:
        TargetPortfolio(
            LATER_SESSION, CONSTRUCTION, make_result(), make_constraints(), positions, 0.2
        )

    problems = excinfo.value.problems
    assert len(problems) == 3
    joined = "\n".join(problems)
    assert "decision date" in joined
    assert "did not select" in joined
    assert "add up to 1.0" in joined


def test_a_state_and_a_target_make_different_statements() -> None:
    """The state says what is held; the target says what should be held."""
    state = make_state()  # 100% AAPL, dated yesterday
    target = make_target()  # 50/50 MSFT/NVDA, dated today

    assert state.symbols == ("AAPL",)
    assert target.target_symbols == ("MSFT", "NVDA")
    assert state.as_of_date < target.as_of_date
    assert not isinstance(state, TargetPortfolio)
    assert not isinstance(target, PortfolioState)


def test_targets_are_compared_by_value() -> None:
    # A StrategyResult compares by identity, so the same result object is reused:
    # two targets built from two *different* selections are different targets.
    result = make_result()
    other_rule = PortfolioConstructionSpec(name="score_weighted", version="1.0.0")

    assert make_target(source_result=result) == make_target(source_result=result)
    assert make_target({"NVDA": 1.0}, source_result=result) != make_target(source_result=result)
    assert make_target(source_result=result, construction_spec=other_rule) != make_target(
        source_result=result
    )


# ---------------------------------------------------------------------------
# Errors -- the layer's own vocabulary
# ---------------------------------------------------------------------------
def test_portfolio_errors_are_their_own_hierarchy() -> None:
    """A portfolio violation is neither a strategy, a factor nor a data error."""
    assert issubclass(PortfolioContractError, PortfolioError)
    assert not issubclass(PortfolioError, StrategyError)
    assert not issubclass(PortfolioError, FactorError)
    assert not issubclass(PortfolioError, DataPipelineError)
    assert not issubclass(PortfolioError, ValueError)
    assert not issubclass(PortfolioError, KeyError)
    assert not issubclass(PortfolioError, TypeError)


def test_portfolio_contract_errors_carry_their_problems() -> None:
    error = PortfolioContractError("invalid thing", ["first problem", "second problem"])

    assert error.problems == ["first problem", "second problem"]
    assert str(error).startswith("invalid thing\n  - ")
    assert "first problem" in str(error)
    assert "second problem" in str(error)
    assert PortfolioContractError("plain").problems == []
    assert str(PortfolioContractError("plain")) == "plain"


@pytest.mark.parametrize(
    "build",
    [
        lambda: PortfolioPosition("aapl", 0.5),
        lambda: PortfolioPosition("AAPL", 0.0),
        lambda: PortfolioState(None, None, None),
        lambda: PortfolioState("2024-01-04", (), 1.5),
        lambda: PortfolioConstraints(None, None),
        lambda: PortfolioConstructionSpec(None, None),
        lambda: PortfolioContext(None, None, None, None),
        lambda: TargetPortfolio(None, None, None, None, None, None),
        lambda: make_context(strategy_result=make_result(as_of_date=LATER_SESSION)),
        lambda: make_target({"AAPL": 1.0}),
    ],
)
def test_the_layer_reports_its_own_errors_never_a_low_level_one(build: Any) -> None:
    with pytest.raises(PortfolioContractError) as excinfo:
        build()

    assert isinstance(excinfo.value, PortfolioError)
    assert excinfo.value.problems


# ---------------------------------------------------------------------------
# Package surface and architectural boundary
# ---------------------------------------------------------------------------
def test_the_contract_is_exported_from_the_package_root() -> None:
    expected = {
        "PortfolioConstraints",
        "PortfolioConstructionSpec",
        "PortfolioConstructor",
        "PortfolioContext",
        "PortfolioContractError",
        "PortfolioError",
        "PortfolioPosition",
        "PortfolioState",
        "TargetPortfolio",
    }

    assert expected <= set(portfolio_sys.__all__)
    assert portfolio_sys.PortfolioPosition is PortfolioPosition
    assert portfolio_sys.PortfolioState is PortfolioState
    assert portfolio_sys.PortfolioConstraints is PortfolioConstraints
    assert portfolio_sys.PortfolioConstructionSpec is PortfolioConstructionSpec
    assert portfolio_sys.PortfolioContext is PortfolioContext
    assert portfolio_sys.TargetPortfolio is TargetPortfolio
    assert portfolio_sys.PortfolioConstructor is PortfolioConstructor
    assert portfolio_sys.PortfolioError is PortfolioError
    assert portfolio_sys.PortfolioContractError is PortfolioContractError
    assert portfolio_sys.CONSTRUCTION_NAME_PATTERN == CONSTRUCTION_NAME_PATTERN


def test_the_package_ships_no_rule_no_registry_and_no_runner() -> None:
    for absent in (
        "EqualWeightConstructor",
        "ScoreWeightedConstructor",
        "VolatilityWeightedConstructor",
        "OptimizerConstructor",
        "PortfolioRegistry",
        "PortfolioRunner",
        "run_portfolio",
        "load_targets",
        "save_targets",
    ):
        assert not hasattr(portfolio_sys, absent), absent

    imported = {name for name in sys.modules if name.startswith("portfolio_sys.")}
    # A weighting rule now exists one level down, so it may appear here -- but only
    # because a *caller* imported it by name, never because the contract layer
    # pulled it in: the root's own imports are pinned on the next line.
    assert imported <= {
        "portfolio_sys.constructors",
        "portfolio_sys.constructors.capped_score_weighted",
        "portfolio_sys.contracts",
        "portfolio_sys.errors",
    }, imported
    assert imported_modules(Path(portfolio_sys.__file__)) == {
        "__future__",
        "portfolio_sys.contracts",
        "portfolio_sys.errors",
    }


def test_the_package_is_made_of_the_three_contract_modules() -> None:
    package = Path(portfolio_sys.__file__).parent

    assert sorted(path.name for path in package.glob("*.py")) == [
        "__init__.py",
        "contracts.py",
        "errors.py",
    ]


#: Modules the contract may import: the standard library it needs, the project's
#: single ticker pattern, the factor layer's universe type, the strategy result it
#: weights, and its own two modules -- nothing else.
ALLOWED_IMPORTS: frozenset[str] = frozenset(
    {
        "__future__",
        "abc",
        "collections.abc",
        "dataclasses",
        "datetime",
        "math",
        "numbers",
        "re",
        "typing",
        "data_sys.schema",
        "factor_sys.universe",
        "portfolio_sys.contracts",
        "portfolio_sys.errors",
        "strategy_sys.contracts",
    }
)

#: Text that must never appear in the contract modules' *code*.  Docstrings are
#: blanked first, so prose may still name what the layer deliberately avoids.
FORBIDDEN_TEXT: tuple[str, ...] = (
    "import pandas",
    "import pyarrow",
    "import duckdb",
    "import yfinance",
    "import requests",
    "import httpx",
    "import socket",
    "import exchange_calendars",
    "data_sys.storage",
    "data_sys.providers",
    "data_sys.duckdb_verify",
    "factor_sys.factor_store",
    "factor_sys.market_data",
    "factor_sys.runner",
    "strategy_sys.registry",
    "strategy_sys.runner",
    "StrategyRunner",
    "TopNByFactorStrategy",
    "ParquetFactorStore",
    "read_parquet",
    "to_parquet",
    "urlopen",
    "csv",
    "open(",
)

#: Identifiers the contract must never use: an order, a fill, a price, a share
#: count or a backtest is another layer's vocabulary.
FORBIDDEN_IDENTIFIERS: frozenset[str] = frozenset(
    {
        "quantity",
        "shares",
        "share_count",
        "price",
        "fill",
        "fill_price",
        "order",
        "orders",
        "commission",
        "slippage",
        "turnover",
        "nav",
        "backtest",
        "rebalance",
        "pnl",
        "execution_date",
        "valid_date",
    }
)


def contract_modules() -> list[Path]:
    """The three modules that make up the portfolio contract layer."""
    package = Path(portfolio_sys.__file__).parent
    return [package / "__init__.py", package / "contracts.py", package / "errors.py"]


def imported_modules(path: Path) -> set[str]:
    """Every module named by an import statement in ``path``."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def contract_code(path: Path) -> str:
    """``path``'s source with every docstring blanked out."""
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines(keepends=True)
    documented = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    for node in ast.walk(ast.parse(source, filename=str(path))):
        if isinstance(node, documented) and ast.get_docstring(node, clean=False) is not None:
            docstring = node.body[0]
            for index in range(docstring.lineno - 1, docstring.end_lineno):
                lines[index] = ""
    return "".join(lines)


def test_the_contract_imports_only_its_own_layers() -> None:
    """No store, no storage, no network, no calendar: the contracts and a pattern."""
    for module in contract_modules():
        imported = imported_modules(module)
        assert imported <= ALLOWED_IMPORTS, (module.name, sorted(imported - ALLOWED_IMPORTS))


def test_the_contract_never_names_a_forbidden_dependency() -> None:
    for module in contract_modules():
        code = contract_code(module)
        offenders = [text for text in FORBIDDEN_TEXT if text in code]
        assert offenders == [], (module.name, offenders)


def test_the_contract_never_opens_prints_or_evaluates_anything() -> None:
    """The layer calls no builtin that reads, writes, prints or compiles."""
    for module in contract_modules():
        tree = ast.parse(module.read_text(encoding="utf-8"))
        called = {ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}
        assert called & {"open", "print", "input", "eval", "exec", "compile"} == set(), module.name


def test_the_contract_contains_no_weighting_arithmetic() -> None:
    """Sharing a budget out -- a product or a quotient -- is a rule, and a rule lives
    in a constructor, never in the objects a constructor agrees on."""
    forbidden = (ast.Mult, ast.Div, ast.FloorDiv, ast.Pow, ast.Mod)

    for module in contract_modules():
        tree = ast.parse(module.read_text(encoding="utf-8"))
        operators = [type(node.op) for node in ast.walk(tree) if isinstance(node, ast.BinOp)]
        assert [operator for operator in operators if operator in forbidden] == [], module.name


def test_the_contract_never_names_an_order_a_price_or_a_backtest() -> None:
    for module in contract_modules():
        tree = ast.parse(module.read_text(encoding="utf-8"))
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        names |= {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        names |= {node.arg for node in ast.walk(tree) if isinstance(node, ast.arg)}
        names |= {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        }
        offenders = sorted(names & FORBIDDEN_IDENTIFIERS)
        assert offenders == [], (module.name, offenders)


def test_the_contract_stays_deliberately_small() -> None:
    """Exactly the fields the task names: no weight, order or cost snuck in."""
    assert {record.name for record in fields(PortfolioPosition)} == {"symbol", "weight"}
    assert {record.name for record in fields(PortfolioState)} == {
        "as_of_date",
        "positions",
        "cash_weight",
    }
    assert {record.name for record in fields(PortfolioConstraints)} == {
        "cash_weight",
        "max_single_weight",
    }
    assert {record.name for record in fields(PortfolioConstructionSpec)} == {"name", "version"}
    assert {record.name for record in fields(PortfolioContext)} == {
        "as_of_date",
        "strategy_result",
        "current_state",
        "constraints",
    }
    assert {record.name for record in fields(TargetPortfolio)} == {
        "as_of_date",
        "construction_spec",
        "source_result",
        "constraints",
        "positions",
        "cash_weight",
    }


def test_the_derived_views_are_properties_not_extra_fields() -> None:
    """A symbol list or a weight map is derived on demand, never stored twice."""
    state_views = ("invested_weight", "symbols", "weights_by_symbol")
    context_views = ("selected_symbols", "strategy_spec", "factor_requirements", "universe")
    target_views = (
        "strategy_spec",
        "factor_requirements",
        "universe",
        "target_symbols",
        "target_weights_by_symbol",
        "invested_weight",
    )

    for contract, views in (
        (PortfolioState, state_views),
        (PortfolioContext, context_views),
        (TargetPortfolio, target_views),
    ):
        for view in views:
            assert isinstance(getattr(contract, view), property), (contract.__name__, view)
        stored = {record.name for record in fields(contract)}
        assert not (set(views) & stored), contract.__name__


def test_the_contract_keeps_the_projects_patterns_and_tolerance() -> None:
    assert contracts_module.WEIGHT_TOLERANCE == 1e-12
    assert portfolio_sys.WEIGHT_TOLERANCE is contracts_module.WEIGHT_TOLERANCE
    assert contracts_module.CONSTRUCTION_NAME_PATTERN == STRATEGY_NAME_PATTERN
    assert portfolio_sys.CONSTRUCTION_NAME_PATTERN is contracts_module.CONSTRUCTION_NAME_PATTERN
