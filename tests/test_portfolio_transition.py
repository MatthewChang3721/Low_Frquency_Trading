"""Tests for the transition layer (:mod:`portfolio_sys.transition`).

Everything here is in memory and offline: the state is a tuple of positions, the target is
built from a ``UniverseSnapshot`` plus a small frame, and the transition is the difference
between the two.  No ``data/`` directory, factor store, Parquet file, DuckDB connection,
trading calendar, network call, registry, constructor or strategy runner is involved -- by
construction, because the module under test may not reach any of them.

Covered in this order:

* the basic transitions: an exit, an entry, an increase, a decrease, an unchanged holding,
* the worked example from the design note,
* all-cash in, all-cash out, and a pure stock swap,
* a mixed rebalance and the convenience views it produces,
* floating-point behaviour around ``WEIGHT_TOLERANCE``,
* ``PortfolioWeightChange``: its symbol rule, its weights, its delta and its properties,
* ``PortfolioTransition``: the object it accepts and every difference it refuses,
* provenance: strategy spec, factor versions, construction spec, constraints and universe,
* purity: inputs untouched, numbers stable, results frozen,
* the package surface and the architectural boundary.
"""

from __future__ import annotations

import ast
import inspect
import math
import sys
from dataclasses import FrozenInstanceError
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest

import portfolio_sys
import portfolio_sys.transition as transition_module
from factor_sys import UniverseSnapshot
from portfolio_sys import (
    WEIGHT_TOLERANCE,
    PortfolioConstraints,
    PortfolioConstructionSpec,
    PortfolioContractError,
    PortfolioError,
    PortfolioPosition,
    PortfolioState,
    PortfolioTransition,
    PortfolioWeightChange,
    TargetPortfolio,
    build_portfolio_transition,
)
from strategy_sys import FactorRequirement, StrategyResult, StrategySpec

#: The in-memory universe every fixture below works with.
SYMBOLS: tuple[str, ...] = ("AAPL", "MSFT", "NVDA", "TSLA")

#: The decision date of every fixture; the state a rebalance starts from is dated one
#: session earlier, and ``LATER_SESSION`` must never be accepted as a state for ``AS_OF``.
AS_OF: date = date(2025, 1, 9)
PREVIOUS_SESSION: date = date(2025, 1, 2)
LATER_SESSION: date = date(2025, 1, 16)

#: Two exact factor identities: the ones the fixture strategy spec reads.
MOMENTUM = FactorRequirement(factor_name="momentum_12_1", factor_version="1.0.0")
VALUE = FactorRequirement(factor_name="book_to_price", factor_version="2.0.0")

#: The strategy identity the fixture result is produced under.
SPEC = StrategySpec(name="top_n_momentum", version="1.0.0", required_factors=(MOMENTUM,))

#: The construction identity the fixture target names.
CONSTRUCTION = PortfolioConstructionSpec(name="equal_weight", version="1.0.0")

#: The limits the fixture target was built within.
CONSTRAINTS = PortfolioConstraints(cash_weight=0.1, max_single_weight=0.12)


# ---------------------------------------------------------------------------
# In-memory fixtures (no disk, no network, no store, no constructor)
# ---------------------------------------------------------------------------
def make_result(
    *,
    symbols: tuple[str, ...] = SYMBOLS,
    selected: tuple[str, ...] = ("MSFT", "NVDA"),
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
        universe=UniverseSnapshot(  # type: ignore[arg-type]
            as_of_date=as_of_date,
            symbols=tuple(symbols),
            source="config/universe_seed.csv",
        ),
        strategy_spec=spec,
        as_of_date=as_of_date,  # type: ignore[arg-type]
    )


def make_state(
    weights: dict[str, float] | None = None,
    *,
    cash_weight: object = 0.0,
    as_of_date: object = PREVIOUS_SESSION,
) -> PortfolioState:
    """The account before the decision: by default entirely in ``AAPL``."""
    holdings = {"AAPL": 1.0} if weights is None else weights
    return PortfolioState(  # type: ignore[arg-type]
        as_of_date=as_of_date,
        positions=tuple(PortfolioPosition(symbol, weight) for symbol, weight in holdings.items()),
        cash_weight=cash_weight,
    )


def make_target(
    weights: dict[str, float] | None = None,
    *,
    cash_weight: object = 0.0,
    as_of_date: object = AS_OF,
    construction_spec: object = CONSTRUCTION,
    constraints: object = CONSTRAINTS,
    source_result: object = None,
) -> TargetPortfolio:
    """The weights a rule wants: by default half ``MSFT`` and half ``NVDA``.

    The source strategy result is derived from the weights, so the target holds exactly the
    symbols its own selection selected -- the contract requires no less.
    """
    holdings = {"MSFT": 0.5, "NVDA": 0.5} if weights is None else weights
    selected = tuple(holdings)
    symbols = tuple(dict.fromkeys(SYMBOLS + selected))
    return TargetPortfolio(  # type: ignore[arg-type]
        as_of_date=as_of_date,
        construction_spec=construction_spec,
        source_result=(
            make_result(symbols=symbols, selected=selected, as_of_date=as_of_date)
            if source_result is None
            else source_result
        ),
        constraints=constraints,
        positions=tuple(PortfolioPosition(symbol, weight) for symbol, weight in holdings.items()),
        cash_weight=cash_weight,
    )


def transition_of(state: PortfolioState, target: TargetPortfolio) -> PortfolioTransition:
    """The transition between two fixtures."""
    return build_portfolio_transition(current_state=state, target_portfolio=target)


def table(transition: PortfolioTransition) -> list[tuple[str, float, float, float]]:
    """The rows of ``transition`` as ``(symbol, current, target, change)`` quadruples."""
    return [
        (change.symbol, change.current_weight, change.target_weight, change.weight_change)
        for change in transition.changes
    ]


def change(
    symbol: str = "MSFT",
    current: object = 0.3,
    target: object = 0.45,
    delta: object = None,
) -> PortfolioWeightChange:
    """A weight change whose delta is ``target - current`` unless one is given."""
    difference = (float(target) - float(current)) if delta is None else delta
    return PortfolioWeightChange(  # type: ignore[arg-type]
        symbol=symbol,
        current_weight=current,  # type: ignore[arg-type]
        target_weight=target,  # type: ignore[arg-type]
        weight_change=difference,  # type: ignore[arg-type]
    )


#: The worked example: ``AAPL 60% + MSFT 30%`` held beside ``10%`` cash ...
EXAMPLE_STATE: dict[str, float] = {"AAPL": 0.6, "MSFT": 0.3}
#: ... and ``MSFT 45% + NVDA 45%`` wanted beside the same ``10%`` cash.
EXAMPLE_TARGET: dict[str, float] = {"MSFT": 0.45, "NVDA": 0.45}


# ---------------------------------------------------------------------------
# The basic transitions: exit, entry, increase, decrease, unchanged
# ---------------------------------------------------------------------------
def test_a_holding_the_target_drops_exits_with_a_zero_target_weight() -> None:
    """A current-only holding is reported, not silently forgotten."""
    state = make_state({"AAPL": 0.6, "MSFT": 0.3}, cash_weight=0.1)
    target = make_target({"MSFT": 0.3}, cash_weight=0.7)

    transition = transition_of(state, target)

    assert table(transition) == [
        ("MSFT", pytest.approx(0.3), pytest.approx(0.3), pytest.approx(0.0)),
        ("AAPL", pytest.approx(0.6), 0.0, pytest.approx(-0.6)),
    ]
    assert transition.exiting_symbols == ("AAPL",)
    assert transition.entering_symbols == ()


def test_a_symbol_the_target_adds_enters_with_a_zero_current_weight() -> None:
    """A target-only holding is reported with a current weight of exactly zero."""
    state = make_state({"AAPL": 1.0})
    target = make_target({"NVDA": 0.8}, cash_weight=0.2)

    transition = transition_of(state, target)

    assert table(transition) == [
        ("NVDA", 0.0, pytest.approx(0.8), pytest.approx(0.8)),
        ("AAPL", 1.0, 0.0, -1.0),
    ]
    assert transition.entering_symbols == ("NVDA",)
    assert transition.exiting_symbols == ("AAPL",)


def test_a_shared_holding_that_grows_reports_a_positive_change() -> None:
    """An increase compares the two positive weights."""
    state = make_state({"MSFT": 0.2}, cash_weight=0.8)
    target = make_target({"MSFT": 0.5}, cash_weight=0.5)

    (row,) = transition_of(state, target).changes

    assert (row.symbol, row.current_weight, row.target_weight) == ("MSFT", 0.2, 0.5)
    assert row.weight_change == pytest.approx(0.3)
    assert row.is_buy and not row.is_sell and not row.is_unchanged


def test_a_shared_holding_that_shrinks_reports_a_negative_change() -> None:
    """A decrease is the same comparison with the other sign."""
    state = make_state({"MSFT": 0.5}, cash_weight=0.5)
    target = make_target({"MSFT": 0.2}, cash_weight=0.8)

    (row,) = transition_of(state, target).changes

    assert row.weight_change == pytest.approx(-0.3)
    assert row.is_sell and not row.is_buy
    assert not row.is_entry and not row.is_exit


def test_a_shared_holding_that_keeps_its_weight_reports_no_change() -> None:
    """The same weight on both sides is an unchanged row, not a missing one."""
    state = make_state({"MSFT": 0.5}, cash_weight=0.5)
    target = make_target({"MSFT": 0.5}, cash_weight=0.5)

    transition = transition_of(state, target)

    assert table(transition) == [("MSFT", 0.5, 0.5, 0.0)]
    assert transition.changed_symbols == ()
    assert transition.turnover == 0.0


def test_the_rows_cover_the_complete_union_of_both_portfolios() -> None:
    """Every symbol of either side appears exactly once, and nothing else does."""
    state = make_state({"AAPL": 0.4, "MSFT": 0.4}, cash_weight=0.2)
    target = make_target({"MSFT": 0.4, "TSLA": 0.4}, cash_weight=0.2)

    transition = transition_of(state, target)

    assert transition.symbols == ("MSFT", "TSLA", "AAPL")
    assert set(transition.symbols) == {"AAPL", "MSFT", "TSLA"}
    assert len(transition.symbols) == len(set(transition.symbols))


def test_the_target_holdings_come_first_in_the_targets_own_order() -> None:
    """The rows start with the target's symbols, in the order the rule wrote them."""
    state = make_state({"AAPL": 1.0})
    target = make_target({"TSLA": 0.3, "NVDA": 0.7})

    transition = transition_of(state, target)

    assert transition.symbols[:2] == ("TSLA", "NVDA")
    assert transition.symbols[:2] == target.target_symbols


def test_the_exits_follow_in_the_current_states_own_order() -> None:
    """Current-only symbols are appended last, in the state's order -- never sorted."""
    state = make_state({"TSLA": 0.4, "AAPL": 0.4}, cash_weight=0.2)
    target = make_target({"NVDA": 0.9}, cash_weight=0.1)

    transition = transition_of(state, target)

    assert transition.symbols == ("NVDA", "TSLA", "AAPL")
    assert transition.symbols[1:] == state.symbols


def test_the_order_is_stable_across_identical_builds() -> None:
    """Two builds of the same pair produce one sequence, so reports stay comparable."""
    state = make_state({"TSLA": 0.4, "AAPL": 0.4}, cash_weight=0.2)
    target = make_target({"NVDA": 0.9}, cash_weight=0.1)

    first = transition_of(state, target)
    second = transition_of(state, target)

    assert first.symbols == second.symbols
    assert table(first) == table(second)


def test_the_transition_is_dated_the_targets_decision_date() -> None:
    """The difference belongs to the decision, so it carries the target's date."""
    transition = transition_of(make_state(), make_target())

    assert transition.as_of_date == AS_OF
    assert transition.as_of_date == transition.target_portfolio.as_of_date
    assert transition.current_state.as_of_date == PREVIOUS_SESSION


# ---------------------------------------------------------------------------
# The worked example from the design note
# ---------------------------------------------------------------------------
def example_transition() -> PortfolioTransition:
    """``AAPL 60% + MSFT 30% + cash 10%`` -> ``MSFT 45% + NVDA 45% + cash 10%``."""
    return transition_of(
        make_state(EXAMPLE_STATE, cash_weight=0.1),
        make_target(EXAMPLE_TARGET, cash_weight=0.1),
    )


def test_the_worked_example_reports_the_three_rows_in_order() -> None:
    """``MSFT`` and ``NVDA`` first, the closing ``AAPL`` last."""
    transition = example_transition()

    assert table(transition) == [
        ("MSFT", pytest.approx(0.30), pytest.approx(0.45), pytest.approx(0.15)),
        ("NVDA", 0.0, pytest.approx(0.45), pytest.approx(0.45)),
        ("AAPL", pytest.approx(0.60), 0.0, pytest.approx(-0.60)),
    ]


def test_the_worked_example_reports_equal_one_sided_volumes() -> None:
    """60% bought and 60% sold: the same weight, in opposite directions."""
    transition = example_transition()

    assert transition.buy_weight == pytest.approx(0.6)
    assert transition.sell_weight == pytest.approx(0.6)


def test_the_worked_example_leaves_the_cash_share_alone() -> None:
    """Both sides keep 10% cash, so the cash difference is zero."""
    transition = example_transition()

    assert transition.cash_weight_change == pytest.approx(0.0)
    assert transition.target_portfolio.cash_weight == 0.1
    assert transition.current_state.cash_weight == 0.1


def test_the_worked_example_turns_over_sixty_percent() -> None:
    """One side of a 60%-in / 60%-out rebalance, never the 120% sum of both."""
    transition = example_transition()

    assert transition.turnover == pytest.approx(0.6)
    assert transition.turnover == pytest.approx(max(transition.buy_weight, transition.sell_weight))
    assert transition.turnover != pytest.approx(transition.buy_weight + transition.sell_weight)


def test_the_worked_example_names_the_entries_the_exits_and_the_moves() -> None:
    """The three convenience views read the rows in row order."""
    transition = example_transition()

    assert transition.entering_symbols == ("NVDA",)
    assert transition.exiting_symbols == ("AAPL",)
    assert transition.changed_symbols == ("MSFT", "NVDA", "AAPL")


# ---------------------------------------------------------------------------
# All-cash in, all-cash out, and a pure swap
# ---------------------------------------------------------------------------
def test_from_all_cash_every_target_symbol_enters() -> None:
    """An all-cash account enters every target holding with a zero current weight."""
    state = make_state({}, cash_weight=1.0)
    target = make_target({"MSFT": 0.45, "NVDA": 0.45}, cash_weight=0.1)

    transition = transition_of(state, target)

    assert table(transition) == [
        ("MSFT", 0.0, pytest.approx(0.45), pytest.approx(0.45)),
        ("NVDA", 0.0, pytest.approx(0.45), pytest.approx(0.45)),
    ]
    assert transition.entering_symbols == ("MSFT", "NVDA")
    assert transition.exiting_symbols == ()


def test_from_all_cash_the_turnover_is_the_invested_weight() -> None:
    """Nothing is sold, so the rebalance is exactly what was put into stock."""
    state = make_state({}, cash_weight=1.0)
    target = make_target({"MSFT": 0.45, "NVDA": 0.45}, cash_weight=0.1)

    transition = transition_of(state, target)

    assert transition.buy_weight == pytest.approx(0.9)
    assert transition.sell_weight == 0.0
    assert transition.turnover == pytest.approx(0.9)
    assert transition.turnover == pytest.approx(target.invested_weight)


def test_from_all_cash_the_cash_share_falls_by_the_invested_weight() -> None:
    """The cash that was spent is reported as a cash change, never as a row."""
    transition = transition_of(
        make_state({}, cash_weight=1.0),
        make_target({"MSFT": 0.45, "NVDA": 0.45}, cash_weight=0.1),
    )

    assert transition.cash_weight_change == pytest.approx(-0.9)
    assert "CASH" not in transition.symbols


def test_into_all_cash_every_holding_exits() -> None:
    """A target that holds nothing closes every current holding at zero."""
    state = make_state({"AAPL": 0.6, "MSFT": 0.3}, cash_weight=0.1)
    target = make_target({}, cash_weight=1.0)

    transition = transition_of(state, target)

    assert table(transition) == [
        ("AAPL", pytest.approx(0.6), 0.0, pytest.approx(-0.6)),
        ("MSFT", pytest.approx(0.3), 0.0, pytest.approx(-0.3)),
    ]
    assert transition.exiting_symbols == ("AAPL", "MSFT")
    assert transition.entering_symbols == ()


def test_into_all_cash_the_turnover_is_the_liquidated_weight() -> None:
    """Nothing is bought, so the rebalance is exactly what left the market."""
    transition = transition_of(
        make_state({"AAPL": 0.6, "MSFT": 0.3}, cash_weight=0.1),
        make_target({}, cash_weight=1.0),
    )

    assert transition.buy_weight == 0.0
    assert transition.sell_weight == pytest.approx(0.9)
    assert transition.turnover == pytest.approx(0.9)
    assert transition.cash_weight_change == pytest.approx(0.9)


def test_a_pure_swap_reports_one_exit_and_one_entry() -> None:
    """Selling ``AAPL`` to buy ``NVDA`` is two rows and no cash movement."""
    transition = transition_of(make_state({"AAPL": 1.0}), make_target({"NVDA": 1.0}))

    assert table(transition) == [
        ("NVDA", 0.0, 1.0, 1.0),
        ("AAPL", 1.0, 0.0, -1.0),
    ]
    assert transition.cash_weight_change == 0.0


def test_a_pure_swap_turns_over_one_side_only() -> None:
    """100% in and 100% out is a 100% rebalance, not 200%."""
    transition = transition_of(make_state({"AAPL": 1.0}), make_target({"NVDA": 1.0}))

    assert transition.buy_weight == 1.0
    assert transition.sell_weight == 1.0
    assert transition.turnover == pytest.approx(1.0)
    assert transition.turnover != pytest.approx(transition.buy_weight + transition.sell_weight), (
        "the two sides of one swap are the same money, counted once"
    )


def test_a_partial_swap_measures_the_smaller_side_once() -> None:
    """A 30%-out / 10%-in / 20%-into-cash move turns over 30%, the sold side."""
    state = make_state({"AAPL": 0.3, "MSFT": 0.7})
    target = make_target({"MSFT": 0.7, "NVDA": 0.1}, cash_weight=0.2)

    transition = transition_of(state, target)

    assert transition.buy_weight == pytest.approx(0.1)
    assert transition.sell_weight == pytest.approx(0.3)
    assert transition.cash_weight_change == pytest.approx(0.2)
    assert transition.turnover == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# A mixed rebalance: entry, exit, increase, decrease and unchanged together
# ---------------------------------------------------------------------------
def mixed_transition() -> PortfolioTransition:
    """One rebalance that does all five things at once."""
    state = make_state({"AAPL": 0.2, "MSFT": 0.4, "TSLA": 0.1}, cash_weight=0.3)
    target = make_target({"MSFT": 0.4, "NVDA": 0.2}, cash_weight=0.4)
    return transition_of(state, target)


def test_a_mixed_rebalance_classifies_every_row() -> None:
    """Unchanged, entered, exited -- and the two exits keep the state's order."""
    transition = mixed_transition()

    assert table(transition) == [
        ("MSFT", pytest.approx(0.4), pytest.approx(0.4), pytest.approx(0.0)),
        ("NVDA", 0.0, pytest.approx(0.2), pytest.approx(0.2)),
        ("AAPL", pytest.approx(0.2), 0.0, pytest.approx(-0.2)),
        ("TSLA", pytest.approx(0.1), 0.0, pytest.approx(-0.1)),
    ]
    assert [row.symbol for row in transition.changes if row.is_unchanged] == ["MSFT"]
    assert transition.entering_symbols == ("NVDA",)
    assert transition.exiting_symbols == ("AAPL", "TSLA")


def test_changed_symbols_leaves_the_unchanged_row_out() -> None:
    """A holding that keeps its weight is not part of the rebalance."""
    transition = mixed_transition()

    assert transition.changed_symbols == ("NVDA", "AAPL", "TSLA")
    assert "MSFT" not in transition.changed_symbols


def test_a_mixed_rebalance_sums_both_sides_and_the_cash_move() -> None:
    """20% into ``NVDA``, 30% out of two names, 10% back into cash."""
    transition = mixed_transition()

    assert transition.buy_weight == pytest.approx(0.2)
    assert transition.sell_weight == pytest.approx(0.3)
    assert transition.cash_weight_change == pytest.approx(0.1)
    assert transition.turnover == pytest.approx(0.3)


def test_a_mixed_rebalance_turns_over_the_larger_side() -> None:
    """The identity ``turnover == max(buy, sell)`` holds for a mixed move too."""
    transition = mixed_transition()

    assert transition.turnover == pytest.approx(max(transition.buy_weight, transition.sell_weight))
    assert transition.turnover == pytest.approx(
        0.5
        * (
            sum(abs(row.weight_change) for row in transition.changes)
            + abs(transition.cash_weight_change)
        )
    )


def test_the_convenience_views_read_the_rows_in_order() -> None:
    """Every view is the row sequence filtered, never re-sorted."""
    transition = mixed_transition()

    assert transition.symbols == tuple(row.symbol for row in transition.changes)
    assert transition.changed_symbols == tuple(
        row.symbol for row in transition.changes if not row.is_unchanged
    )
    assert transition.entering_symbols == tuple(
        row.symbol for row in transition.changes if row.is_entry
    )
    assert transition.exiting_symbols == tuple(
        row.symbol for row in transition.changes if row.is_exit
    )


# ---------------------------------------------------------------------------
# Floating point: the tolerance, and nothing beyond it
# ---------------------------------------------------------------------------
def test_a_change_exactly_at_the_tolerance_is_unchanged() -> None:
    """``abs(change) <= WEIGHT_TOLERANCE`` is unchanged, and not a buy."""
    row = change("AAPL", current=0.0, target=WEIGHT_TOLERANCE, delta=WEIGHT_TOLERANCE)

    assert row.is_unchanged
    assert not row.is_buy
    assert not row.is_sell


def test_a_change_just_past_the_tolerance_is_a_buy() -> None:
    """The tolerance is a boundary, not a rounding fudge that swallows real moves."""
    row = change("AAPL", current=0.0, target=2 * WEIGHT_TOLERANCE, delta=2 * WEIGHT_TOLERANCE)

    assert row.is_buy
    assert not row.is_unchanged


def test_a_change_just_past_the_tolerance_the_other_way_is_a_sell() -> None:
    """The same boundary on the negative side."""
    row = change("AAPL", current=2 * WEIGHT_TOLERANCE, target=0.0, delta=-2 * WEIGHT_TOLERANCE)

    assert row.is_sell
    assert not row.is_unchanged


def test_a_tenth_plus_a_fifth_minus_three_tenths_is_not_a_change() -> None:
    """``0.1 + 0.2 != 0.3`` in binary floating point, and that must not matter."""
    row = PortfolioWeightChange(
        symbol="AAPL",
        current_weight=0.1 + 0.2,
        target_weight=0.3,
        weight_change=0.3 - (0.1 + 0.2),
    )

    assert row.is_unchanged
    assert row.weight_change != 0.0


def test_dust_in_a_transition_is_not_a_changed_symbol() -> None:
    """A hair between the two sides is reported, but never as a rebalance."""
    state = make_state({"MSFT": 0.5}, cash_weight=0.5)
    target = make_target({"MSFT": 0.5 + 1e-13}, cash_weight=0.5 - 1e-13)

    transition = transition_of(state, target)

    assert transition.changed_symbols == ()
    assert transition.entering_symbols == ()
    assert transition.exiting_symbols == ()
    assert transition.turnover < WEIGHT_TOLERANCE


def test_a_real_change_is_still_reported_beside_dust() -> None:
    """A tolerance that hides a 20% move because a 1e-13 hair exists would be a bug."""
    state = make_state({"MSFT": 0.5, "AAPL": 0.5})
    target = make_target({"MSFT": 0.5 + 1e-13, "NVDA": 0.2, "TSLA": 0.3 - 1e-13})

    transition = transition_of(state, target)

    assert transition.changed_symbols == ("NVDA", "TSLA", "AAPL")
    assert transition.entering_symbols == ("NVDA", "TSLA")


def test_turnover_equals_the_larger_side_with_floating_point_weights() -> None:
    """The cash term is what makes the one-sided identity hold to the last bit."""
    transition = example_transition()

    assert transition.turnover == pytest.approx(
        max(transition.buy_weight, transition.sell_weight), abs=WEIGHT_TOLERANCE
    )


def test_no_number_a_transition_reports_is_nan_or_infinite() -> None:
    """Every weight, every delta and every sum is a finite number."""
    transition = mixed_transition()
    numbers = [transition.buy_weight, transition.sell_weight, transition.turnover]
    numbers.append(transition.cash_weight_change)
    for row in transition.changes:
        numbers.extend((row.current_weight, row.target_weight, row.weight_change))

    assert [number for number in numbers if not math.isfinite(number)] == []


# ---------------------------------------------------------------------------
# PortfolioWeightChange: one row, and everything it refuses
# ---------------------------------------------------------------------------
def test_a_change_keeps_the_numbers_it_was_given() -> None:
    """The four fields are stored exactly as given, and the delta is checked, not rebuilt."""
    row = PortfolioWeightChange("MSFT", 0.3, 0.45, 0.15)

    assert (row.symbol, row.current_weight, row.target_weight, row.weight_change) == (
        "MSFT",
        0.3,
        0.45,
        0.15,
    )


def test_an_integer_weight_is_stored_as_a_float() -> None:
    """``1`` and ``1.0`` are the same share of the account."""
    row = PortfolioWeightChange("AAPL", 0, 1, 1)

    assert isinstance(row.current_weight, float)
    assert isinstance(row.target_weight, float)
    assert isinstance(row.weight_change, float)
    assert (row.current_weight, row.target_weight) == (0.0, 1.0)


def test_a_lower_case_symbol_is_refused() -> None:
    """A repaired ticker would silently become a different security."""
    with pytest.raises(PortfolioContractError) as excinfo:
        PortfolioWeightChange("aapl", 0.0, 0.1, 0.1)

    assert "symbol 'aapl'" in str(excinfo.value)


def test_a_symbol_with_whitespace_is_refused() -> None:
    """The ticker rule is the project's own, and it does not tolerate padding."""
    with pytest.raises(PortfolioContractError) as excinfo:
        PortfolioWeightChange(" AAPL", 0.0, 0.1, 0.1)

    assert "upper-case ticker" in str(excinfo.value)


@pytest.mark.parametrize("symbol", [None, 123, b"AAPL", ["AAPL"]])
def test_a_symbol_that_is_not_a_string_is_refused(symbol: object) -> None:
    """A ticker is text, and nothing else is converted into one."""
    with pytest.raises(PortfolioContractError) as excinfo:
        PortfolioWeightChange(symbol, 0.0, 0.1, 0.1)  # type: ignore[arg-type]

    assert "upper-case ticker" in str(excinfo.value)


def test_cash_is_refused_as_a_symbol() -> None:
    """Cash is a field of the portfolios, never a row of the difference."""
    with pytest.raises(PortfolioContractError) as excinfo:
        PortfolioWeightChange("CASH", 0.1, 0.0, -0.1)

    assert "cash_weight_change" in str(excinfo.value)


def test_a_negative_current_weight_is_refused() -> None:
    """A short is not expressible, not even on the current side of a row."""
    with pytest.raises(PortfolioContractError) as excinfo:
        PortfolioWeightChange("AAPL", -0.1, 0.1, 0.2)

    assert "current_weight must be between 0 and 1" in str(excinfo.value)


def test_a_target_weight_above_one_is_refused() -> None:
    """A share of the account cannot exceed the whole account."""
    with pytest.raises(PortfolioContractError) as excinfo:
        PortfolioWeightChange("AAPL", 0.0, 1.5, 1.5)

    assert "target_weight must be between 0 and 1" in str(excinfo.value)


@pytest.mark.parametrize("weight", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("field", ["current", "target", "delta"])
def test_a_weight_that_is_not_finite_is_refused(field: str, weight: float) -> None:
    """``NaN`` and ``±inf`` cannot be held, sized or compared."""
    values: dict[str, object] = {"current": 0.2, "target": 0.3, "delta": 0.1}
    values[field] = weight

    with pytest.raises(PortfolioContractError) as excinfo:
        change("AAPL", current=values["current"], target=values["target"], delta=values["delta"])

    assert "finite" in str(excinfo.value)


@pytest.mark.parametrize("weight", [True, False, "0.3", None])
def test_a_weight_that_is_not_a_number_is_refused_rather_than_converted(weight: object) -> None:
    """A bool is an ``int`` in Python; a string is text.  Neither is a weight."""
    with pytest.raises(PortfolioContractError) as excinfo:
        PortfolioWeightChange("AAPL", weight, 0.3, 0.0)  # type: ignore[arg-type]

    assert "must be a real number" in str(excinfo.value)


def test_an_inconsistent_delta_is_refused() -> None:
    """A row that disagrees with itself is a broken row, not a new opinion."""
    with pytest.raises(PortfolioContractError) as excinfo:
        PortfolioWeightChange("AAPL", 0.3, 0.45, 0.05)

    assert "weight_change is 0.05" in str(excinfo.value)
    assert "0.15000000000000002" in str(excinfo.value)


def test_a_delta_inside_the_tolerance_is_accepted_and_never_recomputed() -> None:
    """The caller's number is stored, even when the arithmetic would round differently."""
    row = PortfolioWeightChange("AAPL", 0.1, 0.3, 0.2)

    assert row.weight_change == 0.2
    assert row.weight_change != 0.3 - 0.1


def test_every_problem_of_a_row_is_collected_in_one_error() -> None:
    """One refusal names the symbol, both weights and the delta."""
    with pytest.raises(PortfolioContractError) as excinfo:
        PortfolioWeightChange("cash", -0.5, 2.0, "lots")  # type: ignore[arg-type]

    problems = excinfo.value.problems
    assert len(problems) == 4
    text = str(excinfo.value)
    assert "invalid PortfolioWeightChange" in text
    assert "upper-case ticker" in text
    assert "current_weight must be between 0 and 1" in text
    assert "target_weight must be between 0 and 1" in text
    assert "must be a real number" in text


def test_a_row_is_frozen() -> None:
    """What a rebalance was cannot change while a report is being written."""
    row = change()

    with pytest.raises(FrozenInstanceError):
        row.weight_change = 0.0  # type: ignore[misc]


def test_a_zero_weight_is_legal_on_both_sides() -> None:
    """Zero is the point of a change: nothing held, or nothing wanted any more."""
    row = PortfolioWeightChange("AAPL", 0.0, 0.0, 0.0)

    assert row.is_unchanged
    assert not row.is_entry
    assert not row.is_exit


@pytest.mark.parametrize(
    ("current", "target", "expected"),
    [
        (0.2, 0.5, (True, False, False)),
        (0.5, 0.2, (False, True, False)),
        (0.3, 0.3, (False, False, True)),
        (0.0, 0.4, (True, False, False)),
        (0.4, 0.0, (False, True, False)),
    ],
)
def test_the_direction_properties_describe_the_target(
    current: float, target: float, expected: tuple[bool, bool, bool]
) -> None:
    """Exactly one of buy, sell and unchanged is true for every row."""
    row = change("AAPL", current=current, target=target)

    assert (row.is_buy, row.is_sell, row.is_unchanged) == expected


def test_an_entry_needs_an_exact_zero_on_the_current_side() -> None:
    """A state either holds a name or does not, so "nearly zero" is not an entry."""
    dust = change("AAPL", current=1e-15, target=0.5)

    assert not dust.is_entry
    assert dust.is_buy
    assert change("AAPL", current=0.0, target=0.5).is_entry


def test_an_exit_needs_an_exact_zero_on_the_target_side() -> None:
    """A target either wants a name or does not, so "nearly zero" is not an exit."""
    dust = change("AAPL", current=0.5, target=1e-15)

    assert not dust.is_exit
    assert dust.is_sell
    assert change("AAPL", current=0.5, target=0.0).is_exit


def test_the_row_error_is_a_portfolio_error() -> None:
    """Every refusal of this layer is catchable as one type."""
    with pytest.raises(PortfolioError) as excinfo:
        PortfolioWeightChange("AAPL", 0.0, 0.0, 0.5)

    assert isinstance(excinfo.value, PortfolioContractError)


# ---------------------------------------------------------------------------
# PortfolioTransition: the object it accepts and every difference it refuses
# ---------------------------------------------------------------------------
def example_pair() -> tuple[PortfolioState, TargetPortfolio]:
    """The worked example as a ``(state, target)`` pair."""
    return (
        make_state(EXAMPLE_STATE, cash_weight=0.1),
        make_target(EXAMPLE_TARGET, cash_weight=0.1),
    )


def example_rows() -> tuple[PortfolioWeightChange, ...]:
    """The three rows the worked example must report."""
    return transition_of(*example_pair()).changes


def test_a_hand_built_transition_of_the_right_rows_is_accepted() -> None:
    """The object is a value, so a caller may state it directly -- if it agrees."""
    state, target = example_pair()
    rows = example_rows()

    transition = PortfolioTransition(
        as_of_date=AS_OF,
        current_state=state,
        target_portfolio=target,
        changes=rows,
    )

    assert transition.changes == rows
    assert transition.symbols == ("MSFT", "NVDA", "AAPL")


def test_a_date_string_is_normalized_to_a_plain_date() -> None:
    """``"YYYY-MM-DD"`` is accepted and stored as a ``date``, exactly like the contract."""
    state, target = example_pair()

    transition = PortfolioTransition(
        as_of_date="2025-01-09",  # type: ignore[arg-type]
        current_state=state,
        target_portfolio=target,
        changes=example_rows(),
    )

    assert transition.as_of_date == AS_OF
    assert isinstance(transition.as_of_date, date)


def test_a_current_state_that_is_not_a_state_is_refused() -> None:
    """The current side is a validated state, never a mapping of weights."""
    _, target = example_pair()

    with pytest.raises(PortfolioContractError) as excinfo:
        PortfolioTransition(
            as_of_date=AS_OF,
            current_state=EXAMPLE_STATE,  # type: ignore[arg-type]
            target_portfolio=target,
            changes=example_rows(),
        )

    assert "current_state must be a PortfolioState" in str(excinfo.value)


def test_a_target_that_is_not_a_target_is_refused() -> None:
    """The other side is a validated target, never the state it came from."""
    state, _ = example_pair()

    with pytest.raises(PortfolioContractError) as excinfo:
        PortfolioTransition(
            as_of_date=AS_OF,
            current_state=state,
            target_portfolio=state,  # type: ignore[arg-type]
            changes=example_rows(),
        )

    assert "target_portfolio must be a TargetPortfolio" in str(excinfo.value)


def test_both_sides_that_are_wrong_are_reported_together() -> None:
    """One refusal lists every broken side, not just the first."""
    with pytest.raises(PortfolioContractError) as excinfo:
        PortfolioTransition(
            as_of_date=AS_OF,
            current_state=None,  # type: ignore[arg-type]
            target_portfolio=None,  # type: ignore[arg-type]
            changes=(),
        )

    assert len(excinfo.value.problems) == 2


def test_a_date_that_is_not_the_targets_date_is_refused() -> None:
    """A transition is dated the decision its target was built for."""
    state, target = example_pair()

    with pytest.raises(PortfolioContractError) as excinfo:
        PortfolioTransition(
            as_of_date=PREVIOUS_SESSION,
            current_state=state,
            target_portfolio=target,
            changes=example_rows(),
        )

    assert "2025-01-02" in str(excinfo.value)
    assert "target_portfolio.as_of_date" in str(excinfo.value)


def test_a_timestamp_is_refused_even_at_midnight() -> None:
    """A moment is not the day a rebalance was decided."""
    state, target = example_pair()

    with pytest.raises(PortfolioContractError) as excinfo:
        PortfolioTransition(
            as_of_date=datetime(2025, 1, 9),  # type: ignore[arg-type]
            current_state=state,
            target_portfolio=target,
            changes=example_rows(),
        )

    assert "timestamp" in str(excinfo.value)


def test_a_state_from_the_future_is_refused_by_the_builder() -> None:
    """A state dated after the decision would leak information that did not exist."""
    with pytest.raises(PortfolioContractError) as excinfo:
        build_portfolio_transition(
            current_state=make_state(as_of_date=LATER_SESSION),
            target_portfolio=make_target(),
        )

    assert "later than" in str(excinfo.value)


def test_changes_that_are_not_a_tuple_are_refused() -> None:
    """A list promises no order and a generator can be read only once."""
    state, target = example_pair()

    with pytest.raises(PortfolioContractError) as excinfo:
        PortfolioTransition(
            as_of_date=AS_OF,
            current_state=state,
            target_portfolio=target,
            changes=list(example_rows()),  # type: ignore[arg-type]
        )

    assert "must be a tuple" in str(excinfo.value)


def test_a_row_that_is_not_a_weight_change_is_refused() -> None:
    """A bare quadruple has never been validated, so it is refused by name."""
    state, target = example_pair()
    rows = example_rows()

    with pytest.raises(PortfolioContractError) as excinfo:
        PortfolioTransition(
            as_of_date=AS_OF,
            current_state=state,
            target_portfolio=target,
            changes=rows + (("TSLA", 0.0, 0.0, 0.0),),  # type: ignore[arg-type]
        )

    assert "must be a PortfolioWeightChange" in str(excinfo.value)


def test_a_duplicate_symbol_is_refused() -> None:
    """Two rows for one holding would make its weight change ambiguous."""
    state, target = example_pair()
    rows = example_rows()

    with pytest.raises(PortfolioContractError) as excinfo:
        PortfolioTransition(
            as_of_date=AS_OF,
            current_state=state,
            target_portfolio=target,
            changes=rows + (rows[0],),
        )

    assert "more than once" in str(excinfo.value)


def test_a_missing_exit_is_refused() -> None:
    """A holding the target drops must be reported, not left out."""
    state, target = example_pair()
    rows = example_rows()

    with pytest.raises(PortfolioContractError) as excinfo:
        PortfolioTransition(
            as_of_date=AS_OF,
            current_state=state,
            target_portfolio=target,
            changes=(rows[0], rows[1]),
        )

    assert "omits" in str(excinfo.value)
    assert "AAPL" in str(excinfo.value)


def test_a_missing_entry_is_refused() -> None:
    """A name the target adds must be reported, not left out."""
    state, target = example_pair()
    rows = example_rows()

    with pytest.raises(PortfolioContractError) as excinfo:
        PortfolioTransition(
            as_of_date=AS_OF,
            current_state=state,
            target_portfolio=target,
            changes=(rows[0], rows[2]),
        )

    assert "omits" in str(excinfo.value)
    assert "NVDA" in str(excinfo.value)


def test_a_symbol_neither_portfolio_holds_is_refused() -> None:
    """A transition cannot invent a third symbol."""
    state, target = example_pair()
    rows = example_rows()

    with pytest.raises(PortfolioContractError) as excinfo:
        PortfolioTransition(
            as_of_date=AS_OF,
            current_state=state,
            target_portfolio=target,
            changes=rows + (PortfolioWeightChange("TSLA", 0.0, 0.0, 0.0),),
        )

    assert "neither portfolio holds" in str(excinfo.value)
    assert "TSLA" in str(excinfo.value)


def test_a_current_weight_that_contradicts_the_state_is_refused() -> None:
    """A row restates what the account holds; it never re-measures it."""
    state, target = example_pair()
    rows = example_rows()

    with pytest.raises(PortfolioContractError) as excinfo:
        PortfolioTransition(
            as_of_date=AS_OF,
            current_state=state,
            target_portfolio=target,
            changes=(
                PortfolioWeightChange("MSFT", 0.4, 0.45, 0.05),
                rows[1],
                rows[2],
            ),
        )

    assert "current_state holds" in str(excinfo.value)


def test_a_target_weight_that_contradicts_the_target_is_refused() -> None:
    """A row restates what the target asked for; it never revises it."""
    state, target = example_pair()
    rows = example_rows()

    with pytest.raises(PortfolioContractError) as excinfo:
        PortfolioTransition(
            as_of_date=AS_OF,
            current_state=state,
            target_portfolio=target,
            changes=(
                PortfolioWeightChange("MSFT", 0.3, 0.5, 0.2),
                rows[1],
                rows[2],
            ),
        )

    assert "target_portfolio holds" in str(excinfo.value)


def test_a_nonzero_current_weight_for_a_symbol_the_state_lacks_is_refused() -> None:
    """The current side of a name the account does not hold is exactly zero."""
    state, target = example_pair()
    rows = example_rows()

    with pytest.raises(PortfolioContractError) as excinfo:
        PortfolioTransition(
            as_of_date=AS_OF,
            current_state=state,
            target_portfolio=target,
            changes=(
                rows[0],
                PortfolioWeightChange("NVDA", 0.1, 0.45, 0.35),
                rows[2],
            ),
        )

    assert "current_state does not hold it" in str(excinfo.value)


def test_a_nonzero_target_weight_for_a_symbol_the_target_lacks_is_refused() -> None:
    """The target side of a name the target does not want is exactly zero."""
    state, target = example_pair()
    rows = example_rows()

    with pytest.raises(PortfolioContractError) as excinfo:
        PortfolioTransition(
            as_of_date=AS_OF,
            current_state=state,
            target_portfolio=target,
            changes=(
                rows[0],
                rows[1],
                PortfolioWeightChange("AAPL", 0.6, 0.1, -0.5),
            ),
        )

    assert "target_portfolio does not hold it" in str(excinfo.value)


def test_a_shuffled_order_is_refused_and_the_first_mismatch_is_named() -> None:
    """Target first, exits last: the order is a promise, not a preference."""
    state, target = example_pair()
    rows = example_rows()

    with pytest.raises(PortfolioContractError) as excinfo:
        PortfolioTransition(
            as_of_date=AS_OF,
            current_state=state,
            target_portfolio=target,
            changes=(rows[1], rows[0], rows[2]),
        )

    text = str(excinfo.value)
    assert "is ordered" in text
    assert "position 0" in text
    assert "'NVDA'" in text
    assert "'MSFT'" in text


def test_several_problems_are_collected_in_one_error() -> None:
    """A caller sees the whole list rather than one line per attempt."""
    state, target = example_pair()
    contradiction = PortfolioWeightChange("MSFT", 0.4, 0.45, 0.05)
    rows = example_rows()

    with pytest.raises(PortfolioContractError) as excinfo:
        PortfolioTransition(
            as_of_date=PREVIOUS_SESSION,
            current_state=state,
            target_portfolio=target,
            changes=(contradiction, rows[1], rows[2], contradiction),
        )

    problems = excinfo.value.problems
    assert len(problems) >= 3
    text = str(excinfo.value)
    assert "as_of_date is 2025-01-02" in text
    assert "more than once" in text
    assert "current_state holds" in text


def test_a_transition_is_frozen() -> None:
    """A rebalance that has been reported cannot be re-stated afterwards."""
    transition = mixed_transition()

    with pytest.raises(FrozenInstanceError):
        transition.changes = ()  # type: ignore[misc]


def test_a_transition_holds_the_very_objects_it_was_given() -> None:
    """The two portfolios travel by reference: nothing is copied, so nothing is lost."""
    state, target = example_pair()

    transition = transition_of(state, target)

    assert transition.current_state is state
    assert transition.target_portfolio is target


def test_the_builder_refuses_a_pair_that_is_not_one() -> None:
    """Two wrong inputs are reported together, before anything is compared."""
    with pytest.raises(PortfolioContractError) as excinfo:
        build_portfolio_transition(
            current_state={"AAPL": 1.0},  # type: ignore[arg-type]
            target_portfolio={"MSFT": 1.0},  # type: ignore[arg-type]
        )

    text = str(excinfo.value)
    assert "cannot build a PortfolioTransition" in text
    assert "current_state must be a PortfolioState" in text
    assert "target_portfolio must be a TargetPortfolio" in text
    assert len(excinfo.value.problems) == 2


def test_the_builder_refuses_a_target_given_as_the_current_side() -> None:
    """Swapping the two arguments is caught, not silently reported as a wild rebalance."""
    _, target = example_pair()

    with pytest.raises(PortfolioContractError) as excinfo:
        build_portfolio_transition(
            current_state=target,  # type: ignore[arg-type]
            target_portfolio=target,
        )

    assert "current_state must be a PortfolioState" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Provenance: the target's own objects travel through the transition
# ---------------------------------------------------------------------------
#: A second strategy identity: the same rule reading one more factor.
SPEC_WITH_VALUE = StrategySpec(
    name="top_n_momentum",
    version="2.0.0",
    required_factors=(MOMENTUM, VALUE),
)

#: A second construction identity: the same rule, a different version.
CONSTRUCTION_V2 = PortfolioConstructionSpec(name="equal_weight", version="2.0.0")


def test_the_strategy_spec_comes_from_the_target() -> None:
    """The rule and its version are the target's, not a re-derived guess."""
    target = make_target(source_result=make_result(spec=SPEC_WITH_VALUE))

    transition = transition_of(make_state(), target)

    assert transition.strategy_spec is target.strategy_spec
    assert transition.strategy_spec.name == "top_n_momentum"
    assert transition.strategy_spec.version == "2.0.0"


def test_the_factor_requirements_keep_their_exact_versions() -> None:
    """Every ``(factor_name, factor_version)`` pair survives the transition."""
    target = make_target(source_result=make_result(spec=SPEC_WITH_VALUE))

    transition = transition_of(make_state(), target)

    assert transition.factor_requirements is target.factor_requirements
    assert {(item.factor_name, item.factor_version) for item in transition.factor_requirements} == {
        ("momentum_12_1", "1.0.0"),
        ("book_to_price", "2.0.0"),
    }


def test_the_construction_spec_comes_from_the_target() -> None:
    """Which rule set the weights, and which version of it, is the target's answer."""
    target = make_target(construction_spec=CONSTRUCTION_V2)

    transition = transition_of(make_state(), target)

    assert transition.construction_spec is CONSTRUCTION_V2
    assert (transition.construction_spec.name, transition.construction_spec.version) == (
        "equal_weight",
        "2.0.0",
    )


def test_the_constraints_come_from_the_target() -> None:
    """The cash share and the single-name cap are reported as they were applied."""
    target = make_target()

    transition = transition_of(make_state(), target)

    assert transition.constraints is CONSTRAINTS
    assert transition.constraints.cash_weight == CONSTRAINTS.cash_weight
    assert transition.constraints.max_single_weight == CONSTRAINTS.max_single_weight


def test_the_universe_comes_from_the_target() -> None:
    """The pool the candidates were selected from is the same snapshot."""
    target = make_target()

    transition = transition_of(make_state(), target)

    assert transition.universe is target.universe
    assert transition.universe.symbols == SYMBOLS


def test_the_transition_restates_no_version_of_its_own() -> None:
    """Every provenance view is the target's object, never a copy or a rebuild."""
    target = make_target()

    transition = transition_of(make_state(), target)

    assert transition.strategy_spec is target.strategy_spec
    assert transition.factor_requirements is target.factor_requirements
    assert transition.construction_spec is target.construction_spec
    assert transition.constraints is target.constraints
    assert transition.universe is target.universe


# ---------------------------------------------------------------------------
# Purity: the inputs are read once and never touched
# ---------------------------------------------------------------------------
def test_the_builder_leaves_both_inputs_untouched() -> None:
    """Two objects in, one object out -- and the two are exactly as they were."""
    state, target = example_pair()
    state_before = (state.as_of_date, state.positions, state.cash_weight)
    target_before = (target.as_of_date, target.positions, target.cash_weight)

    transition_of(state, target)

    assert (state.as_of_date, state.positions, state.cash_weight) == state_before
    assert (target.as_of_date, target.positions, target.cash_weight) == target_before


def test_the_source_result_frame_is_untouched() -> None:
    """The difference reads the target's provenance; it never rewrites the frame."""
    _, target = example_pair()
    before = target.source_result.frame.copy()

    transition_of(make_state(EXAMPLE_STATE, cash_weight=0.1), target)

    pd.testing.assert_frame_equal(target.source_result.frame, before)


def test_two_builds_report_the_same_numbers() -> None:
    """The builder is a function of its two arguments and of nothing else."""
    state, target = example_pair()

    first = transition_of(state, target)
    second = transition_of(state, target)

    assert table(first) == table(second)
    assert (first.buy_weight, first.sell_weight, first.turnover) == (
        second.buy_weight,
        second.sell_weight,
        second.turnover,
    )


def test_two_builds_return_two_equal_objects() -> None:
    """A new object every call, with the same value: no cache and no shared row list."""
    state, target = example_pair()

    first = transition_of(state, target)
    second = transition_of(state, target)

    assert first is not second
    assert first.changes is not second.changes
    assert first == second


# ---------------------------------------------------------------------------
# The package surface and the architectural boundary
# ---------------------------------------------------------------------------
TRANSITION_PATH = Path(transition_module.__file__)

#: Modules ``transition.py`` may import: the standard library it needs and its own layer.
ALLOWED_IMPORTS: frozenset[str] = frozenset(
    {
        "__future__",
        "dataclasses",
        "datetime",
        "typing",
        "portfolio_sys.contracts",
        "portfolio_sys.errors",
    }
)

#: Any layer or library the transition may never reach.
FORBIDDEN_PREFIXES: tuple[str, ...] = (
    "data_sys",
    "factor_sys",
    "strategy_sys",
    "pandas",
    "numpy",
    "pyarrow",
    "duckdb",
    "exchange_calendars",
    "portfolio_sys.registry",
    "portfolio_sys.constructors",
    "portfolio_sys.transition",  # it may not import itself as a module either
)

#: Calls that would make a pure, in-memory difference impure.
FORBIDDEN_CALLS: frozenset[str] = frozenset(
    {"__import__", "compile", "connect", "eval", "exec", "input", "open", "print", "urlopen"}
)

#: Identifiers the module must never use: an order, a fill, a price, a share count or a
#: backtest belongs to another layer's vocabulary.
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
        "nav",
        "pnl",
        "backtest",
        "execution_date",
        "valid_date",
        "construct",
        "register",
        "create",
    }
)

#: Text that must never appear in the module's *code*; docstrings are blanked first, so
#: the prose may still explain what the layer deliberately avoids.
FORBIDDEN_TEXT: tuple[str, ...] = (
    "import pandas",
    "import numpy",
    "import pyarrow",
    "import duckdb",
    "import exchange_calendars",
    "data_sys",
    "factor_sys",
    "strategy_sys",
    "portfolio_sys.registry",
    "portfolio_sys.constructors",
    "read_parquet",
    "to_parquet",
    "urlopen",
    "open(",
    "connect(",
)


def parse(path: Path) -> ast.Module:
    """The syntax tree of ``path``."""
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def imported_modules(path: Path) -> set[str]:
    """Every module named by an import statement anywhere in ``path``."""
    names: set[str] = set()
    for node in ast.walk(parse(path)):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def mentioned_names(path: Path) -> set[str]:
    """Every name, attribute, argument and definition the module's code uses."""
    tree = parse(path)
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    names |= {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    names |= {node.arg for node in ast.walk(tree) if isinstance(node, ast.arg)}
    names |= {
        node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }
    return names


def called_names(path: Path) -> set[str]:
    """Every function or method the module calls, exactly as written."""
    return {ast.unparse(node.func) for node in ast.walk(parse(path)) if isinstance(node, ast.Call)}


def code_of(path: Path) -> str:
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


def test_the_package_root_exports_the_transition_objects() -> None:
    """A caller imports the three names from the root, never from a submodule."""
    assert portfolio_sys.PortfolioTransition is PortfolioTransition
    assert portfolio_sys.PortfolioWeightChange is PortfolioWeightChange
    assert portfolio_sys.build_portfolio_transition is build_portfolio_transition
    assert {"PortfolioTransition", "PortfolioWeightChange", "build_portfolio_transition"} <= set(
        portfolio_sys.__all__
    )


def test_the_module_declares_exactly_three_public_names() -> None:
    """Nothing else is exported: no runner, no registry, no constructor."""
    assert transition_module.__all__ == [
        "PortfolioTransition",
        "PortfolioWeightChange",
        "build_portfolio_transition",
    ]


def test_every_public_name_has_a_docstring() -> None:
    """The module, the three names and every public property explain themselves."""
    assert inspect.getdoc(transition_module)
    for name in transition_module.__all__:
        assert inspect.getdoc(getattr(transition_module, name)), name

    for klass in (PortfolioWeightChange, PortfolioTransition):
        for attribute, value in vars(klass).items():
            if attribute.startswith("_"):
                continue
            assert inspect.getdoc(value), (klass.__name__, attribute)


def test_the_module_imports_only_its_own_layer() -> None:
    """No data system, no factor layer, no strategy layer, no table, no rule, no pandas."""
    imported = imported_modules(TRANSITION_PATH)

    assert imported <= ALLOWED_IMPORTS, sorted(imported - ALLOWED_IMPORTS)
    assert [name for name in imported if name.startswith(FORBIDDEN_PREFIXES)] == []


def test_the_module_never_names_an_order_a_price_or_a_backtest() -> None:
    """The vocabulary of the execution and backtest layers never appears in the code."""
    offenders = sorted(mentioned_names(TRANSITION_PATH) & FORBIDDEN_IDENTIFIERS)

    assert offenders == []


def test_the_module_never_opens_prints_or_evaluates_anything() -> None:
    """A pure in-memory difference: no file, no socket, no print, no eval."""
    code = code_of(TRANSITION_PATH)

    assert called_names(TRANSITION_PATH) & FORBIDDEN_CALLS == set()
    assert [text for text in FORBIDDEN_TEXT if text in code] == []


def test_the_module_hard_codes_no_path_or_data_directory() -> None:
    """No test may depend on the repository's own ``data/`` tree."""
    code = code_of(TRANSITION_PATH).lower()

    for text in ("data/", "data\\", "c:\\", "/home", "/users", ".parquet", ".csv", "os.environ"):
        assert text not in code


def test_the_module_never_calls_construct_or_a_constructor() -> None:
    """A difference reads two finished objects; it never builds weights."""
    called = called_names(TRANSITION_PATH)
    mentioned = mentioned_names(TRANSITION_PATH)

    assert "construct" not in called
    assert "construct" not in mentioned
    assert [name for name in mentioned if name.endswith("Constructor")] == []
    assert [name for name in mentioned if "Registry" in name] == []


def test_the_contract_the_rules_and_the_table_never_import_the_transition() -> None:
    """The dependency runs one way: nothing below a transition knows it exists."""
    package = Path(portfolio_sys.__file__).parent
    below = [
        package / "contracts.py",
        package / "errors.py",
        package / "registry.py",
        package / "constructors" / "__init__.py",
        package / "constructors" / "capped_score_weighted.py",
    ]
    names = {"PortfolioTransition", "PortfolioWeightChange", "build_portfolio_transition"}

    for module in below:
        imported = imported_modules(module)
        assert "portfolio_sys.transition" not in imported, module.name
        assert mentioned_names(module) & names == set(), module.name


def test_the_transition_module_is_the_module_the_package_imports() -> None:
    """One module, imported once: no second copy of the difference under another name."""
    imported = {name for name in sys.modules if name.startswith("portfolio_sys.transition")}

    assert imported == {"portfolio_sys.transition"}
    assert TRANSITION_PATH.name == "transition.py"
