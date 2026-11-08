"""Tests for :class:`~portfolio_sys.constructors.CappedScoreWeightedConstructor`.

The first real weighting rule of the project: split the investable share in
proportion to the selected symbols' scores, pin every name that would exceed
``constraints.max_single_weight`` and re-split the rest -- still by score -- until
nothing is above the cap.

Everything here is in memory and offline.  A ``StrategyResult`` is built from a
``UniverseSnapshot`` plus a small frame, the current state is a tuple of
positions, and the constraints are two numbers: no ``data/`` directory, factor
store, trading calendar, network call or DuckDB connection is involved, which is
exactly the point of a constructor that reads one ``PortfolioContext``.

Covered in this order:

* the interface: identity, statelessness, and the refusal of a non-context,
* pure score-proportional weights, with and without cash,
* one cap and one redistribution (the documented five-name case),
* several rounds of capping,
* cash and the cap together,
* the boundaries: exactly at the cap, equal scores, cap equal to the budget,
  a single holding, and floating-point totals,
* scores at the top of the float range: no overflow, no silent zero weight, and
  ordinary weights unchanged to within a last-bit rounding difference,
* infeasible constraints,
* unusable selected scores (zero, negative, NaN, ±inf, and one that underflows),
* an empty selection,
* provenance and the order of the positions,
* purity of the inputs and of repeated calls,
* errors, package surface and the architectural boundary.
"""

from __future__ import annotations

import ast
import inspect
import math
from dataclasses import FrozenInstanceError
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

import portfolio_sys
import portfolio_sys.constructors as constructors_package
import portfolio_sys.constructors.capped_score_weighted as rule_module
from factor_sys.universe import UniverseSnapshot
from portfolio_sys import (
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
from portfolio_sys.constructors import DEFAULT_VERSION, CappedScoreWeightedConstructor
from portfolio_sys.errors import PortfolioConstructionError
from strategy_sys import (
    STRATEGY_RESULT_COLUMNS,
    FactorRequirement,
    StrategyResult,
    StrategySpec,
)

#: The in-memory universe every fixture below works with: one row per symbol, in
#: this order, so a test can tell "universe order" from "score order".
UNIVERSE: tuple[str, ...] = ("AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOG")

#: The decision date; the state a construction starts from is dated one session
#: earlier, which is the normal "last rebalance" shape.
AS_OF: date = date(2024, 1, 5)
PREVIOUS_SESSION: date = date(2024, 1, 4)

#: The exact factor identity and the spec the fixture results are produced under.
MOMENTUM = FactorRequirement(factor_name="momentum_12_1", factor_version="1.0.0")
SPEC = StrategySpec(name="top_n_momentum", version="1.0.0", required_factors=(MOMENTUM,))

#: The contract's own tolerance: a weight total may miss ``1.0`` by this much and
#: still be one portfolio, so every "adds up" assertion uses it.
TOLERANCE: float = 1e-12


# ---------------------------------------------------------------------------
# In-memory fixtures (no disk, no network, no store)
# ---------------------------------------------------------------------------
@pytest.fixture
def constructor() -> CappedScoreWeightedConstructor:
    """A fresh builder per test: the rule holds no state to carry between runs."""
    return CappedScoreWeightedConstructor()


def make_universe(
    symbols: tuple[str, ...] = UNIVERSE, *, as_of_date: object = AS_OF
) -> UniverseSnapshot:
    """A :class:`UniverseSnapshot` for the fixtures."""
    return UniverseSnapshot(  # type: ignore[arg-type]
        as_of_date=as_of_date,
        symbols=tuple(symbols),
        source="config/universe_seed.csv",
    )


def make_result(
    scores: dict[str, float],
    selected: tuple[str, ...],
    *,
    symbols: tuple[str, ...] = UNIVERSE,
    as_of_date: object = AS_OF,
    spec: StrategySpec = SPEC,
) -> StrategyResult:
    """A strategy result scored by ``scores`` and flagging exactly ``selected``.

    ``score`` is ``NaN`` and ``rank`` missing for a symbol that is not in
    ``scores`` -- the "not scored" shape the strategy contract types -- and the
    ranks of the scored symbols follow descending score with a symbol tie-break,
    so the ranks are the unique positive integers the contract asks for.
    """
    order = sorted(
        (symbol for symbol in symbols if symbol in scores),
        key=lambda symbol: (-scores[symbol], symbol),
    )
    ranks = {symbol: position for position, symbol in enumerate(order, start=1)}
    frame = pd.DataFrame(
        {
            "as_of_date": [pd.Timestamp(as_of_date)] * len(symbols),
            "strategy_name": [spec.name] * len(symbols),
            "strategy_version": [spec.version] * len(symbols),
            "symbol": list(symbols),
            "score": pd.Series(
                [float(scores[symbol]) if symbol in scores else math.nan for symbol in symbols],
                dtype="float64",
            ),
            "rank": pd.Series([ranks.get(symbol) for symbol in symbols], dtype="Int64"),
            "is_selected": pd.Series([symbol in selected for symbol in symbols], dtype="bool"),
        }
    )
    return StrategyResult(
        frame,
        universe=make_universe(symbols, as_of_date=as_of_date),
        strategy_spec=spec,
        as_of_date=as_of_date,  # type: ignore[arg-type]
    )


def make_state(
    positions: tuple[PortfolioPosition, ...] = (),
    *,
    cash_weight: float = 1.0,
    as_of_date: object = PREVIOUS_SESSION,
) -> PortfolioState:
    """The account before the decision: by default entirely in cash."""
    return PortfolioState(  # type: ignore[arg-type]
        as_of_date=as_of_date,
        positions=positions,
        cash_weight=cash_weight,
    )


def make_constraints(
    *, cash_weight: object = 0.0, max_single_weight: object = 1.0
) -> PortfolioConstraints:
    """Fully invested and uncapped unless told otherwise, like the contract fixture."""
    return PortfolioConstraints(  # type: ignore[arg-type]
        cash_weight=cash_weight,
        max_single_weight=max_single_weight,
    )


def make_context(
    scores: dict[str, float],
    selected: tuple[str, ...],
    *,
    cash_weight: object = 0.0,
    max_single_weight: object = 1.0,
    symbols: tuple[str, ...] = UNIVERSE,
    strategy_result: object = None,
    current_state: object = None,
) -> PortfolioContext:
    """A complete construction input for one selection and one set of constraints."""
    return PortfolioContext(  # type: ignore[arg-type]
        as_of_date=AS_OF,
        strategy_result=(
            make_result(scores, selected, symbols=symbols)
            if strategy_result is None
            else strategy_result
        ),
        current_state=make_state() if current_state is None else current_state,
        constraints=make_constraints(cash_weight=cash_weight, max_single_weight=max_single_weight),
    )


def weights_of(target: TargetPortfolio) -> dict[str, float]:
    """``{symbol: weight}`` of a target, as the target itself reports it."""
    return target.target_weights_by_symbol


def corrupt_score(context: PortfolioContext, symbol: str, score: float) -> None:
    """Write ``score`` straight into the result's stored frame.

    ``StrategyResult`` refuses *at construction* to flag a symbol that has no
    finite score, so ``NaN`` and ``±inf`` can only reach a constructor from a
    result that was corrupted afterwards.  The rule must still refuse them, and
    this is the one way to produce that state without touching the strategy layer:
    the stored frame is mutable, the object around it is frozen.
    """
    rows = context.strategy_result._rows  # the private frame of a frozen object
    rows.loc[rows["symbol"] == symbol, "score"] = score


# ---------------------------------------------------------------------------
# The interface: identity, statelessness, and the input it refuses
# ---------------------------------------------------------------------------
def test_the_builder_is_a_portfolio_constructor(
    constructor: CappedScoreWeightedConstructor,
) -> None:
    assert isinstance(constructor, PortfolioConstructor)
    assert issubclass(CappedScoreWeightedConstructor, PortfolioConstructor)


def test_the_builder_identity_is_fixed(constructor: CappedScoreWeightedConstructor) -> None:
    assert constructor.spec == PortfolioConstructionSpec(
        name="capped_score_weighted", version="1.0.0"
    )
    assert constructor.spec.name == "capped_score_weighted"
    assert constructor.spec.version == "1.0.0"
    assert constructor.spec.version == DEFAULT_VERSION == "1.0.0"


def test_the_identity_is_one_shared_immutable_spec(
    constructor: CappedScoreWeightedConstructor,
) -> None:
    """Two instances are the same rule, so they report the very same spec object."""
    assert constructor.spec is CappedScoreWeightedConstructor().spec
    with pytest.raises(FrozenInstanceError):
        constructor.spec.name = "equal_weight"  # type: ignore[misc]


def test_the_builder_takes_no_parameters_of_its_own() -> None:
    """Cash and the cap come from the constraints, never from the constructor."""
    assert list(inspect.signature(CappedScoreWeightedConstructor).parameters) == []

    constructor = CappedScoreWeightedConstructor()
    for absent in ("cash_weight", "max_single_weight", "cap", "top_n", "weights"):
        assert not hasattr(constructor, absent), absent


def test_the_builder_repr_names_its_rule(constructor: CappedScoreWeightedConstructor) -> None:
    assert repr(constructor) == (
        "CappedScoreWeightedConstructor(spec="
        "PortfolioConstructionSpec(name='capped_score_weighted', version='1.0.0'))"
    )


def test_repeated_calls_accumulate_no_state(constructor: CappedScoreWeightedConstructor) -> None:
    """One instance, three calls, two different contexts: no order dependence."""
    first = make_context(
        {"AAPL": 3.0, "MSFT": 1.0}, ("AAPL", "MSFT"), cash_weight=0.2, max_single_weight=0.4
    )
    other = make_context({"NVDA": 1.0, "TSLA": 9.0}, ("NVDA", "TSLA"), cash_weight=0.25)

    before = constructor.construct(first)
    repeated = constructor.construct(first)
    constructor.construct(other)
    after = constructor.construct(first)

    assert weights_of(repeated) == weights_of(before)
    assert weights_of(after) == weights_of(before)
    assert before == repeated == after


def test_a_context_of_the_wrong_type_is_refused(
    constructor: CappedScoreWeightedConstructor,
) -> None:
    """A constructor reads the validated context, never a frame or a ticker list."""
    for not_a_context in (None, "AAPL", {"AAPL": 1.0}, make_state()):
        with pytest.raises(PortfolioContractError) as excinfo:
            constructor.construct(not_a_context)  # type: ignore[arg-type]

        assert "PortfolioContext" in str(excinfo.value)
        assert not isinstance(excinfo.value, PortfolioConstructionError)


# ---------------------------------------------------------------------------
# Pure score-proportional weights
# ---------------------------------------------------------------------------
def test_weights_follow_the_scores_when_nothing_is_capped(
    constructor: CappedScoreWeightedConstructor,
) -> None:
    """6:3:1 over 90% invested is 54%, 27%, 9% and 10% cash."""
    context = make_context(
        {"AAPL": 6.0, "MSFT": 3.0, "NVDA": 1.0},
        ("AAPL", "MSFT", "NVDA"),
        cash_weight=0.1,
    )

    target = constructor.construct(context)

    assert weights_of(target) == pytest.approx({"AAPL": 0.54, "MSFT": 0.27, "NVDA": 0.09})
    assert target.cash_weight == 0.1
    assert target.invested_weight == pytest.approx(0.9, abs=TOLERANCE)
    assert sum(weights_of(target).values()) + target.cash_weight == pytest.approx(1.0)


def test_the_scale_of_the_scores_does_not_matter(
    constructor: CappedScoreWeightedConstructor,
) -> None:
    """Only the ratios matter, so multiplying every score leaves the target unchanged."""
    plain = make_context({"AAPL": 6.0, "MSFT": 3.0, "NVDA": 1.0}, ("AAPL", "MSFT", "NVDA"))
    scaled = make_context(
        {"AAPL": 6000.0, "MSFT": 3000.0, "NVDA": 1000.0}, ("AAPL", "MSFT", "NVDA")
    )

    assert weights_of(constructor.construct(scaled)) == pytest.approx(
        weights_of(constructor.construct(plain))
    )


def test_only_the_selected_symbols_are_weighted(
    constructor: CappedScoreWeightedConstructor,
) -> None:
    """An unselected symbol takes no weight, however high its score is."""
    context = make_context(
        {"AAPL": 1.0, "MSFT": 1000.0, "NVDA": 1.0},
        ("AAPL", "NVDA"),
        cash_weight=0.2,
    )

    target = constructor.construct(context)

    assert target.target_symbols == ("AAPL", "NVDA")
    assert weights_of(target) == pytest.approx({"AAPL": 0.4, "NVDA": 0.4})


# ---------------------------------------------------------------------------
# One cap, one redistribution: the documented five-name case
# ---------------------------------------------------------------------------
def test_the_documented_five_name_case(
    constructor: CappedScoreWeightedConstructor,
) -> None:
    """30/20/20/15/15 at a 25% cap: A pinned, B:C:D:E re-split 20:20:15:15."""
    symbols = ("A", "B", "C", "D", "E")
    context = make_context(
        {"A": 30.0, "B": 20.0, "C": 20.0, "D": 15.0, "E": 15.0},
        symbols,
        symbols=symbols,
        max_single_weight=0.25,
    )

    target = constructor.construct(context)
    weights = weights_of(target)

    assert weights["A"] == pytest.approx(0.25, abs=TOLERANCE)
    assert weights["B"] == pytest.approx(0.75 * 20 / 70, abs=TOLERANCE)
    assert weights["C"] == pytest.approx(0.75 * 20 / 70, abs=TOLERANCE)
    assert weights["D"] == pytest.approx(0.75 * 15 / 70, abs=TOLERANCE)
    assert weights["E"] == pytest.approx(0.75 * 15 / 70, abs=TOLERANCE)
    assert target.cash_weight == 0.0
    assert target.invested_weight == pytest.approx(1.0, abs=TOLERANCE)
    assert max(weights.values()) <= 0.25


def test_the_redistribution_keeps_the_score_ratios(
    constructor: CappedScoreWeightedConstructor,
) -> None:
    """The capped name's share goes to the others *in proportion*, never equally."""
    symbols = ("A", "B", "C", "D", "E")
    context = make_context(
        {"A": 30.0, "B": 20.0, "C": 20.0, "D": 15.0, "E": 15.0},
        symbols,
        symbols=symbols,
        max_single_weight=0.25,
    )

    weights = weights_of(constructor.construct(context))

    assert weights["B"] / weights["C"] == pytest.approx(1.0)
    assert weights["B"] / weights["D"] == pytest.approx(20 / 15)
    assert weights["D"] / weights["E"] == pytest.approx(1.0)
    # Equal weight after the cap would have given every free name 18.75%.
    assert weights["B"] != pytest.approx(0.75 / 4, abs=1e-9)


def test_the_excess_is_not_left_in_cash(
    constructor: CappedScoreWeightedConstructor,
) -> None:
    """Capping the biggest name frees its share for the others; it does not become cash."""
    context = make_context(
        {"AAPL": 90.0, "MSFT": 4.0, "NVDA": 3.0, "TSLA": 3.0},
        ("AAPL", "MSFT", "NVDA", "TSLA"),
        max_single_weight=0.3,
    )

    target = constructor.construct(context)

    assert target.cash_weight == 0.0
    assert weights_of(target)["AAPL"] == pytest.approx(0.3, abs=TOLERANCE)
    assert target.invested_weight == pytest.approx(1.0, abs=TOLERANCE)
    assert weights_of(target) == pytest.approx(
        {"AAPL": 0.3, "MSFT": 0.28, "NVDA": 0.21, "TSLA": 0.21}, abs=1e-12
    )


# ---------------------------------------------------------------------------
# Several rounds of capping
# ---------------------------------------------------------------------------
def test_a_second_round_pins_the_name_that_is_still_too_big(
    constructor: CappedScoreWeightedConstructor,
) -> None:
    """100/10/10/10/10 at a 20% cap: one pin, then the rest sit exactly on the cap."""
    symbols = ("A", "B", "C", "D", "E")
    context = make_context(
        {"A": 100.0, "B": 10.0, "C": 10.0, "D": 10.0, "E": 10.0},
        symbols,
        symbols=symbols,
        max_single_weight=0.2,
    )

    target = constructor.construct(context)

    assert weights_of(target) == pytest.approx(dict.fromkeys(symbols, 0.2), abs=TOLERANCE)
    assert target.invested_weight == pytest.approx(1.0, abs=TOLERANCE)


def test_three_rounds_of_capping_keep_the_proportions(
    constructor: CappedScoreWeightedConstructor,
) -> None:
    """40/30/20/6/4 at a 25% cap needs three rounds: A, then B, then C.

    Round 1 pins A (40%) and B (30%); round 2 pins C (20/30 of what was left);
    round 3 splits the remainder 6:4 between D and E -- not equally, which is the
    whole point of re-splitting by score.
    """
    symbols = ("A", "B", "C", "D", "E")
    context = make_context(
        {"A": 40.0, "B": 30.0, "C": 20.0, "D": 6.0, "E": 4.0},
        symbols,
        symbols=symbols,
        max_single_weight=0.25,
    )

    target = constructor.construct(context)
    weights = weights_of(target)

    assert weights["A"] == pytest.approx(0.25, abs=TOLERANCE)
    assert weights["B"] == pytest.approx(0.25, abs=TOLERANCE)
    assert weights["C"] == pytest.approx(0.25, abs=TOLERANCE)
    assert weights["D"] == pytest.approx(0.15, abs=TOLERANCE)
    assert weights["E"] == pytest.approx(0.10, abs=TOLERANCE)
    assert target.invested_weight == pytest.approx(1.0, abs=TOLERANCE)
    assert max(weights.values()) <= 0.25


def test_a_capped_name_never_gets_more_than_the_cap(
    constructor: CappedScoreWeightedConstructor,
) -> None:
    """Whatever the scores, every weight stays inside the cap and inside [0, 1]."""
    symbols = ("AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOG")
    context = make_context(
        {"AAPL": 500.0, "MSFT": 200.0, "NVDA": 60.0, "TSLA": 6.0, "AMZN": 1.0, "GOOG": 0.5},
        symbols,
        cash_weight=0.05,
        max_single_weight=0.2,
    )

    target = constructor.construct(context)

    assert all(weight <= 0.2 for weight in weights_of(target).values())
    assert all(weight > 0.0 for weight in weights_of(target).values())
    assert target.invested_weight == pytest.approx(0.95, abs=TOLERANCE)
    assert target.cash_weight == 0.05


# ---------------------------------------------------------------------------
# Cash and the cap together
# ---------------------------------------------------------------------------
def test_cash_and_the_cap_hold_at_the_same_time(
    constructor: CappedScoreWeightedConstructor,
) -> None:
    """10% cash at a 25% cap: 90% in stock, A pinned, the rest re-split 20:20:15:15."""
    symbols = ("A", "B", "C", "D", "E")
    context = make_context(
        {"A": 30.0, "B": 20.0, "C": 20.0, "D": 15.0, "E": 15.0},
        symbols,
        symbols=symbols,
        cash_weight=0.1,
        max_single_weight=0.25,
    )

    target = constructor.construct(context)
    weights = weights_of(target)

    assert target.cash_weight == 0.1
    assert target.invested_weight == pytest.approx(0.9, abs=TOLERANCE)
    assert weights["A"] == pytest.approx(0.25, abs=TOLERANCE)
    assert weights["B"] == pytest.approx(0.65 * 20 / 70, abs=TOLERANCE)
    assert weights["C"] == pytest.approx(0.65 * 20 / 70, abs=TOLERANCE)
    assert weights["D"] == pytest.approx(0.65 * 15 / 70, abs=TOLERANCE)
    assert weights["E"] == pytest.approx(0.65 * 15 / 70, abs=TOLERANCE)
    assert max(weights.values()) <= 0.25
    assert sum(weights.values()) + target.cash_weight == pytest.approx(1.0, abs=TOLERANCE)


# ---------------------------------------------------------------------------
# Boundaries
# ---------------------------------------------------------------------------
def test_a_weight_exactly_at_the_cap_is_not_treated_as_over_it(
    constructor: CappedScoreWeightedConstructor,
) -> None:
    """Equal scores at a cap that exactly matches the share: nothing is pinned."""
    context = make_context(
        {"AAPL": 2.0, "MSFT": 2.0}, ("AAPL", "MSFT"), cash_weight=0.5, max_single_weight=0.25
    )

    target = constructor.construct(context)

    assert weights_of(target) == pytest.approx({"AAPL": 0.25, "MSFT": 0.25}, abs=TOLERANCE)
    assert target.cash_weight == 0.5


def test_equal_scores_split_the_investable_share_evenly(
    constructor: CappedScoreWeightedConstructor,
) -> None:
    """Equal scores are the one case where the proportions *are* equal weights."""
    symbols = ("A", "B", "C", "D", "E")
    invested = make_context(dict.fromkeys(symbols, 1.0), symbols, symbols=symbols)
    with_cash = make_context(dict.fromkeys(symbols, 1.0), symbols, symbols=symbols, cash_weight=0.1)

    assert weights_of(constructor.construct(invested)) == pytest.approx(
        dict.fromkeys(symbols, 0.2), abs=TOLERANCE
    )
    assert weights_of(constructor.construct(with_cash)) == pytest.approx(
        dict.fromkeys(symbols, 0.18), abs=TOLERANCE
    )


def test_the_cap_may_equal_the_whole_investable_share(
    constructor: CappedScoreWeightedConstructor,
) -> None:
    """A cap of exactly 90% over a 90% budget never pins, however skewed the scores."""
    context = make_context(
        {"AAPL": 100.0, "MSFT": 1.0, "NVDA": 1.0},
        ("AAPL", "MSFT", "NVDA"),
        cash_weight=0.1,
        max_single_weight=0.9,
    )

    target = constructor.construct(context)

    assert weights_of(target)["AAPL"] == pytest.approx(0.9 * 100 / 102, abs=TOLERANCE)
    assert target.invested_weight == pytest.approx(0.9, abs=TOLERANCE)


def test_a_single_selected_symbol_takes_the_investable_share(
    constructor: CappedScoreWeightedConstructor,
) -> None:
    """``top_n=1``: one name, one weight, the rest in cash."""
    context = make_context({"NVDA": 7.5}, ("NVDA",), cash_weight=0.2, max_single_weight=1.0)

    target = constructor.construct(context)

    assert target.target_symbols == ("NVDA",)
    assert weights_of(target) == pytest.approx({"NVDA": 0.8}, abs=TOLERANCE)
    assert target.cash_weight == 0.2


def test_a_repeating_decimal_total_still_adds_up(
    constructor: CappedScoreWeightedConstructor,
) -> None:
    """Seven equal names give 1/7 each; the total stays inside the contract tolerance."""
    symbols = ("AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOG", "META")
    context = make_context(dict.fromkeys(symbols, 3.0), symbols, symbols=symbols)

    target = constructor.construct(context)

    assert weights_of(target) == pytest.approx(dict.fromkeys(symbols, 1 / 7), abs=TOLERANCE)
    assert target.invested_weight == pytest.approx(1.0, abs=TOLERANCE)
    assert abs(sum(weights_of(target).values()) - 1.0) <= TOLERANCE


# ---------------------------------------------------------------------------
# Scores at the top of the float range: no overflow, no silent zero
# ---------------------------------------------------------------------------
def test_two_scores_at_the_top_of_the_float_range_split_the_account_evenly() -> None:
    """``1e308`` beside ``1e308`` is half the account each, not two zeroes.

    Adding the two raw scores overflows to ``inf``, so ``budget * score / inf`` is
    ``0.0`` for both names.  Scaling by the round's largest score first keeps the
    total at ``2.0`` and the ratio at ``1:1``.
    """
    assert rule_module._capped_weights({"AAPL": 1e308, "MSFT": 1e308}, 1.0, 1.0) == {
        "AAPL": 0.5,
        "MSFT": 0.5,
    }


def test_huge_scores_keep_their_proportional_ratio() -> None:
    """The ratio between two huge scores survives when their sum would not.

    ``1e307 + 2e307`` overflows a float sum as well, yet the split is exactly
    ``1:2``; so is ``8e307 : 1.6e308`` -- the ``1e308 : 2e308`` pair, moved down to
    where both ends and the ratio still fit in a float -- and ``1e308 : 1.7e308``,
    whose sum ``2.7e308`` is past the largest representable float.
    """
    assert rule_module._capped_weights({"A": 1e307, "B": 2e307}, 1.0, 1.0) == {
        "A": 1 / 3,
        "B": 2 / 3,
    }
    assert rule_module._capped_weights({"A": 8e307, "B": 1.6e308}, 1.0, 1.0) == {
        "A": 1 / 3,
        "B": 2 / 3,
    }
    assert rule_module._capped_weights({"A": 1e308, "B": 1.7e308}, 1.0, 1.0) == pytest.approx(
        {"A": 1 / 2.7, "B": 1.7 / 2.7}, abs=TOLERANCE
    )


def test_huge_scores_are_still_capped_and_the_rest_re_split(
    constructor: CappedScoreWeightedConstructor,
) -> None:
    """A cap bites on huge scores exactly as it does on ordinary ones.

    ``1e308, 1e308`` want 28.6% each of the account; at a 25% cap they are pinned
    and the remaining 50% goes to the three equal small names in proportion --
    16.7% each, not re-divided equally across all five.
    """
    scores = {"AAPL": 1e308, "MSFT": 1e308, "NVDA": 5e307, "TSLA": 5e307, "AMZN": 5e307}
    context = make_context(scores, tuple(scores), max_single_weight=0.25)

    target = constructor.construct(context)

    assert weights_of(target) == pytest.approx(
        {"AAPL": 0.25, "MSFT": 0.25, "NVDA": 1 / 6, "TSLA": 1 / 6, "AMZN": 1 / 6},
        abs=TOLERANCE,
    )
    assert target.invested_weight == pytest.approx(1.0, abs=TOLERANCE)
    assert all(weight <= 0.25 for weight in weights_of(target).values())


def test_a_huge_score_can_be_capped_on_its_own() -> None:
    """One dominant huge score is pinned; the two small ones keep their 1:1 ratio."""
    assert rule_module._capped_weights({"A": 1e308, "B": 2e307, "C": 2e307}, 1.0, 0.5) == {
        "A": 0.5,
        "B": 0.25,
        "C": 0.25,
    }


def test_a_huge_selection_no_longer_collapses_into_a_refusal(
    constructor: CappedScoreWeightedConstructor,
) -> None:
    """Before the scaling, this selection "underflowed" and was refused by mistake.

    Both shares came out ``0.0`` -- a zero weight is not a holding -- so the rule
    reported an unusable score for perfectly good input.  The construction now
    succeeds with two 50% positions.
    """
    context = make_context({"AAPL": 1e308, "MSFT": 1e308}, ("AAPL", "MSFT"))

    target = constructor.construct(context)

    assert weights_of(target) == {"AAPL": 0.5, "MSFT": 0.5}
    assert target.invested_weight == 1.0
    assert target.cash_weight == 0.0


def test_a_share_that_underflows_beside_a_huge_score_is_still_refused(
    constructor: CappedScoreWeightedConstructor,
) -> None:
    """Scaling does not rescue a genuinely negligible score: it is refused, by name.

    ``5e-324 / 1e308`` is ``0.0`` even after scaling, exactly as ``5e-324 / 1e300``
    was before, so the refusal the rule already had -- a zero weight is not a
    holding -- is unchanged.
    """
    context = make_context({"AAPL": 1e308, "MSFT": 5e-324}, ("AAPL", "MSFT"))

    with pytest.raises(PortfolioConstructionError) as excinfo:
        constructor.construct(context)

    message = str(excinfo.value)
    assert "MSFT" in message
    assert "5e-324" in message
    assert "AAPL" not in message


@pytest.mark.parametrize(
    ("scores", "cash", "cap"),
    [
        ({"AAPL": 1.0, "MSFT": 1.0}, 0.0, 1.0),
        ({"AAPL": 1.0, "MSFT": 3.0}, 0.0, 1.0),
        ({"AAPL": 6.0, "MSFT": 3.0, "NVDA": 1.0}, 0.1, 1.0),
        ({"A": 30.0, "B": 20.0, "C": 20.0, "D": 15.0, "E": 15.0}, 0.0, 0.25),
        ({"A": 30.0, "B": 20.0, "C": 20.0, "D": 15.0, "E": 15.0}, 0.1, 0.25),
        ({"A": 40.0, "B": 30.0, "C": 20.0, "D": 6.0, "E": 4.0}, 0.0, 0.25),
        ({"A": 100.0, "B": 1.0}, 0.1, 0.95),
        ({"A": 3.0, "B": 3.0, "C": 3.0}, 0.0, 1.0),
        ({"A": math.pi, "B": math.e, "C": math.sqrt(2)}, 0.17, 0.5),
    ],
)
def test_ordinary_scores_are_unchanged_by_the_scaling(
    scores: dict[str, float], cash: float, cap: float
) -> None:
    """The unscaled formula and the scaled one agree to the last bit or so.

    ``pinned`` below is the pre-scaling arithmetic verbatim: the raw total, no
    scaling.  Every ordinary case here -- the documented five-name one included --
    lands within ``1e-15`` of it, four orders of magnitude inside the contract's own
    tolerance, so no existing behaviour changed.
    """
    budget = 1.0 - cash
    free = dict(scores)
    pinned: dict[str, float] = {}
    while free:
        total = sum(free.values())
        allocation = {symbol: budget * score / total for symbol, score in free.items()}
        over = [symbol for symbol, weight in allocation.items() if weight > cap]
        if not over:
            pinned.update(allocation)
            break
        for symbol in over:
            pinned[symbol] = cap
            budget -= cap
            del free[symbol]

    scaled = rule_module._capped_weights(scores, 1.0 - cash, cap)

    assert scaled.keys() == pinned.keys()
    assert scaled == pytest.approx(pinned, rel=1e-15, abs=0.0)


def test_the_documented_five_name_weights_are_still_the_documented_ones() -> None:
    """The canonical example is unchanged, digit for digit, after the fix."""
    weights = rule_module._capped_weights(
        {"A": 30.0, "B": 20.0, "C": 20.0, "D": 15.0, "E": 15.0}, 1.0, 0.25
    )

    assert weights == pytest.approx(
        {"A": 0.25, "B": 3 / 14, "C": 3 / 14, "D": 9 / 56, "E": 9 / 56}, rel=1e-15
    )
    assert round(sum(weights.values()), 12) == 1.0


# ---------------------------------------------------------------------------
# Infeasible constraints
# ---------------------------------------------------------------------------
def test_constraints_the_selection_cannot_satisfy_are_refused(
    constructor: CappedScoreWeightedConstructor,
) -> None:
    """Three names at a 25% cap can carry 75% of the account, not the 90% required."""
    context = make_context(
        {"AAPL": 1.0, "MSFT": 1.0, "NVDA": 1.0},
        ("AAPL", "MSFT", "NVDA"),
        cash_weight=0.1,
        max_single_weight=0.25,
    )
    before = context.strategy_result.frame

    with pytest.raises(PortfolioConstructionError) as excinfo:
        constructor.construct(context)

    message = str(excinfo.value)
    assert "3 selected symbol(s)" in message
    assert "0.9" in message
    assert "0.25" in message
    assert "0.75" in message
    assert excinfo.value.problems
    assert isinstance(excinfo.value, PortfolioError)
    assert not isinstance(excinfo.value, PortfolioContractError)
    assert context.strategy_result.frame.equals(before)


def test_the_refusal_says_what_it_will_not_do(
    constructor: CappedScoreWeightedConstructor,
) -> None:
    """The cash target, the cap and the selection are not quietly rewritten."""
    context = make_context(
        {"AAPL": 1.0, "MSFT": 1.0, "NVDA": 1.0},
        ("AAPL", "MSFT", "NVDA"),
        cash_weight=0.1,
        max_single_weight=0.25,
    )

    with pytest.raises(PortfolioConstructionError) as excinfo:
        constructor.construct(context)

    problems = "\n".join(excinfo.value.problems)
    assert "cash target is not raised" in problems
    assert "cap is not widened" in problems
    assert "no selected symbol is dropped" in problems


def test_one_name_with_a_cap_below_the_budget_is_refused(
    constructor: CappedScoreWeightedConstructor,
) -> None:
    """A 70% cap cannot carry an 80% budget, however few names are selected."""
    context = make_context({"AAPL": 1.0}, ("AAPL",), cash_weight=0.2, max_single_weight=0.7)

    with pytest.raises(PortfolioConstructionError) as excinfo:
        constructor.construct(context)

    assert "1 selected symbol(s)" in str(excinfo.value)
    assert "0.7" in str(excinfo.value)
    assert "0.8" in str(excinfo.value)


def test_a_selection_that_just_fits_is_built(
    constructor: CappedScoreWeightedConstructor,
) -> None:
    """Four names at 25% cover the 90% budget -- the loosest fit that still builds."""
    symbols = ("AAPL", "MSFT", "NVDA", "TSLA")
    context = make_context(
        {"AAPL": 100.0, "MSFT": 1.0, "NVDA": 1.0, "TSLA": 1.0},
        symbols,
        symbols=symbols,
        cash_weight=0.1,
        max_single_weight=0.25,
    )

    target = constructor.construct(context)

    assert target.invested_weight == pytest.approx(0.9, abs=TOLERANCE)
    assert max(weights_of(target).values()) <= 0.25
    assert target.cash_weight == 0.1


def test_an_exact_fit_on_the_capacity_boundary_is_built(
    constructor: CappedScoreWeightedConstructor,
) -> None:
    """Three names at 25% and a 25% cash target: capacity equals the budget exactly."""
    context = make_context(
        {"AAPL": 100.0, "MSFT": 1.0, "NVDA": 1.0},
        ("AAPL", "MSFT", "NVDA"),
        cash_weight=0.25,
        max_single_weight=0.25,
    )

    target = constructor.construct(context)

    assert weights_of(target) == pytest.approx(
        {"AAPL": 0.25, "MSFT": 0.25, "NVDA": 0.25}, abs=TOLERANCE
    )
    assert target.cash_weight == 0.25


# ---------------------------------------------------------------------------
# An unusable score refuses the whole construction
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad_score", [0.0, -1.0, -0.5, -1e-9])
def test_a_non_positive_selected_score_is_refused(
    constructor: CappedScoreWeightedConstructor, bad_score: float
) -> None:
    """A score of zero or below has no positive share to give a long-only holding."""
    context = make_context({"AAPL": bad_score, "MSFT": 1.0}, ("AAPL", "MSFT"))

    with pytest.raises(PortfolioConstructionError) as excinfo:
        constructor.construct(context)

    message = str(excinfo.value)
    assert "AAPL" in message
    assert repr(bad_score) in message
    assert "positive" in message
    assert not isinstance(excinfo.value, PortfolioContractError)


@pytest.mark.parametrize(
    ("bad_score", "rendered"),
    [
        (math.nan, "nan"),
        (math.inf, "inf"),
        (-math.inf, "-inf"),
    ],
)
def test_a_non_finite_selected_score_is_refused(
    constructor: CappedScoreWeightedConstructor, bad_score: float, rendered: str
) -> None:
    """``NaN`` and ``±inf`` cannot be turned into a share of the account."""
    context = make_context({"AAPL": 2.0, "MSFT": 1.0}, ("AAPL", "MSFT"))
    corrupt_score(context, "MSFT", bad_score)

    with pytest.raises(PortfolioConstructionError) as excinfo:
        constructor.construct(context)

    message = str(excinfo.value)
    assert "MSFT" in message
    assert rendered in message
    assert "non-finite" in "\n".join(excinfo.value.problems)


def test_every_unusable_score_is_reported_at_once(
    constructor: CappedScoreWeightedConstructor,
) -> None:
    """One construction, one error, three offending symbols and the closing note."""
    context = make_context(
        {"AAPL": 0.0, "MSFT": -2.0, "NVDA": 3.0, "TSLA": 4.0},
        ("AAPL", "MSFT", "NVDA", "TSLA"),
    )
    corrupt_score(context, "NVDA", math.nan)

    with pytest.raises(PortfolioConstructionError) as excinfo:
        constructor.construct(context)

    problems = excinfo.value.problems
    assert len(problems) == 4
    reported = "\n".join(problems)
    for symbol in ("AAPL", "MSFT", "NVDA"):
        assert symbol in reported, symbol
    assert "equal weight" in problems[-1]


def test_the_refusal_explains_that_nothing_was_repaired(
    constructor: CappedScoreWeightedConstructor,
) -> None:
    """No shift, no absolute value, no clipping, no silent equal weight."""
    context = make_context({"AAPL": -3.0, "MSFT": 1.0}, ("AAPL", "MSFT"))

    with pytest.raises(PortfolioConstructionError) as excinfo:
        constructor.construct(context)

    note = excinfo.value.problems[-1]
    assert "no constant is added" in note
    assert "absolute-valued" in note
    assert "clipped to zero" in note


def test_an_anomalous_unselected_score_is_ignored(
    constructor: CappedScoreWeightedConstructor,
) -> None:
    """Score only matters for the symbols the strategy selected."""
    context = make_context({"AAPL": 1.0, "MSFT": 1.0}, ("AAPL", "MSFT"), cash_weight=0.2)
    corrupt_score(context, "NVDA", math.nan)
    corrupt_score(context, "TSLA", -17.0)

    target = constructor.construct(context)

    assert target.target_symbols == ("AAPL", "MSFT")
    assert target.cash_weight == 0.2


def test_a_negative_score_on_an_unselected_row_is_legal(
    constructor: CappedScoreWeightedConstructor,
) -> None:
    """A score of -5 on a row the strategy did not pick changes nothing at all."""
    scores = {"AAPL": 1.0, "MSFT": 3.0, "NVDA": -5.0}
    context = make_context(scores, ("AAPL", "MSFT"))
    without = make_context({"AAPL": 1.0, "MSFT": 3.0}, ("AAPL", "MSFT"))

    assert weights_of(constructor.construct(context)) == weights_of(constructor.construct(without))


def test_a_score_whose_share_rounds_to_zero_is_refused(
    constructor: CappedScoreWeightedConstructor,
) -> None:
    """A positive score can still be too small to be a holding, and that is refused.

    ``5e-324`` beside ``1e300`` is positive and finite, so the strategy contract
    accepts it -- but dividing it out gives a weight of exactly ``0.0``, which this
    layer cannot hold.  The rule says so in its own words instead of letting a zero
    position reach the contract.
    """
    context = make_context({"AAPL": 5e-324, "MSFT": 1e300}, ("AAPL", "MSFT"))

    with pytest.raises(PortfolioConstructionError) as excinfo:
        constructor.construct(context)

    message = str(excinfo.value)
    assert "AAPL" in message
    assert "5e-324" in message
    assert not isinstance(excinfo.value, PortfolioContractError)


# ---------------------------------------------------------------------------
# An empty selection is an all-cash target
# ---------------------------------------------------------------------------
def test_an_empty_selection_returns_an_all_cash_target(
    constructor: CappedScoreWeightedConstructor,
) -> None:
    """No stock to weight: the one legal portfolio is entirely in cash."""
    context = make_context({"AAPL": 1.0}, ())

    target = constructor.construct(context)

    assert target.positions == ()
    assert target.cash_weight == 1.0
    assert target.invested_weight == 0.0
    assert target.target_symbols == ()


def test_an_empty_selection_ignores_the_cash_constraint(
    constructor: CappedScoreWeightedConstructor,
) -> None:
    """A 0% cash target and a 0.1% cap are both irrelevant when nothing is selected."""
    context = make_context({"AAPL": 1.0}, (), cash_weight=0.0, max_single_weight=0.001)

    target = constructor.construct(context)

    assert target.positions == ()
    assert target.cash_weight == 1.0
    assert target.constraints is context.constraints


def test_an_empty_selection_keeps_the_source_chain(
    constructor: CappedScoreWeightedConstructor,
) -> None:
    context = make_context({"AAPL": 1.0}, ())

    target = constructor.construct(context)

    assert target.source_result is context.strategy_result
    assert target.construction_spec == constructor.spec
    assert target.as_of_date == context.as_of_date
    assert target.strategy_spec is context.strategy_spec
    assert target.universe is context.universe
    assert "CASH" not in target.target_symbols


# ---------------------------------------------------------------------------
# Provenance and the order of the positions
# ---------------------------------------------------------------------------
def test_the_target_points_at_the_context_objects(
    constructor: CappedScoreWeightedConstructor,
) -> None:
    """The result and the constraints are shared, not copied: no second truth."""
    context = make_context(
        {"AAPL": 2.0, "MSFT": 1.0}, ("AAPL", "MSFT"), cash_weight=0.1, max_single_weight=0.6
    )

    target = constructor.construct(context)

    assert target.source_result is context.strategy_result
    assert target.constraints is context.constraints
    assert target.construction_spec == constructor.spec
    assert target.as_of_date == context.as_of_date


def test_the_target_keeps_the_strategy_factor_and_universe_provenance(
    constructor: CappedScoreWeightedConstructor,
) -> None:
    """The exact strategy, the exact factor versions and the universe survive."""
    context = make_context({"AAPL": 2.0, "MSFT": 1.0}, ("AAPL", "MSFT"))

    target = constructor.construct(context)

    assert target.strategy_spec is context.strategy_result.strategy_spec
    assert target.strategy_spec.name == "top_n_momentum"
    assert target.factor_requirements == (MOMENTUM,)
    assert target.factor_requirements is context.strategy_result.factor_requirements
    assert target.universe is context.strategy_result.universe
    assert target.universe.source == "config/universe_seed.csv"


def test_positions_follow_the_selection_order_not_the_score_order(
    constructor: CappedScoreWeightedConstructor,
) -> None:
    """The weights are re-sorted by nothing: universe order is the output order."""
    scores = {"AAPL": 1.0, "MSFT": 5.0, "NVDA": 3.0}
    context = make_context(scores, ("AAPL", "MSFT", "NVDA"), cash_weight=0.1)

    target = constructor.construct(context)

    assert target.target_symbols == ("AAPL", "MSFT", "NVDA")
    assert target.target_symbols == context.selected_symbols
    assert list(weights_of(target)) == ["AAPL", "MSFT", "NVDA"]
    # The score order would have been MSFT, NVDA, AAPL -- it is deliberately not used.
    assert target.target_symbols != ("MSFT", "NVDA", "AAPL")


def test_every_selected_symbol_gets_exactly_one_position(
    constructor: CappedScoreWeightedConstructor,
) -> None:
    symbols = ("AAPL", "MSFT", "NVDA", "TSLA", "AMZN")
    context = make_context(
        dict.fromkeys(symbols, 2.0),
        symbols,
        symbols=symbols,
        cash_weight=0.05,
        max_single_weight=0.25,
    )

    target = constructor.construct(context)

    assert sorted(target.target_symbols) == sorted(symbols)
    assert len(target.positions) == len(set(target.target_symbols))
    assert all(isinstance(position, PortfolioPosition) for position in target.positions)


# ---------------------------------------------------------------------------
# Purity: nothing in, nothing out
# ---------------------------------------------------------------------------
def test_the_builder_does_not_touch_its_inputs(
    constructor: CappedScoreWeightedConstructor,
) -> None:
    """The frame, the state, the constraints and the context all survive intact."""
    context = make_context(
        {"AAPL": 4.0, "MSFT": 1.0}, ("AAPL", "MSFT"), cash_weight=0.1, max_single_weight=0.6
    )
    frame_before = context.strategy_result.frame
    columns_before = list(frame_before.columns)
    state_before = context.current_state
    constraints_before = context.constraints
    selected_before = context.selected_symbols

    constructor.construct(context)

    assert context.strategy_result.frame.equals(frame_before)
    assert list(context.strategy_result.frame.columns) == columns_before
    assert context.current_state == state_before
    assert context.current_state is state_before
    assert context.constraints == constraints_before
    assert context.constraints is constraints_before
    assert context.selected_symbols == selected_before


def test_no_column_is_added_to_the_strategy_result(
    constructor: CappedScoreWeightedConstructor,
) -> None:
    """A weight never becomes a column of the result: it is a separate object."""
    context = make_context({"AAPL": 2.0, "MSFT": 1.0}, ("AAPL", "MSFT"))

    constructor.construct(context)

    assert tuple(context.strategy_result.frame.columns) == STRATEGY_RESULT_COLUMNS


def test_two_calls_return_equal_but_independent_targets(
    constructor: CappedScoreWeightedConstructor,
) -> None:
    """Same context, same numbers -- and two objects that cannot edit each other."""
    context = make_context({"AAPL": 3.0, "MSFT": 1.0}, ("AAPL", "MSFT"), cash_weight=0.1)

    first = constructor.construct(context)
    second = constructor.construct(context)

    assert first == second
    assert first is not second
    assert first.positions == second.positions
    assert first.positions is not second.positions

    first.target_weights_by_symbol["AAPL"] = 0.99
    assert second.target_weights_by_symbol["AAPL"] != 0.99


def test_the_current_state_plays_no_part_in_the_weights(
    constructor: CappedScoreWeightedConstructor,
) -> None:
    """The rule weights the selection from scratch; the state is carried, not read."""
    scores = {"AAPL": 3.0, "MSFT": 1.0}
    from_cash = make_context(scores, ("AAPL", "MSFT"), cash_weight=0.1)
    from_holdings = make_context(
        scores,
        ("AAPL", "MSFT"),
        cash_weight=0.1,
        current_state=make_state(
            (PortfolioPosition("NVDA", 0.7), PortfolioPosition("TSLA", 0.2)), cash_weight=0.1
        ),
    )

    assert weights_of(constructor.construct(from_holdings)) == weights_of(
        constructor.construct(from_cash)
    )


# ---------------------------------------------------------------------------
# Errors and the package surface
# ---------------------------------------------------------------------------
def test_the_construction_error_is_a_portfolio_error() -> None:
    assert issubclass(PortfolioConstructionError, PortfolioError)
    assert not issubclass(PortfolioConstructionError, PortfolioContractError)
    assert not issubclass(PortfolioContractError, PortfolioConstructionError)


def test_the_construction_error_collects_its_problems() -> None:
    without = PortfolioConstructionError("nothing to build")
    assert without.problems == []
    assert str(without) == "nothing to build"

    with_problems = PortfolioConstructionError("nothing to build", ["a", "b"])
    assert with_problems.problems == ["a", "b"]
    assert str(with_problems) == "nothing to build\n  - a\n  - b"


def test_the_contract_error_keeps_its_meaning() -> None:
    """The new error does not take the contract error's place."""
    with pytest.raises(PortfolioContractError):
        PortfolioPosition("AAPL", 0.0)

    with pytest.raises(PortfolioContractError) as excinfo:
        PortfolioConstraints(cash_weight=1.5, max_single_weight=1.0)
    assert isinstance(excinfo.value, PortfolioError)
    assert not isinstance(excinfo.value, PortfolioConstructionError)


def test_the_package_root_exports_the_new_error() -> None:
    assert portfolio_sys.PortfolioConstructionError is PortfolioConstructionError
    assert "PortfolioConstructionError" in portfolio_sys.__all__
    assert sorted(portfolio_sys.__all__) == [
        "CONSTRUCTION_NAME_PATTERN",
        "PortfolioConstraints",
        "PortfolioConstructionError",
        "PortfolioConstructionSpec",
        "PortfolioConstructor",
        "PortfolioConstructorRegistry",
        "PortfolioContext",
        "PortfolioContractError",
        "PortfolioError",
        "PortfolioPosition",
        "PortfolioRegistryError",
        "PortfolioState",
        "PortfolioTransition",
        "PortfolioWeightChange",
        "TargetPortfolio",
        "WEIGHT_TOLERANCE",
        "__version__",
        "build_portfolio_transition",
    ]
    for name in portfolio_sys.__all__:
        assert hasattr(portfolio_sys, name), name


def test_the_rule_is_exported_from_the_constructors_package() -> None:
    assert constructors_package.CappedScoreWeightedConstructor is CappedScoreWeightedConstructor
    assert constructors_package.DEFAULT_VERSION == DEFAULT_VERSION
    assert sorted(constructors_package.__all__) == [
        "CappedScoreWeightedConstructor",
        "DEFAULT_VERSION",
    ]
    assert rule_module.CappedScoreWeightedConstructor is CappedScoreWeightedConstructor


def test_the_root_does_not_re_export_the_rule() -> None:
    """Like :mod:`strategy_sys`, the root stays the contract and never becomes a registry."""
    for absent in (
        "CappedScoreWeightedConstructor",
        "EqualWeightConstructor",
        "ScoreWeightedConstructor",
        "PortfolioRegistry",
        "PortfolioRunner",
        "run_portfolio",
    ):
        assert not hasattr(portfolio_sys, absent), absent


def test_every_public_object_has_a_docstring() -> None:
    for name in constructors_package.__all__:
        assert inspect.getdoc(getattr(constructors_package, name)), name

    for member in (
        CappedScoreWeightedConstructor.spec,
        CappedScoreWeightedConstructor.construct,
    ):
        assert inspect.getdoc(member), member

    assert inspect.getdoc(rule_module), "the rule module has a module docstring"
    assert inspect.getdoc(PortfolioConstructionError), "the new error is documented"


def test_the_rule_module_is_documented_as_the_documented_example() -> None:
    """The docstring arithmetic and the code agree on the five-name case."""
    weights = rule_module._capped_weights(
        {"A": 30.0, "B": 20.0, "C": 20.0, "D": 15.0, "E": 15.0},
        investable_weight=1.0,
        cap=0.25,
    )

    assert weights["A"] == 0.25
    assert weights["B"] == pytest.approx(0.75 * 20 / 70, abs=TOLERANCE)
    assert sum(weights.values()) == pytest.approx(1.0, abs=TOLERANCE)


# ---------------------------------------------------------------------------
# Architectural boundary (read off the source, not off the run)
# ---------------------------------------------------------------------------
def rule_modules() -> list[Path]:
    """The rule's modules: the sub-package and the rule itself."""
    package = Path(portfolio_sys.__file__).parent / "constructors"
    return [package / "__init__.py", package / "capped_score_weighted.py"]


#: Modules the rule may import: the standard library it needs and the portfolio
#: modules it is built on -- nothing else.
ALLOWED_IMPORTS: frozenset[str] = frozenset(
    {
        "__future__",
        "collections.abc",
        "math",
        "portfolio_sys.constructors.capped_score_weighted",
        "portfolio_sys.contracts",
        "portfolio_sys.errors",
    }
)

#: Text that must never appear in the rule's *code*.  Docstrings are blanked
#: first, so the prose may still name what the layer deliberately avoids.
FORBIDDEN_TEXT: tuple[str, ...] = (
    "import pandas",
    "import numpy",
    "import pyarrow",
    "import polars",
    "import duckdb",
    "import yfinance",
    "import requests",
    "import httpx",
    "import socket",
    "import exchange_calendars",
    "data_sys",
    "factor_sys",
    "strategy_sys",
    "read_parquet",
    "to_parquet",
    "read_csv",
    "urlopen",
    "open(",
)

#: Identifiers the rule must never use: a price, a share count, an order, a fill or
#: a backtest is another layer's vocabulary.
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

#: Names the rule may never reach for: the contract's private helpers and the
#: frozen-dataclass escape hatch.
FORBIDDEN_NAMES: frozenset[str] = frozenset(
    {
        "_checked_positions",
        "_completion_problems",
        "_describe",
        "_finite_number",
        "_session_date",
        "__post_init__",
        "__setattr__",
    }
)


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


def code_without_docstrings(path: Path) -> str:
    """``path``'s source with every docstring blanked out."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    lines = source.splitlines(keepends=True)
    holders = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    for node in ast.walk(tree):
        if not isinstance(node, holders) or not node.body:
            continue
        first = node.body[0]
        if not isinstance(first, ast.Expr):
            continue
        if not isinstance(first.value, ast.Constant) or not isinstance(first.value.value, str):
            continue
        for index in range(first.lineno - 1, first.end_lineno):
            lines[index] = "\n"
    return "".join(lines)


def test_the_rule_imports_only_the_standard_library_and_its_own_contract() -> None:
    for module in rule_modules():
        imported = imported_modules(module)
        assert imported <= ALLOWED_IMPORTS, (module.name, sorted(imported - ALLOWED_IMPORTS))


def test_the_rule_never_names_a_data_factor_or_strategy_module() -> None:
    for module in rule_modules():
        code = code_without_docstrings(module)
        for forbidden in FORBIDDEN_TEXT:
            assert forbidden not in code, (module.name, forbidden)


def test_the_rule_does_not_open_a_file_print_or_evaluate() -> None:
    banned = {"open", "print", "input", "eval", "exec", "compile", "__import__"}

    for module in rule_modules():
        tree = ast.parse(module.read_text(encoding="utf-8"))
        called = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert called & banned == set(), (module.name, sorted(called & banned))


def test_the_rule_never_names_a_price_a_share_or_a_backtest() -> None:
    names_gathered: dict[str, set[str]] = {}
    for module in rule_modules():
        tree = ast.parse(module.read_text(encoding="utf-8"))
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        names |= {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        names |= {node.arg for node in ast.walk(tree) if isinstance(node, ast.arg)}
        names |= {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        }
        names_gathered[module.name] = names

    for name, names in names_gathered.items():
        offenders = sorted(names & FORBIDDEN_IDENTIFIERS)
        assert offenders == [], (name, offenders)


def test_the_rule_does_not_re_implement_the_contract() -> None:
    """No second ``__post_init__``, no private helper borrowed, no frozen-dataclass hack."""
    for module in rule_modules():
        tree = ast.parse(module.read_text(encoding="utf-8"))
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        names |= {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        names |= {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        }
        offenders = sorted(names & FORBIDDEN_NAMES)
        assert offenders == [], (module.name, offenders)


def test_the_rule_defines_exactly_one_class() -> None:
    tree = ast.parse(rule_modules()[1].read_text(encoding="utf-8"))
    classes = [node.name for node in tree.body if isinstance(node, ast.ClassDef)]

    assert classes == ["CappedScoreWeightedConstructor"]
    assert issubclass(CappedScoreWeightedConstructor, PortfolioConstructor)


def test_the_rule_answers_in_exactly_two_places() -> None:
    """One target for an empty selection, one for a weighted one -- no third path."""
    tree = ast.parse(rule_modules()[1].read_text(encoding="utf-8"))
    built = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "TargetPortfolio"
    ]

    assert len(built) == 2


def test_the_contract_modules_do_not_know_the_rule() -> None:
    """Adding a rule never edits the objects every rule agrees on."""
    package = Path(portfolio_sys.__file__).parent
    for name in ("__init__.py", "contracts.py", "errors.py"):
        code = code_without_docstrings(package / name)
        assert "CappedScoreWeightedConstructor" not in code, name
        assert "_capped_weights" not in code, name
