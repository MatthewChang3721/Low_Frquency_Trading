"""Tests for :mod:`backtest_sys.execution` -- filling an order at the next open.

Everything here is in memory and offline: no bar, no price file, no Parquet
dataset, no DuckDB connection, no trading calendar and no network.  An execution
model is *handed* the open price it needs, so every expectation below is a
hand-written number against a hand-written price -- which is exactly what makes the
friction rule auditable: the price a trade gets and the money it moves are checked
against arithmetic done in the test, not against another part of the system.

Covered in this order:

* the interface: one abstract method, its keyword-only request, and its return type,
* the configuration: the 5 bps / zero-commission defaults, the frozen hashable
  value, every illegal slippage and commission (``bool``, string, ``None``, ``NaN``,
  ``inf``, negative, ``10_000`` bps), the closed zero and open ``10_000`` boundary,
  both bad parameters named in one error, and ``slippage_fraction`` as a property,
* the arithmetic: a buy, a sell, zero slippage, what the commission moves and what
  it does not, several orders at once, and the two overflows a fill must refuse,
* the request: the empty tuple, a non-tuple, a non-order, a repeated id, one symbol
  on two orders, both dates agreeing with the event, the id order, a foreign event,
  and one error that names every problem of a broken request,
* the price table: every symbol priced, a missing symbol named, an illegal price,
  an illegal key stamped with the project's own ticker rule (``"???"``, lower case,
  punctuation, padding), a non-mapping, extra symbols -- legal ones tolerated,
  illegal ones refused -- and the proof that nothing is modified,
* the wrapping: a ``Fill`` the contract refuses arrives as ``ExecutionError`` with
  its cause kept, so no caller sees a bare contract error in the middle of a fill,
* the package surface and the module boundary: no reader, no frame, no file, no
  network, no account and no private import.
"""

from __future__ import annotations

import ast
import inspect
import math
import re
import sys
from dataclasses import FrozenInstanceError, fields
from datetime import date
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest

import backtest_sys
import backtest_sys.execution as execution_module
from backtest_sys import (
    ExecutionError,
    ExecutionModel,
    Fill,
    FixedBpsExecutionModel,
    OrderIntent,
    RebalanceEvent,
)
from backtest_sys.errors import BacktestContractError, BacktestError
from data_sys.schema import SYMBOL_PATTERN

#: The module under test, as a file (for the boundary tests below).
SOURCE: Path = Path(execution_module.__file__).resolve()

#: The decision / execution pair every fixture trades in: Thursday's close decides,
#: Friday's open fills -- the project's default timeline.
DECISION: date = date(2024, 1, 4)
EXECUTION: date = date(2024, 1, 5)

#: A sentinel for "the factory was not told otherwise", so a test can still pass
#: ``None`` on purpose and mean it.
_UNSET: Any = object()


# ---------------------------------------------------------------------------
# In-memory fixtures (no disk, no network, no calendar, no prices)
# ---------------------------------------------------------------------------
def make_order(
    *,
    order_id: object = "rb-0001",
    decision: object = DECISION,
    scheduled: object = EXECUTION,
    symbol: object = "AAPL",
    side: object = "BUY",
    quantity: object = 10.0,
) -> OrderIntent:
    """An :class:`OrderIntent` for the fixtures: a buy of 10 AAPL at Friday's open."""
    return OrderIntent(  # type: ignore[arg-type]
        order_id=order_id,
        decision_date=decision,
        scheduled_fill_date=scheduled,
        symbol=symbol,
        side=side,
        quantity=quantity,
    )


def make_event(*, decision: object = DECISION, execution: object = EXECUTION) -> RebalanceEvent:
    """The :class:`RebalanceEvent` the fixture orders belong to."""
    return RebalanceEvent(decision, execution)  # type: ignore[arg-type]


def make_model(
    *, slippage_bps: object = 5.0, commission_per_order: object = 0.0
) -> FixedBpsExecutionModel:
    """A :class:`FixedBpsExecutionModel` with the fixtures' friction."""
    return FixedBpsExecutionModel(  # type: ignore[arg-type]
        slippage_bps=slippage_bps,
        commission_per_order=commission_per_order,
    )


def fill_one(
    order: object = _UNSET,
    *,
    open_prices: object = _UNSET,
    event: object = _UNSET,
    model: object = _UNSET,
) -> Fill:
    """The single fill of ``order``, priced from ``open_prices`` (AAPL at 100.00)."""
    trade = make_order() if order is _UNSET else order
    prices = {"AAPL": 100.0} if open_prices is _UNSET else open_prices
    session = make_event() if event is _UNSET else event
    friction = make_model() if model is _UNSET else model
    fills = friction.fill_orders(  # type: ignore[arg-type]
        event=session,
        orders=(trade,),
        open_prices=prices,
    )
    assert len(fills) == 1
    return fills[0]


# ---------------------------------------------------------------------------
# The interface: one method, three keyword-only arguments, one return type
# ---------------------------------------------------------------------------
def test_the_interface_is_abstract_and_declares_one_method() -> None:
    assert inspect.isabstract(ExecutionModel)
    assert ExecutionModel.__abstractmethods__ == frozenset({"fill_orders"})

    with pytest.raises(TypeError):
        ExecutionModel()  # type: ignore[abstract]


def test_the_interface_has_exactly_one_abstract_method_and_no_state() -> None:
    abstract = {
        name
        for name, value in vars(ExecutionModel).items()
        if getattr(value, "__isabstractmethod__", False)
    }

    assert abstract == {"fill_orders"}
    assert ExecutionModel.__slots__ == ()


def test_the_request_is_keyword_only_and_typed() -> None:
    signature = inspect.signature(ExecutionModel.fill_orders)
    parameters = signature.parameters

    assert list(parameters) == ["self", "event", "orders", "open_prices"]
    for name in ("event", "orders", "open_prices"):
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY, name
    assert parameters["event"].annotation == "RebalanceEvent"
    assert parameters["orders"].annotation == "tuple[OrderIntent, ...]"
    assert parameters["open_prices"].annotation == "Mapping[str, float]"
    assert signature.return_annotation == "tuple[Fill, ...]"


def test_the_reference_model_implements_the_interface() -> None:
    assert issubclass(FixedBpsExecutionModel, ExecutionModel)
    assert not inspect.isabstract(FixedBpsExecutionModel)
    assert FixedBpsExecutionModel.__abstractmethods__ == frozenset()


# ---------------------------------------------------------------------------
# The configuration: 5 bps a side, no commission, and nothing else accepted
# ---------------------------------------------------------------------------
def test_the_defaults_are_five_basis_points_and_no_commission() -> None:
    model = FixedBpsExecutionModel()

    assert model.slippage_bps == 5.0
    assert model.commission_per_order == 0.0
    assert model.slippage_fraction == 0.0005


def test_the_model_is_a_frozen_hashable_value_with_a_readable_repr() -> None:
    model = make_model()

    assert repr(model) == "FixedBpsExecutionModel(slippage_bps=5.0, commission_per_order=0.0)"
    assert model == make_model()
    assert model != make_model(slippage_bps=10.0)
    assert hash(model) == hash(make_model())
    assert {model: "five bps"}[make_model()] == "five bps"

    with pytest.raises(FrozenInstanceError):
        model.slippage_bps = 1.0  # type: ignore[misc]


def test_the_model_declares_exactly_two_fields() -> None:
    assert [field.name for field in fields(FixedBpsExecutionModel)] == [
        "slippage_bps",
        "commission_per_order",
    ]


@pytest.mark.parametrize(
    "slippage",
    [
        -0.01,
        -1.0,
        -1e9,
        10_000.0,
        10_000.5,
        1e12,
        math.nan,
        math.inf,
        -math.inf,
        True,
        False,
        "5",
        "5.0",
        None,
        [5.0],
    ],
)
def test_an_illegal_slippage_is_refused(slippage: object) -> None:
    with pytest.raises(BacktestContractError) as excinfo:
        make_model(slippage_bps=slippage)

    assert "slippage_bps" in str(excinfo.value)
    assert len(excinfo.value.problems) == 1


@pytest.mark.parametrize("slippage", [0, 0.0, 1e-9, 5.0, 100.0, 999.9999, 9_999.999])
def test_the_slippage_range_is_closed_at_zero_and_open_at_ten_thousand(slippage: float) -> None:
    model = make_model(slippage_bps=slippage)

    assert model.slippage_bps == float(slippage)
    assert model.slippage_fraction == float(slippage) / 10_000.0
    assert 0.0 <= model.slippage_fraction < 1.0


def test_ten_thousand_basis_points_is_refused_as_the_sell_price_boundary() -> None:
    with pytest.raises(BacktestContractError) as excinfo:
        make_model(slippage_bps=10_000.0)

    assert "10_000" in str(excinfo.value)
    assert "zero" in str(excinfo.value)


@pytest.mark.parametrize(
    "cost",
    [-0.01, -1.0, -1e9, math.nan, math.inf, -math.inf, True, False, "0", None, [0.0]],
)
def test_an_illegal_commission_is_refused(cost: object) -> None:
    with pytest.raises(BacktestContractError) as excinfo:
        make_model(commission_per_order=cost)

    assert "commission_per_order" in str(excinfo.value)
    assert len(excinfo.value.problems) == 1


@pytest.mark.parametrize("cost", [0, 0.0, 0.005, 1.0, 35.0])
def test_the_commission_is_zero_or_any_flat_amount(cost: float) -> None:
    model = make_model(commission_per_order=cost)

    assert model.commission_per_order == float(cost)


def test_one_error_names_both_bad_parameters() -> None:
    with pytest.raises(BacktestContractError) as excinfo:
        make_model(slippage_bps=-1.0, commission_per_order=-2.0)

    problems = excinfo.value.problems
    assert len(problems) == 2
    assert any("slippage_bps" in problem for problem in problems)
    assert any("commission_per_order" in problem for problem in problems)


def test_the_slippage_fraction_is_a_property_not_a_field() -> None:
    assert isinstance(FixedBpsExecutionModel.slippage_fraction, property)
    assert "slippage_fraction" not in FixedBpsExecutionModel.__dataclass_fields__

    model = make_model(slippage_bps=25.0)
    assert model.slippage_fraction == 0.0025

    with pytest.raises(AttributeError):
        model.slippage_fraction = 1.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# The arithmetic: the open, moved by the slippage, less the commission
# ---------------------------------------------------------------------------
def test_a_buy_fills_at_the_open_plus_the_slippage() -> None:
    order = make_order()

    fill = fill_one(order)

    assert fill.order is order
    assert fill.fill_date == EXECUTION
    assert fill.quantity == order.quantity
    assert fill.transaction_cost == 0.0
    assert fill.fill_price == 100.05
    assert fill.gross_notional == 1000.5
    assert fill.cash_change == -1000.5


def test_a_sell_fills_at_the_open_minus_the_slippage() -> None:
    fill = fill_one(make_order(side="SELL"))

    assert fill.fill_price == 99.95
    assert fill.gross_notional == 999.5
    assert fill.cash_change == 999.5


@pytest.mark.parametrize("side", ["BUY", "SELL"])
def test_zero_slippage_fills_at_the_raw_open(side: str) -> None:
    fill = fill_one(make_order(side=side), model=make_model(slippage_bps=0.0))

    assert fill.fill_price == 100.0
    assert fill.gross_notional == 1000.0
    assert fill.cash_change == (-1000.0 if side == "BUY" else 1000.0)


def test_the_commission_moves_the_cash_and_never_the_price() -> None:
    model = make_model(commission_per_order=1.0)

    buy = fill_one(make_order(side="BUY"), model=model)
    sell = fill_one(make_order(side="SELL"), model=model)

    assert (buy.fill_price, sell.fill_price) == (100.05, 99.95)
    assert (buy.transaction_cost, sell.transaction_cost) == (1.0, 1.0)
    assert buy.cash_change == -1001.5
    assert sell.cash_change == 998.5


def test_each_order_fills_in_turn_and_pays_its_own_commission() -> None:
    model = make_model(commission_per_order=0.5)
    buy = make_order(order_id="rb-0001", symbol="AAPL", side="BUY", quantity=10.0)
    sell = make_order(order_id="rb-0002", symbol="MSFT", side="SELL", quantity=4.0)

    fills = model.fill_orders(
        event=make_event(),
        orders=(buy, sell),
        open_prices={"AAPL": 100.0, "MSFT": 50.0},
    )

    assert isinstance(fills, tuple)
    assert len(fills) == 2
    assert all(isinstance(fill, Fill) for fill in fills)
    assert [fill.order for fill in fills] == [buy, sell]
    assert [fill.fill_date for fill in fills] == [EXECUTION, EXECUTION]
    assert [fill.transaction_cost for fill in fills] == [0.5, 0.5]

    assert fills[0].fill_price == 100.05
    assert fills[0].cash_change == -(fills[0].gross_notional + 0.5)
    assert fills[1].fill_price == pytest.approx(49.975)
    assert fills[1].cash_change == pytest.approx(199.4)
    assert fills[1].cash_change == fills[1].gross_notional - 0.5


def test_the_same_request_fills_the_same_way_twice() -> None:
    model = make_model(commission_per_order=1.0)
    request = {
        "event": make_event(),
        "orders": (make_order(),),
        "open_prices": {"AAPL": 100.0},
    }

    assert model.fill_orders(**request) == model.fill_orders(**request)


def test_an_overflowing_slipped_price_is_refused() -> None:
    with pytest.raises(ExecutionError) as excinfo:
        fill_one(open_prices={"AAPL": sys.float_info.max})

    assert "AAPL" in str(excinfo.value)
    assert "rb-0001" in str(excinfo.value)


def test_an_overflowing_notional_is_refused_behind_an_execution_error() -> None:
    with pytest.raises(ExecutionError) as excinfo:
        fill_one(make_order(quantity=1e200), open_prices={"AAPL": 1e200})

    assert isinstance(excinfo.value.__cause__, BacktestContractError)
    assert "gross_notional" in str(excinfo.value)


# ---------------------------------------------------------------------------
# The request: one tuple of this event's orders, in one unambiguous order
# ---------------------------------------------------------------------------
def test_an_empty_tuple_fills_nothing_and_needs_no_price() -> None:
    fills = make_model().fill_orders(event=make_event(), orders=(), open_prices={})

    assert fills == ()
    assert isinstance(fills, tuple)


@pytest.mark.parametrize("orders", [[], {}, set(), make_order(), None])
def test_the_orders_must_be_a_tuple(orders: object) -> None:
    with pytest.raises(ExecutionError) as excinfo:
        make_model().fill_orders(
            event=make_event(),
            orders=orders,  # type: ignore[arg-type]
            open_prices={"AAPL": 100.0},
        )

    assert "tuple[OrderIntent, ...]" in str(excinfo.value)


def test_a_generator_of_orders_is_not_a_tuple_either() -> None:
    with pytest.raises(ExecutionError) as excinfo:
        make_model().fill_orders(
            event=make_event(),
            orders=iter((make_order(),)),  # type: ignore[arg-type]
            open_prices={"AAPL": 100.0},
        )

    assert "generator" in str(excinfo.value)


def test_an_entry_that_is_not_an_order_is_refused() -> None:
    with pytest.raises(ExecutionError) as excinfo:
        make_model().fill_orders(
            event=make_event(),
            orders=(make_order(), "rb-0002"),  # type: ignore[arg-type]
            open_prices={"AAPL": 100.0},
        )

    problems = excinfo.value.problems
    assert len(problems) == 1
    assert "must be an OrderIntent" in problems[0]
    assert "str" in problems[0]


def test_a_repeated_order_id_is_refused() -> None:
    twice = (
        make_order(order_id="rb-0001", symbol="AAPL"),
        make_order(order_id="rb-0001", symbol="MSFT"),
    )

    with pytest.raises(ExecutionError) as excinfo:
        make_model().fill_orders(
            event=make_event(),
            orders=twice,
            open_prices={"AAPL": 100.0, "MSFT": 50.0},
        )

    problems = excinfo.value.problems
    assert len(problems) == 1
    assert "order_id 'rb-0001' appears on 2 orders" in problems[0]


@pytest.mark.parametrize("other_side", ["BUY", "SELL"])
def test_one_symbol_may_appear_on_one_order_only(other_side: str) -> None:
    pair = (
        make_order(order_id="rb-0001", symbol="AAPL", side="BUY"),
        make_order(order_id="rb-0002", symbol="AAPL", side=other_side),
    )

    with pytest.raises(ExecutionError) as excinfo:
        make_model().fill_orders(event=make_event(), orders=pair, open_prices={"AAPL": 100.0})

    problems = excinfo.value.problems
    assert len(problems) == 1
    assert "symbol 'AAPL' appears on 2 orders" in problems[0]


def test_an_order_decided_on_another_day_cannot_fill_here() -> None:
    with pytest.raises(ExecutionError) as excinfo:
        fill_one(make_order(decision=date(2024, 1, 3)))

    problems = excinfo.value.problems
    assert len(problems) == 1
    assert "was decided on 2024-01-03" in problems[0]
    assert "the event decides on 2024-01-04" in problems[0]


def test_an_order_scheduled_for_another_session_cannot_fill_here() -> None:
    with pytest.raises(ExecutionError) as excinfo:
        fill_one(make_order(scheduled=date(2024, 2, 1)))

    problems = excinfo.value.problems
    assert len(problems) == 1
    assert "is scheduled to fill on 2024-02-01" in problems[0]
    assert "the event fills on 2024-01-05" in problems[0]


def test_the_orders_must_be_strictly_ascending_by_id() -> None:
    backwards = (
        make_order(order_id="rb-0002", symbol="MSFT"),
        make_order(order_id="rb-0001", symbol="AAPL"),
    )

    with pytest.raises(ExecutionError) as excinfo:
        make_model().fill_orders(
            event=make_event(),
            orders=backwards,
            open_prices={"AAPL": 100.0, "MSFT": 50.0},
        )

    problems = excinfo.value.problems
    assert len(problems) == 1
    assert "ascending order_id" in problems[0]
    assert "'rb-0001' comes after 'rb-0002' but sorts before it" in problems[0]


def test_sorted_unique_orders_fill_in_that_order() -> None:
    fills = make_model().fill_orders(
        event=make_event(),
        orders=(
            make_order(order_id="rb-0001", symbol="AAPL"),
            make_order(order_id="rb-0002", symbol="MSFT", side="SELL"),
        ),
        open_prices={"AAPL": 100.0, "MSFT": 50.0},
    )

    assert [fill.order.order_id for fill in fills] == ["rb-0001", "rb-0002"]


@pytest.mark.parametrize(
    "event",
    [
        None,
        "2024-01-04",
        {"decision_date": DECISION, "execution_date": EXECUTION},
        (DECISION, EXECUTION),
        EXECUTION,
    ],
)
def test_only_a_rebalance_event_describes_a_fill_session(event: object) -> None:
    with pytest.raises(ExecutionError) as excinfo:
        make_model().fill_orders(
            event=event,  # type: ignore[arg-type]
            orders=(make_order(),),
            open_prices={"AAPL": 100.0},
        )

    assert "must be a RebalanceEvent" in str(excinfo.value)


def test_one_error_collects_every_problem_of_the_request() -> None:
    broken = (
        make_order(order_id="rb-0002", symbol="MSFT", side="SELL"),
        make_order(order_id="rb-0001", symbol="AAPL", decision=date(2024, 1, 3)),
        make_order(order_id="rb-0003", symbol="AAPL", scheduled=date(2024, 2, 1)),
    )

    with pytest.raises(ExecutionError) as excinfo:
        make_model().fill_orders(event=make_event(), orders=broken, open_prices={"MSFT": 50.0})

    error = excinfo.value
    assert str(error).startswith("invalid fill request")
    assert len(error.problems) == 5
    assert any("symbol 'AAPL' appears on 2 orders" in problem for problem in error.problems)
    assert any("was decided on 2024-01-03" in problem for problem in error.problems)
    assert any("is scheduled to fill on 2024-02-01" in problem for problem in error.problems)
    assert any("ascending order_id" in problem for problem in error.problems)
    assert any("no price for 'AAPL'" in problem for problem in error.problems)


# ---------------------------------------------------------------------------
# The price table: one real open per traded symbol, and nothing else is touched
# ---------------------------------------------------------------------------
def test_every_order_is_priced_from_its_own_symbol() -> None:
    fills = make_model().fill_orders(
        event=make_event(),
        orders=(
            make_order(order_id="rb-0001", symbol="AAPL"),
            make_order(order_id="rb-0002", symbol="MSFT", side="SELL"),
        ),
        open_prices={"AAPL": 100.0, "MSFT": 50.0},
    )

    assert fills[0].fill_price == 100.05
    assert fills[1].fill_price == pytest.approx(49.975)


def test_a_missing_price_names_the_symbol_and_the_order() -> None:
    with pytest.raises(ExecutionError) as excinfo:
        fill_one(open_prices={"MSFT": 50.0})

    problems = excinfo.value.problems
    assert len(problems) == 1
    assert "no price for 'AAPL'" in problems[0]
    assert "rb-0001" in problems[0]


def test_the_table_may_price_more_symbols_than_the_orders_trade() -> None:
    fill = fill_one(open_prices={"AAPL": 100.0, "MSFT": 412.0, "TSLA": 250.0})

    assert fill.fill_price == 100.05


@pytest.mark.parametrize(
    "price",
    [0, 0.0, -0.01, -100.0, math.nan, math.inf, -math.inf, "100.0", True, False, None, [100.0]],
)
def test_a_price_must_be_a_finite_number_above_zero(price: object) -> None:
    with pytest.raises(ExecutionError) as excinfo:
        fill_one(open_prices={"AAPL": price})

    problems = excinfo.value.problems
    assert len(problems) == 1
    assert "open_prices['AAPL']" in problems[0]


@pytest.mark.parametrize(
    "open_prices",
    [None, [], [("AAPL", 100.0)], ("AAPL", 100.0), "AAPL", 100.0, 5],
)
def test_the_prices_must_be_a_mapping(open_prices: object) -> None:
    with pytest.raises(ExecutionError) as excinfo:
        fill_one(open_prices=open_prices)

    assert "must be a Mapping[str, float]" in str(excinfo.value)


@pytest.mark.parametrize(
    "key",
    [
        1,
        1.5,
        None,
        b"AAPL",
        (),
        "",
        " AAPL",
        "AAPL ",
        "AA PL",
        "A\tB",
        "AAPL\n",
        "???",
        "aapl",
        "AaPl",
        "AAPL!",
        "$AAPL",
        "AAPL/US",
        "AAPL_1",
        "AAPL?",
    ],
)
def test_a_price_key_must_be_a_project_ticker(key: object) -> None:
    with pytest.raises(ExecutionError) as excinfo:
        fill_one(open_prices={key: 100.0, "AAPL": 100.0})

    problems = excinfo.value.problems
    assert len(problems) == 1
    assert "key" in problems[0]


@pytest.mark.parametrize("key", ["BRK.B", "BF-B", "MSFT", "Z9", "GOOGL"])
def test_a_legal_extra_symbol_is_tolerated(key: str) -> None:
    fill = fill_one(open_prices={"AAPL": 100.0, key: 412.0})

    assert fill.fill_price == 100.05
    assert fill.order.symbol == "AAPL"


@pytest.mark.parametrize(
    "key",
    [
        "AAPL",
        "BRK.B",
        "BF-B",
        "Z9",
        "MMM",
        # "CASH" matches the pattern and is harmless here: no order can ever be cash
        # (``OrderIntent`` refuses it), so a cash row prices nothing.
        "CASH",
        "???",
        "aapl",
        "AaPl",
        "AAPL!",
        "$AAPL",
        "AAPL/US",
        "AAPL_1",
        "AAPL\n",
        "AAPL ",
        " AAPL",
        "A B",
        "",
    ],
)
def test_key_acceptance_is_exactly_the_project_pattern(key: str) -> None:
    """The table takes a key if and only if the project's ticker rule takes it."""
    table: dict[str, float] = {"AAPL": 100.0}
    if key != "AAPL":
        table[key] = 100.0

    if re.fullmatch(SYMBOL_PATTERN, key) is not None:
        assert fill_one(open_prices=table).fill_price == 100.05
        return

    with pytest.raises(ExecutionError) as excinfo:
        fill_one(open_prices=table)
    assert any("open_prices key" in problem for problem in excinfo.value.problems)


def test_the_key_rule_is_the_project_one_not_a_second_one() -> None:
    """The pattern is imported from :mod:`data_sys.schema`, not restated here."""
    assert execution_module.SYMBOL_PATTERN is SYMBOL_PATTERN

    code = SOURCE.read_text(encoding="utf-8")
    assert "[A-Z0-9" not in code  # no second character class, however similar
    assert not re.search(r"""re\.compile\(\s*["']""", code)  # compiled, never a literal


def test_an_unpriced_extra_symbol_is_still_refused() -> None:
    with pytest.raises(ExecutionError) as excinfo:
        fill_one(open_prices={"AAPL": 100.0, "MSFT": None})

    problems = excinfo.value.problems
    assert len(problems) == 1
    assert "open_prices['MSFT']" in problems[0]


def test_the_request_is_read_and_never_modified() -> None:
    order = make_order()
    orders = (order,)
    event = make_event()
    prices = {"AAPL": 100.0, "MSFT": 50.0}
    before = dict(prices)

    fills = make_model(commission_per_order=0.5).fill_orders(
        event=event,
        orders=orders,
        open_prices=prices,
    )

    assert prices == before
    assert orders == (order,)
    assert event == make_event()
    assert fills[0].order is order
    assert fills[0].fill_date == event.execution_date


def test_a_read_only_mapping_is_read_without_complaint() -> None:
    prices = MappingProxyType({"AAPL": 100.0})

    fill = fill_one(open_prices=prices)

    assert fill.fill_price == 100.05
    assert dict(prices) == {"AAPL": 100.0}


# ---------------------------------------------------------------------------
# The wrapping: a fill the contract refuses is still an execution problem
# ---------------------------------------------------------------------------
def test_a_fill_the_contract_refuses_arrives_as_an_execution_error() -> None:
    with pytest.raises(ExecutionError) as excinfo:
        fill_one(make_order(quantity=1e200), open_prices={"AAPL": 1e200})

    error = excinfo.value
    assert isinstance(error.__cause__, BacktestContractError)
    assert "order 'rb-0001' cannot be recorded as a fill at" in str(error)


def test_every_execution_error_is_a_backtest_error() -> None:
    assert issubclass(ExecutionError, BacktestError)
    assert issubclass(BacktestContractError, BacktestError)

    with pytest.raises(BacktestError):
        fill_one(open_prices={})
    with pytest.raises(BacktestError):
        make_model(slippage_bps=-1.0)


def test_the_package_reexports_the_execution_objects() -> None:
    assert backtest_sys.ExecutionModel is ExecutionModel
    assert backtest_sys.FixedBpsExecutionModel is FixedBpsExecutionModel
    assert backtest_sys.ExecutionError is ExecutionError
    assert backtest_sys.errors.ExecutionError is ExecutionError

    for name in ("ExecutionError", "ExecutionModel", "FixedBpsExecutionModel"):
        assert name in backtest_sys.__all__, name


def test_the_module_states_its_public_surface() -> None:
    assert execution_module.__all__ == [
        "BASIS_POINTS_PER_UNIT",
        "DEFAULT_COMMISSION_PER_ORDER",
        "DEFAULT_SLIPPAGE_BPS",
        "MAX_SLIPPAGE_BPS",
        "ExecutionModel",
        "FixedBpsExecutionModel",
    ]
    assert execution_module.BASIS_POINTS_PER_UNIT == 10_000.0
    assert execution_module.MAX_SLIPPAGE_BPS == execution_module.BASIS_POINTS_PER_UNIT
    assert execution_module.DEFAULT_SLIPPAGE_BPS == 5.0
    assert execution_module.DEFAULT_COMMISSION_PER_ORDER == 0.0


def test_the_public_objects_explain_themselves() -> None:
    for name in execution_module.__all__:
        value = getattr(execution_module, name)
        if isinstance(value, float):
            continue
        assert value.__doc__, name


# ---------------------------------------------------------------------------
# The boundary: a friction rule reads no data, so it imports none
# ---------------------------------------------------------------------------
#: Modules the execution layer may import: the standard library it needs, the
#: project's single ticker pattern, and the three backtest modules a fill is stated
#: in -- never a reader, a frame or a file.  ``data_sys.schema`` is the sanctioned
#: ticker source, exactly as it is for :mod:`backtest_sys.contracts`; a key rule of
#: this module's own is what this set exists to prevent.
ALLOWED_IMPORTS: frozenset[str] = frozenset(
    {
        "__future__",
        "abc",
        "backtest_sys.contracts",
        "backtest_sys.errors",
        "backtest_sys.rebalance_policy",
        "collections.abc",
        "data_sys.schema",
        "dataclasses",
        "math",
        "numbers",
        "re",
        "typing",
    }
)

#: Names the execution layer must never mention: an account, a position, a
#: valuation, an engine or a statistic belongs to a later task.
FORBIDDEN_IDENTIFIERS: frozenset[str] = frozenset(
    {
        "BacktestConfig",
        "BacktestPosition",
        "BacktestResult",
        "BacktestState",
        "DailyPortfolioValue",
        "InitialPosition",
        "run_backtest",
        "build_report",
        "plan_orders",
        "update_cash",
        "update_position",
        "portfolio_value",
        "valuation",
        "ledger",
        "nav",
        "turnover",
        "drawdown",
        "sharpe",
    }
)


def execution_tree() -> ast.Module:
    """The parsed source of the module under test."""
    return ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))


def imported_modules() -> set[str]:
    """Every module the module under test imports, by dotted name."""
    names: set[str] = set()
    for node in ast.walk(execution_tree()):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            names.add(node.module)
    return names


def test_the_module_imports_only_the_standard_library_and_its_own_modules() -> None:
    unexpected = imported_modules() - ALLOWED_IMPORTS

    assert unexpected == set(), sorted(unexpected)


@pytest.mark.parametrize(
    "forbidden",
    [
        "pandas",
        "numpy",
        "pyarrow",
        "duckdb",
        "data_sys.storage",
        "data_sys.providers",
        "data_sys.update",
        "data_sys.market_data",
        "factor_sys",
        "strategy_sys",
        "portfolio_sys",
        "backtest_sys.market_data",
        "requests",
        "httpx",
        "urllib",
        "socket",
        "pathlib",
        "os",
        "io",
        "csv",
        "json",
    ],
)
def test_the_module_reads_no_frame_no_file_and_no_network(forbidden: str) -> None:
    assert not any(
        name == forbidden or name.startswith(f"{forbidden}.") for name in imported_modules()
    ), forbidden


def test_the_module_opens_prints_and_evaluates_nothing() -> None:
    banned = {"open", "print", "input", "eval", "exec", "compile", "__import__"}
    called = {
        node.func.id
        for node in ast.walk(execution_tree())
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert called & banned == set()


def test_the_module_names_no_account_no_engine_and_no_statistic() -> None:
    tree = execution_tree()
    mentioned: set[str] = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    mentioned |= {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    mentioned |= {node.arg for node in ast.walk(tree) if isinstance(node, ast.arg)}
    mentioned |= {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }

    assert sorted(mentioned & FORBIDDEN_IDENTIFIERS) == []


def test_the_module_imports_no_private_name_from_another_module() -> None:
    for node in ast.walk(execution_tree()):
        if not isinstance(node, ast.ImportFrom) or node.module == "__future__":
            continue

        assert node.module is not None
        assert not node.module.split(".")[-1].startswith("_"), node.module
        for alias in node.names:
            assert not alias.name.startswith("_"), (node.module, alias.name)


def test_the_module_imports_no_wildcard() -> None:
    for node in ast.walk(execution_tree()):
        if isinstance(node, ast.ImportFrom):
            assert all(alias.name != "*" for alias in node.names), node.module


def test_every_imported_name_is_used_by_the_module() -> None:
    tree = execution_tree()
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            # ``from __future__ import annotations`` imports a compiler flag, not a name.
            if isinstance(node, ast.ImportFrom) and node.module == "__future__":
                continue
            imported.update(alias.asname or alias.name.split(".")[0] for alias in node.names)

    used: set[str] = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    used |= {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    used |= {
        node.value.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
    }

    assert sorted(imported - used) == []
