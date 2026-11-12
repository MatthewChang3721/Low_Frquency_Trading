"""Tests for the backtest data contract (:mod:`backtest_sys.contracts`).

Everything here is in memory and offline: no ``data/`` directory, no factor
store, no price, no trading calendar, no DuckDB connection and no network.  A
contract that needed any of them in order to be *declared* would already be
answering a question the engine is supposed to answer, so these tests build the
objects from literals and pin the vocabulary down before any engine exists.

Covered in this order:

* ``BacktestConfig``: when a replay runs, and with how much money,
* ``BacktestPosition``: one holding, in shares,
* ``BacktestState``: what the account owns at one close,
* ``OrderIntent``: what a decision wants to trade, and when,
* ``Fill``: one order that executed,
* ``DailyPortfolioValue``: what the account was worth at one close,
* ``BacktestResult``: the finished run,
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

import backtest_sys
import backtest_sys.contracts as contracts_module
from backtest_sys import (
    DEFAULT_CALENDAR_NAME,
    NAV_TOLERANCE,
    ORDER_SIDES,
    BacktestConfig,
    BacktestContractError,
    BacktestError,
    BacktestPosition,
    BacktestResult,
    BacktestState,
    DailyPortfolioValue,
    Fill,
    OrderIntent,
)
from data_sys.schema import SYMBOL_PATTERN

#: The window every fixture works in.  ``BEFORE`` and ``AFTER`` sit outside it, so
#: an out-of-range date is never ambiguous.
START: date = date(2024, 1, 4)
END: date = date(2024, 4, 26)
BEFORE: date = date(2024, 1, 3)
AFTER: date = date(2024, 4, 29)

#: The decision / execution pair the fixtures use: decided at one close, filled at
#: the next session's open.
DECISION: date = START
NEXT_SESSION: date = date(2024, 1, 5)
LATER_SESSION: date = date(2024, 2, 1)

#: A decision taken before the window opens, so an out-of-window order can still
#: be scheduled *after* its own decision date.
EXTERNAL_DECISION: date = date(2024, 1, 2)

#: A sentinel for "the factory was not told otherwise", so a test can still pass
#: ``None`` on purpose and mean it.
_UNSET: Any = object()


# ---------------------------------------------------------------------------
# In-memory fixtures (no disk, no network, no calendar, no prices)
# ---------------------------------------------------------------------------
def make_config(
    *,
    start: object = START,
    end: object = END,
    initial_cash: object = 100_000.0,
    calendar_name: object = DEFAULT_CALENDAR_NAME,
) -> BacktestConfig:
    """A :class:`BacktestConfig` for the fixtures."""
    return BacktestConfig(  # type: ignore[arg-type]
        start_date=start,
        end_date=end,
        initial_cash=initial_cash,
        calendar_name=calendar_name,
    )


def make_position(symbol: object = "AAPL", quantity: object = 10.0) -> BacktestPosition:
    """A :class:`BacktestPosition` for the fixtures."""
    return BacktestPosition(symbol=symbol, quantity=quantity)  # type: ignore[arg-type]


def make_state(
    *,
    as_of: object = NEXT_SESSION,
    cash: object = 1_000.0,
    positions: object = _UNSET,
) -> BacktestState:
    """A :class:`BacktestState` holding two sorted positions unless told otherwise."""
    holdings = (
        (BacktestPosition("AAPL", 10.0), BacktestPosition("MSFT", 5.0))
        if positions is _UNSET
        else positions
    )
    return BacktestState(  # type: ignore[arg-type]
        as_of_date=as_of, cash=cash, positions=holdings
    )


def make_order(
    *,
    order_id: object = "rb-0001",
    decision: object = DECISION,
    scheduled: object = NEXT_SESSION,
    symbol: object = "AAPL",
    side: object = "BUY",
    quantity: object = 10.0,
) -> OrderIntent:
    """An :class:`OrderIntent` for the fixtures: a buy of 10 AAPL, one session later."""
    return OrderIntent(  # type: ignore[arg-type]
        order_id=order_id,
        decision_date=decision,
        scheduled_fill_date=scheduled,
        symbol=symbol,
        side=side,
        quantity=quantity,
    )


def make_fill(
    *,
    order: object = None,
    fill_date: object = None,
    fill_price: object = 100.0,
    quantity: object = None,
    transaction_cost: object = 1.0,
) -> Fill:
    """A :class:`Fill` that executes ``order`` completely at its scheduled session."""
    executed = make_order() if order is None else order
    return Fill(  # type: ignore[arg-type]
        order=executed,
        fill_date=executed.scheduled_fill_date if fill_date is None else fill_date,
        fill_price=fill_price,
        quantity=executed.quantity if quantity is None else quantity,
        transaction_cost=transaction_cost,
    )


def make_value(
    *,
    as_of: object = NEXT_SESSION,
    cash: float = 1_000.0,
    market_value: float = 99_000.0,
    nav: object = None,
) -> DailyPortfolioValue:
    """A :class:`DailyPortfolioValue` whose ``nav`` really is ``cash + market_value``."""
    return DailyPortfolioValue(  # type: ignore[arg-type]
        as_of_date=as_of,
        cash=cash,
        market_value=market_value,
        nav=cash + market_value if nav is None else nav,
    )


def make_result(
    *,
    config: object = _UNSET,
    final_state: object = _UNSET,
    daily_values: object = _UNSET,
    fills: object = (),
) -> BacktestResult:
    """A :class:`BacktestResult` of one valued session and no trades by default."""
    values = (make_value(as_of=NEXT_SESSION),) if daily_values is _UNSET else daily_values
    state = make_state(as_of=NEXT_SESSION) if final_state is _UNSET else final_state
    return BacktestResult(  # type: ignore[arg-type]
        config=make_config() if config is _UNSET else config,
        final_state=state,
        daily_values=values,
        fills=fills,
    )


# ---------------------------------------------------------------------------
# BacktestConfig -- when a replay runs, and with how much money
# ---------------------------------------------------------------------------
def test_a_config_holds_the_range_the_cash_and_the_calendar() -> None:
    config = make_config()

    assert (config.start_date, config.end_date) == (START, END)
    assert config.initial_cash == 100_000.0
    assert config.calendar_name == DEFAULT_CALENDAR_NAME


def test_a_config_accepts_iso_text_and_stores_real_dates() -> None:
    config = BacktestConfig(start_date="2024-01-04", end_date="2024-04-26", initial_cash=1_000)

    assert config.start_date == START
    assert config.end_date == END
    assert isinstance(config.start_date, date) and not isinstance(config.start_date, datetime)
    assert isinstance(config.initial_cash, float)
    assert config == BacktestConfig(START, END, 1_000.0)


def test_a_config_accepts_a_single_session_range() -> None:
    config = make_config(start=START, end=START)

    assert config.start_date == config.end_date == START


def test_a_config_defaults_to_the_new_york_calendar() -> None:
    assert DEFAULT_CALENDAR_NAME == "XNYS"
    assert BacktestConfig(START, END, 1.0).calendar_name == "XNYS"


def test_a_config_is_frozen_and_renders_its_fields() -> None:
    config = make_config()

    with pytest.raises(FrozenInstanceError):
        config.initial_cash = 1.0  # type: ignore[misc]

    rendered = repr(config)
    assert rendered.startswith("BacktestConfig(")
    assert "initial_cash=100000.0" in rendered


def test_a_config_declares_exactly_four_fields() -> None:
    assert {record.name for record in fields(BacktestConfig)} == {
        "start_date",
        "end_date",
        "initial_cash",
        "calendar_name",
    }


@pytest.mark.parametrize(
    "moment",
    [
        datetime(2024, 1, 4),
        datetime(2024, 1, 4, 9, 30),
        pd.Timestamp("2024-01-04"),
        pd.Timestamp("2024-01-04 09:30"),
        pd.NaT,
    ],
    ids=["midnight", "intraday", "timestamp", "timestamp-with-time", "nat"],
)
def test_a_config_rejects_a_timestamp_as_a_date(moment: object) -> None:
    with pytest.raises(BacktestContractError) as excinfo:
        make_config(start=moment)

    assert "BacktestConfig.start_date" in str(excinfo.value)


@pytest.mark.parametrize(
    "text",
    [
        "2024-01-04T00:00",
        "2024-01-04 00:00:00",
        "2024-1-4",
        "20240104",
        "04/01/2024",
        "2024-13-01",
        "not a date",
        "",
    ],
)
def test_a_config_rejects_anything_but_a_bare_iso_date(text: str) -> None:
    with pytest.raises(BacktestContractError):
        make_config(end=text)


@pytest.mark.parametrize("value", [None, 20240104, 1.5, object()])
def test_a_config_rejects_a_date_that_is_neither_text_nor_a_date(value: object) -> None:
    with pytest.raises(BacktestContractError):
        make_config(start=value)


def test_a_config_rejects_a_range_that_runs_backwards() -> None:
    with pytest.raises(BacktestContractError) as excinfo:
        make_config(start=END, end=START)

    assert "runs backwards" in str(excinfo.value)


@pytest.mark.parametrize(
    "cash",
    [True, False, 0, 0.0, -1.0, -0.5, float("nan"), float("inf"), float("-inf"), "1000", None],
    ids=[
        "true",
        "false",
        "zero-int",
        "zero",
        "negative",
        "negative-float",
        "nan",
        "inf",
        "-inf",
        "text",
        "none",
    ],
)
def test_a_config_rejects_a_starting_cash_that_cannot_be_invested(cash: object) -> None:
    with pytest.raises(BacktestContractError) as excinfo:
        make_config(initial_cash=cash)

    assert "initial_cash" in str(excinfo.value)


@pytest.mark.parametrize("name", ["", "   ", None, 123, ["XNYS"]])
def test_a_config_rejects_a_nameless_calendar(name: object) -> None:
    with pytest.raises(BacktestContractError) as excinfo:
        make_config(calendar_name=name)

    assert "calendar_name" in str(excinfo.value)


@pytest.mark.parametrize("name", [" XNYS", "XNYS ", " XNYS "])
def test_a_config_refuses_to_trim_a_calendar_name(name: str) -> None:
    with pytest.raises(BacktestContractError) as excinfo:
        make_config(calendar_name=name)

    assert "whitespace" in str(excinfo.value)


def test_a_config_accepts_another_named_calendar() -> None:
    assert make_config(calendar_name="XLON").calendar_name == "XLON"


# ---------------------------------------------------------------------------
# BacktestPosition -- one holding, in shares
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("symbol", ["AAPL", "BRK.B", "RDS-A", "F", "ZZZZ", "DELISTED"])
def test_a_position_accepts_the_projects_ticker_syntax(symbol: str) -> None:
    assert make_position(symbol).symbol == symbol


def test_a_position_keeps_a_fractional_share_count() -> None:
    position = make_position("AAPL", 50.5)

    assert position.quantity == 50.5
    assert isinstance(position.quantity, float)


def test_a_position_normalises_a_whole_share_count_to_a_float() -> None:
    position = make_position("AAPL", 10)

    assert position.quantity == 10.0
    assert isinstance(position.quantity, float)


def test_a_position_is_frozen_and_declares_two_fields() -> None:
    position = make_position()

    with pytest.raises(FrozenInstanceError):
        position.quantity = 1.0  # type: ignore[misc]

    assert {record.name for record in fields(BacktestPosition)} == {"symbol", "quantity"}
    assert position == BacktestPosition("AAPL", 10.0)
    assert "BacktestPosition(" in repr(position)


def test_a_position_explains_that_cash_is_a_field_not_a_security() -> None:
    with pytest.raises(BacktestContractError) as excinfo:
        make_position("CASH")

    assert "cash is the account's own cash amount" in str(excinfo.value)


@pytest.mark.parametrize("symbol", ["aapl", "AA PL", "AA_PL", "", " AAPL", "AAPL ", None, 123])
def test_a_position_rejects_a_symbol_that_is_not_a_ticker(symbol: object) -> None:
    with pytest.raises(BacktestContractError) as excinfo:
        make_position(symbol)

    assert SYMBOL_PATTERN in str(excinfo.value)


@pytest.mark.parametrize(
    "quantity",
    [True, 0, 0.0, -1.0, -0.25, float("nan"), float("inf"), float("-inf"), "10", None],
    ids=[
        "true",
        "zero-int",
        "zero",
        "negative",
        "negative-float",
        "nan",
        "inf",
        "-inf",
        "text",
        "none",
    ],
)
def test_a_position_rejects_a_share_count_that_is_not_positive(quantity: object) -> None:
    with pytest.raises(BacktestContractError) as excinfo:
        make_position("AAPL", quantity)

    assert "quantity" in str(excinfo.value)


# ---------------------------------------------------------------------------
# BacktestState -- what the account owns at one close
# ---------------------------------------------------------------------------
def test_a_state_holds_cash_and_sorted_share_counts() -> None:
    state = make_state()

    assert state.as_of_date == NEXT_SESSION
    assert state.cash == 1_000.0
    assert state.symbols == ("AAPL", "MSFT")
    assert state.quantities_by_symbol == {"AAPL": 10.0, "MSFT": 5.0}
    assert state.positions == (BacktestPosition("AAPL", 10.0), BacktestPosition("MSFT", 5.0))


def test_a_state_accepts_an_all_cash_account() -> None:
    state = make_state(positions=(), cash=0.0)

    assert state.positions == ()
    assert state.symbols == ()
    assert state.quantities_by_symbol == {}
    assert state.cash == 0.0


def test_a_state_accepts_a_single_holding() -> None:
    state = make_state(positions=(BacktestPosition("BRK.B", 3.5),))

    assert state.symbols == ("BRK.B",)
    assert state.quantities_by_symbol["BRK.B"] == 3.5


def test_a_state_accepts_iso_text_for_its_date() -> None:
    assert make_state(as_of="2024-01-05").as_of_date == NEXT_SESSION


def test_a_state_is_frozen_and_declares_three_fields() -> None:
    state = make_state()

    with pytest.raises(FrozenInstanceError):
        state.cash = 1.0  # type: ignore[misc]

    assert {record.name for record in fields(BacktestState)} == {
        "as_of_date",
        "cash",
        "positions",
    }
    assert "BacktestState(" in repr(state)


def test_a_state_exposes_its_views_as_properties_not_fields() -> None:
    for view in ("symbols", "quantities_by_symbol"):
        assert isinstance(getattr(BacktestState, view), property), view
    stored = {record.name for record in fields(BacktestState)}
    assert not ({"symbols", "quantities_by_symbol"} & stored)


@pytest.mark.parametrize(
    "positions",
    [list(()), set(), iter(()), None, BacktestPosition("AAPL", 1.0)],
    ids=["list", "set", "generator", "none", "bare-position"],
)
def test_a_state_requires_a_tuple_of_positions(positions: object) -> None:
    with pytest.raises(BacktestContractError) as excinfo:
        make_state(positions=positions)

    assert "tuple[BacktestPosition, ...]" in str(excinfo.value)


def test_a_state_rejects_an_entry_that_is_not_a_position() -> None:
    with pytest.raises(BacktestContractError) as excinfo:
        make_state(positions=(("AAPL", 10.0),))

    assert "must be a BacktestPosition" in str(excinfo.value)


def test_a_state_rejects_a_symbol_held_twice() -> None:
    with pytest.raises(BacktestContractError) as excinfo:
        make_state(positions=(BacktestPosition("AAPL", 10.0), BacktestPosition("AAPL", 1.0)))

    assert "holds AAPL more than once" in str(excinfo.value)


def test_a_state_rejects_positions_that_are_not_sorted() -> None:
    with pytest.raises(BacktestContractError) as excinfo:
        make_state(positions=(BacktestPosition("MSFT", 5.0), BacktestPosition("AAPL", 10.0)))

    message = str(excinfo.value)
    assert "ascending symbol" in message
    assert "re-sorted" in message


def test_a_state_reports_a_duplicate_once_without_the_ordering_complaint() -> None:
    with pytest.raises(BacktestContractError) as excinfo:
        make_state(
            positions=(
                BacktestPosition("MSFT", 5.0),
                BacktestPosition("MSFT", 1.0),
                BacktestPosition("AAPL", 1.0),
            )
        )

    problems = excinfo.value.problems
    assert len(problems) == 1
    assert "MSFT" in problems[0]


def test_a_state_bounds_the_symbols_it_quotes_in_one_error() -> None:
    with pytest.raises(BacktestContractError) as excinfo:
        make_state(
            positions=(
                BacktestPosition("MSFT", 1.0),
                BacktestPosition("AAPL", 1.0),
                BacktestPosition("NVDA", 1.0),
                BacktestPosition("TSLA", 1.0),
                BacktestPosition("AMZN", 1.0),
            )
        )

    assert "(+2 more)" in str(excinfo.value)


@pytest.mark.parametrize(
    "cash",
    [True, -1.0, -0.01, float("nan"), float("inf"), float("-inf"), "0", None],
    ids=["true", "negative", "negative-small", "nan", "inf", "-inf", "text", "none"],
)
def test_a_state_rejects_cash_that_never_could_have_been_held(cash: object) -> None:
    with pytest.raises(BacktestContractError) as excinfo:
        make_state(cash=cash)

    assert "cash" in str(excinfo.value)


def test_a_state_rejects_a_timestamp_as_its_date() -> None:
    with pytest.raises(BacktestContractError) as excinfo:
        make_state(as_of=pd.Timestamp("2024-01-05"))

    assert "BacktestState.as_of_date" in str(excinfo.value)


# ---------------------------------------------------------------------------
# OrderIntent -- what a decision wants to trade, and when
# ---------------------------------------------------------------------------
def test_an_order_holds_the_timing_the_symbol_the_side_and_the_size() -> None:
    order = make_order()

    assert order.order_id == "rb-0001"
    assert (order.decision_date, order.scheduled_fill_date) == (DECISION, NEXT_SESSION)
    assert (order.symbol, order.side, order.quantity) == ("AAPL", "BUY", 10.0)
    assert order.is_buy
    assert not order.is_sell


def test_an_order_accepts_a_sell_and_reports_the_side() -> None:
    order = make_order(side="SELL")

    assert order.is_sell
    assert not order.is_buy


def test_an_order_accepts_iso_text_for_its_dates() -> None:
    order = make_order(decision="2024-01-04", scheduled="2024-01-05")

    assert order.decision_date == DECISION
    assert order.scheduled_fill_date == NEXT_SESSION


def test_the_two_order_sides_are_the_projects_vocabulary() -> None:
    assert ORDER_SIDES == ("BUY", "SELL")


def test_an_order_is_frozen_and_declares_six_fields() -> None:
    order = make_order()

    with pytest.raises(FrozenInstanceError):
        order.quantity = 1.0  # type: ignore[misc]

    assert {record.name for record in fields(OrderIntent)} == {
        "order_id",
        "decision_date",
        "scheduled_fill_date",
        "symbol",
        "side",
        "quantity",
    }
    assert "OrderIntent(" in repr(order)


def test_an_order_exposes_its_sides_as_properties_not_fields() -> None:
    for view in ("is_buy", "is_sell"):
        assert isinstance(getattr(OrderIntent, view), property), view
    stored = {record.name for record in fields(OrderIntent)}
    assert not ({"is_buy", "is_sell"} & stored)


@pytest.mark.parametrize("side", ["buy", "Buy", "SELL ", "LONG", "SHORT", "", None, 1])
def test_an_order_rejects_a_side_that_is_not_one_of_the_two_spellings(side: object) -> None:
    with pytest.raises(BacktestContractError) as excinfo:
        make_order(side=side)

    assert "side" in str(excinfo.value)


@pytest.mark.parametrize("scheduled", [DECISION, BEFORE], ids=["same-session", "earlier"])
def test_an_order_cannot_be_scheduled_for_its_own_decision_session(scheduled: object) -> None:
    with pytest.raises(BacktestContractError) as excinfo:
        make_order(scheduled=scheduled)

    assert "must be after decision_date" in str(excinfo.value)


def test_an_order_accepts_a_fill_many_sessions_later() -> None:
    assert make_order(scheduled=LATER_SESSION).scheduled_fill_date == LATER_SESSION


@pytest.mark.parametrize("identifier", ["", "   ", " rb-0001", "rb-0001 ", None, 1])
def test_an_order_rejects_an_id_that_is_blank_or_padded(identifier: object) -> None:
    with pytest.raises(BacktestContractError) as excinfo:
        make_order(order_id=identifier)

    assert "order_id" in str(excinfo.value)


def test_an_order_rejects_cash_as_a_symbol() -> None:
    with pytest.raises(BacktestContractError) as excinfo:
        make_order(symbol="CASH")

    assert "not a security" in str(excinfo.value)


@pytest.mark.parametrize(
    "quantity",
    [True, 0, -1.0, float("nan"), float("inf"), float("-inf"), "10", None],
)
def test_an_order_rejects_a_size_that_is_not_a_positive_number(quantity: object) -> None:
    with pytest.raises(BacktestContractError) as excinfo:
        make_order(quantity=quantity)

    assert "quantity" in str(excinfo.value)


def test_an_order_rejects_a_timestamp_as_a_date() -> None:
    with pytest.raises(BacktestContractError) as excinfo:
        make_order(decision=pd.Timestamp("2024-01-04 16:00"))

    assert "OrderIntent.decision_date" in str(excinfo.value)


def test_an_order_rejects_a_timestamp_as_its_scheduled_date() -> None:
    with pytest.raises(BacktestContractError) as excinfo:
        make_order(scheduled=datetime(2024, 1, 5, 9, 30))

    assert "OrderIntent.scheduled_fill_date" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Fill -- one order that executed
# ---------------------------------------------------------------------------
def test_a_buy_fill_pays_its_notional_and_its_cost() -> None:
    fill = make_fill(fill_price=100.0, transaction_cost=1.0)

    assert fill.order == make_order()
    assert fill.fill_date == NEXT_SESSION
    assert fill.fill_price == 100.0
    assert fill.quantity == 10.0
    assert fill.transaction_cost == 1.0
    assert fill.symbol == "AAPL"
    assert fill.gross_notional == 1_000.0
    assert fill.cash_change == -1_001.0


def test_a_sell_fill_receives_its_notional_less_its_cost() -> None:
    fill = make_fill(order=make_order(side="SELL"), fill_price=50.0, transaction_cost=2.5)

    assert fill.gross_notional == 500.0
    assert fill.cash_change == 497.5


def test_a_fill_without_a_commission_still_states_its_cash_change() -> None:
    assert make_fill(transaction_cost=0.0).cash_change == -1_000.0
    assert make_fill(order=make_order(side="SELL"), transaction_cost=0).cash_change == 1_000.0


def test_a_fractional_fill_keeps_its_exact_size() -> None:
    fill = make_fill(order=make_order(quantity=2.5), fill_price=40.0, quantity=2.5)

    assert fill.quantity == 2.5
    assert fill.gross_notional == 100.0
    assert fill.cash_change == -101.0


def test_a_fill_is_frozen_and_declares_five_fields() -> None:
    fill = make_fill()

    with pytest.raises(FrozenInstanceError):
        fill.fill_price = 1.0  # type: ignore[misc]

    assert {record.name for record in fields(Fill)} == {
        "order",
        "fill_date",
        "fill_price",
        "quantity",
        "transaction_cost",
    }
    assert "Fill(" in repr(fill)


def test_a_fill_exposes_its_derived_amounts_as_properties_not_fields() -> None:
    for view in ("symbol", "gross_notional", "cash_change"):
        assert isinstance(getattr(Fill, view), property), view
    stored = {record.name for record in fields(Fill)}
    assert not ({"symbol", "gross_notional", "cash_change"} & stored)


@pytest.mark.parametrize("order", [None, "rb-0001", {"symbol": "AAPL"}])
def test_a_fill_must_quote_an_order_intent(order: object) -> None:
    with pytest.raises(BacktestContractError) as excinfo:
        Fill(  # type: ignore[arg-type]
            order=order,
            fill_date=NEXT_SESSION,
            fill_price=100.0,
            quantity=10.0,
            transaction_cost=1.0,
        )

    assert "must be an OrderIntent" in str(excinfo.value)


@pytest.mark.parametrize(
    "fill_date",
    [DECISION, LATER_SESSION, "2024-01-08"],
    ids=["early", "late", "text-late"],
)
def test_a_fill_is_dated_exactly_its_scheduled_session(fill_date: object) -> None:
    with pytest.raises(BacktestContractError) as excinfo:
        make_fill(fill_date=fill_date)

    assert "not the scheduled fill date" in str(excinfo.value)


@pytest.mark.parametrize("quantity", [5.0, 10.000000001, 20.0], ids=["half", "almost", "double"])
def test_a_fill_refuses_a_partial_or_oversized_execution(quantity: float) -> None:
    with pytest.raises(BacktestContractError) as excinfo:
        make_fill(quantity=quantity)

    assert "not the ordered size" in str(excinfo.value)


@pytest.mark.parametrize(
    "price",
    [True, 0, 0.0, -1.0, float("nan"), float("inf"), float("-inf"), "100", None],
    ids=["true", "zero-int", "zero", "negative", "nan", "inf", "-inf", "text", "none"],
)
def test_a_fill_rejects_a_price_that_is_not_positive(price: object) -> None:
    with pytest.raises(BacktestContractError) as excinfo:
        make_fill(fill_price=price)

    assert "fill_price" in str(excinfo.value)


@pytest.mark.parametrize(
    "cost", [True, -0.01, float("nan"), float("inf"), float("-inf"), "1", None]
)
def test_a_fill_rejects_a_negative_or_unusable_cost(cost: object) -> None:
    with pytest.raises(BacktestContractError) as excinfo:
        make_fill(transaction_cost=cost)

    assert "transaction_cost" in str(excinfo.value)


def test_a_fill_refuses_a_notional_that_overflows() -> None:
    order = make_order(quantity=1e308)

    with pytest.raises(BacktestContractError) as excinfo:
        make_fill(order=order, fill_price=1e308, quantity=1e308)

    assert "gross_notional is not a finite amount" in str(excinfo.value)


def test_a_fill_refuses_a_cash_change_that_overflows() -> None:
    with pytest.raises(BacktestContractError) as excinfo:
        make_fill(
            order=make_order(quantity=1.7),
            fill_price=1e308,
            quantity=1.7,
            transaction_cost=1.7e308,
        )

    assert "cash_change is not a finite amount" in str(excinfo.value)


@pytest.mark.parametrize(
    "quantity",
    [True, 0, 0.0, -1.0, float("nan"), float("inf"), "10", None],
    ids=["true", "zero-int", "zero", "negative", "nan", "inf", "text", "none"],
)
def test_a_fill_rejects_a_size_that_is_not_a_positive_number(quantity: object) -> None:
    with pytest.raises(BacktestContractError) as excinfo:
        Fill(  # type: ignore[arg-type]
            order=make_order(),
            fill_date=NEXT_SESSION,
            fill_price=100.0,
            quantity=quantity,
            transaction_cost=1.0,
        )

    assert "quantity" in str(excinfo.value)


def test_a_fill_rejects_a_timestamp_as_its_date() -> None:
    with pytest.raises(BacktestContractError) as excinfo:
        make_fill(fill_date=pd.Timestamp("2024-01-05 09:30"))

    assert "Fill.fill_date" in str(excinfo.value)


# ---------------------------------------------------------------------------
# DailyPortfolioValue -- what the account was worth at one close
# ---------------------------------------------------------------------------
def test_a_valuation_holds_its_three_amounts() -> None:
    value = make_value()

    assert value.as_of_date == NEXT_SESSION
    assert (value.cash, value.market_value, value.nav) == (1_000.0, 99_000.0, 100_000.0)


def test_a_valuation_accepts_iso_text_and_an_all_cash_account() -> None:
    value = DailyPortfolioValue("2024-01-05", 0.0, 0.0, 0.0)

    assert value.as_of_date == NEXT_SESSION
    assert value.nav == 0.0


def test_a_valuation_tolerates_the_last_bit_of_a_correct_sum() -> None:
    cash = 0.1 + 0.2  # 0.30000000000000004, not 0.3
    value = DailyPortfolioValue(NEXT_SESSION, cash, 0.0, 0.3)

    assert abs(value.nav - (value.cash + value.market_value)) < NAV_TOLERANCE


def test_the_nav_tolerance_is_the_projects_own_constant() -> None:
    assert NAV_TOLERANCE == 1e-9
    assert contracts_module.NAV_TOLERANCE is NAV_TOLERANCE


def test_a_valuation_is_frozen_and_declares_four_fields() -> None:
    value = make_value()

    with pytest.raises(FrozenInstanceError):
        value.nav = 1.0  # type: ignore[misc]

    assert {record.name for record in fields(DailyPortfolioValue)} == {
        "as_of_date",
        "cash",
        "market_value",
        "nav",
    }
    assert "DailyPortfolioValue(" in repr(value)


@pytest.mark.parametrize("delta", [1e-3, -1e-3, 100.0, -100.0])
def test_a_valuation_rejects_a_nav_that_does_not_add_up(delta: float) -> None:
    with pytest.raises(BacktestContractError) as excinfo:
        make_value(nav=100_000.0 + delta)

    assert "nav must equal cash + market_value" in str(excinfo.value)


@pytest.mark.parametrize("field_name", ["cash", "market_value", "nav"])
@pytest.mark.parametrize(
    "amount",
    [True, -1.0, float("nan"), float("inf"), float("-inf"), "0", None],
    ids=["true", "negative", "nan", "inf", "-inf", "text", "none"],
)
def test_a_valuation_rejects_an_amount_that_is_not_a_non_negative_number(
    field_name: str, amount: object
) -> None:
    amounts: dict[str, Any] = {
        "cash": 1_000.0,
        "market_value": 99_000.0,
        "nav": 100_000.0,
        field_name: amount,
    }

    with pytest.raises(BacktestContractError) as excinfo:
        DailyPortfolioValue(  # type: ignore[arg-type]
            as_of_date=NEXT_SESSION,
            cash=amounts["cash"],
            market_value=amounts["market_value"],
            nav=amounts["nav"],
        )

    assert field_name in str(excinfo.value)


def test_a_valuation_rejects_a_timestamp_as_its_date() -> None:
    with pytest.raises(BacktestContractError) as excinfo:
        make_value(as_of=datetime(2024, 1, 5, 16, 0))

    assert "DailyPortfolioValue.as_of_date" in str(excinfo.value)


# ---------------------------------------------------------------------------
# BacktestResult -- the finished run
# ---------------------------------------------------------------------------
def make_close(day: object, *, cash: float = 1_000.0, market_value: float = 99_000.0):
    """One valuation row, dated ``day``, whose nav really adds up."""
    return make_value(as_of=day, cash=cash, market_value=market_value)


def test_a_result_holds_the_run_and_both_histories_in_order() -> None:
    fill = make_fill()
    result = make_result(fills=(fill,))

    assert result.config == make_config()
    assert result.dates == (NEXT_SESSION,)
    assert result.order_ids == ("rb-0001",)
    assert result.final_state.as_of_date == NEXT_SESSION
    assert result.fills == (fill,)


def test_a_result_accepts_two_closes_in_ascending_order() -> None:
    result = make_result(
        daily_values=(make_close(DECISION), make_close(NEXT_SESSION)),
        final_state=make_state(as_of=NEXT_SESSION),
    )

    assert result.dates == (DECISION, NEXT_SESSION)


def test_a_result_accepts_empty_histories() -> None:
    result = BacktestResult(
        config=make_config(),
        final_state=make_state(as_of=END),
        daily_values=(),
        fills=(),
    )

    assert result.dates == ()
    assert result.order_ids == ()


def test_a_result_is_frozen_and_declares_four_fields() -> None:
    result = make_result()

    with pytest.raises(FrozenInstanceError):
        result.fills = ()  # type: ignore[misc]

    assert {record.name for record in fields(BacktestResult)} == {
        "config",
        "final_state",
        "daily_values",
        "fills",
    }
    assert "BacktestResult(" in repr(result)


def test_a_result_exposes_its_histories_as_properties_not_fields() -> None:
    for view in ("dates", "order_ids"):
        assert isinstance(getattr(BacktestResult, view), property), view
    stored = {record.name for record in fields(BacktestResult)}
    assert not ({"dates", "order_ids"} & stored)


@pytest.mark.parametrize("config", [None, "2024-01-04", {"start_date": START}])
def test_a_result_must_name_its_configuration(config: object) -> None:
    with pytest.raises(BacktestContractError) as excinfo:
        make_result(config=config)

    assert "must be a BacktestConfig" in str(excinfo.value)


@pytest.mark.parametrize("state", [None, (), {"cash": 0.0}])
def test_a_result_must_end_in_a_state(state: object) -> None:
    with pytest.raises(BacktestContractError) as excinfo:
        make_result(final_state=state)

    assert "must be a BacktestState" in str(excinfo.value)


@pytest.mark.parametrize("field_name", ["daily_values", "fills"])
@pytest.mark.parametrize("history", [list(()), set(), iter(())], ids=["list", "set", "generator"])
def test_a_result_requires_tuples_for_both_histories(field_name: str, history: object) -> None:
    kwargs: dict[str, Any] = {
        "daily_values": (make_close(NEXT_SESSION),),
        "fills": (make_fill(),),
        field_name: history,
    }

    with pytest.raises(BacktestContractError) as excinfo:
        make_result(**kwargs)

    assert field_name in str(excinfo.value)


def test_a_result_rejects_a_valuation_entry_that_is_not_a_value() -> None:
    with pytest.raises(BacktestContractError) as excinfo:
        make_result(daily_values=(("2024-01-05", 1.0, 2.0, 3.0),))

    assert "must be a DailyPortfolioValue" in str(excinfo.value)


def test_a_result_rejects_a_fill_entry_that_is_not_a_fill() -> None:
    with pytest.raises(BacktestContractError) as excinfo:
        make_result(fills=(make_order(),))

    assert "must be a Fill" in str(excinfo.value)


def test_a_result_rejects_valuations_out_of_order() -> None:
    with pytest.raises(BacktestContractError) as excinfo:
        make_result(daily_values=(make_close(NEXT_SESSION), make_close(DECISION)))

    assert "ascending as_of_date" in str(excinfo.value)


def test_a_result_rejects_a_session_valued_twice() -> None:
    with pytest.raises(BacktestContractError) as excinfo:
        make_result(daily_values=(make_close(NEXT_SESSION), make_close(NEXT_SESSION)))

    assert "more than once" in str(excinfo.value)


@pytest.mark.parametrize("day", [BEFORE, AFTER], ids=["before-start", "after-end"])
def test_a_result_rejects_a_valuation_outside_the_configured_window(day: date) -> None:
    with pytest.raises(BacktestContractError) as excinfo:
        make_result(daily_values=(make_close(day),), final_state=make_state(as_of=day))

    assert "outside the configured range" in str(excinfo.value)


def test_a_result_requires_the_closing_state_of_the_last_close() -> None:
    with pytest.raises(BacktestContractError) as excinfo:
        make_result(
            daily_values=(make_close(DECISION),),
            final_state=make_state(as_of=NEXT_SESSION),
        )

    assert "the last valuation is dated" in str(excinfo.value)


@pytest.mark.parametrize("day", [BEFORE, AFTER], ids=["before-start", "after-end"])
def test_a_result_requires_a_closing_state_inside_the_window(day: date) -> None:
    """Even with no valuation to compare against, the state belongs to this run."""
    with pytest.raises(BacktestContractError) as excinfo:
        make_result(final_state=make_state(as_of=day), daily_values=())

    assert "final_state.as_of_date" in str(excinfo.value)
    assert "outside the configured range" in str(excinfo.value)


@pytest.mark.parametrize("day", [START, END], ids=["first-session", "last-session"])
def test_a_result_accepts_a_closing_state_on_either_edge_of_the_window(day: date) -> None:
    result = make_result(final_state=make_state(as_of=day), daily_values=())

    assert result.final_state.as_of_date == day
    assert result.dates == ()


def test_a_result_requires_the_closing_cash_of_the_last_close() -> None:
    with pytest.raises(BacktestContractError) as excinfo:
        make_result(
            daily_values=(make_close(NEXT_SESSION, cash=1_000.0),),
            final_state=make_state(as_of=NEXT_SESSION, cash=999.0),
        )

    assert "final_state.cash" in str(excinfo.value)
    assert "closing snapshot answer one question" in str(excinfo.value)


def test_a_result_accepts_closing_cash_exactly_at_the_nav_tolerance() -> None:
    # ``1e-9 - 0.0`` is exact in binary floating point, so this really is the edge.
    result = make_result(
        daily_values=(make_close(NEXT_SESSION, cash=0.0),),
        final_state=make_state(as_of=NEXT_SESSION, cash=NAV_TOLERANCE),
    )

    assert result.final_state.cash == NAV_TOLERANCE


def test_a_result_rejects_closing_cash_beyond_the_nav_tolerance() -> None:
    with pytest.raises(BacktestContractError) as excinfo:
        make_result(
            daily_values=(make_close(NEXT_SESSION, cash=0.0),),
            final_state=make_state(as_of=NEXT_SESSION, cash=NAV_TOLERANCE * 2.0),
        )

    assert "final_state.cash" in str(excinfo.value)


def test_a_result_blames_the_date_rather_than_comparing_cash_across_sessions() -> None:
    # Comparing 999 to a close dated another session would be a second complaint
    # about a value that is not comparable, so only the date is named.
    with pytest.raises(BacktestContractError) as excinfo:
        make_result(
            daily_values=(make_close(DECISION, cash=1_000.0),),
            final_state=make_state(as_of=NEXT_SESSION, cash=999.0),
        )

    problems = excinfo.value.problems
    assert len(problems) == 1
    assert "the last valuation is dated" in problems[0]


def test_a_result_rejects_an_order_executed_twice() -> None:
    with pytest.raises(BacktestContractError) as excinfo:
        make_result(fills=(make_fill(), make_fill()))

    assert "more than once" in str(excinfo.value)


def test_a_result_rejects_fills_out_of_date_order() -> None:
    late = make_fill(order=make_order(order_id="rb-0002", scheduled=LATER_SESSION))
    early = make_fill()

    with pytest.raises(BacktestContractError) as excinfo:
        make_result(fills=(late, early))

    assert "(fill_date, order_id)" in str(excinfo.value)


def test_a_result_rejects_two_fills_of_one_session_out_of_id_order() -> None:
    second = make_fill(order=make_order(order_id="rb-b"))
    first = make_fill(order=make_order(order_id="rb-a"))

    with pytest.raises(BacktestContractError) as excinfo:
        make_result(fills=(second, first))

    assert "(fill_date, order_id)" in str(excinfo.value)


@pytest.mark.parametrize("day", [BEFORE, AFTER], ids=["before-start", "after-end"])
def test_a_result_rejects_a_fill_outside_the_configured_window(day: date) -> None:
    # The order is decided outside the window too, so the only complaint left is
    # the fill's own date.
    order = make_order(decision=EXTERNAL_DECISION, scheduled=day)

    with pytest.raises(BacktestContractError) as excinfo:
        make_result(fills=(make_fill(order=order),))

    assert "outside the configured range" in str(excinfo.value)


def test_a_result_accepts_a_fill_on_the_last_session_of_the_window() -> None:
    order = make_order(decision=BEFORE, scheduled=END)
    result = make_result(fills=(make_fill(order=order),))

    assert result.order_ids == ("rb-0001",)
    assert result.fills[0].fill_date == END


# ---------------------------------------------------------------------------
# Collected problems: one error names them all
# ---------------------------------------------------------------------------
def test_a_state_reports_every_problem_in_one_error() -> None:
    with pytest.raises(BacktestContractError) as excinfo:
        BacktestState(  # type: ignore[arg-type]
            as_of_date=pd.Timestamp("2024-01-05 16:00"),
            cash=-5.0,
            positions=(BacktestPosition("MSFT", 1.0), BacktestPosition("AAPL", 1.0)),
        )

    error = excinfo.value
    assert len(error.problems) == 3
    assert "as_of_date" in error.problems[0]
    assert "ascending symbol" in error.problems[1]
    assert "cash" in error.problems[2]
    assert str(error).startswith("invalid BacktestState")
    assert len(str(error).splitlines()) == 4


def test_a_result_reports_every_problem_in_one_error() -> None:
    with pytest.raises(BacktestContractError) as excinfo:
        BacktestResult(  # type: ignore[arg-type]
            config="2024-01-04",
            final_state=None,
            daily_values=(make_close(NEXT_SESSION), make_close(DECISION)),
            fills=(make_fill(), make_fill()),
        )

    error = excinfo.value
    assert len(error.problems) == 4
    assert any("config" in problem for problem in error.problems)
    assert any("final_state" in problem for problem in error.problems)
    assert any("ascending as_of_date" in problem for problem in error.problems)
    assert any("more than once" in problem for problem in error.problems)


def test_a_problem_names_the_field_it_is_about() -> None:
    with pytest.raises(BacktestContractError) as excinfo:
        make_order(quantity=-1.0, symbol="aapl")

    problems = excinfo.value.problems
    assert len(problems) == 2
    assert any("symbol" in problem for problem in problems)
    assert any("quantity" in problem for problem in problems)


# ---------------------------------------------------------------------------
# Errors: the layer's own hierarchy
# ---------------------------------------------------------------------------
def test_the_hierarchy_is_one_catchable_boundary() -> None:
    assert issubclass(BacktestContractError, BacktestError)
    assert issubclass(BacktestError, Exception)
    assert not issubclass(BacktestError, ValueError)


def test_a_contract_error_without_problems_is_a_plain_message() -> None:
    error = BacktestContractError("invalid BacktestConfig")

    assert error.problems == []
    assert str(error) == "invalid BacktestConfig"


def test_a_contract_error_bullets_each_problem() -> None:
    error = BacktestContractError("invalid Fill", ["first", "second"])

    assert error.problems == ["first", "second"]
    assert str(error) == "invalid Fill\n  - first\n  - second"


def test_a_contract_error_copies_the_problems_it_was_given() -> None:
    problems = ["one"]
    error = BacktestContractError("invalid BacktestConfig", problems)

    problems.append("two")

    assert error.problems == ["one"]


def test_the_errors_module_declares_exactly_the_five_errors() -> None:
    assert backtest_sys.errors.__all__ == [
        "BacktestContractError",
        "BacktestError",
        "BacktestMarketDataError",
        "ExecutionError",
        "RebalancePolicyError",
    ]
    assert backtest_sys.errors.BacktestError is BacktestError
    assert backtest_sys.errors.BacktestContractError is BacktestContractError
    assert backtest_sys.errors.BacktestMarketDataError is backtest_sys.BacktestMarketDataError
    assert backtest_sys.errors.ExecutionError is backtest_sys.ExecutionError
    assert backtest_sys.errors.RebalancePolicyError is backtest_sys.RebalancePolicyError


# ---------------------------------------------------------------------------
# Package surface: the root exports the contract, the rule and nothing else
# ---------------------------------------------------------------------------
def test_the_package_exports_exactly_the_contract() -> None:
    assert backtest_sys.__all__ == [
        "BAR_FIELDS",
        "DEFAULT_CALENDAR_NAME",
        "NAV_TOLERANCE",
        "ORDER_SIDES",
        "BacktestConfig",
        "BacktestContractError",
        "BacktestError",
        "BacktestMarketDataError",
        "BacktestMarketDataReader",
        "BacktestPosition",
        "BacktestResult",
        "BacktestState",
        "DailyPortfolioValue",
        "ExecutionError",
        "ExecutionModel",
        "Fill",
        "FixedBpsExecutionModel",
        "OrderIntent",
        "ParquetBacktestMarketDataReader",
        "RebalanceEvent",
        "RebalancePolicy",
        "RebalancePolicyError",
        "RebalanceSchedule",
        "__version__",
        "rebalance_schedule",
    ]
    for name in backtest_sys.__all__:
        assert hasattr(backtest_sys, name), name


def test_the_package_reexports_the_contract_objects_themselves() -> None:
    assert backtest_sys.BacktestConfig is BacktestConfig
    assert backtest_sys.BacktestPosition is BacktestPosition
    assert backtest_sys.BacktestState is BacktestState
    assert backtest_sys.OrderIntent is OrderIntent
    assert backtest_sys.Fill is Fill
    assert backtest_sys.DailyPortfolioValue is DailyPortfolioValue
    assert backtest_sys.BacktestResult is BacktestResult
    assert backtest_sys.BacktestContractError is BacktestContractError
    assert backtest_sys.BacktestError is BacktestError
    assert backtest_sys.ORDER_SIDES == ORDER_SIDES
    assert backtest_sys.DEFAULT_CALENDAR_NAME == DEFAULT_CALENDAR_NAME
    assert backtest_sys.NAV_TOLERANCE == NAV_TOLERANCE
    assert contracts_module.__all__ == [
        "DEFAULT_CALENDAR_NAME",
        "NAV_TOLERANCE",
        "ORDER_SIDES",
        "BacktestConfig",
        "BacktestPosition",
        "BacktestResult",
        "BacktestState",
        "DailyPortfolioValue",
        "Fill",
        "OrderIntent",
    ]


def test_the_package_ships_no_engine_no_ledger_and_no_report() -> None:
    for absent in (
        "BacktestLedger",
        "BacktestEngine",
        "BacktestReport",
        "run_backtest",
        "default_rebalance_policy",
        "load_prices",
        "save_result",
    ):
        assert not hasattr(backtest_sys, absent), absent


def test_the_package_is_made_of_its_five_modules_and_the_errors() -> None:
    package = Path(backtest_sys.__file__).parent

    assert sorted(path.name for path in package.glob("*.py")) == [
        "__init__.py",
        "contracts.py",
        "errors.py",
        "execution.py",
        "market_data.py",
        "rebalance_policy.py",
    ]
    assert imported_modules(Path(backtest_sys.__file__)) == {
        "__future__",
        "backtest_sys.contracts",
        "backtest_sys.errors",
        "backtest_sys.execution",
        "backtest_sys.market_data",
        "backtest_sys.rebalance_policy",
    }
    loaded = {name for name in sys.modules if name.startswith("backtest_sys.")}
    assert loaded <= {
        "backtest_sys.contracts",
        "backtest_sys.errors",
        "backtest_sys.execution",
        "backtest_sys.market_data",
        "backtest_sys.rebalance_policy",
    }, loaded


def test_the_contract_uses_the_projects_ticker_pattern() -> None:
    assert contracts_module.SYMBOL_PATTERN is SYMBOL_PATTERN

    with pytest.raises(BacktestContractError) as excinfo:
        make_position("AA PL")

    assert SYMBOL_PATTERN in str(excinfo.value)


def test_the_public_objects_all_carry_a_docstring() -> None:
    for name in backtest_sys.__all__:
        value = getattr(backtest_sys, name)
        if isinstance(value, (str, float, tuple)):
            continue
        assert value.__doc__, name


# ---------------------------------------------------------------------------
# Architectural boundary: no reader, no runner, no rule, no metric
# ---------------------------------------------------------------------------
#: Modules the contract may import: the standard library it needs, the project's
#: single ticker pattern, and its own modules -- nothing else.  ``data_sys.schema``
#: is the sanctioned ticker source; importing it is the *only* project import here.
ALLOWED_IMPORTS: frozenset[str] = frozenset(
    {
        "__future__",
        "backtest_sys.contracts",
        "backtest_sys.errors",
        "backtest_sys.execution",
        "backtest_sys.market_data",
        "backtest_sys.rebalance_policy",
        "collections.abc",
        "data_sys.schema",
        "dataclasses",
        "datetime",
        "math",
        "numbers",
        "re",
        "typing",
    }
)

#: Text that must never appear in the contract modules' *code*.  Docstrings are
#: blanked first, so prose may still name what the layer deliberately avoids.
FORBIDDEN_TEXT: tuple[str, ...] = (
    "import numpy",
    "import pandas",
    "import pyarrow",
    "import duckdb",
    "import yfinance",
    "import httpx",
    "import requests",
    "import socket",
    "exchange_calendars",
    "data_sys.storage",
    "data_sys.providers",
    "data_sys.update",
    "factor_sys.runner",
    "factor_sys.factor_store",
    "strategy_sys.runner",
    "strategy_sys.registry",
    "portfolio_sys.contracts",
    "portfolio_sys.registry",
    "portfolio_sys.transition",
    "PortfolioConstructor",
    "build_portfolio_transition",
    "read_parquet",
    "to_parquet",
    "urlopen",
    "open(",
)

#: Identifiers the contract must never use: a rule, an engine, a ledger or a
#: performance metric belongs to a later task and a later module.
FORBIDDEN_IDENTIFIERS: frozenset[str] = frozenset(
    {
        "RebalancePolicy",
        "build_report",
        "commission_model",
        "drawdown",
        "engine",
        "ledger",
        "read_prices",
        "rebalance",
        "returns",
        "sharpe",
        "slippage_model",
        "trading_calendar",
        "turnover",
    }
)


def contract_modules() -> list[Path]:
    """The three modules that make up the backtest contract layer."""
    package = Path(backtest_sys.__file__).parent
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


def test_the_contract_imports_only_the_standard_library_and_its_own_modules() -> None:
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
    banned = {"open", "print", "input", "eval", "exec", "compile", "__import__"}

    for module in contract_modules():
        tree = ast.parse(module.read_text(encoding="utf-8"))
        called = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert called & banned == set(), (module.name, sorted(called & banned))


def test_the_contract_never_names_a_rule_an_engine_or_a_metric() -> None:
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


def test_the_contract_divides_nothing() -> None:
    """Every amount is stated, summed or multiplied -- never averaged or scaled."""
    forbidden = (ast.Div, ast.FloorDiv, ast.Pow, ast.Mod)

    for module in contract_modules():
        tree = ast.parse(module.read_text(encoding="utf-8"))
        operators = [type(node.op) for node in ast.walk(tree) if isinstance(node, ast.BinOp)]
        assert [operator for operator in operators if operator in forbidden] == [], module.name
