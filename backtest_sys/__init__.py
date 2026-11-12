"""Backtest system: the objects one replay of history exchanges.

Layering of the project::

    data_sys       market bars and the universe -- the only layer that knows files
        |
    factor_sys     the factor contract, its runner, its panel store and analytics
        |
    strategy_sys   scores and selects symbols from those panels
        |
    portfolio_sys  how much weight each selected symbol should carry
        |
    backtest_sys   what the account actually did on the timeline      <- this package
        |
    (reporting)    returns, drawdown and turnover                     -- a later task

This package is the **backtest data contract**, the **rebalance calendar** and the
**execution rule**: the objects a rebalance policy, an execution model, a ledger, an
engine and a reporting layer exchange, the one function that says *which session
decides and which later session fills*, and the one model that says *at which price
a decided order fills and what that costs*.  Everything here is a description of the
timeline, and of the price and cost of one trade on it, before anything is replayed.

``BacktestConfig``
    when a replay runs and with how much money: an inclusive date range, the
    starting cash and the calendar the range is measured on
``BacktestState``
    what the account **owns** at one close: cash plus a symbol -> share-count tuple,
    sorted by symbol, holding no price, no value and no weight
``OrderIntent``
    what a decision wants to trade: one symbol, a ``"BUY"``/``"SELL"`` side, a share
    count and a fill date **strictly after** the decision date
``Fill``
    one order that executed: the order it quotes, its date, price, size, cost, and
    the two derived amounts ``gross_notional`` and ``cash_change``
``DailyPortfolioValue``
    what the account was **worth** at one close: cash, market value and a ``nav``
    that must equal the other two
``BacktestResult``
    the finished run: its configuration, its closing state, its valuations and its
    fills, both histories ordered as they happened and a closing state that *is* the
    last close -- same session, same cash
``RebalancePolicy``
    the rule: ``weekly`` / ``biweekly`` / ``monthly``, an anchor weekday (``"THU"``
    by default, and a default policy says so), the calendar name, how many sessions
    after the decision the trade executes, and the price convention (``"open"``)
``RebalanceEvent``
    one calendar fact: *this session's close decides, that session's open fills*
``RebalanceSchedule``
    a window resolved into ``decision_dates``, the ``events`` it can complete and
    the ``unexecuted_decision_dates`` it cannot
``rebalance_schedule(...)``
    the function that builds those three from a window and a policy
``BacktestMarketDataReader``
    the read-only market-data interface a replay asks for bars through: symbols,
    an inclusive window, the fields it needs, and nothing about any universe
``ParquetBacktestMarketDataReader``
    the v1 implementation: the published Hive dataset under
    ``data/standardized/market_bars``, read through
    :func:`data_sys.storage.read_partitioned` and filtered here
``ExecutionModel``
    the interface a replay fills orders through: one event, that event's orders and
    the real opens of its session in; exactly one ``Fill`` per order out, in the
    order they were given
``FixedBpsExecutionModel``
    the v1 implementation: the raw open moved by a fixed number of basis points a
    side -- up for a buy, down for a sell -- plus one flat commission per order,
    with the slippage already inside the price
``BacktestError`` / ``BacktestContractError`` / ``RebalancePolicyError`` /
``BacktestMarketDataError`` / ``ExecutionError``
    the backtest layer's own exception hierarchy: every contract problem collected in
    one error, in the style of the layers below, one error for a legal rule that the
    calendar cannot honour, one for a read of the local bars that cannot be honoured,
    and one for legal orders that a model cannot fill at the open the event names

The timing this contract assumes is the project's default, expressed by shape rather
than by a flag::

    decision at the close  ->  fill at the next open  ->  valuation at that close

An ``OrderIntent`` therefore cannot be scheduled for its own decision date, and a
``Fill`` cannot be dated anywhere but its scheduled session.  ``rebalance_schedule``
answers *which* session comes next -- by reusing
:func:`factor_sys.factor_schedule.decision_schedule` for the decision and
:mod:`data_sys.market_calendar` for the sessions, never a second holiday list -- but
it still reads no price, generates no order and keeps no ledger.  Those are the next
tasks, and they implement *against* these objects.

The reader speaks the same timeline in prices, and keeps three of them apart:

* ``open`` -- the real, unadjusted open: the candidate price an execution fills at;
* ``close`` -- the real, unadjusted close: the candidate price a valuation uses;
* ``adjusted_close`` -- research only, because a back-adjusted series is not a price
  anybody traded and not a cash amount anybody booked.

It hands every value through exactly as the dataset stored it and derives nothing,
so a backtest cites prices that were really there.  There is no dividend, split or
delisting event data in this project yet, so no module here adjusts a holding: a
real corporate-action record is a data-source problem, and it is a separate task.

Three statements, never confused:

* ``TargetPortfolio`` (``portfolio_sys``) -- how much **weight** each holding should
  carry;
* ``BacktestState`` (here) -- how many **shares** the account actually holds;
* ``DailyPortfolioValue`` (here) -- what those holdings were **worth** at a close.

A weight is not a share count and a share count is not a value; keeping the three
apart is what lets a backtest be audited line by line.

What this package deliberately does **not** contain: a trading calendar of its own
(it names one and asks ``data_sys`` for the sessions), a corporate-action model, an
order generator, a position or cash ledger, a backtest engine, a command-line entry
point, persistence, and any performance metric -- no return, no Sharpe ratio, no
maximum drawdown, no turnover.  The reader above reads bars and prices nothing; the
execution model above prices one trade and reads nothing; neither is a valuation.
Those arrive later, each with its own task, and none of them changes the objects
here.

The execution rule is deliberately the narrowest thing that can state friction::

    event + orders + the real opens of the session  ->  one Fill per order

It is handed the prices it needs, so the friction rule can be tested without a
Parquet file and the reader can be tested without a fill.  It does not generate
orders, does not size them, does not check the account's cash or holdings, does not
apply the state change and does not value anything: those are the order planner and
the ledger, which come next.

Examples
--------
>>> config = BacktestConfig("2024-01-04", "2024-01-05", 1_000.0)
>>> config.calendar_name
'XNYS'
>>> state = BacktestState("2024-01-05", 1_000.0, ())
>>> state.quantities_by_symbol
{}
>>> OrderIntent("rb-0001", "2024-01-04", "2024-01-05", "AAPL", "BUY", 2.0).is_buy
True
>>> DailyPortfolioValue("2024-01-05", 1_000.0, 0.0, 1_000.0).nav
1000.0
>>> schedule = rebalance_schedule("2024-11-25", "2024-11-29")
>>> schedule.events
(RebalanceEvent(2024-11-27 -> 2024-11-29),)
>>> event = RebalanceEvent("2024-01-04", "2024-01-05")
>>> order = OrderIntent("rb-0001", "2024-01-04", "2024-01-05", "AAPL", "BUY", 10.0)
>>> FixedBpsExecutionModel().fill_orders(
...     event=event, orders=(order,), open_prices={"AAPL": 100.0}
... )[0].fill_price
100.05
"""

from __future__ import annotations

from backtest_sys.contracts import (
    DEFAULT_CALENDAR_NAME,
    NAV_TOLERANCE,
    ORDER_SIDES,
    BacktestConfig,
    BacktestPosition,
    BacktestResult,
    BacktestState,
    DailyPortfolioValue,
    Fill,
    OrderIntent,
)
from backtest_sys.errors import (
    BacktestContractError,
    BacktestError,
    BacktestMarketDataError,
    ExecutionError,
    RebalancePolicyError,
)
from backtest_sys.execution import ExecutionModel, FixedBpsExecutionModel
from backtest_sys.market_data import (
    BAR_FIELDS,
    BacktestMarketDataReader,
    ParquetBacktestMarketDataReader,
)
from backtest_sys.rebalance_policy import (
    RebalanceEvent,
    RebalancePolicy,
    RebalanceSchedule,
    rebalance_schedule,
)

__version__ = "0.1.0"

__all__ = [
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
