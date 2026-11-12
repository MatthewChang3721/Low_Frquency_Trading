"""Exception hierarchy for the backtest system.

The backtest system is a **separate boundary** from the data, factor, strategy and
portfolio systems: a ticket that cannot be executed is not a
:class:`~portfolio_sys.errors.PortfolioContractError` and not a
:class:`~strategy_sys.errors.StrategyError`, so none of those layers has to know
that a backtest exists, and a caller can catch everything this layer raises
through :class:`BacktestError`.

Five errors cover the whole of this layer:

``BacktestError``
    the base class: everything the backtest layer raises is one of these, so
    ``except BacktestError`` is always safe at a call site.
``BacktestContractError``
    one backtest object broke the shared contract, and every problem found while
    checking it travels back in :attr:`~BacktestContractError.problems`.
``BacktestMarketDataError``
    a read of the local bars cannot be honoured -- an illegal request, a dataset
    that is missing or empty, a requested field the data does not provide, a
    ``date`` column that cannot be read, or two bars for the same session -- and
    every reason found travels back in
    :attr:`~BacktestMarketDataError.problems`.
``RebalancePolicyError``
    a legal rule could not be resolved -- the calendar it names cannot be loaded,
    or the sessions it asks for cannot be found -- and the failures of the layers
    it borrows (:mod:`factor_sys.factor_schedule`,
    :mod:`data_sys.market_calendar`) are wrapped rather than leaked.
``ExecutionError``
    a set of legal orders cannot be filled by an execution model at the open the
    event names -- an order decided or scheduled for another session, two orders
    for one symbol, an order symbol the price table does not price, a price that
    is not a price, or a fill the model computed that the ``Fill`` contract
    refuses -- and every reason found travels back in
    :attr:`~ExecutionError.problems`.

Why a legal object gets a second error
--------------------------------------
Malformed and unsatisfiable are different statements.  ``BacktestContractError``
means "this field is not the vocabulary the contract defines" -- a negative delay,
an unknown price convention, a timestamp where a date belongs -- and the fix is in
the object.  ``RebalancePolicyError`` means "the object is fine, but the world it
describes is not available": ``calendar_name="XTKS"`` is a well-formed name this
environment cannot load, and a window the calendar does not cover is a legitimate
request with an empty answer.  Folding the second into the first would tell a
caller to edit a field that is already correct.

``ExecutionError`` is the same kind of statement one step further along: a negative
slippage is a malformed model and is refused as a contract error, while an order
scheduled for a session the event does not fill is a perfectly well-formed order
and a perfectly well-formed event that simply do not describe one trade.  Telling
that caller to "fix a field" would be wrong -- the fix is in the request.  The same
applies to a fill a model computed: when the ``Fill`` contract refuses it, the model
re-raises the failure as an ``ExecutionError`` with the original kept as
``__cause__``, so a caller never receives a contract error that has lost the order
it was about.

The same split exists one layer down, in
:class:`~portfolio_sys.errors.PortfolioConstructionError` and
:class:`~factor_sys.errors.FactorScheduleError`, and it is copied here for the same
reason: the caller decides differently.  A wrapped failure keeps its ``__cause__``,
so the calendar's own message is still readable, but a caller of the backtest layer
never has to import a factor-layer or data-layer error to catch it.

Why problems are collected, not raised one at a time
----------------------------------------------------
A malformed :class:`~backtest_sys.contracts.BacktestResult` usually has more than
one thing wrong with it -- a valuation that does not add up *and* a fill dated
outside the window.  Reporting only the first would mean one edit-run-edit cycle
per defect, so every object validates everything it can and raises once, with one
bullet per finding, in the order the checks ran.

Anything that is *not* a backtest error -- a bare ``ValueError`` from a
conversion, a ``TypeError`` from a mis-typed argument, a ``KeyError`` on a missing
symbol -- never reaches a caller of this layer: the contract validates its inputs
and reports what is wrong in its own vocabulary.

Examples
--------
>>> error = BacktestContractError(
...     "invalid DailyPortfolioValue",
...     ["nav must equal cash + market_value, got 995.0"],
... )
>>> print(error)
invalid DailyPortfolioValue
  - nav must equal cash + market_value, got 995.0
>>> error.problems
['nav must equal cash + market_value, got 995.0']
"""

from __future__ import annotations

from collections.abc import Sequence


class BacktestError(Exception):
    """Base class for every error raised by the backtest system."""


def _with_problems(message: str, problems: Sequence[str]) -> str:
    """Append collected problems to ``message`` in the project's bullet style.

    The format deliberately matches the data, factor, strategy and portfolio
    systems' own helper: one error reads the same way wherever it comes from.
    The function is re-declared here instead of imported, so this package never
    depends on another package's private name.
    """
    if not problems:
        return message
    return f"{message}\n  - " + "\n  - ".join(problems)


class BacktestContractError(BacktestError):
    """Raised when a backtest object violates the shared contract.

    The message names the object that failed (``"invalid Fill"``) and every
    bullet names one field, one entry or one pair that broke a rule -- so a caller
    can point at the value to fix without re-running to find the next problem.

    Parameters
    ----------
    message:
        What was being validated, in the project's ``"invalid <Object>"`` style.
    problems:
        Every violation found while validating that object, in the order the
        checks ran.  Omitted or empty for a rejection that has no detail.

    Attributes
    ----------
    problems:
        Human readable descriptions of every violation found while validating
        one object, in the order the checks ran.

    Examples
    --------
    >>> error = BacktestContractError("invalid BacktestState")
    >>> error.problems
    []
    >>> str(error)
    'invalid BacktestState'
    >>> error = BacktestContractError("invalid Fill", ["fill_date is not the scheduled date"])
    >>> error.problems
    ['fill_date is not the scheduled date']
    >>> isinstance(error, BacktestError)
    True
    """

    def __init__(self, message: str, problems: Sequence[str] | None = None) -> None:
        self.problems: list[str] = list(problems) if problems else []
        super().__init__(_with_problems(message, self.problems))


class BacktestMarketDataError(BacktestError):
    """Raised when a read of the local market bars cannot be honoured.

    Two sides of a read can fail it, and both are reported here:

    * the **request** -- ``symbols`` that are not a tuple of tickers, a window
      that is not two plain calendar days, a window that runs backwards, a field
      list that is empty, repeated, reserved or unknown;
    * the **dataset** -- a root that does not exist, is not a directory or holds
      no parquet file, a requested field the published columns do not provide, a
      ``date`` column that cannot be read, or two bars for the same
      ``(symbol, date)``.

    The second group is the reason this error exists instead of a bare
    ``KeyError``/``FileNotFoundError``/pyarrow message: a caller of the backtest
    layer must be able to tell "my request was wrong" from "the published data
    cannot answer it" without importing the storage layer's vocabulary.

    What is **not** an error
    ------------------------
    A symbol with no partition, a window that matches no row, a history shorter
    than the window and a gap in the middle of a history are *answers*: the read
    returns the rows that exist (or none) with the requested columns.  Whether a
    symbol is eligible or tradable on a session is decided by the execution and
    ledger layers that come next, never by a reader guessing.

    Parameters
    ----------
    message:
        What was being read, in the project's ``"invalid read request"`` style.
    problems:
        Every reason the read was refused, collected the way
        :class:`BacktestContractError` collects them, so one failed read names
        all of them at once.

    Attributes
    ----------
    problems:
        Human readable descriptions of every reason the read was refused, in the
        order the checks ran.

    Examples
    --------
    >>> error = BacktestMarketDataError(
    ...     "invalid read request",
    ...     ["duplicate symbol 'AAPL'"],
    ... )
    >>> print(error)
    invalid read request
      - duplicate symbol 'AAPL'
    >>> error.problems
    ["duplicate symbol 'AAPL'"]
    >>> isinstance(error, BacktestError)
    True
    >>> BacktestMarketDataError("invalid read request").problems
    []
    """

    def __init__(self, message: str, problems: Sequence[str] | None = None) -> None:
        self.problems: list[str] = list(problems) if problems else []
        super().__init__(_with_problems(message, self.problems))


class RebalancePolicyError(BacktestError):
    """Raised when a rebalance rule, or the schedule it implies, cannot be honoured.

    The rule itself was well formed -- :class:`~backtest_sys.rebalance_policy.RebalancePolicy`
    only stores a calendar *name* -- so this error is about the world, not the
    object: the named calendar cannot be loaded, or
    :func:`~factor_sys.factor_schedule.decision_schedule` cannot resolve the cadence
    on it for the requested window.

    Both of those are failures of a borrowed layer, and both are re-raised here
    with ``raise ... from exc``, so :attr:`__cause__` still carries the calendar's or
    the factor schedule's own message while a caller of the backtest layer only has
    to know :class:`BacktestError`.

    Notes
    -----
    * A window the calendar does not cover is **not** an error: it selects no
      decision, which is a legal, empty schedule.
    * A window that runs backwards, or a date that is not a plain ISO day, is refused
      with this error too: the defect arrives through
      :func:`~backtest_sys.rebalance_policy.rebalance_schedule` rather than through an
      object's constructor, so there is no half-built object to hand a
      :class:`BacktestContractError` back for.

    Examples
    --------
    >>> error = RebalancePolicyError("could not load the trading calendar 'XTKS'")
    >>> isinstance(error, BacktestError)
    True
    >>> str(error)
    "could not load the trading calendar 'XTKS'"
    """


class ExecutionError(BacktestError):
    """Raised when legal orders cannot be filled by this model at this event's open.

    Every object involved is well formed when this error is raised: the
    :class:`~backtest_sys.contracts.OrderIntent` values passed their own contract,
    the :class:`~backtest_sys.rebalance_policy.RebalanceEvent` passed its own, and
    the model is a legal configuration.  What fails is the *request*, the statement
    that these orders trade in this session at these prices -- so the fix is in the
    request and not in a field.

    Two sides of a fill request can fail it, and both are reported here:

    * the **orders and the event** -- an order decided on another date or scheduled
      for another session than the one the event names, a repeated ``order_id``, two
      orders for one symbol, or an order list that is not a tuple in strictly
      ascending id order;
    * the **prices and the arithmetic** -- a symbol with no open price, a price that
      is not a finite positive number, a key that is not a ticker, a table that is
      not a mapping, a slipped price that overflows, or a
      :class:`~backtest_sys.contracts.Fill` the contract itself refuses.

    The second group is why this error exists instead of letting the ``Fill``
    contract's own error escape: a caller filling five orders should learn "order
    three has no open price" in the vocabulary of execution, and should still find
    the original failure in ``__cause__`` when a fill could not be recorded.

    Parameters
    ----------
    message:
        What was being filled, in the project's ``"invalid fill request"`` style.
    problems:
        Every reason the request was refused, collected the way
        :class:`BacktestContractError` collects them, so one failed request names
        all of them at once.

    Attributes
    ----------
    problems:
        Human readable descriptions of every reason the request was refused, in the
        order the checks ran.

    Examples
    --------
    >>> error = ExecutionError(
    ...     "invalid fill request",
    ...     ["open_prices has no price for 'AAPL'"],
    ... )
    >>> print(error)
    invalid fill request
      - open_prices has no price for 'AAPL'
    >>> error.problems
    ["open_prices has no price for 'AAPL'"]
    >>> isinstance(error, BacktestError)
    True
    >>> ExecutionError("invalid fill request").problems
    []
    """

    def __init__(self, message: str, problems: Sequence[str] | None = None) -> None:
        self.problems: list[str] = list(problems) if problems else []
        super().__init__(_with_problems(message, self.problems))


__all__ = [
    "BacktestContractError",
    "BacktestError",
    "BacktestMarketDataError",
    "ExecutionError",
    "RebalancePolicyError",
]
