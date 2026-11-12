"""Filling an order at the open of its session: the price it gets and the cost it pays.

The one question
----------------
::

    model = FixedBpsExecutionModel(slippage_bps=5.0)
    fills = model.fill_orders(
        event=RebalanceEvent("2024-01-04", "2024-01-05"),
        orders=(order,),
        open_prices={"AAPL": 100.0},
    )

    fills[0].fill_price        # 100.05 -- the raw open, 5 bps worse
    fills[0].gross_notional    # 1000.5 -- 100.05 x 10 shares
    fills[0].cash_change       # -1000.5 -- what the buy took out of cash

*Given legal orders and the real open of the session they were scheduled for, at
which price does each one trade, and what does it cost?*  That is the whole of this
module.  It reads no bar, prices no other session, moves no cash and holds no
position.

Where a fill sits on the timeline
---------------------------------
::

    Thursday close   OrderIntent      what to trade, and for which later session
    Friday open      this module      at which price, and at what cost  -> Fill
    Friday close     (a later task)   what the account is worth

An :class:`~backtest_sys.contracts.OrderIntent` says *what* to trade and *when*; a
:class:`~backtest_sys.rebalance_policy.RebalanceEvent` says *which sessions* are the
decision and the fill.  Neither knows a price, and neither should: a price is
**data**, and data is read by :mod:`backtest_sys.market_data`.  This module sits
between the two and answers the third question -- *at which price, and at what
cost* -- from a plain mapping the caller has already extracted.  Keeping the
mapping as the interface is what makes the friction rule replaceable without
touching a Parquet file, and the reader replaceable without touching the friction
rule.

The arithmetic, stated once
---------------------------
With ``P_open`` the **real, unadjusted open** of the execution session and ``f`` the
:attr:`~FixedBpsExecutionModel.slippage_fraction` -- ``slippage_bps / 10_000``::

    BUY    fill_price = P_open * (1 + f)
    SELL   fill_price = P_open * (1 - f)

every order pays ``commission_per_order`` -- one flat amount per order, neither a
rate nor a share of anything -- and :class:`~backtest_sys.contracts.Fill` derives the
rest from the two numbers it is given::

    gross_notional = fill_price * quantity
    BUY    cash_change = -(gross_notional + commission)
    SELL   cash_change =  gross_notional - commission

At 5 bps and a raw open of 100.00 a ten-share buy fills at 100.05, moves 1000.50 of
shares and takes 1000.50 out of cash; the same sell receives 99.95 a share and
999.50.  A model with ``slippage_bps=0.0`` fills at the raw open exactly, which is
what makes "the open, with no friction" a configuration rather than another model.

Why the slippage lives in the price and not in the cost
-------------------------------------------------------
Slippage *is* a price effect: the decision hoped for one price and the open offered
a worse one.  A buy that fills at 100.05 instead of 100.00 moved ten shares for
1000.50, and 1000.50 is the money that actually left the account.  Booking the 0.50
again as a fee would charge the same friction twice and leave the account poorer
than the market made it -- and a valuation reading ``fill_price`` while a ledger
reading ``transaction_cost`` would disagree about what the trade cost.  So
``transaction_cost`` carries the **explicit commission only** -- an amount a broker
states -- and the slippage stays inside ``fill_price``, where one number answers
"what did this share cost" for both readers.

What a request must satisfy
---------------------------
* ``event`` is exactly one :class:`~backtest_sys.rebalance_policy.RebalanceEvent`:
  one decision session and one execution session.  A request fills one event's
  orders at that event's open, so an order dated for another decision, or scheduled
  for another fill session, belongs to another request.
* ``orders`` is a **tuple** of :class:`~backtest_sys.contracts.OrderIntent`, in
  strictly ascending ``order_id`` order.  An empty tuple is legal and fills nothing
  -- a rebalance that decided to trade nothing is a legal rebalance, and it does not
  need a price.  Every order's ``decision_date`` must be the event's decision and
  its ``scheduled_fill_date`` the event's execution session; ids do not repeat and
  one symbol appears on one order, because two orders for one symbol in one session
  have no fill order this model could choose between.
* ``open_prices`` is a read-only ``Mapping[str, float]`` holding the real,
  unadjusted opens.  Every order's symbol must be there with a finite price
  strictly greater than zero, and the table may carry other symbols as well: the
  caller usually has a whole cross-section, and this model reads only what it needs.
  A key must be a ticker under the project's one rule --
  :data:`data_sys.schema.SYMBOL_PATTERN`, the same pattern the universe, the panels,
  the strategies, the portfolios and :class:`~backtest_sys.contracts.OrderIntent`
  enforce -- so ``"???"``, ``"aapl"`` and ``"AAPL!"`` are refused even on a row no
  order reads, and a value must be a price, because a table holding ``None`` or
  ``nan`` is not a table of prices whichever row it sits in.  Nothing here
  substitutes another convention: the numbers this interface takes are the **real
  unadjusted open**, never a close and never an adjusted series, and the module
  refuses to look one up on its own so that the caller has to say which price a
  trade is measured against.

Every problem in a request is collected into one
:class:`~backtest_sys.errors.ExecutionError`, whose ``problems`` list names each
finding in the order the checks ran -- the same style as every contract in this
package.  A fill that :class:`~backtest_sys.contracts.Fill` itself refuses (for
example a notional that overflows) is re-raised as an
:class:`~backtest_sys.errors.ExecutionError` with the contract's message kept as
``__cause__``, so a caller of this layer never has to catch a second error type to
learn why a trade could not be recorded.

Why it neither reads nor updates anything
-----------------------------------------
The model is handed the price it needs.  It does not open a dataset, call
:class:`~backtest_sys.market_data.BacktestMarketDataReader`, or ask a calendar which
session this is: reading is one decision (which date's bar) and filling is another
(at which price), and folding them together would make a friction rule impossible
to test without a Parquet file and a bar reader impossible to test without a fill.

It also does not check whether the account can afford the order, or holds enough
shares to sell it, and it does not apply the result.  Cash and positions belong to
:class:`~backtest_sys.contracts.BacktestState`, whose owner -- the future ledger --
decides what an unaffordable order means (shrink it? skip it? fail the run?) and
when the state changes.  An execution model that quietly refused an unaffordable
buy would be making a portfolio decision; one that quietly updated the state would
be a second place where the account's cash is written down.  This version fills what
it is given, completely, and records it: no partial fill, no postponed fill, no
cancelled order.

What this module is not
-----------------------
Not a portfolio transition, not a share-count calculator, not a cash or position
ledger, not a valuation, not an engine, not a report and not a command-line tool.
It turns one legal order into one legal :class:`~backtest_sys.contracts.Fill`, and
it is deliberately the last thing that knows nothing about the account.

Examples
--------
The default model charges 5 bps a side and no commission::

>>> model = FixedBpsExecutionModel()
>>> model.slippage_bps, model.commission_per_order, model.slippage_fraction
(5.0, 0.0, 0.0005)

A ten-share buy of AAPL at an open of 100.00, decided on Thursday and filled on
Friday::

>>> event = RebalanceEvent("2024-01-04", "2024-01-05")
>>> order = OrderIntent("rb-0001", "2024-01-04", "2024-01-05", "AAPL", "BUY", 10.0)
>>> fill = model.fill_orders(event=event, orders=(order,), open_prices={"AAPL": 100.0})[0]
>>> fill.fill_price, fill.gross_notional, fill.cash_change
(100.05, 1000.5, -1000.5)
>>> fill.fill_date
datetime.date(2024, 1, 5)

The same order sold receives the open minus the slippage, and a model with no
slippage fills at the open itself::

>>> sell = OrderIntent("rb-0002", "2024-01-04", "2024-01-05", "AAPL", "SELL", 10.0)
>>> model.fill_orders(event=event, orders=(sell,), open_prices={"AAPL": 100.0})[0].cash_change
999.5
>>> plain = FixedBpsExecutionModel(slippage_bps=0.0)
>>> plain.fill_orders(event=event, orders=(order,), open_prices={"AAPL": 100.0})[0].fill_price
100.0
"""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Any

from backtest_sys.contracts import Fill, OrderIntent
from backtest_sys.errors import BacktestContractError, ExecutionError
from backtest_sys.rebalance_policy import RebalanceEvent
from data_sys.schema import SYMBOL_PATTERN

# ---------------------------------------------------------------------------
# The vocabulary of a fill
# ---------------------------------------------------------------------------
#: One basis point as a fraction: ``1 bp = 1 / 10_000`` of a price.  A basis point
#: is a *rate*, so the divisor is a plain float and no rounding happens anywhere.
BASIS_POINTS_PER_UNIT: float = 10_000.0

#: The slippage a model applies when none is given: 5 bps a side, the project's
#: default -- five hundredths of one percent of the raw open.
DEFAULT_SLIPPAGE_BPS: float = 5.0

#: The commission a model charges when none is given: zero per order, the US equity
#: convention this first version assumes, so by default friction is slippage alone.
DEFAULT_COMMISSION_PER_ORDER: float = 0.0

#: The largest slippage a model may apply, in basis points (100% of the price).
#:
#: ``10_000`` is the boundary and it is **exclusive**: a sell fills at
#: ``open * (1 - bps / 10_000)``, so at ``10_000`` bps it would receive zero -- and
#: above it, less than zero, which is not a price a share can change hands at.
MAX_SLIPPAGE_BPS: float = BASIS_POINTS_PER_UNIT

#: The project's one ticker rule, borrowed rather than restated.
#:
#: The universe, the panels, the strategies, the portfolios and
#: :mod:`backtest_sys.contracts` all read :data:`data_sys.schema.SYMBOL_PATTERN`; a
#: second pattern written here would be the first step towards a price table that
#: prices a ticker nothing else in the project can store.
_SYMBOL_RE = re.compile(SYMBOL_PATTERN)


# ---------------------------------------------------------------------------
# Validation: numbers, keys and tables -- everything below runs before a fill
# ---------------------------------------------------------------------------
def _finite_number(value: Any, label: str) -> tuple[float | None, str | None]:
    """``value`` as a float when it is a finite real number, otherwise why not.

    ``bool`` is refused even though it is an ``int``: ``True`` cannot mean "five
    basis points".  A string is refused rather than parsed, and ``None``, ``NaN``
    and ``+-inf`` are refused because none of them is a rate a price can be
    adjusted by, or a price a share can trade at.
    """
    if isinstance(value, bool) or not isinstance(value, Real):
        return None, (
            f"{label} must be a real number, got {type(value).__name__}: {value!r}; a bool, a "
            "string or None is refused rather than converted, because a rate has to mean exactly "
            "the number it is"
        )
    number = float(value)
    if not math.isfinite(number):
        return None, (
            f"{label} must be finite, got {number!r}: NaN and +-inf cannot adjust a price or "
            "state one, so they never reach a fill"
        )
    return number, None


def _positive_number(value: Any, label: str) -> tuple[float | None, str | None]:
    """A finite real number strictly greater than zero, or why not."""
    number, problem = _finite_number(value, label)
    if problem is not None:
        return None, problem
    if number <= 0.0:
        return None, (
            f"{label} must be greater than zero, got {number!r}: a price of zero or less is not "
            "a price, so no share changes hands at it"
        )
    return number, None


def _duplicates(values: Iterable[str]) -> list[tuple[str, int]]:
    """Each value that appears more than once, with how often, in first-seen order."""
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return [(value, count) for value, count in counts.items() if count > 1]


def _key_problem(key: Any) -> str | None:
    """Why ``key`` is not a ticker a price may be filed under, or ``None``.

    The rule is the project's, not a second one: :data:`data_sys.schema.SYMBOL_PATTERN`
    -- the pattern the universe, the panels, the strategies, the portfolios and
    :class:`~backtest_sys.contracts.OrderIntent` already enforce -- is imported and
    applied here, exactly as :mod:`backtest_sys.contracts` applies it, rather than
    paraphrased as a shape test of this module's own.  A key is checked whichever row
    it sits on, not only when an order happens to read it: a table that prices
    ``"???"`` or ``"aapl"`` disagrees with the data it was extracted from, and if that
    disagreement were tolerated it would stay invisible until the day a universe
    change made one of those rows tradable.
    """
    if not isinstance(key, str):
        return (
            f"open_prices key {key!r} must be a ticker string, got {type(key).__name__}: prices "
            "are filed under the symbol they belong to"
        )
    # ``fullmatch`` rather than ``match``, for the reason the contract layer gives:
    # the pattern is the project's own and anchored, but a trailing newline must not
    # slip through as a second ticker.
    if _SYMBOL_RE.fullmatch(key) is not None:
        return None
    if not key:
        hint = "an empty string is not a ticker"
    elif key != key.strip():
        hint = (
            "it is padded with whitespace, and keys are matched verbatim, so its own symbol "
            "would never be found"
        )
    elif len(key.split()) != 1:
        hint = "a symbol is one word, and a key with whitespace inside it prices nothing"
    else:
        hint = (
            "the data stores a symbol in upper case with only letters, digits, '.' and '-', "
            "so a lower-case or punctuated variant names no symbol at all"
        )
    return (
        f"open_prices key {key!r} must be a ticker matching {SYMBOL_PATTERN} (letters, digits, "
        f"'.' and '-'), and it is not: {hint}"
    )


def _checked_orders(orders: Any) -> tuple[tuple[OrderIntent, ...] | None, list[str]]:
    """The request's orders as a tuple, plus every reason they are not one.

    ``None`` comes back -- with a problem -- when the field is not a tuple at all,
    so a caller can skip the checks that only make sense on a real sequence.
    """
    if not isinstance(orders, tuple):
        return None, [
            f"orders must be a tuple[OrderIntent, ...], got {type(orders).__name__}: a fill "
            "request is an immutable sequence of intents, so a list, a set or a generator is "
            "refused; declare (..., )"
        ]

    problems: list[str] = [
        f"orders entry {order!r} must be an OrderIntent, got {type(order).__name__}: this model "
        "fills decided orders, and it cannot invent the symbol, the side or the size of anything "
        "else"
        for order in orders
        if not isinstance(order, OrderIntent)
    ]
    return tuple(order for order in orders if isinstance(order, OrderIntent)), problems


def _checked_price_table(
    open_prices: Any,
) -> tuple[Mapping[str, Any] | None, list[str]]:
    """The price table as given, plus every reason it is not a table of prices.

    A key must match the project's ticker rule and a value must be a finite price
    above zero, for every row of the table and not only the rows an order reads, so
    the caller cannot hand over a column of a dataframe with holes in it -- or with a
    ``"???"`` row no universe would ever produce -- and learn about it only when a
    trade happens to land on that row.
    """
    if not isinstance(open_prices, Mapping):
        return None, [
            "open_prices must be a Mapping[str, float], got "
            f"{type(open_prices).__name__}: the caller hands over the opens it already extracted, "
            "so the lookup is a mapping -- never a reader, a file or a dataframe"
        ]

    problems: list[str] = []
    for key in open_prices:
        key_problem = _key_problem(key)
        if key_problem is not None:
            problems.append(key_problem)
            continue
        _, value_problem = _positive_number(open_prices[key], f"open_prices[{key!r}]")
        if value_problem is not None:
            problems.append(value_problem)
    return open_prices, problems


def _agreement_problems(orders: tuple[OrderIntent, ...], event: RebalanceEvent) -> list[str]:
    """Why ``orders`` is not one ordered, unambiguous set of this event's orders."""
    problems: list[str] = []

    repeated_ids = _duplicates(order.order_id for order in orders)
    problems.extend(
        f"order_id {order_id!r} appears on {count} orders: one id names one order, and a repeated "
        "one would let two fills claim the same ticket"
        for order_id, count in repeated_ids
    )

    repeated_symbols = _duplicates(order.symbol for order in orders)
    problems.extend(
        f"symbol {symbol!r} appears on {count} orders: one session, one order per symbol -- a buy "
        "and a sell of one symbol in the same rebalance have no fill order this model could "
        "choose between"
        for symbol, count in repeated_symbols
    )

    for order in orders:
        if order.decision_date != event.decision_date:
            problems.append(
                f"order {order.order_id!r} was decided on {order.decision_date.isoformat()}, but "
                f"the event decides on {event.decision_date.isoformat()}: one fill request carries "
                "one decision's orders"
            )
        if order.scheduled_fill_date != event.execution_date:
            problems.append(
                f"order {order.order_id!r} is scheduled to fill on "
                f"{order.scheduled_fill_date.isoformat()}, but the event fills on "
                f"{event.execution_date.isoformat()}: this model fills one event's orders at that "
                "event's open, never on another session"
            )

    # A duplicate already breaks strict ascent, so the ordering complaint is only
    # added when the ids are otherwise unique -- one cause, one bullet.
    if not repeated_ids:
        identifiers = [order.order_id for order in orders]
        if identifiers != sorted(identifiers):
            offender = next(
                f"{identifiers[index]!r} comes after {identifiers[index - 1]!r} but sorts before it"
                for index in range(1, len(identifiers))
                if identifiers[index] < identifiers[index - 1]
            )
            problems.append(
                "orders must be ordered by strictly ascending order_id, got "
                f"{identifiers!r}: {offender}; the fill history is a record, so the order is "
                "refused rather than re-sorted -- declare the orders sorted by id"
            )
    return problems


def _price_problems(orders: tuple[OrderIntent, ...], open_prices: Mapping[str, Any]) -> list[str]:
    """Every ordered symbol the table does not price, once per symbol."""
    problems: list[str] = []
    unpriced: set[str] = set()
    for order in orders:
        if order.symbol in open_prices or order.symbol in unpriced:
            continue
        unpriced.add(order.symbol)
        problems.append(
            f"open_prices has no price for {order.symbol!r}, the symbol of order "
            f"{order.order_id!r}: a fill needs the real unadjusted open of the session it "
            "trades in, and this model refuses to price a trade from any other number"
        )
    return problems


# ---------------------------------------------------------------------------
# The interface every execution model implements
# ---------------------------------------------------------------------------
class ExecutionModel(ABC):
    """The stable interface of a backtest execution model: orders in, fills out.

    One method and one promise: every legal order in ``orders`` becomes exactly one
    :class:`~backtest_sys.contracts.Fill`, in the order it was given, at a price and
    a cost this model decides.  A model may not drop an order, split one, postpone
    one or add one -- a request it cannot honour is refused with
    :class:`~backtest_sys.errors.ExecutionError`, so a caller never has to compare
    the order history with the fill history to find out what happened.

    The interface is stated in shares, not in weights, and in prices, not in
    returns: an order already says how many shares it moves, and this layer only
    says at which price they move and what the trade costs.

    Implementations are expected to be immutable values -- the reference model is a
    frozen dataclass -- so one model can be reused for every event of a replay.
    """

    __slots__ = ()

    @abstractmethod
    def fill_orders(
        self,
        *,
        event: RebalanceEvent,
        orders: tuple[OrderIntent, ...],
        open_prices: Mapping[str, float],
    ) -> tuple[Fill, ...]:
        """Fill ``orders`` at the open of ``event``'s execution session.

        Parameters
        ----------
        event:
            The :class:`~backtest_sys.rebalance_policy.RebalanceEvent` this request
            belongs to: one decision session and the session whose open fills it.
            Every order must agree with both dates.
        orders:
            The decided orders, as a **tuple**, in strictly ascending ``order_id``
            order.  An empty tuple is legal and must be answered with an empty tuple
            -- a rebalance that traded nothing needs no price at all.
        open_prices:
            A read-only ``Mapping`` of ``symbol -> price``: the real, unadjusted open
            of the execution session.  Every ordered symbol must be present with a
            finite price strictly greater than zero.  Other symbols may be present --
            the caller usually has a whole cross-section -- and no order needs them,
            but every key of the table must still be a ticker under the project's one
            rule and every value a price, so a table of ``"???"`` rows or ``None``
            holes is refused whichever row an order reads.  The mapping is never
            modified.

        Returns
        -------
        tuple[Fill, ...]
            One fill per order, in the order the orders were given, each dated
            exactly ``event.execution_date`` and quoting its order back verbatim.

        Raises
        ------
        backtest_sys.errors.ExecutionError
            If the event is not a ``RebalanceEvent``, the orders are not an
            unambiguous tuple of this event's orders, the price table is not a table
            of prices, or a fill this model computed cannot be recorded.  Every
            request problem is collected into one error.
        """


# ---------------------------------------------------------------------------
# The reference model: fixed slippage, flat commission
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FixedBpsExecutionModel(ExecutionModel):
    """Fill at the raw open moved by a fixed number of basis points, plus a commission.

    This is the project's first execution model and deliberately the simplest one
    that is still honest about friction: the price a trade gets is the session's
    **real unadjusted open**, worsened by ``slippage_bps`` in the direction that
    costs the account money -- up for a buy, down for a sell -- and the cost it pays
    is one flat commission per order.

    Parameters
    ----------
    slippage_bps:
        The friction applied to the open, in basis points: a finite real number
        greater than or equal to ``0`` and **strictly below** ``10_000``.  The upper
        bound is not a style choice: a sell fills at ``open * (1 - bps / 10_000)``,
        so at ``10_000`` bps it would receive a price of zero, and above it a
        negative one -- a trade that hands shares away rather than sells them.  The
        default is ``5.0``, i.e. 5 bps a side.
    commission_per_order:
        A flat amount of money per order: a finite real number greater than or equal
        to ``0``.  It is charged once for each order, never per share and never as a
        rate, and a sell pays it out of its proceeds.  The default is ``0.0``, the US
        equity convention this version assumes.

    Raises
    ------
    backtest_sys.errors.BacktestContractError
        If a parameter is not a finite real number (a ``bool``, a string, ``None``,
        ``NaN`` and ``+-inf`` are refused), the slippage is negative or reaches
        ``10_000`` bps, or the commission is negative.  Both parameters are checked
        before either is reported, so one error names every bad one.

    Attributes
    ----------
    slippage_fraction:
        ``slippage_bps / 10_000``: the slippage as a fraction of the open (``5.0``
        bps is ``0.0005``), which is the number the price arithmetic uses.  A
        property, never a field -- storing it would let a model state a fraction that
        disagrees with its own basis points.

    Notes
    -----
    * The slippage is **already inside the fill price**, so ``transaction_cost``
      carries the commission only: charging the slippage twice would leave the
      account poorer than the market made it.
    * Fills are complete: the model never shrinks, skips, postpones or splits an
      order, never checks the account's cash or holdings, and never touches
      :class:`~backtest_sys.contracts.BacktestState`.  Those decisions belong to the
      future order planner and ledger.
    * Frozen and hashable, so one model can be shared by every event of a replay and
      used as a mapping key.

    Examples
    --------
    >>> model = FixedBpsExecutionModel(slippage_bps=5.0, commission_per_order=0.0)
    >>> model
    FixedBpsExecutionModel(slippage_bps=5.0, commission_per_order=0.0)
    >>> model.slippage_fraction
    0.0005
    >>> FixedBpsExecutionModel()
    FixedBpsExecutionModel(slippage_bps=5.0, commission_per_order=0.0)
    >>> FixedBpsExecutionModel(commission_per_order=1.5).commission_per_order
    1.5
    """

    slippage_bps: float = DEFAULT_SLIPPAGE_BPS
    commission_per_order: float = DEFAULT_COMMISSION_PER_ORDER

    def __post_init__(self) -> None:
        problems: list[str] = []

        slippage, slippage_problem = _finite_number(self.slippage_bps, "slippage_bps")
        if slippage_problem is not None:
            problems.append(slippage_problem)
        elif not 0.0 <= slippage < MAX_SLIPPAGE_BPS:
            problems.append(
                f"slippage_bps must be zero or greater and below {MAX_SLIPPAGE_BPS!r} (10_000 bps "
                f"= 100% of the price), got {slippage!r}: a sell fills at "
                "open * (1 - bps / 10_000), so at 10_000 bps it would receive zero and above that "
                "a negative price -- a trade that gives the shares away rather than sells them"
            )

        commission, commission_problem = _finite_number(
            self.commission_per_order, "commission_per_order"
        )
        if commission_problem is not None:
            problems.append(commission_problem)
        elif commission < 0.0:
            problems.append(
                f"commission_per_order must be zero or greater, got {commission!r}: a commission "
                "is money paid to trade, so it is zero for a free execution but never a rebate "
                "this model would have to invent"
            )

        if problems:
            raise BacktestContractError("invalid FixedBpsExecutionModel", problems)

        object.__setattr__(self, "slippage_bps", slippage)
        object.__setattr__(self, "commission_per_order", commission)

    @property
    def slippage_fraction(self) -> float:
        """``slippage_bps / 10_000``: the side's slippage as a fraction of the open."""
        return self.slippage_bps / BASIS_POINTS_PER_UNIT

    def fill_orders(
        self,
        *,
        event: RebalanceEvent,
        orders: tuple[OrderIntent, ...],
        open_prices: Mapping[str, float],
    ) -> tuple[Fill, ...]:
        """Fill every order at ``event``'s open, moved by this model's slippage.

        Parameters
        ----------
        event, orders, open_prices:
            As described on :meth:`ExecutionModel.fill_orders`.  Nothing is mutated:
            the event and the orders are frozen values and the price table is only
            ever read.

        Returns
        -------
        tuple[Fill, ...]
            One fill per order, in the given order: each quotes its own order, is
            dated ``event.execution_date``, carries the ordered quantity unchanged,
            fills at the raw open moved by ``slippage_bps`` in the direction that
            costs the account money, and states ``commission_per_order`` as its
            ``transaction_cost``.  The two amounts a fill derives follow: a buy's
            ``cash_change`` is ``-(notional + commission)`` and a sell's is
            ``notional - commission``.

        Raises
        ------
        backtest_sys.errors.ExecutionError
            If the request is not fillable -- the event is not a ``RebalanceEvent``,
            the orders are not a tuple of this event's unique orders, a symbol has no
            price or a price is not a positive finite number -- with every problem
            collected in one error; or if a single fill cannot be computed or
            recorded, in which case the ``Fill`` contract's own failure is kept as
            ``__cause__``.

        Examples
        --------
        With a one-dollar commission the price is unchanged and the cash change grows
        by the commission, because the slippage is already inside the price::

        >>> model = FixedBpsExecutionModel(slippage_bps=5.0, commission_per_order=1.0)
        >>> event = RebalanceEvent("2024-01-04", "2024-01-05")
        >>> order = OrderIntent("rb-0001", "2024-01-04", "2024-01-05", "AAPL", "BUY", 10.0)
        >>> (fill,) = model.fill_orders(event=event, orders=(order,), open_prices={"AAPL": 100.0})
        >>> fill.fill_price, fill.transaction_cost
        (100.05, 1.0)
        >>> fill.cash_change
        -1001.5
        >>> fill.cash_change == -(fill.gross_notional + fill.transaction_cost)
        True
        """
        problems: list[str] = []

        resolved_event = event if isinstance(event, RebalanceEvent) else None
        if resolved_event is None:
            problems.append(
                f"event must be a RebalanceEvent, got {type(event).__name__}: a fill request names "
                "exactly one decision and the session its orders execute in, and only an event can "
                "say which sessions those are"
            )

        checked_orders, order_problems = _checked_orders(orders)
        problems.extend(order_problems)

        table, table_problems = _checked_price_table(open_prices)
        problems.extend(table_problems)

        if checked_orders is not None and resolved_event is not None:
            problems.extend(_agreement_problems(checked_orders, resolved_event))
        if checked_orders is not None and table is not None:
            problems.extend(_price_problems(checked_orders, table))

        # A missing value always arrives with a problem of its own above, so this
        # single guard is also what narrows the three names below to real values.
        if problems or resolved_event is None or checked_orders is None or table is None:
            raise ExecutionError("invalid fill request", problems)

        fills: list[Fill] = []
        for order in checked_orders:
            open_price = float(table[order.symbol])
            if order.is_buy:
                fill_price = open_price * (1.0 + self.slippage_fraction)
            else:
                fill_price = open_price * (1.0 - self.slippage_fraction)

            if not math.isfinite(fill_price) or fill_price <= 0.0:
                raise ExecutionError(
                    f"slipping the open of {order.symbol!r} by {self.slippage_bps!r} bps does not "
                    f"give a usable price for order {order.order_id!r}: {open_price!r} becomes "
                    f"{fill_price!r}, and a fill at that price would record a trade nobody could "
                    "have made"
                )

            try:
                fills.append(
                    Fill(
                        order=order,
                        fill_date=resolved_event.execution_date,
                        fill_price=fill_price,
                        quantity=order.quantity,
                        transaction_cost=self.commission_per_order,
                    )
                )
            except BacktestContractError as exc:
                raise ExecutionError(
                    f"order {order.order_id!r} cannot be recorded as a fill at {fill_price!r}",
                    [str(exc)],
                ) from exc

        return tuple(fills)


__all__ = [
    "BASIS_POINTS_PER_UNIT",
    "DEFAULT_COMMISSION_PER_ORDER",
    "DEFAULT_SLIPPAGE_BPS",
    "MAX_SLIPPAGE_BPS",
    "ExecutionModel",
    "FixedBpsExecutionModel",
]
