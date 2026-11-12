"""The stable backtest contract: one configuration in, one replayed history out.

Where this layer sits
---------------------
Every layer answers one question and hands the answer on::

    data_sys       market bars and the universe
        |
    factor_sys     one validated factor cross-section per session
        |
    strategy_sys   which symbols this decision selects         -- StrategyResult
        |
    portfolio_sys  how much weight each of them should carry   -- TargetPortfolio
        |
    backtest_sys   what the account actually did on the timeline -- this package

The portfolio layer stops at a *weight* difference: it can say "go from 40% to
25% of the account in AAPL", and it stops there deliberately.  A weight is not a
trade.  It says nothing about how many shares that is, at which price they change
hands, on which day they change hands, what the commission is, or what the account
is worth afterwards.  Those are the questions this package names -- and only
names: it declares the objects one history replay exchanges, it replays nothing.

Six statements
--------------
``BacktestConfig``
    **when** the replay runs and with how much money: a closed date range, the
    starting cash, and the calendar the range is measured on.
``BacktestState``
    what the account **owns** at the close of one session: cash plus a
    ``symbol -> share count`` tuple.  A fact, never a target.
``OrderIntent``
    one order **decided** on a decision date and **scheduled** for the open of a
    later session: which symbol, which side, how many shares.
``Fill``
    one order that **executed**: the price it got, the shares it moved and the
    cost it paid.
``DailyPortfolioValue``
    what the account was **worth** at the close of one session.
``BacktestResult``
    the **finished run**: the configuration it ran under, the closing state and the
    two histories -- valuations and fills.

The default trade timing, fixed by shape
----------------------------------------
The timeline this system replays is not a preference, so it is not a flag::

    Thursday close      the decision is taken, from data known at that close
    Friday open         the orders decided on Thursday execute
    Friday close        the account is valued on the new holdings

Two fields therefore carry the whole timing rule:

* :attr:`OrderIntent.scheduled_fill_date` must be **strictly later** than
  :attr:`OrderIntent.decision_date` -- an order can never fill on the day it was
  decided, so no look-ahead can be written down;
* :attr:`Fill.fill_date` must be **exactly** the scheduled date -- a fill cannot
  quietly arrive a day early or a day late.

Neither object knows *which* session Friday is: "the next session the calendar is
open" is a question for the rebalance policy and the trading calendar, which this
task deliberately does not implement.  The contract only refuses a schedule that
points backwards.

Why share counts, and not weights
---------------------------------
``0.4 of the account`` is true for one instant; ``100 shares`` stays true until
something trades.  A ledger, a fill and a valuation all need the second form: a
weight has to be re-priced to become a share count, while a share count times a
price *is* a value.  So :class:`BacktestPosition` stores ``quantity`` -- a
positive, possibly fractional number of shares (this first version allows
fractional shares) -- and never a weight, a price or a value.  Cash is
:attr:`BacktestState.cash`, a plain non-negative amount of money, never a
``"CASH"`` position: a symbol with no price, no rank and no factor value would
have to answer questions it cannot answer.

Why a fill may not be partial
-----------------------------
First version, one rule: :attr:`Fill.quantity` must equal
:attr:`OrderIntent.quantity` *exactly*.  Allowing a smaller fill would need a
remaining-quantity field on the order, a lifecycle (open, partially filled,
filled, cancelled) and a policy for what happens to the residue at the close --
none of which this task declares.  A partial fill is refused with an explanation
instead of being stored as an order that silently did not do what it said.

Why a valuation must add up
---------------------------
:attr:`DailyPortfolioValue.nav` is ``cash + market_value`` within
:data:`NAV_TOLERANCE`.  A snapshot whose net value does not equal its own parts
cannot be compared with the previous close, and the difference would be a return
that never happened -- invented by a typo, invisible in the output.  The tolerance
is not a judgement call: binary floating point cannot represent every decimal
amount exactly, so two correct sums must not fail on their last bit.

Why the closing state must coincide with the last close
-------------------------------------------------------
:attr:`BacktestResult.final_state` is not a seventh opinion about the run: it is
the last row of the valuation history, stated in the vocabulary of holdings.  So
its date must lie inside the configured window, and when valuations exist it must
both *be* the last valuation's session and *agree* with that row's ``cash`` within
:data:`NAV_TOLERANCE`.  Two cash figures for one close would let a reporting layer
quote 999 from the state and 1000 from the curve, and neither could be called
wrong.  What is *not* checked is the value of the holdings: a
:class:`DailyPortfolioValue` stores no per-symbol price, so the holdings' worth
cannot be recomputed from it and pretending otherwise would mean inventing a
number.

What this package deliberately does not contain
-----------------------------------------------
No rebalance policy (which session decides, which session fills), no trading
calendar lookup, no price or bar reader, no order generation from a
:class:`~portfolio_sys.contracts.PortfolioTransition`, no execution model, no
commission model, no ledger, no engine and no reporting: no return, no Sharpe
ratio, no maximum drawdown, no turnover.  It reads no file, opens no store and
touches no network -- importing it is as safe as importing a dataclass.

Those pieces arrive later and implement *against* these objects, which is why the
objects are frozen, fully validated and free of arithmetic beyond the two derived
amounts a fill can state about itself
(:attr:`~Fill.gross_notional` and :attr:`~Fill.cash_change`).

Conventions borrowed from the layers below
------------------------------------------
* **Dates are dates.**  A plain :class:`datetime.date` or a bare ISO-8601
  ``"YYYY-MM-DD"`` string is accepted; a ``datetime`` is refused even at
  midnight, and with it every ``pandas.Timestamp``: a session is a day, not a
  moment, and dropping a time of day silently is exactly the kind of repair that
  produces a backtest nobody can reproduce.
* **The ticker rule is the project's, not a second one.**  Symbols must match
  :data:`data_sys.schema.SYMBOL_PATTERN` -- the same pattern the universe, the
  panels, the strategies and the portfolios enforce -- so ``BRK.B`` is a symbol
  here too and a lower-case spelling is not.
* **Every collection is a tuple.**  A list, a set and a generator are refused: a
  history a caller can still append to is not a record of what happened.
* **Every problem is collected.**  One error names all of them, in the order the
  checks ran.

Examples
--------
>>> config = BacktestConfig(
...     start_date="2024-01-04",
...     end_date="2024-03-28",
...     initial_cash=100_000.0,
... )
>>> config.start_date.isoformat(), config.calendar_name
('2024-01-04', 'XNYS')
>>> state = BacktestState(
...     as_of_date="2024-01-05",
...     cash=2_500.0,
...     positions=(BacktestPosition("AAPL", 100.0), BacktestPosition("MSFT", 50.5)),
... )
>>> state.symbols
('AAPL', 'MSFT')
>>> state.quantities_by_symbol["MSFT"]
50.5
>>> order = OrderIntent(
...     order_id="rebalance-2024-01-04-0001",
...     decision_date="2024-01-04",
...     scheduled_fill_date="2024-01-05",
...     symbol="AAPL",
...     side="BUY",
...     quantity=25.0,
... )
>>> fill = Fill(
...     order=order,
...     fill_date="2024-01-05",
...     fill_price=187.5,
...     quantity=25.0,
...     transaction_cost=1.0,
... )
>>> fill.gross_notional
4687.5
>>> round(fill.cash_change, 6)
-4688.5
>>> order.is_buy, order.is_sell
(True, False)
>>> DailyPortfolioValue(
...     as_of_date="2024-01-05", cash=2_500.0, market_value=97_500.0, nav=100_000.0
... ).nav
100000.0
>>> result = BacktestResult(
...     config=config,
...     final_state=state,
...     daily_values=(DailyPortfolioValue("2024-01-05", 2_500.0, 97_500.0, 100_000.0),),
...     fills=(fill,),
... )
>>> result.dates[0].isoformat(), result.order_ids
('2024-01-05', ('rebalance-2024-01-04-0001',))
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime
from numbers import Real
from typing import Any

from backtest_sys.errors import BacktestContractError
from data_sys.schema import SYMBOL_PATTERN

# ---------------------------------------------------------------------------
# Defaults, vocabularies and tolerances
# ---------------------------------------------------------------------------
#: The calendar a replay is measured on unless a caller says otherwise: the New
#: York Stock Exchange, the venue the daily bars of :mod:`data_sys` are pulled
#: for.  Storing the name is not loading the calendar -- the sessions of a range
#: are looked up by whoever needs them, never here.
DEFAULT_CALENDAR_NAME: str = "XNYS"

#: The only two sides a simulated order may have, spelled exactly like this.
#: ``"buy"``, ``"Buy"`` and ``"long"`` are refused rather than folded: the side
#: decides the sign of a fill's cash change, so it is never guessed from case.
ORDER_SIDES: tuple[str, ...] = ("BUY", "SELL")

#: How far :attr:`DailyPortfolioValue.nav` may sit from ``cash + market_value``
#: and still be one snapshot.  An absolute amount of money, not a share of the
#: account: ``1e-9`` dollars is a billionth of a cent -- far below any tick or
#: commission, far above the last-bit error of a correct sum.
NAV_TOLERANCE: float = 1e-9

_SYMBOL_RE = re.compile(SYMBOL_PATTERN)
#: A session label written as text: exactly ``YYYY-MM-DD``, nothing else.
_DATE_TEXT_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------
def _describe(values: Iterable[Any], limit: int = 3) -> str:
    """A short, bounded rendering of offending values for an error message."""
    items = list(values)
    shown = ", ".join(repr(value) for value in items[:limit])
    if len(items) > limit:
        shown = f"{shown}, ... (+{len(items) - limit} more)"
    return f"[{shown}]"


def _duplicates(values: Iterable[Any]) -> list[Any]:
    """The values that occur more than once, in first-seen order."""
    seen: set[Any] = set()
    repeated: list[Any] = []
    for value in values:
        if value in seen and value not in repeated:
            repeated.append(value)
        seen.add(value)
    return repeated


def _session_date(value: Any, label: str) -> date:
    """Validate one session label and return it as a plain :class:`datetime.date`.

    Exactly two forms are accepted: a :class:`datetime.date` and a bare ISO-8601
    ``"YYYY-MM-DD"`` string.

    A ``datetime`` is refused *even at midnight*, and with it every
    ``pandas.Timestamp`` and ``pandas.NaT`` (both are ``datetime`` subclasses): a
    timestamp is a moment, not the session a backtest happened on, and accepting
    one would silently drop its time of day.  A string must therefore be a bare
    date too -- ``"2024-01-05T00:00"``, ``"2024-1-5"`` and ``"05/01/2024"`` are
    all refused rather than guessed at.
    """
    if isinstance(value, datetime):
        raise BacktestContractError(
            f"{label} must be a date, not the timestamp {value.isoformat()}: a backtest date "
            "carries no time of day and no timezone -- pass date(...) or 'YYYY-MM-DD'"
        )
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        text = value.strip()
        if _DATE_TEXT_RE.fullmatch(text):
            try:
                return date.fromisoformat(text)
            except ValueError:
                pass
    raise BacktestContractError(
        f"{label} must be a date or an ISO-8601 'YYYY-MM-DD' string, got {value!r}"
    )


def _finite_number(value: Any, label: str) -> tuple[float | None, str | None]:
    """``value`` as a float when it is a finite real number, otherwise why not.

    ``bool`` is refused even though it is an ``int``: ``True`` cannot mean "one
    share".  A string is refused rather than parsed, and ``None``, ``NaN`` and
    ``±inf`` are refused because none of them sizes a holding, prices a fill or
    states what an account is worth.
    """
    if isinstance(value, bool) or not isinstance(value, Real):
        return None, (
            f"{label} must be a real number, got {type(value).__name__}: {value!r}; a bool, a "
            "string or None is refused rather than converted, because an amount has to mean "
            "exactly the number it is"
        )
    number = float(value)
    if not math.isfinite(number):
        return None, (
            f"{label} must be finite, got {number!r}: NaN and ±inf can be held, traded or "
            "valued by nothing, so they are never a record of what happened"
        )
    return number, None


def _positive_number(value: Any, label: str, consequence: str) -> tuple[float | None, str | None]:
    """A finite real number strictly greater than zero, or why not."""
    number, problem = _finite_number(value, label)
    if problem is not None:
        return None, problem
    if number <= 0.0:
        return None, f"{label} must be greater than zero, got {number!r}: {consequence}"
    return number, None


def _non_negative_number(
    value: Any, label: str, consequence: str
) -> tuple[float | None, str | None]:
    """A finite real number greater than or equal to zero, or why not."""
    number, problem = _finite_number(value, label)
    if problem is not None:
        return None, problem
    if number < 0.0:
        return None, f"{label} must be zero or greater, got {number!r}: {consequence}"
    return number, None


def _symbol_problem(symbol: Any, label: str) -> str | None:
    """Why ``symbol`` is not a tradable ticker, or ``None`` when it is one."""
    # ``fullmatch`` rather than ``match``: the pattern is the project's own, but a
    # trailing newline must not slip through as a second ticker.
    if not isinstance(symbol, str) or not _SYMBOL_RE.fullmatch(symbol):
        return (
            f"{label} {symbol!r} must be an upper-case ticker matching {SYMBOL_PATTERN} "
            f"(letters, digits, '.' and '-'), got {type(symbol).__name__}: a holding is a "
            "security, never a blank and never a lower-case spelling"
        )
    if symbol == "CASH":
        return (
            f"{label} 'CASH' is not a security: cash is the account's own cash amount, so it "
            "never becomes a position, an order or a fill with a price of its own"
        )
    return None


def _checked_positions(
    value: Any, label: str
) -> tuple[tuple[BacktestPosition, ...] | None, list[str]]:
    """Validate a ``positions`` field; return the usable holdings and the problems.

    ``None`` comes back (with a problem) when the field is not a tuple at all, so
    a caller can skip the checks that only make sense on a real holding set.
    """
    if not isinstance(value, tuple):
        return None, [
            f"{label} must be a tuple[BacktestPosition, ...], got {type(value).__name__}: one "
            "holding is one immutable (symbol, quantity) pair, so a list, a set or a generator "
            "is refused -- a set has no order and a generator can be read only once; declare "
            "(BacktestPosition(...), ...)"
        ]

    problems: list[str] = [
        f"{label} entry {entry!r} must be a BacktestPosition, got {type(entry).__name__}: a "
        "bare ticker or a bare share count cannot say what is held and how much of it"
        for entry in value
        if not isinstance(entry, BacktestPosition)
    ]
    positions = tuple(entry for entry in value if isinstance(entry, BacktestPosition))

    repeated = _duplicates(position.symbol for position in positions)
    problems.extend(
        f"{label} holds {symbol} more than once: one holding per symbol -- two share counts of "
        "the same security are one position, and adding them up is the caller's job, never a "
        "silent repair here"
        for symbol in repeated
    )

    # A duplicate already breaks strict ascent, so the ordering complaint is only
    # added when the holding set is otherwise sound -- one cause, one bullet.
    if not repeated and len(positions) == len(value):
        symbols = [position.symbol for position in positions]
        if symbols != sorted(symbols):
            offender = next(
                f"{symbols[index]!r} comes after {symbols[index - 1]!r} but sorts before it"
                for index in range(1, len(symbols))
                if symbols[index] < symbols[index - 1]
            )
            problems.append(
                f"{label} must be ordered by strictly ascending symbol, got {_describe(symbols)}: "
                f"{offender}; the order is part of the record, so it is refused rather than "
                "re-sorted -- declare the holdings sorted by symbol"
            )
    return positions, problems


def _checked_tuple(
    value: Any, label: str, entry_type: type, entry_hint: str
) -> tuple[tuple[Any, ...] | None, list[str]]:
    """Validate one ``tuple[entry_type, ...]`` history; return it and the problems.

    ``None`` comes back (with a problem) when the field is not a tuple at all, so
    a caller can skip the checks that only make sense on a real history.
    """
    if not isinstance(value, tuple):
        return None, [
            f"{label} must be a tuple[{entry_type.__name__}, ...], got {type(value).__name__}: "
            f"{entry_hint}; declare (..., )"
        ]

    problems: list[str] = [
        f"{label} entry {entry!r} must be a {entry_type.__name__}, got {type(entry).__name__}"
        for entry in value
        if not isinstance(entry, entry_type)
    ]
    items = tuple(entry for entry in value if isinstance(entry, entry_type))
    return items, problems


# ---------------------------------------------------------------------------
# BacktestConfig -- when a replay runs, and with how much
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BacktestConfig:
    """The date range and the starting cash of one replay: the static half.

    A configuration says *which sessions* are replayed and *with how much money*
    the account starts.  It says nothing about which strategy is scored, which
    constructor weights the selection, when a rebalance is scheduled or what a
    fill costs: those travel with the objects a caller supplies, so one
    configuration can drive any research configuration.

    Parameters
    ----------
    start_date:
        The first session of the range: a plain :class:`datetime.date` or a bare
        ISO-8601 ``"YYYY-MM-DD"`` string.  A ``datetime`` -- and therefore every
        ``pandas.Timestamp`` -- is refused even at midnight.
    end_date:
        The last session of the range, in the same two forms.  It must not be
        earlier than :attr:`start_date`: a range that runs backwards contains no
        session at all.
    initial_cash:
        The account's starting cash: a finite real number strictly greater than
        ``0``.  ``bool``, ``str``, ``None``, ``NaN``, ``±inf``, ``0`` and negative
        amounts are refused -- an account that starts with nothing to invest
        cannot make a decision.
    calendar_name:
        The trading calendar the range is measured on; defaults to
        :data:`DEFAULT_CALENDAR_NAME` (``"XNYS"``).  The name is stored, never
        loaded: which sessions are open is a question for the calendar a future
        engine asks, so this object reads no calendar, no file and no network.  An
        empty name is refused, and so is one padded with whitespace, because
        ``" XNYS"`` and ``"XNYS"`` would silently be two different calendars.

    Raises
    ------
    backtest_sys.errors.BacktestContractError
        If a date is not a plain date, the range runs backwards, ``initial_cash``
        is not a positive finite number, or ``calendar_name`` is not a name.
        Every problem is reported at once.

    Notes
    -----
    * The range is **inclusive** at both ends: both dates are sessions the replay
      may trade and value on.
    * There is deliberately no rebalance policy here.  "Which session decides and
      which session fills" is a rule, and rules arrive with the task that can run
      them; a configuration only says when the window is open.
    * The range is not checked against a calendar: a Saturday is a legal
      ``start_date`` until a calendar says otherwise.

    Examples
    --------
    >>> config = BacktestConfig(date(2024, 1, 4), date(2024, 3, 28), 100_000)
    >>> config.start_date, config.end_date
    (datetime.date(2024, 1, 4), datetime.date(2024, 3, 28))
    >>> config.initial_cash
    100000.0
    >>> config.calendar_name
    'XNYS'
    >>> config == BacktestConfig("2024-01-04", "2024-03-28", 100_000.0)
    True
    """

    start_date: date
    end_date: date
    initial_cash: float
    calendar_name: str = DEFAULT_CALENDAR_NAME

    def __post_init__(self) -> None:
        problems: list[str] = []

        start: date | None = None
        end: date | None = None
        try:
            start = _session_date(self.start_date, "BacktestConfig.start_date")
        except BacktestContractError as exc:
            problems.append(str(exc))
        try:
            end = _session_date(self.end_date, "BacktestConfig.end_date")
        except BacktestContractError as exc:
            problems.append(str(exc))

        cash, cash_problem = _positive_number(
            self.initial_cash,
            "BacktestConfig.initial_cash",
            "a replay needs money to invest, so the starting amount is strictly positive",
        )
        if cash_problem is not None:
            problems.append(cash_problem)

        name = self.calendar_name
        if not isinstance(name, str):
            problems.append(
                f"BacktestConfig.calendar_name must be a str, got {type(name).__name__}: a "
                "calendar is named, never loaded, so the name has to be text"
            )
        elif not name.strip():
            problems.append(
                "BacktestConfig.calendar_name must not be empty: a replay is measured on one "
                f"named calendar (the default is {DEFAULT_CALENDAR_NAME!r})"
            )
        elif name != name.strip():
            problems.append(
                f"BacktestConfig.calendar_name {name!r} has leading or trailing whitespace: the "
                "name is compared verbatim, so it is refused rather than trimmed"
            )

        if start is not None and end is not None and start > end:
            problems.append(
                f"start_date {start.isoformat()} is after end_date {end.isoformat()}: a range "
                "that runs backwards contains no session at all"
            )

        if problems:
            raise BacktestContractError("invalid BacktestConfig", problems)

        object.__setattr__(self, "start_date", start)
        object.__setattr__(self, "end_date", end)
        object.__setattr__(self, "initial_cash", cash)


# ---------------------------------------------------------------------------
# BacktestPosition -- one holding, in shares
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BacktestPosition:
    """One holding exactly as the account owns it: a symbol and a share count.

    This is the smallest fact a ledger states about a security -- *the account
    owns this many shares of this symbol* -- and it stops there.  It has no price,
    no weight and no value: what the holding is *worth* depends on the close of a
    session and belongs to :class:`DailyPortfolioValue`; what share of the account
    it *should* carry belongs to the portfolio layer's
    :class:`~portfolio_sys.contracts.TargetPortfolio`.

    Parameters
    ----------
    symbol:
        Upper-case ticker matching the project's single ticker rule,
        :data:`data_sys.schema.SYMBOL_PATTERN` (``BRK.B`` is a valid symbol).  It is
        stored exactly as given: this contract never upper-cases or strips a
        symbol, because a repaired ticker would silently become a *different*
        security.  ``"CASH"`` is refused by name: cash is
        :attr:`BacktestState.cash`, an amount of money, never a holding with a
        price of its own.
    quantity:
        The number of shares held: a finite real number strictly greater than
        ``0``.  Fractional shares are allowed in this first version, so ``50.5`` is
        a legal holding; ``bool``, ``str``, ``None``, ``NaN``, ``±inf``, ``0`` and
        negative numbers are refused -- a zero holding is not a holding (leave it
        out) and a negative one would be a short, which this system cannot
        express.

    Raises
    ------
    backtest_sys.errors.BacktestContractError
        If the symbol is not a ticker or ``CASH``, or the quantity is not a
        positive finite number; every problem is reported at once.

    Notes
    -----
    Whether ``symbol`` is still listed, still in a universe, or has a price on some
    session is deliberately **not** checked here: a position only knows itself, and
    a delisted holding is exactly the kind of thing a backtest has to be able to
    carry.

    Examples
    --------
    >>> position = BacktestPosition(symbol="NVDA", quantity=12.5)
    >>> position.symbol, position.quantity
    ('NVDA', 12.5)
    >>> position == BacktestPosition("NVDA", 12.5)
    True
    """

    symbol: str
    quantity: float

    def __post_init__(self) -> None:
        problems: list[str] = []

        symbol_problem = _symbol_problem(self.symbol, "symbol")
        if symbol_problem is not None:
            problems.append(symbol_problem)

        quantity, quantity_problem = _positive_number(
            self.quantity,
            "quantity",
            "a holding with no shares is not a holding (leave it out) and a negative one would "
            "be a short, which this long-only system cannot express",
        )
        if quantity_problem is not None:
            problems.append(quantity_problem)

        if problems:
            raise BacktestContractError("invalid BacktestPosition", problems)

        object.__setattr__(self, "quantity", quantity)


# ---------------------------------------------------------------------------
# BacktestState -- what the account actually holds
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BacktestState:
    """What the account owns at the close of one session: cash and share counts.

    This is a statement of fact, not of intent, and it is deliberately different
    from every other object in the project: a
    :class:`~portfolio_sys.contracts.PortfolioState` says how the account is
    *allocated* (weights), a :class:`~portfolio_sys.contracts.TargetPortfolio`
    says how it *should* be allocated, and this object says what is actually in it
    -- including the mistakes of previous decisions and names that have since left
    the universe.

    Parameters
    ----------
    as_of_date:
        The session this state is true at the close of: a plain
        :class:`datetime.date` or a bare ISO-8601 ``"YYYY-MM-DD"`` string.  A
        ``datetime``, a ``pandas.Timestamp`` and a timezone are refused even at
        midnight -- a state is printed on a session, not stamped with a moment.
    cash:
        The uninvested money in the account: a finite real number greater than or
        equal to ``0``.  This system holds no margin, so a negative balance cannot
        be expressed; ``NaN``, ``±inf``, ``bool`` and ``str`` are refused.
    positions:
        A **tuple** of :class:`BacktestPosition`, one per symbol, ordered by
        **strictly ascending symbol**.  A list, a set, a generator and a bare
        position are refused, duplicate symbols are refused, and an out-of-order
        tuple is refused rather than quietly sorted: the tuple *is* the record, and
        re-ordering it would change what the record says about the account.  An
        empty tuple is legal and means an all-cash account -- exactly what a replay
        starts from.

    Raises
    ------
    backtest_sys.errors.BacktestContractError
        If the date is not a plain date, ``cash`` is not a finite non-negative
        number, or ``positions`` is not a symbol-sorted, duplicate-free tuple of
        holdings.  Every problem is reported at once.

    Notes
    -----
    * **No price, no value, no weight.**  What the account is worth is a
      :class:`DailyPortfolioValue`, computed from prices the future engine reads;
      what share of it a symbol should carry is the portfolio layer's vocabulary.
      Adding either here would let two valuations disagree.
    * **A holding is not required to be tradable.**  A delisted symbol may still be
      owned; it simply cannot be sold, which the engine will have to decide about,
      not this object.
    * The state is immutable and holds no list, so nothing a caller keeps can
      change what the account held.

    Examples
    --------
    >>> state = BacktestState(
    ...     as_of_date="2024-01-05",
    ...     cash=1_000.0,
    ...     positions=(BacktestPosition("AAPL", 10.0), BacktestPosition("MSFT", 5.0)),
    ... )
    >>> state.symbols
    ('AAPL', 'MSFT')
    >>> state.quantities_by_symbol
    {'AAPL': 10.0, 'MSFT': 5.0}
    >>> BacktestState("2024-01-05", 0.0, ()).quantities_by_symbol
    {}
    """

    as_of_date: date
    cash: float
    positions: tuple[BacktestPosition, ...]

    def __post_init__(self) -> None:
        problems: list[str] = []

        resolved: date | None = None
        try:
            resolved = _session_date(self.as_of_date, "BacktestState.as_of_date")
        except BacktestContractError as exc:
            problems.append(str(exc))

        positions, position_problems = _checked_positions(self.positions, "BacktestState.positions")
        problems.extend(position_problems)

        cash, cash_problem = _non_negative_number(
            self.cash,
            "BacktestState.cash",
            "this system holds no margin, so cash is what the trades left behind and never a loan",
        )
        if cash_problem is not None:
            problems.append(cash_problem)

        if problems:
            raise BacktestContractError("invalid BacktestState", problems)

        object.__setattr__(self, "as_of_date", resolved)
        object.__setattr__(self, "cash", cash)

    @property
    def symbols(self) -> tuple[str, ...]:
        """The held symbols, in the stored (strictly ascending) order."""
        return tuple(position.symbol for position in self.positions)

    @property
    def quantities_by_symbol(self) -> dict[str, float]:
        """``symbol -> share count`` for every holding of this state."""
        return {position.symbol: position.quantity for position in self.positions}


# ---------------------------------------------------------------------------
# OrderIntent -- what the decision wants to trade, and when
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OrderIntent:
    """One order decided on a decision date and scheduled for a later session.

    An intent is the bridge between a decision and a trade: it says *this symbol,
    this side, this many shares, at the next session's open* and nothing else.  It
    carries no price, no slippage, no commission and no target weight, and it has
    no status -- an intent is not "partially filled" or "cancelled": either a fill
    quotes it back, or it does not exist.

    Parameters
    ----------
    order_id:
        The identifier a :class:`Fill` points back at: a non-empty string without
        leading or trailing whitespace.  This task does not decide how an id is
        generated -- date-prefixed counters, UUIDs and hashes are all fine -- but it
        refuses a blank id and a padded one, because ids are compared verbatim and
        ``"a1 "`` must not be a second order next to ``"a1"``.
    decision_date:
        The session the decision was taken at the close of: a plain
        :class:`datetime.date` or a bare ISO-8601 string, with the same strictness
        as every other date in the project.
    scheduled_fill_date:
        The session this order is due to execute in: **strictly later** than
        :attr:`decision_date`.  That is the whole first-version timing rule --
        decide after the close, fill at the next open -- and it is enforced by shape
        so that no look-ahead can be written down.  Which session that is (the next
        one the calendar is open) is decided by the future rebalance policy, never
        here.
    symbol:
        The security to trade, under the project's single ticker rule.  ``"CASH"``
        is refused: cash is not a security.
    side:
        Exactly ``"BUY"`` or ``"SELL"`` -- one of :data:`ORDER_SIDES`.  There is no
        case folding and no synonym: the side decides the sign of a fill's cash
        change, so it is never guessed.
    quantity:
        How many shares the order wants to move: a finite real number strictly
        greater than ``0``, fractional allowed.  A zero order is not an order and a
        negative one is a side, not a size.

    Raises
    ------
    backtest_sys.errors.BacktestContractError
        If the id is blank, a date is not a plain date, the schedule does not point
        forward, the symbol is not a ticker, the side is not one of the two
        spellings, or the quantity is not a positive finite number.  Every problem
        is reported at once.

    Examples
    --------
    >>> order = OrderIntent(
    ...     order_id="rb-0001",
    ...     decision_date=date(2024, 1, 4),
    ...     scheduled_fill_date=date(2024, 1, 5),
    ...     symbol="AAPL",
    ...     side="SELL",
    ...     quantity=40.0,
    ... )
    >>> order.symbol, order.quantity
    ('AAPL', 40.0)
    >>> order.is_buy, order.is_sell
    (False, True)
    """

    order_id: str
    decision_date: date
    scheduled_fill_date: date
    symbol: str
    side: str
    quantity: float

    def __post_init__(self) -> None:
        problems: list[str] = []

        identifier = self.order_id
        if not isinstance(identifier, str):
            problems.append(
                f"order_id must be a str, got {type(identifier).__name__}: a fill names the "
                "order it executed by id, so the id has to be text"
            )
        elif not identifier.strip():
            problems.append(
                "order_id must not be empty: every order needs an id a fill can point back at"
            )
        elif identifier != identifier.strip():
            problems.append(
                f"order_id {identifier!r} has leading or trailing whitespace: ids are compared "
                "verbatim, so it is refused rather than trimmed"
            )

        decision: date | None = None
        scheduled: date | None = None
        try:
            decision = _session_date(self.decision_date, "OrderIntent.decision_date")
        except BacktestContractError as exc:
            problems.append(str(exc))
        try:
            scheduled = _session_date(self.scheduled_fill_date, "OrderIntent.scheduled_fill_date")
        except BacktestContractError as exc:
            problems.append(str(exc))

        symbol_problem = _symbol_problem(self.symbol, "symbol")
        if symbol_problem is not None:
            problems.append(symbol_problem)

        if self.side not in ORDER_SIDES:
            problems.append(
                f"side {self.side!r} is not an order side: exactly 'BUY' or 'SELL', never folded "
                "from case and never a synonym -- the side decides the sign of the cash change"
            )

        quantity, quantity_problem = _positive_number(
            self.quantity,
            "quantity",
            "an order has to move shares: a zero order is not an order and a negative one is a "
            "side, not a size",
        )
        if quantity_problem is not None:
            problems.append(quantity_problem)

        if decision is not None and scheduled is not None and scheduled <= decision:
            problems.append(
                f"scheduled_fill_date {scheduled.isoformat()} must be after decision_date "
                f"{decision.isoformat()}: a decision is taken after the close and filled at the "
                "next open, so an order can never execute on the session it was decided on"
            )

        if problems:
            raise BacktestContractError("invalid OrderIntent", problems)

        object.__setattr__(self, "decision_date", decision)
        object.__setattr__(self, "scheduled_fill_date", scheduled)
        object.__setattr__(self, "quantity", quantity)

    @property
    def is_buy(self) -> bool:
        """Whether this intent buys: the side whose cash change is negative."""
        return self.side == "BUY"

    @property
    def is_sell(self) -> bool:
        """Whether this intent sells: the side whose cash change is positive."""
        return self.side == "SELL"


# ---------------------------------------------------------------------------
# Fill -- an order that actually executed
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Fill:
    """One executed order: the price it got, the shares it moved, the cost it paid.

    A fill is the only object in the project that states what a trade *cost*.  It
    is derived from an :class:`OrderIntent` rather than repeating its symbol and
    side, so a fill can never disagree with the order it claims to execute, and it
    is immutable, so a recorded execution cannot drift.

    Parameters
    ----------
    order:
        The :class:`OrderIntent` this fill executes.  A fill quotes its order back
        verbatim; it never re-states the symbol, the side or the size as its own
        fields.
    fill_date:
        The session the trade happened in: a plain :class:`datetime.date` or a bare
        ISO-8601 string.  It must be **exactly** :attr:`OrderIntent.scheduled_fill_date`
        -- a fill cannot arrive early, late, or on a session nobody scheduled.
    fill_price:
        The price per share actually paid or received: a finite real number strictly
        greater than ``0``.  A zero or negative price is not a trade.
    quantity:
        The number of shares moved: a finite real number strictly greater than ``0``
        that must equal :attr:`OrderIntent.quantity` **exactly**.  This first
        version has no partial fills: a partially filled order would need a
        remaining quantity and a lifecycle the contract does not declare yet, so it
        is refused instead of being stored as an order that did not do what it said.
    transaction_cost:
        Commission and fees paid for this execution: a finite real number greater
        than or equal to ``0``.  It is an amount of money, not a rate and not a share
        of the account.

    Raises
    ------
    backtest_sys.errors.BacktestContractError
        If ``order`` is not an :class:`OrderIntent`, ``fill_date`` is not the
        scheduled date, a price or quantity is not a positive finite number,
        ``quantity`` differs from the ordered size, ``transaction_cost`` is
        negative, or price times quantity is not a finite amount.  Every problem is
        reported at once.

    Attributes
    ----------
    gross_notional:
        ``fill_price * quantity``: the money the shares themselves moved, before the
        cost and with no sign -- a buy and a sell of the same size have the same
        notional.
    cash_change:
        The signed effect on the account's cash: ``-(gross_notional +
        transaction_cost)`` for a ``BUY`` and ``gross_notional - transaction_cost``
        for a ``SELL``.  A buy spends its notional *and* its cost; a sell receives
        its notional and pays its cost out of it.

    Notes
    -----
    Both derived amounts are properties, not fields: storing them would let a
    caller construct a fill whose stated cash change disagrees with its own price
    and quantity.  Their multiplication is checked for overflow at construction, so
    a fill can never hand back ``inf`` as the money it moved.

    Examples
    --------
    >>> order = OrderIntent("rb-0001", "2024-01-04", "2024-01-05", "AAPL", "BUY", 40.0)
    >>> fill = Fill(
    ...     order=order,
    ...     fill_date="2024-01-05",
    ...     fill_price=180.0,
    ...     quantity=40.0,
    ...     transaction_cost=2.5,
    ... )
    >>> fill.gross_notional
    7200.0
    >>> fill.cash_change
    -7202.5
    >>> fill.symbol
    'AAPL'
    """

    order: OrderIntent
    fill_date: date
    fill_price: float
    quantity: float
    transaction_cost: float

    def __post_init__(self) -> None:
        problems: list[str] = []

        order: OrderIntent | None = self.order
        if not isinstance(order, OrderIntent):
            problems.append(
                f"order must be an OrderIntent, got {type(order).__name__}: a fill quotes the "
                "order it executed, so it can never trade something nobody decided to trade"
            )
            order = None

        resolved: date | None = None
        try:
            resolved = _session_date(self.fill_date, "Fill.fill_date")
        except BacktestContractError as exc:
            problems.append(str(exc))

        price, price_problem = _positive_number(
            self.fill_price,
            "fill_price",
            "a price of zero or less is not a trade",
        )
        if price_problem is not None:
            problems.append(price_problem)

        quantity, quantity_problem = _positive_number(
            self.quantity,
            "quantity",
            "a fill moves shares: zero shares did not execute and a negative size is a side",
        )
        if quantity_problem is not None:
            problems.append(quantity_problem)

        cost, cost_problem = _non_negative_number(
            self.transaction_cost,
            "transaction_cost",
            "a cost is money paid, so it is zero for a free execution but never negative",
        )
        if cost_problem is not None:
            problems.append(cost_problem)

        if order is not None and resolved is not None and resolved != order.scheduled_fill_date:
            problems.append(
                f"fill_date {resolved.isoformat()} is not the scheduled fill date "
                f"{order.scheduled_fill_date.isoformat()} of order {order.order_id!r}: an order "
                "fills on the session it was scheduled for, never early and never late"
            )

        if order is not None and quantity is not None and quantity != order.quantity:
            problems.append(
                f"quantity {quantity!r} is not the ordered size {order.quantity!r} of order "
                f"{order.order_id!r}: this version fills completely or not at all, so a partial "
                "fill would need a remaining quantity the contract does not declare"
            )

        if price is not None and quantity is not None:
            notional = price * quantity
            if not math.isfinite(notional):
                problems.append(
                    f"gross_notional is not a finite amount: fill_price {price!r} x quantity "
                    f"{quantity!r} overflows to {notional!r}, and no account can hold it"
                )
            elif cost is not None and not math.isfinite(notional + cost):
                problems.append(
                    f"cash_change is not a finite amount: notional {notional!r} and cost {cost!r} "
                    "overflow when they meet, and no account can hold that either"
                )

        if problems:
            raise BacktestContractError("invalid Fill", problems)

        object.__setattr__(self, "fill_date", resolved)
        object.__setattr__(self, "fill_price", price)
        object.__setattr__(self, "quantity", quantity)
        object.__setattr__(self, "transaction_cost", cost)

    @property
    def symbol(self) -> str:
        """The symbol this fill traded, taken from the order it executed."""
        return self.order.symbol

    @property
    def gross_notional(self) -> float:
        """``fill_price * quantity``: the money the shares moved, without a sign."""
        return self.fill_price * self.quantity

    @property
    def cash_change(self) -> float:
        """The signed effect on cash: a buy pays notional and cost, a sell receives net."""
        notional = self.gross_notional
        if self.order.is_buy:
            return -(notional + self.transaction_cost)
        return notional - self.transaction_cost


def _valuation_problems(
    values: tuple[DailyPortfolioValue, ...], config: BacktestConfig | None
) -> list[str]:
    """Why the valuation history is not one ordered, in-window series, or nothing."""
    problems: list[str] = []

    repeated = _duplicates(value.as_of_date for value in values)
    problems.extend(
        f"daily_values holds {day.isoformat()} more than once: one close, one value -- two "
        "valuations of the same session are two answers to one question"
        for day in repeated
    )
    if not repeated:
        dates = [value.as_of_date for value in values]
        if dates != sorted(dates):
            offender = next(
                f"{dates[index].isoformat()} comes after {dates[index - 1].isoformat()} but sorts "
                "before it"
                for index in range(1, len(dates))
                if dates[index] < dates[index - 1]
            )
            problems.append(
                "daily_values must be ordered by strictly ascending as_of_date, got "
                f"{_describe(day.isoformat() for day in dates)}: {offender}; the history is a "
                "record, so it is refused rather than re-sorted"
            )

    if config is not None:
        outside = [
            value.as_of_date
            for value in values
            if not config.start_date <= value.as_of_date <= config.end_date
        ]
        if outside:
            window = f"{config.start_date.isoformat()}..{config.end_date.isoformat()}"
            problems.append(
                f"daily_values dated {_describe(day.isoformat() for day in outside)} fall outside "
                f"the configured range {window}: a valuation outside the window was never part "
                "of this run"
            )
    return problems


def _fill_problems(fills: tuple[Fill, ...], config: BacktestConfig | None) -> list[str]:
    """Why the execution history is not one ordered, unique, in-window series, or nothing."""
    problems: list[str] = []

    repeated = _duplicates(fill.order.order_id for fill in fills)
    problems.extend(
        f"fills holds order id {order_id!r} more than once: one order fills once, so a repeated "
        "id is a trade the record would count twice"
        for order_id in repeated
    )

    if not repeated:
        keys = [(fill.fill_date, fill.order.order_id) for fill in fills]
        if keys != sorted(keys):
            offender = next(
                f"{_describe(keys[index])} comes after {_describe(keys[index - 1])} but sorts "
                "before it"
                for index in range(1, len(keys))
                if keys[index] < keys[index - 1]
            )
            problems.append(
                "fills must be ordered by ascending (fill_date, order_id), got "
                f"{_describe(keys)}: {offender}; the execution history is a record, so it is "
                "refused rather than re-sorted"
            )

    if config is not None:
        outside = [
            fill.fill_date
            for fill in fills
            if not config.start_date <= fill.fill_date <= config.end_date
        ]
        if outside:
            window = f"{config.start_date.isoformat()}..{config.end_date.isoformat()}"
            problems.append(
                f"fills dated {_describe(day.isoformat() for day in outside)} fall outside the "
                f"configured range {window}: a trade outside the window was never part of this run"
            )
    return problems


# ---------------------------------------------------------------------------
# DailyPortfolioValue -- what the account was worth at one close
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DailyPortfolioValue:
    """The account's worth at one session's close: cash, holdings and the total.

    This is the valuation half of a replay: one row per session the backtest
    valued, holding the three amounts a net-value curve is built from.  It carries
    no return and no drawdown -- those compare two rows, and comparing rows is
    reporting, which this task does not implement.

    Parameters
    ----------
    as_of_date:
        The session this valuation is the close of: a plain :class:`datetime.date`
        or a bare ISO-8601 ``"YYYY-MM-DD"`` string, with the usual strictness.
    cash:
        The uninvested money at that close: a finite real number greater than or
        equal to ``0``.
    market_value:
        What the held shares are worth at that close: a finite real number greater
        than or equal to ``0``.  Summing holdings is the future engine's job; this
        object records the total it reached.
    nav:
        The net asset value of the account: a finite real number greater than or
        equal to ``0`` that must equal ``cash + market_value`` within
        :data:`NAV_TOLERANCE`.  A caller that already computed the total is welcome
        to pass it; a caller that passes a different one is refused, because a
        snapshot that does not add up would enter the curve as a return that never
        happened.

    Raises
    ------
    backtest_sys.errors.BacktestContractError
        If the date is not a plain date, an amount is not a finite non-negative
        number, or ``nav`` is not ``cash + market_value``.  Every problem is
        reported at once.

    Notes
    -----
    * ``nav`` is deliberately **not** derived from the other two fields.  Deriving
      it would hide the caller's own arithmetic, and a *stated* total that
      disagrees with its parts is exactly the mistake this contract exists to
      catch -- the difference would otherwise become a return nobody can explain.
    * A valuation is not a state: the state says how many shares are held, the
      valuation says what they were worth.  Both are needed and neither implies the
      other.

    Examples
    --------
    >>> value = DailyPortfolioValue("2024-01-05", 1_000.0, 99_000.0, 100_000.0)
    >>> value.nav
    100000.0
    >>> value.cash + value.market_value
    100000.0
    """

    as_of_date: date
    cash: float
    market_value: float
    nav: float

    def __post_init__(self) -> None:
        problems: list[str] = []

        resolved: date | None = None
        try:
            resolved = _session_date(self.as_of_date, "DailyPortfolioValue.as_of_date")
        except BacktestContractError as exc:
            problems.append(str(exc))

        cash, cash_problem = _non_negative_number(
            self.cash,
            "cash",
            "an account cannot hold less than no money, and this system holds no margin",
        )
        if cash_problem is not None:
            problems.append(cash_problem)

        market_value, market_problem = _non_negative_number(
            self.market_value,
            "market_value",
            "a valuation is a value, so what the holdings are worth is never negative",
        )
        if market_problem is not None:
            problems.append(market_problem)

        nav, nav_problem = _non_negative_number(
            self.nav,
            "nav",
            "a valuation is a value, so the net asset value is never negative",
        )
        if nav_problem is not None:
            problems.append(nav_problem)

        if cash is not None and market_value is not None and nav is not None:
            total = cash + market_value
            if abs(nav - total) > NAV_TOLERANCE:
                problems.append(
                    f"nav must equal cash + market_value, got {nav!r} against {cash!r} + "
                    f"{market_value!r} = {total!r} (tolerance {NAV_TOLERANCE!r}): a snapshot that "
                    "does not add up cannot be compared with the next close"
                )

        if problems:
            raise BacktestContractError("invalid DailyPortfolioValue", problems)

        object.__setattr__(self, "as_of_date", resolved)
        object.__setattr__(self, "cash", cash)
        object.__setattr__(self, "market_value", market_value)
        object.__setattr__(self, "nav", nav)


# ---------------------------------------------------------------------------
# BacktestResult -- the finished run
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BacktestResult:
    """A completed replay: its configuration, its closing state and its two histories.

    A result is what a reporting layer reads and what a caller keeps: the exact
    configuration the run was measured under, the state the account ended in, one
    valuation per valued session, and every fill that happened in the order it
    happened.

    Parameters
    ----------
    config:
        The :class:`BacktestConfig` the run was measured under.  A result that cannot
        name its own window cannot be interpreted: "a 12% return" means nothing
        without the dates and the starting cash it started from.
    final_state:
        The :class:`BacktestState` the account ended in.  Its date must fall inside
        ``[config.start_date, config.end_date]`` -- a closing state dated into a
        session the run never covered is not this run's closing state -- and, when
        :attr:`daily_values` is not empty, it must be the last valuation's session
        and its ``cash`` must match that row's ``cash`` within
        :data:`NAV_TOLERANCE`.  Two cash figures for one close would be two
        contradictory answers to the same question.
    daily_values:
        A **tuple** of :class:`DailyPortfolioValue`, one per valued session, ordered
        by **strictly ascending** ``as_of_date``, every date inside
        ``[config.start_date, config.end_date]``.  A list, a set and a generator are
        refused, and so is a repeated date: one close, one value.  An empty tuple is
        legal -- a window in which no session was valued is a meaningful result (for
        example a range that holds no open session), and refusing it would force the
        future engine to invent a placeholder row.
    fills:
        A **tuple** of :class:`Fill`, ordered by ascending
        ``(fill_date, order_id)``, with unique order ids and every fill date inside
        the window.  A list, a set and a generator are refused.  An empty tuple is
        legal: a replay that never traded is a legal replay.

    Raises
    ------
    backtest_sys.errors.BacktestContractError
        If ``config`` or ``final_state`` is the wrong type, a history is not a
        tuple, an entry is not the right object, a date is outside the window, the
        valuations or the fills are out of order or duplicated, the closing state is
        dated differently from the last valuation, or its cash disagrees with that
        valuation's cash.  Every problem is reported at once.

    Notes
    -----
    * **Nothing is computed here.**  No return, no Sharpe ratio, no maximum
      drawdown, no turnover, no trade count.  Those are reporting: they compare
      rows, and they will be written as a separate module that *reads* a result --
      never as a property of it, because a stored metric is a second source of
      truth that can disagree with the rows it came from.
    * **The order of the two histories is part of the record.**  Sorting is refused
      rather than performed, for the same reason the account's holdings are: a
      caller handing over a history must say when things happened, and a library
      that quietly re-sorts it would hide a bug in the engine that produced it.
    * **A result is not a ledger.**  It states the closes and the executions; the
      running cash balance and the position-by-position history belong to the
      future ledger, which will build them from exactly these two tuples.
    * **The closing state is the last close, not a second row.**  Its date lies in
      the window, and when valuations exist it repeats the last row's session and
      its cash within :data:`NAV_TOLERANCE`, so a reporting layer cannot read two
      different cash figures for the same session.

    Examples
    --------
    >>> config = BacktestConfig("2024-01-04", "2024-01-05", 1_000.0)
    >>> order = OrderIntent("rb-0001", "2024-01-04", "2024-01-05", "AAPL", "BUY", 2.0)
    >>> fill = Fill(order, "2024-01-05", 100.0, 2.0, 0.5)
    >>> result = BacktestResult(
    ...     config=config,
    ...     final_state=BacktestState("2024-01-05", 799.5, (BacktestPosition("AAPL", 2.0),)),
    ...     daily_values=(DailyPortfolioValue("2024-01-05", 799.5, 200.0, 999.5),),
    ...     fills=(fill,),
    ... )
    >>> result.dates
    (datetime.date(2024, 1, 5),)
    >>> result.order_ids
    ('rb-0001',)
    >>> result.final_state.symbols
    ('AAPL',)
    """

    config: BacktestConfig
    final_state: BacktestState
    daily_values: tuple[DailyPortfolioValue, ...]
    fills: tuple[Fill, ...]

    def __post_init__(self) -> None:
        problems: list[str] = []

        config: BacktestConfig | None = self.config
        if not isinstance(config, BacktestConfig):
            problems.append(
                f"config must be a BacktestConfig, got {type(config).__name__}: a result that "
                "cannot name its own window of dates and starting cash cannot be interpreted"
            )
            config = None

        state: BacktestState | None = self.final_state
        if not isinstance(state, BacktestState):
            problems.append(
                f"final_state must be a BacktestState, got {type(state).__name__}: a result ends "
                "in exactly one account state"
            )
            state = None

        daily_values, daily_problems = _checked_tuple(
            self.daily_values,
            "BacktestResult.daily_values",
            DailyPortfolioValue,
            "a valuation history is an immutable sequence of closes, so a list, a set or a "
            "generator is refused",
        )
        problems.extend(daily_problems)

        fills, fills_problems = _checked_tuple(
            self.fills,
            "BacktestResult.fills",
            Fill,
            "an execution history is an immutable sequence of fills, so a list, a set or a "
            "generator is refused",
        )
        problems.extend(fills_problems)

        if state is not None and config is not None:
            start, end = config.start_date, config.end_date
            if not start <= state.as_of_date <= end:
                problems.append(
                    f"final_state.as_of_date is {state.as_of_date.isoformat()}, outside the "
                    f"configured range {start.isoformat()}..{end.isoformat()}: the closing "
                    "state belongs to this run, so it cannot be dated into a session the run "
                    "never covered"
                )

        if daily_values is not None:
            problems.extend(_valuation_problems(daily_values, config))
            if state is not None and daily_values:
                last = daily_values[-1]
                if state.as_of_date != last.as_of_date:
                    problems.append(
                        f"final_state.as_of_date is {state.as_of_date.isoformat()}, but the last "
                        f"valuation is dated {last.as_of_date.isoformat()}: the closing state is "
                        "the state of the last close, never of another session"
                    )
                elif abs(state.cash - last.cash) > NAV_TOLERANCE:
                    problems.append(
                        f"final_state.cash is {state.cash!r}, but the "
                        f"{last.as_of_date.isoformat()} valuation closed with cash "
                        f"{last.cash!r} (tolerance {NAV_TOLERANCE!r}): the closing state and the "
                        "closing snapshot answer one question, so they must not disagree"
                    )

        if fills is not None:
            problems.extend(_fill_problems(fills, config))

        if problems:
            raise BacktestContractError("invalid BacktestResult", problems)

    @property
    def dates(self) -> tuple[date, ...]:
        """The valued sessions, in the stored (strictly ascending) order."""
        return tuple(value.as_of_date for value in self.daily_values)

    @property
    def order_ids(self) -> tuple[str, ...]:
        """The executed order ids, in the order the fills were recorded."""
        return tuple(fill.order.order_id for fill in self.fills)


__all__ = [
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
