"""The stable portfolio contract: what a portfolio is, and what may build one.

Where this layer sits
---------------------
Every layer answers one question and hands the answer on::

    data_sys       market bars and the universe
        |
    factor_sys     one validated factor cross-section per session
        |
    strategy_sys   which symbols this decision selects         -- StrategyResult
        |
    portfolio_sys  how much weight each of them should carry   -- this package
        |
    (execution)    orders, fills and costs                     -- a later task

Three objects, three different statements
-----------------------------------------
``StrategyResult``
    *what* is selected on the decision date: one row per symbol with a score, a
    rank and a selection flag, plus the exact strategy **and factor versions**
    that produced it.
``PortfolioState``
    what the account **already holds**: the weights of the portfolio that is
    being rebalanced, dated the day that state became known.
``TargetPortfolio``
    what the account **should hold after** this decision: the weights a
    constructor wants, the constraints they were built under, and the strategy
    result they came from.

A result is not a portfolio, and a portfolio is not an order.  Nothing in this
module carries a share count, a price, a fill, a commission or a return: a
weight is a share of the account, and turning weights into trades belongs to the
execution layer.

What this package deliberately does not contain
-----------------------------------------------
No weighting algorithm, no constructor registry, no runner, no CLI, no file, no
Parquet, no DuckDB and no factor formula.  :class:`PortfolioConstructor` is only
the interface that a later ``EqualWeightConstructor``,
``ScoreWeightedConstructor``, ``VolatilityWeightedConstructor`` or
``OptimizerConstructor`` will implement -- this module is what they agree on, so
that one backtest can run any of them.

Why cash is a field and not a ticker
------------------------------------
``"CASH"`` is not a symbol: it has no price, no rank and no factor value, and a
universe that contained it would have to answer questions it cannot answer.  So
cash is :attr:`PortfolioState.cash_weight` / :attr:`TargetPortfolio.cash_weight`
-- a plain non-negative share of the account, separate from every holding.

Why long-only is not a switch
-----------------------------
The first version of this system is long-only, permanently, and the contract
says so by *shape* rather than by an option: every stock weight is strictly
positive and cash is never negative, so a short position and a negative cash
balance cannot be spelled at all.  There is no ``long_only=False`` to pass --
a flag that can be turned off is a guarantee that can be turned off.

Why weights must add up to 1.0
------------------------------
A portfolio is a complete answer: every dollar of the account is either in a
stock or in cash.  ``sum(weight) + cash_weight`` must therefore be ``1.0``
within :data:`WEIGHT_TOLERANCE` -- not roughly, not approximately.  A total below
1.0 is unallocated money the next layer would have to invent a home for, and a
total above 1.0 is leverage the system does not support.  The tolerance exists
only because binary floating point cannot represent every decimal share exactly,
so two correct constructions must not fail on their last bit.

An empty holding set is legal only with ``cash_weight == 1.0``: a portfolio that
holds nothing *is* an all-cash portfolio, which is exactly what a backtest starts
from and what a constructor may choose to return.

Dates are labels, not sessions
------------------------------
A portfolio date says *"this state, or this decision, was known on this day"*.
The contract does not know a trading calendar, does not check that a date is a
session and does not decide when a rebalance happens: that belongs to a future
rebalance-calendar or backtest layer.  What it does enforce is the direction of
knowledge:

* a :class:`TargetPortfolio` is dated exactly its
  :attr:`~TargetPortfolio.source_result`'s date -- a target cannot be built from
  another day's selection;
* a :class:`PortfolioContext` may carry a current state dated *earlier* than the
  decision (the last rebalance usually was), but never *later*: a state from the
  future would leak information that did not exist on the decision date.

Traceability
------------
A :class:`TargetPortfolio` never restates the strategy: it holds the whole
:class:`~strategy_sys.contracts.StrategyResult` it was built from and reads
:attr:`~TargetPortfolio.strategy_spec`,
:attr:`~TargetPortfolio.factor_requirements` and
:attr:`~TargetPortfolio.universe` straight out of it.  The strategy name, its
version and the exact ``(factor_name, factor_version)`` pairs therefore reach a
future backtest ledger, report or cache key without being copied, re-derived or
dropped.

Conventions reused, never re-declared
-------------------------------------
* The ticker syntax is the project's single one,
  :data:`data_sys.schema.SYMBOL_PATTERN` -- the same constant the factor layer
  uses, imported rather than copied.  (It is a constants module: importing it
  reads no file and touches no market data.)
* The construction-name syntax is the strategy layer's own
  :data:`strategy_sys.contracts.STRATEGY_NAME_PATTERN`, so a constructor and a
  strategy are named the same way.
* Validation collects every problem in one
  :class:`~portfolio_sys.errors.PortfolioContractError`, in the style of the
  factor and strategy contracts.
"""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime
from numbers import Real
from typing import Any

from data_sys.schema import SYMBOL_PATTERN
from factor_sys.universe import UniverseSnapshot
from portfolio_sys.errors import PortfolioContractError
from strategy_sys.contracts import (
    STRATEGY_NAME_PATTERN,
    FactorRequirement,
    StrategyResult,
    StrategySpec,
)

# ---------------------------------------------------------------------------
# Patterns and tolerances
# ---------------------------------------------------------------------------
#: A construction name is the contract's snake_case identifier; reused from the
#: strategy layer, not redefined, so a constructor is named like a strategy.
CONSTRUCTION_NAME_PATTERN: str = STRATEGY_NAME_PATTERN

#: How far ``sum(weight) + cash_weight`` may sit from ``1.0`` and still be one
#: portfolio: binary floating point cannot represent every decimal share
#: exactly, and two correct constructions must not fail on their last bit.
WEIGHT_TOLERANCE: float = 1e-12

_SYMBOL_RE = re.compile(SYMBOL_PATTERN)
_NAME_RE = re.compile(CONSTRUCTION_NAME_PATTERN)
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
    """Validate one date label and return it as a plain :class:`datetime.date`.

    Exactly two forms are accepted: a :class:`datetime.date` and a plain
    ISO-8601 ``"YYYY-MM-DD"`` string.

    A ``datetime`` is refused *even at midnight*, and with it every
    ``pandas.Timestamp`` and ``pandas.NaT`` (both are ``datetime`` subclasses): a
    timestamp is a moment, not the day a portfolio was known to exist, and
    accepting one would silently drop its time of day.  A string must therefore
    be a bare date too -- ``"2024-01-05T00:00"`` is refused rather than parsed.
    """
    if isinstance(value, datetime):
        raise PortfolioContractError(
            f"{label} must be a date, not the timestamp {value.isoformat()}: a portfolio date "
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
    raise PortfolioContractError(
        f"{label} must be a date or an ISO-8601 'YYYY-MM-DD' string, got {value!r}"
    )


def _finite_number(value: Any, label: str) -> tuple[float | None, str | None]:
    """``value`` as a float when it is a finite real number, otherwise why not.

    ``bool`` is refused even though it is an ``int``: ``True`` cannot mean "all of
    the account".  A string is refused rather than parsed, and ``None``, ``NaN``
    and ``±inf`` are refused because none of them sizes a holding.
    """
    if isinstance(value, bool) or not isinstance(value, Real):
        return None, (
            f"{label} must be a real number, got {type(value).__name__}: {value!r}; a bool, a "
            "string or None is refused rather than converted, because a weight has to mean "
            "exactly the number it is"
        )
    number = float(value)
    if not math.isfinite(number):
        return None, (
            f"{label} must be finite, got {value!r}: NaN and ±inf cannot be held, sized or compared"
        )
    return number, None


def _checked_positions(
    value: Any, label: str
) -> tuple[tuple[PortfolioPosition, ...] | None, list[str]]:
    """Validate a ``positions`` field; return the usable holdings and the problems.

    ``None`` comes back (with a problem) when the field is not a tuple at all, so
    a caller can skip the checks that only make sense on a real holding set.
    """
    if not isinstance(value, tuple):
        return None, [
            f"{label} must be a tuple[PortfolioPosition, ...], got {type(value).__name__}: one "
            "holding is one immutable (symbol, weight) pair, so a list, a set or a generator is "
            "refused -- a set has no order and a generator can be read only once; declare "
            "(PortfolioPosition(...), ...)"
        ]

    problems: list[str] = []
    positions = tuple(entry for entry in value if isinstance(entry, PortfolioPosition))
    problems.extend(
        f"{label} entry {entry!r} must be a PortfolioPosition, got {type(entry).__name__}: a "
        "bare ticker or a bare number cannot say which symbol weighs what"
        for entry in value
        if not isinstance(entry, PortfolioPosition)
    )
    problems.extend(
        f"{label} holds {symbol} more than once: one holding per symbol -- two weights of the "
        "same symbol are one position, and adding them up is the caller's job, never a silent "
        "repair here"
        for symbol in _duplicates(position.symbol for position in positions)
    )
    return positions, problems


def _completion_problems(
    label: str, positions: tuple[PortfolioPosition, ...], cash: float
) -> list[str]:
    """Why ``positions`` plus ``cash`` is not one complete portfolio, or nothing.

    A portfolio is complete when its holdings and its cash together are the whole
    account: less is unallocated money, more is leverage.
    """
    stock = sum(position.weight for position in positions)
    total = stock + cash
    if abs(total - 1.0) <= WEIGHT_TOLERANCE:
        return []
    if not positions:
        return [
            f"{label} holds no position and a cash_weight of {cash!r}: a portfolio with no "
            "stock must be entirely in cash (cash_weight 1.0), because the rest of the account "
            "cannot be left unaccounted for"
        ]
    return [
        f"{label} weights must add up to 1.0, got {stock!r} in stock plus {cash!r} in cash = "
        f"{total!r}: a long-only portfolio is either fully allocated or in cash, so nothing may "
        "be left unallocated or borrowed"
    ]


# ---------------------------------------------------------------------------
# PortfolioPosition -- one non-cash holding
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PortfolioPosition:
    """One non-cash holding: a symbol and the share of the account it carries.

    A position is the smallest statement the portfolio layer makes -- *this
    symbol carries this much of the account* -- and it stops there.  It has no
    price, no share count and no market value: 9% of the account is a decision,
    "buy 100 shares" is an execution.

    Parameters
    ----------
    symbol:
        Upper-case ticker matching the project's single ticker rule,
        :data:`data_sys.schema.SYMBOL_PATTERN` (``BRK.B`` is a valid symbol).
        It is stored exactly as given: this contract never upper-cases or strips
        a symbol, because a repaired ticker would silently become a *different*
        security.  ``"CASH"`` is refused by name: cash is the portfolio's own
        :attr:`cash_weight <PortfolioState.cash_weight>` field, never a holding
        that could be priced or ranked.
    weight:
        The share of the account this holding carries: a finite real number
        strictly greater than ``0`` and, being a share, normally at most ``1``.
        ``bool``, ``str``, ``None``, ``NaN``, ``±inf``, ``0`` and negative
        numbers are refused: a zero weight is not a holding (leave it out) and a
        negative one would be a short, which this long-only layer cannot express.
        The value is stored as a ``float``, so ``PortfolioPosition("AAPL", 1)``
        and ``PortfolioPosition("AAPL", 1.0)`` are the same position.

    Raises
    ------
    portfolio_sys.errors.PortfolioContractError
        If the symbol is not a ticker or the weight is not a positive finite
        number; every problem is reported at once.

    Notes
    -----
    Whether ``symbol`` belongs to a universe is deliberately **not** checked
    here: a position only knows itself.  The surrounding
    :class:`PortfolioState` decides what a state may hold, and
    :class:`TargetPortfolio` refuses a holding the strategy did not select.

    Examples
    --------
    >>> position = PortfolioPosition(symbol="NVDA", weight=0.09)
    >>> position.symbol, position.weight
    ('NVDA', 0.09)
    >>> position == PortfolioPosition("NVDA", 0.09)
    True
    """

    symbol: str
    weight: float

    def __post_init__(self) -> None:
        problems: list[str] = []

        symbol = self.symbol
        # ``fullmatch`` rather than ``match``: the pattern is the project's own,
        # but a trailing newline must not slip through as a second ticker.
        if not isinstance(symbol, str) or not _SYMBOL_RE.fullmatch(symbol):
            problems.append(
                f"symbol {symbol!r} must be an upper-case ticker matching {SYMBOL_PATTERN} "
                f"(letters, digits, '.' and '-'), got {type(symbol).__name__}: a holding is a "
                "security, never a blank and never a lower-case spelling"
            )
        elif symbol == "CASH":
            problems.append(
                "symbol 'CASH' is not a holding: cash is the portfolio's own cash_weight field, "
                "so it never becomes a position with a weight of its own"
            )

        weight, problem = _finite_number(self.weight, "weight")
        if problem is not None:
            problems.append(problem)
        elif weight <= 0.0:
            problems.append(
                f"weight must be greater than zero, got {weight!r}: this layer is long-only, so "
                "a zero weight is not a holding (leave it out of the portfolio) and a negative "
                "one would be a short"
            )

        if problems:
            raise PortfolioContractError("invalid PortfolioPosition", problems)

        object.__setattr__(self, "weight", weight)


# ---------------------------------------------------------------------------
# PortfolioState -- what the account actually holds now
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PortfolioState:
    """The portfolio an account actually holds on one date, before a rebalance.

    This is the *current* side of a rebalance, and it is a different statement
    from :class:`TargetPortfolio`: a state says what is held, a target says what
    should be held.  A state is never derived from a strategy result -- it is
    what the account really contains, including the mistakes of previous
    decisions.

    Parameters
    ----------
    as_of_date:
        The day this state is known as of: a plain :class:`datetime.date` or a
        bare ISO-8601 ``"YYYY-MM-DD"`` string.  A ``datetime``, a
        ``pandas.Timestamp`` and a timezone are refused even at midnight -- this
        is a date the ledger can be printed on, not a moment.  The state may be
        dated *earlier* than the decision it is used for (the last rebalance
        usually was); being dated *later* is refused by
        :class:`PortfolioContext`, which is where two dates meet.
    positions:
        A **tuple** of :class:`PortfolioPosition`, duplicate-free by symbol.  A
        list, a set, a generator and a bare position are refused: the tuple is
        the holding set, and its order is the caller's.
    cash_weight:
        The share of the account held in cash: a finite real number in
        ``[0, 1]``.  Cash is a field, never a ``"CASH"`` position.

    Raises
    ------
    portfolio_sys.errors.PortfolioContractError
        If the date is not a plain date, ``positions`` is not a tuple, holds
        something that is not a position or repeats a symbol, ``cash_weight`` is
        not a finite number in ``[0, 1]``, or the holdings plus the cash do not
        add up to ``1.0``.  Every problem is reported at once.

    Notes
    -----
    * **Holdings are not required to be in today's universe.**  A symbol the
      strategy has since dropped, or one that has been delisted, may still be
      held -- and a future backtest has to know it should be taken to zero.  What
      a state *may* hold is therefore the account's own history, not a snapshot
      the strategy chose.
    * An **empty holding set** with ``cash_weight == 1.0`` is the legal
      all-cash portfolio: the state a backtest starts from.
    * The state is immutable and holds no list, so nothing a caller keeps can
      change what the account held.

    Examples
    --------
    >>> state = PortfolioState(
    ...     as_of_date="2024-01-04",
    ...     positions=(PortfolioPosition("AAPL", 0.5), PortfolioPosition("MSFT", 0.4)),
    ...     cash_weight=0.1,
    ... )
    >>> state.symbols
    ('AAPL', 'MSFT')
    >>> state.invested_weight
    0.9
    >>> state.weights_by_symbol["MSFT"]
    0.4
    >>> PortfolioState(date(2024, 1, 4), (), 1.0).invested_weight
    0.0
    """

    as_of_date: date
    positions: tuple[PortfolioPosition, ...]
    cash_weight: float

    def __post_init__(self) -> None:
        problems: list[str] = []

        resolved: date | None = None
        try:
            resolved = _session_date(self.as_of_date, "PortfolioState.as_of_date")
        except PortfolioContractError as exc:
            problems.append(str(exc))

        positions, position_problems = _checked_positions(
            self.positions, "PortfolioState.positions"
        )
        problems.extend(position_problems)

        cash, cash_problem = _finite_number(self.cash_weight, "PortfolioState.cash_weight")
        if cash_problem is not None:
            problems.append(cash_problem)
            cash = None
        elif not 0.0 <= cash <= 1.0:
            problems.append(
                f"PortfolioState.cash_weight must be between 0 and 1 inclusive, got {cash!r}: "
                "cash is a share of the account, so it can neither be borrowed (negative) nor "
                "exceed the whole portfolio"
            )
            cash = None

        if positions is not None and cash is not None:
            problems.extend(_completion_problems("PortfolioState", positions, cash))

        if problems:
            raise PortfolioContractError("invalid PortfolioState", problems)

        object.__setattr__(self, "as_of_date", resolved)
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "cash_weight", cash)

    @property
    def invested_weight(self) -> float:
        """The share of the account held in stock: ``sum(position.weight)``."""
        return sum((position.weight for position in self.positions), 0.0)

    @property
    def symbols(self) -> tuple[str, ...]:
        """The held symbols, in the order the positions were given."""
        return tuple(position.symbol for position in self.positions)

    @property
    def weights_by_symbol(self) -> dict[str, float]:
        """A **new** ``{symbol: weight}`` mapping, in position order.

        Built on every access, so a caller may sort, pop or overwrite it without
        changing the state it came from.
        """
        return {position.symbol: position.weight for position in self.positions}


# ---------------------------------------------------------------------------
# PortfolioConstraints -- the limits a constructor builds within
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PortfolioConstraints:
    """The base limits a first-version constructor has to build within.

    Constraints are the *rules*, not the *answer*: this object says how much cash
    a target should keep and how large one holding may become, while
    :class:`TargetPortfolio` says what was actually chosen.  It is deliberately
    not a constructor, has no algorithm and knows nothing about a selection.

    Parameters
    ----------
    cash_weight:
        The share of the account the target should leave in cash: a finite real
        number in ``[0, 1]``.  ``1`` is the legal all-cash target.
    max_single_weight:
        The cap on any one holding: a finite real number in ``(0, 1]``.  Zero is
        refused because no positive weight could ever meet it, and a cap above
        ``1`` caps nothing.

    Raises
    ------
    portfolio_sys.errors.PortfolioContractError
        If either value is a ``bool``, a string, ``None``, ``NaN``, ``±inf`` or
        outside its range; every problem is reported at once.

    Notes
    -----
    Long-only is **not** a field here, not even a ``True`` one: it is a hard
    property of the whole layer, expressed by the shape of
    :class:`PortfolioPosition` (every weight strictly positive) and by
    ``cash_weight`` never being negative.  A flag that can be turned off is a
    guarantee that can be turned off.

    Whether a particular selection is *feasible* under these limits -- five names
    with 10% cash and a 12% single-name cap, say -- is deliberately not decided
    here.  That question belongs to the constructor that has to solve it, and to
    the test of that constructor; a constraints object that guessed would be a
    constructor in disguise.

    Examples
    --------
    >>> constraints = PortfolioConstraints(cash_weight=0.10, max_single_weight=0.12)
    >>> constraints.cash_weight
    0.1
    >>> constraints.invested_weight
    0.9
    >>> PortfolioConstraints(0.0, 1.0).max_single_weight
    1.0
    """

    cash_weight: float
    max_single_weight: float

    def __post_init__(self) -> None:
        problems: list[str] = []

        cash, cash_problem = _finite_number(self.cash_weight, "cash_weight")
        if cash_problem is not None:
            problems.append(cash_problem)
            cash = None
        elif not 0.0 <= cash <= 1.0:
            problems.append(
                f"cash_weight must be between 0 and 1 inclusive, got {cash!r}: a cash "
                "constraint is a share of the account, so it can neither be negative nor exceed "
                "the whole portfolio"
            )
            cash = None

        limit, limit_problem = _finite_number(self.max_single_weight, "max_single_weight")
        if limit_problem is not None:
            problems.append(limit_problem)
            limit = None
        elif not 0.0 < limit <= 1.0:
            problems.append(
                f"max_single_weight must be greater than 0 and at most 1, got {limit!r}: no "
                "positive holding could meet a cap of zero, and a cap above 1 caps nothing"
            )
            limit = None

        if problems:
            raise PortfolioContractError("invalid PortfolioConstraints", problems)

        object.__setattr__(self, "cash_weight", cash)
        object.__setattr__(self, "max_single_weight", limit)

    @property
    def invested_weight(self) -> float:
        """The share of the account the constraints leave for stock: ``1 - cash``."""
        return 1.0 - self.cash_weight


# ---------------------------------------------------------------------------
# PortfolioConstructionSpec -- which rule built the weights
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PortfolioConstructionSpec:
    """The complete identity of one portfolio construction rule.

    ``equal_weight / 1.0.0`` or ``score_weighted / 1.0.0``: a name and a version,
    and nothing else.  The pair is what a stored target, a report row or a
    backtest ledger needs in order to say *which rule* produced these weights,
    the same way :class:`~strategy_sys.contracts.StrategySpec` says which rule
    produced a selection.

    Parameters
    ----------
    name:
        snake_case identifier -- lower-case letters, digits and ``_`` only,
        validated with the strategy layer's own
        :data:`~strategy_sys.contracts.STRATEGY_NAME_PATTERN` (reused, not
        redefined, so a constructor is named exactly like a strategy).  The name
        is stored exactly as given: this contract never lower-cases or strips it,
        because a repaired name would silently become a *different* rule.
    version:
        Non-empty label of the rule's behaviour, e.g. ``"1.0.0"``.  Two
        constructions that assign weights differently need two versions: the
        same name plus a new version is a different rule.

    Raises
    ------
    portfolio_sys.errors.PortfolioContractError
        If the name is not snake_case or the version is blank; every problem is
        reported at once.

    Notes
    -----
    * **No factor version here.**  Which factors a target was built from is
      already recorded by the :class:`~strategy_sys.contracts.StrategyResult` it
      came from: a constructor reads a selection, so the factor versions travel
      with the result, never with the construction rule.
    * **No parameters here either.**  How much cash to keep and how large one
      holding may become are per-run decisions and live in
      :class:`PortfolioConstraints`.  The spec names the *rule*; the constraints
      hold the *numbers* it was applied with.
    * Frozen and hashable, so a spec can key a registry or a cache later.

    Examples
    --------
    >>> spec = PortfolioConstructionSpec(name="equal_weight", version="1.0.0")
    >>> spec.name, spec.version
    ('equal_weight', '1.0.0')
    >>> spec == PortfolioConstructionSpec("equal_weight", "1.0.0")
    True
    >>> len({spec, PortfolioConstructionSpec("equal_weight", "2.0.0")})
    2
    """

    name: str
    version: str

    def __post_init__(self) -> None:
        problems: list[str] = []

        name = self.name
        if not isinstance(name, str) or not name.strip():
            problems.append(f"name must be a non-empty string, got {name!r}")
        elif not _NAME_RE.fullmatch(name):
            problems.append(
                f"name {name!r} must be snake_case: lower-case letters, digits and '_' only "
                f"(matches {CONSTRUCTION_NAME_PATTERN})"
            )

        version = self.version
        if not isinstance(version, str) or not version.strip():
            problems.append(f"version must be a non-empty string, got {version!r}")

        if problems:
            raise PortfolioContractError("invalid PortfolioConstructionSpec", problems)


# ---------------------------------------------------------------------------
# PortfolioContext -- everything one construction may read
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PortfolioContext:
    """The complete input of one portfolio construction.

    A constructor receives this object and nothing else, so its whole input is
    visible in one place: *on this date, this strategy selected these symbols,
    the account currently holds this, and these are the limits*.  Everything a
    constructor must not touch -- prices, calendar, files, factor panels -- is
    absent, which is exactly why a constructor can be tested without any data on
    disk.

    Parameters
    ----------
    as_of_date:
        The decision date: a plain :class:`datetime.date` or a bare ISO-8601
        ``"YYYY-MM-DD"`` string.  A ``datetime`` is refused even at midnight.  It
        must equal ``strategy_result.as_of_date``: a target built from another
        day's selection would silently mix two decisions.
    strategy_result:
        The :class:`~strategy_sys.contracts.StrategyResult` to weight -- *what*
        is selected today, with the exact strategy and factor versions that
        selected it.  It is stored as given, never copied, so the context cannot
        drift away from the result it describes.
    current_state:
        The :class:`PortfolioState` the account holds before this rebalance.
        Dated **no later than** ``as_of_date``: an older state is normal (the
        last rebalance was a while ago), a newer one is look-ahead.
    constraints:
        The :class:`PortfolioConstraints` to build within.

    Raises
    ------
    portfolio_sys.errors.PortfolioContractError
        If the date is not a plain date, any of the three objects has the wrong
        type, the result is dated another day, or the current state is dated
        after the decision.  Wrong types are collected together, so one context
        reports every field that is wrong.

    Notes
    -----
    * The context does **not** check that the constraints can be met by this
      selection, and does not check that the current state's holdings are in
      today's universe: the first is the constructor's feasibility question, the
      second is the account's history.  Only *knowledge order* is enforced.
    * Nothing here is copied or re-derived: :attr:`strategy_spec`,
      :attr:`factor_requirements`, :attr:`universe` and
      :attr:`selected_symbols` are read straight off :attr:`strategy_result`, so
      a constructor can label its output with the exact rule and factor versions
      that produced the selection.
    """

    as_of_date: date
    strategy_result: StrategyResult
    current_state: PortfolioState
    constraints: PortfolioConstraints

    def __post_init__(self) -> None:
        problems: list[str] = []

        resolved: date | None = None
        try:
            resolved = _session_date(self.as_of_date, "PortfolioContext.as_of_date")
        except PortfolioContractError as exc:
            problems.append(str(exc))

        result = self.strategy_result
        if not isinstance(result, StrategyResult):
            problems.append(
                "strategy_result must be a StrategyResult (the strategy layer's validated "
                f"cross-section), got {type(result).__name__}: a constructor weights a selection "
                "plus the exact strategy and factor versions it came from, never a bare frame or "
                "a list of tickers"
            )
            result = None  # type: ignore[assignment]

        state = self.current_state
        if not isinstance(state, PortfolioState):
            problems.append(
                "current_state must be a PortfolioState (what the account holds now), got "
                f"{type(state).__name__}: a target cannot be a diff against nothing"
            )
            state = None  # type: ignore[assignment]

        if not isinstance(self.constraints, PortfolioConstraints):
            problems.append(
                "constraints must be a PortfolioConstraints, got "
                f"{type(self.constraints).__name__}: the rules a target was built within travel "
                "with the target"
            )

        if resolved is not None and result is not None and result.as_of_date != resolved:
            problems.append(
                f"strategy_result.as_of_date is {result.as_of_date.isoformat()}, but the context "
                f"is dated {resolved.isoformat()}: a constructor weights the selection of the day "
                "it is asked about, never another day's"
            )

        if resolved is not None and state is not None and state.as_of_date > resolved:
            problems.append(
                f"current_state.as_of_date is {state.as_of_date.isoformat()}, later than the "
                f"decision date {resolved.isoformat()}: a portfolio cannot be built from a state "
                "that only becomes known after the decision -- that is look-ahead"
            )

        if problems:
            raise PortfolioContractError("invalid PortfolioContext", problems)

        object.__setattr__(self, "as_of_date", resolved)

    @property
    def selected_symbols(self) -> tuple[str, ...]:
        """``strategy_result.selected_symbols``: what this decision selected."""
        return self.strategy_result.selected_symbols

    @property
    def strategy_spec(self) -> StrategySpec:
        """``strategy_result.strategy_spec``: which rule selected these symbols."""
        return self.strategy_result.strategy_spec

    @property
    def factor_requirements(self) -> tuple[FactorRequirement, ...]:
        """``strategy_result.factor_requirements``: the exact factors *and* versions."""
        return self.strategy_result.factor_requirements

    @property
    def universe(self) -> UniverseSnapshot:
        """``strategy_result.universe``: the pool the selection was made in."""
        return self.strategy_result.universe


# ---------------------------------------------------------------------------
# TargetPortfolio -- what the account should hold after the decision
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TargetPortfolio:
    """The weights a constructor wants the account to hold after this decision.

    A target is **not** an order and **not** a fill: it says *9% in AVGO, 9% in
    NVDA, 10% in cash*, and it is the execution layer's job -- a later task -- to
    turn that into trades, prices and share counts.  Nothing here has a price, a
    quantity, a commission, a turnover or a return, and nothing here is dated
    later than the decision.

    Parameters
    ----------
    as_of_date:
        The decision date: a plain :class:`datetime.date` or a bare ISO-8601
        ``"YYYY-MM-DD"`` string (a ``datetime`` is refused even at midnight).
        It must equal ``source_result.as_of_date``.
    construction_spec:
        The :class:`PortfolioConstructionSpec` of the rule that produced these
        weights -- which constructor, at which version.
    source_result:
        The :class:`~strategy_sys.contracts.StrategyResult` the weights were
        built from.  It is stored as given, and every strategy/factor/universe
        property below reads through it, so the chain
        *result -> target -> future backtest ledger* cannot lose the exact
        ``(factor_name, factor_version)`` pairs.
    constraints:
        The :class:`PortfolioConstraints` the weights were built within, as
        given.  See *Notes* for what that does and does not promise.
    positions:
        A **tuple** of :class:`PortfolioPosition`, duplicate-free by symbol, each
        symbol taken from ``source_result.selected_symbols``.  A subset is legal:
        a constructor may drop a candidate the constraints leave no room for, but
        it may never add a symbol the strategy did not select.
    cash_weight:
        The share left in cash: a finite real number in ``[0, 1]``.
        ``sum(position.weight) + cash_weight`` must be ``1.0`` within
        :data:`WEIGHT_TOLERANCE`.

    Raises
    ------
    portfolio_sys.errors.PortfolioContractError
        If a field has the wrong type, the date is not a plain date, the target is
        dated differently from its result, ``positions`` is not a tuple, holds
        something that is not a position, repeats a symbol or names a symbol the
        strategy did not select, a weight is not positive and finite,
        ``cash_weight`` is outside ``[0, 1]``, or the weights do not add up to
        ``1.0``.  Every problem is reported at once.

    Notes
    -----
    * **The constraints are recorded, not re-enforced.**  Whether a selection is
      *feasible* under ``cash_weight`` and ``max_single_weight`` -- five names,
      10% cash, a 12% cap -- is the constructor's own decision, made when it had
      the selection in front of it; this object stores what it built under and
      refuses to guess a second opinion.  Its numbers must agree with
      :class:`PortfolioConstraints`, whose job that is.
    * **Order is the constructor's.**  :attr:`positions` keeps the order it was
      given and :attr:`target_symbols` reports it; nothing is re-sorted by weight
      or by symbol, because a silent re-order would change a stored target's
      bytes for no reason.
    * An **empty target** is legal only with ``cash_weight == 1.0``: "hold
      nothing" and "be 100% in cash" are the same statement, and a target that
      holds nothing with cash left over is an incomplete answer.

    """

    as_of_date: date
    construction_spec: PortfolioConstructionSpec
    source_result: StrategyResult
    constraints: PortfolioConstraints
    positions: tuple[PortfolioPosition, ...]
    cash_weight: float

    def __post_init__(self) -> None:
        problems: list[str] = []

        resolved: date | None = None
        try:
            resolved = _session_date(self.as_of_date, "TargetPortfolio.as_of_date")
        except PortfolioContractError as exc:
            problems.append(str(exc))

        if not isinstance(self.construction_spec, PortfolioConstructionSpec):
            problems.append(
                "construction_spec must be a PortfolioConstructionSpec, got "
                f"{type(self.construction_spec).__name__}: a target has to name the rule *and* "
                "the version that produced its weights"
            )

        result = self.source_result
        if not isinstance(result, StrategyResult):
            problems.append(
                "source_result must be a StrategyResult, got "
                f"{type(result).__name__}: a target that cannot name the selection it weights "
                "cannot be traced back to a strategy or to an exact factor version"
            )
            result = None  # type: ignore[assignment]

        if not isinstance(self.constraints, PortfolioConstraints):
            problems.append(
                "constraints must be a PortfolioConstraints, got "
                f"{type(self.constraints).__name__}: the limits a target was built within travel "
                "with the target, so a later layer never has to re-invent them"
            )

        positions, position_problems = _checked_positions(
            self.positions, "TargetPortfolio.positions"
        )
        problems.extend(position_problems)

        cash, cash_problem = _finite_number(self.cash_weight, "TargetPortfolio.cash_weight")
        if cash_problem is not None:
            problems.append(cash_problem)
            cash = None
        elif not 0.0 <= cash <= 1.0:
            problems.append(
                f"TargetPortfolio.cash_weight must be between 0 and 1 inclusive, got {cash!r}: "
                "cash is a share of the account, so a target can neither borrow (negative) nor "
                "hold more than the whole portfolio"
            )
            cash = None

        if resolved is not None and result is not None and result.as_of_date != resolved:
            problems.append(
                f"as_of_date is {resolved.isoformat()}, but source_result.as_of_date is "
                f"{result.as_of_date.isoformat()}: a target weights the selection of its own "
                "decision date, never another day's"
            )

        if result is not None and positions is not None:
            selected = result.selected_symbols
            outsiders = [
                position.symbol for position in positions if position.symbol not in selected
            ]
            if outsiders:
                problems.append(
                    f"positions holds {_describe(outsiders)}, which {result.strategy_name} did "
                    f"not select on {result.as_of_date.isoformat()}: a target may hold a subset "
                    f"of the selected symbols {_describe(selected, limit=5)}, never a name the "
                    "strategy did not choose"
                )

        if positions is not None and cash is not None:
            problems.extend(_completion_problems("TargetPortfolio", positions, cash))

        if problems:
            raise PortfolioContractError("invalid TargetPortfolio", problems)

        object.__setattr__(self, "as_of_date", resolved)
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "cash_weight", cash)

    @property
    def strategy_spec(self) -> StrategySpec:
        """``source_result.strategy_spec``: which rule selected the candidates."""
        return self.source_result.strategy_spec

    @property
    def factor_requirements(self) -> tuple[FactorRequirement, ...]:
        """``source_result.factor_requirements``: the exact factors *and* versions."""
        return self.source_result.factor_requirements

    @property
    def universe(self) -> UniverseSnapshot:
        """``source_result.universe``: the pool the candidates were selected from."""
        return self.source_result.universe

    @property
    def target_symbols(self) -> tuple[str, ...]:
        """The held symbols, in the order the constructor gave them."""
        return tuple(position.symbol for position in self.positions)

    @property
    def target_weights_by_symbol(self) -> dict[str, float]:
        """A **new** ``{symbol: weight}`` mapping, in position order.

        Built on every access, so a caller may edit, sort or discard it without
        changing the target it came from.
        """
        return {position.symbol: position.weight for position in self.positions}

    @property
    def invested_weight(self) -> float:
        """The share of the account the target puts in stock: ``sum(weight)``."""
        return sum((position.weight for position in self.positions), 0.0)


# ---------------------------------------------------------------------------
# PortfolioConstructor -- the interface, and nothing else
# ---------------------------------------------------------------------------
class PortfolioConstructor(ABC):
    """Abstract base class every portfolio constructor implements.

    A constructor is a pure function of a :class:`PortfolioContext`: it declares
    what it is through :attr:`spec`, reads *only* the context it is handed, and
    returns a validated :class:`TargetPortfolio`.  It must not read files, know a
    path, reach into a factor store or a strategy runner, call a trading
    calendar, fetch a price or produce orders, fills or trades -- a constructor
    that can be built from an in-memory context is a constructor that can be
    tested without any data on disk.

    This class deliberately contains **no algorithm**: *how* weights are decided
    is what later implementations differ in::

        class EqualWeightConstructor(PortfolioConstructor):
            spec = PortfolioConstructionSpec(name="equal_weight", version="1.0.0")

            def construct(self, context: PortfolioContext) -> TargetPortfolio:
                ...  # a later task

    Subclasses either declare ``spec`` as a class attribute (the usual, simplest
    form) or implement it as a property; both satisfy the abstract requirement.
    Because the interface does not change when a rule is added, a caller can hold
    "some constructor" and hand it any context -- the way a strategy runner holds
    "some strategy".
    """

    @property
    @abstractmethod
    def spec(self) -> PortfolioConstructionSpec:
        """The constructor's immutable identity (:class:`PortfolioConstructionSpec`)."""

    @abstractmethod
    def construct(self, context: PortfolioContext) -> TargetPortfolio:
        """Return the target weights this rule wants for ``context``.

        Parameters
        ----------
        context:
            The point-in-time input a constructor may read: one decision date, one
            strategy result, the current portfolio state and the constraints.  A
            ``construct`` call must not mutate it and must not depend on anything
            outside it, so the same context always yields the same target.

        Returns
        -------
        TargetPortfolio
            Already validated: dated the context's ``as_of_date``, naming this
            constructor's own :attr:`spec`, holding this context's ``constraints``
            and ``strategy_result``, and covering only symbols the strategy
            selected -- so the target names the exact strategy and the exact
            factor versions it was built from.

        Notes
        -----
        Whether the constraints can be satisfied by this selection is the
        constructor's own decision (and its own test): it may drop candidates,
        keep more cash or refuse the context in its own words.  The contract only
        checks that the answer is arithmetically a long-only portfolio whose
        holdings were really selected.
        """

    def __repr__(self) -> str:
        return f"{type(self).__name__}(spec={self.spec!r})"


__all__ = [
    "CONSTRUCTION_NAME_PATTERN",
    "WEIGHT_TOLERANCE",
    "PortfolioConstraints",
    "PortfolioConstructionSpec",
    "PortfolioConstructor",
    "PortfolioContext",
    "PortfolioPosition",
    "PortfolioState",
    "TargetPortfolio",
]
