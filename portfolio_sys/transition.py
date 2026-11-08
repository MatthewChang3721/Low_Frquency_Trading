"""From the portfolio the account holds to the portfolio a rule wants: the difference.

What this module is
-------------------
``portfolio_sys.transition`` answers the one question that sits between a
:class:`~portfolio_sys.contracts.TargetPortfolio` and anything that could act on it:
*between what the account holds and what the rule asked for, what actually changes?*::

    PortfolioState + TargetPortfolio  ->  PortfolioTransition  ->  (execution)

Three objects, three statements
-------------------------------
``PortfolioWeightChange``
    one symbol: the weight it carries now, the weight it should carry, and the
    difference between them.
``PortfolioTransition``
    one decision date: every symbol of the two portfolios with one row each, in a fixed
    order, plus the two objects they came from and the arithmetic implied by them.
``build_portfolio_transition``
    the one function that turns two validated portfolios into one validated transition.

What it answers
---------------
* how much of the account is held now, and how much should be held after the decision;
* what each symbol's weight change is -- including a holding that leaves (target weight
  ``0``) and a name that arrives (current weight ``0``);
* which holdings are unchanged, increased, decreased, entered and exited;
* the one-sided weight volume this rebalance needs (:attr:`PortfolioTransition.turnover`).

What it deliberately does not answer
------------------------------------
* how many shares to trade, at what price, on which day, and whether it filled;
* what the commission, the slippage or the market impact of that would be;
* what the net value, the return or the profit becomes.

A weight change is a *decision*; a share count needs a price and a fill, and those need a
market-data read and a trading calendar -- neither of which this layer has.  Turning
weights into trades belongs to the execution layer, and the cost and net-value curve of
doing so belongs to the backtest layer after it.  Nothing here is an order.

Cash is a field, and so is a cash change
----------------------------------------
``"CASH"`` is not a symbol here either.  The cash difference is
:attr:`PortfolioTransition.cash_weight_change`, computed from the two ``cash_weight``
fields the portfolios already carry; a row for ``"CASH"`` is refused by name, because a
pseudo-holding that could be priced, ranked or selected is exactly what the two portfolio
objects refuse to have.  The complete account is therefore described by the rows *plus*
one cash number, never by a row that pretends to be cash.

Zero is a legal weight here, unlike in a holding
------------------------------------------------
:class:`~portfolio_sys.contracts.PortfolioPosition` refuses a weight of zero, because a
zero-weight holding is not a holding.  A change is the one place where zero is the whole
point: the current side of an entry is ``0`` (nothing held yet) and the target side of an
exit is ``0`` (nothing wanted any more).  Both are stored exactly as ``0.0``, so
"not held" is one value, not a small number that means almost the same thing.

Turnover: one side, never both
------------------------------
:attr:`PortfolioTransition.turnover` is the single-sided weight volume needed to move the
account from the current weights to the target weights::

    turnover = 0.5 * (sum(|weight_change|) + |cash_weight_change|)

On both sides a portfolio is complete -- ``sum(weight) + cash_weight == 1`` -- so the
stock changes and the cash change cancel out and the formula reduces to
``max(buy_weight, sell_weight)``: one side of the rebalance, counted once.

A pure swap makes the difference clear.  Selling ``A`` at 100% and buying ``B`` at 100%
is a 100% rebalance, not 200%: ``buy_weight`` and ``sell_weight`` are each 100%, and
adding them would count the same dollar twice, once as a sale and once as a purchase.
All-cash to fully invested is ``turnover == buy_weight`` (there is nothing to sell); the
reverse is ``turnover == sell_weight``.  Cash moving into or out of stock is counted once
as well, through the stock side, which is why the cash term is added rather than treated
as a third movement.

Turnover is a **weight**, not an amount: it says "10% of the account changed hands", not
"ten thousand dollars", and it carries no tax, no spread and no commission.

The order of the rows
---------------------
:attr:`PortfolioTransition.changes` is the complete union of the two symbol sets, in one
stable order:

1. every symbol of ``target_portfolio.target_symbols``, in the target's own order;
2. then every symbol of ``current_state.symbols`` the target does not hold, in the
   state's own order -- the exits.

So the portfolio *after* the decision is read first, top to bottom, exactly as the
constructor ordered it, and the names that leave follow.  Nothing is re-sorted by weight,
by symbol, by score or by direction, and nothing depends on the order of a dictionary:
two identical inputs produce one identical sequence, which is what makes two transitions
comparable line by line.

Complete, or refused
--------------------
:class:`PortfolioTransition` accepts no hand-made difference it cannot verify.  It checks
that the two sides are the portfolio objects, that the knowledge they carry runs forwards
in time, and that ``changes`` is a tuple with exactly one row per symbol of the union, in
the documented order, agreeing with the weights of the state and of the target and
consistent with its own ``weight_change``.  Every problem found is reported at once, in
one :class:`~portfolio_sys.errors.PortfolioContractError`, so a caller sees the whole list
rather than one line per attempt.

What it cannot check is where the target came from: a
:class:`~portfolio_sys.contracts.TargetPortfolio` does not record the state it was
constructed from, so nobody can prove that this target was built *for* this state.  That is
deliberate -- a future turnover-aware constructor reads ``PortfolioContext.current_state``
itself -- and it is why a transition only ever asserts "these are the weights of the two
objects you handed me", never "this was the right decision".

Not implemented, on purpose
---------------------------
No order, no share count, no execution date, no price, no fill, no commission, no
slippage, no market impact, no net value, no return, no backtest, no CLI, no runner, no
registry and no constructor.  Also no rebalance calendar: whether the decision date is a
trading day is nobody's question here, because nothing is executed.

Purity, layering and reuse
--------------------------
Standard library plus :mod:`portfolio_sys.contracts` and :mod:`portfolio_sys.errors`.
No pandas, no numpy, no ``data_sys``, no ``factor_sys``, no ``strategy_sys``, no market
data, no factor panel, no calendar, no Parquet/DuckDB, no file, no network, no runner and
no registry.  It never calls ``construct()``, never reads a frame and never writes
anything: two objects in, one object out, and the inputs are left exactly as they were.

The rules it needs are the contract's own, imported rather than re-declared -- the ticker
pattern, the date rule, the finite-number rule, the duplicate rule and the bounded list
rendering of an error message -- so a symbol this module accepts is exactly a symbol a
:class:`~portfolio_sys.contracts.PortfolioPosition` accepts, and a date it accepts is
exactly a date a portfolio accepts.  One rule, one definition: the same reuse
``factor_sys.universe`` makes of ``factor_sys.contracts``.

A note on three annotations
---------------------------
:attr:`PortfolioTransition.strategy_spec`,
:attr:`PortfolioTransition.factor_requirements` and
:attr:`PortfolioTransition.universe` are pass-throughs: they return the exact objects the
target already carries, so a transition never restates a strategy, a factor version or a
universe.  Their types live in ``strategy_sys`` and ``factor_sys``, and this module may
not import those layers -- that is the boundary -- so those three properties name their
return type in the docstring instead of in an annotation.
:attr:`PortfolioTransition.construction_spec` and :attr:`PortfolioTransition.constraints`
*are* annotated, because ``portfolio_sys.contracts`` already owns those two types.

Example
-------
Two portfolios, one decision, and the rows it produces (every number is a weight of the
account, never an amount)::

    current  2025-01-02:  AAPL 60%,  MSFT 30%,  cash 10%
    target   2025-01-09:  MSFT 45%,  NVDA 45%,  cash 10%

    symbol  current  target    change
    MSFT    30%      45%       +15%
    NVDA     0%      45%       +45%
    AAPL    60%       0%       -60%

    buy_weight 60%, sell_weight 60%, cash_weight_change 0%, turnover 60%
    entering (NVDA,), exiting (AAPL,), changed (MSFT, NVDA, AAPL)

``MSFT`` and ``NVDA`` come first because they are the target's holdings; ``AAPL``, which
the target no longer holds, is appended as the exit.

>>> change = PortfolioWeightChange(
...     symbol="AAPL", current_weight=0.6, target_weight=0.0, weight_change=-0.6
... )
>>> change.is_exit, change.is_sell, change.is_unchanged
(True, True, False)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

from portfolio_sys.contracts import (
    _SYMBOL_RE,
    WEIGHT_TOLERANCE,
    PortfolioConstraints,
    PortfolioConstructionSpec,
    PortfolioState,
    TargetPortfolio,
    _describe,
    _duplicates,
    _finite_number,
    _session_date,
)
from portfolio_sys.errors import PortfolioContractError


# ---------------------------------------------------------------------------
# Small shared helpers: the contract's rules, composed -- never re-declared
# ---------------------------------------------------------------------------
def _symbol_problems(symbol: Any) -> list[str]:
    """Why ``symbol`` cannot name a weight change, or nothing.

    The ticker rule is the contract's own compiled pattern, so a name accepted here is
    exactly a name a :class:`~portfolio_sys.contracts.PortfolioPosition` accepts.  Only the
    ``"CASH"`` refusal needs a sentence of its own, because here cash *is* part of the
    comparison and must not arrive as a pseudo-holding row.
    """
    if not isinstance(symbol, str) or not _SYMBOL_RE.fullmatch(symbol):
        return [
            f"symbol {symbol!r} must be an upper-case ticker matching {_SYMBOL_RE.pattern} "
            f"(letters, digits, '.' and '-'), got {type(symbol).__name__}: a weight change names "
            "a security, and a stripped or lower-case spelling would silently name a different one"
        ]
    if symbol == "CASH":
        return [
            "symbol 'CASH' is not a weight change: cash is a field of the portfolio, so the cash "
            "difference is reported as cash_weight_change, never as a row that pretends to hold "
            "cash as a security"
        ]
    return []


def _weight_problems(value: Any, label: str) -> tuple[float | None, list[str]]:
    """``value`` as a share of the account in ``[0, 1]``, and why it is not one.

    Zero is legal here -- unlike a holding's weight -- because a change is exactly where
    "nothing is held on this side" is spelled: ``0`` on the current side of an entry, ``0``
    on the target side of an exit.
    """
    number, problem = _finite_number(value, label)
    if problem is not None:
        return None, [problem]
    if not 0.0 <= number <= 1.0:
        return None, [
            f"{label} must be between 0 and 1 inclusive, got {number!r}: a weight is a share of "
            "the account, so it can neither be negative nor exceed the whole portfolio; zero is "
            "legal here and means nothing is held on that side"
        ]
    return number, []


# ---------------------------------------------------------------------------
# PortfolioWeightChange -- one symbol, before and after
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PortfolioWeightChange:
    """One symbol's weight before the decision, after it, and the difference.

    This is the smallest statement the transition layer makes -- *this symbol carried this
    much of the account, should carry that much, so its weight moves by the difference* --
    and it stops there.  It holds no price, no share count and no order side: ``is_buy``
    describes the *target*, not a trade, and whether anything is bought and how much
    belongs to the execution layer.

    Parameters
    ----------
    symbol:
        Upper-case ticker matching the contract's single ticker rule (the project's one
        pattern, reused rather than copied).  It is stored exactly as given: a repaired
        ticker would silently become a *different* security.  ``"CASH"`` is refused by name
        -- cash is the transition's
        :attr:`~PortfolioTransition.cash_weight_change`, never a row.
    current_weight:
        The share of the account this symbol carried *before* the decision: a finite real
        number in ``[0, 1]``.  Zero is legal and means the symbol was not held, which is
        what the current side of an entry says.
    target_weight:
        The share of the account the target wants: a finite real number in ``[0, 1]``.
        Zero is legal and means the holding is closed, which is what the target side of an
        exit says.
    weight_change:
        ``target_weight - current_weight``.  It is **checked, never recomputed**: a caller
        whose number disagrees with its own two weights gets a refusal, because silently
        overwriting a supplied field would hide the disagreement instead of reporting it.

    Raises
    ------
    portfolio_sys.errors.PortfolioContractError
        If the symbol is not a ticker or is ``"CASH"``, if either weight is not a finite
        number in ``[0, 1]``, if ``weight_change`` is not a finite number, or if it differs
        from ``target_weight - current_weight`` by more than
        :data:`~portfolio_sys.contracts.WEIGHT_TOLERANCE`.  Every problem is reported at
        once.

    Notes
    -----
    * **Immutable.**  The object is frozen, so a change cannot drift after it has been
      compared, summed or reported.
    * **Not an order.**  ``is_buy`` and ``is_sell`` say which way the *target weight* moves.
      They are not an order side, not a fill and not a tax event.
    * The tolerance exists for the same reason it does everywhere in this layer: binary
      floating point cannot represent every decimal share exactly, so a change of
      ``0.1 + 0.2 - 0.3`` is the same statement as ``0``.

    Examples
    --------
    >>> change = PortfolioWeightChange(
    ...     symbol="NVDA", current_weight=0.0, target_weight=0.2, weight_change=0.2
    ... )
    >>> change.is_entry, change.is_buy, change.is_sell
    (True, True, False)
    >>> PortfolioWeightChange("AAPL", 0.3, 0.3, 0.0).is_unchanged
    True
    >>> PortfolioWeightChange("AAPL", 0.6, 0.0, -0.6).is_exit
    True
    """

    symbol: str
    current_weight: float
    target_weight: float
    weight_change: float

    def __post_init__(self) -> None:
        problems: list[str] = list(_symbol_problems(self.symbol))

        current, current_problems = _weight_problems(
            self.current_weight, "PortfolioWeightChange.current_weight"
        )
        problems.extend(current_problems)

        target, target_problems = _weight_problems(
            self.target_weight, "PortfolioWeightChange.target_weight"
        )
        problems.extend(target_problems)

        change, change_problem = _finite_number(
            self.weight_change, "PortfolioWeightChange.weight_change"
        )
        if change_problem is not None:
            problems.append(change_problem)
            change = None

        if current is not None and target is not None and change is not None:
            expected = target - current
            if abs(change - expected) > WEIGHT_TOLERANCE:
                problems.append(
                    f"weight_change is {change!r}, but target_weight - current_weight is "
                    f"{expected!r} for {self.symbol!r}: a row restates the difference it was "
                    "computed from, so a row that disagrees with itself is refused rather than "
                    "repaired -- overwriting the caller's number would hide the disagreement"
                )

        if problems:
            raise PortfolioContractError("invalid PortfolioWeightChange", problems)

        object.__setattr__(self, "current_weight", current)
        object.__setattr__(self, "target_weight", target)
        object.__setattr__(self, "weight_change", change)

    @property
    def is_buy(self) -> bool:
        """Whether the target asks for **more** of this symbol: ``change > tolerance``.

        A direction of the target weight, never an order side and never a fill.  A symbol
        whose change sits inside :data:`~portfolio_sys.contracts.WEIGHT_TOLERANCE` is neither
        bought nor sold, so a floating-point hair is never reported as a trade.
        """
        return self.weight_change > WEIGHT_TOLERANCE

    @property
    def is_sell(self) -> bool:
        """Whether the target asks for **less** of this symbol: ``change < -tolerance``.

        A direction of the target weight, never an order side and never a fill.
        """
        return self.weight_change < -WEIGHT_TOLERANCE

    @property
    def is_unchanged(self) -> bool:
        """Whether the symbol keeps its weight, within the tolerance.

        Exactly one of :attr:`is_buy`, :attr:`is_sell` and :attr:`is_unchanged` is true for
        every row, so no symbol falls between them.
        """
        return abs(self.weight_change) <= WEIGHT_TOLERANCE

    @property
    def is_entry(self) -> bool:
        """Whether this is a brand-new holding: nothing carried, something wanted.

        ``current_weight == 0`` and ``target_weight > 0``, both read exactly: a state either
        holds a name or does not, so there is no tolerance on "was not held".
        """
        return self.current_weight == 0.0 and self.target_weight > 0.0

    @property
    def is_exit(self) -> bool:
        """Whether this holding is closed: something carried, nothing wanted.

        ``current_weight > 0`` and ``target_weight == 0``, both read exactly -- a target
        either wants a name or does not.
        """
        return self.current_weight > 0.0 and self.target_weight == 0.0


# ---------------------------------------------------------------------------
# Transition-level checks: coverage, order and agreement with the two sources
# ---------------------------------------------------------------------------
def _expected_symbols(state: PortfolioState, target: TargetPortfolio) -> tuple[str, ...]:
    """The union of the two symbol sets, in the order a transition must report it.

    The target's symbols come first, in the target's own order -- the portfolio *after* the
    decision, exactly as the constructor ordered it -- and the names only the current state
    holds follow, in the state's own order.  Nothing is sorted: the order is a promise, not
    a preference.
    """
    wanted = target.target_symbols
    wanted_set = set(wanted)
    return wanted + tuple(symbol for symbol in state.symbols if symbol not in wanted_set)


def _coverage_problems(
    rows: tuple[PortfolioWeightChange, ...],
    state: PortfolioState | None,
    target: TargetPortfolio | None,
) -> list[str]:
    """Why ``rows`` is not one row per symbol of the union, in the promised order."""
    if state is None or target is None:
        return []

    expected = _expected_symbols(state, target)
    actual = tuple(row.symbol for row in rows)
    expected_set = set(expected)
    actual_set = set(actual)
    problems: list[str] = []

    missing = [symbol for symbol in expected if symbol not in actual_set]
    if missing:
        problems.append(
            f"changes omits {_describe(missing)}, which the two portfolios hold: every symbol of "
            "the union gets exactly one row, because a missing exit leaves a holding the target "
            "no longer wants and a missing entry leaves one it does"
        )

    extra = [symbol for symbol in actual if symbol not in expected_set]
    if extra:
        problems.append(
            f"changes holds {_describe(extra)}, which neither portfolio holds: a transition "
            "describes the move between these two portfolios, so it cannot invent a third symbol"
        )

    repeated = _duplicates(actual)
    if not missing and not extra and not repeated and actual != expected:
        mismatches = [
            position
            for position, (got, want) in enumerate(zip(actual, expected, strict=False))
            if got != want
        ]
        index = mismatches[0] if mismatches else min(len(actual), len(expected))
        got = repr(actual[index]) if index < len(actual) else "nothing (changes ends)"
        want = repr(expected[index]) if index < len(expected) else "nothing (the union ends)"
        problems.append(
            f"changes is ordered {_describe(actual, limit=4)}, but the target's symbols come "
            f"first and the exits after them, {_describe(expected, limit=4)}: position {index} "
            f"holds {got} where {want} belongs -- rows follow target_portfolio.target_symbols "
            "and then the current-only symbols in current_state order, never a weight, a score "
            "or a direction"
        )
    return problems


def _source_problems(
    rows: tuple[PortfolioWeightChange, ...],
    state: PortfolioState | None,
    target: TargetPortfolio | None,
) -> list[str]:
    """Why a row's weights are not the weights of the portfolio it claims to describe.

    A row restates two numbers it did not measure: the weight the state already carries and
    the weight the target already asked for.  Restating is not re-measuring, so a
    disagreement is a broken row, never a new opinion about the account.
    """
    problems: list[str] = []

    if state is not None:
        held = state.weights_by_symbol
        for row in rows:
            weight = held.get(row.symbol)
            if weight is None:
                if abs(row.current_weight) > WEIGHT_TOLERANCE:
                    problems.append(
                        f"changes gives {row.symbol} a current_weight of "
                        f"{row.current_weight!r}, but current_state does not hold it: the "
                        "current side of a name the account does not hold is exactly 0.0"
                    )
            elif abs(row.current_weight - weight) > WEIGHT_TOLERANCE:
                problems.append(
                    f"changes gives {row.symbol} a current_weight of {row.current_weight!r}, "
                    f"but current_state holds {weight!r}: a row restates what the account holds"
                )

    if target is not None:
        wanted = target.target_weights_by_symbol
        for row in rows:
            weight = wanted.get(row.symbol)
            if weight is None:
                if abs(row.target_weight) > WEIGHT_TOLERANCE:
                    problems.append(
                        f"changes gives {row.symbol} a target_weight of {row.target_weight!r}, "
                        "but target_portfolio does not hold it: the target side of a name the "
                        "target does not want is exactly 0.0"
                    )
            elif abs(row.target_weight - weight) > WEIGHT_TOLERANCE:
                problems.append(
                    f"changes gives {row.symbol} a target_weight of {row.target_weight!r}, but "
                    f"target_portfolio holds {weight!r}: a row restates what the target asked for"
                )

    return problems


# ---------------------------------------------------------------------------
# PortfolioTransition -- the whole difference, on one decision date
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PortfolioTransition:
    """The complete weight difference between what is held and what is wanted.

    A transition is one decision read from both ends: it holds the
    :class:`~portfolio_sys.contracts.PortfolioState` the account was in, the
    :class:`~portfolio_sys.contracts.TargetPortfolio` the rule asked for, the date the
    decision belongs to, and one :class:`PortfolioWeightChange` per symbol of their union.
    Everything above the portfolio layer can therefore describe a rebalance without ever
    touching either portfolio again -- and still find the strategy, its version, the exact
    factor versions, the construction rule and the constraints it was built under, because
    those are read straight out of the target.

    Nothing here is a trade: no share count, no price, no execution date, no fill, no
    commission and no return.  :attr:`turnover` is a *weight* that changed hands once.

    Parameters
    ----------
    as_of_date:
        The decision date: a plain :class:`datetime.date` or a bare ISO-8601
        ``"YYYY-MM-DD"`` string (a ``datetime`` is refused even at midnight).  It must equal
        ``target_portfolio.as_of_date``, and ``current_state.as_of_date`` may not be later --
        a state from the future would carry knowledge the decision could not have had.
        Whether that date is a trading day is not this layer's question.
    current_state:
        The account before the decision, as a
        :class:`~portfolio_sys.contracts.PortfolioState`.
    target_portfolio:
        The weights the rule wants after the decision, as a
        :class:`~portfolio_sys.contracts.TargetPortfolio`.  It travels whole, so
        traceability is never re-derived.
    changes:
        A **tuple** of :class:`PortfolioWeightChange`, one row per symbol of
        ``set(current_state.symbols) | set(target_portfolio.target_symbols)``, ordered
        target-first and exits last (see the module docstring).  A list or a generator is
        refused, and so is a hand-built difference that does not match the two portfolios.

    Raises
    ------
    portfolio_sys.errors.PortfolioContractError
        If a date is not a plain date, the two sides are not the portfolio objects, the
        dates disagree, ``changes`` is not a tuple, an entry is not a weight change, a symbol
        repeats, a symbol of the union is missing or an outsider appears, the order is not
        the promised one, or a row disagrees with the state, with the target or with itself.
        Every problem is reported at once.

    Notes
    -----
    * **Immutable.**  Both the transition and every row are frozen: what a rebalance was
      cannot change while a report is being written about it.
    * **A view, not a decision.**  ``buy_weight``, ``sell_weight``, :attr:`turnover`, the
      entry list and the exit list are convenience readings of the rows; none of them is an
      instruction to trade anything.
    * **The target carries the provenance.**  :attr:`strategy_spec`,
      :attr:`factor_requirements`, :attr:`construction_spec`, :attr:`constraints` and
      :attr:`universe` return the target's own objects -- the same instances, not copies --
      so a transition cannot lose or nudge a version on the way out.
    * **It cannot prove the pair.**  A target does not record the state it was built from,
      so a transition states the weights of the two objects it was given and never claims
      that the target was the right answer to that state.

    Examples
    --------
    With ``AAPL 60%`` and ``MSFT 30%`` held beside ``10%`` cash, and ``MSFT 45%``,
    ``NVDA 45%`` and ``10%`` cash wanted on the next decision date, the transition
    reports::

        symbols             ('MSFT', 'NVDA', 'AAPL')
        buy_weight          0.6
        sell_weight         0.6
        cash_weight_change  0.0
        turnover            0.6
        entering_symbols    ('NVDA',)
        exiting_symbols     ('AAPL',)
        changed_symbols     ('MSFT', 'NVDA', 'AAPL')

    and every row agrees with the portfolio it restates.
    :func:`build_portfolio_transition` produces exactly this from two validated objects.
    """

    as_of_date: date
    current_state: PortfolioState
    target_portfolio: TargetPortfolio
    changes: tuple[PortfolioWeightChange, ...]

    def __post_init__(self) -> None:
        problems: list[str] = []

        resolved: date | None = None
        try:
            resolved = _session_date(self.as_of_date, "PortfolioTransition.as_of_date")
        except PortfolioContractError as exc:
            problems.append(str(exc))

        state = self.current_state
        if not isinstance(state, PortfolioState):
            problems.append(
                "current_state must be a PortfolioState, got "
                f"{type(state).__name__}: a transition starts from what the account actually "
                "holds, so the current side has to be a validated state and not a target, a "
                "mapping or a bare tuple"
            )
            state = None

        target = self.target_portfolio
        if not isinstance(target, TargetPortfolio):
            problems.append(
                "target_portfolio must be a TargetPortfolio, got "
                f"{type(target).__name__}: the target side of a transition is the validated "
                "object a constructor returned, because the rule, its version and the strategy "
                "result it came from travel with it"
            )
            target = None

        if resolved is not None and target is not None and target.as_of_date != resolved:
            problems.append(
                f"as_of_date is {resolved.isoformat()}, but target_portfolio.as_of_date is "
                f"{target.as_of_date.isoformat()}: a transition is dated the decision its target "
                "was built for, never another day"
            )

        if resolved is not None and state is not None and state.as_of_date > resolved:
            problems.append(
                f"current_state.as_of_date is {state.as_of_date.isoformat()}, which is later "
                f"than as_of_date {resolved.isoformat()}: a transition cannot start from a state "
                "of the future, because no decision could have known it"
            )

        changes = self.changes
        if not isinstance(changes, tuple):
            problems.append(
                "changes must be a tuple[PortfolioWeightChange, ...], got "
                f"{type(changes).__name__}: one row per symbol in a promised order -- a list "
                "promises nothing and a generator can be read only once"
            )
            changes = None

        rows: tuple[PortfolioWeightChange, ...] | None = None
        if changes is not None:
            rows = tuple(row for row in changes if isinstance(row, PortfolioWeightChange))
            problems.extend(
                f"changes entry {row!r} must be a PortfolioWeightChange, got "
                f"{type(row).__name__}: a bare triple has not been validated, so it could carry "
                "a symbol that is not a ticker or a weight that is not a share of the account"
                for row in changes
                if not isinstance(row, PortfolioWeightChange)
            )

        if rows is not None:
            problems.extend(
                f"changes holds {symbol} more than once: every symbol of the union appears "
                "exactly once, because two rows for one holding make its weight change ambiguous"
                for symbol in _duplicates(tuple(row.symbol for row in rows))
            )
            problems.extend(_coverage_problems(rows, state, target))
            problems.extend(_source_problems(rows, state, target))

        if problems:
            raise PortfolioContractError("invalid PortfolioTransition", problems)

        object.__setattr__(self, "as_of_date", resolved)
        object.__setattr__(self, "changes", rows)

    @property
    def symbols(self) -> tuple[str, ...]:
        """The symbols the transition covers, in row order: the full union, once each."""
        return tuple(change.symbol for change in self.changes)

    @property
    def strategy_spec(self):
        """``target_portfolio.strategy_spec``: which rule selected the candidates.

        The ``StrategySpec`` itself -- the rule's name *and* version -- not a copy.
        """
        return self.target_portfolio.strategy_spec

    @property
    def factor_requirements(self):
        """``target_portfolio.factor_requirements``: the exact factors *and* versions.

        The same tuple of ``FactorRequirement`` the strategy result carries, so a transition
        can never drop or nudge a factor version on the way to a report.
        """
        return self.target_portfolio.factor_requirements

    @property
    def construction_spec(self) -> PortfolioConstructionSpec:
        """``target_portfolio.construction_spec``: the rule and version that set the weights."""
        return self.target_portfolio.construction_spec

    @property
    def constraints(self) -> PortfolioConstraints:
        """``target_portfolio.constraints``: the cash share and single-name cap it honoured."""
        return self.target_portfolio.constraints

    @property
    def universe(self):
        """``target_portfolio.universe``: the pool the candidates were selected from.

        The ``UniverseSnapshot`` itself, never a re-built or re-dated one.
        """
        return self.target_portfolio.universe

    @property
    def buy_weight(self) -> float:
        """The weight added across all stocks: the sum of the positive changes.

        ``sum(max(weight_change, 0))`` over the rows.  It is the weight the target puts *into*
        stocks that it wants to hold more of, and it says nothing about how anything would be
        bought -- no amount, no order, no fill.  A change inside the tolerance still
        contributes its own value, so the sum is literal and never rounded behind a caller's
        back.
        """
        return sum((max(change.weight_change, 0.0) for change in self.changes), 0.0)

    @property
    def sell_weight(self) -> float:
        """The weight removed across all stocks: the sum of the negative changes, unsigned.

        ``sum(max(-weight_change, 0))`` over the rows -- the mirror of :attr:`buy_weight`, and
        equally not an order.
        """
        return sum((max(-change.weight_change, 0.0) for change in self.changes), 0.0)

    @property
    def cash_weight_change(self) -> float:
        """``target.cash_weight - current_state.cash_weight``: the cash share's move.

        Cash is a field of a portfolio, not a symbol, so it is reported here and never as a
        row.  The number may be anything in ``[-1, 1]``: positive when stock is sold into
        cash, negative when cash is spent on stock.
        """
        return self.target_portfolio.cash_weight - self.current_state.cash_weight

    @property
    def turnover(self) -> float:
        """The one-sided weight volume this rebalance needs, counted once.

        ``0.5 * (sum(|weight_change|) + |cash_weight_change|)``.  Because a portfolio is
        complete on both sides, the stock changes and the cash change cancel out and this is
        ``max(buy_weight, sell_weight)``: selling 100% of one name to buy 100% of another is a
        100% turnover, not 200%, and moving between all-cash and fully invested is the stock
        weight that moved.  It is a share of the account, never an amount of money, and it
        carries no cost, no spread and no tax.
        """
        moved = sum((abs(change.weight_change) for change in self.changes), 0.0)
        return 0.5 * (moved + abs(self.cash_weight_change))

    @property
    def changed_symbols(self) -> tuple[str, ...]:
        """The symbols whose weight really moves, in row order.

        A row inside :data:`~portfolio_sys.contracts.WEIGHT_TOLERANCE` is unchanged and is
        left out, so a floating-point hair never shows up as a rebalance.
        """
        return tuple(change.symbol for change in self.changes if not change.is_unchanged)

    @property
    def entering_symbols(self) -> tuple[str, ...]:
        """The brand-new holdings (:attr:`PortfolioWeightChange.is_entry`), in row order."""
        return tuple(change.symbol for change in self.changes if change.is_entry)

    @property
    def exiting_symbols(self) -> tuple[str, ...]:
        """The closed holdings (:attr:`PortfolioWeightChange.is_exit`), in row order."""
        return tuple(change.symbol for change in self.changes if change.is_exit)


# ---------------------------------------------------------------------------
# build_portfolio_transition -- two validated portfolios, one difference
# ---------------------------------------------------------------------------
def build_portfolio_transition(
    *,
    current_state: PortfolioState,
    target_portfolio: TargetPortfolio,
) -> PortfolioTransition:
    """Build the complete weight difference between a state and a target.

    The whole job: read the two portfolios, walk the union of their symbols in the promised
    order, give the missing side of each row a weight of ``0.0``, and let the resulting
    :class:`PortfolioTransition` validate itself.  Nothing is read, written, re-weighted,
    re-priced or executed on the way.

    Parameters
    ----------
    current_state:
        The account before the decision: a
        :class:`~portfolio_sys.contracts.PortfolioState`.  It is not modified, and it is
        stored in the result as the very same object.
    target_portfolio:
        The weights the rule wants after the decision: a
        :class:`~portfolio_sys.contracts.TargetPortfolio`, likewise stored as the same
        object.  Its ``as_of_date`` becomes the transition's date, and the strategy, factor
        requirements, construction spec, constraints and universe are read from it -- never
        copied, re-derived or guessed.

    Returns
    -------
    PortfolioTransition
        A **new**, already validated transition: one row per symbol of
        ``set(current_state.symbols) | set(target_portfolio.target_symbols)``, the target's
        symbols first in the target's order and the exits after them in the state's order.
        Two calls on the same inputs return two equal objects with identical numbers, which
        is what makes a caller free to keep either.

    Raises
    ------
    portfolio_sys.errors.PortfolioContractError
        If either argument is not the portfolio object it must be, or if the transition the
        pair implies is not legal -- a state dated after the target, for example.  A refused
        build leaves both inputs exactly as they were.

    Notes
    -----
    * **It does not judge the decision.**  A :class:`~portfolio_sys.contracts.TargetPortfolio`
      does not record the state it was built from, so this function cannot know whether the
      target was built *for* this state; it reports the difference between the two objects it
      was handed, and nothing more.  A future turnover-aware constructor reads
      ``PortfolioContext.current_state`` itself, which is where that comparison belongs.
    * **No zero-weight holdings are invented.**  A row with ``current_weight == 0`` (an
      entry) or ``target_weight == 0`` (an exit) is a statement about the difference, not a
      :class:`~portfolio_sys.contracts.PortfolioPosition` -- neither portfolio gains or loses
      a holding here.
    * **Pure and offline.**  No file, no price, no calendar, no factor panel, no strategy
      run, no constructor, no registry, no order and no share count: two in-memory objects
      in, one in-memory object out.

    Examples
    --------
    A target that keeps ``MSFT``, adds ``NVDA`` and drops ``AAPL`` gives three rows::

        current  2025-01-02:  AAPL 60%,  MSFT 30%,  cash 10%
        target   2025-01-09:  MSFT 45%,  NVDA 45%,  cash 10%

        transition.as_of_date          2025-01-09
        transition.symbols             ('MSFT', 'NVDA', 'AAPL')
        transition.buy_weight          0.6
        transition.sell_weight         0.6
        transition.cash_weight_change  0.0
        transition.turnover            0.6
        transition.entering_symbols    ('NVDA',)
        transition.exiting_symbols     ('AAPL',)
    """
    problems: list[str] = []

    if not isinstance(current_state, PortfolioState):
        problems.append(
            "current_state must be a PortfolioState, got "
            f"{type(current_state).__name__}: the current side of a difference is what the "
            "account holds, so it has to be a validated state rather than a target, a mapping "
            "or a tuple of pairs"
        )

    if not isinstance(target_portfolio, TargetPortfolio):
        problems.append(
            "target_portfolio must be a TargetPortfolio, got "
            f"{type(target_portfolio).__name__}: the other side of a difference is the target a "
            "constructor returned, because the difference reports the weights it asked for and "
            "the rules and versions it was built under"
        )

    if problems:
        raise PortfolioContractError("cannot build a PortfolioTransition", problems)

    held = current_state.weights_by_symbol
    wanted = target_portfolio.target_weights_by_symbol

    ordered = list(target_portfolio.target_symbols)
    ordered.extend(symbol for symbol in current_state.symbols if symbol not in wanted)

    rows: list[PortfolioWeightChange] = []
    for symbol in ordered:
        current = held.get(symbol, 0.0)
        target = wanted.get(symbol, 0.0)
        rows.append(
            PortfolioWeightChange(
                symbol=symbol,
                current_weight=current,
                target_weight=target,
                weight_change=target - current,
            )
        )

    return PortfolioTransition(
        as_of_date=target_portfolio.as_of_date,
        current_state=current_state,
        target_portfolio=target_portfolio,
        changes=tuple(rows),
    )


__all__ = [
    "PortfolioTransition",
    "PortfolioWeightChange",
    "build_portfolio_transition",
]
