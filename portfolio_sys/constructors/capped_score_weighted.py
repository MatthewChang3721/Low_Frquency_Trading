"""Score-proportional weights under a single-name cap -- the capped water-filling rule.

Where this module sits
----------------------
Every layer answers one question and hands the answer on::

    strategy_sys   which symbols this decision selects        -- StrategyResult
        |
    portfolio_sys  how much weight each of them should carry  -- TargetPortfolio
        |

:class:`~portfolio_sys.contracts.PortfolioConstructor` defines the interface and
this module implements the first rule of it.  A constructor reads a validated
:class:`~portfolio_sys.contracts.PortfolioContext` -- one decision date, one
strategy result, the current state and the constraints -- and returns a validated
:class:`~portfolio_sys.contracts.TargetPortfolio`.  It sees no price, no share
count, no account value, no calendar and no file: a weight is a share of the
account, and turning shares of an account into trades belongs to the execution
layer, which does not exist yet.

The rule, exactly
-----------------
1. ``selected = context.selected_symbols``: the strategy's own ``is_selected``
   flags, in the universe's row order.  Nothing is re-scored, re-ranked, filtered
   or dropped here.
2. The budget is ``investable = 1 - constraints.cash_weight`` and the target's
   cash is *exactly* ``constraints.cash_weight``: with 10% cash the selected
   stocks carry 90% between them.  Nothing that is not needed is parked in cash.
3. Every selected symbol must have a finite score strictly greater than ``0``.
   This is a *positive* proportional rule, so a score of ``0``, a negative score,
   ``NaN`` or ``±inf`` is refused instead of absorbed -- see below.
4. The selection must be able to absorb the budget:
   ``len(selected) * cap >= investable`` (``cap`` being
   ``constraints.max_single_weight``), within :data:`~portfolio_sys.WEIGHT_TOLERANCE`.
   If it cannot, the context is refused -- see below.
5. Otherwise the score-proportional weights are computed and then *water-filled*:
   a symbol whose proportional weight lands above ``cap`` is pinned at ``cap``,
   the budget shrinks by each pinned weight, and the unpinned symbols re-split
   what remains -- still in proportion to their **original** scores -- until no
   weight is above the cap.
6. The positions come back in ``context.selected_symbols`` order, at the exact
   date, rule identity, strategy result and constraints the context carried.

Why water-filling and not truncation
------------------------------------
The obvious shortcut -- cap the offending names and leave the difference in cash
-- is a different portfolio: it silently changes the cash share the constraints
asked for, and it says "we cannot invest as much as planned" when the truth is
only "one name was too big".  The second shortcut -- cap the offenders and hand
what is left to the rest *equally* -- throws the scores away the moment the cap
bites.  Neither is what a score-proportional portfolio means, so this rule pins the
offender and repeats the proportional split among the names that are still free.
The five-name case from the documentation::

    scores   A 30, B 20, C 20, D 15, E 15     cap 25%, cash 0%
    round 1  A 30.000%  B 20.000%  C 20.000%  D 15.000%  E 15.000%
             A is above the cap -> pinned at 25%, its share leaves the budget
    round 2  budget 75% over the remaining scores 20:20:15:15
             B 21.429%  C 21.429%  D 16.071%  E 16.071%
    result   A 25.000%  B 21.429%  C 21.429%  D 16.071%  E 16.071%  cash 0%
             total 100%, nothing above the cap, B:C:D:E still 20:20:15:15

Numbers, not just ratios
------------------------
A score is any finite float, so a selection can carry the largest ones a float
holds.  No round of the rule ever adds raw scores together: each round is scaled by
the largest score still free, so two scores of ``1e308`` divide by their true ratio
instead of overflowing the total to ``inf`` -- which would make every share
``0.0`` -- or raising ``OverflowError``.  Scaling changes no ratio and therefore no
weight; for ordinary scores the only difference is a last-bit rounding difference,
four orders of magnitude inside :data:`~portfolio_sys.WEIGHT_TOLERANCE`.

Why it refuses instead of repairing
-----------------------------------
The rule has exactly two refusals, and both are
:class:`~portfolio_sys.errors.PortfolioConstructionError` -- never a contract
error, because the objects themselves are legal:

* **an unusable score.**  A score that is not a positive finite number -- or one
  so small beside the others that its share rounds down to ``0.0`` -- cannot become
  a holding.  Adding a constant, taking an absolute value, clipping to zero or
  falling back to equal weight would invent a decision the strategy never made, and
  dropping the symbol would rewrite the selection.  The error names the symbol and
  the score it carries.
* **infeasible constraints.**  Three names at a 25% cap can carry at most 75% of
  the account, so 90% invested is impossible.  Raising the cash target, widening
  the cap or deleting a candidate would each rewrite either the constraints or the
  selection; the error reports the arithmetic and stops.

An empty selection is *not* a refusal: with nothing to weight, the only legal
answer is an all-cash target (``positions == ()``, ``cash_weight == 1.0``), which
is the one case where ``constraints.cash_weight`` is deliberately not honoured --
there is no stock left to carry the investable share, and inventing a ``"CASH"``
holding is what this layer refuses to do in the first place.

What this module must never do
------------------------------
Only the standard library and :mod:`portfolio_sys.contracts` /
:mod:`portfolio_sys.errors` are imported: no :mod:`pandas`, no :mod:`numpy`, no
PyArrow, no DuckDB, no storage, no factor runner, no strategy, no market data, no
calendar, no network, no file.  It reads ``context.strategy_result`` (through the
defensive ``frame`` copy the result already hands over) and never writes to the
context, the result, the state, the constraints or their frames; it builds no
order, no fill, no price, no turnover and no return; and it re-implements no check
that :class:`~portfolio_sys.contracts.TargetPortfolio` already performs -- the
target validates itself.

``context.current_state`` is read by nobody here.  It travels with the context
because every constructor receives the same input, and it will matter to a rule
with a turnover budget; weighting a fresh selection from scratch does not need it.

Versioning
----------
``spec.version`` labels *this behaviour*.  Changing the weighting (equal weight,
inverse volatility), the cap semantics (a cap on the initial weight only, a cap
plus a drift buffer) or the treatment of an empty selection changes what a target
means, so it requires a new version.  The name ``capped_score_weighted`` already
says which family of rule this is.

Examples
--------
The allocation on its own -- it is the whole rule, and it is pure:

>>> scores = {"A": 30.0, "B": 20.0, "C": 20.0, "D": 15.0, "E": 15.0}
>>> weights = _capped_weights(scores, investable_weight=1.0, cap=0.25)
>>> {symbol: round(weight, 6) for symbol, weight in weights.items()}
{'A': 0.25, 'B': 0.214286, 'C': 0.214286, 'D': 0.160714, 'E': 0.160714}
>>> round(sum(weights.values()), 12)
1.0

An uncapped budget simply follows the scores:

>>> weights = _capped_weights({"AAPL": 6.0, "MSFT": 3.0, "NVDA": 1.0}, 0.9, 1.0)
>>> [round(weights[symbol], 6) for symbol in weights]
[0.54, 0.27, 0.09]

The scores may sit anywhere in the float range; they are never added up raw:

>>> huge = _capped_weights({"AAPL": 1e308, "MSFT": 1e308}, 1.0, 1.0)
>>> huge["AAPL"] == huge["MSFT"] == 0.5
True

...and the cap still bites on them, with the rest re-split by original ratio:

>>> _capped_weights({"A": 1e308, "B": 2e307, "C": 2e307}, 1.0, 0.5)
{'A': 0.5, 'B': 0.25, 'C': 0.25}
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from portfolio_sys.contracts import (
    WEIGHT_TOLERANCE,
    PortfolioConstructionSpec,
    PortfolioConstructor,
    PortfolioContext,
    PortfolioPosition,
    TargetPortfolio,
)
from portfolio_sys.errors import PortfolioConstructionError, PortfolioContractError

#: Default version of this rule (see "Versioning" above).
DEFAULT_VERSION: str = "1.0.0"

#: The rule's name; with :data:`DEFAULT_VERSION` it is the rule's whole identity.
_NAME: str = "capped_score_weighted"

#: The strategy result's own column names.  They are read positionally out of the
#: frame the result hands over, so no second vocabulary is declared here -- and no
#: pandas import is needed to read a column of a frame that already exists.
_SYMBOL_COLUMN: str = "symbol"
_SCORE_COLUMN: str = "score"
_SELECTED_COLUMN: str = "is_selected"

#: The immutable identity every target built by this rule carries.
_SPEC: PortfolioConstructionSpec = PortfolioConstructionSpec(name=_NAME, version=DEFAULT_VERSION)


# ---------------------------------------------------------------------------
# Reading the selection, and refusing an unusable one
# ---------------------------------------------------------------------------
def _selected_scores(context: PortfolioContext) -> dict[str, float]:
    """The score of every selected symbol, in ``context.selected_symbols`` order.

    The result validates itself, so its frame holds one row per universe symbol
    with a boolean ``is_selected`` flag and a float ``score``; reading the three
    columns positionally keeps the symbols in the frame's own (universe) order and
    makes the lookup below a plain dictionary read.  ``frame`` hands over a
    defensive copy, so nothing here can change the result it came from, and no
    column is added, renamed or written.

    Selecting a symbol without a finite score is refused *by the strategy
    contract*, so the values returned here are finite for every symbol that
    ``is_selected`` -- unless a result was corrupted after it was built, which is
    exactly why :func:`_score_problems` checks the values again instead of trusting
    the flag.
    """
    frame = context.strategy_result.frame
    names = frame[_SYMBOL_COLUMN].tolist()
    flags = frame[_SELECTED_COLUMN].astype(bool).tolist()
    values = frame[_SCORE_COLUMN].astype("float64").tolist()

    scored = {
        name: float(value)
        for name, selected, value in zip(names, flags, values, strict=True)
        if selected
    }
    return {symbol: scored[symbol] for symbol in context.selected_symbols}


def _score_problems(scores: Mapping[str, float]) -> list[str]:
    """Why these scores cannot become long-only weights, or nothing at all.

    ``0`` and every negative number are *legal* scores in a strategy result -- a
    score is any finite number -- so this rule has to check the one property it
    needs and report each offender instead of guessing what the strategy meant.
    Symbols that were not selected never reach this function: their scores play no
    part in the allocation, so a missing or nonsensical score there is not a
    problem.
    """
    problems: list[str] = []
    for symbol, score in scores.items():
        if not math.isfinite(score):
            problems.append(
                f"{symbol} carries the non-finite score {score!r}: NaN and ±inf cannot be "
                "ordered, summed or divided, so no share of the account follows from them"
            )
        elif score <= 0.0:
            problems.append(
                f"{symbol} carries the score {score!r}: this rule splits the investable share "
                "in proportion to positive scores, and a zero or negative score has no "
                "positive share to give a long-only holding"
            )
    if problems:
        problems.append(
            "the scores are used exactly as the strategy wrote them: no constant is added, "
            "nothing is absolute-valued, clipped to zero or shifted, and the rule does not "
            "fall back to equal weight -- each of those would report a decision the strategy "
            "never made"
        )
    return problems


def _feasibility_problems(count: int, cash: float, investable: float, cap: float) -> list[str]:
    """Why ``count`` names at ``cap`` cannot carry ``investable``, or nothing.

    A single-name cap is a ceiling on every holding, so ``count`` names can hold at
    most ``count * cap`` of the account.  If the investable share is larger, no
    allocation can honour both the cap and the cash target, and the arithmetic is
    reported instead of relaxed.  The comparison allows
    :data:`~portfolio_sys.WEIGHT_TOLERANCE`, because ``3 * 0.3`` and ``1 - 0.1``
    disagree in their last bit in binary floating point.
    """
    capacity = count * cap
    if capacity + WEIGHT_TOLERANCE >= investable:
        return []
    return [
        f"{count} selected symbol(s) x max_single_weight {cap!r} = {capacity!r}, but "
        f"{investable!r} of the account must be invested (cash_weight {cash!r}): the "
        f"selection can carry at most {capacity:.2%} of the account, "
        f"which is {investable - capacity:.2%} short",
        "the cash target is not raised, the single-name cap is not widened and no selected "
        "symbol is dropped to make the numbers fit: each of those would rewrite the "
        "constraints or the selection the caller already decided on",
    ]


def _starved_problems(scores: Mapping[str, float], weights: Mapping[str, float]) -> list[str]:
    """Why some share rounded down to nothing, or nothing at all.

    A score can be positive and finite and still be a rounding error against the
    total (``5e-324`` beside ``1e300``): dividing it out gives a weight of exactly
    ``0.0``, and the round's scaling does not change that -- ``5e-324 / 1e308`` is
    ``0.0`` as well.  A zero weight is not a holding in this layer -- every position
    is strictly positive -- and dropping the symbol would quietly change the
    selection, so the construction is refused with the rule's own error instead of
    letting a position be built that the contract would then reject.
    """
    starved = [symbol for symbol, weight in weights.items() if weight <= 0.0]
    if not starved:
        return []
    total = sum(scores.values())
    return [
        f"{symbol} carries the score {scores[symbol]!r} against a total of {total!r}, so its "
        f"share rounds down to {weights[symbol]!r} of the account: a zero weight is not a "
        "holding, and the symbol is not dropped to make room for the others either"
        for symbol in starved
    ]


# ---------------------------------------------------------------------------
# The allocation itself
# ---------------------------------------------------------------------------
def _capped_weights(
    scores: Mapping[str, float], investable_weight: float, cap: float
) -> dict[str, float]:
    """Spend ``investable_weight`` in proportion to ``scores``, under ``cap``.

    This is the whole rule, isolated so it can be read (and doctested) on its own.
    Water-filling, in three lines of arithmetic per round: every *free* symbol gets
    ``budget * score / sum(free scores)``; the symbols whose share lands strictly
    above ``cap`` are pinned at ``cap`` (never "whatever they asked for"), each pin
    removes exactly ``cap`` from the budget, and the rest -- still in proportion to
    their original scores, never to their shares, and never equally -- re-split what
    remains.

    The sum in that formula is never taken over the raw scores.  A score is any
    finite float, so a handful of ordinary-looking large ones (``1e308`` twice) add
    up past the float range: the total becomes ``inf``, every share
    ``budget * score / inf`` collapses to ``0.0``, and the round would hand out
    nothing -- and :func:`math.fsum` refuses such a sum outright with
    ``OverflowError`` rather than return it.  Scaling by the round's largest score
    first is the fix: every scaled score is at most ``1`` and their total at most
    ``len(free)``, nowhere near overflow, while the *ratio* between any two scores --
    which is all the allocation depends on -- is untouched.  Ordinary scores
    therefore come out the same as the unscaled formula, to within a last-bit
    rounding difference.

    Two facts make it terminate and stay inside the cap.  Each round pins at least
    one symbol, so there are at most ``len(scores)`` rounds; and a round can never
    pin *every* remaining symbol, because their shares add up to the budget and
    every one of them is then above ``cap`` -- which would mean the budget exceeds
    its own total.  The budget therefore stays positive, and the last unpinned
    round lands below the cap by construction.

    No post-hoc correction is applied to the total.  The accumulated error of
    ``len(scores)`` multiplications and divisions is around ``1e-16``, four orders
    of magnitude inside :data:`~portfolio_sys.WEIGHT_TOLERANCE`, whereas re-scaling
    the answer at the end could push a pinned name past its own cap -- a wrong
    portfolio is worse than a last-bit difference.

    Parameters
    ----------
    scores:
        The original score of every symbol still to weight; all of them positive
        and finite (the caller has already refused anything else).
    investable_weight:
        The share of the account to hand out: ``1 - cash_weight``.
    cap:
        The largest share any single symbol may receive.
    """
    pinned: dict[str, float] = {}
    free = dict(scores)
    budget = investable_weight

    while free:
        # Scale by this round's largest score before summing: two scores of ``1e308``
        # overflow the float range when added, and a total of ``inf`` turns every
        # share below into ``0.0`` (``math.fsum`` raises ``OverflowError`` on such a
        # sum instead of returning it).  Every scaled score is in ``(0, 1]``, so the
        # total is at most ``len(free)`` and the ratios the allocation depends on are
        # unchanged -- the same weights, one division later.
        scale = max(free.values())
        scaled = {symbol: score / scale for symbol, score in free.items()}
        total = math.fsum(scaled.values())
        allocation = {symbol: budget * part / total for symbol, part in scaled.items()}
        over = [symbol for symbol, weight in allocation.items() if weight > cap]

        if not over:
            pinned.update(allocation)
            break

        for symbol in over:
            pinned[symbol] = cap
            budget -= cap
            del free[symbol]

    return pinned


# ---------------------------------------------------------------------------
# CappedScoreWeightedConstructor -- the rule
# ---------------------------------------------------------------------------
class CappedScoreWeightedConstructor(PortfolioConstructor):
    """Weight the selection by score, under the single-name cap of the constraints.

    ``investable = 1 - constraints.cash_weight`` is split in proportion to the
    selected symbols' ``score``; every symbol whose share would exceed
    ``constraints.max_single_weight`` is pinned at that cap and what is left is
    re-split, still by original score, among the symbols that are not capped --
    until no weight is above the cap.  The five-name example in the module
    docstring is the canonical case.

    The rule has no parameters of its own.  Cash and the cap come from
    ``context.constraints``, never from the constructor, so a caller cannot
    accidentally run "the same rule" with a private cap: two constructions differ
    only if one of them was given different constraints, and the target then
    carries those constraints with it.  An instance is stateless and reusable.

    Raises
    ------
    portfolio_sys.errors.PortfolioConstructionError
        If a selected symbol's score is not a positive finite number (or is so small
        beside the others that its share rounds down to zero), or if
        ``len(selected) * max_single_weight`` cannot cover the investable share.
        Both are legal inputs this rule declines, and both keep their own type
        rather than pretending to be a malformed object.
    portfolio_sys.errors.PortfolioContractError
        If ``context`` is not a :class:`~portfolio_sys.contracts.PortfolioContext`.

    Examples
    --------
    >>> constructor = CappedScoreWeightedConstructor()
    >>> f"{constructor.spec.name}/{constructor.spec.version}"
    'capped_score_weighted/1.0.0'
    >>> constructor.spec is CappedScoreWeightedConstructor().spec
    True
    """

    @property
    def spec(self) -> PortfolioConstructionSpec:
        """This rule's immutable identity: ``capped_score_weighted``/``1.0.0``."""
        return _SPEC

    def construct(self, context: PortfolioContext) -> TargetPortfolio:
        """Return the capped score-proportional target for ``context``.

        Parameters
        ----------
        context:
            The decision to weight: its date, its strategy result, the current
            state (unused here) and the constraints (``cash_weight`` and
            ``max_single_weight`` are read, nothing else).

        Returns
        -------
        TargetPortfolio
            One position per selected symbol, in ``context.selected_symbols``
            order, each strictly positive and no larger than
            ``constraints.max_single_weight``; ``sum(weight)`` is
            ``1 - constraints.cash_weight`` within
            :data:`~portfolio_sys.WEIGHT_TOLERANCE`, and the target carries the
            context's own ``strategy_result`` and ``constraints`` objects so the
            provenance chain stays unbroken.  An empty selection yields
            ``positions == ()`` and ``cash_weight == 1.0``.

        Notes
        -----
        The context is read, never written: the strategy result's frame is a copy,
        no column is added, and the state, constraints and context come back
        untouched, so the same context always produces the same target.
        """
        if not isinstance(context, PortfolioContext):
            raise PortfolioContractError(
                f"{_NAME} needs a PortfolioContext, got {type(context).__name__}: a constructor "
                "reads the validated input it is handed -- one date, one strategy result, one "
                "state, one set of constraints -- never a bare frame or a list of tickers"
            )

        selected = context.selected_symbols
        constraints = context.constraints
        cash_weight = constraints.cash_weight

        if not selected:
            # Nothing to weight: the only legal portfolio is entirely in cash, and
            # that is the one exception to "cash equals constraints.cash_weight" --
            # no holding is left to carry the investable share.
            return TargetPortfolio(
                as_of_date=context.as_of_date,
                construction_spec=_SPEC,
                source_result=context.strategy_result,
                constraints=constraints,
                positions=(),
                cash_weight=1.0,
            )

        scores = _selected_scores(context)
        problems = _score_problems(scores)
        if problems:
            raise PortfolioConstructionError(
                f"{_NAME}/{DEFAULT_VERSION} cannot weight this selection", problems
            )

        investable_weight = 1.0 - cash_weight
        cap = constraints.max_single_weight
        capacity = len(selected) * cap
        problems = _feasibility_problems(
            count=len(selected), cash=cash_weight, investable=investable_weight, cap=cap
        )
        if problems:
            raise PortfolioConstructionError(
                f"{_NAME}/{DEFAULT_VERSION} cannot honour PortfolioConstraints with "
                f"{len(selected)} selected symbol(s): {len(selected)} x max_single_weight "
                f"{cap!r} = {capacity!r} of the account, but {investable_weight!r} must be "
                f"invested (cash_weight {cash_weight!r})",
                problems,
            )

        weights = _capped_weights(scores, investable_weight=investable_weight, cap=cap)
        problems = _starved_problems(scores, weights)
        if problems:
            raise PortfolioConstructionError(
                f"{_NAME}/{DEFAULT_VERSION} cannot weight this selection", problems
            )

        return TargetPortfolio(
            as_of_date=context.as_of_date,
            construction_spec=_SPEC,
            source_result=context.strategy_result,
            constraints=constraints,
            positions=tuple(PortfolioPosition(symbol, weights[symbol]) for symbol in selected),
            cash_weight=cash_weight,
        )


__all__ = ["DEFAULT_VERSION", "CappedScoreWeightedConstructor"]
