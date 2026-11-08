"""Run one portfolio construction for one date: selection -> target -> weight difference.

What this script is
-------------------
The minimal, real entry point of the portfolio layer, and the third command of the
project.  The strategy command answers *which symbols*; this one answers *how much
weight each of them carries*, and what that would change.  It owns no scoring rule,
no weighting rule and no storage rule: it joins the modules that already exist, in
exactly this order::

     1  run_strategy(...)                     the strategy layer's own entry function
     2  StrategyResult                        one validated cross-section for the date
     3  PortfolioConstraints                  cash share + single-name cap for this run
     4  PortfolioState                        what the account holds now (all cash, or given)
     5  PortfolioContext(date, result,
                         state, constraints)  the four inputs of one construction
     6  PortfolioConstructorRegistry()        a table for this invocation only
     7  registry.register(
             CappedScoreWeightedConstructor)  one rule, registered under its whole spec
     8  registry.create(spec)                 the instance this run really uses
     9  constructor.construct(context)        the target weights
    10  build_portfolio_transition(...)       current weights -> target weights
    11  print the target and the difference

Usage (project root, virtual environment active)::

    # a first purchase: the account starts entirely in cash
    python scripts/run_portfolio.py --factor momentum_12_1 --factor-version 1.0.0 \
        --top-n 10 --as-of 2024-12-31 --target-cash-weight 0.10 \
        --max-single-weight 0.25 --initial-cash

    # a rebalance: describe the account as it stands today
    python scripts/run_portfolio.py --factor momentum_12_1 --factor-version 1.0.0 \
        --top-n 10 --as-of 2024-12-31 --target-cash-weight 0.10 \
        --max-single-weight 0.25 --current-as-of 2024-12-24 \
        --current-position AAPL=0.60 --current-position MSFT=0.30 \
        --current-cash-weight 0.10

Reading a selection, making weights
-----------------------------------
The command *reads* the factor panel that the factor system already stored, under
``<data-root>/factors/panels``, and it reads it exactly the way the strategy command
does: through :func:`scripts.run_strategy.run_strategy`, with the same flags, the same
exact-version lookup and the same refusals.  No factor is computed here, no panel is
written here, no strategy rule is rebuilt here, and no date is moved to a
neighbouring session: a panel that is not on disk is an error, not a reason to
download.

What the registry is doing here
-------------------------------
``PortfolioConstructorRegistry`` has no default instance and no auto-registration,
and this script does not add one.  It builds a table for this one command, registers
the one rule that exists under its complete
:class:`~portfolio_sys.contracts.PortfolioConstructionSpec`, and constructs with
whatever ``create`` hands back.  The instance that weights the account is therefore a
registry-built instance -- never a literal ``CappedScoreWeightedConstructor()``
written inline -- and the spec it was created under is the spec the target carries.
There is deliberately no ``--constructor`` flag: with one real rule it would invent a
generality the layer does not have, and a second rule is a registry entry, not a new
flag.

The state is an argument, never a file
--------------------------------------
The current portfolio is an *input to this command*: either ``--initial-cash`` (an
account that holds nothing but cash, the way a first purchase starts) or an explicit
``--current-as-of`` + ``--current-position SYMBOL=WEIGHT`` + ``--current-cash-weight``
description.  Both modes are validated by
:class:`~portfolio_sys.contracts.PortfolioState` itself: the weights must add up to
``1.0`` within :data:`~portfolio_sys.WEIGHT_TOLERANCE`, every weight must be a
positive finite share, duplicates are refused, and the state may not be dated later
than the decision (that would be look-ahead).  Nothing is read from a state file and
nothing is written to one: this command has no ``--load-state``, no ``--save-state``
and no notion of an account that persists between runs.

The output
----------
One stable ``key: value`` block, then the target and the difference::

    as_of_date: 2024-12-31
    current_state_as_of_date: 2024-12-31
    strategy_name: top_3_by_momentum_12_1
    strategy_version: 1.0.0
    required_factor: momentum_12_1/1.0.0
    construction_name: capped_score_weighted
    construction_version: 1.0.0
    target_cash_weight: 0.100000
    max_single_weight: 0.500000
    selected_count: 3
    target_position_count: 3
    buy_weight: 0.900000
    sell_weight: 0.000000
    cash_weight_change: -0.900000
    turnover: 0.900000
    target:
    symbol  target_weight
    META    0.257143
    MSFT    0.293878
    NVDA    0.348980
    transition:
    symbol  current_weight  target_weight  weight_change  status
    META    0.000000        0.257143       0.257143       entry
    MSFT    0.000000        0.293878       0.293878       entry
    NVDA    0.000000        0.348980       0.348980       entry
    cash:
    current_weight: 1.000000
    target_weight: 0.100000
    weight_change: -0.900000

Every weight is printed with six decimals; ``target:`` lists the target's positions
in the constructor's own order, ``transition:`` lists the complete weight difference
-- one row per symbol of the union of the two portfolios, target names first and the
names only the current state holds after them -- as ``entry`` (nothing held,
something wanted), ``exit`` (held, nothing wanted), ``increase``, ``decrease`` or
``unchanged``.  ``turnover`` is a *share of the account moved once*:
``0.5 * (sum(|weight_change|) + |cash_weight_change|)``, which for two complete
portfolios equals ``max(buy_weight, sell_weight)`` -- selling one name to buy another
is one turnover, not two.

Those status words describe what the **target weights** ask for.  They are not order
sides, not fills and not trades: nothing here has a share count, a price, a
commission, a slippage estimate or a return.

Exit codes
----------
``0``
    The construction ran and its target and difference were printed, including an
    empty selection, which is an all-cash target rather than a failure.
``1``
    A domain or command error: a missing or damaged panel
    (:class:`~factor_sys.errors.FactorStoreError`), an unusable strategy or factor
    request (:class:`~strategy_sys.errors.StrategyError` /
    :class:`~factor_sys.errors.FactorError`), a malformed portfolio object
    (:class:`~portfolio_sys.errors.PortfolioContractError`), a legal selection this
    rule cannot honour (:class:`~portfolio_sys.errors.PortfolioConstructionError`), a
    registry failure (:class:`~portfolio_sys.errors.PortfolioRegistryError`) or an
    argument combination this command refuses (:class:`PortfolioCommandError`).  The
    message goes to stderr as one ``error: ...`` line -- in the project's bullet
    rendering when several problems were collected, never a traceback -- and nothing
    is written anywhere.
``2``
    argparse's own code for a missing, malformed or unknown flag.

This is not an execution layer
------------------------------
There are no orders, no share counts, no prices, no fills, no commissions, no
slippage, no costs, no persisted account and no returns here, and no second date:
one command, one decision date, one target and one difference between two portfolios.
Turning that difference into trades, or measuring it over time, is the work of the
execution and backtest layers that come later.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

# Running this file directly puts ``scripts/`` (not the project root) on
# ``sys.path``; add the root so ``python scripts/run_portfolio.py`` works from a
# checkout, with or without the package installed in the environment.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from data_sys.config import PipelineConfig  # noqa: E402
from factor_sys import FactorError  # noqa: E402
from portfolio_sys import (  # noqa: E402
    PortfolioConstraints,
    PortfolioConstructionSpec,
    PortfolioConstructorRegistry,
    PortfolioContext,
    PortfolioContractError,
    PortfolioError,
    PortfolioPosition,
    PortfolioState,
    PortfolioTransition,
    PortfolioWeightChange,
    TargetPortfolio,
    build_portfolio_transition,
)
from portfolio_sys.constructors import CappedScoreWeightedConstructor  # noqa: E402
from scripts.run_strategy import (  # noqa: E402
    DEFAULT_STRATEGY_VERSION,
    StrategyRunOutcome,
    run_strategy,
)
from strategy_sys import StrategyError, StrategyResult  # noqa: E402

#: Decimals every weight of the report is printed with.
WEIGHT_DECIMALS = 6

#: Columns of the ``target:`` table.
TARGET_COLUMNS: tuple[str, ...] = ("symbol", "target_weight")

#: Columns of the ``transition:`` table.
TRANSITION_COLUMNS: tuple[str, ...] = (
    "symbol",
    "current_weight",
    "target_weight",
    "weight_change",
    "status",
)

#: Spaces that separate two columns of a printed table.
COLUMN_GAP = 2

#: Printed under a table header when the table has no rows.
EMPTY_TABLE_TEXT = "none"

#: The cash share an account starts with under ``--initial-cash``.
INITIAL_CASH_WEIGHT = 1.0

#: The status words of the ``transition:`` table -- a direction of the target weight,
#: never an order side, a fill or a trade.
ENTRY_STATUS = "entry"
EXIT_STATUS = "exit"
INCREASE_STATUS = "increase"
DECREASE_STATUS = "decrease"
UNCHANGED_STATUS = "unchanged"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class PortfolioCommandError(PortfolioError):
    """A command request this entry point refuses before any module could answer it.

    Only *argument semantics* land here: ``--initial-cash`` combined with a
    ``--current-*`` flag, a missing ``--current-as-of`` or ``--current-cash-weight``,
    or a ``--current-position`` that is not ``SYMBOL=WEIGHT``.  Everything else stays
    with the layer that owns the rule -- a weight outside ``[0, 1]`` is a
    :class:`~portfolio_sys.errors.PortfolioContractError` raised by
    :class:`~portfolio_sys.contracts.PortfolioConstraints`, a state whose weights do
    not add up is one raised by :class:`~portfolio_sys.contracts.PortfolioState`, and
    a selection this rule cannot weight is a
    :class:`~portfolio_sys.errors.PortfolioConstructionError`.  This error therefore
    re-implements no business rule; it only says which combination of flags cannot be
    meant.

    It derives from :class:`~portfolio_sys.errors.PortfolioError` so that a caller of
    :func:`run_portfolio` -- and this command, with its one ``except`` -- catches
    every refusal of the portfolio layer, whichever part of it spoke.

    Examples
    --------
    >>> isinstance(PortfolioCommandError("--initial-cash needs no state"), PortfolioError)
    True
    >>> f"{PortfolioCommandError('no state')}"
    'no state'
    """


# ---------------------------------------------------------------------------
# What one command produced
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PortfolioRunOutcome:
    """Everything one :func:`run_portfolio` call produced -- and nothing else.

    The object carries the run itself, not a rendering of it: the strategy
    :class:`~scripts.run_strategy.StrategyRunOutcome` whose selection was weighted, the
    :class:`~portfolio_sys.contracts.TargetPortfolio` the constructor accepted and the
    :class:`~portfolio_sys.transition.PortfolioTransition` between the state this run
    was given and that target.  :func:`format_summary` is the only thing that turns it
    into text, so a test can assert on numbers instead of parsing stdout, and a
    notebook can call :func:`run_portfolio` directly.

    No frame and no weight table is copied into the outcome: every property reads
    through to the object that owns the number, so a target and the transition of one
    run can never disagree about a weight.

    Parameters
    ----------
    strategy_outcome:
        The strategy layer's own result object, exactly as
        :func:`scripts.run_strategy.run_strategy` returned it.  It is the *source* of
        the target, never a re-read of it.
    target_portfolio:
        The weights the registry-created constructor produced.  Its
        ``source_result`` must be ``strategy_outcome.result`` -- the same object, not
        an equal one.
    transition:
        The complete difference between the state this run was given and
        ``target_portfolio``.  Its ``target_portfolio`` must be that same target
        object.

    Raises
    ------
    PortfolioContractError
        If the three parts do not describe one construction: a part of the wrong type,
        a target built from another result, or a transition of another target -- so a
        printed report can never mix two runs.
    """

    def __post_init__(self) -> None:
        """Refuse an outcome whose parts do not describe one construction."""
        if not isinstance(self.strategy_outcome, StrategyRunOutcome):
            raise PortfolioContractError(
                "strategy_outcome must be a StrategyRunOutcome, got "
                f"{type(self.strategy_outcome).__name__}: a bare StrategyResult cannot say "
                "which data root and which selected count it came with"
            )
        if not isinstance(self.target_portfolio, TargetPortfolio):
            raise PortfolioContractError(
                "target_portfolio must be a TargetPortfolio, got "
                f"{type(self.target_portfolio).__name__}: weights outside the portfolio "
                "contract cannot be diffed against a state"
            )
        if not isinstance(self.transition, PortfolioTransition):
            raise PortfolioContractError(
                "transition must be a PortfolioTransition, got "
                f"{type(self.transition).__name__}: the difference between the state and the "
                "target is built by ``build_portfolio_transition``, never re-derived here"
            )

        target = self.target_portfolio
        transition = self.transition
        result = self.strategy_outcome.result
        problems: list[str] = []

        if target.source_result is not result:
            problems.append(
                "target_portfolio.source_result is not strategy_outcome.result: the target must "
                "be the one built from this run's cross-section, not a re-read or a copy"
            )
        if transition.target_portfolio is not target:
            problems.append(
                "transition.target_portfolio is not target_portfolio: the difference must be "
                "taken against the target this run produced"
            )
        if transition.as_of_date != target.as_of_date:
            problems.append(
                f"transition.as_of_date is {transition.as_of_date.isoformat()}, but the target is "
                f"dated {target.as_of_date.isoformat()}"
            )
        if target.strategy_spec != result.strategy_spec:
            problems.append(
                f"target_portfolio.strategy_spec is {target.strategy_spec!r}, but the result was "
                f"scored by {result.strategy_spec!r}"
            )
        if target.factor_requirements != self.strategy_outcome.factor_requirements:
            problems.append(
                "target_portfolio.factor_requirements differ from strategy_outcome's: the exact "
                "factor version a target was built from travels with it, unchanged"
            )
        if transition.construction_spec != target.construction_spec:
            problems.append(
                f"transition.construction_spec is {transition.construction_spec!r}, but the target "
                f"was built by {target.construction_spec!r}"
            )
        if transition.constraints != target.constraints:
            problems.append(
                f"transition.constraints are {transition.constraints!r}, but the target honoured "
                f"{target.constraints!r}"
            )

        if problems:
            raise PortfolioContractError("invalid PortfolioRunOutcome", problems)

    @property
    def strategy_result(self) -> StrategyResult:
        """``strategy_outcome.result``: the cross-section that was weighted."""
        return self.strategy_outcome.result

    @property
    def current_state(self) -> PortfolioState:
        """``transition.current_state``: the state this run was given, unchanged.

        It is the object the transition re-checks every row against, so the printed
        current weights and the printed changes always come from one portfolio.
        """
        return self.transition.current_state

    @property
    def constraints(self) -> PortfolioConstraints:
        """``target_portfolio.constraints``: the cash target and cap this run honoured."""
        return self.target_portfolio.constraints

    @property
    def construction_spec(self) -> PortfolioConstructionSpec:
        """``target_portfolio.construction_spec``: the rule *and* version that set it."""
        return self.target_portfolio.construction_spec

    @property
    def as_of_date(self) -> date:
        """The decision date this construction belongs to."""
        return self.target_portfolio.as_of_date

    @property
    def selected_count(self) -> int:
        """How many symbols the strategy selected -- the target's candidates."""
        return self.strategy_outcome.selected_count

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(as_of_date={self.as_of_date.isoformat()!r}, "
            f"construction={self.construction_spec.name!r}, "
            f"construction_version={self.construction_spec.version!r}, "
            f"selected_count={self.selected_count!r}, "
            f"target_position_count={len(self.target_portfolio.positions)!r}, "
            f"turnover={self.transition.turnover!r})"
        )

    strategy_outcome: StrategyRunOutcome
    target_portfolio: TargetPortfolio
    transition: PortfolioTransition


# ---------------------------------------------------------------------------
# The two current-state modes: arguments of one command, never a file
# ---------------------------------------------------------------------------
def _joined(arguments: Sequence[str]) -> str:
    """``--a``, ``--a and --b`` or ``--a, --b and --c``: flag names as a message."""
    names = list(arguments)
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + f" and {names[-1]}"


def check_current_state_arguments(
    *,
    initial_cash: bool,
    current_as_of_date: date | str | None,
    current_positions: Sequence[PortfolioPosition] | None,
    current_cash_weight: float | None,
) -> None:
    """Refuse a current-state request that is not exactly one of the two modes.

    Called before anything is read, so a mistyped command never opens a panel.  It
    only decides *which* mode was meant -- ``--initial-cash`` alone, or
    ``--current-as-of`` + ``--current-cash-weight`` (+ any ``--current-position``) --
    and never what a valid state is: the weights, the symbols, the sum and the date
    belong to :class:`~portfolio_sys.contracts.PortfolioState`, which validates the
    object this command then hands it.

    Parameters
    ----------
    initial_cash:
        Whether the all-cash mode was requested.
    current_as_of_date:
        ``--current-as-of`` as typed, or ``None`` when it was not given.
    current_positions:
        The parsed ``--current-position`` values, or ``None`` when the flag was not
        given at all.  An empty sequence is a *given* manual state with no holdings,
        which is legal: the account is entirely in cash.
    current_cash_weight:
        ``--current-cash-weight`` as typed, or ``None`` when it was not given.

    Raises
    ------
    PortfolioCommandError
        If ``--initial-cash`` was combined with any ``--current-*`` value, or if the
        manual mode is missing ``--current-as-of`` or ``--current-cash-weight``.
    """
    given = [
        name
        for name, value in (
            ("--current-as-of", current_as_of_date),
            ("--current-position", current_positions),
            ("--current-cash-weight", current_cash_weight),
        )
        if value is not None
    ]

    if initial_cash:
        if given:
            raise PortfolioCommandError(
                f"--initial-cash cannot be combined with {_joined(given)}: an account that "
                "starts in cash has no positions and no date of its own, so pass either "
                "--initial-cash or the --current-* arguments that describe the account, "
                "never both"
            )
        return

    missing = [
        name
        for name, value in (
            ("--current-as-of", current_as_of_date),
            ("--current-cash-weight", current_cash_weight),
        )
        if value is None
    ]
    if missing:
        raise PortfolioCommandError(
            f"missing {_joined(missing)}: pass --initial-cash for a first purchase, or "
            "describe the account as it stands today with --current-as-of, "
            "--current-cash-weight and any --current-position"
        )


def parse_current_position(text: str) -> PortfolioPosition:
    """Parse one ``--current-position SYMBOL=WEIGHT`` argument into a position.

    The text is split on the single ``=`` it must contain and *nothing else* is
    repaired: the symbol is not trimmed, not upper-cased and not re-spelled, and the
    weight is only converted to a float.  Whether that symbol is a ticker and whether
    that weight is a share of the account is decided by
    :class:`~portfolio_sys.contracts.PortfolioPosition`, so a ``CASH=0.1``, a negative
    or ``nan`` weight or an empty symbol is refused by the contract rather than
    guessed at here.

    Parameters
    ----------
    text:
        One ``SYMBOL=WEIGHT`` value, e.g. ``"AAPL=0.60"``.

    Returns
    -------
    PortfolioPosition
        The validated position.

    Raises
    ------
    PortfolioCommandError
        If the text is not a string, does not contain exactly one ``=``, has an empty
        symbol or an empty weight, or has a weight that is not a number.
    portfolio_sys.errors.PortfolioContractError
        If the symbol or the weight is not a legal holding.
    """
    if not isinstance(text, str):
        raise PortfolioCommandError(
            f"--current-position must be a SYMBOL=WEIGHT string, got {type(text).__name__}"
        )

    if text.count("=") != 1:
        detail = (
            "it contains no '='"
            if "=" not in text
            else f"it contains {text.count('=')} '=' characters"
        )
        raise PortfolioCommandError(
            f"--current-position must be SYMBOL=WEIGHT, got {text!r}: {detail}, so the symbol "
            "and the weight cannot be told apart"
        )

    symbol, _, weight_text = text.partition("=")

    if not symbol:
        raise PortfolioCommandError(
            f"--current-position must be SYMBOL=WEIGHT, got {text!r}: the symbol is empty"
        )
    if not weight_text:
        raise PortfolioCommandError(
            f"--current-position must be SYMBOL=WEIGHT, got {text!r}: the weight is empty"
        )

    try:
        weight = float(weight_text)
    except ValueError:
        raise PortfolioCommandError(
            f"--current-position must be SYMBOL=WEIGHT, got {text!r}: {weight_text!r} is not a "
            "number"
        ) from None

    return PortfolioPosition(symbol, weight)


def parse_current_positions(values: Sequence[str]) -> tuple[PortfolioPosition, ...]:
    """Parse every ``--current-position`` value, in the order it was given."""
    return tuple(parse_current_position(value) for value in values)


def _current_state(
    *,
    initial_cash: bool,
    decision_date: date,
    current_as_of_date: date | str | None,
    current_positions: Sequence[PortfolioPosition] | None,
    current_cash_weight: float | None,
) -> PortfolioState:
    """The portfolio this run diffs against, validated by ``PortfolioState`` itself.

    ``--initial-cash`` means an account that holds nothing but cash and is dated the
    decision date -- it has no history of its own to state.  The manual mode states
    the account as it stands: its own date (which the contract refuses if it is later
    than the decision), the holdings it carries and the cash that completes them.

    No weight is normalised, no symbol is repaired and no cash share is topped up
    here: :class:`~portfolio_sys.contracts.PortfolioState` owns all of that, and a
    state it refuses is a legal refusal of this command.

    Parameters
    ----------
    initial_cash:
        Whether the all-cash mode was requested; already checked for conflicts.
    decision_date:
        The decision date, as validated by the strategy layer, used as the date of an
        all-cash account.
    current_as_of_date:
        ``--current-as-of``, only read in the manual mode.
    current_positions:
        The parsed holdings, only read in the manual mode.  ``None`` and ``()`` both
        mean "no holdings": a manual all-cash account.
    current_cash_weight:
        ``--current-cash-weight``, only read in the manual mode.

    Returns
    -------
    PortfolioState
        The validated state.

    Raises
    ------
    portfolio_sys.errors.PortfolioContractError
        If the state the manual mode describes is not a portfolio: weights that do not
        add up to ``1.0``, a duplicate or unusable symbol, a non-positive or
        non-finite weight, a date that is not a bare ISO date.
    """
    if initial_cash:
        return PortfolioState(
            as_of_date=decision_date,
            positions=(),
            cash_weight=INITIAL_CASH_WEIGHT,
        )

    return PortfolioState(
        as_of_date=current_as_of_date,
        positions=() if current_positions is None else tuple(current_positions),
        cash_weight=current_cash_weight,
    )


# ---------------------------------------------------------------------------
# The composition
# ---------------------------------------------------------------------------
def run_portfolio(
    *,
    factor_name: str,
    factor_version: str,
    top_n: int,
    as_of_date: date | str,
    target_cash_weight: float,
    max_single_weight: float,
    initial_cash: bool = False,
    current_as_of_date: date | str | None = None,
    current_positions: Sequence[PortfolioPosition] | None = None,
    current_cash_weight: float | None = None,
    strategy_version: str = DEFAULT_STRATEGY_VERSION,
    data_root: Path | str | None = None,
    universe_path: Path | str | None = None,
) -> PortfolioRunOutcome:
    """Construct one target portfolio for one session and diff it against a state.

    The steps of the module docstring run strictly in order, each through the public
    interface of the module that owns it: the selection comes from
    :func:`scripts.run_strategy.run_strategy` (so no panel is opened, no factor is
    computed and no strategy rule is rebuilt here), the weights come from the
    constructor the registry built, and the difference comes from
    :func:`portfolio_sys.transition.build_portfolio_transition`.  Nothing is written:
    no panel, no target file, no state file, no order.

    The two current-state modes are mutually exclusive.  ``initial_cash=True`` is an
    account that holds nothing but cash, dated the decision date.  Otherwise the
    account is described by ``current_as_of_date``, ``current_cash_weight`` and
    ``current_positions`` -- three *arguments of this call*, never a file, never a
    database and never a default the next run inherits.

    Parameters
    ----------
    factor_name:
        Factor the strategy ranks by, e.g. ``"momentum_12_1"``.
    factor_version:
        The *exact* version of that factor to read, e.g. ``"1.0.0"``.  It is required
        (not defaulted) exactly as on the strategy command: a target is only traceable
        if the panel it was scored from is named.
    top_n:
        How many rankable symbols the strategy selects: a positive integer.
    as_of_date:
        The decision date, as a :class:`datetime.date` or a strict ISO-8601
        ``"YYYY-MM-DD"`` string.  Its panel must already be stored; the date is never
        adjusted.
    target_cash_weight:
        The cash share the target must leave uninvested, in ``[0, 1]``.  Validated by
        :class:`~portfolio_sys.contracts.PortfolioConstraints`.
    max_single_weight:
        The largest share one symbol may carry, in ``(0, 1]``.  Same validation.
    initial_cash:
        ``True`` for the all-cash mode; it cannot be combined with any ``current_*``
        value.
    current_as_of_date:
        The date the stated account belongs to; required in the manual mode and refused
        by :class:`~portfolio_sys.contracts.PortfolioContext` if it is later than
        ``as_of_date``.
    current_positions:
        The holdings of the stated account, as
        :class:`~portfolio_sys.contracts.PortfolioPosition` values.  ``None`` (the
        default) and ``()`` both mean "no holdings", which with a cash weight of ``1.0``
        is a manual all-cash account.
    current_cash_weight:
        The cash share of the stated account; required in the manual mode.
    strategy_version:
        The strategy version recorded in the spec and in every row of the selection.
    data_root:
        Root the panels are read from: ``<data_root>/factors/panels``.  ``None`` means
        :attr:`data_sys.config.PipelineConfig.data_root`.
    universe_path:
        The static stock-pool CSV.  ``None`` means the committed
        ``config/universe_seed.csv``.

    Returns
    -------
    PortfolioRunOutcome
        The strategy outcome it was built on, the target the constructor produced and
        the transition from the state to that target.

    Raises
    ------
    PortfolioCommandError
        If the two state modes were mixed, if the manual mode is missing one of its
        required values, or if the mode combination is otherwise impossible.
    strategy_sys.errors.StrategyError
        An invalid strategy parameter, a refused date, a registry failure.
    factor_sys.errors.FactorError
        A missing or damaged panel, an unreadable universe.
    portfolio_sys.errors.PortfolioContractError
        A constraint, a state or a context that is not a portfolio.
    portfolio_sys.errors.PortfolioConstructionError
        A legal selection ``capped_score_weighted`` cannot weight: a non-positive
        selected score, or a cap and cash target that leave no room for it.
    portfolio_sys.errors.PortfolioRegistryError
        If the rule could not be registered or created, which no caller can provoke
        with a legal argument.
    """
    # (0) Which current-state mode was meant, and whether this run's constraints are
    # legal: both are pure argument questions, so a mistyped command fails before any
    # panel is opened.
    check_current_state_arguments(
        initial_cash=initial_cash,
        current_as_of_date=current_as_of_date,
        current_positions=current_positions,
        current_cash_weight=current_cash_weight,
    )
    constraints = PortfolioConstraints(
        cash_weight=target_cash_weight,
        max_single_weight=max_single_weight,
    )

    # (1)-(2) the selection: the strategy layer's own entry function, called with the
    # flags it already owns.  No panel is opened here, no rule is rebuilt here.
    strategy_outcome = run_strategy(
        factor_name=factor_name,
        factor_version=factor_version,
        top_n=top_n,
        as_of_date=as_of_date,
        strategy_version=strategy_version,
        data_root=data_root,
        universe_path=universe_path,
    )

    # (3)-(4) what the account holds now.  Dated by the decision date the strategy
    # layer validated, so an all-cash start and the target always share one date.
    current_state = _current_state(
        initial_cash=initial_cash,
        decision_date=strategy_outcome.as_of_date,
        current_as_of_date=current_as_of_date,
        current_positions=current_positions,
        current_cash_weight=current_cash_weight,
    )

    # (5) the four inputs of one construction, validated together: the date, the
    # cross-section, the state and the constraints.
    context = PortfolioContext(
        as_of_date=strategy_outcome.as_of_date,
        strategy_result=strategy_outcome.result,
        current_state=current_state,
        constraints=constraints,
    )

    # (6)-(8) a table for this invocation only.  The registration probe validates the
    # rule, and ``create`` hands back the instance that actually constructs -- the
    # probe itself never weights anything.
    registry = PortfolioConstructorRegistry()
    construction_spec = registry.register(CappedScoreWeightedConstructor)
    constructor = registry.create(construction_spec)

    # (9) the target weights, from the registry-built instance.
    target_portfolio = constructor.construct(context)

    # (10) the complete weight difference between the two portfolios.
    transition = build_portfolio_transition(
        current_state=current_state,
        target_portfolio=target_portfolio,
    )

    # (11) happens in ``main``: this function answers with data, never with text.
    return PortfolioRunOutcome(
        strategy_outcome=strategy_outcome,
        target_portfolio=target_portfolio,
        transition=transition,
    )


# ---------------------------------------------------------------------------
# The summary -- the only rendering of an outcome
# ---------------------------------------------------------------------------
def _format_weight(value: float) -> str:
    """``0.100000`` -- one weight at :data:`WEIGHT_DECIMALS`, never a negative zero."""
    return f"{0.0 if value == 0.0 else value:.{WEIGHT_DECIMALS}f}"


def _transition_status(change: PortfolioWeightChange) -> str:
    """What the *target* asks for on one row -- never an order side, a fill or a trade.

    ``entry`` and ``exit`` are read exactly (a portfolio either holds a name or does
    not) and take precedence, so a brand-new holding is reported as an entry rather
    than as "more of it"; every other row is an ``increase``, a ``decrease`` or
    ``unchanged`` from the change's own tolerance.
    """
    if change.is_entry:
        return ENTRY_STATUS
    if change.is_exit:
        return EXIT_STATUS
    if change.is_buy:
        return INCREASE_STATUS
    if change.is_sell:
        return DECREASE_STATUS
    return UNCHANGED_STATUS


def _table(columns: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    """A header, one line per row, then nothing: ``none`` when the table is empty.

    Column widths follow the header names (plus :data:`COLUMN_GAP`), so the numbers
    under ``current_weight`` and ``target_weight`` never shift, and the last column is
    left unpadded, so no line ends in whitespace.
    """
    widths = [len(column) + COLUMN_GAP for column in columns]
    widths[-1] = len(columns[-1])

    def line(cells: Sequence[str]) -> str:
        return "".join(
            f"{cell:<{width}}" for cell, width in zip(cells, widths, strict=True)
        ).rstrip()

    lines = [line(columns)]
    if not rows:
        lines.append(EMPTY_TABLE_TEXT)
        return lines
    lines.extend(line([str(cell) for cell in row]) for row in rows)
    return lines


def format_summary(outcome: PortfolioRunOutcome) -> str:
    """Render ``outcome`` as the command's stable, one-field-per-line report.

    The text is a reporting surface, never a storage format: every number it prints is
    already in ``outcome`` at full precision, and nothing is recomputed here -- the
    weights are the target's own, the changes are the transition's own rows, the
    status words restate two flags of those rows and ``turnover`` is the transition's
    one-sided figure.  Printing therefore cannot change a portfolio.

    The block is always present and always in the same order: the decision date and the
    date of the stated account, the strategy's name and version, one
    ``required_factor`` line per exact ``(name, version)`` pair, the construction's name
    and version, this run's cash target and cap, the selected and target counts, the
    three weight aggregates, ``target:`` (the positions, or ``none``), ``transition:``
    (the complete difference, or ``none``) and ``cash:``.

    Parameters
    ----------
    outcome:
        The run to render.

    Returns
    -------
    str
        The summary, without a trailing newline (``print`` adds one).
    """
    target = outcome.target_portfolio
    transition = outcome.transition

    lines = [
        f"as_of_date: {outcome.as_of_date.isoformat()}",
        f"current_state_as_of_date: {outcome.current_state.as_of_date.isoformat()}",
        f"strategy_name: {outcome.strategy_result.strategy_spec.name}",
        f"strategy_version: {outcome.strategy_result.strategy_spec.version}",
    ]
    lines.extend(f"required_factor: {requirement}" for requirement in target.factor_requirements)
    lines.extend(
        [
            f"construction_name: {outcome.construction_spec.name}",
            f"construction_version: {outcome.construction_spec.version}",
            f"target_cash_weight: {_format_weight(outcome.constraints.cash_weight)}",
            f"max_single_weight: {_format_weight(outcome.constraints.max_single_weight)}",
            f"selected_count: {outcome.selected_count}",
            f"target_position_count: {len(target.positions)}",
            f"buy_weight: {_format_weight(transition.buy_weight)}",
            f"sell_weight: {_format_weight(transition.sell_weight)}",
            f"cash_weight_change: {_format_weight(transition.cash_weight_change)}",
            f"turnover: {_format_weight(transition.turnover)}",
            "target:",
        ]
    )
    lines.extend(
        _table(
            TARGET_COLUMNS,
            [(position.symbol, _format_weight(position.weight)) for position in target.positions],
        )
    )
    lines.append("transition:")
    lines.extend(
        _table(
            TRANSITION_COLUMNS,
            [
                (
                    change.symbol,
                    _format_weight(change.current_weight),
                    _format_weight(change.target_weight),
                    _format_weight(change.weight_change),
                    _transition_status(change),
                )
                for change in transition.changes
            ],
        )
    )
    lines.extend(
        [
            "cash:",
            f"current_weight: {_format_weight(outcome.current_state.cash_weight)}",
            f"target_weight: {_format_weight(target.cash_weight)}",
            f"weight_change: {_format_weight(transition.cash_weight_change)}",
        ]
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """The argument parser of ``python scripts/run_portfolio.py``.

    Seven flags are the strategy command's own -- *which* factor at *which* exact
    version, *how many* names, *which* session, *where* the panels live, *which* pool --
    and they are handed to :func:`scripts.run_strategy.run_strategy` unchanged.  Two
    more are this run's :class:`~portfolio_sys.contracts.PortfolioConstraints`.  The
    last four describe the current account in exactly one of the two documented modes.

    There is deliberately no ``--constructor`` (one real rule), no date range, no batch,
    no frequency or calendar, no state file, no save/load, no order, share, price,
    notional, cost or backtest flag; a missing, malformed or unknown flag is argparse's
    own exit code ``2``.
    """
    return _parser_for(PipelineConfig().data_root)


def _parser_for(default_data_root: Path) -> argparse.ArgumentParser:
    """Build the parser, naming the defaults the run will actually use."""
    parser = argparse.ArgumentParser(
        prog="run_portfolio.py",
        description=(
            "Build one target portfolio for one decision date from the stored factor "
            "panels and print the weight difference against the current account."
        ),
    )

    # -- the strategy's own flags -------------------------------------------
    parser.add_argument(
        "--factor",
        required=True,
        metavar="FACTOR_NAME",
        help="factor the strategy ranks by, e.g. momentum_12_1",
    )
    parser.add_argument(
        "--factor-version",
        dest="factor_version",
        required=True,
        metavar="VERSION",
        help=(
            "exact factor version to read, e.g. 1.0.0 (required: a target is only "
            "traceable if the panel it was scored from is named)"
        ),
    )
    parser.add_argument(
        "--top-n",
        dest="top_n",
        required=True,
        type=int,
        metavar="N",
        help="how many rankable symbols the strategy selects, and so how many candidates "
        "the constructor weights (a positive integer)",
    )
    parser.add_argument(
        "--as-of",
        dest="as_of",
        required=True,
        metavar="YYYY-MM-DD",
        help=(
            "decision date as strict ISO-8601; its panel must already be stored, and the "
            "date is never adjusted to a neighbouring session"
        ),
    )
    parser.add_argument(
        "--strategy-version",
        dest="strategy_version",
        metavar="VERSION",
        default=DEFAULT_STRATEGY_VERSION,
        help=f"version recorded in the strategy's identity (default: {DEFAULT_STRATEGY_VERSION})",
    )
    parser.add_argument(
        "--data-root",
        dest="data_root",
        metavar="PATH",
        default=None,
        help=f"root holding factors/panels (default: {default_data_root})",
    )
    parser.add_argument(
        "--universe-path",
        dest="universe_path",
        metavar="PATH",
        default=None,
        help="static stock-pool CSV (default: the committed config/universe_seed.csv)",
    )

    # -- this run's constraints --------------------------------------------
    parser.add_argument(
        "--target-cash-weight",
        dest="target_cash_weight",
        required=True,
        type=float,
        metavar="WEIGHT",
        help=(
            "cash share the target must leave uninvested, in [0, 1]; it becomes "
            "PortfolioConstraints.cash_weight"
        ),
    )
    parser.add_argument(
        "--max-single-weight",
        dest="max_single_weight",
        required=True,
        type=float,
        metavar="WEIGHT",
        help="largest share one symbol may carry, in (0, 1]; it becomes "
        "PortfolioConstraints.max_single_weight",
    )

    # -- the current account: mode A, or mode B ----------------------------
    parser.add_argument(
        "--initial-cash",
        dest="initial_cash",
        action="store_true",
        help=(
            "the account holds nothing but cash, dated --as-of: the first purchase.  "
            "Cannot be combined with any --current-* flag"
        ),
    )
    parser.add_argument(
        "--current-as-of",
        dest="current_as_of",
        metavar="YYYY-MM-DD",
        default=None,
        help=(
            "date the stated account belongs to; required without --initial-cash, and "
            "never later than --as-of (that would be look-ahead)"
        ),
    )
    parser.add_argument(
        "--current-position",
        dest="current_position",
        metavar="SYMBOL=WEIGHT",
        action="append",
        default=None,
        help=(
            "one current holding, e.g. AAPL=0.60; repeat the flag for each name.  The "
            "weights plus --current-cash-weight must add up to 1.0"
        ),
    )
    parser.add_argument(
        "--current-cash-weight",
        dest="current_cash_weight",
        type=float,
        metavar="WEIGHT",
        default=None,
        help="cash share of the stated account; required without --initial-cash",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse ``argv``, run the construction, print the report and return the exit code.

    Parameters
    ----------
    argv:
        Command-line arguments without the program name (``None`` means
        ``sys.argv[1:]``).  Passing a list lets a test drive the command exactly as a
        shell would, without starting a second process.

    Returns
    -------
    int
        ``0`` when the target and the difference were printed -- including an empty
        selection, which is an all-cash target, not a failure.  ``1`` when a
        :class:`~strategy_sys.errors.StrategyError`, a
        :class:`~factor_sys.errors.FactorError`, a
        :class:`~portfolio_sys.errors.PortfolioError` or this module's
        :class:`PortfolioCommandError` stopped the run; that message goes to stderr as
        one ``error: ...`` line, never as a traceback.  argparse keeps its own ``2``.
    """
    args = build_parser().parse_args(argv)

    try:
        outcome = run_portfolio(
            factor_name=args.factor,
            factor_version=args.factor_version,
            top_n=args.top_n,
            as_of_date=args.as_of,
            target_cash_weight=args.target_cash_weight,
            max_single_weight=args.max_single_weight,
            initial_cash=args.initial_cash,
            current_as_of_date=args.current_as_of,
            current_positions=(
                None
                if args.current_position is None
                else parse_current_positions(args.current_position)
            ),
            current_cash_weight=args.current_cash_weight,
            strategy_version=args.strategy_version,
            data_root=args.data_root,
            universe_path=args.universe_path,
        )
    except (FactorError, StrategyError, PortfolioError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(format_summary(outcome))
    return 0


__all__ = [
    "DEFAULT_STRATEGY_VERSION",
    "EMPTY_TABLE_TEXT",
    "PortfolioCommandError",
    "PortfolioRunOutcome",
    "WEIGHT_DECIMALS",
    "build_parser",
    "check_current_state_arguments",
    "format_summary",
    "main",
    "parse_current_position",
    "parse_current_positions",
    "run_portfolio",
]


if __name__ == "__main__":
    raise SystemExit(main())
