"""Exception hierarchy for the portfolio system.

The portfolio system is a **separate boundary** from the data, factor and
strategy systems: an unusable weight is not a
:class:`~strategy_sys.errors.StrategyError` and not a
:class:`~factor_sys.errors.FactorError`, so neither of those layers has to know
that portfolios exist, and a caller can catch everything this layer raises
through :class:`PortfolioError`.

Three errors cover the whole of this layer:

``PortfolioError``
    the base class: everything the portfolio layer raises is one of these, so
    ``except PortfolioError`` is always safe at a call site.
``PortfolioContractError``
    one portfolio input or output broke the shared contract, and every problem
    found while checking it travels back in :attr:`~PortfolioContractError.problems`.
``PortfolioConstructionError``
    a **legal** input could not be turned into weights by one construction rule:
    the selection carries a score that cannot become a long position, or the
    constraints leave the selection no room.  The objects are well formed, the
    rule simply cannot honour them.

The two are deliberately different types.  A contract error says "this object is
not a portfolio"; a construction error says "this portfolio is a portfolio, and
this rule still cannot build it".  A caller can therefore repair its data in the
first case and choose another rule, another selection or wider constraints in the
second -- and a construction failure is never disguised as a malformed object.

Nothing else is declared yet, deliberately: a constructor-registry error and a
persistence error belong to the tasks that can actually produce them, and a
caller should not have to guess which of two empty branches to catch.

Anything that is *not* a portfolio error -- a bare ``ValueError`` from a
conversion, a ``KeyError`` on a missing symbol, a ``TypeError`` from a mis-typed
argument -- never reaches a caller of this layer: the contract validates its
inputs and reports what is wrong in its own vocabulary.

Examples
--------
>>> error = PortfolioContractError(
...     "invalid PortfolioState",
...     ["cash_weight must be between 0 and 1 inclusive, got 1.5"],
... )
>>> print(error)
invalid PortfolioState
  - cash_weight must be between 0 and 1 inclusive, got 1.5
>>> error.problems
['cash_weight must be between 0 and 1 inclusive, got 1.5']
"""

from __future__ import annotations

from collections.abc import Sequence


class PortfolioError(Exception):
    """Base class for every error raised by the portfolio system."""


def _with_problems(message: str, problems: Sequence[str]) -> str:
    """Append collected problems to ``message`` in the project's bullet style.

    The format deliberately matches the factor and strategy systems' own helper:
    one error reads the same way wherever it comes from.  The function is
    re-declared here instead of imported, so this package never depends on
    another package's private name.
    """
    if not problems:
        return message
    return f"{message}\n  - " + "\n  - ".join(problems)


class PortfolioContractError(PortfolioError):
    """Raised when portfolio input or output violates the shared contract.

    Validation *collects* instead of stopping at the first bad field (the style
    of :class:`factor_sys.errors.FactorContractError` and
    :class:`strategy_sys.errors.StrategyContractError`), so one construction
    reports every problem the input has as a whole: a portfolio state is checked
    for its date, its holdings, its cash and its total in one pass.

    Attributes
    ----------
    problems:
        Human readable descriptions of every violation found while validating
        one object, in the order the checks ran.

    Examples
    --------
    >>> error = PortfolioContractError("invalid TargetPortfolio")
    >>> error.problems
    []
    >>> str(error)
    'invalid TargetPortfolio'
    """

    def __init__(self, message: str, problems: Sequence[str] | None = None) -> None:
        self.problems: list[str] = list(problems) if problems else []
        super().__init__(_with_problems(message, self.problems))


class PortfolioConstructionError(PortfolioError):
    """Raised when a construction rule cannot turn a context into target weights.

    The dividing line with :class:`PortfolioContractError` is the input, not the
    severity: a contract error means an *object* is malformed (a date that is a
    timestamp, weights that do not add up to ``1.0``), and this error means the
    objects are fine but one rule cannot honour them -- a selected score of ``0``
    cannot be a long-only share of the account, and a selection of three names at
    a 25% single-name cap cannot carry 90% of the account.

    It is raised per *rule*, not per contract: two constructors may disagree about
    what is buildable (one may drop a candidate, another may refuse), so a caller
    that wants to try another rule catches this error and keeps going.  Nothing
    here ever claims the input was malformed, and nothing ever repairs it quietly:
    the rule refuses and says why.

    Attributes
    ----------
    problems:
        Human readable descriptions of every finding, in the order the checks ran:
        one explanation per unusable score, or the arithmetic of an infeasible cap.

    Examples
    --------
    >>> error = PortfolioConstructionError(
    ...     "capped_score_weighted cannot honour PortfolioConstraints",
    ...     ["3 selected symbol(s) x 0.25 = 0.75 < 0.9"],
    ... )
    >>> print(error)
    capped_score_weighted cannot honour PortfolioConstraints
      - 3 selected symbol(s) x 0.25 = 0.75 < 0.9
    >>> error.problems
    ['3 selected symbol(s) x 0.25 = 0.75 < 0.9']
    """

    def __init__(self, message: str, problems: Sequence[str] | None = None) -> None:
        self.problems: list[str] = list(problems) if problems else []
        super().__init__(_with_problems(message, self.problems))


__all__ = [
    "PortfolioConstructionError",
    "PortfolioContractError",
    "PortfolioError",
]
