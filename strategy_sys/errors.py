"""Exception hierarchy for the strategy system.

The strategy system is a **separate boundary** from both the data pipeline and
the factor system: a strategy contract violation is not a
:class:`~factor_sys.errors.FactorError`, so neither ``data_sys`` nor
``factor_sys`` has to know that strategies exist, and a caller can catch the
whole strategy layer through :class:`StrategyError`.

Four errors cover the contract, the table and the run:

``StrategyError``
    the base class: everything the strategy layer raises is one of these, so
    ``except StrategyError`` is always safe at a call site.
``StrategyContractError``
    one strategy input or output broke the shared contract, and every problem
    found while checking it travels back in :attr:`~StrategyContractError.problems`.
``StrategyRegistryError``
    a strategy could not be registered, or could not be created from a spec:
    a factory that is not callable, does not build a strategy, changes what it
    builds, or reuses an instance; a duplicate spec; an unregistered request.
``StrategyRunError``
    one run could not be completed, or its output does not belong to it: a
    dependency that is not the interface, a strategy that is not a
    :class:`~strategy_sys.contracts.Strategy`, an unusable decision date, a
    universe provider or a factor store answering with the wrong object, or a
    result naming another strategy, session or universe than the one that was
    asked for.  Every mismatch found in one result travels back in
    :attr:`~StrategyRunError.problems`.

Anything that is *not* a strategy error -- a bare ``ValueError`` from pandas, a
``KeyError`` on a missing column, a ``TypeError`` from a mis-typed argument --
never reaches a caller of this layer: the contract validates its inputs and
reports what is wrong in its own vocabulary.

Examples
--------
>>> error = StrategyContractError(
...     "invalid StrategySpec",
...     ["name must be a non-empty string, got ''"],
... )
>>> print(error)
invalid StrategySpec
  - name must be a non-empty string, got ''
>>> error.problems
["name must be a non-empty string, got ''"]
"""

from __future__ import annotations

from collections.abc import Sequence


class StrategyError(Exception):
    """Base class for every error raised by the strategy system."""


def _with_problems(message: str, problems: Sequence[str]) -> str:
    """Append collected problems to ``message`` in the project's bullet style.

    The format deliberately matches the factor system's own helper: one error
    reads the same way wherever it comes from.  The function is re-declared here
    instead of imported, so this package never depends on another package's
    private name.
    """
    if not problems:
        return message
    return f"{message}\n  - " + "\n  - ".join(problems)


class StrategyContractError(StrategyError):
    """Raised when strategy input or output violates the shared contract.

    Validation *collects* instead of stopping at the first bad row (the style of
    :class:`factor_sys.errors.FactorContractError`), so one construction reports
    every problem the input has as a whole: a frame is checked for its dates, its
    symbol coverage, its scores, its ranks and its selection in one pass.

    Attributes
    ----------
    problems:
        Human readable descriptions of every violation found while validating one
        object, in the order the checks ran.

    Notes
    -----
    A *quality* finding is never an error.  An empty selection, a symbol the
    strategy chose not to score (``score`` missing) or a rank it did not assign
    are facts about the cross-section that the contract types and allows; what
    they mean is decided by the strategy and, afterwards, by the portfolio layer.
    """

    def __init__(self, message: str, problems: Sequence[str] | None = None) -> None:
        self.problems: list[str] = list(problems) if problems else []
        super().__init__(_with_problems(message, self.problems))


class StrategyRegistryError(StrategyError):
    """Raised when a strategy cannot be registered, or created from a spec.

    This is the registry's only error: ``register`` and ``create`` either honour
    a request or explain, in one message, why they cannot.  That covers a factory
    that is not callable, one that raises, one that builds something other than a
    :class:`~strategy_sys.contracts.Strategy`, one whose strategy declares no
    :class:`~strategy_sys.contracts.StrategySpec`, a duplicate registration, a
    request that is not a ``StrategySpec``, a spec the table does not hold, a
    factory that now builds a different spec, and a factory that hands back an
    instance it has already used.

    Unlike :class:`StrategyContractError` this error collects nothing: there is
    exactly one decision to report, and it is reported with the whole strategy
    identity -- name, version and the exact factor versions -- so a message is
    never ambiguous between two rules that share a name.

    Notes
    -----
    A :class:`~strategy_sys.errors.StrategyError` raised *inside* a factory is
    never wrapped: the layer that knows what went wrong already said it
    precisely, and a caller may branch on that type.  Any other exception from a
    factory travels back wrapped in this error, with the original as
    ``__cause__``.
    """


class StrategyRunError(StrategyError):
    """Raised when a strategy run cannot produce a valid strategy result.

    Only *orchestration* failures are reported here: a runner built with
    something that is not a
    :class:`~factor_sys.universe.UniverseProvider` or a
    :class:`~factor_sys.factor_store.FactorStore`, a ``run`` whose strategy is
    not a :class:`~strategy_sys.contracts.Strategy` (or declares no
    :class:`~strategy_sys.contracts.StrategySpec`), a decision date that is not
    a plain :class:`datetime.date` or a strict ISO ``"YYYY-MM-DD"`` string, a
    provider that answers with something other than a
    :class:`~factor_sys.universe.UniverseSnapshot` for the date it was asked
    about, a store that returns something other than a
    :class:`~factor_sys.contracts.FactorPanel`, a strategy that raises an
    unrelated exception or returns something other than a
    :class:`~strategy_sys.contracts.StrategyResult`, and a result that names
    another strategy, session or universe than the run asked for.

    Errors raised by the layers underneath keep their own type, so a caller can
    tell "the panels were not there" from "the factor and the context disagree"
    from "the run itself was impossible": a
    :class:`~strategy_sys.errors.StrategyError` raised inside a strategy is
    never wrapped, a :class:`~strategy_sys.errors.StrategyContractError` from
    :class:`~strategy_sys.contracts.StrategyContext` or from the result travels
    back untouched, and a
    :class:`~factor_sys.errors.FactorStoreError` stays exactly what the store
    raised.  Only an *unknown* exception -- one no layer of this project
    declared -- is wrapped here, with the original kept as ``__cause__``.

    Attributes
    ----------
    problems:
        Every reason a result was refused, collected the way
        :class:`StrategyContractError` collects them, so one failed run names
        all the ways its output mismatched at once.

    Examples
    --------
    >>> error = StrategyRunError(
    ...     "the result does not belong to this run",
    ...     ["as_of_date is 2024-01-08, expected 2024-01-05"],
    ... )
    >>> print(error)
    the result does not belong to this run
      - as_of_date is 2024-01-08, expected 2024-01-05
    >>> error.problems
    ['as_of_date is 2024-01-08, expected 2024-01-05']
    """

    def __init__(self, message: str, problems: Sequence[str] | None = None) -> None:
        self.problems: list[str] = list(problems) if problems else []
        super().__init__(_with_problems(message, self.problems))


__all__ = [
    "StrategyContractError",
    "StrategyError",
    "StrategyRegistryError",
    "StrategyRunError",
]
