"""Strategy system: score and select symbols from validated factor panels.

Layering of the project::

    data_sys      market bars and the universe -- the only layer that knows files
        |
    factor_sys    the factor contract, its market-data reader, the runner, the
        |         registry, the panel store and the research analytics
        |
    strategy_sys  scores and selects symbols from those panels  <- this package
        |
    (portfolio)   weights, cash and position sizing             -- a later task
        |
    (execution)   orders, fills and costs                       -- a later task

This package is the **strategy data contract**: the objects a strategy and the
strategy runner exchange, so that every strategy is validated the same way
before anything is computed, stored or turned into a portfolio.

``FactorRequirement``
    one exact factor identity a strategy depends on
    (``factor_name`` + ``factor_version``)
``StrategySpec``
    what a strategy is and what it needs: name, version, required factors
``StrategyContext``
    what a strategy may look at on one decision date: the universe snapshot and
    one validated ``FactorPanel`` per factor
``StrategyResult``
    what a strategy returns: one row per symbol of the universe, with a score, a
    rank and a selection flag -- plus the strategy's full ``StrategySpec``, so the
    exact ``(factor_name, factor_version)`` pairs it depended on travel with the
    result (``result.factor_requirements``) towards a future portfolio or backtest.
``Strategy``
    the abstract scoring interface
``StrategyRegistry``
    the explicit ``StrategySpec -> factory`` table: one registered identity, one
    fresh strategy per ``create``, never a default rule chosen by the library
``StrategyRunner``
    one strategy on one decision date: it reads the panels the spec declares,
    builds the ``StrategyContext``, calls the strategy and accepts only *this*
    run's result -- so one run is one traceable cross-section
``StrategyError`` / ``StrategyContractError`` / ``StrategyRegistryError`` /
``StrategyRunError``
    the strategy layer's own exception hierarchy, with every contract problem
    collected in one error

Concrete rules live one level down, in :mod:`strategy_sys.strategies`
(``strategy_sys.strategies.single_factor.TopNByFactorStrategy`` to start with),
and are deliberately *not* re-exported here: the root stays the shared contract,
the table and the runner, so importing it never drags a particular rule into a
caller.
Which rules a project runs is a research decision, so nothing is registered
automatically: :class:`StrategyRegistry` is filled by explicit ``register`` calls
in the caller, keyed by the complete ``StrategySpec`` (name, version *and* the
exact factor versions).  :class:`StrategyRunner` then runs one of those
strategies for one date; it never chooses which.

What this package deliberately does **not** contain: a CLI, a default strategy
configuration, batch scheduling or any notion of a frequency, result storage,
portfolio construction, orders, fills or a backtest.  A strategy scores and
selects, the runner runs one of them for one decision date, and what that
selection is worth -- weighting, trading, measuring -- comes later, on top of the
objects defined here.

Examples
--------
>>> spec = StrategySpec(
...     name="top_n_momentum",
...     version="1.0.0",
...     required_factors=(FactorRequirement("momentum_12_1", "1.0.0"),),
... )
>>> spec.name
'top_n_momentum'
>>> [str(requirement) for requirement in spec.required_factors]
['momentum_12_1/1.0.0']
"""

from __future__ import annotations

from strategy_sys.contracts import (
    FORBIDDEN_RESULT_COLUMNS,
    STRATEGY_NAME_PATTERN,
    STRATEGY_RESULT_COLUMNS,
    FactorRequirement,
    Strategy,
    StrategyContext,
    StrategyResult,
    StrategySpec,
)
from strategy_sys.errors import (
    StrategyContractError,
    StrategyError,
    StrategyRegistryError,
    StrategyRunError,
)
from strategy_sys.registry import StrategyRegistry
from strategy_sys.runner import StrategyRunner

__version__ = "0.1.0"

__all__ = [
    "FORBIDDEN_RESULT_COLUMNS",
    "STRATEGY_NAME_PATTERN",
    "STRATEGY_RESULT_COLUMNS",
    "FactorRequirement",
    "Strategy",
    "StrategyContext",
    "StrategyContractError",
    "StrategyError",
    "StrategyRegistry",
    "StrategyRegistryError",
    "StrategyResult",
    "StrategyRunError",
    "StrategyRunner",
    "StrategySpec",
    "__version__",
]
