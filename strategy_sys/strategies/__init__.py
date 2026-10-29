"""One-factor selection rules: rank a cross-section by one factor, select the top.

What belongs here
-----------------
``strategy_sys.strategies`` holds *rules*, not plumbing::

    strategy_sys.contracts   the shapes a strategy exchanges (spec, context, result)
        |
    strategy_sys.strategies  the rules themselves -- this package
        |
    (strategy runner)        building a context, iterating dates, storing results

A module in this package may import the standard library, :mod:`pandas`,
:mod:`strategy_sys.contracts` and :mod:`strategy_sys.errors`.  It must not touch
storage, the factor runner, market data, a trading calendar, the network or a
file, and it must not produce weights, quantities, orders or fills: a strategy
scores and selects, everything after that belongs to a later layer.

The strategies
--------------
``TopNByFactorStrategy``
    the generic single-factor rule: given one exact
    :class:`~strategy_sys.contracts.FactorRequirement`, select the ``top_n``
    rankable symbols with the highest ``raw_value``
    (:mod:`strategy_sys.strategies.single_factor.top_n`)

Nothing here is registered anywhere.  A concrete strategy is imported by whoever
builds it -- by name, from the sub-package that holds it -- so the package root
:mod:`strategy_sys` stays the contract layer and never becomes a registry of
rules.  A momentum rule is just this class with the momentum factor::

    TopNByFactorStrategy(FactorRequirement("momentum_12_1", "1.0.0"), top_n=10)

Not implemented yet, on purpose: a strategy registry, a strategy runner, a CLI,
result storage, multi-factor blends, portfolio construction, orders, fills and a
backtest.

Examples
--------
>>> from strategy_sys import FactorRequirement
>>> from strategy_sys.strategies import TopNByFactorStrategy
>>> strategy = TopNByFactorStrategy(
...     FactorRequirement("momentum_12_1", "1.0.0"), top_n=10
... )
>>> strategy.spec.name
'top_10_by_momentum_12_1'
"""

from __future__ import annotations

from strategy_sys.strategies.single_factor import TopNByFactorStrategy

__all__ = ["TopNByFactorStrategy"]
