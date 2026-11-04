"""Construction rules: given a selection, decide how much weight each name carries.

What belongs here
-----------------
``portfolio_sys`` holds the *contract*; this sub-package holds the *rules*::

    strategy_sys.contracts    what is selected        (score, rank, is_selected)
        |
    portfolio_sys.contracts   the shapes a construction exchanges
        |                     (context, constraints, target, interface)
    portfolio_sys.constructors  the rules themselves -- this package

A module in this package may import the standard library and
:mod:`portfolio_sys.contracts` / :mod:`portfolio_sys.errors`, and nothing else:
no market data, no factor panel, no strategy runner, no store, no file, no
calendar, no network.  A rule is a pure function of the
:class:`~portfolio_sys.contracts.PortfolioContext` it is handed, which is exactly
why every rule here can be tested in memory.

The rules
---------
``CappedScoreWeightedConstructor``
    split the investable share in proportion to the selected symbols' ``score``,
    pin every name whose proportional weight would exceed
    ``constraints.max_single_weight`` at that cap, and re-split what is left among
    the unpinned names -- still in proportion to their original scores -- until
    nothing exceeds the cap (:mod:`portfolio_sys.constructors.capped_score_weighted`)

Nothing here is registered anywhere.  A concrete rule is imported by whoever
builds it -- by name, from the module that holds it -- so the package root
:mod:`portfolio_sys` stays the contract layer and never becomes a registry of
rules, exactly as :mod:`strategy_sys` keeps its rules in
:mod:`strategy_sys.strategies`:

    from portfolio_sys.constructors import CappedScoreWeightedConstructor

    constructor = CappedScoreWeightedConstructor()
    target = constructor.construct(context)

Not implemented here, on purpose: an equal-weight rule, a volatility-weighted
rule, an optimizer, a registry instance, a runner, a command line, persistence of
any kind, the trade list that turns a current state into a target, orders, fills,
costs, turnover and a backtest.

Examples
--------
>>> from portfolio_sys.constructors import (
...     DEFAULT_VERSION,
...     CappedScoreWeightedConstructor,
... )
>>> constructor = CappedScoreWeightedConstructor()
>>> constructor.spec.name
'capped_score_weighted'
>>> constructor.spec.version == DEFAULT_VERSION
True
"""

from __future__ import annotations

from portfolio_sys.constructors.capped_score_weighted import (
    DEFAULT_VERSION,
    CappedScoreWeightedConstructor,
)

__all__ = ["DEFAULT_VERSION", "CappedScoreWeightedConstructor"]
