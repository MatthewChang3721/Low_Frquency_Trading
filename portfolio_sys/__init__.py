"""Portfolio system: turn a strategy's selection into target weights.

Layering of the project::

    data_sys       market bars and the universe -- the only layer that knows files
        |
    factor_sys     the factor contract, its runner, its panel store and analytics
        |
    strategy_sys   scores and selects symbols from those panels
        |
    portfolio_sys  how much weight each selected symbol should carry  <- this package
        |
    (execution)    orders, fills and costs                            -- a later task
        |
    (backtest)     turnover, costs and a net-value curve              -- a later task

This package is the **portfolio data contract**: the objects a constructor, its
caller and (later) an execution/backtest layer exchange, so that every
construction is validated the same way before anything is traded.

``PortfolioPosition``
    one non-cash holding: a symbol and the share of the account it carries
``PortfolioState``
    what the account actually holds on one date -- cumulative, including names
    no longer in the universe
``PortfolioConstraints``
    the base limits a construction works within: cash share and single-name cap
``PortfolioConstructionSpec``
    the complete identity of a construction rule: ``name`` + ``version``
``PortfolioContext``
    everything one construction may read: the decision date, the strategy
    result, the current state and the constraints
``TargetPortfolio``
    what the account should hold after the decision -- the weights, plus the
    strategy result, rule identity and constraints they came from
``PortfolioConstructor``
    the abstract construction interface
``PortfolioError`` / ``PortfolioContractError`` / ``PortfolioConstructionError``
    the portfolio layer's own exception hierarchy: every contract problem
    collected in one error, and a construction that a legal input cannot satisfy
    reported as its own type

Three statements, never confused:

* ``StrategyResult`` (``strategy_sys``) -- **what** is selected;
* ``TargetPortfolio`` (here) -- **how much weight** each holding should carry;
* ``PortfolioState`` (here) -- **what the account holds right now**.

A weight is a share of the account, never a share count, a price or a trade:
turning weights into orders belongs to the execution layer.  Cash is a field
(``cash_weight``), never a ``"CASH"`` position, and long-only is a property of
the objects rather than a flag -- every stock weight is strictly positive.

What this package deliberately does **not** contain: an equal-weight rule, a
volatility-weighted rule, an optimizer, a constructor registry, a runner, a
command-line entry point, batch scheduling, a rebalance calendar, persistence of
any kind, the trade list that turns a current state into a target, orders, fills,
costs or a backtest.  Adding a rule is a new :class:`PortfolioConstructor`
subclass with its own :class:`PortfolioConstructionSpec`; it never changes the
objects here.

The contract and the rules are kept apart, the way :mod:`strategy_sys` keeps its
contract at the root and its strategies in :mod:`strategy_sys.strategies`: the
first rule lives one level down, in :mod:`portfolio_sys.constructors`
(``CappedScoreWeightedConstructor``: split the investable share in proportion to
the selected symbols' scores, pin every name that would exceed
``max_single_weight`` and re-split the rest), and it is deliberately *not*
re-exported here, so importing this root never drags a particular rule along.

Examples
--------
>>> state = PortfolioState("2024-01-04", (PortfolioPosition("AAPL", 0.6),), 0.4)
>>> state.weights_by_symbol
{'AAPL': 0.6}
>>> spec = PortfolioConstructionSpec(name="equal_weight", version="1.0.0")
>>> f"{spec.name}/{spec.version}"
'equal_weight/1.0.0'
>>> constraints = PortfolioConstraints(cash_weight=0.1, max_single_weight=0.12)
>>> constraints.invested_weight
0.9
"""

from __future__ import annotations

from portfolio_sys.contracts import (
    CONSTRUCTION_NAME_PATTERN,
    WEIGHT_TOLERANCE,
    PortfolioConstraints,
    PortfolioConstructionSpec,
    PortfolioConstructor,
    PortfolioContext,
    PortfolioPosition,
    PortfolioState,
    TargetPortfolio,
)
from portfolio_sys.errors import (
    PortfolioConstructionError,
    PortfolioContractError,
    PortfolioError,
)

__version__ = "0.1.0"

__all__ = [
    "CONSTRUCTION_NAME_PATTERN",
    "WEIGHT_TOLERANCE",
    "PortfolioConstraints",
    "PortfolioConstructionError",
    "PortfolioConstructionSpec",
    "PortfolioConstructor",
    "PortfolioContext",
    "PortfolioContractError",
    "PortfolioError",
    "PortfolioPosition",
    "PortfolioState",
    "TargetPortfolio",
    "__version__",
]
