"""Selection rules that need exactly one factor.

One sub-package per rule family keeps a strategy's module path readable:
``strategy_sys.strategies.single_factor.top_n`` says "a single-factor rule named
top-n" without a second imported convention.  Multi-factor blends will get their
own sibling package instead of growing parameters here, because a blend is a
different rule with a different identity.

Every rule in this package is pure: in-memory context in, validated result out.

Examples
--------
>>> from strategy_sys import FactorRequirement
>>> from strategy_sys.strategies.single_factor import TopNByFactorStrategy
>>> strategy = TopNByFactorStrategy(
...     FactorRequirement("book_to_price", "2.0.0"), top_n=20
... )
>>> strategy.spec.name
'top_20_by_book_to_price'
"""

from __future__ import annotations

from strategy_sys.strategies.single_factor.top_n import (
    DEFAULT_VERSION,
    TopNByFactorStrategy,
)

__all__ = ["DEFAULT_VERSION", "TopNByFactorStrategy"]
