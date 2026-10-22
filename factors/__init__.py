"""Factor formulas and their parameters -- and nothing else.

What belongs here
-----------------
``factors/`` is the formula layer of the project::

    data_sys      market bars, the universe and the trading calendar
        |
    factor_sys    the contract, the market-data reader, the runner, the name
        |         registry and the panel store; then transforms and analytics
    factors/      formulas only: ``adjusted_close(t-21) / adjusted_close(t-252) - 1``

A module in this package may import :mod:`factor_sys.contracts` (to subclass
:class:`~factor_sys.contracts.Factor`) and :mod:`factor_sys.errors` (to report a
contract violation), plus the standard library and pandas for the arithmetic.
It must not import :mod:`data_sys` -- no storage, no providers, no update path --
and it must not touch a file, a connection or the network.  Getting the bars is
the runner's job; a factor only ever sees a :class:`FactorContext`, which is what
makes every factor here testable from an in-memory frame.

The factors
-----------
``Momentum12_1``
    twelve-month price momentum that skips the most recent month
    (``momentum_12_1``, version ``1.0.0``, lookback 252 sessions)

Every factor here is reachable by name through
:func:`factor_sys.registry.default_factor_registry`: a new module becomes
available to callers that only know a name once it is registered there
explicitly.  The runner never imports a factor, so nothing else has to change.

Not implemented yet, on purpose: cross-sectional transforms (``rank``/``zscore``),
``QuickAnalysis``, multi-factor composition, IC analysis, strategies, portfolio
construction and backtesting.  Storing a panel is no longer on that list: a
validated panel already goes to :class:`factor_sys.factor_store.ParquetFactorStore`.

Examples
--------
>>> from factors import Momentum12_1
>>> Momentum12_1().spec.version
'1.0.0'
"""

from __future__ import annotations

from factors.momentum import Momentum12_1

__all__ = ["Momentum12_1"]
