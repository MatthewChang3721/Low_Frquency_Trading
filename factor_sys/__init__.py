"""Factor system: contracts first, everything else later.

Layering of the project::

    data_sys      market bars and the universe -- the only layer that knows files
        |
    factor_sys    the factor contract, its market-data reader and the
        |         single-factor runner; then transforms, storage and analytics
        |
    factors/      factor *formulas and parameters only* -- no I/O, no paths

This step implements the **factor data contract**: the objects a formula and the
factor runner exchange, so that every factor is validated the same way before
anything is computed or stored.

``FactorSpec``
    what a factor needs: name, version, required fields, lookback
``FactorContext``
    what a factor may look at: the history visible on ``as_of_date`` (never
    anything after it)
``FactorPanel``
    what a factor returns: one validated cross-section in long format
``Factor``
    the abstract formula interface
``UniverseSnapshot``
    which symbols a factor may be computed for on one date, read from the
    committed pool
``UniverseProvider`` / ``StaticUniverseProvider``
    the interface a future historical universe must implement, and the current
    *static* implementation of it (today's pool for every date: not
    point-in-time, survivorship bias not removed)
``MarketDataReader`` / ``ParquetMarketDataReader``
    how a factor gets its market history: one interface that hands out a long
    ``date``/``symbol``/field frame for the window
    ``[start_date, snapshot.as_of_date]`` (never anything later), and the v1
    implementation that reads it from the published Hive dataset
``MarketDataReadError``
    a read that cannot be honoured, or a local dataset that cannot answer it
``FactorRunner``
    one factor, one session: snapshot -> read window -> context -> compute ->
    validated panel, with every dependency reached through its interface
``FactorRunError``
    an orchestration failure (an unusable dependency, a date that is not a
    session, a factor that raises, a panel from another run)

Not implemented yet, on purpose: concrete factors, factor storage, transforms
(``zscore``/``rank``) and IC analysis.

Example
-------
>>> from factor_sys import FactorContext, FactorSpec
>>> spec = FactorSpec(name="momentum_12_1", version="1.0.0", required_fields=("close",))
>>> spec.lookback_sessions
0
"""

from __future__ import annotations

from factor_sys.contracts import (
    BAR_REQUIRED_COLUMNS,
    FACTOR_NAME_PATTERN,
    FACTOR_PANEL_COLUMNS,
    Factor,
    FactorContext,
    FactorPanel,
    FactorSpec,
)
from factor_sys.errors import (
    FactorContractError,
    FactorError,
    FactorRunError,
    MarketDataReadError,
)
from factor_sys.market_data import MarketDataReader, ParquetMarketDataReader
from factor_sys.runner import FactorRunner
from factor_sys.universe import (
    StaticUniverseProvider,
    UniverseProvider,
    UniverseSnapshot,
)

__version__ = "0.1.0"

__all__ = [
    "BAR_REQUIRED_COLUMNS",
    "FACTOR_NAME_PATTERN",
    "FACTOR_PANEL_COLUMNS",
    "Factor",
    "FactorContext",
    "FactorContractError",
    "FactorError",
    "FactorPanel",
    "FactorRunError",
    "FactorRunner",
    "FactorSpec",
    "MarketDataReadError",
    "MarketDataReader",
    "ParquetMarketDataReader",
    "StaticUniverseProvider",
    "UniverseProvider",
    "UniverseSnapshot",
    "__version__",
]
