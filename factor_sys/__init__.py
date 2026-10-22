"""Factor system: contracts first, everything else later.

Layering of the project::

    data_sys      market bars and the universe -- the only layer that knows files
        |
    factor_sys    the factor contract, its market-data reader, the
        |         single-factor runner, the name registry and the panel store;
        |         then transforms and analytics
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
    validated panel -> quality check, with every dependency reached through
    its interface
``FactorRunResult``
    what one run produced: the validated ``FactorPanel`` and the quality report
    (``None`` when nothing is wrong)
``FactorQualityChecker`` / ``DefaultFactorQualityChecker``
    the replaceable question "is this panel usable as it stands?", and the v1
    rule set behind it: coverage, and finiteness where it must hold
``FactorQualityIssue`` / ``FactorQualityReport``
    the *soft* problems a well-formed panel can still carry: one frozen issue
    (``code``/``severity``/``message``/``affected_symbols``) and the report
    that lists at least one of them
``FactorRunError``
    an orchestration failure (an unusable dependency, a date that is not a
    session, a factor that raises, a panel from another run)
``FactorRegistry`` / ``default_factor_registry``
    the explicit table from a factor name to a parameterless factory: a caller
    that knows only the name (a CLI flag, an analysis, a strategy, a backtest)
    creates a fresh factor instance without importing its class.  Adding a factor
    is one line in the default builder, never a change to the runner
``FactorRegistryError``
    a registration or lookup that cannot be honoured (a name that is not
    snake_case or is already taken, an unknown name, a factory that does not
    build a ``Factor``, a factory that hands back an instance it has already
    used)
``FactorStore`` / ``ParquetFactorStore``
    where a validated panel becomes durable: one interface (``write``/``read``/
    ``read_range``/``exists`` keyed by factor, version and session) and the v1
    implementation that keeps exactly one Parquet file per panel, replaced
    atomically.  ``read_range`` hands back the panels already stored between two
    days as a tuple in date order: it reads history, it never computes it
``FactorStoreError``
    a panel that cannot be stored, or a stored panel that cannot be trusted
``QuickAnalyzer`` / ``FactorPanelStatistics``
    the first, purely descriptive look at a panel: one frozen statistics object
    per session (counts, eligible coverage, missingness, and the
    mean/std/min/max of the usable values) and a date-ordered tuple of them for
    a history -- it reads no store, computes no factor and reads no returns
``QuickAnalysisError``
    a sequence of panels that cannot be described as one factor's history
    (mixed factors or versions, a repeated date, something that is not a panel)

Not implemented yet, on purpose: multi-date factor batches, a cache-hit policy,
transforms (``zscore``/``rank``), registry auto-discovery (plugins, entry points,
config files), forward returns, IC/ICIR and quantile-portfolio analysis,
strategies and backtests.

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
    FactorRegistryError,
    FactorRunError,
    FactorStoreError,
    MarketDataReadError,
    QuickAnalysisError,
)
from factor_sys.factor_store import FactorStore, ParquetFactorStore
from factor_sys.market_data import MarketDataReader, ParquetMarketDataReader
from factor_sys.quality import (
    DefaultFactorQualityChecker,
    FactorQualityChecker,
    FactorQualityIssue,
    FactorQualityReport,
)
from factor_sys.quick_analysis import FactorPanelStatistics, QuickAnalyzer
from factor_sys.registry import FactorRegistry, default_factor_registry
from factor_sys.runner import FactorRunner, FactorRunResult
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
    "DefaultFactorQualityChecker",
    "Factor",
    "FactorContext",
    "FactorContractError",
    "FactorError",
    "FactorPanel",
    "FactorPanelStatistics",
    "FactorQualityChecker",
    "FactorQualityIssue",
    "FactorQualityReport",
    "FactorRegistry",
    "FactorRegistryError",
    "FactorRunError",
    "FactorRunResult",
    "FactorRunner",
    "FactorSpec",
    "FactorStore",
    "FactorStoreError",
    "MarketDataReadError",
    "MarketDataReader",
    "ParquetFactorStore",
    "ParquetMarketDataReader",
    "QuickAnalysisError",
    "QuickAnalyzer",
    "StaticUniverseProvider",
    "UniverseProvider",
    "UniverseSnapshot",
    "__version__",
    "default_factor_registry",
]
