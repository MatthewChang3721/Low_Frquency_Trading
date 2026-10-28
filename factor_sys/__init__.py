"""Factor system: contracts first, everything else later.

Layering of the project::

    data_sys      market bars and the universe -- the only layer that knows files
        |
    factor_sys    the factor contract, its market-data reader, the
        |         single-factor runner, the name registry and the panel store;
        |         then transforms and analytics
        |
    factors/      factor *formulas and parameters only* -- no I/O, no paths

This package started with the **factor data contract**: the objects a formula and
the factor runner exchange, so that every factor is validated the same way before
anything is computed or stored.  It now also owns the research additions that
live *outside* that contract: the forward-return labels measured after a
decision, the Spearman IC evaluation of a factor against them, and the quantile
grouping of a cross-section that asks which end of the ranking earned what.

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
``ForwardReturnPanel`` / ``ForwardReturnCalculator``
    the research label measured *after* a decision: for one factor, one session
    and one horizon, the close-to-close total return of every universe symbol
    ``h`` exchange sessions later (``adjusted_close(end) / adjusted_close(t) -
    1``), with unusable prices marked ``is_available=False`` instead of guessed.
    Labels read bars through ``MarketDataReader`` at the very end of the
    pipeline and never travel back into a context, a panel or a factor
``ForwardReturnError``
    a label that cannot be built or cannot be trusted (not a panel, an invalid
    horizon, a date that is not a session, an untrustworthy reader answer)
``ICObservation`` / ``ICReport`` / ``ICAnalyzer``
    the after-the-fact question "did higher factor values precede higher
    returns?": one cross-sectional Spearman rank IC per session (average ranks,
    ties included), and the frozen report that summarizes a history of them --
    mean IC, sample deviation (``ddof=1``) and ICIR.  The ranks live inside the
    statistic and are thrown away: no ``FactorPanel`` is written back, no
    transformed factor is produced, and nothing is stored.  Panels and labels
    in, a report out
``ICAnalysisError``
    a request or a report that contradicts itself: an invalid ``min_pairs``, two
    date sets that do not match, a universe that moved between a panel and its
    labels, a hand-built report whose averages disagree with its observations
``QuantileObservation`` / ``QuantileSpreadObservation`` / ``QuantileReport`` /
``QuantileAnalyzer``
    the second after-the-fact question, "did the *ranked* cross-section behave
    as the factor promised?": one decision date's usable symbols sorted by
    ``raw_value`` into ``Q`` equal-rank groups, each group's equal-weighted mean
    forward return, and the top-minus-bottom spread.  Ties share an average rank
    and are never split, an empty group is reported as ``None`` instead of
    ``0.0``, and the whole history is frozen into one report.  It is research
    evidence over realized labels, not a tradable portfolio: no order, no fill,
    no cost, no holding period
``QuantileAnalysisError``
    a request or a report that contradicts itself: an invalid ``quantiles``, two
    date sets that do not match, a universe that moved between a panel and its
    labels, a spread that is not the difference of its own extremes, or a report
    whose member counts do not add up
``DecisionSchedule`` / ``decision_schedule`` / ``select_decision_dates``
    the pure date rule behind a multi-date run: given a closed window, a cadence
    (``weekly``/``biweekly``/``monthly``), an anchor weekday and a calendar, the
    sessions a study decides on -- one per week (the anchor, or the session
    before it when the exchange is closed), every second of those weeks, or the
    last session of each month.  It selects dates only: no bars, no factor, no
    file, so the factor batch and the future IC / quantile batches share one
    definition of "the same dates"
``FactorScheduleError``
    a cadence, anchor, window or calendar that cannot be honoured (an inverted
    window, an unknown weekday, an anchor passed for a monthly schedule)
``FactorResearchError``
    a multi-date research analysis that cannot be produced: an unusable horizon,
    group count or path, a schedule that selected no decision date, a selected
    date that failed, or a report that cannot be published.  The layers
    underneath keep their own errors, so "the data was missing" stays
    distinguishable from "these objects do not belong together"

Not implemented yet, on purpose: transforms (``zscore``/``rank``), registry
auto-discovery (plugins, entry points, config files), multi-factor combination,
industry neutralization, annualized ICIR and risk-adjusted statistics (Sharpe,
t-statistics, Newey-West), execution, costs, slippage, strategies and backtests.
Two multi-date entry points exist -- ``scripts/run_factor_batch.py`` (panels and
their manifest) and ``scripts/analyze_factor.py`` (panels, forward-return
labels, IC and quantile reports) -- and neither introduces a cache-invalidation
policy beyond the per-key lookup of ``scripts/run_factor.py``.

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
    FactorResearchError,
    FactorRunError,
    FactorScheduleError,
    FactorStoreError,
    ForwardReturnError,
    ICAnalysisError,
    MarketDataReadError,
    QuantileAnalysisError,
    QuickAnalysisError,
)
from factor_sys.factor_schedule import (
    ANCHOR_WEEKDAYS,
    BIWEEKLY,
    BIWEEKLY_ANCHOR,
    DEFAULT_ANCHOR_WEEKDAY,
    DEFAULT_FREQUENCY,
    FREQUENCIES,
    MONTHLY,
    WEEKLY,
    DecisionSchedule,
    decision_schedule,
    select_decision_dates,
)
from factor_sys.factor_store import FactorStore, ParquetFactorStore
from factor_sys.forward_returns import (
    FORWARD_RETURN_COLUMNS,
    ForwardReturnCalculator,
    ForwardReturnPanel,
)
from factor_sys.ic_analysis import ICAnalyzer, ICObservation, ICReport
from factor_sys.market_data import MarketDataReader, ParquetMarketDataReader
from factor_sys.quality import (
    DefaultFactorQualityChecker,
    FactorQualityChecker,
    FactorQualityIssue,
    FactorQualityReport,
)
from factor_sys.quantile_analysis import (
    QuantileAnalyzer,
    QuantileObservation,
    QuantileReport,
    QuantileSpreadObservation,
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
    "ANCHOR_WEEKDAYS",
    "BAR_REQUIRED_COLUMNS",
    "BIWEEKLY",
    "BIWEEKLY_ANCHOR",
    "DEFAULT_ANCHOR_WEEKDAY",
    "DEFAULT_FREQUENCY",
    "DecisionSchedule",
    "FACTOR_NAME_PATTERN",
    "FACTOR_PANEL_COLUMNS",
    "FORWARD_RETURN_COLUMNS",
    "FREQUENCIES",
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
    "FactorResearchError",
    "FactorRunError",
    "FactorRunResult",
    "FactorRunner",
    "FactorScheduleError",
    "FactorSpec",
    "FactorStore",
    "FactorStoreError",
    "ForwardReturnCalculator",
    "ForwardReturnError",
    "ForwardReturnPanel",
    "ICAnalysisError",
    "ICAnalyzer",
    "ICObservation",
    "ICReport",
    "MONTHLY",
    "MarketDataReadError",
    "MarketDataReader",
    "ParquetFactorStore",
    "ParquetMarketDataReader",
    "QuantileAnalysisError",
    "QuantileAnalyzer",
    "QuantileObservation",
    "QuantileReport",
    "QuantileSpreadObservation",
    "QuickAnalysisError",
    "QuickAnalyzer",
    "StaticUniverseProvider",
    "UniverseProvider",
    "UniverseSnapshot",
    "WEEKLY",
    "__version__",
    "decision_schedule",
    "default_factor_registry",
    "select_decision_dates",
]
