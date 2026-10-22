"""One factor, one date: the runner that turns three interfaces into a panel.

What this module is
-------------------
``factor_sys.runner`` is the only place in the factor system where the
replaceable parts meet::

    UniverseProvider.snapshot(as_of_date)      -> UniverseSnapshot
    MarketDataReader.read_history(...)         -> long bars frame
    FactorContext(as_of_date, universe, bars)  -> the point-in-time slice
    Factor.compute(context)                    -> FactorPanel
    FactorQualityChecker.inspect(panel)        -> FactorQualityReport | None

:class:`FactorRunner` walks those steps in a fixed order, works out how far back
the read has to start, checks that the panel it got back really belongs to the
run it was asked for, and only then asks a checker how usable that panel is.  It
is deliberately thin: it never rewrites a value, never fills a gap, never decides
eligibility, never judges quality itself and never opens a file.

Why the runner exists at all
----------------------------
A factor stays pure (no paths, no I/O, no clock) and a reader does not know what
a factor is, so *something* has to know all three interfaces at once -- that is
this module.  Because it only ever calls interfaces, a point-in-time universe, a
database-backed reader or an in-memory test fixture drop in without a single
formula changing.

How much history a run reads
----------------------------
``FactorSpec.lookback_sessions`` counts the sessions **before** ``as_of_date``,
which is *not* included in that number::

    lookback_sessions = 0    -> start on as_of_date          (1 session visible)
    lookback_sessions = 1    -> start on the session before  (2 sessions visible)
    lookback_sessions = 252  -> start 252 sessions back      (253 sessions visible)

The start date is always a *session* taken from
:func:`data_sys.market_calendar.last_sessions`, never "``as_of_date`` minus N
calendar days": the Monday after a weekend must reach back to Friday, and the
Monday after Good Friday must reach back to the Thursday, because the market was
shut in between.  When the calendar does not reach back far enough, its earliest
session is used -- a short history is the factor's problem, expressed through
``is_eligible``, not an orchestration error.

What the runner refuses to decide
---------------------------------
A reader may legitimately answer with an empty frame, with a symbol that has two
sessions of history, or with nothing at all for the window.  The runner hands
that on untouched: warm-up, missing-data policy and eligibility stay in the
formula, where the same rule is applied to every symbol.

The last step: a quality report
-------------------------------
A validated panel is not automatically a usable cross-section: half the universe
may be missing a year of history.  That is a *soft* problem, so the runner asks a
:class:`~factor_sys.quality.FactorQualityChecker` -- the default one unless
another is injected -- to inspect the panel, and hands back both objects::

    result = runner.run(Momentum12_1(), "2024-12-31")
    result.panel            # the validated FactorPanel
    result.quality_report   # a FactorQualityReport, or None when nothing is wrong

The checker runs only after every hard check has passed, so a broken read, an
invalid context or a panel from another run never reaches it.  A quality warning
or error is never raised as an exception here: whether "only 40 of 50 symbols"
is acceptable belongs to the strategy, the CLI and the backtest.

Failed runs
-----------
Errors stay as specific as the layer that raised them.  A broken read stays a
:class:`~factor_sys.errors.MarketDataReadError`, and a contract violation raised
by :class:`~factor_sys.contracts.FactorContext` or by a factor's own frame stays
a :class:`~factor_sys.errors.FactorContractError`.  Only orchestration failures
-- a dependency that does not implement its interface, a decision date that is
not a session, an unrelated exception from ``compute``, a panel that belongs to
a different run, or a checker that answers with something other than a report
about this panel -- become :class:`~factor_sys.errors.FactorRunError`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

from data_sys.errors import CalendarError
from data_sys.market_calendar import (
    DEFAULT_CALENDAR_NAME,
    calendar_bounds,
    get_calendar,
    is_session,
    last_sessions,
)
from factor_sys.contracts import (
    Factor,
    FactorContext,
    FactorPanel,
    FactorSpec,
    _normalize_date,
)
from factor_sys.errors import FactorContractError, FactorError, FactorRunError
from factor_sys.market_data import MarketDataReader
from factor_sys.quality import (
    DefaultFactorQualityChecker,
    FactorQualityChecker,
    FactorQualityReport,
)
from factor_sys.universe import UniverseProvider, UniverseSnapshot


def _calendar_named(calendar_name: Any) -> tuple[str, Any]:
    """Resolve ``calendar_name`` once: ``(name, calendar)`` or a clear failure."""
    if not isinstance(calendar_name, str) or not calendar_name.strip():
        raise FactorRunError(
            "calendar_name must be a non-empty trading calendar name such as "
            f"{DEFAULT_CALENDAR_NAME!r}, got {calendar_name!r}"
        )
    name = calendar_name.strip()
    try:
        return name, get_calendar(name)
    except CalendarError as exc:
        raise FactorRunError(f"could not load the trading calendar {name!r}: {exc}") from exc


# ---------------------------------------------------------------------------
# FactorRunResult
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FactorRunResult:
    """What one run produced: the panel, and what the quality check saw.

    ``run`` always answers with this one shape -- never a bare panel on a good
    day and a report on a bad one -- so calling code is the same in both cases::

        result = runner.run(Momentum12_1(), "2024-12-31")
        result.panel             # the validated FactorPanel, unchanged
        result.quality_report    # None, or a FactorQualityReport about that panel

    Parameters
    ----------
    panel:
        The object ``factor.compute`` returned, already validated against the
        run's date, universe, name and version.  It stays a
        :class:`~factor_sys.contracts.FactorPanel`, whose ``frame`` property
        hands out a copy: this result never exposes a mutable frame of its own.
    quality_report:
        ``None`` when the checker found nothing, otherwise a report whose
        identity (date, name, version, universe size) and non-empty issue list
        are checked here, so a forged or stale report can never be handed on.

    Raises
    ------
    factor_sys.errors.FactorRunError
        If ``panel`` is not a :class:`~factor_sys.contracts.FactorPanel`, if the
        report is neither ``None`` nor a
        :class:`~factor_sys.quality.FactorQualityReport`, or if the report does
        not describe exactly this panel.
    """

    panel: FactorPanel
    quality_report: FactorQualityReport | None = None

    def __post_init__(self) -> None:
        """Refuse a result whose two halves do not belong together."""
        if not isinstance(self.panel, FactorPanel):
            raise FactorRunError(
                f"FactorRunResult.panel must be a FactorPanel, got "
                f"{type(self.panel).__name__}: a run result carries the validated "
                "cross-section, not a raw frame"
            )

        report = self.quality_report
        if report is None:
            return
        if not isinstance(report, FactorQualityReport):
            raise FactorRunError(
                "FactorRunResult.quality_report must be a FactorQualityReport or None, got "
                f"{type(report).__name__}: a quality check either finds something or nothing"
            )

        problems: list[str] = []
        if report.as_of_date != self.panel.as_of_date:
            problems.append(
                f"as_of_date is {report.as_of_date.isoformat()}, expected "
                f"{self.panel.as_of_date.isoformat()}"
            )
        if report.factor_name != self.panel.factor_name:
            problems.append(
                f"factor_name is {report.factor_name!r}, expected {self.panel.factor_name!r}"
            )
        if report.factor_version != self.panel.factor_version:
            problems.append(
                f"factor_version is {report.factor_version!r}, expected "
                f"{self.panel.factor_version!r}"
            )
        if report.universe_size != len(self.panel.universe):
            problems.append(
                f"universe_size is {report.universe_size}, expected {len(self.panel.universe)}"
            )
        if not report.issues:
            problems.append(
                "issues is empty: a report exists only because something was found "
                "(a clean panel must be reported as None)"
            )

        if problems:
            raise FactorRunError(
                "the quality report does not describe the panel this run returned",
                problems,
            )


# ---------------------------------------------------------------------------
# FactorRunner
# ---------------------------------------------------------------------------
class FactorRunner:
    """Run one factor for one session: prepare its inputs, verify its output.

    One run is always these steps, in this order:

    1. the factor must be a :class:`~factor_sys.contracts.Factor` with a valid
       :class:`~factor_sys.contracts.FactorSpec` -- its declaration is what says
       which fields to read and how far back to start;
    2. :meth:`~factor_sys.universe.UniverseProvider.snapshot` decides which
       symbols may be computed, for exactly the requested date;
    3. that date must be a real session of the runner's calendar (a weekend, a
       holiday or a date the calendar does not cover fails here, before anything
       is read -- the decision date is never nudged to a neighbouring session);
    4. ``lookback_sessions`` sessions of history are requested *in addition to*
       ``as_of_date``;
    5. the bars come from
       :meth:`~factor_sys.market_data.MarketDataReader.read_history`, so the
       runner never touches Parquet itself;
    6. a :class:`~factor_sys.contracts.FactorContext` is built from the snapshot
       and those bars, which is where the no-look-ahead guard runs again;
    7. ``factor.compute(context)`` produces the panel;
    8. the panel must be a :class:`~factor_sys.contracts.FactorPanel` dated and
       populated exactly as this run asked;
    9. the panel is handed to a
       :class:`~factor_sys.quality.FactorQualityChecker`, whose answer travels
       back with it.

    The runner then returns a :class:`FactorRunResult` holding that *same* panel
    object plus the checker's report (``None`` when the panel is clean): it
    validates a panel, it never rewrites one, so a factor's values, eligibility
    flags and row order travel out exactly as they were produced.  A quality
    warning or error is reported, never raised -- the strategy, the CLI and the
    backtest decide what an "only 40 of 50 symbols" panel is worth.

    Parameters
    ----------
    universe_provider:
        Where the symbols for a date come from.  Only ``snapshot`` is ever
        called, so today's static pool can be replaced by a point-in-time
        implementation without this class changing.
    market_data_reader:
        Where the bars come from.  Every row a factor sees arrived through
        ``read_history``.
    calendar_name:
        Trading calendar that defines "a session" (``"XNYS"`` by default).  It is
        resolved once, when the runner is built, so a misspelled name fails
        immediately instead of at the first run.
    quality_checker:
        What decides whether a valid panel is a *usable* one
        (:class:`~factor_sys.quality.DefaultFactorQualityChecker` when omitted).
        It runs last, only on a panel that already passed every hard check, and
        it may be replaced without the runner learning a single rule.

    Raises
    ------
    factor_sys.errors.FactorRunError
        If a dependency does not implement its interface, the quality checker is
        neither ``None`` nor a
        :class:`~factor_sys.quality.FactorQualityChecker`, or the calendar name
        is blank or unknown.

    Notes
    -----
    A runner keeps no state between runs -- the same instance can run different
    factors for different dates, in any order -- and nothing is memoised except
    the calendar object itself (which
    :func:`data_sys.market_calendar.get_calendar` caches anyway).

    Examples
    --------
    .. code-block:: python

        runner = FactorRunner(StaticUniverseProvider(), ParquetMarketDataReader())
        result = runner.run(Momentum12_1(), "2024-01-05")
        result.panel          # one validated row per symbol of that day's pool
        result.quality_report  # None, or what the checker found
    """

    def __init__(
        self,
        universe_provider: UniverseProvider,
        market_data_reader: MarketDataReader,
        *,
        calendar_name: str = DEFAULT_CALENDAR_NAME,
        quality_checker: FactorQualityChecker | None = None,
    ) -> None:
        if not isinstance(universe_provider, UniverseProvider):
            raise FactorRunError(
                "universe_provider must be a UniverseProvider, got "
                f"{type(universe_provider).__name__}: the runner only knows the "
                "snapshot interface, so any pool implementation can be used"
            )
        if not isinstance(market_data_reader, MarketDataReader):
            raise FactorRunError(
                "market_data_reader must be a MarketDataReader, got "
                f"{type(market_data_reader).__name__}: the runner reads bars through "
                "that interface only and never parses Parquet itself"
            )
        if quality_checker is None:
            quality_checker = DefaultFactorQualityChecker()
        elif not isinstance(quality_checker, FactorQualityChecker):
            raise FactorRunError(
                "quality_checker must be a FactorQualityChecker, got "
                f"{type(quality_checker).__name__}: the runner asks one object "
                "inspect(panel) and never applies a quality rule itself"
            )

        name, calendar = _calendar_named(calendar_name)

        self._universe_provider = universe_provider
        self._market_data_reader = market_data_reader
        self._calendar_name = name
        self._calendar = calendar
        self._quality_checker = quality_checker

    @property
    def universe_provider(self) -> UniverseProvider:
        """The provider this runner asks for a universe snapshot."""
        return self._universe_provider

    @property
    def market_data_reader(self) -> MarketDataReader:
        """The reader this runner asks for bars."""
        return self._market_data_reader

    @property
    def calendar_name(self) -> str:
        """Name of the trading calendar that defines "a session" for this runner."""
        return self._calendar_name

    @property
    def calendar(self) -> Any:
        """The resolved calendar object (an ``exchange_calendars`` calendar)."""
        return self._calendar

    @property
    def quality_checker(self) -> FactorQualityChecker:
        """The checker every run asks about the panel it just produced."""
        return self._quality_checker

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(calendar_name={self._calendar_name!r}, "
            f"universe_provider={type(self._universe_provider).__name__}, "
            f"market_data_reader={type(self._market_data_reader).__name__}, "
            f"quality_checker={type(self._quality_checker).__name__})"
        )

    # -- the run -------------------------------------------------------------
    def run(self, factor: Factor, as_of_date: date | str) -> FactorRunResult:
        """Compute ``factor`` for the session ``as_of_date`` and inspect the panel.

        Parameters
        ----------
        factor:
            The formula to run.  Only ``factor.spec`` and ``factor.compute`` are
            used.
        as_of_date:
            The decision date, as a :class:`datetime.date` or an ISO-8601
            ``"YYYY-MM-DD"`` string (a ``datetime`` is accepted and
            day-normalized, because the decision date is a label).  It must be a
            session of the runner's calendar.

        Returns
        -------
        FactorRunResult
            ``.panel`` is the object ``factor.compute`` returned, unchanged and
            already validated against this run's date, universe, name and
            version; ``.quality_report`` is ``None`` when the checker found
            nothing, or a report about that very panel (never an exception: a
            soft problem is the caller's decision).

        Raises
        ------
        FactorRunError
            If the factor is not runnable, the provider answers with something
            that is not a snapshot for this date, the date is not a session, the
            factor raises an unrelated exception, the panel does not belong to
            this run, or the checker answers with something that is not a report
            about that panel.
        MarketDataReadError
            Propagated unchanged from the reader: an impossible request or a
            dataset that cannot answer it is the reader's failure to report.
        FactorContractError
            Propagated unchanged from :class:`FactorContext` or from a factor
            that produced an invalid panel.
        """
        spec = _runnable_spec(factor)
        decision = _decision_date(as_of_date)
        snapshot = _snapshot_for(self._universe_provider, decision)

        self._require_session(snapshot.as_of_date)
        start = self._history_start(snapshot.as_of_date, spec.lookback_sessions)

        bars = self._market_data_reader.read_history(
            snapshot=snapshot,
            start_date=start,
            required_fields=spec.required_fields,
        )
        context = FactorContext(
            as_of_date=snapshot.as_of_date,
            universe=snapshot.symbols,
            bars=bars,
        )

        panel = _computed_panel(factor, context, spec)
        _require_this_run(panel, snapshot, spec)
        return FactorRunResult(
            panel=panel,
            quality_report=_quality_report(self._quality_checker, panel),
        )

    # -- steps that need the calendar ---------------------------------------
    def _require_session(self, as_of: date) -> None:
        """Refuse a decision date that is not a session of this calendar."""
        first, last = calendar_bounds(self._calendar)
        if as_of < first or as_of > last:
            raise FactorRunError(
                f"{as_of.isoformat()} lies outside the {self._calendar_name} calendar "
                f"({first.isoformat()}..{last.isoformat()}): the decision date must be a "
                "session the calendar actually covers"
            )
        if not is_session(self._calendar, as_of):
            raise FactorRunError(
                f"{as_of.isoformat()} is not a {self._calendar_name} trading session (a "
                "weekend or an exchange holiday): the decision date must be a real "
                "session, and the runner never moves it to a neighbouring one"
            )

    def _history_start(self, as_of: date, lookback_sessions: int) -> date:
        """First date of the read window: ``as_of`` plus the sessions before it.

        ``lookback_sessions`` counts the sessions *before* ``as_of_date``, so the
        window starts ``lookback_sessions + 1`` sessions back -- and
        :func:`data_sys.market_calendar.last_sessions` already stops at the
        calendar's first session, which is exactly the "use the history that
        exists" rule: a calendar that begins late yields its earliest session
        instead of an invented date or an error.
        """
        sessions = last_sessions(self._calendar, as_of, lookback_sessions + 1)
        if not sessions:
            # Unreachable after ``_require_session``: the calendar just proved it
            # has a session on ``as_of``.  Kept so a broken calendar can never
            # turn into a read from an arbitrary date.
            raise FactorRunError(
                f"the {self._calendar_name} calendar holds no session up to "
                f"{as_of.isoformat()}, so the read window has no start"
            )
        return sessions[0]


# ---------------------------------------------------------------------------
# Request validation and answer checking
# ---------------------------------------------------------------------------
def _runnable_spec(factor: Any) -> FactorSpec:
    """The validated :class:`FactorSpec` of ``factor``: what this run must read."""
    if not isinstance(factor, Factor):
        raise FactorRunError(
            f"factor must be a Factor, got {type(factor).__name__}: a factor needs a spec "
            "and a compute method, so a plain callable cannot be run"
        )
    spec = factor.spec
    if not isinstance(spec, FactorSpec):
        raise FactorRunError(
            f"{type(factor).__name__}.spec must be a FactorSpec, got {type(spec).__name__}: "
            "the runner reads the required fields and the lookback from there"
        )
    return spec


def _decision_date(value: Any) -> date:
    """The run's decision date as a plain :class:`datetime.date`."""
    try:
        return _normalize_date(value, "as_of_date")
    except FactorContractError as exc:
        raise FactorRunError(
            f"as_of_date must be a date or an ISO-8601 'YYYY-MM-DD' string, got {value!r}"
        ) from exc


def _snapshot_for(provider: UniverseProvider, as_of: date) -> UniverseSnapshot:
    """Ask ``provider`` for the pool of ``as_of`` and check what came back."""
    snapshot = provider.snapshot(as_of)
    if not isinstance(snapshot, UniverseSnapshot):
        raise FactorRunError(
            f"{type(provider).__name__}.snapshot() must return a UniverseSnapshot, got "
            f"{type(snapshot).__name__}"
        )
    if snapshot.as_of_date != as_of:
        raise FactorRunError(
            f"{type(provider).__name__}.snapshot() answered for "
            f"{snapshot.as_of_date.isoformat()} when asked about {as_of.isoformat()}: a "
            "provider must echo the decision date it was given, never move it"
        )
    return snapshot


def _computed_panel(factor: Factor, context: FactorContext, spec: FactorSpec) -> FactorPanel:
    """``factor.compute(context)``, with unrelated failures given a domain type."""
    try:
        panel = factor.compute(context)
    except FactorError:
        # Already precise: a FactorContractError (an invalid panel), a
        # FactorRunError or anything else from the factor system is what the
        # caller expects to catch, so re-wrapping it would only hide it.
        raise
    except Exception as exc:
        raise FactorRunError(
            f"factor {spec.name!r} v{spec.version} raised {type(exc).__name__} while "
            f"computing {context.as_of_date.isoformat()}: {exc}"
        ) from exc

    if not isinstance(panel, FactorPanel):
        raise FactorRunError(
            f"factor {spec.name!r} must return a FactorPanel, got {type(panel).__name__}: "
            "an unvalidated frame must not travel towards storage"
        )
    return panel


def _require_this_run(
    panel: FactorPanel,
    snapshot: UniverseSnapshot,
    spec: FactorSpec,
) -> None:
    """Refuse a panel that describes a different date, pool, name or version."""
    problems: list[str] = []

    if panel.as_of_date != snapshot.as_of_date:
        problems.append(
            f"as_of_date is {panel.as_of_date.isoformat()}, expected "
            f"{snapshot.as_of_date.isoformat()}"
        )
    if panel.universe != snapshot.symbols:
        missing = [symbol for symbol in snapshot.symbols if symbol not in panel.universe]
        unexpected = [symbol for symbol in panel.universe if symbol not in snapshot.symbols]
        if missing or unexpected:
            problems.append(
                f"universe differs from the snapshot: missing {missing}, unexpected {unexpected}"
            )
        else:
            problems.append(
                "universe holds the snapshot's symbols in a different order: "
                f"{list(panel.universe)} vs {list(snapshot.symbols)}"
            )
    if panel.factor_name != spec.name:
        problems.append(f"factor_name is {panel.factor_name!r}, expected {spec.name!r}")
    if panel.factor_version != spec.version:
        problems.append(f"factor_version is {panel.factor_version!r}, expected {spec.version!r}")

    if problems:
        raise FactorRunError(
            f"the panel returned for {snapshot.as_of_date.isoformat()} does not belong to "
            f"this run of {spec.name!r} v{spec.version}",
            problems,
        )


def _quality_report(
    checker: FactorQualityChecker,
    panel: FactorPanel,
) -> FactorQualityReport | None:
    """Ask ``checker`` about ``panel``: last step, and never a rule of our own.

    Only reached after the panel has been accepted, so a hard failure can never
    be mistaken for a quality finding.  The answer must be ``None`` or a report;
    a report about another panel (or one with no issues) is refused later by
    :class:`FactorRunResult`, which is the single place that owns that invariant.
    """
    try:
        report = checker.inspect(panel)
    except FactorError:
        # Already precise: a FactorContractError from a custom checker, or any
        # other factor-system error, is what the caller expects to catch.
        raise
    except Exception as exc:
        raise FactorRunError(
            f"{type(checker).__name__}.inspect() raised {type(exc).__name__} while "
            f"checking {panel.factor_name!r} v{panel.factor_version} on "
            f"{panel.as_of_date.isoformat()}: {exc}"
        ) from exc

    if report is None or isinstance(report, FactorQualityReport):
        return report
    raise FactorRunError(
        f"{type(checker).__name__}.inspect() must return a FactorQualityReport or None, "
        f"got {type(report).__name__}: a quality check either finds something or nothing"
    )


__all__ = ["FactorRunResult", "FactorRunner"]
