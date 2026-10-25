"""Run one factor for one decision date: name -> panel -> stored file -> summary.

What this script is
-------------------
The minimal, real entry point of the factor research layer.  It owns no formula,
no storage rule and no quality rule: it joins the modules that already exist, in
exactly this order::

     1  default_factor_registry()          the name -> factory table
     2  registry.create(factor_name)       a fresh factor, created by name
     3  the date and the trading calendar  what "a session" means for this run
     4  ParquetFactorStore(...).exists()   is this key already stored?
     5  store.read(...)                    -- hit: the stored panel, re-checked
        + quality checker + analyzer          and summarized; no bars, no run,
                                              no write
     6  StaticUniverseProvider(...)        -- miss or ``--force``: the stock pool
     7  ParquetMarketDataReader(...)       the published local bars
     8  FactorRunner(...)                  the three interfaces, wired together
     9  runner.run(factor, as_of_date)     one validated panel + a quality report
    10  ParquetFactorStore(...).write(...) one Parquet file per key, replaced atomically
    11  QuickAnalyzer().summarize(panel)   counts and the four moments
    12  print the summary                  stable ``key: value`` lines on stdout

Usage (project root, virtual environment active)::

    python scripts/run_factor.py --factor momentum_12_1 --as-of 2024-12-31
    python scripts/run_factor.py --factor momentum_12_1 --as-of 2024-12-31 `
        --data-root C:\\research\\data --universe-path C:\\research\\pool.csv

``--data-root`` decides **both** ends of a run: the bars are read from
``<data-root>/standardized/market_bars`` and the panel is written to
``<data-root>/factors/panels``.  ``--universe-path`` replaces only the static
pool CSV.  ``--calendar`` names the trading calendar that decides what "a
session" is -- ``XNYS`` by default, taken from the data system rather than
re-declared here.

Nothing here downloads anything
-------------------------------
The command reads the standardized Parquet dataset ``data_sys`` already
published and writes one factor panel.  It never contacts a vendor, never calls
the universe or market-data updater, and never creates a raw or staging
directory: on a machine without ``data/standardized/market_bars`` it fails with a
readable error instead of trying to fetch one.

One date, one factor, one key
-----------------------------
There is deliberately no date range, no batch loop, no rebalance schedule and no
date-series flag: one invocation answers for one session.  What it does have is
a light cache.  A panel is stored under ``(factor_name, factor_version,
as_of_date)``, so a second run of the same key reads that panel back instead of
recomputing it -- no bars are read, no factor runs and the stored file is not
rewritten.  ``--force`` skips the read, recomputes the cross-section and replaces
the file atomically; on a missing key it simply computes once and says so.

That cache is a storage-key lookup and nothing more.  It carries no lineage: it
cannot see that the bars, the static pool, the formula or the code changed since
the panel was written, so it keeps answering with what is stored until
``--force`` (or a new ``factor_version``) says otherwise.  A stored file that
cannot be restored is a :class:`~factor_sys.errors.FactorStoreError`, never a
silent recomputation: damaged evidence is reported, not overwritten.

Forward returns, IC, quantile spreads, strategies and backtests all need a date
*series*, so they belong to the research layers above this entry point and are
not reachable from here.

Exit codes
----------
``0``
    the panel was computed or read back, and summarized.  A non-``None`` quality
    report is **not** a failure: it is printed, and the exit code stays ``0``.
``1``
    a domain error (:class:`factor_sys.errors.FactorError`): an unknown factor, a
    date that is not a session, an unknown calendar, an unreadable or empty
    universe, a missing or empty standardized dataset, a factor that failed, or a
    store that could not write (or could not restore the panel it holds).  The
    message goes to stderr as one ``error: ...`` line, and no panel is written,
    because the store is only reached once the run succeeded.
``2``
    argparse refused the command line (a missing or unknown flag).

What is printed
---------------
Sixteen stable ``key: value`` lines -- ``factor_name``, ``factor_version``,
``as_of_date``, ``universe_size``, ``eligible_count``, ``eligible_coverage``,
``valid_count``, ``missing_count``, ``missing_rate``, ``mean``, ``std``,
``minimum``, ``maximum``, ``stored_panel_path``, ``quality_status`` and
``cache_status`` (``hit``, ``miss`` or ``forced``) -- and then, only when the
checker found something, four lines per issue (``quality_issue_code``,
``_severity``, ``_message``, ``_affected_symbols``).  Numbers are printed to six
decimals; a statistic with no value (an empty usable sample) is printed as
``n/a``.  A clean run prints no issue line at all -- nothing is invented to fill
the gap.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

# Running this file directly puts ``scripts/`` (not the project root) on
# ``sys.path``; add the root so ``python scripts/run_factor.py`` works from a
# checkout, with or without the package installed in the environment.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from data_sys.config import PipelineConfig  # noqa: E402
from data_sys.errors import CalendarError  # noqa: E402
from data_sys.market_calendar import (  # noqa: E402
    DEFAULT_CALENDAR_NAME,
    calendar_bounds,
    get_calendar,
    is_session,
)
from factor_sys import (  # noqa: E402
    DefaultFactorQualityChecker,
    FactorError,
    FactorPanel,
    FactorPanelStatistics,
    FactorQualityIssue,
    FactorQualityReport,
    FactorRunError,
    FactorRunner,
    ParquetFactorStore,
    ParquetMarketDataReader,
    QuickAnalyzer,
    StaticUniverseProvider,
    default_factor_registry,
)

#: Printed for a statistic that has no value (an empty usable sample).
MISSING_TEXT = "n/a"

#: Decimals used for the coverage, the rates and the four moments.
DECIMALS = 6

#: Every answer the store can give about a key: read back, computed, or forced.
CACHE_STATUSES: tuple[str, ...] = ("hit", "miss", "forced")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class RunFactorError(FactorError):
    """A request this entry point refuses before any module could answer it.

    Only *argument* problems land here -- a blank ``--data-root``, a path of the
    wrong type -- because everything else stays with the layer that owns the
    rule: an unknown factor is a
    :class:`~factor_sys.errors.FactorRegistryError`, a non-session date is a
    :class:`~factor_sys.errors.FactorRunError`, a missing dataset is a
    :class:`~factor_sys.errors.MarketDataReadError` and a failed write is a
    :class:`~factor_sys.errors.FactorStoreError`.  All of them share
    :class:`~factor_sys.errors.FactorError`, so the command line catches them the
    same way and never has to guess.
    """


# ---------------------------------------------------------------------------
# What one command produced
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FactorRunOutcome:
    """Everything one :func:`run_factor` call produced, already stored.

    The object carries the run itself, not a rendering of it: the panel the
    factor produced (or the one the store handed back), the statistics computed
    *from* that panel, the quality report the default checker produced for it
    (``None`` when the panel was clean), the file that panel lives in and how that
    file came to be.  :func:`format_summary` is the only thing that turns this
    into text, so a test can assert on numbers instead of parsing stdout, and a
    notebook can reuse the same call.

    Parameters
    ----------
    panel:
        The validated cross-section: what the factor returned, or -- for a cache
        hit -- the panel the store restored and validated.
    statistics:
        The :class:`~factor_sys.quick_analysis.FactorPanelStatistics` of that
        very panel: counts, coverage, missing rate and the four moments.
    quality_report:
        ``None`` when the checker found nothing, otherwise a report about this
        panel.  It is a *soft* finding: the panel is stored either way, and a
        stored panel is inspected again every time it is read back.
    stored_panel_path:
        The Parquet file of ``(factor_name, factor_version, as_of_date)``: the
        path :meth:`~factor_sys.factor_store.ParquetFactorStore.write` returned
        for a computed panel, or the existing file a hit was read from.
    cache_status:
        ``"hit"`` when the panel was already stored and read back, ``"miss"``
        when nothing was stored and the run computed and wrote it, ``"forced"``
        when ``--force`` made the run recompute and replace it.

    Raises
    ------
    RunFactorError
        If the parts do not describe one run (a hand-built outcome whose
        statistics or report belong to another factor, version or date, or a
        ``cache_status`` outside :data:`CACHE_STATUSES`).
    """

    panel: FactorPanel
    statistics: FactorPanelStatistics
    quality_report: FactorQualityReport | None
    stored_panel_path: Path
    cache_status: str

    def __post_init__(self) -> None:
        """Refuse an outcome whose parts do not belong together."""
        problems: list[str] = []

        if not isinstance(self.panel, FactorPanel):
            problems.append(f"panel must be a FactorPanel, got {type(self.panel).__name__}")
            panel = None
        else:
            panel = self.panel

        if not isinstance(self.statistics, FactorPanelStatistics):
            problems.append(
                f"statistics must be a FactorPanelStatistics, got {type(self.statistics).__name__}"
            )
            statistics = None
        else:
            statistics = self.statistics

        if self.quality_report is not None and not isinstance(
            self.quality_report, FactorQualityReport
        ):
            problems.append(
                "quality_report must be a FactorQualityReport or None, got "
                f"{type(self.quality_report).__name__}: a clean panel is reported as "
                "None, never as an empty report"
            )
            report = None
        else:
            report = self.quality_report

        if not isinstance(self.stored_panel_path, Path):
            problems.append(
                "stored_panel_path must be a pathlib.Path, got "
                f"{type(self.stored_panel_path).__name__}"
            )

        if self.cache_status not in CACHE_STATUSES:
            problems.append(
                f"cache_status must be one of {', '.join(repr(s) for s in CACHE_STATUSES)}, "
                f"got {self.cache_status!r}: a stored panel is either read back, computed "
                "or recomputed on request, and nothing else"
            )

        for label, described in (("statistics", statistics), ("quality_report", report)):
            if described is None or panel is None:
                continue
            if (
                described.as_of_date != panel.as_of_date
                or described.factor_name != panel.factor_name
                or described.factor_version != panel.factor_version
            ):
                problems.append(
                    f"{label} describes {described.factor_name!r} v"
                    f"{described.factor_version} on {described.as_of_date.isoformat()}, "
                    f"not {panel.factor_name!r} v{panel.factor_version} on "
                    f"{panel.as_of_date.isoformat()}"
                )

        if problems:
            raise RunFactorError(
                "the run outcome does not describe one run: " + "; ".join(problems)
            )

    @property
    def quality_status(self) -> str:
        """``"clean"`` when the panel raised no issue, ``"reported"`` otherwise."""
        return "clean" if self.quality_report is None else "reported"

    @property
    def issues(self) -> tuple[FactorQualityIssue, ...]:
        """The issues of this run in report order; empty for a clean panel."""
        return () if self.quality_report is None else self.quality_report.issues

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(factor_name={self.statistics.factor_name!r}, "
            f"factor_version={self.statistics.factor_version!r}, "
            f"as_of_date={self.statistics.as_of_date.isoformat()!r}, "
            f"quality_status={self.quality_status!r}, "
            f"cache_status={self.cache_status!r})"
        )


# ---------------------------------------------------------------------------
# The composition
# ---------------------------------------------------------------------------
def run_factor(
    factor_name: str,
    as_of_date: date | str,
    *,
    data_root: Path | str | None = None,
    universe_path: Path | str | None = None,
    calendar_name: str | None = None,
    force: bool = False,
) -> FactorRunOutcome:
    """Compute one factor for one session, store the panel and describe it.

    The steps of the module docstring run strictly in order, each through the
    public interface of the module that owns it: no formula is imported here, no
    :class:`~factor_sys.contracts.FactorContext` is built by hand, no Parquet file
    is opened here, no quality rule lives in this file and the store's layout is
    never reproduced here.

    A stored panel short-circuits the computation.  The factor, its version, the
    decision date and the calendar are checked first, then the store is asked
    whether ``(spec.name, spec.version, as_of_date)`` is already there; when it is
    and ``force`` is false, that panel is read back, inspected by the default
    quality checker and summarized.  No bars are read, no factor runs and the file
    is not rewritten.  A damaged stored panel is an error, never a reason to
    recompute over it.

    Parameters
    ----------
    factor_name:
        Registered name of the factor, e.g. ``"momentum_12_1"``.  It is resolved
        through :func:`factor_sys.default_factor_registry`, so a typo fails
        before any file is opened.
    as_of_date:
        The decision date, as a :class:`datetime.date` or a strict ISO-8601
        ``"YYYY-MM-DD"`` string.  It must be a session of ``calendar_name``.
    data_root:
        Root the whole run is derived from: bars are read from
        ``<data_root>/standardized/market_bars`` and the panel is stored in
        ``<data_root>/factors/panels``.  ``None`` means
        :attr:`data_sys.config.PipelineConfig.data_root`.
    universe_path:
        The static stock-pool CSV.  ``None`` means the committed
        ``config/universe_seed.csv``.  A cache hit does not read it at all.
    calendar_name:
        Trading calendar that decides what "a session" is.  ``None`` means
        :data:`data_sys.market_calendar.DEFAULT_CALENDAR_NAME`.
    force:
        Recompute even when the key is already stored, replacing that file
        atomically.  ``False`` (the default) reads a stored panel back instead.

    Returns
    -------
    FactorRunOutcome
        The panel, its statistics, the quality report (``None`` when clean), the
        file that panel lives in and ``cache_status`` (``"hit"``, ``"miss"`` or
        ``"forced"``).  For a hit, that file already existed and still holds the
        panel that was read.

    Raises
    ------
    factor_sys.FactorError
        Any domain failure: an unknown factor
        (:class:`~factor_sys.errors.FactorRegistryError`), an unreadable or empty
        universe (:class:`~factor_sys.errors.FactorContractError`), a missing or
        unusable dataset (:class:`~factor_sys.errors.MarketDataReadError`), an
        unknown calendar, a date that is not a session or a factor that raised
        (:class:`~factor_sys.errors.FactorRunError`), or a failed write or an
        unreadable stored panel (:class:`~factor_sys.errors.FactorStoreError`).
        Nothing is written unless the run itself succeeded, and nothing is
        overwritten to hide a panel that cannot be read.
    RunFactorError
        If ``data_root`` is blank or of a type no path can be made from.
    """
    config = _pipeline_config(data_root)

    # (1)-(2): the factor is created *by name* -- this file never imports a
    # formula, so adding one is a change to the registry alone.
    factor = default_factor_registry().create(factor_name)
    spec = factor.spec

    # (3): a day the calendar actually trades.  The calendar is resolved before
    # the store is asked anything, so a weekend, a holiday or an unknown calendar
    # can never be "satisfied" by a file that happens to carry that date.
    day = _decision_day(as_of_date)
    calendar_label, calendar = _trading_calendar(
        DEFAULT_CALENDAR_NAME if calendar_name is None else calendar_name
    )
    _require_session(calendar_label, calendar, day)

    # (4)-(5): one key, one stored file -- read it back instead of recomputing it.
    store = ParquetFactorStore(_panels_root(config))
    if not force and store.exists(spec.name, spec.version, day):
        panel = store.read(spec.name, spec.version, day)
        return FactorRunOutcome(
            panel=panel,
            statistics=QuickAnalyzer().summarize(panel),
            quality_report=DefaultFactorQualityChecker().inspect(panel),
            stored_panel_path=store.path_for(spec.name, spec.version, day),
            cache_status="hit",
        )

    # (6)-(8): the replaceable parts, each through its own public constructor.
    universe_provider = StaticUniverseProvider(universe_path)
    market_data_reader = ParquetMarketDataReader(config.standardized_dir)
    runner = FactorRunner(
        universe_provider,
        market_data_reader,
        calendar_name=calendar_label,
    )

    # (9): one validated panel, plus the soft findings about it.
    run = runner.run(factor, as_of_date)

    # (10): written before the summary exists, so a warning never costs the panel
    # and a printing problem can never leave an undescribed run on disk.
    stored_panel_path = store.write(run.panel)

    # (11): the description is derived from the panel's own published rows.
    statistics = QuickAnalyzer().summarize(run.panel)

    # (12) happens in ``main``: this function answers with data, never with text.
    return FactorRunOutcome(
        panel=run.panel,
        statistics=statistics,
        quality_report=run.quality_report,
        stored_panel_path=stored_panel_path,
        cache_status="forced" if force else "miss",
    )


def _decision_day(value: date | str) -> date:
    """The decision date as a plain :class:`datetime.date`, or a clear refusal.

    A ``datetime`` is day-normalized (its time of day is meaningless for a daily
    cross-section) and an ISO-8601 string must name a real day, so ``2024-02-30``
    and ``31/12/2024`` are refused here instead of being passed on as a key.
    """
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return datetime.fromisoformat(value.strip()).date()
        except ValueError:
            pass
    raise FactorRunError(
        f"as_of_date must be a date or an ISO-8601 'YYYY-MM-DD' string, got {value!r}"
    )


def _trading_calendar(calendar_name: Any) -> tuple[str, Any]:
    """Resolve ``calendar_name`` once: ``(name, calendar)`` or a clear failure.

    The calendar belongs to the data system, so it is asked for by name rather
    than rebuilt here; only the failure is translated, so an unknown calendar is
    one readable ``error: ...`` line instead of a data-pipeline exception.
    """
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


def _require_session(calendar_label: str, calendar: Any, day: date) -> None:
    """Refuse a decision date the calendar does not trade.

    The check happens *before* the store is consulted: a stored file whose name
    happens to carry that date is not evidence that the exchange was open, so a
    weekend, a holiday or a day outside the calendar's own range always fails.
    """
    first, last = calendar_bounds(calendar)
    if day < first or day > last:
        raise FactorRunError(
            f"{day.isoformat()} lies outside the {calendar_label} calendar "
            f"({first.isoformat()}..{last.isoformat()}): the decision date must be a "
            "session the calendar actually covers, and a stored panel is never moved "
            "to a neighbouring day"
        )
    if not is_session(calendar, day):
        raise FactorRunError(
            f"{day.isoformat()} is not a {calendar_label} trading session (a weekend or "
            "an exchange holiday): the decision date must be a real session, and a "
            "stored panel is never reused for a day the calendar does not trade"
        )


def _pipeline_config(data_root: Path | str | None) -> PipelineConfig:
    """The layout every path of this run is derived from."""
    if data_root is None:
        return PipelineConfig()
    if isinstance(data_root, str) and not data_root.strip():
        raise RunFactorError(
            "--data-root must not be blank: pass a directory, or omit it to read and "
            f"write the project's default {PipelineConfig().data_root}"
        )
    if not isinstance(data_root, (str, Path)):
        raise RunFactorError(
            f"data_root must be a path or a string, got {type(data_root).__name__}"
        )
    return PipelineConfig(data_root=Path(data_root))


def _panels_root(config: PipelineConfig) -> Path:
    """Where the factor store lives for this run's data root."""
    return config.data_root / "factors" / "panels"


# ---------------------------------------------------------------------------
# The summary -- the only rendering of an outcome
# ---------------------------------------------------------------------------
def format_summary(outcome: FactorRunOutcome) -> str:
    """Render ``outcome`` as the command's stable, one-field-per-line summary.

    The text is a reporting surface, never a storage format: every number it
    prints is already in ``outcome.statistics`` at full precision, and the four
    fields that can be ``None`` (an empty usable sample cannot be described) are
    printed as :data:`MISSING_TEXT` rather than invented as ``0``.

    The first sixteen lines are always present and always in this order, so a
    caller may parse them by prefix; ``cache_status`` is the last of them and says
    whether the panel was read back, computed or recomputed.  Issue lines follow
    **only** when the quality checker found something: four lines per issue, in the
    report's own order, and nothing at all for a clean panel.

    Parameters
    ----------
    outcome:
        The run to render.

    Returns
    -------
    str
        The summary, without a trailing newline (``print`` adds one).
    """
    statistics = outcome.statistics
    lines = [
        f"factor_name: {statistics.factor_name}",
        f"factor_version: {statistics.factor_version}",
        f"as_of_date: {statistics.as_of_date.isoformat()}",
        f"universe_size: {statistics.universe_size}",
        f"eligible_count: {statistics.eligible_count}",
        f"eligible_coverage: {_number(statistics.eligible_coverage)}",
        f"valid_count: {statistics.valid_count}",
        f"missing_count: {statistics.missing_count}",
        f"missing_rate: {_number(statistics.missing_rate)}",
        f"mean: {_number(statistics.mean)}",
        f"std: {_number(statistics.std)}",
        f"minimum: {_number(statistics.minimum)}",
        f"maximum: {_number(statistics.maximum)}",
        f"stored_panel_path: {outcome.stored_panel_path}",
        f"quality_status: {outcome.quality_status}",
        f"cache_status: {outcome.cache_status}",
    ]

    for issue in outcome.issues:
        lines.extend(
            [
                f"quality_issue_code: {issue.code}",
                f"quality_issue_severity: {issue.severity}",
                f"quality_issue_message: {issue.message}",
                f"quality_issue_affected_symbols: {', '.join(issue.affected_symbols)}",
            ]
        )

    return "\n".join(lines)


def _number(value: float | None) -> str:
    """``value`` to :data:`DECIMALS` decimals, or :data:`MISSING_TEXT` for ``None``."""
    return MISSING_TEXT if value is None else f"{value:.{DECIMALS}f}"


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """The argument parser of ``python scripts/run_factor.py``.

    One factor, one date: every flag below chooses *which* factor, *which*
    session, *where* the local data lives, *which* calendar decides a session or
    whether a stored panel may be reused.  There is deliberately no range, no
    batch, no horizon and no cache-management flag; a missing or unknown flag is
    argparse's own exit code ``2``.
    """
    return _parser_for(PipelineConfig().data_root)


def _parser_for(default_data_root: Path) -> argparse.ArgumentParser:
    """Build the parser, naming the defaults the run will actually use."""
    parser = argparse.ArgumentParser(
        prog="run_factor.py",
        description=(
            "Compute one factor for one decision date from the local standardized "
            "bars, store the panel and print its cross-section statistics."
        ),
    )
    parser.add_argument(
        "--factor",
        required=True,
        metavar="FACTOR_NAME",
        help="registered factor name, e.g. momentum_12_1 (resolved through the registry)",
    )
    parser.add_argument(
        "--as-of",
        dest="as_of",
        required=True,
        metavar="YYYY-MM-DD",
        help="decision date; it must be a session of the trading calendar",
    )
    parser.add_argument(
        "--data-root",
        dest="data_root",
        metavar="PATH",
        default=None,
        help=(
            "root holding standardized/market_bars and receiving factors/panels "
            f"(default: {default_data_root})"
        ),
    )
    parser.add_argument(
        "--universe-path",
        dest="universe_path",
        metavar="PATH",
        default=None,
        help="static stock-pool CSV (default: the committed config/universe_seed.csv)",
    )
    parser.add_argument(
        "--calendar",
        metavar="NAME",
        default=None,
        help=f"trading calendar that defines a session (default: {DEFAULT_CALENDAR_NAME})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "recompute even when (factor, version, date) is already stored, replacing "
            "that panel atomically (default: read the stored panel back)"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse ``argv``, run the factor, print the summary and return the exit code.

    Parameters
    ----------
    argv:
        Command-line arguments without the program name (``None`` means
        ``sys.argv[1:]``).  Passing a list lets a test drive the command exactly
        as a shell would, without starting a second process.

    Returns
    -------
    int
        ``0`` when the panel was computed or read back, and printed; ``1`` when a
        :class:`factor_sys.errors.FactorError` (or this module's
        :class:`RunFactorError`) stopped the run.  That message goes to stderr as
        one ``error: ...`` line, never as a traceback, and nothing is written (or
        overwritten) when the run itself failed.
    """
    args = build_parser().parse_args(argv)

    try:
        outcome = run_factor(
            args.factor,
            args.as_of,
            data_root=args.data_root,
            universe_path=args.universe_path,
            calendar_name=args.calendar,
            force=args.force,
        )
    except FactorError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(format_summary(outcome))
    return 0


__all__ = [
    "CACHE_STATUSES",
    "FactorRunOutcome",
    "RunFactorError",
    "build_parser",
    "format_summary",
    "main",
    "run_factor",
]


if __name__ == "__main__":
    raise SystemExit(main())
