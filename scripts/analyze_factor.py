"""Measure one factor across a schedule of decision dates: panels, labels, ranks.

What this script is
-------------------
The research face of the multi-date entry point.  It answers one question -- *did
this factor's cross-section rank the returns that followed it?* -- over exactly
the dates a cadence selects, and writes the answer down::

     1  run_factor_batch(...)          every selected panel: hit, computed or forced
     2  ParquetMarketDataReader        bars for the label window, and nothing later
     3  ForwardReturnCalculator        one forward-return panel per factor panel
     4  QuickAnalyzer                  counts, coverage, missing rate, moments
     5  ICAnalyzer                     Spearman IC per date, mean, std, ICIR
     6  QuantileAnalyzer               mean forward return per group, top - bottom
     7  the atomic report publish      one new directory: metadata + five CSVs

Step 1 is the *same function* the batch entry point exposes, so the schedule
rule, the panel cache, the per-date failure isolation and the batch manifest are
the ones that already exist.  Steps 4-6 are the *same analyzers* the package
exposes, so the Spearman ranking, the ICIR, the tie handling, the grouping and
the spread are computed once, in one place, for every caller.  Nothing here
computes a factor value, reads a stored panel by path, ranks a cross-section or
writes a label: this module only decides *which* dates, *which* horizon and
*where* the report goes.

Cadence and horizon are two different things
--------------------------------------------
``--frequency`` / ``--anchor-weekday`` decide *where* cross-sections are formed::

    weekly    one session per week (the anchor, or the session before it when
              the exchange is closed); ``biweekly`` keeps every second of those,
              ``monthly`` takes the last session of each month and has no anchor

``--horizon-sessions`` decides *how far forward* every one of those dates is
measured, counted in exchange sessions, never in calendar days::

    5     roughly one week ahead
    10    roughly two weeks ahead
    20    roughly one month ahead

The two are independent on purpose.  ``weekly + THU + 20`` means "a cross-section
forms on every Thursday's close, and each one is tested against the return of the
next twenty sessions"; forcing ``weekly`` to mean five sessions would answer a
different question than the one asked.  A long horizon against a short cadence
also means the label windows **overlap**, so consecutive ICs are not independent
observations.

Only this run's dates are measured
----------------------------------
``ParquetFactorStore.read_range`` would hand back every panel the directory
happens to hold: Wednesdays, Thursdays, biweekly, month-end and one-off manual
dates, all mixed into one IC series whose average describes no sampling scheme
at all.  So the dates come from :func:`scripts.run_factor_batch.run_factor_batch`
and the panels are taken from ``FactorBatchOutcome.outcomes`` -- nothing else is
read.  The batch already answers the three cases that follow::

    an existing panel for this factor/version/date   -> hit, read back
    a missing panel                                  -> computed and stored
    --force                                          -> recomputed, replaced

Labels are research labels
--------------------------
The forward return is
:class:`factor_sys.forward_returns.ForwardReturnCalculator`'s own
``adjusted_close(t + horizon) / adjusted_close(t) - 1``: a decision close to a
future close, on split- and dividend-adjusted prices.  It is **not** the return
a backtest would earn, because a backtest fills at an open, pays a spread and a
commission, is constrained by liquidity and turnover, and sizes positions.  None
of that exists here.

A missing price is a fact, not a failure: a symbol without a usable close at
either end gets ``is_available=False``, a date whose future window is not
published yet keeps its row with a low or zero ``pair_coverage`` and ``ic=None``,
and nothing is filled, interpolated, carried forward or dropped silently.

What is refused, and how
------------------------
The analysis measures nothing at all unless the batch delivered *every* selected
date.  A failed date means the batch manifest records a hole, and analysing the
survivors would describe a sampling scheme nobody chose; the command stops with
:class:`~factor_sys.errors.FactorResearchError` naming the failure count and the
manifest path, and writes no report.  A selection with no date is refused for the
same reason (there is nothing to measure), *before* the analyzers can complain
about an empty sequence.  A hard refusal of the label, IC or quantile layer keeps
its own error type, so "the data was missing" stays distinguishable from "these
objects do not belong together".

The same question is asked once more when the result object is built: the
concluding :class:`FactorResearchOutcome` reconciles the batch *record* against
the sample that was measured -- the selected dates, the date of every successful
outcome and the panels of ``factor_panels`` must be the same dates, in the same
order, and the *same objects*, so a panel that merely looks like the batch's
panel (same date, factor, version and numbers, built somewhere else) is refused.
Nothing is recomputed there either; it only proves that the metadata, the
manifest, the IC series and the quantile returns all point at one set of data.

Exit codes
----------
``0``
    every selected date succeeded, the three analyses ran and the report was
    published completely.
``1``
    a date failed, a label / IC / quantile layer refused its input, or the report
    could not be written; the message goes to stderr as one ``error: ...`` line,
    never as a traceback.  A quality finding is **not** a failure: it shows up as
    coverage and pair counts inside the report.
``2``
    argparse refused the command line: a missing or unknown flag, an unknown
    ``--frequency`` / ``--anchor-weekday``, a non-positive ``--horizon-sessions``,
    a ``--quantiles`` below two, a ``--min-pairs`` below two, or
    ``--anchor-weekday`` combined with ``--frequency monthly``.

The report
----------
Every successful analysis publishes a *new* directory under
``<data_root>/factors/research_runs/<research_run_id>/``::

    metadata.json               the request, the dates and the IC summary
    panel_statistics.csv        QuickAnalyzer.to_frame(...)
    forward_returns.csv         every symbol, every date, availability included
    ic_observations.csv         ICReport.to_frame()
    quantile_observations.csv   QuantileReport.to_frame()
    quantile_spreads.csv        QuantileReport.spread_to_frame()

The directory is a *record of one analysis*, never a cache key: the same request
run twice produces two independent reports.  It is written into a unique staging
directory first and published with one :func:`os.replace`, so a failure leaves the
previous reports exactly as they were and no half-finished directory behind.  The
panels and the batch manifest of step 1 are already on disk by then and are never
rolled back.

What this script is not
-----------------------
It places no order, holds no position and computes no net value; it has no open
fill, cost, slippage or turnover model; it fits no regression, reports no R²,
t-statistic or Newey-West error; it combines no factors and defines no new
formula; it prints no per-symbol detail; and it creates no permanent
forward-return panel store.  The IC and the quantile spread are *research
statistics*, not a strategy: with overlapping windows they are a first look, not
a significance test, and a full report is still several steps away from a
backtest.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

# Running this file directly puts ``scripts/`` (not the project root) on
# ``sys.path``; add the root so ``python scripts/analyze_factor.py`` works from a
# checkout, with or without the package installed in the environment.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from data_sys.config import PipelineConfig  # noqa: E402
from data_sys.utils import new_run_id, utc_now_iso  # noqa: E402
from factor_sys import (  # noqa: E402
    ANCHOR_WEEKDAYS,
    DEFAULT_FREQUENCY,
    FORWARD_RETURN_COLUMNS,
    FREQUENCIES,
    MONTHLY,
    FactorError,
    FactorPanel,
    FactorPanelStatistics,
    ForwardReturnCalculator,
    ForwardReturnPanel,
    ICAnalyzer,
    ICReport,
    ParquetMarketDataReader,
    QuantileAnalyzer,
    QuantileReport,
    QuickAnalyzer,
)
from factor_sys.errors import FactorResearchError  # noqa: E402
from scripts.run_factor_batch import FactorBatchOutcome, run_factor_batch  # noqa: E402

#: Printed for a field that does not apply (a monthly anchor, an undefined ICIR).
MISSING_TEXT = "n/a"

#: Version of the report layout this script writes.
REPORT_FORMAT_VERSION = 1

#: Directory of research reports, relative to the run's data root.
RESEARCH_DIRECTORY: tuple[str, ...] = ("factors", "research_runs")

#: Groups a cross-section is cut into when ``--quantiles`` is not given.  It is
#: the :class:`QuantileAnalyzer` default, named here so the CLI can print it.
DEFAULT_QUANTILES = 5

#: Decimals used by the summary block (the report keeps full precision).
DECIMALS = 6

#: The metadata file of one report.
METADATA_FILE = "metadata.json"

#: The five tables of one report: the key used by this module, and the file name
#: written for it.  This order is the order they are written in.
TABLE_FILES: Mapping[str, str] = {
    "panel_statistics": "panel_statistics.csv",
    "forward_returns": "forward_returns.csv",
    "ic_observations": "ic_observations.csv",
    "quantile_observations": "quantile_observations.csv",
    "quantile_spreads": "quantile_spreads.csv",
}


# ---------------------------------------------------------------------------
# The request: what one research run may ask for
# ---------------------------------------------------------------------------
def _session_count(value: Any, label: str) -> int:
    """``value`` as a positive number of exchange sessions, or a clear refusal.

    ``bool`` is refused even though it is an ``int``: ``True`` sessions would
    silently mean one.  The horizon is counted in sessions -- never in calendar
    days -- because the exchange, not a 24-hour clock, decides what "five days
    later" means.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise FactorResearchError(
            f"{label} must be a whole number of sessions, got {value!r} "
            f"({type(value).__name__}): a horizon is counted in exchange sessions"
        )
    if value < 1:
        raise FactorResearchError(
            f"{label} must be at least 1 session, got {value}: a label needs a forward window"
        )
    return value


def _group_count(value: Any, label: str) -> int:
    """``value`` as a number of quantile groups of at least two, or a refusal."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise FactorResearchError(
            f"{label} must be a whole number of groups, got {value!r} ({type(value).__name__})"
        )
    if value < 2:
        raise FactorResearchError(
            f"{label} must be at least 2, got {value}: a top-minus-bottom spread needs two groups"
        )
    return value


def _pair_count(value: Any, label: str) -> int:
    """``value`` as a usable-pair threshold of at least two, or a refusal."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise FactorResearchError(
            f"{label} must be a whole number of pairs, got {value!r} ({type(value).__name__})"
        )
    if value < 2:
        raise FactorResearchError(
            f"{label} must be at least 2, got {value}: a rank correlation needs two pairs"
        )
    return value


def _unusable_path(value: Any, label: str) -> None:
    """Refuse a blank or non-path value for one path argument."""
    if isinstance(value, str) and not value.strip():
        raise FactorResearchError(f"{label} must not be blank: pass a directory, or omit it")
    if not isinstance(value, (str, Path)):
        raise FactorResearchError(f"{label} must be a path or a string, got {type(value).__name__}")


def _data_root(value: Path | str | None) -> Path:
    """The root every path of this analysis is derived from, or a clear refusal.

    ``None`` means the project's own :class:`PipelineConfig` default, so nothing
    here embeds a machine-specific directory.  The value is validated once,
    before the batch runs, because the report root and the label reader are both
    derived from it.
    """
    if value is None:
        return PipelineConfig().data_root
    _unusable_path(value, "data_root")
    return PipelineConfig(data_root=Path(value)).data_root


def _report_root(value: Path | str | None, data_root: Path) -> Path:
    """Where this analysis publishes its report directory.

    ``None`` derives ``<data_root>/factors/research_runs`` from the run's own
    data root, so the default location follows ``--data-root`` instead of being
    hard-coded anywhere.
    """
    if value is None:
        return data_root.joinpath(*RESEARCH_DIRECTORY)
    _unusable_path(value, "report_root")
    return Path(value)


# ---------------------------------------------------------------------------
# What one analysis produced
# ---------------------------------------------------------------------------
def _objects(value: Any, kind: type, label: str, problems: list[str]) -> tuple[Any, ...]:
    """``value`` as a tuple of ``kind`` objects, recording every offender."""
    if not isinstance(value, tuple):
        problems.append(
            f"{label} must be a tuple of {kind.__name__}, got {type(value).__name__}: "
            "the result of one run is a tuple, never a list or None"
        )
        return ()
    collected: list[Any] = []
    for position, item in enumerate(value):
        if not isinstance(item, kind):
            problems.append(
                f"{label}[{position}] is a {type(item).__name__}, not a {kind.__name__}"
            )
            continue
        collected.append(item)
    return tuple(collected)


def _reconcile_batch(
    batch: FactorBatchOutcome,
    panels: tuple[FactorPanel, ...],
    problems: list[str],
) -> None:
    """Cross-check the batch record against the panels the analysis measured.

    Nothing here is recomputed: every date and every panel is read off an
    object that already exists, so this step can only fail when the parts were
    never one analysis.  It is the only place that ties the *record* of the run
    (``selected_dates``, the per-date outcomes, the manifest) to the *sample*
    that was actually measured, so a reader of the report can trust that the
    metadata, the IC series and the quantile returns describe exactly the same
    dates:

    * the batch recorded **no failed date** -- an analysis measures a sample,
      and a hole in the sample is not a smaller sample;
    * the batch **wrote its manifest** -- the report's metadata names that
      manifest, so an unwritten record would leave the run untraceable;
    * the successful outcomes are the selected dates, one each, same order --
      a selected date without a panel, or a panel for a date that was never
      selected, breaks the chain;
    * ``panels`` are those outcomes' panels by **position, date and identity**.
      A panel that merely *looks* the same -- same factor, same version, same
      numbers, built here instead of read from the batch -- is a different
      sample and is refused;
    * the batch's factor identity is the panels' identity, because
      :attr:`FactorResearchOutcome.factor_name` is read from the batch and would
      otherwise relabel the report.

    In short::

        selected_dates == [o.panel.as_of_date for o in outcomes]
                       == [p.as_of_date for p in factor_panels]

    Parameters
    ----------
    batch:
        The :class:`~scripts.run_factor_batch.FactorBatchOutcome` carried by the
        outcome; already known to be one.
    panels:
        The validated ``factor_panels`` of the outcome.
    problems:
        The refusal messages collected so far; appended to, never raised here.
    """
    recorded = tuple(item.panel.as_of_date for item in batch.outcomes)

    if batch.manifest_error is not None:
        problems.append(
            "batch_outcome records that its manifest could not be written "
            f"({batch.manifest_error}): the report's metadata points at that manifest, so "
            "publishing it would leave the run untraceable"
        )

    if batch.failures:
        shown = ", ".join(failure.as_of_date.isoformat() for failure in batch.failures[:5])
        more = f" (and {batch.failed_count - 5} more)" if batch.failed_count > 5 else ""
        problems.append(
            f"batch_outcome records {batch.failed_count} failed date(s) -- {shown}{more} -- "
            "for an analysis it published: a report covers every selected date or none"
        )

    if len(recorded) != batch.selection_size:
        problems.append(
            f"batch_outcome holds {len(recorded)} successful outcome(s) for "
            f"{batch.selection_size} selected date(s): every selected date needs one panel"
        )

    if len(panels) != len(recorded):
        problems.append(
            f"factor_panels holds {len(panels)} panel(s) for {len(recorded)} successful date(s) "
            "of the batch: the analysis measures exactly the batch's panels"
        )

    for position, day in enumerate(batch.selected_dates):
        if position >= len(recorded):
            break
        if recorded[position] != day:
            problems.append(
                f"batch_outcome.outcomes[{position}] is dated "
                f"{recorded[position].isoformat()}, but selected_dates[{position}] is "
                f"{day.isoformat()}: a batch and its own schedule must agree"
            )

    for position, outcome in enumerate(batch.outcomes):
        if position >= len(panels):
            break
        panel = panels[position]
        if panel is not outcome.panel:
            problems.append(
                f"factor_panels[{position}] is not the panel the batch recorded for "
                f"{outcome.panel.as_of_date.isoformat()}: a second panel with the same date, "
                "factor and version is a different sample, and the report would no longer be "
                "traceable to this batch"
            )
        if panel.as_of_date != outcome.panel.as_of_date:
            problems.append(
                f"factor_panels[{position}] is dated {panel.as_of_date.isoformat()}, but the "
                f"batch recorded {outcome.panel.as_of_date.isoformat()} at that position: the "
                "dates must pair up one to one"
            )

    if panels and (panels[0].factor_name, panels[0].factor_version) != (
        batch.factor_name,
        batch.factor_version,
    ):
        problems.append(
            f"the batch measured {batch.factor_name}/{batch.factor_version}, but the panels "
            f"are {panels[0].factor_name}/{panels[0].factor_version}: the report takes its "
            "factor name from the batch record"
        )


@dataclass(frozen=True)
class FactorResearchOutcome:
    """One research analysis: the batch, its labels, the three reports and where.

    Immutable and self-checking, like every other result object in the project.
    It carries the objects the analysis really produced, not a rendering of
    them:

    ``batch_outcome``
        The untouched :class:`scripts.run_factor_batch.FactorBatchOutcome` of
        stage A -- the schedule that was used, the per-date cache status, the
        failures (there are none in a published report) and the manifest path.
    ``factor_panels``
        The ``panel`` of every successful outcome, in schedule order.  Panels of
        another run are never mixed in: this tuple *is* the analysed sample.
    ``forward_return_panels``
        One label panel per factor panel, in the same order, for the same date,
        the same factor identity and the same universe order.
    ``panel_statistics``
        ``QuickAnalyzer.summarize_history(...)``: counts, coverage, missing rate
        and the four moments, one entry per date.
    ``ic_report``
        ``ICAnalyzer.analyze(...)``: the Spearman IC of every date (``None``
        where it is undefined), plus the mean, the sample deviation and the ICIR.
    ``quantile_report``
        ``QuantileAnalyzer.analyze(...)``: the mean forward return of every group
        of every date, and the top-minus-bottom spread.
    ``min_pairs``
        The usable-pair threshold the IC analyzer actually used, kept because it
        is a policy knob of *this* run and cannot be recovered from the report.
    ``report_path``
        The published directory of this analysis: one new directory per run,
        holding ``metadata.json`` and the five tables.

    Raises
    ------
    factor_sys.errors.FactorResearchError
        If the parts do not describe one analysis: a foreign object, a label or a
        statistics entry that belongs to another date, a report about another
        factor, version or horizon, a count that disagrees with the panels, or a
        batch record that is not the source of the panels -- see
        :func:`_reconcile_batch`, which refuses a batch with a failed date or an
        unwritten manifest, a selected date without its panel, and a panel that
        merely looks like the batch's panel instead of *being* it.

    Notes
    -----
    Nothing here is recomputed: the numbers live in the reports the analyzers
    returned, and :func:`format_summary` is the only rendering of them.  The
    construction is also the last chance to notice that the parts came from
    different runs, which is why the batch is checked against the panels and not
    merely stored next to them.
    """

    batch_outcome: FactorBatchOutcome
    factor_panels: tuple[FactorPanel, ...]
    forward_return_panels: tuple[ForwardReturnPanel, ...]
    panel_statistics: tuple[FactorPanelStatistics, ...]
    ic_report: ICReport
    quantile_report: QuantileReport
    min_pairs: int
    report_path: Path

    def __post_init__(self) -> None:
        """Refuse an outcome whose parts do not describe one analysis."""
        problems: list[str] = []

        if not isinstance(self.batch_outcome, FactorBatchOutcome):
            problems.append(
                "batch_outcome must be the FactorBatchOutcome of this run, got "
                f"{type(self.batch_outcome).__name__}: the analysis never re-derives the schedule"
            )

        panels = _objects(self.factor_panels, FactorPanel, "factor_panels", problems)
        labels = _objects(
            self.forward_return_panels, ForwardReturnPanel, "forward_return_panels", problems
        )
        statistics = _objects(
            self.panel_statistics, FactorPanelStatistics, "panel_statistics", problems
        )

        ic: ICReport | None = None
        if not isinstance(self.ic_report, ICReport):
            problems.append(f"ic_report must be an ICReport, got {type(self.ic_report).__name__}")
        else:
            ic = self.ic_report

        groups: QuantileReport | None = None
        if not isinstance(self.quantile_report, QuantileReport):
            problems.append(
                "quantile_report must be a QuantileReport, got "
                f"{type(self.quantile_report).__name__}"
            )
        else:
            groups = self.quantile_report

        if isinstance(self.min_pairs, bool) or not isinstance(self.min_pairs, int):
            problems.append(
                f"min_pairs must be a whole number of pairs, got {self.min_pairs!r} "
                f"({type(self.min_pairs).__name__})"
            )
        elif self.min_pairs < 2:
            problems.append(f"min_pairs must be at least 2, got {self.min_pairs}")

        if not isinstance(self.report_path, Path):
            problems.append(
                f"report_path must be a pathlib.Path, got {type(self.report_path).__name__}"
            )

        if problems:
            self._reject(problems)

        # -- the three sequences must describe the same sessions ---------------
        if not panels:
            problems.append(
                "factor_panels must hold at least one panel: an analysis of no date measures "
                "nothing"
            )
        if len(labels) != len(panels):
            problems.append(
                f"forward_return_panels holds {len(labels)} panel(s) for {len(panels)} factor "
                "panel(s): every panel needs exactly one set of labels"
            )
        if len(statistics) != len(panels):
            problems.append(
                f"panel_statistics holds {len(statistics)} entrie(s) for {len(panels)} factor "
                "panel(s): every panel is described exactly once"
            )

        identity = (panels[0].factor_name, panels[0].factor_version) if panels else None
        horizon = labels[0].horizon_sessions if labels else None
        for position, panel in enumerate(panels):
            current = (panel.factor_name, panel.factor_version)
            if identity is not None and current != identity:
                problems.append(
                    f"factor_panels[{position}] is {current[0]}/{current[1]}, but this analysis "
                    f"is {identity[0]}/{identity[1]}: one report measures one factor at one version"
                )
            if position and panel.as_of_date <= panels[position - 1].as_of_date:
                problems.append(
                    f"factor_panels[{position}] is dated {panel.as_of_date.isoformat()}, which "
                    f"does not follow {panels[position - 1].as_of_date.isoformat()}: the analysis "
                    "follows the schedule in strictly ascending order"
                )

        for position, label in enumerate(labels):
            if position >= len(panels):
                break
            panel = panels[position]
            if label.as_of_date != panel.as_of_date:
                problems.append(
                    f"forward_return_panels[{position}] is labelled for "
                    f"{label.as_of_date.isoformat()}, not the panel's "
                    f"{panel.as_of_date.isoformat()}"
                )
            if (label.factor_name, label.factor_version) != (
                panel.factor_name,
                panel.factor_version,
            ):
                problems.append(
                    f"forward_return_panels[{position}] labels {label.factor_name}/"
                    f"{label.factor_version}, not {panel.factor_name}/{panel.factor_version}"
                )
            if label.universe != panel.universe:
                problems.append(
                    f"forward_return_panels[{position}] covers symbols in another order than the "
                    f"panel's {len(panel.universe)}: a label is built from its own panel's universe"
                )
            if label.horizon_sessions != horizon:
                problems.append(
                    f"forward_return_panels[{position}] measures {label.horizon_sessions} "
                    f"session(s) ahead, but the first label measured {horizon}: one analysis "
                    "measures one horizon"
                )

        for position, described in enumerate(statistics):
            if position >= len(panels):
                break
            panel = panels[position]
            if (described.as_of_date, described.factor_name, described.factor_version) != (
                panel.as_of_date,
                panel.factor_name,
                panel.factor_version,
            ):
                problems.append(
                    f"panel_statistics[{position}] describes {described.factor_name!r} v"
                    f"{described.factor_version} on {described.as_of_date.isoformat()}, not "
                    f"{panel.factor_name!r} v{panel.factor_version} on "
                    f"{panel.as_of_date.isoformat()}"
                )

        # -- the two reports must agree with the panels they summarize ---------
        if ic is not None and identity is not None:
            if (ic.factor_name, ic.factor_version) != identity:
                problems.append(
                    f"ic_report measures {ic.factor_name}/{ic.factor_version}, not "
                    f"{identity[0]}/{identity[1]}"
                )
            if horizon is not None and ic.horizon_sessions != horizon:
                problems.append(
                    f"ic_report measures {ic.horizon_sessions} session(s) ahead, but the labels "
                    f"measure {horizon}"
                )
            if len(ic.observations) != len(panels):
                problems.append(
                    f"ic_report holds {len(ic.observations)} observation(s) for {len(panels)} "
                    "panel(s): every date is measured once, including the undefined ones"
                )

        if groups is not None and identity is not None:
            if (groups.factor_name, groups.factor_version) != identity:
                problems.append(
                    f"quantile_report measures {groups.factor_name}/{groups.factor_version}, not "
                    f"{identity[0]}/{identity[1]}"
                )
            if horizon is not None and groups.horizon_sessions != horizon:
                problems.append(
                    f"quantile_report measures {groups.horizon_sessions} session(s) ahead, but the "
                    f"labels measure {horizon}"
                )
            if len(groups.spreads) != len(panels):
                problems.append(
                    f"quantile_report holds {len(groups.spreads)} spread(s) for {len(panels)} "
                    "panel(s): every date contributes one top-minus-bottom spread"
                )

        # -- the batch record and the measured panels are the same sample ------
        _reconcile_batch(self.batch_outcome, panels, problems)

        if problems:
            self._reject(problems)

    def _reject(self, problems: list[str]) -> None:
        """Raise :class:`FactorResearchError` naming every problem found."""
        raise FactorResearchError(
            "the research outcome does not describe one analysis: " + "; ".join(problems)
        )

    # -- derived counts, so they cannot disagree with the panels ---------------
    @property
    def selected_panel_count(self) -> int:
        """How many decision dates this analysis measured."""
        return len(self.factor_panels)

    @property
    def factor_name(self) -> str:
        """The factor every panel, label and report belongs to."""
        return self.batch_outcome.factor_name

    @property
    def factor_version(self) -> str:
        """The factor version every panel, label and report belongs to."""
        return self.batch_outcome.factor_version

    @property
    def horizon_sessions(self) -> int:
        """Sessions between a decision date and the end date of its label."""
        return self.ic_report.horizon_sessions

    @property
    def valid_ic_count(self) -> int:
        """How many dates had a defined IC."""
        return self.ic_report.valid_ic_count

    @property
    def mean_ic(self) -> float | None:
        """The mean of the defined ICs, or ``None`` when there is none."""
        return self.ic_report.mean_ic

    @property
    def ic_std(self) -> float | None:
        """The sample deviation (``ddof=1``) of the defined ICs, or ``None``."""
        return self.ic_report.ic_std

    @property
    def icir(self) -> float | None:
        """``mean_ic / ic_std``, or ``None`` when the deviation is undefined or zero."""
        return self.ic_report.icir

    def __len__(self) -> int:
        """How many decision dates this analysis measured."""
        return len(self.factor_panels)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(factor={self.factor_name}@{self.factor_version}, "
            f"dates={self.selected_panel_count}, horizon_sessions={self.horizon_sessions}, "
            f"valid_ic_count={self.valid_ic_count}, mean_ic={self.mean_ic!r}, "
            f"report_path={str(self.report_path)!r})"
        )


# ---------------------------------------------------------------------------
# Stage A: the batch must deliver every selected date
# ---------------------------------------------------------------------------
def _require_complete(batch: FactorBatchOutcome) -> None:
    """Refuse a batch that cannot support a complete analysis, and say why.

    Three states are refused, each with the evidence a reader needs:

    * a selection with **no** date -- there is nothing to measure, and letting
      ``ICAnalyzer`` answer "empty sequence" would describe a broken analysis
      rather than an empty request;
    * **any** failed date -- the analysis measures a sample, and a hole in the
      sample is not a smaller sample; the message names the count, the dates and
      the manifest that records them, and no report is written;
    * a manifest that could **not be written** -- the report's metadata points at
      this manifest, so publishing a report that references a missing record
      would make the run untraceable.
    """
    if not batch.selected_dates:
        raise FactorResearchError(
            f"the {batch.frequency} schedule of {batch.start_date.isoformat()}.."
            f"{batch.end_date.isoformat()} selected no decision date, so there is no "
            "cross-section to analyze: widen the window, or choose a cadence and anchor that "
            "select at least one session"
        )

    if batch.failures:
        shown = ", ".join(failure.as_of_date.isoformat() for failure in batch.failures[:5])
        more = f" (and {batch.failed_count - 5} more)" if batch.failed_count > 5 else ""
        raise FactorResearchError(
            f"{batch.failed_count} of {batch.selection_size} selected dates failed, so the "
            f"analysis would measure an incomplete history: {shown}{more} "
            f"(batch manifest: {batch.manifest_path})"
        )

    if batch.manifest_error is not None:
        raise FactorResearchError(
            "the batch manifest of this run could not be written, so the report would point at "
            f"a record that does not exist: {batch.manifest_error}"
        )

    if batch.succeeded_count != batch.selection_size:
        raise FactorResearchError(
            f"the batch recorded {batch.succeeded_count} panel(s) for "
            f"{batch.selection_size} selected date(s): every date needs exactly one outcome"
        )


# ---------------------------------------------------------------------------
# Stage B: one set of labels per panel
# ---------------------------------------------------------------------------
def _label_panels(
    panels: Sequence[FactorPanel],
    *,
    data_root: Path,
    calendar_name: str,
    horizon_sessions: int,
) -> tuple[ForwardReturnPanel, ...]:
    """Label every panel over ``horizon_sessions`` sessions, in panel order.

    One reader and one calculator are built for the whole analysis, so every
    date is measured against the same published bars and the same trading
    calendar -- the calendar the batch itself used, so a date the schedule
    selected is a date the labels can measure.

    A symbol without a usable price at either end, or a date whose future window
    is not published yet, comes back as an *unavailable* label: that is a fact
    about the data and stays in the report.  A hard error of the calculator (an
    impossible horizon, a reader that answers with something untrustworthy) keeps
    its own type and travels back unchanged.
    """
    reader = ParquetMarketDataReader(PipelineConfig(data_root=data_root).standardized_dir)
    calculator = ForwardReturnCalculator(reader, calendar_name=calendar_name)
    return tuple(calculator.compute(panel, horizon_sessions) for panel in panels)


def _forward_return_frame(labels: Sequence[ForwardReturnPanel]) -> pd.DataFrame:
    """Every label of this analysis as one table: one row per symbol and date.

    The frames are concatenated exactly as the objects publish them, so the
    availability flag and the horizon travel with every row and no per-symbol
    value is dropped, rounded or re-ordered on the way out.
    """
    if not labels:
        return pd.DataFrame(columns=list(FORWARD_RETURN_COLUMNS))
    return pd.concat([label.frame for label in labels], ignore_index=True)


# ---------------------------------------------------------------------------
# The report: one new directory, written completely or not at all
# ---------------------------------------------------------------------------
def report_metadata(
    outcome_batch: FactorBatchOutcome,
    *,
    research_run_id: str,
    created_at_utc: str,
    data_root: Path,
    horizon_sessions: int,
    quantiles: int,
    min_pairs: int,
    ic_report: ICReport,
) -> dict[str, Any]:
    """The JSON-safe record of one analysis.

    Everything a later reader needs is here: the window that was requested, the
    cadence and anchor that turned it into dates, the complete ``selected_dates``
    list the batch really used, the horizon and the two analysis knobs, the
    manifest of stage A and the IC summary.

    Two rules make the file publishable as-is: a path is stored **relative to
    ``data_root``** with ``/`` separators when it lives there (a report must stay
    meaningful after the tree moves to another machine), and a ``None`` of the
    domain objects becomes JSON ``null`` -- never ``NaN`` or ``Infinity``, which
    are not JSON and would break every strict reader.
    """
    return {
        "format_version": REPORT_FORMAT_VERSION,
        "research_run_id": research_run_id,
        "created_at_utc": created_at_utc,
        "factor_name": outcome_batch.factor_name,
        "factor_version": outcome_batch.factor_version,
        "requested_start_date": outcome_batch.start_date.isoformat(),
        "requested_end_date": outcome_batch.end_date.isoformat(),
        "frequency": outcome_batch.frequency,
        "anchor_weekday": outcome_batch.anchor_weekday,
        "biweekly_anchor": outcome_batch.biweekly_anchor,
        "calendar_name": outcome_batch.calendar_name,
        "horizon_sessions": horizon_sessions,
        "quantiles": quantiles,
        "min_pairs": min_pairs,
        "selected_dates": [day.isoformat() for day in outcome_batch.selected_dates],
        "batch_manifest_path": _portable(outcome_batch.manifest_path, data_root),
        "summary": {
            "panel_count": outcome_batch.succeeded_count,
            "valid_ic_count": ic_report.valid_ic_count,
            "mean_ic": _json_number(ic_report.mean_ic),
            "ic_std": _json_number(ic_report.ic_std),
            "icir": _json_number(ic_report.icir),
        },
    }


def _portable(path: Path, data_root: Path) -> str:
    """``path`` relative to ``data_root`` when it lives there, otherwise verbatim."""
    try:
        return path.relative_to(data_root).as_posix()
    except ValueError:
        return path.as_posix()


def _json_number(value: float | None) -> float | None:
    """``value`` as a JSON number, or ``None`` when it is missing or not finite.

    The domain objects already answer ``None`` for "undefined", so this is the
    last guard before ``json.dump``: a non-finite float would be written as
    ``NaN``/``Infinity`` -- invalid JSON -- by default, and a report nobody can
    parse is worse than a missing number.
    """
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def write_report(
    report_root: Path,
    research_run_id: str,
    metadata: Mapping[str, Any],
    tables: Mapping[str, pd.DataFrame],
) -> Path:
    """Publish one research report under ``report_root``, completely or not at all.

    The five steps of an atomic publish, in this order:

    1. the report root is created (a missing root is a failure, not a silent
       no-op -- the caller asked for a report);
    2. a **unique staging directory** is created inside it, so the final rename
       stays a rename within one filesystem and two concurrent runs can never
       write into the same place;
    3. every table and then the metadata are written there, the metadata last;
    4. the directory is checked for existence and non-emptiness;
    5. one :func:`os.replace` moves it to ``<report_root>/<research_run_id>``.

    Any failure removes the staging directory and raises
    :class:`~factor_sys.errors.FactorResearchError`: no half-finished directory is
    ever visible under its final name, and the reports of earlier runs are
    untouched.  The panels and the batch manifest of this run were written by an
    earlier stage and are *not* rolled back -- they are independent results.

    Parameters
    ----------
    report_root:
        Directory holding research reports, normally
        ``<data_root>/factors/research_runs``.
    research_run_id:
        Name of the new directory.  It must be unique; an existing directory of
        that name is refused rather than overwritten.
    metadata:
        The JSON-safe record from :func:`report_metadata`.
    tables:
        Every table named by :data:`TABLE_FILES`, keyed by its short name.

    Returns
    -------
    Path
        The published directory.
    """
    missing = [name for name in TABLE_FILES if name not in tables]
    if missing:
        raise FactorResearchError(
            "the research report is missing table(s) " + ", ".join(sorted(missing))
        )

    try:
        report_root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise FactorResearchError(
            f"cannot create the research-report directory {str(report_root)!r}: {exc}"
        ) from exc

    final = report_root / research_run_id
    if final.exists():
        raise FactorResearchError(
            f"the research-report directory {str(final)!r} already exists: every run publishes "
            "a new directory"
        )

    try:
        staging = Path(
            tempfile.mkdtemp(dir=str(report_root), prefix=f".{research_run_id}.", suffix=".tmp")
        )
    except OSError as exc:
        raise FactorResearchError(
            f"cannot create a staging directory under {str(report_root)!r}: {exc}"
        ) from exc

    try:
        for name, filename in TABLE_FILES.items():
            _write_table(tables[name], staging / filename)
        _write_metadata(metadata, staging / METADATA_FILE)
        _verify_staged(staging, (*TABLE_FILES.values(), METADATA_FILE))
        _rename_report(staging, final)
    except FactorResearchError:
        _remove_quietly(staging)
        raise
    except (OSError, ValueError, TypeError) as exc:
        _remove_quietly(staging)
        raise FactorResearchError(
            f"could not publish the research report to {str(final)!r}: {exc}"
        ) from exc
    return final


def _write_table(frame: pd.DataFrame, path: Path) -> None:
    """Write one CSV table with ``\\n`` line endings, or a clear refusal."""
    if not isinstance(frame, pd.DataFrame):
        raise FactorResearchError(
            f"the table written to {str(path)!r} must be a pandas DataFrame, got "
            f"{type(frame).__name__}: a report file renders an object, it never is one"
        )
    try:
        frame.to_csv(path, index=False, lineterminator="\n")
    except (OSError, ValueError) as exc:
        raise FactorResearchError(f"cannot write the table {str(path)!r}: {exc}") from exc


def _write_metadata(metadata: Mapping[str, Any], path: Path) -> None:
    """Write ``metadata`` as one JSON document, refusing a non-finite number."""
    try:
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            # ``allow_nan=False`` turns a stray NaN or infinity into a hard error
            # here instead of a file no strict reader can parse.
            json.dump(metadata, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except (OSError, ValueError, TypeError) as exc:
        raise FactorResearchError(
            f"cannot write the research metadata to {str(path)!r}: {exc}"
        ) from exc


def _verify_staged(directory: Path, names: Sequence[str]) -> None:
    """Refuse to publish a directory that misses a file or holds an empty one."""
    missing = [name for name in names if not (directory / name).is_file()]
    empty = [
        name
        for name in names
        if (directory / name).is_file() and (directory / name).stat().st_size == 0
    ]
    problems: list[str] = []
    if missing:
        problems.append("missing file(s) " + ", ".join(sorted(missing)))
    if empty:
        problems.append("empty file(s) " + ", ".join(sorted(empty)))
    if problems:
        raise FactorResearchError(
            "the staged research report is incomplete, so it was not published: "
            + "; ".join(problems)
        )


def _rename_report(staging: Path, final: Path) -> None:
    """Move the staging directory onto its final name in one atomic step."""
    os.replace(str(staging), str(final))


def _remove_quietly(path: Path) -> None:
    """Delete ``path`` if it exists; a leftover staging directory is never the story."""
    shutil.rmtree(path, ignore_errors=True)


# ---------------------------------------------------------------------------
# The composition: panels -> labels -> three analyses -> one report
# ---------------------------------------------------------------------------
def analyze_factor_history(
    factor_name: str,
    start_date: date | str,
    end_date: date | str,
    *,
    horizon_sessions: int,
    frequency: str = DEFAULT_FREQUENCY,
    anchor_weekday: str | None = None,
    quantiles: int = DEFAULT_QUANTILES,
    min_pairs: int | None = None,
    data_root: Path | str | None = None,
    universe_path: Path | str | None = None,
    calendar_name: str | None = None,
    force: bool = False,
    report_root: Path | str | None = None,
) -> FactorResearchOutcome:
    """Analyze one factor over every decision date a cadence selects.

    The seven steps of the module docstring run in order, each through the public
    interface of the layer that owns it::

        batch   = run_factor_batch(...)                     which panels, cached or new
        panels  = tuple(item.panel for item in batch.outcomes)
        labels  = _label_panels(panels, ...)                one horizon, one calendar
        stats   = QuickAnalyzer().summarize_history(panels)
        ic      = ICAnalyzer(min_pairs=...).analyze(panels, labels)
        groups  = QuantileAnalyzer(quantiles=...).analyze(panels, labels)
        report  = write_report(...)                          one new directory

    The cadence and the horizon are passed separately and mean different things:
    ``frequency`` decides *which* sessions form a cross-section, while
    ``horizon_sessions`` decides *how far ahead* each one is measured.  Nothing
    derives one from the other -- ``weekly`` with a 20-session horizon is a
    legitimate, and deliberately overlapping, request.

    Parameters
    ----------
    factor_name:
        Registered factor name (``momentum_12_1``), resolved by the batch through
        the registry.  This module never imports a formula.
    start_date / end_date:
        The closed window the schedule selects from, as dates or strict ISO
        ``"YYYY-MM-DD"`` strings.  Labels may reach past ``end_date``: the
        horizon, not the window, decides where a label ends.
    horizon_sessions:
        Sessions between a decision date and its label's end date, a positive
        whole number (``bool``, ``0``, negatives, floats and strings are refused).
    frequency:
        ``"weekly"`` (the default), ``"biweekly"`` or ``"monthly"``.
    anchor_weekday:
        ``"MON"``..``"FRI"`` for weekly/biweekly (default ``"THU"``); ``None`` for
        monthly, which has no anchor.
    quantiles:
        Groups every cross-section is cut into, at least ``2`` (default ``5``).
    min_pairs:
        Usable pairs a date needs before its IC is computed; ``None`` keeps the
        :class:`~factor_sys.ic_analysis.ICAnalyzer` default.
    data_root:
        Root the whole analysis is derived from: panels and the batch manifest are
        written by the batch, the label window is read from
        ``<data_root>/standardized/market_bars`` and the report goes to
        ``<data_root>/factors/research_runs`` unless ``report_root`` says
        otherwise.
    universe_path:
        The static stock-pool CSV, passed to the batch unchanged.
    calendar_name:
        Trading calendar that defines a session, passed to the batch and reused
        for the labels so the two cannot disagree.
    force:
        Recompute every selected panel, replacing each stored file atomically.
    report_root:
        Where the report directory is published (default:
        ``<data_root>/factors/research_runs``).

    Returns
    -------
    FactorResearchOutcome
        The batch, the analysed panels, their labels, the three reports, the pair
        threshold and the published report directory.

    Raises
    ------
    FactorResearchError
        A request this entry point refuses (a bad horizon, group count or path), a
        selection with no date or with failed dates, or a report that could not be
        written.
    factor_sys.FactorError
        Whatever the batch, the label calculator, the IC analyzer or the quantile
        analyzer refused, unchanged: an unknown factor or window
        (:class:`~factor_sys.errors.FactorScheduleError`,
        :class:`~factor_sys.errors.FactorRegistryError`), an untrustworthy price
        window (:class:`~factor_sys.errors.ForwardReturnError`) or a join that does
        not close (:class:`~factor_sys.errors.ICAnalysisError`,
        :class:`~factor_sys.errors.QuantileAnalysisError`).
    """
    horizon = _session_count(horizon_sessions, "horizon_sessions")
    groups = _group_count(quantiles, "quantiles")
    threshold = None if min_pairs is None else _pair_count(min_pairs, "min_pairs")
    root = _data_root(data_root)
    reports = _report_root(report_root, root)

    batch = run_factor_batch(
        factor_name,
        start_date,
        end_date,
        frequency=frequency,
        anchor_weekday=anchor_weekday,
        data_root=root,
        universe_path=universe_path,
        calendar_name=calendar_name,
        force=force,
    )
    _require_complete(batch)

    panels = tuple(item.panel for item in batch.outcomes)
    labels = _label_panels(
        panels,
        data_root=root,
        calendar_name=batch.calendar_name,
        horizon_sessions=horizon,
    )

    quick = QuickAnalyzer()
    ic_analyzer = ICAnalyzer() if threshold is None else ICAnalyzer(min_pairs=threshold)
    quantile_analyzer = QuantileAnalyzer(quantiles=groups)

    statistics = quick.summarize_history(panels)
    ic_report = ic_analyzer.analyze(panels, labels)
    quantile_report = quantile_analyzer.analyze(panels, labels)

    research_run_id = new_run_id()
    metadata = report_metadata(
        batch,
        research_run_id=research_run_id,
        created_at_utc=utc_now_iso(),
        data_root=root,
        horizon_sessions=horizon,
        quantiles=groups,
        min_pairs=ic_analyzer.min_pairs,
        ic_report=ic_report,
    )
    tables = {
        "panel_statistics": quick.to_frame(statistics),
        "forward_returns": _forward_return_frame(labels),
        "ic_observations": ic_report.to_frame(),
        "quantile_observations": quantile_report.to_frame(),
        "quantile_spreads": quantile_report.spread_to_frame(),
    }
    report_path = write_report(reports, research_run_id, metadata, tables)

    return FactorResearchOutcome(
        batch_outcome=batch,
        factor_panels=panels,
        forward_return_panels=labels,
        panel_statistics=statistics,
        ic_report=ic_report,
        quantile_report=quantile_report,
        min_pairs=ic_analyzer.min_pairs,
        report_path=report_path,
    )


# ---------------------------------------------------------------------------
# The summary -- the only rendering of an analysis
# ---------------------------------------------------------------------------
def format_summary(outcome: FactorResearchOutcome) -> str:
    """Render ``outcome`` as the command's stable, one-field-per-line summary.

    The text is a reporting surface, never a storage format: the numbers are the
    reports' own values at :data:`DECIMALS` decimals, and every field that can be
    ``None`` -- no defined IC, no deviation, no ICIR -- is printed as
    :data:`MISSING_TEXT` rather than invented as ``0``.  A missing monthly anchor
    is printed the same way.

    The lines are always present and always in this order, so a caller may parse
    them by prefix: it starts with the identity of the factor, continues with the
    request (window, cadence, horizon, knobs), then the batch counts, then the IC
    summary, and ends with the two paths -- the report this run published and the
    manifest that records the panels behind it.  No table, no per-symbol label and
    no object ``repr`` is ever printed here.

    Parameters
    ----------
    outcome:
        The analysis to render.

    Returns
    -------
    str
        The summary, without a trailing newline (``print`` adds one).
    """
    batch = outcome.batch_outcome
    lines = [
        f"factor_name: {batch.factor_name}",
        f"factor_version: {batch.factor_version}",
        f"start_date: {batch.start_date.isoformat()}",
        f"end_date: {batch.end_date.isoformat()}",
        f"frequency: {batch.frequency}",
        f"anchor_weekday: {batch.anchor_weekday or MISSING_TEXT}",
        f"calendar_name: {batch.calendar_name}",
        f"horizon_sessions: {outcome.horizon_sessions}",
        f"quantiles: {outcome.quantile_report.quantiles}",
        f"min_pairs: {outcome.min_pairs}",
        f"selected_panel_count: {outcome.selected_panel_count}",
        f"computed_count: {batch.computed_count}",
        f"cached_count: {batch.cached_count}",
        f"forced_count: {batch.forced_count}",
        f"valid_ic_count: {outcome.valid_ic_count}",
        f"mean_ic: {_number(outcome.mean_ic)}",
        f"ic_std: {_number(outcome.ic_std)}",
        f"icir: {_number(outcome.icir)}",
        f"report_path: {outcome.report_path}",
        f"batch_manifest_path: {batch.manifest_path}",
    ]
    return "\n".join(lines)


def _number(value: float | None) -> str:
    """``value`` to :data:`DECIMALS` decimals, or :data:`MISSING_TEXT` for ``None``."""
    return MISSING_TEXT if value is None else f"{value:.{DECIMALS}f}"


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------
def _session_argument(text: str) -> int:
    """``--horizon-sessions``: a positive count of sessions, or argparse's ``2``.

    The conversion belongs to the parser so a typo fails before a single panel is
    read: a horizon is not a preference the analysis may quietly repair.
    """
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected a whole number of sessions, got {text!r}"
        ) from None
    if value < 1:
        raise argparse.ArgumentTypeError(
            f"horizon-sessions must be at least 1 session, got {value}"
        )
    return value


def _group_argument(text: str) -> int:
    """``--quantiles``: at least two groups, or argparse refuses the command line."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected a whole number of groups, got {text!r}"
        ) from None
    if value < 2:
        raise argparse.ArgumentTypeError(
            f"quantiles must be at least 2, got {value}: a spread needs two groups"
        )
    return value


def _pair_argument(text: str) -> int:
    """``--min-pairs``: at least two usable pairs, or argparse refuses it."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected a whole number of pairs, got {text!r}"
        ) from None
    if value < 2:
        raise argparse.ArgumentTypeError(
            f"min-pairs must be at least 2, got {value}: a rank correlation needs two pairs"
        )
    return value


def build_parser() -> argparse.ArgumentParser:
    """Build the parser, naming the defaults the analysis will actually use."""
    default_data_root = PipelineConfig().data_root
    parser = argparse.ArgumentParser(
        prog="analyze_factor.py",
        description=(
            "Measure one factor's Spearman IC and quantile returns across every session a "
            "weekly/biweekly/monthly schedule selects, and write one research report."
        ),
    )
    parser.add_argument(
        "--factor",
        required=True,
        metavar="FACTOR_NAME",
        help="registered factor name, e.g. momentum_12_1 (resolved through the registry)",
    )
    parser.add_argument(
        "--start",
        dest="start_date",
        required=True,
        metavar="YYYY-MM-DD",
        help="first day of the closed window the schedule selects from (inclusive)",
    )
    parser.add_argument(
        "--end",
        dest="end_date",
        required=True,
        metavar="YYYY-MM-DD",
        help=(
            "last day of that window (inclusive); labels may reach past it, because the "
            "horizon decides where a label ends"
        ),
    )
    parser.add_argument(
        "--frequency",
        choices=FREQUENCIES,
        default=DEFAULT_FREQUENCY,
        help=(
            "date cadence: weekly (one session per week: the anchor, or the session before it "
            "when the exchange is closed), biweekly (every second such week) or monthly (the "
            f"last session of each month) (default: {DEFAULT_FREQUENCY})"
        ),
    )
    parser.add_argument(
        "--anchor-weekday",
        dest="anchor_weekday",
        choices=ANCHOR_WEEKDAYS,
        default=None,
        help=(
            "weekday a weekly/biweekly schedule probes (default: THU); it does not apply to "
            "--frequency monthly"
        ),
    )
    parser.add_argument(
        "--horizon-sessions",
        dest="horizon_sessions",
        required=True,
        type=_session_argument,
        metavar="N",
        help=(
            "sessions between a decision date and its label's end date: 5 is about one week, "
            "20 about one month (required, and independent of --frequency)"
        ),
    )
    parser.add_argument(
        "--quantiles",
        type=_group_argument,
        default=DEFAULT_QUANTILES,
        metavar="N",
        help=f"groups every cross-section is cut into, at least 2 (default: {DEFAULT_QUANTILES})",
    )
    parser.add_argument(
        "--min-pairs",
        dest="min_pairs",
        type=_pair_argument,
        default=None,
        metavar="N",
        help=(
            "usable pairs a date needs before its IC is computed (default: the ICAnalyzer default)"
        ),
    )
    parser.add_argument(
        "--data-root",
        dest="data_root",
        metavar="PATH",
        default=None,
        help=(
            "root holding standardized/market_bars, factors/panels, factors/batch_runs and "
            f"factors/research_runs (default: {default_data_root})"
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
        help="trading calendar that defines a session (default: XNYS)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "recompute every selected date even when its panel is already stored, replacing "
            "each file atomically (default: read stored panels back)"
        ),
    )
    parser.add_argument(
        "--report-root",
        dest="report_root",
        metavar="PATH",
        default=None,
        help=("directory the report is published in (default: <data_root>/factors/research_runs)"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse ``argv``, run the analysis, print its summary and return the exit code.

    Parameters
    ----------
    argv:
        Command-line arguments without the program name (``None`` means
        ``sys.argv[1:]``).  Passing a list lets a test drive the command exactly as
        a shell would, without starting a second process.

    Returns
    -------
    int
        ``0`` when every selected date succeeded and the report was published;
        ``1`` when the batch lost a date, a label / IC / quantile layer refused its
        input, or the report could not be written.  Domain errors are printed to
        stderr as one ``error: ...`` line, never as a traceback, and no partial
        report is left behind.  ``argparse`` keeps its own exit code ``2``.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.frequency == MONTHLY and args.anchor_weekday is not None:
        parser.error(
            "--anchor-weekday does not apply to --frequency monthly: a monthly schedule "
            "already selects the last session of each month, so omit the flag"
        )

    try:
        outcome = analyze_factor_history(
            args.factor,
            args.start_date,
            args.end_date,
            horizon_sessions=args.horizon_sessions,
            frequency=args.frequency,
            anchor_weekday=args.anchor_weekday,
            quantiles=args.quantiles,
            min_pairs=args.min_pairs,
            data_root=args.data_root,
            universe_path=args.universe_path,
            calendar_name=args.calendar,
            force=args.force,
            report_root=args.report_root,
        )
    except FactorError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(format_summary(outcome))
    return 0


__all__ = [
    "DEFAULT_QUANTILES",
    "DECIMALS",
    "METADATA_FILE",
    "MISSING_TEXT",
    "REPORT_FORMAT_VERSION",
    "RESEARCH_DIRECTORY",
    "TABLE_FILES",
    "FactorResearchOutcome",
    "analyze_factor_history",
    "build_parser",
    "format_summary",
    "main",
    "report_metadata",
    "write_report",
]


if __name__ == "__main__":
    raise SystemExit(main())
