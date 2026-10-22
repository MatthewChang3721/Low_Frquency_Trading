"""Quick analysis: the plain shape of one cross-section, and of a history of them.

What this module is
-------------------
``factor_sys.quick_analysis`` answers the cheapest question about a
:class:`~factor_sys.contracts.FactorPanel`: *how many numbers did the factor
actually produce, and what do they look like?*  One panel becomes one frozen
:class:`FactorPanelStatistics`; a sequence of panels becomes a tuple of them in
date order.  Nothing else happens here: no forward return is read, no rank is
taken, nothing is filled and no opinion about whether the factor works is
expressed.

Why this is not the quality check
---------------------------------
:mod:`factor_sys.quality` asks a *veto* question -- "is this panel usable as it
stands?" -- and answers with issues.  This module asks a *descriptive* one and
never complains: an eligible symbol whose ``raw_value`` is ``NaN`` is a quality
*warning* there, and here it is simply not a usable number, so it lands in
``missing_count``.  A quality issue is a judgement; a statistic is a fact.

What is counted
---------------
===========================  =====================================================
``universe_size``            rows of the panel: one per symbol of the pool
``eligible_count``           rows with ``is_eligible=True``
``eligible_coverage``        ``eligible_count / universe_size``
``valid_count``              **eligible** rows whose ``raw_value`` is a finite
                             number (not ``NaN``, ``+inf`` or ``-inf``)
``missing_count``            ``universe_size - valid_count``
``missing_rate``             ``missing_count / universe_size``
``mean`` / ``std`` /         over the ``valid_count`` finite values only;
``minimum`` / ``maximum``    ``std`` is the population standard deviation
                             (``ddof=0``), taken about the reported mean
===========================  =====================================================

Three decisions are worth spelling out, because they are what makes the numbers
comparable across days:

*Missing is measured against the whole universe, not against the eligible
part.*  A symbol the factor never computed is absent from the cross-section just
as much as one that came back ``NaN``, so both are missing.  ``eligible_count``
is reported next to it for the opposite reason: it says how much of the pool the
formula was even willing to speak about, and ``eligible_coverage`` separates "the
formula excluded it" from "the formula could not produce a number".

*``None`` is not ``NaN``.*  When ``valid_count == 0`` there is no mean, so the
four numbers are ``None`` -- an honest "nothing to describe" -- instead of
``NaN``, which would poison every later arithmetic operation silently.  A
``valid_count`` of zero is a normal result of a legal panel, never an error.
``missing_rate`` then equals ``1.0``, which is the whole point of reporting it.

*Finite input, finite statistics.*  Every moment is formed from the usable
values only, and when their magnitude approaches the float limit the sample is
divided by its largest absolute value first and the moments are scaled back
afterwards: two ``1e308`` values are described by ``mean == 1e308`` and
``std == 0.0``, and ``1e308`` next to ``-1e308`` by ``mean == 0.0`` and
``std == 1e308``.  Finite values are never demoted to missing to dodge an
overflow, and an infinite statistic is never reported as a result --
:class:`FactorPanelStatistics` refuses one.

What this module deliberately does not do
-----------------------------------------
No forward returns, no IC/ICIR, no quantile portfolios, no rank, z-score,
winsorizing or filling, no multi-factor combination, no rebalancing, no storage
and no scheduling.  ``std`` is the population standard deviation (``ddof=0``) by
contract: a cross-section is the whole population of symbols on that day, and one
convention for every panel keeps histories comparable.

Purity
------
Only the panel the caller passes in is read, through its public ``frame``
property (a copy), so the panel is never modified.  No file, no path, no clock,
no calendar, no environment variable and no market-data client is involved: the
same panel always produces the same statistics, which makes a stored history
cheap to re-describe at any time.

Example
-------
>>> import pandas as pd
>>> from factor_sys.contracts import FactorPanel
>>> frame = pd.DataFrame(
...     {
...         "as_of_date": [pd.Timestamp("2024-01-05")] * 3,
...         "symbol": ["AAPL", "MSFT", "NVDA"],
...         "factor_name": ["momentum_12_1"] * 3,
...         "factor_version": ["1.0.0"] * 3,
...         "raw_value": [0.25, float("nan"), 0.75],
...         "is_eligible": [True, False, True],
...     }
... )
>>> panel = FactorPanel(frame, universe=["AAPL", "MSFT", "NVDA"])
>>> stats = QuickAnalyzer().summarize(panel)
>>> (stats.universe_size, stats.eligible_count, stats.valid_count, stats.missing_count)
(3, 2, 2, 1)
>>> (stats.mean, stats.std, stats.minimum, stats.maximum)
(0.5, 0.25, 0.25, 0.75)
>>> QuickAnalyzer().summarize_history(())
()
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd

from factor_sys.contracts import FACTOR_NAME_PATTERN, FactorPanel
from factor_sys.errors import QuickAnalysisError

# ---------------------------------------------------------------------------
# Table shape
# ---------------------------------------------------------------------------
#: Fixed column order of :meth:`QuickAnalyzer.to_frame` (one row per session).
STATISTICS_COLUMNS: tuple[str, ...] = (
    "as_of_date",
    "factor_name",
    "factor_version",
    "universe_size",
    "eligible_count",
    "eligible_coverage",
    "valid_count",
    "missing_count",
    "missing_rate",
    "mean",
    "std",
    "minimum",
    "maximum",
)

#: Panel columns this module reads (the contract's own names).
_VALUE_COLUMN = "raw_value"
_ELIGIBLE_COLUMN = "is_eligible"

#: A factor name is the contract's snake_case identifier; reused, not redefined.
_NAME_RE = re.compile(FACTOR_NAME_PATTERN)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _is_count(value: object) -> bool:
    """``True`` for a real count: a ``bool`` is a flag, not a count."""
    return isinstance(value, int) and not isinstance(value, bool)


def _is_real(value: object) -> bool:
    """``True`` for a real-valued statistic: a ``bool`` is not a number here."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


#: The smallest positive normal float.  A value below it still rounds, but with
#: fewer than 53 significant bits, so a sum or a variance there has lost digits.
_SMALLEST_NORMAL = math.ldexp(1.0, -1022)


def _total(sample: Sequence[float]) -> float | None:
    """The exact-then-rounded sum, or ``None`` when it leaves the float range.

    ``math.fsum`` is used because it is exact-then-rounded: ``mean`` does not
    depend on row order, and no digit is lost to accumulation.  It raises
    ``OverflowError`` when the total itself is beyond the float range, which is
    a fact about the sample (``1e308 + 1e308``), not a reason to answer ``inf``.
    """
    try:
        total = math.fsum(sample)
    except OverflowError:
        return None
    return total if math.isfinite(total) else None


def _deviation(sample: Sequence[float], mean: float) -> float | None:
    """Population deviation about ``mean``, or ``None`` when that loses digits.

    ``std`` is the population standard deviation (``ddof=0``) about the reported
    mean, so one value -- or many identical values -- is exactly ``0.0``, never a
    rounding leftover.

    ``None`` reports the two magnitudes where the plain formula stops being
    trustworthy, and the caller measures a scaled copy instead:

    * a sample near the float limit overflows in ``(value - mean) ** 2``
      (``(2e308) ** 2``), which would otherwise answer ``inf``;
    * a sample near ``1e-162`` squares into the subnormal range, where a float
      carries fewer than 53 significant bits, so the variance -- and the
      deviation read from it -- would be silently coarse rather than wrong.

    A variance of exactly ``0.0`` is not reported as untrustworthy here: for a
    flat sample it is the right answer, and for the other samples that reach it
    the caller rescales to recover the spread the squares lost.
    """
    count = len(sample)
    try:
        squared_deviations = [(value - mean) ** 2 for value in sample]
        variance = math.fsum(squared_deviations) / count
    except OverflowError:
        return None
    if 0.0 < variance < _SMALLEST_NORMAL:
        return None
    std = math.sqrt(variance)
    return std if math.isfinite(std) else None


def _scaled_moments(sample: Sequence[float], scale: float) -> tuple[float, float]:
    """``(mean, std)`` of ``sample``, formed on ``value / scale`` and scaled back.

    Both moments are invariant under a positive rescaling, so dividing by the
    sample's largest absolute value keeps every intermediate inside ``[-1, 1]``
    -- a range in which ``fsum`` cannot overflow and a squared deviation cannot
    leave the float range -- and multiplying the two results by ``scale``
    afterwards lands on the same numbers, finite.

    Two bounds make the scaling back safe: a mean of values in ``[-1, 1]`` is
    itself in ``[-1, 1]`` (so ``mean * scale`` cannot exceed ``scale``), and by
    Popoviciu's inequality a population variance over an interval of length two
    is at most one.  A normalized deviation a rounding step above ``1.0`` is
    therefore clamped, which is a sub-ulp correction that keeps ``std * scale``
    finite even when ``scale`` is the largest float.
    """
    count = len(sample)
    normalized = [value / scale for value in sample]
    normalized_mean = math.fsum(normalized) / count
    squared_deviations = [(value - normalized_mean) ** 2 for value in normalized]
    normalized_std = min(math.sqrt(math.fsum(squared_deviations) / count), 1.0)
    return normalized_mean * scale, normalized_std * scale


def _sample_statistics(
    sample: Sequence[float],
) -> tuple[float | None, float | None, float | None, float | None]:
    """``(mean, std, minimum, maximum)`` of a sample, or four times ``None``.

    ``sample`` must already hold exactly the finite values the statistics are
    about.  ``std`` is the population standard deviation (``ddof=0``) taken
    about the reported ``mean``, so one value -- or many identical values --
    comes back with ``std == 0.0`` exactly, never a rounding leftover.

    Finite input always produces finite output.  The exact-then-rounded total is
    kept whenever it fits (it is the most accurate mean there is), the direct
    deviation is kept whenever it fits, and only what the direct formula cannot
    answer is measured on the scaled copy: ``1e308`` twice becomes
    ``mean == 1e308`` with ``std == 0.0`` instead of two infinities.  The scaled
    copy is also taken when the direct deviation collapses to ``0.0`` for a
    sample that is not flat, or lands in the subnormal range, where floats carry
    fewer than 53 bits -- the mirror image of the overflow, caused by squared
    deviations of ``1e-200`` values underflowing.  No path turns a usable value
    into a missing one, so ``valid_count`` and the four statistics always
    describe the same rows.
    """
    if not sample:
        return None, None, None, None

    minimum = min(sample)
    maximum = max(sample)
    scale = max(abs(minimum), abs(maximum))

    mean: float | None = None
    std: float | None = None
    total = _total(sample)
    if total is not None:
        mean = total / len(sample)
        std = _deviation(sample, mean)

    if mean is None or std is None or (std == 0.0 and minimum != maximum):
        if scale == 0.0:
            mean, std = 0.0, 0.0
        else:
            scaled_mean, std = _scaled_moments(sample, scale)
            mean = scaled_mean if mean is None else mean

    return mean, std, minimum, maximum


def _rejected_panels(panels: object) -> str:
    """Explain why ``panels`` is not a sequence of panels for ``summarize_history``."""
    if isinstance(panels, FactorPanel):
        return (
            "QuickAnalyzer.summarize_history() needs an iterable of FactorPanel objects, "
            "got a single FactorPanel: summarize() describes one panel, "
            "summarize_history() describes a sequence of them"
        )
    return (
        "QuickAnalyzer.summarize_history() needs an iterable of FactorPanel objects (a "
        f"tuple, a list or a generator), got {type(panels).__name__}"
    )


# ---------------------------------------------------------------------------
# FactorPanelStatistics -- one described cross-section
# ---------------------------------------------------------------------------
@dataclass(frozen=True, repr=False)
class FactorPanelStatistics:
    """What one cross-section holds, counted and measured.

    Every field is copied from the panel or derived from its rows, so a
    statistics object can always be matched back to the panel it describes.
    Because the derived fields really are derived, they are cross-checked at
    construction: an object cannot claim a coverage, a rate, a count, a moment
    that is not finite or a deviation that contradicts its own numbers, so a
    hand-built or half-updated statistics object is a bug now instead of a
    subtle analysis error later.

    Parameters
    ----------
    as_of_date / factor_name / factor_version:
        The panel's identity, copied unchanged.
    universe_size:
        How many symbols the factor was asked about (``len(panel.universe)``).
    eligible_count:
        How many rows carry ``is_eligible=True``.
    eligible_coverage:
        ``eligible_count / universe_size``, exactly.
    valid_count:
        How many *eligible* rows hold a finite ``raw_value``.
    missing_count:
        ``universe_size - valid_count``: everything that is not a usable number,
        whether because it was ineligible or because its value was not finite.
    missing_rate:
        ``missing_count / universe_size``, exactly.
    mean / std / minimum / maximum:
        Over the ``valid_count`` finite values only; ``std`` is the population
        standard deviation (``ddof=0``).  All four are ``None`` when
        ``valid_count == 0``, because there is nothing to describe -- a normal
        result of a legal panel, not an error.  When ``valid_count > 0`` all four
        are finite numbers: a finite sample always has a finite mean and
        deviation, so ``NaN``, ``+inf`` and ``-inf`` are refused here.

    Examples
    --------
    >>> from datetime import date
    >>> stats = FactorPanelStatistics(
    ...     as_of_date=date(2024, 1, 5),
    ...     factor_name="momentum_12_1",
    ...     factor_version="1.0.0",
    ...     universe_size=4,
    ...     eligible_count=3,
    ...     eligible_coverage=0.75,
    ...     valid_count=2,
    ...     missing_count=2,
    ...     missing_rate=0.5,
    ...     mean=0.5,
    ...     std=0.25,
    ...     minimum=0.25,
    ...     maximum=0.75,
    ... )
    >>> stats.missing_rate
    0.5
    >>> (stats.valid_count, stats.missing_count)
    (2, 2)
    """

    as_of_date: date
    factor_name: str
    factor_version: str
    universe_size: int
    eligible_count: int
    eligible_coverage: float
    valid_count: int
    missing_count: int
    missing_rate: float
    mean: float | None
    std: float | None
    minimum: float | None
    maximum: float | None

    def __post_init__(self) -> None:
        """Validate the object and coerce the derived numbers to ``float``."""
        problems: list[str] = []

        if not isinstance(self.as_of_date, date) or isinstance(self.as_of_date, datetime):
            problems.append(
                "as_of_date must be a datetime.date (a day, not a datetime), got "
                f"{self.as_of_date!r}"
            )
        if not isinstance(self.factor_name, str) or not _NAME_RE.fullmatch(self.factor_name):
            problems.append(
                f"factor_name {self.factor_name!r} must be snake_case ({FACTOR_NAME_PATTERN})"
            )
        if not isinstance(self.factor_version, str) or not self.factor_version.strip():
            problems.append(
                f"factor_version must be a non-empty string, got {self.factor_version!r}"
            )

        counts = (
            ("universe_size", self.universe_size),
            ("eligible_count", self.eligible_count),
            ("valid_count", self.valid_count),
            ("missing_count", self.missing_count),
        )
        for label, value in counts:
            if not _is_count(value) or value < 0:
                problems.append(f"{label} must be a non-negative integer, got {value!r}")

        if problems:
            self._reject(problems)

        # -- the counts must agree with one another ---------------------------
        if self.universe_size < 1:
            problems.append(
                "universe_size must be at least 1: a panel always describes at least one symbol"
            )
        if self.eligible_count > self.universe_size:
            problems.append(
                f"eligible_count {self.eligible_count} cannot exceed universe_size "
                f"{self.universe_size}"
            )
        if self.valid_count > self.eligible_count:
            problems.append(
                f"valid_count {self.valid_count} cannot exceed eligible_count "
                f"{self.eligible_count}: a usable number requires an eligible row"
            )
        if self.missing_count != self.universe_size - self.valid_count:
            problems.append(
                f"missing_count {self.missing_count} must be universe_size - valid_count = "
                f"{self.universe_size - self.valid_count}: every symbol is either usable or "
                "missing"
            )
        if self.universe_size >= 1:
            expected_coverage = self.eligible_count / self.universe_size
            if self.eligible_coverage != expected_coverage:
                problems.append(
                    f"eligible_coverage {self.eligible_coverage!r} must be eligible_count / "
                    f"universe_size = {expected_coverage!r}"
                )
            expected_rate = self.missing_count / self.universe_size
            if self.missing_rate != expected_rate:
                problems.append(
                    f"missing_rate {self.missing_rate!r} must be missing_count / universe_size"
                    f" = {expected_rate!r}"
                )
        problems.extend(self._statistic_problems())

        if problems:
            self._reject(problems)

        object.__setattr__(self, "eligible_coverage", float(self.eligible_coverage))
        object.__setattr__(self, "missing_rate", float(self.missing_rate))
        for label in ("mean", "std", "minimum", "maximum"):
            value = getattr(self, label)
            if value is not None:
                object.__setattr__(self, label, float(value))

    def _statistic_problems(self) -> list[str]:
        """Check the four derived statistics against ``valid_count``."""
        problems: list[str] = []
        labels = ("mean", "std", "minimum", "maximum")
        values = (self.mean, self.std, self.minimum, self.maximum)

        if self.valid_count == 0:
            filled = [
                label for label, value in zip(labels, values, strict=True) if value is not None
            ]
            if filled:
                problems.append(
                    f"{', '.join(filled)} must be None when valid_count is 0: there is no "
                    "finite value to describe"
                )
            return problems

        absent = [label for label, value in zip(labels, values, strict=True) if value is None]
        if absent:
            problems.append(
                f"{', '.join(absent)} must be a number when valid_count is {self.valid_count}"
            )
            return problems

        not_representable = [
            label
            for label, value in zip(labels, values, strict=True)
            if not _is_real(value) or not math.isfinite(float(value))  # type: ignore[arg-type]
        ]
        if not_representable:
            problems.append(
                f"{', '.join(not_representable)} must be finite real numbers: a finite "
                "sample always has finite moments, so a NaN or an infinity is never a "
                "statistic"
            )
            return problems

        std = float(self.std)  # type: ignore[arg-type]
        minimum = float(self.minimum)  # type: ignore[arg-type]
        maximum = float(self.maximum)  # type: ignore[arg-type]

        # ``mean`` is deliberately not required to sit inside
        # ``[minimum, maximum]``: rounding can put it one ulp outside for a flat
        # cross-section (three copies of 0.1 average to 0.10000000000000002),
        # and that is arithmetic, not a contradiction.  That is the only freedom
        # left here: every moment is finite, because a finite sample has finite
        # moments and the analyzer guarantees it.
        if std < 0.0:
            problems.append(f"std {std!r} cannot be negative")
        if minimum > maximum:
            problems.append(f"minimum {minimum!r} cannot exceed maximum {maximum!r}")
        if self.valid_count == 1 and std != 0.0:
            problems.append(
                f"std must be 0.0 for a single valid value (population deviation), got {std!r}"
            )
        return problems

    def _reject(self, problems: list[str]) -> None:
        """Raise :class:`QuickAnalysisError` naming every problem found."""
        raise QuickAnalysisError(
            "FactorPanelStatistics("
            f"as_of_date={self.as_of_date!r}, factor_name={self.factor_name!r}) is not a "
            "valid description of a cross-section",
            problems,
        )

    def __repr__(self) -> str:
        """A one-line summary for logs; the panel's rows stay out of it."""
        return (
            f"FactorPanelStatistics(as_of_date={self.as_of_date.isoformat()!r}, "
            f"factor={self.factor_name}@{self.factor_version}, "
            f"universe={self.universe_size}, eligible={self.eligible_count}, "
            f"valid={self.valid_count}, missing={self.missing_count}, mean={self.mean!r})"
        )


# ---------------------------------------------------------------------------
# QuickAnalyzer -- panels in, statistics out
# ---------------------------------------------------------------------------
class QuickAnalyzer:
    """Describe a panel, or a history of panels, without judging them.

    The analyzer is stateless and takes no constructor arguments: it is a set of
    pure questions about objects the caller already holds.  It never reads a
    store, never asks a runner for a panel and never opens a file -- a caller
    that has panels (from a run, from ``ParquetFactorStore.read_range`` or from a
    test literal) gets the same numbers for the same rows.

    Two levels, one object:

    * :meth:`summarize` -- one :class:`~factor_sys.contracts.FactorPanel` in, one
      :class:`FactorPanelStatistics` out;
    * :meth:`summarize_history` -- any iterable of panels in, a tuple of
      statistics in ascending ``as_of_date`` order out.  The panels must share
      one ``factor_name`` and one ``factor_version`` and must not repeat a date:
      a description of "one factor over time" is the only thing a history means.

    :meth:`to_frame` turns that tuple into a table for a notebook or a CLI, with
    :data:`STATISTICS_COLUMNS` in a fixed order; it is a rendering, never a
    storage format.

    Examples
    --------
    >>> analyzer = QuickAnalyzer()
    >>> analyzer.summarize_history(())
    ()
    >>> repr(analyzer)
    'QuickAnalyzer()'
    """

    __slots__ = ()

    def summarize(self, panel: FactorPanel) -> FactorPanelStatistics:
        """Count and measure the cross-section ``panel`` carries.

        Only the panel's published rows are read (``raw_value`` and
        ``is_eligible``, through the ``frame`` copy), so the panel is never
        modified and the same panel always yields the same statistics.

        An eligible row whose value is ``NaN``, ``+inf`` or ``-inf`` is *not* a
        usable number: it is excluded from the mean/std/min/max and counted as
        missing, exactly like a symbol the factor never computed.  A panel with
        no usable value at all is described, not rejected: all four numbers come
        back as ``None``.

        A value that is merely *large* stays usable: ``1e308`` is a finite
        number of the same standing as ``0.5``, so it enters the moments and the
        moments stay finite (the sample is rescaled internally when the direct
        arithmetic would overflow, never demoted to missing).

        Parameters
        ----------
        panel:
            The validated cross-section to describe.

        Returns
        -------
        FactorPanelStatistics
            The frozen description of this one session.

        Raises
        ------
        QuickAnalysisError
            If ``panel`` is not a :class:`~factor_sys.contracts.FactorPanel` (a
            raw frame, a price series or ``None`` is not a described
            cross-section).
        """
        if not isinstance(panel, FactorPanel):
            raise QuickAnalysisError(
                f"QuickAnalyzer.summarize() needs a FactorPanel, got {type(panel).__name__}: "
                "statistics describe one validated cross-section, not a raw frame or a raw "
                "price series"
            )

        rows = panel.frame
        values = rows[_VALUE_COLUMN].tolist()
        flags = rows[_ELIGIBLE_COLUMN].tolist()

        universe_size = len(panel.universe)
        eligible_count = 0
        sample: list[float] = []
        for value, eligible in zip(values, flags, strict=True):
            if not eligible:
                continue
            eligible_count += 1
            if math.isfinite(value):
                sample.append(value)

        valid_count = len(sample)
        missing_count = universe_size - valid_count
        mean, std, minimum, maximum = _sample_statistics(sample)

        return FactorPanelStatistics(
            as_of_date=panel.as_of_date,
            factor_name=panel.factor_name,
            factor_version=panel.factor_version,
            universe_size=universe_size,
            eligible_count=eligible_count,
            eligible_coverage=eligible_count / universe_size,
            valid_count=valid_count,
            missing_count=missing_count,
            missing_rate=missing_count / universe_size,
            mean=mean,
            std=std,
            minimum=minimum,
            maximum=maximum,
        )

    def summarize_history(self, panels: Iterable[FactorPanel]) -> tuple[FactorPanelStatistics, ...]:
        """Describe every panel in ``panels`` as one factor's history.

        One call describes **one** factor at **one** version, because that is what
        a history is: panels of the same ``factor_name``/``factor_version``, at
        most one per ``as_of_date``, in any input order.  The result is sorted by
        ``as_of_date`` ascending, so a caller never has to sort a generator and a
        shuffled list produces the same tuple as a sorted one.

        A date missing from the input is *not* filled in: sessions without a panel
        simply do not appear, because inventing an empty cross-section would
        describe a day the factor never ran.  Universes may differ from day to
        day (a point-in-time pool changes over time), so panels are described one
        by one and never compared symbol-wise.

        Parameters
        ----------
        panels:
            Any iterable of panels -- a tuple, a list, a generator or the tuple
            ``ParquetFactorStore.read_range`` returns.  Empty input is a legal,
            empty history, not an error.

        Returns
        -------
        tuple[FactorPanelStatistics, ...]
            One immutable description per panel, ascending by ``as_of_date``.
            ``()`` for an empty input.

        Raises
        ------
        QuickAnalysisError
            If ``panels`` is not an iterable, holds something that is not a
            :class:`~factor_sys.contracts.FactorPanel`, mixes factor names or
            versions, or describes the same date twice.  Every problem found is
            reported at once, the way the rest of the factor system collects
            them.
        """
        if isinstance(panels, (str, bytes)) or not isinstance(panels, Iterable):
            raise QuickAnalysisError(_rejected_panels(panels))

        statistics: list[FactorPanelStatistics] = []
        problems: list[str] = []
        identity: tuple[str, str] | None = None
        seen: dict[date, int] = {}

        for position, panel in enumerate(panels):
            if not isinstance(panel, FactorPanel):
                problems.append(
                    f"panels[{position}] is a {type(panel).__name__}, not a FactorPanel: a "
                    "history of panel statistics can only be built from panels"
                )
                continue

            current = (panel.factor_name, panel.factor_version)
            if identity is None:
                identity = current
            elif current != identity:
                problems.append(
                    f"panels[{position}] is {current[0]}/{current[1]}, but this history is "
                    f"{identity[0]}/{identity[1]}: one call describes one factor at one version"
                )
                continue

            first = seen.setdefault(panel.as_of_date, position)
            if first != position:
                problems.append(
                    f"panels[{position}] repeats as_of_date "
                    f"{panel.as_of_date.isoformat()} (already given by panels[{first}]): a "
                    "history holds one cross-section per session"
                )
                continue

            statistics.append(self.summarize(panel))

        if problems:
            raise QuickAnalysisError(
                "QuickAnalyzer.summarize_history() cannot describe these panels as one factor "
                "history",
                problems,
            )

        statistics.sort(key=lambda item: item.as_of_date)
        return tuple(statistics)

    def to_frame(self, history: Iterable[FactorPanelStatistics]) -> pd.DataFrame:
        """Render a statistics sequence as a table, one row per session.

        This is a presentation helper for a notebook or a future CLI -- the tuple
        :meth:`summarize_history` returns stays the canonical result.  The
        columns are exactly :data:`STATISTICS_COLUMNS`, in that order, whatever
        the input order is; rows keep the order they were given in (which
        ``summarize_history`` already made ascending), and the panel's per-symbol
        rows are never mixed into this table.

        An empty history still produces a table with the full column set and
        usable dtypes (``datetime64[s]``, ``object``, ``int64``, ``float64``), so
        a caller can concatenate or plot without special-casing day one.

        Parameters
        ----------
        history:
            Any iterable of :class:`FactorPanelStatistics`, typically the tuple
            returned by :meth:`summarize_history`.

        Returns
        -------
        pandas.DataFrame
            A new frame; the statistics objects it was built from are untouched.

        Raises
        ------
        QuickAnalysisError
            If an item is not a :class:`FactorPanelStatistics` (a single object
            is refused too: wrap it as ``(stats,)``), since silently rendering
            something else would produce a table nobody could trust.
        """
        if isinstance(history, FactorPanelStatistics):
            raise QuickAnalysisError(
                "QuickAnalyzer.to_frame() needs an iterable of FactorPanelStatistics objects, "
                "got a single one: wrap it as (statistics,) -- one table row is one session"
            )
        if isinstance(history, (str, bytes)) or not isinstance(history, Iterable):
            raise QuickAnalysisError(
                "QuickAnalyzer.to_frame() needs an iterable of FactorPanelStatistics objects "
                f"(such as the tuple summarize_history() returns), got {type(history).__name__}"
            )

        items = list(history)
        problems = [
            f"history[{position}] is a {type(item).__name__}, not a FactorPanelStatistics"
            for position, item in enumerate(items)
            if not isinstance(item, FactorPanelStatistics)
        ]
        if problems:
            raise QuickAnalysisError(
                "QuickAnalyzer.to_frame() can only render FactorPanelStatistics objects",
                problems,
            )

        columns = {
            "as_of_date": pd.Series([item.as_of_date for item in items], dtype="datetime64[s]"),
            "factor_name": pd.Series([item.factor_name for item in items], dtype="object"),
            "factor_version": pd.Series([item.factor_version for item in items], dtype="object"),
            "universe_size": pd.Series([item.universe_size for item in items], dtype="int64"),
            "eligible_count": pd.Series([item.eligible_count for item in items], dtype="int64"),
            "eligible_coverage": pd.Series(
                [item.eligible_coverage for item in items], dtype="float64"
            ),
            "valid_count": pd.Series([item.valid_count for item in items], dtype="int64"),
            "missing_count": pd.Series([item.missing_count for item in items], dtype="int64"),
            "missing_rate": pd.Series([item.missing_rate for item in items], dtype="float64"),
            "mean": pd.Series([item.mean for item in items], dtype="float64"),
            "std": pd.Series([item.std for item in items], dtype="float64"),
            "minimum": pd.Series([item.minimum for item in items], dtype="float64"),
            "maximum": pd.Series([item.maximum for item in items], dtype="float64"),
        }
        return pd.DataFrame(columns, columns=list(STATISTICS_COLUMNS))

    def __repr__(self) -> str:
        """``QuickAnalyzer()``: the analyzer holds no state, so its type says everything."""
        return f"{type(self).__name__}()"


__all__ = [
    "STATISTICS_COLUMNS",
    "FactorPanelStatistics",
    "QuickAnalysisError",
    "QuickAnalyzer",
]
