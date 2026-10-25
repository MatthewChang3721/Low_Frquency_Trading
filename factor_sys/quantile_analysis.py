"""Quantile analysis: what did the ranked cross-section do afterwards?

What this module is
-------------------
``factor_sys.quantile_analysis`` sorts one session's usable stocks by their
factor value and asks a blunt question: did the stocks with the highest values
go on to earn more than the stocks with the lowest ones?  For one decision date
``t`` and a horizon of ``h`` sessions it measures, for every quantile
``q = 1..Q``::

    mean_forward_return(q, t) = mean of forward_return(t, h) over the symbols
                                whose raw_value(t) belongs to quantile q

and one number for the session as a whole::

    top_minus_bottom(t) = mean_forward_return(Q, t) - mean_forward_return(1, t)

One session becomes ``Q`` frozen :class:`QuantileObservation` objects plus one
:class:`QuantileSpreadObservation`; a history of sessions becomes one frozen
:class:`QuantileReport`.  :class:`QuantileAnalyzer` is the only object here: it
holds no data and reads nothing but the two panel sequences the caller hands it.

Research evidence, not a backtest
---------------------------------
The returns averaged here are
:class:`~factor_sys.forward_returns.ForwardReturnPanel` labels, that is
``adjusted_close(end_date) / adjusted_close(t) - 1``.  A group's mean is
therefore a **hindsight** statement about stocks that were sorted by a factor
on ``t``.  Nothing is traded: no order, no fill at the next open, no holding
period, no turnover, no commission, no slippage, no borrow and no cash
constraint.  A two-percent spread is a property of the sample, not a return
anybody could have earned; execution, costs and capacity belong to a backtest
built *on top of* this evidence, never inside it.

Which symbols are grouped
-------------------------
A symbol enters the sorted sample of one session only when all four hold:

===========================  ==============================================
``is_eligible`` is ``True``  the factor itself was willing to speak about it
``raw_value`` is finite       ``NaN``/``+inf``/``-inf`` is not a value
``is_available`` is ``True``  the label found two usable prices
``forward_return`` finite     a ``NaN`` label carries no information
===========================  ==============================================

``paired_count`` counts those symbols and ``paired_coverage`` is
``paired_count / universe_size`` -- the same denominator
:meth:`factor_sys.ic_analysis.ICAnalyzer` uses, so the two analyses can be read
side by side.  Nothing is filled, imputed, interpolated, borrowed from a
neighbouring session or dropped silently: an excluded symbol leaves every group
untouched and its absence is visible in the counts.

How a quantile is assigned
--------------------------
Within one session:

1. the usable symbols are ordered by ``raw_value``, ascending;
2. symbols with **exactly** the same factor value form one tie group, and a tie
   group is never split to make the groups even;
3. a tie group is placed by its *average rank* ``r`` (the mean of the 1-based
   ranks it would occupy), so ``r`` is a half-integer when it is tied;
4. that average rank decides the quantile::

       quantile = ceil(r * Q / paired_count)

5. the result is clamped to ``[1, Q]``, so the lowest value can never fall below
   the first group and the highest never above the last;

``Q`` is :attr:`QuantileAnalyzer.quantiles`, five by default.  No tiers,
``pandas.qcut`` edges or interpolation are involved, and nothing depends on the
order the rows arrive in: the same symbols with the same values always land in
the same quantile, ties or not.

Some groups may be empty, and that is a result
----------------------------------------------
Ties and small samples both produce empty groups -- two usable symbols in five
quantiles fill quantile 3 and quantile 5; a cross-section whose values are all
equal fills a single group.  An empty group is still reported: ``member_count``
is ``0``, ``member_coverage`` is ``0.0`` and ``mean_forward_return`` is ``None``
-- never ``NaN``, never ``0.0`` -- because "this group has no member" and "this
group returned nothing" are different statements.  When the extreme quantiles
are empty the spread is ``None`` too: the middle of the cross-section is never
substituted for the top or the bottom.  It is also ``None`` when the two means
differ by more than a float can hold (``1e308`` against ``-1e308``): an
unrepresentable spread is undefined, never an ``inf``.

What is averaged
----------------
Every group is an **equal-weighted** mean of its members' forward returns,
summed with ``math.fsum`` so the number does not depend on the order of the
symbols.  A group whose total leaves the float range is measured as a scaled
copy instead, so two ``1e308`` labels average to ``1e308`` rather than
overflowing the analysis.  There is no size, volatility or industry weighting,
no winsorizing, no clipping, no z-score, no rank transform written back and no
neutralization.
The grouping reads ``raw_value`` as the factor produced it -- a transformed
value would be a different factor with a different contract.

The history, and the holes in it
--------------------------------
Observations are kept in strictly ascending ``(as_of_date, quantile)`` order and
spreads in strictly ascending ``as_of_date`` order, so two input orders produce
the same object.  A calendar hole stays a hole: a date without a panel is absent
from the report, because a group return for a session the factor never ran on
would be invented data.

Boundaries of the request, never of the data
--------------------------------------------
Structure is refused, findings are not.  An empty sequence, a foreign object, a
single panel instead of an iterable, two factor names or versions, two label
horizons, a repeated date, a date present on one side only, or a universe that
moved or was re-ordered between a panel and its labels all raise
:class:`~factor_sys.errors.QuantileAnalysisError`, with every problem collected
so one failed call names them all.  A session with no usable symbol is *not* an
error: it is reported with its empty groups.

Not implemented here, on purpose: cumulative, annualized or risk-adjusted
statistics (Sharpe, t-statistics, Newey-West, ICIR), multi-factor combination,
industry or style neutralization, execution, costs, slippage and backtests,
label or report persistence, caching, batch scheduling, a CLI, cross-sectional
transforms (``rank``/``zscore``/winsorize/fill) and per-symbol member detail.

Example
-------
>>> from datetime import date
>>> import pandas as pd
>>> from factor_sys.contracts import FactorPanel
>>> from factor_sys.forward_returns import ForwardReturnPanel
>>> symbols = ["AAPL", "MSFT", "NVDA", "AMZN"]
>>> factor_rows = pd.DataFrame(
...     {
...         "as_of_date": [pd.Timestamp("2024-01-05")] * 4,
...         "symbol": symbols,
...         "factor_name": ["momentum_12_1"] * 4,
...         "factor_version": ["1.0.0"] * 4,
...         "raw_value": [0.1, 0.2, 0.3, 0.4],
...         "is_eligible": [True] * 4,
...     }
... )
>>> label_rows = factor_rows.drop(columns=["raw_value", "is_eligible"]).assign(
...     end_date=pd.Timestamp("2024-01-08"),
...     horizon_sessions=1,
...     forward_return=[0.5, 0.25, 0.25, -0.5],
...     is_available=True,
... )
>>> panel = FactorPanel(factor_rows, universe=symbols)
>>> labels = ForwardReturnPanel(
...     label_rows,
...     universe=symbols,
...     as_of_date=date(2024, 1, 5),
...     end_date=date(2024, 1, 8),
...     factor_name="momentum_12_1",
...     factor_version="1.0.0",
...     horizon_sessions=1,
... )
>>> report = QuantileAnalyzer(quantiles=2).analyze([panel], [labels])
>>> report.to_frame()["mean_forward_return"].tolist()
[0.375, -0.125]
>>> report.spread_to_frame()["top_minus_bottom"].tolist()
[-0.5]
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from numbers import Integral, Real

import pandas as pd

from factor_sys.contracts import FACTOR_NAME_PATTERN, FactorPanel
from factor_sys.errors import QuantileAnalysisError
from factor_sys.forward_returns import ForwardReturnPanel

# ---------------------------------------------------------------------------
# Table shape
# ---------------------------------------------------------------------------
#: Fixed column order of :meth:`QuantileReport.to_frame` (one row per group).
QUANTILE_OBSERVATION_COLUMNS: tuple[str, ...] = (
    "as_of_date",
    "end_date",
    "factor_name",
    "factor_version",
    "horizon_sessions",
    "quantiles",
    "quantile",
    "universe_size",
    "paired_count",
    "paired_coverage",
    "member_count",
    "member_coverage",
    "mean_forward_return",
)

#: Fixed column order of :meth:`QuantileReport.spread_to_frame` (one row per date).
QUANTILE_SPREAD_COLUMNS: tuple[str, ...] = (
    "as_of_date",
    "end_date",
    "factor_name",
    "factor_version",
    "horizon_sessions",
    "quantiles",
    "bottom_mean_forward_return",
    "top_mean_forward_return",
    "top_minus_bottom",
)

#: Panel and label columns this module reads (the contracts' own names).
_VALUE_COLUMN = "raw_value"
_ELIGIBLE_COLUMN = "is_eligible"
_RETURN_COLUMN = "forward_return"
_AVAILABLE_COLUMN = "is_available"

#: A factor name is the contract's snake_case identifier; reused, not redefined.
_NAME_RE = re.compile(FACTOR_NAME_PATTERN)

#: Groups a cross-section is cut into when the caller does not say.
_DEFAULT_QUANTILES = 5

#: Fewest groups a quantile analysis can compare: a bottom and a top.
_MIN_QUANTILES = 2

#: How far a hand-written derived number may sit from the recomputed one.
_CONSISTENCY_TOLERANCE = 1e-12


# ---------------------------------------------------------------------------
# Small validation helpers
# ---------------------------------------------------------------------------
def _is_count(value: object) -> bool:
    """``True`` for a real count: a ``bool`` is a flag, not a count."""
    return isinstance(value, Integral) and not isinstance(value, bool)


def _is_real(value: object) -> bool:
    """``True`` for a real-valued statistic: a ``bool`` is not a number here."""
    return isinstance(value, Real) and not isinstance(value, bool)


def _plain_date(value: object, label: str, problems: list[str]) -> date | None:
    """``value`` as a day, or a problem: a session is a day, never an instant."""
    if isinstance(value, datetime):
        problems.append(
            f"{label} must be a datetime.date, got a datetime {value!r}: a group return is "
            "measured for a session, not for an instant"
        )
        return None
    if isinstance(value, date):
        return value
    problems.append(f"{label} must be a datetime.date, got {value!r} ({type(value).__name__})")
    return None


def _factor_name(value: object, label: str, problems: list[str]) -> str | None:
    """``value`` as a snake_case factor name, or a problem."""
    if not isinstance(value, str) or not _NAME_RE.match(value):
        problems.append(
            f"{label} must be a snake_case factor name such as 'momentum_12_1', got {value!r}"
        )
        return None
    return value


def _factor_version(value: object, label: str, problems: list[str]) -> str | None:
    """``value`` as a non-empty version string, or a problem."""
    if not isinstance(value, str) or not value.strip():
        problems.append(f"{label} must be a non-empty version string, got {value!r}")
        return None
    return value


def _horizon_sessions(value: object, label: str, problems: list[str]) -> int | None:
    """``value`` as a positive number of sessions, or a problem."""
    if not _is_count(value) or value < 1:
        problems.append(
            f"{label} must be a positive integer number of sessions, got {value!r} "
            f"({type(value).__name__})"
        )
        return None
    return int(value)


def _count(value: object, label: str, problems: list[str]) -> int | None:
    """``value`` as a non-negative count, or a problem."""
    if not _is_count(value) or value < 0:
        problems.append(f"{label} must be a non-negative integer, got {value!r}")
        return None
    return int(value)


def _quantile_count(value: object, label: str, problems: list[str]) -> int | None:
    """``value`` as a usable quantile count (an integer of at least two), or a problem."""
    if not _is_count(value) or value < _MIN_QUANTILES:
        problems.append(
            f"{label} must be an integer of at least {_MIN_QUANTILES}, got {value!r} "
            f"({type(value).__name__}): a quantile analysis compares a bottom group with a top "
            "group, so one group is nothing to compare"
        )
        return None
    return int(value)


def _describe(items: Sequence[object], limit: int = 4) -> str:
    """Render a few items for an error message, saying how many were dropped."""
    shown = ", ".join(str(item) for item in items[:limit])
    if len(items) > limit:
        shown = f"{shown}, ... (+{len(items) - limit} more)"
    return shown or "none"


def _key_text(values: Sequence[date | int]) -> str:
    """Render a ``(date, quantile)`` sort key for an error message."""
    return ", ".join(item.isoformat() if isinstance(item, date) else str(item) for item in values)


def _quantiles(value: object) -> int:
    """``value`` as an analyzer's quantile count, or a refusal."""
    problems: list[str] = []
    quantiles = _quantile_count(value, "QuantileAnalyzer.quantiles", problems)
    if problems or quantiles is None:
        raise QuantileAnalysisError(
            "QuantileAnalyzer.quantiles is not a usable group count",
            problems or [f"got {value!r}"],
        )
    return quantiles


# ---------------------------------------------------------------------------
# Grouping: pairs in, quantile numbers out
# ---------------------------------------------------------------------------
def _paired_values(
    panel: FactorPanel, labels: ForwardReturnPanel
) -> tuple[list[float], list[float]]:
    """The usable ``(factor value, forward return)`` pairs of one session.

    A pair needs an eligible row, a finite ``raw_value``, an available label and
    a finite ``forward_return``; anything else is simply absent from the sample,
    never filled.  Both frames follow their own universe order -- which the
    caller has already proved identical -- so position *is* symbol here.
    """
    factor_rows = panel.frame
    label_rows = labels.frame
    values: list[float] = []
    returns: list[float] = []
    for value, eligible, forward, available in zip(
        factor_rows[_VALUE_COLUMN].tolist(),
        factor_rows[_ELIGIBLE_COLUMN].tolist(),
        label_rows[_RETURN_COLUMN].tolist(),
        label_rows[_AVAILABLE_COLUMN].tolist(),
        strict=True,
    ):
        if not eligible or not available:
            continue
        if not math.isfinite(value) or not math.isfinite(forward):
            continue
        values.append(float(value))
        returns.append(float(forward))
    return values, returns


def _quantile_assignments(values: Sequence[float], quantiles: int) -> list[int]:
    """The quantile of every value, by tie-group average rank, in input order.

    The values are ordered ascending; a run of exactly equal values is one tie
    group and receives the mean of the 1-based ranks the run covers, so a tie is
    never broken by row position.  That average rank ``r`` is placed with
    ``ceil(r * quantiles / len(values))`` and clamped to ``[1, quantiles]``.

    The result depends on the multiset of values only: two sessions holding the
    same symbols with the same values get the same assignment whatever order the
    rows arrive in.  With ``n`` values in ``Q`` groups, ties and small samples
    both leave groups empty -- ``n = 2`` in ``Q = 5`` fills quantiles 3 and 5 --
    and ``n`` equal values all land in one group.  An empty sample assigns
    nothing.
    """
    count = len(values)
    if count == 0:
        return []

    order = sorted(range(count), key=values.__getitem__)
    assignments = [0] * count
    start = 0
    while start < count:
        stop = start + 1
        while stop < count and values[order[stop]] == values[order[start]]:
            stop += 1
        average_rank = (start + 1 + stop) / 2.0
        group = math.ceil(average_rank * quantiles / count)
        group = min(quantiles, max(1, group))
        for position in range(start, stop):
            assignments[order[position]] = group
        start = stop
    return assignments


def _mean(values: Sequence[float]) -> float | None:
    """The equal-weighted mean of ``values``, or ``None`` when there is none.

    ``math.fsum`` is exactly rounded, so the mean of a group does not change when
    the symbols inside it are re-ordered: the same sample always gives the same
    digits, which is what makes two runs comparable at all.

    ``fsum`` raises ``OverflowError`` when the total itself leaves the float
    range (``1e308 + 1e308``).  That is a fact about the sample -- a group of
    very large returns -- not a reason to answer ``inf`` or to fail the
    analysis, so the mean is then formed on the sample divided by its largest
    absolute value and scaled back: every scaled value lies in ``[-1, 1]``,
    where no total can overflow, and a mean in ``[-1, 1]`` times a finite
    ``scale`` is itself finite.  Two labels of ``1e308`` average to ``1e308``;
    ``1e308`` against ``-1e308`` averages to ``0.0``.
    """
    if not values:
        return None
    try:
        total = math.fsum(values)
    except OverflowError:
        total = None
    if total is not None and math.isfinite(total):
        return total / len(values)
    scale = max(abs(value) for value in values)
    return math.fsum(value / scale for value in values) / len(values) * scale


def _difference(top: float | None, bottom: float | None) -> float | None:
    """``top - bottom``, or ``None`` when that difference has no float value.

    Both ends are finite, but their difference need not be: ``1e308`` against
    ``-1e308`` is ``2e308``, which is past the largest float.  An unrepresentable
    spread is reported as ``None`` -- an undefined number -- rather than as an
    ``inf`` that would quietly poison every total built on it.
    """
    if top is None or bottom is None:
        return None
    difference = float(top) - float(bottom)
    return difference if math.isfinite(difference) else None


def _agrees(reported: float | None, expected: float | None) -> bool:
    """``True`` when a hand-written derived number matches the recomputed one."""
    if reported is None or expected is None:
        return reported is None and expected is None
    return math.isclose(
        float(reported),
        float(expected),
        rel_tol=_CONSISTENCY_TOLERANCE,
        abs_tol=_CONSISTENCY_TOLERANCE,
    )


# ---------------------------------------------------------------------------
# QuantileObservation -- one session's group
# ---------------------------------------------------------------------------
@dataclass(frozen=True, repr=False)
class QuantileObservation:
    """One quantile of one decision date, and what that group went on to earn.

    An observation is the smallest honest unit of quantile evidence: one
    decision date, its label window, the identity of the factor measured, the
    group it describes, how much of the cross-section could be grouped at all,
    and the equal-weighted mean forward return of the group's members.  It is
    built by :class:`QuantileAnalyzer` and re-validated on construction, so a
    hand-written observation cannot claim a coverage its counts contradict or a
    mean no group could produce.

    Parameters
    ----------
    as_of_date:
        The decision session the factor was computed for.
    end_date:
        The session the labels were measured to (``as_of_date + h`` sessions).
        It must be strictly later, because a forward return covers at least one
        session.
    factor_name, factor_version:
        The identity of the measured factor, copied from the panel.
    horizon_sessions:
        ``h``: the label horizon, in exchange sessions.
    quantiles:
        ``Q``: how many groups the cross-section was cut into.
    quantile:
        Which group this observation is, from ``1`` (lowest factor values) to
        ``Q`` (highest).
    universe_size:
        Rows of the cross-section: symbols the factor was asked about.
    paired_count:
        Symbols that had *both* a usable factor value and a usable label, and
        therefore took part in the grouping.
    paired_coverage:
        ``paired_count / universe_size``, checked against the counts.
    member_count:
        Symbols in this group; the counts of one date's observations add up to
        their shared ``paired_count``.
    member_coverage:
        ``member_count / universe_size``, checked against the counts; the
        coverages of one date add up to ``paired_coverage``.
    mean_forward_return:
        The equal-weighted mean label of the members, or ``None`` when the group
        is empty -- never ``NaN``, never ``0.0``.  When present it is finite.

    Notes
    -----
    Observations are compared by value (they hold no frame and no member list),
    so two runs over the same inputs produce equal objects.
    """

    as_of_date: date
    end_date: date
    factor_name: str
    factor_version: str
    horizon_sessions: int
    quantiles: int
    quantile: int
    universe_size: int
    paired_count: int
    paired_coverage: float
    member_count: int
    member_coverage: float
    mean_forward_return: float | None

    def __post_init__(self) -> None:
        """Check the identity, the counts and the group mean, then freeze them."""
        problems: list[str] = []

        _plain_date(self.as_of_date, "QuantileObservation.as_of_date", problems)
        _plain_date(self.end_date, "QuantileObservation.end_date", problems)
        _factor_name(self.factor_name, "QuantileObservation.factor_name", problems)
        _factor_version(self.factor_version, "QuantileObservation.factor_version", problems)
        _horizon_sessions(self.horizon_sessions, "QuantileObservation.horizon_sessions", problems)
        quantiles = _quantile_count(self.quantiles, "QuantileObservation.quantiles", problems)
        quantile = _count(self.quantile, "QuantileObservation.quantile", problems)
        _count(self.universe_size, "QuantileObservation.universe_size", problems)
        _count(self.paired_count, "QuantileObservation.paired_count", problems)
        _count(self.member_count, "QuantileObservation.member_count", problems)

        for label in ("paired_coverage", "member_coverage"):
            value = getattr(self, label)
            if not _is_real(value) or not math.isfinite(float(value)):
                problems.append(
                    f"QuantileObservation.{label} must be a finite real number, got "
                    f"{value!r} ({type(value).__name__})"
                )

        mean = self.mean_forward_return
        if mean is not None and (not _is_real(mean) or not math.isfinite(float(mean))):
            problems.append(
                "QuantileObservation.mean_forward_return must be a finite real number or None, "
                f"got {mean!r} ({type(mean).__name__}): an undefined group mean is None, never "
                "NaN or an infinity"
            )

        if problems:
            self._reject(problems)

        self._check_fields(quantiles, quantile, problems)

        if problems:
            self._reject(problems)

        object.__setattr__(self, "horizon_sessions", int(self.horizon_sessions))
        object.__setattr__(self, "quantiles", int(self.quantiles))
        object.__setattr__(self, "quantile", int(self.quantile))
        object.__setattr__(self, "universe_size", int(self.universe_size))
        object.__setattr__(self, "paired_count", int(self.paired_count))
        object.__setattr__(self, "member_count", int(self.member_count))
        object.__setattr__(self, "paired_coverage", float(self.paired_coverage))
        object.__setattr__(self, "member_coverage", float(self.member_coverage))
        if self.mean_forward_return is not None:
            object.__setattr__(self, "mean_forward_return", float(self.mean_forward_return))

    def _check_fields(
        self, quantiles: int | None, quantile: int | None, problems: list[str]
    ) -> None:
        """The contradictions between fields, once each field is well typed."""
        if self.end_date <= self.as_of_date:
            problems.append(
                f"end_date {self.end_date.isoformat()} must be later than as_of_date "
                f"{self.as_of_date.isoformat()}: a forward return always covers at least one "
                "session"
            )
        if self.universe_size < 1:
            problems.append(
                "universe_size must be at least 1: a cross-section always holds at least one symbol"
            )
        if self.paired_count > self.universe_size:
            problems.append(
                f"paired_count {self.paired_count} cannot exceed universe_size "
                f"{self.universe_size}: a pair is one symbol of the cross-section"
            )
        if self.member_count > self.paired_count:
            problems.append(
                f"member_count {self.member_count} cannot exceed paired_count "
                f"{self.paired_count}: a group is drawn from the grouped symbols"
            )
        if quantiles is not None and quantile is not None and not 1 <= self.quantile <= quantiles:
            problems.append(
                f"quantile {self.quantile} must lie in [1, {quantiles}]: the groups of a "
                "cross-section are numbered from its lowest factor values to its highest"
            )
        if self.universe_size >= 1:
            for label, members, source in (
                ("paired_coverage", self.paired_count, "paired_count"),
                ("member_coverage", self.member_count, "member_count"),
            ):
                expected = members / self.universe_size
                if getattr(self, label) != expected:
                    problems.append(
                        f"{label} {getattr(self, label)!r} must be {source} / universe_size = "
                        f"{expected!r}"
                    )
        if self.member_count == 0:
            if self.mean_forward_return is not None:
                problems.append(
                    f"mean_forward_return {self.mean_forward_return!r} cannot exist for an empty "
                    "group: a group with no member has no mean, which is None"
                )
        elif self.mean_forward_return is None:
            problems.append(
                f"mean_forward_return is None although member_count is {self.member_count}: a "
                "non-empty group always has a mean"
            )

    def _reject(self, problems: list[str]) -> None:
        """Raise :class:`QuantileAnalysisError` naming every problem found."""
        raise QuantileAnalysisError(
            "QuantileObservation("
            f"as_of_date={self.as_of_date!r}, factor_name={self.factor_name!r}, "
            f"quantile={self.quantile!r}) is not a valid single-session group observation",
            problems,
        )

    def __repr__(self) -> str:
        """A one-line summary for logs; no per-symbol value is ever printed."""
        return (
            f"QuantileObservation(as_of_date={self.as_of_date.isoformat()!r}, "
            f"end_date={self.end_date.isoformat()!r}, "
            f"factor={self.factor_name}@{self.factor_version}, "
            f"horizon_sessions={self.horizon_sessions}, quantile={self.quantile}/"
            f"{self.quantiles}, paired={self.paired_count}/{self.universe_size}, "
            f"coverage={self.paired_coverage!r}, members={self.member_count}, "
            f"mean_forward_return={self.mean_forward_return!r})"
        )


# ---------------------------------------------------------------------------
# QuantileSpreadObservation -- the top group minus the bottom group
# ---------------------------------------------------------------------------
@dataclass(frozen=True, repr=False)
class QuantileSpreadObservation:
    """One decision date's top-minus-bottom return.

    The spread is the headline number of a quantile analysis and the one most
    easily misread, so it keeps its two ingredients: ``top_minus_bottom`` is
    always ``top_mean_forward_return - bottom_mean_forward_return``, recomputed
    and checked on construction, and it is ``None`` whenever either extreme group
    is empty.  An empty group is *not* a zero return and the second-highest group
    is never promoted to "top".

    Parameters
    ----------
    as_of_date / end_date:
        The decision session and the session its labels end on (strictly later).
    factor_name, factor_version, horizon_sessions, quantiles:
        The identity and shape of the analysis this spread belongs to.
    bottom_mean_forward_return:
        The equal-weighted mean label of quantile ``1``, ``None`` when empty.
    top_mean_forward_return:
        The equal-weighted mean label of quantile ``Q``, ``None`` when empty.
    top_minus_bottom:
        ``top - bottom``, or ``None`` when either of the two is ``None`` or when
        that difference is too large for a float (``1e308`` against ``-1e308``).
        Finite when present; never ``NaN`` and never an ``inf``.

    Notes
    -----
    This is a research statistic over realized labels, not the return of a
    portfolio: nothing was bought, sold, held, borrowed or charged a commission.
    """

    as_of_date: date
    end_date: date
    factor_name: str
    factor_version: str
    horizon_sessions: int
    quantiles: int
    bottom_mean_forward_return: float | None
    top_mean_forward_return: float | None
    top_minus_bottom: float | None

    def __post_init__(self) -> None:
        """Check the identity, the two extremes and their difference, then freeze."""
        problems: list[str] = []

        _plain_date(self.as_of_date, "QuantileSpreadObservation.as_of_date", problems)
        _plain_date(self.end_date, "QuantileSpreadObservation.end_date", problems)
        _factor_name(self.factor_name, "QuantileSpreadObservation.factor_name", problems)
        _factor_version(self.factor_version, "QuantileSpreadObservation.factor_version", problems)
        _horizon_sessions(
            self.horizon_sessions, "QuantileSpreadObservation.horizon_sessions", problems
        )
        _quantile_count(self.quantiles, "QuantileSpreadObservation.quantiles", problems)

        for label in (
            "bottom_mean_forward_return",
            "top_mean_forward_return",
            "top_minus_bottom",
        ):
            value = getattr(self, label)
            if value is None:
                continue
            if not _is_real(value) or not math.isfinite(float(value)):
                problems.append(
                    f"QuantileSpreadObservation.{label} must be a finite real number or None, "
                    f"got {value!r} ({type(value).__name__}): an undefined spread is None, never "
                    "NaN or an infinity"
                )

        if problems:
            self._reject(problems)

        if self.end_date <= self.as_of_date:
            problems.append(
                f"end_date {self.end_date.isoformat()} must be later than as_of_date "
                f"{self.as_of_date.isoformat()}: a forward return always covers at least one "
                "session"
            )

        bottom = self.bottom_mean_forward_return
        top = self.top_mean_forward_return
        spread = self.top_minus_bottom
        if bottom is None or top is None:
            empty = "bottom" if bottom is None else "top"
            if spread is not None:
                problems.append(
                    f"top_minus_bottom {spread!r} cannot exist when the {empty} group is empty "
                    "(None): an empty group has no mean to subtract"
                )
        else:
            difference = _difference(top, bottom)
            if difference is None:
                if spread is not None:
                    problems.append(
                        f"top_minus_bottom {float(spread)!r} cannot exist: "
                        f"top_mean_forward_return - bottom_mean_forward_return = "
                        f"{float(top)!r} - {float(bottom)!r} is past the float range, so the "
                        "only representable answer is None"
                    )
            elif spread is None:
                problems.append(
                    "top_minus_bottom is None although both extreme groups have a mean and "
                    "their difference is representable: the difference of two numbers always "
                    "exists"
                )
            elif not _agrees(spread, difference):
                problems.append(
                    f"top_minus_bottom {float(spread)!r} must be top_mean_forward_return - "
                    f"bottom_mean_forward_return = {difference!r}"
                )

        if problems:
            self._reject(problems)

        object.__setattr__(self, "horizon_sessions", int(self.horizon_sessions))
        object.__setattr__(self, "quantiles", int(self.quantiles))
        for label in (
            "bottom_mean_forward_return",
            "top_mean_forward_return",
            "top_minus_bottom",
        ):
            value = getattr(self, label)
            if value is not None:
                object.__setattr__(self, label, float(value))

    def _reject(self, problems: list[str]) -> None:
        """Raise :class:`QuantileAnalysisError` naming every problem found."""
        raise QuantileAnalysisError(
            "QuantileSpreadObservation("
            f"as_of_date={self.as_of_date!r}, factor_name={self.factor_name!r}) is not a valid "
            "top-minus-bottom observation",
            problems,
        )

    def __repr__(self) -> str:
        """A one-line summary for logs."""
        return (
            f"QuantileSpreadObservation(as_of_date={self.as_of_date.isoformat()!r}, "
            f"factor={self.factor_name}@{self.factor_version}, "
            f"horizon_sessions={self.horizon_sessions}, quantiles={self.quantiles}, "
            f"bottom={self.bottom_mean_forward_return!r}, top={self.top_mean_forward_return!r}, "
            f"top_minus_bottom={self.top_minus_bottom!r})"
        )


# ---------------------------------------------------------------------------
# QuantileReport -- a factor's quantile history over many sessions
# ---------------------------------------------------------------------------
@dataclass(frozen=True, repr=False)
class QuantileReport:
    """A factor's quantile history: one row per date and group, one row per spread.

    One report is **one** factor at **one** version and **one** horizon, measured
    over a sequence of decision dates with a fixed number of groups.  It keeps
    every observation -- including the empty groups, because "the top group had
    no member" is evidence in its own right -- in strictly ascending
    ``(as_of_date, quantile)`` order, and one spread per date in ascending date
    order.

    A hand-built report is checked against itself before it is frozen: every
    date must carry exactly ``quantiles`` observations numbered ``1..Q``, those
    observations must agree on their end date, universe size, paired count and
    coverage, their member counts must add up to that paired count, their member
    coverages must add up to that coverage, and each spread must reproduce the
    means of quantile ``1`` and quantile ``Q`` of its own date.  A report that
    disagrees with itself cannot exist.

    Parameters
    ----------
    factor_name, factor_version, horizon_sessions:
        The identity every observation and spread must share.
    quantiles:
        ``Q``: the number of groups every session was cut into.
    observations:
        ``Q`` observations per session, ascending by ``(as_of_date, quantile)``,
        at least one session.
    spreads:
        One top-minus-bottom observation per session of ``observations``, in
        ascending ``as_of_date`` order.

    Notes
    -----
    ``to_frame`` and ``spread_to_frame`` render the two tables; the object itself
    stays the canonical result, and neither table is ever written back into it.
    """

    factor_name: str
    factor_version: str
    horizon_sessions: int
    quantiles: int
    observations: tuple[QuantileObservation, ...]
    spreads: tuple[QuantileSpreadObservation, ...]

    def __post_init__(self) -> None:
        """Check the shape, every row and the dates, then freeze the report."""
        problems: list[str] = []

        name = _factor_name(self.factor_name, "QuantileReport.factor_name", problems)
        version = _factor_version(self.factor_version, "QuantileReport.factor_version", problems)
        horizon = _horizon_sessions(
            self.horizon_sessions, "QuantileReport.horizon_sessions", problems
        )
        quantiles = _quantile_count(self.quantiles, "QuantileReport.quantiles", problems)

        observations = self._collected(self.observations, QuantileObservation, problems)
        spreads = self._collected(self.spreads, QuantileSpreadObservation, problems)

        if problems:
            self._reject(problems)

        self._check_rows(observations, spreads, name, version, horizon, quantiles, problems)
        self._check_dates(observations, spreads, quantiles, problems)

        if problems:
            self._reject(problems)

        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "spreads", spreads)
        object.__setattr__(self, "horizon_sessions", int(self.horizon_sessions))
        object.__setattr__(self, "quantiles", int(self.quantiles))

    def _collected(
        self,
        items: object,
        kind: type,
        problems: list[str],
    ) -> tuple:
        """``items`` as a tuple of ``kind`` objects, plus what is wrong with them."""
        label = "observations" if kind is QuantileObservation else "spreads"
        if isinstance(items, kind):
            problems.append(
                f"QuantileReport.{label} must be a sequence of {kind.__name__} objects, got a "
                f"single one: wrap it as (item,) -- one group is one table row"
            )
            return (items,)
        if isinstance(items, (str, bytes)) or not isinstance(items, Iterable):
            problems.append(
                f"QuantileReport.{label} must be an iterable of {kind.__name__} objects, got "
                f"{type(items).__name__}"
            )
            return ()
        collected = tuple(items)
        problems.extend(
            f"QuantileReport.{label}[{position}] is a {type(item).__name__}, not a {kind.__name__}"
            for position, item in enumerate(collected)
            if not isinstance(item, kind)
        )
        if not collected:
            problems.append(
                f"QuantileReport.{label} must hold at least one {kind.__name__}: a report of "
                "nothing describes no session"
            )
        return collected

    def _check_rows(
        self,
        observations: Sequence[QuantileObservation],
        spreads: Sequence[QuantileSpreadObservation],
        name: str | None,
        version: str | None,
        horizon: int | None,
        quantiles: int | None,
        problems: list[str],
    ) -> None:
        """Every row must belong to this report and sit in ascending order."""
        for position, observation in enumerate(observations):
            self._check_row_identity(
                position,
                "observations",
                observation.factor_name,
                observation.factor_version,
                observation.horizon_sessions,
                observation.quantiles,
                name,
                version,
                horizon,
                quantiles,
                problems,
            )
            if position == 0:
                continue
            previous = observations[position - 1]
            self._check_order(
                position,
                "observations",
                (observation.as_of_date, observation.quantile),
                (previous.as_of_date, previous.quantile),
                problems,
            )

        for position, spread in enumerate(spreads):
            self._check_row_identity(
                position,
                "spreads",
                spread.factor_name,
                spread.factor_version,
                spread.horizon_sessions,
                spread.quantiles,
                name,
                version,
                horizon,
                quantiles,
                problems,
            )
            if position == 0:
                continue
            previous = spreads[position - 1]
            self._check_order(
                position,
                "spreads",
                (spread.as_of_date,),
                (previous.as_of_date,),
                problems,
            )

    def _check_row_identity(
        self,
        position: int,
        label: str,
        name: str,
        version: str,
        horizon: int,
        row_quantiles: int,
        expected_name: str | None,
        expected_version: str | None,
        expected_horizon: int | None,
        expected_quantiles: int | None,
        problems: list[str],
    ) -> None:
        """One row's identity must equal the report's identity."""
        if (name, version) != (expected_name, expected_version):
            problems.append(
                f"{label}[{position}] is {name}/{version}, but this report is "
                f"{expected_name}/{expected_version}: one report measures one factor at one "
                "version"
            )
        if horizon != expected_horizon:
            problems.append(
                f"{label}[{position}] was measured over {horizon} session(s), but this report "
                f"says {expected_horizon}: one report measures one horizon"
            )
        if row_quantiles != expected_quantiles:
            problems.append(
                f"{label}[{position}] was cut into {row_quantiles} group(s), but this report "
                f"says {expected_quantiles}: one report has one grouping"
            )

    def _check_order(
        self,
        position: int,
        label: str,
        current: tuple[date, ...],
        previous: tuple[date, ...],
        problems: list[str],
    ) -> None:
        """Rows are strictly ascending; a repeat or a step back is a contradiction."""
        shown = _key_text(current)
        if current == previous:
            problems.append(
                f"{label}[{position}] repeats {shown} (already given by {label}[{position - 1}]): "
                "one group is measured once per session"
            )
        elif current < previous:
            problems.append(
                f"{label}[{position}] is at {shown}, before {label}[{position - 1}] at "
                f"{_key_text(previous)}: rows are kept in ascending date order"
            )

    def _check_dates(
        self,
        observations: Sequence[QuantileObservation],
        spreads: Sequence[QuantileSpreadObservation],
        quantiles: int | None,
        problems: list[str],
    ) -> None:
        """The two tables must describe the same sessions, group by group."""
        blocks: dict[date, list[QuantileObservation]] = {}
        for observation in observations:
            blocks.setdefault(observation.as_of_date, []).append(observation)

        spread_by_date: dict[date, QuantileSpreadObservation] = {}
        for spread in spreads:
            spread_by_date.setdefault(spread.as_of_date, spread)

        for day, group in blocks.items():
            self._check_block(day, group, quantiles, problems)

        missing = sorted(set(blocks) - set(spread_by_date))
        extra = sorted(set(spread_by_date) - set(blocks))
        if missing:
            problems.append(
                f"spreads has no top-minus-bottom row for as_of_date "
                f"{_describe([day.isoformat() for day in missing])}: every measured session "
                "carries exactly one spread"
            )
        if extra:
            problems.append(
                f"spreads has a top-minus-bottom row for as_of_date "
                f"{_describe([day.isoformat() for day in extra])} with no observation: a spread "
                "is the difference of two measured groups"
            )

        for day in sorted(set(blocks) & set(spread_by_date)):
            self._check_spread_agrees(day, blocks[day], spread_by_date[day], quantiles, problems)

    def _check_block(
        self,
        day: date,
        group: Sequence[QuantileObservation],
        quantiles: int | None,
        problems: list[str],
    ) -> None:
        """One session's groups must be a complete, coherent set for that date."""
        numbers = [item.quantile for item in group]
        if quantiles is not None:
            if len(group) != quantiles:
                problems.append(
                    f"{day.isoformat()} holds {len(group)} group observation(s), but this report "
                    f"says {quantiles} quantiles: every session reports every group, empty or not"
                )
            elif numbers != list(range(1, quantiles + 1)):
                problems.append(
                    f"{day.isoformat()} reports quantiles {numbers} instead of 1..{quantiles} in "
                    "ascending order: one session has one group per quantile"
                )

        first = group[0]
        for observation in group[1:]:
            if observation.end_date != first.end_date:
                problems.append(
                    f"{day.isoformat()} group {observation.quantile} ends on "
                    f"{observation.end_date.isoformat()}, but group {first.quantile} ends on "
                    f"{first.end_date.isoformat()}: one session's labels share one window"
                )
            if observation.universe_size != first.universe_size:
                problems.append(
                    f"{day.isoformat()} group {observation.quantile} covers "
                    f"{observation.universe_size} symbol(s), but group {first.quantile} covers "
                    f"{first.universe_size}: one session has one cross-section"
                )
            if observation.paired_count != first.paired_count:
                problems.append(
                    f"{day.isoformat()} group {observation.quantile} groups "
                    f"{observation.paired_count} symbol(s), but group {first.quantile} groups "
                    f"{first.paired_count}: the groups of one session share one sample"
                )
            if observation.paired_coverage != first.paired_coverage:
                problems.append(
                    f"{day.isoformat()} group {observation.quantile} reports coverage "
                    f"{observation.paired_coverage!r}, but group {first.quantile} reports "
                    f"{first.paired_coverage!r}: the groups of one session share one coverage"
                )

        members = sum(item.member_count for item in group)
        if members != first.paired_count:
            problems.append(
                f"{day.isoformat()}: member_count sums to {members}, but paired_count is "
                f"{first.paired_count}: every grouped symbol belongs to exactly one group"
            )
        coverage = math.fsum(item.member_coverage for item in group)
        if not _agrees(coverage, first.paired_coverage):
            problems.append(
                f"{day.isoformat()}: member_coverage sums to {coverage!r}, but paired_coverage is "
                f"{first.paired_coverage!r}: every grouped symbol belongs to exactly one group"
            )

    def _check_spread_agrees(
        self,
        day: date,
        group: Sequence[QuantileObservation],
        spread: QuantileSpreadObservation,
        quantiles: int | None,
        problems: list[str],
    ) -> None:
        """A session's spread must be exactly its own bottom and top group."""
        first = group[0]
        if spread.end_date != first.end_date:
            problems.append(
                f"the spread of {day.isoformat()} ends on {spread.end_date.isoformat()}, but its "
                f"groups end on {first.end_date.isoformat()}: one session has one label window"
            )

        by_number = {item.quantile: item for item in group}
        lower = by_number.get(1)
        upper = by_number.get(quantiles) if quantiles is not None else None

        for label, observation in (
            ("bottom_mean_forward_return", lower),
            ("top_mean_forward_return", upper),
        ):
            if observation is None:
                continue
            reported = getattr(spread, label)
            if not _agrees(reported, observation.mean_forward_return):
                problems.append(
                    f"the spread of {day.isoformat()} reports {label} {reported!r}, but quantile "
                    f"{observation.quantile} has mean_forward_return "
                    f"{observation.mean_forward_return!r}: the spread is built from these groups"
                )

        expected = _difference(spread.top_mean_forward_return, spread.bottom_mean_forward_return)
        if not _agrees(spread.top_minus_bottom, expected):
            problems.append(
                f"the spread of {day.isoformat()} reports top_minus_bottom "
                f"{spread.top_minus_bottom!r}, but its own extremes give {expected!r}"
            )

    def _reject(self, problems: list[str]) -> None:
        """Raise :class:`QuantileAnalysisError` naming every problem found."""
        raise QuantileAnalysisError(
            "QuantileReport("
            f"factor_name={self.factor_name!r}, factor_version={self.factor_version!r}, "
            f"horizon_sessions={self.horizon_sessions!r}, quantiles={self.quantiles!r}) is not a "
            "valid quantile report",
            problems,
        )

    def to_frame(self) -> pd.DataFrame:
        """Render the group observations as one table row per date and quantile.

        The columns are exactly :data:`QUANTILE_OBSERVATION_COLUMNS`, in that
        order, whatever the caller does with the frame afterwards.  Every date
        contributes ``quantiles`` rows; an empty group keeps its row, with
        ``member_count == 0`` and ``NaN`` in ``mean_forward_return`` -- the
        absence of a mean is reported, never hidden by dropping the row.  No
        per-symbol data is ever mixed into this table.

        Returns
        -------
        pandas.DataFrame
            A new frame; the report and its observations are untouched.
        """
        observations = self.observations
        columns = {
            "as_of_date": pd.Series(
                [item.as_of_date for item in observations], dtype="datetime64[s]"
            ),
            "end_date": pd.Series([item.end_date for item in observations], dtype="datetime64[s]"),
            "factor_name": pd.Series([item.factor_name for item in observations], dtype="object"),
            "factor_version": pd.Series(
                [item.factor_version for item in observations], dtype="object"
            ),
            "horizon_sessions": pd.Series(
                [item.horizon_sessions for item in observations], dtype="int64"
            ),
            "quantiles": pd.Series([item.quantiles for item in observations], dtype="int64"),
            "quantile": pd.Series([item.quantile for item in observations], dtype="int64"),
            "universe_size": pd.Series(
                [item.universe_size for item in observations], dtype="int64"
            ),
            "paired_count": pd.Series([item.paired_count for item in observations], dtype="int64"),
            "paired_coverage": pd.Series(
                [item.paired_coverage for item in observations], dtype="float64"
            ),
            "member_count": pd.Series([item.member_count for item in observations], dtype="int64"),
            "member_coverage": pd.Series(
                [item.member_coverage for item in observations], dtype="float64"
            ),
            "mean_forward_return": pd.Series(
                [item.mean_forward_return for item in observations], dtype="float64"
            ),
        }
        return pd.DataFrame(columns, columns=list(QUANTILE_OBSERVATION_COLUMNS))

    def spread_to_frame(self) -> pd.DataFrame:
        """Render the top-minus-bottom history as one table row per session.

        The columns are exactly :data:`QUANTILE_SPREAD_COLUMNS`, in that order.
        A session whose extreme group is empty keeps its row with ``NaN`` in the
        three return columns: the spread is undefined, which is not the same as
        zero.

        Returns
        -------
        pandas.DataFrame
            A new frame; the report and its spreads are untouched.
        """
        spreads = self.spreads
        columns = {
            "as_of_date": pd.Series([item.as_of_date for item in spreads], dtype="datetime64[s]"),
            "end_date": pd.Series([item.end_date for item in spreads], dtype="datetime64[s]"),
            "factor_name": pd.Series([item.factor_name for item in spreads], dtype="object"),
            "factor_version": pd.Series([item.factor_version for item in spreads], dtype="object"),
            "horizon_sessions": pd.Series(
                [item.horizon_sessions for item in spreads], dtype="int64"
            ),
            "quantiles": pd.Series([item.quantiles for item in spreads], dtype="int64"),
            "bottom_mean_forward_return": pd.Series(
                [item.bottom_mean_forward_return for item in spreads], dtype="float64"
            ),
            "top_mean_forward_return": pd.Series(
                [item.top_mean_forward_return for item in spreads], dtype="float64"
            ),
            "top_minus_bottom": pd.Series(
                [item.top_minus_bottom for item in spreads], dtype="float64"
            ),
        }
        return pd.DataFrame(columns, columns=list(QUANTILE_SPREAD_COLUMNS))

    def __repr__(self) -> str:
        """A one-line summary for logs; the observations stay out of it."""
        return (
            f"QuantileReport(factor={self.factor_name}@{self.factor_version}, "
            f"horizon_sessions={self.horizon_sessions}, quantiles={self.quantiles}, "
            f"observations={len(self.observations)}, spreads={len(self.spreads)})"
        )


# ---------------------------------------------------------------------------
# QuantileAnalyzer -- panels and labels in, one report out
# ---------------------------------------------------------------------------
def _items(items: object, kind: type, argument: str) -> list[object]:
    """``items`` as a list of ``kind`` objects, or a refusal naming what is wrong."""
    if isinstance(items, kind):
        raise QuantileAnalysisError(
            f"QuantileAnalyzer.analyze() needs an iterable of {kind.__name__} objects for "
            f"{argument}, got a single {type(items).__name__}: one panel is one session, and a "
            "group return needs every session with its own labels"
        )
    if isinstance(items, (str, bytes)) or not isinstance(items, Iterable):
        raise QuantileAnalysisError(
            f"QuantileAnalyzer.analyze() needs an iterable of {kind.__name__} objects for "
            f"{argument} (a tuple, a list or a generator), got {type(items).__name__}"
        )

    collected = list(items)
    if not collected:
        raise QuantileAnalysisError(
            f"QuantileAnalyzer.analyze() needs at least one {kind.__name__} in {argument}, got an "
            "empty sequence: a group return is measured per session, so an empty history has "
            "nothing to group"
        )

    problems = [
        f"{argument}[{position}] is a {type(item).__name__}, not a {kind.__name__}"
        for position, item in enumerate(collected)
        if not isinstance(item, kind)
    ]
    if problems:
        raise QuantileAnalysisError(
            f"QuantileAnalyzer.analyze() can only group {kind.__name__} objects: {argument} must "
            "hold nothing else",
            problems,
        )
    return collected


def _factor_index(panels: Sequence[FactorPanel], problems: list[str]) -> dict[date, FactorPanel]:
    """One panel per session, or the reasons a sequence is not one factor's history."""
    index: dict[date, FactorPanel] = {}
    identity = (panels[0].factor_name, panels[0].factor_version)
    first_seen: dict[date, int] = {}

    for position, panel in enumerate(panels):
        current = (panel.factor_name, panel.factor_version)
        if current != identity:
            problems.append(
                f"factor_panels[{position}] is {current[0]}/{current[1]}, but factor_panels[0] "
                f"is {identity[0]}/{identity[1]}: one call measures one factor at one version"
            )
            continue

        first = first_seen.setdefault(panel.as_of_date, position)
        if first != position:
            problems.append(
                f"factor_panels[{position}] repeats as_of_date "
                f"{panel.as_of_date.isoformat()} (already given by factor_panels[{first}]): the "
                "groups of a session are measured once"
            )
            continue

        index[panel.as_of_date] = panel
    return index


def _label_index(
    labels: Sequence[ForwardReturnPanel], problems: list[str]
) -> dict[date, ForwardReturnPanel]:
    """One label panel per session, or the reasons they are not one factor's labels."""
    index: dict[date, ForwardReturnPanel] = {}
    identity = (labels[0].factor_name, labels[0].factor_version, labels[0].horizon_sessions)
    first_seen: dict[date, int] = {}

    for position, panel in enumerate(labels):
        current = (panel.factor_name, panel.factor_version, panel.horizon_sessions)
        if current != identity:
            problems.append(
                f"forward_return_panels[{position}] is {current[0]}/{current[1]} at "
                f"horizon_sessions={current[2]}, but forward_return_panels[0] is "
                f"{identity[0]}/{identity[1]} at horizon_sessions={identity[2]}: one report "
                "measures one factor at one horizon"
            )
            continue

        first = first_seen.setdefault(panel.as_of_date, position)
        if first != position:
            problems.append(
                f"forward_return_panels[{position}] repeats as_of_date "
                f"{panel.as_of_date.isoformat()} (already given by forward_return_panels["
                f"{first}]): the groups of a session are measured once"
            )
            continue

        index[panel.as_of_date] = panel
    return index


def _universe_reason(factors: tuple[str, ...], labels: tuple[str, ...]) -> str:
    """Explain how two universes differ, for the error that refuses the pair."""
    if len(factors) != len(labels):
        return f"{len(factors)} symbol(s) in the panel against {len(labels)} in the labels"

    only_factors = sorted(set(factors) - set(labels))
    only_labels = sorted(set(labels) - set(factors))
    if only_factors or only_labels:
        return (
            f"symbols only in the panel: {_describe(only_factors)}; only in the labels: "
            f"{_describe(only_labels)}"
        )

    position = next(
        index
        for index, (factor, label) in enumerate(zip(factors, labels, strict=True))
        if factor != label
    )
    return (
        f"the same symbols in a different order (position {position}: {factors[position]} in "
        f"the panel, {labels[position]} in the labels)"
    )


class QuantileAnalyzer:
    """Measure the forward return of every factor quantile.

    The analyzer is a pure function of the panels it is given: no reader, no
    store, no calendar and no clock.  A caller holding panels (from a run, from
    ``ParquetFactorStore.read_range`` or from a test literal) and labels (from
    :class:`~factor_sys.forward_returns.ForwardReturnCalculator`) gets the same
    report for the same rows, today and a year from now.

    ``quantiles`` is the only policy knob: how many groups a cross-section is cut
    into.  It is a count of groups, not a list of edges, so the grouping can
    never depend on a threshold estimated from another session's sample.

    Examples
    --------
    >>> QuantileAnalyzer().quantiles
    5
    >>> repr(QuantileAnalyzer(quantiles=10))
    'QuantileAnalyzer(quantiles=10)'
    """

    __slots__ = ("_quantiles",)

    def __init__(self, quantiles: int = _DEFAULT_QUANTILES) -> None:
        """Store the group count, refusing one a comparison cannot use."""
        self._quantiles = _quantiles(quantiles)

    @property
    def quantiles(self) -> int:
        """Groups one cross-section is cut into (``2`` or more)."""
        return self._quantiles

    def analyze(
        self,
        factor_panels: Iterable[FactorPanel],
        forward_return_panels: Iterable[ForwardReturnPanel],
    ) -> QuantileReport:
        """Measure one factor's quantile history against one set of labels.

        The two sequences must describe **the same sessions**: every
        ``FactorPanel`` is paired with the ``ForwardReturnPanel`` of the same
        ``as_of_date`` -- both are keyed by that date, so a label can never be
        grouped with another session's panel -- and a date present on one side
        only is an error, not something to skip.  Analysing the intersection
        would hide exactly the broken join this analysis is supposed to catch.

        Within one session the two universes must be identical *and in the same
        order*: the labels were built from the panel's universe, so a difference
        means the two objects do not belong together, and re-ordering or
        intersecting symbols here would silently group the wrong stock.  Different
        sessions may perfectly well carry different universes: a point-in-time
        pool changes over time.

        Parameters
        ----------
        factor_panels:
            Any iterable of :class:`~factor_sys.contracts.FactorPanel` objects,
            one per session, in any order.
        forward_return_panels:
            Any iterable of
            :class:`~factor_sys.forward_returns.ForwardReturnPanel` objects, one
            per session of ``factor_panels``, in any order.

        Returns
        -------
        QuantileReport
            ``quantiles`` observations per session plus one spread per session,
            both in ascending ``as_of_date`` order.

        Raises
        ------
        factor_sys.errors.QuantileAnalysisError
            Every problem found, collected: a foreign object, an empty sequence,
            mixed identities or horizons, a repeated date, a date set that does
            not close, or a universe that moved between a panel and its labels.
        """
        panels = _items(factor_panels, FactorPanel, "factor_panels")
        labels = _items(forward_return_panels, ForwardReturnPanel, "forward_return_panels")

        problems: list[str] = []
        factors = _factor_index(panels, problems)
        labelled = _label_index(labels, problems)

        # -- the two sides must describe the same sessions ---------------------
        missing = sorted(set(factors) - set(labelled))
        extra = sorted(set(labelled) - set(factors))
        if missing:
            problems.append(
                "forward_return_panels has no labels for as_of_date "
                f"{_describe([day.isoformat() for day in missing])}: every factor panel needs "
                "its own labels, and analysing only the dates both sides happen to share would "
                "hide a broken join"
            )
        if extra:
            problems.append(
                "forward_return_panels has labels for as_of_date "
                f"{_describe([day.isoformat() for day in extra])} with no factor panel: a label "
                "without a factor value groups nothing"
            )

        # -- and one session must line up symbol for symbol --------------------
        for as_of in sorted(set(factors) & set(labelled)):
            panel = factors[as_of]
            label = labelled[as_of]
            identity = (panel.factor_name, panel.factor_version)
            if identity != (label.factor_name, label.factor_version):
                problems.append(
                    f"the panel and the labels for {as_of.isoformat()} disagree on identity: "
                    f"{identity[0]}/{identity[1]} against "
                    f"{label.factor_name}/{label.factor_version}"
                )
            if panel.universe != label.universe:
                problems.append(
                    f"the panel and the labels for {as_of.isoformat()} do not share a universe: "
                    f"{_universe_reason(panel.universe, label.universe)}"
                )

        if problems:
            raise QuantileAnalysisError(
                "QuantileAnalyzer.analyze() cannot join these panels into one quantile history",
                problems,
            )

        sessions = sorted(factors)
        observations: list[QuantileObservation] = []
        spreads: list[QuantileSpreadObservation] = []
        for day in sessions:
            groups, spread = self._observe(factors[day], labelled[day])
            observations.extend(groups)
            spreads.append(spread)

        first = factors[sessions[0]]
        return QuantileReport(
            factor_name=first.factor_name,
            factor_version=first.factor_version,
            horizon_sessions=labelled[sessions[0]].horizon_sessions,
            quantiles=self._quantiles,
            observations=tuple(observations),
            spreads=tuple(spreads),
        )

    def _observe(
        self, panel: FactorPanel, labels: ForwardReturnPanel
    ) -> tuple[tuple[QuantileObservation, ...], QuantileSpreadObservation]:
        """Measure one session: pair the usable symbols, group them, average.

        Every quantile is emitted, empty or not, so a caller reading the table
        never has to guess whether a missing row means "no members" or "not
        computed".  The two extremes are then subtracted with plain float
        arithmetic -- the same operation a reader would do with a calculator --
        and the difference is ``None`` as soon as either extreme is empty.
        """
        values, returns = _paired_values(panel, labels)
        quantiles = self._quantiles
        universe_size = len(panel.universe)
        paired_count = len(values)
        paired_coverage = paired_count / universe_size

        members: dict[int, list[float]] = {number: [] for number in range(1, quantiles + 1)}
        for number, forward in zip(_quantile_assignments(values, quantiles), returns, strict=True):
            members[number].append(forward)

        observations: list[QuantileObservation] = []
        for number in range(1, quantiles + 1):
            group = members[number]
            observations.append(
                QuantileObservation(
                    as_of_date=panel.as_of_date,
                    end_date=labels.end_date,
                    factor_name=panel.factor_name,
                    factor_version=panel.factor_version,
                    horizon_sessions=labels.horizon_sessions,
                    quantiles=quantiles,
                    quantile=number,
                    universe_size=universe_size,
                    paired_count=paired_count,
                    paired_coverage=paired_coverage,
                    member_count=len(group),
                    member_coverage=len(group) / universe_size,
                    mean_forward_return=_mean(group),
                )
            )

        bottom = observations[0].mean_forward_return
        top = observations[-1].mean_forward_return
        spread = QuantileSpreadObservation(
            as_of_date=panel.as_of_date,
            end_date=labels.end_date,
            factor_name=panel.factor_name,
            factor_version=panel.factor_version,
            horizon_sessions=labels.horizon_sessions,
            quantiles=quantiles,
            bottom_mean_forward_return=bottom,
            top_mean_forward_return=top,
            top_minus_bottom=_difference(top, bottom),
        )
        return tuple(observations), spread

    def __repr__(self) -> str:
        """``QuantileAnalyzer(quantiles=5)``: the group count is all the state."""
        return f"{type(self).__name__}(quantiles={self._quantiles})"


__all__ = [
    "QUANTILE_OBSERVATION_COLUMNS",
    "QUANTILE_SPREAD_COLUMNS",
    "QuantileAnalysisError",
    "QuantileAnalyzer",
    "QuantileObservation",
    "QuantileReport",
    "QuantileSpreadObservation",
]
