"""IC analysis: did higher factor values precede higher returns?

What this module is
-------------------
``factor_sys.ic_analysis`` turns a factor and its labels into evidence.  For one
decision date ``t`` and a horizon of ``h`` sessions it measures::

    IC(t) = SpearmanCorrelation(factor.raw_value(t), forward_return(t, h))

One session becomes one frozen :class:`ICObservation`; a history of sessions
becomes one frozen :class:`ICReport`.  :class:`ICAnalyzer` is the only object
here: it holds no data and reads nothing but the two panel sequences the caller
hands it.

Rank IC is a statistic, not a transformed factor
------------------------------------------------
The rank is an internal step of the correlation.  Inside one session's sample,
factor values and forward returns are each replaced by their average ranks, the
two rank vectors are correlated with Pearson's formula, and both rank vectors
are then thrown away.  Nothing is written back: no ``raw_value`` is replaced by
a score, no ``FactorPanel`` gains a ``rank``/``score`` column, no new factor is
produced and no transformed panel is stored.  Cross-sectional transforms
(``rank``, ``zscore``, winsorizing) are a different feature with a different
contract.

Which pairs count
-----------------
A symbol enters one session's sample only when all four hold:

===========================  ==============================================
``is_eligible`` is ``True``  the factor itself was willing to speak about it
``raw_value`` is finite       ``NaN``/``+inf``/``-inf`` is not a value
``is_available`` is ``True``  the label found two usable prices
``forward_return`` finite     a ``NaN`` label carries no information
===========================  ==============================================

``paired_count`` counts those symbols, and ``pair_coverage`` is
``paired_count / universe_size`` measured against the whole cross-section the
factor was computed for -- the same denominator
:meth:`factor_sys.quick_analysis.QuickAnalyzer.summarize` uses for its coverage,
so the two numbers can be read side by side.  Nothing is filled, imputed or
borrowed from a neighbouring session, and a symbol is never paired by luck of
row order: factor and labels must carry the *same* universe in the same order,
or the request is refused.

``None`` is a result, not an error
----------------------------------
``ic`` is ``None`` -- never ``NaN``, never an infinity -- in two cases:

* fewer than ``min_pairs`` usable pairs (``ICAnalyzer(min_pairs=2)`` by
  default): a correlation over a single point does not exist;
* the sample is constant on one side: when every usable factor value is the
  same, or every usable return is, the ranks have no variance and Spearman's
  correlation is undefined.

Both are normal outcomes of a legal cross-section.  The observation is still
reported, with its exact ``paired_count`` and ``pair_coverage``; it simply does
not enter the summary statistics.

Average ranks, so ties never depend on row order
------------------------------------------------
Tied factor values -- and tied returns, which happen as soon as a label repeats
a price -- share the **average** of the ranks they would occupy: not the first,
dense or ordinal rank.  A perfect monotone relationship therefore gives exactly
``+1.0`` and the reverse exactly ``-1.0``, ties or not, and shuffling the rows
of both inputs changes neither the ranks nor the IC.  The correlation is
Pearson's formula on those ranks, summed with ``math.fsum``: no SciPy, and no
dependence on the order the symbols happen to arrive in.

From one session to a judgement about a factor
----------------------------------------------
Only observations with a defined IC enter the summary:

===================  ==========================================================
``valid_ic_count``   sessions with ``ic is not None``
``mean_ic``          their arithmetic mean; ``None`` when there are none
``ic_std``           their **sample** standard deviation (``ddof=1``);
                     ``None`` for fewer than two sessions
``icir``             ``mean_ic / ic_std``; ``None`` for fewer than two sessions
                     and whenever ``ic_std == 0.0`` -- a zero denominator is
                     reported as "undefined", never as ``inf``
===================  ==========================================================

Observations are kept in strictly ascending ``as_of_date`` order, so a caller
never has to sort a generator and two input orders produce the same object.  A
calendar hole is preserved: a date without a panel is simply absent from the
report, because an IC for a session the factor never ran would be invented data.

IC is after the fact
--------------------
An IC is measured *after* a decision and can only look backwards at one.  This
module imports no market data, no factor store, no runner, no registry, no
quality checker, no calendar and no ``data_sys``, so it cannot read a bar,
cannot compute a label and cannot write anything.  What it produces is
evaluation output: it may be reported, plotted or compared between factors, but
it never travels back into a ``FactorContext``, a ``FactorPanel``, a factor
formula or a trading decision -- a factor that saw its own IC would be fitted to
its own evaluation sample.

Not implemented here, on purpose: quantile portfolios and the top-minus-bottom
spread, backtest returns, fills, costs and slippage, cross-sectional transforms
(``rank``/``zscore``/winsorize/fill), multi-factor combination, label or IC
storage and caching, batch scheduling, a CLI, annualized ICIR, t-statistics,
p-values and any industry or style neutralization.

Example
-------
>>> from datetime import date
>>> import pandas as pd
>>> from factor_sys.contracts import FactorPanel
>>> from factor_sys.forward_returns import ForwardReturnPanel
>>> symbols = ["AAPL", "MSFT", "NVDA"]
>>> def cross_section(values, returns):
...     frame = pd.DataFrame(
...         {
...             "as_of_date": [pd.Timestamp("2024-01-05")] * 3,
...             "symbol": symbols,
...             "factor_name": ["momentum_12_1"] * 3,
...             "factor_version": ["1.0.0"] * 3,
...             "raw_value": values,
...             "is_eligible": [True] * 3,
...         }
...     )
...     labels = frame.drop(columns=["raw_value", "is_eligible"]).assign(
...         end_date=pd.Timestamp("2024-01-08"),
...         horizon_sessions=1,
...         forward_return=returns,
...         is_available=True,
...     )
...     return (
...         FactorPanel(frame, universe=symbols),
...         ForwardReturnPanel(
...             labels,
...             universe=symbols,
...             as_of_date=date(2024, 1, 5),
...             end_date=date(2024, 1, 8),
...             factor_name="momentum_12_1",
...             factor_version="1.0.0",
...             horizon_sessions=1,
...         ),
...     )
>>> panel, labels = cross_section([0.1, 0.2, 0.3], [0.02, 0.01, -0.03])
>>> report = ICAnalyzer().analyze([panel], [labels])
>>> report.valid_ic_count, report.mean_ic, report.icir
(1, -1.0, None)
>>> report.to_frame()["ic"].tolist()
[-1.0]
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
from factor_sys.errors import ICAnalysisError
from factor_sys.forward_returns import ForwardReturnPanel

# ---------------------------------------------------------------------------
# Table shape
# ---------------------------------------------------------------------------
#: Fixed column order of :meth:`ICReport.to_frame` (one row per session).
IC_OBSERVATION_COLUMNS: tuple[str, ...] = (
    "as_of_date",
    "end_date",
    "factor_name",
    "factor_version",
    "horizon_sessions",
    "universe_size",
    "paired_count",
    "pair_coverage",
    "ic",
)

#: Panel and label columns this module reads (the contracts' own names).
_VALUE_COLUMN = "raw_value"
_ELIGIBLE_COLUMN = "is_eligible"
_RETURN_COLUMN = "forward_return"
_AVAILABLE_COLUMN = "is_available"

#: A factor name is the contract's snake_case identifier; reused, not redefined.
_NAME_RE = re.compile(FACTOR_NAME_PATTERN)

#: Fewest usable pairs a correlation can be measured on.
_DEFAULT_MIN_PAIRS = 2

#: How far a hand-written derived statistic may sit from the recomputed one.
_SUMMARY_TOLERANCE = 1e-12


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
            f"{label} must be a datetime.date, got a datetime {value!r}: an IC is measured for "
            "a session, not for an instant"
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


def _describe(items: Sequence[object], limit: int = 4) -> str:
    """Render a few items for an error message, saying how many were dropped."""
    shown = ", ".join(str(item) for item in items[:limit])
    if len(items) > limit:
        shown = f"{shown}, ... (+{len(items) - limit} more)"
    return shown or "none"


def _min_pairs(value: object) -> int:
    """``value`` as a pair threshold: an integer of at least two."""
    if not _is_count(value):
        raise ICAnalysisError(
            "ICAnalyzer.min_pairs must be an integer of at least "
            f"{_DEFAULT_MIN_PAIRS}, got {value!r} ({type(value).__name__}): two paired symbols "
            "are the smallest sample a correlation is defined on"
        )
    if value < _DEFAULT_MIN_PAIRS:
        raise ICAnalysisError(
            f"ICAnalyzer.min_pairs must be at least {_DEFAULT_MIN_PAIRS}, got {value}: a "
            "correlation over a single pair is undefined, so a threshold below two would ask "
            "for a number that cannot exist"
        )
    return int(value)


# ---------------------------------------------------------------------------
# ICObservation -- one session's rank IC
# ---------------------------------------------------------------------------
@dataclass(frozen=True, repr=False)
class ICObservation:
    """What one session's cross-section says about a factor.

    An observation is the smallest honest unit of IC evidence: one decision
    date, its label window, the identity of the factor measured, how much of the
    cross-section could actually be paired, and the correlation itself.  It is
    built by :class:`ICAnalyzer` and re-validated on construction, so a
    hand-written observation cannot claim a coverage its counts contradict or a
    correlation no sample could produce.

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
    universe_size:
        Rows of the cross-section: symbols the factor was asked about.
    paired_count:
        Symbols that had *both* a usable factor value and a usable label.
    pair_coverage:
        ``paired_count / universe_size``, checked against the counts.
    ic:
        The Spearman rank IC, or ``None`` when it is undefined -- fewer than two
        pairs, or a constant side.  When present it is finite and inside
        ``[-1, 1]``; it is never ``NaN``.

    Notes
    -----
    Observations are compared by value (they hold no frame), so two runs over
    the same inputs produce equal objects.
    """

    as_of_date: date
    end_date: date
    factor_name: str
    factor_version: str
    horizon_sessions: int
    universe_size: int
    paired_count: int
    pair_coverage: float
    ic: float | None

    def __post_init__(self) -> None:
        """Check the identity, the counts and the correlation, then freeze them."""
        problems: list[str] = []

        _plain_date(self.as_of_date, "ICObservation.as_of_date", problems)
        _plain_date(self.end_date, "ICObservation.end_date", problems)
        _factor_name(self.factor_name, "ICObservation.factor_name", problems)
        _factor_version(self.factor_version, "ICObservation.factor_version", problems)
        _horizon_sessions(self.horizon_sessions, "ICObservation.horizon_sessions", problems)
        _count(self.universe_size, "ICObservation.universe_size", problems)
        _count(self.paired_count, "ICObservation.paired_count", problems)

        coverage = self.pair_coverage
        if not _is_real(coverage) or not math.isfinite(float(coverage)):
            problems.append(
                "ICObservation.pair_coverage must be a finite real number, got "
                f"{coverage!r} ({type(coverage).__name__})"
            )

        ic = self.ic
        if ic is not None and (not _is_real(ic) or not math.isfinite(float(ic))):
            problems.append(
                "ICObservation.ic must be a finite real number or None, got "
                f"{ic!r} ({type(ic).__name__}): an undefined IC is None, never NaN or an "
                "infinity"
            )

        if problems:
            self._reject(problems)

        # -- fields that only make sense once the other ones are well typed ----
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
        if self.universe_size >= 1:
            expected_coverage = self.paired_count / self.universe_size
            if self.pair_coverage != expected_coverage:
                problems.append(
                    f"pair_coverage {self.pair_coverage!r} must be paired_count / universe_size "
                    f"= {expected_coverage!r}"
                )
        if self.ic is not None:
            if self.paired_count < _DEFAULT_MIN_PAIRS:
                problems.append(
                    f"ic {self.ic!r} needs at least {_DEFAULT_MIN_PAIRS} paired symbols, but "
                    f"paired_count is {self.paired_count}: a correlation over one pair does not "
                    "exist"
                )
            elif not -1.0 <= float(self.ic) <= 1.0:
                problems.append(
                    f"ic {float(self.ic)!r} must lie in [-1, 1]: a rank correlation cannot "
                    "leave that range"
                )

        if problems:
            self._reject(problems)

        object.__setattr__(self, "universe_size", int(self.universe_size))
        object.__setattr__(self, "paired_count", int(self.paired_count))
        object.__setattr__(self, "pair_coverage", float(self.pair_coverage))
        if self.ic is not None:
            object.__setattr__(self, "ic", float(self.ic))

    def _reject(self, problems: list[str]) -> None:
        """Raise :class:`ICAnalysisError` naming every problem found."""
        raise ICAnalysisError(
            "ICObservation("
            f"as_of_date={self.as_of_date!r}, factor_name={self.factor_name!r}) is not a valid "
            "single-session IC observation",
            problems,
        )

    def __repr__(self) -> str:
        """A one-line summary for logs; no per-symbol value is ever printed."""
        return (
            f"ICObservation(as_of_date={self.as_of_date.isoformat()!r}, "
            f"end_date={self.end_date.isoformat()!r}, "
            f"factor={self.factor_name}@{self.factor_version}, "
            f"horizon_sessions={self.horizon_sessions}, paired={self.paired_count}/"
            f"{self.universe_size}, coverage={self.pair_coverage!r}, ic={self.ic!r})"
        )


# ---------------------------------------------------------------------------
# Spearman IC and summary statistics
# ---------------------------------------------------------------------------
#: What each derived summary field means, for the error that rejects a wrong one.
_SUMMARY_MEANING: dict[str, str] = {
    "mean_ic": "the mean of the defined ICs",
    "ic_std": "their sample deviation (ddof=1), None for fewer than two",
    "icir": "mean_ic / ic_std, None when that deviation is missing or zero",
}


def _average_ranks(values: Sequence[float]) -> list[float]:
    """Average ranks of ``values``: ties share the mean of the ranks they cover.

    ``method="average"`` is the only tie rule Spearman's correlation is defined
    with: breaking ties by position would make the IC depend on the order the
    symbols happen to arrive in.  Ranks are half-integers, which is why the sums
    below are exact.
    """
    return pd.Series(values, dtype="float64").rank(method="average").tolist()


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    """Pearson's correlation of two rank vectors, or ``None`` when undefined.

    The deviations are centred and summed with ``math.fsum``, so the result is
    exactly rounded and independent of the order of the pairs.  A rank vector
    with no variance has no correlation to report -- that is ``None``, never a
    ``NaN`` -- and a perfect monotone relationship is returned as exactly
    ``+1.0`` or ``-1.0`` instead of ``0.9999999999999998``: with half-integer
    ranks the sums are exact, so Cauchy-Schwarz equality is *tested*
    (``covariance ** 2 == left_variance * right_variance``) rather than guessed
    from a tolerance.
    """
    count = len(left)
    left_mean = math.fsum(left) / count
    right_mean = math.fsum(right) / count
    left_deviations = [value - left_mean for value in left]
    right_deviations = [value - right_mean for value in right]

    covariance = math.fsum(a * b for a, b in zip(left_deviations, right_deviations, strict=True))
    left_variance = math.fsum(a * a for a in left_deviations)
    right_variance = math.fsum(b * b for b in right_deviations)
    if left_variance == 0.0 or right_variance == 0.0:
        return None

    if covariance * covariance == left_variance * right_variance:
        return 1.0 if covariance > 0.0 else -1.0

    value = covariance / (math.sqrt(left_variance) * math.sqrt(right_variance))
    if not math.isfinite(value):
        return None
    # Rounding can leave a near-perfect sample one ulp outside the range; the
    # clamp keeps the promise that a reported IC lies in [-1, 1].
    return min(1.0, max(-1.0, value))


def _spearman(factors: Sequence[float], returns: Sequence[float]) -> float | None:
    """The Spearman rank IC of two paired samples, or ``None`` when undefined."""
    if len(factors) < _DEFAULT_MIN_PAIRS:
        return None
    return _pearson(_average_ranks(factors), _average_ranks(returns))


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
    factors: list[float] = []
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
        factors.append(float(value))
        returns.append(float(forward))
    return factors, returns


def _summarize(
    observations: Sequence[ICObservation],
) -> tuple[int, float | None, float | None, float | None]:
    """``(valid_ic_count, mean_ic, ic_std, icir)`` of a whole history.

    Only defined ICs are summarized.  The deviation is the **sample** standard
    deviation (``ddof=1``): a history of ICs is a sample of the sessions the
    factor happened to run on, not a population.  A flat history has a deviation
    of ``0.0`` and therefore no ratio -- ``icir`` is ``None``, because ``inf``
    would be a number nobody could compare with another factor's.
    """
    values = [float(item.ic) for item in observations if item.ic is not None]
    count = len(values)
    if count == 0:
        return 0, None, None, None
    mean = math.fsum(values) / count
    if count == 1:
        return 1, mean, None, None
    deviation = math.sqrt(math.fsum((value - mean) ** 2 for value in values) / (count - 1))
    if deviation == 0.0:
        return count, mean, 0.0, None
    return count, mean, deviation, mean / deviation


def _agrees(reported: float | None, expected: float | None) -> bool:
    """``True`` when a hand-written statistic matches the recomputed one.

    Counts are compared exactly; derived floats are compared to within
    :data:`_SUMMARY_TOLERANCE`, because writing ``0.2`` by hand for a mean of
    ``0.20000000000000004`` is a rounding difference, not a contradiction.
    """
    if reported is None or expected is None:
        return reported is None and expected is None
    return math.isclose(
        float(reported), expected, rel_tol=_SUMMARY_TOLERANCE, abs_tol=_SUMMARY_TOLERANCE
    )


# ---------------------------------------------------------------------------
# ICReport -- a factor measured over a history of sessions
# ---------------------------------------------------------------------------
@dataclass(frozen=True, repr=False)
class ICReport:
    """A factor's IC history and the three numbers that summarize it.

    One report is **one** factor at **one** version and **one** horizon, measured
    over a sequence of decision dates.  It keeps every observation -- including
    the sessions whose IC is undefined -- because ``paired_count`` and
    ``pair_coverage`` are evidence in their own right, and it keeps them in
    strictly ascending ``as_of_date`` order.

    The summary fields are derived, and a hand-built report is checked against
    its own observations: the count must match, every average must agree with
    the observations to within rounding, and the ``None`` boundaries below must
    be respected.  A report that disagrees with itself cannot exist.

    Parameters
    ----------
    factor_name, factor_version, horizon_sessions:
        The identity every observation must share.
    observations:
        One :class:`ICObservation` per measured session, strictly ascending by
        ``as_of_date``, at least one.
    valid_ic_count:
        How many observations have a defined ``ic``.
    mean_ic:
        The mean of those ICs, or ``None`` when there is none.
    ic_std:
        Their **sample** standard deviation (``ddof=1``): ``None`` for fewer than
        two valid ICs, and ``0.0`` when they are all equal.
    icir:
        ``mean_ic / ic_std``, or ``None`` when the deviation is undefined or
        zero.  An ICIR is never ``inf``: a flat IC history has no ratio.

    Notes
    -----
    ``to_frame`` renders the observations as a table; the object itself stays the
    canonical result.
    """

    factor_name: str
    factor_version: str
    horizon_sessions: int
    observations: tuple[ICObservation, ...]
    valid_ic_count: int
    mean_ic: float | None
    ic_std: float | None
    icir: float | None

    def __post_init__(self) -> None:
        """Check the identity, the observations and their summary, then freeze."""
        problems: list[str] = []

        name = _factor_name(self.factor_name, "ICReport.factor_name", problems)
        version = _factor_version(self.factor_version, "ICReport.factor_version", problems)
        horizon = _horizon_sessions(self.horizon_sessions, "ICReport.horizon_sessions", problems)
        count = _count(self.valid_ic_count, "ICReport.valid_ic_count", problems)

        observations = self._collected_observations(problems)

        for label in ("mean_ic", "ic_std", "icir"):
            value = getattr(self, label)
            if value is None:
                continue
            if not _is_real(value) or not math.isfinite(float(value)):
                problems.append(
                    f"ICReport.{label} must be a finite real number or None, got "
                    f"{value!r} ({type(value).__name__})"
                )

        if problems:
            self._reject(problems)

        # -- identity and date order ------------------------------------------
        for position, observation in enumerate(observations):
            if (observation.factor_name, observation.factor_version) != (name, version):
                problems.append(
                    f"observations[{position}] is {observation.factor_name}/"
                    f"{observation.factor_version}, but this report is {name}/{version}: one "
                    "report measures one factor at one version"
                )
            if observation.horizon_sessions != horizon:
                problems.append(
                    f"observations[{position}] was measured over "
                    f"{observation.horizon_sessions} session(s), but this report says "
                    f"{horizon}: one report measures one horizon"
                )
            if position == 0:
                continue
            previous = observations[position - 1]
            if observation.as_of_date == previous.as_of_date:
                problems.append(
                    f"observations[{position}] repeats as_of_date "
                    f"{observation.as_of_date.isoformat()} (already given by "
                    f"observations[{position - 1}]): one IC is measured per session"
                )
            elif observation.as_of_date < previous.as_of_date:
                problems.append(
                    f"observations[{position}] is dated "
                    f"{observation.as_of_date.isoformat()}, before observations[{position - 1}] "
                    f"({previous.as_of_date.isoformat()}): observations are kept in ascending "
                    "date order"
                )

        if problems:
            self._reject(problems)

        # -- the summary must describe exactly these observations --------------
        expected_count, expected_mean, expected_std, expected_icir = _summarize(observations)
        if count != expected_count:
            problems.append(
                f"valid_ic_count {count} must be {expected_count}: only observations with a "
                "defined ic enter the summary"
            )
        for label, reported, expected in (
            ("mean_ic", self.mean_ic, expected_mean),
            ("ic_std", self.ic_std, expected_std),
            ("icir", self.icir, expected_icir),
        ):
            if _agrees(reported, expected):
                continue
            problems.append(
                f"ICReport.{label} must be {expected!r} for these observations "
                f"(valid_ic_count {expected_count}): {_SUMMARY_MEANING[label]}"
            )

        if problems:
            self._reject(problems)

        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "valid_ic_count", int(count))
        for label in ("mean_ic", "ic_std", "icir"):
            value = getattr(self, label)
            if value is not None:
                object.__setattr__(self, label, float(value))

    def _collected_observations(self, problems: list[str]) -> tuple[ICObservation, ...]:
        """``self.observations`` as a tuple of observations, plus what was wrong."""
        observations = self.observations
        if isinstance(observations, ICObservation):
            problems.append(
                "ICReport.observations must be a sequence of ICObservation objects, got a "
                "single one: wrap it as (observation,) -- one session is one table row"
            )
            return (observations,)
        if isinstance(observations, (str, bytes)) or not isinstance(observations, Iterable):
            problems.append(
                "ICReport.observations must be a sequence of ICObservation objects, got "
                f"{type(observations).__name__}"
            )
            return ()

        collected = tuple(observations)
        if not collected:
            problems.append(
                "ICReport.observations must hold at least one observation: a report of no "
                "session summarizes nothing"
            )
            return ()

        usable: list[ICObservation] = []
        for position, item in enumerate(collected):
            if not isinstance(item, ICObservation):
                problems.append(
                    f"observations[{position}] is a {type(item).__name__}, not an ICObservation:"
                    " a report is built from single-session observations, never from panels or "
                    "frames"
                )
                continue
            usable.append(item)
        return tuple(usable)

    def _reject(self, problems: list[str]) -> None:
        """Raise :class:`ICAnalysisError` naming every problem found."""
        raise ICAnalysisError(
            f"ICReport(factor_name={self.factor_name!r}, horizon_sessions="
            f"{self.horizon_sessions!r}) is not a valid IC summary",
            problems,
        )

    def to_frame(self) -> pd.DataFrame:
        """Render the observations as one table row per session.

        The columns are exactly :data:`IC_OBSERVATION_COLUMNS`, in that order,
        whatever the caller does with the frame afterwards.  A session whose
        ``ic`` is undefined keeps its row and holds ``NaN`` in that column: the
        absence of a number is reported, never hidden by dropping the row, and
        no per-symbol data is ever mixed into this table.

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
            "universe_size": pd.Series(
                [item.universe_size for item in observations], dtype="int64"
            ),
            "paired_count": pd.Series([item.paired_count for item in observations], dtype="int64"),
            "pair_coverage": pd.Series(
                [item.pair_coverage for item in observations], dtype="float64"
            ),
            "ic": pd.Series([item.ic for item in observations], dtype="float64"),
        }
        return pd.DataFrame(columns, columns=list(IC_OBSERVATION_COLUMNS))

    def __repr__(self) -> str:
        """A one-line summary for logs; the observations stay out of it."""
        return (
            f"ICReport(factor={self.factor_name}@{self.factor_version}, "
            f"horizon_sessions={self.horizon_sessions}, "
            f"observations={len(self.observations)}, valid_ic_count={self.valid_ic_count}, "
            f"mean_ic={self.mean_ic!r}, ic_std={self.ic_std!r}, icir={self.icir!r})"
        )


# ---------------------------------------------------------------------------
# ICAnalyzer -- panels and labels in, one report out
# ---------------------------------------------------------------------------
def _items(items: object, kind: type, argument: str) -> list[object]:
    """``items`` as a list of ``kind`` objects, or a refusal naming what is wrong."""
    if isinstance(items, kind):
        raise ICAnalysisError(
            f"ICAnalyzer.analyze() needs an iterable of {kind.__name__} objects for {argument}, "
            f"got a single {type(items).__name__}: one panel is one session, and an IC pairs "
            "every session with its labels"
        )
    if isinstance(items, (str, bytes)) or not isinstance(items, Iterable):
        raise ICAnalysisError(
            f"ICAnalyzer.analyze() needs an iterable of {kind.__name__} objects for {argument} "
            f"(a tuple, a list or a generator), got {type(items).__name__}"
        )

    collected = list(items)
    if not collected:
        raise ICAnalysisError(
            f"ICAnalyzer.analyze() needs at least one {kind.__name__} in {argument}, got an "
            "empty sequence: an IC is measured per session, so an empty history has nothing to "
            "summarize"
        )

    problems = [
        f"{argument}[{position}] is a {type(item).__name__}, not a {kind.__name__}"
        for position, item in enumerate(collected)
        if not isinstance(item, kind)
    ]
    if problems:
        raise ICAnalysisError(
            f"ICAnalyzer.analyze() can only pair {kind.__name__} objects: {argument} must hold "
            "nothing else",
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
                f"{panel.as_of_date.isoformat()} (already given by factor_panels[{first}]): an "
                "IC is measured once per session"
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
                f"{first}]): an IC is measured once per session"
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


class ICAnalyzer:
    """Measure a factor's rank IC against its forward-return labels.

    The analyzer is a pure function of the panels it is given: no reader, no
    store, no calendar and no clock.  A caller holding panels (from a run, from
    ``ParquetFactorStore.read_range`` or from a test literal) and labels (from
    :class:`~factor_sys.forward_returns.ForwardReturnCalculator`) gets the same
    report for the same rows, today and a year from now.

    ``min_pairs`` is the only policy knob.  It is the number of usable pairs a
    session needs before an IC is reported for it; a session below the threshold
    is not dropped, it is reported with its exact ``paired_count``,
    ``pair_coverage`` and ``ic=None``.

    Examples
    --------
    >>> ICAnalyzer().min_pairs
    2
    >>> repr(ICAnalyzer(min_pairs=5))
    'ICAnalyzer(min_pairs=5)'
    """

    __slots__ = ("_min_pairs",)

    def __init__(self, min_pairs: int = _DEFAULT_MIN_PAIRS) -> None:
        """Store the pair threshold, refusing one a correlation cannot use."""
        self._min_pairs = _min_pairs(min_pairs)

    @property
    def min_pairs(self) -> int:
        """Usable pairs a session needs before its IC is computed."""
        return self._min_pairs

    def analyze(
        self,
        factor_panels: Iterable[FactorPanel],
        forward_return_panels: Iterable[ForwardReturnPanel],
    ) -> ICReport:
        """Measure one factor's IC history against one set of labels.

        The two sequences must describe **the same sessions**: every
        ``FactorPanel`` is paired with the ``ForwardReturnPanel`` of the same
        ``as_of_date`` -- both are keyed by that date, so a label can never be
        paired with another session's panel -- and a date present on one side
        only is an error, not something to skip.  Analysing the intersection
        would hide exactly the broken join this analysis is supposed to catch.

        Within one session the two universes must be identical *and in the same
        order*: the labels were built from the panel's universe, so a difference
        means the two objects do not belong together, and re-ordering or
        intersecting symbols here would silently pair values with the wrong
        stock.  Different sessions may perfectly well carry different universes:
        a point-in-time pool changes over time.

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
        ICReport
            One observation per session in ascending ``as_of_date`` order, plus
            the mean IC, the sample deviation (``ddof=1``) and the ICIR.

        Raises
        ------
        ICAnalysisError
            If either argument is not an iterable of the right panel type, is
            empty, mixes factor names, versions or horizons, repeats a date, or
            if the two sides disagree on dates, identity or universe.  Every
            problem found is reported at once, the way the rest of the factor
            system collects them.  An undefined IC is **not** one of them: it is
            a result, and travels back as ``ic=None``.
        """
        panels = _items(factor_panels, FactorPanel, "factor_panels")
        labels = _items(forward_return_panels, ForwardReturnPanel, "forward_return_panels")

        problems: list[str] = []
        factors = _factor_index(panels, problems)
        labelled = _label_index(labels, problems)
        if problems:
            raise ICAnalysisError(
                "ICAnalyzer.analyze() cannot read these inputs as one factor history and its "
                "labels",
                problems,
            )

        # -- the two sequences must describe exactly the same sessions --------
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
                "without a factor value measures nothing"
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
            raise ICAnalysisError(
                "ICAnalyzer.analyze() cannot join these panels into one IC history", problems
            )

        sessions = sorted(factors)
        observations = tuple(self._observe(factors[day], labelled[day]) for day in sessions)
        valid_ic_count, mean_ic, ic_std, icir = _summarize(observations)
        first = factors[sessions[0]]
        return ICReport(
            factor_name=first.factor_name,
            factor_version=first.factor_version,
            horizon_sessions=labelled[sessions[0]].horizon_sessions,
            observations=observations,
            valid_ic_count=valid_ic_count,
            mean_ic=mean_ic,
            ic_std=ic_std,
            icir=icir,
        )

    def _observe(self, panel: FactorPanel, labels: ForwardReturnPanel) -> ICObservation:
        """Measure one session: pair the usable symbols, rank them, correlate."""
        factors, returns = _paired_values(panel, labels)
        paired_count = len(factors)
        ic = _spearman(factors, returns) if paired_count >= self._min_pairs else None
        universe_size = len(panel.universe)
        return ICObservation(
            as_of_date=panel.as_of_date,
            end_date=labels.end_date,
            factor_name=panel.factor_name,
            factor_version=panel.factor_version,
            horizon_sessions=labels.horizon_sessions,
            universe_size=universe_size,
            paired_count=paired_count,
            pair_coverage=paired_count / universe_size,
            ic=ic,
        )

    def __repr__(self) -> str:
        """``ICAnalyzer(min_pairs=2)``: the threshold is all the state there is."""
        return f"{type(self).__name__}(min_pairs={self._min_pairs})"


__all__ = [
    "IC_OBSERVATION_COLUMNS",
    "ICAnalysisError",
    "ICAnalyzer",
    "ICObservation",
    "ICReport",
]
