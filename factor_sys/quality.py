"""Factor result quality: the *soft* problems a structurally valid panel can carry.

What this module is
-------------------
``factor_sys.quality`` answers the one question about a
:class:`~factor_sys.contracts.FactorPanel` that the contract itself cannot
answer: *is this cross-section usable as it stands?*

Two very different things can be wrong with a factor's output, and they are
never confused here::

    hard contract error   the panel is malformed                -> exception
                          (wrong columns, a symbol outside the     (raised by
                          universe, a duplicated row, a           contracts or
                          non-numeric value, a missing date,      the runner)
                          a panel from another run)
    soft quality issue    the panel is well formed, but not     -> report
                          every symbol could be computed, or an   (returned,
                          eligible symbol holds NaN / inf)        never raised

The first kind is raised by :class:`~factor_sys.contracts.FactorContext`,
:class:`~factor_sys.contracts.FactorPanel` and the runner *before* a checker is
ever reached.  The second kind is **reported**: the panel is still a legal
panel, so discarding it is the caller's decision, not this module's.

Why a report instead of an exception
------------------------------------
A factor that could compute 40 of 50 symbols still produced 40 usable numbers.
Whether that is good enough depends on the strategy, the CLI and the backtest --
none of which exist yet -- so the runner must not turn a warning into a failure.
A checker therefore returns ``None`` (nothing to say) or a
:class:`FactorQualityReport` (at least one issue), and the runner attaches that
report to its :class:`~factor_sys.runner.FactorRunResult` unchanged.  Nothing in
the factor system escalates a quality issue to an exception; a later layer may.

Rules of the default checker
----------------------------
:class:`DefaultFactorQualityChecker` implements exactly four rules, all derived
from the panel's own columns -- no prices, no calendar, no second data source:

============================  ===========  ==============================================
code                          severity     raised when
============================  ===========  ==============================================
``zero_eligible_coverage``    ``error``    ``eligible_count == 0``: nothing was computed
``partial_eligible_coverage`` ``warning``  ``0 < eligible_count < universe_size``
``eligible_non_finite_value`` ``error``    an **eligible** row holds NaN, ``+inf`` or ``-inf``
``ineligible_infinite_value`` ``warning``  an **ineligible** row holds ``+inf`` or ``-inf``
============================  ===========  ==============================================

``is_eligible=False`` with ``NaN`` is the normal case and is never reported, and
``is_eligible=False`` with a finite value is not an error either: it is how some
formulas express "computed, but excluded" (a filter, a blacklist, a liquidity
cut-off).  Full coverage with finite eligible values produces ``None``.

What the default checker deliberately does not judge
----------------------------------------------------
Per-symbol missingness *is* worth a warning, but outliers are not: a value that
is large, small, negative, skewed or heavy-tailed is a matter for transforms
(``zscore``/``winsorize``), strategy constraints, IC analysis or a later
``QuickAnalysis`` pass, not for a structural check.  Industry concentration,
turnover, look-ahead and every comparison against future returns need data this
object never sees.  The checker therefore reads ``symbol``, ``raw_value`` and
``is_eligible`` and nothing else, and it never modifies the panel: it does not
reorder rows, fill NaN, clip extremes or recompute anything.

Purity
------
Pure, in-memory and offline: no file is read or written, no network client is
imported, no Parquet/DuckDB layer is involved and no concrete factor is named.
The same panel always yields the same report, so a stored panel can be re-checked
cheaply at any time.

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
...         "raw_value": [0.12, float("nan"), float("inf")],
...         "is_eligible": [True, False, False],
...     }
... )
>>> panel = FactorPanel(frame, universe=["AAPL", "MSFT", "NVDA"])
>>> report = DefaultFactorQualityChecker().inspect(panel)
>>> [issue.code for issue in report.issues]
['partial_eligible_coverage', 'ineligible_infinite_value']
>>> report.eligible_coverage == 1 / 3
True
"""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from typing import Literal

from factor_sys.contracts import FACTOR_NAME_PATTERN, FactorPanel, _normalize_date
from factor_sys.errors import FactorContractError

# ---------------------------------------------------------------------------
# Codes and severities
# ---------------------------------------------------------------------------
#: The only severities an issue may carry, least severe first.
SEVERITIES: tuple[str, ...] = ("warning", "error")

#: ``eligible_count == 0``: the factor computed nothing this run can use (error).
ZERO_ELIGIBLE_COVERAGE = "zero_eligible_coverage"

#: ``0 < eligible_count < universe_size``: some symbols are missing (warning).
PARTIAL_ELIGIBLE_COVERAGE = "partial_eligible_coverage"

#: An eligible row holds NaN, ``+inf`` or ``-inf``: it needs a number (error).
ELIGIBLE_NON_FINITE_VALUE = "eligible_non_finite_value"

#: An ineligible row holds ``+inf``/``-inf``: it should hold NaN instead (warning).
INELIGIBLE_INFINITE_VALUE = "ineligible_infinite_value"

#: A quality code is a snake_case identifier, never a sentence.
_CODE_PATTERN: str = r"[a-z][a-z0-9_]*"
_CODE_RE = re.compile(_CODE_PATTERN)

#: How many symbols a message renders before switching to a count.
_SYMBOL_LIMIT: int = 5

#: Panel columns the default checker reads (the contract's own names).
_SYMBOL_COLUMN = "symbol"
_VALUE_COLUMN = "raw_value"
_ELIGIBLE_COLUMN = "is_eligible"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _render(symbols: tuple[str, ...], limit: int = _SYMBOL_LIMIT) -> str:
    """Render ``symbols`` for a message: all of them, or the first few and a count."""
    if len(symbols) <= limit:
        return ", ".join(symbols)
    return f"{', '.join(symbols[:limit])}, ... ({len(symbols)} in total)"


def _is_integer(value: object) -> bool:
    """``True`` for a real integer: a ``bool`` is a flag, not a count."""
    return isinstance(value, int) and not isinstance(value, bool)


def _validated_symbols(values: object) -> tuple[tuple[str, ...], list[str]]:
    """``(symbols, problems)`` for an ``affected_symbols`` field."""
    if isinstance(values, str) or not isinstance(values, Iterable):
        return (), [
            "affected_symbols must be a sequence of symbol strings (use a one-element "
            f"tuple for a single symbol), got {values!r}"
        ]

    symbols = tuple(values)
    problems: list[str] = []

    malformed = [symbol for symbol in symbols if not isinstance(symbol, str) or not symbol.strip()]
    if malformed:
        problems.append(
            f"affected_symbols must hold non-empty symbol strings, got {malformed[:3]!r}"
        )

    seen: set[str] = set()
    repeated: list[str] = []
    for symbol in symbols:
        if not isinstance(symbol, str):
            continue
        if symbol in seen and symbol not in repeated:
            repeated.append(symbol)
        seen.add(symbol)
    if repeated:
        problems.append(
            f"affected_symbols must not repeat a symbol ({repeated[:3]!r}): one entry per "
            "affected symbol keeps the count honest"
        )

    return symbols, problems


# ---------------------------------------------------------------------------
# The two value objects
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FactorQualityIssue:
    """One soft problem found in one panel.

    An issue is a *description*, not a reaction: it carries a stable ``code``
    (machine-readable, snake_case), a ``severity`` and a human ``message``, plus
    the symbols it is about.  Nothing here decides whether the run may continue.

    Parameters
    ----------
    code:
        Stable identifier of the rule that fired, e.g.
        ``"partial_eligible_coverage"``.  Callers branch on this, never on
        ``message``.
    severity:
        ``"warning"`` (look at it) or ``"error"`` (this panel is not usable as it
        stands: nothing, or not a number, was computed where a number was
        promised).  Neither stops the runner.
    message:
        One readable sentence for a human, including the counts it is about.
    affected_symbols:
        The symbols the issue is about, in panel row order.  Empty when the
        issue is about the run as a whole rather than about named symbols.

    Examples
    --------
    >>> issue = FactorQualityIssue(
    ...     code="partial_eligible_coverage",
    ...     severity="warning",
    ...     message="1 of 2 symbol(s) are eligible",
    ...     affected_symbols=("MSFT",),
    ... )
    >>> issue.affected_symbols
    ('MSFT',)
    >>> str(issue)
    'warning: partial_eligible_coverage: 1 of 2 symbol(s) are eligible [MSFT]'
    """

    code: str
    severity: Literal["warning", "error"]
    message: str
    affected_symbols: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate the issue and freeze ``affected_symbols`` as a tuple."""
        problems: list[str] = []

        if not isinstance(self.code, str) or not _CODE_RE.fullmatch(self.code):
            problems.append(
                f"code {self.code!r} must be snake_case ({_CODE_PATTERN}): a stable "
                "identifier a caller can branch on, not a sentence"
            )
        if self.severity not in SEVERITIES:
            problems.append(f"severity must be one of {list(SEVERITIES)}, got {self.severity!r}")
        if not isinstance(self.message, str) or not self.message.strip():
            problems.append(f"message must be a non-empty string, got {self.message!r}")

        symbols, symbol_problems = _validated_symbols(self.affected_symbols)
        problems.extend(symbol_problems)

        if problems:
            raise FactorContractError(
                f"FactorQualityIssue(code={self.code!r}, severity={self.severity!r}) is not "
                "a valid quality issue",
                problems,
            )
        object.__setattr__(self, "affected_symbols", symbols)

    def __str__(self) -> str:
        """``severity: code: message [SYMBOL, ...]`` -- one readable line."""
        symbols = f" [{_render(self.affected_symbols)}]" if self.affected_symbols else ""
        return f"{self.severity}: {self.code}: {self.message}{symbols}"

    def __repr__(self) -> str:
        """A short, unambiguous form for logs (the message is left to ``str``)."""
        return (
            f"FactorQualityIssue(code={self.code!r}, severity={self.severity!r}, "
            f"affected_symbols={self.affected_symbols!r})"
        )


@dataclass(frozen=True)
class FactorQualityReport:
    """Everything one quality check found about one panel.

    A report exists only because something was found, so it always holds at
    least one :class:`FactorQualityIssue`; a clean panel produces ``None``
    instead of an empty report.  Identity fields are copied from the panel that
    was inspected, never invented, so a report can be matched back to the run it
    came from and a mismatch is a bug -- which is why it is rejected here.

    Parameters
    ----------
    as_of_date:
        The panel's session.
    factor_name / factor_version:
        The panel's identity.
    universe_size:
        How many symbols the factor was asked about (``len(panel.universe)``).
    eligible_count:
        How many rows carry ``is_eligible=True``.
    eligible_coverage:
        ``eligible_count / universe_size``, exactly -- a derived value, so a
        report cannot carry a coverage that contradicts its own counts.
    issues:
        At least one issue, in the order the checker found them.

    Examples
    --------
    >>> report = FactorQualityReport(
    ...     as_of_date=date(2024, 1, 5),
    ...     factor_name="momentum_12_1",
    ...     factor_version="1.0.0",
    ...     universe_size=2,
    ...     eligible_count=1,
    ...     eligible_coverage=0.5,
    ...     issues=(
    ...         FactorQualityIssue(
    ...             code="partial_eligible_coverage",
    ...             severity="warning",
    ...             message="1 of 2 symbol(s) are eligible",
    ...             affected_symbols=("MSFT",),
    ...         ),
    ...     ),
    ... )
    >>> report.eligible_coverage
    0.5
    >>> print(report)
    momentum_12_1 v1.0.0 2024-01-05: 1/2 eligible (50.0%)
      - warning: partial_eligible_coverage: 1 of 2 symbol(s) are eligible [MSFT]
    """

    as_of_date: date
    factor_name: str
    factor_version: str
    universe_size: int
    eligible_count: int
    eligible_coverage: float
    issues: tuple[FactorQualityIssue, ...]

    def __post_init__(self) -> None:
        """Validate the identity, the counts, the coverage and the issues."""
        problems: list[str] = []

        try:
            when: date | None = _normalize_date(self.as_of_date, "FactorQualityReport.as_of_date")
        except FactorContractError:
            when = None
            problems.append(
                "as_of_date must be a date or an ISO-8601 'YYYY-MM-DD' string, got "
                f"{self.as_of_date!r}"
            )

        if not isinstance(self.factor_name, str) or not re.fullmatch(
            FACTOR_NAME_PATTERN, self.factor_name
        ):
            problems.append(
                f"factor_name {self.factor_name!r} must be snake_case (matches "
                f"{FACTOR_NAME_PATTERN}): it must be the inspected panel's own name"
            )
        if not isinstance(self.factor_version, str) or not self.factor_version.strip():
            problems.append(
                f"factor_version must be a non-empty string, got {self.factor_version!r}"
            )

        expected_coverage: float | None = None
        if not _is_integer(self.universe_size) or self.universe_size < 1:
            problems.append(f"universe_size must be a positive integer, got {self.universe_size!r}")
        elif (
            not _is_integer(self.eligible_count)
            or self.eligible_count < 0
            or self.eligible_count > self.universe_size
        ):
            problems.append(
                "eligible_count must be an integer between 0 and universe_size "
                f"({self.universe_size}), got {self.eligible_count!r}"
            )
        else:
            expected_coverage = self.eligible_count / self.universe_size

        coverage: float | None = None
        if isinstance(self.eligible_coverage, bool) or not isinstance(
            self.eligible_coverage, (int, float)
        ):
            problems.append(
                f"eligible_coverage must be a number, got {type(self.eligible_coverage).__name__}"
            )
        else:
            coverage = float(self.eligible_coverage)
            if not math.isfinite(coverage) or not 0.0 <= coverage <= 1.0:
                problems.append(
                    f"eligible_coverage must lie in [0, 1], got {self.eligible_coverage!r}"
                )
                coverage = None
            elif expected_coverage is not None and coverage != expected_coverage:
                problems.append(
                    f"eligible_coverage is {coverage!r} but eligible_count / universe_size "
                    f"is {expected_coverage!r}: coverage is derived from the counts, never "
                    "carried on its own"
                )

        issues: tuple[FactorQualityIssue, ...] = ()
        if isinstance(self.issues, (str, bytes)) or not isinstance(self.issues, Iterable):
            problems.append(
                f"issues must be a sequence of FactorQualityIssue, got {type(self.issues).__name__}"
            )
        else:
            issues = tuple(self.issues)
            if not issues:
                problems.append(
                    "issues must not be empty: a report exists only because something was "
                    "found (a clean panel is reported as None, not as an empty report)"
                )
            offenders = [
                type(issue).__name__
                for issue in issues
                if not isinstance(issue, FactorQualityIssue)
            ]
            if offenders:
                problems.append(f"issues must hold FactorQualityIssue objects, got {offenders}")

        if problems:
            raise FactorContractError(
                f"FactorQualityReport for {self.factor_name!r} v{self.factor_version} on "
                f"{self.as_of_date!r} is not valid",
                problems,
            )

        object.__setattr__(self, "as_of_date", when)
        object.__setattr__(self, "eligible_coverage", coverage)
        object.__setattr__(self, "issues", issues)

    def __str__(self) -> str:
        """A short human summary: one header line, then one line per issue."""
        header = (
            f"{self.factor_name} v{self.factor_version} {self.as_of_date.isoformat()}: "
            f"{self.eligible_count}/{self.universe_size} eligible "
            f"({self.eligible_coverage:.1%})"
        )
        return "\n".join([header, *(f"  - {issue}" for issue in self.issues)])

    def __repr__(self) -> str:
        """One line, enough to log: identity, coverage and the issue codes."""
        codes = ", ".join(issue.code for issue in self.issues)
        return (
            f"FactorQualityReport(as_of_date={self.as_of_date.isoformat()}, "
            f"factor={self.factor_name!r} v{self.factor_version}, "
            f"eligible={self.eligible_count}/{self.universe_size} "
            f"({self.eligible_coverage:.1%}), issues=[{codes}])"
        )


# ---------------------------------------------------------------------------
# The checker interface and the default rule set
# ---------------------------------------------------------------------------
class FactorQualityChecker(ABC):
    """The replaceable question "is this panel usable as it stands?".

    One method, one panel in, ``None`` or a report out.  An implementation must
    not modify the panel, must not raise for a *quality* problem (that is what
    the report is for) and must not need anything but the panel: no reader, no
    file, no clock.

    A custom checker is how a project adds its own soft rules -- "an eligible
    symbol must have at least three peers in its sector", "at least half of the
    universe must be eligible" -- without the runner knowing them.

    Examples
    --------
    >>> class IgnoringChecker(FactorQualityChecker):
    ...     def inspect(self, panel):
    ...         return None
    >>> IgnoringChecker().inspect(None) is None
    True
    """

    @abstractmethod
    def inspect(self, panel: FactorPanel) -> FactorQualityReport | None:
        """Report what is wrong with ``panel``, or return ``None`` if nothing is."""

    def __repr__(self) -> str:
        """``ClassName()``: a checker is stateless, so its type says everything."""
        return f"{type(self).__name__}()"


class DefaultFactorQualityChecker(FactorQualityChecker):
    """The v1 rule set: coverage, and finiteness where it must hold.

    Four rules, checked in this fixed order so two runs of the same panel produce
    identical reports:

    1. ``zero_eligible_coverage`` (error) or ``partial_eligible_coverage``
       (warning), with every ``is_eligible=False`` symbol in row order;
    2. ``eligible_non_finite_value`` (error): an eligible row must hold a finite
       number -- NaN, ``+inf`` and ``-inf`` are all reported;
    3. ``ineligible_infinite_value`` (warning): an excluded row may be NaN (the
       normal case) but not infinite.

    Full coverage with finite eligible values is clean, so ``inspect`` returns
    ``None``.  The panel is read through its public ``frame`` property (a copy)
    and is never written to: no row is reordered, no NaN is filled, no extreme is
    clipped and no value is recomputed.
    """

    __slots__ = ()

    def inspect(self, panel: FactorPanel) -> FactorQualityReport | None:
        """Inspect ``panel`` without touching it; ``None`` means "nothing to report"."""
        if not isinstance(panel, FactorPanel):
            raise FactorContractError(
                f"{type(self).__name__}.inspect() needs a FactorPanel, got "
                f"{type(panel).__name__}: quality is a question about a validated "
                "cross-section, not about a raw frame"
            )

        rows = panel.frame
        symbols = tuple(rows[_SYMBOL_COLUMN].tolist())
        values = rows[_VALUE_COLUMN].tolist()
        flags = rows[_ELIGIBLE_COLUMN].tolist()

        eligible_count = 0
        ineligible: list[str] = []
        eligible_not_finite: list[str] = []
        ineligible_infinite: list[str] = []

        for symbol, value, eligible in zip(symbols, values, flags, strict=True):
            finite = math.isfinite(value)
            if eligible:
                eligible_count += 1
                if not finite:
                    eligible_not_finite.append(symbol)
                continue
            ineligible.append(symbol)
            if not finite and not math.isnan(value):
                ineligible_infinite.append(symbol)

        universe_size = len(symbols)
        coverage = eligible_count / universe_size

        issues: list[FactorQualityIssue] = []
        if eligible_count == 0:
            issues.append(
                FactorQualityIssue(
                    code=ZERO_ELIGIBLE_COVERAGE,
                    severity="error",
                    message=(
                        f"no symbol of the {universe_size}-symbol universe is eligible on "
                        f"{panel.as_of_date.isoformat()}: the factor computed nothing this "
                        "run can use"
                    ),
                    affected_symbols=tuple(ineligible),
                )
            )
        elif eligible_count < universe_size:
            issues.append(
                FactorQualityIssue(
                    code=PARTIAL_ELIGIBLE_COVERAGE,
                    severity="warning",
                    message=(
                        f"{eligible_count} of {universe_size} symbol(s) are eligible "
                        f"({coverage:.1%}): {len(ineligible)} symbol(s) produced no usable "
                        "value"
                    ),
                    affected_symbols=tuple(ineligible),
                )
            )

        if eligible_not_finite:
            issues.append(
                FactorQualityIssue(
                    code=ELIGIBLE_NON_FINITE_VALUE,
                    severity="error",
                    message=(
                        f"{len(eligible_not_finite)} eligible symbol(s) hold a non-finite "
                        "raw_value (NaN, +inf or -inf): an eligible row must carry a finite "
                        "number"
                    ),
                    affected_symbols=tuple(eligible_not_finite),
                )
            )

        if ineligible_infinite:
            issues.append(
                FactorQualityIssue(
                    code=INELIGIBLE_INFINITE_VALUE,
                    severity="warning",
                    message=(
                        f"{len(ineligible_infinite)} ineligible symbol(s) hold an infinite "
                        "raw_value: an excluded symbol should hold NaN, not +inf/-inf"
                    ),
                    affected_symbols=tuple(ineligible_infinite),
                )
            )

        if not issues:
            return None

        return FactorQualityReport(
            as_of_date=panel.as_of_date,
            factor_name=panel.factor_name,
            factor_version=panel.factor_version,
            universe_size=universe_size,
            eligible_count=eligible_count,
            eligible_coverage=coverage,
            issues=tuple(issues),
        )


__all__ = [
    "ELIGIBLE_NON_FINITE_VALUE",
    "INELIGIBLE_INFINITE_VALUE",
    "PARTIAL_ELIGIBLE_COVERAGE",
    "SEVERITIES",
    "ZERO_ELIGIBLE_COVERAGE",
    "DefaultFactorQualityChecker",
    "FactorQualityChecker",
    "FactorQualityIssue",
    "FactorQualityReport",
]
