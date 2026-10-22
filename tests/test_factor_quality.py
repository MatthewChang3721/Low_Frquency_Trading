"""Tests for the factor quality check (:mod:`factor_sys.quality`).

Everything here is in memory and offline: a panel is built from a literal frame,
the runner is wired to in-memory doubles and the checkers are plain objects, so
nothing reads ``data/``, calls a vendor or needs a clock.  The only test that
touches a file writes a panel to ``tmp_path`` through ``ParquetFactorStore``, to
prove that the runner's new return object still feeds the store unchanged.

Covered in this order:

* the two frozen value objects: validation, every rejected construction, and the
  readable ``repr``/``str`` a caller prints,
* the checker interface: it cannot be instantiated directly, the default checker
  implements it, and a custom checker really replaces the default rules,
* the default rules, one at a time and in combination: full coverage (``None``),
  zero coverage (error), partial coverage (warning, symbols in row order), a
  non-finite value on an eligible row (error), NaN on an ineligible row (silent),
  an infinite value on an ineligible row (warning), and the outliers that are
  deliberately *not* judged,
* purity: the panel the checker reads is byte-identical afterwards,
* the runner's last step: a checker never runs after a hard failure, a forged or
  stale report is refused, a ``FactorError`` is propagated unchanged, an
  ordinary exception is wrapped with its cause kept,
* ``ParquetFactorStore.write(runner.run(...).panel)`` still stores a panel,
* the architectural boundary of the module and the package exports.
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Sequence
from dataclasses import FrozenInstanceError
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import factor_sys
import factor_sys.quality as quality_module
import factor_sys.runner as runner_module
from factor_sys import (
    DefaultFactorQualityChecker,
    Factor,
    FactorContext,
    FactorContractError,
    FactorError,
    FactorPanel,
    FactorQualityChecker,
    FactorQualityIssue,
    FactorQualityReport,
    FactorRunError,
    FactorRunner,
    FactorSpec,
    MarketDataReader,
    MarketDataReadError,
    ParquetFactorStore,
    UniverseProvider,
    UniverseSnapshot,
)

#: The pool every in-memory provider answers with.
SYMBOLS: tuple[str, ...] = ("AAPL", "MSFT", "NVDA")

#: A real XNYS session (a Friday).
AS_OF: date = date(2024, 1, 5)

NAME: str = "momentum_12_1"
VERSION: str = "1.0.0"
SPEC = FactorSpec(name=NAME, version=VERSION, required_fields=("close",))

NAN = float("nan")
INF = float("inf")


# ---------------------------------------------------------------------------
# Builders: a panel, a report, a checker
# ---------------------------------------------------------------------------
def make_panel(
    *,
    values: Sequence[float] | None = None,
    eligible: Sequence[bool] | None = None,
    symbols: Sequence[str] = SYMBOLS,
    name: str = NAME,
    version: str = VERSION,
    as_of: date | str = AS_OF,
) -> FactorPanel:
    """A valid panel whose values, flags and row order the test chooses."""
    universe = list(symbols)
    count = len(universe)
    frame = pd.DataFrame(
        {
            "as_of_date": [pd.Timestamp(as_of)] * count,
            "symbol": universe,
            "factor_name": [name] * count,
            "factor_version": [version] * count,
            "raw_value": list(values) if values is not None else [0.5] * count,
            "is_eligible": list(eligible) if eligible is not None else [True] * count,
        }
    )
    return FactorPanel(frame, universe=universe, as_of_date=as_of)


def inspect(panel: FactorPanel) -> FactorQualityReport | None:
    """What the default checker says about ``panel``."""
    return DefaultFactorQualityChecker().inspect(panel)


def codes(report: FactorQualityReport | None) -> tuple[str, ...]:
    """The issue codes of ``report``, in order (``()`` for ``None``)."""
    return () if report is None else tuple(issue.code for issue in report.issues)


def report_about(panel: FactorPanel, *, code: str = "custom_rule") -> FactorQualityReport:
    """A valid report that describes exactly ``panel``."""
    return FactorQualityReport(
        as_of_date=panel.as_of_date,
        factor_name=panel.factor_name,
        factor_version=panel.factor_version,
        universe_size=len(panel.universe),
        eligible_count=len(panel.universe),
        eligible_coverage=1.0,
        issues=(
            FactorQualityIssue(
                code=code,
                severity="warning",
                message="a custom rule fired",
                affected_symbols=panel.symbols[:1],
            ),
        ),
    )


def forged(report: Any, field: str, value: Any) -> FactorQualityReport:
    """Bend one field of an already validated report, as a buggy checker might."""
    object.__setattr__(report, field, value)
    return report


# ---------------------------------------------------------------------------
# The two value objects
# ---------------------------------------------------------------------------
def test_an_issue_is_frozen_and_freezes_its_symbols() -> None:
    """A list becomes a tuple, and neither the issue nor the tuple can change."""
    issue = FactorQualityIssue(
        code="partial_eligible_coverage",
        severity="warning",
        message="1 of 2 symbol(s) are eligible",
        affected_symbols=["MSFT"],  # type: ignore[arg-type]
    )

    assert issue.affected_symbols == ("MSFT",)
    assert isinstance(issue.affected_symbols, tuple)
    with pytest.raises(FrozenInstanceError):
        issue.code = "another_rule"  # type: ignore[misc]


def test_an_issue_without_symbols_is_a_run_level_finding() -> None:
    """``affected_symbols`` is optional, and ``str`` reads as one line."""
    issue = FactorQualityIssue(code="run_level", severity="error", message="the run is odd")

    assert issue.affected_symbols == ()
    assert str(issue) == "error: run_level: the run is odd"
    assert repr(issue) == (
        "FactorQualityIssue(code='run_level', severity='error', affected_symbols=())"
    )


def test_a_report_is_frozen_and_freezes_its_issues() -> None:
    """The report and everything it holds are immutable, so it can be shared."""
    report = inspect(make_panel(eligible=[True, False, True]))

    assert isinstance(report.issues, tuple)
    assert isinstance(report.issues[0].affected_symbols, tuple)
    assert report.issues[0].affected_symbols == ("MSFT",)
    with pytest.raises(FrozenInstanceError):
        report.eligible_count = 3  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        report.issues = ()  # type: ignore[misc]


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"code": "Not Snaky", "severity": "warning", "message": "m"}, "must be snake_case"),
        ({"code": "rule", "severity": "fatal", "message": "m"}, "severity must be one of"),
        ({"code": "rule", "severity": None, "message": "m"}, "severity must be one of"),
        ({"code": "rule", "severity": "error", "message": "   "}, "message must be a non-empty"),
        ({"code": "rule", "severity": "error", "message": 42}, "message must be a non-empty"),
        (
            {"code": "rule", "severity": "error", "message": "m", "affected_symbols": "MSFT"},
            "affected_symbols must be a sequence",
        ),
        (
            {
                "code": "rule",
                "severity": "error",
                "message": "m",
                "affected_symbols": ("MSFT", "MSFT"),
            },
            "must not repeat",
        ),
        (
            {"code": "rule", "severity": "error", "message": "m", "affected_symbols": ("",)},
            "non-empty symbol strings",
        ),
    ],
)
def test_an_invalid_issue_is_refused(kwargs: dict[str, Any], expected: str) -> None:
    """An issue a caller cannot branch on (or read) must not exist at all."""
    with pytest.raises(FactorContractError) as excinfo:
        FactorQualityIssue(**kwargs)

    assert expected in str(excinfo.value)
    assert excinfo.value.problems


def report_kwargs() -> dict[str, Any]:
    """The keyword arguments of one valid report (a test breaks exactly one)."""
    issue = FactorQualityIssue(
        code="custom_rule",
        severity="warning",
        message="a custom rule fired",
        affected_symbols=("MSFT",),
    )
    return {
        "as_of_date": AS_OF,
        "factor_name": NAME,
        "factor_version": VERSION,
        "universe_size": 3,
        "eligible_count": 2,
        "eligible_coverage": 2 / 3,
        "issues": (issue,),
    }


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"issues": ()}, "issues must not be empty"),
        ({"issues": ("not an issue",)}, "must hold FactorQualityIssue objects"),
        ({"issues": "custom_rule"}, "issues must be a sequence"),
        ({"eligible_coverage": 0.5}, "coverage is derived"),
        ({"eligible_coverage": 1.0}, "coverage is derived"),
        ({"eligible_coverage": float("nan")}, "must lie in [0, 1]"),
        ({"eligible_coverage": float("inf")}, "must lie in [0, 1]"),
        ({"eligible_coverage": "two thirds"}, "must be a number"),
        ({"eligible_coverage": True}, "must be a number"),
        ({"eligible_count": -1}, "eligible_count must be an integer"),
        ({"eligible_count": True}, "eligible_count must be an integer"),
        ({"eligible_count": 4, "eligible_coverage": 4 / 3}, "eligible_count must be an integer"),
        (
            {"universe_size": 0, "eligible_count": 0, "eligible_coverage": 0.0},
            "universe_size must be a positive integer",
        ),
        ({"universe_size": 3.0}, "universe_size must be a positive integer"),
        ({"universe_size": True}, "universe_size must be a positive integer"),
        ({"factor_name": "Momentum 12-1"}, "must be snake_case"),
        ({"factor_name": None}, "must be snake_case"),
        ({"factor_version": "   "}, "factor_version must be a non-empty string"),
        ({"as_of_date": "05/01/2024"}, "as_of_date must be a date"),
        ({"as_of_date": 20240105}, "as_of_date must be a date"),
    ],
)
def test_an_invalid_report_is_refused(overrides: dict[str, Any], expected: str) -> None:
    """A report carries its own counts, so it must not be able to lie about them."""
    kwargs = report_kwargs() | overrides

    with pytest.raises(FactorContractError) as excinfo:
        FactorQualityReport(**kwargs)

    assert expected in str(excinfo.value)
    assert excinfo.value.problems


def test_a_report_normalizes_its_date_and_issues() -> None:
    """A datetime is a label, and an issue list is frozen into a tuple."""
    kwargs = report_kwargs() | {
        "as_of_date": pd.Timestamp(AS_OF).to_pydatetime(),
        "issues": [report_kwargs()["issues"][0]],
    }

    report = FactorQualityReport(**kwargs)

    assert report.as_of_date == AS_OF
    assert type(report.as_of_date) is date
    assert isinstance(report.issues, tuple)


def test_a_report_prints_its_identity_coverage_and_issues() -> None:
    """``repr`` is one log line; ``str`` is what ``print(report)`` shows."""
    report = inspect(make_panel(eligible=[True, False, True]))

    assert repr(report) == (
        "FactorQualityReport(as_of_date=2024-01-05, factor='momentum_12_1' v1.0.0, "
        "eligible=2/3 (66.7%), issues=[partial_eligible_coverage])"
    )
    lines = str(report).splitlines()
    assert lines[0] == "momentum_12_1 v1.0.0 2024-01-05: 2/3 eligible (66.7%)"
    assert lines[1] == (
        "  - warning: partial_eligible_coverage: 2 of 3 symbol(s) are eligible (66.7%): "
        "1 symbol(s) produced no usable value [MSFT]"
    )


# ---------------------------------------------------------------------------
# The checker interface
# ---------------------------------------------------------------------------
def test_the_checker_interface_cannot_be_instantiated() -> None:
    """A checker is an interface: ``inspect`` is the one thing that makes one."""
    with pytest.raises(TypeError) as excinfo:
        FactorQualityChecker()  # type: ignore[abstract]

    assert "inspect" in str(excinfo.value)

    class Incomplete(FactorQualityChecker):
        """A checker that forgot the only method a checker has."""

    with pytest.raises(TypeError) as excinfo:
        Incomplete()  # type: ignore[abstract]

    assert "inspect" in str(excinfo.value)


def test_the_default_checker_implements_the_interface() -> None:
    """The v1 rules are one implementation, not a special case in the runner."""
    checker = DefaultFactorQualityChecker()

    assert isinstance(checker, FactorQualityChecker)
    assert repr(checker) == "DefaultFactorQualityChecker()"
    assert str(checker) == "DefaultFactorQualityChecker()"


@pytest.mark.parametrize("value", [pd.DataFrame(), None, "panel", object()])
def test_the_default_checker_refuses_anything_but_a_panel(value: object) -> None:
    """Quality is a question about a validated cross-section, not a raw frame."""
    with pytest.raises(FactorContractError) as excinfo:
        DefaultFactorQualityChecker().inspect(value)  # type: ignore[arg-type]

    assert "needs a FactorPanel" in str(excinfo.value)
    assert type(value).__name__ in str(excinfo.value)


# ---------------------------------------------------------------------------
# Runner doubles: a provider, a reader and a factor that never touch a file
# ---------------------------------------------------------------------------
class FixedProvider(UniverseProvider):
    """A provider over a literal pool; it records the dates it was asked about."""

    def __init__(self, symbols: Sequence[str] = SYMBOLS) -> None:
        self._symbols = tuple(symbols)
        self.asked: list[date] = []

    def snapshot(self, as_of_date: date | str) -> UniverseSnapshot:
        moment = pd.Timestamp(as_of_date).date()
        self.asked.append(moment)
        return UniverseSnapshot(as_of_date=moment, symbols=self._symbols, source="quality test")


class FixedReader(MarketDataReader):
    """A reader that answers one session of bars, or raises what it was given."""

    def __init__(
        self,
        frame: pd.DataFrame | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self.frame = frame
        self.error = error
        self.asked: list[date] = []

    def read_history(
        self,
        snapshot: UniverseSnapshot,
        start_date: date,
        required_fields: Sequence[str],
    ) -> pd.DataFrame:
        self.asked.append(start_date)
        if self.error is not None:
            raise self.error
        if self.frame is not None:
            return self.frame.copy(deep=True)
        return session_bars(snapshot.symbols, tuple(required_fields), snapshot.as_of_date)


def session_bars(
    symbols: Sequence[str] = SYMBOLS,
    fields: Sequence[str] = ("close",),
    session: date = AS_OF,
) -> pd.DataFrame:
    """One bar per symbol on ``session``: enough for a factor, no history at all."""
    return pd.DataFrame(
        {
            "date": pd.Series([pd.Timestamp(session)] * len(symbols), dtype="datetime64[s]"),
            "symbol": list(symbols),
            **{field: [100.0] * len(symbols) for field in fields},
        }
    )


def panel_from(context: FactorContext, **choices: Any) -> FactorPanel:
    """A panel for ``context``, with the values and flags a test asks for."""
    return make_panel(symbols=context.universe, as_of=context.as_of_date, **choices)


class PanelFactor(Factor):
    """A factor that answers with whatever ``build`` returns (or raises ``error``)."""

    def __init__(
        self,
        build: Callable[[FactorContext], object] | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self.build = build
        self.error = error

    @property
    def spec(self) -> FactorSpec:
        return SPEC

    def compute(self, context: FactorContext) -> FactorPanel:
        if self.error is not None:
            raise self.error
        if self.build is not None:
            return self.build(context)  # type: ignore[return-value]
        return panel_from(context)


class StubChecker(FactorQualityChecker):
    """A checker double: it records the panels it was asked about, then answers."""

    def __init__(self, *, answer: object = None, error: Exception | None = None) -> None:
        self.answer = answer
        self.error = error
        self.calls: list[FactorPanel] = []

    def inspect(self, panel: FactorPanel) -> FactorQualityReport | None:
        self.calls.append(panel)
        if self.error is not None:
            raise self.error
        return self.answer  # type: ignore[return-value]


class MutatingChecker(FactorQualityChecker):
    """A checker that answers about the panel, then bends the report on purpose."""

    def __init__(self, mutate: Callable[[FactorQualityReport], object]) -> None:
        self.mutate = mutate
        self.calls: list[FactorPanel] = []

    def inspect(self, panel: FactorPanel) -> FactorQualityReport | None:
        self.calls.append(panel)
        return self.mutate(report_about(panel))  # type: ignore[return-value]


def make_runner(
    checker: FactorQualityChecker | None = None,
    *,
    symbols: Sequence[str] = SYMBOLS,
    reader: MarketDataReader | None = None,
) -> FactorRunner:
    """The real runner, wired to in-memory doubles."""
    return FactorRunner(
        FixedProvider(symbols),
        FixedReader() if reader is None else reader,
        quality_checker=checker,
    )


# ---------------------------------------------------------------------------
# Rule A: eligible coverage
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "values",
    [
        [0.1, 0.2, 0.3],
        [-1e9, 0.0, 1e9],  # large, small and negative are not quality problems
        [1e-12, -0.0, 1e6],
    ],
)
def test_full_coverage_with_finite_values_is_clean(values: list[float]) -> None:
    """Every symbol eligible with a finite number: the checker says nothing."""
    assert inspect(make_panel(values=values)) is None


def test_zero_eligible_coverage_is_an_error() -> None:
    """A factor that computed nothing for anyone is reported, not raised."""
    report = inspect(make_panel(values=[NAN, NAN, NAN], eligible=[False, False, False]))

    assert report is not None
    assert codes(report) == ("zero_eligible_coverage",)

    issue = report.issues[0]
    assert issue.severity == "error"
    assert issue.affected_symbols == SYMBOLS
    assert "no symbol of the 3-symbol universe is eligible" in issue.message
    assert report.eligible_count == 0
    assert report.universe_size == 3
    assert report.eligible_coverage == 0.0


def test_zero_coverage_does_not_need_missing_values() -> None:
    """Eligibility is the factor's decision, so a finite value can still be excluded."""
    report = inspect(make_panel(values=[0.1, 0.2, 0.3], eligible=[False, False, False]))

    assert codes(report) == ("zero_eligible_coverage",)
    assert report.eligible_coverage == 0.0


def test_partial_coverage_is_a_warning_with_the_symbols_in_row_order() -> None:
    """One symbol out of three is a warning, and it names exactly that symbol."""
    report = inspect(make_panel(values=[0.5, NAN, 0.5], eligible=[True, False, True]))

    assert report is not None
    assert codes(report) == ("partial_eligible_coverage",)

    issue = report.issues[0]
    assert issue.severity == "warning"
    assert issue.affected_symbols == ("MSFT",)
    assert "2 of 3 symbol(s) are eligible (66.7%)" in issue.message
    assert report.eligible_count == 2
    assert report.universe_size == 3
    assert report.eligible_coverage == pytest.approx(2 / 3)


def test_the_affected_symbols_follow_the_panels_row_order() -> None:
    """The report repeats the panel's order, not the alphabet's."""
    panel = make_panel(
        symbols=("NVDA", "AAPL", "MSFT"),
        values=[NAN, 0.5, NAN],
        eligible=[False, True, False],
    )

    report = inspect(panel)

    assert report.issues[0].affected_symbols == ("NVDA", "MSFT")


def test_an_ineligible_symbol_with_a_finite_value_is_only_a_coverage_warning() -> None:
    """A filter that keeps the value but drops the symbol is legal: warn, do not error."""
    report = inspect(make_panel(values=[0.5, 0.25, 0.5], eligible=[True, False, True]))

    assert codes(report) == ("partial_eligible_coverage",)
    assert [issue.severity for issue in report.issues] == ["warning"]


# ---------------------------------------------------------------------------
# Rule B: an eligible row must hold a finite number
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("value", [NAN, INF, -INF], ids=["nan", "+inf", "-inf"])
def test_an_eligible_non_finite_value_is_an_error(value: float) -> None:
    """An eligible row promises a number; NaN and either infinity break that promise."""
    panel = make_panel(symbols=("AAPL", "MSFT"), values=[0.5, value])

    report = inspect(panel)

    assert report is not None
    assert codes(report) == ("eligible_non_finite_value",)

    issue = report.issues[0]
    assert issue.severity == "error"
    assert issue.affected_symbols == ("MSFT",)
    assert report.eligible_count == 2
    assert report.eligible_coverage == 1.0


def test_every_eligible_non_finite_symbol_is_listed() -> None:
    """The issue names all of them at once, in row order."""
    report = inspect(make_panel(values=[NAN, INF, 0.5]))

    assert report.issues[0].affected_symbols == ("AAPL", "MSFT")
    assert "2 eligible symbol(s) hold a non-finite raw_value" in report.issues[0].message


# ---------------------------------------------------------------------------
# Rule C: an ineligible row may be NaN, but not infinite
# ---------------------------------------------------------------------------
def test_an_ineligible_nan_is_the_normal_case() -> None:
    """The usual "no value because the symbol was excluded" row is never reported."""
    report = inspect(make_panel(values=[0.5, NAN, 0.5], eligible=[True, False, True]))

    assert codes(report) == ("partial_eligible_coverage",)


@pytest.mark.parametrize("value", [INF, -INF], ids=["+inf", "-inf"])
def test_an_ineligible_infinite_value_is_a_warning(value: float) -> None:
    """An excluded row should hold NaN; an infinity there is a formula bug to look at."""
    panel = make_panel(values=[0.5, value, 0.5], eligible=[True, False, True])

    report = inspect(panel)

    assert report is not None
    assert codes(report) == ("partial_eligible_coverage", "ineligible_infinite_value")

    infinite = report.issues[1]
    assert infinite.severity == "warning"
    assert infinite.affected_symbols == ("MSFT",)
    assert "ineligible symbol(s) hold an infinite raw_value" in infinite.message


# ---------------------------------------------------------------------------
# Several problems at once, reported once and in a fixed order
# ---------------------------------------------------------------------------
def test_every_problem_is_reported_once_and_in_a_fixed_order() -> None:
    """One report lists all three findings, coverage first, symbols in row order."""
    panel = make_panel(values=[NAN, INF, NAN], eligible=[True, False, False])

    report = inspect(panel)

    assert report is not None
    assert codes(report) == (
        "partial_eligible_coverage",
        "eligible_non_finite_value",
        "ineligible_infinite_value",
    )
    assert report.issues[0].affected_symbols == ("MSFT", "NVDA")
    assert report.issues[1].affected_symbols == ("AAPL",)
    assert report.issues[2].affected_symbols == ("MSFT",)
    assert report.eligible_count == 1
    assert report.eligible_coverage == pytest.approx(1 / 3)
    # Deterministic: the same panel always yields the same report.
    assert inspect(panel) == report


def test_the_report_keeps_the_identity_of_the_panel_it_inspected() -> None:
    """Date, name, version, pool size, eligible count and coverage all come from the panel."""
    panel = make_panel(
        symbols=("MSFT", "AAPL"),
        name="value_ep",
        version="2.1.0",
        values=[0.5, NAN],
        eligible=[True, False],
        as_of="2024-02-02",
    )

    report = inspect(panel)

    assert report.as_of_date == date(2024, 2, 2)
    assert report.factor_name == "value_ep"
    assert report.factor_version == "2.1.0"
    assert report.universe_size == 2
    assert report.eligible_count == 1
    assert report.eligible_coverage == 0.5
    assert report.issues[0].affected_symbols == ("AAPL",)


def test_the_checker_never_modifies_the_panel() -> None:
    """The checker reads a copy: values, flags and row order are exactly as they were."""
    panel = make_panel(values=[NAN, INF, 0.5], eligible=[True, False, True])
    before = panel.frame

    report = inspect(panel)

    pd.testing.assert_frame_equal(panel.frame, before)
    assert panel.symbols == SYMBOLS
    assert len(panel) == 3
    assert report is not None
    assert report.universe_size == len(panel.universe)


# ---------------------------------------------------------------------------
# The runner: a quality check is the last step, and never a hard failure
# ---------------------------------------------------------------------------
def test_the_runner_uses_the_default_checker_when_none_is_given() -> None:
    """The default rules run on the panel the factor returned."""

    def build(context: FactorContext) -> FactorPanel:
        return panel_from(context, values=[0.5, NAN, 0.5], eligible=[True, False, True])

    result = make_runner().run(PanelFactor(build), AS_OF)

    assert result.quality_report is not None
    assert codes(result.quality_report) == ("partial_eligible_coverage",)
    assert result.quality_report.issues[0].affected_symbols == ("MSFT",)
    assert result.quality_report.as_of_date == result.panel.as_of_date
    assert result.quality_report.factor_name == result.panel.factor_name
    assert result.quality_report.factor_version == result.panel.factor_version


def test_a_custom_checker_replaces_the_default_rules() -> None:
    """The rules are the checker's, so swapping it silences them completely."""
    checker = StubChecker(answer=None)
    runner = make_runner(checker)

    result = runner.run(PanelFactor(), AS_OF)

    assert checker.calls == [result.panel]
    assert result.quality_report is None


def test_a_clean_run_carries_no_report() -> None:
    """Nothing to say is ``None``, never an empty report."""
    result = make_runner().run(PanelFactor(), AS_OF)

    assert result.quality_report is None
    assert result.panel.universe == SYMBOLS


def test_a_quality_error_does_not_fail_the_run() -> None:
    """Zero coverage is an issue, not an exception: the caller decides."""

    def build(context: FactorContext) -> FactorPanel:
        return panel_from(context, values=[NAN, NAN, NAN], eligible=[False, False, False])

    result = make_runner().run(PanelFactor(build), AS_OF)

    assert result.panel.universe == SYMBOLS
    assert codes(result.quality_report) == ("zero_eligible_coverage",)
    assert result.quality_report.issues[0].severity == "error"


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("read_error", MarketDataReadError),
        ("bad_context", FactorContractError),
        ("factor_error", FactorRunError),
        ("non_panel", FactorRunError),
        ("foreign_panel", FactorRunError),
        ("non_session", FactorRunError),
        ("unknown_factor", FactorRunError),
    ],
)
def test_a_hard_failure_never_reaches_the_checker(
    case: str,
    expected: type[Exception],
) -> None:
    """A broken read, context, factor or panel is refused before any inspection."""
    checker = StubChecker()
    factor: Any = PanelFactor()
    reader = FixedReader()
    as_of: Any = AS_OF

    if case == "read_error":
        reader = FixedReader(error=MarketDataReadError("no bars under the root"))
    elif case == "bad_context":
        reader = FixedReader(frame=session_bars(session=date(2024, 1, 8)))
    elif case == "factor_error":
        factor = PanelFactor(error=ValueError("boom"))
    elif case == "non_panel":
        factor = PanelFactor(lambda context: pd.DataFrame())
    elif case == "foreign_panel":
        factor = PanelFactor(lambda context: panel_from(context, as_of=date(2024, 1, 4)))
    elif case == "non_session":
        as_of = date(2024, 1, 6)  # a Saturday
    elif case == "unknown_factor":
        factor = object()

    with pytest.raises(expected):
        FactorRunner(FixedProvider(), reader, quality_checker=checker).run(factor, as_of)

    assert checker.calls == []


def test_a_report_about_the_panel_is_accepted() -> None:
    """A consistent custom report travels out unchanged."""
    checker = MutatingChecker(lambda report: report)

    result = make_runner(checker).run(PanelFactor(), AS_OF)

    assert result.quality_report == report_about(result.panel)
    assert checker.calls == [result.panel]


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("as_of_date", date(2024, 1, 4), "as_of_date is 2024-01-04"),
        ("factor_name", "value_ep", "factor_name is 'value_ep'"),
        ("factor_version", "9.9.9", "factor_version is '9.9.9'"),
        ("universe_size", 2, "universe_size is 2"),
    ],
)
def test_a_report_about_another_panel_is_refused(
    field: str,
    value: object,
    expected: str,
) -> None:
    """A checker cannot hand on a report about a panel this run did not return."""
    checker = MutatingChecker(lambda report: forged(report, field, value))

    with pytest.raises(FactorRunError) as excinfo:
        make_runner(checker).run(PanelFactor(), AS_OF)

    assert expected in str(excinfo.value)
    assert expected in excinfo.value.problems[0]
    assert len(checker.calls) == 1


def test_a_report_without_issues_is_refused() -> None:
    """A report that says nothing is a bug, not a clean run."""
    checker = MutatingChecker(lambda report: forged(report, "issues", ()))

    with pytest.raises(FactorRunError) as excinfo:
        make_runner(checker).run(PanelFactor(), AS_OF)

    assert "issues is empty" in str(excinfo.value)


@pytest.mark.parametrize("answer", ["clean", 42, pd.DataFrame(), FactorPanel])
def test_a_checker_must_answer_with_a_report_or_none(answer: object) -> None:
    """The answer is checked: a run result may not carry an arbitrary object."""
    checker = StubChecker(answer=answer)

    with pytest.raises(FactorRunError) as excinfo:
        make_runner(checker).run(PanelFactor(), AS_OF)

    assert "must return a FactorQualityReport or None" in str(excinfo.value)
    assert type(answer).__name__ in str(excinfo.value)


@pytest.mark.parametrize(
    "error",
    [
        FactorContractError("the checker refused the panel"),
        FactorRunError("a nested orchestration failure"),
        MarketDataReadError("a read performed inside the checker"),
    ],
)
def test_a_factor_system_error_from_the_checker_is_never_re_wrapped(error: FactorError) -> None:
    """A checker's own domain error keeps its type and its identity."""
    checker = StubChecker(error=error)

    with pytest.raises(type(error)) as excinfo:
        make_runner(checker).run(PanelFactor(), AS_OF)

    assert excinfo.value is error


def test_an_unrelated_checker_failure_is_wrapped_with_its_cause() -> None:
    """A bug inside a checker becomes an orchestration error, cause kept."""
    error = ValueError("boom in the checker")
    checker = StubChecker(error=error)

    with pytest.raises(FactorRunError) as excinfo:
        make_runner(checker).run(PanelFactor(), AS_OF)

    assert excinfo.value.__cause__ is error
    assert "ValueError" in str(excinfo.value)
    assert "boom in the checker" in str(excinfo.value)
    assert "momentum_12_1" in str(excinfo.value)
    assert "2024-01-05" in str(excinfo.value)


def test_the_run_result_still_feeds_the_store(tmp_path: Path) -> None:
    """``store.write(runner.run(...).panel)`` is still the supported pipeline."""
    result = make_runner().run(PanelFactor(), AS_OF)
    store = ParquetFactorStore(tmp_path / "panels")

    path = store.write(result.panel)
    again = store.read(NAME, VERSION, AS_OF)

    assert path.is_file()
    assert store.exists(NAME, VERSION, AS_OF) is True
    pd.testing.assert_frame_equal(again.frame, result.panel.frame)


# ---------------------------------------------------------------------------
# The module's own boundary
# ---------------------------------------------------------------------------
#: Modules ``quality.py`` may import: its own contract, the errors it raises and
#: the standard library it needs.  No pandas, no storage, no network, no factor.
ALLOWED_IMPORTS: frozenset[str] = frozenset(
    {
        "__future__",
        "abc",
        "collections.abc",
        "dataclasses",
        "datetime",
        "math",
        "re",
        "typing",
        "factor_sys.contracts",
        "factor_sys.errors",
    }
)

#: Module prefixes that would put another layer inside the quality check.
FORBIDDEN_PREFIXES: tuple[str, ...] = (
    "data_sys",
    "factors",
    "pandas",
    "numpy",
    "pyarrow",
    "duckdb",
    "yfinance",
    "requests",
    "httpx",
    "socket",
    "urllib",
)

#: Calls that would make a pure, in-memory module impure.
FORBIDDEN_CALLS: frozenset[str] = frozenset(
    {
        "__import__",
        "compile",
        "connect",
        "eval",
        "exec",
        "input",
        "open",
        "print",
        "read_bytes",
        "read_json",
        "read_parquet",
        "read_text",
        "to_csv",
        "to_parquet",
        "urlopen",
        "write_bytes",
        "write_text",
    }
)

#: The rule codes the default checker owns; the runner must not know them.
QUALITY_CODES: tuple[str, ...] = (
    "zero_eligible_coverage",
    "partial_eligible_coverage",
    "eligible_non_finite_value",
    "ineligible_infinite_value",
)

QUALITY_PATH = Path(quality_module.__file__)
RUNNER_PATH = Path(runner_module.__file__)


def parse(path: Path) -> ast.Module:
    """The syntax tree of ``path``."""
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def imported_modules(path: Path) -> set[str]:
    """Every module named by an import statement in ``path``."""
    names: set[str] = set()
    for node in ast.walk(parse(path)):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def called_names(path: Path) -> set[str]:
    """Every function or method the module calls, exactly as written."""
    return {ast.unparse(node.func) for node in ast.walk(parse(path)) if isinstance(node, ast.Call)}


def code_strings(path: Path) -> list[str]:
    """Every string literal that is code, not a docstring."""
    tree = parse(path)
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                docstrings.add(id(body[0].value))
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
    ]


def test_the_quality_module_imports_only_its_own_layer() -> None:
    """No data system, no storage, no columnar engine, no vendor client, no factor."""
    imported = imported_modules(QUALITY_PATH)

    assert imported <= ALLOWED_IMPORTS
    assert not [name for name in imported if name.startswith(FORBIDDEN_PREFIXES)]


def test_the_quality_module_never_reads_or_writes_anything() -> None:
    """A pure check: no file, no path, no print, no eval."""
    called = called_names(QUALITY_PATH)
    names = {node.id for node in ast.walk(parse(QUALITY_PATH)) if isinstance(node, ast.Name)}

    assert called & FORBIDDEN_CALLS == set()
    assert not names & {"open", "Path", "os", "shutil", "tempfile", "subprocess"}


def test_the_quality_module_hard_codes_no_data_directory() -> None:
    """No test may depend on the repository's own ``data/`` tree."""
    suspicious = [
        text
        for text in code_strings(QUALITY_PATH)
        if text == "data" or text.startswith(("data/", "data\\", "C:", "/home", "/Users"))
    ]

    assert suspicious == []


def test_the_quality_module_never_touches_the_network_or_duckdb() -> None:
    """No vendor client, no database engine: a panel is all the input there is."""
    attributes = {
        node.attr
        for node in ast.walk(parse(QUALITY_PATH))
        if isinstance(node, ast.Attribute) and isinstance(node.attr, str)
    }

    assert not attributes & {"urlopen", "connect", "read_html", "execute", "download"}
    assert not [name for name in imported_modules(QUALITY_PATH) if "duckdb" in name]


def test_the_quality_module_names_no_concrete_factor() -> None:
    """Quality is a property of a panel, not of momentum or of any other formula."""
    tree = parse(QUALITY_PATH)
    words = list(code_strings(QUALITY_PATH))
    words.extend(node.id for node in ast.walk(tree) if isinstance(node, ast.Name))

    assert not [word for word in words if "momentum" in word.lower()]


def test_the_runner_asks_the_checker_and_owns_no_rule() -> None:
    """The runner calls ``inspect``; the codes, the columns and the maths live here."""
    called = called_names(RUNNER_PATH)
    strings = code_strings(RUNNER_PATH)

    assert "checker.inspect" in called
    assert not [name for name in called if name.split(".")[-1] in {"isnan", "isinf", "isfinite"}]
    assert [code for code in QUALITY_CODES if code in strings] == []
    assert [text for text in strings if text in {"is_eligible", "raw_value"}] == []


def test_the_rule_codes_are_exported_constants() -> None:
    """Callers branch on constants, so a typo in a rule code cannot go unnoticed."""
    assert {
        quality_module.ZERO_ELIGIBLE_COVERAGE,
        quality_module.PARTIAL_ELIGIBLE_COVERAGE,
        quality_module.ELIGIBLE_NON_FINITE_VALUE,
        quality_module.INELIGIBLE_INFINITE_VALUE,
    } == set(QUALITY_CODES)
    assert quality_module.SEVERITIES == ("warning", "error")


def test_the_package_root_exports_the_quality_objects() -> None:
    """The package root is the one place to import the quality objects from."""
    assert factor_sys.FactorQualityIssue is quality_module.FactorQualityIssue
    assert factor_sys.FactorQualityReport is quality_module.FactorQualityReport
    assert factor_sys.FactorQualityChecker is quality_module.FactorQualityChecker
    assert factor_sys.DefaultFactorQualityChecker is quality_module.DefaultFactorQualityChecker
    assert factor_sys.FactorRunResult is runner_module.FactorRunResult

    for name in (
        "FactorQualityIssue",
        "FactorQualityReport",
        "FactorQualityChecker",
        "DefaultFactorQualityChecker",
        "FactorRunResult",
    ):
        assert name in factor_sys.__all__

    assert set(quality_module.__all__) == {
        "ELIGIBLE_NON_FINITE_VALUE",
        "INELIGIBLE_INFINITE_VALUE",
        "PARTIAL_ELIGIBLE_COVERAGE",
        "SEVERITIES",
        "ZERO_ELIGIBLE_COVERAGE",
        "DefaultFactorQualityChecker",
        "FactorQualityChecker",
        "FactorQualityIssue",
        "FactorQualityReport",
    }
