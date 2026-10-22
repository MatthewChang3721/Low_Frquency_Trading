"""Tests for quick cross-sectional statistics (:mod:`factor_sys.quick_analysis`).

Everything here is in memory and offline: a panel is built from a literal frame
inside the test, and the analyzer is a plain object, so nothing reads ``data/``,
calls a vendor, writes a file or needs a clock.  The statistics are re-derived
independently with the standard library (``statistics.fmean``/``pstdev``) where
the expected number is not exactly representable, so a change of convention --
sample instead of population deviation, say -- fails a test instead of silently
moving a number.

Covered in this order:

* the frozen statistics object: its fields, its equality, the ``repr`` a log
  prints, and every construction it must refuse (counts that disagree with each
  other, a coverage or a rate that contradicts them, a ``NaN`` or infinite
  moment, a missing moment where a number is promised, a negative deviation, a
  single value with a non-zero deviation),
* ``summarize``: exact statistics for fully valid panels, ineligible rows and
  non-finite values both counted as missing, no usable value at all (four
  ``None``), one usable value (``std == 0.0``), the population convention, and
  values at the float limit (``1e308``) measured finitely instead of overflowing,
  plus the rejection of anything that is not a ``FactorPanel``,
* ``summarize_history``: date order independent of input order, normal date
  holes, ``()`` for empty input, list/tuple/generator equivalence, a changing
  universe, and every inconsistency it refuses (a foreign factor, a foreign
  version, a repeated date, a non-panel, a single panel, a non-iterable),
* ``to_frame``: fixed column order, the values and dtypes of the table, the
  schema of an empty history, order preservation, and the input it refuses,
* purity and determinism: the panel, the input list and the statistics survive a
  call unchanged, and two calls always agree,
* the error type and the package exports,
* the architectural boundary of the module: no store, no runner, no data system,
  no registry, no quality checker, no columnar engine, no file, no network.
"""

from __future__ import annotations

import ast
import math
import statistics as stdlib_statistics
from collections.abc import Sequence
from dataclasses import FrozenInstanceError
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import factor_sys
import factor_sys.quick_analysis as quick_analysis_module
from factor_sys import (
    FactorError,
    FactorPanel,
    FactorPanelStatistics,
    QuickAnalysisError,
    QuickAnalyzer,
)
from factor_sys.quick_analysis import STATISTICS_COLUMNS

#: A three-symbol pool (the small, usual case).
SYMBOLS: tuple[str, ...] = ("AAPL", "MSFT", "NVDA")

#: A five-symbol pool, for the "the universe is allowed to change" test.
WIDER: tuple[str, ...] = ("AAPL", "AMD", "MSFT", "NVDA", "TSLA")

AS_OF: date = date(2024, 1, 5)
LATER: date = date(2024, 1, 8)
LATEST: date = date(2024, 1, 10)

NAME: str = "momentum_12_1"
VERSION: str = "1.0.0"

NAN = float("nan")
INF = float("inf")


# ---------------------------------------------------------------------------
# Builders
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
    """A valid panel whose values, flags, pool, identity and date the test chooses."""
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


def describe(**kwargs: Any) -> FactorPanelStatistics:
    """The statistics of one panel built from ``kwargs`` (see :func:`make_panel`)."""
    return QuickAnalyzer().summarize(make_panel(**kwargs))


#: A valid statistics object; every rejected construction below bends one field.
BASELINE: dict[str, Any] = {
    "as_of_date": AS_OF,
    "factor_name": NAME,
    "factor_version": VERSION,
    "universe_size": 4,
    "eligible_count": 3,
    "eligible_coverage": 0.75,
    "valid_count": 2,
    "missing_count": 2,
    "missing_rate": 0.5,
    "mean": 0.5,
    "std": 0.25,
    "minimum": 0.25,
    "maximum": 0.75,
}

#: The same object with exactly one usable value (where ``std`` must be ``0.0``).
SINGLE: dict[str, Any] = {
    **BASELINE,
    "valid_count": 1,
    "missing_count": 3,
    "missing_rate": 0.75,
    "mean": 0.5,
    "std": 0.0,
    "minimum": 0.5,
    "maximum": 0.5,
}


def statistics_with(**overrides: Any) -> FactorPanelStatistics:
    """A statistics object built from :data:`BASELINE` with ``overrides`` applied."""
    return FactorPanelStatistics(**{**BASELINE, **overrides})


def single_with(**overrides: Any) -> FactorPanelStatistics:
    """A one-usable-value statistics object built from :data:`SINGLE`."""
    return FactorPanelStatistics(**{**SINGLE, **overrides})


# ---------------------------------------------------------------------------
# The statistics object
# ---------------------------------------------------------------------------
def test_the_statistics_object_is_frozen() -> None:
    """A description of a past session must not be editable in place."""
    stats = statistics_with()

    with pytest.raises(FrozenInstanceError):
        stats.mean = 1.0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        stats.valid_count = 1  # type: ignore[misc]


def test_two_statistics_objects_with_the_same_numbers_are_equal() -> None:
    """Value equality, so two runs over the same panel compare equal."""
    assert statistics_with() == statistics_with()
    assert statistics_with() != statistics_with(mean=0.6)


def test_the_statistics_object_prints_one_readable_line() -> None:
    """The ``repr`` names the session and the counts, not thirteen POSIX fields."""
    assert repr(statistics_with()) == (
        "FactorPanelStatistics(as_of_date='2024-01-05', factor=momentum_12_1@1.0.0, "
        "universe=4, eligible=3, valid=2, missing=2, mean=0.5)"
    )


def test_integer_moments_and_rates_are_coerced_to_float() -> None:
    """A hand-built object may say ``1``; it is stored as the float ``1.0``."""
    stats = statistics_with(
        universe_size=4,
        eligible_count=4,
        eligible_coverage=1,
        valid_count=4,
        missing_count=0,
        missing_rate=0,
        mean=0.5,
        std=0.25,
        minimum=0.25,
        maximum=0.75,
    )

    assert isinstance(stats.eligible_coverage, float)
    assert isinstance(stats.missing_rate, float)
    assert (stats.eligible_coverage, stats.missing_rate) == (1.0, 0.0)
    assert all(isinstance(value, float) for value in (stats.mean, stats.std))
    assert (stats.minimum, stats.maximum) == (0.25, 0.75)


def test_a_statistics_object_with_no_usable_value_is_legal() -> None:
    """Four ``None`` moments with every count at zero-useful is a normal result."""
    stats = statistics_with(
        eligible_count=0,
        eligible_coverage=0.0,
        valid_count=0,
        missing_count=4,
        missing_rate=1.0,
        mean=None,
        std=None,
        minimum=None,
        maximum=None,
    )

    assert (stats.mean, stats.std, stats.minimum, stats.maximum) == (None, None, None, None)
    assert stats.missing_rate == 1.0


def test_a_single_usable_value_must_have_zero_deviation() -> None:
    """The population deviation of one value is exactly zero, never rounded."""
    assert single_with().std == 0.0

    with pytest.raises(QuickAnalysisError) as excinfo:
        single_with(std=0.1)

    assert "single valid value" in str(excinfo.value)


@pytest.mark.parametrize(
    ("field", "value", "fragment"),
    [
        ("as_of_date", datetime(2024, 1, 5), "datetime.date"),
        ("as_of_date", "2024-01-05", "datetime.date"),
        ("factor_name", "Momentum", "snake_case"),
        ("factor_name", 1, "snake_case"),
        ("factor_version", "", "non-empty string"),
        ("factor_version", "   ", "non-empty string"),
        ("universe_size", -1, "non-negative integer"),
        ("universe_size", 0, "at least 1"),
        ("eligible_count", True, "non-negative integer"),
        ("eligible_count", 5, "cannot exceed universe_size"),
        ("valid_count", 4, "cannot exceed eligible_count"),
        ("missing_count", 1, "universe_size - valid_count"),
        ("eligible_coverage", 0.7, "eligible_count /"),
        ("missing_rate", 0.4, "missing_count / universe_size"),
        ("mean", None, "must be a number when valid_count is 2"),
        ("mean", NAN, "finite real numbers"),
        ("mean", INF, "finite real numbers"),
        ("mean", -INF, "finite real numbers"),
        ("mean", "0.5", "finite real numbers"),
        ("std", INF, "finite real numbers"),
        ("std", -INF, "finite real numbers"),
        ("minimum", INF, "finite real numbers"),
        ("maximum", -INF, "finite real numbers"),
        ("std", -0.25, "cannot be negative"),
        ("minimum", 0.9, "cannot exceed maximum"),
        ("maximum", None, "must be a number when valid_count is 2"),
    ],
)
def test_an_inconsistent_statistics_object_is_rejected(
    field: str, value: Any, fragment: str
) -> None:
    """No object may claim numbers that contradict its own counts."""
    with pytest.raises(QuickAnalysisError) as excinfo:
        statistics_with(**{field: value})

    assert fragment in str(excinfo.value)
    assert excinfo.value.problems


@pytest.mark.parametrize("field", ["mean", "std", "minimum", "maximum"])
@pytest.mark.parametrize("value", [INF, -INF, NAN])
def test_a_hand_built_moment_must_be_finite(field: str, value: float) -> None:
    """A finite sample has finite moments, so no member may be ``inf`` or ``NaN``.

    Each field is refused on its own name, whatever the sign of the infinity:
    an infinite statistic never describes a cross-section this module measured.
    """
    with pytest.raises(QuickAnalysisError) as excinfo:
        statistics_with(**{field: value})

    assert field in str(excinfo.value)
    assert "finite real numbers" in str(excinfo.value)


def test_a_statistics_object_without_a_usable_value_may_not_carry_a_moment() -> None:
    """``valid_count == 0`` allows four ``None``, not a number and not an ``inf``."""
    empty = {
        "eligible_count": 0,
        "eligible_coverage": 0.0,
        "valid_count": 0,
        "missing_count": 4,
        "missing_rate": 1.0,
        "mean": None,
        "std": None,
        "minimum": None,
    }

    steady = statistics_with(**empty, maximum=None)
    assert (steady.mean, steady.std, steady.minimum, steady.maximum) == (
        None,
        None,
        None,
        None,
    )

    with pytest.raises(QuickAnalysisError) as excinfo:
        statistics_with(**empty, maximum=INF)

    assert "must be None when valid_count is 0" in str(excinfo.value)


# ---------------------------------------------------------------------------
# summarize -- one panel
# ---------------------------------------------------------------------------
def test_a_fully_valid_panel_is_measured_exactly() -> None:
    """Counts, coverage and the four moments of two usable values."""
    stats = describe(values=[0.25, 0.75], symbols=("AAPL", "MSFT"))

    assert (stats.universe_size, stats.eligible_count) == (2, 2)
    assert (stats.valid_count, stats.missing_count) == (2, 0)
    assert (stats.eligible_coverage, stats.missing_rate) == (1.0, 0.0)
    assert (stats.mean, stats.std, stats.minimum, stats.maximum) == (0.5, 0.25, 0.25, 0.75)


def test_the_standard_deviation_is_the_population_one() -> None:
    """``ddof=0``: the sample deviation of ``[1, 2, 3]`` (1.0) must not appear."""
    values = (1.0, 2.0, 3.0)
    stats = describe(values=values)

    assert stats.mean == pytest.approx(stdlib_statistics.fmean(values))
    assert stats.std == pytest.approx(stdlib_statistics.pstdev(values))
    assert stats.std == pytest.approx(math.sqrt(2 / 3))
    assert stats.std != pytest.approx(stdlib_statistics.stdev(values))
    assert (stats.minimum, stats.maximum) == (1.0, 3.0)


def test_ineligible_rows_are_missing_not_measured() -> None:
    """Two of four rows are eligible, so two of four symbols are usable."""
    stats = describe(
        values=[0.25, 1.0, 0.75, 0.5],
        eligible=[True, False, True, False],
        symbols=WIDER[:4],
    )

    assert (stats.universe_size, stats.eligible_count) == (4, 2)
    assert (stats.eligible_coverage, stats.missing_rate) == (0.5, 0.5)
    assert (stats.valid_count, stats.missing_count) == (2, 2)
    assert (stats.mean, stats.std, stats.minimum, stats.maximum) == (0.5, 0.25, 0.25, 0.75)


def test_an_ineligible_finite_value_never_enters_the_moments() -> None:
    """Computed but excluded is still excluded: it cannot move the mean."""
    stats = describe(values=[0.5, 100.0], eligible=[True, False], symbols=("AAPL", "MSFT"))

    assert (stats.valid_count, stats.missing_count) == (1, 1)
    assert (stats.mean, stats.std, stats.minimum, stats.maximum) == (0.5, 0.0, 0.5, 0.5)


@pytest.mark.parametrize("bad", [NAN, INF, -INF])
def test_an_eligible_non_finite_value_is_counted_as_missing(bad: float) -> None:
    """``NaN``/``+inf``/``-inf`` do not crash the analysis, and are not numbers."""
    stats = describe(values=[0.25, bad, 0.75])

    assert (stats.universe_size, stats.eligible_count) == (3, 3)
    assert (stats.valid_count, stats.missing_count) == (2, 1)
    assert stats.missing_rate == pytest.approx(1 / 3)
    assert (stats.mean, stats.std, stats.minimum, stats.maximum) == (0.5, 0.25, 0.25, 0.75)


def test_a_panel_without_a_single_usable_value_has_no_moments() -> None:
    """Nothing to describe is a result: four ``None`` and a missing rate of 1."""
    stats = describe(values=[NAN, INF, -INF])

    assert (stats.universe_size, stats.eligible_count) == (3, 3)
    assert (stats.valid_count, stats.missing_count) == (0, 3)
    assert stats.missing_rate == 1.0
    assert (stats.mean, stats.std, stats.minimum, stats.maximum) == (None, None, None, None)


def test_a_fully_ineligible_panel_has_zero_coverage() -> None:
    """Eligibility of zero says the formula excluded the whole pool."""
    stats = describe(values=[0.25, 0.75, 0.5], eligible=[False, False, False])

    assert (stats.eligible_count, stats.eligible_coverage) == (0, 0.0)
    assert (stats.valid_count, stats.missing_count) == (0, 3)
    assert stats.missing_rate == 1.0
    assert (stats.mean, stats.std) == (None, None)


def test_a_single_usable_value_has_a_zero_deviation() -> None:
    """One eligible number and two missing symbols: ``std`` is exactly ``0.0``."""
    stats = describe(values=[0.7, NAN, NAN])

    assert (stats.valid_count, stats.missing_count) == (1, 2)
    assert stats.std == 0.0
    assert (stats.mean, stats.minimum, stats.maximum) == (0.7, 0.7, 0.7)


def test_identical_values_have_no_deviation() -> None:
    """A flat cross-section is not a rounding error's plaything."""
    stats = describe(values=[0.3, 0.3, 0.3])

    assert stats.mean == pytest.approx(0.3)
    assert stats.std == 0.0


def test_two_values_at_the_float_limit_are_measured_finitely() -> None:
    """``1e308`` twice is a flat cross-section, not ``mean == inf``.

    The total of the two values leaves the float range, so the moments are
    formed on the scaled sample and scaled back: the answer is the input, and
    the deviation is exactly zero.
    """
    stats = describe(values=[1e308, 1e308], symbols=("AAPL", "MSFT"))

    assert (stats.universe_size, stats.eligible_count) == (2, 2)
    assert (stats.valid_count, stats.missing_count) == (2, 0)
    assert all(
        math.isfinite(moment)  # type: ignore[arg-type]
        for moment in (stats.mean, stats.std, stats.minimum, stats.maximum)
    )
    assert stats.mean == 1e308
    assert stats.std == 0.0
    assert (stats.minimum, stats.maximum) == (1e308, 1e308)


def test_opposite_values_at_the_float_limit_keep_a_finite_deviation() -> None:
    """``1e308`` next to ``-1e308``: zero mean, one full scale of deviation."""
    stats = describe(values=[1e308, -1e308], symbols=("AAPL", "MSFT"))

    assert stats.valid_count == 2
    assert all(
        math.isfinite(moment)  # type: ignore[arg-type]
        for moment in (stats.mean, stats.std, stats.minimum, stats.maximum)
    )
    assert stats.mean == 0.0
    assert stats.std == 1e308
    assert (stats.minimum, stats.maximum) == (-1e308, 1e308)


def test_a_huge_cross_section_keeps_the_population_convention() -> None:
    """The scaled path still answers ``ddof=0``, and still ignores row order."""
    values = [1e308, 1e308, -1e308, -1e308, 0.0]
    stats = describe(values=values, symbols=WIDER)

    assert stats.valid_count == 5
    assert stats.mean == 0.0
    assert stats.std == pytest.approx(stdlib_statistics.pstdev(values), rel=1e-12)
    assert stats.std != pytest.approx(stdlib_statistics.stdev(values), rel=1e-6)
    assert (stats.minimum, stats.maximum) == (-1e308, 1e308)


def test_the_order_of_a_huge_cross_section_does_not_move_its_moments() -> None:
    """Exact-then-rounded sums on the scaled copy: reordering changes nothing."""
    values = [1e308, -1e308, 1e308, -1e308]
    forward = describe(values=values, symbols=WIDER[:4])
    backward = describe(values=list(reversed(values)), symbols=WIDER[:4])

    assert (forward.mean, forward.std) == (backward.mean, backward.std)


def test_a_tiny_cross_section_keeps_its_spread() -> None:
    """The mirror image of the overflow: ``[1e-200, 2e-200]`` is not flat.

    Squaring a deviation of ``5e-201`` underflows to zero, so the direct
    arithmetic would call this cross-section flat; the scaled copy measures the
    real spread instead of collapsing it.
    """
    values = [1e-200, 2e-200]
    stats = describe(values=values, symbols=("AAPL", "MSFT"))

    assert stats.std > 0.0
    assert stats.mean == pytest.approx(1.5e-200, rel=1e-12, abs=0.0)
    assert stats.std == pytest.approx(5e-201, rel=1e-12, abs=0.0)
    assert stats.std == pytest.approx(stdlib_statistics.pstdev(values), rel=1e-12, abs=0.0)


def test_a_subnormal_variance_is_measured_at_full_precision() -> None:
    """``[1e-161, -1e-161]`` squares into the subnormal range, not to zero.

    A variance of ``1e-322`` is still representable, but only as a subnormal,
    where a float keeps well under 53 significant bits; the plain formula would
    answer a spread near, but not at, ``1e-161``.  The scaled copy measures it
    at full precision instead, and the mean stays exact.
    """
    stats = describe(values=[1e-161, -1e-161], symbols=("AAPL", "MSFT"))

    assert stats.mean == 0.0
    assert stats.std == 1e-161
    assert (stats.minimum, stats.maximum) == (-1e-161, 1e-161)


def test_a_huge_value_is_measured_not_demoted_to_missing() -> None:
    """Finite is finite: ``1e308`` stays usable and does not become missing."""
    stats = describe(values=[1e308, 0.75, NAN])

    assert (stats.valid_count, stats.missing_count) == (2, 1)
    assert stats.mean == pytest.approx(5e307, rel=1e-12)
    assert stats.std == pytest.approx(5e307, rel=1e-12)
    assert (stats.minimum, stats.maximum) == (0.75, 1e308)


def test_negative_fractional_values_are_kept_as_they_are() -> None:
    """No clipping, no rounding: a negative momentum stays negative."""
    values = (-1.5, 0.0, 2.5)
    stats = describe(values=values)

    assert stats.mean == pytest.approx(stdlib_statistics.fmean(values))
    assert stats.std == pytest.approx(stdlib_statistics.pstdev(values))
    assert (stats.minimum, stats.maximum) == (-1.5, 2.5)


@pytest.mark.parametrize(
    "bad",
    [
        pd.DataFrame({"raw_value": [1.0], "is_eligible": [True]}),
        None,
        {"AAPL": 1.0},
        "AAPL",
        42,
    ],
)
def test_summarize_refuses_anything_that_is_not_a_panel(bad: object) -> None:
    """A frame, a mapping, a name or a number is not a described cross-section."""
    with pytest.raises(QuickAnalysisError) as excinfo:
        QuickAnalyzer().summarize(bad)  # type: ignore[arg-type]

    assert "needs a FactorPanel" in str(excinfo.value)


def test_summarize_refuses_a_sequence_of_panels() -> None:
    """A history belongs to ``summarize_history``; a list is not a panel."""
    with pytest.raises(QuickAnalysisError) as excinfo:
        QuickAnalyzer().summarize([make_panel()])  # type: ignore[arg-type]

    assert "needs a FactorPanel" in str(excinfo.value)


def test_summarize_does_not_modify_the_panel() -> None:
    """The panel and its frame are identical before and after."""
    panel = make_panel(values=[0.25, NAN, 0.75], eligible=[True, True, False])
    snapshot = panel.frame.copy(deep=True)
    symbols = panel.symbols

    QuickAnalyzer().summarize(panel)

    pd.testing.assert_frame_equal(panel.frame, snapshot)
    assert panel.symbols == symbols
    assert len(panel) == 3


def test_summarize_is_a_pure_function_of_the_panel() -> None:
    """Two calls give equal but distinct objects with the same ``repr``."""
    panel = make_panel(values=[0.25, 0.75, 0.5])

    first = QuickAnalyzer().summarize(panel)
    second = QuickAnalyzer().summarize(panel)

    assert first == second
    assert first is not second
    assert repr(first) == repr(second)
    assert (first.as_of_date, first.factor_name, first.factor_version) == (AS_OF, NAME, VERSION)


# ---------------------------------------------------------------------------
# summarize_history -- a sequence of panels
# ---------------------------------------------------------------------------
def dated_panels() -> tuple[FactorPanel, ...]:
    """Three panels of one factor at three dates (mean 0.1, 0.2, 0.3)."""
    return (
        make_panel(as_of=AS_OF, values=[0.1, 0.1, 0.1]),
        make_panel(as_of=LATER, values=[0.2, 0.2, 0.2]),
        make_panel(as_of=LATEST, values=[0.3, 0.3, 0.3]),
    )


def test_history_is_sorted_by_date_whatever_the_input_order() -> None:
    """A shuffled input describes the same sessions in the same order."""
    analyzer = QuickAnalyzer()
    january_5, january_8, january_10 = dated_panels()

    history = analyzer.summarize_history([january_10, january_5, january_8])

    assert [item.as_of_date for item in history] == [AS_OF, LATER, LATEST]
    assert [item.mean for item in history] == pytest.approx([0.1, 0.2, 0.3])
    assert all(isinstance(item.as_of_date, date) for item in history)


def test_history_keeps_the_date_holes_it_was_given() -> None:
    """A missing session is not invented: only the dates handed in appear."""
    earlier = date(2024, 1, 2)
    analyzer = QuickAnalyzer()

    history = analyzer.summarize_history((make_panel(as_of=AS_OF), make_panel(as_of=earlier)))

    assert [item.as_of_date for item in history] == [earlier, AS_OF]
    assert len(history) == 2


def test_an_empty_history_is_an_empty_tuple() -> None:
    """No panels is a legal, empty history -- not an error and not a list."""
    analyzer = QuickAnalyzer()

    empty = analyzer.summarize_history(())

    assert empty == ()
    assert isinstance(empty, tuple)
    assert analyzer.summarize_history([]) == ()
    assert analyzer.summarize_history(iter([])) == ()


def test_history_accepts_a_list_a_tuple_and_a_generator() -> None:
    """The three ways a caller can hold a history all describe it identically."""
    analyzer = QuickAnalyzer()
    panels = dated_panels()

    from_list = analyzer.summarize_history(list(panels))
    from_tuple = analyzer.summarize_history(panels)
    from_generator = analyzer.summarize_history(panel for panel in reversed(panels))

    assert from_list == from_tuple == from_generator
    assert isinstance(from_generator, tuple)
    assert len(from_generator) == 3


def test_history_of_one_panel_is_a_one_element_tuple() -> None:
    """A single session is a history too."""
    panel = make_panel(as_of=AS_OF, values=[0.25, 0.75, 0.5])

    history = QuickAnalyzer().summarize_history([panel])

    assert len(history) == 1
    assert history[0] == QuickAnalyzer().summarize(panel)


def test_the_universe_may_change_from_day_to_day() -> None:
    """A point-in-time pool is not held constant, and is not compared across days."""
    analyzer = QuickAnalyzer()
    panels = (
        make_panel(as_of=AS_OF, values=[0.5, 0.5, 0.5]),
        make_panel(as_of=LATER, symbols=WIDER, values=[0.1, 0.2, 0.3, 0.4, 0.5]),
    )

    history = analyzer.summarize_history(panels)

    assert [item.universe_size for item in history] == [3, 5]
    assert [item.valid_count for item in history] == [3, 5]
    assert history[1].mean == pytest.approx(0.3)
    assert history[0].mean == pytest.approx(0.5)


def test_history_refuses_a_different_factor_name() -> None:
    """One history describes one factor; a second name is a mistake."""
    analyzer = QuickAnalyzer()
    panels = (
        make_panel(as_of=AS_OF),
        make_panel(as_of=LATER, name="value_ratio"),
    )

    with pytest.raises(QuickAnalysisError) as excinfo:
        analyzer.summarize_history(panels)

    message = str(excinfo.value)
    assert "value_ratio" in message
    assert NAME in message


def test_history_refuses_a_different_factor_version() -> None:
    """A new version is a new factor; its statistics must not be glued on."""
    analyzer = QuickAnalyzer()
    panels = (
        make_panel(as_of=AS_OF),
        make_panel(as_of=LATER, version="2.0.0"),
    )

    with pytest.raises(QuickAnalysisError) as excinfo:
        analyzer.summarize_history(panels)

    assert "2.0.0" in str(excinfo.value)
    assert VERSION in str(excinfo.value)


def test_history_refuses_a_repeated_date() -> None:
    """Two cross-sections for one session cannot both be the session."""
    analyzer = QuickAnalyzer()
    panels = (make_panel(as_of=AS_OF), make_panel(as_of=AS_OF, values=[0.9, 0.9, 0.9]))

    with pytest.raises(QuickAnalysisError) as excinfo:
        analyzer.summarize_history(panels)

    message = str(excinfo.value)
    assert "repeats as_of_date 2024-01-05" in message
    assert "panels[1]" in message


def test_history_refuses_something_that_is_not_a_panel() -> None:
    """A frame, a name or a number inside the sequence is named by position."""
    analyzer = QuickAnalyzer()

    with pytest.raises(QuickAnalysisError) as excinfo:
        analyzer.summarize_history((make_panel(as_of=AS_OF), "AAPL"))

    assert "panels[1] is a str" in str(excinfo.value)


def test_history_reports_every_problem_in_one_error() -> None:
    """Three broken entries are named together, not one per failed call."""
    analyzer = QuickAnalyzer()
    panels = [
        make_panel(as_of=AS_OF),
        make_panel(as_of=AS_OF, values=[0.9, 0.9, 0.9]),
        make_panel(as_of=LATER, name="value_ratio"),
        42,
    ]

    with pytest.raises(QuickAnalysisError) as excinfo:
        analyzer.summarize_history(panels)

    problems = excinfo.value.problems
    assert len(problems) == 3
    assert any("repeats as_of_date" in problem for problem in problems)
    assert any("value_ratio" in problem for problem in problems)
    assert any("not a FactorPanel" in problem for problem in problems)


@pytest.mark.parametrize("bad", [None, 42, "AAPL", b"AAPL"])
def test_history_refuses_a_non_iterable(bad: object) -> None:
    """A history has to be something the analyzer can walk."""
    with pytest.raises(QuickAnalysisError) as excinfo:
        QuickAnalyzer().summarize_history(bad)  # type: ignore[arg-type]

    assert "needs an iterable of FactorPanel" in str(excinfo.value)


def test_history_refuses_a_single_panel() -> None:
    """The single-panel question has its own method, and the error says so."""
    with pytest.raises(QuickAnalysisError) as excinfo:
        QuickAnalyzer().summarize_history(make_panel())  # type: ignore[arg-type]

    message = str(excinfo.value)
    assert "single FactorPanel" in message
    assert "summarize_history" in message


def test_history_does_not_modify_the_panels_it_was_given() -> None:
    """The list and the panels inside it come back untouched."""
    analyzer = QuickAnalyzer()
    panels = list(dated_panels())
    snapshot = tuple(panels)

    analyzer.summarize_history(panels)

    assert len(panels) == 3
    assert all(left is right for left, right in zip(panels, snapshot, strict=True))


def test_history_is_repeatable() -> None:
    """The same panels always describe the same sessions in the same order."""
    analyzer = QuickAnalyzer()
    panels = dated_panels()

    first = analyzer.summarize_history(panels)
    second = analyzer.summarize_history(panels)

    assert first == second
    assert first is not second


# ---------------------------------------------------------------------------
# to_frame -- the same statistics as a table
# ---------------------------------------------------------------------------
def test_to_frame_has_the_documented_columns_in_order() -> None:
    """One fixed order, so two notebooks print the same table."""
    analyzer = QuickAnalyzer()
    frame = analyzer.to_frame(analyzer.summarize_history(dated_panels()))

    assert list(frame.columns) == list(STATISTICS_COLUMNS)
    assert tuple(frame.columns) == STATISTICS_COLUMNS
    assert len(STATISTICS_COLUMNS) == len(set(STATISTICS_COLUMNS))


def test_to_frame_carries_the_statistics_values() -> None:
    """A row is the statistics object it came from, field by field."""
    analyzer = QuickAnalyzer()
    history = analyzer.summarize_history(dated_panels())

    frame = analyzer.to_frame(history)

    assert len(frame) == 3
    assert frame["as_of_date"].dt.date.tolist() == [AS_OF, LATER, LATEST]
    assert frame["factor_name"].tolist() == [NAME] * 3
    assert frame["factor_version"].tolist() == [VERSION] * 3
    assert frame["universe_size"].tolist() == [3, 3, 3]
    assert frame["eligible_count"].tolist() == [3, 3, 3]
    assert frame["valid_count"].tolist() == [3, 3, 3]
    assert frame["missing_count"].tolist() == [0, 0, 0]
    assert frame["mean"].tolist() == pytest.approx([0.1, 0.2, 0.3])
    assert frame["std"].tolist() == pytest.approx([0.0, 0.0, 0.0])
    assert frame["minimum"].tolist() == pytest.approx([0.1, 0.2, 0.3])
    assert frame["missing_rate"].tolist() == [0.0, 0.0, 0.0]


def test_to_frame_uses_stable_dtypes() -> None:
    """A table a caller can concatenate: born-typed, not all ``object``."""
    analyzer = QuickAnalyzer()
    frame = analyzer.to_frame(analyzer.summarize_history(dated_panels()))

    assert str(frame["as_of_date"].dtype) == "datetime64[s]"
    for column in ("universe_size", "eligible_count", "valid_count", "missing_count"):
        assert str(frame[column].dtype) == "int64"
    for column in ("eligible_coverage", "missing_rate", "mean", "std", "minimum", "maximum"):
        assert str(frame[column].dtype) == "float64"
    for column in ("factor_name", "factor_version"):
        assert str(frame[column].dtype) == "object"


def test_to_frame_of_an_empty_history_keeps_the_schema() -> None:
    """A caller can concatenate day one: the empty table has the full schema."""
    frame = QuickAnalyzer().to_frame(())

    assert frame.empty
    assert frame.shape == (0, len(STATISTICS_COLUMNS))
    assert tuple(frame.columns) == STATISTICS_COLUMNS
    assert str(frame["as_of_date"].dtype) == "datetime64[s]"
    assert str(frame["universe_size"].dtype) == "int64"
    assert str(frame["mean"].dtype) == "float64"
    assert str(frame["factor_name"].dtype) == "object"


def test_to_frame_renders_no_usable_value_as_an_empty_moment() -> None:
    """``None`` becomes ``NaN`` in a float column, and stays a float column."""
    stats = QuickAnalyzer().summarize(make_panel(values=[NAN, NAN, NAN]))

    frame = QuickAnalyzer().to_frame((stats,))

    assert frame["mean"].isna().all()
    assert frame["std"].isna().all()
    assert frame["missing_rate"].tolist() == [1.0]
    assert str(frame["mean"].dtype) == "float64"


def test_to_frame_keeps_a_huge_but_finite_moment_finite() -> None:
    """The table shows the number the analyzer reported, never ``inf``."""
    stats = QuickAnalyzer().summarize(make_panel(values=[1e308, 1e308], symbols=("AAPL", "MSFT")))

    frame = QuickAnalyzer().to_frame((stats,))

    assert frame["mean"].tolist() == [1e308]
    assert frame["std"].tolist() == [0.0]
    assert frame["maximum"].tolist() == [1e308]
    assert frame[["mean", "std", "minimum", "maximum"]].notna().all().all()


def test_to_frame_keeps_the_order_it_was_given() -> None:
    """Row order is the caller's business; the columns are the fixed part."""
    analyzer = QuickAnalyzer()
    history = analyzer.summarize_history(dated_panels())

    frame = analyzer.to_frame(tuple(reversed(history)))

    assert frame["as_of_date"].dt.date.tolist() == [LATEST, LATER, AS_OF]


def test_to_frame_holds_statistics_not_symbol_rows() -> None:
    """One row per session: the panel's per-symbol rows never leak into it."""
    analyzer = QuickAnalyzer()
    panels = dated_panels()

    frame = analyzer.to_frame(analyzer.summarize_history(panels))

    assert len(frame) == len(panels)
    assert "symbol" not in frame.columns
    assert "raw_value" not in frame.columns
    assert frame["universe_size"].tolist() == [len(panel.universe) for panel in panels]


def test_to_frame_refuses_a_single_statistics_object() -> None:
    """One object is a row, not a table; the error says how to fix it."""
    with pytest.raises(QuickAnalysisError) as excinfo:
        QuickAnalyzer().to_frame(statistics_with())  # type: ignore[arg-type]

    assert "wrap it as (statistics,)" in str(excinfo.value)


def test_to_frame_refuses_a_foreign_item() -> None:
    """Anything that is not a statistics object is named by position."""
    with pytest.raises(QuickAnalysisError) as excinfo:
        QuickAnalyzer().to_frame([statistics_with(), "AAPL"])  # type: ignore[list-item]

    message = str(excinfo.value)
    assert "history[1] is a str" in message
    assert "not a FactorPanelStatistics" in message


@pytest.mark.parametrize("bad", [None, 42, "AAPL", {"as_of_date": AS_OF}])
def test_to_frame_refuses_a_non_iterable(bad: object) -> None:
    """A table needs a sequence of statistics to render."""
    with pytest.raises(QuickAnalysisError):
        QuickAnalyzer().to_frame(bad)  # type: ignore[arg-type]


def test_to_frame_does_not_modify_the_history() -> None:
    """The table is a copy; editing it cannot rewrite the statistics."""
    analyzer = QuickAnalyzer()
    history = analyzer.summarize_history(dated_panels())
    snapshot = tuple(history)

    frame = analyzer.to_frame(history)
    frame.loc[0, "mean"] = -1.0

    assert history == snapshot
    assert tuple(history) == snapshot
    assert history[0].mean == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# The error type and the package exports
# ---------------------------------------------------------------------------
def test_quick_analysis_error_is_a_factor_error() -> None:
    """One family of errors, so a caller can catch ``FactorError`` once."""
    assert issubclass(QuickAnalysisError, FactorError)

    with pytest.raises(FactorError):
        QuickAnalyzer().summarize(None)  # type: ignore[arg-type]


def test_quick_analysis_error_keeps_its_problems() -> None:
    """Every problem is kept, and printed under the message."""
    error = QuickAnalysisError("cannot describe", ["panels[1] is a str", "dated twice"])

    assert error.problems == ["panels[1] is a str", "dated twice"]
    assert "cannot describe" in str(error)
    assert "panels[1] is a str" in str(error)
    assert "dated twice" in str(error)
    assert QuickAnalysisError("just a message").problems == []


def test_the_package_root_exports_the_quick_analysis_api() -> None:
    """``from factor_sys import ...`` is the supported way to reach this module."""
    assert factor_sys.FactorPanelStatistics is FactorPanelStatistics
    assert factor_sys.QuickAnalyzer is QuickAnalyzer
    assert factor_sys.QuickAnalysisError is QuickAnalysisError
    assert quick_analysis_module.FactorPanelStatistics is FactorPanelStatistics
    assert quick_analysis_module.QuickAnalysisError is QuickAnalysisError
    assert issubclass(factor_sys.QuickAnalysisError, factor_sys.FactorError)

    for name in ("FactorPanelStatistics", "QuickAnalyzer", "QuickAnalysisError"):
        assert name in factor_sys.__all__
        assert name in quick_analysis_module.__all__


# ---------------------------------------------------------------------------
# The architectural boundary of the module
# ---------------------------------------------------------------------------
#: The module under test, read as source: the boundary is a property of the code.
MODULE_PATH: Path = Path(quick_analysis_module.__file__)

#: Exactly the modules a descriptive statistics module may depend on.
ALLOWED_IMPORTS: frozenset[str] = frozenset(
    {
        "__future__",
        "collections.abc",
        "dataclasses",
        "datetime",
        "math",
        "re",
        "factor_sys.contracts",
        "factor_sys.errors",
        "pandas",
    }
)

#: Layers the analyzer must not reach for: data, storage, execution, judgement,
#: columnar engines and the outside world (files, network, clock, calendar).
FORBIDDEN_IMPORT_PREFIXES: tuple[str, ...] = (
    "data_sys",
    "factors",
    "factor_sys.factor_store",
    "factor_sys.runner",
    "factor_sys.market_data",
    "factor_sys.registry",
    "factor_sys.quality",
    "factor_sys.store",
    "factor_sys.analytics",
    "pyarrow",
    "duckdb",
    "numpy",
    "requests",
    "httpx",
    "urllib",
    "socket",
    "yfinance",
    "exchange_calendars",
    "os",
    "pathlib",
    "shutil",
    "tempfile",
    "glob",
    "subprocess",
    "time",
    "calendar",
    "pickle",
    "sqlite3",
    "csv",
    "io",
)

#: Functions that would mean the analyzer loads, computes or persists something.
FORBIDDEN_CALLS: frozenset[str] = frozenset(
    {
        "__import__",
        "eval",
        "exec",
        "input",
        "open",
        "read_csv",
        "read_excel",
        "read_html",
        "read_json",
        "read_parquet",
        "read_text",
        "read_bytes",
        "to_csv",
        "to_parquet",
        "to_pickle",
        "write_text",
        "write_bytes",
        "urlopen",
        "connect",
        "glob",
        "iterdir",
        "listdir",
        "mkdir",
        "system",
        "popen",
    }
)

#: Objects belonging to the layers this module must stay away from.
FORBIDDEN_NAMES: frozenset[str] = frozenset(
    {
        "FactorStore",
        "ParquetFactorStore",
        "FactorRunner",
        "MarketDataReader",
        "ParquetMarketDataReader",
        "FactorRegistry",
        "FactorQualityReport",
        "FactorQualityChecker",
        "DefaultFactorQualityChecker",
        "os",
        "Path",
        "pa",
        "duckdb",
        "np",
    }
)


def module_tree() -> ast.Module:
    """The parsed source of :mod:`factor_sys.quick_analysis`."""
    return ast.parse(MODULE_PATH.read_text(encoding="utf-8"), filename=str(MODULE_PATH))


def module_imports(tree: ast.Module) -> set[str]:
    """Every module named by an ``import`` or a ``from ... import`` statement."""
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            imports.add(node.module or "")
    return imports


def matches(name: str, prefix: str) -> bool:
    """``True`` if ``name`` is ``prefix`` or lives under it."""
    return name == prefix or name.startswith(f"{prefix}.")


def test_the_module_depends_only_on_stdlib_contracts_and_errors() -> None:
    """No store, no runner, no data system, no registry, no quality checker."""
    imports = module_imports(module_tree())

    assert imports <= ALLOWED_IMPORTS, sorted(imports - ALLOWED_IMPORTS)


def test_the_module_imports_nothing_from_the_outside_world() -> None:
    """Every forbidden layer, named one by one, so the intent stays readable."""
    imports = module_imports(module_tree())

    for name in sorted(imports):
        for prefix in FORBIDDEN_IMPORT_PREFIXES:
            assert not matches(name, prefix), f"{name} must not be imported"


def test_the_module_calls_no_loader_persistence_or_network_function() -> None:
    """Nothing here opens, reads, writes, connects or shells out."""
    tree = module_tree()
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    called_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }

    assert not (called_names | called_attributes) & FORBIDDEN_CALLS


def test_the_module_mentions_no_object_from_a_forbidden_layer() -> None:
    """No store, runner, reader, registry or quality checker is even named."""
    tree = module_tree()
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}

    assert not names & FORBIDDEN_NAMES
    assert not attributes & FORBIDDEN_NAMES


def test_the_module_reads_the_panel_through_its_public_frame() -> None:
    """Published columns only: no ``panel._frame`` shortcut, no private state."""
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert "panel._" not in source
    assert "frame._" not in source


def test_the_module_does_not_import_the_package_root() -> None:
    """Importing ``factor_sys`` would drag the store in; contracts and errors are used."""
    imports = module_imports(module_tree())

    assert "factor_sys" not in imports


def test_the_module_publishes_exactly_its_documented_api() -> None:
    """A small, named surface: the statistics, the analyzer and the error."""
    assert set(quick_analysis_module.__all__) == {
        "STATISTICS_COLUMNS",
        "FactorPanelStatistics",
        "QuickAnalysisError",
        "QuickAnalyzer",
    }
