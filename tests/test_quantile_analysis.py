"""Tests for quantile analysis (:mod:`factor_sys.quantile_analysis`).

Everything here is in memory and offline: panels and labels are literal frames,
no reader is called, no file is written and no calendar is consulted.  The
grouping is checked against an independently computed expectation -- the
average rank of each tie group, ``ceil(rank * Q / n)``, written out by hand --
so a passing test says the assignment is right, not that two copies of the same
formula agree.

Covered in this order:

* the analyzer: the default five groups, every legal and illegal group count,
  and the read-only property behind them,
* the grouping rule: ascending factor values, ``Q`` rows per session, exact
  member counts and coverages, equal-weighted means, and the top-minus-bottom
  difference,
* ties: average ranks, a tie group that is never split, and the fact that
  neither row order nor input order can move a symbol from one group to another,
* the awkward cross-sections: fewer symbols than groups, all values equal, and a
  session where nothing can be paired -- ``Q`` rows, empty groups, ``None``
  means, ``None`` spread,
* the pairing rule: ineligible rows and non-finite values are excluded, and the
  usable sample is exactly the one the IC analysis counts,
* many sessions: sorting, calendar holes, changing universes, generators,
* every refused request: empty or non-iterable inputs, foreign objects, mixed
  names/versions/horizons, repeated dates, date sets that do not close, and
  universes that moved or were re-ordered,
* the three frozen objects: every self-contradiction they refuse, the coverage
  identities, and the ``repr`` a log prints,
* the two tables: fixed columns, row order, dtypes, ``NaN`` for an undefined
  number, and a copy the caller cannot corrupt,
* purity: panels, labels and caller sequences survive a call unchanged,
* the architectural boundary: the imports, the objects the module may not name,
  and the fact that the factor and label paths know nothing about quantiles,
* the error type, the package exports, the docstrings and the doctests.
"""

from __future__ import annotations

import ast
import doctest
import inspect
import math
from collections.abc import Sequence
from dataclasses import FrozenInstanceError, replace
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest

import factor_sys
import factor_sys.quantile_analysis as quantile_analysis_module
from factor_sys import (
    FactorError,
    FactorPanel,
    ForwardReturnError,
    ForwardReturnPanel,
    ICAnalyzer,
    QuantileAnalysisError,
    QuantileAnalyzer,
    QuantileObservation,
    QuantileReport,
    QuantileSpreadObservation,
)
from factor_sys.quantile_analysis import (
    QUANTILE_OBSERVATION_COLUMNS,
    QUANTILE_SPREAD_COLUMNS,
)

NAN = float("nan")
INF = float("inf")

#: Ten symbols, so five groups hold two members each without ties.
SYMBOLS: tuple[str, ...] = (
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOG",
    "META",
    "TSLA",
    "AVGO",
    "AMD",
    "NFLX",
)

#: Identity every fixture uses.
NAME: str = "momentum_12_1"
VERSION: str = "1.0.0"

#: Decision dates and the sessions their one-session labels end on.
DAY_ONE: date = date(2024, 1, 5)
END_ONE: date = date(2024, 1, 8)
DAY_TWO: date = date(2024, 1, 8)
END_TWO: date = date(2024, 1, 9)
DAY_THREE: date = date(2024, 1, 9)
END_THREE: date = date(2024, 1, 10)
#: A later date with a calendar hole between it and :data:`DAY_THREE`.
DAY_FOUR: date = date(2024, 1, 16)
END_FOUR: date = date(2024, 1, 17)

#: The module under test, read as source for the boundary tests near the end.
MODULE_PATH: Path = Path(quantile_analysis_module.__file__)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def universe_of(count: int, symbols: Sequence[str] | None = None) -> list[str]:
    """``count`` symbols: the caller's, or the first ``count`` fixture symbols."""
    if symbols is not None:
        return list(symbols)
    return list(SYMBOLS[:count])


def panel_frame(
    symbols: Sequence[str] | None = None,
    values: Sequence[float] | None = None,
    *,
    as_of: date = DAY_ONE,
    name: str = NAME,
    version: str = VERSION,
    eligible: Sequence[bool] | None = None,
) -> pd.DataFrame:
    """A factor frame as a formula would hand it over: one row per symbol."""
    universe = universe_of(len(values or SYMBOLS), symbols)
    count = len(universe)
    default = [float(position) for position in range(count)]
    return pd.DataFrame(
        {
            "as_of_date": [pd.Timestamp(as_of)] * count,
            "symbol": universe,
            "factor_name": [name] * count,
            "factor_version": [version] * count,
            "raw_value": list(values) if values is not None else default,
            "is_eligible": list(eligible) if eligible is not None else [True] * count,
        }
    )


def make_panel(
    symbols: Sequence[str] | None = None,
    values: Sequence[float] | None = None,
    *,
    as_of: date = DAY_ONE,
    name: str = NAME,
    version: str = VERSION,
    eligible: Sequence[bool] | None = None,
) -> FactorPanel:
    """A valid factor panel whose pool, values, date and identity the test chooses."""
    universe = universe_of(len(values or SYMBOLS), symbols)
    frame = panel_frame(
        universe, values, as_of=as_of, name=name, version=version, eligible=eligible
    )
    return FactorPanel(frame, universe=universe, as_of_date=as_of)


def label_frame(
    symbols: Sequence[str] | None = None,
    returns: Sequence[float] | None = None,
    *,
    available: Sequence[bool] | None = None,
    as_of: date = DAY_ONE,
    end: date = END_ONE,
    name: str = NAME,
    version: str = VERSION,
    horizon: int = 1,
) -> pd.DataFrame:
    """A label frame as :class:`ForwardReturnCalculator` would hand it over."""
    universe = universe_of(len(returns or SYMBOLS), symbols)
    count = len(universe)
    return pd.DataFrame(
        {
            "as_of_date": [pd.Timestamp(as_of)] * count,
            "end_date": [pd.Timestamp(end)] * count,
            "symbol": universe,
            "factor_name": [name] * count,
            "factor_version": [version] * count,
            "horizon_sessions": [horizon] * count,
            "forward_return": list(returns) if returns is not None else [0.01] * count,
            "is_available": list(available) if available is not None else [True] * count,
        }
    )


def make_labels(
    symbols: Sequence[str] | None = None,
    returns: Sequence[float] | None = None,
    *,
    available: Sequence[bool] | None = None,
    as_of: date = DAY_ONE,
    end: date = END_ONE,
    name: str = NAME,
    version: str = VERSION,
    horizon: int = 1,
) -> ForwardReturnPanel:
    """A valid label object; refusals are exercised by bending one field."""
    universe = universe_of(len(returns or SYMBOLS), symbols)
    return ForwardReturnPanel(
        label_frame(
            universe,
            returns,
            available=available,
            as_of=as_of,
            end=end,
            name=name,
            version=version,
            horizon=horizon,
        ),
        universe=universe,
        as_of_date=as_of,
        end_date=end,
        factor_name=name,
        factor_version=version,
        horizon_sessions=horizon,
    )


def make_session(
    values: Sequence[float],
    returns: Sequence[float],
    *,
    symbols: Sequence[str] | None = None,
    eligible: Sequence[bool] | None = None,
    available: Sequence[bool] | None = None,
    as_of: date = DAY_ONE,
    end: date = END_ONE,
    horizon: int = 1,
    name: str = NAME,
    version: str = VERSION,
) -> tuple[FactorPanel, ForwardReturnPanel]:
    """One decision date as a factor panel plus its labels.

    A symbol marked unavailable gets a ``NaN`` label, the way the calculator
    leaves it: the contract refuses an unavailable row that holds a number.
    """
    flags = list(available) if available is not None else [True] * len(returns)
    weights = [value if flag else NAN for value, flag in zip(returns, flags, strict=True)]
    panel = make_panel(symbols, values, as_of=as_of, name=name, version=version, eligible=eligible)
    labels = make_labels(
        symbols,
        weights,
        available=flags,
        as_of=as_of,
        end=end,
        name=name,
        version=version,
        horizon=horizon,
    )
    return panel, labels


def report_for(
    values: Sequence[float],
    returns: Sequence[float],
    *,
    quantiles: int = 5,
    **kwargs: object,
) -> QuantileReport:
    """Analyse one session and return its report."""
    panel, labels = make_session(values, returns, **kwargs)
    return QuantileAnalyzer(quantiles=quantiles).analyze([panel], [labels])


def groups_of(report: QuantileReport, day: date = DAY_ONE) -> dict[int, QuantileObservation]:
    """The observations of one session, keyed by quantile number."""
    return {item.quantile: item for item in report.observations if item.as_of_date == day}


def spread_of(report: QuantileReport, day: date = DAY_ONE) -> QuantileSpreadObservation:
    """The top-minus-bottom observation of one session."""
    return next(item for item in report.spreads if item.as_of_date == day)


def decode_members(report: QuantileReport, universe: Sequence[str]) -> dict[int, list[str]]:
    """Decode which symbols each group holds, from power-of-two labels.

    Every symbol's label is a distinct power of two, so the sum of a group's
    labels is a bit mask of its members and the group mean times its member count
    recovers that mask exactly.  That lets a test assert *membership*, not merely
    a mean that could coincide.
    """
    decoded: dict[int, list[str]] = {}
    for item in report.observations:
        if item.member_count == 0:
            decoded[item.quantile] = []
            continue
        mask = round(float(item.mean_forward_return) * item.member_count)
        decoded[item.quantile] = [
            universe[position] for position in range(len(universe)) if mask & (1 << position)
        ]
    return decoded


def member_sets(
    values: Sequence[float],
    quantiles: int = 5,
    *,
    symbols: Sequence[str] | None = None,
) -> dict[int, list[str]]:
    """Analyse one session with power-of-two labels and decode its groups."""
    universe = universe_of(len(values), symbols)
    weights = [float(2**position) for position in range(len(universe))]
    report = report_for(values, weights, quantiles=quantiles, symbols=universe)
    return decode_members(report, universe)


def permuted(
    panel: FactorPanel,
    labels: ForwardReturnPanel,
    order: Sequence[int],
) -> tuple[FactorPanel, ForwardReturnPanel]:
    """The same session with both frames -- and therefore the universe -- re-ordered."""
    universe = [panel.universe[index] for index in order]
    return (
        FactorPanel(
            panel.frame.iloc[list(order)].reset_index(drop=True),
            universe=universe,
            as_of_date=panel.as_of_date,
        ),
        ForwardReturnPanel(
            labels.frame.iloc[list(order)].reset_index(drop=True),
            universe=universe,
            as_of_date=labels.as_of_date,
            end_date=labels.end_date,
            factor_name=labels.factor_name,
            factor_version=labels.factor_version,
            horizon_sessions=labels.horizon_sessions,
        ),
    )


def observation(**changes: object) -> QuantileObservation:
    """A valid observation, with the fields a test wants to bend."""
    fields: dict[str, object] = {
        "as_of_date": DAY_ONE,
        "end_date": END_ONE,
        "factor_name": NAME,
        "factor_version": VERSION,
        "horizon_sessions": 1,
        "quantiles": 5,
        "quantile": 1,
        "universe_size": 10,
        "paired_count": 10,
        "paired_coverage": 1.0,
        "member_count": 2,
        "member_coverage": 0.2,
        "mean_forward_return": 0.01,
    }
    fields.update(changes)
    return QuantileObservation(**fields)  # type: ignore[arg-type]


def spread_observation(**changes: object) -> QuantileSpreadObservation:
    """A valid spread observation, with the fields a test wants to bend."""
    fields: dict[str, object] = {
        "as_of_date": DAY_ONE,
        "end_date": END_ONE,
        "factor_name": NAME,
        "factor_version": VERSION,
        "horizon_sessions": 1,
        "quantiles": 5,
        "bottom_mean_forward_return": 0.01,
        "top_mean_forward_return": 0.03,
        "top_minus_bottom": 0.02,
    }
    fields.update(changes)
    return QuantileSpreadObservation(**fields)  # type: ignore[arg-type]


def five_groups(
    *,
    day: date = DAY_ONE,
    end: date = END_ONE,
    means: Sequence[float] = (0.05, 0.03, 0.01, -0.01, -0.03),
    paired: int = 10,
    universe_size: int = 10,
) -> tuple[QuantileObservation, ...]:
    """A hand-built, fully coherent set of five groups for one session."""
    return tuple(
        observation(
            as_of_date=day,
            end_date=end,
            quantile=number,
            universe_size=universe_size,
            paired_count=paired,
            paired_coverage=paired / universe_size,
            member_count=2,
            member_coverage=2 / universe_size,
            mean_forward_return=means[number - 1],
        )
        for number in range(1, 6)
    )


def report_of(
    observations: Sequence[QuantileObservation] | object,
    spreads: Sequence[QuantileSpreadObservation] | object,
    *,
    quantiles: int = 5,
    name: str = NAME,
    version: str = VERSION,
    horizon: int = 1,
) -> QuantileReport:
    """A hand-built report around the rows a test supplies."""
    return QuantileReport(
        factor_name=name,
        factor_version=version,
        horizon_sessions=horizon,
        quantiles=quantiles,
        observations=observations,  # type: ignore[arg-type]
        spreads=spreads,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# The analyzer: one knob, and it must be a real group count
# ---------------------------------------------------------------------------
def test_the_default_analyzer_cuts_the_cross_section_into_five_groups() -> None:
    """Five quantiles is the documented default, and the property says so."""
    assert QuantileAnalyzer().quantiles == 5
    assert QuantileAnalyzer(quantiles=5).quantiles == 5


@pytest.mark.parametrize("quantiles", [2, 3, 4, 10, 20, 100])
def test_every_group_count_of_two_or_more_is_accepted(quantiles: int) -> None:
    """Any integer of at least two describes a real bottom-to-top comparison."""
    assert QuantileAnalyzer(quantiles=quantiles).quantiles == quantiles


@pytest.mark.parametrize(
    "quantiles",
    [1, 0, -1, -5, True, False, 2.0, 2.5, "5", "", None, [5], (5,), {"quantiles": 5}],
)
def test_an_unusable_group_count_is_refused(quantiles: object) -> None:
    """A flag, a float, a string, one group or none: nothing to compare."""
    with pytest.raises(QuantileAnalysisError) as caught:
        QuantileAnalyzer(quantiles=quantiles)  # type: ignore[arg-type]

    assert "quantiles" in str(caught.value)
    assert caught.value.problems


def test_the_group_count_is_read_only() -> None:
    """``quantiles`` is a property of an immutable analyzer, not a mutable field."""
    analyzer = QuantileAnalyzer(quantiles=7)

    with pytest.raises(AttributeError):
        analyzer.quantiles = 3  # type: ignore[misc]

    assert not hasattr(analyzer, "__dict__")
    assert analyzer.quantiles == 7


def test_the_analyzer_repr_names_its_only_knob() -> None:
    """A log line says how the cross-section was cut."""
    assert repr(QuantileAnalyzer()) == "QuantileAnalyzer(quantiles=5)"
    assert repr(QuantileAnalyzer(quantiles=10)) == "QuantileAnalyzer(quantiles=10)"


# ---------------------------------------------------------------------------
# Grouping: low factor values first, exact counts, equal-weighted means
# ---------------------------------------------------------------------------
def test_low_factor_values_land_in_the_first_group() -> None:
    """Ten ascending values and ascending labels fill five groups of two."""
    report = report_for(list(range(10)), [0.01 * index for index in range(10)])
    groups = groups_of(report)

    assert sorted(groups) == [1, 2, 3, 4, 5]
    assert [groups[number].member_count for number in range(1, 6)] == [2] * 5
    assert [groups[number].mean_forward_return for number in range(1, 6)] == pytest.approx(
        [0.005, 0.025, 0.045, 0.065, 0.085]
    )


def test_each_group_holds_the_symbols_its_mean_claims() -> None:
    """Membership, decoded from power-of-two labels, follows the sort order."""
    assert member_sets(list(range(10))) == {
        1: ["AAPL", "MSFT"],
        2: ["NVDA", "AMZN"],
        3: ["GOOG", "META"],
        4: ["TSLA", "AVGO"],
        5: ["AMD", "NFLX"],
    }


def test_the_assignment_follows_the_documented_arithmetic() -> None:
    """Seven symbols in three groups: ranks 1..7 give ``ceil(r * 3 / 7)``."""
    assert member_sets([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], quantiles=3) == {
        1: ["AAPL", "MSFT"],
        2: ["NVDA", "AMZN"],
        3: ["GOOG", "META", "TSLA"],
    }


def test_only_the_order_of_the_values_matters() -> None:
    """A factor's scale is irrelevant to a rank-based grouping."""
    small = member_sets([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], quantiles=4)
    huge = member_sets([-1e9, 0.0, 1e-9, 2e-9, 3e-9, 4e-9, 5e-9, 6e-9], quantiles=4)

    assert small == huge


def test_a_group_mean_is_the_equal_weighted_mean_of_its_members() -> None:
    """A group of two averages its members; it is not one member's number."""
    report = report_for([1.0, 2.0, 3.0, 4.0], [0.5, 0.25, 0.125, -0.5], quantiles=2)
    groups = groups_of(report)

    assert groups[1].mean_forward_return == 0.375
    assert groups[2].mean_forward_return == -0.1875


def test_every_session_reports_exactly_one_row_per_group() -> None:
    """A report with three sessions has ``3 * quantiles`` observations."""
    sessions = [
        make_session(list(range(10)), [0.01] * 10, as_of=DAY_ONE, end=END_ONE),
        make_session(list(range(10)), [0.01] * 10, as_of=DAY_TWO, end=END_TWO),
        make_session(list(range(10)), [0.01] * 10, as_of=DAY_THREE, end=END_THREE),
    ]

    report = QuantileAnalyzer(quantiles=4).analyze(
        [panel for panel, _ in sessions], [label for _, label in sessions]
    )

    assert len(report.observations) == 12
    assert len(report.spreads) == 3
    for day in (DAY_ONE, DAY_TWO, DAY_THREE):
        assert sorted(groups_of(report, day)) == [1, 2, 3, 4]


def test_member_counts_and_coverages_add_up_to_the_paired_sample() -> None:
    """Every grouped symbol sits in exactly one group, once."""
    report = report_for(list(range(10)), [0.01] * 10, quantiles=3)
    groups = groups_of(report)

    assert sum(item.member_count for item in groups.values()) == 10
    assert math.isclose(
        math.fsum(item.member_coverage for item in groups.values()),
        groups[1].paired_coverage,
    )
    for item in groups.values():
        assert item.member_coverage == item.member_count / item.universe_size
        assert item.paired_coverage == item.paired_count / item.universe_size
        assert item.universe_size == 10
        assert item.paired_count == 10


def test_the_spread_is_the_top_mean_minus_the_bottom_mean() -> None:
    """Top-minus-bottom is a subtraction of two reported numbers, nothing else."""
    report = report_for(list(range(10)), [0.01 * index for index in range(10)])
    groups = groups_of(report)
    spread = spread_of(report)

    assert spread.bottom_mean_forward_return == groups[1].mean_forward_return
    assert spread.top_mean_forward_return == groups[5].mean_forward_return
    assert spread.top_minus_bottom == pytest.approx(0.08)
    assert spread.top_minus_bottom == groups[5].mean_forward_return - groups[1].mean_forward_return


def test_a_falling_label_makes_the_spread_negative() -> None:
    """When the bottom group earns more, the sign says so instead of hiding it."""
    report = report_for(list(range(10)), [-0.01 * index for index in range(10)])

    assert spread_of(report).top_minus_bottom == pytest.approx(-0.08)


def test_a_group_of_huge_labels_averages_without_overflowing() -> None:
    """A total past the float range is averaged on a scaled copy, not answered as inf."""
    report = report_for([1.0, 1.0, 2.0, 2.0], [1e308] * 4, quantiles=2)
    groups = groups_of(report)

    assert groups[1].member_count == 2
    assert groups[2].member_count == 2
    assert groups[1].mean_forward_return == 1e308
    assert groups[2].mean_forward_return == 1e308
    assert math.isfinite(groups[1].mean_forward_return)
    assert math.isfinite(groups[2].mean_forward_return)
    assert spread_of(report).top_minus_bottom == 0.0


def test_huge_labels_of_opposite_sign_average_to_zero() -> None:
    """Scaling the sample keeps the sign cancellation instead of overflowing it."""
    report = report_for([1.0, 1.0, 1.0, 1.0], [1e308, 1e308, 1e308, -1e308], quantiles=2)
    group = groups_of(report)[2]

    assert group.member_count == 4
    assert group.mean_forward_return == pytest.approx(5e307)
    assert math.isfinite(group.mean_forward_return)
    assert spread_of(report).top_minus_bottom is None


def test_a_spread_too_large_for_a_float_is_none_not_an_infinity() -> None:
    """Two finite means may still differ by more than float64 can hold."""
    report = report_for([1.0, 2.0], [-1e308, 1e308], quantiles=2)
    spread = spread_of(report)

    assert spread.bottom_mean_forward_return == -1e308
    assert spread.top_mean_forward_return == 1e308
    assert math.isfinite(spread.bottom_mean_forward_return)
    assert math.isfinite(spread.top_mean_forward_return)
    assert spread.top_minus_bottom is None


def test_ordinary_labels_average_to_the_same_numbers_as_before() -> None:
    """The overflow path is a fallback: everyday samples keep their exact digits."""
    report = report_for(list(range(10)), [0.01 * index for index in range(10)])
    groups = groups_of(report)

    assert [groups[number].mean_forward_return for number in range(1, 6)] == pytest.approx(
        [0.005, 0.025, 0.045, 0.065, 0.085]
    )
    assert spread_of(report).top_minus_bottom == pytest.approx(0.08)


def test_the_report_copies_the_identity_of_the_panels() -> None:
    """Name, version, horizon and grouping travel from the inputs to the report."""
    report = report_for(list(range(10)), [0.01] * 10, quantiles=4)

    assert report.factor_name == NAME
    assert report.factor_version == VERSION
    assert report.horizon_sessions == 1
    assert report.quantiles == 4
    for item in report.observations:
        assert (item.factor_name, item.factor_version) == (NAME, VERSION)
        assert item.horizon_sessions == 1
        assert item.quantiles == 4
        assert item.end_date == END_ONE
    assert report.spreads[0].end_date == END_ONE


# ---------------------------------------------------------------------------
# Ties: average ranks, never split, never order-dependent
# ---------------------------------------------------------------------------
def test_tied_values_share_an_average_rank_and_are_never_split() -> None:
    """Two equal values occupy ranks 1.5, so both land in quantile 1 of two."""
    assert member_sets([1.0, 1.0, 2.0, 3.0], quantiles=2) == {
        1: ["AAPL", "MSFT"],
        2: ["NVDA", "AMZN"],
    }


def test_a_tie_group_larger_than_a_group_is_still_kept_together() -> None:
    """Four equal values cannot be spread over two groups to make them even."""
    report = report_for([1.0, 1.0, 1.0, 1.0, 2.0, 3.0], [0.1] * 6, quantiles=3)
    groups = groups_of(report)

    assert groups[1].member_count + groups[2].member_count + groups[3].member_count == 6
    assert member_sets([1.0, 1.0, 1.0, 1.0, 2.0, 3.0], quantiles=3) == {
        1: [],
        2: ["AAPL", "MSFT", "NVDA", "AMZN"],
        3: ["GOOG", "META"],
    }


def test_shuffling_the_rows_moves_nobody_between_groups() -> None:
    """The assignment is a function of the values, not of the row positions."""
    values = [3.0, 1.0, 1.0, 2.0, 5.0, 4.0]
    universe = SYMBOLS[:6]
    weights = [float(2**position) for position in range(len(universe))]
    panel, labels = make_session(values, weights, symbols=universe)

    straight = QuantileAnalyzer(quantiles=3).analyze([panel], [labels])
    shuffled_panel, shuffled_labels = permuted(panel, labels, [4, 2, 0, 5, 1, 3])
    shuffled = QuantileAnalyzer(quantiles=3).analyze([shuffled_panel], [shuffled_labels])

    assert decode_members(straight, universe) == decode_members(shuffled, universe)
    assert decode_members(straight, universe) == {
        1: ["MSFT", "NVDA"],
        2: ["AAPL", "AMZN"],
        3: ["GOOG", "META"],
    }
    assert spread_of(straight).top_minus_bottom == pytest.approx(
        spread_of(shuffled).top_minus_bottom
    )


def test_reversing_the_input_order_returns_the_same_report() -> None:
    """A report is independent of the order its panels arrived in."""
    first = make_session(list(range(10)), [0.01] * 10, as_of=DAY_ONE, end=END_ONE)
    second = make_session(list(range(10)), [0.02] * 10, as_of=DAY_TWO, end=END_TWO)

    straight = QuantileAnalyzer().analyze([first[0], second[0]], [first[1], second[1]])
    reversed_ = QuantileAnalyzer().analyze([second[0], first[0]], [second[1], first[1]])

    assert straight == reversed_
    assert [item.as_of_date for item in straight.spreads] == [DAY_ONE, DAY_TWO]


# ---------------------------------------------------------------------------
# Small samples, empty groups, and the sessions nothing can be grouped in
# ---------------------------------------------------------------------------
def test_a_sample_smaller_than_the_group_count_fills_scattered_groups() -> None:
    """Two symbols in five groups fill quantile 3 and quantile 5, and say so."""
    assert member_sets([1.0, 2.0], quantiles=5) == {
        1: [],
        2: [],
        3: ["AAPL"],
        4: [],
        5: ["MSFT"],
    }


def test_identical_factor_values_fill_a_single_group() -> None:
    """Four equal values are one tie group, placed by its average rank."""
    assert member_sets([2.0, 2.0, 2.0, 2.0], quantiles=3) == {
        1: [],
        2: ["AAPL", "MSFT", "NVDA", "AMZN"],
        3: [],
    }


def test_a_session_with_nothing_to_group_still_reports_every_group() -> None:
    """No usable pair is a result: ``Q`` empty rows and no spread."""
    report = report_for(
        [1.0, 2.0, 3.0, 4.0],
        [0.01, 0.02, 0.03, 0.04],
        eligible=[False] * 4,
    )
    groups = groups_of(report)
    spread = spread_of(report)

    assert sorted(groups) == [1, 2, 3, 4, 5]
    for item in groups.values():
        assert item.universe_size == 4
        assert item.paired_count == 0
        assert item.paired_coverage == 0.0
        assert item.member_count == 0
        assert item.member_coverage == 0.0
        assert item.mean_forward_return is None

    assert spread.bottom_mean_forward_return is None
    assert spread.top_mean_forward_return is None
    assert spread.top_minus_bottom is None


def test_the_top_group_is_never_empty_when_anything_can_be_grouped() -> None:
    """The highest value always lands in the highest group; the bottom can be empty."""
    report = report_for([1.0, 2.0], [0.01, 0.02], quantiles=5)
    groups = groups_of(report)

    assert groups[1].member_count == 0
    assert groups[5].member_count == 1
    assert spread_of(report).top_minus_bottom is None


def test_an_empty_group_never_receives_a_zero_mean() -> None:
    """``None`` and ``0.0`` are different statements, and the object keeps them apart."""
    report = report_for([1.0, 2.0], [0.0, 0.0], quantiles=5)
    groups = groups_of(report)

    assert groups[1].mean_forward_return is None
    assert groups[3].mean_forward_return == 0.0
    assert groups[5].mean_forward_return == 0.0


# ---------------------------------------------------------------------------
# Pairing: which rows may enter the sort at all
# ---------------------------------------------------------------------------
def test_an_ineligible_row_enters_no_group() -> None:
    """Eligibility is the factor's own veto, and it is respected."""
    values = list(range(10))
    eligible = [False] + [True] * 8 + [False]
    weights = [float(2**position) for position in range(10)]

    panel, labels = make_session(values, weights, eligible=eligible)
    report = QuantileAnalyzer(quantiles=5).analyze([panel], [labels])

    assert all(item.paired_count == 8 for item in report.observations)
    assert all(item.paired_coverage == 0.8 for item in report.observations)
    members = decode_members(report, SYMBOLS)
    assert "AAPL" not in members[1]
    assert "NFLX" not in members[5]
    assert sum(len(group) for group in members.values()) == 8


@pytest.mark.parametrize("value", [NAN, INF, -INF])
def test_a_non_finite_factor_value_enters_no_group(value: float) -> None:
    """``NaN`` and infinities have no order, so they cannot be ranked."""
    report = report_for([1.0, value, 3.0, 4.0], [0.01, 0.02, 0.03, 0.04], quantiles=2)

    assert all(item.paired_count == 3 for item in report.observations)
    assert all(item.paired_coverage == 0.75 for item in report.observations)
    assert sum(item.member_count for item in report.observations) == 3


def test_an_unavailable_label_enters_no_group() -> None:
    """The horizon has not been traded out yet, so the row is not grouped."""
    report = report_for(
        [1.0, 2.0, 3.0, 4.0],
        [0.01, 0.02, 0.03, 0.04],
        quantiles=2,
        available=[True, False, True, True],
    )

    assert all(item.paired_count == 3 for item in report.observations)
    assert sum(item.member_count for item in report.observations) == 3
    # the unavailable symbol is the second one, which would be in quantile 1
    assert groups_of(report)[1].member_count == 1


def test_a_label_that_contradicts_itself_never_reaches_the_analysis() -> None:
    """The label contract refuses a ``NaN`` or ``inf`` that claims to be available."""
    with pytest.raises(ForwardReturnError, match="is_available=True"):
        make_labels(SYMBOLS, [0.01, NAN, 0.03, 0.04] + [0.05] * 6)

    with pytest.raises(ForwardReturnError):
        make_labels(SYMBOLS, [0.01, INF, 0.03, 0.04] + [0.05] * 6)


def test_every_reason_at_once_leaves_only_the_usable_rows() -> None:
    """Coverage counts the whole cross-section, not just the grouped rows."""
    report = report_for(
        [1.0, 2.0, 3.0, 4.0],
        [0.01, 0.02, 0.03, 0.04],
        quantiles=2,
        eligible=[True, False, True, True],
        available=[True, True, False, True],
    )
    groups = groups_of(report)

    assert all(item.universe_size == 4 for item in groups.values())
    assert all(item.paired_count == 2 for item in groups.values())
    assert all(item.paired_coverage == 0.5 for item in groups.values())
    assert sum(item.member_count for item in groups.values()) == 2
    assert groups[1].member_count == 1
    assert groups[2].member_count == 1


def test_the_grouped_sample_is_exactly_the_sample_the_ic_analysis_counts() -> None:
    """The two analyses must never disagree about which rows exist."""
    panel, labels = make_session(
        [1.0, NAN, 3.0, 4.0, 5.0, 6.0, 7.0, INF, 9.0, 10.0],
        [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10],
        eligible=[True, True, False, True, True, True, True, True, True, True],
        available=[True, True, True, True, False, True, True, True, True, True],
    )

    quantiles = QuantileAnalyzer(quantiles=4).analyze([panel], [labels])
    ic = ICAnalyzer().analyze([panel], [labels])

    assert quantiles.observations[0].universe_size == ic.observations[0].universe_size
    assert quantiles.observations[0].paired_count == ic.observations[0].paired_count
    assert quantiles.observations[0].paired_coverage == ic.observations[0].pair_coverage


# ---------------------------------------------------------------------------
# Many sessions: order, holes and changing universes
# ---------------------------------------------------------------------------
def test_sessions_are_reported_in_date_order_and_holes_stay_holes() -> None:
    """Input order is irrelevant; a date without a panel is not invented."""
    sessions = [
        make_session(list(range(10)), [0.01] * 10, as_of=DAY_FOUR, end=END_FOUR),
        make_session(list(range(10)), [0.02] * 10, as_of=DAY_ONE, end=END_ONE),
        make_session(list(range(10)), [0.03] * 10, as_of=DAY_THREE, end=END_THREE),
    ]

    report = QuantileAnalyzer(quantiles=3).analyze(
        [panel for panel, _ in sessions], [label for _, label in sessions]
    )

    assert [item.as_of_date for item in report.spreads] == [DAY_ONE, DAY_THREE, DAY_FOUR]
    assert [item.as_of_date for item in report.observations] == [
        DAY_ONE,
        DAY_ONE,
        DAY_ONE,
        DAY_THREE,
        DAY_THREE,
        DAY_THREE,
        DAY_FOUR,
        DAY_FOUR,
        DAY_FOUR,
    ]
    assert [item.quantile for item in report.observations[:3]] == [1, 2, 3]
    assert DAY_TWO not in {item.as_of_date for item in report.observations}


def test_a_different_universe_on_each_session_is_allowed() -> None:
    """A point-in-time pool changes; each session is grouped within itself."""
    small = make_session([1.0, 2.0, 3.0, 4.0], [0.01] * 4, as_of=DAY_ONE, end=END_ONE)
    large = make_session(list(range(10)), [0.02] * 10, as_of=DAY_TWO, end=END_TWO)

    report = QuantileAnalyzer(quantiles=2).analyze([small[0], large[0]], [small[1], large[1]])

    assert {item.universe_size for item in groups_of(report, DAY_ONE).values()} == {4}
    assert {item.universe_size for item in groups_of(report, DAY_TWO).values()} == {10}
    assert all(item.end_date == END_ONE for item in groups_of(report, DAY_ONE).values())
    assert all(item.end_date == END_TWO for item in groups_of(report, DAY_TWO).values())


def test_generators_are_accepted_and_consumed_once() -> None:
    """A caller may hand over a generator; nothing re-reads it behind their back."""
    panel, labels = make_session(list(range(10)), [0.01] * 10)

    report = QuantileAnalyzer(quantiles=2).analyze(
        (item for item in [panel]), (item for item in [labels])
    )

    assert len(report.observations) == 2
    assert spread_of(report) is not None


def test_one_session_produces_one_spread() -> None:
    """The spread table is per session, whatever the number of groups."""
    report = report_for(list(range(10)), [0.01] * 10, quantiles=10)

    assert len(report.observations) == 10
    assert len(report.spreads) == 1
    assert report.spreads[0].as_of_date == DAY_ONE


# ---------------------------------------------------------------------------
# Refusals: an input that cannot be joined is refused, never intersected
# ---------------------------------------------------------------------------
def test_an_empty_panel_sequence_is_refused() -> None:
    """An empty history has nothing to group."""
    _, labels = make_session([1.0], [0.01])

    with pytest.raises(QuantileAnalysisError, match="at least one FactorPanel"):
        QuantileAnalyzer().analyze([], [labels])


def test_an_empty_label_sequence_is_refused() -> None:
    """A factor panel without labels measures nothing."""
    panel, _ = make_session([1.0], [0.01])

    with pytest.raises(QuantileAnalysisError, match="at least one ForwardReturnPanel"):
        QuantileAnalyzer().analyze([panel], [])


def test_a_single_panel_is_refused() -> None:
    """One panel is one session; the argument is a sequence of them."""
    panel, labels = make_session([1.0], [0.01])

    with pytest.raises(QuantileAnalysisError, match="single FactorPanel"):
        QuantileAnalyzer().analyze(panel, [labels])  # type: ignore[arg-type]

    with pytest.raises(QuantileAnalysisError, match="single ForwardReturnPanel"):
        QuantileAnalyzer().analyze([panel], labels)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", ["panels", b"panels", None, 5, 3.5, object()])
def test_a_non_iterable_input_is_refused(value: object) -> None:
    """A string, a number or ``None`` is not a sequence of panels."""
    with pytest.raises(QuantileAnalysisError):
        QuantileAnalyzer().analyze(value, [])  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [{"panel": 1}, pd.DataFrame(), ("AAPL",)])
def test_a_foreign_object_in_the_sequence_is_refused(value: object) -> None:
    """Something that is not a panel cannot be grouped with labels."""
    _, labels = make_session([1.0], [0.01])

    with pytest.raises(QuantileAnalysisError, match="not a FactorPanel"):
        QuantileAnalyzer().analyze([value], [labels])  # type: ignore[list-item]


def test_a_label_object_in_the_panel_argument_is_refused() -> None:
    """The two sequences have different types for a reason."""
    panel, labels = make_session([1.0], [0.01])

    with pytest.raises(QuantileAnalysisError, match="not a FactorPanel"):
        QuantileAnalyzer().analyze([labels], [labels])  # type: ignore[list-item]

    with pytest.raises(QuantileAnalysisError, match="not a ForwardReturnPanel"):
        QuantileAnalyzer().analyze([panel], [panel])  # type: ignore[list-item]


def test_mixed_factor_names_are_refused() -> None:
    """One call measures one factor."""
    first = make_session([1.0, 2.0], [0.01, 0.02], as_of=DAY_ONE, end=END_ONE)
    second = make_session(
        [1.0, 2.0], [0.01, 0.02], as_of=DAY_TWO, end=END_TWO, name="book_to_price"
    )

    with pytest.raises(QuantileAnalysisError, match="one factor at one version") as caught:
        QuantileAnalyzer().analyze([first[0], second[0]], [first[1], second[1]])

    assert any("book_to_price" in problem for problem in caught.value.problems)


def test_mixed_factor_versions_are_refused() -> None:
    """A version is part of the identity of the factor being measured."""
    first = make_session([1.0, 2.0], [0.01, 0.02], as_of=DAY_ONE, end=END_ONE)
    second = make_session([1.0, 2.0], [0.01, 0.02], as_of=DAY_TWO, end=END_TWO, version="2.0.0")

    with pytest.raises(QuantileAnalysisError, match="one factor at one version"):
        QuantileAnalyzer().analyze([first[0], second[0]], [first[1], second[1]])


def test_mixed_label_horizons_are_refused() -> None:
    """Groups measured over one session cannot be averaged with another horizon."""
    first = make_session([1.0, 2.0], [0.01, 0.02], as_of=DAY_ONE, end=END_ONE)
    second = make_session([1.0, 2.0], [0.01, 0.02], as_of=DAY_TWO, end=END_TWO, horizon=5)

    with pytest.raises(QuantileAnalysisError, match="one factor at one horizon"):
        QuantileAnalyzer().analyze([first[0], second[0]], [first[1], second[1]])


def test_a_repeated_panel_date_is_refused() -> None:
    """A session is grouped once."""
    first = make_session([1.0, 2.0], [0.01, 0.02], as_of=DAY_ONE, end=END_ONE)
    second = make_session([3.0, 4.0], [0.03, 0.04], as_of=DAY_ONE, end=END_ONE)

    with pytest.raises(QuantileAnalysisError, match="repeats as_of_date") as caught:
        QuantileAnalyzer().analyze([first[0], second[0]], [first[1], second[1]])

    assert any("factor_panels[1]" in problem for problem in caught.value.problems)


def test_a_repeated_label_date_is_refused() -> None:
    """Two label objects for one session are ambiguous, so they are refused."""
    first = make_session([1.0, 2.0], [0.01, 0.02], as_of=DAY_ONE, end=END_ONE)
    second = make_session([1.0, 2.0], [0.03, 0.04], as_of=DAY_ONE, end=END_ONE)

    with pytest.raises(QuantileAnalysisError, match="repeats as_of_date"):
        QuantileAnalyzer().analyze([first[0]], [first[1], second[1]])


def test_a_date_present_on_one_side_only_is_refused() -> None:
    """Analysing the intersection would hide exactly the broken join."""
    first = make_session([1.0, 2.0], [0.01, 0.02], as_of=DAY_ONE, end=END_ONE)
    second = make_session([1.0, 2.0], [0.01, 0.02], as_of=DAY_TWO, end=END_TWO)

    with pytest.raises(QuantileAnalysisError, match="no labels for as_of_date") as caught:
        QuantileAnalyzer().analyze([first[0], second[0]], [first[1]])

    assert any(DAY_TWO.isoformat() in problem for problem in caught.value.problems)

    with pytest.raises(QuantileAnalysisError, match="with no factor panel"):
        QuantileAnalyzer().analyze([first[0]], [first[1], second[1]])


def test_a_panel_and_its_labels_must_share_an_identity() -> None:
    """The labels were measured for one factor; another factor's labels fit nothing."""
    panel, _ = make_session([1.0, 2.0], [0.01, 0.02])
    labels = make_labels(None, [0.01, 0.02], name="book_to_price")

    with pytest.raises(QuantileAnalysisError, match="disagree on identity") as caught:
        QuantileAnalyzer().analyze([panel], [labels])

    assert any("book_to_price" in problem for problem in caught.value.problems)


def test_a_universe_of_another_length_is_refused() -> None:
    """Symbols cannot be paired positionally when the pools differ in size."""
    panel, _ = make_session([1.0, 2.0, 3.0, 4.0], [0.01] * 4)
    labels = make_labels(SYMBOLS[:3], [0.01, 0.02, 0.03])

    with pytest.raises(QuantileAnalysisError, match="do not share a universe") as caught:
        QuantileAnalyzer().analyze([panel], [labels])

    assert any("4 symbol(s)" in problem for problem in caught.value.problems)


def test_a_universe_with_other_members_is_refused() -> None:
    """A different ticker in the same slot would group the wrong stock."""
    panel, _ = make_session([1.0, 2.0, 3.0, 4.0], [0.01] * 4)
    labels = make_labels(("AAPL", "MSFT", "NVDA", "ZZZZ"), [0.01] * 4)

    with pytest.raises(QuantileAnalysisError, match="do not share a universe") as caught:
        QuantileAnalyzer().analyze([panel], [labels])

    assert any("ZZZZ" in problem for problem in caught.value.problems)


def test_a_universe_in_a_different_order_is_refused() -> None:
    """Same symbols in another order is still a different pairing."""
    panel, labels = make_session([1.0, 2.0, 3.0, 4.0], [0.01, 0.02, 0.03, 0.04])
    reordered = make_labels(
        tuple(reversed(labels.universe)),
        labels.frame["forward_return"].tolist()[::-1],
    )

    with pytest.raises(QuantileAnalysisError, match="do not share a universe") as caught:
        QuantileAnalyzer().analyze([panel], [reordered])

    assert any("different order" in problem for problem in caught.value.problems)


def test_every_problem_is_collected_in_one_refusal() -> None:
    """One failed call names all of them, not just the first."""
    first = make_session([1.0, 2.0], [0.01, 0.02], as_of=DAY_ONE, end=END_ONE)
    second = make_session(
        [1.0, 2.0], [0.01, 0.02], as_of=DAY_TWO, end=END_TWO, name="book_to_price"
    )
    third = make_session([1.0, 2.0], [0.01, 0.02], as_of=DAY_THREE, end=END_THREE)

    with pytest.raises(QuantileAnalysisError) as caught:
        QuantileAnalyzer().analyze([first[0], second[0], third[0]], [first[1], second[1]])

    problems = caught.value.problems
    assert len(problems) >= 2
    assert any("book_to_price" in problem for problem in problems)
    assert any("no labels for as_of_date" in problem for problem in problems)


# ---------------------------------------------------------------------------
# QuantileObservation: a frozen row that cannot contradict itself
# ---------------------------------------------------------------------------
def test_an_observation_is_frozen() -> None:
    """A reported group is evidence, not a scratch row to be edited."""
    item = observation()

    with pytest.raises(FrozenInstanceError):
        item.member_count = 3  # type: ignore[misc]

    assert item.quantile == 1
    assert item.mean_forward_return == 0.01


def test_an_observation_accepts_a_coherent_empty_group() -> None:
    """No member, no coverage, no mean -- a legal and meaningful row."""
    item = observation(member_count=0, member_coverage=0.0, mean_forward_return=None)

    assert item.member_count == 0
    assert item.mean_forward_return is None


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"as_of_date": datetime(2024, 1, 5, 16)}, "datetime.date"),
        ({"as_of_date": "2024-01-05"}, "datetime.date"),
        ({"end_date": DAY_ONE}, "later than as_of_date"),
        ({"end_date": date(2024, 1, 4)}, "later than as_of_date"),
        ({"factor_name": "Momentum 12"}, "snake_case"),
        ({"factor_version": ""}, "non-empty version"),
        ({"factor_version": 1}, "non-empty version"),
        ({"horizon_sessions": 0}, "positive integer"),
        ({"horizon_sessions": 1.5}, "positive integer"),
        ({"quantiles": 1}, "at least 2"),
        ({"quantiles": True}, "at least 2"),
        ({"quantile": 6}, "must lie in"),
        ({"quantile": 0}, "must lie in"),
        ({"universe_size": 0}, "at least 1"),
        ({"paired_count": 11}, "cannot exceed universe_size"),
        ({"member_count": 11}, "cannot exceed paired_count"),
        ({"paired_coverage": 0.9}, "paired_count / universe_size"),
        ({"member_coverage": 0.3}, "member_count / universe_size"),
        ({"mean_forward_return": NAN}, "never NaN"),
        ({"mean_forward_return": INF}, "finite real number"),
        ({"mean_forward_return": "-0.01"}, "finite real number"),
        ({"mean_forward_return": None}, "always has a mean"),
        ({"member_count": 0, "member_coverage": 0.0}, "cannot exist for an empty group"),
        ({"member_coverage": NAN}, "finite real number"),
    ],
)
def test_an_observation_refuses_every_self_contradiction(
    changes: dict[str, object], expected: str
) -> None:
    """A hand-built row that lies about itself cannot exist."""
    with pytest.raises(QuantileAnalysisError, match=expected) as caught:
        observation(**changes)

    assert caught.value.problems


def test_the_observation_repr_fits_on_one_line() -> None:
    """A log line shows the identity, the group and the number, and nothing else."""
    text = repr(observation(member_count=3, member_coverage=0.3, mean_forward_return=0.02))

    assert "\n" not in text
    assert text.startswith("QuantileObservation(as_of_date='2024-01-05'")
    assert "quantile=1/5" in text
    assert "members=3" in text
    assert "mean_forward_return=0.02" in text


# ---------------------------------------------------------------------------
# QuantileSpreadObservation: the extremes, and only the extremes
# ---------------------------------------------------------------------------
def test_a_spread_observation_is_frozen() -> None:
    """The headline number is frozen like the groups it came from."""
    item = spread_observation()

    with pytest.raises(FrozenInstanceError):
        item.top_minus_bottom = 0.5  # type: ignore[misc]

    assert item.top_minus_bottom == 0.02


def test_a_spread_observation_accepts_two_empty_extremes() -> None:
    """No bottom mean, no top mean, no spread -- one coherent statement."""
    item = spread_observation(
        bottom_mean_forward_return=None,
        top_mean_forward_return=None,
        top_minus_bottom=None,
    )

    assert item.top_minus_bottom is None


def test_a_spread_observation_accepts_an_unrepresentable_difference_as_none() -> None:
    """Two finite extremes may still differ past the float range: None is the answer."""
    item = spread_observation(
        bottom_mean_forward_return=-1e308,
        top_mean_forward_return=1e308,
        top_minus_bottom=None,
    )

    assert item.top_minus_bottom is None


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"as_of_date": datetime(2024, 1, 5)}, "datetime.date"),
        ({"end_date": END_ONE.replace(day=4)}, "later than as_of_date"),
        ({"factor_name": "not a name"}, "snake_case"),
        ({"factor_version": ""}, "non-empty version"),
        ({"horizon_sessions": -1}, "positive integer"),
        ({"quantiles": 1}, "at least 2"),
        ({"top_minus_bottom": NAN}, "never NaN"),
        ({"bottom_mean_forward_return": INF}, "finite real number"),
        ({"bottom_mean_forward_return": None}, "bottom group is empty"),
        ({"top_mean_forward_return": None}, "top group is empty"),
        (
            {"bottom_mean_forward_return": None, "top_minus_bottom": 0.02},
            "bottom group is empty",
        ),
        (
            {
                "bottom_mean_forward_return": 0.01,
                "top_mean_forward_return": 0.03,
                "top_minus_bottom": None,
            },
            "difference of two numbers always exists",
        ),
        (
            {
                "bottom_mean_forward_return": 0.01,
                "top_mean_forward_return": 0.03,
                "top_minus_bottom": 0.5,
            },
            "must be top_mean_forward_return - bottom_mean_forward_return",
        ),
        (
            {
                "bottom_mean_forward_return": -1e308,
                "top_mean_forward_return": 1e308,
                "top_minus_bottom": 0.0,
            },
            "past the float range",
        ),
    ],
)
def test_a_spread_observation_refuses_every_self_contradiction(
    changes: dict[str, object], expected: str
) -> None:
    """A spread that is not its own difference cannot exist."""
    with pytest.raises(QuantileAnalysisError, match=expected) as caught:
        spread_observation(**changes)

    assert caught.value.problems


def test_the_spread_repr_names_both_extremes() -> None:
    """A log line shows the subtraction that produced the number."""
    text = repr(spread_observation())

    assert "\n" not in text
    assert "bottom=0.01" in text
    assert "top=0.03" in text
    assert "top_minus_bottom=0.02" in text


# ---------------------------------------------------------------------------
# QuantileReport: the two tables must agree with each other
# ---------------------------------------------------------------------------
def coherent_report() -> QuantileReport:
    """A hand-built report whose five groups and single spread agree."""
    observations = five_groups()
    spread = spread_observation(
        bottom_mean_forward_return=0.05,
        top_mean_forward_return=-0.03,
        top_minus_bottom=-0.08,
    )
    return report_of(observations, (spread,))


def test_a_coherent_report_is_accepted_and_frozen() -> None:
    """A report built from consistent rows exists, and cannot be edited."""
    report = coherent_report()

    assert len(report.observations) == 5
    assert len(report.spreads) == 1
    assert report.quantiles == 5

    with pytest.raises(FrozenInstanceError):
        report.quantiles = 3  # type: ignore[misc]

    assert report.observations[0].quantile == 1
    assert report.spreads[0].top_minus_bottom == -0.08


def test_the_report_repr_summarizes_without_listing_the_rows() -> None:
    """A log line says which factor, which horizon, how many rows."""
    text = repr(coherent_report())

    assert "\n" not in text
    assert text.startswith("QuantileReport(factor=momentum_12_1@1.0.0")
    assert "quantiles=5" in text
    assert "observations=5" in text
    assert "spreads=1" in text


def test_a_report_cannot_hold_a_single_observation_object() -> None:
    """One group is one table row; the field is the sequence of them."""
    with pytest.raises(QuantileAnalysisError, match="wrap it as") as caught:
        report_of(five_groups()[0], ())

    assert caught.value.problems


def test_a_report_cannot_hold_an_empty_table() -> None:
    """A report of nothing describes no session."""
    with pytest.raises(QuantileAnalysisError, match="at least one QuantileObservation"):
        report_of((), (spread_observation(),))

    with pytest.raises(QuantileAnalysisError, match="at least one QuantileSpreadObservation"):
        report_of(five_groups(), ())


@pytest.mark.parametrize("value", ["observations", b"observations", None, 5])
def test_a_report_refuses_a_non_iterable_table(value: object) -> None:
    """A string or a number is not a sequence of rows."""
    with pytest.raises(QuantileAnalysisError):
        report_of(value, (spread_observation(),))


def test_a_report_refuses_a_row_of_the_wrong_type() -> None:
    """A spread is not an observation, and the other way round."""
    with pytest.raises(QuantileAnalysisError, match="not a QuantileObservation"):
        report_of((*five_groups()[:4], spread_observation()), (spread_observation(),))

    with pytest.raises(QuantileAnalysisError, match="not a QuantileSpreadObservation"):
        report_of(five_groups(), (observation(),))


def test_a_report_refuses_a_missing_group() -> None:
    """Every session reports every group, empty or not."""
    with pytest.raises(QuantileAnalysisError, match="holds 4 group observation") as caught:
        report_of(five_groups()[:4], (spread_observation(),))

    assert any("every session reports every group" in problem for problem in caught.value.problems)


def test_a_report_refuses_rows_out_of_order() -> None:
    """Observations ascend by date and then by quantile."""
    observations = list(five_groups())
    observations[0], observations[1] = observations[1], observations[0]

    with pytest.raises(QuantileAnalysisError, match="rows are kept in ascending date order"):
        report_of(tuple(observations), (spread_observation(),))


def test_a_report_refuses_a_group_numbered_twice() -> None:
    """Two rows for one quantile of one session cannot both be true."""
    observations = (*five_groups()[:4], replace(five_groups()[4], quantile=4))

    with pytest.raises(QuantileAnalysisError, match="repeats") as caught:
        report_of(observations, (spread_observation(),))

    problems = caught.value.problems
    assert any("one group is measured once per session" in problem for problem in problems)
    assert any("reports quantiles" in problem for problem in problems)


def test_a_report_refuses_a_row_from_another_factor() -> None:
    """One report measures one factor at one version."""
    observations = (replace(five_groups()[0], factor_version="2.0.0"), *five_groups()[1:])

    with pytest.raises(QuantileAnalysisError, match="one report measures one factor"):
        report_of(observations, (spread_observation(),))


def test_a_report_refuses_a_row_with_another_horizon() -> None:
    """One report measures one horizon."""
    observations = (replace(five_groups()[0], horizon_sessions=5), *five_groups()[1:])

    with pytest.raises(QuantileAnalysisError, match="one report measures one horizon"):
        report_of(observations, (spread_observation(),))


def test_a_report_refuses_a_row_with_another_grouping() -> None:
    """One report has one grouping."""
    observations = (replace(five_groups()[0], quantiles=3), *five_groups()[1:])

    with pytest.raises(QuantileAnalysisError, match="one report has one grouping"):
        report_of(observations, (spread_observation(),))


def spread_for(
    day: date = DAY_ONE,
    end: date = END_ONE,
    means: Sequence[float] = (0.05, 0.03, 0.01, -0.01, -0.03),
) -> QuantileSpreadObservation:
    """The spread implied by a set of group means, for one session."""
    return spread_observation(
        as_of_date=day,
        end_date=end,
        bottom_mean_forward_return=means[0],
        top_mean_forward_return=means[-1],
        top_minus_bottom=means[-1] - means[0],
    )


def two_day_report() -> QuantileReport:
    """Two coherent sessions, each with its own groups and spread."""
    second_means = (0.02, 0.01, 0.0, -0.01, -0.02)
    observations = (
        *five_groups(day=DAY_ONE, end=END_ONE),
        *five_groups(day=DAY_TWO, end=END_TWO, means=second_means),
    )
    spreads = (spread_for(DAY_ONE, END_ONE), spread_for(DAY_TWO, END_TWO, second_means))
    return report_of(observations, spreads)


def test_two_coherent_sessions_are_accepted() -> None:
    """A multi-session report keeps its blocks apart and its spreads in step."""
    report = two_day_report()

    assert len(report.observations) == 10
    assert len(report.spreads) == 2
    assert [item.as_of_date for item in report.spreads] == [DAY_ONE, DAY_TWO]
    assert groups_of(report, DAY_TWO)[5].mean_forward_return == -0.02


def test_a_report_refuses_groups_of_one_session_with_different_windows() -> None:
    """One session's labels share one end date."""
    observations = (
        replace(five_groups()[0], end_date=date(2024, 1, 9)),
        *five_groups()[1:],
    )

    with pytest.raises(QuantileAnalysisError, match="share one window"):
        report_of(observations, (spread_observation(),))


def test_a_report_refuses_groups_of_one_session_with_different_universes() -> None:
    """One session has one cross-section."""
    odd = replace(
        five_groups()[0],
        universe_size=20,
        paired_count=10,
        paired_coverage=0.5,
        member_count=2,
        member_coverage=0.1,
    )

    with pytest.raises(QuantileAnalysisError, match="one session has one cross-section"):
        report_of((odd, *five_groups()[1:]), (spread_observation(),))


def test_a_report_refuses_groups_that_do_not_share_one_sample() -> None:
    """The groups of one session are cut from the same paired sample."""
    odd = replace(
        five_groups()[0],
        paired_count=8,
        paired_coverage=0.8,
        member_count=2,
        member_coverage=0.2,
    )

    with pytest.raises(QuantileAnalysisError, match="share one sample"):
        report_of((odd, *five_groups()[1:]), (spread_observation(),))


def test_a_report_refuses_member_counts_that_do_not_add_up() -> None:
    """Every grouped symbol belongs to exactly one group."""
    odd = replace(five_groups()[0], member_count=1, member_coverage=0.1)

    with pytest.raises(QuantileAnalysisError) as caught:
        report_of((odd, *five_groups()[1:]), (spread_observation(),))

    problems = caught.value.problems
    assert any("member_count sums to 9" in problem for problem in problems)
    assert any("member_coverage sums to" in problem for problem in problems)


def test_a_report_refuses_a_session_without_a_spread() -> None:
    """Every measured session carries exactly one top-minus-bottom row."""
    observations = (
        *five_groups(day=DAY_ONE, end=END_ONE),
        *five_groups(day=DAY_TWO, end=END_TWO),
    )

    with pytest.raises(QuantileAnalysisError, match="no top-minus-bottom row") as caught:
        report_of(observations, (spread_for(),))

    assert any(DAY_TWO.isoformat() in problem for problem in caught.value.problems)


def test_a_report_refuses_a_spread_without_its_session() -> None:
    """A spread is the difference of two measured groups."""
    with pytest.raises(QuantileAnalysisError, match="with no observation"):
        report_of(five_groups(), (spread_observation(), spread_for(DAY_TWO, END_TWO)))


def test_a_report_refuses_spreads_out_of_order() -> None:
    """Spreads ascend by date, exactly like the observations."""
    report = two_day_report()

    with pytest.raises(QuantileAnalysisError, match="before spreads"):
        report_of(report.observations, tuple(reversed(report.spreads)))


def test_a_report_refuses_a_spread_from_another_factor() -> None:
    """The spread belongs to the same factor as the groups it subtracts."""
    with pytest.raises(QuantileAnalysisError, match="one report measures one factor"):
        report_of(five_groups(), (replace(spread_for(), factor_version="2.0.0"),))


def test_a_report_refuses_a_spread_whose_extremes_disagree_with_its_groups() -> None:
    """The top of the spread must be the top group's own mean."""
    wrong_top = spread_for(means=(0.05, 0.03, 0.01, -0.01, -0.02))

    with pytest.raises(QuantileAnalysisError, match="reports top_mean_forward_return"):
        report_of(five_groups(), (wrong_top,))


def test_a_report_refuses_a_spread_of_another_window() -> None:
    """One session has one label window."""
    with pytest.raises(QuantileAnalysisError, match="one session has one label window"):
        report_of(five_groups(), (spread_for(DAY_ONE, date(2024, 1, 9)),))


# ---------------------------------------------------------------------------
# The two tables: fixed columns, fixed order, always a copy
# ---------------------------------------------------------------------------
def test_the_observation_table_has_the_documented_columns_in_order() -> None:
    """The column constant *is* the table's shape."""
    frame = report_for(list(range(10)), [0.01] * 10).to_frame()

    assert tuple(frame.columns) == QUANTILE_OBSERVATION_COLUMNS
    assert QUANTILE_OBSERVATION_COLUMNS == (
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


def test_the_observation_table_holds_one_row_per_date_and_group() -> None:
    """Rows ascend by date, then by quantile; no per-symbol data is mixed in."""
    report = two_day_report()
    frame = report.to_frame()

    assert len(frame) == 10
    assert frame["as_of_date"].tolist() == [pd.Timestamp(DAY_ONE)] * 5 + [pd.Timestamp(DAY_TWO)] * 5
    assert frame["quantile"].tolist() == [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
    assert frame["mean_forward_return"].tolist() == pytest.approx(
        [item.mean_forward_return for item in report.observations]
    )
    assert frame["member_count"].tolist() == [item.member_count for item in report.observations]


def test_the_observation_table_has_stable_dtypes() -> None:
    """Dates, counts and returns keep the dtypes a downstream analysis expects."""
    frame = report_for(list(range(10)), [0.01] * 10).to_frame()

    expected = {
        "as_of_date": "datetime64[s]",
        "end_date": "datetime64[s]",
        "factor_name": "object",
        "factor_version": "object",
        "horizon_sessions": "int64",
        "quantiles": "int64",
        "quantile": "int64",
        "universe_size": "int64",
        "paired_count": "int64",
        "paired_coverage": "float64",
        "member_count": "int64",
        "member_coverage": "float64",
        "mean_forward_return": "float64",
    }

    assert {column: str(dtype) for column, dtype in frame.dtypes.items()} == expected


def test_an_empty_group_is_nan_in_the_table_but_none_in_the_object() -> None:
    """The frame carries the absence; the observation keeps the meaning."""
    report = report_for([1.0, 2.0], [0.01, 0.02], quantiles=5)
    frame = report.to_frame()

    assert frame["member_count"].tolist() == [0, 0, 1, 0, 1]
    assert math.isnan(frame["mean_forward_return"].tolist()[0])
    assert groups_of(report)[1].mean_forward_return is None


def test_the_observation_table_is_a_copy() -> None:
    """Editing the frame cannot reach back into the report."""
    report = report_for(list(range(10)), [0.01] * 10)
    first = report.to_frame()
    first.loc[0, "member_count"] = 99
    first.loc[0, "mean_forward_return"] = -1.0

    second = report.to_frame()

    assert second.loc[0, "member_count"] == 2
    assert second.loc[0, "mean_forward_return"] == pytest.approx(0.01)
    assert report.observations[0].member_count == 2


def test_the_spread_table_has_the_documented_columns_in_order() -> None:
    """The spread table has its own fixed shape, one row per session."""
    frame = two_day_report().spread_to_frame()

    assert tuple(frame.columns) == QUANTILE_SPREAD_COLUMNS
    assert QUANTILE_SPREAD_COLUMNS == (
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
    assert len(frame) == 2
    assert frame["as_of_date"].tolist() == [pd.Timestamp(DAY_ONE), pd.Timestamp(DAY_TWO)]
    assert frame["top_minus_bottom"].tolist() == pytest.approx([-0.08, -0.04])


def test_the_spread_table_has_stable_dtypes() -> None:
    """Three numeric columns, one date pair, integers for the shape."""
    frame = two_day_report().spread_to_frame()

    expected = {
        "as_of_date": "datetime64[s]",
        "end_date": "datetime64[s]",
        "factor_name": "object",
        "factor_version": "object",
        "horizon_sessions": "int64",
        "quantiles": "int64",
        "bottom_mean_forward_return": "float64",
        "top_mean_forward_return": "float64",
        "top_minus_bottom": "float64",
    }

    assert {column: str(dtype) for column, dtype in frame.dtypes.items()} == expected


def test_an_undefined_spread_is_nan_in_the_table_but_none_in_the_object() -> None:
    """An undefined spread is reported, never replaced by zero."""
    report = report_for([1.0, 2.0], [0.01, 0.02], quantiles=5)
    frame = report.spread_to_frame()

    assert math.isnan(frame.loc[0, "bottom_mean_forward_return"])
    assert math.isnan(frame.loc[0, "top_minus_bottom"])
    assert frame.loc[0, "top_mean_forward_return"] == pytest.approx(0.02)
    assert report.spreads[0].top_minus_bottom is None


def test_the_spread_table_is_a_copy() -> None:
    """Editing the spread frame cannot reach back into the report."""
    report = two_day_report()
    first = report.spread_to_frame()
    first.loc[0, "top_minus_bottom"] = 0.0

    assert report.spread_to_frame().loc[0, "top_minus_bottom"] == pytest.approx(-0.08)
    assert report.spreads[0].top_minus_bottom == -0.08


def test_the_tables_of_a_real_multi_session_report_line_up() -> None:
    """Both tables describe the same sessions, in the same order."""
    sessions = [
        make_session(list(range(10)), [0.01] * 10, as_of=DAY_ONE, end=END_ONE),
        make_session(list(range(10)), [0.02] * 10, as_of=DAY_THREE, end=END_THREE),
    ]
    report = QuantileAnalyzer(quantiles=3).analyze(
        [panel for panel, _ in sessions], [label for _, label in sessions]
    )

    observations = report.to_frame()
    spreads = report.spread_to_frame()

    assert len(observations) == 6
    assert len(spreads) == 2
    assert spreads["as_of_date"].tolist() == [pd.Timestamp(DAY_ONE), pd.Timestamp(DAY_THREE)]
    assert spreads["top_minus_bottom"].tolist() == pytest.approx([0.0, 0.0])
    assert observations["paired_coverage"].tolist() == [1.0] * 6


# ---------------------------------------------------------------------------
# Purity: the inputs are read, never written
# ---------------------------------------------------------------------------
def test_the_inputs_survive_a_call_unchanged() -> None:
    """Panels and labels are evidence, and the analysis does not edit them."""
    panel, labels = make_session(
        [3.0, 1.0, 2.0, 4.0],
        [0.03, 0.01, 0.02, 0.04],
        eligible=[True, False, True, True],
        available=[True, True, True, False],
    )
    panel_frame_before = panel.frame.copy(deep=True)
    label_frame_before = labels.frame.copy(deep=True)
    panel_universe = list(panel.universe)
    label_universe = list(labels.universe)

    QuantileAnalyzer(quantiles=2).analyze([panel], [labels])

    pd.testing.assert_frame_equal(panel.frame, panel_frame_before)
    pd.testing.assert_frame_equal(labels.frame, label_frame_before)
    assert list(panel.universe) == panel_universe
    assert list(labels.universe) == label_universe
    assert list(panel.frame.columns) == list(factor_sys.FACTOR_PANEL_COLUMNS)
    assert list(labels.frame.columns) == list(factor_sys.FORWARD_RETURN_COLUMNS)


def test_the_caller_sequences_survive_a_call_unchanged() -> None:
    """Appending to the caller's list afterwards still works, and re-analysis agrees."""
    panel, labels = make_session([1.0, 2.0], [0.01, 0.02])
    panels = [panel]
    labels_list = [labels]

    first = QuantileAnalyzer(quantiles=2).analyze(panels, labels_list)
    second = QuantileAnalyzer(quantiles=2).analyze(panels, labels_list)

    assert panels == [panel]
    assert labels_list == [labels]
    assert first == second
    assert first.to_frame().equals(second.to_frame())


def test_the_report_holds_no_frame() -> None:
    """The canonical result is the frozen object, not a table."""
    report = report_for(list(range(10)), [0.01] * 10)

    assert isinstance(report.observations, tuple)
    assert isinstance(report.spreads, tuple)
    for name in ("frame", "table", "data"):
        assert not hasattr(report, name)


def test_the_public_objects_are_documented() -> None:
    """Every exported class carries the docstring a reader needs, not a placeholder."""
    for name in (
        "QuantileAnalyzer",
        "QuantileObservation",
        "QuantileReport",
        "QuantileSpreadObservation",
    ):
        doc = getattr(quantile_analysis_module, name).__doc__

        assert doc, name
        assert len(doc.strip().splitlines()) > 3, name
        assert not doc.strip().startswith(("TODO", "FIXME", "...")), name

    assert quantile_analysis_module.__doc__
    assert len(quantile_analysis_module.__doc__.strip().splitlines()) > 40
    assert inspect.getdoc(QuantileAnalyzer.analyze)


# ---------------------------------------------------------------------------
# The architectural boundary: stdlib, pandas and the two contracts only
# ---------------------------------------------------------------------------
ALLOWED_IMPORTS: frozenset[str] = frozenset(
    {
        "__future__",
        "collections.abc",
        "dataclasses",
        "datetime",
        "math",
        "numbers",
        "pandas",
        "re",
        "factor_sys.contracts",
        "factor_sys.errors",
        "factor_sys.forward_returns",
    }
)

FORBIDDEN_IMPORT_PREFIXES: tuple[str, ...] = (
    "factor_sys.factor_store",
    "factor_sys.runner",
    "factor_sys.quick_analysis",
    "factor_sys.ic_analysis",
    "factor_sys.registry",
    "factor_sys.quality",
    "factor_sys.market_data",
    "factor_sys.universe",
    "factor_sys.factors",
    "factors",
    "data_sys",
    "duckdb",
    "numpy",
    "pyarrow",
    "scipy",
    "sklearn",
    "requests",
    "httpx",
    "urllib",
    "socket",
    "yfinance",
    "os",
    "pathlib",
    "shutil",
    "subprocess",
    "tempfile",
    "time",
    "calendar",
    "sqlite3",
    "csv",
    "pickle",
)

FORBIDDEN_CALLS: frozenset[str] = frozenset(
    {
        "__import__",
        "compute",
        "connect",
        "eval",
        "exec",
        "get_calendar",
        "glob",
        "iterdir",
        "listdir",
        "mkdir",
        "now",
        "open",
        "popen",
        "read_csv",
        "read_parquet",
        "sessions_in_range",
        "system",
        "to_csv",
        "to_parquet",
        "to_pickle",
        "today",
        "urlopen",
        "write_text",
    }
)

FORBIDDEN_NAMES: frozenset[str] = frozenset(
    {
        "DefaultFactorQualityChecker",
        "FactorContext",
        "FactorQualityChecker",
        "FactorRegistry",
        "FactorRunner",
        "FactorStore",
        "ForwardReturnCalculator",
        "ICAnalyzer",
        "MarketDataReader",
        "ParquetFactorStore",
        "ParquetMarketDataReader",
        "QuickAnalyzer",
    }
)

#: The project root, for the "the factor path knows nothing about quantiles" test.
PROJECT_ROOT: Path = Path(factor_sys.__file__).parent.parent


def module_tree() -> ast.Module:
    """The parsed source of :mod:`factor_sys.quantile_analysis`."""
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


def test_the_module_depends_only_on_the_allowed_layers() -> None:
    """The stdlib, pandas, the panel contract, the error and the label contract."""
    imports = module_imports(module_tree())

    assert imports <= ALLOWED_IMPORTS, sorted(imports - ALLOWED_IMPORTS)


def test_the_module_imports_nothing_from_the_factor_path_or_the_outside_world() -> None:
    """No runner, store, registry, IC module, factor formula, file or network call."""
    imports = module_imports(module_tree())

    for name in sorted(imports):
        for prefix in FORBIDDEN_IMPORT_PREFIXES:
            assert not matches(name, prefix), f"{name} must not be imported"


def test_the_module_does_not_import_the_package_root() -> None:
    """Importing ``factor_sys`` would drag the store and the runner in."""
    assert "factor_sys" not in module_imports(module_tree())


def test_the_module_calls_no_calculator_loader_or_writer() -> None:
    """The groups are measured from the panels it is given, never recomputed or stored."""
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


def test_the_module_mentions_no_object_of_the_factor_path() -> None:
    """No calculator, runner, context, store, registry, IC analyzer or reader is named."""
    tree = module_tree()
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}

    assert not names & FORBIDDEN_NAMES
    assert not attributes & FORBIDDEN_NAMES


def test_the_module_never_writes_a_column_back_into_a_panel() -> None:
    """Reading ``raw_value`` is the module's job; assigning into a frame is not."""
    tree = module_tree()
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assigned = {
        node.targets[0].attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and node.targets
        and isinstance(node.targets[0], ast.Attribute)
    }

    assert "to_parquet" not in attributes
    assert "to_csv" not in attributes
    assert "frame" not in assigned
    assert "raw_value" not in assigned


#: Files of the factor path this module must leave completely alone.
FACTOR_PATH_FILES: tuple[Path, ...] = (
    PROJECT_ROOT / "factor_sys" / "runner.py",
    PROJECT_ROOT / "factor_sys" / "quick_analysis.py",
    PROJECT_ROOT / "factor_sys" / "ic_analysis.py",
    PROJECT_ROOT / "factor_sys" / "registry.py",
    PROJECT_ROOT / "factor_sys" / "factor_store.py",
    *sorted((PROJECT_ROOT / "factors").glob("*.py")),
)


def test_the_factor_path_never_carries_a_quantile_report() -> None:
    """Producing factors and measuring ICs stays free of the quantile grouping."""
    assert FACTOR_PATH_FILES

    for path in FACTOR_PATH_FILES:
        text = path.read_text(encoding="utf-8")
        for token in (
            "quantile_analysis",
            "QuantileAnalyzer",
            "QuantileReport",
            "QuantileObservation",
            "QuantileSpreadObservation",
        ):
            assert token not in text, f"{path.name} must not mention {token}"


def test_the_label_module_does_not_depend_on_the_quantile_module() -> None:
    """Labels are an input to the grouping, so the dependency points one way only."""
    text = (PROJECT_ROOT / "factor_sys" / "forward_returns.py").read_text(encoding="utf-8")

    assert "quantile_analysis" not in text
    assert "QuantileAnalyzer" not in text


def test_the_ic_module_does_not_depend_on_the_quantile_module() -> None:
    """Two evaluations of the same labels; neither needs the other to exist."""
    text = (PROJECT_ROOT / "factor_sys" / "ic_analysis.py").read_text(encoding="utf-8")

    assert "quantile_analysis" not in text
    assert "QuantileAnalyzer" not in text


# ---------------------------------------------------------------------------
# The error type and the package surface
# ---------------------------------------------------------------------------
def test_the_quantile_error_collects_problems_like_the_rest_of_the_package() -> None:
    """A refusal carries the list of what is wrong, not just the first symptom."""
    error = QuantileAnalysisError("cannot group this", ["first problem", "second problem"])

    assert isinstance(error, FactorError)
    assert error.problems == ["first problem", "second problem"]
    assert "cannot group this" in str(error)
    assert "first problem" in str(error)
    assert "second problem" in str(error)
    assert QuantileAnalysisError("just a message").problems == []


def test_the_quantile_error_does_not_hide_later_problems() -> None:
    """The message renders every problem as its own line, ready for a log."""
    error = QuantileAnalysisError("nope", ["one", "two", "three"])
    lines = str(error).splitlines()

    assert lines[0] == "nope"
    assert [line.strip() for line in lines[1:]] == ["- one", "- two", "- three"]


def test_the_quantile_error_is_its_own_type() -> None:
    """A broken join is a quantile problem, not an IC problem."""
    assert not issubclass(QuantileAnalysisError, factor_sys.ICAnalysisError)
    assert not issubclass(factor_sys.ICAnalysisError, QuantileAnalysisError)


def test_the_package_root_exports_the_quantile_api() -> None:
    """A caller writes ``from factor_sys import QuantileAnalyzer``, not the module path."""
    assert factor_sys.QuantileAnalyzer is QuantileAnalyzer
    assert factor_sys.QuantileObservation is QuantileObservation
    assert factor_sys.QuantileSpreadObservation is QuantileSpreadObservation
    assert factor_sys.QuantileReport is QuantileReport
    assert factor_sys.QuantileAnalysisError is QuantileAnalysisError
    assert quantile_analysis_module.QuantileAnalyzer is QuantileAnalyzer
    assert issubclass(factor_sys.QuantileAnalysisError, factor_sys.FactorError)

    for name in (
        "QuantileAnalysisError",
        "QuantileAnalyzer",
        "QuantileObservation",
        "QuantileReport",
        "QuantileSpreadObservation",
    ):
        assert name in factor_sys.__all__
        assert getattr(factor_sys, name) is getattr(quantile_analysis_module, name)


def test_the_module_exports_the_documented_names() -> None:
    """``__all__`` is the module's public surface and holds nothing else."""
    assert set(quantile_analysis_module.__all__) == {
        "QUANTILE_OBSERVATION_COLUMNS",
        "QUANTILE_SPREAD_COLUMNS",
        "QuantileAnalysisError",
        "QuantileAnalyzer",
        "QuantileObservation",
        "QuantileReport",
        "QuantileSpreadObservation",
    }


def test_the_documented_example_still_runs() -> None:
    """The module's own doctests are executed here, so the example cannot rot."""
    results = doctest.testmod(quantile_analysis_module, verbose=False)

    assert results.failed == 0
    assert results.attempted > 0


def test_the_earlier_layers_are_still_exported() -> None:
    """Adding the quantile analysis must not disturb the layers built before it."""
    for name in (
        "FactorContext",
        "FactorPanel",
        "FactorRunner",
        "FactorStore",
        "ForwardReturnCalculator",
        "ForwardReturnPanel",
        "ICAnalyzer",
        "ICReport",
        "ParquetFactorStore",
        "QuickAnalyzer",
    ):
        assert name in factor_sys.__all__, name
        assert getattr(factor_sys, name) is not None


def test_the_ic_analysis_still_measures_the_same_fixture() -> None:
    """The new module reads the same panels without changing what the IC says."""
    panel, labels = make_session(list(range(10)), [0.01 * index for index in range(10)])

    ic = ICAnalyzer().analyze([panel], [labels])
    quantiles = QuantileAnalyzer().analyze([panel], [labels])

    assert ic.observations[0].ic == 1.0
    assert quantiles.spreads[0].top_minus_bottom == pytest.approx(0.08)
    assert ic.observations[0].paired_count == quantiles.observations[0].paired_count
