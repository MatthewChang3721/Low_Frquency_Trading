"""Tests for IC analysis (:mod:`factor_sys.ic_analysis`).

Everything here is in memory and offline: panels and labels are literal frames,
no reader is called, no file is written and no calendar is consulted.  The
correlation is checked against an independently written reference -- ranks
computed by hand plus :func:`statistics.correlation` -- so a passing test says
the number is right, not that two copies of the same formula agree.

Covered in this order:

* the frozen observation: identity fields, the coverage identity
  ``paired_count / universe_size``, every self-contradiction it must refuse, and
  the ``repr`` a log prints,
* the frozen report: mean, sample deviation (``ddof=1``), ICIR, the ``None``
  boundaries (0 valid ICs, 1 valid IC, a flat history), ascending dates, and the
  contradictions a hand-built report cannot get away with,
* the ``min_pairs`` threshold: integers of two and up only,
* the IC itself: ``+1`` for a perfect monotone factor, ``-1`` for the reverse,
  average ranks for ties, invariance to row order, and ``None`` -- never ``NaN``
  -- for a constant side or too few pairs,
* the pairing rule: ineligible rows, non-finite factor values, missing labels
  and non-finite labels are all excluded, with the counts and coverage that
  follow,
* many sessions: changing universes, input order, calendar holes kept as holes,
  and summary statistics that agree with the definition,
* every refused request: empty or non-iterable inputs, foreign objects, mixed
  names/versions/horizons, repeated dates, date sets that do not close, and
  universes that moved,
* purity: the panels and labels survive a call unchanged, nothing is written
  back and the same inputs always give the same report,
* the architectural boundary: the module's imports, the objects it may not
  name, the calculator it may not call, and the fact that the factor path knows
  nothing about ICs,
* the error type, the package exports, and the fact that the layers built before
  this one are still exported and still agree with their contracts.
"""

from __future__ import annotations

import ast
import doctest
import inspect
import math
import statistics
from collections.abc import Sequence
from dataclasses import FrozenInstanceError
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

import factor_sys
import factor_sys.ic_analysis as ic_analysis_module
from factor_sys import (
    FactorError,
    FactorPanel,
    ForwardReturnError,
    ForwardReturnPanel,
    ICAnalysisError,
    ICAnalyzer,
    ICObservation,
    ICReport,
)
from factor_sys.ic_analysis import IC_OBSERVATION_COLUMNS

NAN = float("nan")
INF = float("inf")

#: A four-symbol cross-section; four points leave room for a real tie pattern.
SYMBOLS: tuple[str, ...] = ("AAPL", "MSFT", "NVDA", "AMZN")

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

#: The module under test, read as source for the boundary tests near the end.
MODULE_PATH: Path = Path(ic_analysis_module.__file__)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def panel_frame(
    symbols: Sequence[str] = SYMBOLS,
    values: Sequence[float] | None = None,
    *,
    as_of: date = DAY_ONE,
    name: str = NAME,
    version: str = VERSION,
    eligible: Sequence[bool] | None = None,
) -> pd.DataFrame:
    """A factor frame as a formula would hand it over: one row per symbol."""
    universe = list(symbols)
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
    symbols: Sequence[str] = SYMBOLS,
    values: Sequence[float] | None = None,
    *,
    as_of: date = DAY_ONE,
    name: str = NAME,
    version: str = VERSION,
    eligible: Sequence[bool] | None = None,
) -> FactorPanel:
    """A valid factor panel whose pool, values, date and identity the test chooses."""
    universe = list(symbols)
    frame = panel_frame(
        universe, values, as_of=as_of, name=name, version=version, eligible=eligible
    )
    return FactorPanel(frame, universe=universe, as_of_date=as_of)


def label_frame(
    symbols: Sequence[str] = SYMBOLS,
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
    universe = list(symbols)
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
    symbols: Sequence[str] = SYMBOLS,
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
    universe = list(symbols)
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
    symbols: Sequence[str] = SYMBOLS,
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


def one_session(values: Sequence[float], returns: Sequence[float], **kwargs: object) -> ICReport:
    """Analyze exactly one session built from the two value lists."""
    panel, labels = make_session(values, returns, **kwargs)  # type: ignore[arg-type]
    return ICAnalyzer().analyze([panel], [labels])


def average_ranks(values: Sequence[float]) -> list[float]:
    """Ranks written out by hand: tied values share the mean of their positions.

    This is the reference the module is checked against, and it is deliberately
    written the slow, obvious way -- sort, look at neighbours, average -- so a
    bug in the implementation cannot hide behind the same shortcut.
    """
    ordered = sorted(values)
    ranks: list[float] = []
    for value in values:
        positions = [index for index, other in enumerate(ordered) if other == value]
        ranks.append((positions[0] + positions[-1]) / 2.0 + 1.0)
    return ranks


def reference_ic(values: Sequence[float], returns: Sequence[float]) -> float | None:
    """The Spearman rank IC from the standard library, for comparison."""
    if len(set(values)) == 1 or len(set(returns)) == 1:
        return None
    return statistics.correlation(average_ranks(values), average_ranks(returns))


def summarize(
    observations: Sequence[ICObservation],
) -> tuple[int, float | None, float | None, float | None]:
    """The documented summary, from :mod:`statistics` instead of the module."""
    values = [item.ic for item in observations if item.ic is not None]
    if not values:
        return 0, None, None, None
    mean = statistics.mean(values)
    if len(values) == 1:
        return 1, mean, None, None
    deviation = statistics.stdev(values)
    if deviation == 0.0:
        return len(values), mean, 0.0, None
    return len(values), mean, deviation, mean / deviation


def make_observation(**overrides: object) -> ICObservation:
    """A valid single-session observation; a refusal test bends one field."""
    fields: dict[str, object] = {
        "as_of_date": DAY_ONE,
        "end_date": END_ONE,
        "factor_name": NAME,
        "factor_version": VERSION,
        "horizon_sessions": 1,
        "universe_size": 4,
        "paired_count": 4,
        "pair_coverage": 1.0,
        "ic": 0.5,
    }
    fields.update(overrides)
    return ICObservation(**fields)  # type: ignore[arg-type]


def observation_on(day: date, ic: float | None, *, paired: int = 4) -> ICObservation:
    """A valid observation for ``day`` holding ``ic`` and a matching coverage."""
    return make_observation(
        as_of_date=day,
        end_date=day + timedelta(days=3),
        paired_count=paired,
        pair_coverage=paired / 4,
        ic=ic,
    )


def make_report(
    observations: Sequence[ICObservation] | None = None,
    **overrides: object,
) -> ICReport:
    """A valid report; the summary is computed from the observations actually used."""
    if observations is None:
        observations = (observation_on(DAY_ONE, 0.5),)
    items = tuple(overrides.pop("observations", observations))
    if all(isinstance(item, ICObservation) for item in items):
        count, mean, deviation, icir = summarize(items)
    else:
        # a deliberately broken sequence: let the report say what is wrong with it
        count, mean, deviation, icir = 0, None, None, None
    fields: dict[str, object] = {
        "factor_name": NAME,
        "factor_version": VERSION,
        "horizon_sessions": 1,
        "observations": items,
        "valid_ic_count": count,
        "mean_ic": mean,
        "ic_std": deviation,
        "icir": icir,
    }
    fields.update(overrides)
    return ICReport(**fields)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# The frozen observation
# ---------------------------------------------------------------------------
def test_a_valid_observation_exposes_one_session_and_its_coverage() -> None:
    """Every field names the one session, factor, horizon and sample it measured."""
    observation = observation_on(DAY_ONE, 1.0)

    assert observation.as_of_date == DAY_ONE
    assert observation.end_date == END_ONE
    assert observation.factor_name == NAME
    assert observation.factor_version == VERSION
    assert observation.horizon_sessions == 1
    assert observation.universe_size == 4
    assert observation.paired_count == 4
    assert observation.pair_coverage == 1.0
    assert observation.ic == 1.0


def test_an_observation_without_an_ic_is_perfectly_valid() -> None:
    """``ic=None`` is a result: the counts and the coverage still have to be right."""
    observation = observation_on(DAY_ONE, None, paired=3)

    assert observation.ic is None
    assert observation.paired_count == 3
    assert observation.pair_coverage == pytest.approx(0.75)


def test_the_observation_is_frozen() -> None:
    """An IC of a session that already happened cannot be edited afterwards."""
    observation = observation_on(DAY_ONE, 1.0)

    with pytest.raises(FrozenInstanceError):
        observation.ic = 0.0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"pair_coverage": 0.5}, "pair_coverage"),
        ({"paired_count": 5}, "cannot exceed universe_size"),
        ({"universe_size": 0, "paired_count": 0, "pair_coverage": 0.0}, "at least 1"),
        ({"ic": 1.5}, "must lie in"),
        ({"ic": -1.5}, "must lie in"),
        ({"ic": 0.5, "paired_count": 1, "pair_coverage": 0.25}, "at least 2 paired symbols"),
        ({"end_date": DAY_ONE}, "must be later than as_of_date"),
        ({"end_date": date(2024, 1, 4)}, "must be later than as_of_date"),
        ({"ic": NAN}, "never NaN"),
        ({"ic": INF}, "never NaN"),
        ({"pair_coverage": NAN}, "finite real number"),
        ({"as_of_date": datetime(2024, 1, 5, 16, 0)}, "session, not for an instant"),
        ({"as_of_date": "2024-01-05"}, "must be a datetime.date"),
        ({"factor_name": "Momentum"}, "snake_case"),
        ({"factor_version": "  "}, "non-empty version string"),
        ({"horizon_sessions": 0}, "positive integer"),
        ({"universe_size": True}, "non-negative integer"),
        ({"paired_count": -1}, "non-negative integer"),
    ],
)
def test_an_observation_that_contradicts_itself_is_refused(
    overrides: dict[str, object], message: str
) -> None:
    """A hand-built observation cannot claim something no sample could produce."""
    with pytest.raises(ICAnalysisError, match=message):
        make_observation(**overrides)


def test_the_observation_error_collects_every_problem() -> None:
    """One refusal names every broken field of the same kind, as the package does."""
    with pytest.raises(ICAnalysisError) as caught:
        make_observation(pair_coverage=0.5, ic=2.0, end_date=DAY_ONE)

    assert isinstance(caught.value, FactorError)
    assert len(caught.value.problems) == 3
    assert any("pair_coverage" in problem for problem in caught.value.problems)
    assert any("must be later" in problem for problem in caught.value.problems)
    assert any("must lie in" in problem for problem in caught.value.problems)


def test_the_observation_repr_summarises_one_session() -> None:
    """The ``repr`` a log prints says which session, which factor and which IC."""
    text = repr(observation_on(DAY_ONE, 1.0))

    assert text.startswith("ICObservation(")
    assert "'2024-01-05'" in text
    assert "'2024-01-08'" in text
    assert "momentum_12_1@1.0.0" in text
    assert "paired=4/4" in text
    assert "coverage=1.0" in text
    assert "ic=1.0" in text


# ---------------------------------------------------------------------------
# The frozen report
# ---------------------------------------------------------------------------
def test_a_report_exposes_its_summary_and_its_observations() -> None:
    """Two opposite ICs average to zero with a sample deviation of ``sqrt(2)``."""
    first = observation_on(DAY_ONE, 1.0)
    second = observation_on(DAY_TWO, -1.0)

    report = make_report([first, second])

    assert report.factor_name == NAME
    assert report.factor_version == VERSION
    assert report.horizon_sessions == 1
    assert report.observations == (first, second)
    assert report.valid_ic_count == 2
    assert report.mean_ic == 0.0
    assert report.ic_std == pytest.approx(math.sqrt(2.0))
    assert report.icir == 0.0


def test_the_deviation_is_the_sample_deviation() -> None:
    """``ddof=1``: dividing by ``n`` would give ``sqrt(2/3)``, not ``1``."""
    report = make_report(
        [
            observation_on(DAY_ONE, 1.0),
            observation_on(DAY_TWO, 0.0),
            observation_on(DAY_THREE, -1.0),
        ]
    )

    assert report.valid_ic_count == 3
    assert report.mean_ic == 0.0
    assert report.ic_std == pytest.approx(1.0)
    assert report.ic_std != pytest.approx(math.sqrt(2.0 / 3.0))
    assert report.icir == 0.0


def test_a_flat_ic_history_has_no_icir() -> None:
    """A zero denominator is reported as "undefined", never as an infinity."""
    report = make_report([observation_on(DAY_ONE, 1.0), observation_on(DAY_TWO, 1.0)])

    assert report.valid_ic_count == 2
    assert report.mean_ic == 1.0
    assert report.ic_std == 0.0
    assert report.icir is None


def test_a_single_valid_ic_has_no_deviation_and_no_icir() -> None:
    """A deviation needs two observations; an undefined session keeps its row."""
    report = make_report([observation_on(DAY_ONE, 1.0), observation_on(DAY_TWO, None, paired=1)])

    assert report.valid_ic_count == 1
    assert report.mean_ic == 1.0
    assert report.ic_std is None
    assert report.icir is None
    assert len(report.observations) == 2
    assert report.observations[1].ic is None


def test_a_report_without_a_defined_ic_summarizes_to_none() -> None:
    """No session could be measured: the summary says so instead of dividing by zero."""
    report = make_report([observation_on(DAY_ONE, None), observation_on(DAY_TWO, None)])

    assert report.valid_ic_count == 0
    assert report.mean_ic is None
    assert report.ic_std is None
    assert report.icir is None


def test_the_report_is_frozen() -> None:
    """A summary cannot be edited after the observations that produced it."""
    report = make_report()

    with pytest.raises(FrozenInstanceError):
        report.mean_ic = 0.1  # type: ignore[misc]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"valid_ic_count": 5}, "valid_ic_count"),
        ({"mean_ic": 0.25}, "mean_ic"),
        ({"ic_std": 0.0}, "ic_std"),
        ({"icir": 2.0}, "icir"),
        (
            {
                "observations": (observation_on(DAY_ONE, 1.0), observation_on(DAY_TWO, -1.0)),
                "icir": None,
            },
            "icir",
        ),
        ({"mean_ic": NAN}, "finite real number"),
        ({"ic_std": INF}, "finite real number"),
        ({"icir": NAN}, "finite real number"),
        ({"observations": ()}, "at least one observation"),
        ({"observations": "not a sequence"}, "not an ICObservation"),
        (
            {"observations": (observation_on(DAY_TWO, 0.5), observation_on(DAY_ONE, 0.5))},
            "ascending",
        ),
        (
            {"observations": (observation_on(DAY_ONE, 0.5), observation_on(DAY_ONE, 0.5))},
            "repeats as_of_date",
        ),
        ({"observations": (make_observation(factor_name="value_12_1"),)}, "one factor"),
        ({"observations": (make_observation(factor_version="2.0.0"),)}, "one factor"),
        ({"observations": (make_observation(horizon_sessions=5),)}, "one horizon"),
        ({"factor_name": "Momentum"}, "snake_case"),
        ({"horizon_sessions": 0}, "positive integer"),
        ({"valid_ic_count": True}, "non-negative integer"),
    ],
)
def test_a_report_that_contradicts_its_observations_is_refused(
    overrides: dict[str, object], message: str
) -> None:
    """The summary is derived: a hand-built report cannot restate it differently."""
    with pytest.raises(ICAnalysisError, match=message):
        make_report(**overrides)


def test_the_report_error_collects_every_problem() -> None:
    """A refusal names the identity and the horizon of one bad observation at once."""
    with pytest.raises(ICAnalysisError) as caught:
        make_report([make_observation(factor_name="value_12_1", horizon_sessions=5)])

    assert isinstance(caught.value, FactorError)
    assert len(caught.value.problems) == 2
    assert any("one factor" in problem for problem in caught.value.problems)
    assert any("one horizon" in problem for problem in caught.value.problems)


def test_the_report_repr_summarises_the_history() -> None:
    """The ``repr`` a log prints says how many sessions and what the IC averaged."""
    text = repr(make_report([observation_on(DAY_ONE, 1.0), observation_on(DAY_TWO, 0.0)]))

    assert text.startswith("ICReport(")
    assert "momentum_12_1@1.0.0" in text
    assert "observations=2" in text
    assert "valid_ic_count=2" in text
    assert "mean_ic=0.5" in text
    assert "ic_std=" in text
    assert "icir=" in text


def test_the_observation_columns_are_the_documented_order() -> None:
    """Downstream joins read this frame, so the column order is part of the API."""
    assert IC_OBSERVATION_COLUMNS == (
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


def test_to_frame_lays_the_observations_out_for_a_join() -> None:
    """One row per session, the fixed columns in order, ``NaN`` for an undefined IC."""
    report = make_report([observation_on(DAY_ONE, 1.0), observation_on(DAY_TWO, None, paired=1)])

    frame = report.to_frame()

    assert list(frame.columns) == list(IC_OBSERVATION_COLUMNS)
    assert len(frame) == len(report.observations)
    assert frame["as_of_date"].dt.date.tolist() == [DAY_ONE, DAY_TWO]
    assert frame["end_date"].dt.date.tolist() == [END_ONE, DAY_TWO + timedelta(days=3)]
    assert frame["factor_name"].tolist() == [NAME, NAME]
    assert frame["factor_version"].tolist() == [VERSION, VERSION]
    assert frame["horizon_sessions"].tolist() == [1, 1]
    assert frame["universe_size"].tolist() == [4, 4]
    assert frame["paired_count"].tolist() == [4, 1]
    assert frame["pair_coverage"].tolist() == [1.0, 0.25]
    ics = frame["ic"].tolist()
    assert ics[0] == 1.0
    assert math.isnan(ics[1])


def test_to_frame_keeps_the_dtypes_a_join_expects() -> None:
    """Sessions stay sessions and counts stay integers, even for one session."""
    frame = make_report().to_frame()

    assert str(frame["as_of_date"].dtype) == "datetime64[s]"
    assert str(frame["end_date"].dtype) == "datetime64[s]"
    assert str(frame["horizon_sessions"].dtype) == "int64"
    assert str(frame["universe_size"].dtype) == "int64"
    assert str(frame["paired_count"].dtype) == "int64"
    assert str(frame["pair_coverage"].dtype) == "float64"
    assert str(frame["ic"].dtype) == "float64"
    assert str(frame["factor_name"].dtype) == "object"


def test_to_frame_returns_a_copy() -> None:
    """Editing the frame cannot reach back into the frozen observations."""
    report = make_report([observation_on(DAY_ONE, 1.0)])

    frame = report.to_frame()
    frame.loc[0, "ic"] = -1.0

    assert report.observations[0].ic == 1.0
    assert report.to_frame() is not frame


# ---------------------------------------------------------------------------
# The analyzer's only knob
# ---------------------------------------------------------------------------
def test_min_pairs_defaults_to_two_and_is_reported_in_the_repr() -> None:
    """One analyzer, one number: the fewest pairs an IC may be measured on."""
    assert ICAnalyzer().min_pairs == 2
    assert ICAnalyzer(min_pairs=7).min_pairs == 7
    assert repr(ICAnalyzer()) == "ICAnalyzer(min_pairs=2)"
    assert repr(ICAnalyzer(min_pairs=5)) == "ICAnalyzer(min_pairs=5)"


@pytest.mark.parametrize("value", [True, False, 0, 1, -3, 2.0, 2.5, "2", None, [2], (2,)])
def test_min_pairs_refuses_anything_a_correlation_cannot_use(value: object) -> None:
    """A threshold below two, or one that is not a count, would ask for nothing."""
    with pytest.raises(ICAnalysisError, match="min_pairs"):
        ICAnalyzer(min_pairs=value)  # type: ignore[arg-type]


def test_the_analyzer_holds_no_other_state() -> None:
    """No reader, no store, no clock: the constructor takes the threshold only."""
    assert list(inspect.signature(ICAnalyzer).parameters) == ["min_pairs"]
    analyzer = ICAnalyzer()
    assert list(analyzer.__slots__) == ["_min_pairs"]
    assert not hasattr(analyzer, "__dict__")


# ---------------------------------------------------------------------------
# A perfect rank correlation is exactly +1 or -1
# ---------------------------------------------------------------------------
def test_a_monotone_factor_has_ic_plus_one() -> None:
    """Strictly increasing values against strictly increasing returns."""
    report = one_session([0.1, 0.2, 0.4, 0.8], [0.01, 0.02, 0.03, 0.04])

    observation = report.observations[0]
    assert observation.ic == 1.0
    assert observation.paired_count == 4
    assert observation.pair_coverage == 1.0
    assert report.mean_ic == 1.0
    assert report.valid_ic_count == 1


def test_an_inverse_factor_has_ic_minus_one() -> None:
    """Strictly increasing values against strictly decreasing returns."""
    report = one_session([0.1, 0.2, 0.4, 0.8], [0.04, 0.03, 0.02, 0.01])

    assert report.observations[0].ic == -1.0
    assert report.mean_ic == -1.0


def test_a_rank_ic_ignores_scale_offset_and_curvature() -> None:
    """Only the order matters: any strictly increasing transform is the same factor."""
    returns = [0.01, 0.02, 0.03, 0.04]

    plain = one_session([0.1, 0.2, 0.4, 0.8], returns).observations[0]
    transformed = one_session([-500.0, 3.0, 4.0, 1e6], returns).observations[0]
    skewed = one_session([1.0, 1.1, 4.0, 4.000001], returns).observations[0]

    assert plain.ic == 1.0
    assert transformed.ic == 1.0
    assert skewed.ic == 1.0


def test_two_sessions_can_be_correlated_in_opposite_directions() -> None:
    """The report keeps one observation per session instead of aggregating rows."""
    first_panel, first_labels = make_session(
        [1.0, 2.0, 3.0, 4.0], [0.01, 0.02, 0.03, 0.04], as_of=DAY_ONE, end=END_ONE
    )
    second_panel, second_labels = make_session(
        [1.0, 2.0, 3.0, 4.0], [0.04, 0.03, 0.02, 0.01], as_of=DAY_TWO, end=END_TWO
    )

    report = ICAnalyzer().analyze([first_panel, second_panel], [first_labels, second_labels])

    assert [item.ic for item in report.observations] == [1.0, -1.0]
    assert report.mean_ic == 0.0
    assert report.ic_std == pytest.approx(math.sqrt(2.0))
    assert report.icir == 0.0


def test_a_modest_rank_ic_is_the_one_a_hand_calculation_gives() -> None:
    """``[1, 2, 3, 4]`` against ``[3, 1, 4, 2]``: the ranks cancel exactly."""
    report = one_session([1.0, 2.0, 3.0, 4.0], [0.03, 0.01, 0.04, 0.02])
    values = [1.0, 2.0, 3.0, 4.0]
    returns = [0.03, 0.01, 0.04, 0.02]

    assert report.observations[0].ic == pytest.approx(0.0, abs=1e-12)
    assert report.observations[0].ic == pytest.approx(reference_ic(values, returns), abs=1e-12)
    assert report.valid_ic_count == 1
    assert report.mean_ic == pytest.approx(0.0, abs=1e-12)
    assert report.ic_std is None


# ---------------------------------------------------------------------------
# Ties are averaged, never broken by row position
# ---------------------------------------------------------------------------
def test_a_tie_in_the_factor_uses_the_average_rank() -> None:
    """Two equal values share ranks 1.5 and 2.5; breaking the tie would give ``+1``."""
    values = [1.0, 1.0, 2.0, 3.0]
    returns = [0.01, 0.02, 0.03, 0.04]

    ic = one_session(values, returns).observations[0].ic

    assert ic == pytest.approx(reference_ic(values, returns), abs=1e-12)
    assert ic == pytest.approx(0.9486832980505138, abs=1e-12)
    assert ic is not None and ic < 1.0
    # what a "first occurrence wins" rank would have claimed instead
    assert statistics.correlation([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]) == 1.0


def test_a_tie_in_the_returns_uses_the_average_rank() -> None:
    """The same rule applies to the right-hand side of the correlation."""
    values = [0.1, 0.2, 0.3, 0.4]
    returns = [0.02, 0.01, 0.01, 0.03]

    ic = one_session(values, returns).observations[0].ic

    assert ic == pytest.approx(reference_ic(values, returns), abs=1e-12)
    assert ic == pytest.approx(0.31622776601683794, abs=1e-12)


def test_every_value_tied_leaves_no_ic() -> None:
    """A tie across the whole cross-section is the constant case, not a correlation."""
    report = one_session([5.0, 5.0, 5.0, 5.0], [0.01, 0.01, 0.01, 0.01])

    assert report.observations[0].ic is None
    assert report.observations[0].paired_count == 4


def test_the_hand_written_ranks_agree_with_pandas() -> None:
    """The reference the tests compare against is itself checked against pandas."""
    values = [3.0, 1.0, 3.0, 2.0, 1.0]

    assert average_ranks(values) == pd.Series(values).rank(method="average").tolist()


def test_the_ic_does_not_depend_on_the_row_order() -> None:
    """Re-ordering both frames cannot change a rank correlation built from averages."""
    panel, labels = make_session([2.0, 1.0, 1.0, 3.0], [0.03, 0.01, 0.02, 0.01])
    first = ICAnalyzer().analyze([panel], [labels]).observations[0]

    for order in ([3, 2, 1, 0], [1, 2, 0, 3], [2, 0, 3, 1]):
        shuffled, shuffled_labels = permuted(panel, labels, order)
        other = ICAnalyzer().analyze([shuffled], [shuffled_labels]).observations[0]
        assert other.ic == pytest.approx(first.ic, abs=1e-12)
        assert other.paired_count == first.paired_count
        assert other.universe_size == first.universe_size


def test_a_column_permutation_cannot_change_the_report() -> None:
    """The whole report is a function of the rows, not of their order."""
    panel, labels = make_session([2.0, 1.0, 1.0, 3.0], [0.03, 0.01, 0.02, 0.01])
    report = ICAnalyzer().analyze([panel], [labels])

    shuffled, shuffled_labels = permuted(panel, labels, [2, 3, 0, 1])

    assert ICAnalyzer().analyze([shuffled], [shuffled_labels]) == report


# ---------------------------------------------------------------------------
# When there is no IC: None, never NaN and never NaN-shaped zeros
# ---------------------------------------------------------------------------
def test_a_constant_factor_value_has_no_ic() -> None:
    """A zero-variance side has no rank correlation; the row still reports its counts."""
    report = one_session([2.0, 2.0, 2.0, 2.0], [0.01, 0.02, 0.03, 0.04])

    observation = report.observations[0]
    assert observation.ic is None
    assert observation.paired_count == 4
    assert observation.pair_coverage == 1.0
    assert report.valid_ic_count == 0
    assert report.mean_ic is None
    assert report.ic_std is None
    assert report.icir is None


def test_a_constant_forward_return_has_no_ic() -> None:
    """A flat return column is just as undefined as a flat factor column."""
    report = one_session([1.0, 2.0, 3.0, 4.0], [0.02, 0.02, 0.02, 0.02])

    assert report.observations[0].ic is None
    assert report.observations[0].paired_count == 4


def test_fewer_pairs_than_min_pairs_has_no_ic() -> None:
    """The threshold decides whether to measure; it never hides the counts."""
    panel, labels = make_session(
        [1.0, 2.0, 3.0, 4.0],
        [0.01, 0.02, 0.03, 0.04],
        eligible=[True, True, False, False],
    )

    strict = ICAnalyzer(min_pairs=3).analyze([panel], [labels])
    observation = strict.observations[0]
    assert observation.paired_count == 2
    assert observation.pair_coverage == 0.5
    assert observation.ic is None
    assert strict.valid_ic_count == 0

    lenient = ICAnalyzer(min_pairs=2).analyze([panel], [labels])
    assert lenient.observations[0].paired_count == 2
    assert lenient.observations[0].ic == 1.0


def test_the_threshold_never_changes_the_counts() -> None:
    """Only the IC appears or disappears; the sample description stays the same."""
    panel, labels = make_session(
        [1.0, 2.0, 3.0, 4.0],
        [0.01, 0.02, 0.03, 0.04],
        eligible=[True, True, False, True],
    )

    counts = {
        (item.paired_count, item.pair_coverage)
        for item in (
            ICAnalyzer(min_pairs=2).analyze([panel], [labels]).observations[0],
            ICAnalyzer(min_pairs=3).analyze([panel], [labels]).observations[0],
            ICAnalyzer(min_pairs=4).analyze([panel], [labels]).observations[0],
        )
    }

    assert counts == {(3, 0.75)}


# ---------------------------------------------------------------------------
# Which symbol is paired: eligibility, a finite value and a known label
# ---------------------------------------------------------------------------
def test_an_ineligible_row_is_excluded_from_the_correlation() -> None:
    """A symbol the factor itself ruled out contributes nothing, not a zero."""
    report = one_session(
        [1.0, 2.0, 3.0, 4.0],
        [0.01, 0.02, 0.03, 0.04],
        eligible=[True, False, True, True],
    )

    observation = report.observations[0]
    assert observation.universe_size == 4
    assert observation.paired_count == 3
    assert observation.pair_coverage == pytest.approx(0.75)
    assert observation.ic == 1.0


@pytest.mark.parametrize("value", [NAN, INF, -INF])
def test_a_non_finite_factor_value_is_excluded(value: float) -> None:
    """A ``NaN`` or an infinity has no rank; the row stays out of the sample."""
    report = one_session([1.0, value, 3.0, 4.0], [0.01, 0.02, 0.03, 0.04])

    observation = report.observations[0]
    assert observation.paired_count == 3
    assert observation.pair_coverage == pytest.approx(0.75)
    assert observation.ic == 1.0


def test_a_label_that_is_not_available_is_excluded() -> None:
    """The horizon has not been traded out yet: the symbol is not a pair."""
    report = one_session(
        [1.0, 2.0, 3.0, 4.0],
        [0.01, 0.02, 0.03, 0.04],
        available=[True, False, True, True],
    )

    observation = report.observations[0]
    assert observation.paired_count == 3
    assert observation.pair_coverage == pytest.approx(0.75)
    assert observation.ic == 1.0


def test_a_label_is_never_nan_without_saying_so() -> None:
    """The contract refuses both halves of the half-truth, so an IC never sees one."""
    with pytest.raises(ForwardReturnError, match="is_available=False"):
        make_labels(SYMBOLS, [0.01, 0.02, 0.03, 0.04], available=[True, False, True, True])

    with pytest.raises(ForwardReturnError, match="is_available=True"):
        make_labels(SYMBOLS, [0.01, NAN, 0.03, 0.04], available=[True, True, True, True])


def test_both_reasons_at_once_leave_two_pairs() -> None:
    """Coverage counts the whole cross-section, not just the usable rows."""
    report = one_session(
        [1.0, 2.0, 3.0, 4.0],
        [0.01, 0.02, 0.03, 0.04],
        eligible=[True, False, True, True],
        available=[True, True, False, True],
    )

    observation = report.observations[0]
    assert observation.universe_size == 4
    assert observation.paired_count == 2
    assert observation.pair_coverage == 0.5
    assert observation.ic == 1.0


def test_an_eligible_but_undefined_value_is_still_excluded() -> None:
    """Eligibility is necessary, not sufficient: the value must exist as well."""
    report = one_session(
        [1.0, NAN, 3.0, 4.0],
        [0.01, 0.02, 0.03, 0.04],
        eligible=[True, True, True, True],
    )

    assert report.observations[0].paired_count == 3
    assert report.observations[0].ic == 1.0


def test_a_session_with_no_usable_pair_still_reports_itself() -> None:
    """Nothing paired is a fact about the session, so the row survives with zero coverage."""
    report = one_session(
        [1.0, 2.0, 3.0, 4.0],
        [0.01, 0.02, 0.03, 0.04],
        eligible=[False, False, False, False],
    )

    observation = report.observations[0]
    assert observation.universe_size == 4
    assert observation.paired_count == 0
    assert observation.pair_coverage == 0.0
    assert observation.ic is None
    assert report.valid_ic_count == 0
    assert report.mean_ic is None


def test_a_two_symbol_universe_can_still_be_measured() -> None:
    """Two pairs are the smallest sample a correlation is defined on."""
    report = one_session(
        [1.0, 2.0],
        [0.01, 0.02],
        symbols=("AAPL", "MSFT"),
    )

    observation = report.observations[0]
    assert observation.universe_size == 2
    assert observation.paired_count == 2
    assert observation.pair_coverage == 1.0
    assert observation.ic == 1.0


# ---------------------------------------------------------------------------
# A history of sessions
# ---------------------------------------------------------------------------
def test_a_different_universe_on_each_session_is_allowed() -> None:
    """A point-in-time pool changes; each session is measured on its own symbols."""
    first_panel, first_labels = make_session(
        [1.0, 2.0, 3.0, 4.0], [0.01, 0.02, 0.03, 0.04], as_of=DAY_ONE, end=END_ONE
    )
    second_panel, second_labels = make_session(
        [0.1, 0.2, 0.3],
        [0.03, 0.02, 0.01],
        symbols=("AAPL", "MSFT", "NVDA"),
        as_of=DAY_TWO,
        end=END_TWO,
    )

    report = ICAnalyzer().analyze([first_panel, second_panel], [first_labels, second_labels])

    assert [item.universe_size for item in report.observations] == [4, 3]
    assert [item.paired_count for item in report.observations] == [4, 3]
    assert [item.ic for item in report.observations] == [1.0, -1.0]


def test_observations_come_back_in_ascending_date_order() -> None:
    """However the caller orders the input, the report reads like a time series."""
    sessions = [
        make_session([1.0, 2.0, 3.0, 4.0], [0.01, 0.02, 0.03, 0.04], as_of=day, end=end)
        for day, end in ((DAY_ONE, END_ONE), (DAY_TWO, END_TWO), (DAY_THREE, END_THREE))
    ]
    panels = [panel for panel, _ in sessions]
    labels = [label for _, label in sessions]

    forward = ICAnalyzer().analyze(panels, labels)
    backward = ICAnalyzer().analyze(list(reversed(panels)), list(reversed(labels)))
    generated = ICAnalyzer().analyze(iter(panels), iter(labels))

    expected = [DAY_ONE, DAY_TWO, DAY_THREE]
    assert [item.as_of_date for item in forward.observations] == expected
    assert backward == forward
    assert generated == forward


def test_a_calendar_hole_stays_a_hole() -> None:
    """Two sessions in January are two rows: no missing date is invented or filled."""
    first_panel, first_labels = make_session(
        [1.0, 2.0, 3.0, 4.0], [0.01, 0.02, 0.03, 0.04], as_of=DAY_ONE, end=END_ONE
    )
    third_panel, third_labels = make_session(
        [4.0, 3.0, 2.0, 1.0], [0.01, 0.02, 0.03, 0.04], as_of=DAY_THREE, end=END_THREE
    )

    report = ICAnalyzer().analyze([first_panel, third_panel], [first_labels, third_labels])

    assert [item.as_of_date for item in report.observations] == [DAY_ONE, DAY_THREE]
    assert [item.ic for item in report.observations] == [1.0, -1.0]
    assert len(report.to_frame()) == 2
    assert DAY_TWO not in set(report.to_frame()["as_of_date"].dt.date)


def test_the_summary_is_the_mean_and_the_sample_deviation_of_the_ic_series() -> None:
    """``+1``, ``0`` and ``-1``: a mean of zero, a deviation of one, an ICIR of zero."""
    sessions = [
        make_session([1.0, 2.0, 3.0, 4.0], [0.01, 0.02, 0.03, 0.04], as_of=DAY_ONE, end=END_ONE),
        make_session([1.0, 2.0, 3.0, 4.0], [0.03, 0.01, 0.04, 0.02], as_of=DAY_TWO, end=END_TWO),
        make_session(
            [1.0, 2.0, 3.0, 4.0], [0.04, 0.03, 0.02, 0.01], as_of=DAY_THREE, end=END_THREE
        ),
    ]

    report = ICAnalyzer().analyze(
        [panel for panel, _ in sessions], [label for _, label in sessions]
    )

    assert [item.ic for item in report.observations] == pytest.approx([1.0, 0.0, -1.0])
    assert report.valid_ic_count == 3
    assert report.mean_ic == pytest.approx(0.0, abs=1e-12)
    assert report.ic_std == pytest.approx(1.0)
    assert report.icir == pytest.approx(0.0, abs=1e-12)


def test_a_persistently_positive_ic_has_no_icir_because_it_never_varies() -> None:
    """Three perfect but differently-sized samples still average to ``1`` with no spread."""
    sessions = [
        make_session([1.0, 2.0, 3.0, 4.0], [0.01, 0.02, 0.03, 0.04], as_of=DAY_ONE, end=END_ONE),
        make_session(
            [1.0, 2.0], [0.01, 0.02], symbols=("AAPL", "MSFT"), as_of=DAY_TWO, end=END_TWO
        ),
        make_session(
            [1.0, 2.0, 3.0, 4.0], [0.001, 0.05, 0.06, 0.2], as_of=DAY_THREE, end=END_THREE
        ),
    ]

    report = ICAnalyzer().analyze(
        [panel for panel, _ in sessions], [label for _, label in sessions]
    )

    assert [item.ic for item in report.observations] == [1.0, 1.0, 1.0]
    assert report.mean_ic == 1.0
    assert report.ic_std == 0.0
    assert report.icir is None


def test_an_undefined_session_does_not_enter_the_summary() -> None:
    """It stays a row with its coverage, but it is not an IC of zero."""
    defined = make_session(
        [1.0, 2.0, 3.0, 4.0], [0.01, 0.02, 0.03, 0.04], as_of=DAY_ONE, end=END_ONE
    )
    flat = make_session([2.0, 2.0, 2.0, 2.0], [0.01, 0.02, 0.03, 0.04], as_of=DAY_TWO, end=END_TWO)
    inverse = make_session(
        [1.0, 2.0, 3.0, 4.0], [0.04, 0.03, 0.02, 0.01], as_of=DAY_THREE, end=END_THREE
    )

    report = ICAnalyzer().analyze(
        [defined[0], flat[0], inverse[0]], [defined[1], flat[1], inverse[1]]
    )

    assert [item.ic for item in report.observations] == [1.0, None, -1.0]
    assert report.observations[1].paired_count == 4
    assert report.observations[1].pair_coverage == 1.0
    assert report.valid_ic_count == 2
    assert report.mean_ic == 0.0
    assert report.ic_std == pytest.approx(math.sqrt(2.0))


def test_a_label_horizon_travels_with_the_observation() -> None:
    """A five-session label measures a five-session forward return."""
    panel, labels = make_session(
        [1.0, 2.0, 3.0, 4.0],
        [0.01, 0.02, 0.03, 0.04],
        end=date(2024, 1, 12),
        horizon=5,
    )

    report = ICAnalyzer().analyze([panel], [labels])

    assert report.horizon_sessions == 5
    assert report.observations[0].horizon_sessions == 5
    assert report.observations[0].end_date == date(2024, 1, 12)


def test_the_same_inputs_always_give_the_same_report() -> None:
    """No clock, no randomness: two calls produce equal, independent reports."""
    panel, labels = make_session([2.0, 1.0, 1.0, 3.0], [0.03, 0.01, 0.02, 0.01])

    first = ICAnalyzer().analyze([panel], [labels])
    second = ICAnalyzer().analyze([panel], [labels])

    assert first == second
    assert first is not second
    assert first.to_frame().equals(second.to_frame())


def test_reports_of_different_factors_are_not_equal() -> None:
    """The identity is part of the value, so two factors never compare equal."""
    panel, labels = make_session([1.0, 2.0, 3.0, 4.0], [0.01, 0.02, 0.03, 0.04])
    other_panel, other_labels = make_session(
        [1.0, 2.0, 3.0, 4.0], [0.01, 0.02, 0.03, 0.04], name="value_12_1"
    )

    first = ICAnalyzer().analyze([panel], [labels])
    second = ICAnalyzer().analyze([other_panel], [other_labels])

    assert first != second
    assert (first.factor_name, second.factor_name) == (NAME, "value_12_1")


# ---------------------------------------------------------------------------
# Every request that cannot be answered is refused, never guessed at
# ---------------------------------------------------------------------------
def test_an_empty_history_is_refused() -> None:
    """An IC is measured per session, so no session is nothing to summarize."""
    panel, labels = make_session([1.0, 2.0], [0.01, 0.02], symbols=("AAPL", "MSFT"))

    with pytest.raises(ICAnalysisError, match="at least one FactorPanel"):
        ICAnalyzer().analyze([], [labels])
    with pytest.raises(ICAnalysisError, match="at least one ForwardReturnPanel"):
        ICAnalyzer().analyze([panel], [])
    with pytest.raises(ICAnalysisError, match="at least one FactorPanel"):
        ICAnalyzer().analyze([], [])


def test_a_single_panel_is_not_a_history() -> None:
    """A bare panel is a common mistake, so the refusal says how to wrap it."""
    panel, labels = make_session([1.0, 2.0], [0.01, 0.02], symbols=("AAPL", "MSFT"))

    with pytest.raises(ICAnalysisError, match="got a single FactorPanel"):
        ICAnalyzer().analyze(panel, [labels])  # type: ignore[arg-type]
    with pytest.raises(ICAnalysisError, match="got a single ForwardReturnPanel"):
        ICAnalyzer().analyze([panel], labels)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", ["AAPL", b"AAPL", None, 3, 3.5, object()])
def test_something_that_is_not_a_sequence_of_panels_is_refused(value: object) -> None:
    """A string is iterable, which is exactly why it has to be refused by name."""
    panel, labels = make_session([1.0, 2.0], [0.01, 0.02], symbols=("AAPL", "MSFT"))

    with pytest.raises(ICAnalysisError, match="needs an iterable"):
        ICAnalyzer().analyze(value, [labels])  # type: ignore[arg-type]
    with pytest.raises(ICAnalysisError, match="needs an iterable"):
        ICAnalyzer().analyze([panel], value)  # type: ignore[arg-type]


def test_a_foreign_object_inside_the_sequence_is_refused() -> None:
    """A frame, a dict or a panel of the other kind is a type error, not a skip."""
    panel, labels = make_session([1.0, 2.0], [0.01, 0.02], symbols=("AAPL", "MSFT"))

    with pytest.raises(ICAnalysisError, match="can only pair FactorPanel objects"):
        ICAnalyzer().analyze([panel, labels], [labels, labels])  # type: ignore[list-item]
    with pytest.raises(ICAnalysisError, match="can only pair ForwardReturnPanel objects"):
        ICAnalyzer().analyze([panel, panel], [labels, panel.frame])  # type: ignore[list-item]
    with pytest.raises(ICAnalysisError, match="factor_panels\\[0\\] is a DataFrame"):
        ICAnalyzer().analyze([panel.frame], [labels])  # type: ignore[list-item]


def test_a_dataframe_is_not_a_sequence_of_panels() -> None:
    """A frame iterates its column names, which the type check then refuses."""
    panel, labels = make_session([1.0, 2.0], [0.01, 0.02], symbols=("AAPL", "MSFT"))

    with pytest.raises(ICAnalysisError, match="can only pair FactorPanel objects"):
        ICAnalyzer().analyze(panel.frame, [labels])  # type: ignore[arg-type]


def test_a_mixed_factor_name_or_version_is_refused() -> None:
    """One call measures one factor at one version: two never average together."""
    first = make_session([1.0, 2.0], [0.01, 0.02], symbols=("AAPL", "MSFT"), as_of=DAY_ONE)
    second = make_session(
        [1.0, 2.0],
        [0.01, 0.02],
        symbols=("AAPL", "MSFT"),
        as_of=DAY_TWO,
        end=END_TWO,
        name="value_12_1",
    )
    third = make_session(
        [1.0, 2.0],
        [0.01, 0.02],
        symbols=("AAPL", "MSFT"),
        as_of=DAY_TWO,
        end=END_TWO,
        version="2.0.0",
    )

    with pytest.raises(ICAnalysisError, match="one factor at one version"):
        ICAnalyzer().analyze([first[0], second[0]], [first[1], second[1]])
    with pytest.raises(ICAnalysisError, match="one factor at one version"):
        ICAnalyzer().analyze([first[0], third[0]], [first[1], third[1]])


def test_a_mixed_horizon_is_refused() -> None:
    """ICs measured over one session and over five are not the same series."""
    first = make_session([1.0, 2.0], [0.01, 0.02], symbols=("AAPL", "MSFT"), as_of=DAY_ONE)
    second = make_session(
        [1.0, 2.0],
        [0.01, 0.02],
        symbols=("AAPL", "MSFT"),
        as_of=DAY_TWO,
        end=END_TWO,
        horizon=5,
    )

    with pytest.raises(ICAnalysisError, match="one factor at one horizon"):
        ICAnalyzer().analyze([first[0], second[0]], [first[1], second[1]])


def test_a_repeated_session_is_refused() -> None:
    """The same date twice would silently weight one session twice."""
    panel, labels = make_session([1.0, 2.0], [0.01, 0.02], symbols=("AAPL", "MSFT"))

    with pytest.raises(ICAnalysisError, match="repeats as_of_date"):
        ICAnalyzer().analyze([panel, panel], [labels])
    with pytest.raises(ICAnalysisError, match="repeats as_of_date"):
        ICAnalyzer().analyze([panel], [labels, labels])


def test_a_session_without_labels_is_refused_instead_of_dropped() -> None:
    """Analysing the intersection would hide exactly the broken join to catch."""
    first = make_session([1.0, 2.0], [0.01, 0.02], symbols=("AAPL", "MSFT"), as_of=DAY_ONE)
    second = make_session(
        [1.0, 2.0], [0.01, 0.02], symbols=("AAPL", "MSFT"), as_of=DAY_TWO, end=END_TWO
    )

    with pytest.raises(ICAnalysisError, match="has no labels for as_of_date") as caught:
        ICAnalyzer().analyze([first[0], second[0]], [first[1]])

    assert "2024-01-08" in str(caught.value)


def test_labels_without_a_panel_are_refused() -> None:
    """A label with no factor value measures nothing, so it cannot just be ignored."""
    first = make_session([1.0, 2.0], [0.01, 0.02], symbols=("AAPL", "MSFT"), as_of=DAY_ONE)
    second = make_session(
        [1.0, 2.0], [0.01, 0.02], symbols=("AAPL", "MSFT"), as_of=DAY_TWO, end=END_TWO
    )

    with pytest.raises(ICAnalysisError, match="with no factor panel") as caught:
        ICAnalyzer().analyze([first[0]], [first[1], second[1]])

    assert "2024-01-08" in str(caught.value)


def test_a_panel_and_its_labels_must_share_an_identity() -> None:
    """The labels were built for a factor; labels of another one are a foreign object."""
    panel, _ = make_session([1.0, 2.0], [0.01, 0.02], symbols=("AAPL", "MSFT"))
    wrong_name = make_labels(("AAPL", "MSFT"), [0.01, 0.02], name="value_12_1")
    wrong_version = make_labels(("AAPL", "MSFT"), [0.01, 0.02], version="2.0.0")

    with pytest.raises(ICAnalysisError, match="disagree on identity"):
        ICAnalyzer().analyze([panel], [wrong_name])
    with pytest.raises(ICAnalysisError, match="disagree on identity"):
        ICAnalyzer().analyze([panel], [wrong_version])


@pytest.mark.parametrize(
    ("symbols", "returns", "message"),
    [
        (("AAPL", "MSFT", "NVDA"), [0.01, 0.02, 0.03], r"4 symbol\(s\) in the panel against 3"),
        (
            ("AAPL", "MSFT", "NVDA", "GOOGL"),
            [0.01, 0.02, 0.03, 0.04],
            "only in the panel: AMZN",
        ),
        (
            ("AMZN", "MSFT", "NVDA", "AAPL"),
            [0.01, 0.02, 0.03, 0.04],
            "the same symbols in a different order",
        ),
    ],
)
def test_a_universe_that_moved_between_panel_and_labels_is_refused(
    symbols: Sequence[str], returns: Sequence[float], message: str
) -> None:
    """Re-ordering or intersecting symbols here would pair values with the wrong stock."""
    panel, _ = make_session([1.0, 2.0, 3.0, 4.0], [0.01, 0.02, 0.03, 0.04])
    labels = make_labels(symbols, returns)

    with pytest.raises(ICAnalysisError, match=message):
        ICAnalyzer().analyze([panel], [labels])


def test_a_reordered_universe_names_the_first_position_that_differs() -> None:
    """The refusal says where the two universes stopped agreeing."""
    panel, _ = make_session([1.0, 2.0, 3.0, 4.0], [0.01, 0.02, 0.03, 0.04])
    labels = make_labels(("AAPL", "MSFT", "AMZN", "NVDA"), [0.01, 0.02, 0.03, 0.04])

    with pytest.raises(ICAnalysisError) as caught:
        ICAnalyzer().analyze([panel], [labels])

    assert "position 2" in str(caught.value)
    assert "NVDA in the panel, AMZN in the labels" in str(caught.value)


def test_every_join_problem_is_collected_in_one_error() -> None:
    """One call reports the missing date and the moved universe together."""
    first = make_session([1.0, 2.0], [0.01, 0.02], symbols=("AAPL", "MSFT"), as_of=DAY_ONE)
    second = make_session(
        [1.0, 2.0], [0.01, 0.02], symbols=("AAPL", "MSFT"), as_of=DAY_TWO, end=END_TWO
    )
    moved = make_labels(("AAPL", "MSFT"), [0.01, 0.02], as_of=DAY_THREE, end=END_THREE)

    with pytest.raises(ICAnalysisError) as caught:
        ICAnalyzer().analyze([first[0], second[0]], [first[1], moved])

    problems = caught.value.problems
    assert isinstance(caught.value, FactorError)
    assert any("has no labels for as_of_date" in problem for problem in problems)
    assert any("with no factor panel" in problem for problem in problems)


def test_the_refusal_names_the_operation_that_failed() -> None:
    """The message says which call refused, so a log needs no stack trace."""
    panel, labels = make_session([1.0, 2.0], [0.01, 0.02], symbols=("AAPL", "MSFT"))

    with pytest.raises(ICAnalysisError, match="ICAnalyzer.analyze\\(\\)"):
        ICAnalyzer().analyze([], [labels])


# ---------------------------------------------------------------------------
# Purity: the analysis reads its inputs and writes nothing back
# ---------------------------------------------------------------------------
def test_the_inputs_survive_a_call_unchanged() -> None:
    """No rank column, no score column: the frames come back byte for byte."""
    panel, labels = make_session([2.0, 1.0, 1.0, 3.0], [0.03, 0.01, 0.02, 0.01])
    panel_before = panel.frame.copy()
    labels_before = labels.frame.copy()
    identity_before = (panel.factor_name, panel.factor_version, panel.as_of_date, panel.universe)

    ICAnalyzer().analyze([panel], [labels])

    pd.testing.assert_frame_equal(panel.frame, panel_before)
    pd.testing.assert_frame_equal(labels.frame, labels_before)
    assert (panel.factor_name, panel.factor_version, panel.as_of_date, panel.universe) == (
        identity_before
    )
    assert not {"rank", "ic", "score", "forward_return", "paired_count"} & set(panel.frame.columns)


def test_the_caller_s_lists_are_left_alone() -> None:
    """The analyzer copies what it needs; it never pops, sorts or reverses the input."""
    first = make_session([1.0, 2.0], [0.01, 0.02], symbols=("AAPL", "MSFT"), as_of=DAY_ONE)
    second = make_session(
        [1.0, 2.0], [0.01, 0.02], symbols=("AAPL", "MSFT"), as_of=DAY_TWO, end=END_TWO
    )
    panels = [second[0], first[0]]
    labels = [second[1], first[1]]
    panels_before = list(panels)
    labels_before = list(labels)

    ICAnalyzer().analyze(panels, labels)

    assert panels == panels_before
    assert labels == labels_before


def test_a_generator_is_consumed_once_and_still_works() -> None:
    """A caller may hand over a generator; nothing re-reads it behind their back."""
    panel, labels = make_session([1.0, 2.0], [0.01, 0.02], symbols=("AAPL", "MSFT"))

    report = ICAnalyzer().analyze((item for item in [panel]), (item for item in [labels]))

    assert report.observations[0].ic == 1.0


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
        "MarketDataReader",
        "ParquetFactorStore",
        "ParquetMarketDataReader",
        "QuickAnalyzer",
    }
)


#: The project root, for the "the factor path knows nothing about ICs" test.
PROJECT_ROOT: Path = Path(factor_sys.__file__).parent.parent


def module_tree() -> ast.Module:
    """The parsed source of :mod:`factor_sys.ic_analysis`."""
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
    """No runner, store, registry, factor formula, file or network call."""
    imports = module_imports(module_tree())

    for name in sorted(imports):
        for prefix in FORBIDDEN_IMPORT_PREFIXES:
            assert not matches(name, prefix), f"{name} must not be imported"


def test_the_module_does_not_import_the_package_root() -> None:
    """Importing ``factor_sys`` would drag the store and the runner in."""
    assert "factor_sys" not in module_imports(module_tree())


def test_the_module_calls_no_calculator_loader_or_writer() -> None:
    """The IC is measured from the panels it is given, never recomputed or stored."""
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
    """No calculator, runner, context, store, registry or market reader is named."""
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
    assert "rank" not in assigned
    assert "ic" not in assigned
    assert "frame" not in assigned


#: Files of the factor path this module must leave completely alone.
FACTOR_PATH_FILES: tuple[Path, ...] = (
    PROJECT_ROOT / "factor_sys" / "runner.py",
    PROJECT_ROOT / "factor_sys" / "quick_analysis.py",
    *sorted((PROJECT_ROOT / "factors").glob("*.py")),
)


def test_the_factor_path_never_carries_an_ic() -> None:
    """Producing factors stays free of the analysis of those factors."""
    assert FACTOR_PATH_FILES

    for path in FACTOR_PATH_FILES:
        text = path.read_text(encoding="utf-8")
        assert "ic_analysis" not in text, path.name
        assert "ICAnalyzer" not in text, path.name
        assert "ICReport" not in text, path.name
        assert "ICObservation" not in text, path.name


def test_the_label_module_does_not_depend_on_the_ic_module() -> None:
    """Labels are an input to the IC, so the dependency only ever points one way."""
    text = (PROJECT_ROOT / "factor_sys" / "forward_returns.py").read_text(encoding="utf-8")

    assert "ic_analysis" not in text
    assert "ICAnalyzer" not in text


# ---------------------------------------------------------------------------
# The error type and the package surface
# ---------------------------------------------------------------------------
def test_the_ic_error_collects_problems_like_the_rest_of_the_package() -> None:
    """A refusal carries the list of what is wrong, not just the first symptom."""
    error = ICAnalysisError("cannot measure this", ["first problem", "second problem"])

    assert isinstance(error, FactorError)
    assert error.problems == ["first problem", "second problem"]
    assert "cannot measure this" in str(error)
    assert "first problem" in str(error)
    assert "second problem" in str(error)
    assert ICAnalysisError("just a message").problems == []


def test_the_ic_error_does_not_hide_later_problems() -> None:
    """The message renders every problem as its own line, ready for a log."""
    error = ICAnalysisError("nope", ["one", "two", "three"])
    lines = str(error).splitlines()

    assert lines[0] == "nope"
    assert [line.strip() for line in lines[1:]] == ["- one", "- two", "- three"]


def test_the_package_root_exports_the_ic_api() -> None:
    """A caller writes ``from factor_sys import ICAnalyzer``, not the module path."""
    assert factor_sys.ICAnalyzer is ICAnalyzer
    assert factor_sys.ICObservation is ICObservation
    assert factor_sys.ICReport is ICReport
    assert factor_sys.ICAnalysisError is ICAnalysisError
    assert ic_analysis_module.ICAnalyzer is ICAnalyzer
    assert issubclass(factor_sys.ICAnalysisError, factor_sys.FactorError)

    for name in ("ICAnalysisError", "ICAnalyzer", "ICObservation", "ICReport"):
        assert name in factor_sys.__all__
        assert getattr(factor_sys, name) is getattr(ic_analysis_module, name)


def test_the_module_exports_the_documented_names() -> None:
    """``__all__`` is the module's public surface and holds nothing else."""
    assert set(ic_analysis_module.__all__) == {
        "IC_OBSERVATION_COLUMNS",
        "ICAnalysisError",
        "ICAnalyzer",
        "ICObservation",
        "ICReport",
    }


def test_the_public_objects_are_documented() -> None:
    """Every exported class carries the docstring a reader needs, not a placeholder."""
    for name in ("ICAnalysisError", "ICAnalyzer", "ICObservation", "ICReport"):
        doc = getattr(ic_analysis_module, name).__doc__

        assert doc, name
        assert len(doc.strip().splitlines()) > 3, name
        assert not doc.strip().startswith(("TODO", "FIXME", "...")), name


def test_the_documented_example_still_runs() -> None:
    """The module's own doctests are executed here, so the example cannot rot."""
    results = doctest.testmod(ic_analysis_module, verbose=False)

    assert results.failed == 0
    assert results.attempted > 0


def test_the_earlier_layers_are_still_exported() -> None:
    """Adding the IC analysis must not disturb the layers built before it."""
    for name in (
        "FactorContext",
        "FactorPanel",
        "FactorRunner",
        "FactorStore",
        "ForwardReturnCalculator",
        "ForwardReturnPanel",
        "ParquetFactorStore",
        "QuickAnalyzer",
    ):
        assert name in factor_sys.__all__, name
        assert getattr(factor_sys, name) is not None


def test_the_forward_return_contract_is_unchanged() -> None:
    """The labels this module consumes keep their documented columns and order."""
    frame = make_labels(SYMBOLS, [0.01, 0.02, 0.03, 0.04]).frame

    assert list(frame.columns) == list(factor_sys.FORWARD_RETURN_COLUMNS)
    assert list(IC_OBSERVATION_COLUMNS)[:4] == [
        "as_of_date",
        "end_date",
        "factor_name",
        "factor_version",
    ]
