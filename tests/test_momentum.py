"""Tests for the first real factor: 12-1 momentum (:mod:`factors.momentum`).

Everything here is in memory and offline: bars are ordinary pandas frames, a
:class:`~factor_sys.contracts.FactorContext` is built directly for the unit tests,
and the integration test wires the *real* :class:`~factor_sys.runner.FactorRunner`
to an in-memory provider and an in-memory reader.  No ``data/`` directory is read,
no Parquet file is written, no vendor is called and no clock is consulted.

The unit-test session sequences are synthetic weekdays on purpose: a factor must
not know what a trading calendar is, so the tests must not need one either.  The
integration test uses the real XNYS sessions, because that is where the runner's
window comes from.

Covered in this order:

* identity: the factor is a :class:`Factor`, its spec is exact, immutable and
  shared, and the package exports it,
* the formula: ``adjusted_close(t-21) / adjusted_close(t-252) - 1`` on exactly
  the two anchor rows, with the most recent 21 sessions skipped and the decision
  day's own price ignored,
* ``adjusted_close`` only -- ``close`` can differ and changes nothing,
* eligibility: 253 observations required, a missing session or a missing
  decision-day bar making a symbol ineligible, unusable anchors (missing, text,
  infinite, zero, negative), a mixed universe and a completely empty frame,
* the panel contract: columns, identity, universe order, raw (untransformed)
  values,
* purity: the context and its bars are untouched, shuffled input rows do not
  change the answer, and repeated runs are identical,
* the contract error for a reader that forgot ``required_fields``,
* the runner integration: the reader is asked for ``("adjusted_close",)``, the
  window really is 253 XNYS sessions ending on the decision date, and the panel
  that comes back is the one the factor computed,
* the architectural boundary of the module and its exports.
"""

from __future__ import annotations

import ast
import inspect
import math
from collections.abc import Mapping, Sequence
from dataclasses import FrozenInstanceError
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import factors
import factors.momentum as momentum_module
from data_sys.market_calendar import get_calendar, sessions_in_range
from factor_sys import (
    FACTOR_PANEL_COLUMNS,
    Factor,
    FactorContext,
    FactorContractError,
    FactorError,
    FactorPanel,
    FactorRunner,
    MarketDataReader,
    UniverseProvider,
    UniverseSnapshot,
)
from factors import Momentum12_1
from factors.momentum import (
    LOOKBACK_SESSIONS,
    PRICE_FIELD,
    REQUIRED_OBSERVATIONS,
    SKIP_SESSIONS,
)

#: A real XNYS session (a half day, but a session): the decision date used here.
AS_OF: date = date(2024, 12, 31)

#: The real calendar.  Only the runner-integration tests use it.
CALENDAR = get_calendar("XNYS")


# ---------------------------------------------------------------------------
# Builders: sessions, bars, contexts
# ---------------------------------------------------------------------------
def sessions(count: int, *, end: date = AS_OF) -> list[date]:
    """``count`` consecutive weekday "sessions" ending on ``end``.

    Synthetic on purpose: the factor is position-based and must never consult an
    exchange calendar, so neither do its unit tests.
    """
    return [stamp.date() for stamp in pd.bdate_range(end=pd.Timestamp(end), periods=count)]


def momentum_prices(
    *,
    start: float = 100.0,
    end: float = 130.0,
    fill: float = 100.0,
    count: int = REQUIRED_OBSERVATIONS,
) -> list[float]:
    """A full 253-observation series whose two anchors are ``start`` and ``end``."""
    prices = [fill] * count
    prices[-count] = start
    prices[-SKIP_SESSIONS - 1] = end
    return prices


def bars_for(
    symbol: str,
    prices: Sequence[Any],
    dates: Sequence[date],
    *,
    field: str = PRICE_FIELD,
    dtype: str | None = None,
    **extra: Sequence[Any],
) -> pd.DataFrame:
    """One symbol's long bars: ``prices[i]`` is ``field`` on ``dates[i]``.

    Extra keyword columns (``close=...``) are appended in the order given.  Pass
    ``dtype="object"`` to keep values exactly as supplied (a ``bool`` or a text
    price would otherwise be converted by pandas' inference).
    """
    data: dict[str, Any] = {
        "date": pd.Series(list(dates), dtype="datetime64[s]"),
        "symbol": [symbol] * len(prices),
        field: pd.Series(list(prices), dtype=dtype),
    }
    data.update({name: pd.Series(list(values), dtype=dtype) for name, values in extra.items()})
    return pd.DataFrame(data)


def empty_bars(*, field: str = PRICE_FIELD) -> pd.DataFrame:
    """A correctly shaped bars frame with no rows at all."""
    return pd.DataFrame(
        {
            "date": pd.Series(dtype="datetime64[s]"),
            "symbol": pd.Series(dtype="object"),
            field: pd.Series(dtype="float64"),
        }
    )


def make_context(
    *parts: pd.DataFrame,
    universe: Sequence[str] = ("AAPL",),
    as_of: date = AS_OF,
) -> FactorContext:
    """A context over ``parts`` concatenated; no parts means an empty frame."""
    bars = pd.concat(parts, ignore_index=True) if parts else empty_bars()
    return FactorContext(as_of_date=as_of, universe=list(universe), bars=bars)


def values(panel: FactorPanel) -> dict[str, float]:
    """The panel's ``raw_value`` per symbol."""
    frame = panel.frame
    return dict(zip(frame["symbol"], frame["raw_value"], strict=True))


def eligibility(panel: FactorPanel) -> dict[str, bool]:
    """The panel's ``is_eligible`` per symbol."""
    frame = panel.frame
    pairs = zip(frame["symbol"], frame["is_eligible"], strict=True)
    return {symbol: bool(flag) for symbol, flag in pairs}


# ---------------------------------------------------------------------------
# Identity: a Factor with one exact, immutable spec
# ---------------------------------------------------------------------------
def test_momentum_is_a_concrete_factor() -> None:
    """It is a :class:`Factor` and it can be instantiated (no abstract parts left)."""
    factor = Momentum12_1()

    assert isinstance(factor, Factor)
    assert not inspect.isabstract(type(factor))


def test_the_spec_is_the_documented_one() -> None:
    """Name, version, field and lookback are exactly what the task defines."""
    spec = Momentum12_1().spec

    assert spec.name == "momentum_12_1"
    assert spec.version == "1.0.0"
    assert spec.required_fields == ("adjusted_close",)
    assert spec.lookback_sessions == 252


def test_the_spec_is_shared_and_frozen() -> None:
    """One instance per process, and nobody can edit it afterwards."""
    spec = Momentum12_1().spec

    assert spec is momentum_module.SPEC
    assert spec is Momentum12_1().spec
    assert isinstance(spec.required_fields, tuple)
    with pytest.raises(FrozenInstanceError):
        spec.version = "2.0.0"  # type: ignore[misc]


def test_the_declared_constants_agree_with_the_formula() -> None:
    """The skip, the lookback and the required row count cannot drift apart."""
    assert PRICE_FIELD == "adjusted_close"
    assert LOOKBACK_SESSIONS == 252
    assert SKIP_SESSIONS == 21
    assert REQUIRED_OBSERVATIONS == 253
    assert LOOKBACK_SESSIONS == REQUIRED_OBSERVATIONS - 1
    assert Momentum12_1().spec.lookback_sessions == LOOKBACK_SESSIONS


def test_the_factor_is_exported_by_the_package() -> None:
    """``from factors import Momentum12_1`` is the one public door."""
    assert factors.Momentum12_1 is Momentum12_1
    assert factors.__all__ == ["Momentum12_1"]
    assert momentum_module.__all__ == ["Momentum12_1"]


# ---------------------------------------------------------------------------
# The formula
# ---------------------------------------------------------------------------
def test_the_basic_formula_is_the_one_declared() -> None:
    """``t-252`` = 100 and ``t-21`` = 130 give a raw return of 0.30."""
    dates = sessions(REQUIRED_OBSERVATIONS)
    context = make_context(bars_for("AAPL", momentum_prices(start=100.0, end=130.0), dates))

    panel = Momentum12_1().compute(context)

    assert values(panel) == {"AAPL": pytest.approx(0.30)}
    assert eligibility(panel) == {"AAPL": True}


def test_a_negative_return_stays_negative() -> None:
    """The sign is the price ratio's sign, not an absolute value."""
    dates = sessions(REQUIRED_OBSERVATIONS)
    context = make_context(bars_for("AAPL", momentum_prices(start=200.0, end=150.0), dates))

    assert values(Momentum12_1().compute(context)) == {"AAPL": pytest.approx(-0.25)}


def test_a_flat_history_has_exactly_zero_momentum() -> None:
    """Equal anchors are a 0% return, computed as ``value / value - 1``."""
    dates = sessions(REQUIRED_OBSERVATIONS)
    context = make_context(bars_for("AAPL", [50.0] * REQUIRED_OBSERVATIONS, dates))

    assert values(Momentum12_1().compute(context)) == {"AAPL": 0.0}


def test_only_the_two_anchor_rows_change_the_value() -> None:
    """Every row except ``t-252`` and ``t-21`` is invisible to the formula."""
    dates = sessions(REQUIRED_OBSERVATIONS)
    base = momentum_prices(start=100.0, end=130.0)
    distorted = [price * 3.5 + 7.0 for price in base]
    distorted[0] = base[0]
    distorted[-SKIP_SESSIONS - 1] = base[-SKIP_SESSIONS - 1]

    expected = Momentum12_1().compute(make_context(bars_for("AAPL", base, dates)))
    actual = Momentum12_1().compute(make_context(bars_for("AAPL", distorted, dates)))

    assert values(actual) == values(expected) == {"AAPL": pytest.approx(0.30)}


def test_the_decision_day_price_never_enters_the_return() -> None:
    """``t`` is only the information horizon; its own close is not used."""
    dates = sessions(REQUIRED_OBSERVATIONS)
    base = momentum_prices()
    changed = list(base)
    changed[-1] = 5_000.0

    expected = Momentum12_1().compute(make_context(bars_for("AAPL", base, dates)))
    actual = Momentum12_1().compute(make_context(bars_for("AAPL", changed, dates)))

    assert values(actual) == values(expected)


def test_the_start_anchor_is_the_first_row_of_the_window() -> None:
    """``t-252`` is the 253rd row from the end -- one year of sessions, not 250."""
    dates = sessions(REQUIRED_OBSERVATIONS)
    prices = [50.0] * REQUIRED_OBSERVATIONS
    prices[0] = 80.0
    prices[-SKIP_SESSIONS - 1] = 100.0

    panel = Momentum12_1().compute(make_context(bars_for("AAPL", prices, dates)))

    assert values(panel) == {"AAPL": pytest.approx(0.25)}


# ---------------------------------------------------------------------------
# The skipped month
# ---------------------------------------------------------------------------
def test_the_most_recent_twenty_one_sessions_are_skipped() -> None:
    """Rewriting ``t-20`` .. ``t`` cannot move a 12-1 value."""
    dates = sessions(REQUIRED_OBSERVATIONS)
    base = momentum_prices(start=100.0, end=130.0)
    noisy = list(base)
    for offset in range(1, SKIP_SESSIONS + 1):
        noisy[-offset] = 10_000.0 * offset

    expected = Momentum12_1().compute(make_context(bars_for("AAPL", base, dates)))
    actual = Momentum12_1().compute(make_context(bars_for("AAPL", noisy, dates)))

    assert values(actual) == values(expected) == {"AAPL": pytest.approx(0.30)}


def test_the_twenty_second_row_from_the_end_is_the_end_anchor() -> None:
    """``t-21`` *does* matter: it is the last price of the measured year."""
    dates = sessions(REQUIRED_OBSERVATIONS)
    prices = momentum_prices(start=100.0, end=130.0)
    prices[-SKIP_SESSIONS - 1] = 65.0

    panel = Momentum12_1().compute(make_context(bars_for("AAPL", prices, dates)))

    assert values(panel) == {"AAPL": pytest.approx(-0.35)}


def test_the_decision_day_cannot_stand_in_for_the_skipped_month() -> None:
    """A price at ``t`` that is close to nothing never becomes the end anchor."""
    dates = sessions(REQUIRED_OBSERVATIONS)
    prices = momentum_prices(start=100.0, end=100.0)
    prices[-1] = 5_000.0

    panel = Momentum12_1().compute(make_context(bars_for("AAPL", prices, dates)))

    assert values(panel) == {"AAPL": 0.0}


# ---------------------------------------------------------------------------
# adjusted_close only
# ---------------------------------------------------------------------------
def test_close_cannot_substitute_for_adjusted_close() -> None:
    """``close`` shows a 200% year; the factor still reports the adjusted 30%."""
    dates = sessions(REQUIRED_OBSERVATIONS)
    adjusted = momentum_prices(start=100.0, end=130.0)
    close = momentum_prices(start=100.0, end=300.0)
    bars = bars_for("AAPL", adjusted, dates, close=close)

    panel = Momentum12_1().compute(make_context(bars))

    assert values(panel) == {"AAPL": pytest.approx(0.30)}


def test_changing_close_changes_nothing() -> None:
    """Two contexts differing only in ``close`` produce the same panel."""
    dates = sessions(REQUIRED_OBSERVATIONS)
    adjusted = momentum_prices(start=100.0, end=130.0)
    cheap = bars_for("AAPL", adjusted, dates, close=[1.0] * REQUIRED_OBSERVATIONS)
    dear = bars_for("AAPL", adjusted, dates, close=[9_999.0] * REQUIRED_OBSERVATIONS)

    first = Momentum12_1().compute(make_context(cheap))
    second = Momentum12_1().compute(make_context(dear))

    pd.testing.assert_frame_equal(first.frame, second.frame)


def test_the_spec_asks_only_for_adjusted_close() -> None:
    """Nothing else can be required, so a reader is never asked for ``close``."""
    spec = Momentum12_1().spec

    assert spec.required_fields == ("adjusted_close",)
    assert "close" not in spec.required_fields


# ---------------------------------------------------------------------------
# Eligibility: enough data, the right anchors, a bar on the decision day
# ---------------------------------------------------------------------------
def test_a_complete_window_of_253_observations_is_eligible() -> None:
    """252 past sessions plus the decision day is exactly enough."""
    dates = sessions(REQUIRED_OBSERVATIONS)
    context = make_context(bars_for("AAPL", momentum_prices(), dates))

    panel = Momentum12_1().compute(context)

    assert len(panel) == 1
    assert eligibility(panel) == {"AAPL": True}
    assert math.isfinite(values(panel)["AAPL"])


@pytest.mark.parametrize("count", [1, 2, 22, 100, 251, 252])
def test_a_short_history_is_ineligible(count: int) -> None:
    """One row short is already too short: no interpolation, no nearest price."""
    dates = sessions(count)
    prices = [100.0 + position for position in range(count)]
    context = make_context(bars_for("AAPL", prices, dates))

    panel = Momentum12_1().compute(context)

    assert eligibility(panel) == {"AAPL": False}
    assert math.isnan(values(panel)["AAPL"])


def test_a_missing_session_inside_the_window_is_ineligible() -> None:
    """A hole leaves 252 rows, so the anchors are no longer a year/month apart."""
    all_dates = sessions(REQUIRED_OBSERVATIONS)
    all_prices = momentum_prices(start=100.0, end=130.0)
    kept = [
        (day, price)
        for position, (day, price) in enumerate(zip(all_dates, all_prices, strict=True))
        if position != 100
    ]
    dates = [day for day, _ in kept]
    prices = [price for _, price in kept]
    assert len(prices) == REQUIRED_OBSERVATIONS - 1
    assert dates[-1] == AS_OF

    panel = Momentum12_1().compute(make_context(bars_for("AAPL", prices, dates)))

    assert eligibility(panel) == {"AAPL": False}
    assert math.isnan(values(panel)["AAPL"])


def test_a_symbol_without_a_decision_day_bar_is_ineligible() -> None:
    """A history that stops the session before ``t`` is stale, not computable."""
    dates = sessions(REQUIRED_OBSERVATIONS, end=date(2024, 12, 30))
    assert dates[-1] < AS_OF
    context = make_context(bars_for("AAPL", momentum_prices(), dates))

    panel = Momentum12_1().compute(context)

    assert eligibility(panel) == {"AAPL": False}
    assert math.isnan(values(panel)["AAPL"])


def test_a_symbol_with_extra_history_still_uses_the_last_253_rows() -> None:
    """More rows than the runner's window is harmless: the anchors are from the end."""
    dates = sessions(REQUIRED_OBSERVATIONS + 40)
    prices = [1.0] * 40 + momentum_prices(start=100.0, end=130.0)
    context = make_context(bars_for("AAPL", prices, dates))

    panel = Momentum12_1().compute(context)

    assert eligibility(panel) == {"AAPL": True}
    assert values(panel) == {"AAPL": pytest.approx(0.30)}


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, 0.0, -5.0, "100.0", None, True])
@pytest.mark.parametrize("position", [-REQUIRED_OBSERVATIONS, -SKIP_SESSIONS - 1])
def test_an_unusable_anchor_makes_the_symbol_ineligible(position: int, bad: Any) -> None:
    """Missing, text, infinite, zero and negative prices are not returns."""
    dates = sessions(REQUIRED_OBSERVATIONS)
    prices = momentum_prices(start=100.0, end=130.0)
    prices[position] = bad
    bars = bars_for("AAPL", prices, dates, dtype="object")

    panel = Momentum12_1().compute(make_context(bars))

    assert eligibility(panel) == {"AAPL": False}
    assert math.isnan(values(panel)["AAPL"])


def test_an_integer_price_is_accepted() -> None:
    """A whole-number price is a number like any other."""
    dates = sessions(REQUIRED_OBSERVATIONS)
    prices = [100] * REQUIRED_OBSERVATIONS
    prices[-SKIP_SESSIONS - 1] = 130
    context = make_context(bars_for("AAPL", prices, dates))

    panel = Momentum12_1().compute(context)

    assert eligibility(panel) == {"AAPL": True}
    assert values(panel) == {"AAPL": pytest.approx(0.30)}


def test_a_mixed_universe_still_covers_every_symbol() -> None:
    """Complete, short and absent histories all get exactly one row."""
    dates = sessions(REQUIRED_OBSERVATIONS)
    complete = bars_for("AAPL", momentum_prices(start=100.0, end=130.0), dates)
    short = bars_for("MSFT", [100.0 + position for position in range(10)], dates[-10:])
    context = make_context(complete, short, universe=("AAPL", "MSFT", "NVDA"))

    panel = Momentum12_1().compute(context)

    assert len(panel) == 3
    assert panel.symbols == ("AAPL", "MSFT", "NVDA")
    assert eligibility(panel) == {"AAPL": True, "MSFT": False, "NVDA": False}
    assert panel.frame["is_eligible"].tolist() == [True, False, False]
    assert values(panel)["AAPL"] == pytest.approx(0.30)
    assert math.isnan(values(panel)["MSFT"])
    assert math.isnan(values(panel)["NVDA"])


def test_an_empty_frame_yields_a_complete_panel_of_ineligible_rows() -> None:
    """No rows at all is a valid context, not an error and not an empty panel."""
    context = make_context(universe=("AAPL", "MSFT", "NVDA"))

    panel = Momentum12_1().compute(context)

    frame = panel.frame
    assert len(panel) == 3
    assert panel.symbols == ("AAPL", "MSFT", "NVDA")
    assert frame["raw_value"].isna().all()
    assert frame["raw_value"].dtype == "float64"
    assert frame["is_eligible"].dtype == bool
    assert not frame["is_eligible"].any()


def test_a_frame_without_adjusted_close_is_a_contract_error() -> None:
    """A reader that ignored ``required_fields`` must fail loudly, not fill NaN."""
    dates = sessions(REQUIRED_OBSERVATIONS)
    bars = bars_for("AAPL", momentum_prices(), dates, field="close")
    context = make_context(bars)

    with pytest.raises(FactorContractError) as excinfo:
        Momentum12_1().compute(context)

    message = str(excinfo.value)
    assert "adjusted_close" in message
    assert "'close'" in message
    assert "required_fields" in message
    assert isinstance(excinfo.value, FactorError)


def test_an_empty_frame_without_adjusted_close_is_still_a_contract_error() -> None:
    """Even with no rows the missing column is the caller's mistake."""
    context = make_context(empty_bars(field="close"), universe=("AAPL",))

    with pytest.raises(FactorContractError):
        Momentum12_1().compute(context)


# ---------------------------------------------------------------------------
# The panel contract
# ---------------------------------------------------------------------------
def test_the_output_is_a_validated_factor_panel() -> None:
    """Columns, labels and dtypes are exactly what the contract demands."""
    dates = sessions(REQUIRED_OBSERVATIONS)
    context = make_context(
        bars_for("AAPL", momentum_prices(start=100.0, end=130.0), dates),
        bars_for("MSFT", momentum_prices(start=50.0, end=45.0), dates),
        universe=("AAPL", "MSFT"),
    )

    panel = Momentum12_1().compute(context)

    assert isinstance(panel, FactorPanel)
    frame = panel.frame
    assert list(frame.columns) == list(FACTOR_PANEL_COLUMNS)
    assert frame["as_of_date"].dt.date.tolist() == [AS_OF, AS_OF]
    assert frame["factor_name"].tolist() == ["momentum_12_1", "momentum_12_1"]
    assert frame["factor_version"].tolist() == ["1.0.0", "1.0.0"]
    assert frame["raw_value"].dtype == "float64"
    assert frame["is_eligible"].dtype == bool
    assert panel.as_of_date == AS_OF
    assert panel.universe == ("AAPL", "MSFT")
    assert panel.factor_name == "momentum_12_1"
    assert panel.factor_version == "1.0.0"


def test_the_rows_follow_the_universe_order() -> None:
    """Row order is the requested pool's order: no sorting, no ranking."""
    dates = sessions(REQUIRED_OBSERVATIONS)
    prices = {
        "NVDA": momentum_prices(start=100.0, end=300.0),
        "AAPL": momentum_prices(start=100.0, end=110.0),
        "MSFT": momentum_prices(start=100.0, end=90.0),
    }
    parts = [bars_for(symbol, series, dates) for symbol, series in prices.items()]
    universe = ("NVDA", "AAPL", "MSFT")

    panel = Momentum12_1().compute(make_context(*parts, universe=universe))

    assert panel.symbols == universe
    assert panel.frame["symbol"].tolist() == list(universe)


def test_the_values_are_raw_returns_not_scores() -> None:
    """No winsorizing, no ranking, no z-scoring: 300% and -90% stay as they are."""
    dates = sessions(REQUIRED_OBSERVATIONS)
    context = make_context(
        bars_for("AAPL", momentum_prices(start=10.0, end=40.0), dates),
        bars_for("MSFT", momentum_prices(start=100.0, end=10.0), dates),
        universe=("AAPL", "MSFT"),
    )

    panel = Momentum12_1().compute(context)

    assert values(panel) == {"AAPL": pytest.approx(3.0), "MSFT": pytest.approx(-0.9)}
    assert eligibility(panel) == {"AAPL": True, "MSFT": True}


def test_the_panel_carries_the_contexts_decision_date() -> None:
    """The output is labelled with the context's session, not with "today"."""
    other = date(2024, 6, 28)
    dates = sessions(REQUIRED_OBSERVATIONS, end=other)
    context = make_context(bars_for("AAPL", momentum_prices(), dates), as_of=other)

    panel = Momentum12_1().compute(context)

    assert panel.as_of_date == other
    assert panel.frame["as_of_date"].dt.date.tolist() == [other]


# ---------------------------------------------------------------------------
# Purity: nothing is read, nothing is written, nothing is mutated
# ---------------------------------------------------------------------------
def test_the_context_and_its_bars_are_not_mutated() -> None:
    """``compute`` reads the context; it never edits it or its frame."""
    dates = sessions(REQUIRED_OBSERVATIONS)
    context = make_context(
        bars_for("AAPL", momentum_prices(), dates),
        bars_for("MSFT", momentum_prices(start=50.0, end=60.0), dates),
        universe=("AAPL", "MSFT"),
    )
    before = context.bars.copy(deep=True)
    index_before = context.bars.index

    Momentum12_1().compute(context)

    assert context.as_of_date == AS_OF
    assert context.universe == ("AAPL", "MSFT")
    pd.testing.assert_frame_equal(context.bars, before)
    pd.testing.assert_index_equal(context.bars.index, index_before)


def test_the_same_context_can_be_computed_twice() -> None:
    """The factor keeps no state: a second run repeats the first exactly."""
    dates = sessions(REQUIRED_OBSERVATIONS)
    context = make_context(bars_for("AAPL", momentum_prices(), dates))
    factor = Momentum12_1()

    first = factor.compute(context)
    second = factor.compute(context)

    pd.testing.assert_frame_equal(first.frame, second.frame)
    assert values(second) == values(first)


def test_two_identical_contexts_give_identical_panels() -> None:
    """Determinism across instances and across frame objects, not just calls."""
    dates = sessions(REQUIRED_OBSERVATIONS)
    bars = bars_for("AAPL", momentum_prices(), dates)

    first = Momentum12_1().compute(make_context(bars.copy(deep=True)))
    second = Momentum12_1().compute(make_context(bars.copy(deep=True)))

    pd.testing.assert_frame_equal(first.frame, second.frame)


def test_shuffled_input_rows_do_not_change_the_result() -> None:
    """Rows arrive in any order; each symbol's history is sorted by date first."""
    dates = sessions(REQUIRED_OBSERVATIONS)
    bars = pd.concat(
        [
            bars_for("AAPL", momentum_prices(start=100.0, end=130.0), dates),
            bars_for("MSFT", momentum_prices(start=40.0, end=30.0), dates),
        ],
        ignore_index=True,
    )
    ordered = Momentum12_1().compute(make_context(bars, universe=("AAPL", "MSFT")))
    shuffled = Momentum12_1().compute(make_context(bars.iloc[::-1], universe=("AAPL", "MSFT")))

    assert shuffled.symbols == ("AAPL", "MSFT")
    assert values(shuffled) == values(ordered)
    pd.testing.assert_frame_equal(shuffled.frame, ordered.frame)


def test_a_symbol_whose_rows_are_reversed_is_still_computable() -> None:
    """One symbol's own rows in reverse order is the same history."""
    dates = sessions(REQUIRED_OBSERVATIONS)
    prices = momentum_prices(start=100.0, end=130.0)
    reversed_bars = bars_for("AAPL", list(reversed(prices)), list(reversed(dates)))

    panel = Momentum12_1().compute(make_context(reversed_bars))

    assert eligibility(panel) == {"AAPL": True}
    assert values(panel) == {"AAPL": pytest.approx(0.30)}


# ---------------------------------------------------------------------------
# Integration: the real runner, in-memory dependencies
# ---------------------------------------------------------------------------
class MemoryUniverseProvider(UniverseProvider):
    """A universe provider over a literal list: no CSV, no path, no network."""

    def __init__(self, symbols: Sequence[str]) -> None:
        self._symbols = tuple(symbols)
        self.asked: list[date] = []

    def snapshot(self, as_of_date: date | str) -> UniverseSnapshot:
        self.asked.append(pd.Timestamp(as_of_date).date())
        return UniverseSnapshot(
            as_of_date=as_of_date,
            symbols=self._symbols,
            source="in-memory pool",
        )


class WindowMarketDataReader(MarketDataReader):
    """Answers exactly the window it was asked for, from in-memory prices.

    The prices are aligned to the *requested* sessions, so a runner that asked for
    the wrong window would hand the factor the wrong anchors.  A symbol listed in
    ``short`` is truncated to its last ``n`` requested sessions (a recent
    listing); a symbol missing from ``prices`` is not returned at all (a ticker
    with no local history); ``gaps`` drops requested sessions by position (a
    missing bar inside an otherwise complete window).
    """

    def __init__(
        self,
        calendar: Any,
        prices: Mapping[str, Sequence[Any]],
        *,
        short: Mapping[str, int] | None = None,
        gaps: Mapping[str, Sequence[int]] | None = None,
    ) -> None:
        self.calendar = calendar
        self.prices = {symbol: list(series) for symbol, series in prices.items()}
        self.short = dict(short or {})
        self.gaps = dict(gaps or {})
        self.requests: list[dict[str, Any]] = []
        self.answers: list[pd.DataFrame] = []

    def read_history(
        self,
        snapshot: UniverseSnapshot,
        start_date: date,
        required_fields: Sequence[str],
    ) -> pd.DataFrame:
        fields = tuple(required_fields)
        self.requests.append(
            {"snapshot": snapshot, "start_date": start_date, "required_fields": fields}
        )

        window = sessions_in_range(self.calendar, start_date, snapshot.as_of_date)
        frames: list[pd.DataFrame] = []
        for symbol in snapshot.symbols:
            if symbol not in self.prices:
                continue
            count = self.short.get(symbol, len(window))
            dates = window[-count:]
            values = self.prices[symbol][-count:]
            dropped = set(self.gaps.get(symbol, ()))
            kept = [position for position in range(len(dates)) if position not in dropped]
            frames.append(
                bars_for(
                    symbol,
                    [values[position] for position in kept],
                    [dates[position] for position in kept],
                )
            )

        columns = ["date", "symbol", *fields]
        if frames:
            answer = pd.concat(frames, ignore_index=True)[columns]
        else:
            answer = pd.DataFrame(
                {
                    "date": pd.Series(dtype="datetime64[s]"),
                    "symbol": pd.Series(dtype="object"),
                    **{field: pd.Series(dtype="float64") for field in fields},
                }
            )[columns]

        self.answers.append(answer.copy(deep=True))
        return answer


def session_before(count: int) -> date:
    """The XNYS session ``count`` sessions before :data:`AS_OF`.

    Taken from the calendar's own session index rather than from the runner's
    helper, so the test checks the runner against the exchange, not itself.
    """
    window = CALENDAR.sessions_in_range(
        pd.Timestamp(AS_OF) - pd.Timedelta(days=2 * count + 40), pd.Timestamp(AS_OF)
    )
    return window[-(count + 1)].date()


def test_the_runner_asks_for_adjusted_close_over_a_253_session_window() -> None:
    """The real runner asks for one field, over the window the lookback implies."""
    prices = {
        "AAPL": momentum_prices(start=100.0, end=130.0),
        "MSFT": [100.0] * REQUIRED_OBSERVATIONS,
    }
    provider = MemoryUniverseProvider(("AAPL", "MSFT", "NVDA"))
    reader = WindowMarketDataReader(CALENDAR, prices, short={"MSFT": 100})
    runner = FactorRunner(provider, reader)

    panel = runner.run(Momentum12_1(), AS_OF.isoformat())

    # -- what the runner asked for -----------------------------------------
    assert provider.asked == [AS_OF]
    assert len(reader.requests) == 1
    request = reader.requests[0]
    assert request["required_fields"] == ("adjusted_close",)
    assert request["snapshot"].as_of_date == AS_OF
    assert request["snapshot"].symbols == ("AAPL", "MSFT", "NVDA")
    assert request["start_date"] == session_before(LOOKBACK_SESSIONS)

    # -- the window really is 252 past sessions plus the decision day -------
    window = sessions_in_range(CALENDAR, request["start_date"], AS_OF)
    assert len(window) == REQUIRED_OBSERVATIONS
    assert window[0] == session_before(LOOKBACK_SESSIONS)
    assert window[-SKIP_SESSIONS - 1] == session_before(SKIP_SESSIONS)
    assert window[-1] == AS_OF

    # -- and the factor was computed from exactly that data -----------------
    assert isinstance(panel, FactorPanel)
    assert panel.as_of_date == AS_OF
    assert panel.factor_name == "momentum_12_1"
    assert panel.factor_version == "1.0.0"
    assert panel.symbols == ("AAPL", "MSFT", "NVDA")
    assert eligibility(panel) == {"AAPL": True, "MSFT": False, "NVDA": False}
    assert values(panel)["AAPL"] == pytest.approx(0.30)
    assert math.isnan(values(panel)["MSFT"])
    assert math.isnan(values(panel)["NVDA"])


def test_the_runner_returns_exactly_what_the_factor_computed() -> None:
    """The panel the runner returns is the factor's own panel, unrewritten."""
    prices = {
        "AAPL": momentum_prices(start=100.0, end=130.0),
        "MSFT": momentum_prices(start=80.0, end=60.0),
    }
    provider = MemoryUniverseProvider(("AAPL", "MSFT", "NVDA"))
    reader = WindowMarketDataReader(CALENDAR, prices)
    runner = FactorRunner(provider, reader)

    panel = runner.run(Momentum12_1(), AS_OF)

    bars = reader.answers[0]
    assert list(bars.columns) == ["date", "symbol", "adjusted_close"]
    assert len(bars) == 2 * REQUIRED_OBSERVATIONS

    context = FactorContext(as_of_date=AS_OF, universe=("AAPL", "MSFT", "NVDA"), bars=bars)
    direct = Momentum12_1().compute(context)

    pd.testing.assert_frame_equal(panel.frame, direct.frame)
    computed = values(panel)
    assert computed["AAPL"] == pytest.approx(0.30)
    assert computed["MSFT"] == pytest.approx(-0.25)
    assert math.isnan(computed["NVDA"])


def test_a_gap_in_the_read_window_makes_only_that_symbol_ineligible() -> None:
    """252 rows in the requested window is not a year: the gap is never filled."""
    prices = {
        "AAPL": momentum_prices(start=100.0, end=130.0),
        "MSFT": [100.0] * REQUIRED_OBSERVATIONS,
    }
    provider = MemoryUniverseProvider(("AAPL", "MSFT"))
    reader = WindowMarketDataReader(CALENDAR, prices, gaps={"AAPL": (100,)})
    runner = FactorRunner(provider, reader)

    panel = runner.run(Momentum12_1(), AS_OF)

    assert len(reader.answers[0]) == 2 * REQUIRED_OBSERVATIONS - 1
    assert eligibility(panel) == {"AAPL": False, "MSFT": True}
    assert math.isnan(values(panel)["AAPL"])
    assert values(panel)["MSFT"] == 0.0


# ---------------------------------------------------------------------------
# The module's own boundary
# ---------------------------------------------------------------------------
#: Modules ``momentum.py`` may import: the factor contract, its errors, and the
#: standard library/pandas needed for the arithmetic.  No ``data_sys``, no
#: storage, no provider, no network client, no columnar engine.
ALLOWED_IMPORTS: frozenset[str] = frozenset(
    {
        "__future__",
        "datetime",
        "math",
        "numbers",
        "typing",
        "pandas",
        "factor_sys.contracts",
        "factor_sys.errors",
    }
)

#: Text that must never appear in the module: storage, network, file output.
FORBIDDEN_TEXT: tuple[str, ...] = (
    "data_sys",
    "duckdb",
    "pyarrow",
    "yfinance",
    "httpx",
    "requests",
    "socket",
    "urllib",
    "read_parquet",
    "to_parquet",
    "to_csv",
    "to_pickle",
    "write_text",
    "write_bytes",
)


def imported_modules(path: Path) -> set[str]:
    """Every module named by an import statement in ``path``."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def called_names(path: Path) -> set[str]:
    """Every function or method the module calls, exactly as written."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}


def test_the_factor_imports_only_its_own_layer() -> None:
    """No storage, no providers, no Parquet, no DuckDB, no network."""
    path = Path(momentum_module.__file__)

    assert imported_modules(path) <= ALLOWED_IMPORTS
    source = path.read_text(encoding="utf-8").lower()
    assert [text for text in FORBIDDEN_TEXT if text in source] == []


def test_the_factor_never_opens_prints_or_evaluates_anything() -> None:
    """The module calls no builtin that reads, writes, prints or evaluates."""
    called = called_names(Path(momentum_module.__file__))

    assert called & {"open", "print", "input", "eval", "exec", "compile", "__import__"} == set()


def test_the_factor_never_writes_a_file() -> None:
    """A factor hands its panel back; persisting it belongs to a later module."""
    called = called_names(Path(momentum_module.__file__))

    writers = {
        "to_parquet",
        "to_csv",
        "to_pickle",
        "to_feather",
        "to_hdf",
        "write_text",
        "write_bytes",
        "mkdir",
        "makedirs",
    }
    assert called & writers == set()


def test_the_factor_reaches_for_no_calendar_reader_or_provider() -> None:
    """Sessions and bars come from the runner; a formula must not fetch them."""
    called = called_names(Path(momentum_module.__file__))

    forbidden = {
        "get_calendar",
        "sessions_in_range",
        "last_sessions",
        "session_index",
        "read_history",
        "snapshot",
        "read_partitioned",
    }
    assert called & forbidden == set()


def test_the_factor_does_not_import_the_data_layer() -> None:
    """``data_sys`` is below ``factor_sys``; a formula must not see it at all."""
    modules = imported_modules(Path(momentum_module.__file__))

    assert [name for name in modules if name.startswith("data_sys")] == []


def test_the_package_root_imports_nothing_but_the_formula() -> None:
    """``factors/__init__.py`` is an export list, not a place for machinery."""
    assert imported_modules(Path(factors.__file__)) == {"__future__", "factors.momentum"}
