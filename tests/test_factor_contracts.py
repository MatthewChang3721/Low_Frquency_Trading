"""Tests for the factor data contract (:mod:`factor_sys.contracts`).

Everything here is in memory: no ``data/`` directory is read, no Parquet file is
touched and nothing hits the network.  The contract only ever sees plain
``DataFrame``\\ s, which is exactly how the future runner will hand history to a
factor -- so these tests pin down the shared vocabulary before any factor exists.

Covered in this order:

* valid ``FactorSpec`` / ``FactorContext`` / ``FactorPanel`` objects,
* every rejection the contract promises (identity, dates, universe, symbols,
  coverage, duplicates, dtypes),
* the point-in-time guard (no bar dated after ``as_of_date``),
* immutability and defensive copying (mutating an input never changes an object),
* the abstract :class:`Factor` interface,
* the architectural boundary: the contract never reaches into the storage layer.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import FrozenInstanceError, fields
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest

import factor_sys
from data_sys.errors import DataPipelineError
from factor_sys import (
    FACTOR_PANEL_COLUMNS,
    Factor,
    FactorContext,
    FactorContractError,
    FactorError,
    FactorPanel,
    FactorSpec,
)

#: The in-memory universe every fixture below works with.
SYMBOLS: tuple[str, ...] = ("AAPL", "MSFT", "NVDA")

#: The last session of the generated history, i.e. the context's ``as_of_date``.
AS_OF: date = date(2024, 1, 5)

#: How many sessions :func:`make_bars` generates by default.
SESSIONS: int = 5


# ---------------------------------------------------------------------------
# In-memory fixtures (pure DataFrame builders, no disk, no network)
# ---------------------------------------------------------------------------
def make_bars(
    symbols: Sequence[str] = SYMBOLS,
    *,
    sessions: int = SESSIONS,
    end: str = "2024-01-05",
    extra_column: bool = False,
) -> pd.DataFrame:
    """Long-format daily bars covering ``symbols`` up to ``end``."""
    sessions_range = pd.bdate_range(end=end, periods=sessions)
    records: list[dict[str, object]] = []
    for symbol in symbols:
        for offset, session in enumerate(sessions_range):
            record: dict[str, object] = {
                "date": session,
                "symbol": symbol,
                "close": 100.0 + offset,
                "volume": 1_000_000 + offset,
            }
            if extra_column:
                record["adjusted_close"] = 99.5 + offset
            records.append(record)
    return pd.DataFrame(records)


def make_context(
    symbols: Sequence[str] = SYMBOLS,
    *,
    bars: pd.DataFrame | None = None,
    as_of_date: object = AS_OF,
    universe: Sequence[str] | None = None,
    **bar_options: object,
) -> FactorContext:
    """A :class:`FactorContext` built from in-memory bars."""
    frame = make_bars(symbols, **bar_options) if bars is None else bars
    return FactorContext(
        as_of_date=as_of_date,  # type: ignore[arg-type]
        universe=list(symbols) if universe is None else universe,
        bars=frame,
    )


def make_panel_frame(
    symbols: Sequence[str] = SYMBOLS,
    *,
    as_of_date: object = AS_OF,
    factor_name: object = "momentum_12_1",
    factor_version: object = "1.0.0",
    raw_value: object = None,
    is_eligible: object = True,
) -> pd.DataFrame:
    """A complete, valid raw panel output for one factor and one session."""
    values = range(1, len(symbols) + 1) if raw_value is None else raw_value
    return pd.DataFrame(
        [
            {
                "as_of_date": as_of_date,
                "symbol": symbol,
                "factor_name": factor_name,
                "factor_version": factor_version,
                "raw_value": value,
                "is_eligible": is_eligible,
            }
            for symbol, value in zip(symbols, values, strict=True)
        ],
        columns=list(FACTOR_PANEL_COLUMNS),
    )


def make_panel(
    frame: pd.DataFrame | None = None,
    *,
    symbols: Sequence[str] = SYMBOLS,
    **panel_options: object,
) -> FactorPanel:
    """A :class:`FactorPanel` built from in-memory output."""
    rows = make_panel_frame(symbols, **panel_options) if frame is None else frame
    return FactorPanel(rows, universe=list(symbols))


# ---------------------------------------------------------------------------
# FactorSpec: what a factor needs
# ---------------------------------------------------------------------------
def test_factor_spec_accepts_a_valid_declaration() -> None:
    spec = FactorSpec(
        name="momentum_12_1",
        version="1.0.0",
        required_fields=("adjusted_close", "volume"),
        lookback_sessions=252,
    )

    assert spec.name == "momentum_12_1"
    assert spec.version == "1.0.0"
    assert spec.required_fields == ("adjusted_close", "volume")
    assert spec.lookback_sessions == 252


def test_factor_spec_defaults_lookback_to_zero() -> None:
    """A factor without warm-up does not have to think about a lookback."""
    spec = FactorSpec(name="reversal_1m", version="0.1.0", required_fields=("close",))

    assert spec.lookback_sessions == 0


def test_factor_spec_freezes_the_required_fields() -> None:
    """The caller's list is copied into a tuple, so later mutation cannot leak in."""
    declared = ["close", "volume"]
    spec = FactorSpec(
        name="reversal_1m",
        version="0.1.0",
        required_fields=declared,  # type: ignore[arg-type]
        lookback_sessions=21,
    )

    declared.append("open")

    assert spec.required_fields == ("close", "volume")
    assert isinstance(spec.required_fields, tuple)
    assert isinstance(spec.lookback_sessions, int)


def test_factor_spec_is_immutable() -> None:
    spec = FactorSpec(name="momentum_12_1", version="1.0.0", required_fields=("close",))

    with pytest.raises(FrozenInstanceError):
        spec.name = "other"  # type: ignore[misc]


@pytest.mark.parametrize(
    "name",
    ["", "   ", "Momentum_12_1", "momentum-12-1", "momentum 12", "momentum.12", "动量", 12, None],
)
def test_factor_spec_rejects_an_invalid_name(name: object) -> None:
    with pytest.raises(FactorContractError) as excinfo:
        FactorSpec(name=name, version="1.0.0", required_fields=("close",))  # type: ignore[arg-type]

    assert "name" in str(excinfo.value)


def test_factor_spec_rejects_duplicate_fields() -> None:
    with pytest.raises(FactorContractError) as excinfo:
        FactorSpec(name="momentum_12_1", version="1.0.0", required_fields=("close", "close"))

    assert "duplicate" in str(excinfo.value).lower()


@pytest.mark.parametrize("version", ["", "   ", 1.5, None])
def test_factor_spec_rejects_an_unusable_version(version: object) -> None:
    with pytest.raises(FactorContractError) as excinfo:
        FactorSpec(name="momentum_12_1", version=version, required_fields=("close",))  # type: ignore[arg-type]

    assert "version" in str(excinfo.value)


@pytest.mark.parametrize("required", [(), [], "close", ("close", ""), ("close", 1)])
def test_factor_spec_rejects_bad_required_fields(required: object) -> None:
    with pytest.raises(FactorContractError) as excinfo:
        FactorSpec(
            name="momentum_12_1",
            version="1.0.0",
            required_fields=required,  # type: ignore[arg-type]
        )

    assert "required_fields" in str(excinfo.value)


@pytest.mark.parametrize("lookback", [-1, -252, 1.5, "252", None, True])
def test_factor_spec_rejects_a_bad_lookback(lookback: object) -> None:
    with pytest.raises(FactorContractError) as excinfo:
        FactorSpec(
            name="momentum_12_1",
            version="1.0.0",
            required_fields=("close",),
            lookback_sessions=lookback,  # type: ignore[arg-type]
        )

    assert "lookback_sessions" in str(excinfo.value)


def test_factor_spec_reports_every_problem_at_once() -> None:
    """Validation collects: a factor author sees all mistakes in one run."""
    with pytest.raises(FactorContractError) as excinfo:
        FactorSpec(
            name="Momentum",
            version="",
            required_fields=("close", "close"),
            lookback_sessions=-5,
        )

    message = str(excinfo.value)
    assert "name" in message
    assert "version" in message
    assert "duplicate" in message.lower()
    assert "lookback_sessions" in message
    assert len(excinfo.value.problems) == 4


@pytest.mark.parametrize("label", ["date", "symbol"])
def test_factor_spec_rejects_a_label_column_as_a_required_field(label: str) -> None:
    """``date`` and ``symbol`` are always present; a factor cannot request them.

    The market-data reader refuses to be asked for them (they are the frame's
    label columns), so a spec that declared them could never run: the mistake is
    caught where the factor is written, not only when the runner reads history.
    """
    with pytest.raises(FactorContractError) as excinfo:
        FactorSpec(name="momentum_12_1", version="1.0.0", required_fields=(label,))

    assert "required_fields" in str(excinfo.value)
    assert "label column" in str(excinfo.value)
    assert label in str(excinfo.value)
    assert len(excinfo.value.problems) == 1


def test_factor_spec_collects_both_label_column_problems() -> None:
    """Each offending entry is reported once, in the order it was declared."""
    with pytest.raises(FactorContractError) as excinfo:
        FactorSpec(
            name="momentum_12_1",
            version="1.0.0",
            required_fields=("close", "date", "symbol"),
        )

    assert len(excinfo.value.problems) == 2
    assert "'date'" in excinfo.value.problems[0]
    assert "'symbol'" in excinfo.value.problems[1]


def test_factor_spec_reports_label_columns_with_the_other_problems() -> None:
    """The label rule joins the collected problems instead of short-circuiting."""
    with pytest.raises(FactorContractError) as excinfo:
        FactorSpec(name="Momentum", version="", required_fields=("date",), lookback_sessions=-1)

    message = str(excinfo.value)
    assert "name" in message
    assert "version" in message
    assert "label column" in message
    assert "lookback_sessions" in message
    assert len(excinfo.value.problems) == 4


# ---------------------------------------------------------------------------
# FactorContext: what a factor may look at
# ---------------------------------------------------------------------------
def test_factor_context_accepts_clean_history() -> None:
    context = make_context()

    assert context.as_of_date == AS_OF
    assert isinstance(context.as_of_date, date)
    assert context.universe == SYMBOLS
    assert list(context.bars.columns) == ["date", "symbol", "close", "volume"]
    assert str(context.bars["date"].dtype) == "datetime64[s]"
    assert context.bars["date"].max().date() == AS_OF
    assert len(context.bars) == len(SYMBOLS) * SESSIONS


@pytest.mark.parametrize("as_of_date", [AS_OF, datetime(2024, 1, 5, 21, 30), "2024-01-05"])
def test_factor_context_normalizes_as_of_date(as_of_date: object) -> None:
    """A date, a datetime or an ISO string all become a plain ``date``."""
    context = make_context(as_of_date=as_of_date)

    assert context.as_of_date == AS_OF
    assert type(context.as_of_date) is date


def test_factor_context_accepts_extra_bar_columns() -> None:
    """A factor declares required fields; the context only demands date+symbol."""
    context = make_context(bars=make_bars(extra_column=True))

    assert "adjusted_close" in context.bars.columns


def test_factor_context_accepts_a_bar_dated_on_the_as_of_date() -> None:
    """The boundary: a bar *on* the as_of_date is visible, not future data."""
    bars = make_bars(("AAPL",), sessions=1)
    bars.loc[0, "date"] = pd.Timestamp(AS_OF) + pd.Timedelta(hours=16)

    context = FactorContext(as_of_date=AS_OF, universe=SYMBOLS, bars=bars)

    assert context.bars["date"].max().date() == AS_OF


def test_factor_context_accepts_partial_coverage_and_short_history() -> None:
    """A subset of the universe with two sessions of history is valid input."""
    bars = make_bars(("AAPL",), sessions=2, end="2023-06-02")
    context = FactorContext(as_of_date=date(2023, 6, 2), universe=SYMBOLS, bars=bars)

    assert context.bars["symbol"].unique().tolist() == ["AAPL"]
    assert len(context.bars) == 2


def test_factor_context_accepts_an_empty_history() -> None:
    """No visible history yet is a valid -- if useless -- slice for a factor."""
    context = make_context(bars=make_bars(("AAPL",), sessions=1).iloc[0:0])

    assert context.bars.empty
    assert str(context.bars["date"].dtype) == "datetime64[s]"


def test_factor_context_string_dates_are_parsed() -> None:
    """Bars read from a CSV-like source still work: dates are parsed, not trusted."""
    bars = make_bars(("AAPL",), sessions=2)
    bars["date"] = bars["date"].dt.date.astype(str)

    context = FactorContext(as_of_date=AS_OF, universe=SYMBOLS, bars=bars)

    assert str(context.bars["date"].dtype) == "datetime64[s]"
    assert context.bars["date"].max().date() == AS_OF


def test_factor_context_rejects_future_bars() -> None:
    """A look-ahead leak fails loudly instead of silently inflating a backtest."""
    bars = make_bars()
    bars.loc[0, "date"] = pd.Timestamp("2024-01-08")

    with pytest.raises(FactorContractError) as excinfo:
        make_context(bars=bars)

    message = str(excinfo.value)
    assert "future" in message
    assert "2024-01-08" in message


def test_factor_context_rejects_a_single_future_row() -> None:
    """One leaked session is enough, even when every other row is in range."""
    bars = make_bars()
    bars.loc[len(bars) - 1, "date"] = pd.Timestamp("2024-02-01")

    with pytest.raises(FactorContractError) as excinfo:
        make_context(bars=bars)

    assert "future" in str(excinfo.value)
    assert "2024-02-01" in str(excinfo.value)


def test_factor_context_rejects_a_future_time_on_the_next_day() -> None:
    """A 16:00 bar *after* the as_of_date is still future data."""
    bars = make_bars(("AAPL",), sessions=1)
    bars.loc[0, "date"] = pd.Timestamp(AS_OF) + pd.Timedelta(days=1, hours=16)

    with pytest.raises(FactorContractError) as excinfo:
        FactorContext(as_of_date=AS_OF, universe=SYMBOLS, bars=bars)

    assert "future" in str(excinfo.value)


@pytest.mark.parametrize("column", ["date", "symbol"])
def test_factor_context_requires_date_and_symbol(column: str) -> None:
    bars = make_bars().drop(columns=[column])

    with pytest.raises(FactorContractError) as excinfo:
        make_context(bars=bars)

    assert column in str(excinfo.value)


def test_factor_context_rejects_unparseable_dates() -> None:
    bars = make_bars()
    bars["date"] = bars["date"].dt.date.astype(str)
    bars.loc[0, "date"] = "not-a-date"

    with pytest.raises(FactorContractError) as excinfo:
        make_context(bars=bars)

    assert "date" in str(excinfo.value)


def test_factor_context_rejects_missing_dates() -> None:
    bars = make_bars()
    bars.loc[0, "date"] = None

    with pytest.raises(FactorContractError) as excinfo:
        make_context(bars=bars)

    assert "date" in str(excinfo.value)


def test_factor_context_rejects_symbols_outside_the_universe() -> None:
    bars = make_bars(("AAPL", "GOOGL"))

    with pytest.raises(FactorContractError) as excinfo:
        make_context(bars=bars)

    assert "GOOGL" in str(excinfo.value)


def test_factor_context_rejects_duplicate_symbol_date_rows() -> None:
    bars = make_bars(("AAPL",), sessions=2)
    bars = pd.concat([bars, bars.iloc[[0]]], ignore_index=True)

    with pytest.raises(FactorContractError) as excinfo:
        make_context(bars=bars, symbols=("AAPL",))

    assert "duplicate" in str(excinfo.value).lower()
    assert "AAPL" in str(excinfo.value)


def test_factor_context_rejects_a_non_frame() -> None:
    with pytest.raises(FactorContractError) as excinfo:
        FactorContext(
            as_of_date=AS_OF,
            universe=SYMBOLS,
            bars=[{"date": AS_OF, "symbol": "AAPL"}],  # type: ignore[arg-type]
        )

    assert "DataFrame" in str(excinfo.value)


@pytest.mark.parametrize(
    "universe",
    [(), [], ["AAPL", "AAPL"], ["aapl"], ["AAP L"], [""], ["AAPL", None], ["AAPL", 1], "AAPL"],
)
def test_factor_context_rejects_an_invalid_universe(universe: object) -> None:
    with pytest.raises(FactorContractError) as excinfo:
        FactorContext(
            as_of_date=AS_OF,
            universe=universe,  # type: ignore[arg-type]
            bars=make_bars(),
        )

    assert "universe" in str(excinfo.value)


@pytest.mark.parametrize("as_of_date", [None, "", "not-a-date", 20240105])
def test_factor_context_rejects_an_unusable_as_of_date(as_of_date: object) -> None:
    with pytest.raises(FactorContractError) as excinfo:
        FactorContext(
            as_of_date=as_of_date,  # type: ignore[arg-type]
            universe=SYMBOLS,
            bars=make_bars(),
        )

    assert "as_of_date" in str(excinfo.value)


def test_factor_context_is_immutable() -> None:
    context = make_context()

    with pytest.raises(FrozenInstanceError):
        context.as_of_date = date(2024, 1, 8)  # type: ignore[misc]


def test_factor_context_bars_are_a_defensive_copy() -> None:
    """Mutating the caller's frame afterwards cannot change the context."""
    bars = make_bars()
    context = FactorContext(as_of_date=AS_OF, universe=SYMBOLS, bars=bars)

    bars.loc[:, "close"] = -1.0
    bars.loc[0, "symbol"] = "GOOGL"
    bars.loc[0, "date"] = pd.Timestamp("2024-12-31")

    assert context.bars["close"].min() > 0.0
    assert "GOOGL" not in context.bars["symbol"].tolist()
    assert context.bars["date"].max().date() == AS_OF
    assert context.bars is not bars


def test_factor_context_keeps_the_caller_row_order() -> None:
    """The contract normalizes dtypes; it never reorders or drops history."""
    shuffled = make_bars().sample(frac=1.0, random_state=7).reset_index(drop=True)

    context = FactorContext(as_of_date=AS_OF, universe=SYMBOLS, bars=shuffled)

    assert context.bars["symbol"].tolist() == shuffled["symbol"].tolist()
    assert list(context.bars.index) == list(range(len(shuffled)))


# ---------------------------------------------------------------------------
# FactorPanel: what a factor returns
# ---------------------------------------------------------------------------
def test_factor_panel_accepts_a_complete_cross_section() -> None:
    panel = FactorPanel(make_panel_frame(), universe=SYMBOLS)

    assert list(panel.frame.columns) == list(FACTOR_PANEL_COLUMNS)
    assert panel.as_of_date == AS_OF
    assert panel.universe == SYMBOLS
    assert panel.symbols == SYMBOLS
    assert panel.factor_name == "momentum_12_1"
    assert panel.factor_version == "1.0.0"
    assert len(panel) == len(SYMBOLS)
    assert str(panel.frame["as_of_date"].dtype) == "datetime64[s]"
    assert str(panel.frame["raw_value"].dtype) == "float64"
    assert str(panel.frame["is_eligible"].dtype) == "bool"
    assert panel.frame["is_eligible"].tolist() == [True, True, True]
    assert panel.frame["raw_value"].tolist() == [1.0, 2.0, 3.0]


def test_factor_panel_normalizes_the_column_order() -> None:
    """Any column order is accepted; the panel always exposes the canonical one."""
    shuffled = list(reversed(FACTOR_PANEL_COLUMNS))

    panel = FactorPanel(make_panel_frame()[shuffled], universe=SYMBOLS)

    assert list(panel.frame.columns) == list(FACTOR_PANEL_COLUMNS)


def test_factor_panel_accepts_dates_as_strings_and_datetimes() -> None:
    frame = make_panel_frame(as_of_date="2024-01-05")

    panel = FactorPanel(frame, universe=SYMBOLS)

    assert panel.as_of_date == AS_OF


def test_factor_panel_accepts_a_matching_explicit_as_of_date() -> None:
    panel = FactorPanel(make_panel_frame(), universe=SYMBOLS, as_of_date="2024-01-05")

    assert panel.as_of_date == AS_OF


def test_factor_panel_accepts_string_numbers_for_raw_value() -> None:
    """``raw_value`` only has to be *convertible* to float."""
    frame = make_panel_frame()
    frame["raw_value"] = ["1.5", None, "3"]

    panel = FactorPanel(frame, universe=SYMBOLS)
    values = panel.frame["raw_value"]

    assert values.iloc[0] == pytest.approx(1.5)
    assert pd.isna(values.iloc[1])
    assert values.iloc[2] == pytest.approx(3.0)


def test_factor_panel_allows_a_missing_value_for_an_ineligible_symbol() -> None:
    frame = make_panel_frame()
    frame.loc[1, "raw_value"] = None
    frame.loc[1, "is_eligible"] = False

    panel = FactorPanel(frame, universe=SYMBOLS)

    assert pd.isna(panel.frame.loc[1, "raw_value"])
    assert panel.frame["is_eligible"].tolist() == [True, False, True]


def test_factor_panel_does_not_police_eligibility_semantics() -> None:
    """``is_eligible`` is the factor's decision; the contract only types it."""
    frame = make_panel_frame()
    frame.loc[1, "raw_value"] = None  # eligible but unknown
    frame.loc[2, "is_eligible"] = False  # ineligible but valued

    panel = FactorPanel(frame, universe=SYMBOLS)

    assert panel.frame["is_eligible"].tolist() == [True, True, False]


def test_factor_panel_frame_is_a_defensive_copy() -> None:
    """The returned frame is a copy: mutating it cannot change the panel."""
    panel = FactorPanel(make_panel_frame(), universe=SYMBOLS)

    exposed = panel.frame
    exposed.loc[:, "raw_value"] = 0.0
    exposed.loc[0, "symbol"] = "GOOGL"

    assert panel.frame["raw_value"].tolist() == [1.0, 2.0, 3.0]
    assert panel.symbols == SYMBOLS


def test_factor_panel_does_not_alias_the_caller_frame() -> None:
    """Mutating the input frame afterwards cannot change the panel."""
    frame = make_panel_frame()
    panel = FactorPanel(frame, universe=SYMBOLS)

    frame.loc[:, "raw_value"] = 99.0
    frame.loc[0, "symbol"] = "GOOGL"

    assert panel.frame["raw_value"].tolist() == [1.0, 2.0, 3.0]
    assert panel.symbols == SYMBOLS


def test_factor_panel_is_immutable() -> None:
    panel = make_panel()

    with pytest.raises(FrozenInstanceError):
        panel.factor_name = "other"  # type: ignore[misc]


def test_factor_panel_can_be_built_from_a_context() -> None:
    """The intended call site: a factor returns a panel for its own context."""
    context = make_context(("AAPL", "MSFT"))

    panel = FactorPanel(
        make_panel_frame(("AAPL", "MSFT")),
        universe=context.universe,
        as_of_date=context.as_of_date,
    )

    assert panel.universe == context.universe
    assert panel.as_of_date == context.as_of_date
    assert len(panel) == len(context.universe)


def test_factor_panel_rejects_missing_output_symbols() -> None:
    """Every universe symbol needs a row: silently dropping one is a bug."""
    frame = make_panel_frame(("AAPL", "MSFT"))

    with pytest.raises(FactorContractError) as excinfo:
        FactorPanel(frame, universe=SYMBOLS)

    message = str(excinfo.value)
    assert "NVDA" in message
    assert "cover" in message


def test_factor_panel_rejects_symbols_outside_the_universe() -> None:
    frame = pd.concat([make_panel_frame(), make_panel_frame(("GOOGL",))], ignore_index=True)

    with pytest.raises(FactorContractError) as excinfo:
        FactorPanel(frame, universe=SYMBOLS)

    assert "GOOGL" in str(excinfo.value)


def test_factor_panel_rejects_duplicate_output_rows() -> None:
    frame = make_panel_frame()
    duplicated = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)

    with pytest.raises(FactorContractError) as excinfo:
        FactorPanel(duplicated, universe=SYMBOLS)

    assert "duplicate" in str(excinfo.value).lower()


def test_factor_panel_rejects_multiple_dates_in_one_cross_section() -> None:
    frame = make_panel_frame()
    frame.loc[0, "as_of_date"] = date(2024, 1, 8)

    with pytest.raises(FactorContractError) as excinfo:
        FactorPanel(frame, universe=SYMBOLS)

    assert "as_of_date" in str(excinfo.value)


def test_factor_panel_rejects_a_mismatched_as_of_date() -> None:
    with pytest.raises(FactorContractError) as excinfo:
        FactorPanel(make_panel_frame(), universe=SYMBOLS, as_of_date=date(2024, 1, 8))

    message = str(excinfo.value)
    assert "2024-01-05" in message
    assert "2024-01-08" in message


def test_factor_panel_rejects_missing_dates() -> None:
    frame = make_panel_frame()
    frame.loc[0, "as_of_date"] = None

    with pytest.raises(FactorContractError) as excinfo:
        FactorPanel(frame, universe=SYMBOLS)

    assert "as_of_date" in str(excinfo.value)


def test_factor_panel_rejects_multiple_factor_names() -> None:
    """One panel is one cross-section of one factor, not a multi-factor bundle."""
    frame = make_panel_frame()
    frame.loc[0, "factor_name"] = "reversal_1m"

    with pytest.raises(FactorContractError) as excinfo:
        FactorPanel(frame, universe=SYMBOLS)

    assert "factor_name" in str(excinfo.value)


def test_factor_panel_rejects_multiple_factor_versions() -> None:
    frame = make_panel_frame()
    frame.loc[0, "factor_version"] = "2.0.0"

    with pytest.raises(FactorContractError) as excinfo:
        FactorPanel(frame, universe=SYMBOLS)

    assert "factor_version" in str(excinfo.value)


@pytest.mark.parametrize("overrides", [{"factor_name": "Momentum"}, {"factor_version": ""}])
def test_factor_panel_rejects_an_invalid_identity(overrides: dict[str, object]) -> None:
    with pytest.raises(FactorContractError) as excinfo:
        FactorPanel(make_panel_frame(**overrides), universe=SYMBOLS)

    assert "must be" in str(excinfo.value)


def test_factor_panel_rejects_a_non_numeric_raw_value() -> None:
    frame = make_panel_frame()
    frame["raw_value"] = ["1.0", "2.0", "not-a-number"]

    with pytest.raises(FactorContractError) as excinfo:
        FactorPanel(frame, universe=SYMBOLS)

    assert "raw_value" in str(excinfo.value)


@pytest.mark.parametrize("eligible", [[1, 0, 1], ["yes", "no", "yes"], [None, True, False]])
def test_factor_panel_rejects_a_non_boolean_is_eligible(eligible: list[object]) -> None:
    frame = make_panel_frame()
    frame["is_eligible"] = eligible

    with pytest.raises(FactorContractError) as excinfo:
        FactorPanel(frame, universe=SYMBOLS)

    assert "is_eligible" in str(excinfo.value)


def test_factor_panel_rejects_missing_eligibility() -> None:
    frame = make_panel_frame()
    frame["is_eligible"] = pd.array([True, pd.NA, True], dtype="boolean")

    with pytest.raises(FactorContractError) as excinfo:
        FactorPanel(frame, universe=SYMBOLS)

    assert "is_eligible" in str(excinfo.value)


def test_factor_panel_rejects_missing_or_extra_columns() -> None:
    with pytest.raises(FactorContractError) as excinfo:
        FactorPanel(make_panel_frame().drop(columns=["is_eligible"]), universe=SYMBOLS)
    assert "is_eligible" in str(excinfo.value)

    extra = make_panel_frame()
    extra["zscore"] = 0.0  # transforms belong to a later module
    with pytest.raises(FactorContractError) as excinfo:
        FactorPanel(extra, universe=SYMBOLS)
    assert "zscore" in str(excinfo.value)


def test_factor_panel_rejects_an_empty_frame() -> None:
    with pytest.raises(FactorContractError) as excinfo:
        FactorPanel(make_panel_frame().iloc[0:0], universe=SYMBOLS)

    assert "not be empty" in str(excinfo.value)


def test_factor_panel_rejects_a_non_frame() -> None:
    with pytest.raises(FactorContractError) as excinfo:
        FactorPanel([{"symbol": "AAPL"}], universe=SYMBOLS)  # type: ignore[arg-type]

    assert "DataFrame" in str(excinfo.value)


def test_factor_panel_rejects_an_empty_universe() -> None:
    with pytest.raises(FactorContractError) as excinfo:
        FactorPanel(make_panel_frame(), universe=[])

    assert "universe" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Factor: the formula interface
# ---------------------------------------------------------------------------
def test_factor_is_an_abstract_base_class() -> None:
    assert {"spec", "compute"} <= set(Factor.__abstractmethods__)

    with pytest.raises(TypeError):
        Factor()  # type: ignore[abstract]


def test_factor_requires_both_spec_and_compute() -> None:
    class SpecOnly(Factor):
        spec = FactorSpec(name="spec_only", version="1.0.0", required_fields=("close",))

    class ComputeOnly(Factor):
        def compute(self, context: FactorContext) -> FactorPanel:  # pragma: no cover
            raise NotImplementedError

    for incomplete in (SpecOnly, ComputeOnly):
        with pytest.raises(TypeError):
            incomplete()  # type: ignore[abstract]


def test_factor_may_declare_its_spec_as_a_property_or_a_class_attribute() -> None:
    class ClassAttributeFactor(Factor):
        spec = FactorSpec(name="class_attr", version="1.0.0", required_fields=("close",))

        def compute(self, context: FactorContext) -> FactorPanel:  # pragma: no cover
            raise NotImplementedError

    class PropertyFactor(Factor):
        @property
        def spec(self) -> FactorSpec:
            return FactorSpec(name="property", version="1.0.0", required_fields=("close",))

        def compute(self, context: FactorContext) -> FactorPanel:  # pragma: no cover
            raise NotImplementedError

    assert ClassAttributeFactor().spec.name == "class_attr"
    assert PropertyFactor().spec.name == "property"


def test_factor_compute_returns_a_validated_panel() -> None:
    """The smallest useful factor: read the context, return a panel."""

    class LastCloseFactor(Factor):
        spec = FactorSpec(
            name="last_close",
            version="1.0.0",
            required_fields=("close",),
            lookback_sessions=1,
        )

        def compute(self, context: FactorContext) -> FactorPanel:
            last_close = context.bars.sort_values("date").groupby("symbol")["close"].last()
            frame = pd.DataFrame(
                {
                    "as_of_date": context.as_of_date,
                    "symbol": list(context.universe),
                    "factor_name": self.spec.name,
                    "factor_version": self.spec.version,
                    "raw_value": [
                        float(last_close.get(symbol, float("nan"))) for symbol in context.universe
                    ],
                    "is_eligible": [symbol in last_close.index for symbol in context.universe],
                }
            )
            return FactorPanel(
                frame,
                universe=context.universe,
                as_of_date=context.as_of_date,
            )

    factor = LastCloseFactor()
    panel = factor.compute(make_context())

    assert isinstance(panel, FactorPanel)
    assert panel.factor_name == "last_close"
    assert panel.factor_version == "1.0.0"
    assert panel.symbols == SYMBOLS
    assert panel.as_of_date == AS_OF
    assert panel.frame["raw_value"].tolist() == [104.0, 104.0, 104.0]
    assert panel.frame["is_eligible"].tolist() == [True, True, True]
    assert repr(factor).startswith("LastCloseFactor(spec=FactorSpec(")


def test_factor_output_is_validated_by_the_contract() -> None:
    """A factor that forgets a symbol cannot smuggle a broken panel through."""

    class SloppyFactor(Factor):
        spec = FactorSpec(name="sloppy", version="1.0.0", required_fields=("close",))

        def compute(self, context: FactorContext) -> FactorPanel:
            frame = pd.DataFrame(
                {
                    "as_of_date": context.as_of_date,
                    "symbol": list(context.universe)[:-1],
                    "factor_name": self.spec.name,
                    "factor_version": self.spec.version,
                    "raw_value": 1.0,
                    "is_eligible": True,
                }
            )
            return FactorPanel(frame, universe=context.universe)

    with pytest.raises(FactorContractError):
        SloppyFactor().compute(make_context())


def test_a_long_lookback_does_not_invalidate_a_short_context() -> None:
    """Warm-up is the factor's business: a short context is still valid input."""
    context = make_context(("AAPL",), sessions=2)
    spec = FactorSpec(
        name="long_warmup",
        version="1.0.0",
        required_fields=("close",),
        lookback_sessions=252,
    )

    assert spec.lookback_sessions == 252
    assert len(context.bars) == 2


# ---------------------------------------------------------------------------
# Package surface and architectural boundary
# ---------------------------------------------------------------------------
def test_the_contract_is_exported_from_the_package_root() -> None:
    expected = {
        "FACTOR_PANEL_COLUMNS",
        "Factor",
        "FactorContext",
        "FactorContractError",
        "FactorError",
        "FactorPanel",
        "FactorSpec",
    }

    assert expected <= set(factor_sys.__all__)
    assert factor_sys.FactorSpec is FactorSpec
    assert factor_sys.FactorContext is FactorContext
    assert factor_sys.FactorPanel is FactorPanel
    assert factor_sys.Factor is Factor
    assert factor_sys.FactorContractError is FactorContractError


def test_factor_errors_are_a_separate_hierarchy() -> None:
    """A factor contract violation must not masquerade as a data-pipeline error."""
    assert issubclass(FactorContractError, FactorError)
    assert not issubclass(FactorContractError, DataPipelineError)
    assert not issubclass(FactorError, DataPipelineError)


def test_contract_errors_carry_their_problems() -> None:
    error = FactorContractError("invalid thing", ["first problem", "second problem"])

    assert error.problems == ["first problem", "second problem"]
    assert "first problem" in str(error)
    assert "second problem" in str(error)
    assert FactorContractError("plain").problems == []


def test_the_contract_modules_never_reach_into_the_storage_layer() -> None:
    """``Factor`` and the contract objects stay free of files, Parquet and I/O.

    Only the contract modules are guarded: the future runner and factor storage
    modules are *supposed* to do I/O.
    """
    package = Path(factor_sys.__file__).parent
    guarded = [package / "__init__.py", package / "contracts.py", package / "errors.py"]
    forbidden_import = re.compile(
        r"^\s*(?:import|from)\s+(?:pathlib|pyarrow|duckdb|data_sys\.storage)\b",
        re.MULTILINE,
    )
    forbidden_calls = ("read_parquet", "to_parquet", "write_parquet", "open(")

    offenders: list[str] = []
    for module in guarded:
        source = module.read_text(encoding="utf-8")
        if forbidden_import.search(source):
            offenders.append(f"{module.name}: forbidden import")
        offenders.extend(f"{module.name}: {token}" for token in forbidden_calls if token in source)

    assert not offenders, offenders


def test_the_contract_stays_deliberately_small() -> None:
    """No transforms, no readers, no runner: this step is the contract only."""
    spec_fields = {record.name for record in fields(FactorSpec)}

    assert spec_fields == {"name", "version", "required_fields", "lookback_sessions"}
    assert not hasattr(factor_sys, "zscore")
    assert not hasattr(factor_sys, "load_bars")
    assert not hasattr(factor_sys, "run_factor")
    assert not hasattr(factor_sys, "storage")
