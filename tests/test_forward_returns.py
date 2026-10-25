"""Tests for forward-return labels (:mod:`factor_sys.forward_returns`).

Everything here is in memory and offline: the market-data reader is a plain
object serving a literal frame, so nothing reads ``data/``, calls a vendor,
writes a file or needs a clock.  Session arithmetic is checked against the real
``XNYS`` calendar, because "the next session" is a fact about the exchange, not
about this module.

Covered in this order:

* the frozen label object: its identity fields, column order, universe order,
  the ``frame`` copy that cannot change it, and the ``repr`` a log prints,
* every construction it must refuse: a row order that is not the universe order,
  a missing/repeated/foreign symbol, an identity that disagrees with the rows, a
  horizon that is not a positive integer, an ``end_date`` that is not later, an
  available row holding ``NaN``/``inf``, an unavailable row holding a number,
  bad dates, names, versions, columns and dtypes,
* the calculation: the exact close-to-close formula, ``horizon_sessions=1`` as
  the *next* session, weekends and Good Friday, longer horizons, the exact read
  window the reader receives (universe, start, ``as_of_date == end_date``, the
  single field ``adjusted_close``),
* per-symbol handling: a missing start or end bar, ``NaN``/``inf``/``0``/
  negative/text prices, an empty answer, no borrowing of a neighbouring close,
  missing intermediate sessions,
* every untrusted-reader rejection, and ``MarketDataReadError`` propagating
  unchanged,
* the request rejections: a non-panel, an impossible horizon, a decision date
  that is not a session, an unusable reader or calendar name,
* purity and point-in-time safety: the panel survives a call unchanged, labels
  ignore ``raw_value`` completely, an identical call answers identically, and
  the labels never reach past ``end_date``,
* the error type and the package exports,
* the architectural boundary: the module's imports, the objects it mentions, the
  factor columns it never reads, and the fact that neither ``factor_sys/runner``
  nor ``factors/`` knows what a forward return is.
"""

from __future__ import annotations

import ast
import math
from collections.abc import Sequence
from dataclasses import FrozenInstanceError
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest

import factor_sys
import factor_sys.forward_returns as forward_returns_module
from data_sys.errors import CalendarError
from data_sys.market_calendar import get_calendar, sessions_in_range
from factor_sys import (
    FactorError,
    FactorPanel,
    ForwardReturnCalculator,
    ForwardReturnError,
    ForwardReturnPanel,
    MarketDataReader,
    MarketDataReadError,
    UniverseSnapshot,
)
from factor_sys.forward_returns import FORWARD_RETURN_COLUMNS, PRICE_FIELD

#: A three-symbol pool, in the order the labels must follow.
SYMBOLS: tuple[str, ...] = ("AAPL", "MSFT", "NVDA")

#: A real XNYS session (Maundy Thursday, the day before Good Friday).
AS_OF: date = date(2024, 3, 28)

#: The next session after ``AS_OF``: Good Friday is not a session.
END_ONE: date = date(2024, 4, 1)

#: Identity every fixture uses.
NAME: str = "momentum_12_1"
VERSION: str = "1.0.0"

#: The real calendar, for expectations that must come from the exchange itself.
CALENDAR = get_calendar("XNYS")

NAN = float("nan")
INF = float("inf")

#: The module under test, read as source for the boundary tests at the end.
MODULE_PATH: Path = Path(forward_returns_module.__file__)


# ---------------------------------------------------------------------------
# Doubles and builders
# ---------------------------------------------------------------------------
class FakeReader(MarketDataReader):
    """An in-memory reader that records its request and answers from a frame.

    Exactly one of ``frame`` (a literal answer), ``answer`` (any object, to test
    a non-frame) or ``error`` (to test propagation) is normally supplied.
    """

    def __init__(
        self,
        frame: pd.DataFrame | None = None,
        *,
        answer: object | None = None,
        error: Exception | None = None,
    ) -> None:
        self.frame = frame
        self.answer = answer
        self.error = error
        self.calls: list[dict[str, object]] = []

    def read_history(
        self,
        snapshot: object,
        start_date: object,
        required_fields: Sequence[str],
    ) -> pd.DataFrame:
        self.calls.append(
            {
                "snapshot": snapshot,
                "start_date": start_date,
                "required_fields": required_fields,
            }
        )
        if self.error is not None:
            raise self.error
        if self.answer is not None:
            return self.answer  # type: ignore[return-value]
        if self.frame is not None:
            return self.frame.copy(deep=True)
        return empty_prices()

    @property
    def request(self) -> dict[str, object]:
        """The single recorded request (fails if the reader ran != once)."""
        assert len(self.calls) == 1, f"read_history ran {len(self.calls)} time(s)"
        return self.calls[0]


def empty_prices() -> pd.DataFrame:
    """A correctly shaped answer that holds no bar at all."""
    return pd.DataFrame(
        {
            "date": pd.Series([], dtype="datetime64[s]"),
            "symbol": pd.Series([], dtype="object"),
            PRICE_FIELD: pd.Series([], dtype="float64"),
        }
    )


def price_frame(rows: Sequence[tuple[str, date, object]]) -> pd.DataFrame:
    """A reader frame built from ``(symbol, date, adjusted_close)`` triples."""
    return pd.DataFrame(
        [
            {"date": pd.Timestamp(day), "symbol": symbol, PRICE_FIELD: price}
            for symbol, day, price in rows
        ],
        columns=["date", "symbol", PRICE_FIELD],
    )


def ends(
    mapping: dict[str, tuple[float, float]],
    *,
    as_of: date = AS_OF,
    end: date = END_ONE,
) -> pd.DataFrame:
    """Two bars per symbol: ``{symbol: (price at as_of, price at end)}``."""
    rows: list[tuple[str, date, object]] = []
    for symbol, (start_price, end_price) in mapping.items():
        rows.append((symbol, as_of, start_price))
        rows.append((symbol, end, end_price))
    return price_frame(rows)


def default_answer(*, as_of: date = AS_OF, end: date = END_ONE) -> pd.DataFrame:
    """Every symbol at 100 on ``as_of`` and 110 on ``end`` (a 10% label)."""
    return ends({symbol: (100.0, 110.0) for symbol in SYMBOLS}, as_of=as_of, end=end)


def make_panel(
    *,
    symbols: Sequence[str] = SYMBOLS,
    values: Sequence[float] | None = None,
    as_of: date = AS_OF,
    name: str = NAME,
    version: str = VERSION,
) -> FactorPanel:
    """A valid factor panel whose pool, values, date and identity the test chooses."""
    universe = list(symbols)
    count = len(universe)
    frame = pd.DataFrame(
        {
            "as_of_date": [pd.Timestamp(as_of)] * count,
            "symbol": universe,
            "factor_name": [name] * count,
            "factor_version": [version] * count,
            "raw_value": list(values) if values is not None else [0.5] * count,
            "is_eligible": [True] * count,
        }
    )
    return FactorPanel(frame, universe=universe, as_of_date=as_of)


def make_calculator(
    frame: pd.DataFrame | None = None,
    *,
    answer: object | None = None,
    error: Exception | None = None,
) -> tuple[ForwardReturnCalculator, FakeReader]:
    """A calculator over a recording in-memory reader, plus that reader."""
    reader = FakeReader(frame, answer=answer, error=error)
    return ForwardReturnCalculator(reader), reader


def session_after(day: date, horizon: int) -> date:
    """The ``horizon``-th ``XNYS`` session after ``day``, straight from the exchange."""
    sessions = sessions_in_range(CALENDAR, day, CALENDAR.last_session.date())
    return sessions[horizon]


def drop_bar(frame: pd.DataFrame, symbol: str, day: date) -> pd.DataFrame:
    """``frame`` without the bar of ``symbol`` on ``day``."""
    match = (frame["symbol"] == symbol) & (frame["date"] == pd.Timestamp(day))
    return frame.drop(index=frame.index[match])


def set_price(frame: pd.DataFrame, symbol: str, day: date, value: object) -> pd.DataFrame:
    """``frame`` with one bar's ``adjusted_close`` replaced by ``value``."""
    changed = frame.copy(deep=True)
    changed[PRICE_FIELD] = changed[PRICE_FIELD].astype(object)
    changed.loc[
        (changed["symbol"] == symbol) & (changed["date"] == pd.Timestamp(day)), PRICE_FIELD
    ] = value
    return changed


def label_frame(symbols: Sequence[str] = SYMBOLS, **overrides: object) -> pd.DataFrame:
    """A valid one-session label table; a rejected construction bends one field."""
    universe = list(symbols)
    count = len(universe)
    default_return = [0.1, NAN, -0.05]
    default_available = [True, False, True]
    base: dict[str, object] = {
        "as_of_date": [pd.Timestamp(AS_OF)] * count,
        "end_date": [pd.Timestamp(END_ONE)] * count,
        "symbol": universe,
        "factor_name": [NAME] * count,
        "factor_version": [VERSION] * count,
        "horizon_sessions": [1] * count,
        "forward_return": [default_return[index % 3] for index in range(count)],
        "is_available": [default_available[index % 3] for index in range(count)],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def make_labels(**overrides: object) -> ForwardReturnPanel:
    """A valid label object; every rejected construction below bends one field."""
    fields: dict[str, object] = {
        "frame": label_frame(),
        "universe": list(SYMBOLS),
        "as_of_date": AS_OF,
        "end_date": END_ONE,
        "factor_name": NAME,
        "factor_version": VERSION,
        "horizon_sessions": 1,
    }
    fields.update(overrides)
    return ForwardReturnPanel(**fields)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# The frozen label object
# ---------------------------------------------------------------------------
def test_a_valid_label_table_exposes_its_identity_and_rows() -> None:
    """Every field names the one factor, session, horizon and pool it describes."""
    labels = make_labels()

    assert labels.as_of_date == AS_OF
    assert labels.end_date == END_ONE
    assert labels.factor_name == NAME
    assert labels.factor_version == VERSION
    assert labels.horizon_sessions == 1
    assert labels.universe == SYMBOLS
    assert labels.symbols == SYMBOLS
    assert len(labels) == 3

    frame = labels.frame
    assert list(frame.columns) == list(FORWARD_RETURN_COLUMNS)
    assert frame["symbol"].tolist() == list(SYMBOLS)
    assert frame["as_of_date"].dt.date.tolist() == [AS_OF] * 3
    assert frame["end_date"].dt.date.tolist() == [END_ONE] * 3
    assert frame["horizon_sessions"].tolist() == [1, 1, 1]
    assert frame["is_available"].tolist() == [True, False, True]
    values = frame["forward_return"].tolist()
    assert values[0] == 0.1
    assert math.isnan(values[1])
    assert values[2] == -0.05

    assert str(frame["as_of_date"].dtype) == "datetime64[s]"
    assert str(frame["end_date"].dtype) == "datetime64[s]"
    assert str(frame["horizon_sessions"].dtype) == "int64"
    assert str(frame["forward_return"].dtype) == "float64"
    assert str(frame["is_available"].dtype) == "bool"


def test_the_panel_holds_no_factor_value_or_trade_information() -> None:
    """A label table is labels only: no raw value, rank, IC, group or fill."""
    frame = make_labels().frame

    assert set(frame.columns) == set(FORWARD_RETURN_COLUMNS)
    for forbidden in ("raw_value", "rank", "ic", "group", "quantile", "weight", "fill_price"):
        assert forbidden not in frame.columns


def test_the_label_object_is_frozen() -> None:
    """A label object cannot be edited after it was validated."""
    labels = make_labels()

    with pytest.raises(FrozenInstanceError):
        labels.horizon_sessions = 2  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        labels.end_date = date(2024, 4, 2)  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        labels.universe = ("AAPL",)  # type: ignore[misc]

    assert labels.horizon_sessions == 1
    assert labels.universe == SYMBOLS


def test_the_frame_property_returns_an_independent_copy() -> None:
    """Mutating a returned frame changes neither the panel nor a later copy."""
    labels = make_labels()

    first = labels.frame
    second = labels.frame
    assert first is not second

    first.loc[0, "symbol"] = "TSLA"
    first.loc[0, "forward_return"] = 999.0
    first.loc[1, "is_available"] = True
    first.drop(index=2, inplace=True)

    assert labels.symbols == SYMBOLS
    assert len(labels) == 3
    assert labels.frame["symbol"].tolist() == list(SYMBOLS)
    assert labels.frame["forward_return"].tolist()[0] == 0.1
    assert labels.frame["is_available"].tolist() == [True, False, True]


def test_the_repr_summarises_the_labels_without_the_rows() -> None:
    """A log line names the window, the factor and the coverage, not the values."""
    description = repr(make_labels())

    assert description.startswith("ForwardReturnPanel(")
    assert "as_of_date='2024-03-28'" in description
    assert "end_date='2024-04-01'" in description
    assert f"factor={NAME}@{VERSION}" in description
    assert "horizon_sessions=1" in description
    assert "universe=3" in description
    assert "available=2" in description
    assert "0.1" not in description


# ---------------------------------------------------------------------------
# Every label table the contract must refuse
# ---------------------------------------------------------------------------
def test_a_row_order_that_is_not_the_universe_order_is_refused() -> None:
    """Row order is part of the contract: the labels follow ``universe``."""
    with pytest.raises(ForwardReturnError) as excinfo:
        make_labels(frame=label_frame(symbols=["MSFT", "AAPL", "NVDA"]))

    assert "row order must equal the universe order" in str(excinfo.value)
    assert "['MSFT', 'AAPL', 'NVDA']" in str(excinfo.value)


@pytest.mark.parametrize(
    ("symbols", "message"),
    [
        (["AAPL", "MSFT", "AMZN"], "outside the universe"),
        (["AAPL", "MSFT"], "must cover every symbol"),
        (["AAPL", "AAPL", "NVDA"], "repeats symbol"),
        (["AAPL", "msft", "NVDA"], "invalid value"),
        (["AAPL", "MSFT", "NVDA", "TSLA"], "row(s) for a universe"),
    ],
)
def test_a_universe_that_the_rows_do_not_cover_is_refused(symbols: list[str], message: str) -> None:
    """Exactly one row per universe symbol: no gap, no repeat, no stranger."""
    with pytest.raises(ForwardReturnError) as excinfo:
        make_labels(frame=label_frame(symbols=symbols))

    assert message in str(excinfo.value)


@pytest.mark.parametrize(
    "universe",
    [
        ["AAPL", "MSFT"],
        ["AAPL", "AAPL", "NVDA"],
        [],
        "AAPL",
        None,
        ["AAPL", "MSFT", "NVDA", "TSLA"],
    ],
)
def test_a_universe_that_is_not_a_clean_pool_is_refused(universe: object) -> None:
    """The universe is validated by the contract's own pool rules, not a new dialect."""
    with pytest.raises(ForwardReturnError):
        make_labels(universe=universe)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("as_of_date", date(2024, 3, 27)),
        ("end_date", date(2024, 4, 2)),
        ("factor_name", "value_1"),
        ("factor_version", "2.0.0"),
        ("horizon_sessions", 5),
    ],
)
def test_an_identity_that_disagrees_with_the_rows_is_refused(field: str, value: object) -> None:
    """The object's identity and its own columns must say the same thing."""
    with pytest.raises(ForwardReturnError) as excinfo:
        make_labels(**{field: value})

    assert f"does not match the label's {field}" in str(excinfo.value)


@pytest.mark.parametrize("horizon", [0, -1, -20, True, 1.0, "1", None])
def test_a_horizon_that_is_not_a_positive_integer_is_refused(horizon: object) -> None:
    """A horizon is a whole number of sessions: a flag, a float or text is not."""
    with pytest.raises(ForwardReturnError) as excinfo:
        make_labels(horizon_sessions=horizon)

    assert "positive integer number of sessions" in str(excinfo.value)


@pytest.mark.parametrize(
    "horizon_column",
    [[1.0, 1.0, 1.0], [0, 0, 0], [1, 2, 1], [True, True, True]],
)
def test_a_horizon_column_that_is_not_one_positive_integer_is_refused(
    horizon_column: list[object],
) -> None:
    """The column is checked too, not only the value passed to the constructor."""
    with pytest.raises(ForwardReturnError):
        make_labels(frame=label_frame(horizon_sessions=horizon_column))


@pytest.mark.parametrize(
    "end_date",
    [AS_OF, date(2024, 3, 1)],
)
def test_an_end_date_that_is_not_later_is_refused(end_date: date) -> None:
    """A forward return is measured to a later session, never to the same one."""
    with pytest.raises(ForwardReturnError) as excinfo:
        make_labels(frame=label_frame(end_date=[pd.Timestamp(end_date)] * 3))

    assert "must be after as_of_date" in str(excinfo.value)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("as_of_date", datetime(2024, 3, 28, 16, 0)),
        ("as_of_date", "2024-03-28"),
        ("as_of_date", None),
        ("end_date", datetime(2024, 4, 1)),
        ("end_date", "2024-04-01"),
    ],
)
def test_a_date_that_is_not_a_real_day_is_refused(field: str, value: object) -> None:
    """A label names a session: a timestamp or a string is not a validated day."""
    with pytest.raises(ForwardReturnError) as excinfo:
        make_labels(**{field: value})

    assert f"ForwardReturnPanel.{field} must be a datetime.date" in str(excinfo.value)


def test_unparseable_or_mixed_dates_in_the_rows_are_refused() -> None:
    """The rows carry real days: text that is not a date, and a mixed column, fail."""
    with pytest.raises(ForwardReturnError) as excinfo:
        make_labels(frame=label_frame(as_of_date=["not-a-date"] * 3))
    assert "missing or unparseable" in str(excinfo.value)

    with pytest.raises(ForwardReturnError) as excinfo:
        make_labels(
            frame=label_frame(
                as_of_date=[pd.Timestamp(AS_OF), pd.Timestamp("2024-03-27"), pd.Timestamp(AS_OF)]
            )
        )
    assert "must hold exactly one date" in str(excinfo.value)


@pytest.mark.parametrize("value", [NAN, INF, -INF])
def test_an_available_row_must_hold_a_finite_return(value: float) -> None:
    """``is_available=True`` promises a number a consumer may sum."""
    with pytest.raises(ForwardReturnError) as excinfo:
        make_labels(
            frame=label_frame(forward_return=[0.1, value, 0.2], is_available=[True, True, True])
        )

    assert "is_available=True" in str(excinfo.value)
    assert "MSFT" in str(excinfo.value)


@pytest.mark.parametrize("value", [0.0, -0.25, 100.0, INF, -INF])
def test_an_unavailable_row_must_hold_nan(value: float) -> None:
    """``is_available=False`` promises NaN -- never a number, never an infinity."""
    with pytest.raises(ForwardReturnError) as excinfo:
        make_labels(
            frame=label_frame(forward_return=[0.1, value, 0.2], is_available=[True, False, True])
        )

    assert "is_available=False" in str(excinfo.value)
    assert "MSFT" in str(excinfo.value)


def test_a_return_that_is_not_a_number_is_refused() -> None:
    """A label is a float (or NaN): a piece of text is not a return."""
    with pytest.raises(ForwardReturnError) as excinfo:
        make_labels(
            frame=label_frame(forward_return=["abc", NAN, NAN], is_available=[False, False, False])
        )

    assert "convertible to float" in str(excinfo.value)


@pytest.mark.parametrize(
    ("frame", "message"),
    [
        (lambda: label_frame().drop(columns=["is_available"]), "missing column(s)"),
        (lambda: label_frame().assign(rank=[1, 2, 3]), "unexpected column(s)"),
        (lambda: [[1, 2, 3]], "pandas DataFrame"),
        (lambda: label_frame(symbols=[]), "must not be empty"),
        (lambda: label_frame(is_available=[1, 0, 1]), "boolean column"),
        (
            lambda: label_frame(is_available=pd.array([True, None, True], dtype="boolean")),
            "must not be missing",
        ),
        (lambda: label_frame(factor_name=[NAME, "value_1", NAME]), "exactly one value"),
        (lambda: label_frame(factor_name=["Momentum"] * 3), "must be snake_case"),
        (lambda: label_frame(factor_version=["  "] * 3), "non-empty string"),
    ],
)
def test_a_frame_that_is_not_a_label_table_is_refused(frame: object, message: str) -> None:
    """Wrong columns, wrong dtypes and a foreign identity are all refused."""
    built = frame() if callable(frame) else frame

    with pytest.raises(ForwardReturnError) as excinfo:
        make_labels(frame=built)

    assert message in str(excinfo.value)


@pytest.mark.parametrize("name", ["Momentum", "Momentum-12", "", None, 5])
def test_a_factor_name_outside_the_contract_dialect_is_refused(name: object) -> None:
    """One name dialect for the whole system: the panel's, reused unchanged."""
    with pytest.raises(ForwardReturnError) as excinfo:
        make_labels(factor_name=name)

    assert "ForwardReturnPanel.factor_name" in str(excinfo.value)


@pytest.mark.parametrize("version", ["", "   ", None, 1])
def test_a_factor_version_that_is_not_a_non_empty_string_is_refused(version: object) -> None:
    """A version is a label a reader must be able to print and compare."""
    with pytest.raises(ForwardReturnError) as excinfo:
        make_labels(factor_version=version)

    assert "ForwardReturnPanel.factor_version" in str(excinfo.value)


def test_the_error_collects_every_problem_of_one_table() -> None:
    """One failed construction names all the ways the table is wrong."""
    frame = label_frame(symbols=["MSFT", "AAPL", "NVDA"])
    frame["factor_name"] = ["Momentum"] * 3

    with pytest.raises(ForwardReturnError) as excinfo:
        make_labels(frame=frame, horizon_sessions=0)

    problems = excinfo.value.problems
    assert len(problems) >= 3
    assert any("row order" in problem for problem in problems)
    assert any("horizon_sessions" in problem for problem in problems)
    assert any("snake_case" in problem for problem in problems)
    assert "is not a valid set of forward-return labels" in str(excinfo.value)
    for problem in problems:
        assert problem in str(excinfo.value)


# ---------------------------------------------------------------------------
# The calculation: sessions, prices and the read window
# ---------------------------------------------------------------------------
def test_the_label_is_the_close_to_close_adjusted_return() -> None:
    """``adjusted_close(end) / adjusted_close(as_of) - 1`` on the exact prices."""
    calculator, _ = make_calculator(
        ends({"AAPL": (100.0, 130.0), "MSFT": (250.0, 200.0), "NVDA": (80.5, 80.5)})
    )
    labels = calculator.compute(make_panel(), 1)

    frame = labels.frame
    assert frame["forward_return"].tolist() == [
        130.0 / 100.0 - 1.0,
        200.0 / 250.0 - 1.0,
        80.5 / 80.5 - 1.0,
    ]
    assert frame["is_available"].tolist() == [True, True, True]
    assert labels.as_of_date == AS_OF
    assert labels.end_date == END_ONE
    assert labels.horizon_sessions == 1


@pytest.mark.parametrize("horizon", [1, 5, 20])
def test_a_longer_horizon_measures_the_documented_pair(horizon: int) -> None:
    """The end price is the session ``horizon`` later, never the nearest one."""
    end = session_after(AS_OF, horizon)
    calculator, _ = make_calculator(ends({s: (50.0, 57.5) for s in SYMBOLS}, end=end))

    labels = calculator.compute(make_panel(), horizon)

    assert labels.horizon_sessions == horizon
    assert labels.end_date == end
    assert labels.frame["horizon_sessions"].tolist() == [horizon] * 3
    assert labels.frame["forward_return"].tolist() == [57.5 / 50.0 - 1.0] * 3


def test_the_horizon_is_counted_on_the_exchange_calendar() -> None:
    """Weekends and exchange holidays cost no session: the calendar decides."""
    expected = {
        (date(2024, 3, 28), 1): date(2024, 4, 1),  # Good Friday sits in between
        (date(2024, 3, 22), 1): date(2024, 3, 25),  # the weekend sits in between
        (date(2024, 3, 27), 1): date(2024, 3, 28),
        (date(2024, 3, 28), 5): date(2024, 4, 5),
        (date(2024, 1, 5), 20): date(2024, 2, 5),
    }
    for (as_of, horizon), end in expected.items():
        calculator, _ = make_calculator(
            ends({symbol: (100.0, 100.0) for symbol in SYMBOLS}, as_of=as_of, end=end)
        )

        labels = calculator.compute(make_panel(as_of=as_of), horizon)

        assert labels.end_date == end
        assert labels.end_date == session_after(as_of, horizon)


def test_horizon_one_is_the_next_session_not_the_same_day() -> None:
    """The label of ``h=1`` never collapses to a zero return on the decision day."""
    calculator, _ = make_calculator(ends({symbol: (100.0, 111.0) for symbol in SYMBOLS}))

    labels = calculator.compute(make_panel(), 1)

    assert labels.end_date == date(2024, 4, 1)
    assert labels.end_date > labels.as_of_date
    assert labels.frame["forward_return"].tolist() == [111.0 / 100.0 - 1.0] * 3


def test_the_reader_receives_exactly_the_label_window() -> None:
    """One read: the panel's universe, ``[as_of_date, end_date]``, one field."""
    panel = make_panel()
    calculator, reader = make_calculator(default_answer())

    labels = calculator.compute(panel, 1)

    request = reader.request
    snapshot = request["snapshot"]
    assert isinstance(snapshot, UniverseSnapshot)
    assert snapshot.symbols == panel.universe
    assert snapshot.as_of_date == labels.end_date == END_ONE
    assert "forward-return labels" in snapshot.source
    assert request["start_date"] == panel.as_of_date == AS_OF
    assert tuple(request["required_fields"]) == (PRICE_FIELD,)
    assert PRICE_FIELD == "adjusted_close"
    assert len(reader.calls) == 1


@pytest.mark.parametrize("horizon", [1, 5, 20])
def test_the_read_window_never_extends_past_the_end_date(horizon: int) -> None:
    """The label read is bounded by the horizon: no bar after ``end_date`` is asked for."""
    end = session_after(AS_OF, horizon)
    calculator, reader = make_calculator(
        ends({symbol: (100.0, 102.0) for symbol in SYMBOLS}, end=end)
    )

    labels = calculator.compute(make_panel(), horizon)

    request = reader.request
    snapshot = request["snapshot"]
    assert snapshot.as_of_date == end == labels.end_date
    assert request["start_date"] == AS_OF
    assert labels.frame["end_date"].dt.date.tolist() == [end] * 3


def test_columns_the_reader_adds_are_ignored() -> None:
    """A reader may volunteer more fields; only ``adjusted_close`` becomes a label."""
    frame = default_answer().assign(close=1.0, volume=1_000, adjusted=9.0)
    calculator, _ = make_calculator(frame)

    labels = calculator.compute(make_panel(), 1)

    assert labels.frame["forward_return"].tolist() == [110.0 / 100.0 - 1.0] * 3


# ---------------------------------------------------------------------------
# One symbol at a time: a gap is a missing label, never a guess
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "mutate",
    [
        lambda frame: drop_bar(frame, "MSFT", AS_OF),
        lambda frame: drop_bar(frame, "MSFT", END_ONE),
        lambda frame: set_price(frame, "MSFT", END_ONE, NAN),
        lambda frame: set_price(frame, "MSFT", AS_OF, NAN),
        lambda frame: set_price(frame, "MSFT", AS_OF, INF),
        lambda frame: set_price(frame, "MSFT", END_ONE, -INF),
        lambda frame: set_price(frame, "MSFT", AS_OF, 0.0),
        lambda frame: set_price(frame, "MSFT", AS_OF, -12.5),
        lambda frame: set_price(frame, "MSFT", END_ONE, "110.0"),
        lambda frame: set_price(frame, "MSFT", END_ONE, None),
    ],
)
def test_a_symbol_without_a_usable_pair_is_unavailable_alone(mutate: object) -> None:
    """A missing or unusable end leaves that label NaN and the others intact."""
    calculator, _ = make_calculator(mutate(default_answer()))  # type: ignore[operator]

    labels = calculator.compute(make_panel(), 1)

    frame = labels.frame
    assert frame["is_available"].tolist() == [True, False, True]
    values = frame["forward_return"].tolist()
    assert values[0] == 110.0 / 100.0 - 1.0
    assert math.isnan(values[1])
    assert values[2] == 110.0 / 100.0 - 1.0


def test_an_empty_answer_marks_every_symbol_unavailable() -> None:
    """No bars at all is an answer (nothing to label), never a failed computation."""
    calculator, _ = make_calculator()

    labels = calculator.compute(make_panel(), 1)

    assert labels.end_date == END_ONE
    assert labels.frame["is_available"].tolist() == [False, False, False]
    assert labels.frame["forward_return"].isna().all()
    assert len(labels) == 3


def test_no_neighbouring_close_is_borrowed() -> None:
    """A bar one session early is not the end date: no stale label is produced."""
    horizon = 5
    end = session_after(AS_OF, horizon)
    previous = session_after(AS_OF, horizon - 1)
    frame = price_frame([(symbol, AS_OF, 100.0) for symbol in SYMBOLS] + [("NVDA", previous, 99.0)])
    calculator, _ = make_calculator(frame)

    labels = calculator.compute(make_panel(), horizon)

    assert labels.frame["is_available"].tolist() == [False, False, False]
    assert labels.frame["forward_return"].isna().all()
    assert end not in frame["date"].dt.date.tolist()


def test_intermediate_sessions_may_be_missing() -> None:
    """Only the two ends of the window matter; the sessions between them may be absent."""
    horizon = 5
    end = session_after(AS_OF, horizon)
    calculator, _ = make_calculator(ends({symbol: (100.0, 104.0) for symbol in SYMBOLS}, end=end))

    labels = calculator.compute(make_panel(), horizon)

    assert labels.frame["forward_return"].tolist() == [104.0 / 100.0 - 1.0] * 3
    assert labels.frame["is_available"].tolist() == [True, True, True]


def test_a_return_outside_the_float_range_is_reported_missing() -> None:
    """A ratio of finite prices can still overflow; that is a gap, not an infinity."""
    calculator, _ = make_calculator(ends({symbol: (1e-308, 1e308) for symbol in SYMBOLS}))

    labels = calculator.compute(make_panel(), 1)

    assert labels.frame["is_available"].tolist() == [False, False, False]
    assert labels.frame["forward_return"].isna().all()


# ---------------------------------------------------------------------------
# The reader's answer cannot be trusted: refuse, never interpret
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("answer", "message"),
    [
        ([[1, 2, 3]], "pandas DataFrame"),
        ("bars", "pandas DataFrame"),
        (0, "pandas DataFrame"),
        (default_answer().drop(columns=["date"]), "missing required column"),
        (default_answer().drop(columns=["symbol"]), "missing required column"),
        (default_answer().drop(columns=[PRICE_FIELD]), "missing required column"),
        (
            pd.concat([default_answer(), default_answer().head(1)], ignore_index=True),
            "duplicate (symbol, date)",
        ),
        (
            price_frame([(symbol, AS_OF, 100.0) for symbol in SYMBOLS] + [("TSLA", END_ONE, 1.0)]),
            "outside the universe",
        ),
        (
            price_frame(
                [(symbol, AS_OF, 100.0) for symbol in SYMBOLS]
                + [(symbol, END_ONE, 110.0) for symbol in SYMBOLS]
                + [("AAPL", date(2024, 3, 27), 99.0)]
            ),
            "dated outside",
        ),
        (
            price_frame(
                [(symbol, AS_OF, 100.0) for symbol in SYMBOLS]
                + [(symbol, END_ONE, 110.0) for symbol in SYMBOLS]
                + [("AAPL", date(2024, 4, 2), 111.0)]
            ),
            "dated outside",
        ),
        (
            pd.DataFrame(
                {
                    "date": ["not-a-date", pd.Timestamp(END_ONE)],
                    "symbol": ["AAPL", "AAPL"],
                    PRICE_FIELD: [100.0, 110.0],
                }
            ),
            "missing or unparseable",
        ),
        (price_frame([("aapl", AS_OF, 100.0)]), "invalid value"),
    ],
)
def test_an_untrusted_reader_answer_is_refused(answer: object, message: str) -> None:
    """A malformed answer would make a label silently wrong, so it is refused."""
    calculator, _ = make_calculator(answer=answer)

    with pytest.raises(ForwardReturnError) as excinfo:
        calculator.compute(make_panel(), 1)

    assert message in str(excinfo.value)
    assert excinfo.value.problems


def test_a_read_error_is_propagated_unchanged() -> None:
    """The reader's own failure keeps its type: it is never re-wrapped here."""
    error = MarketDataReadError("the dataset cannot answer this window", ["no parquet file"])
    calculator, _ = make_calculator(error=error)

    with pytest.raises(MarketDataReadError) as excinfo:
        calculator.compute(make_panel(), 1)

    assert excinfo.value is error
    assert excinfo.value.problems == ["no parquet file"]


# ---------------------------------------------------------------------------
# Requests that cannot be honoured
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "panel",
    [None, "panel", pd.DataFrame({"symbol": ["AAPL"]}), [1, 2, 3]],
)
def test_compute_needs_a_validated_factor_panel(panel: object) -> None:
    """Labels describe a panel that exists: a raw frame is not one."""
    calculator, reader = make_calculator(default_answer())

    with pytest.raises(ForwardReturnError) as excinfo:
        calculator.compute(panel, 1)  # type: ignore[arg-type]

    assert "FactorPanel" in str(excinfo.value)
    assert reader.calls == []


@pytest.mark.parametrize("horizon", [0, -1, -20, True, 1.5, "5", None])
def test_compute_rejects_a_horizon_that_is_not_a_positive_integer(horizon: object) -> None:
    """The horizon is validated before the calendar or the reader is touched."""
    calculator, reader = make_calculator(default_answer())

    with pytest.raises(ForwardReturnError) as excinfo:
        calculator.compute(make_panel(), horizon)  # type: ignore[arg-type]

    assert "positive integer number of sessions" in str(excinfo.value)
    assert reader.calls == []


@pytest.mark.parametrize("as_of", [date(2024, 3, 29), date(2024, 3, 30)])
def test_a_weekend_or_a_holiday_is_not_a_decision_date(as_of: date) -> None:
    """Good Friday and Saturday are refused instead of being nudged to a session."""
    calculator, reader = make_calculator(
        ends({symbol: (100.0, 110.0) for symbol in SYMBOLS}, as_of=as_of)
    )

    with pytest.raises(ForwardReturnError) as excinfo:
        calculator.compute(make_panel(as_of=as_of), 1)

    assert "not a XNYS trading session" in str(excinfo.value)
    assert "never moved to a neighbouring one" in str(excinfo.value)
    assert reader.calls == []


def test_a_date_outside_the_calendar_is_refused() -> None:
    """A decision date the calendar does not cover has no sessions to count."""
    calculator, reader = make_calculator(
        ends({symbol: (100.0, 110.0) for symbol in SYMBOLS}, as_of=date(1990, 1, 2))
    )

    with pytest.raises(ForwardReturnError) as excinfo:
        calculator.compute(make_panel(as_of=date(1990, 1, 2)), 1)

    assert "outside the XNYS calendar" in str(excinfo.value)
    assert reader.calls == []


def test_the_calendar_must_cover_the_whole_horizon() -> None:
    """Asking past the last known session is refused, not answered with a guess."""
    last = CALENDAR.last_session.date()
    calculator, reader = make_calculator()

    with pytest.raises(ForwardReturnError) as excinfo:
        calculator.compute(make_panel(as_of=last), 1)

    assert "has no end date to measure to" in str(excinfo.value)
    assert reader.calls == []


@pytest.mark.parametrize("reader", [None, object(), "reader", pd.DataFrame()])
def test_the_calculator_needs_a_market_data_reader(reader: object) -> None:
    """Labels read bars through the reader interface; anything else is refused."""
    with pytest.raises(ForwardReturnError) as excinfo:
        ForwardReturnCalculator(reader)  # type: ignore[arg-type]

    assert "MarketDataReader" in str(excinfo.value)


@pytest.mark.parametrize("calendar_name", ["", "   ", None, 5])
def test_the_calculator_needs_a_usable_calendar_name(calendar_name: object) -> None:
    """A blank or non-text calendar name is a construction error, not a late one."""
    with pytest.raises(ForwardReturnError) as excinfo:
        ForwardReturnCalculator(FakeReader(), calendar_name=calendar_name)  # type: ignore[arg-type]

    assert "calendar_name" in str(excinfo.value)


def test_an_unknown_calendar_is_reported_with_its_cause() -> None:
    """A misspelled calendar fails at construction and keeps the library error."""
    with pytest.raises(ForwardReturnError) as excinfo:
        ForwardReturnCalculator(FakeReader(), calendar_name="NOT-A-CALENDAR")

    assert "could not load the trading calendar" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, CalendarError)


def test_the_calculator_exposes_its_dependencies() -> None:
    """The reader and calendar a calculator uses are inspectable, never hidden."""
    reader = FakeReader()
    calculator = ForwardReturnCalculator(reader)

    assert calculator.market_data_reader is reader
    assert calculator.calendar_name == "XNYS"
    assert calculator.calendar is CALENDAR
    assert repr(calculator) == (
        "ForwardReturnCalculator(calendar_name='XNYS', market_data_reader=FakeReader)"
    )


# ---------------------------------------------------------------------------
# Purity: the panel is read, never written; the labels ignore raw values
# ---------------------------------------------------------------------------
def test_the_panel_survives_a_computation_unchanged() -> None:
    """A call reads ``universe``, dates and identity -- and changes nothing."""
    panel = make_panel(values=[0.25, NAN, -1.0])
    before = panel.frame
    calculator, _ = make_calculator(default_answer())

    calculator.compute(panel, 1)

    pd.testing.assert_frame_equal(panel.frame, before)
    assert str(panel.frame["raw_value"].dtype) == "float64"


def test_the_labels_ignore_the_factor_values() -> None:
    """Same identity, same pool, different values: byte-identical labels."""
    calculator, _ = make_calculator(default_answer())
    left = make_panel(values=[0.25, NAN, 1e308])
    right = make_panel(values=[-0.75, INF, 0.0])

    first = calculator.compute(left, 1)
    second = calculator.compute(right, 1)

    pd.testing.assert_frame_equal(first.frame, second.frame)
    assert first.universe == second.universe


def test_the_labels_copy_the_panel_identity() -> None:
    """A label table belongs to the factor that produced the panel."""
    panel = make_panel(name="value_1", version="2.0.0")
    calculator, _ = make_calculator(default_answer())

    labels = calculator.compute(panel, 1)

    assert labels.factor_name == "value_1"
    assert labels.factor_version == "2.0.0"
    assert labels.frame["factor_name"].tolist() == ["value_1"] * 3
    assert labels.frame["factor_version"].tolist() == ["2.0.0"] * 3


def test_the_labels_follow_the_panel_universe_order() -> None:
    """Output order is the panel's pool order, not an alphabetical one."""
    symbols = ("NVDA", "AAPL", "MSFT")
    panel = make_panel(symbols=symbols)
    calculator, _ = make_calculator(ends({symbol: (100.0, 110.0) for symbol in symbols}))

    labels = calculator.compute(panel, 1)

    assert labels.universe == symbols
    assert labels.symbols == symbols
    assert labels.frame["symbol"].tolist() == list(symbols)


def test_the_same_call_always_answers_the_same_way() -> None:
    """No clock, no cache and no randomness: two calls agree exactly."""
    calculator, reader = make_calculator(default_answer())
    panel = make_panel()

    first = calculator.compute(panel, 1)
    second = calculator.compute(panel, 1)

    pd.testing.assert_frame_equal(first.frame, second.frame)
    assert len(reader.calls) == 2


def test_the_labels_only_ever_describe_the_one_window() -> None:
    """Every row is dated ``as_of_date``/``end_date`` and carries the one horizon."""
    calculator, _ = make_calculator(default_answer())

    labels = calculator.compute(make_panel(), 1)
    frame = labels.frame

    assert frame["as_of_date"].dt.date.tolist() == [AS_OF] * 3
    assert frame["end_date"].dt.date.tolist() == [END_ONE] * 3
    assert frame["horizon_sessions"].tolist() == [1] * 3
    assert frame["end_date"].max().date() == labels.end_date


# ---------------------------------------------------------------------------
# The architectural boundary of the module
# ---------------------------------------------------------------------------
#: Exactly the modules a forward-return label module may depend on.
ALLOWED_IMPORTS: frozenset[str] = frozenset(
    {
        "__future__",
        "collections.abc",
        "dataclasses",
        "datetime",
        "math",
        "numbers",
        "re",
        "typing",
        "pandas",
        "data_sys.errors",
        "data_sys.market_calendar",
        "data_sys.schema",
        "factor_sys.contracts",
        "factor_sys.errors",
        "factor_sys.market_data",
        "factor_sys.universe",
    }
)

#: Layers the label module must not reach for: storage, execution, judgement,
#: columnar engines and the outside world (files, network, clock, environment).
FORBIDDEN_IMPORT_PREFIXES: tuple[str, ...] = (
    "factors",
    "factor_sys.factor_store",
    "factor_sys.runner",
    "factor_sys.quick_analysis",
    "factor_sys.registry",
    "factor_sys.quality",
    "data_sys.storage",
    "data_sys.providers",
    "data_sys.update",
    "data_sys.batch_query",
    "data_sys.config",
    "data_sys.duckdb_verify",
    "pyarrow",
    "duckdb",
    "numpy",
    "requests",
    "httpx",
    "urllib",
    "socket",
    "yfinance",
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

#: Functions that would mean the module loads, computes or persists something.
FORBIDDEN_CALLS: frozenset[str] = frozenset(
    {
        "__import__",
        "eval",
        "exec",
        "open",
        "read_csv",
        "read_excel",
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

#: Objects belonging to the factor path this module must never call into.
FORBIDDEN_NAMES: frozenset[str] = frozenset(
    {
        "Factor",
        "FactorContext",
        "FactorRunner",
        "FactorStore",
        "ParquetFactorStore",
        "FactorRegistry",
        "QuickAnalyzer",
        "FactorQualityChecker",
        "DefaultFactorQualityChecker",
        "os",
        "Path",
        "pa",
        "duckdb",
        "np",
    }
)

#: The project root, for the "the factor path knows nothing about labels" test.
PROJECT_ROOT: Path = Path(factor_sys.__file__).parent.parent


def module_tree() -> ast.Module:
    """The parsed source of :mod:`factor_sys.forward_returns`."""
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
    """Contracts, errors, the reader, the snapshot and the calendar -- nothing else."""
    imports = module_imports(module_tree())

    assert imports <= ALLOWED_IMPORTS, sorted(imports - ALLOWED_IMPORTS)


def test_the_module_imports_nothing_from_the_factor_path_or_the_outside_world() -> None:
    """The runner, the store, a factor formula, a file or a network call: none."""
    imports = module_imports(module_tree())

    for name in sorted(imports):
        for prefix in FORBIDDEN_IMPORT_PREFIXES:
            assert not matches(name, prefix), f"{name} must not be imported"


def test_the_module_does_not_import_the_package_root() -> None:
    """Importing ``factor_sys`` would drag the store and the runner in."""
    assert "factor_sys" not in module_imports(module_tree())


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


def test_the_module_mentions_no_object_of_the_factor_path() -> None:
    """No runner, context, formula, store, registry or analyzer is even named."""
    tree = module_tree()
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}

    assert not names & FORBIDDEN_NAMES
    assert not attributes & FORBIDDEN_NAMES


def test_the_module_never_reads_the_factor_columns() -> None:
    """``raw_value``, ``is_eligible`` and the panel frame are not read here."""
    tree = module_tree()
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}

    assert "raw_value" not in attributes
    assert "is_eligible" not in attributes
    assert "frame" not in attributes


#: Files of the factor path this module must leave completely alone.
FACTOR_PATH_FILES: tuple[Path, ...] = (
    PROJECT_ROOT / "factor_sys" / "runner.py",
    PROJECT_ROOT / "factor_sys" / "quick_analysis.py",
    *sorted((PROJECT_ROOT / "factors").glob("*.py")),
)


def test_the_factor_path_never_carries_a_forward_return() -> None:
    """The runner and every formula know nothing about labels."""
    assert FACTOR_PATH_FILES

    for path in FACTOR_PATH_FILES:
        text = path.read_text(encoding="utf-8")
        assert "forward_return" not in text, path.name
        assert "ForwardReturnPanel" not in text, path.name
        assert "ForwardReturnCalculator" not in text, path.name
        assert "ForwardReturnError" not in text, path.name


# ---------------------------------------------------------------------------
# The error type and the exported API
# ---------------------------------------------------------------------------
def test_the_forward_return_error_is_a_collecting_factor_error() -> None:
    """One domain error, carrying every problem the way the contract does."""
    error = ForwardReturnError("cannot label", ["first", "second"])

    assert isinstance(error, FactorError)
    assert error.problems == ["first", "second"]
    assert "cannot label" in str(error)
    assert "first" in str(error)
    assert "second" in str(error)
    assert ForwardReturnError("just a message").problems == []


def test_the_package_root_exports_the_forward_return_api() -> None:
    """``from factor_sys import ForwardReturnCalculator, ...`` is supported."""
    assert factor_sys.ForwardReturnCalculator is ForwardReturnCalculator
    assert factor_sys.ForwardReturnPanel is ForwardReturnPanel
    assert factor_sys.ForwardReturnError is ForwardReturnError
    assert factor_sys.FORWARD_RETURN_COLUMNS == FORWARD_RETURN_COLUMNS
    assert forward_returns_module.ForwardReturnPanel is ForwardReturnPanel
    assert forward_returns_module.ForwardReturnError is ForwardReturnError
    assert issubclass(factor_sys.ForwardReturnError, factor_sys.FactorError)

    for name in (
        "FORWARD_RETURN_COLUMNS",
        "ForwardReturnCalculator",
        "ForwardReturnPanel",
        "ForwardReturnError",
    ):
        assert name in factor_sys.__all__

    assert set(forward_returns_module.__all__) == {
        "FORWARD_RETURN_COLUMNS",
        "PRICE_FIELD",
        "ForwardReturnCalculator",
        "ForwardReturnError",
        "ForwardReturnPanel",
    }
    assert PRICE_FIELD == "adjusted_close"


def test_a_reader_may_answer_with_iso_days() -> None:
    """A custom reader may spell days as text; the label table still reads them."""
    frame = default_answer()
    frame["date"] = frame["date"].dt.strftime("%Y-%m-%d")
    calculator, _ = make_calculator(frame)

    labels = calculator.compute(make_panel(), 1)

    assert labels.frame["forward_return"].tolist() == [110.0 / 100.0 - 1.0] * 3
    assert str(labels.frame["as_of_date"].dtype) == "datetime64[s]"
