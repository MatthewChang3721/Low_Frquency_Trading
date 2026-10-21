"""Tests for the single-factor, single-date runner (:mod:`factor_sys.runner`).

Everything here is in memory and offline: the universe provider, the market-data
reader and the factor are doubles that *record* what they were asked, so a test
can assert not only what the runner returned but in which order it asked, and
whether a failure happened before or after the next step.  No ``data/``
directory is read, no Parquet file is written, nothing touches the network and
neither DuckDB nor PyArrow is imported.

Covered in this order:

* the pipeline itself: provider -> reader -> context -> factor -> panel, with the
  exact arguments each step received and the panel returned untouched,
* the runner as an interface-only object: another provider and another reader
  drop in without the runner changing,
* the read window: ``lookback_sessions`` counts the sessions *before*
  ``as_of_date``, taken from the real NYSE calendar (weekends, Good Friday, July
  4th, Martin Luther King Day) and clamped to the calendar's first session,
* history the runner refuses to judge: a short window, an empty frame,
* every failure: a non-session decision date, a read error, a context contract
  violation, a non-panel answer, a panel from another run, an unrelated factor
  exception, unusable dependencies and a blank/unknown calendar,
* the architectural boundary of the module and its exports.
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Iterable, Sequence
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import factor_sys
import factor_sys.runner as runner_module
from data_sys.errors import CalendarError, DataPipelineError
from data_sys.market_calendar import calendar_bounds, get_calendar, sessions_in_range
from factor_sys import (
    Factor,
    FactorContext,
    FactorContractError,
    FactorError,
    FactorPanel,
    FactorRunError,
    FactorRunner,
    FactorSpec,
    MarketDataReader,
    MarketDataReadError,
    StaticUniverseProvider,
    UniverseProvider,
    UniverseSnapshot,
)
from tests.helpers import write_universe_file

#: The pool every in-memory provider answers with (a snapshot may ask for fewer).
SYMBOLS: tuple[str, ...] = ("AAPL", "MSFT", "NVDA")

#: A real XNYS session (a Friday) that most tests decide on.
AS_OF: date = date(2024, 1, 5)

#: The real calendar, for expectations that must come from the exchange itself.
CALENDAR = get_calendar("XNYS")

#: A pool wide enough for :class:`StaticUniverseProvider` (>= 10 rows).
POOL: tuple[str, ...] = (
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "META",
    "TSLA",
    "AVGO",
    "COST",
    "JPM",
    "V",
    "ORCL",
)


# ---------------------------------------------------------------------------
# Doubles: they record, they never read a file
# ---------------------------------------------------------------------------
class Recorder:
    """A shared, ordered call log, so a test can assert *when* each step ran."""

    def __init__(self) -> None:
        self.events: list[tuple[str, Any]] = []

    def record(self, event: str, detail: Any = None) -> None:
        """Append ``event`` (with an optional ``detail``) to the log."""
        self.events.append((event, detail))

    @property
    def names(self) -> list[str]:
        """The recorded event names, in call order."""
        return [name for name, _ in self.events]

    def detail(self, event: str) -> Any:
        """The single detail recorded for ``event`` (fails if it ran != once)."""
        details = [detail for name, detail in self.events if name == event]
        assert len(details) == 1, f"{event!r} was called {len(details)} time(s)"
        return details[0]


class RecordingProvider(UniverseProvider):
    """A universe provider that records the date it was asked about.

    ``answer`` short-circuits the snapshot (for misbehaving providers) and
    ``as_of`` answers for a *different* date on purpose.
    """

    def __init__(
        self,
        recorder: Recorder,
        symbols: Sequence[str] = SYMBOLS,
        *,
        answer: object | None = None,
        as_of: date | str | None = None,
    ) -> None:
        self.recorder = recorder
        self.symbols = tuple(symbols)
        self.answer = answer
        self.as_of = as_of

    def snapshot(self, as_of_date: date | str) -> UniverseSnapshot:
        self.recorder.record("snapshot", as_of_date)
        if self.answer is not None:
            return self.answer  # type: ignore[return-value]
        return UniverseSnapshot(
            as_of_date=as_of_date if self.as_of is None else self.as_of,
            symbols=self.symbols,
            source="recording provider",
        )


class RecordingReader(MarketDataReader):
    """A reader that records its request and answers from memory.

    Exactly one of ``frame`` (a literal answer), ``answer`` (any object, to test
    a non-frame) or ``error`` (to test propagation) is normally supplied; the
    default is a one-session frame for the snapshot's own date.
    """

    def __init__(
        self,
        recorder: Recorder,
        frame: pd.DataFrame | None = None,
        *,
        answer: object | None = None,
        error: Exception | None = None,
    ) -> None:
        self.recorder = recorder
        self.frame = frame
        self.answer = answer
        self.error = error

    def read_history(
        self,
        snapshot: UniverseSnapshot,
        start_date: date,
        required_fields: Sequence[str],
    ) -> pd.DataFrame:
        self.recorder.record(
            "read_history",
            {
                "snapshot": snapshot,
                "start_date": start_date,
                "required_fields": required_fields,
            },
        )
        if self.error is not None:
            raise self.error
        if self.answer is not None:
            return self.answer  # type: ignore[return-value]
        if self.frame is not None:
            return self.frame.copy(deep=True)
        return bars_for(snapshot.symbols, required_fields, sessions=(snapshot.as_of_date,))


class RecordingFactor(Factor):
    """A factor whose whole behaviour the test supplies; it only records."""

    def __init__(
        self,
        recorder: Recorder,
        *,
        spec: FactorSpec | None = None,
        name: str = "momentum_12_1",
        version: str = "1.0.0",
        required_fields: Sequence[str] = ("close",),
        lookback_sessions: int = 0,
        respond: Callable[[FactorContext], object] | None = None,
    ) -> None:
        self.recorder = recorder
        self._spec = (
            spec
            if spec is not None
            else FactorSpec(
                name=name,
                version=version,
                required_fields=required_fields,
                lookback_sessions=lookback_sessions,
            )
        )
        self.respond = respond

    @property
    def spec(self) -> FactorSpec:
        return self._spec

    def compute(self, context: FactorContext) -> FactorPanel:
        self.recorder.record("compute", context)
        if self.respond is not None:
            return self.respond(context)  # type: ignore[return-value]
        return panel_for(context, self._spec)


class LiteralReader(MarketDataReader):
    """The smallest legal reader: one method, no recording, no file."""

    def read_history(
        self,
        snapshot: UniverseSnapshot,
        start_date: date,
        required_fields: Sequence[str],
    ) -> pd.DataFrame:
        return bars_for(snapshot.symbols, required_fields, sessions=(snapshot.as_of_date,))


class LiteralFactor(Factor):
    """The smallest legal factor: one constant value, everything eligible."""

    spec = FactorSpec(name="constant_one", version="0.1.0", required_fields=("close",))

    def compute(self, context: FactorContext) -> FactorPanel:
        return panel_for(context, self.spec, values=[1.0] * len(context.universe))


# ---------------------------------------------------------------------------
# Builders for the two frames a run touches
# ---------------------------------------------------------------------------
def bars_for(
    symbols: Sequence[str] = SYMBOLS,
    fields: Iterable[str] = ("close",),
    *,
    sessions: Sequence[date] = (AS_OF,),
) -> pd.DataFrame:
    """A tiny long bars frame: one row per (symbol, session), deterministic values."""
    names = tuple(fields)
    return pd.DataFrame(
        [
            {
                "date": pd.Timestamp(session),
                "symbol": symbol,
                **{name: 100.0 + position for name in names},
            }
            for symbol in symbols
            for position, session in enumerate(sessions)
        ],
        columns=["date", "symbol", *names],
    )


def empty_bars(fields: Iterable[str] = ("close",)) -> pd.DataFrame:
    """A correctly shaped but completely empty bars frame."""
    names = tuple(fields)
    return pd.DataFrame(
        {
            "date": pd.Series(dtype="datetime64[s]"),
            "symbol": pd.Series(dtype="object"),
            **{name: pd.Series(dtype="float64") for name in names},
        }
    )


def panel_for(
    context: FactorContext,
    spec: FactorSpec,
    *,
    values: Sequence[float] | None = None,
    eligible: Sequence[bool] | None = None,
    universe: Sequence[str] | None = None,
    name: str | None = None,
    version: str | None = None,
    as_of: date | str | None = None,
) -> FactorPanel:
    """A panel for ``context`` -- or a deliberately mismatched one."""
    symbols = list(context.universe if universe is None else universe)
    count = len(symbols)
    decision = as_of if as_of is not None else context.as_of_date
    frame = pd.DataFrame(
        {
            "as_of_date": [pd.Timestamp(decision)] * count,
            "symbol": symbols,
            "factor_name": [name if name is not None else spec.name] * count,
            "factor_version": [version if version is not None else spec.version] * count,
            "raw_value": list(values) if values is not None else [0.5] * count,
            "is_eligible": list(eligible) if eligible is not None else [True] * count,
        }
    )
    return FactorPanel(frame, universe=symbols, as_of_date=decision)


def nth_session_before(as_of: date, count: int) -> date:
    """The session ``count`` sessions before ``as_of``, straight from the exchange.

    Deliberately *not* ``last_sessions``: the tests check the runner's answer
    against the calendar's own session index instead of against its own helper.
    """
    window = CALENDAR.sessions_in_range(
        pd.Timestamp(as_of) - pd.Timedelta(days=2 * count + 40),
        pd.Timestamp(as_of),
    )
    return window[-(count + 1)].date()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def recorder() -> Recorder:
    """A fresh, empty call log shared by the doubles of one test."""
    return Recorder()


@pytest.fixture
def provider(recorder: Recorder) -> RecordingProvider:
    """A well-behaved universe provider over :data:`SYMBOLS`."""
    return RecordingProvider(recorder)


@pytest.fixture
def reader(recorder: Recorder) -> RecordingReader:
    """A well-behaved reader answering one session of bars for :data:`SYMBOLS`."""
    return RecordingReader(recorder)


@pytest.fixture
def runner(provider: RecordingProvider, reader: RecordingReader) -> FactorRunner:
    """The runner under test, wired to the two recording doubles."""
    return FactorRunner(provider, reader)


#: The spec the recording factor declares by default.
SPEC = FactorSpec(name="momentum_12_1", version="1.0.0", required_fields=("close",))


# ---------------------------------------------------------------------------
# The pipeline: provider -> reader -> context -> factor -> panel
# ---------------------------------------------------------------------------
def test_a_run_walks_the_pipeline_in_a_fixed_order(
    runner: FactorRunner,
    recorder: Recorder,
) -> None:
    """Each step runs exactly once, in the documented order, and a panel comes back."""
    panel = runner.run(RecordingFactor(recorder, lookback_sessions=2), AS_OF)

    assert recorder.names == ["snapshot", "read_history", "compute"]
    assert recorder.detail("snapshot") == AS_OF
    assert isinstance(panel, FactorPanel)
    assert panel.as_of_date == AS_OF
    assert panel.universe == SYMBOLS


def test_the_reader_gets_the_spec_fields_and_the_computed_window(
    runner: FactorRunner,
    recorder: Recorder,
) -> None:
    """``required_fields`` and ``start_date`` come from the spec, not from luck."""
    factor = RecordingFactor(recorder, required_fields=("volume", "close"), lookback_sessions=3)

    runner.run(factor, AS_OF)

    request = recorder.detail("read_history")
    assert request["required_fields"] == ("volume", "close")
    assert request["start_date"] == nth_session_before(AS_OF, 3)
    assert request["snapshot"].as_of_date == AS_OF
    assert request["snapshot"].symbols == SYMBOLS


def test_the_factor_gets_a_context_built_from_the_snapshot_and_the_bars(
    recorder: Recorder,
) -> None:
    """The context is the runner's own construction, not the reader's own frame."""
    bars = bars_for(sessions=(date(2024, 1, 4), AS_OF))
    runner = FactorRunner(RecordingProvider(recorder), RecordingReader(recorder, frame=bars))

    runner.run(RecordingFactor(recorder), AS_OF)

    context = recorder.detail("compute")
    assert isinstance(context, FactorContext)
    assert context.as_of_date == AS_OF
    assert context.universe == SYMBOLS
    assert context.bars["symbol"].tolist() == bars["symbol"].tolist()
    assert context.bars["close"].tolist() == bars["close"].tolist()


def test_the_panel_comes_back_untouched(runner: FactorRunner, recorder: Recorder) -> None:
    """The runner validates a panel; it never rewrites values, flags or row order."""
    returned: list[FactorPanel] = []

    def respond(context: FactorContext) -> FactorPanel:
        panel = panel_for(
            context,
            SPEC,
            values=[0.25, float("nan"), -3.5],
            eligible=[True, False, True],
        )
        returned.append(panel)
        return panel

    panel = runner.run(RecordingFactor(recorder, respond=respond), AS_OF)

    assert panel is returned[0]
    rows = panel.frame
    assert rows["symbol"].tolist() == list(SYMBOLS)
    assert rows["raw_value"].iloc[0] == 0.25
    assert pd.isna(rows["raw_value"].iloc[1])
    assert rows["raw_value"].iloc[2] == -3.5
    assert rows["is_eligible"].tolist() == [True, False, True]


@pytest.mark.parametrize("value", [AS_OF, "2024-01-05", datetime(2024, 1, 5, 21, 0)])
def test_the_decision_date_is_normalized_to_a_plain_date(
    runner: FactorRunner,
    recorder: Recorder,
    value: date | str | datetime,
) -> None:
    """A date, an ISO string and a datetime all decide on the same session."""
    panel = runner.run(RecordingFactor(recorder), value)

    assert recorder.detail("snapshot") == AS_OF
    assert panel.as_of_date == AS_OF


def test_one_runner_serves_many_runs(runner: FactorRunner, recorder: Recorder) -> None:
    """The runner keeps no state: the same instance runs more work afterwards."""
    first = runner.run(RecordingFactor(recorder), AS_OF)
    second = runner.run(
        RecordingFactor(recorder, name="value_ep", lookback_sessions=1),
        date(2024, 1, 8),
    )

    assert recorder.names == ["snapshot", "read_history", "compute"] * 2
    assert (first.factor_name, second.factor_name) == ("momentum_12_1", "value_ep")
    assert (first.as_of_date, second.as_of_date) == (AS_OF, date(2024, 1, 8))


def test_the_runner_exposes_its_wiring(
    provider: RecordingProvider,
    reader: RecordingReader,
    runner: FactorRunner,
) -> None:
    """The defaults are visible, and the calendar is resolved once."""
    assert runner.universe_provider is provider
    assert runner.market_data_reader is reader
    assert runner.calendar_name == "XNYS"
    assert runner.calendar is get_calendar("XNYS")
    assert repr(runner) == (
        "FactorRunner(calendar_name='XNYS', universe_provider=RecordingProvider, "
        "market_data_reader=RecordingReader)"
    )


# ---------------------------------------------------------------------------
# The runner only needs the interfaces
# ---------------------------------------------------------------------------
def test_other_implementations_drop_in_unchanged(tmp_path: Path) -> None:
    """A different provider and a different reader run through the same runner."""
    path = write_universe_file(tmp_path / "universe.csv", POOL)

    runner = FactorRunner(StaticUniverseProvider(path), LiteralReader())
    panel = runner.run(LiteralFactor(), AS_OF)

    assert panel.factor_name == "constant_one"
    assert panel.factor_version == "0.1.0"
    assert panel.universe == tuple(sorted(POOL))
    rows = panel.frame
    assert rows["raw_value"].tolist() == [1.0] * len(POOL)
    assert rows["is_eligible"].all()


def test_the_runner_is_bound_to_no_implementation(runner: FactorRunner) -> None:
    """Nothing about the runner mentions a concrete provider, reader or factor."""
    assert isinstance(runner.universe_provider, UniverseProvider)
    assert isinstance(runner.market_data_reader, MarketDataReader)
    assert [base.__name__ for base in type(runner).__mro__] == ["FactorRunner", "object"]


# ---------------------------------------------------------------------------
# The read window: lookback_sessions counts the sessions *before* as_of_date
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("lookback", "expected"),
    [
        (0, "2024-01-05"),
        (1, "2024-01-04"),
        (2, "2024-01-03"),
        (3, "2024-01-02"),
        (4, "2023-12-29"),  # New Year's Day plus a weekend sit in between
        (5, "2023-12-28"),
    ],
)
def test_the_window_starts_the_sessions_before_the_decision_date(
    runner: FactorRunner,
    recorder: Recorder,
    lookback: int,
    expected: str,
) -> None:
    """The start date is real NYSE sessions, counted back from ``as_of_date``."""
    runner.run(RecordingFactor(recorder, lookback_sessions=lookback), AS_OF)

    assert recorder.detail("read_history")["start_date"] == date.fromisoformat(expected)


def test_lookback_zero_reads_the_decision_date_only(
    runner: FactorRunner,
    recorder: Recorder,
) -> None:
    """``lookback_sessions=0`` means one visible session: ``as_of_date`` itself."""
    runner.run(RecordingFactor(recorder, lookback_sessions=0), AS_OF)

    assert recorder.detail("read_history")["start_date"] == AS_OF


@pytest.mark.parametrize("lookback", [0, 2, 21, 252])
def test_the_window_is_never_calendar_day_arithmetic(
    runner: FactorRunner,
    recorder: Recorder,
    lookback: int,
) -> None:
    """The window holds ``lookback`` sessions plus the decision date -- not days."""
    runner.run(RecordingFactor(recorder, lookback_sessions=lookback), AS_OF)

    start = recorder.detail("read_history")["start_date"]
    assert start == nth_session_before(AS_OF, lookback)
    assert start <= AS_OF
    # The visible sessions are exactly the decision date and its predecessors.
    assert len(sessions_in_range(CALENDAR, start, AS_OF)) == lookback + 1
    if lookback:
        assert start < AS_OF


def test_a_year_of_sessions_is_more_than_a_year_of_days(
    runner: FactorRunner,
    recorder: Recorder,
) -> None:
    """252 sessions means ~352 calendar days: subtracting days would be wrong."""
    runner.run(RecordingFactor(recorder, lookback_sessions=252), AS_OF)

    start = recorder.detail("read_history")["start_date"]
    assert (AS_OF - start).days > 252
    assert start != AS_OF - timedelta(days=252)
    assert CALENDAR.is_session(pd.Timestamp(start))


@pytest.mark.parametrize(
    ("as_of", "lookback", "expected"),
    [
        ("2024-04-01", 1, "2024-03-28"),  # the Monday after Good Friday
        ("2024-04-02", 4, "2024-03-26"),  # Good Friday inside the window
        ("2024-07-08", 2, "2024-07-03"),  # Independence Day 2024-07-04
        ("2024-01-08", 1, "2024-01-05"),  # the Monday after a weekend
        ("2024-01-16", 1, "2024-01-12"),  # Martin Luther King Day 2024-01-15
    ],
)
def test_exchange_holidays_and_weekends_are_skipped(
    runner: FactorRunner,
    recorder: Recorder,
    as_of: str,
    lookback: int,
    expected: str,
) -> None:
    """Every start date is a session the exchange was actually open on."""
    runner.run(RecordingFactor(recorder, lookback_sessions=lookback), date.fromisoformat(as_of))

    start = recorder.detail("read_history")["start_date"]
    assert start == date.fromisoformat(expected)
    assert get_calendar("XNYS").is_session(pd.Timestamp(start))


def test_good_friday_is_not_a_calendar_day(runner: FactorRunner, recorder: Recorder) -> None:
    """One session before Monday 2024-04-01 is Thursday, never the 29th or the 31st."""
    runner.run(RecordingFactor(recorder, lookback_sessions=1), date(2024, 4, 1))

    start = recorder.detail("read_history")["start_date"]
    assert start == date(2024, 3, 28)
    assert start != date(2024, 3, 29)  # Good Friday: the exchange was shut
    assert start != date(2024, 3, 31)  # Sunday
    assert start != date(2024, 4, 1) - timedelta(days=1)


def test_a_short_calendar_history_uses_its_first_session(recorder: Recorder) -> None:
    """A calendar that begins late yields its earliest session, not an error."""
    first, _ = calendar_bounds(CALENDAR)
    early = sessions_in_range(CALENDAR, first, first + timedelta(days=14))[2]
    runner = FactorRunner(RecordingProvider(recorder), RecordingReader(recorder))

    panel = runner.run(RecordingFactor(recorder, lookback_sessions=252), early)

    assert recorder.detail("read_history")["start_date"] == first
    assert recorder.names == ["snapshot", "read_history", "compute"]
    assert panel.as_of_date == early


# ---------------------------------------------------------------------------
# History the runner refuses to judge
# ---------------------------------------------------------------------------
def test_a_short_history_still_reaches_the_factor(recorder: Recorder) -> None:
    """Two sessions for a 21-session lookback: the factor decides, not the runner."""
    short = bars_for(sessions=(date(2024, 1, 4), AS_OF))
    runner = FactorRunner(RecordingProvider(recorder), RecordingReader(recorder, frame=short))

    def respond(context: FactorContext) -> FactorPanel:
        counts = context.bars.groupby("symbol").size()
        eligible = [int(counts.get(symbol, 0)) >= 3 for symbol in context.universe]
        values = [1.0 if ok else float("nan") for ok in eligible]
        return panel_for(context, SPEC, values=values, eligible=eligible)

    panel = runner.run(RecordingFactor(recorder, lookback_sessions=21, respond=respond), AS_OF)

    assert recorder.detail("read_history")["start_date"] == nth_session_before(AS_OF, 21)
    rows = panel.frame
    assert rows["is_eligible"].tolist() == [False, False, False]
    assert rows["raw_value"].isna().all()


def test_a_missing_symbol_is_neither_filled_nor_dropped(recorder: Recorder) -> None:
    """The context keeps exactly the rows the reader returned, and no more."""
    partial = bars_for(SYMBOLS[:2])
    runner = FactorRunner(RecordingProvider(recorder), RecordingReader(recorder, frame=partial))

    seen: list[FactorContext] = []

    def respond(context: FactorContext) -> FactorPanel:
        seen.append(context)
        eligible = [symbol in set(context.bars["symbol"]) for symbol in context.universe]
        return panel_for(context, SPEC, eligible=eligible)

    panel = runner.run(RecordingFactor(recorder, respond=respond), AS_OF)

    assert seen[0].bars["symbol"].tolist() == ["AAPL", "MSFT"]
    assert len(seen[0].bars) == 2
    assert panel.universe == SYMBOLS
    assert panel.frame["is_eligible"].tolist() == [True, True, False]


def test_an_empty_frame_is_passed_through_to_the_factor(recorder: Recorder) -> None:
    """An empty answer is a factor's problem, not an orchestration error."""
    reader = RecordingReader(recorder, frame=empty_bars())
    runner = FactorRunner(RecordingProvider(recorder), reader)

    def respond(context: FactorContext) -> FactorPanel:
        assert context.bars.empty
        return panel_for(
            context,
            SPEC,
            values=[float("nan")] * len(context.universe),
            eligible=[False] * len(context.universe),
        )

    panel = runner.run(RecordingFactor(recorder, respond=respond), AS_OF)

    rows = panel.frame
    assert rows["symbol"].tolist() == list(SYMBOLS)
    assert not rows["is_eligible"].any()
    assert rows["raw_value"].isna().all()


# ---------------------------------------------------------------------------
# Failures that belong to the orchestration
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("as_of", "expected"),
    [
        ("2024-01-06", "is not a XNYS trading session"),  # Saturday
        ("2024-01-07", "is not a XNYS trading session"),  # Sunday
        ("2024-03-29", "is not a XNYS trading session"),  # Good Friday
        ("2024-01-15", "is not a XNYS trading session"),  # Martin Luther King Day
        ("2024-07-04", "is not a XNYS trading session"),  # Independence Day
        ("1990-01-02", "lies outside the XNYS calendar"),  # before the first session
    ],
)
def test_a_non_session_decision_date_fails_before_anything_is_read(
    runner: FactorRunner,
    recorder: Recorder,
    as_of: str,
    expected: str,
) -> None:
    """The reader and the factor are never asked; the date is never moved."""
    with pytest.raises(FactorRunError) as excinfo:
        runner.run(RecordingFactor(recorder), as_of)

    assert expected in str(excinfo.value)
    assert recorder.names == ["snapshot"]


def test_a_date_after_the_calendar_ends_is_refused(recorder: Recorder) -> None:
    """A decision date the calendar does not cover is an orchestration failure."""
    _, last = calendar_bounds(CALENDAR)
    runner = FactorRunner(RecordingProvider(recorder), RecordingReader(recorder))

    with pytest.raises(FactorRunError) as excinfo:
        runner.run(RecordingFactor(recorder), last + timedelta(days=1))

    assert "lies outside the XNYS calendar" in str(excinfo.value)
    assert recorder.names == ["snapshot"]


def test_the_decision_date_is_never_moved_to_a_neighbouring_session(
    runner: FactorRunner,
    recorder: Recorder,
) -> None:
    """A holiday is refused instead of being traded as the day before or after."""
    with pytest.raises(FactorRunError) as excinfo:
        runner.run(RecordingFactor(recorder), date(2024, 3, 29))

    assert "never moves it to a neighbouring one" in str(excinfo.value)
    assert recorder.names == ["snapshot"]


# ---------------------------------------------------------------------------
# Failures that belong to the layer underneath: propagated unchanged
# ---------------------------------------------------------------------------
def test_a_read_error_propagates_unchanged(recorder: Recorder) -> None:
    """The reader's own failure keeps its type, its data and its identity."""
    error = MarketDataReadError("bars root 'data/standardized/market_bars' does not exist")
    runner = FactorRunner(RecordingProvider(recorder), RecordingReader(recorder, error=error))

    with pytest.raises(MarketDataReadError) as excinfo:
        runner.run(RecordingFactor(recorder), AS_OF)

    assert excinfo.value is error
    assert recorder.names == ["snapshot", "read_history"]


@pytest.mark.parametrize(
    ("frame", "expected"),
    [
        (bars_for(sessions=(AS_OF, date(2024, 1, 8))), "dated after as_of_date"),
        (bars_for(SYMBOLS).assign(symbol="TSLA"), "outside the universe"),
        (pd.concat([bars_for(), bars_for()]), "duplicate (symbol, date)"),
        (bars_for().drop(columns=["symbol"]), "missing required column"),
    ],
)
def test_a_context_contract_violation_propagates_unchanged(
    recorder: Recorder,
    frame: pd.DataFrame,
    expected: str,
) -> None:
    """The no-look-ahead guard is the context's; the runner only passes it on."""
    runner = FactorRunner(RecordingProvider(recorder), RecordingReader(recorder, frame=frame))

    with pytest.raises(FactorContractError) as excinfo:
        runner.run(RecordingFactor(recorder), AS_OF)

    assert expected in str(excinfo.value)
    assert "compute" not in recorder.names


def test_a_non_frame_answer_is_a_context_violation(recorder: Recorder) -> None:
    """A reader that answers with something else fails in the contract, once."""
    runner = FactorRunner(RecordingProvider(recorder), RecordingReader(recorder, answer="bars"))

    with pytest.raises(FactorContractError) as excinfo:
        runner.run(RecordingFactor(recorder), AS_OF)

    assert "must be a pandas DataFrame" in str(excinfo.value)
    assert "compute" not in recorder.names


# ---------------------------------------------------------------------------
# The panel must belong to this run
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("answer", [None, object(), "panel", pd.DataFrame()])
def test_a_factor_must_return_a_factor_panel(recorder: Recorder, answer: object) -> None:
    """An unvalidated frame (or anything else) never leaves the runner."""
    runner = FactorRunner(RecordingProvider(recorder), RecordingReader(recorder))

    def respond(context: FactorContext) -> object:
        return answer

    with pytest.raises(FactorRunError) as excinfo:
        runner.run(RecordingFactor(recorder, respond=respond), AS_OF)

    assert "must return a FactorPanel" in str(excinfo.value)


@pytest.mark.parametrize(
    ("respond", "expected"),
    [
        (
            lambda context: panel_for(context, SPEC, as_of=date(2024, 1, 4)),
            "as_of_date is 2024-01-04",
        ),
        (
            lambda context: panel_for(context, SPEC, universe=SYMBOLS[:2]),
            "missing ['NVDA']",
        ),
        (
            lambda context: panel_for(context, SPEC, name="value_ep"),
            "factor_name is 'value_ep'",
        ),
        (
            lambda context: panel_for(context, SPEC, version="9.9.9"),
            "factor_version is '9.9.9'",
        ),
    ],
)
def test_a_panel_from_another_run_is_refused(
    recorder: Recorder,
    respond: Callable[[FactorContext], FactorPanel],
    expected: str,
) -> None:
    """A panel that is valid on its own terms is still refused for this run."""
    runner = FactorRunner(RecordingProvider(recorder), RecordingReader(recorder))

    with pytest.raises(FactorRunError) as excinfo:
        runner.run(RecordingFactor(recorder, respond=respond), AS_OF)

    assert len(excinfo.value.problems) == 1
    assert expected in excinfo.value.problems[0]
    assert expected in str(excinfo.value)


def test_a_panel_with_the_universe_in_another_order_is_refused(recorder: Recorder) -> None:
    """Same symbols, different order: still not the panel this run asked for."""
    runner = FactorRunner(RecordingProvider(recorder), RecordingReader(recorder))

    def respond(context: FactorContext) -> FactorPanel:
        return panel_for(context, SPEC, universe=list(reversed(context.universe)))

    with pytest.raises(FactorRunError) as excinfo:
        runner.run(RecordingFactor(recorder, respond=respond), AS_OF)

    assert "different order" in excinfo.value.problems[0]


def test_every_mismatch_is_reported_at_once(recorder: Recorder) -> None:
    """One failed run names all the ways its panel did not fit."""
    runner = FactorRunner(RecordingProvider(recorder), RecordingReader(recorder))

    def respond(context: FactorContext) -> FactorPanel:
        return panel_for(context, SPEC, as_of=date(2024, 1, 4), name="value_ep")

    with pytest.raises(FactorRunError) as excinfo:
        runner.run(RecordingFactor(recorder, respond=respond), AS_OF)

    assert len(excinfo.value.problems) == 2
    assert "as_of_date is 2024-01-04" in excinfo.value.problems[0]
    assert "factor_name is 'value_ep'" in excinfo.value.problems[1]


# ---------------------------------------------------------------------------
# A factor's own failures: wrapped only when they are not ours
# ---------------------------------------------------------------------------
class ExplodingFactor(Factor):
    """A factor with a bug in it: an ordinary exception, not a domain error."""

    def __init__(self, recorder: Recorder) -> None:
        self.recorder = recorder

    @property
    def spec(self) -> FactorSpec:
        return SPEC

    def compute(self, context: FactorContext) -> FactorPanel:
        self.recorder.record("compute", context)
        raise ValueError("boom")


def test_an_unrelated_factor_failure_is_wrapped(runner: FactorRunner, recorder: Recorder) -> None:
    """A formula's bug becomes a domain error, with the original cause kept."""
    with pytest.raises(FactorRunError) as excinfo:
        runner.run(ExplodingFactor(recorder), AS_OF)

    assert isinstance(excinfo.value.__cause__, ValueError)
    assert "ValueError" in str(excinfo.value)
    assert "boom" in str(excinfo.value)
    assert "momentum_12_1" in str(excinfo.value)
    assert recorder.names == ["snapshot", "read_history", "compute"]


@pytest.mark.parametrize(
    "error",
    [
        FactorContractError("the factor built an invalid panel"),
        FactorRunError("a nested orchestration failure"),
        MarketDataReadError("a read performed inside the factor"),
    ],
)
def test_a_factor_system_error_is_never_re_wrapped(
    recorder: Recorder,
    error: FactorError,
) -> None:
    """Callers must still catch the precise error a factor raised."""
    runner = FactorRunner(RecordingProvider(recorder), RecordingReader(recorder))

    def respond(context: FactorContext) -> FactorPanel:
        raise error

    with pytest.raises(type(error)) as excinfo:
        runner.run(RecordingFactor(recorder, respond=respond), AS_OF)

    assert excinfo.value is error


# ---------------------------------------------------------------------------
# Dependencies that do not implement their interface
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("value", [None, object(), "provider", RecordingProvider, pd.DataFrame()])
def test_a_provider_that_is_not_a_universe_provider_is_refused(
    reader: RecordingReader,
    value: object,
) -> None:
    """Construction is where a wrong dependency is caught, not the first run."""
    with pytest.raises(FactorRunError) as excinfo:
        FactorRunner(value, reader)  # type: ignore[arg-type]

    assert "universe_provider must be a UniverseProvider" in str(excinfo.value)


@pytest.mark.parametrize("value", [None, object(), "reader", RecordingReader, pd.DataFrame()])
def test_a_reader_that_is_not_a_market_data_reader_is_refused(
    provider: RecordingProvider,
    value: object,
) -> None:
    """The runner only knows the reader interface, so anything else is refused."""
    with pytest.raises(FactorRunError) as excinfo:
        FactorRunner(provider, value)  # type: ignore[arg-type]

    assert "market_data_reader must be a MarketDataReader" in str(excinfo.value)


class ForeignSpecFactor(Factor):
    """A factor that declares something which is not a :class:`FactorSpec`."""

    spec = "momentum_12_1"

    def compute(self, context: FactorContext) -> FactorPanel:  # pragma: no cover
        raise AssertionError("compute must never run without a valid spec")


@pytest.mark.parametrize("value", [None, object(), lambda context: None, pd.DataFrame()])
def test_a_factor_that_is_not_a_factor_is_refused(
    runner: FactorRunner,
    recorder: Recorder,
    value: object,
) -> None:
    """A bare callable is not runnable: it has no spec to read fields from."""
    with pytest.raises(FactorRunError) as excinfo:
        runner.run(value, AS_OF)  # type: ignore[arg-type]

    assert "factor must be a Factor" in str(excinfo.value)
    assert recorder.names == []


def test_a_spec_that_is_not_a_factor_spec_is_refused(
    runner: FactorRunner,
    recorder: Recorder,
) -> None:
    """The declaration is checked before the provider or the reader is asked."""
    with pytest.raises(FactorRunError) as excinfo:
        runner.run(ForeignSpecFactor(), AS_OF)

    assert "spec must be a FactorSpec" in str(excinfo.value)
    assert recorder.names == []


@pytest.mark.parametrize("value", ["05/01/2024", "", None, 20240105, date])
def test_an_unusable_decision_date_is_refused(
    runner: FactorRunner,
    recorder: Recorder,
    value: object,
) -> None:
    """Only a date or an ISO-8601 string can name a decision date."""
    with pytest.raises(FactorRunError) as excinfo:
        runner.run(RecordingFactor(recorder), value)  # type: ignore[arg-type]

    assert "as_of_date must be a date" in str(excinfo.value)
    assert recorder.names == []


@pytest.mark.parametrize("answer", [42, ("AAPL", "MSFT"), pd.DataFrame(), object()])
def test_a_provider_must_return_a_universe_snapshot(recorder: Recorder, answer: object) -> None:
    """Anything else is an orchestration failure, not a mysterious attribute error."""
    provider = RecordingProvider(recorder, answer=answer)
    runner = FactorRunner(provider, RecordingReader(recorder))

    with pytest.raises(FactorRunError) as excinfo:
        runner.run(RecordingFactor(recorder), AS_OF)

    assert "must return a UniverseSnapshot" in str(excinfo.value)
    assert recorder.names == ["snapshot"]


def test_a_provider_must_echo_the_decision_date(recorder: Recorder) -> None:
    """A provider that quietly moves the date would move the whole factor."""
    provider = RecordingProvider(recorder, as_of=date(2024, 1, 4))
    runner = FactorRunner(provider, RecordingReader(recorder))

    with pytest.raises(FactorRunError) as excinfo:
        runner.run(RecordingFactor(recorder), AS_OF)

    assert "answered for 2024-01-04" in str(excinfo.value)
    assert recorder.names == ["snapshot"]


# ---------------------------------------------------------------------------
# The calendar is part of the wiring
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["", "   ", None, 42])
def test_a_blank_calendar_name_is_refused(
    provider: RecordingProvider,
    reader: RecordingReader,
    name: object,
) -> None:
    """A blank calendar would silently mean "no session at all"."""
    with pytest.raises(FactorRunError) as excinfo:
        FactorRunner(provider, reader, calendar_name=name)  # type: ignore[arg-type]

    assert "calendar_name must be a non-empty trading calendar name" in str(excinfo.value)


def test_an_unknown_calendar_name_is_refused(
    provider: RecordingProvider,
    reader: RecordingReader,
) -> None:
    """A misspelled calendar fails when the runner is built, with its cause kept."""
    with pytest.raises(FactorRunError) as excinfo:
        FactorRunner(provider, reader, calendar_name="NOPE")

    assert isinstance(excinfo.value.__cause__, CalendarError)
    assert "NOPE" in str(excinfo.value)


def test_the_calendar_name_is_trimmed(provider: RecordingProvider, reader: RecordingReader) -> None:
    """``"  XNYS  "`` is the same calendar as ``"XNYS"``."""
    runner = FactorRunner(provider, reader, calendar_name="  XNYS  ")

    assert runner.calendar_name == "XNYS"
    assert runner.calendar is get_calendar("XNYS")


# ---------------------------------------------------------------------------
# The module's own boundary
# ---------------------------------------------------------------------------
#: Modules ``runner.py`` may import: its own contract and interfaces, the errors
#: they raise, the trading calendar -- and the standard library it needs.
ALLOWED_IMPORTS: frozenset[str] = frozenset(
    {
        "__future__",
        "datetime",
        "typing",
        "data_sys.errors",
        "data_sys.market_calendar",
        "factor_sys.contracts",
        "factor_sys.errors",
        "factor_sys.market_data",
        "factor_sys.universe",
    }
)

#: Text that must never appear in the module: storage, network, columnar engines.
FORBIDDEN_TEXT: tuple[str, ...] = (
    "pandas",
    "pyarrow",
    "duckdb",
    "requests",
    "httpx",
    "socket",
    "yfinance",
    "data_sys.storage",
    "data_sys.providers",
    "data_sys.update",
    "data_sys.batch_query",
    "read_parquet",
    "to_parquet",
    "urlopen",
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


def test_the_runner_imports_only_its_own_layers() -> None:
    """No storage, no providers, no Parquet, no network: interfaces only."""
    path = Path(runner_module.__file__)

    assert imported_modules(path) <= ALLOWED_IMPORTS
    source = path.read_text(encoding="utf-8").lower()
    assert [text for text in FORBIDDEN_TEXT if text in source] == []


def test_the_runner_never_opens_or_prints_anything() -> None:
    """The module calls no builtin that reads, writes or prints."""
    tree = ast.parse(Path(runner_module.__file__).read_text(encoding="utf-8"))
    called = {ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}

    assert called & {"open", "print", "input", "eval", "exec", "compile"} == set()


def test_the_runner_and_its_error_are_exported() -> None:
    """The package root is the one place to import them from."""
    assert factor_sys.FactorRunner is FactorRunner
    assert factor_sys.FactorRunError is FactorRunError
    assert {"FactorRunner", "FactorRunError"} <= set(factor_sys.__all__)
    assert runner_module.__all__ == ["FactorRunner"]


def test_factor_run_error_is_a_factor_system_error_only() -> None:
    """It is catchable with ``FactorError`` -- and never with a data-system error."""
    assert issubclass(FactorRunError, FactorError)
    assert not issubclass(FactorRunError, DataPipelineError)

    error = FactorRunError("the panel does not belong to this run", ["one", "two"])
    assert error.problems == ["one", "two"]
    assert "one" in str(error)
    assert "two" in str(error)
    assert FactorRunError("a plain failure").problems == []
