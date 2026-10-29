"""Tests for the single-date strategy runner.

Everything here is in memory and offline: a fake ``UniverseProvider`` that
serves a ``UniverseSnapshot`` built from a tuple of tickers, a fake
``FactorStore`` that serves ``FactorPanel`` objects from a dictionary, and
strategies that are either the real ``TopNByFactorStrategy`` or a recording
double.  No ``data/`` directory, no Parquet file, no network call and no trading
calendar is involved anywhere.

The order follows one run:

* the constructor: what it accepts, what it exposes, what it touches;
* a successful run: the calls made, the object returned and its content;
* the read order of a multi-factor strategy;
* the decision date: every accepted form and every refused one;
* a provider that answers with the wrong thing or the wrong date;
* a store that answers with the wrong thing, or with its own error;
* the panel checks ``StrategyContext`` performs on the runner's behalf;
* a strategy that returns something other than a ``StrategyResult``;
* a result that belongs to another strategy, date or universe;
* how an exception from a strategy travels;
* the absence of side effects, and the module's architectural boundary.
"""

from __future__ import annotations

import ast
import inspect
from collections.abc import Callable, Mapping
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import strategy_sys
import strategy_sys.runner as runner_module
from factor_sys import FactorPanel, UniverseSnapshot
from factor_sys.errors import FactorStoreError
from factor_sys.factor_store import FactorStore
from factor_sys.universe import UniverseProvider
from strategy_sys import (
    FactorRequirement,
    Strategy,
    StrategyContext,
    StrategyContractError,
    StrategyError,
    StrategyRegistryError,
    StrategyResult,
    StrategyRunError,
    StrategyRunner,
    StrategySpec,
)
from strategy_sys.strategies import TopNByFactorStrategy

# ---------------------------------------------------------------------------
# The fixtures every test builds on: three tickers, two sessions, two factors
# ---------------------------------------------------------------------------
SYMBOLS: tuple[str, ...] = ("AAPL", "MSFT", "NVDA")
AS_OF: date = date(2024, 1, 5)
NEXT_DAY: date = date(2024, 1, 8)

MOMENTUM = FactorRequirement(factor_name="momentum_12_1", factor_version="1.0.0")
VALUE = FactorRequirement(factor_name="book_to_price", factor_version="2.0.0")


def make_universe(
    symbols: tuple[str, ...] = SYMBOLS,
    as_of: date = AS_OF,
) -> UniverseSnapshot:
    """A snapshot of ``symbols`` for one session."""
    return UniverseSnapshot(
        as_of_date=as_of,
        symbols=tuple(symbols),
        source="tests/fake_universe",
    )


def make_panel(
    requirement: FactorRequirement = MOMENTUM,
    *,
    symbols: tuple[str, ...] = SYMBOLS,
    raw_value: tuple[float, ...] = (0.20, 0.50, 0.10),
    as_of: date = AS_OF,
    is_eligible: tuple[bool, ...] | None = None,
) -> FactorPanel:
    """A valid panel of one factor for one date and one universe."""
    eligible = is_eligible if is_eligible is not None else (True,) * len(symbols)
    frame = pd.DataFrame(
        {
            "as_of_date": [pd.Timestamp(as_of)] * len(symbols),
            "symbol": list(symbols),
            "factor_name": [requirement.factor_name] * len(symbols),
            "factor_version": [requirement.factor_version] * len(symbols),
            "raw_value": list(raw_value),
            "is_eligible": list(eligible),
        }
    )
    return FactorPanel(frame, universe=tuple(symbols), as_of_date=as_of)


def make_spec(
    *,
    name: str = "blend_momentum_value",
    version: str = "1.0.0",
    required_factors: tuple[FactorRequirement, ...] = (MOMENTUM,),
) -> StrategySpec:
    """The identity a strategy double declares."""
    return StrategySpec(name=name, version=version, required_factors=required_factors)


def make_result(
    spec: StrategySpec,
    *,
    universe: UniverseSnapshot | None = None,
    as_of: date = AS_OF,
    symbols: tuple[str, ...] = SYMBOLS,
    scores: tuple[float, ...] = (0.20, 0.50, 0.10),
    ranks: tuple[int, ...] = (2, 1, 3),
    selected: tuple[bool, ...] = (True, True, False),
) -> StrategyResult:
    """A valid ``StrategyResult`` naming ``spec``, as a strategy would build it."""
    frame = pd.DataFrame(
        {
            "as_of_date": [pd.Timestamp(as_of)] * len(symbols),
            "strategy_name": [spec.name] * len(symbols),
            "strategy_version": [spec.version] * len(symbols),
            "symbol": list(symbols),
            "score": list(scores),
            "rank": list(ranks),
            "is_selected": list(selected),
        }
    )
    return StrategyResult(
        frame,
        universe=universe if universe is not None else make_universe(symbols, as_of),
        strategy_spec=spec,
        as_of_date=as_of,
    )


def result_from_context(spec: StrategySpec) -> Callable[[StrategyContext], StrategyResult]:
    """A builder that answers with a result valid for whatever context it is given."""

    def build(context: StrategyContext) -> StrategyResult:
        return make_result(spec, universe=context.universe, as_of=context.as_of_date)

    return build


# ---------------------------------------------------------------------------
# The doubles a run is wired to
# ---------------------------------------------------------------------------
class RecordingProvider(UniverseProvider):
    """A ``UniverseProvider`` that answers from memory and remembers the dates.

    ``answer`` is returned verbatim, whatever it is, so a provider that breaks
    its own interface can be tested; ``fixed`` replaces the snapshot the
    provider would otherwise build for the date it was asked about, so a
    provider that answers for another session can be tested.
    """

    def __init__(
        self,
        *,
        answer: Any = None,
        fixed: UniverseSnapshot | None = None,
        symbols: tuple[str, ...] = SYMBOLS,
    ) -> None:
        self.calls: list[date] = []
        self.snapshots: list[Any] = []
        self._answer = answer
        self._fixed = fixed
        self._symbols = tuple(symbols)

    def snapshot(self, as_of_date: date) -> Any:
        """Record the question and answer it."""
        self.calls.append(as_of_date)
        if self._answer is not None:
            return self._answer
        snapshot = (
            self._fixed if self._fixed is not None else make_universe(self._symbols, as_of_date)
        )
        self.snapshots.append(snapshot)
        return snapshot


class RecordingStore(FactorStore):
    """A ``FactorStore`` that serves panels from a dictionary and records reads.

    Every method the runner is *not* allowed to call raises ``AssertionError``
    and is recorded in ``forbidden`` instead of succeeding, so "the runner never
    probes ``exists`` and never writes" is checked by a normal successful run
    rather than by inspecting the runner's source.
    """

    def __init__(
        self,
        panels: Mapping[tuple[str, str, date], Any] | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self.panels = dict(panels or {})
        self.error = error
        self.reads: list[tuple[str, str, date]] = []
        self.forbidden: list[str] = []

    def read(self, factor_name: str, factor_version: str, as_of_date: date | str) -> Any:
        """Record the exact key and serve the panel stored under it."""
        self.reads.append((factor_name, factor_version, as_of_date))
        if self.error is not None:
            raise self.error
        key = (factor_name, factor_version, as_of_date)
        if key not in self.panels:
            raise FactorStoreError(
                f"no panel is stored for {factor_name}/{factor_version} on {as_of_date.isoformat()}"
            )
        return self.panels[key]

    def write(self, panel: FactorPanel) -> Path:
        """Never called: the runner stores nothing."""
        self.forbidden.append("write")
        raise AssertionError("the runner must never write a panel")

    def read_range(self, *args: Any, **kwargs: Any) -> tuple[FactorPanel, ...]:
        """Never called: a run is one date, not a range."""
        self.forbidden.append("read_range")
        raise AssertionError("the runner must never read a range")

    def exists(self, *args: Any, **kwargs: Any) -> bool:
        """Never called: a missing panel is an error to report, not a branch."""
        self.forbidden.append("exists")
        raise AssertionError("the runner must call read(), never exists()")

    def path_for(self, *args: Any, **kwargs: Any) -> Path:
        """Never called: the runner does not know a storage layout."""
        self.forbidden.append("path_for")
        raise AssertionError("the runner must never ask for a path")


class FixedStrategy(Strategy):
    """A strategy double: a declared spec, a prepared answer, a recalled context."""

    def __init__(
        self,
        spec: StrategySpec,
        build_result: Callable[[StrategyContext], Any] | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self._spec = spec
        self._build_result = build_result
        self._error = error
        self.contexts: list[StrategyContext] = []

    @property
    def spec(self) -> StrategySpec:
        """The identity this double declares."""
        return self._spec

    def compute(self, context: StrategyContext) -> Any:
        """Record the context, then raise the prepared error or return the answer."""
        self.contexts.append(context)
        if self._error is not None:
            raise self._error
        if self._build_result is None:
            raise AssertionError("this strategy double has no prepared result")
        return self._build_result(context)


class DelegatingStrategy(Strategy):
    """A strategy that forwards ``compute`` to a real one and keeps its result.

    The runner promises to return the *same* object the strategy produced, which
    can only be checked against an object the test can still reach; wrapping the
    real ``TopNByFactorStrategy`` keeps both the real scoring and the identity.
    """

    def __init__(self, inner: Strategy) -> None:
        self._inner = inner
        self.results: list[Any] = []

    @property
    def spec(self) -> StrategySpec:
        """The wrapped strategy's spec, unchanged."""
        return self._inner.spec

    def compute(self, context: StrategyContext) -> Any:
        """Delegate, remember the object, return it untouched."""
        result = self._inner.compute(context)
        self.results.append(result)
        return result


class MislabelledResult(StrategyResult):
    """A result whose declared factor requirements are not its spec's.

    ``StrategyResult`` derives ``factor_requirements`` from ``strategy_spec``, so
    the runner's fourth runtime check can only be exercised by an object that
    reports different requirements -- exactly the shape a future result variant
    would have.  Everything else about the result stays valid.
    """

    @property
    def factor_requirements(self) -> tuple[FactorRequirement, ...]:
        """Requirements that do not match the spec carried next to them."""
        return (VALUE,)


def make_runner(
    store: RecordingStore | None = None,
    provider: RecordingProvider | None = None,
) -> tuple[StrategyRunner, RecordingProvider, RecordingStore]:
    """A runner wired to fakes, plus the fakes themselves for inspection."""
    provider = RecordingProvider() if provider is None else provider
    store = RecordingStore() if store is None else store
    runner = StrategyRunner(universe_provider=provider, factor_store=store)
    return runner, provider, store


def one_factor_store(
    *,
    as_of: date = AS_OF,
    raw_value: tuple[float, ...] = (0.20, 0.50, 0.10),
) -> RecordingStore:
    """A store holding this session's momentum panel, and nothing else."""
    store = RecordingStore()
    store.panels[(MOMENTUM.factor_name, MOMENTUM.factor_version, as_of)] = make_panel(
        MOMENTUM, as_of=as_of, raw_value=raw_value
    )
    return store


# ---------------------------------------------------------------------------
# The constructor: two interfaces, nothing else, no I/O
# ---------------------------------------------------------------------------
def test_the_runner_refuses_a_universe_provider_that_is_not_one() -> None:
    assert not isinstance(object(), UniverseProvider)
    with pytest.raises(StrategyRunError, match="universe_provider must be a UniverseProvider"):
        StrategyRunner(universe_provider=object(), factor_store=RecordingStore())


def test_the_runner_refuses_a_factor_store_that_is_not_one() -> None:
    assert not isinstance(object(), FactorStore)
    with pytest.raises(StrategyRunError, match="factor_store must be a FactorStore"):
        StrategyRunner(universe_provider=RecordingProvider(), factor_store=object())


def test_the_runner_holds_exactly_the_two_dependencies() -> None:
    provider = RecordingProvider()
    store = RecordingStore()
    runner = StrategyRunner(provider, store)

    assert runner.universe_provider is provider
    assert runner.factor_store is store
    # No registry, no calendar, no result store and no frequency: none of those
    # is a runner's business, and an attribute for one would be the first step
    # towards the runner choosing a strategy.
    for absent in (
        "registry",
        "strategy_registry",
        "create",
        "calendar",
        "market_calendar",
        "frequency",
        "schedule",
        "results",
        "store",
        "run_batch",
    ):
        assert not hasattr(runner, absent)


def test_the_repr_names_both_dependencies() -> None:
    runner, _, _ = make_runner()
    assert repr(runner) == (
        "StrategyRunner(universe_provider=RecordingProvider, factor_store=RecordingStore)"
    )


def test_construction_reads_nothing_and_writes_nothing() -> None:
    provider = RecordingProvider()
    store = RecordingStore()
    StrategyRunner(universe_provider=provider, factor_store=store)

    assert provider.calls == []
    assert store.reads == []
    assert store.forbidden == []


# ---------------------------------------------------------------------------
# One successful run: one factor, the real Top N rule
# ---------------------------------------------------------------------------
def test_one_factor_run_end_to_end() -> None:
    """A real TopNByFactorStrategy scoring the panel a fake store serves."""
    strategy = DelegatingStrategy(TopNByFactorStrategy(MOMENTUM, 2))
    runner, provider, store = make_runner(one_factor_store())

    result = runner.run(strategy, as_of_date="2024-01-05")

    # the store was asked for exactly one key: this factor, this version, this
    # date -- and for a date, not for a timestamp
    assert store.reads == [("momentum_12_1", "1.0.0", AS_OF)]
    assert isinstance(store.reads[0][2], date)
    assert not isinstance(store.reads[0][2], datetime)
    # the provider decided the universe exactly once, for the date it was given
    assert provider.calls == [AS_OF]
    # the runner handed back the strategy's own object, not a copy of it
    assert result is strategy.results[0]
    # and that object carries the real rule's scoring
    assert result.selected_symbols == ("AAPL", "MSFT")
    rows = result.frame.to_dict("records")
    assert [(row["symbol"], int(row["rank"]), bool(row["is_selected"])) for row in rows] == [
        ("AAPL", 2, True),
        ("MSFT", 1, True),
        ("NVDA", 3, False),
    ]
    # provenance: the exact spec, the exact factor version, the exact pool
    assert result.strategy_spec is strategy.spec
    assert result.strategy_spec.name == "top_2_by_momentum_12_1"
    assert result.factor_requirements == (MOMENTUM,)
    assert result.as_of_date == AS_OF
    assert result.universe is provider.snapshots[0]
    # and the runner never probed the store or wrote to it
    assert store.forbidden == []


def test_a_second_strategy_runs_on_the_same_runner() -> None:
    """The runner is stateless: which rule runs is the caller's argument."""
    runner, _, store = make_runner(one_factor_store())
    first = DelegatingStrategy(TopNByFactorStrategy(MOMENTUM, 1))
    second = DelegatingStrategy(TopNByFactorStrategy(MOMENTUM, 2))

    assert runner.run(first, AS_OF).selected_symbols == ("MSFT",)
    assert runner.run(second, AS_OF).selected_symbols == ("AAPL", "MSFT")
    assert len(store.reads) == 2


# ---------------------------------------------------------------------------
# A multi-factor strategy: declared order, nothing extra
# ---------------------------------------------------------------------------
def test_a_multi_factor_strategy_reads_its_factors_in_declared_order() -> None:
    spec = make_spec(required_factors=(VALUE, MOMENTUM))
    strategy = FixedStrategy(spec, result_from_context(spec))
    extra = FactorRequirement(factor_name="size", factor_version="1.0.0")
    store = RecordingStore(
        {
            (VALUE.factor_name, VALUE.factor_version, AS_OF): make_panel(VALUE),
            (MOMENTUM.factor_name, MOMENTUM.factor_version, AS_OF): make_panel(MOMENTUM),
            (extra.factor_name, extra.factor_version, AS_OF): make_panel(extra),
        }
    )
    runner, _, _ = make_runner(store)

    runner.run(strategy, AS_OF)

    # the value panel was declared first, so it is read first
    assert store.reads == [
        (VALUE.factor_name, VALUE.factor_version, AS_OF),
        (MOMENTUM.factor_name, MOMENTUM.factor_version, AS_OF),
    ]
    # the undeclared factor was never touched, even though the store has it
    assert (extra.factor_name, extra.factor_version, AS_OF) not in store.reads
    context = strategy.contexts[0]
    assert list(context.factor_panels) == [VALUE, MOMENTUM]
    assert len(context.factor_panels) == 2
    assert set(context.factor_panels) == {VALUE, MOMENTUM}
    assert all(isinstance(panel, FactorPanel) for panel in context.factor_panels.values())


# ---------------------------------------------------------------------------
# The decision date: two forms, and no third
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "value",
    [
        pytest.param(AS_OF, id="date"),
        pytest.param("2024-01-05", id="iso-string"),
        # the strategy contract strips surrounding whitespace before matching, so
        # the runner does too: a date it accepts must never be refused later by
        # the context or the result it feeds
        pytest.param(" 2024-01-05 ", id="padded-iso-string"),
    ],
)
def test_a_date_or_a_strict_iso_string_is_accepted(value: date | str) -> None:
    spec = make_spec()
    strategy = FixedStrategy(spec, result_from_context(spec))
    runner, provider, store = make_runner(one_factor_store())

    result = runner.run(strategy, value)

    assert result.as_of_date == AS_OF
    assert provider.calls == [AS_OF]
    assert store.reads == [(MOMENTUM.factor_name, MOMENTUM.factor_version, AS_OF)]


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(datetime(2024, 1, 5), id="midnight-datetime"),
        pytest.param(datetime(2024, 1, 5, 16, 0), id="closing-datetime"),
        pytest.param(pd.Timestamp("2024-01-05"), id="midnight-timestamp"),
        pytest.param(pd.Timestamp("2024-01-05 16:00"), id="closing-timestamp"),
    ],
)
def test_a_timestamp_is_refused_even_at_midnight(value: Any) -> None:
    spec = make_spec()
    strategy = FixedStrategy(spec, result_from_context(spec))
    runner, provider, store = make_runner(one_factor_store())

    with pytest.raises(StrategyRunError, match="must be a date, not the timestamp"):
        runner.run(strategy, value)

    # refused before anything is asked of the data layers
    assert provider.calls == []
    assert store.reads == []
    assert strategy.contexts == []


@pytest.mark.parametrize(
    "value",
    [
        pytest.param("2024-1-5", id="short-month"),
        pytest.param("2024-01-5", id="short-day"),
        pytest.param("24-01-05", id="short-year"),
        pytest.param("2024-01-05T00:00", id="iso-datetime"),
        pytest.param("2024-01-05 00:00", id="iso-with-space"),
        pytest.param("2024-01-05Z", id="with-zulu"),
        pytest.param("05/01/2024", id="us-slash"),
        pytest.param("January 5, 2024", id="prose"),
        pytest.param("2024-02-30", id="impossible-day"),
        pytest.param("2024-13-01", id="impossible-month"),
        pytest.param("2024-W01-5", id="iso-week"),
        pytest.param("20240105", id="compact"),
        pytest.param("", id="empty"),
        pytest.param("   ", id="blank"),
        pytest.param(None, id="none"),
        pytest.param(20240105, id="int"),
        pytest.param(20240105.0, id="float"),
        pytest.param(b"2024-01-05", id="bytes"),
        pytest.param(("2024", "01", "05"), id="tuple"),
    ],
)
def test_anything_else_is_refused_as_a_date(value: Any) -> None:
    spec = make_spec()
    strategy = FixedStrategy(spec, result_from_context(spec))
    runner, provider, store = make_runner(one_factor_store())

    with pytest.raises(StrategyRunError, match="strict ISO-8601 'YYYY-MM-DD' string"):
        runner.run(strategy, value)

    assert provider.calls == []
    assert store.reads == []
    assert strategy.contexts == []


def test_a_session_without_a_stored_panel_is_the_stores_error() -> None:
    """The runner never moves a date to a neighbouring session."""
    spec = make_spec()
    strategy = FixedStrategy(spec, result_from_context(spec))
    runner, provider, store = make_runner(one_factor_store())  # only AS_OF is stored

    with pytest.raises(
        FactorStoreError, match="no panel is stored for momentum_12_1/1.0.0 on 2024-01-08"
    ):
        runner.run(strategy, NEXT_DAY)

    # the missing date was asked for as given: no fallback day was tried
    assert provider.calls == [NEXT_DAY]
    assert store.reads == [(MOMENTUM.factor_name, MOMENTUM.factor_version, NEXT_DAY)]
    assert strategy.contexts == []


def test_the_strategy_is_checked_before_the_date() -> None:
    """Step 1 fails first, so a bad date never masks a bad strategy."""
    runner, provider, store = make_runner(one_factor_store())

    with pytest.raises(StrategyRunError, match="strategy must be a Strategy"):
        runner.run(object(), "not-a-date")

    assert provider.calls == []
    assert store.reads == []


class SpecLessStrategy(Strategy):
    """A strategy whose ``spec`` is a bare label instead of a real spec."""

    spec = "top_10_by_momentum_12_1"

    def compute(self, context: StrategyContext) -> Any:
        """Never reached: the runner refuses this strategy's spec first."""
        raise AssertionError("compute() must not be reached")


def test_a_strategy_without_a_real_spec_is_refused() -> None:
    runner, provider, store = make_runner(one_factor_store())

    with pytest.raises(StrategyRunError, match=r"\.spec must be a StrategySpec"):
        runner.run(SpecLessStrategy(), AS_OF)

    assert provider.calls == []
    assert store.reads == []


# ---------------------------------------------------------------------------
# The universe provider: one snapshot, for this date
# ---------------------------------------------------------------------------
def test_a_provider_that_returns_something_else_is_refused() -> None:
    spec = make_spec()
    strategy = FixedStrategy(spec, result_from_context(spec))
    runner, _, store = make_runner(one_factor_store(), RecordingProvider(answer={"AAPL": 1.0}))

    with pytest.raises(StrategyRunError, match="must return a UniverseSnapshot"):
        runner.run(strategy, AS_OF)

    # refused before a single panel is read
    assert store.reads == []
    assert strategy.contexts == []


def test_a_snapshot_for_another_session_is_refused() -> None:
    spec = make_spec()
    strategy = FixedStrategy(spec, result_from_context(spec))
    provider = RecordingProvider(fixed=make_universe(as_of=NEXT_DAY))
    runner, _, store = make_runner(one_factor_store(), provider)

    with pytest.raises(
        StrategyRunError,
        match="answered for 2024-01-08 when asked about 2024-01-05",
    ):
        runner.run(strategy, AS_OF)

    assert provider.calls == [AS_OF]
    assert store.reads == []
    assert strategy.contexts == []


# ---------------------------------------------------------------------------
# The factor store: read, never probed, never substituted
# ---------------------------------------------------------------------------
def test_a_store_that_returns_something_else_is_refused() -> None:
    spec = make_spec()
    strategy = FixedStrategy(spec, result_from_context(spec))
    store = RecordingStore({(MOMENTUM.factor_name, MOMENTUM.factor_version, AS_OF): object()})
    runner, _, _ = make_runner(store)

    with pytest.raises(StrategyRunError, match="must return a FactorPanel, got object"):
        runner.run(strategy, AS_OF)

    assert strategy.contexts == []


def test_a_missing_panel_is_the_stores_error_and_nothing_else() -> None:
    spec = make_spec()
    strategy = FixedStrategy(spec, result_from_context(spec))
    runner, _, store = make_runner(RecordingStore())  # nothing stored at all

    with pytest.raises(FactorStoreError, match="no panel is stored for momentum_12_1/1.0.0"):
        runner.run(strategy, AS_OF)

    # the runner never computed, substituted or probed: it read once and stopped
    assert store.reads == [(MOMENTUM.factor_name, MOMENTUM.factor_version, AS_OF)]
    assert store.forbidden == []
    assert strategy.contexts == []


def test_a_store_error_travels_unchanged() -> None:
    error = FactorStoreError(
        "the file stored for momentum_12_1/1.0.0 on 2024-01-05 cannot be restored as a panel"
    )
    spec = make_spec()
    strategy = FixedStrategy(spec, result_from_context(spec))
    runner, _, _ = make_runner(RecordingStore(error=error))

    with pytest.raises(FactorStoreError) as raised:
        runner.run(strategy, AS_OF)

    assert raised.value is error
    assert not isinstance(raised.value, StrategyRunError)
    assert strategy.contexts == []


# ---------------------------------------------------------------------------
# The context's own panel checks, reported as they are
# ---------------------------------------------------------------------------
def panel_context_error(panel: FactorPanel) -> StrategyContractError:
    """Run one panel through the runner and return the error it caused."""
    spec = make_spec()
    strategy = FixedStrategy(spec, result_from_context(spec))
    store = RecordingStore({(MOMENTUM.factor_name, MOMENTUM.factor_version, AS_OF): panel})
    runner, _, _ = make_runner(store)

    with pytest.raises(StrategyContractError) as raised:
        runner.run(strategy, AS_OF)
    assert strategy.contexts == []
    return raised.value


def test_a_panel_dated_for_another_session_stops_the_run() -> None:
    error = panel_context_error(make_panel(MOMENTUM, as_of=NEXT_DAY))

    assert "is dated 2024-01-08, but the context is dated 2024-01-05" in str(error)
    assert not isinstance(error, StrategyRunError)


def test_a_panel_for_another_universe_stops_the_run() -> None:
    error = panel_context_error(
        make_panel(MOMENTUM, symbols=("AAPL", "MSFT"), raw_value=(0.2, 0.5))
    )

    assert "missing universe symbol(s) ['NVDA']" in str(error)


def test_a_panel_in_another_symbol_order_stops_the_run() -> None:
    error = panel_context_error(
        make_panel(MOMENTUM, symbols=("NVDA", "MSFT", "AAPL"), raw_value=(0.1, 0.5, 0.2))
    )

    assert "in a different order" in str(error)


def test_a_panel_of_another_factor_version_stops_the_run() -> None:
    mislabelled = FactorRequirement(factor_name="momentum_12_1", factor_version="9.9.9")
    error = panel_context_error(make_panel(mislabelled))

    assert "holds the panel of momentum_12_1/9.9.9" in str(error)


# ---------------------------------------------------------------------------
# The runtime double check: this run's strategy, date, pool and requirements
# ---------------------------------------------------------------------------
def test_a_result_that_is_not_a_strategy_result_is_refused() -> None:
    spec = make_spec()
    strategy = FixedStrategy(spec, lambda context: pd.DataFrame({"symbol": list(SYMBOLS)}))
    runner, _, _ = make_runner(one_factor_store())

    with pytest.raises(StrategyRunError, match="must return a StrategyResult"):
        runner.run(strategy, AS_OF)


def test_a_result_for_another_strategy_is_refused() -> None:
    spec = make_spec()
    other = make_spec(name="blend_momentum_value_v2", version="2.0.0")

    def build(context: StrategyContext) -> StrategyResult:
        return make_result(other, universe=context.universe, as_of=context.as_of_date)

    strategy = FixedStrategy(spec, build)
    runner, _, _ = make_runner(one_factor_store())

    with pytest.raises(StrategyRunError) as raised:
        runner.run(strategy, AS_OF)

    [problem] = raised.value.problems
    assert problem.startswith("strategy_spec is blend_momentum_value_v2/2.0.0")
    assert "expected blend_momentum_value/1.0.0 requiring [momentum_12_1/1.0.0]" in problem
    assert "does not belong to this run of 'blend_momentum_value' v1.0.0" in str(raised.value)


def test_a_result_for_another_session_is_refused() -> None:
    spec = make_spec()

    def build(context: StrategyContext) -> StrategyResult:
        return make_result(spec, universe=context.universe, as_of=NEXT_DAY)

    strategy = FixedStrategy(spec, build)
    runner, _, _ = make_runner(one_factor_store())

    with pytest.raises(StrategyRunError) as raised:
        runner.run(strategy, AS_OF)

    assert raised.value.problems == ["as_of_date is 2024-01-08, expected 2024-01-05"]


def test_a_result_scored_against_a_look_alike_universe_is_refused() -> None:
    """The pool is compared by identity: an equal snapshot is still another pool."""
    spec = make_spec()
    look_alike = make_universe()

    def build(context: StrategyContext) -> StrategyResult:
        return make_result(spec, universe=look_alike, as_of=context.as_of_date)

    strategy = FixedStrategy(spec, build)
    runner, provider, _ = make_runner(one_factor_store())

    with pytest.raises(StrategyRunError) as raised:
        runner.run(strategy, AS_OF)

    assert look_alike is not provider.snapshots[0]
    assert look_alike == provider.snapshots[0]
    [problem] = raised.value.problems
    assert problem.startswith("universe is not the snapshot this run was given")
    assert "the same symbols but is another object" in problem


def test_a_result_reporting_other_factor_requirements_is_refused() -> None:
    spec = make_spec()

    def build(context: StrategyContext) -> StrategyResult:
        base = make_result(spec, universe=context.universe, as_of=context.as_of_date)
        return MislabelledResult(
            base.frame,
            universe=context.universe,
            strategy_spec=spec,
            as_of_date=context.as_of_date,
        )

    strategy = FixedStrategy(spec, build)
    runner, _, _ = make_runner(one_factor_store())

    with pytest.raises(StrategyRunError) as raised:
        runner.run(strategy, AS_OF)

    assert raised.value.problems == [
        "factor_requirements are [book_to_price/2.0.0], expected [momentum_12_1/1.0.0]"
    ]


def test_every_mismatch_is_reported_in_one_error() -> None:
    spec = make_spec()
    other = make_spec(name="value_only_blend", version="2.0.0", required_factors=(VALUE,))
    stranger = make_result(other, universe=make_universe(), as_of=NEXT_DAY)

    def build(context: StrategyContext) -> Any:
        return MislabelledResult(
            stranger.frame,
            universe=stranger.universe,
            strategy_spec=other,
            as_of_date=NEXT_DAY,
        )

    strategy = FixedStrategy(spec, build)
    runner, _, _ = make_runner(one_factor_store())

    with pytest.raises(StrategyRunError) as raised:
        runner.run(strategy, AS_OF)

    problems = raised.value.problems
    assert len(problems) == 4
    assert problems[0].startswith("strategy_spec is value_only_blend/2.0.0")
    assert problems[1] == "as_of_date is 2024-01-08, expected 2024-01-05"
    assert problems[2].startswith("universe is not the snapshot this run was given")
    assert problems[3] == (
        "factor_requirements are [book_to_price/2.0.0], expected [momentum_12_1/1.0.0]"
    )
    for problem in problems:
        assert f"  - {problem}" in str(raised.value)


# ---------------------------------------------------------------------------
# How an exception from a strategy travels
# ---------------------------------------------------------------------------
def test_a_strategy_contract_error_travels_unchanged() -> None:
    error = StrategyContractError(
        "the panel has no eligible symbol",
        ["all 3 symbols are ineligible"],
    )
    strategy = FixedStrategy(make_spec(), error=error)
    runner, _, _ = make_runner(one_factor_store())

    with pytest.raises(StrategyContractError) as raised:
        runner.run(strategy, AS_OF)

    assert raised.value is error
    assert not isinstance(raised.value, StrategyRunError)


def test_any_other_strategy_error_travels_unchanged_too() -> None:
    error = StrategyRegistryError("no strategy is registered for 'blend_momentum_value' v1.0.0")
    strategy = FixedStrategy(make_spec(), error=error)
    runner, _, _ = make_runner(one_factor_store())

    with pytest.raises(StrategyRegistryError) as raised:
        runner.run(strategy, AS_OF)

    assert raised.value is error


def test_an_unrelated_exception_is_wrapped_with_its_cause() -> None:
    original = ZeroDivisionError("division by zero")
    strategy = FixedStrategy(make_spec(), error=original)
    runner, _, _ = make_runner(one_factor_store())

    with pytest.raises(StrategyRunError) as raised:
        runner.run(strategy, AS_OF)

    assert raised.value.__cause__ is original
    assert raised.value.problems == []
    message = str(raised.value)
    assert "raised ZeroDivisionError while scoring 2024-01-05" in message
    assert "blend_momentum_value" in message
    assert "division by zero" in message


def test_a_run_error_collects_its_problems() -> None:
    error = StrategyRunError("the result does not belong to this run", ["first", "second"])

    assert error.problems == ["first", "second"]
    assert str(error) == "the result does not belong to this run\n  - first\n  - second"
    assert StrategyRunError("no problems").problems == []


# ---------------------------------------------------------------------------
# No side effects: nothing written, nothing probed, nothing stored
# ---------------------------------------------------------------------------
def test_a_run_never_writes_probes_a_range_or_asks_for_a_path() -> None:
    """Every forbidden store method raises, so a clean run proves none was called."""
    strategy = DelegatingStrategy(TopNByFactorStrategy(MOMENTUM, 2))
    runner, provider, store = make_runner(one_factor_store())

    runner.run(strategy, AS_OF)

    assert store.forbidden == []
    assert provider.calls == [AS_OF]
    assert len(store.reads) == 1  # one factor, read once, never re-read


# ---------------------------------------------------------------------------
# The module's own boundary
# ---------------------------------------------------------------------------
RUNNER_PATH = Path(runner_module.__file__)

#: Modules ``runner.py`` may import: the standard library, the factor layer's
#: interfaces and validated objects, and its own contract and errors.
ALLOWED_MODULE_IMPORTS: frozenset[str] = frozenset(
    {
        "__future__",
        "datetime",
        "re",
        "typing",
        "factor_sys.contracts",
        "factor_sys.errors",
        "factor_sys.factor_store",
        "factor_sys.universe",
        "strategy_sys.contracts",
        "strategy_sys.errors",
    }
)

#: Names and modules whose presence would mean the runner knows a rule, a
#: formula, a storage format, a calendar or a later layer.
FORBIDDEN_TEXT: tuple[str, ...] = (
    "pandas",
    "data_sys",
    "pyarrow",
    "duckdb",
    "pathlib",
    "socket",
    "requests",
    "urllib",
    "subprocess",
    "exchange_calendars",
    "market_calendar",
    "factor_sys.runner",
    "factor_sys.market_data",
    "factor_sys.registry",
    "FactorRunner",
    "FactorRegistry",
    "strategy_sys.registry",
    "StrategyRegistry",
    "strategy_sys.strategies",
    "TopNByFactorStrategy",
    "open(",
    "print(",
)


def runner_source() -> str:
    """The runner's source, as written."""
    return RUNNER_PATH.read_text(encoding="utf-8")


def runner_tree() -> ast.Module:
    """The runner's syntax tree."""
    return ast.parse(runner_source(), filename=str(RUNNER_PATH))


def runner_imports() -> list[str]:
    """Every module the runner imports, at module level or inside a function."""
    imported: list[str] = []
    for node in ast.walk(runner_tree()):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.append(node.module)
    return imported


def runner_docstrings() -> set[str]:
    """Every docstring in the runner: prose, which may *describe* what it avoids."""
    documented = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    found: set[str] = set()
    for node in ast.walk(runner_tree()):
        if isinstance(node, documented):
            doc = ast.get_docstring(node, clean=False)
            if doc is not None:
                found.add(doc)
    return found


def runner_code_names() -> set[str]:
    """Every identifier, parameter, attribute and string literal the runner *uses*."""
    names: set[str] = set()
    for node in ast.walk(runner_tree()):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            names.add(node.value)
    return names - runner_docstrings()


def runner_called_names() -> set[str]:
    """Everything the runner calls, rendered as source text (e.g. ``store.read``)."""
    return {
        ast.unparse(node.func) for node in ast.walk(runner_tree()) if isinstance(node, ast.Call)
    }


def test_the_runner_imports_only_its_own_layer_and_the_factor_interfaces() -> None:
    imported = runner_imports()

    assert imported, "the boundary test must actually see the runner's imports"
    assert set(imported) <= ALLOWED_MODULE_IMPORTS
    # spelled out, because these are the mistakes that would actually be made
    for forbidden in ("factor_sys.runner", "factor_sys.market_data", "strategy_sys.strategies"):
        assert forbidden not in imported


def test_the_runner_never_names_a_forbidden_dependency() -> None:
    """Prose may say what the runner avoids; the code itself must never name it."""
    named = runner_code_names()

    assert named, "the boundary test must actually see the runner's code"
    for forbidden in FORBIDDEN_TEXT:
        offenders = sorted(name for name in named if forbidden in name)
        assert offenders == [], f"runner.py must not mention {forbidden!r}: {offenders}"


def test_the_runner_never_opens_prints_or_evaluates_anything() -> None:
    called = runner_called_names()

    assert called & {"open", "print", "input", "eval", "exec", "compile", "__import__"} == set()
    assert called, "the runner does call the interfaces it was given"


def test_the_runner_calls_only_the_three_interfaces_it_was_given() -> None:
    called = runner_called_names()

    # one snapshot per run, one read per declared factor, one compute per run
    assert {"provider.snapshot", "store.read", "strategy.compute"} <= called
    # and never a probe, a range, a path or a write
    for forbidden in ("exists", "read_range", "path_for", "write"):
        assert not any(name.endswith(f".{forbidden}") for name in called)


# ---------------------------------------------------------------------------
# The runner's public surface
# ---------------------------------------------------------------------------
def test_the_runner_is_exported_from_the_package_root() -> None:
    assert strategy_sys.StrategyRunner is StrategyRunner
    assert strategy_sys.StrategyRunError is StrategyRunError
    assert {"StrategyRunner", "StrategyRunError"} <= set(strategy_sys.__all__)
    assert runner_module.__all__ == ["StrategyRunner"]


def test_a_run_error_is_a_strategy_error_and_nothing_else() -> None:
    assert issubclass(StrategyRunError, StrategyError)
    assert not issubclass(StrategyRunError, StrategyContractError)
    assert not issubclass(StrategyRunError, ValueError)
    assert not issubclass(StrategyRunError, KeyError)


def test_the_runner_takes_one_strategy_and_one_date_and_nothing_else() -> None:
    assert list(inspect.signature(StrategyRunner.__init__).parameters) == [
        "self",
        "universe_provider",
        "factor_store",
    ]
    assert list(inspect.signature(StrategyRunner.run).parameters) == [
        "self",
        "strategy",
        "as_of_date",
    ]
    # a result, not a run report: there is no wrapper object to unwrap
    assert inspect.signature(StrategyRunner.run).return_annotation == "StrategyResult"
