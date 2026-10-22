"""Tests for the explicit factor registry (:mod:`factor_sys.registry`).

Everything here is in memory and offline: the table is filled with tiny local
``Factor`` subclasses, and the one integration test runs a registry-created
``Momentum12_1`` through the *real* :class:`~factor_sys.runner.FactorRunner`, fed by
an in-memory universe provider and an in-memory market-data reader.  No ``data/``
directory is read, no Parquet file is written and nothing touches the network.

Covered in this order:

* an empty table: ``names``, ``len``, membership, ``repr``,
* registration: the name comes from the spec, a factory is called once at
  registration and once per ``create``, every ``create`` returns a new object,
* the default table: exactly ``momentum_12_1``, independent per call,
* refusals: duplicate names, unknown names, illegal names (never normalized),
  non-string names, non-callable factories, factories that raise (with their
  cause), factories that do not build a ``Factor``, a spec that is not a
  ``FactorSpec``, a factory that changes what it builds,
* ``names`` as a sorted, unmodifiable tuple,
* the integration: a factor created by name runs through the runner and yields a
  ``FactorRunResult.panel``,
* the architectural boundary of the module and its exports.
"""

from __future__ import annotations

import ast
import inspect
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import factor_sys
import factor_sys.registry as registry_module
import factor_sys.runner as runner_module
from data_sys.errors import DataPipelineError
from factor_sys import (
    FACTOR_NAME_PATTERN,
    Factor,
    FactorContext,
    FactorContractError,
    FactorError,
    FactorPanel,
    FactorRegistry,
    FactorRegistryError,
    FactorRunner,
    FactorRunResult,
    FactorSpec,
    MarketDataReader,
    UniverseProvider,
    UniverseSnapshot,
    default_factor_registry,
)
from factors import Momentum12_1
from factors.momentum import REQUIRED_OBSERVATIONS, SKIP_SESSIONS


# ---------------------------------------------------------------------------
# Factors to register: tiny, local, and never computed
# ---------------------------------------------------------------------------
def make_factor_class(name: str, *, version: str = "1.0.0") -> type[Factor]:
    """A minimal concrete factor whose spec carries ``name``.

    A class per call, so every test starts from a fresh identity.  A class *is*
    the normal factory: calling it builds a new instance every time.
    """

    class Built(Factor):
        spec = FactorSpec(name=name, version=version, required_fields=("close",))

        def compute(self, context: FactorContext) -> FactorPanel:
            raise NotImplementedError("this test double never computes a panel")

    return Built


def forged_spec(name: str) -> FactorSpec:
    """A ``FactorSpec`` built *past* its own validation, to test the guard."""
    spec = object.__new__(FactorSpec)
    object.__setattr__(spec, "name", name)
    object.__setattr__(spec, "version", "1.0.0")
    object.__setattr__(spec, "required_fields", ("close",))
    object.__setattr__(spec, "lookback_sessions", 0)
    return spec


def forged_factor_class(name: str) -> type[Factor]:
    """A ``Factor`` whose spec never went through ``FactorSpec`` validation."""

    class Forged(Factor):
        @property
        def spec(self) -> FactorSpec:
            return forged_spec(name)

        def compute(self, context: FactorContext) -> FactorPanel:
            raise NotImplementedError("this test double never computes a panel")

    return Forged


# ---------------------------------------------------------------------------
# An empty table
# ---------------------------------------------------------------------------
def test_a_new_registry_is_empty() -> None:
    """Nothing registered: no names, no members, and a readable ``repr``."""
    registry = FactorRegistry()

    assert registry.names == ()
    assert len(registry) == 0
    assert "momentum_12_1" not in registry
    assert repr(registry) == "FactorRegistry()"


# ---------------------------------------------------------------------------
# Registration and creation
# ---------------------------------------------------------------------------
def test_a_registered_factor_can_be_created_by_name() -> None:
    """``register`` returns the spec's name; ``create`` builds that factor."""
    registry = FactorRegistry()
    factor_class = make_factor_class("other_factor")

    name = registry.register(factor_class)

    assert name == "other_factor"
    assert registry.names == ("other_factor",)
    assert len(registry) == 1
    assert "other_factor" in registry
    assert isinstance(registry.create("other_factor"), factor_class)
    assert repr(registry) == "FactorRegistry(names=('other_factor',))"


def test_every_create_builds_a_new_instance() -> None:
    """The table stores a factory, never an instance: no state crosses a run."""
    built: list[Factor] = []
    factor_class = make_factor_class("counted_factor")

    def build() -> Factor:
        factor = factor_class()
        built.append(factor)
        return factor

    registry = FactorRegistry()
    registry.register(build)

    first = registry.create("counted_factor")
    second = registry.create("counted_factor")

    # one build at registration (to learn the name), one per create
    assert len(built) == 3
    assert first is built[1]
    assert second is built[2]
    assert first is not second
    assert isinstance(first, factor_class) and isinstance(second, factor_class)

    first.marker = "first"  # type: ignore[attr-defined]
    assert not hasattr(second, "marker")


def test_the_name_always_comes_from_the_spec() -> None:
    """``register`` takes no name, so the table cannot disagree with the panel."""
    registry = FactorRegistry()
    factor_class = make_factor_class("called_what_its_spec_says")

    assert list(inspect.signature(FactorRegistry.register).parameters) == ["self", "factory"]
    assert registry.register(factor_class) == "called_what_its_spec_says"
    assert registry.names == ("called_what_its_spec_says",)
    assert isinstance(registry.create("called_what_its_spec_says"), factor_class)


def test_a_forged_spec_name_is_read_instead_of_guessed() -> None:
    """A spec that skipped validation is still the only source of the name."""
    registry = FactorRegistry()

    name = registry.register(forged_factor_class("legal_but_forged"))

    assert name == "legal_but_forged"
    assert registry.names == ("legal_but_forged",)


# ---------------------------------------------------------------------------
# The built-in table
# ---------------------------------------------------------------------------
def test_the_default_registry_holds_exactly_the_built_in_factors() -> None:
    """Today there is one built-in factor, and it is reachable by its name."""
    registry = default_factor_registry()

    assert registry.names == ("momentum_12_1",)
    assert len(registry) == 1
    assert "momentum_12_1" in registry
    assert repr(registry) == "FactorRegistry(names=('momentum_12_1',))"

    factor = registry.create("momentum_12_1")

    assert isinstance(factor, Momentum12_1)
    assert factor.spec.name == "momentum_12_1"
    assert factor.spec.version == "1.0.0"


def test_two_creates_of_the_built_in_factor_are_independent() -> None:
    """The registry never hands out a shared ``Momentum12_1``."""
    registry = default_factor_registry()

    first = registry.create("momentum_12_1")
    second = registry.create("momentum_12_1")

    assert first is not second
    first.marker = "first"  # type: ignore[attr-defined]
    assert not hasattr(second, "marker")


def test_default_registries_are_independent() -> None:
    """Registering something in one table must not be visible in the next."""
    first = default_factor_registry()
    second = default_factor_registry()

    assert first is not second
    assert first.names == second.names

    first.register(make_factor_class("extra_factor"))

    assert first.names == ("extra_factor", "momentum_12_1")
    assert second.names == ("momentum_12_1",)
    assert "extra_factor" not in second
    with pytest.raises(FactorRegistryError):
        second.create("extra_factor")


def test_the_default_registry_refuses_to_register_a_builtin_twice() -> None:
    """The built-in is registered once, and registering it again is an error."""
    registry = default_factor_registry()

    with pytest.raises(FactorRegistryError) as excinfo:
        registry.register(Momentum12_1)

    assert "already registered" in str(excinfo.value)
    assert registry.names == ("momentum_12_1",)


# ---------------------------------------------------------------------------
# Refusals: names
# ---------------------------------------------------------------------------
def test_a_duplicate_name_is_refused_and_never_overwrites() -> None:
    """The first registration wins; a second one is an error naming the factor."""
    registry = FactorRegistry()
    original = make_factor_class("duplicate_factor")
    replacement = make_factor_class("duplicate_factor", version="2.0.0")
    registry.register(original)

    with pytest.raises(FactorRegistryError) as excinfo:
        registry.register(replacement)

    message = str(excinfo.value)
    assert "'duplicate_factor'" in message
    assert "already registered" in message
    assert "never overwrites" in message
    assert "would replace" in message
    assert registry.names == ("duplicate_factor",)
    assert isinstance(registry.create("duplicate_factor"), original)
    assert registry.create("duplicate_factor").spec.version == "1.0.0"


def test_registering_the_same_factory_twice_is_still_a_duplicate() -> None:
    """No idempotent shortcut: the rule is one factory per name, always."""
    registry = FactorRegistry()
    factor_class = make_factor_class("same_factor")
    registry.register(factor_class)

    with pytest.raises(FactorRegistryError) as excinfo:
        registry.register(factor_class)

    message = str(excinfo.value)
    assert "already registered under that name" in message
    assert "would replace" not in message
    assert registry.names == ("same_factor",)


def test_an_unknown_name_lists_what_is_registered() -> None:
    """A lookup failure names the request *and* everything the table does hold."""
    registry = FactorRegistry()
    registry.register(make_factor_class("alpha_factor"))
    registry.register(make_factor_class("beta_factor"))

    with pytest.raises(FactorRegistryError) as excinfo:
        registry.create("gamma_factor")

    message = str(excinfo.value)
    assert "'gamma_factor'" in message
    assert "'alpha_factor'" in message
    assert "'beta_factor'" in message
    assert "not registered" in message


def test_an_unknown_name_in_an_empty_registry_says_none_are_registered() -> None:
    """An empty table answers with ``none`` rather than an empty list."""
    with pytest.raises(FactorRegistryError) as excinfo:
        FactorRegistry().create("momentum_12_1")

    message = str(excinfo.value)
    assert "'momentum_12_1'" in message
    assert "none" in message


@pytest.mark.parametrize(
    "name",
    [
        "Momentum_12_1",
        " momentum_12_1",
        "momentum_12_1 ",
        "momentum-12-1",
        "momentum 12 1",
        "momentum.12.1",
        "动量",
        "",
    ],
)
def test_an_illegal_name_is_refused_instead_of_normalized(name: str) -> None:
    """No trimming, no lower-casing, no guessing: a typo is an error."""
    registry = default_factor_registry()

    with pytest.raises(FactorRegistryError) as excinfo:
        registry.create(name)

    message = str(excinfo.value)
    assert "not a legal factor name" in message
    assert FACTOR_NAME_PATTERN in message
    assert "never trimmed" in message
    assert registry.names == ("momentum_12_1",)


@pytest.mark.parametrize(
    "name",
    ["Momentum_12_1", " momentum_12_1", "momentum_12_1 ", "momentum-12-1"],
)
def test_a_forged_spec_name_is_checked_at_registration(name: str) -> None:
    """Even a spec that skipped its own validation cannot smuggle a name in."""
    registry = FactorRegistry()

    with pytest.raises(FactorRegistryError) as excinfo:
        registry.register(forged_factor_class(name))

    message = str(excinfo.value)
    assert name in message
    assert "not a legal factor name" in message
    assert registry.names == ()


@pytest.mark.parametrize("value", [None, 12, 3.5, ["momentum_12_1"], ("momentum_12_1",), object()])
def test_a_name_that_is_not_a_string_is_refused(value: object) -> None:
    """A list of names is not a name, and ``str()`` coercion is not a fallback."""
    registry = default_factor_registry()

    with pytest.raises(FactorRegistryError) as excinfo:
        registry.create(value)  # type: ignore[arg-type]

    message = str(excinfo.value)
    assert "must be a string" in message
    assert type(value).__name__ in message


@pytest.mark.parametrize("value", [None, 12, ["momentum_12_1"]])
def test_membership_of_a_non_string_is_simply_false(value: object) -> None:
    """``in`` never raises: an impossible key is just absent."""
    registry = default_factor_registry()

    assert value not in registry


# ---------------------------------------------------------------------------
# Refusals: factories
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("factory", [None, 3, "Momentum12_1", Momentum12_1(), {"a": 1}])
def test_a_non_callable_factory_is_refused(factory: object) -> None:
    """An instance is not a factory: registering one would share it across runs."""
    registry = FactorRegistry()

    with pytest.raises(FactorRegistryError) as excinfo:
        registry.register(factory)  # type: ignore[arg-type]

    message = str(excinfo.value)
    assert "must be callable" in message
    assert type(factory).__name__ in message
    assert registry.names == ()


def test_a_factory_that_raises_is_reported_with_its_cause() -> None:
    """An ordinary exception is wrapped, and the original is kept as ``__cause__``."""

    def broken() -> Factor:
        raise RuntimeError("no bars at all")

    registry = FactorRegistry()

    with pytest.raises(FactorRegistryError) as excinfo:
        registry.register(broken)

    error = excinfo.value
    assert "no bars at all" in str(error)
    assert "RuntimeError" in str(error)
    assert isinstance(error.__cause__, RuntimeError)
    assert str(error.__cause__) == "no bars at all"
    assert registry.names == ()


def test_a_factory_that_fails_only_when_creating_keeps_its_cause() -> None:
    """``create`` wraps too: registration succeeded, the *build* did not."""
    calls: list[int] = []

    def flaky() -> Factor:
        calls.append(len(calls) + 1)
        if len(calls) > 1:
            raise ValueError("the second build fails")
        return make_factor_class("flaky_factor")()

    registry = FactorRegistry()
    registry.register(flaky)

    with pytest.raises(FactorRegistryError) as excinfo:
        registry.create("flaky_factor")

    message = str(excinfo.value)
    assert "the second build fails" in message
    assert "ValueError" in message
    assert isinstance(excinfo.value.__cause__, ValueError)
    assert registry.names == ("flaky_factor",)


def test_a_factor_system_error_from_the_factory_is_not_wrapped() -> None:
    """A ``FactorError`` already says what is wrong: it keeps its own type."""
    contract_error = FactorContractError("invalid FactorSpec", ["name must be snake_case"])

    def broken() -> Factor:
        raise contract_error

    registry = FactorRegistry()

    with pytest.raises(FactorContractError) as excinfo:
        registry.register(broken)

    assert excinfo.value is contract_error
    assert not isinstance(excinfo.value, FactorRegistryError)
    assert registry.names == ()


def test_a_factor_system_error_at_create_time_keeps_its_type() -> None:
    """The pass-through is a rule, not a one-off in ``register``."""
    calls: list[int] = []

    def once_then_broken() -> Factor:
        calls.append(len(calls) + 1)
        if len(calls) > 1:
            raise FactorContractError("invalid FactorSpec", ["version must not be empty"])
        return make_factor_class("once_then_broken_factor")()

    registry = FactorRegistry()
    registry.register(once_then_broken)

    with pytest.raises(FactorContractError) as excinfo:
        registry.create("once_then_broken_factor")

    assert "version must not be empty" in str(excinfo.value)
    assert not isinstance(excinfo.value, FactorRegistryError)


@pytest.mark.parametrize(
    "answer",
    [
        None,
        7,
        "a factor",
        {"spec": None},
        SimpleNamespace(spec=forged_spec("looks_like_a_factor"), compute=lambda context: None),
    ],
)
def test_a_factory_that_does_not_build_a_factor_is_refused(answer: object) -> None:
    """Only a real ``Factor`` counts -- a lookalike is not one, and no cause exists."""
    registry = FactorRegistry()

    with pytest.raises(FactorRegistryError) as excinfo:
        registry.register(lambda: answer)  # type: ignore[arg-type,return-value]

    message = str(excinfo.value)
    assert "must build a Factor" in message
    assert type(answer).__name__ in message
    assert excinfo.value.__cause__ is None
    assert registry.names == ()


def test_a_factor_without_a_valid_spec_is_refused() -> None:
    """A ``Factor`` whose ``spec`` is not a ``FactorSpec`` has no usable name."""

    class NoSpec(Factor):
        spec = SimpleNamespace(name="no_spec_factor", version="1.0.0")

        def compute(self, context: FactorContext) -> FactorPanel:
            raise NotImplementedError("this test double never computes a panel")

    with pytest.raises(FactorRegistryError) as excinfo:
        FactorRegistry().register(NoSpec)

    message = str(excinfo.value)
    assert "not a FactorSpec" in message
    assert "SimpleNamespace" in message


def test_a_factory_that_changes_what_it_builds_is_caught_at_create() -> None:
    """A closure that drifts is refused instead of producing a foreign panel."""
    current = {"name": "first_factor"}

    def drifting() -> Factor:
        return make_factor_class(current["name"])()

    registry = FactorRegistry()
    registry.register(drifting)
    current["name"] = "second_factor"

    with pytest.raises(FactorRegistryError) as excinfo:
        registry.create("first_factor")

    message = str(excinfo.value)
    assert "'first_factor'" in message
    assert "'second_factor'" in message
    assert "must always build the factor it was registered under" in message
    assert registry.names == ("first_factor",)


# ---------------------------------------------------------------------------
# One instance, once: a factory may not hand the same object out twice
# ---------------------------------------------------------------------------
def test_a_singleton_factory_is_refused_at_the_first_create() -> None:
    """A lambda over one pre-built factor is a legal factory but a shared object."""
    singleton = Momentum12_1()
    registry = FactorRegistry()

    assert registry.register(lambda: singleton) == "momentum_12_1"

    with pytest.raises(FactorRegistryError) as excinfo:
        registry.create("momentum_12_1")

    message = str(excinfo.value)
    assert "'momentum_12_1'" in message
    assert "reused" in message
    assert "new factor" in message

    # the registry refuses the factory; it neither resets nor copies the factor
    assert singleton.spec.name == "momentum_12_1"
    assert registry.names == ("momentum_12_1",)
    assert len(registry) == 1
    with pytest.raises(FactorRegistryError):
        registry.create("momentum_12_1")


def test_the_probe_counts_as_used_even_after_good_creates() -> None:
    """The instance built at registration never leaves, but it is not fresh either."""
    probe = Momentum12_1()
    plan = iter([probe, Momentum12_1(), Momentum12_1(), probe])
    calls: list[Factor] = []

    def factory() -> Factor:
        factor = next(plan)
        calls.append(factor)
        return factor

    registry = FactorRegistry()
    registry.register(factory)

    first = registry.create("momentum_12_1")
    second = registry.create("momentum_12_1")

    assert first is calls[1]
    assert second is calls[2]
    assert first is not second

    with pytest.raises(FactorRegistryError) as excinfo:
        registry.create("momentum_12_1")  # the probe, handed out for the first time

    message = str(excinfo.value)
    assert "'momentum_12_1'" in message
    assert "reused" in message
    assert registry.names == ("momentum_12_1",)


def test_a_late_repeat_of_an_already_returned_instance_is_refused() -> None:
    """Not an adjacent-pair check: the third create replays the first one."""
    probe = Momentum12_1()
    first = Momentum12_1()
    second = Momentum12_1()
    plan = iter([probe, first, second, first])

    def factory() -> Factor:
        return next(plan)

    registry = FactorRegistry()
    registry.register(factory)

    returned_first = registry.create("momentum_12_1")
    returned_second = registry.create("momentum_12_1")

    assert returned_first is first
    assert returned_second is second
    assert returned_first is not returned_second  # neighbours differ ...

    with pytest.raises(FactorRegistryError) as excinfo:
        registry.create("momentum_12_1")  # ... and replaying one is still reuse

    message = str(excinfo.value)
    assert "'momentum_12_1'" in message
    assert "reused" in message
    assert registry.names == ("momentum_12_1",)


def test_a_class_factory_still_builds_a_fresh_factor_every_time() -> None:
    """The guard only catches shared objects: a class builds a new one per call."""
    registry = FactorRegistry()
    registry.register(Momentum12_1)

    factors = [registry.create("momentum_12_1") for _ in range(3)]

    assert all(isinstance(factor, Momentum12_1) for factor in factors)
    assert len({id(factor) for factor in factors}) == 3

    built_in = default_factor_registry()
    first = built_in.create("momentum_12_1")
    second = built_in.create("momentum_12_1")

    assert first is not second
    assert first.spec.name == second.spec.name == "momentum_12_1"
    assert first.spec.version == second.spec.version == "1.0.0"


def test_identity_not_equality_decides_whether_an_instance_is_new() -> None:
    """Two distinct objects that compare equal are two fresh factors."""
    factor_class = make_factor_class("equal_factor")
    factor_class.__eq__ = lambda self, other: True  # every instance equals any
    factor_class.__hash__ = None  # and therefore need not be hashable

    registry = FactorRegistry()
    registry.register(factor_class)

    first = registry.create("equal_factor")
    second = registry.create("equal_factor")

    assert first == second  # equal ...
    assert first is not second  # ... but not the same object
    assert registry.names == ("equal_factor",)


def test_a_refused_reuse_leaves_the_table_and_other_registries_alone() -> None:
    """No registration is dropped, and one table's ledger never leaks into another."""
    factor_class = make_factor_class("kept_factor")
    singleton = factor_class()

    registry = FactorRegistry()
    registry.register(factor_class)
    shared = FactorRegistry()
    shared.register(lambda: singleton)

    with pytest.raises(FactorRegistryError):
        shared.create("kept_factor")

    assert registry.names == ("kept_factor",)
    assert shared.names == ("kept_factor",)
    assert isinstance(registry.create("kept_factor"), factor_class)

    # the duplicate path is unchanged: still refused, still no overwrite
    with pytest.raises(FactorRegistryError) as excinfo:
        registry.register(factor_class)
    assert "already registered" in str(excinfo.value)
    assert registry.names == ("kept_factor",)


# ---------------------------------------------------------------------------
# ``names``: sorted, stable, unmodifiable
# ---------------------------------------------------------------------------
def test_names_are_sorted_and_cannot_be_mutated() -> None:
    """A tuple in alphabetical order, and no path back into the table's dict."""
    registry = FactorRegistry()
    for name in ("zeta_factor", "alpha_factor", "middle_factor"):
        registry.register(make_factor_class(name))

    names = registry.names

    assert isinstance(names, tuple)
    assert names == ("alpha_factor", "middle_factor", "zeta_factor")
    assert registry.names == names
    with pytest.raises(TypeError):
        names[-1] = "hacked"  # type: ignore[index]
    with pytest.raises(AttributeError):
        registry.names = names  # type: ignore[misc]

    # a tuple handed out earlier is not a window into the table
    registry.register(make_factor_class("beta_factor"))
    assert names == ("alpha_factor", "middle_factor", "zeta_factor")
    assert registry.names == ("alpha_factor", "beta_factor", "middle_factor", "zeta_factor")


# ---------------------------------------------------------------------------
# The error type
# ---------------------------------------------------------------------------
def test_a_registry_error_is_a_factor_system_error_only() -> None:
    """Catchable with ``FactorError`` -- and never with a data-system error."""
    assert issubclass(FactorRegistryError, FactorError)
    assert not issubclass(FactorRegistryError, DataPipelineError)
    assert str(FactorRegistryError("factor 'x' is not registered")) == (
        "factor 'x' is not registered"
    )


# ---------------------------------------------------------------------------
# Integration: a factor created by name runs through the real runner
# ---------------------------------------------------------------------------
#: A real XNYS session (a half day, but a session): the decision date used here.
AS_OF: date = date(2024, 12, 31)

#: The pool the in-memory provider answers with.
SYMBOLS: tuple[str, ...] = ("AAPL", "MSFT")


def sessions(count: int, *, end: date = AS_OF) -> list[date]:
    """``count`` consecutive weekdays ending on ``end``.

    Synthetic on purpose: the momentum formula is position-based, so this test
    needs no exchange calendar to build a 253-session history.
    """
    return [stamp.date() for stamp in pd.bdate_range(end=pd.Timestamp(end), periods=count)]


def momentum_prices(*, start: float = 100.0, end: float = 130.0) -> list[float]:
    """A complete window whose anchors are ``start`` (``t-252``) and ``end`` (``t-21``)."""
    prices = [100.0] * REQUIRED_OBSERVATIONS
    prices[-REQUIRED_OBSERVATIONS] = start
    prices[-SKIP_SESSIONS - 1] = end
    return prices


class MemoryUniverseProvider(UniverseProvider):
    """A provider over a literal pool: no seed file, no clock, no cache."""

    def __init__(self, symbols: Sequence[str] = SYMBOLS) -> None:
        self.symbols = tuple(symbols)
        self.asked: list[date] = []

    def snapshot(self, as_of_date: date) -> UniverseSnapshot:
        self.asked.append(as_of_date)
        return UniverseSnapshot(
            as_of_date=as_of_date, symbols=self.symbols, source="memory provider"
        )


class MemoryMarketDataReader(MarketDataReader):
    """``REQUIRED_OBSERVATIONS`` weekday bars per symbol, ending on the request date."""

    def __init__(self, prices: Mapping[str, Sequence[float]]) -> None:
        self.prices = {symbol: list(series) for symbol, series in prices.items()}
        self.requests: list[tuple[date, date]] = []

    def read_history(
        self,
        snapshot: UniverseSnapshot,
        start_date: date,
        required_fields: Sequence[str],
    ) -> pd.DataFrame:
        self.requests.append((start_date, snapshot.as_of_date))
        dates = sessions(REQUIRED_OBSERVATIONS, end=snapshot.as_of_date)
        frames = [
            pd.DataFrame(
                {
                    "date": pd.Series(dates, dtype="datetime64[s]"),
                    "symbol": [symbol] * len(dates),
                    "adjusted_close": pd.Series(self.prices[symbol], dtype="float64"),
                }
            )
            for symbol in snapshot.symbols
            if symbol in self.prices
        ]
        return pd.concat(frames, ignore_index=True)


def test_a_factor_created_by_name_runs_through_the_runner() -> None:
    """The point of the table: a caller that knows only the name gets a panel."""
    registry = default_factor_registry()
    provider = MemoryUniverseProvider()
    reader = MemoryMarketDataReader(
        {
            "AAPL": momentum_prices(start=100.0, end=130.0),
            "MSFT": momentum_prices(start=80.0, end=60.0),
        }
    )
    runner = FactorRunner(provider, reader)

    result = runner.run(registry.create("momentum_12_1"), AS_OF)

    assert isinstance(result, FactorRunResult)
    panel = result.panel
    assert panel.factor_name == "momentum_12_1"
    assert panel.factor_version == "1.0.0"
    assert panel.as_of_date == AS_OF
    assert panel.symbols == SYMBOLS

    frame = panel.frame
    values = dict(zip(frame["symbol"], frame["raw_value"], strict=True))
    assert values["AAPL"] == pytest.approx(0.30)
    assert values["MSFT"] == pytest.approx(-0.25)
    assert list(frame["is_eligible"]) == [True, True]
    assert result.quality_report is None

    assert provider.asked == [AS_OF]
    assert len(reader.requests) == 1


def test_two_runs_from_the_same_name_produce_the_same_panel() -> None:
    """``create`` is deterministic: a new object every time, the same numbers."""
    registry = default_factor_registry()
    prices = {"AAPL": momentum_prices(start=100.0, end=130.0), "MSFT": momentum_prices()}
    runner = FactorRunner(MemoryUniverseProvider(), MemoryMarketDataReader(prices))

    first = runner.run(registry.create("momentum_12_1"), AS_OF)
    second = runner.run(registry.create("momentum_12_1"), AS_OF)

    assert first.panel is not second.panel
    pd.testing.assert_frame_equal(first.panel.frame, second.panel.frame)


# ---------------------------------------------------------------------------
# The module's own boundary
# ---------------------------------------------------------------------------
REGISTRY_PATH = Path(registry_module.__file__)

#: Modules ``registry.py`` may import at module level: its own contract, its own
#: errors and the standard library.  No layer above or below it, no pandas.
ALLOWED_MODULE_IMPORTS: frozenset[str] = frozenset(
    {"__future__", "collections.abc", "factor_sys.contracts", "factor_sys.errors", "re"}
)

#: Imports allowed *inside* a function, and only there: the built-in factors, in
#: the default builder and nowhere else.
ALLOWED_FUNCTION_IMPORTS: dict[str, frozenset[str]] = {
    "default_factor_registry": frozenset({"factors.momentum"})
}

#: Module prefixes that would drag another layer into the mechanism.
FORBIDDEN_PREFIXES: tuple[str, ...] = (
    "data_sys",
    "pandas",
    "numpy",
    "pyarrow",
    "duckdb",
    "yfinance",
    "requests",
    "httpx",
    "socket",
    "urllib",
    "os",
    "pathlib",
    "shutil",
    "tempfile",
    "subprocess",
    "json",
    "factor_sys.market_data",
    "factor_sys.runner",
    "factor_sys.factor_store",
    "factor_sys.quality",
    "factor_sys.universe",
)

#: Calls that would make a pure, in-memory table impure.
FORBIDDEN_CALLS: frozenset[str] = frozenset(
    {"__import__", "compile", "connect", "eval", "exec", "input", "open", "print", "urlopen"}
)


def parse(path: Path) -> ast.Module:
    """The syntax tree of ``path``."""
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def scoped_imports(path: Path) -> list[tuple[str | None, str]]:
    """Every ``(enclosing function, module)`` pair named by an import statement."""
    found: list[tuple[str | None, str]] = []
    scope: list[str | None] = [None]

    class Visitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            scope.append(node.name)
            self.generic_visit(node)
            scope.pop()

        def visit_Import(self, node: ast.Import) -> None:
            found.extend((scope[-1], alias.name) for alias in node.names)

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            if node.module:
                found.append((scope[-1], node.module))

    Visitor().visit(parse(path))
    return found


def called_names(path: Path) -> set[str]:
    """Every function or method the module calls, exactly as written."""
    return {ast.unparse(node.func) for node in ast.walk(parse(path)) if isinstance(node, ast.Call)}


def referenced_names(path: Path) -> set[str]:
    """Every bare name the module's code mentions (docstrings excluded)."""
    return {node.id for node in ast.walk(parse(path)) if isinstance(node, ast.Name)}


def test_the_registry_imports_only_its_own_layer() -> None:
    """No data system, no reader, no runner, no store, no quality check, no pandas."""
    imports = scoped_imports(REGISTRY_PATH)

    module_level = {module for scope, module in imports if scope is None}
    assert module_level <= ALLOWED_MODULE_IMPORTS
    assert [module for _, module in imports if module.startswith(FORBIDDEN_PREFIXES)] == []
    for scope, module in imports:
        if scope is not None:
            allowed = ALLOWED_FUNCTION_IMPORTS.get(scope, frozenset())
            assert module in allowed, f"{module!r} may not be imported inside {scope}()"


def test_the_only_factor_import_lives_in_the_default_builder() -> None:
    """The formulas are reached from one explicit place, and from nowhere else."""
    factor_imports = [
        (scope, module)
        for scope, module in scoped_imports(REGISTRY_PATH)
        if module.startswith("factors")
    ]

    assert factor_imports == [("default_factor_registry", "factors.momentum")]


def test_the_registry_never_reads_writes_or_prints_anything() -> None:
    """A pure table: no file, no path, no print, no eval."""
    names = referenced_names(REGISTRY_PATH)

    assert called_names(REGISTRY_PATH) & FORBIDDEN_CALLS == set()
    assert not names & {"Path", "environ", "getenv", "os", "shutil", "subprocess", "tempfile"}


def test_the_registry_hard_codes_no_path_or_data_directory() -> None:
    """No test may depend on the repository's own ``data/`` tree."""
    source = REGISTRY_PATH.read_text(encoding="utf-8").lower()

    for text in ("data/", "data\\", "c:\\", "/home", "/users", ".parquet", ".csv", "os.environ"):
        assert text not in source


def test_the_registry_and_the_runner_do_not_know_each_other() -> None:
    """A new factor is one registration -- never an edit to the runner."""
    runner_names = referenced_names(Path(runner_module.__file__))

    assert not runner_names & {"FactorRegistry", "default_factor_registry", "FactorRegistryError"}
    assert not referenced_names(REGISTRY_PATH) & {"FactorRunner", "FactorRunResult"}


def test_the_registry_its_error_and_the_default_builder_are_exported() -> None:
    """The package root is the one place to import them from."""
    assert factor_sys.FactorRegistry is FactorRegistry
    assert factor_sys.FactorRegistryError is FactorRegistryError
    assert factor_sys.default_factor_registry is default_factor_registry
    assert {"FactorRegistry", "FactorRegistryError", "default_factor_registry"} <= set(
        factor_sys.__all__
    )
    assert registry_module.__all__ == [
        "FactorRegistry",
        "FactorRegistryError",
        "default_factor_registry",
    ]
