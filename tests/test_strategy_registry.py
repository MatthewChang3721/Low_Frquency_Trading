"""Tests for the explicit strategy registry (:mod:`strategy_sys.registry`).

Everything here is in memory and offline: the table is filled with tiny local
``Strategy`` doubles and closure factories.  The registry itself imports no
concrete rule, no data, no pandas and no factor layer, so most of this file needs
none of them; the two integration tests at the end register the *real*
:class:`~strategy_sys.strategies.TopNByFactorStrategy` to prove the flow, and the
boundary tests assert that the link exists only there, in the caller's hands.  No
``data/`` directory is read, no Parquet file is written and nothing touches the
network.

Covered in this order:

* an empty table: ``specs``, ``len``, ``find``, membership, ``repr``,
* registration and creation by the *whole* spec, a new instance per ``create``,
* the whole spec as the key: two factor versions of one rule, two multi-factor
  orderings and two multi-factor versions are all separate registrations,
* refusals: duplicate specs, non-callable factories, factories that raise (with
  their cause), factories that build the wrong thing, an invalid spec, an
  unhashable spec, requests that are not a spec, unregistered specs, drift,
* ``StrategyError`` from a factory travelling back unwrapped,
* the instance ledger: a singleton, the probe, a late replay, one instance under
  two specs (refused at the second registration, or at the create), identity (not
  equality) as the rule,
* ``specs`` sorted and read-only, ``find`` read-only,
* the exception type, the package surface and the absence of a default table,
* the architectural boundary of the module,
* the integration with the real rule.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

import strategy_sys
import strategy_sys.registry as registry_module
from strategy_sys import (
    FactorRequirement,
    Strategy,
    StrategyContext,
    StrategyContractError,
    StrategyError,
    StrategyRegistry,
    StrategyRegistryError,
    StrategyResult,
    StrategySpec,
)
from strategy_sys.strategies import TopNByFactorStrategy

#: Exact factor identities: one, the same name on another version, and two more.
MOMENTUM = FactorRequirement(factor_name="momentum_12_1", factor_version="1.0.0")
MOMENTUM_V2 = FactorRequirement(factor_name="momentum_12_1", factor_version="2.0.0")
VALUE = FactorRequirement(factor_name="book_to_price", factor_version="2.0.0")
QUALITY = FactorRequirement(factor_name="gross_profitability", factor_version="1.0.0")

#: The identity the doubles below declare.
SPEC = StrategySpec(
    name="top_n_momentum",
    version="1.0.0",
    required_factors=(MOMENTUM,),
)

#: The same rule name and version, computed from another version of the factor.
SPEC_ON_MOMENTUM_V2 = StrategySpec(
    name="top_n_momentum",
    version="1.0.0",
    required_factors=(MOMENTUM_V2,),
)


# ---------------------------------------------------------------------------
# Doubles to register: tiny, local, and never computed
# ---------------------------------------------------------------------------
def make_strategy_class(spec: StrategySpec) -> type[Strategy]:
    """A minimal concrete strategy whose spec is ``spec``.

    A class per call, so every test starts from a fresh identity.  A class *is*
    the normal factory: calling it builds a new instance every time.
    """

    class Built(Strategy):
        def __init__(self) -> None:
            self.created = True

        @property
        def spec(self) -> StrategySpec:
            return spec

        def compute(self, context: StrategyContext) -> StrategyResult:
            raise NotImplementedError("this test double never scores anything")

    return Built


def make_spec(
    name: str = "top_n_momentum",
    *,
    version: str = "1.0.0",
    required: tuple[FactorRequirement, ...] = (MOMENTUM,),
) -> StrategySpec:
    """A spec with a chosen identity, so tests read as the strategy they mean."""
    return StrategySpec(name=name, version=version, required_factors=required)


def forged_spec(**overrides: object) -> StrategySpec:
    """A ``StrategySpec`` built *past* its own validation, to test the guard."""
    spec = object.__new__(StrategySpec)
    values: dict[str, object] = {
        "name": "top_n_momentum",
        "version": "1.0.0",
        "required_factors": (MOMENTUM,),
    }
    values.update(overrides)
    for field, value in values.items():
        object.__setattr__(spec, field, value)
    return spec


def make_chameleon_class(initial: StrategySpec) -> type[Strategy]:
    """A strategy whose ``spec`` can be re-labelled after it was built.

    Only a mutable spec lets a test move one object between two identities,
    which is exactly the manoeuvre the instance ledger has to catch.
    """

    class Chameleon(Strategy):
        def __init__(self) -> None:
            self.current = initial

        @property
        def spec(self) -> StrategySpec:
            return self.current

        def compute(self, context: StrategyContext) -> StrategyResult:
            raise NotImplementedError("this test double never scores anything")

    return Chameleon


def register_all(*specs: StrategySpec) -> StrategyRegistry:
    """A registry holding one double per spec, registered in the given order."""
    registry = StrategyRegistry()
    for spec in specs:
        registry.register(make_strategy_class(spec))
    return registry


# ---------------------------------------------------------------------------
# An empty table
# ---------------------------------------------------------------------------
def test_a_new_registry_is_empty() -> None:
    registry = StrategyRegistry()

    assert len(registry) == 0
    assert registry.specs == ()
    assert registry.find("top_n_momentum") == ()
    assert SPEC not in registry
    assert repr(registry) == "StrategyRegistry()"


@pytest.mark.parametrize("value", [None, "top_n_momentum", (), SPEC.required_factors, object()])
def test_membership_of_something_that_is_not_a_spec_is_simply_false(value: object) -> None:
    """``in`` never raises: an object that cannot be a spec is not in the table."""
    registry = register_all(SPEC)

    assert value not in registry
    assert SPEC in registry


def test_find_returns_nothing_for_a_name_that_is_registered_nowhere() -> None:
    registry = register_all(SPEC)

    assert registry.find("top_10_by_momentum_12_1") == ()
    assert registry.find("top_n_momentum") == (SPEC,)


@pytest.mark.parametrize("name", ["", " ", "Top_N_Momentum", "top-n-momentum", "top_n_momentum "])
def test_an_illegal_name_is_refused_instead_of_normalized(name: str) -> None:
    """A name is either registered as written or it is a typo -- never repaired."""
    registry = register_all(SPEC)

    with pytest.raises(StrategyRegistryError) as error:
        registry.find(name)

    assert "not a legal strategy name" in str(error.value)


@pytest.mark.parametrize("name", [None, 42, SPEC, ("top_n_momentum",)])
def test_find_needs_a_string(name: object) -> None:
    registry = register_all(SPEC)

    with pytest.raises(StrategyRegistryError, match="must be a string"):
        registry.find(name)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Registering and creating by the whole spec
# ---------------------------------------------------------------------------
def test_a_registered_strategy_can_be_created_by_its_whole_spec() -> None:
    built = make_strategy_class(SPEC)
    registry = StrategyRegistry()

    spec = registry.register(built)
    strategy = registry.create(spec)

    assert spec == built().spec
    assert spec == SPEC
    assert isinstance(strategy, built)
    assert strategy.spec == spec
    assert len(registry) == 1
    assert registry.specs == (SPEC,)


def test_the_registration_returns_the_strategys_own_identity() -> None:
    """The table never names a strategy itself: the spec is the authority."""
    registry = StrategyRegistry()
    spec = make_spec("top_5_by_book_to_price", required=(VALUE,))

    returned = registry.register(lambda: make_strategy_class(spec)())

    assert returned is spec
    assert returned.name == "top_5_by_book_to_price"
    assert returned.required_factors == (VALUE,)


def test_an_equal_spec_built_elsewhere_creates_the_registered_strategy() -> None:
    """A spec is a value: the same identity passed as another object is the same."""
    registry = register_all(SPEC)
    equivalent = StrategySpec(
        name="top_n_momentum",
        version="1.0.0",
        required_factors=(FactorRequirement("momentum_12_1", "1.0.0"),),
    )

    assert equivalent is not SPEC
    assert equivalent == SPEC
    assert equivalent in registry
    assert registry.create(equivalent).spec == SPEC


def test_every_create_builds_a_new_instance() -> None:
    registry = register_all(SPEC)

    first = registry.create(SPEC)
    second = registry.create(SPEC)

    assert first is not second
    assert first.spec == second.spec == SPEC


def test_the_factory_is_called_once_at_registration_and_once_per_create() -> None:
    built = make_strategy_class(SPEC)
    calls: list[int] = []

    def factory() -> Strategy:
        calls.append(len(calls))
        return built()

    registry = StrategyRegistry()
    registry.register(factory)

    assert len(calls) == 1  # the probe
    registry.create(SPEC)
    registry.create(SPEC)
    assert len(calls) == 3


def test_a_class_factory_builds_a_fresh_strategy_every_time() -> None:
    built = make_strategy_class(SPEC)
    registry = StrategyRegistry()
    registry.register(built)

    strategies = [registry.create(SPEC) for _ in range(3)]

    assert len({id(strategy) for strategy in strategies}) == 3
    assert all(isinstance(strategy, built) for strategy in strategies)


# ---------------------------------------------------------------------------
# The whole spec is the key
# ---------------------------------------------------------------------------
def test_two_factor_versions_of_the_same_rule_are_two_registrations() -> None:
    """Same name, same version, different factor version: two strategies."""
    registry = register_all(SPEC, SPEC_ON_MOMENTUM_V2)

    assert SPEC.name == SPEC_ON_MOMENTUM_V2.name == "top_n_momentum"
    assert SPEC.version == SPEC_ON_MOMENTUM_V2.version == "1.0.0"
    assert SPEC != SPEC_ON_MOMENTUM_V2
    assert len(registry) == 2
    assert set(registry.specs) == {SPEC, SPEC_ON_MOMENTUM_V2}
    assert registry.find("top_n_momentum") == (SPEC, SPEC_ON_MOMENTUM_V2)


def test_creating_one_factor_version_never_returns_the_other() -> None:
    registry = register_all(SPEC, SPEC_ON_MOMENTUM_V2)

    on_v1 = registry.create(SPEC)
    on_v2 = registry.create(SPEC_ON_MOMENTUM_V2)

    assert on_v1.spec == SPEC
    assert on_v1.spec.required_factors == (MOMENTUM,)
    assert on_v2.spec == SPEC_ON_MOMENTUM_V2
    assert on_v2.spec.required_factors == (MOMENTUM_V2,)
    assert type(on_v1) is not type(on_v2)


def test_a_multi_factor_declaration_is_keyed_by_its_whole_dependency_tuple() -> None:
    """The same blend name and version, read in the other order, is a third rule."""
    momentum_then_value = make_spec("blend", required=(MOMENTUM, VALUE))
    value_then_momentum = make_spec("blend", required=(VALUE, MOMENTUM))
    registry = register_all(momentum_then_value, value_then_momentum)

    assert momentum_then_value.name == value_then_momentum.name
    assert momentum_then_value.version == value_then_momentum.version
    assert momentum_then_value != value_then_momentum
    assert len(registry) == 2
    assert registry.find("blend") == (value_then_momentum, momentum_then_value)
    assert registry.create(value_then_momentum).spec == value_then_momentum
    assert registry.create(momentum_then_value).spec == momentum_then_value


def test_a_multi_factor_version_change_is_a_different_registration() -> None:
    on_v1 = make_spec("blend", required=(MOMENTUM, VALUE))
    on_v2 = make_spec("blend", required=(MOMENTUM, QUALITY))
    registry = register_all(on_v1, on_v2)

    assert len(registry) == 2
    assert [str(spec.required_factors[1]) for spec in registry.specs] == [
        "book_to_price/2.0.0",
        "gross_profitability/1.0.0",
    ]
    assert registry.create(on_v2).spec.required_factors == (MOMENTUM, QUALITY)
    assert registry.create(on_v1).spec.required_factors == (MOMENTUM, VALUE)


def test_one_registry_never_sees_another_registrys_registrations() -> None:
    first = register_all(SPEC)
    second = StrategyRegistry()

    assert SPEC in first
    assert SPEC not in second
    assert second.specs == ()
    with pytest.raises(StrategyRegistryError, match="not registered"):
        second.create(SPEC)


# ---------------------------------------------------------------------------
# Refusals: duplicates, types, factories
# ---------------------------------------------------------------------------
def test_a_duplicate_spec_is_refused_and_never_overwrites() -> None:
    first = make_strategy_class(SPEC)
    second = make_strategy_class(SPEC)
    registry = StrategyRegistry()
    registry.register(first)

    with pytest.raises(StrategyRegistryError) as error:
        registry.register(second)

    message = str(error.value)
    assert "already registered" in message
    assert "top_n_momentum/1.0.0" in message and "momentum_12_1/1.0.0" in message
    assert len(registry) == 1
    assert isinstance(registry.create(SPEC), first)  # the first factory still wins


def test_registering_the_same_factory_twice_is_still_a_duplicate() -> None:
    built = make_strategy_class(SPEC)
    registry = StrategyRegistry()
    registry.register(built)

    with pytest.raises(StrategyRegistryError, match="already registered"):
        registry.register(built)

    assert len(registry) == 1


@pytest.mark.parametrize("factory", [None, 42, "strategy", SPEC, object(), (), 1.0])
def test_a_non_callable_factory_is_refused(factory: object) -> None:
    registry = StrategyRegistry()

    with pytest.raises(StrategyRegistryError) as error:
        registry.register(factory)  # type: ignore[arg-type]

    assert "must be callable" in str(error.value)
    assert len(registry) == 0


def test_a_factory_that_raises_is_reported_with_its_cause() -> None:
    def factory() -> Strategy:
        raise ValueError("no parameters at all")

    registry = StrategyRegistry()
    with pytest.raises(StrategyRegistryError) as error:
        registry.register(factory)

    assert "ValueError" in str(error.value)
    assert "no parameters at all" in str(error.value)
    assert isinstance(error.value.__cause__, ValueError)
    assert len(registry) == 0


def test_a_factory_that_fails_only_when_creating_keeps_its_cause() -> None:
    calls: list[int] = []
    built = make_strategy_class(SPEC)

    def factory() -> Strategy:
        calls.append(len(calls))
        if len(calls) > 1:
            raise RuntimeError("the second call cannot happen")
        return built()

    registry = StrategyRegistry()
    registry.register(factory)

    with pytest.raises(StrategyRegistryError) as error:
        registry.create(SPEC)

    assert "RuntimeError" in str(error.value)
    assert isinstance(error.value.__cause__, RuntimeError)
    assert len(registry) == 1


@pytest.mark.parametrize("answer", [None, 42, "strategy", SPEC, object(), [SPEC]])
def test_a_factory_that_does_not_build_a_strategy_is_refused(answer: object) -> None:
    registry = StrategyRegistry()

    with pytest.raises(StrategyRegistryError) as error:
        registry.register(lambda: answer)  # type: ignore[return-value, arg-type]

    assert "must build a Strategy" in str(error.value)
    assert len(registry) == 0


@pytest.mark.parametrize("declared", [None, "top_n_momentum", {}, (SPEC,)])
def test_a_strategy_without_a_valid_spec_is_refused(declared: object) -> None:
    class Bad(Strategy):
        spec = declared

        def compute(self, context: StrategyContext) -> StrategyResult:
            raise NotImplementedError

    registry = StrategyRegistry()
    with pytest.raises(StrategyRegistryError) as error:
        registry.register(Bad)

    assert "not a StrategySpec" in str(error.value)
    assert len(registry) == 0


def test_a_strategy_whose_spec_cannot_be_read_is_refused() -> None:
    """A broken ``spec`` property is a broken factory, reported as one."""

    class Broken(Strategy):
        @property
        def spec(self) -> StrategySpec:
            raise AttributeError("no declaration at all")

        def compute(self, context: StrategyContext) -> StrategyResult:
            raise NotImplementedError

    registry = StrategyRegistry()
    with pytest.raises(StrategyRegistryError) as error:
        registry.register(Broken)

    assert "could not be read" in str(error.value)
    assert "AttributeError" in str(error.value)
    assert isinstance(error.value.__cause__, AttributeError)
    assert len(registry) == 0


def test_a_spec_that_cannot_be_a_table_key_is_refused() -> None:
    """A spec forged past validation with a list of factors is not a key."""
    unhashable = forged_spec(required_factors=[MOMENTUM])
    registry = StrategyRegistry()

    with pytest.raises(StrategyRegistryError) as error:
        registry.register(lambda: make_strategy_class(unhashable)())

    assert "cannot be a table key" in str(error.value)
    assert len(registry) == 0
    assert unhashable not in registry
    with pytest.raises(StrategyRegistryError, match="not hashable"):
        registry.create(unhashable)


@pytest.mark.parametrize(
    "value", ["top_n_momentum", None, 42, (SPEC,), {"name": "top_n_momentum"}, SPEC.name]
)
def test_create_needs_a_spec_not_a_name(value: object) -> None:
    registry = register_all(SPEC)

    with pytest.raises(StrategyRegistryError) as error:
        registry.create(value)  # type: ignore[arg-type]

    message = str(error.value)
    assert "whole StrategySpec" in message
    assert "find(name)" in message
    assert len(registry) == 1


def test_create_of_an_unregistered_spec_lists_what_is_registered() -> None:
    registered = make_spec("top_5_by_book_to_price", required=(VALUE,))
    registry = register_all(registered)
    unknown = make_spec("top_3_by_gross_profitability", required=(QUALITY,))

    with pytest.raises(StrategyRegistryError) as error:
        registry.create(unknown)

    message = str(error.value)
    assert "not registered" in message
    assert "top_3_by_gross_profitability/1.0.0" in message
    assert "gross_profitability/1.0.0" in message
    assert "top_5_by_book_to_price/1.0.0" in message  # what the table does hold
    assert len(registry) == 1


def test_create_points_at_the_registered_factor_versions_of_the_same_name() -> None:
    """A bare name is not enough: the message says which spec is registered."""
    registry = register_all(SPEC)

    with pytest.raises(StrategyRegistryError) as error:
        registry.create(SPEC_ON_MOMENTUM_V2)

    message = str(error.value)
    assert "top_n_momentum/1.0.0" in message
    assert "momentum_12_1/2.0.0" in message  # what was asked for
    assert "registered for 1 factor version" in message
    assert "momentum_12_1/1.0.0" in message  # what the table holds
    assert "find(name)" in message


def test_a_factory_that_changes_what_it_builds_is_caught_at_create() -> None:
    holder = {"spec": SPEC}

    class Drifting(Strategy):
        @property
        def spec(self) -> StrategySpec:
            return holder["spec"]

        def compute(self, context: StrategyContext) -> StrategyResult:
            raise NotImplementedError

    registry = StrategyRegistry()
    registry.register(Drifting)
    registry.create(SPEC)  # fine while the declaration is unchanged

    holder["spec"] = SPEC_ON_MOMENTUM_V2
    with pytest.raises(StrategyRegistryError) as error:
        registry.create(SPEC)

    message = str(error.value)
    assert "now builds" in message
    assert "momentum_12_1/1.0.0" in message and "momentum_12_1/2.0.0" in message
    assert len(registry) == 1


# ---------------------------------------------------------------------------
# Strategy errors travel untouched
# ---------------------------------------------------------------------------
def test_a_strategy_error_from_the_factory_is_not_wrapped() -> None:
    def factory() -> Strategy:
        raise StrategyContractError("invalid TopNByFactorStrategy", ["top_n must be positive"])

    registry = StrategyRegistry()
    with pytest.raises(StrategyContractError) as error:
        registry.register(factory)

    assert error.value.problems == ["top_n must be positive"]
    assert error.value.__cause__ is None


def test_a_strategy_error_at_create_time_keeps_its_type() -> None:
    built = make_strategy_class(SPEC)
    calls: list[int] = []

    def factory() -> Strategy:
        calls.append(len(calls))
        if len(calls) > 1:
            raise StrategyError("the panel is missing")
        return built()

    registry = StrategyRegistry()
    registry.register(factory)

    with pytest.raises(StrategyError, match="the panel is missing") as error:
        registry.create(SPEC)

    assert not isinstance(error.value, StrategyRegistryError)


# ---------------------------------------------------------------------------
# One instance per create
# ---------------------------------------------------------------------------
def test_a_singleton_factory_is_refused_at_the_first_create() -> None:
    """A closure over one pre-built strategy is not a factory."""
    singleton = make_strategy_class(SPEC)()
    registry = StrategyRegistry()

    spec = registry.register(lambda: singleton)  # legal: it is a Strategy

    with pytest.raises(StrategyRegistryError) as error:
        registry.create(spec)

    assert "reused" in str(error.value)
    assert "build a new strategy for every registration and every create()" in str(error.value)
    assert len(registry) == 1


def test_the_probe_counts_as_used_even_after_good_creates() -> None:
    probe = make_strategy_class(SPEC)()
    fresh = make_strategy_class(SPEC)
    calls: list[int] = []

    def factory() -> Strategy:
        calls.append(len(calls) + 1)
        return probe if calls[-1] in {1, 4} else fresh()

    registry = StrategyRegistry()
    spec = registry.register(factory)  # call 1 hands over the probe

    registry.create(spec)  # call 2
    registry.create(spec)  # call 3
    with pytest.raises(StrategyRegistryError, match="reused"):
        registry.create(spec)  # call 4 replays the probe

    assert len(registry) == 1


def test_a_late_repeat_of_an_already_returned_instance_is_refused() -> None:
    """Every instance is on record, not just the one handed out last."""
    fresh = make_strategy_class(SPEC)
    handed_back: list[Strategy] = []
    calls: list[int] = []

    def factory() -> Strategy:
        calls.append(len(calls) + 1)
        if calls[-1] == 1:  # the registration probe
            return fresh()
        if calls[-1] == 5:  # four calls later: replay the very first answer
            return handed_back[0]
        strategy = fresh()
        handed_back.append(strategy)
        return strategy

    registry = StrategyRegistry()
    spec = registry.register(factory)  # call 1: the probe
    first = registry.create(spec)  # call 2
    registry.create(spec)  # call 3
    registry.create(spec)  # call 4
    assert handed_back[0] is first

    with pytest.raises(StrategyRegistryError, match="reused"):
        registry.create(spec)  # call 5

    assert len(registry) == 1


def test_one_instance_cannot_be_laundered_through_a_second_spec() -> None:
    """The ledger is global: another spec is not a way around the guard.

    Registration writes to the ledger, so it must read from it too.  A probe
    that comes back re-labelled is refused *at that second register*, not later
    at some create, and the refusal leaves the table exactly as it was.
    """
    chameleon = make_chameleon_class(SPEC)()
    registry = StrategyRegistry()
    first = registry.register(lambda: chameleon)  # the probe, under SPEC
    assert first == SPEC

    chameleon.current = SPEC_ON_MOMENTUM_V2
    with pytest.raises(StrategyRegistryError, match="reused") as error:
        registry.register(lambda: chameleon)  # the same object, re-labelled

    assert "already used" in str(error.value)
    assert registry.specs == (SPEC,)
    assert len(registry) == 1
    assert SPEC_ON_MOMENTUM_V2 not in registry
    # the second spec was never added, so it cannot be created either
    with pytest.raises(StrategyRegistryError, match="is not registered"):
        registry.create(SPEC_ON_MOMENTUM_V2)
    # and the refusal poisoned nothing: the table still accepts a real factory
    registry.register(make_chameleon_class(SPEC_ON_MOMENTUM_V2))
    assert registry.specs == (SPEC, SPEC_ON_MOMENTUM_V2)


def test_an_instance_already_created_cannot_become_a_registration_probe() -> None:
    """Registration is no loophole either: a created strategy is on record too."""
    chameleon = make_chameleon_class(SPEC)()
    throwaway = make_chameleon_class(SPEC)()
    calls: list[int] = []

    def factory() -> Strategy:
        calls.append(len(calls) + 1)
        return throwaway if calls[-1] == 1 else chameleon

    registry = StrategyRegistry()
    registry.register(factory)  # call 1: a throwaway probe
    created = registry.create(SPEC)  # call 2: the chameleon is handed out
    assert created is chameleon

    chameleon.current = SPEC_ON_MOMENTUM_V2
    with pytest.raises(StrategyRegistryError, match="reused"):
        registry.register(lambda: chameleon)  # the same object, as a new probe

    assert registry.specs == (SPEC,)
    assert len(registry) == 1
    assert SPEC_ON_MOMENTUM_V2 not in registry


def test_identity_not_equality_decides_whether_an_instance_is_new() -> None:
    """Two equal, unhashable strategies are two instances, and both are legal."""

    class EqualStrategy(Strategy):
        spec = SPEC

        def __eq__(self, other: object) -> bool:
            return isinstance(other, Strategy)

        def compute(self, context: StrategyContext) -> StrategyResult:
            raise NotImplementedError

    registry = StrategyRegistry()
    spec = registry.register(EqualStrategy)

    first = registry.create(spec)
    second = registry.create(spec)

    assert first is not second
    assert first == second  # equal by ==, and still accepted
    with pytest.raises(TypeError):
        hash(first)  # not hashable either
    assert len(registry) == 1


def test_a_refused_reuse_leaves_the_table_and_other_registries_alone() -> None:
    singleton = make_strategy_class(SPEC)()
    other_spec = make_spec("top_5_by_value", required=(VALUE,))

    registry = StrategyRegistry()
    other = register_all(other_spec)
    spec = registry.register(lambda: singleton)
    registry.register(make_strategy_class(other_spec))

    with pytest.raises(StrategyRegistryError, match="reused"):
        registry.create(spec)

    assert len(registry) == 2
    assert set(registry.specs) == {SPEC, other_spec}
    assert isinstance(registry.create(other_spec), Strategy)  # still usable
    assert other.specs == (other_spec,)  # untouched
    assert len(other) == 1


# ---------------------------------------------------------------------------
# specs and find are sorted and read-only
# ---------------------------------------------------------------------------
def test_specs_are_sorted_by_name_version_and_factors() -> None:
    momentum_v1 = make_spec("top_10_by_momentum_12_1", required=(MOMENTUM,))
    momentum_v2 = make_spec("top_10_by_momentum_12_1", required=(MOMENTUM_V2,))
    momentum_next = make_spec("top_10_by_momentum_12_1", version="2.0.0", required=(MOMENTUM,))
    value = make_spec("top_5_by_value", required=(VALUE,))

    registry = register_all(value, momentum_next, momentum_v2, momentum_v1)

    assert registry.specs == (momentum_v1, momentum_v2, momentum_next, value)
    assert [spec.version for spec in registry.specs] == ["1.0.0", "1.0.0", "2.0.0", "1.0.0"]


def test_specs_and_find_hand_out_unmodifiable_tuples() -> None:
    registry = register_all(SPEC, SPEC_ON_MOMENTUM_V2)

    specs = registry.specs
    found = registry.find("top_n_momentum")

    assert isinstance(specs, tuple)
    assert isinstance(found, tuple)
    assert specs is not registry.specs  # a fresh tuple per call
    with pytest.raises(TypeError):
        specs[0] = SPEC  # type: ignore[index]
    with pytest.raises(TypeError):
        found[0] = SPEC  # type: ignore[index]


def test_repr_names_every_registered_identity() -> None:
    registry = register_all(SPEC, SPEC_ON_MOMENTUM_V2)

    text = repr(registry)

    assert text.startswith("StrategyRegistry(specs=(")
    assert "top_n_momentum/1.0.0 (requires momentum_12_1/1.0.0)" in text
    assert "momentum_12_1/2.0.0" in text


# ---------------------------------------------------------------------------
# The error type and the package surface
# ---------------------------------------------------------------------------
def test_a_registry_error_is_a_strategy_error_only() -> None:
    assert issubclass(StrategyRegistryError, StrategyError)
    assert not issubclass(StrategyRegistryError, StrategyContractError)
    assert not issubclass(StrategyRegistryError, ValueError)
    assert not issubclass(StrategyRegistryError, KeyError)


def test_the_registry_is_exported_from_the_package_root() -> None:
    assert strategy_sys.StrategyRegistry is StrategyRegistry
    assert strategy_sys.StrategyRegistryError is StrategyRegistryError
    assert {"StrategyRegistry", "StrategyRegistryError"} <= set(strategy_sys.__all__)
    assert registry_module.__all__ == ["StrategyRegistry"]


def test_there_is_no_default_table_and_no_global_registry() -> None:
    """Which rule a project runs is a research decision, never an import default."""
    assert not hasattr(registry_module, "default_strategy_registry")
    assert not hasattr(strategy_sys, "default_strategy_registry")
    assert not hasattr(StrategyRegistry, "unregister")
    assert not hasattr(StrategyRegistry, "register_many")
    assert not hasattr(StrategyRegistry, "create_by_name")


def test_create_takes_a_spec_and_register_takes_a_factory() -> None:
    """No name-only entry point can be added without breaking this test."""
    assert list(inspect.signature(StrategyRegistry.create).parameters) == ["self", "spec"]
    assert list(inspect.signature(StrategyRegistry.register).parameters) == ["self", "factory"]


# ---------------------------------------------------------------------------
# The module's own boundary
# ---------------------------------------------------------------------------
REGISTRY_PATH = Path(registry_module.__file__)

#: Modules ``registry.py`` may import at module level: its own contract, its own
#: errors and the standard library.  No concrete rule, no data, no pandas, no
#: factor layer -- the strategy arrives as a factory the caller passes in.
ALLOWED_MODULE_IMPORTS: frozenset[str] = frozenset(
    {"__future__", "collections.abc", "re", "strategy_sys.contracts", "strategy_sys.errors"}
)

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
    "exchange_calendars",
    "factor_sys",
    "factors",
    "strategy_sys.strategies",
)

#: Calls that would make a pure, in-memory table impure.
FORBIDDEN_CALLS: frozenset[str] = frozenset(
    {"__import__", "compile", "connect", "eval", "exec", "input", "open", "print", "urlopen"}
)

#: Names that would mean the table knows a concrete rule, a runner or a portfolio.
FORBIDDEN_NAMES: frozenset[str] = frozenset(
    {
        "TopNByFactorStrategy",
        "MomentumTopNStrategy",
        "StrategyRunner",
        "StrategyResult",
        "StrategyContext",
        "FactorStore",
        "FactorRunner",
        "FactorPanel",
        "MarketDataReader",
        "PortfolioConstructor",
        "load_panels",
        "run_strategy",
    }
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
    """No data system, no factor layer, no concrete rule, no pandas, no runner."""
    imports = scoped_imports(REGISTRY_PATH)

    module_level = {module for scope, module in imports if scope is None}
    assert module_level <= ALLOWED_MODULE_IMPORTS
    assert [module for _, module in imports if module.startswith(FORBIDDEN_PREFIXES)] == []
    assert [scope for scope, _ in imports if scope is not None] == []  # no lazy imports


def test_the_registry_knows_no_concrete_strategy() -> None:
    """A rule arrives as a factory: no code path here names or reaches one.

    The module docstring may name a rule while explaining what a caller no longer
    has to import, so the check reads the tree rather than the text: a name or an
    attribute that *executes* must never be a concrete strategy.
    """
    tree = parse(REGISTRY_PATH)

    assert referenced_names(REGISTRY_PATH) & FORBIDDEN_NAMES == set()
    assert {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)} & (
        FORBIDDEN_NAMES
    ) == set()


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


# ---------------------------------------------------------------------------
# Integration: the real rule, registered by the caller
# ---------------------------------------------------------------------------
def test_two_factor_versions_of_top_n_are_registered_and_created_separately() -> None:
    """The whole spec is what keeps the factor-version chain intact."""
    registry = StrategyRegistry()

    momentum_v1 = registry.register(
        lambda: TopNByFactorStrategy(FactorRequirement("momentum_12_1", "1.0.0"), top_n=10)
    )
    momentum_v2 = registry.register(
        lambda: TopNByFactorStrategy(FactorRequirement("momentum_12_1", "2.0.0"), top_n=10)
    )

    assert momentum_v1 != momentum_v2
    assert momentum_v1.name == momentum_v2.name == "top_10_by_momentum_12_1"
    assert momentum_v1.version == momentum_v2.version == "1.0.0"
    assert len(registry) == 2
    assert registry.find("top_10_by_momentum_12_1") == (momentum_v1, momentum_v2)

    on_v2 = registry.create(momentum_v2)
    on_v1 = registry.create(momentum_v1)

    assert isinstance(on_v2, TopNByFactorStrategy)
    assert on_v2.spec == momentum_v2
    assert on_v2.factor_requirement == FactorRequirement("momentum_12_1", "2.0.0")
    assert on_v1.spec == momentum_v1
    assert on_v1.factor_requirement == FactorRequirement("momentum_12_1", "1.0.0")


def test_a_registry_created_strategy_keeps_its_whole_identity() -> None:
    registry = StrategyRegistry()
    spec = registry.register(
        lambda: TopNByFactorStrategy(FactorRequirement("book_to_price", "2.0.0"), top_n=5)
    )

    strategy = registry.create(spec)

    assert spec.name == "top_5_by_book_to_price"
    assert spec.version == "1.0.0"
    assert [str(requirement) for requirement in spec.required_factors] == ["book_to_price/2.0.0"]
    assert [str(requirement) for requirement in strategy.spec.required_factors] == [
        "book_to_price/2.0.0"
    ]
    assert strategy.top_n == 5
    assert registry.find("top_5_by_book_to_price") == (spec,)
    assert registry.find("top_10_by_momentum_12_1") == ()
