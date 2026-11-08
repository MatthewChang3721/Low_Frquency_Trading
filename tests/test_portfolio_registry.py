"""Tests for the explicit construction registry (:mod:`portfolio_sys.registry`).

Everything here is in memory and offline: the table is filled with tiny local
``PortfolioConstructor`` doubles and closure factories.  The registry itself imports no
concrete rule, no data, no pandas, no factor layer and no strategy layer, so most of
this file needs none of them; the last section registers the *real*
:class:`~portfolio_sys.constructors.CappedScoreWeightedConstructor` to prove the flow,
and the boundary tests assert that the link exists only there, in the caller's hands.
Nothing here builds a weight: no context is constructed, no ``construct()`` is called,
no ``data/`` directory is read and nothing touches the network.

Covered in this order:

* an empty table: ``specs``, ``len``, ``find``, membership, ``repr``,
* registration and creation by the *whole* spec, a new instance per ``create``, and the
  registry never calling ``construct()``,
* the whole spec as the key: two versions of one rule, ``find`` listing both, and
  constraints / bare names refused as keys,
* refusals: duplicate specs, non-callable factories, constructor instances passed as
  factories, factories that raise (with their cause), factories that build the wrong
  thing, an invalid spec, an unhashable spec, requests that are not a spec, unregistered
  specs, drift,
* ``PortfolioError`` from a factory travelling back unwrapped,
* the instance ledger: a singleton, the probe, a late replay, one instance under two
  specs, identity (not equality) as the rule,
* ``specs`` sorted and read-only, ``find`` read-only,
* the exception type, the package surface, the public docstrings and ``__all__``,
* the architectural boundary of the module and of its neighbours,
* the integration with the real rule.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

import portfolio_sys
import portfolio_sys.registry as registry_module
from portfolio_sys import (
    PortfolioConstraints,
    PortfolioConstructionError,
    PortfolioConstructionSpec,
    PortfolioConstructor,
    PortfolioConstructorRegistry,
    PortfolioContext,
    PortfolioContractError,
    PortfolioError,
    PortfolioRegistryError,
    TargetPortfolio,
)
from portfolio_sys.constructors import CappedScoreWeightedConstructor

#: The identity the doubles below declare by default: the first real rule's own spec.
SPEC = PortfolioConstructionSpec(name="capped_score_weighted", version="1.0.0")

#: The same rule name, one version later: a different rule, a different entry.
SPEC_V2 = PortfolioConstructionSpec(name="capped_score_weighted", version="2.0.0")

#: Another rule name entirely.
EQUAL = PortfolioConstructionSpec(name="equal_weight", version="1.0.0")


# ---------------------------------------------------------------------------
# Doubles to register: tiny, local, and never computed
# ---------------------------------------------------------------------------
def make_constructor_class(spec: PortfolioConstructionSpec) -> type[PortfolioConstructor]:
    """A minimal concrete constructor whose spec is ``spec``.

    A class per call, so every test starts from a fresh identity.  A class *is* the
    normal factory: calling it builds a new instance every time.
    """

    class Built(PortfolioConstructor):
        def __init__(self) -> None:
            self.created = True

        @property
        def spec(self) -> PortfolioConstructionSpec:
            return spec

        def construct(self, context: PortfolioContext) -> TargetPortfolio:
            raise NotImplementedError("this test double never builds weights")

    return Built


def make_spec(
    name: str = "capped_score_weighted",
    *,
    version: str = "1.0.0",
) -> PortfolioConstructionSpec:
    """A spec with a chosen identity, so tests read as the rule they mean."""
    return PortfolioConstructionSpec(name=name, version=version)


def forged_spec(**overrides: object) -> PortfolioConstructionSpec:
    """A ``PortfolioConstructionSpec`` built *past* its own validation, to test the guard."""
    spec = object.__new__(PortfolioConstructionSpec)
    values: dict[str, object] = {"name": "capped_score_weighted", "version": "1.0.0"}
    values.update(overrides)
    for field, value in values.items():
        object.__setattr__(spec, field, value)
    return spec


def make_chameleon_class(
    initial: PortfolioConstructionSpec,
) -> type[PortfolioConstructor]:
    """A constructor whose ``spec`` can be re-labelled after it was built.

    Only a mutable spec lets a test move one object between two identities, which is
    exactly the manoeuvre the instance ledger has to catch.
    """

    class Chameleon(PortfolioConstructor):
        def __init__(self) -> None:
            self.current = initial

        @property
        def spec(self) -> PortfolioConstructionSpec:
            return self.current

        def construct(self, context: PortfolioContext) -> TargetPortfolio:
            raise NotImplementedError("this test double never builds weights")

    return Chameleon


def register_all(*specs: PortfolioConstructionSpec) -> PortfolioConstructorRegistry:
    """A registry holding one double per spec, registered in the given order."""
    registry = PortfolioConstructorRegistry()
    for spec in specs:
        registry.register(make_constructor_class(spec))
    return registry


# ---------------------------------------------------------------------------
# An empty table
# ---------------------------------------------------------------------------
def test_a_new_registry_is_empty() -> None:
    registry = PortfolioConstructorRegistry()

    assert len(registry) == 0
    assert registry.specs == ()
    assert registry.find("capped_score_weighted") == ()
    assert SPEC not in registry
    assert repr(registry) == "PortfolioConstructorRegistry()"


@pytest.mark.parametrize("value", [None, "capped_score_weighted", (), ("a", "b"), object()])
def test_membership_of_something_that_is_not_a_spec_is_simply_false(value: object) -> None:
    """``in`` never raises: an object that cannot be a spec is not in the table."""
    registry = register_all(SPEC)

    assert value not in registry
    assert SPEC in registry


def test_membership_of_an_unhashable_spec_is_simply_false() -> None:
    """``in`` never raises, even for a forged spec that cannot be a key."""
    registry = register_all(SPEC)

    assert forged_spec(version=["1.0.0"]) not in registry
    assert forged_spec() in registry  # the same identity, hashable: present


def test_find_returns_nothing_for_a_name_that_is_registered_nowhere() -> None:
    registry = register_all(SPEC)

    assert registry.find("equal_weight") == ()
    assert registry.find("capped_score_weighted") == (SPEC,)


@pytest.mark.parametrize("name", ["", " ", "Capped_Score_Weighted", "capped-score-weighted", "x "])
def test_an_illegal_name_is_refused_instead_of_normalized(name: str) -> None:
    """A name is either registered as written or it is a typo -- never repaired."""
    registry = register_all(SPEC)

    with pytest.raises(PortfolioRegistryError) as error:
        registry.find(name)

    assert "not a legal construction name" in str(error.value)


@pytest.mark.parametrize("name", [None, 42, SPEC, ("capped_score_weighted",)])
def test_find_needs_a_string(name: object) -> None:
    registry = register_all(SPEC)

    with pytest.raises(PortfolioRegistryError, match="must be a string"):
        registry.find(name)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Registering and creating by the whole spec
# ---------------------------------------------------------------------------
def test_a_registered_constructor_can_be_created_by_its_whole_spec() -> None:
    built = make_constructor_class(SPEC)
    registry = PortfolioConstructorRegistry()

    spec = registry.register(built)
    constructor = registry.create(spec)

    assert spec == built().spec
    assert spec == SPEC
    assert isinstance(constructor, built)
    assert isinstance(constructor, PortfolioConstructor)
    assert constructor.spec == spec
    assert len(registry) == 1
    assert registry.specs == (SPEC,)


def test_the_registration_returns_the_constructors_own_identity() -> None:
    """The table never names a rule itself: the spec is the authority."""
    registry = PortfolioConstructorRegistry()
    spec = make_spec("rank_weighted", version="1.0.0")

    returned = registry.register(lambda: make_constructor_class(spec)())

    assert returned is spec
    assert returned.name == "rank_weighted"
    assert returned.version == "1.0.0"


def test_an_equal_spec_built_elsewhere_creates_the_registered_constructor() -> None:
    """A spec is a value: the same identity passed as another object is the same."""
    registry = register_all(SPEC)
    equivalent = PortfolioConstructionSpec(name="capped_score_weighted", version="1.0.0")

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
    built = make_constructor_class(SPEC)
    calls: list[int] = []

    def factory() -> PortfolioConstructor:
        calls.append(len(calls))
        return built()

    registry = PortfolioConstructorRegistry()
    registry.register(factory)

    assert len(calls) == 1  # the probe
    registry.create(SPEC)
    registry.create(SPEC)
    assert len(calls) == 3


def test_a_class_factory_builds_a_fresh_constructor_every_time() -> None:
    built = make_constructor_class(SPEC)
    registry = PortfolioConstructorRegistry()
    registry.register(built)

    constructors = [registry.create(SPEC) for _ in range(3)]

    assert len({id(constructor) for constructor in constructors}) == 3
    assert all(isinstance(constructor, built) for constructor in constructors)


def test_the_registry_never_builds_weights() -> None:
    """The table hands out rules; deciding weights is ``construct()``'s job."""
    calls: list[int] = []

    class Counting(PortfolioConstructor):
        spec = SPEC

        def construct(self, context: PortfolioContext) -> TargetPortfolio:
            calls.append(len(calls))
            raise NotImplementedError("this test double never builds weights")

    registry = PortfolioConstructorRegistry()
    spec = registry.register(Counting)
    registry.create(spec)
    registry.create(spec)

    assert calls == []  # never called, not even once
    assert spec == SPEC


# ---------------------------------------------------------------------------
# The whole spec is the key
# ---------------------------------------------------------------------------
def test_two_versions_of_the_same_rule_are_two_registrations() -> None:
    """Same name, different version: two rules, two entries."""
    registry = register_all(SPEC, SPEC_V2)

    assert SPEC.name == SPEC_V2.name == "capped_score_weighted"
    assert SPEC.version == "1.0.0"
    assert SPEC_V2.version == "2.0.0"
    assert SPEC != SPEC_V2
    assert len(registry) == 2
    assert set(registry.specs) == {SPEC, SPEC_V2}
    assert registry.find("capped_score_weighted") == (SPEC, SPEC_V2)


def test_creating_one_version_never_returns_the_other() -> None:
    registry = register_all(SPEC, SPEC_V2)

    on_v1 = registry.create(SPEC)
    on_v2 = registry.create(SPEC_V2)

    assert on_v1.spec == SPEC
    assert on_v2.spec == SPEC_V2
    assert type(on_v1) is not type(on_v2)


def test_find_returns_every_registered_version_of_a_name_in_order() -> None:
    """Versions are labels, so they sort as text: stable, never guessed at."""
    third = make_spec("capped_score_weighted", version="1.10.0")
    other = make_spec("equal_weight", version="1.0.0")
    registry = register_all(third, SPEC, other, SPEC_V2)

    assert registry.find("capped_score_weighted") == (SPEC, third, SPEC_V2)
    assert registry.find("equal_weight") == (other,)


def test_a_duplicate_spec_is_refused_and_never_overwrites() -> None:
    first = make_constructor_class(SPEC)
    second = make_constructor_class(SPEC)
    registry = PortfolioConstructorRegistry()
    registry.register(first)

    with pytest.raises(PortfolioRegistryError) as error:
        registry.register(second)  # same name, same version: one rule

    assert "already registered" in str(error.value)
    assert "would replace" in str(error.value)
    assert registry.specs == (SPEC,)
    assert isinstance(registry.create(SPEC), first)  # the first factory still answers


def test_registering_the_same_factory_twice_is_still_a_duplicate() -> None:
    built = make_constructor_class(SPEC)
    registry = PortfolioConstructorRegistry()
    registry.register(built)

    with pytest.raises(PortfolioRegistryError) as error:
        registry.register(built)

    assert "already registered" in str(error.value)
    assert "is already registered under it" in str(error.value)
    assert len(registry) == 1


def test_a_factory_differing_only_by_hidden_parameters_is_a_duplicate() -> None:
    """Same name and version with other numbers behind it needs a new version."""
    registry = PortfolioConstructorRegistry()
    registry.register(lambda: make_constructor_class(SPEC)())

    with pytest.raises(PortfolioRegistryError, match="already registered"):
        registry.register(lambda: make_constructor_class(SPEC)())

    assert registry.specs == (SPEC,)


@pytest.mark.parametrize(
    "value",
    [
        "capped_score_weighted",
        ("capped_score_weighted", "1.0.0"),
        {"name": "capped_score_weighted", "version": "1.0.0"},
        PortfolioConstraints(cash_weight=0.1, max_single_weight=0.3),
        None,
        42,
    ],
)
def test_nothing_but_a_spec_can_be_created(value: object) -> None:
    """A name is not a request; nor are the constraints part of one."""
    registry = register_all(SPEC)

    with pytest.raises(PortfolioRegistryError) as error:
        registry.create(value)  # type: ignore[arg-type]

    message = str(error.value)
    assert "needs the whole PortfolioConstructionSpec" in message
    assert "find(name)" in message
    assert "PortfolioConstraints" in message


# ---------------------------------------------------------------------------
# Refusals: types, factories, specs
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("factory", [None, 42, "capped_score_weighted", SPEC, object(), []])
def test_a_non_callable_factory_is_refused(factory: object) -> None:
    registry = PortfolioConstructorRegistry()

    with pytest.raises(PortfolioRegistryError) as error:
        registry.register(factory)  # type: ignore[arg-type]

    assert "must be callable" in str(error.value)
    assert len(registry) == 0


def test_a_constructor_instance_is_not_a_factory() -> None:
    """An instance is not a maker: registering one would hand out a shared object."""
    instance = make_constructor_class(SPEC)()
    registry = PortfolioConstructorRegistry()

    with pytest.raises(PortfolioRegistryError) as error:
        registry.register(instance)  # type: ignore[arg-type]

    assert "must be callable" in str(error.value)
    assert "not an instance" in str(error.value)
    assert len(registry) == 0


def test_a_callable_object_is_a_legal_factory() -> None:
    """A maker may be any parameterless callable, not only a class or a closure."""
    built = make_constructor_class(SPEC)
    registry = PortfolioConstructorRegistry()

    class Maker:
        def __call__(self) -> PortfolioConstructor:
            return built()

    spec = registry.register(Maker())

    assert spec == SPEC
    assert isinstance(registry.create(spec), built)


def test_a_factory_that_raises_is_reported_with_its_cause() -> None:
    def factory() -> PortfolioConstructor:
        raise ValueError("no spec today")

    registry = PortfolioConstructorRegistry()

    with pytest.raises(PortfolioRegistryError) as error:
        registry.register(factory)

    assert "raised ValueError" in str(error.value)
    assert isinstance(error.value.__cause__, ValueError)
    assert len(registry) == 0


def test_a_factory_that_fails_only_when_creating_keeps_its_cause() -> None:
    built = make_constructor_class(SPEC)
    calls: list[int] = []

    def factory() -> PortfolioConstructor:
        calls.append(len(calls) + 1)
        if calls[-1] == 1:
            return built()
        raise RuntimeError("the second call broke")

    registry = PortfolioConstructorRegistry()
    spec = registry.register(factory)

    with pytest.raises(PortfolioRegistryError) as error:
        registry.create(spec)

    assert "raised RuntimeError" in str(error.value)
    assert isinstance(error.value.__cause__, RuntimeError)
    assert registry.specs == (SPEC,)  # the registration survives a failed create


@pytest.mark.parametrize("answer", [None, {}, object(), "capped_score_weighted", 3.5])
def test_a_factory_that_does_not_build_a_constructor_is_refused(answer: object) -> None:
    registry = PortfolioConstructorRegistry()

    with pytest.raises(PortfolioRegistryError) as error:
        registry.register(lambda: answer)  # type: ignore[return-value]

    assert "must build a PortfolioConstructor" in str(error.value)
    assert len(registry) == 0


def test_a_duck_typed_lookalike_is_not_a_constructor() -> None:
    """``isinstance`` decides, not a ``spec`` attribute that happens to be there."""

    class Lookalike:
        spec = SPEC

        def construct(self, context: object) -> object:
            raise NotImplementedError

    registry = PortfolioConstructorRegistry()

    with pytest.raises(PortfolioRegistryError, match="must build a PortfolioConstructor"):
        registry.register(Lookalike)  # type: ignore[arg-type]

    assert len(registry) == 0


@pytest.mark.parametrize("declared", [None, "capped_score_weighted", {}, ("name", "version")])
def test_a_constructor_without_a_valid_spec_is_refused(declared: object) -> None:
    class BadSpec(PortfolioConstructor):
        spec = declared

        def construct(self, context: PortfolioContext) -> TargetPortfolio:
            raise NotImplementedError

    registry = PortfolioConstructorRegistry()

    with pytest.raises(PortfolioRegistryError) as error:
        registry.register(BadSpec)

    assert "spec is not a PortfolioConstructionSpec" in str(error.value)
    assert len(registry) == 0


def test_a_constructor_whose_spec_cannot_be_read_is_refused() -> None:
    class Broken(PortfolioConstructor):
        @property
        def spec(self) -> PortfolioConstructionSpec:
            raise RuntimeError("no identity today")

        def construct(self, context: PortfolioContext) -> TargetPortfolio:
            raise NotImplementedError

    registry = PortfolioConstructorRegistry()

    with pytest.raises(PortfolioRegistryError) as error:
        registry.register(Broken)

    assert "spec could not be read" in str(error.value)
    assert isinstance(error.value.__cause__, RuntimeError)


def test_a_spec_property_that_raises_a_portfolio_error_keeps_its_type() -> None:
    """A broken identity is reported by the layer that knows it, not by the table."""

    class BrokenSpec(PortfolioConstructor):
        @property
        def spec(self) -> PortfolioConstructionSpec:
            raise PortfolioContractError("invalid PortfolioConstructionSpec", ["version is blank"])

        def construct(self, context: PortfolioContext) -> TargetPortfolio:
            raise NotImplementedError

    registry = PortfolioConstructorRegistry()

    with pytest.raises(PortfolioContractError) as error:
        registry.register(BrokenSpec)

    assert not isinstance(error.value, PortfolioRegistryError)
    assert error.value.problems == ["version is blank"]
    assert len(registry) == 0


def test_a_spec_that_cannot_be_a_table_key_is_refused() -> None:
    unhashable = forged_spec(version=["1.0.0"])

    class Forged(PortfolioConstructor):
        spec = unhashable

        def construct(self, context: PortfolioContext) -> TargetPortfolio:
            raise NotImplementedError

    registry = PortfolioConstructorRegistry()

    with pytest.raises(PortfolioRegistryError) as error:
        registry.register(Forged)

    assert "cannot be a table key" in str(error.value)
    assert len(registry) == 0


def test_create_of_an_unregistered_spec_lists_what_is_registered() -> None:
    registry = register_all(SPEC)

    with pytest.raises(PortfolioRegistryError) as error:
        registry.create(EQUAL)

    message = str(error.value)
    assert "equal_weight/1.0.0 is not registered" in message
    assert "registered constructions: capped_score_weighted/1.0.0" in message


def test_create_points_at_the_registered_versions_of_the_same_name() -> None:
    """Asking for one version of a rule that exists in another is the common slip."""
    registry = register_all(SPEC)

    with pytest.raises(PortfolioRegistryError) as error:
        registry.create(SPEC_V2)

    message = str(error.value)
    assert "is registered for 1 version(s)" in message
    assert "capped_score_weighted/1.0.0" in message
    assert "find(name)" in message


def test_create_of_an_unhashable_spec_is_refused() -> None:
    registry = register_all(SPEC)

    with pytest.raises(PortfolioRegistryError, match="is not hashable"):
        registry.create(forged_spec(version=["1.0.0"]))


# ---------------------------------------------------------------------------
# Portfolio errors travel untouched
# ---------------------------------------------------------------------------
def test_a_portfolio_error_from_the_factory_is_not_wrapped() -> None:
    def factory() -> PortfolioConstructor:
        raise PortfolioConstructionError("this selection cannot be weighted")

    registry = PortfolioConstructorRegistry()

    with pytest.raises(PortfolioConstructionError) as error:
        registry.register(factory)

    assert not isinstance(error.value, PortfolioRegistryError)
    assert error.value.__cause__ is None


def test_a_portfolio_error_at_create_time_keeps_its_type() -> None:
    built = make_constructor_class(SPEC)
    calls: list[int] = []

    def factory() -> PortfolioConstructor:
        calls.append(len(calls) + 1)
        if calls[-1] == 1:
            return built()
        raise PortfolioConstructionError("the constraints leave no room")

    registry = PortfolioConstructorRegistry()
    spec = registry.register(factory)

    with pytest.raises(PortfolioConstructionError, match="leave no room"):
        registry.create(spec)

    assert registry.specs == (SPEC,)


# ---------------------------------------------------------------------------
# Factory drift: the identity at create must be the identity registered
# ---------------------------------------------------------------------------
def test_a_factory_that_changes_what_it_builds_is_caught_at_create() -> None:
    """The table never returns a constructor of a version nobody asked for."""
    calls: list[int] = []
    chameleon = make_chameleon_class(SPEC)

    def factory() -> PortfolioConstructor:
        calls.append(len(calls) + 1)
        constructor = chameleon()
        if calls[-1] > 1:  # the probe is honest; every later build drifts
            constructor.current = SPEC_V2
        return constructor

    registry = PortfolioConstructorRegistry()
    spec = registry.register(factory)  # registers under 1.0.0
    assert spec == SPEC

    with pytest.raises(PortfolioRegistryError) as error:
        registry.create(SPEC)

    message = str(error.value)
    assert "now builds capped_score_weighted/2.0.0" in message
    assert "must always build the constructor it was registered under" in message
    assert registry.specs == (SPEC,)  # the entry is untouched


def test_a_drifting_factory_leaves_the_rest_of_the_table_usable() -> None:
    calls: list[int] = []
    chameleon = make_chameleon_class(SPEC)
    healthy = make_constructor_class(EQUAL)

    def drifting() -> PortfolioConstructor:
        calls.append(len(calls) + 1)
        constructor = chameleon()
        if calls[-1] > 1:
            constructor.current = SPEC_V2
        return constructor

    registry = PortfolioConstructorRegistry()
    registry.register(drifting)
    registry.register(healthy)

    with pytest.raises(PortfolioRegistryError, match="now builds"):
        registry.create(SPEC)

    assert set(registry.specs) == {SPEC, EQUAL}
    assert len(registry) == 2
    assert isinstance(registry.create(EQUAL), healthy)  # still answers normally


def test_drift_is_checked_on_every_create_not_only_the_first() -> None:
    """Two honest answers do not license a third, different version."""
    calls: list[int] = []
    chameleon = make_chameleon_class(SPEC)

    def factory() -> PortfolioConstructor:
        calls.append(len(calls) + 1)
        constructor = chameleon()
        if calls[-1] > 3:  # the probe and two creates are honest
            constructor.current = SPEC_V2
        return constructor

    registry = PortfolioConstructorRegistry()
    registry.register(factory)

    assert registry.create(SPEC).spec == SPEC
    assert registry.create(SPEC).spec == SPEC
    with pytest.raises(PortfolioRegistryError, match="now builds"):
        registry.create(SPEC)


# ---------------------------------------------------------------------------
# One instance per create
# ---------------------------------------------------------------------------
def test_a_singleton_factory_is_refused_at_the_first_create() -> None:
    """A closure over one pre-built constructor is not a factory."""
    singleton = make_constructor_class(SPEC)()
    registry = PortfolioConstructorRegistry()

    spec = registry.register(lambda: singleton)  # legal: it is a PortfolioConstructor

    with pytest.raises(PortfolioRegistryError) as error:
        registry.create(spec)

    assert "reused" in str(error.value)
    assert "build a new constructor for every registration and every create()" in str(error.value)
    assert len(registry) == 1


def test_the_probe_counts_as_used_even_after_good_creates() -> None:
    probe = make_constructor_class(SPEC)()
    fresh = make_constructor_class(SPEC)
    calls: list[int] = []

    def factory() -> PortfolioConstructor:
        calls.append(len(calls) + 1)
        return probe if calls[-1] in {1, 4} else fresh()

    registry = PortfolioConstructorRegistry()
    spec = registry.register(factory)  # call 1 hands over the probe

    registry.create(spec)  # call 2
    registry.create(spec)  # call 3
    with pytest.raises(PortfolioRegistryError, match="reused"):
        registry.create(spec)  # call 4 replays the probe

    assert len(registry) == 1


def test_a_late_repeat_of_an_already_returned_instance_is_refused() -> None:
    """Every instance is on record, not just the one handed out last."""
    fresh = make_constructor_class(SPEC)
    handed_back: list[PortfolioConstructor] = []
    calls: list[int] = []

    def factory() -> PortfolioConstructor:
        calls.append(len(calls) + 1)
        if calls[-1] == 1:  # the registration probe
            return fresh()
        if calls[-1] == 5:  # four calls later: replay the very first answer
            return handed_back[0]
        constructor = fresh()
        handed_back.append(constructor)
        return constructor

    registry = PortfolioConstructorRegistry()
    spec = registry.register(factory)  # call 1: the probe
    first = registry.create(spec)  # call 2
    registry.create(spec)  # call 3
    registry.create(spec)  # call 4
    assert handed_back[0] is first

    with pytest.raises(PortfolioRegistryError, match="reused"):
        registry.create(spec)  # call 5

    assert len(registry) == 1


def test_one_instance_cannot_be_laundered_through_a_second_spec() -> None:
    """The ledger is global: another spec is not a way around the guard.

    Registration writes to the ledger, so it must read from it too.  A probe that comes
    back re-labelled is refused *at that second register*, not later at some create, and
    the refusal leaves the table exactly as it was.
    """
    chameleon = make_chameleon_class(SPEC)()
    registry = PortfolioConstructorRegistry()
    first = registry.register(lambda: chameleon)  # the probe, under SPEC
    assert first == SPEC

    chameleon.current = SPEC_V2
    with pytest.raises(PortfolioRegistryError, match="reused") as error:
        registry.register(lambda: chameleon)  # the same object, re-labelled

    assert "already used" in str(error.value)
    assert registry.specs == (SPEC,)
    assert len(registry) == 1
    assert SPEC_V2 not in registry
    # the second spec was never added, so it cannot be created either
    with pytest.raises(PortfolioRegistryError, match="is not registered"):
        registry.create(SPEC_V2)
    # and the refusal poisoned nothing: the table still accepts a real factory
    registry.register(make_constructor_class(SPEC_V2))
    assert set(registry.specs) == {SPEC, SPEC_V2}


def test_an_instance_already_created_cannot_become_a_registration_probe() -> None:
    """A create() result is used too: it can never be filed under another identity."""
    chameleon = make_chameleon_class(SPEC)
    registry = PortfolioConstructorRegistry()
    spec = registry.register(chameleon)

    created = registry.create(spec)
    assert isinstance(created, chameleon)

    created.current = SPEC_V2
    with pytest.raises(PortfolioRegistryError, match="reused"):
        registry.register(lambda: created)  # the same object, as a new probe

    assert registry.specs == (SPEC,)
    assert len(registry) == 1
    assert SPEC_V2 not in registry


def test_identity_not_equality_decides_whether_an_instance_is_new() -> None:
    """Two equal, unhashable constructors are two instances, and both are legal."""

    class EqualConstructor(PortfolioConstructor):
        spec = SPEC

        def __eq__(self, other: object) -> bool:
            return isinstance(other, PortfolioConstructor)

        def construct(self, context: PortfolioContext) -> TargetPortfolio:
            raise NotImplementedError

    registry = PortfolioConstructorRegistry()
    spec = registry.register(EqualConstructor)

    first = registry.create(spec)
    second = registry.create(spec)

    assert first is not second
    assert first == second  # equal by ==, and still accepted
    with pytest.raises(TypeError):
        hash(first)  # not hashable either
    assert len(registry) == 1


def test_a_refused_reuse_leaves_the_table_and_other_registries_alone() -> None:
    singleton = make_constructor_class(SPEC)()
    other_spec = make_spec("equal_weight", version="1.0.0")

    registry = PortfolioConstructorRegistry()
    other = register_all(other_spec)
    spec = registry.register(lambda: singleton)
    registry.register(make_constructor_class(other_spec))

    with pytest.raises(PortfolioRegistryError, match="reused"):
        registry.create(spec)

    assert len(registry) == 2
    assert set(registry.specs) == {SPEC, other_spec}
    assert isinstance(registry.create(other_spec), PortfolioConstructor)  # still usable
    assert other.specs == (other_spec,)  # untouched
    assert len(other) == 1


def test_a_refused_registration_does_not_register_the_probe() -> None:
    """The probe is only filed once every check has passed."""
    registry = PortfolioConstructorRegistry()
    registry.register(make_constructor_class(SPEC))

    with pytest.raises(PortfolioRegistryError, match="already registered"):
        registry.register(make_constructor_class(SPEC))

    assert registry.specs == (SPEC,)
    assert len(registry) == 1


# ---------------------------------------------------------------------------
# specs and find are sorted and read-only
# ---------------------------------------------------------------------------
def test_specs_are_sorted_by_name_and_then_version() -> None:
    momentum_v2 = make_spec("capped_score_weighted", version="2.0.0")
    momentum_v1 = make_spec("capped_score_weighted", version="1.0.0")
    equal = make_spec("equal_weight", version="1.0.0")
    rank = make_spec("rank_weighted", version="1.0.0")

    registry = register_all(rank, momentum_v2, equal, momentum_v1)

    assert registry.specs == (momentum_v1, momentum_v2, equal, rank)
    assert [spec.name for spec in registry.specs] == [
        "capped_score_weighted",
        "capped_score_weighted",
        "equal_weight",
        "rank_weighted",
    ]


def test_specs_and_find_hand_out_unmodifiable_tuples() -> None:
    registry = register_all(SPEC, SPEC_V2)

    specs = registry.specs
    found = registry.find("capped_score_weighted")

    assert isinstance(specs, tuple)
    assert isinstance(found, tuple)
    assert specs is not registry.specs  # a fresh tuple per call
    with pytest.raises(TypeError):
        specs[0] = SPEC  # type: ignore[index]
    with pytest.raises(TypeError):
        found[0] = SPEC  # type: ignore[index]


def test_the_table_cannot_be_reached_through_the_listings() -> None:
    """A caller gets values, never the dictionary the table is built on."""
    registry = register_all(SPEC)

    with pytest.raises(AttributeError):
        registry.specs.append(SPEC_V2)  # type: ignore[attr-defined]
    assert len(registry) == 1
    registry.register(make_constructor_class(SPEC_V2))
    assert registry.specs == (SPEC, SPEC_V2)


def test_repr_names_every_registered_identity() -> None:
    registry = register_all(SPEC, SPEC_V2)

    text = repr(registry)

    assert text.startswith("PortfolioConstructorRegistry(specs=(")
    assert "capped_score_weighted/1.0.0" in text
    assert "capped_score_weighted/2.0.0" in text
    assert "factory" not in text  # no internal detail leaks out


# ---------------------------------------------------------------------------
# The error type and the package surface
# ---------------------------------------------------------------------------
def test_a_registry_error_is_a_portfolio_error_only() -> None:
    assert issubclass(PortfolioRegistryError, PortfolioError)
    assert not issubclass(PortfolioRegistryError, PortfolioConstructionError)
    assert not issubclass(PortfolioRegistryError, ValueError)
    assert not issubclass(PortfolioRegistryError, KeyError)


def test_the_registry_is_exported_from_the_package_root() -> None:
    assert portfolio_sys.PortfolioConstructorRegistry is PortfolioConstructorRegistry
    assert portfolio_sys.PortfolioRegistryError is PortfolioRegistryError
    assert registry_module.PortfolioConstructorRegistry is PortfolioConstructorRegistry
    assert {"PortfolioConstructorRegistry", "PortfolioRegistryError"} <= set(portfolio_sys.__all__)
    assert registry_module.__all__ == ["PortfolioConstructorRegistry"]


def test_the_package_root_never_exports_a_concrete_rule() -> None:
    """A rule is imported from ``portfolio_sys.constructors``, never from the root."""
    for absent in (
        "CappedScoreWeightedConstructor",
        "EqualWeightConstructor",
        "VolatilityWeightedConstructor",
        "OptimizerConstructor",
        "default_portfolio_registry",
        "DEFAULT_VERSION",
    ):
        assert not hasattr(portfolio_sys, absent), absent


def test_there_is_no_default_table_and_no_global_registry() -> None:
    """Which rule a project runs is a research decision, never an import default."""
    assert not hasattr(registry_module, "default_portfolio_registry")
    assert not hasattr(registry_module, "_REGISTRY")
    for absent in ("unregister", "register_many", "create_by_name", "clear", "default"):
        assert not hasattr(PortfolioConstructorRegistry, absent), absent


def test_create_takes_a_spec_and_register_takes_a_factory() -> None:
    """No name-only entry point can be added without breaking this test."""
    assert list(inspect.signature(PortfolioConstructorRegistry.create).parameters) == [
        "self",
        "spec",
    ]
    assert list(inspect.signature(PortfolioConstructorRegistry.register).parameters) == [
        "self",
        "factory",
    ]
    assert list(inspect.signature(PortfolioConstructorRegistry.find).parameters) == [
        "self",
        "name",
    ]


def test_every_public_object_is_documented() -> None:
    assert PortfolioConstructorRegistry.__doc__
    assert PortfolioRegistryError.__doc__
    for owner, name in (
        (PortfolioConstructorRegistry, "register"),
        (PortfolioConstructorRegistry, "create"),
        (PortfolioConstructorRegistry, "find"),
        (PortfolioConstructorRegistry, "specs"),
        (PortfolioConstructorRegistry, "__init__"),
    ):
        assert getattr(owner, name).__doc__, name


def test_the_error_module_declares_the_registry_error() -> None:
    from portfolio_sys import errors as errors_module

    assert errors_module.__all__ == [
        "PortfolioConstructionError",
        "PortfolioContractError",
        "PortfolioError",
        "PortfolioRegistryError",
    ]
    assert errors_module.PortfolioRegistryError is PortfolioRegistryError


# ---------------------------------------------------------------------------
# The module's own boundary
# ---------------------------------------------------------------------------
REGISTRY_PATH = Path(registry_module.__file__)

#: Modules ``registry.py`` may import at module level: its own contract, its own errors
#: and the standard library.  No concrete rule, no data, no pandas, no factor layer, no
#: strategy layer -- a rule arrives as a factory the caller passes in.
ALLOWED_MODULE_IMPORTS: frozenset[str] = frozenset(
    {
        "__future__",
        "collections.abc",
        "re",
        "portfolio_sys.contracts",
        "portfolio_sys.errors",
    }
)

#: Module prefixes that would drag another layer into the mechanism.
FORBIDDEN_PREFIXES: tuple[str, ...] = (
    "data_sys",
    "pandas",
    "numpy",
    "pyarrow",
    "polars",
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
    "strategy_sys",
    "portfolio_sys.constructors",
)

#: Calls that would make a pure, in-memory table impure.
FORBIDDEN_CALLS: frozenset[str] = frozenset(
    {"__import__", "compile", "connect", "eval", "exec", "input", "open", "print", "urlopen"}
)

#: Names that would mean the table knows a concrete rule, a runner or a target.
FORBIDDEN_NAMES: frozenset[str] = frozenset(
    {
        "CappedScoreWeightedConstructor",
        "EqualWeightConstructor",
        "ScoreWeightedConstructor",
        "RankWeightedConstructor",
        "VolatilityWeightedConstructor",
        "OptimizerConstructor",
        "PortfolioRunner",
        "run_portfolio",
        "load_targets",
        "save_targets",
        "StrategyResult",
        "StrategyContext",
        "FactorPanel",
        "FactorStore",
        "PortfolioContext",
        "TargetPortfolio",
        "PortfolioConstraints",
        "PortfolioPosition",
        "PortfolioState",
    }
)

#: Identifiers the table must never use: a weight, a cap, a cash share or a score is a
#: *rule's* arithmetic, and an order, a fill or a backtest is another layer's vocabulary.
FORBIDDEN_IDENTIFIERS: frozenset[str] = frozenset(
    {
        "weight",
        "weights",
        "budget",
        "investable",
        "cap",
        "cash_weight",
        "max_single_weight",
        "score",
        "scores",
        "target_weight",
        "quantity",
        "shares",
        "price",
        "fill",
        "order",
        "orders",
        "turnover",
        "nav",
        "backtest",
        "rebalance",
        "pnl",
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


def mentioned_names(path: Path) -> set[str]:
    """Every name, attribute, argument and definition the module's code uses."""
    tree = parse(path)
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    names |= {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    names |= {node.arg for node in ast.walk(tree) if isinstance(node, ast.arg)}
    names |= {
        node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }
    return names


def test_the_registry_imports_only_its_own_layer() -> None:
    """No data system, no factor layer, no strategy layer, no concrete rule, no pandas."""
    imports = scoped_imports(REGISTRY_PATH)

    module_level = {module for scope, module in imports if scope is None}
    assert module_level <= ALLOWED_MODULE_IMPORTS, sorted(module_level - ALLOWED_MODULE_IMPORTS)
    assert [module for _, module in imports if module.startswith(FORBIDDEN_PREFIXES)] == []
    assert [scope for scope, _ in imports if scope is not None] == []  # no lazy imports


def test_the_registry_knows_no_concrete_constructor() -> None:
    """A rule arrives as a factory: no code path here names or reaches one.

    The module docstring may name a rule while explaining what a caller no longer has to
    import, so the check reads the tree rather than the text: a name or an attribute that
    *executes* must never be a concrete constructor.
    """
    tree = parse(REGISTRY_PATH)

    assert referenced_names(REGISTRY_PATH) & FORBIDDEN_NAMES == set()
    assert {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)} & (
        FORBIDDEN_NAMES
    ) == set()
    assert mentioned_names(REGISTRY_PATH) & FORBIDDEN_NAMES == set()


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


def test_the_registry_never_calls_construct() -> None:
    """Handing out a rule is the whole job; deciding weights belongs to ``construct()``."""
    assert "construct" not in called_names(REGISTRY_PATH)
    assert "construct" not in mentioned_names(REGISTRY_PATH)


def test_the_registry_never_names_a_weight_a_cap_a_cash_share_or_a_score() -> None:
    offenders = sorted(mentioned_names(REGISTRY_PATH) & FORBIDDEN_IDENTIFIERS)

    assert offenders == []


def test_the_contract_and_the_rules_never_import_the_registry() -> None:
    """The dependency runs one way: a rule never knows a table holds it."""
    package = Path(portfolio_sys.__file__).parent
    modules = [
        package / "contracts.py",
        package / "constructors" / "__init__.py",
        package / "constructors" / "capped_score_weighted.py",
    ]

    for module in modules:
        imported = {name for _, name in scoped_imports(module)}
        assert not [name for name in imported if name.startswith("portfolio_sys.registry")], (
            module.name
        )
        assert "PortfolioConstructorRegistry" not in mentioned_names(module), module.name


def test_the_package_is_made_of_the_contract_the_registry_and_the_errors() -> None:
    package = Path(portfolio_sys.__file__).parent

    assert sorted(path.name for path in package.glob("*.py")) == [
        "__init__.py",
        "contracts.py",
        "errors.py",
        "registry.py",
        "transition.py",
    ]


# ---------------------------------------------------------------------------
# Integration: the real rule, registered by the caller
# ---------------------------------------------------------------------------
def test_the_real_capped_rule_is_registered_and_created() -> None:
    """The flow the task is about: one explicit register, then a fresh constructor."""
    registry = PortfolioConstructorRegistry()

    spec = registry.register(CappedScoreWeightedConstructor)

    assert spec == PortfolioConstructionSpec(name="capped_score_weighted", version="1.0.0")
    assert spec.name == "capped_score_weighted"
    assert spec.version == "1.0.0"
    assert spec in registry
    assert registry.specs == (spec,)
    assert registry.find("capped_score_weighted") == (spec,)

    constructor = registry.create(spec)

    assert isinstance(constructor, CappedScoreWeightedConstructor)
    assert constructor.spec == spec
    assert registry.create(spec) is not constructor  # a new object every time


def test_the_real_rule_arrives_only_as_a_factory_from_the_caller() -> None:
    """The table holds names and factories; nothing here imports a rule."""
    registry = PortfolioConstructorRegistry()
    spec = registry.register(lambda: CappedScoreWeightedConstructor())

    assert spec == make_spec("capped_score_weighted")
    assert isinstance(registry.create(spec), CappedScoreWeightedConstructor)
    assert not hasattr(registry_module, "CappedScoreWeightedConstructor")


def test_a_second_version_of_the_same_name_coexists_with_the_real_rule() -> None:
    """Publishing v2 of a rule must not evict v1: the version is the identity."""
    registry = PortfolioConstructorRegistry()
    registered = registry.register(CappedScoreWeightedConstructor)
    next_version = registry.register(make_constructor_class(make_spec(version="2.0.0")))

    assert registered.version == "1.0.0"
    assert next_version.version == "2.0.0"
    assert len(registry) == 2
    assert registry.find("capped_score_weighted") == (registered, next_version)
    assert isinstance(registry.create(registered), CappedScoreWeightedConstructor)
    assert type(registry.create(next_version)) is not CappedScoreWeightedConstructor


def test_the_real_rule_is_never_registered_automatically() -> None:
    """A fresh table is empty: which rule runs is the caller's decision."""
    registry = PortfolioConstructorRegistry()

    assert registry.specs == ()
    assert len(registry) == 0
    with pytest.raises(PortfolioRegistryError, match="is not registered"):
        registry.create(make_spec("capped_score_weighted"))
