"""The explicit strategy table: a full ``StrategySpec``, a factory, nothing else.

What this module is
-------------------
``strategy_sys.registry`` answers one question for everything above the strategy
layer: *given the strategy a caller means, how do I get an instance of it?*::

    registry = StrategyRegistry()
    registry.register(lambda: TopNByFactorStrategy(MOMENTUM_V1, top_n=10))
    strategy = registry.create(spec)      # a brand-new TopNByFactorStrategy

Nothing above the strategy layer has to write ``from strategy_sys.strategies
import TopNByFactorStrategy``: a future runner, a research notebook or a backtest
can carry a :class:`~strategy_sys.contracts.StrategySpec`, and this table turns
that spec into a strategy.  Adding a rule is then one explicit ``register`` line
in the caller -- never an edit to the registry and never an edit to a runner.

``StrategyRegistry``
    a table of ``StrategySpec -> factory``, where a *factory* is a parameterless
    callable returning a :class:`~strategy_sys.contracts.Strategy`
``StrategyRegistryError``
    a registration or a request that cannot be honoured

The key is the whole spec
-------------------------
The identity of a strategy is not its name.  It is the declaration that produced
a result: the rule, its version *and* the exact factor versions it read::

    StrategySpec("top_10_by_momentum_12_1", "1.0.0", (FactorRequirement("momentum_12_1", "1.0.0"),))
    StrategySpec("top_10_by_momentum_12_1", "1.0.0", (FactorRequirement("momentum_12_1", "2.0.0"),))

Those two share a name and a version and are still different strategies: they read
different factors, so their results are not comparable and must not be
interchangeable.  A table keyed by ``name`` -- or by ``(name, version)`` -- would
collapse them into one entry, and the second registration would either overwrite
the first or be refused as a duplicate of an unrelated rule.  Either way the
factor-version provenance the contract carries on every result would be lost at
the first lookup.

:class:`~strategy_sys.contracts.StrategySpec` is already a frozen, hashable value
object, so the table uses it directly as its key: no ``StrategyKey`` type, no
glued ``"name@version+factors"`` string, and nothing to parse back.


Registration
------------
``register(factory)`` builds the strategy once, reads ``strategy.spec`` and files
the factory under that complete spec -- the *canonical* identity, taken from the
strategy itself::

    registry = StrategyRegistry()
    MOMENTUM_V1 = FactorRequirement("momentum_12_1", "1.0.0")
    MOMENTUM_V2 = FactorRequirement("momentum_12_1", "2.0.0")
    v1 = registry.register(lambda: TopNByFactorStrategy(MOMENTUM_V1, top_n=10))
    v2 = registry.register(lambda: TopNByFactorStrategy(MOMENTUM_V2, top_n=10))
    v1 == v2        # False: same name and version, different factor versions
    len(registry)   # 2

There is no spec argument, so the table can never disagree with the identity the
strategy declares.  The same complete spec twice is an error, never a silent
overwrite: which of two implementations won would depend on registration order,
and a backtest must never depend on that.  The factory is called once here (to
learn the spec) and once per :meth:`StrategyRegistry.create` call.

Creating
--------
``create(spec)`` looks the *whole* spec up and calls the factory again.  There is
deliberately no ``create(name)``: a name can map to several registered factor
versions, so "the strategy called X" is not a well-formed request.  A caller that
has only a name asks :meth:`StrategyRegistry.find` and passes the spec it means::

    for spec in registry.find("top_10_by_momentum_12_1"):
        ...                              # one entry per registered factor version
    strategy = registry.create(spec)

The answer is checked before it leaves the table: it must be a
:class:`~strategy_sys.contracts.Strategy` whose ``spec`` is a
:class:`~strategy_sys.contracts.StrategySpec` **equal to the registered one** --
name, version and required factors included.  A factory that has drifted (a
closure over a mutable variable, a strategy whose requirement was edited) is
caught here, instead of producing results whose provenance is a lie.

One instance per create
-----------------------
A factory must build a *new* strategy every time.  The table keeps every instance
it has built on record -- the registration probe and every strategy it has
already returned, under *any* spec -- and refuses to hand any of them out twice::

    singleton = TopNByFactorStrategy(FactorRequirement("momentum_12_1", "1.0.0"), top_n=10)
    registry.register(lambda: singleton)   # accepted: it is a legal Strategy
    registry.create(singleton.spec)        # StrategyRegistryError: reused

Sharing the object is what breaks the promise, not calling the factory twice: a
strategy that later grows runtime state (a cache, a counter, a lazily built
context) would carry that state from one run into the next, and neither the
future runner nor the portfolio layer could see it.  The registry does not copy,
reset or otherwise touch the strategy; it only refuses the factory.  The record
keeps a strong reference to each instance, so a hit means *that exact object*
rather than an address that happened to be reused -- affordable because a
strategy is its spec and parameters, never the data it scored.  Identity, never
``==`` and never ``hash``, decides, which is also why two equal-but-distinct
strategies are both accepted.

Errors
------
:class:`StrategyRegistryError` covers registration, lookup and an untrustworthy
factory: a non-callable factory, one that raises, one that builds something other
than a :class:`~strategy_sys.contracts.Strategy`, a strategy whose ``spec`` is not
a :class:`~strategy_sys.contracts.StrategySpec`, a duplicate spec, a request that
is not a ``StrategySpec``, a spec the table does not hold, a factory that now
builds a different spec, and a factory that hands back an instance it has already
used.  Every message names the strategy by its whole identity -- name, version
and factor versions -- never by a bare name, because a bare name is exactly the
thing that cannot tell two registered strategies apart.

A factory that raises a :class:`~strategy_sys.errors.StrategyError` keeps it: the
layer that knows what went wrong (``StrategyContractError`` for an invalid spec,
for example) already says it precisely, and callers may branch on that type.
Every other exception is wrapped in :class:`StrategyRegistryError` with the
original as ``__cause__``.

What is not implemented, on purpose
-----------------------------------
Unregistering, overwriting, aliases, name-only lookup, parameterised factories,
plugin discovery, entry points, config-file or environment-driven registration, a
``default_strategy_registry()``, and a mutable global "the registry".  Which
``top_n``, which factor version and which blend a project runs are research
decisions, so the library never picks a default rule at import time and nothing
is registered automatically; a caller builds its own table and registers exactly
what it means to run.

Purity and layering
-------------------
Standard library plus :mod:`strategy_sys.contracts` and
:mod:`strategy_sys.errors`.  No pandas, no ``data_sys``, no ``factor_sys``, no
factor store, no market data, no calendar, no Parquet/DuckDB, no file, no
environment variable, no network, no runner, no portfolio and no backtest.  The
table knows no strategy by name: a rule arrives only as a factory the caller
passes in, so a new strategy is one ``register`` call and never an edit here.

Example
-------
>>> from strategy_sys import FactorRequirement, Strategy, StrategySpec
>>> class TopOneByMomentum(Strategy):
...     spec = StrategySpec(
...         name="top_1_by_momentum_12_1",
...         version="1.0.0",
...         required_factors=(FactorRequirement("momentum_12_1", "1.0.0"),),
...     )
...     def compute(self, context):
...         raise NotImplementedError("this example never scores anything")
>>> registry = StrategyRegistry()
>>> registry.specs
()
>>> spec = registry.register(TopOneByMomentum)
>>> spec.name, spec.version
('top_1_by_momentum_12_1', '1.0.0')
>>> spec in registry
True
>>> registry.create(spec) is not registry.create(spec)
True
>>> [str(requirement) for requirement in registry.find(spec.name)[0].required_factors]
['momentum_12_1/1.0.0']
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence

from strategy_sys.contracts import STRATEGY_NAME_PATTERN, Strategy, StrategySpec
from strategy_sys.errors import StrategyError, StrategyRegistryError

#: The one name rule, shared with ``StrategySpec``: never a second dialect.
_STRATEGY_NAME_RE = re.compile(STRATEGY_NAME_PATTERN)


# ---------------------------------------------------------------------------
# Readable identities
# ---------------------------------------------------------------------------
def _describe(spec: object) -> str:
    """A whole strategy identity as one readable token.

    ``name/version (requires factor/version, ...)`` -- never just the name, so
    two rules that share a name are still told apart in an error message.  Only
    for messages and logs; specs are compared as objects, never as strings.
    """
    name = getattr(spec, "name", None)
    version = getattr(spec, "version", None)
    required = getattr(spec, "required_factors", ())
    try:
        factors = ", ".join(str(requirement) for requirement in required)
    except TypeError:  # a forged spec whose requirements are not iterable
        factors = repr(required)
    return f"{name}/{version} (requires {factors})"


def _listed(specs: Sequence[StrategySpec]) -> str:
    """``specs`` as a readable list of whole identities, or ``"none"``."""
    return ", ".join(_describe(spec) for spec in specs) if specs else "none"


def _sort_key(spec: StrategySpec) -> tuple[str, str, tuple[tuple[str, str], ...]]:
    """Order specs by name, then version, then the declared factor identities.

    The declared factor order is part of the key: a blend that reads ``momentum``
    before ``value`` is a different declaration from one that reads them the other
    way round, so the two must not sort as the same entry.
    """
    required = getattr(spec, "required_factors", ())
    factors = tuple(
        (
            str(getattr(requirement, "factor_name", requirement)),
            str(getattr(requirement, "factor_version", "")),
        )
        for requirement in (required if isinstance(required, tuple) else ())
    )
    return (str(getattr(spec, "name", "")), str(getattr(spec, "version", "")), factors)


def _factory_label(factory: object) -> str:
    """A short, stable name for a factory: the class or function it came from."""
    name = getattr(factory, "__name__", None)
    if isinstance(name, str) and name:
        module = getattr(factory, "__module__", None)
        return f"{module}.{name}" if isinstance(module, str) and module else name
    return f"{type(factory).__name__} instance"


def _hashable(value: object) -> bool:
    """Whether ``value`` can be a table key at all (a forged spec may not be)."""
    try:
        hash(value)
    except TypeError:
        return False
    return True


def _validated_name(value: object) -> str:
    """``value`` as a strategy name, or a :class:`StrategyRegistryError`.

    The name is returned exactly as given: this validates, it never repairs.
    """
    if not isinstance(value, str):
        raise StrategyRegistryError(
            f"a strategy name must be a string, got {type(value).__name__}: {value!r}"
        )
    if not _STRATEGY_NAME_RE.fullmatch(value):
        raise StrategyRegistryError(
            f"{value!r} is not a legal strategy name: lower-case letters, digits and '_' "
            f"only (matches {STRATEGY_NAME_PATTERN}).  Names are used exactly as written "
            "-- they are never trimmed, lower-cased or otherwise repaired"
        )
    return value


def _spec_of(strategy: Strategy, *, label: str) -> StrategySpec:
    """The validated spec of ``strategy``, or a :class:`StrategyRegistryError`."""
    try:
        spec = strategy.spec
    except StrategyError:
        raise
    except Exception as exc:  # a broken spec property is a broken factory
        raise StrategyRegistryError(
            f"{label} built a strategy whose spec could not be read: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(spec, StrategySpec):
        raise StrategyRegistryError(
            f"{label} built a strategy whose spec is not a StrategySpec "
            f"({type(spec).__name__}): a strategy's identity comes from a validated spec, "
            "because a name alone cannot say which factor versions it read"
        )
    return spec


def _built(factory: Callable[[], Strategy], *, when: str) -> Strategy:
    """Call ``factory`` once and check that it answered with a ``Strategy``."""
    label = _factory_label(factory)
    try:
        strategy = factory()
    except StrategyError:
        # The layer that knows what went wrong already said it precisely.
        raise
    except Exception as exc:
        raise StrategyRegistryError(
            f"{label} raised {type(exc).__name__} while {when} a strategy: {exc}"
        ) from exc

    if not isinstance(strategy, Strategy):
        raise StrategyRegistryError(
            f"{label} must build a Strategy, got {type(strategy).__name__}: a factory is a "
            "parameterless callable returning a Strategy (a closure over the rule's "
            "parameters, or the strategy class itself)"
        )
    return strategy


# ---------------------------------------------------------------------------
# StrategyRegistry -- the table
# ---------------------------------------------------------------------------
class StrategyRegistry:
    """An explicit ``StrategySpec -> factory`` table; every ``create`` is new.

    The table is built by registering factories, not instances, and there is no
    way to register under an identity of the caller's choosing: the identity is
    always the whole :class:`~strategy_sys.contracts.StrategySpec` the strategy
    declares, required factor versions included.  Lookup never renames, sorts or
    normalizes anything -- a spec is either in the table or it is not.

    A factory has to build a *new* object per call.  The table keeps every
    instance it has built on record -- the registration probe and every strategy
    already returned -- and refuses one that comes back a second time, whether it
    comes back at a further ``register`` or at a ``create``, because a shared
    instance would carry state from one run into the next.

    Examples
    --------
    >>> from strategy_sys.registry import StrategyRegistry
    >>> registry = StrategyRegistry()
    >>> registry.specs
    ()
    >>> len(registry)
    0
    """

    def __init__(self) -> None:
        """Start an empty table; register factories to fill it."""
        self._factories: dict[StrategySpec, Callable[[], Strategy]] = {}
        # Every instance this table has built, keyed by ``id`` and never by spec:
        # the strong reference pins the address, so a hit means *that object*, and
        # one instance cannot be laundered through a second spec.
        self._issued: dict[int, Strategy] = {}

    # -- writing ------------------------------------------------------------
    def register(self, factory: Callable[[], Strategy]) -> StrategySpec:
        """Register ``factory`` under its strategy's whole spec and return it.

        Parameters
        ----------
        factory:
            A parameterless callable returning a :class:`Strategy` -- a closure
            over the rule's parameters, or the strategy class itself.  It is
            called once here, to learn the spec, and once per :meth:`create`
            call.  The instance built here is on record as used, so a factory
            that hands it out again -- as a later registration probe, or at a
            :meth:`create` -- is refused rather than allowed to share one object.

        Returns
        -------
        StrategySpec
            The complete identity taken from the strategy's own
            :class:`~strategy_sys.contracts.StrategySpec`: name, version and the
            exact factor versions it declared.

        Raises
        ------
        StrategyRegistryError
            If ``factory`` is not callable, if it raises (unless it raises a
            :class:`~strategy_sys.errors.StrategyError`, which is passed
            through), if it does not build a :class:`Strategy` with a valid
            :class:`~strategy_sys.contracts.StrategySpec`, if that spec is not
            hashable, if that complete spec is already registered, or if the
            instance it built is one this table has already used.
        """
        if not callable(factory):
            raise StrategyRegistryError(
                f"a strategy factory must be callable, got {type(factory).__name__}: pass a "
                "callable returning a Strategy (a closure over the rule's parameters, or the "
                "strategy class), not an instance"
            )

        label = _factory_label(factory)
        probe = _built(factory, when="registering")
        spec = _spec_of(probe, label=label)
        if not _hashable(spec):
            raise StrategyRegistryError(
                f"{label} built a strategy whose spec cannot be a table key "
                f"({_describe(spec)}): an unhashable spec is a spec that broke its own "
                "validation, and a registry is keyed by exactly that value"
            )
        existing = self._factories.get(spec)
        if existing is not None:
            problem = (
                f"{label} is already registered under it"
                if existing is factory
                else f"{label} would replace {_factory_label(existing)}"
            )
            raise StrategyRegistryError(
                f"strategy {_describe(spec)} is already registered: {problem}.  A registry "
                "never overwrites one factory with another, because which of the two won "
                f"would depend on registration order (registered: {_listed(self.specs)})"
            )

        # The probe counts as used from here on: it never leaves the table, but a
        # factory that returns it again is a singleton, not a factory.  Claiming it
        # through the ledger -- rather than writing to it -- is what makes a second
        # registration of an already used object fail *here*, instead of silently
        # replacing the ledger entry and only failing at the next create().
        self._claim(probe, label=_describe(spec))
        self._factories[spec] = factory
        return spec

    # -- reading ------------------------------------------------------------
    def create(self, spec: StrategySpec) -> Strategy:
        """Build a **new** strategy registered under the whole ``spec``.

        Parameters
        ----------
        spec:
            The complete registered identity: name, version *and* required
            factors.  A name alone cannot be passed -- one name may be
            registered under several factor versions; ask :meth:`find` for them
            and pass the spec that is meant.

        Returns
        -------
        Strategy
            A fresh instance from the registered factory, checked to be a
            :class:`Strategy` whose ``spec`` equals ``spec`` exactly, required
            factor versions included.  No object is ever returned twice -- not
            the registration probe, not an earlier answer under any spec -- so no
            state can travel from one run to the next.

        Raises
        ------
        StrategyRegistryError
            If ``spec`` is not a ``StrategySpec``, if it is not registered, if
            the factory raises (except ``StrategyError``, which is passed
            through), if it does not build a :class:`Strategy`, if it now builds
            a different spec, or if it hands back an instance this table has
            already used.
        """
        if not isinstance(spec, StrategySpec):
            raise StrategyRegistryError(
                "StrategyRegistry.create() needs the whole StrategySpec (name, version and "
                f"the exact factor versions), got {type(spec).__name__}: {spec!r}.  A strategy "
                "name can map to several factor versions, so a name alone can never select "
                "one -- use find(name) and pass the spec you mean"
            )
        if not _hashable(spec):
            raise StrategyRegistryError(
                f"strategy spec {_describe(spec)} cannot be looked up: it is not hashable, "
                "so it is not a value this table could have registered"
            )

        factory = self._factories.get(spec)
        if factory is None:
            raise StrategyRegistryError(
                f"strategy {_describe(spec)} is not registered; {self._available_hint(spec)}"
            )

        label = _factory_label(factory)
        strategy = _built(factory, when="building")
        built = _spec_of(strategy, label=label)
        if built != spec:
            raise StrategyRegistryError(
                f"the factory registered for {_describe(spec)} now builds "
                f"{_describe(built)}: a factory must always build the strategy it was "
                "registered under, factor versions included"
            )
        self._claim(strategy, label=_describe(spec))
        return strategy

    # -- the instance ledger ------------------------------------------------
    def _claim(self, strategy: Strategy, *, label: str) -> None:
        """Record ``strategy`` as used, or refuse a factory that reuses one.

        Called at both points where the table accepts an object -- the
        registration probe and every ``create`` -- so the ledger is written once,
        by one rule, and an instance is refused on its second appearance wherever
        that appearance happens.

        The check is by identity against *every* instance this table has built,
        not just the previous one and not just under this spec: a factory that
        replays an old object after several good calls, or one that smuggles a
        single object in under a second spec, is sharing state across runs just
        the same.  Nothing about the strategy is copied or reset; only the
        factory is refused, and nothing is recorded, so a refusal cannot leave
        the table half-registered.
        """
        if self._issued.get(id(strategy)) is strategy:
            raise StrategyRegistryError(
                f"the factory for {label} reused a {type(strategy).__name__} instance this "
                "registry had already used: a factory must build a new strategy for every "
                "registration and every create() call, because a shared instance would "
                "carry state from one run into the next"
            )
        self._issued[id(strategy)] = strategy

    def _available_hint(self, spec: StrategySpec) -> str:
        """What the table does hold, pointing at ``find`` when the name matches.

        A request for one factor version of a name that is registered under
        another version is the mistake this hint exists for: listing every spec
        again would hide it behind a wall of text, so the same-name siblings are
        named instead and :meth:`find` is offered as the way to choose.
        """
        name = getattr(spec, "name", None)
        siblings = (
            [candidate for candidate in self.specs if candidate.name == name]
            if isinstance(name, str)
            else []
        )
        if not siblings:
            return f"available strategies: {_listed(self.specs)}"
        return (
            f"the name {name!r} is registered for {len(siblings)} factor version(s) "
            f"({_listed(siblings)}).  A strategy is created by its whole spec, factor "
            "versions included -- ask find(name) for them and pass the one you mean"
        )

    @property
    def specs(self) -> tuple[StrategySpec, ...]:
        """Every registered spec, ordered by name, version and declared factors.

        A new tuple per call, sorted the same way every time, so a caller can
        print the table, diff two tables or build a report without the order
        depending on registration order.  Declared factor order is part of the
        sort key, because a blend that reads ``momentum`` before ``value`` is a
        different declaration from the reverse.
        """
        return tuple(sorted(self._factories, key=_sort_key))

    def find(self, name: str) -> tuple[StrategySpec, ...]:
        """Every registered spec whose name is ``name``, sorted; ``()`` if none.

        More than one entry means the same rule is registered against several
        factor versions (or several multi-factor blends).  The table never picks
        a default among them: choosing which factor version a run uses is a
        research decision, so the caller receives all of them and passes the one
        it means to :meth:`create`.

        Raises
        ------
        StrategyRegistryError
            If ``name`` is not a string, or not a legal strategy name (names are
            never trimmed or lower-cased).  A legal name that matches nothing is
            not an error: it returns an empty tuple.
        """
        canonical = _validated_name(name)
        return tuple(spec for spec in self.specs if spec.name == canonical)

    def __contains__(self, spec: object) -> bool:
        """``spec in registry``; anything that is not a registered spec is absent."""
        return isinstance(spec, StrategySpec) and _hashable(spec) and spec in self._factories

    def __len__(self) -> int:
        """How many complete strategy specs the table holds."""
        return len(self._factories)

    def __repr__(self) -> str:
        if not self._factories:
            return "StrategyRegistry()"
        return f"StrategyRegistry(specs=({_listed(self.specs)}))"


__all__ = ["StrategyRegistry"]
