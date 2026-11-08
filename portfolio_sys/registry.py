"""The explicit construction table: a full ``PortfolioConstructionSpec``, a factory, nothing else.

What this module is
-------------------
``portfolio_sys.registry`` answers one question for everything above the portfolio
layer: *given the construction a caller means, how do I get an instance of it?*::

    registry = PortfolioConstructorRegistry()
    registry.register(CappedScoreWeightedConstructor)
    constructor = registry.create(spec)      # a brand-new constructor

Nothing above the portfolio layer has to write ``from portfolio_sys.constructors
import CappedScoreWeightedConstructor``: a future research script, a scheduler or a
backtest can carry a
:class:`~portfolio_sys.contracts.PortfolioConstructionSpec`, and this table turns
that spec into a constructor.  Adding a rule is then one explicit ``register`` line
in the caller -- never an edit to the registry and never an edit to a runner.

``PortfolioConstructorRegistry``
    a table of ``PortfolioConstructionSpec -> factory``, where a *factory* is a
    parameterless callable returning a
    :class:`~portfolio_sys.contracts.PortfolioConstructor`
``PortfolioRegistryError``
    a registration or a request that cannot be honoured

The key is the whole spec
-------------------------
The identity of a construction is the rule **and its version**, never the numbers it
was applied with::

    PortfolioConstructionSpec("capped_score_weighted", "1.0.0")
    PortfolioConstructionSpec("capped_score_weighted", "2.0.0")

Those two share a name and assign weights differently, so they are two rules and must
be able to sit in one table side by side -- that is how a new weighting scheme is
published without losing the old one.  A table keyed by ``name`` alone would collapse
them: the second registration would either overwrite the first or be refused as a
duplicate of an unrelated rule, and a stored target could no longer say which of the
two built it.

``cash_weight`` and ``max_single_weight`` are deliberately **not** part of the key.
They are the numbers one construction was run with, they live in
:class:`~portfolio_sys.contracts.PortfolioConstraints`, and the
:class:`~portfolio_sys.contracts.TargetPortfolio` records them beside the spec.  Two
runs of one rule with different constraints are two runs of the *same* rule, not two
rules -- so a factory that changes behaviour through hidden parameters is not a second
entry here: it needs a new version, and the caller decides that.

:class:`~portfolio_sys.contracts.PortfolioConstructionSpec` is already a frozen,
hashable value object, so the table uses it directly as its key: no ``Key`` type, no
glued ``"name@version"`` string, and nothing to parse back.

Registration
------------
``register(factory)`` builds the constructor once, reads ``constructor.spec`` and
files the factory under that complete spec -- the *canonical* identity, taken from the
constructor itself::

    registry = PortfolioConstructorRegistry()
    v1 = registry.register(lambda: CappedScoreWeightedConstructor())
    v1.name, v1.version        # ('capped_score_weighted', '1.0.0')
    len(registry)              # 1

There is no spec argument, so the table can never disagree with the identity the
constructor declares.  The same complete spec twice is an error, never a silent
overwrite: which of two implementations won would depend on registration order, and a
backtest must never depend on that.  The factory is called once here (to learn the
spec) and once per :meth:`PortfolioConstructorRegistry.create` call.

Creating
--------
``create(spec)`` looks the *whole* spec up and calls the factory again.  There is
deliberately no ``create(name)``: a name can map to several registered versions, so
"the constructor called X" is not a well-formed request.  A caller that has only a name
asks :meth:`PortfolioConstructorRegistry.find` and passes the spec it means.

Every ``create`` returns a **new** object.  The table keeps the instances it has handed
out on record -- the registration probe and every constructor already returned -- and
refuses one that comes back a second time: a shared instance would carry state (a
cached budget, a mutated candidate list) from one run into the next.

The ledger
----------
A factory is a *maker*, not a supplier.  One closure over a pre-built constructor::

    singleton = CappedScoreWeightedConstructor()
    registry.register(lambda: singleton)   # legal: it is a PortfolioConstructor
    registry.create(spec)                  # refused: the probe is already in use

and the refusal names the reason instead of handing one object to two callers.  The
record is keyed by ``id`` and holds a strong reference, so an address cannot be
recycled under it, and every hit is confirmed with ``is``: two constructors that
compare equal are still two instances, and both are accepted.  Identity -- never ``==``
and never ``hash`` -- decides.

Errors
------
:class:`PortfolioRegistryError` covers registration, lookup and an untrustworthy
factory: a non-callable factory, one that raises, one that builds something other than
a :class:`~portfolio_sys.contracts.PortfolioConstructor`, a constructor whose ``spec``
is not a :class:`~portfolio_sys.contracts.PortfolioConstructionSpec`, a duplicate spec,
a request that is not a ``PortfolioConstructionSpec``, a spec the table does not hold, a
factory that now builds a different spec, and a factory that hands back an instance it
has already used.  Every message names the construction by its whole identity -- name
and version -- never by a bare name, because a bare name is exactly the thing that
cannot tell two registered versions apart.

A factory that raises a :class:`~portfolio_sys.errors.PortfolioError` keeps it: the
layer that knows what went wrong (``PortfolioContractError`` for an invalid spec, for
example) already says it precisely, and callers may branch on that type.  Every other
exception is wrapped in :class:`PortfolioRegistryError` with the original as
``__cause__``.  A failure never leaves the table half-registered: a refused
registration adds nothing, and a refused create removes nothing.

What is not implemented, on purpose
-----------------------------------
Unregistering, overwriting, aliases, name-only lookup, parameterised factories, plugin
discovery, entry points, config-file or environment-driven registration, a
``default_portfolio_registry()``, and a mutable global "the registry".  Which rule,
which version, how much cash and which single-name cap a project runs are research
decisions, so the library never picks a default rule at import time and nothing is
registered automatically; a caller builds its own table and registers exactly what it
means to run.

Purity and layering
-------------------
Standard library plus :mod:`portfolio_sys.contracts` and :mod:`portfolio_sys.errors`.
No pandas, no numpy, no ``data_sys``, no ``factor_sys``, no ``strategy_sys``, no market
data, no factor panel, no calendar, no Parquet/DuckDB, no file, no environment
variable, no network, no runner and no backtest.  It never builds weights: it never
calls ``construct()``, never reads a ``StrategyResult`` and never saves a
``TargetPortfolio``.  The table knows no constructor by name: a rule arrives only as a
factory the caller passes in, so a new construction is one ``register`` call and never
an edit here.

Example
-------
>>> from portfolio_sys import PortfolioConstructionSpec, PortfolioConstructor
>>> class EqualWeight(PortfolioConstructor):
...     spec = PortfolioConstructionSpec(name="equal_weight", version="1.0.0")
...     def construct(self, context):
...         raise NotImplementedError("this example never builds weights")
>>> registry = PortfolioConstructorRegistry()
>>> registry.specs
()
>>> spec = registry.register(EqualWeight)
>>> spec.name, spec.version
('equal_weight', '1.0.0')
>>> spec in registry
True
>>> registry.create(spec) is not registry.create(spec)
True
>>> [f"{found.name}/{found.version}" for found in registry.find("equal_weight")]
['equal_weight/1.0.0']
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence

from portfolio_sys.contracts import (
    CONSTRUCTION_NAME_PATTERN,
    PortfolioConstructionSpec,
    PortfolioConstructor,
)
from portfolio_sys.errors import PortfolioError, PortfolioRegistryError

#: The one name rule, shared with ``PortfolioConstructionSpec``: never a second dialect.
_CONSTRUCTION_NAME_RE = re.compile(CONSTRUCTION_NAME_PATTERN)


# ---------------------------------------------------------------------------
# Readable identities
# ---------------------------------------------------------------------------
def _describe(spec: object) -> str:
    """A whole construction identity as one readable token: ``name/version``.

    Only for messages and logs; specs are compared as objects, never as strings.
    """
    name = getattr(spec, "name", None)
    version = getattr(spec, "version", None)
    return f"{name}/{version}"


def _listed(specs: Sequence[PortfolioConstructionSpec]) -> str:
    """``specs`` as a readable list of whole identities, or ``"none"``."""
    return ", ".join(_describe(spec) for spec in specs) if specs else "none"


def _sort_key(spec: PortfolioConstructionSpec) -> tuple[str, str]:
    """Order specs by name, then version -- the two halves of their identity.

    The pair is read defensively (a forged spec may hold anything) but never
    repaired: sorting only decides the order of a listing, never membership.
    """
    return (str(getattr(spec, "name", "")), str(getattr(spec, "version", "")))


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
    """``value`` as a construction name, or a :class:`PortfolioRegistryError`.

    The name is returned exactly as given: this validates, it never repairs.
    """
    if not isinstance(value, str):
        raise PortfolioRegistryError(
            f"a construction name must be a string, got {type(value).__name__}: {value!r}"
        )
    if not _CONSTRUCTION_NAME_RE.fullmatch(value):
        raise PortfolioRegistryError(
            f"{value!r} is not a legal construction name: lower-case letters, digits and '_' "
            f"only (matches {CONSTRUCTION_NAME_PATTERN}).  Names are used exactly as written "
            "-- they are never trimmed, lower-cased or otherwise repaired"
        )
    return value


def _spec_of(constructor: PortfolioConstructor, *, label: str) -> PortfolioConstructionSpec:
    """The validated spec of ``constructor``, or a :class:`PortfolioRegistryError`."""
    try:
        spec = constructor.spec
    except PortfolioError:
        raise
    except Exception as exc:  # a broken spec property is a broken factory
        raise PortfolioRegistryError(
            f"{label} built a constructor whose spec could not be read: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(spec, PortfolioConstructionSpec):
        raise PortfolioRegistryError(
            f"{label} built a constructor whose spec is not a PortfolioConstructionSpec "
            f"({type(spec).__name__}): a rule's identity comes from a validated spec, because "
            "a name alone cannot say which version of the rule built the weights"
        )
    return spec


def _built(factory: Callable[[], PortfolioConstructor], *, when: str) -> PortfolioConstructor:
    """Call ``factory`` once and check that it answered with a ``PortfolioConstructor``."""
    label = _factory_label(factory)
    try:
        constructor = factory()
    except PortfolioError:
        # The layer that knows what went wrong already said it precisely.
        raise
    except Exception as exc:
        raise PortfolioRegistryError(
            f"{label} raised {type(exc).__name__} while {when} a constructor: {exc}"
        ) from exc

    if not isinstance(constructor, PortfolioConstructor):
        raise PortfolioRegistryError(
            f"{label} must build a PortfolioConstructor, got {type(constructor).__name__}: a "
            "factory is a parameterless callable returning a PortfolioConstructor (the "
            "constructor class itself, or a closure over the rule's parameters)"
        )
    return constructor


# ---------------------------------------------------------------------------
# PortfolioConstructorRegistry -- the table
# ---------------------------------------------------------------------------
class PortfolioConstructorRegistry:
    """An explicit ``PortfolioConstructionSpec -> factory`` table; every ``create`` is new.

    The table is built by registering factories, not instances, and there is no way to
    register under an identity of the caller's choosing: the identity is always the
    whole :class:`~portfolio_sys.contracts.PortfolioConstructionSpec` the constructor
    declares -- name *and* version.  Lookup never renames, sorts or normalizes anything:
    a spec is either in the table or it is not, and ``cash_weight`` /
    ``max_single_weight`` are constraints of a construction, never part of this key.

    A factory has to build a *new* object per call.  The table keeps every instance it
    has built on record -- the registration probe and every constructor already
    returned -- and refuses one that comes back a second time, whether it comes back at
    a further ``register`` or at a ``create``, because a shared instance would carry
    state from one run into the next.

    Examples
    --------
    >>> from portfolio_sys.registry import PortfolioConstructorRegistry
    >>> registry = PortfolioConstructorRegistry()
    >>> registry.specs
    ()
    >>> len(registry)
    0
    >>> repr(registry)
    'PortfolioConstructorRegistry()'
    """

    def __init__(self) -> None:
        """Start an empty table; register factories to fill it."""
        self._factories: dict[PortfolioConstructionSpec, Callable[[], PortfolioConstructor]] = {}
        # Every instance this table has built, keyed by ``id`` and never by spec: the
        # strong reference pins the address, so a hit means *that object*, and one
        # instance cannot be laundered through a second spec.
        self._issued: dict[int, PortfolioConstructor] = {}

    # -- writing ------------------------------------------------------------
    def register(self, factory: Callable[[], PortfolioConstructor]) -> PortfolioConstructionSpec:
        """Register ``factory`` under its constructor's whole spec and return it.

        Parameters
        ----------
        factory:
            A parameterless callable returning a
            :class:`~portfolio_sys.contracts.PortfolioConstructor` -- the constructor
            class itself, or a closure over the rule's parameters.  It is called once
            here, to learn the spec, and once per :meth:`create` call.  The instance
            built here is on record as used, so a factory that hands it out again -- as
            a later registration probe, or at a :meth:`create` -- is refused rather than
            allowed to share one object.

        Returns
        -------
        PortfolioConstructionSpec
            The complete identity taken from the constructor's own
            :class:`~portfolio_sys.contracts.PortfolioConstructionSpec`: the rule's
            name and the version of its behaviour.

        Raises
        ------
        PortfolioRegistryError
            If ``factory`` is not callable, if it raises (unless it raises a
            :class:`~portfolio_sys.errors.PortfolioError`, which is passed through), if
            it does not build a :class:`PortfolioConstructor` with a valid
            :class:`~portfolio_sys.contracts.PortfolioConstructionSpec`, if that spec is
            not hashable, if that complete spec is already registered, or if the
            instance it built is one this table has already used.
        """
        if not callable(factory):
            raise PortfolioRegistryError(
                f"a constructor factory must be callable, got {type(factory).__name__}: pass a "
                "callable returning a PortfolioConstructor (the constructor class, or a closure "
                "over the rule's parameters), not an instance"
            )

        label = _factory_label(factory)
        probe = _built(factory, when="registering")
        spec = _spec_of(probe, label=label)
        if not _hashable(spec):
            raise PortfolioRegistryError(
                f"{label} built a constructor whose spec cannot be a table key "
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
            raise PortfolioRegistryError(
                f"construction {_describe(spec)} is already registered: {problem}.  A registry "
                "never overwrites one factory with another, because which of the two won would "
                f"depend on registration order (registered: {_listed(self.specs)})"
            )

        # The probe counts as used from here on: it never leaves the table, but a factory
        # that returns it again is a singleton, not a factory.  Claiming it through the
        # ledger -- rather than writing to it -- is what makes a second registration of an
        # already used object fail *here*, instead of silently replacing the ledger entry
        # and only failing at the next create().
        self._claim(probe, label=_describe(spec))
        self._factories[spec] = factory
        return spec

    # -- reading ------------------------------------------------------------
    def create(self, spec: PortfolioConstructionSpec) -> PortfolioConstructor:
        """Build a **new** constructor registered under the whole ``spec``.

        Parameters
        ----------
        spec:
            The complete registered identity: the rule's name *and* the version of its
            behaviour.  A name alone cannot be passed -- one name may be registered
            under several versions; ask :meth:`find` for them and pass the spec that is
            meant.  The constraints are not part of it either: ``cash_weight`` and
            ``max_single_weight`` are handed to ``construct()`` through the context.

        Returns
        -------
        PortfolioConstructor
            A fresh instance from the registered factory, checked to be a
            :class:`PortfolioConstructor` whose ``spec`` equals ``spec`` exactly.  No
            object is ever returned twice -- not the registration probe, not an earlier
            answer under any spec -- so no state can travel from one run to the next.

        Raises
        ------
        PortfolioRegistryError
            If ``spec`` is not a ``PortfolioConstructionSpec``, if it is not registered,
            if the factory raises (except ``PortfolioError``, which is passed through),
            if it does not build a :class:`PortfolioConstructor`, if it now builds a
            different spec, or if it hands back an instance this table has already used.
        """
        if not isinstance(spec, PortfolioConstructionSpec):
            raise PortfolioRegistryError(
                "PortfolioConstructorRegistry.create() needs the whole PortfolioConstructionSpec "
                f"(name and version), got {type(spec).__name__}: {spec!r}.  A construction name "
                "can map to several versions, so a name alone can never select one -- use "
                "find(name) and pass the spec you mean.  cash_weight and max_single_weight are "
                "not part of it either: they are PortfolioConstraints, handed to construct() "
                "through the context"
            )
        if not _hashable(spec):
            raise PortfolioRegistryError(
                f"construction spec {_describe(spec)} cannot be looked up: it is not hashable, "
                "so it is not a value this table could have registered"
            )

        factory = self._factories.get(spec)
        if factory is None:
            raise PortfolioRegistryError(
                f"construction {_describe(spec)} is not registered; {self._available_hint(spec)}"
            )

        label = _factory_label(factory)
        constructor = _built(factory, when="building")
        built = _spec_of(constructor, label=label)
        if built != spec:
            raise PortfolioRegistryError(
                f"the factory registered for {_describe(spec)} now builds "
                f"{_describe(built)}: a factory must always build the constructor it was "
                "registered under, version included"
            )
        self._claim(constructor, label=_describe(spec))
        return constructor

    # -- the instance ledger ------------------------------------------------
    def _claim(self, constructor: PortfolioConstructor, *, label: str) -> None:
        """Record ``constructor`` as used, or refuse a factory that reuses one.

        Called at both points where the table accepts an object -- the registration probe
        and every ``create`` -- so the ledger is written once, by one rule, and an
        instance is refused on its second appearance wherever that appearance happens.

        The check is by identity against *every* instance this table has built, not just
        the previous one and not just under this spec: a factory that replays an old
        object after several good calls, or one that smuggles a single object in under a
        second spec, is sharing state across runs just the same.  Nothing about the
        constructor is copied or reset; only the factory is refused, and nothing is
        recorded, so a refusal cannot leave the table half-registered.
        """
        if self._issued.get(id(constructor)) is constructor:
            raise PortfolioRegistryError(
                f"the factory for {label} reused a {type(constructor).__name__} instance this "
                "registry had already used: a factory must build a new constructor for every "
                "registration and every create() call, because a shared instance would carry "
                "state from one run into the next"
            )
        self._issued[id(constructor)] = constructor

    def _available_hint(self, spec: PortfolioConstructionSpec) -> str:
        """What the table does hold, pointing at ``find`` when the name matches.

        A request for one version of a name that is registered under another version is
        the mistake this hint exists for: listing every spec again would hide it behind a
        wall of text, so the same-name siblings are named instead and :meth:`find` is
        offered as the way to choose.
        """
        name = getattr(spec, "name", None)
        siblings = (
            [candidate for candidate in self.specs if candidate.name == name]
            if isinstance(name, str)
            else []
        )
        if not siblings:
            return f"registered constructions: {_listed(self.specs)}"
        return (
            f"the name {name!r} is registered for {len(siblings)} version(s) "
            f"({_listed(siblings)}).  A constructor is created by its whole spec, version "
            "included -- ask find(name) for them and pass the one you mean"
        )

    @property
    def specs(self) -> tuple[PortfolioConstructionSpec, ...]:
        """Every registered spec, ordered by name and then by version.

        A new tuple per call, sorted the same way every time, so a caller can print the
        table, diff two tables or build a report without the order depending on
        registration order.
        """
        return tuple(sorted(self._factories, key=_sort_key))

    def find(self, name: str) -> tuple[PortfolioConstructionSpec, ...]:
        """Every registered spec whose name is ``name``, sorted; ``()`` if none.

        More than one entry means the same rule is registered under several versions.
        The table never picks a default among them: choosing which version a run uses is
        a research decision, so the caller receives all of them and passes the one it
        means to :meth:`create`.

        Raises
        ------
        PortfolioRegistryError
            If ``name`` is not a string, or not a legal construction name (names are never
            trimmed or lower-cased).  A legal name that matches nothing is not an error:
            it returns an empty tuple.
        """
        canonical = _validated_name(name)
        return tuple(spec for spec in self.specs if spec.name == canonical)

    def __contains__(self, spec: object) -> bool:
        """``spec in registry``; anything that is not a registered spec is absent."""
        return (
            isinstance(spec, PortfolioConstructionSpec)
            and _hashable(spec)
            and spec in self._factories
        )

    def __len__(self) -> int:
        """How many complete construction specs the table holds."""
        return len(self._factories)

    def __repr__(self) -> str:
        if not self._factories:
            return "PortfolioConstructorRegistry()"
        return f"PortfolioConstructorRegistry(specs=({_listed(self.specs)}))"


__all__ = ["PortfolioConstructorRegistry"]
