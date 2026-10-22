"""The explicit factor table: a name, a factory, and nothing else.

What this module is
-------------------
``factor_sys.registry`` answers one question for everything above the runner:
*given the name a strategy mentions, how do I get a factor?*::

    registry = default_factor_registry()
    factor = registry.create("momentum_12_1")   # a fresh Momentum12_1
    result = runner.run(factor, "2024-12-31")

Nothing above the formula layer has to write ``from factors.momentum import
Momentum12_1``: a CLI flag, an analysis notebook, a strategy file or a backtest
can carry only the name, and this table turns that name into a factor.  Adding a
factor is then one explicit line in :func:`default_factor_registry` -- never an
edit to :class:`~factor_sys.runner.FactorRunner`.

``FactorRegistry``
    a table of ``name -> factory``, where a *factory* is a parameterless callable
    returning a :class:`~factor_sys.contracts.Factor` (usually the factor class
    itself)
``default_factor_registry()``
    a brand-new registry holding every factor the project ships with (today:
    ``momentum_12_1``)
``FactorRegistryError``
    a registration or lookup that cannot be honoured

What a registry is not
----------------------
* **not a cache**: it stores factories, and it never hands the same instance out
  twice.  Every ``create`` builds a new factor, so a factor that grows internal
  state (a cached frame, a lazily computed value, a counter) cannot leak that
  state into the next run.  A factory that ignores this -- a closure over one
  pre-built factor, a module-level singleton, a class with a cached instance --
  is refused instead of smuggling a shared object into two runs.
* **not a plugin loader**: no scanning, no entry points, no config file, no
  environment variable.  The whole table is explicit and readable in one place.
* **not a parameter space**: a factory takes no arguments, so one name is one
  factor with one immutable :class:`~factor_sys.contracts.FactorSpec`.
  Hyper-parameters belong to a later analysis layer.
* **not a formula layer**: it imports no pandas, does no arithmetic and knows no
  factor by name except in :func:`default_factor_registry`.

Registration
------------
``register(factory)`` builds the factor once, reads ``factor.spec.name`` and
files the factory under that name -- the *canonical* name, taken from the spec::

    registry = FactorRegistry()
    registry.register(Momentum12_1)        # -> "momentum_12_1"

There is no name argument, so the table can never disagree with the name the
factor writes into its panel.  The same name twice is an error, never a silent
overwrite: which of two implementations won would depend on import order, and a
backtest must never depend on that.  The factory is called once here (to learn
the name) and once per ``create`` call.  The instance built here counts as used:
it never leaves the registry, but a factory that would hand it back later has one
object, not a factory, and is refused at its first ``create``.

Names are used verbatim
-----------------------
A name is a factor name exactly as :class:`~factor_sys.contracts.FactorSpec`
defines it -- ``FACTOR_NAME_PATTERN``: lower-case letters, digits and ``_``.
Nothing here trims, lower-cases or otherwise repairs a name.  ``"Momentum_12_1"``
and ``"momentum_12_1 "`` are refused with the pattern in the message, because a
strategy file that says ``"Momentum_12_1"`` has a typo, and quietly resolving it
would hide that typo until the day it picks the wrong factor.

Creating
--------
``create(name)`` validates the requested name, looks up the factory, calls it and
checks the answer: a :class:`Factor` whose ``spec`` is a
:class:`~factor_sys.contracts.FactorSpec` and whose ``spec.name`` is still the
name it was registered under.  A factory that has been rewritten to build
something else (a closure over a mutable variable, an edited class) is caught
here, instead of producing a panel with an identity nobody registered.

A factory must also build a *new* object every time.  The registry keeps every
instance it has built for a name on record -- the registration probe and every
factor it has already returned -- and refuses to hand any of them out twice::

    singleton = Momentum12_1()
    registry.register(lambda: singleton)     # accepted: it is a legal Factor
    registry.create("momentum_12_1")         # FactorRegistryError: reused

Sharing the object is what breaks the promise, not calling the factory twice: a
factor with a cached frame, a lazily computed value or a counter would carry that
state from one run into the next, and neither the runner nor the store could see
it.  The registry does not copy, reset or otherwise touch the factor's state; it
only refuses the factory.  The record keeps a strong reference to each instance,
which is what makes a hit mean *that exact object* rather than an address that
happened to be reused -- affordable because a factor is its spec and parameters,
never the bars it computed on.

Errors
------
:class:`FactorRegistryError` covers registration, lookup and an untrustworthy
factory: a name that is not a string, a name that is not snake_case, a duplicate,
an unknown name, a non-callable factory, a factory that returns something other
than a :class:`Factor`, a factory that hands back an instance it has already
used, or a factory that raises.

A factory that raises a :class:`~factor_sys.errors.FactorError` keeps it: the
layer that knows what went wrong (``FactorContractError`` for an invalid spec,
for example) already says it precisely, and callers may branch on that type.
Every other exception is wrapped in :class:`FactorRegistryError` with the
original as ``__cause__``.

What is not implemented, on purpose
-----------------------------------
Unregistering, overwriting, aliases, versions as part of the key, parameterised
factories, plugin discovery, entry points, config-file or environment-driven
registration, and a mutable global "the registry".  Every call to
:func:`default_factor_registry` returns an independent table.

Purity and layering
-------------------
Standard library plus :mod:`factor_sys.contracts` and :mod:`factor_sys.errors`.
No ``data_sys``, no market-data reader, no runner, no store, no quality check, no
pandas/PyArrow/DuckDB, no file, no environment variable, no network.  The
``factors.momentum`` import lives *inside* :func:`default_factor_registry`, so the
mechanism never depends on the formulas it happens to know today.

Example
-------
>>> from factor_sys.registry import default_factor_registry
>>> registry = default_factor_registry()
>>> registry.names
('momentum_12_1',)
>>> "momentum_12_1" in registry
True
>>> factor = registry.create("momentum_12_1")
>>> factor.spec.name
'momentum_12_1'
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence

from factor_sys.contracts import FACTOR_NAME_PATTERN, Factor, FactorSpec
from factor_sys.errors import FactorError, FactorRegistryError

#: The one name rule, shared with ``FactorSpec``: never a second dialect.
_FACTOR_NAME_RE = re.compile(FACTOR_NAME_PATTERN)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def _listed(names: Sequence[str]) -> str:
    """``names`` as a readable list, or ``"none"`` for an empty table."""
    return ", ".join(repr(name) for name in names) if names else "none"


def _factory_label(factory: object) -> str:
    """A short, stable name for a factory: the class or function it came from."""
    name = getattr(factory, "__name__", None)
    if isinstance(name, str) and name:
        module = getattr(factory, "__module__", None)
        return f"{module}.{name}" if isinstance(module, str) and module else name
    return f"{type(factory).__name__} instance"


def _validated_name(value: object) -> str:
    """``value`` as a factor name, or a :class:`FactorRegistryError`.

    The name is returned exactly as given: this validates, it never repairs.
    """
    if not isinstance(value, str):
        raise FactorRegistryError(
            f"a factor name must be a string, got {type(value).__name__}: {value!r}"
        )
    if not _FACTOR_NAME_RE.fullmatch(value):
        raise FactorRegistryError(
            f"{value!r} is not a legal factor name: lower-case letters, digits and '_' "
            f"only (matches {FACTOR_NAME_PATTERN}).  Names are used exactly as written "
            "-- they are never trimmed, lower-cased or otherwise repaired"
        )
    return value


def _spec_of(factor: Factor, *, label: str) -> FactorSpec:
    """The validated spec of ``factor``, or a :class:`FactorRegistryError`."""
    spec = getattr(factor, "spec", None)
    if not isinstance(spec, FactorSpec):
        raise FactorRegistryError(
            f"{label} built a factor whose spec is not a FactorSpec "
            f"({type(spec).__name__}): a factor's name comes from a validated spec"
        )
    return spec


def _built(factory: Callable[[], Factor]) -> Factor:
    """Call ``factory`` once and check that it answered with a ``Factor``."""
    label = _factory_label(factory)
    try:
        factor = factory()
    except FactorError:
        # The layer that knows what went wrong already said it precisely.
        raise
    except Exception as exc:
        raise FactorRegistryError(
            f"{label} raised {type(exc).__name__} while building a factor: {exc}"
        ) from exc

    if not isinstance(factor, Factor):
        raise FactorRegistryError(
            f"{label} must build a Factor, got {type(factor).__name__}: a factory is a "
            "parameterless callable returning a Factor (usually the factor class itself)"
        )
    return factor


# ---------------------------------------------------------------------------
# FactorRegistry -- the table
# ---------------------------------------------------------------------------
class FactorRegistry:
    """An explicit ``name -> factory`` table; every ``create`` builds a new factor.

    The table is built by registering factories, not instances, and there is no
    way to register under a name of the caller's choosing: the name is always the
    one ``factory().spec.name`` declares.  Lookup never renames, sorts or
    normalizes anything -- a name is either in the table or it is not.

    A factory has to build a *new* object per call.  The table keeps every
    instance it has built for a name on record -- the registration probe and every
    factor already returned -- and refuses one that comes back a second time,
    because a shared instance would carry state from one run into the next.  That
    record is cheap: a factor holds its spec and parameters, never market data.

    Examples
    --------
    >>> from factor_sys.registry import FactorRegistry
    >>> registry = FactorRegistry()
    >>> registry.names
    ()
    >>> "momentum_12_1" in registry
    False
    """

    def __init__(self) -> None:
        """Start an empty table; register factories to fill it."""
        self._factories: dict[str, Callable[[], Factor]] = {}
        # Every instance this table has built for a name, keyed by ``id``: the
        # strong reference pins the address, so a hit means *that object*, and a
        # factory cannot launder one instance through many ``create`` calls.
        self._issued: dict[str, dict[int, Factor]] = {}

    # -- writing ------------------------------------------------------------
    def register(self, factory: Callable[[], Factor]) -> str:
        """Register ``factory`` under ``factory().spec.name`` and return that name.

        Parameters
        ----------
        factory:
            A parameterless callable returning a :class:`Factor` -- normally the
            factor class itself.  It is called once here, to learn the name, and
            once per :meth:`create` call.  The instance built here is on record as
            used: a factory that would hand it out again is refused at its first
            :meth:`create`.

        Returns
        -------
        str
            The canonical name, taken from the factor's own
            :class:`~factor_sys.contracts.FactorSpec`.

        Raises
        ------
        FactorRegistryError
            If ``factory`` is not callable, if it raises (unless it raises a
            :class:`~factor_sys.errors.FactorError`, which is passed through), if
            it does not return a :class:`Factor` with a valid
            :class:`~factor_sys.contracts.FactorSpec`, if the spec name is not a
            legal factor name, or if that name is already registered.
        """
        if not callable(factory):
            raise FactorRegistryError(
                f"a factor factory must be callable, got {type(factory).__name__}: pass "
                "the factor class (or any callable returning a Factor), not an instance"
            )

        label = _factory_label(factory)
        probe = _built(factory)
        name = _validated_name(_spec_of(probe, label=label).name)
        existing = self._factories.get(name)
        if existing is not None:
            problem = (
                f"{label} is already registered under that name"
                if existing is factory
                else f"{label} would replace {_factory_label(existing)}"
            )
            raise FactorRegistryError(
                f"factor {name!r} is already registered: {problem}.  A registry never "
                f"overwrites one factory with another (registered: {_listed(self.names)})"
            )

        # The probe counts as used from here on: it never leaves the table, but a
        # factory that returns it again is a singleton, not a factory.
        self._issued[name] = {id(probe): probe}
        self._factories[name] = factory
        return name

    # -- reading ------------------------------------------------------------
    def create(self, name: str) -> Factor:
        """Build a **new** factor registered as ``name``.

        Parameters
        ----------
        name:
            A legal factor name, exactly as registered (no trimming, no case
            folding).  An unknown name lists what the table does hold.

        Returns
        -------
        Factor
            A fresh instance from the registered factory, checked to be a
            :class:`Factor` with a valid spec whose ``name`` is still ``name``.
            No object is ever returned twice -- not the registration probe, not
            an earlier answer -- so no state can travel from one run to the next.

        Raises
        ------
        FactorRegistryError
            If ``name`` is not a string or not a legal factor name, if nothing is
            registered under it, if the factory raises (except ``FactorError``,
            which is passed through), if it does not build a :class:`Factor`, if
            it now builds a factor registered under a different name, or if it
            hands back an instance this name has already used.
        """
        canonical = _validated_name(name)
        factory = self._factories.get(canonical)
        if factory is None:
            raise FactorRegistryError(
                f"factor {canonical!r} is not registered; available factors: {_listed(self.names)}"
            )

        label = _factory_label(factory)
        factor = _built(factory)
        spec = _spec_of(factor, label=label)
        if spec.name != canonical:
            raise FactorRegistryError(
                f"the factory registered as {canonical!r} now builds {spec.name!r}: a "
                "factory must always build the factor it was registered under"
            )
        self._claim(canonical, factor, label=label)
        return factor

    # -- the instance ledger ------------------------------------------------
    def _claim(self, name: str, factor: Factor, *, label: str) -> None:
        """Record ``factor`` as handed out, or refuse a factory that reuses one.

        The check is by identity against *every* instance built for ``name``, not
        just the previous one: a factory that replays an old object after several
        good calls is sharing state across runs just the same.  Nothing about the
        factor is copied or reset -- only the factory is refused.
        """
        issued = self._issued.get(name)
        if issued is not None and issued.get(id(factor)) is factor:
            raise FactorRegistryError(
                f"the factory registered as {name!r} reused a factor instance it had "
                f"already used ({label}): every create() must build a new factor, because "
                "a shared instance would carry state from one run into the next"
            )
        self._issued.setdefault(name, {})[id(factor)] = factor

    @property
    def names(self) -> tuple[str, ...]:
        """Every registered name, in alphabetical order (a new tuple per call)."""
        return tuple(sorted(self._factories))

    def __contains__(self, name: object) -> bool:
        """``name in registry``; anything that is not a registered string is absent."""
        return isinstance(name, str) and name in self._factories

    def __len__(self) -> int:
        """How many factors the table holds."""
        return len(self._factories)

    def __repr__(self) -> str:
        return f"FactorRegistry(names={self.names!r})" if self._factories else "FactorRegistry()"


# ---------------------------------------------------------------------------
# The built-in table
# ---------------------------------------------------------------------------
def default_factor_registry() -> FactorRegistry:
    """A new registry holding every built-in factor (today: ``momentum_12_1``).

    Each call builds an independent table, so a caller that registers its own
    factor cannot change what the next caller sees.  Adding a factor to the
    project means writing its module and adding one explicit line here; nothing
    else -- the runner, the store, the quality check -- has to change.

    Examples
    --------
    >>> registry = default_factor_registry()
    >>> registry.names
    ('momentum_12_1',)
    """
    # Imported here, not at module level: the mechanism does not depend on the
    # formulas at import time, and a factor module may import factor_sys itself.
    from factors.momentum import Momentum12_1

    registry = FactorRegistry()
    registry.register(Momentum12_1)
    return registry


__all__ = ["FactorRegistry", "FactorRegistryError", "default_factor_registry"]
