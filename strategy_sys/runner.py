"""One strategy run for one decision date: panels in, validated result out.

What this module is
-------------------
``strategy_sys.runner`` connects the parts of the strategy layer that already
exist, and adds nothing of its own::

    StrategySpec.required_factors
            |
            v
    FactorStore.read(factor_name, factor_version, as_of_date)
            |                                       one panel per requirement
            v
    UniverseProvider.snapshot(as_of_date)
            |                                       the pool of that session
            v
    StrategyContext                                 one date, one universe, exact
            |                                       factor identities
            v
    Strategy.compute(context)
            |
            v
    StrategyResult                                  already validated, with the
                                                    full spec attached

``StrategyRunner``
    the orchestrator: one strategy, one date, exactly one result -- the very
    object the strategy returned, never a copy, a wrapper or a re-serialized
    version of it.
``StrategyRunError``
    the runner's own error, raised for orchestration failures only
    (:class:`~strategy_sys.errors.StrategyRunError`, defined with the rest of the
    strategy hierarchy).

What the runner deliberately does *not* know
--------------------------------------------
* **No concrete strategy.**  The caller hands over a strategy it has already
  built, so which rule runs is a decision made in one place -- the caller's,
  typically ``registry.create(spec)``.  The runner never receives the registry,
  never looks a strategy up by name and never picks a factor version: choosing
  a rule, and choosing which version of a factor it reads, is research
  configuration, not orchestration.
* **No frequency.**  A run is one strategy on one date.  Weekly, monthly or
  any other cadence is a decision-date schedule that produces those dates; the
  runner is the thing that runs *one* of them, and reusing the same runner
  across dates is exactly what makes a schedule cheap.
* **No trading calendar.**  A decision date is a label, not a session the
  runner validates: if nothing is stored for it,
  :meth:`~factor_sys.factor_store.FactorStore.read` says so, and that is the
  layer that owns the data -- the runner never nudges a date to a neighbouring
  session.
* **No storage, no formulas, no files.**  Panels arrive through the injected
  :class:`~factor_sys.factor_store.FactorStore`, so the runner never computes a
  factor, never reads a Parquet file, never caches, never writes a result and
  never needs to know a path.
* **No judgement about the output.**  An empty selection, all symbols scoring
  missing or a strategy that selects nothing are facts the result carries;
  turning a selection into positions is the portfolio layer's job.

The double check, and why it matters for provenance
---------------------------------------------------
A strategy is expected to return a result that names *its* spec, *this* date
and *this* universe.  Instead of trusting it, the runner compares the four
things a later backtest will rely on: ``strategy_spec`` (so the exact
``factor_name``/``factor_version`` pairs travel with the numbers),
``as_of_date`` (no look-ahead), ``universe`` (identity -- the same snapshot
object, not merely equal symbols) and ``factor_requirements``.  A difference in
any of them is refused rather than stored, because a result attributed to the
wrong strategy version or the wrong factor version is indistinguishable from a
correct one once it is a file.

Layering
--------
This module imports the standard library, the factor layer's *interfaces* and
validated objects, and the strategy contract.  It never imports pandas, the
market-data reader, the factor runner, the factor registry, a trading calendar,
a concrete strategy or a storage backend.

Examples
--------
.. code-block:: python

    runner = StrategyRunner(universe_provider=provider, factor_store=store)

    strategy = registry.create(spec)  # the caller's choice, never the runner's
    result = runner.run(strategy, as_of_date="2026-08-31")

    result.selected_symbols  # the selection of that one session
    result.strategy_spec  # the exact strategy, and the exact factor versions
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any

from factor_sys.contracts import FactorPanel
from factor_sys.factor_store import FactorStore
from factor_sys.universe import UniverseProvider, UniverseSnapshot
from strategy_sys.contracts import (
    FactorRequirement,
    Strategy,
    StrategyContext,
    StrategyResult,
    StrategySpec,
)
from strategy_sys.errors import StrategyError, StrategyRunError


class StrategyRunner:
    """Run one strategy for one decision date and verify what comes back.

    One run is always these steps, in this order:

    1. the strategy must be a :class:`~strategy_sys.contracts.Strategy` whose
       ``spec`` is a :class:`~strategy_sys.contracts.StrategySpec` -- that spec
       is what says which factors, and which factor versions, are read;
    2. ``as_of_date`` is validated on its own: a plain :class:`datetime.date` or
       a strict ISO ``"YYYY-MM-DD"`` string, never a timestamp and never a date
       moved to a neighbouring session;
    3. :meth:`~factor_sys.universe.UniverseProvider.snapshot` decides which
       symbols may be scored, and the snapshot must echo the date it was asked
       about;
    4. one panel per declared requirement is read from the
       :class:`~factor_sys.factor_store.FactorStore`, in the order the spec
       declares them, with the exact ``(factor_name, factor_version, date)`` key
       -- read, not tested for existence, and never computed or substituted;
    5. a :class:`~strategy_sys.contracts.StrategyContext` is built from that one
       snapshot and exactly those panels, which is where the panels are checked
       against the date, the universe and their own identity;
    6. ``strategy.compute(context)`` produces the result;
    7. the result must be a
       :class:`~strategy_sys.contracts.StrategyResult` describing *this* run --
       same spec, same date, same snapshot object, same requirements.

    The runner then returns that *same* object.  It validates a result, it never
    rewrites one, so a strategy's scores, ranks and selection travel out exactly
    as they were produced, with the provenance a later portfolio or backtest
    needs already attached.

    Parameters
    ----------
    universe_provider:
        Where the symbols for a date come from.  Only ``snapshot`` is ever
        called, so today's static pool can be replaced by a point-in-time
        implementation without this class changing.
    factor_store:
        Where the panels come from.  Only ``read`` is ever called: a missing
        panel is the store's :class:`~factor_sys.errors.FactorStoreError` to
        report, not something the runner papers over.

    Raises
    ------
    strategy_sys.errors.StrategyRunError
        If a dependency does not implement its interface.  Nothing else happens
        at construction: no universe is read, no panel is loaded, no calendar is
        resolved and no file is touched.

    Notes
    -----
    A runner keeps no state between runs -- the same instance can run different
    strategies for different dates, in any order -- and it holds no registry, no
    calendar and no store of results.  Building one costs nothing, which makes
    it safe to construct per run and safe to keep for a whole schedule.

    Examples
    --------
    .. code-block:: python

        runner = StrategyRunner(
            universe_provider=provider,
            factor_store=store,
        )
        result = runner.run(strategy, as_of_date="2026-08-31")
    """

    def __init__(
        self,
        universe_provider: UniverseProvider,
        factor_store: FactorStore,
    ) -> None:
        """Store the two dependencies this orchestrator is allowed to call."""
        if not isinstance(universe_provider, UniverseProvider):
            raise StrategyRunError(
                "universe_provider must be a UniverseProvider, got "
                f"{type(universe_provider).__name__}: the runner only knows the "
                "snapshot interface, so any pool implementation can be used"
            )
        if not isinstance(factor_store, FactorStore):
            raise StrategyRunError(
                "factor_store must be a FactorStore, got "
                f"{type(factor_store).__name__}: the runner reads stored panels "
                "through that interface only and never computes or parses one itself"
            )

        self._universe_provider = universe_provider
        self._factor_store = factor_store

    @property
    def universe_provider(self) -> UniverseProvider:
        """The provider this runner asks for a universe snapshot."""
        return self._universe_provider

    @property
    def factor_store(self) -> FactorStore:
        """The store this runner reads one panel per required factor from."""
        return self._factor_store

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"universe_provider={type(self._universe_provider).__name__}, "
            f"factor_store={type(self._factor_store).__name__})"
        )

    # -- the run -------------------------------------------------------------
    def run(self, strategy: Strategy, as_of_date: date | str) -> StrategyResult:
        """Score ``strategy`` for the session ``as_of_date`` and verify the result.

        Parameters
        ----------
        strategy:
            The rule to run, already built.  Only ``strategy.spec`` and
            ``strategy.compute`` are used, so the caller -- not this runner --
            decides which strategy and which factor versions are involved.
        as_of_date:
            The decision session, as a :class:`datetime.date` or a bare
            ISO-8601 ``"YYYY-MM-DD"`` string.  A ``datetime`` is refused even at
            midnight: a decision date carries no time of day and no timezone.
            The date is never moved to a trading day; if no panel is stored for
            it, that is the store's answer to give.

        Returns
        -------
        StrategyResult
            The object ``strategy.compute`` returned, unchanged and already
            validated against this run's spec, date, universe and factor
            requirements.

        Raises
        ------
        StrategyRunError
            If the strategy is not a strategy, its spec is not a
            ``StrategySpec``, the date is unusable, the provider answers with
            something that is not a snapshot for that date, the store returns
            something that is not a panel, the strategy raises an unrelated
            exception or returns something that is not a ``StrategyResult``, or
            the result describes a different strategy, date, universe or set of
            factor requirements than this run.
        strategy_sys.errors.StrategyContractError
            Propagated unchanged: the context or the result broke the contract.
        strategy_sys.errors.StrategyError
            Propagated unchanged from inside the strategy.
        factor_sys.errors.FactorError
            Propagated unchanged from the store (a missing or damaged panel) or
            from the provider.
        """
        spec = _declared_spec(strategy)
        decision = _decision_date(as_of_date)

        snapshot = _snapshot_for(self._universe_provider, decision)
        panels = _panels_for(self._factor_store, spec, decision)

        # StrategyContext owns the panel checks -- date, symbol coverage and the
        # requirement <-> panel identity -- so a panel that does not fit this
        # session is refused here, before the strategy can read it.
        context = StrategyContext(as_of_date=decision, universe=snapshot, factor_panels=panels)

        result = _computed_result(strategy, context, spec)
        _require_this_run(result, spec, decision, snapshot)
        return result


# ---------------------------------------------------------------------------
# The steps of a run, one function each
# ---------------------------------------------------------------------------
_ISO_DAY_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")


def _declared_spec(strategy: Any) -> StrategySpec:
    """The validated :class:`StrategySpec` of ``strategy``: what this run reads."""
    if not isinstance(strategy, Strategy):
        raise StrategyRunError(
            f"strategy must be a Strategy, got {type(strategy).__name__}: a run starts from "
            "an already built strategy (registry.create(spec) or a literal), never from a "
            "name the runner would have to look up"
        )
    spec = strategy.spec
    if not isinstance(spec, StrategySpec):
        raise StrategyRunError(
            f"{type(strategy).__name__}.spec must be a StrategySpec, got "
            f"{type(spec).__name__}: the runner reads which factors -- and which factor "
            "versions -- to store-read from there"
        )
    return spec


def _decision_date(value: Any) -> date:
    """The run's decision date as a plain :class:`datetime.date`.

    Exactly two forms are accepted: a :class:`datetime.date` and a bare ISO-8601
    ``"YYYY-MM-DD"`` string, which is what the strategy contract accepts as a
    session label.  A ``datetime`` is refused *even at midnight*, so a timestamp
    -- including a pandas ``Timestamp``, which is one -- can never arrive as a
    decision date with its time silently dropped.

    The check is deliberately re-declared here rather than imported from
    :mod:`strategy_sys.contracts`: the contract's parser is private, and the
    runner only needs to agree with it, not to reuse its internals.  The two
    accept exactly the same two forms, so a date the runner accepts always
    survives the context and the result that follow.
    """
    if isinstance(value, datetime):
        raise StrategyRunError(
            f"as_of_date must be a date, not the timestamp {value.isoformat()}: a decision "
            "date carries no time of day and no timezone -- pass date(...) or 'YYYY-MM-DD'"
        )
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        text = value.strip()
        if _ISO_DAY_RE.fullmatch(text):
            try:
                return date.fromisoformat(text)
            except ValueError:
                pass
    raise StrategyRunError(
        f"as_of_date must be a date or a strict ISO-8601 'YYYY-MM-DD' string, got {value!r}: "
        "a session label is never guessed, parsed from another format or moved to the "
        "nearest trading day"
    )


def _snapshot_for(provider: UniverseProvider, as_of: date) -> UniverseSnapshot:
    """Ask ``provider`` for the pool of ``as_of`` and check what came back."""
    snapshot = provider.snapshot(as_of)
    if not isinstance(snapshot, UniverseSnapshot):
        raise StrategyRunError(
            f"{type(provider).__name__}.snapshot() must return a UniverseSnapshot, got "
            f"{type(snapshot).__name__}"
        )
    if snapshot.as_of_date != as_of:
        raise StrategyRunError(
            f"{type(provider).__name__}.snapshot() answered for "
            f"{snapshot.as_of_date.isoformat()} when asked about {as_of.isoformat()}: a "
            "provider must echo the decision date it was given, never move it"
        )
    return snapshot


def _panels_for(
    store: FactorStore,
    spec: StrategySpec,
    as_of: date,
) -> dict[FactorRequirement, FactorPanel]:
    """Read one panel per declared requirement, in the spec's own order.

    The key is the requirement's exact ``(factor_name, factor_version)`` and the
    date is this run's session, so a strategy cannot silently depend on a
    version it did not declare.  ``read`` is the only store method used: no
    ``exists`` probe first (an absent panel is an error to report, not a branch
    to take) and no fallback, recomputation or substitution -- a missing panel
    stops the run with the store's own error.
    """
    panels: dict[FactorRequirement, FactorPanel] = {}
    for requirement in spec.required_factors:
        panel = store.read(requirement.factor_name, requirement.factor_version, as_of)
        if not isinstance(panel, FactorPanel):
            raise StrategyRunError(
                f"{type(store).__name__}.read({requirement.factor_name!r}, "
                f"{requirement.factor_version!r}, {as_of.isoformat()}) must return a "
                f"FactorPanel, got {type(panel).__name__}: a strategy only ever sees validated "
                "panels, never a raw frame read around the contract"
            )
        panels[requirement] = panel
    return panels


def _computed_result(
    strategy: Strategy,
    context: StrategyContext,
    spec: StrategySpec,
) -> StrategyResult:
    """``strategy.compute(context)``, with unrelated failures given a domain type."""
    try:
        result = strategy.compute(context)
    except StrategyError:
        # Already precise: a StrategyContractError, a StrategyRunError from a
        # nested run or anything else the strategy layer declared is what the
        # caller expects to catch, so re-wrapping it would only hide it.
        raise
    except Exception as exc:
        raise StrategyRunError(
            f"strategy {spec.name!r} v{spec.version} raised {type(exc).__name__} while "
            f"scoring {context.as_of_date.isoformat()}: {exc}"
        ) from exc

    if not isinstance(result, StrategyResult):
        raise StrategyRunError(
            f"strategy {spec.name!r} v{spec.version} must return a StrategyResult, got "
            f"{type(result).__name__}: an unvalidated frame, a bare tuple or a hand-made "
            "dictionary must not travel towards a portfolio"
        )
    return result


def _describe_requirements(requirements: tuple[FactorRequirement, ...]) -> str:
    """``[momentum_12_1/1.0.0, book_to_price/2.0.0]`` -- exact identities, in order."""
    return "[" + ", ".join(str(requirement) for requirement in requirements) + "]"


def _describe_spec(spec: StrategySpec) -> str:
    """A spec rendered with everything that makes it one identity."""
    return f"{spec.name}/{spec.version} requiring {_describe_requirements(spec.required_factors)}"


def _require_this_run(
    result: StrategyResult,
    spec: StrategySpec,
    as_of: date,
    snapshot: UniverseSnapshot,
) -> None:
    """Refuse a result that describes a different strategy, date or universe.

    This is the runtime half of the provenance chain.  The result already passed
    ``StrategyResult`` validation, so it is *a* valid cross-section; what is
    checked here is that it is *this run's* cross-section.  Comparing
    ``strategy_spec`` covers the strategy version **and** the exact
    ``(factor_name, factor_version)`` pairs it declares, so a result scored from
    another factor version can never be filed under this one -- which is what
    makes a later backtest able to explain its own numbers.

    The universe is compared by identity: the result must hold the very snapshot
    object this run was given, not a look-alike built from the same symbols, so
    the pool a result was computed against is as traceable as the date.

    Every mismatch is collected and reported at once: a runner that stopped at
    the first one would hide the fact that a strategy disagreed about two
    things at the same time.
    """
    problems: list[str] = []

    if result.strategy_spec != spec:
        problems.append(
            f"strategy_spec is {_describe_spec(result.strategy_spec)}, "
            f"expected {_describe_spec(spec)}"
        )
    if result.as_of_date != as_of:
        problems.append(
            f"as_of_date is {result.as_of_date.isoformat()}, expected {as_of.isoformat()}"
        )
    if result.universe is not snapshot:
        detail = (
            " (it holds the same symbols but is another object)"
            if result.universe == snapshot
            else f" (it is dated {result.universe.as_of_date.isoformat()} "
            f"with {len(result.universe)} symbol(s))"
        )
        problems.append(f"universe is not the snapshot this run was given{detail}")
    if result.factor_requirements != spec.required_factors:
        problems.append(
            "factor_requirements are "
            f"{_describe_requirements(result.factor_requirements)}, expected "
            f"{_describe_requirements(spec.required_factors)}"
        )

    if problems:
        raise StrategyRunError(
            f"the result returned for {as_of.isoformat()} does not belong to this run of "
            f"{spec.name!r} v{spec.version}",
            problems,
        )


__all__ = ["StrategyRunner"]
