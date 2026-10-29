"""The stable strategy contract: what a strategy is given, and what it returns.

What the contract is
--------------------
``strategy_sys.contracts`` defines the only objects a strategy, its runner and
(later) the portfolio constructor exchange:

``FactorRequirement``
    one exact factor identity a strategy depends on: a name *and* a version.
``StrategySpec``
    what a strategy is and what it needs: its own name and version, plus the
    factors it scores.
``StrategyContext``
    what a strategy may look at on one decision date: the universe snapshot and
    one already-computed panel per factor.
``StrategyResult``
    what a strategy returns: one row per symbol of the universe -- a score, a
    rank and a selection flag -- plus the full :class:`StrategySpec` it was
    produced under, so the exact factor versions travel with it.
``Strategy``
    the abstract scoring interface itself.

What a strategy is *not*
------------------------
A strategy answers exactly one question -- *on this decision date, which symbols
do these panels select, with which score, in which order?*  The questions after
that belong to other layers and are deliberately absent here:

* weights, cash and position sizing -- the portfolio constructor;
* orders, sides, fill prices and executions -- the execution layer;
* turnover, costs and rebalancing -- the backtester;
* reading panels, files, market data or a trading calendar -- the strategy
  runner.

A :class:`StrategyResult` therefore has no column that could express a portfolio
or a trade.  :data:`FORBIDDEN_RESULT_COLUMNS` names those fields explicitly, so
the boundary fails with an explanation ("this belongs to a later layer") instead
of a silent extra column that some later step might honour.

Why the input is a mapping of validated panels
----------------------------------------------
The factor layer hands out :class:`~factor_sys.contracts.FactorPanel` objects:
one already-validated cross-section of one factor, one row per symbol of *its*
universe.  A strategy never sees bars and never sees a frame of raw values; it
receives a mapping from :class:`FactorRequirement` to ``FactorPanel`` and reads
one number per symbol from it.  Three things are checked at that boundary,
because each of them would otherwise produce a plausible-looking but meaningless
result:

* the panel is dated the same session as the context and as its universe;
* the panel covers the universe's symbols, **in the universe's order**;
* the requirement that keys the panel is the panel's own factor identity, so a
  context cannot be labelled ``momentum_12_1/2.0.0`` while holding the 1.0.0
  panel.

A context does not know which factors *a particular strategy* declared: "is this
context enough for that spec?" is asked by the strategy, or by its runner, by
comparing ``spec.required_factors`` with ``context.factor_panels``.  Keeping that
question out of the context is what lets one context serve any strategy.

Why a result carries its whole spec
-----------------------------------
"Top 10 by ``momentum_12_1``" computed from factor version ``1.0.0`` and the same
selection computed from version ``2.0.0`` may share both the *strategy* name and
the *strategy* version -- the ranking rule did not change -- while resting on
different numbers.  A result that recorded only ``strategy_name`` and
``strategy_version`` could not say which factor versions produced it, and that is
exactly what a later portfolio constructor, backtest ledger, report or cache key
has to know.

So :class:`StrategyResult` stores the strategy's immutable
:class:`StrategySpec` (:attr:`StrategyResult.strategy_spec`) and *derives* its
``strategy_name`` and ``strategy_version`` from it, instead of holding a second,
independently supplied identity that could drift away from the declared
dependencies.  :attr:`StrategyResult.factor_requirements` is the spec's own
``required_factors`` tuple, in the declared order, each entry carrying a factor
name **and** a factor version.

Those requirements are deliberately *not* repeated as per-row columns.  A
multi-factor strategy depends on several exact factors at once, so a flat
``factor_name`` / ``factor_version`` column would either multiply the rows by the
number of factors or silently keep one factor and drop the others.  A dependency
is a property of the whole cross-section, not of a symbol, so it lives on the
immutable result object -- and only there.

What a result cannot prove, and must not pretend to
---------------------------------------------------
A result holds no :class:`StrategyContext` and no panel: it cannot verify that
its frame really was computed from the factors it names.  That check needs the
invocation, and it belongs to the future strategy runner -- that
``result.strategy_spec == invoked_strategy.spec``, and that the context carried a
panel for every requirement in that spec.  This layer guarantees only what it
can: the dependency declaration is complete, immutable, and impossible to
desynchronize from the identity written into the frame's own columns.

Conventions reused from the factor system
-----------------------------------------
* A factor name follows :data:`factor_sys.contracts.FACTOR_NAME_PATTERN`, and a
  strategy name follows the same snake_case rule -- one naming convention for the
  whole project rather than a second one invented here.
* The universe is a :class:`~factor_sys.universe.UniverseSnapshot`, never a list
  of tickers: the snapshot already proves the symbols are valid, unique and
  dated, so the strategy layer only has to compare against them.
* Dates are timezone-naive :class:`datetime.date` values, exactly like every other
  date in the project.  A ``datetime`` is refused *even at midnight*, and a string
  must be a bare ``"YYYY-MM-DD"``: a decision date has no time of day and no
  timezone, so nothing is silently truncated.  A frame's ``datetime64`` date
  column (how pandas stores a date) is the one place a timestamp is normalized
  row by row -- and an intraday row is still refused.
* Validation collects every problem it finds into one
  :class:`~strategy_sys.errors.StrategyContractError`.
* :class:`StrategyContext` and :class:`StrategyResult` are compared by identity
  (``eq=False``), like :class:`~factor_sys.contracts.FactorPanel`: two results
  that merely hold equal frames are still two different measurements.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from types import MappingProxyType
from typing import Any

import pandas as pd

from factor_sys.contracts import FACTOR_NAME_PATTERN, FactorPanel
from factor_sys.universe import UniverseSnapshot
from strategy_sys.errors import StrategyContractError

# ---------------------------------------------------------------------------
# Column and naming contracts
# ---------------------------------------------------------------------------
#: Fixed column order of one strategy result (one strategy, one ``as_of_date``).
STRATEGY_RESULT_COLUMNS: tuple[str, ...] = (
    "as_of_date",
    "strategy_name",
    "strategy_version",
    "symbol",
    "score",
    "rank",
    "is_selected",
)

#: A strategy name is a snake_case identifier, e.g. ``top_n_momentum``.
#:
#: The rule is the factor system's own
#: (:data:`factor_sys.contracts.FACTOR_NAME_PATTERN`): a strategy and a factor are
#: named the same way, so a run directory, a log line or a report can hold both
#: without a second convention.
STRATEGY_NAME_PATTERN: str = FACTOR_NAME_PATTERN

#: Columns a strategy result must never carry: they belong to later layers.
#:
#: Naming them (instead of only rejecting "unknown" columns) turns a boundary
#: violation into an explained refusal: a weight, a quantity, an order side or a
#: fill price is not "one more column", it is a different layer's answer.
FORBIDDEN_RESULT_COLUMNS: tuple[str, ...] = (
    "weight",
    "target_weight",
    "current_weight",
    "market_value",
    "cash",
    "quantity",
    "shares",
    "side",
    "order",
    "order_id",
    "order_type",
    "fill_price",
    "price",
    "cost",
    "turnover",
)

_STRATEGY_NAME_RE = re.compile(STRATEGY_NAME_PATTERN)
#: A session label written as text: exactly ``YYYY-MM-DD``, nothing else.
_DATE_TEXT_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")

_DATE_COLUMN = "as_of_date"
_SYMBOL_COLUMN = "symbol"
_NAME_COLUMN = "strategy_name"
_VERSION_COLUMN = "strategy_version"
_SCORE_COLUMN = "score"
_RANK_COLUMN = "rank"
_SELECTED_COLUMN = "is_selected"


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------
def _describe(values: Iterable[Any], limit: int = 3) -> str:
    """A short, bounded rendering of offending values for an error message."""
    items = list(values)
    shown = ", ".join(repr(value) for value in items[:limit])
    if len(items) > limit:
        shown = f"{shown}, ... (+{len(items) - limit} more)"
    return f"[{shown}]"


def _unique(values: Iterable[Any]) -> list[Any]:
    """The distinct values in first-seen order."""
    distinct: list[Any] = []
    for value in values:
        if value not in distinct:
            distinct.append(value)
    return distinct


def _duplicates(values: Iterable[Any]) -> list[Any]:
    """The values that occur more than once, in first-seen order."""
    seen: set[Any] = set()
    repeated: list[Any] = []
    for value in values:
        if value in seen and value not in repeated:
            repeated.append(value)
        seen.add(value)
    return repeated


def _is_missing(value: Any) -> bool:
    """``True`` for ``None``, ``NaN``, ``NaT`` and pandas' ``NA``.

    Scalar cells only: a container is not missing, it is malformed, and the
    caller says so in its own words instead of asking pandas to guess.
    """
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):  # e.g. a list: not a scalar to test
        return False


def _session_date(value: Any, label: str) -> date:
    """Validate one session label and return it as a plain :class:`datetime.date`.

    Exactly two forms are accepted: a :class:`datetime.date` and a plain
    ISO-8601 ``"YYYY-MM-DD"`` string.

    A ``datetime`` is refused *even at midnight*.  ``2024-01-05 00:00`` is a
    timestamp, not a session label: keeping its date would let a timezone-aware
    or intraday moment arrive as a decision date that never was one, and the
    conversion would happen silently.  A string must therefore be a bare date
    too -- ``"2024-01-05T00:00"`` is refused rather than parsed.  The single
    place a timestamp is accepted at all is a frame's ``datetime64`` date column,
    which pandas uses to store dates; see :func:`_normalized_date_column`.
    """
    if isinstance(value, datetime):
        raise StrategyContractError(
            f"{label} must be a date, not the timestamp {value.isoformat()}: a decision date "
            "carries no time of day and no timezone -- pass date(...) or 'YYYY-MM-DD'"
        )
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        text = value.strip()
        if _DATE_TEXT_RE.fullmatch(text):
            try:
                return date.fromisoformat(text)
            except ValueError:
                pass
    raise StrategyContractError(
        f"{label} must be a date or an ISO-8601 'YYYY-MM-DD' string, got {value!r}"
    )


def _normalized_date_column(column: pd.Series, problems: list[str]) -> pd.Series:
    """Return ``column`` with ``datetime64`` cells turned into plain dates.

    A ``datetime64`` column is how pandas stores a session label, so it is
    normalized row by row instead of refused: a midnight timestamp becomes a
    plain :class:`datetime.date`, while a timestamp with a time of day -- or a
    whole timezone-aware column -- is reported exactly like the scalar rule of
    :func:`_session_date` reports it.  Any other dtype is handed on untouched, so
    :func:`_row_dates` sees a string or an object cell precisely as written.
    """
    if not pd.api.types.is_datetime64_any_dtype(column.dtype):
        return column

    moments = pd.to_datetime(column)
    if isinstance(moments.dtype, pd.DatetimeTZDtype):
        problems.append(
            "as_of_date must be a plain date column without a timezone, got dtype "
            f"{column.dtype}: a decision date is a session, not a moment in a timezone"
        )
        moments = moments.dt.tz_localize(None)

    normalized = moments.dt.normalize()
    intraday = moments.notna() & (moments != normalized)
    if bool(intraday.any()):
        problems.append(
            "as_of_date must be a session date, not a timestamp: a decision date has no time "
            f"of day, yet the column holds {_describe(moments[intraday].tolist())}"
        )
    return pd.Series(normalized.dt.date, index=column.index, dtype="object")


def _row_dates(values: pd.Series) -> tuple[list[date | None], list[str]]:
    """Validate every ``as_of_date`` cell; return ``(dates, problems)``."""
    dates: list[date | None] = []
    problems: list[str] = []
    for position, value in enumerate(values.tolist()):
        if _is_missing(value):
            problems.append(f"as_of_date is missing in row {position}")
            dates.append(None)
            continue
        try:
            dates.append(_session_date(value, "as_of_date"))
        except StrategyContractError as exc:
            problems.append(f"row {position}: {exc}")
            dates.append(None)
    return dates, problems


# ---------------------------------------------------------------------------
# FactorRequirement -- one exact factor identity
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FactorRequirement:
    """One exact factor identity a strategy depends on.

    The pair ``(factor_name, factor_version)`` *is* the identity: the same name
    with a new version is a different factor, so a strategy states both and never
    a single glued-together string.  Requirements are hashable, which is what lets
    a :class:`StrategyContext` key its panels by requirement and a strategy look
    one up with the same value it declared.

    Parameters
    ----------
    factor_name:
        snake_case identifier, validated with the project's own rule
        (:data:`factor_sys.contracts.FACTOR_NAME_PATTERN`), e.g.
        ``momentum_12_1``.
    factor_version:
        Non-empty label of the formula version, e.g. ``"1.0.0"``.  It is stored
        exactly as given, like :class:`~factor_sys.contracts.FactorSpec` stores
        it, so a requirement can be compared with a panel's own identity without
        guessing.

    Raises
    ------
    strategy_sys.errors.StrategyContractError
        If the name is not snake_case or the version is blank; every problem is
        reported at once.

    Examples
    --------
    >>> requirement = FactorRequirement(
    ...     factor_name="momentum_12_1",
    ...     factor_version="1.0.0",
    ... )
    >>> requirement.factor_name
    'momentum_12_1'
    >>> print(requirement)
    momentum_12_1/1.0.0
    >>> lookup = {requirement: "panel"}
    >>> lookup[FactorRequirement("momentum_12_1", "1.0.0")]
    'panel'
    """

    factor_name: str
    factor_version: str

    def __post_init__(self) -> None:
        problems: list[str] = []

        name = self.factor_name
        if not isinstance(name, str) or not name.strip():
            problems.append(f"factor_name must be a non-empty string, got {name!r}")
        elif not _STRATEGY_NAME_RE.match(name):
            problems.append(
                f"factor_name {name!r} must be snake_case: lower-case letters, digits and "
                f"'_' only (matches {FACTOR_NAME_PATTERN})"
            )

        version = self.factor_version
        if not isinstance(version, str) or not version.strip():
            problems.append(f"factor_version must be a non-empty string, got {version!r}")

        if problems:
            raise StrategyContractError("invalid FactorRequirement", problems)

    def __str__(self) -> str:
        """``momentum_12_1/1.0.0``: the identity as one readable token.

        Only for messages and logs.  Two requirements are compared as objects,
        never as strings, so this rendering is never parsed back.
        """
        return f"{self.factor_name}/{self.factor_version}"


# ---------------------------------------------------------------------------
# StrategySpec -- what a strategy is and what it needs
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StrategySpec:
    """Immutable declaration of one strategy: identity plus required factors.

    The spec is defined once next to the strategy and never changes while it
    runs; it is what makes a result traceable back to the code that produced it.
    It deliberately holds *no* parameters -- no ``top_n``, no weight, no cash
    ratio -- because those belong to a concrete strategy or to the portfolio
    layer, not to the identity of "which strategy, on which version".

    Parameters
    ----------
    name:
        snake_case identifier, e.g. ``top_n_momentum``.  It is written into every
        row of a :class:`StrategyResult` and later names the stored result.
    version:
        Non-empty string, e.g. ``"1.0.0"``.  A scoring change needs a new version:
        the same name plus a new version is a different strategy.
    required_factors:
        Non-empty, duplicate-free **tuple** of :class:`FactorRequirement`.  A
        single-factor strategy declares one, a multi-factor strategy declares
        several, and the tuple *is* the order in which the strategy reads its
        panels.  A list, a set or a generator is refused: a set has no order, a
        generator can be read only once, and neither should be able to decide
        which factor a blend sees first (a tuple also keeps the spec hashable,
        which a frozen dataclass holding a list would not).

    Raises
    ------
    strategy_sys.errors.StrategyContractError
        If the identity is unusable, or the requirements are not a tuple, are
        empty, hold something that is not a ``FactorRequirement``, or list the
        same factor twice.  All problems are reported at once.

    Examples
    --------
    >>> spec = StrategySpec(
    ...     name="top_n_momentum",
    ...     version="1.0.0",
    ...     required_factors=(FactorRequirement("momentum_12_1", "1.0.0"),),
    ... )
    >>> spec.name
    'top_n_momentum'
    >>> [str(requirement) for requirement in spec.required_factors]
    ['momentum_12_1/1.0.0']
    """

    name: str
    version: str
    required_factors: tuple[FactorRequirement, ...]

    def __post_init__(self) -> None:
        problems: list[str] = []

        name = self.name
        if not isinstance(name, str) or not name.strip():
            problems.append(f"name must be a non-empty string, got {name!r}")
        elif not _STRATEGY_NAME_RE.match(name):
            problems.append(
                f"name {name!r} must be snake_case: lower-case letters, digits and '_' only "
                f"(matches {STRATEGY_NAME_PATTERN})"
            )

        version = self.version
        if not isinstance(version, str) or not version.strip():
            problems.append(f"version must be a non-empty string, got {version!r}")

        declared = self.required_factors
        requirements: list[FactorRequirement] = []
        if not isinstance(declared, tuple):
            problems.append(
                "required_factors must be a tuple[FactorRequirement, ...], got "
                f"{type(declared).__name__}: a strategy states its factor order once, so a "
                "list, a set or a generator is refused -- declare "
                "(FactorRequirement(...), ...)"
            )
        else:
            if not declared:
                problems.append(
                    "required_factors must not be empty: a strategy scores at least one factor"
                )
            problems.extend(
                f"required_factors entry {value!r} must be a FactorRequirement, got "
                f"{type(value).__name__}"
                for value in declared
                if not isinstance(value, FactorRequirement)
            )
            requirements = [value for value in declared if isinstance(value, FactorRequirement)]
            identities = [
                (requirement.factor_name, requirement.factor_version)
                for requirement in requirements
            ]
            problems.extend(
                f"required_factors lists {identity[0]}/{identity[1]} more than once: a "
                "strategy declares one requirement per exact factor identity"
                for identity in _duplicates(identities)
            )

        if problems:
            raise StrategyContractError("invalid StrategySpec", problems)

        object.__setattr__(self, "required_factors", tuple(requirements))


# ---------------------------------------------------------------------------
# StrategyContext -- what a strategy may look at
# ---------------------------------------------------------------------------
def _panel_symbol_problems(
    label: str,
    actual: tuple[str, ...],
    expected: tuple[str, ...],
) -> list[str]:
    """Every way a panel's symbols can disagree with the universe's."""
    known = set(expected)
    present = set(actual)
    missing = [symbol for symbol in expected if symbol not in present]
    unknown = [symbol for symbol in actual if symbol not in known]
    if missing or unknown:
        problems: list[str] = []
        if missing:
            problems.append(f"{label} is missing universe symbol(s) {_describe(missing)}")
        if unknown:
            problems.append(
                f"{label} contains symbol(s) outside the universe: {_describe(unknown)}"
            )
        return problems
    if actual != expected:
        return [
            f"{label} lists the universe's symbols in a different order: expected "
            f"{_describe(expected)} in that order, got {_describe(actual)}"
        ]
    return []


def _validated_panels(
    panels: Any,
    as_of_date: date,
    universe: UniverseSnapshot | None,
    problems: list[str],
) -> Mapping[FactorRequirement, FactorPanel]:
    """Validate the requirement -> panel mapping; return the contract's own copy.

    ``problems`` is extended in place, so a context reports the dates, the symbol
    coverage and the identities of *all* its panels in one error.  The returned
    mapping is a read-only view of a fresh ``dict``: the caller's mapping can be
    changed afterwards without touching the context, and the context's mapping
    cannot be changed at all.
    """
    if isinstance(panels, pd.DataFrame):
        problems.append(
            "StrategyContext.factor_panels must be a mapping from FactorRequirement to "
            "FactorPanel, not a DataFrame: a bare frame has no factor identity"
        )
        return MappingProxyType({})
    if not isinstance(panels, Mapping):
        problems.append(
            "StrategyContext.factor_panels must be a mapping from FactorRequirement to "
            f"FactorPanel, got {type(panels).__name__}"
        )
        return MappingProxyType({})
    if not panels:
        problems.append(
            "StrategyContext.factor_panels must not be empty: a context carries one panel "
            "per factor it is given"
        )

    checked: dict[FactorRequirement, FactorPanel] = {}
    for key, panel in panels.items():
        if not isinstance(key, FactorRequirement):
            problems.append(
                "StrategyContext.factor_panels key must be a FactorRequirement, got "
                f"{type(key).__name__}: {key!r}"
            )
            continue
        label = f"StrategyContext.factor_panels[{str(key)!r}]"
        if not isinstance(panel, FactorPanel):
            problems.append(
                f"{label} must be a FactorPanel, got {type(panel).__name__}: the context "
                "never takes a raw DataFrame"
            )
            continue

        if panel.as_of_date != as_of_date:
            problems.append(
                f"{label} is dated {panel.as_of_date.isoformat()}, but the context is dated "
                f"{as_of_date.isoformat()}: every panel must describe this decision date"
            )
        if (panel.factor_name, panel.factor_version) != (key.factor_name, key.factor_version):
            problems.append(
                f"{label} holds the panel of {panel.factor_name}/{panel.factor_version}: a "
                "context key is the exact factor identity, never a label for another factor"
            )
        if universe is not None:
            problems.extend(_panel_symbol_problems(label, panel.universe, universe.symbols))

        checked[key] = panel

    return MappingProxyType(checked)


@dataclass(frozen=True, eq=False)
class StrategyContext:
    """Everything a strategy may look at on one decision date.

    A context is built by the strategy runner from panels the factor layer has
    already produced; a strategy only reads it.  Four guarantees make a strategy
    safe to reuse in a backtest:

    1. **One decision date.**  ``as_of_date`` is a timezone-naive
       :class:`datetime.date`, and every panel is dated the same session -- a
       panel from another date is refused instead of silently mixing two
       cross-sections.  A ``datetime`` is refused even at midnight, so no clock
       is ever dropped on the way in.
    2. **One universe.**  ``universe`` is the factor layer's
       :class:`~factor_sys.universe.UniverseSnapshot`, and every panel covers its
       symbols in its order, so ``context.factor_panels[requirement].frame``
       lines up with ``context.universe.symbols`` row for row.
    3. **Exact factor identity.**  Each key is the panel's own
       ``(factor_name, factor_version)``, so a panel cannot be mislabelled.
    4. **Isolation.**  The panels and the snapshot are immutable objects the
       context shares rather than copies, and ``factor_panels`` is a read-only
       view of the context's own dictionary -- changing the mapping a caller
       passed in afterwards cannot change the context.

    Parameters
    ----------
    as_of_date:
        The decision session: a :class:`datetime.date` or a bare ISO-8601
        ``"YYYY-MM-DD"`` string.  A ``datetime`` is refused even at midnight.
    universe:
        The :class:`~factor_sys.universe.UniverseSnapshot` for the same session.
    factor_panels:
        Mapping from :class:`FactorRequirement` to the
        :class:`~factor_sys.contracts.FactorPanel` of exactly that factor.

    Raises
    ------
    strategy_sys.errors.StrategyContractError
        If the date is unusable -- refused on its own, because nothing else can be
        checked without a session -- or if the universe is not a snapshot (or is
        dated another session), the mapping is empty or not a mapping, an entry is
        not a ``FactorRequirement``/``FactorPanel`` pair, a panel is dated another
        session, a panel's symbols differ from the universe's, or a key is not the
        panel's own identity.  Everything but the date is reported at once.

    Notes
    -----
    A context says nothing about which factors a particular strategy needs: it
    carries the panels it was given, and the strategy (or the runner) compares
    ``spec.required_factors`` with ``context.factor_panels``.  That is what lets
    one context serve a single-factor and a multi-factor strategy unchanged.

    Contexts are compared by identity (``eq=False``), because they hold panels
    whose equality is identity too.

    Examples
    --------
    >>> universe = UniverseSnapshot(
    ...     as_of_date="2024-01-05",
    ...     symbols=("AAPL", "MSFT"),
    ...     source="config/universe_seed.csv",
    ... )
    >>> momentum = pd.DataFrame(
    ...     {
    ...         "as_of_date": [pd.Timestamp("2024-01-05")] * 2,
    ...         "symbol": ["AAPL", "MSFT"],
    ...         "factor_name": ["momentum_12_1"] * 2,
    ...         "factor_version": ["1.0.0"] * 2,
    ...         "raw_value": [0.12, 0.35],
    ...         "is_eligible": [True, True],
    ...     }
    ... )
    >>> panel = FactorPanel(
    ...     momentum, universe=universe.symbols, as_of_date=universe.as_of_date
    ... )
    >>> context = StrategyContext(
    ...     as_of_date="2024-01-05",
    ...     universe=universe,
    ...     factor_panels={FactorRequirement("momentum_12_1", "1.0.0"): panel},
    ... )
    >>> context.as_of_date
    datetime.date(2024, 1, 5)
    >>> [str(requirement) for requirement in context.factor_panels]
    ['momentum_12_1/1.0.0']
    """

    as_of_date: date
    universe: UniverseSnapshot
    factor_panels: Mapping[FactorRequirement, FactorPanel] = field(repr=False)

    def __post_init__(self) -> None:
        as_of_date = _session_date(self.as_of_date, "StrategyContext.as_of_date")
        problems: list[str] = []

        universe = self.universe
        if not isinstance(universe, UniverseSnapshot):
            problems.append(
                "StrategyContext.universe must be a UniverseSnapshot (the factor layer's "
                f"validated pool), got {type(universe).__name__}"
            )
            # Nothing below can be compared against a pool that is not one; the
            # context still reports every other problem it can see.
            universe = None
        elif universe.as_of_date != as_of_date:
            problems.append(
                f"StrategyContext.universe is dated {universe.as_of_date.isoformat()}, but "
                f"the context is dated {as_of_date.isoformat()}: a context is one session"
            )

        panels = _validated_panels(self.factor_panels, as_of_date, universe, problems)

        if problems:
            raise StrategyContractError("invalid StrategyContext", problems)

        object.__setattr__(self, "as_of_date", as_of_date)
        object.__setattr__(self, "factor_panels", panels)


# ---------------------------------------------------------------------------
# StrategyResult -- what a strategy returns
# ---------------------------------------------------------------------------
def _validated_result_rows(
    frame: Any,
    universe: tuple[str, ...],
    as_of_date: date | None,
    strategy_spec: StrategySpec,
) -> tuple[pd.DataFrame, date]:
    """Validate a raw strategy result; return ``(rows, date)``.

    ``rows`` is the contract's own copy in canonical column order, with
    ``as_of_date`` as ``datetime64[s]``, ``score`` as ``float64``, ``rank`` as a
    nullable ``Int64`` and ``is_selected`` as a plain ``bool`` column.

    The frame's ``strategy_name`` / ``strategy_version`` columns are validated as
    columns first -- exactly one value each, and well formed -- and then
    cross-checked against ``strategy_spec``, which is the authoritative identity:
    a frame naming a different strategy or version than the spec it is handed is
    refused instead of being stored under a label it does not match.
    """
    if not isinstance(frame, pd.DataFrame):
        raise StrategyContractError(
            f"StrategyResult frame must be a pandas DataFrame, got {type(frame).__name__}"
        )
    if len(frame) == 0:
        raise StrategyContractError(
            "StrategyResult frame must not be empty: the universe of "
            f"{len(universe)} symbol(s) needs one row per symbol"
        )

    forbidden = [str(column) for column in frame.columns if column in FORBIDDEN_RESULT_COLUMNS]
    missing_columns = [column for column in STRATEGY_RESULT_COLUMNS if column not in frame.columns]
    unexpected = [
        str(column)
        for column in frame.columns
        if column not in STRATEGY_RESULT_COLUMNS and column not in FORBIDDEN_RESULT_COLUMNS
    ]
    if forbidden or missing_columns or unexpected:
        problems: list[str] = []
        if forbidden:
            problems.append(
                f"portfolio/trading column(s) {forbidden} belong to a later layer: a strategy "
                "result carries a score, a rank and a selection flag, never a weight, a "
                "quantity, a side or a fill"
            )
        if missing_columns:
            problems.append(f"missing column(s) {missing_columns}")
        if unexpected:
            problems.append(f"unexpected column(s) {unexpected}")
        raise StrategyContractError(
            "StrategyResult frame must have exactly the result columns "
            f"{list(STRATEGY_RESULT_COLUMNS)}",
            problems,
        )

    rows = frame.loc[:, list(STRATEGY_RESULT_COLUMNS)].copy(deep=True).reset_index(drop=True)
    problems = []

    # -- as_of_date: exactly one session, never missing ---------------------
    date_column = _normalized_date_column(rows[_DATE_COLUMN], problems)
    dates, date_problems = _row_dates(date_column)
    problems.extend(date_problems)
    distinct_dates = _unique(value for value in dates if value is not None)
    if len(distinct_dates) > 1:
        problems.append(
            "a strategy result carries exactly one as_of_date, but the frame has "
            f"{len(distinct_dates)}: {_describe(value.isoformat() for value in distinct_dates)}"
        )
    elif len(distinct_dates) == 1 and as_of_date is not None and distinct_dates[0] != as_of_date:
        problems.append(
            f"the frame is dated {distinct_dates[0].isoformat()} but was built for "
            f"as_of_date {as_of_date.isoformat()}"
        )

    # -- identity: one strategy name, one version, matching the spec --------
    names = _unique(rows[_NAME_COLUMN].tolist())
    if len(names) != 1:
        problems.append(
            f"the frame must carry exactly one strategy_name, got {len(names)}: {_describe(names)}"
        )
    elif not isinstance(names[0], str) or not _STRATEGY_NAME_RE.match(names[0]):
        problems.append(
            f"strategy_name {names[0]!r} must be snake_case: lower-case letters, digits and "
            f"'_' only (matches {STRATEGY_NAME_PATTERN})"
        )
    elif names[0] != strategy_spec.name:
        problems.append(
            "the frame identity disagrees with strategy_spec: the frame names strategy "
            f"{names[0]!r} but strategy_spec.name is {strategy_spec.name!r} "
            f"({strategy_spec.name}/{strategy_spec.version})"
        )

    versions = _unique(rows[_VERSION_COLUMN].tolist())
    if len(versions) != 1:
        problems.append(
            "the frame must carry exactly one strategy_version, got "
            f"{len(versions)}: {_describe(versions)}"
        )
    elif not isinstance(versions[0], str) or not versions[0].strip():
        problems.append(f"strategy_version must be a non-empty string, got {versions[0]!r}")
    elif versions[0] != strategy_spec.version:
        problems.append(
            "the frame identity disagrees with strategy_spec: the frame carries strategy "
            f"version {versions[0]!r} but strategy_spec.version is {strategy_spec.version!r} "
            f"({strategy_spec.name}/{strategy_spec.version})"
        )

    # -- symbol: the universe, exactly, in its order -----------------------
    symbols = rows[_SYMBOL_COLUMN].tolist()
    named = [value for value in symbols if isinstance(value, str) and value.strip()]
    malformed = [value for value in symbols if not (isinstance(value, str) and value.strip())]
    if malformed:
        problems.append(f"symbol values must be non-empty strings, got {_describe(malformed)}")
    repeated_symbols = _duplicates(named)
    if repeated_symbols:
        problems.append(
            f"symbol appears more than once: {_describe(repeated_symbols)}: a result holds "
            "one row per symbol of the universe"
        )
    known = set(universe)
    missing_symbols = [symbol for symbol in universe if symbol not in set(named)]
    if missing_symbols:
        problems.append(
            "the frame must cover every symbol of the universe; missing "
            f"{_describe(missing_symbols)}"
        )
    unknown_symbols = _unique(symbol for symbol in named if symbol not in known)
    if unknown_symbols:
        problems.append(
            f"the frame contains symbol(s) outside the universe: {_describe(unknown_symbols)}"
        )
    ordered = not malformed and not repeated_symbols and len(named) == len(universe)
    if ordered and set(named) == known and tuple(named) != universe:
        position = next(
            index
            for index, (actual, expected) in enumerate(zip(named, universe, strict=True))
            if actual != expected
        )
        problems.append(
            f"rows must follow the universe's order: row {position} is {named[position]!r} "
            f"but the universe has {universe[position]!r} there"
        )

    # -- score: numeric or missing, finite when present --------------------
    scores = rows[_SCORE_COLUMN]
    numeric_scores = pd.Series(float("nan"), index=rows.index, dtype="float64")
    if pd.api.types.is_bool_dtype(scores.dtype):
        problems.append(
            "score must be a number or missing, got a boolean column: is_selected says "
            "which symbols were picked, score says how strongly"
        )
    elif not pd.api.types.is_numeric_dtype(scores.dtype):
        problems.append(
            f"score must be numeric or missing, got dtype {scores.dtype}: a score is "
            "computed, never formatted text"
        )
    else:
        numeric_scores = scores.astype("float64")
        infinite = numeric_scores.abs() == float("inf")
        if bool(infinite.any()):
            problems.append(
                "score must be finite when present; ±inf in row(s) "
                f"{_describe(rows.loc[infinite, _SYMBOL_COLUMN].tolist())}: no ordering "
                "of an unbounded score is a decision"
            )

    # -- rank: a positive whole number when present, unique -----------------
    ranks = rows[_RANK_COLUMN]
    numeric_ranks = pd.Series(float("nan"), index=rows.index, dtype="float64")
    integral = pd.Series(False, index=rows.index, dtype="bool")
    if pd.api.types.is_bool_dtype(ranks.dtype):
        problems.append("rank must be a positive whole number or missing, got a boolean column")
    elif not pd.api.types.is_numeric_dtype(ranks.dtype):
        problems.append(
            f"rank must be integer-valued or missing, got dtype {ranks.dtype}: a rank is "
            "computed, never formatted text"
        )
    else:
        numeric_ranks = ranks.astype("float64")
        finite = numeric_ranks.notna() & (numeric_ranks.abs() != float("inf"))
        integral = finite & (numeric_ranks == numeric_ranks.round())
        malformed = numeric_ranks.notna() & ~integral
        if bool(malformed.any()):
            problems.append(
                "rank must be a positive whole number or missing; "
                f"{_describe(ranks[malformed].tolist())} are neither"
            )
        below_one = integral & (numeric_ranks < 1)
        if bool(below_one.any()):
            problems.append(f"rank starts at 1; {_describe(ranks[below_one].tolist())} are lower")
        repeated_ranks = _duplicates(numeric_ranks[integral & (numeric_ranks >= 1)].tolist())
        if repeated_ranks:
            problems.append(
                "rank must be unique within the cross-section; repeated "
                f"{_describe(repeated_ranks)}"
            )

    # -- is_selected: a real boolean, never missing ------------------------
    selected_column = rows[_SELECTED_COLUMN]
    selected = pd.Series(False, index=rows.index, dtype="bool")
    if not pd.api.types.is_bool_dtype(selected_column.dtype):
        problems.append(
            f"is_selected must be a boolean column (True/False), got dtype {selected_column.dtype}"
        )
    elif bool(selected_column.isna().any()):
        problems.append(
            f"is_selected must not be missing ({int(selected_column.isna().sum())} row(s) are NA)"
        )
        selected = selected_column.fillna(False).astype(bool)
    else:
        selected = selected_column.astype(bool)

    # -- a selection needs a finite score and a rank -----------------------
    scored = numeric_scores.notna() & (numeric_scores.abs() != float("inf"))
    unranked = selected & numeric_ranks.isna()
    if bool((selected & ~scored).any()):
        problems.append(
            "selected symbol(s) "
            f"{_describe(rows.loc[selected & ~scored, _SYMBOL_COLUMN].tolist())} have no finite "
            "score: a selection must be backed by a number"
        )
    if bool(unranked.any()):
        problems.append(
            "selected symbol(s) "
            f"{_describe(rows.loc[unranked, _SYMBOL_COLUMN].tolist())} have no rank: a selection "
            "needs its position in the cross-section"
        )

    if problems:
        raise StrategyContractError("invalid StrategyResult frame", problems)

    # Nothing raised, so every row and both identity columns were valid.
    session = next(value for value in dates if value is not None)
    rows[_DATE_COLUMN] = pd.Series([pd.Timestamp(session)] * len(rows), dtype="datetime64[s]")
    rows[_SCORE_COLUMN] = numeric_scores
    rows[_RANK_COLUMN] = numeric_ranks.astype("Int64")
    rows[_SELECTED_COLUMN] = selected.astype(bool)
    return rows, session


@dataclass(frozen=True, eq=False)
class StrategyResult:
    """The one cross-section a strategy produced for one decision date.

    One row per symbol of the universe, in the universe's order, with exactly the
    :data:`STRATEGY_RESULT_COLUMNS` columns:

    ===================  =====================================================
    ``as_of_date``       the decision session
    ``strategy_name``    the scoring strategy's snake_case name
    ``strategy_version`` that strategy's version
    ``symbol``           ticker, taken from the universe
    ``score``            numeric or missing; finite when present
    ``rank``             positive whole number or missing; unique
    ``is_selected``      boolean, never missing
    ===================  =====================================================

    Anything the portfolio or execution layer owns -- a weight, a quantity, an
    order side, a fill price -- is refused by name
    (:data:`FORBIDDEN_RESULT_COLUMNS`), so this object can never quietly become a
    portfolio.

    Besides the frame, the result exposes its provenance:

    * :attr:`strategy_spec` -- the full immutable :class:`StrategySpec` it was
      produced under.  This is the **authoritative** identity,
    * :attr:`strategy_name` / :attr:`strategy_version` -- read-only views of
      ``strategy_spec.name`` / ``strategy_spec.version``, never a second identity
      that could disagree with the declared factors,
    * :attr:`factor_requirements` -- ``strategy_spec.required_factors``, in the
      declared order, each entry an exact ``(factor_name, factor_version)`` pair,
    * :attr:`universe`, :attr:`as_of_date`, :attr:`symbols`,
      :attr:`selected_symbols`.

    Parameters
    ----------
    frame:
        Long-format result with exactly the result columns (any column order is
        accepted and normalized).  It is validated before the object exists, so
        an invalid result never travels on towards a portfolio constructor.
    universe:
        The :class:`~factor_sys.universe.UniverseSnapshot` the strategy scored.
        Every symbol must appear exactly once, in the snapshot's order.
    strategy_spec:
        The producing strategy's :class:`StrategySpec`: its name, its version and,
        above all, the exact factors it read.  It is stored as given (never
        copied, so a result and its strategy cannot disagree), the frame's
        ``strategy_name`` / ``strategy_version`` columns must match it, and the
        object's own identity properties are read from it.
    as_of_date:
        Optional cross-check against the frame's own ``as_of_date`` values, as a
        :class:`datetime.date` or a bare ISO-8601 ``"YYYY-MM-DD"`` string.  A
        ``datetime`` is refused even at midnight.

    Raises
    ------
    strategy_sys.errors.StrategyContractError
        If the frame is not a frame, is empty, has missing, unexpected or
        forbidden columns, carries more than one date or identity, does not cover
        the universe exactly once in order, holds a non-numeric or infinite
        ``score``, a non-positive, non-integral or repeated ``rank``, a
        non-boolean or missing ``is_selected``, selects a symbol without a finite
        score or a rank, disagrees with ``strategy_spec`` on the strategy name or
        version, or contradicts the ``as_of_date`` argument.  A ``strategy_spec``
        that is not a :class:`StrategySpec` is refused on its own.  Every problem
        is reported at once.

    Notes
    -----
    * An **empty selection** is legal: the contract types the flag, it does not
      decide how many symbols a portfolio needs.
    * A **missing score** is legal too -- it means "this symbol was not scored",
      which a strategy may well decide for an ineligible input.  A *selected*
      symbol is the one case that must have both a finite score and a rank.
    * ``frame["as_of_date"]`` is a ``datetime64[s]`` column -- how pandas stores a
      date -- and is normalized to the session: an intraday row or a
      timezone-aware column is refused rather than truncated, exactly like a
      scalar ``datetime`` passed to :class:`StrategyContext`.
    * ``frame`` returns a defensive copy; mutating it cannot change the result.
    * Results are compared by identity (``eq=False``), like
      :class:`~factor_sys.contracts.FactorPanel`: comparing frames elementwise is
      ambiguous -- use :meth:`frame` for value comparisons.
    * The spec is what makes two otherwise identical results distinguishable:
      ``top_10_by_momentum_12_1`` on factor ``1.0.0`` and on factor ``2.0.0``
      share ``strategy_name`` and ``strategy_version`` but carry different
      :attr:`strategy_spec` / :attr:`factor_requirements` values, which is the
      key a later portfolio, backtest ledger or cache needs.
    * A result **cannot** verify that its frame was actually computed from the
      factors it names -- it holds no context and no panel.  Comparing
      ``result.strategy_spec`` with the invoked strategy's ``spec``, and checking
      that the context carried every requirement, is the future strategy runner's
      job; this layer never pretends to have done it.

    Examples
    --------
    >>> universe = UniverseSnapshot(
    ...     as_of_date="2024-01-05",
    ...     symbols=("AAPL", "MSFT"),
    ...     source="config/universe_seed.csv",
    ... )
    >>> frame = pd.DataFrame(
    ...     {
    ...         "as_of_date": [pd.Timestamp("2024-01-05")] * 2,
    ...         "strategy_name": ["top_n_momentum"] * 2,
    ...         "strategy_version": ["1.0.0"] * 2,
    ...         "symbol": ["AAPL", "MSFT"],
    ...         "score": [0.35, float("nan")],
    ...         "rank": [1, None],
    ...         "is_selected": [True, False],
    ...     }
    ... )
    >>> spec = StrategySpec(
    ...     name="top_n_momentum",
    ...     version="1.0.0",
    ...     required_factors=(FactorRequirement("momentum_12_1", "1.0.0"),),
    ... )
    >>> result = StrategyResult(
    ...     frame, universe=universe, strategy_spec=spec, as_of_date="2024-01-05"
    ... )
    >>> result.selected_symbols
    ('AAPL',)
    >>> result.strategy_spec is spec
    True
    >>> [str(requirement) for requirement in result.factor_requirements]
    ['momentum_12_1/1.0.0']
    >>> result.as_of_date
    datetime.date(2024, 1, 5)
    >>> tuple(result.frame.columns) == STRATEGY_RESULT_COLUMNS
    True
    """

    as_of_date: date
    universe: UniverseSnapshot
    strategy_spec: StrategySpec
    _rows: pd.DataFrame = field(repr=False)

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        universe: UniverseSnapshot,
        strategy_spec: object,
        as_of_date: object = None,
    ) -> None:
        """Validate ``frame`` as one strategy result for ``strategy_spec`` and freeze it."""
        if not isinstance(universe, UniverseSnapshot):
            raise StrategyContractError(
                "StrategyResult.universe must be a UniverseSnapshot (the factor layer's "
                f"validated pool), got {type(universe).__name__}"
            )
        if not isinstance(strategy_spec, StrategySpec):
            raise StrategyContractError(
                "StrategyResult.strategy_spec must be a StrategySpec (the strategy's immutable "
                f"declaration), got {type(strategy_spec).__name__}: a result carries the exact "
                "strategy and the exact factor versions it came from, not a bare name or a "
                "dictionary of labels"
            )
        expected_date = (
            None if as_of_date is None else _session_date(as_of_date, "StrategyResult.as_of_date")
        )

        rows, resolved = _validated_result_rows(
            frame,
            universe.symbols,
            expected_date,
            strategy_spec,
        )

        object.__setattr__(self, "as_of_date", resolved)
        object.__setattr__(self, "universe", universe)
        object.__setattr__(self, "strategy_spec", strategy_spec)
        object.__setattr__(self, "_rows", rows)

    @property
    def strategy_name(self) -> str:
        """``strategy_spec.name`` -- never an identity of its own."""
        return self.strategy_spec.name

    @property
    def strategy_version(self) -> str:
        """``strategy_spec.version`` -- the version of the rule that scored this."""
        return self.strategy_spec.version

    @property
    def factor_requirements(self) -> tuple[FactorRequirement, ...]:
        """``strategy_spec.required_factors``: the exact factors *and* versions read.

        The spec's own tuple, in its declared order, so the dependencies cannot
        drift away from the identity the frame carries.
        """
        return self.strategy_spec.required_factors

    @property
    def frame(self) -> pd.DataFrame:
        """A copy of the long-format result in canonical column order.

        Mutating the returned frame cannot change the result.
        """
        return self._rows.copy(deep=True)

    @property
    def symbols(self) -> tuple[str, ...]:
        """The result's symbols in row order: the universe's own order."""
        return tuple(self._rows[_SYMBOL_COLUMN].tolist())

    @property
    def selected_symbols(self) -> tuple[str, ...]:
        """The symbols flagged ``is_selected``, in row order."""
        flags = self._rows[_SELECTED_COLUMN].astype(bool)
        return tuple(self._rows.loc[flags, _SYMBOL_COLUMN].tolist())

    def __len__(self) -> int:
        """Number of rows: one per symbol of the universe."""
        return int(len(self._rows))


# ---------------------------------------------------------------------------
# Strategy -- the scoring interface
# ---------------------------------------------------------------------------
class Strategy(ABC):
    """Abstract base class every strategy implements.

    A strategy is pure computation over a :class:`StrategyContext`: it declares
    what it is and what it needs through :attr:`spec`, reads only
    ``context.factor_panels`` (through the requirements it declared) and the
    labels on the context, and hands back a validated :class:`StrategyResult`.
    It must not read files, know a path, reach into the factor store, call a
    trading calendar, hit the network or produce weights, orders or fills -- a
    strategy that can be built from an in-memory context is a strategy that can
    be tested without any data on disk.

    Subclasses either declare ``spec`` as a class attribute (the usual, simplest
    form) or implement it as a property; both satisfy the abstract requirement.
    Swapping one strategy for another -- a single-factor ``top_n`` rule today, a
    multi-factor blend later -- never needs a change at the call site, because
    the runner only ever sees this interface::

        class TopNByMomentum(Strategy):
            spec = StrategySpec(
                name="top_n_momentum",
                version="1.0.0",
                required_factors=(FactorRequirement("momentum_12_1", "1.0.0"),),
            )

            def compute(self, context: StrategyContext) -> StrategyResult:
                panel = context.factor_panels[self.spec.required_factors[0]]
                ...
    """

    @property
    @abstractmethod
    def spec(self) -> StrategySpec:
        """The strategy's immutable declaration (:class:`StrategySpec`)."""

    @abstractmethod
    def compute(self, context: StrategyContext) -> StrategyResult:
        """Return this strategy's cross-section for ``context``.

        Parameters
        ----------
        context:
            The point-in-time input the strategy may read: one decision date, one
            universe, one panel per factor it was given.  A ``compute`` call must
            not mutate it and must not depend on anything outside it, so the same
            context always yields the same result.

        Returns
        -------
        StrategyResult
            Already validated: one row per ``context.universe`` symbol, in that
            order, with the context's ``as_of_date`` and this strategy's own
            :attr:`spec` attached -- so the result names the exact strategy and
            the exact factor versions that produced it.

        Notes
        -----
        Whether ``context`` actually carries the factors named in
        :attr:`spec` is the strategy's own business (or its runner's): it looks
        each requirement up in ``context.factor_panels`` and reports a missing one
        in its own words.  The context itself deliberately does not know.
        """

    def __repr__(self) -> str:
        return f"{type(self).__name__}(spec={self.spec!r})"


__all__ = [
    "FORBIDDEN_RESULT_COLUMNS",
    "STRATEGY_NAME_PATTERN",
    "STRATEGY_RESULT_COLUMNS",
    "FactorRequirement",
    "Strategy",
    "StrategyContext",
    "StrategyResult",
    "StrategySpec",
]
