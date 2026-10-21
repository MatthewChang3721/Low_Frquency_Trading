"""The stable data contract shared by every factor.

What the contract is
--------------------
``factor_sys.contracts`` defines the only objects a factor formula and the
factor runner exchange:

``FactorSpec``
    what a factor needs: its name, version, required bar fields, lookback.
``FactorContext``
    what a factor may look at: the history visible on ``as_of_date``.
``FactorPanel``
    what a factor returns: one cross-section, one row per symbol, long format.
``Factor``
    the abstract formula interface itself.

Why the contract lives here and not in ``factors/``
---------------------------------------------------
``factors/`` holds *formulas and parameters only*: a factor must not read
Parquet, must not know a file path and must not reach into the storage layer.
Turning stored bars into a :class:`FactorContext` and writing a
:class:`FactorPanel` back out is the runner's job (``factor_sys.runner`` and
``factor_sys.storage``, added later).  Keeping the shapes here means every
factor is validated the same way, in memory, before anything touches disk.

Point-in-time safety
--------------------
A factor may only use information that was already public when ``as_of_date``
closed.  The part of that rule which is decidable from the data itself is
enforced here: :class:`FactorContext` refuses any bar dated *after*
``as_of_date``, so a look-ahead leak fails loudly at construction instead of
quietly inflating a backtest.  How much history a factor needs, and how it
treats a short history, stays the factor's own business.

Conventions reused from the data system
---------------------------------------
* ``SYMBOL_PATTERN`` comes from :mod:`data_sys.schema`: one definition of a valid
  ticker for the whole project.
* Dates are timezone-naive, day-normalized ``datetime64[s]`` values, exactly
  like the standardized market bars.
* Validation collects everything it finds in one frame (the style of
  :class:`data_sys.errors.UniverseError`) instead of failing on the first row.
"""

from __future__ import annotations

import re
import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date, datetime
from numbers import Integral
from typing import Any

import pandas as pd

from data_sys.schema import SYMBOL_PATTERN
from factor_sys.errors import FactorContractError

# ---------------------------------------------------------------------------
# Column and pattern contracts
# ---------------------------------------------------------------------------
#: Fixed column order of one factor panel (one factor, one ``as_of_date``).
FACTOR_PANEL_COLUMNS: tuple[str, ...] = (
    "as_of_date",
    "symbol",
    "factor_name",
    "factor_version",
    "raw_value",
    "is_eligible",
)

#: Columns a factor's ``bars`` frame must provide (see :mod:`data_sys.schema`).
BAR_REQUIRED_COLUMNS: tuple[str, ...] = ("date", "symbol")

#: A factor name is a snake_case identifier, e.g. ``momentum_12_1``.
FACTOR_NAME_PATTERN: str = r"^[a-z0-9_]+$"

_SYMBOL_RE = re.compile(SYMBOL_PATTERN)
_FACTOR_NAME_RE = re.compile(FACTOR_NAME_PATTERN)

_PANEL_DATE_COLUMN = "as_of_date"
_PANEL_SYMBOL_COLUMN = "symbol"
_PANEL_NAME_COLUMN = "factor_name"
_PANEL_VERSION_COLUMN = "factor_version"
_PANEL_VALUE_COLUMN = "raw_value"
_PANEL_ELIGIBLE_COLUMN = "is_eligible"


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
    """The distinct values in first-seen order (mixed types are allowed)."""
    distinct: list[Any] = []
    for value in values:
        if value not in distinct:
            distinct.append(value)
    return distinct


def _duplicates(values: Iterable[str]) -> list[str]:
    """The values that occur more than once, in first-seen order."""
    seen: set[str] = set()
    repeated: list[str] = []
    for value in values:
        if value in seen and value not in repeated:
            repeated.append(value)
        seen.add(value)
    return repeated


def _normalize_date(value: Any, label: str) -> date:
    """Validate one scalar date and return it as a plain :class:`datetime.date`.

    Accepts a ``date``, a ``datetime`` (its time component is meaningless for a
    daily system) or an ISO-8601 ``"YYYY-MM-DD"`` string.  The result is always
    timezone-naive and day-normalized.
    """
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            value = value.replace(tzinfo=None)
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        text = value.strip()
        if text:
            try:
                return datetime.fromisoformat(text).date()
            except ValueError:
                pass
    raise FactorContractError(
        f"{label} must be a date or an ISO-8601 'YYYY-MM-DD' string, got {value!r}"
    )


def _canonical_universe(values: Iterable[str], label: str) -> tuple[str, ...]:
    """Validate and freeze a universe: non-empty, canonical, de-duplicated."""
    if values is None or isinstance(values, str):
        raise FactorContractError(
            f"{label} must be an iterable of ticker symbols (not a bare string), got {values!r}"
        )
    try:
        candidates = list(values)
    except TypeError as exc:
        raise FactorContractError(
            f"{label} must be an iterable of ticker symbols, got {type(values).__name__}"
        ) from exc

    if not candidates:
        raise FactorContractError(f"{label} must not be empty")

    symbols: list[str] = []
    problems: list[str] = []
    for value in candidates:
        if not isinstance(value, str) or not _SYMBOL_RE.match(value):
            problems.append(
                f"invalid {label} symbol {value!r}: expected {SYMBOL_PATTERN} "
                "(upper-case letters, digits, '.' and '-')"
            )
        elif value in symbols:
            problems.append(f"duplicate {label} symbol {value!r}")
        else:
            symbols.append(value)

    if problems:
        raise FactorContractError(f"invalid {label}", problems)
    return tuple(symbols)


def _parse_dates(values: pd.Series, label: str) -> pd.Series:
    """``values`` as timezone-naive, day-normalized ``datetime64[s]`` dates.

    Day-normalizing matters for the point-in-time guard: without it a bar
    stamped ``2024-01-05 16:00`` would look like future data on
    ``as_of_date`` 2024-01-05, which it is not.
    """
    try:
        with warnings.catch_warnings():
            # pandas emits "Could not infer format, so each element will be
            # parsed individually" for the very columns the contract is about to
            # reject; the violation is reported as a FactorContractError below,
            # so the pandas hint would only add noise.
            warnings.simplefilter("ignore", UserWarning)
            parsed = pd.to_datetime(values, errors="coerce")
    except (TypeError, ValueError) as exc:
        raise FactorContractError(f"{label} cannot be parsed as dates: {exc}") from exc

    missing = parsed.isna()
    if bool(missing.any()):
        offenders = _describe(
            [
                value
                for value, is_missing in zip(values.tolist(), missing.tolist(), strict=True)
                if is_missing
            ]
        )
        raise FactorContractError(
            f"{label} contains {int(missing.sum())} value(s) that are missing or "
            f"unparseable, e.g. {offenders}"
        )

    if parsed.dt.tz is not None:
        parsed = parsed.dt.tz_localize(None)
    return parsed.dt.normalize().astype("datetime64[s]")


# ---------------------------------------------------------------------------
# FactorSpec -- what a factor needs
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FactorSpec:
    """Immutable declaration of one factor: identity plus input requirements.

    The spec is defined once next to the formula and never changes while the
    factor runs; it is what makes results reproducible and comparable across
    runs.  All validation problems are collected, so a factor author sees every
    mistake in one go.

    Parameters
    ----------
    name:
        snake_case identifier, e.g. ``momentum_12_1``.  It is written into the
        panel and later used to name stored factor files.
    version:
        Non-empty string, e.g. ``"1.0.0"``.  A formula change needs a new
        version: the same name plus a new version is a different factor.
    required_fields:
        Non-empty, duplicate-free sequence of bar field names the formula reads
        (``"close"``, ``"volume"``, ``"adjusted_close"``, ...).  It is frozen
        into a tuple so a caller's list cannot be mutated behind the spec.
        ``"date"`` and ``"symbol"`` may not be declared: they are the label
        columns every read already returns (:data:`BAR_REQUIRED_COLUMNS`), not
        fields a factor requests.
    lookback_sessions:
        Sessions of history the formula needs before its output is meaningful
        (0 when it does not need warm-up).  Purely declarative: the contract
        does not truncate history for a factor that asks for less, and a short
        history is not an error here.

    Examples
    --------
    >>> spec = FactorSpec(
    ...     name="momentum_12_1",
    ...     version="1.0.0",
    ...     required_fields=("adjusted_close",),
    ...     lookback_sessions=252,
    ... )
    >>> spec.required_fields
    ('adjusted_close',)
    """

    name: str
    version: str
    required_fields: tuple[str, ...]
    lookback_sessions: int = 0

    def __post_init__(self) -> None:
        problems: list[str] = []

        name = self.name
        if not isinstance(name, str) or not name.strip():
            problems.append(f"name must be a non-empty string, got {name!r}")
        elif not _FACTOR_NAME_RE.match(name):
            problems.append(
                f"name {name!r} must be snake_case: lower-case letters, digits and "
                f"'_' only (matches {FACTOR_NAME_PATTERN})"
            )

        version = self.version
        if not isinstance(version, str) or not version.strip():
            problems.append(f"version must be a non-empty string, got {version!r}")

        declared = self.required_fields
        fields_: list[str] = []
        if isinstance(declared, str) or not isinstance(declared, Iterable):
            problems.append(
                f"required_fields must be a non-empty sequence of field names, got {declared!r}"
            )
        else:
            candidates = list(declared)
            problems.extend(
                f"required_fields entry {value!r} must be a non-empty string"
                for value in candidates
                if not (isinstance(value, str) and value.strip())
            )
            fields_ = [value for value in candidates if isinstance(value, str) and value.strip()]
            problems.extend(
                f"duplicate required_fields entry {value!r}" for value in _duplicates(fields_)
            )
            # ``date`` and ``symbol`` are the label columns of every bars frame,
            # not fields a factor asks for: the market-data reader refuses to be
            # asked for them, so a spec declaring them could never run.
            problems.extend(
                f"required_fields entry {value!r} is a label column, not a bar field: "
                f"{list(BAR_REQUIRED_COLUMNS)} are always provided"
                for value in fields_
                if value in BAR_REQUIRED_COLUMNS
            )
            if not fields_:
                problems.append("required_fields must not be empty")

        lookback = self.lookback_sessions
        if isinstance(lookback, bool) or not isinstance(lookback, Integral):
            problems.append(f"lookback_sessions must be a non-negative integer, got {lookback!r}")
            lookback = 0
        elif lookback < 0:
            problems.append(f"lookback_sessions must not be negative, got {lookback!r}")
            lookback = 0

        if problems:
            raise FactorContractError("invalid FactorSpec", problems)

        object.__setattr__(self, "required_fields", tuple(fields_))
        object.__setattr__(self, "lookback_sessions", int(lookback))


# ---------------------------------------------------------------------------
# FactorContext -- what a factor may look at
# ---------------------------------------------------------------------------
def _validated_bars(
    bars: Any,
    as_of_date: date,
    universe: tuple[str, ...],
) -> pd.DataFrame:
    """Validate ``bars`` and return the contract's own defensive copy.

    The copy is deliberately normalized: a clean ``RangeIndex`` (so no factor
    comparison can trip over the caller's index labels) and a day-normalized
    ``datetime64[s]`` ``date`` column (so the point-in-time guard is a plain
    comparison).  The caller's frame is never modified.
    """
    if not isinstance(bars, pd.DataFrame):
        raise FactorContractError(
            f"FactorContext.bars must be a pandas DataFrame, got {type(bars).__name__}"
        )

    missing = [column for column in BAR_REQUIRED_COLUMNS if column not in bars.columns]
    if missing:
        raise FactorContractError(
            f"FactorContext.bars is missing required column(s) {missing}; got {list(bars.columns)}"
        )

    frame = bars.copy(deep=True).reset_index(drop=True)
    frame["date"] = _parse_dates(frame["date"], "FactorContext.bars.date")

    # Point-in-time guard: nothing after as_of_date may be visible.
    horizon = pd.Timestamp(as_of_date)
    is_future = frame["date"] > horizon
    if bool(is_future.any()):
        count = int(is_future.sum())
        latest = pd.Timestamp(frame.loc[is_future, "date"].max()).date().isoformat()
        raise FactorContractError(
            f"FactorContext.bars contains {count} row(s) dated after as_of_date "
            f"{as_of_date.isoformat()} (latest {latest}): a factor must never read "
            "future data"
        )

    known = set(universe)
    unknown = [symbol for symbol in _unique(frame["symbol"].tolist()) if symbol not in known]
    if unknown:
        raise FactorContractError(
            f"FactorContext.bars contains symbol(s) outside the universe: {_describe(unknown)}"
        )

    duplicated = frame.duplicated(subset=["symbol", "date"])
    if bool(duplicated.any()):
        offenders = _describe(_unique(frame.loc[duplicated, "symbol"].tolist()))
        raise FactorContractError(
            f"FactorContext.bars has {int(duplicated.sum())} duplicate (symbol, date) "
            f"row(s): {offenders}"
        )

    return frame


@dataclass(frozen=True, eq=False)
class FactorContext:
    """One point-in-time slice of the world a factor is allowed to see.

    The context is built by the runner from stored market bars; a factor only
    reads it.  Three guarantees make a factor safe to reuse in a backtest:

    1. **No future data.**  Every row of ``bars`` is dated on or before
       ``as_of_date``; a later bar raises :class:`FactorContractError` instead
       of silently leaking into the result.
    2. **Correct labels.**  ``as_of_date`` is a timezone-naive
       :class:`datetime.date`, and ``universe`` is a canonical, de-duplicated
       tuple of tickers, so a factor can iterate it directly.
    3. **Isolation.**  ``bars`` is a defensive deep copy with a clean index and a
       day-normalized ``datetime64[s]`` ``date`` column; mutating the frame the
       runner passed in cannot change this context.

    It is *not* an error for ``bars`` to cover only part of the universe or a
    shorter history than a factor would like: a factor decides for itself how to
    treat a missing symbol or a short warm-up (``FactorSpec.lookback_sessions``
    declares the need).  An empty ``bars`` frame is likewise accepted.

    Parameters
    ----------
    as_of_date:
        The last session the factor may see, as a ``date``, ``datetime`` or
        ISO-8601 ``"YYYY-MM-DD"`` string.
    universe:
        Symbols the factor must produce output for, e.g.
        ``("AAPL", "MSFT", "NVDA")``.
    bars:
        Long-format daily bars; at least the ``date`` and ``symbol`` columns,
        plus whatever fields the spec declares.

    Examples
    --------
    >>> bars = pd.DataFrame(
    ...     {
    ...         "date": [pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-05")],
    ...         "symbol": ["AAPL", "MSFT"],
    ...         "close": [189.5, 372.1],
    ...     }
    ... )
    >>> context = FactorContext(
    ...     as_of_date="2024-01-05",
    ...     universe=["AAPL", "MSFT"],
    ...     bars=bars,
    ... )
    >>> context.as_of_date
    datetime.date(2024, 1, 5)
    """

    as_of_date: date
    universe: tuple[str, ...]
    bars: pd.DataFrame = field(repr=False)

    def __post_init__(self) -> None:
        as_of_date = _normalize_date(self.as_of_date, "FactorContext.as_of_date")
        universe = _canonical_universe(self.universe, "FactorContext.universe")
        bars = _validated_bars(self.bars, as_of_date, universe)

        object.__setattr__(self, "as_of_date", as_of_date)
        object.__setattr__(self, "universe", universe)
        object.__setattr__(self, "bars", bars)


# ---------------------------------------------------------------------------
# FactorPanel -- what a factor returns
# ---------------------------------------------------------------------------
def _validated_panel_rows(
    frame: Any,
    universe: tuple[str, ...],
    as_of_date: date | None,
) -> tuple[pd.DataFrame, date, str, str]:
    """Validate a raw panel frame; return ``(rows, date, name, version)``.

    ``rows`` is the contract's own copy in canonical column order, with
    ``as_of_date`` as ``datetime64[s]``, ``raw_value`` as ``float64`` and
    ``is_eligible`` as a plain ``bool`` column.
    """
    if not isinstance(frame, pd.DataFrame):
        raise FactorContractError(
            f"FactorPanel output must be a pandas DataFrame, got {type(frame).__name__}"
        )
    if len(frame) == 0:
        raise FactorContractError(
            "FactorPanel output must not be empty: the universe of "
            f"{len(universe)} symbol(s) needs one row per symbol"
        )

    missing = [column for column in FACTOR_PANEL_COLUMNS if column not in frame.columns]
    unexpected = [column for column in frame.columns if column not in FACTOR_PANEL_COLUMNS]
    if missing or unexpected:
        problems = []
        if missing:
            problems.append(f"missing column(s) {missing}")
        if unexpected:
            problems.append(f"unexpected column(s) {list(unexpected)}")
        raise FactorContractError(
            f"FactorPanel output must have exactly the panel columns {list(FACTOR_PANEL_COLUMNS)}",
            problems,
        )

    # Canonical column order on a deep copy: neither the caller's frame nor the
    # panel can be changed through the other one.
    rows = frame.loc[:, list(FACTOR_PANEL_COLUMNS)].copy(deep=True).reset_index(drop=True)

    # -- as_of_date: exactly one, never missing, matching the caller's date ---
    dates = _parse_dates(rows[_PANEL_DATE_COLUMN], "FactorPanel.as_of_date")
    distinct_dates = _unique(pd.Timestamp(value) for value in dates.tolist())
    if len(distinct_dates) != 1:
        rendered = _describe(str(value.date()) for value in distinct_dates)
        raise FactorContractError(
            "FactorPanel output must carry exactly one as_of_date, got "
            f"{len(distinct_dates)}: {rendered}"
        )
    resolved = distinct_dates[0].date()
    if as_of_date is not None and resolved != as_of_date:
        raise FactorContractError(
            f"FactorPanel output is dated {resolved.isoformat()} but was built for "
            f"as_of_date {as_of_date.isoformat()}"
        )
    rows[_PANEL_DATE_COLUMN] = dates

    # -- symbol: canonical, inside the universe, and covering all of it ------
    symbols = rows[_PANEL_SYMBOL_COLUMN].tolist()
    malformed = _unique(
        value for value in symbols if not isinstance(value, str) or not _SYMBOL_RE.match(value)
    )
    if malformed:
        raise FactorContractError(
            f"FactorPanel output has invalid symbol(s) {_describe(malformed)}: expected "
            f"{SYMBOL_PATTERN} (upper-case letters, digits, '.' and '-')"
        )
    known = set(universe)
    unknown = [symbol for symbol in _unique(symbols) if symbol not in known]
    if unknown:
        raise FactorContractError(
            f"FactorPanel output contains symbol(s) outside the universe: {_describe(unknown)}"
        )
    present = set(symbols)
    absent = [symbol for symbol in universe if symbol not in present]
    if absent:
        raise FactorContractError(
            "FactorPanel output must cover every symbol of the universe; missing: "
            f"{_describe(absent)}"
        )

    # -- identity: one factor name, one version ------------------------------
    names = _unique(rows[_PANEL_NAME_COLUMN].tolist())
    if len(names) != 1:
        raise FactorContractError(
            "FactorPanel output must carry exactly one factor_name, got "
            f"{len(names)}: {_describe(names)}"
        )
    if not isinstance(names[0], str) or not _FACTOR_NAME_RE.match(names[0]):
        raise FactorContractError(
            f"FactorPanel factor_name {names[0]!r} must be snake_case: lower-case "
            f"letters, digits and '_' only (matches {FACTOR_NAME_PATTERN})"
        )
    versions = _unique(rows[_PANEL_VERSION_COLUMN].tolist())
    if len(versions) != 1:
        raise FactorContractError(
            "FactorPanel output must carry exactly one factor_version, got "
            f"{len(versions)}: {_describe(versions)}"
        )
    if not isinstance(versions[0], str) or not versions[0].strip():
        raise FactorContractError(
            f"FactorPanel factor_version must be a non-empty string, got {versions[0]!r}"
        )

    # -- one row per (as_of_date, symbol, factor_name) -----------------------
    key_columns = [_PANEL_DATE_COLUMN, _PANEL_SYMBOL_COLUMN, _PANEL_NAME_COLUMN]
    duplicated = rows.duplicated(subset=key_columns)
    if bool(duplicated.any()):
        offenders = _describe(_unique(rows.loc[duplicated, _PANEL_SYMBOL_COLUMN].tolist()))
        raise FactorContractError(
            "FactorPanel output has duplicate (as_of_date, symbol, factor_name) "
            f"row(s): {offenders}"
        )

    # -- raw_value: numeric, missing allowed (e.g. an ineligible symbol) -----
    raw_values = rows[_PANEL_VALUE_COLUMN]
    filled = raw_values.notna()
    try:
        numeric = pd.to_numeric(raw_values, errors="coerce")
    except (TypeError, ValueError) as exc:
        raise FactorContractError(
            f"FactorPanel raw_value must be numeric, got dtype {raw_values.dtype}: {exc}"
        ) from exc
    unconvertible = filled & numeric.isna()
    if bool(unconvertible.any()):
        offenders = _describe(raw_values[unconvertible].tolist())
        raise FactorContractError(
            "FactorPanel raw_value must be convertible to float; "
            f"{int(unconvertible.sum())} row(s) are not, e.g. {offenders}"
        )
    rows[_PANEL_VALUE_COLUMN] = numeric.astype("float64")

    # -- is_eligible: a real boolean, never missing --------------------------
    eligibility = rows[_PANEL_ELIGIBLE_COLUMN]
    if not pd.api.types.is_bool_dtype(eligibility.dtype):
        raise FactorContractError(
            "FactorPanel is_eligible must be a boolean column (True/False), got "
            f"dtype {eligibility.dtype}"
        )
    if bool(eligibility.isna().any()):
        raise FactorContractError(
            "FactorPanel is_eligible must not be missing "
            f"({int(eligibility.isna().sum())} row(s) are NA)"
        )
    rows[_PANEL_ELIGIBLE_COLUMN] = eligibility.astype(bool)

    return rows, resolved, names[0], versions[0]


@dataclass(frozen=True, init=False, eq=False)
class FactorPanel:
    """The validated long-format cross-section one factor produced.

    One panel holds exactly one ``as_of_date``, one ``factor_name`` and one
    ``factor_version``, and exactly one row per symbol of the universe.  Long
    format (one row per symbol, not one column per symbol) keeps a future
    database write, an IC computation and a "factor vs. forward return" join
    trivial.

    A factor builds its own panel -- typically ``FactorPanel(frame,
    universe=context.universe, as_of_date=context.as_of_date)`` -- and the frame
    is validated before the object exists, so an invalid panel never travels on
    towards storage or analysis.  ``raw_value`` may be missing (for example for
    an ineligible symbol); what eligibility *means* stays the factor's decision,
    the contract only types the column.  Transforms such as ``zscore`` or
    ``rank`` belong to a later module and are not accepted here.

    Parameters
    ----------
    frame:
        Long-format output with exactly the :data:`FACTOR_PANEL_COLUMNS` columns
        (any column order is accepted and normalized):

        ==================  =================================================
        ``as_of_date``      the cross-section's session
        ``symbol``          ticker, canonical upper case, from the universe
        ``factor_name``     snake_case factor identifier
        ``factor_version``  non-empty version string
        ``raw_value``       float or ``NaN`` (not a transformed score)
        ``is_eligible``     boolean, never missing
        ==================  =================================================

    universe:
        Symbols the factor was asked about; every one must appear exactly once.
    as_of_date:
        Optional cross-check against the frame's own ``as_of_date`` values.

    Notes
    -----
    Panels are compared by identity (``eq=False``) because comparing two frames
    elementwise is ambiguous; use :meth:`frame` for value comparisons.

    Examples
    --------
    >>> frame = pd.DataFrame(
    ...     {
    ...         "as_of_date": [pd.Timestamp("2024-01-05")] * 2,
    ...         "symbol": ["AAPL", "MSFT"],
    ...         "factor_name": ["momentum_12_1"] * 2,
    ...         "factor_version": ["1.0.0"] * 2,
    ...         "raw_value": [0.12, 0.35],
    ...         "is_eligible": [True, True],
    ...     }
    ... )
    >>> panel = FactorPanel(frame, universe=["AAPL", "MSFT"])
    >>> panel.factor_name
    'momentum_12_1'
    >>> len(panel)
    2
    """

    as_of_date: date
    universe: tuple[str, ...]
    factor_name: str
    factor_version: str
    _rows: pd.DataFrame = field(repr=False)

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        universe: Iterable[str],
        as_of_date: date | datetime | str | None = None,
    ) -> None:
        """Validate ``frame`` as one cross-section and freeze it."""
        symbols = _canonical_universe(universe, "FactorPanel.universe")
        expected = (
            None if as_of_date is None else _normalize_date(as_of_date, "FactorPanel.as_of_date")
        )
        rows, resolved, name, version = _validated_panel_rows(frame, symbols, expected)

        object.__setattr__(self, "as_of_date", resolved)
        object.__setattr__(self, "universe", symbols)
        object.__setattr__(self, "factor_name", name)
        object.__setattr__(self, "factor_version", version)
        object.__setattr__(self, "_rows", rows)

    @property
    def frame(self) -> pd.DataFrame:
        """A copy of the long-format panel in canonical column order.

        Mutating the returned frame cannot change the panel.
        """
        return self._rows.copy(deep=True)

    @property
    def symbols(self) -> tuple[str, ...]:
        """The output symbols, in row order."""
        return tuple(self._rows[_PANEL_SYMBOL_COLUMN].tolist())

    def __len__(self) -> int:
        """Number of rows: one per symbol of the universe."""
        return int(len(self._rows))


# ---------------------------------------------------------------------------
# Factor -- the formula interface
# ---------------------------------------------------------------------------
class Factor(ABC):
    """Abstract base class every factor formula implements.

    A factor is pure computation over a :class:`FactorContext`: it declares what
    it needs through :attr:`spec`, reads only ``context.bars`` and the labels on
    the context, and hands back a validated :class:`FactorPanel`.  It must not
    read files, know a path, or reach into the storage layer -- a factor that
    can be built from an in-memory context is a factor that can be tested
    without any data on disk.

    Subclasses either declare ``spec`` as a class attribute (the usual, simplest
    form) or implement it as a property; both satisfy the abstract requirement::

        class Momentum12_1(Factor):
            spec = FactorSpec(
                name="momentum_12_1",
                version="1.0.0",
                required_fields=("adjusted_close",),
                lookback_sessions=252,
            )

            def compute(self, context: FactorContext) -> FactorPanel:
                ...
    """

    @property
    @abstractmethod
    def spec(self) -> FactorSpec:
        """The factor's immutable declaration (:class:`FactorSpec`)."""

    @abstractmethod
    def compute(self, context: FactorContext) -> FactorPanel:
        """Return this factor's cross-section for ``context``.

        Parameters
        ----------
        context:
            The point-in-time slice the factor may read.  A ``compute`` call must
            not mutate it and must not depend on anything outside it, so the same
            context always yields the same panel.

        Returns
        -------
        FactorPanel
            Already validated: one row per ``context.universe`` symbol, this
            factor's name and version, and the context's ``as_of_date``.
        """

    def __repr__(self) -> str:
        return f"{type(self).__name__}(spec={self.spec!r})"


__all__ = [
    "BAR_REQUIRED_COLUMNS",
    "FACTOR_NAME_PATTERN",
    "FACTOR_PANEL_COLUMNS",
    "Factor",
    "FactorContext",
    "FactorPanel",
    "FactorSpec",
]
