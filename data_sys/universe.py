"""Security master: the version-controlled initial stock universe.

The seed universe lives in ``config/universe_seed.csv``, so every collaborator
reads the exact same list with no runtime scrape of any index or screener.  This
module is the *only* place that knows how that file is shaped: downstream
modules receive normalized :class:`SecurityMasterEntry` objects (or a plain,
de-duplicated symbol list) and never parse the CSV themselves.

Contract of the seed file -- five columns, extra columns are ignored::

    symbol,name,exchange,security_type,is_active
    AAPL,Apple Inc.,NASDAQ,common_stock,true

Validation is *collecting*: every mistake in the file is reported at once via
:class:`~data_sys.errors.UniverseError`, so a typo in row 17 is not hidden behind
a typo in row 3.

Scope note: CIK numbers, delisting history and ticker-change mapping are
deliberately out of scope here -- they belong to the SEC / reference-data stage.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Collection, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from data_sys.config import PROJECT_ROOT
from data_sys.errors import UniverseError
from data_sys.schema import SYMBOL_PATTERN

logger = logging.getLogger(__name__)

#: Default location of the committed seed universe.
DEFAULT_UNIVERSE_PATH: Path = PROJECT_ROOT / "config" / "universe_seed.csv"

#: Columns the seed file must define, in the order they are stored.
REQUIRED_COLUMNS: tuple[str, ...] = (
    "symbol",
    "name",
    "exchange",
    "security_type",
    "is_active",
)

#: Only plain common stock is tradable in v1 (architecture document, section 3.3).
ALLOWED_SECURITY_TYPES: frozenset[str] = frozenset({"common_stock"})

#: US listing venues accepted in the seed file.  Overridable per call, so adding
#: an OTC or foreign listing later does not require editing this module.
ALLOWED_EXCHANGES: frozenset[str] = frozenset(
    {"NASDAQ", "NYSE", "NYSE AMERICAN", "NYSE ARCA", "BATS", "OTC"}
)

#: Guard rails on the file itself.
MIN_UNIVERSE_SIZE: int = 10


@dataclass(frozen=True)
class SecurityMasterEntry:
    """One normalized row of the security master."""

    symbol: str
    name: str
    exchange: str
    security_type: str
    is_active: bool

    def as_dict(self) -> dict[str, Any]:
        """Return the entry as an ordered, JSON-friendly mapping."""
        return {
            "symbol": self.symbol,
            "name": self.name,
            "exchange": self.exchange,
            "security_type": self.security_type,
            "is_active": self.is_active,
        }


@dataclass(frozen=True)
class Universe:
    """A validated, normalized snapshot of a universe file.

    Instances can only be produced by :func:`read_universe`, so every entry is
    guaranteed to be non-empty, upper-case, format-valid and unique.
    """

    entries: tuple[SecurityMasterEntry, ...]
    source: str = ""

    @property
    def symbols(self) -> list[str]:
        """Sorted, de-duplicated symbols of every entry (active or not)."""
        return sorted({entry.symbol for entry in self.entries})

    @property
    def active_symbols(self) -> list[str]:
        """Sorted, de-duplicated symbols of the entries flagged as active."""
        return sorted({entry.symbol for entry in self.entries if entry.is_active})

    def by_symbol(self) -> dict[str, SecurityMasterEntry]:
        """Map symbol -> entry for direct lookups."""
        return {entry.symbol: entry for entry in self.entries}

    def to_frame(self) -> pd.DataFrame:
        """Return the universe as a frame with the contract column order."""
        if not self.entries:
            return pd.DataFrame(columns=list(REQUIRED_COLUMNS))
        return pd.DataFrame(
            [entry.as_dict() for entry in self.entries], columns=list(REQUIRED_COLUMNS)
        )

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[SecurityMasterEntry]:
        return iter(self.entries)


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------
def _row_number(index: Any) -> int | None:
    """1-based data-row number as a human counts it (header excluded)."""
    try:
        return int(index) + 1
    except (TypeError, ValueError):
        return None


def _row_label(index: Any) -> str:
    """Human-readable location of a row, used in every problem message."""
    number = _row_number(index)
    if number is not None:
        return f"data row {number}"
    return f"index {index!r}"


def _as_text(values: pd.Series) -> pd.Series:
    """Whitespace-trimmed text where a missing value becomes ``""``, never NaN."""
    return values.astype("string").fillna("").str.strip()


def _parse_bool(value: Any) -> bool | None:
    """Parse a boolean-like cell; ``None`` means "not a valid boolean"."""
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):  # pragma: no cover - exotic scalar types
        pass

    text = str(value).strip().lower()
    if text in _TRUTHY:
        return True
    if text in _FALSY:
        return False
    return None


def _normalize(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``frame`` with the five contract columns normalized.

    Only loss-free normalizations are applied here (case folding and whitespace
    trimming).  Case folding happens *before* uniqueness is checked, so
    ``aapl`` and ``AAPL`` in the same file are correctly reported as a duplicate
    rather than silently kept as two securities.
    """
    out = pd.DataFrame(index=frame.index)
    for column in ("symbol", "name", "exchange", "security_type"):
        out[column] = _as_text(frame[column])
    out["is_active"] = frame["is_active"]

    out["symbol"] = out["symbol"].str.upper()
    out["exchange"] = out["exchange"].str.upper()
    out["security_type"] = out["security_type"].str.lower()
    return out


def _duplicated_headers(names: list[str]) -> list[str]:
    """Header names that occurred more than once in the file.

    ``pd.read_csv`` renames a repeated header instead of complaining, so a file
    with two ``symbol`` columns arrives as ``symbol`` + ``symbol.1``.  A suffixed
    name is therefore reported as duplicated when its base name is also present.
    """
    found: list[str] = []
    for name in names:
        match = _MANGLED_SUFFIX_RE.match(name)
        if match is None:
            continue
        base = match.group("base")
        if base in names:
            found.append(base)
    return sorted(set(found))


def _check_columns(frame: pd.DataFrame) -> list[str]:
    """Structural problems that must be fixed before any value can be checked."""
    problems: list[str] = []
    names = [str(column) for column in frame.columns]

    duplicated = _duplicated_headers(names)
    if duplicated:
        problems.append(
            f"duplicate header column(s) {duplicated}: column names must be unique"
        )

    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        problems.append(
            f"missing required column(s) {missing}; "
            f"expected at least {list(REQUIRED_COLUMNS)}, found {names}"
        )
    return problems


def _check_values(
    normalized: pd.DataFrame,
    *,
    min_size: int,
    allowed_security_types: Collection[str],
    allowed_exchanges: Collection[str],
) -> list[str]:
    """Value problems for an already-normalized frame."""
    row_count = len(normalized)
    if row_count == 0:
        return ["the universe file has a header but no data rows"]

    problems: list[str] = []
    if row_count < min_size:
        problems.append(
            f"the universe has {row_count} securit{'y' if row_count == 1 else 'ies'}; "
            f"at least {min_size} are required"
        )

    # -- symbol: non-empty, sane length, allowed characters ------------------
    symbols = normalized["symbol"]
    for index, value in symbols.items():
        where = _row_label(index)
        if not value:
            problems.append(f"{where}: 'symbol' must not be empty")
        elif len(value) > MAX_SYMBOL_LENGTH:
            problems.append(
                f"{where}: symbol {value!r} is longer than {MAX_SYMBOL_LENGTH} characters"
            )
        elif not _SYMBOL_RE.match(value):
            problems.append(
                f"{where}: symbol {value!r} does not match {SYMBOL_PATTERN} "
                "(upper-case letters, digits, '.' and '-' only)"
            )

    # -- symbol uniqueness (checked on normalized values) --------------------
    populated = symbols[symbols.ne("")]
    duplicated = populated[populated.duplicated(keep=False)]
    for symbol in sorted(duplicated.unique()):
        rows = [_row_label(index) for index, value in duplicated.items() if value == symbol]
        problems.append(
            f"duplicate symbol {symbol!r} at {', '.join(rows)}; "
            "each symbol must appear exactly once"
        )

    # -- name ---------------------------------------------------------------
    for index, value in normalized["name"].items():
        if not value:
            problems.append(f"{_row_label(index)}: 'name' must not be empty")

    # -- exchange -----------------------------------------------------------
    exchanges = {str(value).upper() for value in allowed_exchanges}
    for index, value in normalized["exchange"].items():
        where = _row_label(index)
        if not value:
            problems.append(f"{where}: 'exchange' must not be empty")
        elif value not in exchanges:
            problems.append(f"{where}: unknown exchange {value!r}; allowed: {sorted(exchanges)}")

    # -- security_type ------------------------------------------------------
    security_types = {str(value).lower() for value in allowed_security_types}
    for index, value in normalized["security_type"].items():
        where = _row_label(index)
        if not value:
            problems.append(f"{where}: 'security_type' must not be empty")
        elif value not in security_types:
            problems.append(
                f"{where}: security_type {value!r} is not supported yet; "
                f"allowed: {sorted(security_types)}"
            )

    # -- is_active ----------------------------------------------------------
    for index, value in normalized["is_active"].items():
        if _parse_bool(value) is None:
            problems.append(
                f"{_row_label(index)}: 'is_active' must be a boolean "
                f"(true/false, yes/no, 1/0); got {value!r}"
            )

    return problems


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def _read_csv(path: Path) -> pd.DataFrame:
    """Read the seed file as all-text so no ticker is ever coerced to NaN.

    ``dtype=str`` + ``keep_default_na=False`` matters here: without them pandas
    would turn a symbol such as ``NA`` or ``TRUE`` into a missing value.  Header
    names are whitespace-trimmed, values keep whatever the file contains.
    """
    if not path.exists():
        raise UniverseError(f"universe file not found: '{path}'")
    if path.is_dir():
        raise UniverseError(f"universe path is a directory, not a file: '{path}'")

    try:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False, skip_blank_lines=True)
    except pd.errors.EmptyDataError as exc:
        raise UniverseError(f"universe file is empty: '{path}'") from exc
    except (OSError, UnicodeDecodeError, pd.errors.ParserError) as exc:
        raise UniverseError(f"could not read universe file '{path}': {exc}") from exc

    frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def _build_entries(normalized: pd.DataFrame) -> tuple[SecurityMasterEntry, ...]:
    """Build entries from a frame that has already passed validation."""
    return tuple(
        SecurityMasterEntry(
            symbol=row["symbol"],
            name=row["name"],
            exchange=row["exchange"],
            security_type=row["security_type"],
            is_active=bool(_parse_bool(row["is_active"])),
        )
        for _, row in normalized.iterrows()
    )


def validate_universe(
    frame: pd.DataFrame | None,
    *,
    min_size: int = MIN_UNIVERSE_SIZE,
    allowed_security_types: Collection[str] = ALLOWED_SECURITY_TYPES,
    allowed_exchanges: Collection[str] = ALLOWED_EXCHANGES,
) -> list[str]:
    """Return every problem found in ``frame``; an empty list means valid.

    Mirrors :func:`data_sys.universe.read_universe` without raising, which is
    handy for tests and for validating a frame that was built in memory.
    """
    if frame is None:
        return ["universe frame is missing (None)"]

    problems = _check_columns(frame)
    if problems:
        return problems

    return _check_values(
        _normalize(frame),
        min_size=min_size,
        allowed_security_types=allowed_security_types,
        allowed_exchanges=allowed_exchanges,
    )


def read_universe(
    path: Path | str | None = None,
    *,
    min_size: int = MIN_UNIVERSE_SIZE,
    allowed_security_types: Collection[str] = ALLOWED_SECURITY_TYPES,
    allowed_exchanges: Collection[str] = ALLOWED_EXCHANGES,
) -> Universe:
    """Read and validate a universe file, defaulting to the committed seed file.

    Parameters
    ----------
    path:
        CSV to read.  ``None`` means :data:`DEFAULT_UNIVERSE_PATH`.
    min_size:
        Minimum number of data rows required.
    allowed_security_types:
        Permitted ``security_type`` values (case-insensitive).
    allowed_exchanges:
        Permitted ``exchange`` values (case-insensitive).

    Returns
    -------
    Universe
        Entries normalized to upper-case symbols, upper-case exchanges and
        lower-case security types, in file order.

    Raises
    ------
    data_sys.errors.UniverseError
        If the file is missing/unreadable, a required column is absent, or any
        value violates the contract.  All problems are reported together.
    """
    target = Path(path) if path is not None else DEFAULT_UNIVERSE_PATH

    frame = _read_csv(target)

    problems = _check_columns(frame)
    if problems:
        raise UniverseError(f"invalid universe configuration in '{target}'", problems)

    normalized = _normalize(frame)
    problems = _check_values(
        normalized,
        min_size=min_size,
        allowed_security_types=allowed_security_types,
        allowed_exchanges=allowed_exchanges,
    )
    if problems:
        raise UniverseError(f"invalid universe configuration in '{target}'", problems)

    logger.info("universe validated: %d securit(ies) from '%s'", len(normalized), target)
    return Universe(entries=_build_entries(normalized), source=str(target))


def load_universe_symbols(
    path: Path | str | None = None,
    *,
    active_only: bool = True,
    min_size: int = MIN_UNIVERSE_SIZE,
    allowed_security_types: Collection[str] = ALLOWED_SECURITY_TYPES,
    allowed_exchanges: Collection[str] = ALLOWED_EXCHANGES,
) -> list[str]:
    """Return the normalized, de-duplicated, sorted symbols of the universe.

    This is the entry point downstream stages should use: they get a plain list
    of validated tickers and never see the CSV or its columns.
    """
    universe = read_universe(
        path,
        min_size=min_size,
        allowed_security_types=allowed_security_types,
        allowed_exchanges=allowed_exchanges,
    )
    return universe.active_symbols if active_only else universe.symbols


__all__ = [
    "DEFAULT_UNIVERSE_PATH",
    "REQUIRED_COLUMNS",
    "ALLOWED_SECURITY_TYPES",
    "ALLOWED_EXCHANGES",
    "MIN_UNIVERSE_SIZE",
    "MAX_SYMBOL_LENGTH",
    "SecurityMasterEntry",
    "Universe",
    "validate_universe",
    "read_universe",
    "load_universe_symbols",
]

MAX_SYMBOL_LENGTH: int = 10

_SYMBOL_RE = re.compile(SYMBOL_PATTERN)
#: Signature of a header name that ``pd.read_csv`` had to rename (``symbol`` -> ``symbol.1``).
_MANGLED_SUFFIX_RE = re.compile(r"^(?P<base>.+)\.(?P<index>\d+)$")
_TRUTHY = frozenset({"true", "t", "yes", "y", "1"})
_FALSY = frozenset({"false", "f", "no", "n", "0"})
