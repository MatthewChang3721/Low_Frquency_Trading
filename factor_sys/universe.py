"""Which symbols a factor may be computed for, and on which date.

What this module is
-------------------
``factor_sys.universe`` answers one question for the rest of the factor system:
*which symbols may be computed for a given* ``as_of_date``?  The answer travels
as a :class:`UniverseSnapshot` -- the date that was asked about, the symbols, and
where they came from.

``UniverseProvider``
    the interface: ``snapshot(as_of_date) -> UniverseSnapshot``
``StaticUniverseProvider``
    the v1 implementation: the committed, modern stock pool of
    ``config/universe_seed.csv`` (``is_active=true``), identical for every date

Static, *not* point-in-time -- read this before trusting a backtest
------------------------------------------------------------------
:class:`StaticUniverseProvider` is a **static** universe: it returns today's
active pool for every date it is asked about, including dates years before those
companies listed.  It therefore cannot express

* historical index membership (which stocks were in an index in 2015),
* delistings -- symbols that no longer trade are silently absent,
* ticker changes (``FB`` -> ``META`` is one security that appears once),

and it does **not** remove survivorship bias: a 2010-2020 backtest measured on
the 2026 pool only ever sees the winners that survived until 2026.

``as_of_date`` is already part of the interface so that a future point-in-time
provider can honour it; it is *not* a promise that this implementation filters by
date.  Replacing :class:`StaticUniverseProvider` with a historical implementation
must not require touching a single ``Factor``, the runner or a strategy, because
they only ever see :class:`UniverseSnapshot` through :class:`UniverseProvider`.

Layering
--------
Universe *reading and validation* belong to :mod:`data_sys.universe` and are
reused here: this module never parses the seed file itself, never touches market
bars and never reaches into the storage layer.  It only adds the factor system's
own vocabulary on top.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from data_sys.errors import UniverseError
from data_sys.universe import DEFAULT_UNIVERSE_PATH, read_universe
from factor_sys.contracts import _canonical_universe, _normalize_date
from factor_sys.errors import FactorContractError


# ---------------------------------------------------------------------------
# UniverseSnapshot -- the answer
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class UniverseSnapshot:
    """The symbols a factor may be computed for on one ``as_of_date``.

    The snapshot is the only universe object the rest of the factor system sees:
    a :class:`~factor_sys.contracts.FactorContext` is built for exactly these
    symbols, so a stored result can always be traced back to the pool and the
    configuration it came from.

    Parameters
    ----------
    as_of_date:
        The session the pool is requested for, as a ``date``, ``datetime`` or
        ISO-8601 ``"YYYY-MM-DD"`` string.  Stored timezone-naive and
        day-normalized, exactly like every other date in the contract.
    symbols:
        The requested symbols: non-empty, upper-case, unique and each matching
        :data:`data_sys.schema.SYMBOL_PATTERN`.  Frozen into a tuple, so a
        caller's list cannot be changed behind the snapshot.
    source:
        Non-empty description of where the pool came from -- for
        :class:`StaticUniverseProvider` the universe file it read.  It travels
        with the snapshot so a result can be explained later.

    Raises
    ------
    factor_sys.errors.FactorContractError
        If ``as_of_date`` is not a usable date, ``symbols`` is not a non-empty
        pool of valid tickers, or ``source`` is not a non-empty string.

    Examples
    --------
    >>> snapshot = UniverseSnapshot(
    ...     as_of_date="2024-01-05",
    ...     symbols=("AAPL", "MSFT"),
    ...     source="config/universe_seed.csv",
    ... )
    >>> snapshot.as_of_date
    datetime.date(2024, 1, 5)
    >>> "MSFT" in snapshot
    True
    """

    as_of_date: date
    symbols: tuple[str, ...]
    source: str

    def __post_init__(self) -> None:
        # ``factor_sys.contracts`` owns "a valid date" and "a valid universe" for
        # the whole factor system, so the snapshot reuses those validators instead
        # of re-declaring the ticker pattern or a second date parser here.
        as_of_date = _normalize_date(self.as_of_date, "UniverseSnapshot.as_of_date")
        symbols = _canonical_universe(self.symbols, "UniverseSnapshot.symbols")

        source = self.source
        if not isinstance(source, str) or not source.strip():
            raise FactorContractError(
                "UniverseSnapshot.source must be a non-empty description of where the "
                f"universe came from, got {source!r}"
            )

        object.__setattr__(self, "as_of_date", as_of_date)
        object.__setattr__(self, "symbols", symbols)
        object.__setattr__(self, "source", source.strip())

    def __len__(self) -> int:
        """Number of symbols in the pool."""
        return len(self.symbols)

    def __contains__(self, symbol: object) -> bool:
        """``"AAPL" in snapshot``: membership in the frozen pool."""
        return symbol in self.symbols


# ---------------------------------------------------------------------------
# UniverseProvider -- the interface
# ---------------------------------------------------------------------------
class UniverseProvider(ABC):
    """The one interface every stock-universe implementation must satisfy.

    A provider maps a requested date to the pool of symbols that may be computed
    then::

        provider = StaticUniverseProvider()
        snapshot = provider.snapshot("2024-01-05")

    ``as_of_date`` is a plain ``date`` or an ISO-8601 ``"YYYY-MM-DD"`` string, and
    the returned :class:`UniverseSnapshot` echoes it.  An implementation is
    expected to be *pure*: the same date yields an equal snapshot, and a caller
    can never change the provider's pool through the objects it receives.

    Two implementations are expected over time:

    * :class:`StaticUniverseProvider` -- today's committed pool, for every date
      (the current v1, deliberately biased towards surviving companies);
    * a point-in-time provider reading historical membership and delistings,
      added later.

    Because a factor, the runner and a strategy only ever call :meth:`snapshot`,
    swapping the implementation needs no change at any call site.
    """

    @abstractmethod
    def snapshot(self, as_of_date: date | str) -> UniverseSnapshot:
        """Return the stock pool that may be computed for ``as_of_date``.

        Parameters
        ----------
        as_of_date:
            The session the pool is requested for, as a ``date`` or an ISO-8601
            ``"YYYY-MM-DD"`` string.

        Returns
        -------
        UniverseSnapshot
            The validated pool for that date, with its source recorded.

        Raises
        ------
        factor_sys.errors.FactorContractError
            If the date cannot be used or the implementation cannot supply a
            valid pool for it.
        """


# ---------------------------------------------------------------------------
# StaticUniverseProvider -- the v1 implementation
# ---------------------------------------------------------------------------
def _universe_path(path: Path | str | None) -> Path:
    """Resolve the file to read: ``None`` means the committed seed universe."""
    if path is None:
        return DEFAULT_UNIVERSE_PATH
    if isinstance(path, (str, Path)):
        return Path(path)
    raise FactorContractError(
        f"universe path must be a str or a pathlib.Path, got {type(path).__name__}"
    )


class StaticUniverseProvider(UniverseProvider):
    """The committed, modern stock pool: identical for every ``as_of_date``.

    This is a **static** universe, not a point-in-time historical one:

    * the pool is read once from ``config/universe_seed.csv`` (or from the file
      passed to the constructor), keeping only the rows flagged
      ``is_active=true``;
    * :meth:`snapshot` ignores the requested date apart from echoing it back, so
      asking for 2001 returns the same symbols as asking for today;
    * it therefore models **no** index-membership history, no delisting and no
      ticker change, and it does **not** remove survivorship bias.  Any result
      computed through it inherits that limitation and must be reported as such.

    The pool is frozen at construction: editing or deleting the universe file
    afterwards does not change an existing provider -- only a newly constructed
    one sees the new content.  This is what makes a run reproducible, since one
    run reads one pool once.

    Universe reading is delegated to :func:`data_sys.universe.read_universe`, so
    the CSV is parsed and validated by exactly one module (the ticker pattern, the
    exchange list, the minimum size and the boolean parsing all come from there).
    A file that violates that contract is reported as a
    :class:`~factor_sys.errors.FactorContractError`, so callers can catch the whole
    factor system through :class:`~factor_sys.errors.FactorError`, with the
    original :class:`~data_sys.errors.UniverseError` kept as ``__cause__``.

    Parameters
    ----------
    path:
        Universe CSV to read.  ``None`` (the default) means
        :data:`data_sys.universe.DEFAULT_UNIVERSE_PATH`.

    Raises
    ------
    factor_sys.errors.FactorContractError
        If the file is missing or unreadable, violates the universe contract, or
        contains no ``is_active=true`` row.
    """

    def __init__(self, path: Path | str | None = None) -> None:
        target = _universe_path(path)

        try:
            universe = read_universe(target)
        except UniverseError as exc:
            raise FactorContractError(
                f"could not load the static universe from '{target}': {exc}"
            ) from exc

        symbols = tuple(universe.active_symbols)
        if not symbols:
            raise FactorContractError(
                f"the universe file '{target}' has no symbol flagged is_active=true: "
                "a static universe must never be empty"
            )

        # Read once, here, and never again: the pool below is what every snapshot
        # of this provider will carry.
        self._path = target
        self._symbols = symbols
        self._source = universe.source or str(target)

    @property
    def path(self) -> Path:
        """The universe file the pool was read from."""
        return self._path

    @property
    def source(self) -> str:
        """The ``source`` string every snapshot of this provider records."""
        return self._source

    @property
    def symbols(self) -> tuple[str, ...]:
        """The frozen pool: sorted, upper-case, de-duplicated tickers."""
        return self._symbols

    def snapshot(self, as_of_date: date | str) -> UniverseSnapshot:
        """Return this fixed pool for ``as_of_date``.

        Only ``as_of_date`` varies between calls -- the symbols are the same on
        every date, which is exactly what makes this pool static.  Each call
        returns a new :class:`UniverseSnapshot`.
        """
        return UniverseSnapshot(
            as_of_date=as_of_date,
            symbols=self._symbols,
            source=self._source,
        )

    def __len__(self) -> int:
        """Number of symbols in the frozen pool."""
        return len(self._symbols)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(path={str(self._path)!r}, symbols={len(self._symbols)})"


__all__ = [
    "StaticUniverseProvider",
    "UniverseProvider",
    "UniverseSnapshot",
]
