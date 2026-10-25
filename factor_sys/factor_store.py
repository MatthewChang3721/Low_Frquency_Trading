"""Local Parquet storage for factor panels: one panel, one file, one atomic replace.

What this module is
-------------------
``factor_sys.factor_store`` is the only place in the project that turns a
validated :class:`~factor_sys.contracts.FactorPanel` into something durable::

    result = runner.run(Momentum12_1(), "2024-12-31")
    path = ParquetFactorStore().write(result.panel)
    again = ParquetFactorStore().read("momentum_12_1", "1.0.0", "2024-12-31")

``FactorStore``
    the interface every backend implements: ``write(panel) -> Path``,
    ``read(factor_name, factor_version, as_of_date) -> FactorPanel``,
    ``read_range(name, version, start_date, end_date) -> tuple[FactorPanel, ...]``,
    ``exists(factor_name, factor_version, as_of_date) -> bool`` and
    ``path_for(factor_name, factor_version, as_of_date) -> Path``
``ParquetFactorStore``
    the v1 implementation: exactly one Parquet file per stored panel, under the
    project's ``data/factors/panels`` by default

One key, one file
-----------------
A panel is stored under the Hive-style path::

    <root>/factor_name=momentum_12_1/
        factor_version=1.0.0/
            year=2024/
                as_of_date=2024-12-31.parquet

The unique key is ``(factor_name, factor_version, as_of_date)``; the file name
is derived from the same key, so writing a key twice replaces the panel instead
of adding a second copy.  Partition names follow the Hive ``key=value``
convention so a later analysis step can open the whole tree with
``pyarrow.dataset`` (``partitioning="hive"``) without this module adding an
index.  There is deliberately no single database file and no DuckDB catalog: a
batch run writes many small independent files, and replacing one of them never
touches another.

Factor names are already snake_case (``^[a-z0-9_]+$``, see
:data:`factor_sys.contracts.FACTOR_NAME_PATTERN`), so they are used as they are.
Versions are free-form strings, so they are **percent-encoded** before they
become a directory name: a version can therefore never contribute a path
separator, a drive letter or a traversal segment.  ``"."`` and ``".."`` are
refused outright, and every final path is resolved and checked to stay inside
the store root.  Two different version strings are always two different
directories (``%`` itself is encoded), so the encoding never merges two keys.
Trailing dots and spaces, which Windows silently strips, are encoded too, and a
version a Windows device already owns (``CON``, ``NUL``, ``LPT1``, ...) is
spelled with its first character encoded, so a version always names a directory
the filesystem accepts as written.

``path_for`` answers *where* a key lives without touching it: it applies exactly
the key checks and the encoding ``write``, ``read`` and ``exists`` apply, and
returns the path those methods would use.  It never creates, reads, writes or
requires a file.  A caller that has to report or inspect the location of a
stored panel (a command printing where its panel went, a test asserting an
overwrite landed in the same place) therefore asks the store instead of copying
its layout -- the layout is this module's private business.

Metadata: how one file becomes one panel again
----------------------------------------------
The six panel columns are the table; everything else needed to rebuild the
:class:`~factor_sys.contracts.FactorPanel` object is written into the Parquet
**schema metadata** (never as extra columns, so the table stays exactly
``FACTOR_PANEL_COLUMNS``):

=================================  ===========================================
``factor_store.format_version``    layout version of this store (``"1"``)
``factor_store.factor_name``       the panel's factor name
``factor_store.factor_version``    the panel's version
``factor_store.as_of_date``        ISO ``YYYY-MM-DD``
``factor_store.universe``          JSON array of symbols, in the panel's order
``factor_store.stored_at_utc``     ISO-8601 UTC timestamp of the write
=================================  ===========================================

``universe`` cannot be recovered from the table: the table holds one row per
symbol, but row order is the factor's business, while
``FactorPanel.universe`` is the ordered universe the factor was asked about.
Storing it explicitly keeps that order (and the "one row per universe symbol"
check) recoverable.  ``stored_at_utc`` records *when* a result was produced
without pretending it is part of the factor's identity; a re-run overwrites it.

A read validates before it answers: the metadata must exist and be complete, its
format version must be supported, its name/version/date must equal the requested
key, the table must carry exactly the panel columns, and the table's own
``factor_name``/``factor_version``/``as_of_date`` values must agree with the
metadata.  Any violation is a :class:`~factor_sys.errors.FactorStoreError` -- a
damaged file is never partially read, guessed at or silently skipped.

A range of stored panels
------------------------
``read_range`` answers "what is already on disk for this factor and version
between these two days" without computing anything::

    panels = store.read_range("momentum_12_1", "1.0.0", "2024-01-01", "2024-12-31")

The answer is a ``tuple`` of validated panels in strictly ascending
``as_of_date`` order, one object per stored file.  It is deliberately not one
concatenated frame: a second date may have been produced for a different
point-in-time universe, and every panel carries its own date, its own symbol
order and its own factor identity.  A later analysis decides how to align them.

The interval is closed (``start_date <= as_of_date <= end_date``) and the answer
holds only panels that are actually stored: sessions are never invented, gaps
are not filled, and a legal range with nothing stored is an empty tuple rather
than an error.  Each file is turned back into a panel by the same routine
``read`` uses, so the metadata and table checks apply to a range exactly as they
apply to a single key.

The tree is scanned, not trusted.  Any ``*.parquet`` file under the requested
factor/version whose name is not ``as_of_date=YYYY-MM-DD.parquet``, a directory
that is not a ``year=YYYY`` partition, a year partition that disagrees with the
date of a file inside it, and a nested directory where the layout expects a file
are all refused with :class:`~factor_sys.errors.FactorStoreError`; files that are
not panel files at all (a ``.tmp`` left by an interrupted write) are ignored.  The
layout of the whole version directory is checked, so a misnamed file or a wrong
year partition is reported even when it sits outside the range; the *contents* of
a panel are read for the days inside the range, and a damaged panel there fails
the whole call -- a partial history would look complete to the analysis above.


Atomic overwrite
----------------
``write`` never truncates the live file.  It writes a complete temporary Parquet
file *in the target directory* (same filesystem, so the final step is atomic),
re-reads its schema to confirm it is a readable Parquet file, and only then
swaps it into place with :func:`os.replace`.  A failure at any point removes the
temporary file, leaves the previous panel untouched and raises
:class:`~factor_sys.errors.FactorStoreError`.  Unlike the data pipeline's publish
transaction (:mod:`data_sys.storage`) there is no staging tree, no trash area and
no journal: one panel is one file, so one replace is the whole transaction.

What the store does not do
--------------------------
It does not compute a factor, read market bars, decide a universe, call a
:class:`~factor_sys.runner.FactorRunner`, schedule dates, or implement a cache
policy.  It stores and returns what it was given, and nothing else: a caller may
treat a stored panel as a cache by asking :meth:`FactorStore.exists` before
computing (as ``scripts/run_factor.py`` does), but that decision -- and what a
hit means -- belongs to the caller, not to this module.

Layering
--------
This module depends on :mod:`factor_sys.contracts`, :mod:`factor_sys.errors`,
:mod:`data_sys.config` (for the default root), the standard library, pandas and
PyArrow.  It must not import the data pipeline's storage, update, batch-query or
provider modules, DuckDB, a network client, ``factors`` or the runner -- the
store is a leaf of the factor system, not a second pipeline.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from abc import ABC, abstractmethod
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, NamedTuple
from urllib.parse import quote

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from data_sys.config import PipelineConfig
from factor_sys.contracts import FACTOR_NAME_PATTERN, FACTOR_PANEL_COLUMNS, FactorPanel
from factor_sys.errors import FactorContractError, FactorStoreError

#: Layout version written into every file.  A reader refuses anything else.
FORMAT_VERSION: str = "1"

#: Every layout version this build can read.
SUPPORTED_FORMAT_VERSIONS: tuple[str, ...] = (FORMAT_VERSION,)

#: Prefix of every Parquet schema-metadata key this module owns.
METADATA_PREFIX: str = "factor_store."

#: The metadata entries a stored panel must carry, without the prefix.
METADATA_KEYS: tuple[str, ...] = (
    "format_version",
    "factor_name",
    "factor_version",
    "as_of_date",
    "universe",
    "stored_at_utc",
)

#: Suffix of a stored panel file.
PANEL_SUFFIX: str = ".parquet"

#: Device names Windows refuses as a directory name, whatever the extension.
_WINDOWS_RESERVED: frozenset[str] = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{number}" for number in range(1, 10)}
    | {f"LPT{number}" for number in range(1, 10)}
)

_NAME_RE = re.compile(FACTOR_NAME_PATTERN)
_ISO_DAY_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


class FactorStore(ABC):
    """The stable storage interface of the factor system.

    One panel is one object with one key: ``(factor_name, factor_version,
    as_of_date)``.  ``write`` returns the path it wrote, ``read`` returns a
    freshly validated :class:`~factor_sys.contracts.FactorPanel` -- an object the
    caller may mutate through :attr:`FactorPanel.frame` without affecting the
    store or a later read -- ``read_range`` returns every stored panel of one
    factor/version between two days, in date order, ``exists`` answers whether
    one key is stored without raising for a merely absent panel, and ``path_for``
    answers where that key's file is (or would be) without touching it.

    The interface says nothing about *how* a backend stores a panel; it only
    fixes the operations and their error contract: an impossible request or a
    damaged store is always a :class:`~factor_sys.errors.FactorStoreError`.
    """

    @abstractmethod
    def write(self, panel: FactorPanel) -> Path:
        """Store ``panel`` under its own key and return the path written.

        Storing the same key again replaces the previous panel atomically: a
        reader either sees the complete old panel or the complete new one.

        Raises
        ------
        FactorStoreError
            If ``panel`` is not a :class:`~factor_sys.contracts.FactorPanel`, or
            the backend cannot store it.
        """

    @abstractmethod
    def read(
        self,
        factor_name: str,
        factor_version: str,
        as_of_date: date | str,
    ) -> FactorPanel:
        """Return the stored panel for one key.

        Parameters
        ----------
        factor_name:
            The snake_case factor identifier the panel was written with.
        factor_version:
            The exact version string the panel was written with.
        as_of_date:
            The cross-section's session, as a :class:`datetime.date` or a strict
            ISO ``"YYYY-MM-DD"`` string.

        Raises
        ------
        FactorStoreError
            If the key is illegal, nothing is stored for it, or the stored file
            cannot be restored as a valid panel.
        """

    @abstractmethod
    def read_range(
        self,
        factor_name: str,
        factor_version: str,
        start_date: date | str,
        end_date: date | str,
    ) -> tuple[FactorPanel, ...]:
        """Return every stored panel of one factor/version within a closed range.

        The result is a ``tuple`` with one freshly validated panel per stored
        file, in strictly ascending ``as_of_date`` order.  It is not a single
        concatenated frame: two dates may have been produced for different
        universes, and each panel keeps its own date, symbol order and factor
        identity for the analysis layer to align.

        Only panels that are actually stored are returned.  Sessions are not
        filled in from a calendar, an absent day is simply absent, and a legal
        range that holds nothing is an empty tuple -- not an error.  Reading a
        range computes nothing: it never builds a factor, reads market data or
        writes a file.

        Parameters
        ----------
        factor_name:
            The snake_case factor identifier the panels were written with.
        factor_version:
            The exact version string the panels were written with.
        start_date, end_date:
            The inclusive bounds of the range, each a :class:`datetime.date` or a
            strict ISO ``"YYYY-MM-DD"`` string.

        Raises
        ------
        FactorStoreError
            If a name, version or bound is illegal, if ``start_date`` is after
            ``end_date``, or if a panel stored *inside the range* cannot be
            restored as a valid panel.  Such a panel fails the whole call; it is
            never skipped to return a partial history.
        """

    @abstractmethod
    def exists(
        self,
        factor_name: str,
        factor_version: str,
        as_of_date: date | str,
    ) -> bool:
        """Whether one key is stored.

        A legal key with no panel is ``False``; an illegal key is still a
        :class:`~factor_sys.errors.FactorStoreError`, so a typo never looks like
        a cache miss.
        """

    @abstractmethod
    def path_for(
        self,
        factor_name: str,
        factor_version: str,
        as_of_date: date | str,
    ) -> Path:
        """Return the location one key's panel has (or would have).

        This is a *pure* question about a key, not about the disk: the key is
        validated and encoded exactly as :meth:`write`, :meth:`read` and
        :meth:`exists` validate and encode it, so the answer is the path those
        methods use -- but nothing is created, read, written or required to
        exist.  A key that was never stored therefore still has a path.

        It exists so that a caller can report or inspect *where* a panel lives
        without copying the backend's layout, which stays private to the
        backend.

        Raises
        ------
        FactorStoreError
            If the key is illegal (an unknown name shape, a version that cannot
            name a directory, a date that is not a day, a key that would resolve
            outside the store).
        """


class _StoredMetadata(NamedTuple):
    """The decoded schema metadata of one stored panel."""

    format_version: str
    factor_name: str
    factor_version: str
    as_of_date: date
    universe: tuple[str, ...]
    stored_at_utc: str


class _Key(NamedTuple):
    """One validated storage key and the one file it maps to."""

    target: Path
    factor_name: str
    factor_version: str
    as_of_date: date


# ---------------------------------------------------------------------------
# Key validation: the request side
# ---------------------------------------------------------------------------
def _checked_name(factor_name: Any) -> str:
    """``factor_name`` as a usable key part, or raise.

    The pattern has no path character at all, so a name can never name anything
    outside the store; it is validated rather than encoded because every factor
    name already has to match it (:data:`FACTOR_NAME_PATTERN`).
    """
    if not isinstance(factor_name, str) or not _NAME_RE.fullmatch(factor_name):
        raise FactorStoreError(
            f"factor_name must be a snake_case identifier matching {FACTOR_NAME_PATTERN}, "
            f"got {factor_name!r}"
        )
    return factor_name


def _checked_version(factor_version: Any) -> str:
    """``factor_version`` as the exact string a key is built from, or raise.

    The value is not normalized (no stripping): the version is part of the key,
    so ``"1.0.0"`` and ``" 1.0.0 "`` are different keys, and silently folding
    them together would return one factor's results for another one's version.
    """
    if not isinstance(factor_version, str) or not factor_version.strip():
        raise FactorStoreError(f"factor_version must be a non-empty string, got {factor_version!r}")
    return factor_version


def _version_segment(factor_version: Any) -> str:
    """The directory-name segment of ``factor_version``.

    Percent-encoding (with no character left safe) makes the segment a single
    path name whatever the version contains: ``"1.0.0"`` stays ``"1.0.0"``,
    ``"a/b"`` becomes ``"a%2Fb"`` and ``"..\\\\escape"`` cannot traverse anything.
    ``"."`` and ``".."`` carry no version information, so they are refused
    instead of naming a directory that means "this" or "the parent".

    Two filesystem rules are applied on top of the encoding, because Windows
    would otherwise refuse or silently change the name: trailing dots and spaces
    are encoded (Windows strips them), and a name a Windows device already owns
    (``CON``, ``NUL``, ``LPT1``, ...) has its first character encoded.  Both
    repairs only touch characters that ``quote`` leaves alone, so the mapping
    stays one-to-one: two different versions never share a directory.
    """
    version = _checked_version(factor_version)
    if version in {".", ".."}:
        raise FactorStoreError(
            f"factor_version {version!r} cannot name a directory: '.' and '..' are "
            "path segments, not versions"
        )

    segment = quote(version, safe="")
    stripped = segment.rstrip(". ")
    if stripped != segment:
        tail = segment[len(stripped) :]
        segment = stripped + "".join(f"%{ord(character):02X}" for character in tail)
    if segment.split(".", 1)[0].upper() in _WINDOWS_RESERVED:
        segment = f"%{ord(segment[0]):02X}{segment[1:]}"
    return segment


def _checked_as_of_date(value: Any, *, label: str = "as_of_date") -> date:
    """``value`` as a plain day, or raise.

    A ``datetime`` is refused even though it *is* a ``date``: a panel is a daily
    cross-section, and silently dropping a time of day would let the store, not
    the caller, decide which session the key names.  Strings must be strict ISO
    ``"YYYY-MM-DD"`` and must name a real day.

    ``read_range`` passes ``label=`` so that a bad bound is reported as
    ``start_date``/``end_date`` rather than as the single-panel key; the rules
    themselves are shared, never re-implemented for ranges.
    """
    if isinstance(value, datetime):
        raise FactorStoreError(
            f"{label} must be a date or a strict ISO 'YYYY-MM-DD' string, got the "
            f"datetime {value!r}: a panel is stored per session, pass {label}.date()"
        )
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        text = value.strip()
        if _ISO_DAY_RE.fullmatch(text):
            try:
                return date.fromisoformat(text)
            except ValueError:
                pass  # right shape, but no such day (e.g. '2024-02-30')
    raise FactorStoreError(
        f"{label} must be a date or a strict ISO 'YYYY-MM-DD' string, got {value!r}"
    )


def _key_label(factor_name: str, factor_version: str, as_of_date: date) -> str:
    """How one key is named in an error message."""
    return (
        f"factor {factor_name!r} version {factor_version!r} on as_of_date {as_of_date.isoformat()}"
    )


# ---------------------------------------------------------------------------
# Root validation
# ---------------------------------------------------------------------------
def _checked_root(value: Path | str | None) -> Path:
    """The store root: the given directory, or the pipeline's default one.

    An existing *file* is refused here, because it can never become a directory
    of panels.  A root that does not exist yet is accepted: ``write`` creates it,
    ``exists`` answers ``False`` and ``read`` explains that there is nothing to
    read.
    """
    if value is None:
        return PipelineConfig().data_root / "factors" / "panels"
    if isinstance(value, str):
        if not value.strip():
            raise FactorStoreError(
                "factor store root must not be blank: pass a directory, or omit it to use "
                "the project's data/factors/panels"
            )
        root = Path(value)
    elif isinstance(value, Path):
        root = value
    else:
        raise FactorStoreError(
            f"factor store root must be a path or a string, got {type(value).__name__}"
        )

    if root.exists() and not root.is_dir():
        raise FactorStoreError(
            f"factor store root {str(root)!r} is not a directory: it cannot hold stored "
            "factor panels"
        )
    return root


def _path_inside(root: Path, target: Path) -> bool:
    """Whether ``target`` resolves to a location strictly below ``root``."""
    try:
        root_parts = Path(os.path.normcase(str(root.resolve())))
        target_parts = Path(os.path.normcase(str(target.resolve())))
    except OSError as exc:  # e.g. a path Windows refuses to resolve
        raise FactorStoreError(
            f"cannot resolve the factor store path {str(target)!r}: {exc}"
        ) from exc
    return target_parts != root_parts and target_parts.is_relative_to(root_parts)


# ---------------------------------------------------------------------------
# The Parquet implementation
# ---------------------------------------------------------------------------
class ParquetFactorStore(FactorStore):
    """One validated panel per Parquet file, under a Hive-style tree.

    The store is a directory of independent files::

        <root>/factor_name=<name>/factor_version=<version>/year=<YYYY>/
            as_of_date=<YYYY-MM-DD>.parquet

    Each file holds the six panel columns plus the schema metadata described in
    the module docstring, so one file is a complete, self-describing panel and
    replacing it never touches another key.

    Parameters
    ----------
    root:
        Directory holding the panels.  Defaults to
        ``<PipelineConfig().data_root>/factors/panels``, the project's
        ``data/factors/panels``; tests pass a temporary directory.  A root that
        does not exist yet is created on the first write.

    Notes
    -----
    Building a store touches nothing on disk.  An existing root that is a file,
    a blank root string and a root of the wrong type are refused immediately,
    because none of them can ever hold a panel.

    ``read_range`` walks this tree instead of guessing paths: every file it
    passes must have a legal panel name and sit in the year partition its date
    implies, and only the files whose own date falls inside the window are read
    and validated.
    """

    def __init__(self, root: Path | str | None = None) -> None:
        self._root: Path = _checked_root(root)

    @property
    def root(self) -> Path:
        """The directory this store reads and writes."""
        return self._root

    def write(self, panel: FactorPanel) -> Path:
        """Store ``panel`` under its own key, replacing that key atomically."""
        if not isinstance(panel, FactorPanel):
            raise FactorStoreError(
                "write() needs a validated FactorPanel, got "
                f"{type(panel).__name__}: only a panel that passed the contract may be stored"
            )

        key = self._key(panel.factor_name, panel.factor_version, panel.as_of_date)
        _atomic_write(_panel_table(panel), key.target)
        return key.target

    def read(
        self,
        factor_name: str,
        factor_version: str,
        as_of_date: date | str,
    ) -> FactorPanel:
        """Read one stored panel back, validating file and metadata first."""
        return self._panel_at(self._key(factor_name, factor_version, as_of_date))

    def read_range(
        self,
        factor_name: str,
        factor_version: str,
        start_date: date | str,
        end_date: date | str,
    ) -> tuple[FactorPanel, ...]:
        """Read every stored panel of one factor/version inside a closed range.

        The directory tree is only used to *find* candidate files.  Each one is
        rebuilt through :meth:`_panel_at` -- the same path ``read`` takes -- so
        its metadata, table and contract checks decide what the panel really is.
        Anything under this factor/version that looks like a panel but cannot be
        one fails the call instead of being skipped.
        """
        name = _checked_name(factor_name)
        version = _checked_version(factor_version)
        start = _checked_as_of_date(start_date, label="start_date")
        end = _checked_as_of_date(end_date, label="end_date")
        if start > end:
            raise FactorStoreError(
                f"start_date {start.isoformat()} is after end_date {end.isoformat()}: a range "
                "must run forward in time, so the end can never precede the start"
            )

        root = self._root_directory()
        version_dir = root / f"factor_name={name}" / f"factor_version={_version_segment(version)}"
        if not _path_inside(root, version_dir):
            raise FactorStoreError(
                f"factor {name!r} version {version!r} would resolve outside the factor store "
                f"root {str(root)!r}: refusing to scan {str(version_dir)!r}"
            )

        panels: list[FactorPanel] = []
        for key in _scanned_keys(
            version_dir,
            factor_name=name,
            factor_version=version,
            start_date=start,
            end_date=end,
        ):
            if not _path_inside(root, key.target):
                raise FactorStoreError(
                    f"the stored panel {str(key.target)!r} resolves outside the factor store "
                    f"root {str(root)!r}: refusing to read it"
                )
            panels.append(self._panel_at(key))
        return tuple(panels)

    def exists(
        self,
        factor_name: str,
        factor_version: str,
        as_of_date: date | str,
    ) -> bool:
        """Whether a panel is stored for one key; never raises for a missing key."""
        target = self._key(factor_name, factor_version, as_of_date).target
        return target.is_file()

    def path_for(
        self,
        factor_name: str,
        factor_version: str,
        as_of_date: date | str,
    ) -> Path:
        """Where this key's panel is (or would be): the same file the other methods use.

        The single key of ``write``/``read``/``exists`` is validated and encoded
        here, so the answer is that file's path whatever the version contains.
        The path is computed, never checked: a key that was never stored still
        has one, and nothing on disk is touched.
        """
        return self._key(factor_name, factor_version, as_of_date).target

    def _panel_at(self, key: _Key) -> FactorPanel:
        """Rebuild and validate the one panel stored at ``key``'s file.

        ``read`` and ``read_range`` both end here, so a range read applies
        exactly the metadata, table and contract checks a single read applies.
        """
        target = key.target
        if not target.is_file():
            label = _key_label(key.factor_name, key.factor_version, key.as_of_date)
            raise FactorStoreError(
                f"no stored factor panel for {label}: {str(target)!r} does not exist"
            )

        table = _read_table(target)
        metadata = _metadata_from(table, target)
        _check_metadata_identity(metadata, key, target)
        frame = _frame_from(table, target)
        _check_table_identity(frame, metadata, target)

        try:
            return FactorPanel(
                frame,
                universe=metadata.universe,
                as_of_date=metadata.as_of_date,
            )
        except FactorContractError as exc:
            raise FactorStoreError(
                f"the stored panel {str(target)!r} does not restore as a FactorPanel: {exc}"
            ) from exc

    def _root_directory(self) -> Path:
        """The store root, refusing one that has become a file since construction."""
        root = self._root
        if root.exists() and not root.is_dir():
            raise FactorStoreError(
                f"factor store root {str(root)!r} is not a directory: it cannot hold stored "
                "factor panels"
            )
        return root

    def _key(
        self,
        factor_name: str,
        factor_version: str,
        as_of_date: date | str,
    ) -> _Key:
        """Turn a request into its one file, validated to stay inside the root."""
        name = _checked_name(factor_name)
        version = _checked_version(factor_version)
        day = _checked_as_of_date(as_of_date)
        root = self._root_directory()
        target = (
            root
            / f"factor_name={name}"
            / f"factor_version={_version_segment(version)}"
            / f"year={day.year:04d}"
            / f"as_of_date={day.isoformat()}{PANEL_SUFFIX}"
        )
        if not _path_inside(self._root, target):
            raise FactorStoreError(
                f"the key {_key_label(name, version, day)} would resolve outside the factor "
                f"store root {str(self._root)!r}: refusing to touch {str(target)!r}"
            )
        return _Key(
            target=target,
            factor_name=name,
            factor_version=version,
            as_of_date=day,
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(root={self._root!r})"


# ---------------------------------------------------------------------------
# Writing one panel
# ---------------------------------------------------------------------------
def _metadata_key(name: str) -> str:
    """The schema-metadata key of one documented entry."""
    return f"{METADATA_PREFIX}{name}"


def _panel_metadata(panel: FactorPanel) -> dict[str, str]:
    """Everything a reader needs that the six columns cannot carry."""
    return {
        _metadata_key("format_version"): FORMAT_VERSION,
        _metadata_key("factor_name"): panel.factor_name,
        _metadata_key("factor_version"): panel.factor_version,
        _metadata_key("as_of_date"): panel.as_of_date.isoformat(),
        _metadata_key("universe"): json.dumps(list(panel.universe), ensure_ascii=False),
        _metadata_key("stored_at_utc"): datetime.now(UTC).isoformat(timespec="seconds"),
    }


def _panel_table(panel: FactorPanel) -> pa.Table:
    """The Arrow table of one panel: the panel columns plus schema metadata.

    ``preserve_index=False`` keeps the pandas index out of the file, and the
    ``pandas`` metadata PyArrow adds is kept so the dtypes (``datetime64[s]``,
    ``float64``, ``bool``) survive the round trip.
    """
    frame = panel.frame.loc[:, list(FACTOR_PANEL_COLUMNS)]
    table = pa.Table.from_pandas(frame, preserve_index=False)
    # pandas 3 stores text as ``large_string``; the plain ``string`` type keeps the
    # file readable by every Arrow consumer and lets the Hive partition columns
    # (same names, ``string`` type) merge when the tree is opened as a dataset.
    for column in ("symbol", "factor_name", "factor_version"):
        index = table.schema.get_field_index(column)
        if index >= 0 and pa.types.is_large_string(table.schema.field(index).type):
            table = table.set_column(index, column, table.column(column).cast(pa.string()))
    metadata = dict(table.schema.metadata or {})
    metadata.update({key.encode("utf-8"): value for key, value in _panel_metadata(panel).items()})
    return table.replace_schema_metadata(metadata)


def _remove_quietly(path: Path) -> None:
    """Delete a temporary file, ignoring the fact that it may already be gone."""
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass  # a leftover temp file is never worth masking the real failure


def _atomic_write(table: pa.Table, target: Path) -> None:
    """Write ``table`` to ``target`` through a temporary file and one replace.

    The temporary file lives in ``target``'s own directory, so
    :func:`os.replace` is a rename within one filesystem and therefore atomic:
    a concurrent reader sees either the previous file or the complete new one.
    Any failure removes the temporary file and re-raises as
    :class:`~factor_sys.errors.FactorStoreError`; the live file is untouched
    because it is only ever the *source* of nothing and the *target* of the
    final rename.
    """
    parent = target.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise FactorStoreError(
            f"cannot create the factor store directory {str(parent)!r}: {exc}"
        ) from exc

    try:
        handle, temp_name = tempfile.mkstemp(
            dir=str(parent), prefix=f".{target.name}.", suffix=".tmp"
        )
        os.close(handle)
    except OSError as exc:
        raise FactorStoreError(
            f"cannot create a temporary file next to {str(target)!r}: {exc}"
        ) from exc

    temp = Path(temp_name)
    try:
        pq.write_table(table, str(temp), compression="snappy")
        # Confirm the temporary file is a complete, readable Parquet file before
        # it is allowed to replace anything.
        if not temp.is_file():
            raise FactorStoreError(f"the temporary parquet file {str(temp)!r} was not created")
        if temp.stat().st_size == 0:
            raise FactorStoreError(f"the temporary parquet file {str(temp)!r} is empty")
        pq.read_schema(str(temp))
        os.replace(str(temp), str(target))
    except Exception as exc:  # noqa: BLE001 - keep the old file, then normalize the error
        _remove_quietly(temp)
        if isinstance(exc, FactorStoreError):
            raise
        raise FactorStoreError(
            f"could not write the factor panel to {str(target)!r}: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Reading one panel
# ---------------------------------------------------------------------------
def _read_table(path: Path) -> pa.Table:
    """Read one stored Parquet file, or explain why it cannot be read."""
    try:
        return pq.read_table(str(path))
    except Exception as exc:  # noqa: BLE001 - normalize every pyarrow/OS failure
        raise FactorStoreError(
            f"the stored factor panel {str(path)!r} cannot be read as parquet: {exc}"
        ) from exc


def _metadata_text(raw: dict[bytes, bytes], entry: str, path: Path) -> str:
    """One decoded, non-blank metadata value, or raise."""
    value = raw[f"{METADATA_PREFIX}{entry}".encode()]
    try:
        text = value.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise FactorStoreError(
            f"{str(path)!r} has a metadata entry {entry!r} that is not UTF-8: {exc}"
        ) from exc
    if not text.strip():
        raise FactorStoreError(
            f"{str(path)!r} has a blank metadata entry {entry!r}: a stored panel must "
            "describe itself completely"
        )
    return text


def _stored_date(text: str, path: Path) -> date:
    """The ``as_of_date`` recorded in a file, as a plain day."""
    if _ISO_DAY_RE.fullmatch(text.strip()):
        try:
            return date.fromisoformat(text.strip())
        except ValueError:
            pass
    raise FactorStoreError(
        f"{str(path)!r} records as_of_date {text!r}: expected a real ISO 'YYYY-MM-DD' day"
    )


def _stored_universe(text: str, path: Path) -> tuple[str, ...]:
    """The universe recorded in a file, in its original order."""
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise FactorStoreError(
            f"{str(path)!r} records a universe that is not valid JSON: {exc}"
        ) from exc
    if not isinstance(value, list) or not value:
        raise FactorStoreError(
            f"{str(path)!r} records a universe that is not a non-empty array: {value!r}"
        )
    symbols: list[str] = []
    for symbol in value:
        if not isinstance(symbol, str) or not symbol.strip():
            raise FactorStoreError(
                f"{str(path)!r} records a universe entry {symbol!r} that is not a symbol"
            )
        symbols.append(symbol)
    if len(set(symbols)) != len(symbols):
        raise FactorStoreError(
            f"{str(path)!r} records a universe with repeated symbols: {symbols!r}"
        )
    return tuple(symbols)


def _metadata_from(table: pa.Table, path: Path) -> _StoredMetadata:
    """Decode and validate the schema metadata of one stored panel."""
    raw = table.schema.metadata
    if not raw:
        raise FactorStoreError(
            f"{str(path)!r} carries no parquet schema metadata: it is not a stored factor "
            "panel (or it was written without its metadata)"
        )

    missing = [entry for entry in METADATA_KEYS if f"{METADATA_PREFIX}{entry}".encode() not in raw]
    if missing:
        raise FactorStoreError(
            f"{str(path)!r} is missing the metadata entries {missing}; a stored panel must "
            f"carry {list(METADATA_KEYS)}"
        )

    format_version = _metadata_text(raw, "format_version", path)
    if format_version not in SUPPORTED_FORMAT_VERSIONS:
        raise FactorStoreError(
            f"{str(path)!r} was written in factor-store format {format_version!r}; this "
            f"build supports {list(SUPPORTED_FORMAT_VERSIONS)}"
        )

    return _StoredMetadata(
        format_version=format_version,
        factor_name=_metadata_text(raw, "factor_name", path),
        factor_version=_metadata_text(raw, "factor_version", path),
        as_of_date=_stored_date(_metadata_text(raw, "as_of_date", path), path),
        universe=_stored_universe(_metadata_text(raw, "universe", path), path),
        stored_at_utc=_metadata_text(raw, "stored_at_utc", path),
    )


def _check_metadata_identity(metadata: _StoredMetadata, key: _Key, path: Path) -> None:
    """Refuse a file whose metadata describes a different key than the request."""
    recorded = (
        ("factor_name", metadata.factor_name, key.factor_name),
        ("factor_version", metadata.factor_version, key.factor_version),
        ("as_of_date", metadata.as_of_date.isoformat(), key.as_of_date.isoformat()),
    )
    mismatches = [
        f"it records {label} {actual!r} but {wanted!r} was requested"
        for label, actual, wanted in recorded
        if actual != wanted
    ]
    if mismatches:
        raise FactorStoreError(
            f"the stored panel {str(path)!r} belongs to another key: " + "; ".join(mismatches)
        )


def _frame_from(table: pa.Table, path: Path) -> pd.DataFrame:
    """The table as a pandas frame holding exactly the panel columns."""
    try:
        frame = table.to_pandas()
    except Exception as exc:  # noqa: BLE001 - any Arrow/pandas failure means "corrupt"
        raise FactorStoreError(
            f"the stored panel {str(path)!r} cannot be converted to a pandas frame: {exc}"
        ) from exc

    if list(frame.columns) != list(FACTOR_PANEL_COLUMNS):
        raise FactorStoreError(
            f"the stored panel {str(path)!r} does not hold the panel columns: expected "
            f"{list(FACTOR_PANEL_COLUMNS)}, got {[str(column) for column in frame.columns]}"
        )
    return frame.reset_index(drop=True)


def _check_table_identity(frame: pd.DataFrame, metadata: _StoredMetadata, path: Path) -> None:
    """Refuse a table whose own values contradict the file's metadata."""
    if len(frame) != len(metadata.universe):
        raise FactorStoreError(
            f"the stored panel {str(path)!r} holds {len(frame)} row(s) but its metadata "
            f"lists {len(metadata.universe)} universe symbol(s)"
        )

    try:
        dates = pd.to_datetime(frame["as_of_date"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise FactorStoreError(
            f"the stored panel {str(path)!r} has an as_of_date column that is not a date: {exc}"
        ) from exc
    if bool(pd.isna(dates).any()):
        raise FactorStoreError(
            f"the stored panel {str(path)!r} has missing as_of_date value(s) in its table"
        )
    distinct = {pd.Timestamp(value).date() for value in dates}
    if distinct != {metadata.as_of_date}:
        rendered = sorted(day.isoformat() for day in distinct)
        raise FactorStoreError(
            f"the stored panel {str(path)!r} is dated {rendered} in its table but "
            f"{metadata.as_of_date.isoformat()} in its metadata"
        )

    for column, expected in (
        ("factor_name", metadata.factor_name),
        ("factor_version", metadata.factor_version),
    ):
        values = {str(value) for value in frame[column].unique()}
        if values != {expected}:
            raise FactorStoreError(
                f"the stored panel {str(path)!r} carries {column} {sorted(values)} in its "
                f"table but {expected!r} in its metadata"
            )


# ---------------------------------------------------------------------------
# Finding the files of a range
# ---------------------------------------------------------------------------
_YEAR_DIR_RE = re.compile(r"year=[0-9]{4}")
_PANEL_PREFIX = "as_of_date="


def _entries(directory: Path) -> list[Path]:
    """The entries of one store directory, sorted by name; unreadable is an error."""
    try:
        return sorted(directory.iterdir(), key=lambda item: item.name)
    except OSError as exc:
        raise FactorStoreError(
            f"cannot list the factor store directory {str(directory)!r}: {exc}"
        ) from exc


def _year_of(path: Path) -> int:
    """The year a ``year=YYYY`` directory claims, or raise."""
    text = path.name.removeprefix("year=")
    if _YEAR_DIR_RE.fullmatch(path.name) and 1 <= int(text) <= 9999:
        return int(text)
    raise FactorStoreError(
        f"{str(path)!r} is not a factor-store year partition: expected a directory named "
        "'year=YYYY'"
    )


def _panel_day(path: Path) -> date:
    """The day a stored panel's file name claims, or raise.

    The name is only a claim: :meth:`ParquetFactorStore._panel_at` still checks
    it against the file's metadata.  A file that *looks* like a panel but carries
    no readable day is an error rather than something to skip -- a silently
    ignored file would make a partial history look complete.
    """
    stem = path.name.removesuffix(PANEL_SUFFIX)
    day_text = stem.removeprefix(_PANEL_PREFIX)
    if day_text != stem and _ISO_DAY_RE.fullmatch(day_text):
        try:
            return date.fromisoformat(day_text)
        except ValueError:
            pass  # right shape, but no such day (e.g. '2024-02-30')
    raise FactorStoreError(
        f"{str(path)!r} is not a stored factor panel file: expected a name of the form "
        f"'{_PANEL_PREFIX}YYYY-MM-DD{PANEL_SUFFIX}' naming a real day"
    )


def _scanned_keys(
    version_dir: Path,
    *,
    factor_name: str,
    factor_version: str,
    start_date: date,
    end_date: date,
) -> tuple[_Key, ...]:
    """The stored files of one factor/version inside the closed range, in date order.

    The whole ``factor_version=`` directory is scanned, not just the years the
    range touches: a misnamed file or a wrong year partition must not be able to
    hide outside the window and surface as a surprise later.  Only the files
    inside the range are read afterwards, so a panel damaged long ago does not
    block a newer window; every candidate that *is* returned is read and
    validated, so a name or a year partition never decides a panel's identity on
    its own.
    """
    for directory in (version_dir.parent, version_dir):
        if directory.exists() and not directory.is_dir():
            raise FactorStoreError(
                f"{str(directory)!r} is not a directory: the store tree of factor "
                f"{factor_name!r} version {factor_version!r} is damaged"
            )
    if not version_dir.exists():
        return ()

    keys: list[_Key] = []
    for partition in _entries(version_dir):
        if not partition.is_dir():
            if partition.name.endswith(PANEL_SUFFIX):
                raise FactorStoreError(
                    f"{str(partition)!r} is a stored panel file outside a 'year=YYYY' "
                    "partition: the factor store tree is damaged"
                )
            continue  # a temporary file from an interrupted write is not a panel
        year = _year_of(partition)
        for file in _entries(partition):
            if file.is_dir():
                raise FactorStoreError(
                    f"{str(file)!r} is a directory where a stored panel file belongs: expected "
                    f"'{_PANEL_PREFIX}YYYY-MM-DD{PANEL_SUFFIX}'"
                )
            if not file.name.endswith(PANEL_SUFFIX):
                continue  # a temporary file from an interrupted write is not a panel
            day = _panel_day(file)
            if day.year != year:
                raise FactorStoreError(
                    f"{str(file)!r} sits in the year={year:04d} partition but is dated "
                    f"{day.isoformat()}: the partition and the panel disagree"
                )
            if start_date <= day <= end_date:
                keys.append(
                    _Key(
                        target=file,
                        factor_name=factor_name,
                        factor_version=factor_version,
                        as_of_date=day,
                    )
                )

    keys.sort(key=lambda key: key.as_of_date)
    return tuple(keys)


__all__ = ["FactorStore", "ParquetFactorStore"]
