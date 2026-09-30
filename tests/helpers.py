"""Deterministic test-data builders shared by the data-system test-suite.

The generated frames deliberately satisfy every quality rule (positive OHLC,
``high >= max(open, close, low)``, ``low <= min(open, close, high)``, unique
``(symbol, date)``) so that a failing test always points at the code under test
rather than at the fixture.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from data_sys.errors import DownloadError
from data_sys.providers.base import RawBars
from data_sys.standardize import standardize_bars
from data_sys.universe import MIN_UNIVERSE_SIZE, REQUIRED_COLUMNS
from data_sys.utils import parse_date, utc_now_iso

RAW_COLUMNS: list[str] = ["Adj Close", "Close", "High", "Low", "Open", "Volume"]

#: Columns the published parquet files never leave null.  Mirrors
#: ``data_sys.duckdb_verify.REQUIRED_NON_NULL_COLUMNS``; the published dataset is
#: asserted through PyArrow, so this stays an independent expectation.
NEVER_NULL_COLUMNS: tuple[str, ...] = (
    "open",
    "high",
    "low",
    "close",
    "adjusted_close",
    "volume",
    "dollar_volume",
)


def make_raw_frame(periods: int = 250, start: str = "2020-01-02") -> pd.DataFrame:
    """Build a raw, Yahoo-shaped frame (unadjusted OHLC + separate ``Adj Close``)."""
    rng = np.random.default_rng(20200102)

    index = pd.bdate_range(start=start, periods=periods, name="Date").astype("datetime64[s]")

    close = np.round(100.0 + np.cumsum(rng.normal(0.0, 1.0, periods)), 2)
    close = np.maximum(close, 10.0)

    open_ = np.round(close * (1.0 + rng.normal(0.0, 0.002, periods)), 2)

    high = np.round(
        np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, 0.003, periods))), 2
    )
    low = np.round(
        np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, 0.003, periods))), 2
    )
    # enforce the OHLC invariant exactly (rounding can nibble at the edges)
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))

    volume = rng.integers(1_000_000, 5_000_000, periods).astype("int64")
    adjusted_close = np.round(close * 0.97, 2)

    return pd.DataFrame(
        {
            "Adj Close": adjusted_close,
            "Close": close,
            "High": high,
            "Low": low,
            "Open": open_,
            "Volume": volume,
        },
        index=index,
    )


def make_standardized_frame(symbol: str = "AAPL", **kwargs) -> pd.DataFrame:
    """Build a standardized frame by running :func:`standardize_bars`."""
    return standardize_bars(make_raw_frame(**kwargs), symbol)


class FakeProvider:
    """In-memory ``DataProvider`` used to exercise the orchestrator."""

    name = "fake"

    def __init__(
        self,
        frame: pd.DataFrame | None = None,
        error: Exception | None = None,
    ) -> None:
        self.frame = frame
        self.error = error
        self.calls: list[tuple[str, date, date]] = []

    def fetch_bars(self, symbol: str, start: date, end: date) -> RawBars:
        self.calls.append((symbol, start, end))
        if self.error is not None:
            raise self.error

        frame = self.frame.copy() if self.frame is not None else make_raw_frame()
        meta = {
            "provider": self.name,
            "symbol": symbol,
            "requested_start": start.isoformat(),
            "requested_end": end.isoformat(),
            "fetched_at_utc": utc_now_iso(),
            "raw_rows": int(len(frame)),
            "raw_columns": [str(column) for column in frame.columns],
        }
        return RawBars(frame=frame, meta=meta)


# ---------------------------------------------------------------------------
# Session-aligned fixtures for the incremental-update tests
# ---------------------------------------------------------------------------
# Sessions are used instead of ``pd.bdate_range`` so that generated data lines up
# with the NYSE calendar: ``bdate_range`` includes exchange holidays, which the
# gap detector would (correctly) refuse to treat as trading days.
def session_dates(start: str | date, end: str | date, calendar_name: str = "XNYS") -> list[date]:
    """The NYSE sessions in ``[start, end]`` as plain dates."""
    from data_sys.market_calendar import get_calendar, sessions_in_range

    return sessions_in_range(get_calendar(calendar_name), parse_date(start), parse_date(end))


def make_session_prices(
    dates: Sequence[date] | pd.DatetimeIndex,
    *,
    seed: int = 20200102,
    price_scale: float = 1.0,
    close_override: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Deterministic raw OHLCV on exactly ``dates``; every quality rule holds.

    The frame is indexed by ``dates`` (named ``Date``, ``datetime64[s]``) *before*
    any override is applied, so ``close_override`` addresses a bar by its ISO date:
    it maps such a date to a close price, which lets a test inject one bad bar
    (e.g. a negative close) without corrupting the rest.  An override date that is
    not one of ``dates`` is an error, never a silent no-op.
    """
    index = pd.DatetimeIndex(list(dates), name="Date").astype("datetime64[s]")
    n_sessions = len(index)
    if n_sessions == 0:
        raise ValueError("make_session_prices needs at least one session")

    rng = np.random.default_rng(seed)

    close = np.round((100.0 + np.cumsum(rng.normal(0.0, 1.0, n_sessions))) * price_scale, 2)
    close = np.maximum(close, 10.0)
    open_ = np.round(close * (1.0 + rng.normal(0.0, 0.002, n_sessions)), 2)
    high = np.round(
        np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, 0.003, n_sessions))), 2
    )
    low = np.round(
        np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, 0.003, n_sessions))), 2
    )
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))

    frame = pd.DataFrame(
        {
            "Adj Close": np.round(close * 0.97, 2),
            "Close": close,
            "High": high,
            "Low": low,
            "Open": open_,
            "Volume": rng.integers(1_000_000, 5_000_000, n_sessions).astype("int64"),
        },
        index=index,
    )
    close_column = frame.columns.get_loc("Close")
    for iso_date, value in (close_override or {}).items():
        position = index.get_indexer([pd.Timestamp(parse_date(iso_date))])[0]
        if position < 0:
            raise ValueError(f"close_override date {iso_date} is not one of the sessions")
        frame.iloc[position, close_column] = value
    return frame


class SessionFakeProvider:
    """A provider whose data exists exactly on the sessions it is asked for.

    Unlike :class:`FakeProvider` (which ignores the requested range), this one
    *slices* a master frame, so a test can assert precisely which windows an
    incremental update decided to download.  Asking for a window with no data
    raises ``DownloadError``, mirroring the real provider.
    """

    name = "session_fake"

    def __init__(
        self,
        frame: pd.DataFrame | None = None,
        *,
        error: Exception | None = None,
        errors_by_start: dict[date, Exception] | None = None,
        allow_empty: bool = False,
        start: str = "2020-01-02",
        end: str = "2021-12-31",
        seed: int = 20200102,
        price_scale: float = 1.0,
        close_override: dict[str, float] | None = None,
    ) -> None:
        if frame is None:
            # the master frame is indexed by the sessions themselves, so a
            # ``close_override`` can address a bar by its ISO date
            frame = make_session_prices(
                session_dates(start, end),
                seed=seed,
                price_scale=price_scale,
                close_override=close_override,
            )
        self.frame = frame
        self.error = error
        self.errors_by_start = dict(errors_by_start or {})
        self.allow_empty = allow_empty
        self.calls: list[tuple[str, date, date]] = []

    def fetch_bars(self, symbol: str, start: date, end: date) -> RawBars:
        """Return the master rows inside ``[start, end]`` (or raise)."""
        self.calls.append((symbol, start, end))
        if self.error is not None:
            raise self.error
        if start in self.errors_by_start:
            raise self.errors_by_start[start]

        index = pd.to_datetime(self.frame.index)
        mask = (index >= pd.Timestamp(start)) & (index <= pd.Timestamp(end))
        window = self.frame.loc[mask]
        if window.empty and not self.allow_empty:
            raise DownloadError(
                f"no rows for {symbol} ({start.isoformat()}..{end.isoformat()})"
            )

        meta = {
            "provider": self.name,
            "symbol": symbol,
            "requested_start": start.isoformat(),
            "requested_end": end.isoformat(),
            "fetched_at_utc": utc_now_iso(),
            "raw_rows": int(len(window)),
            "raw_columns": [str(column) for column in window.columns],
        }
        return RawBars(frame=window.copy(), meta=meta)

    def windows(self) -> list[tuple[date, date]]:
        """The ``(start, end)`` of every request this provider received."""
        return [(start, end) for _, start, end in self.calls]


def make_provider_factory(
    *,
    start: str = "2020-01-02",
    end: str = "2021-12-31",
    errors: dict[str, Exception] | None = None,
    price_scale: float = 1.0,
    seed: int = 20200102,
) -> Callable[[str], SessionFakeProvider]:
    """A per-symbol provider factory for batch tests.

    Symbols listed in ``errors`` always fail; every other symbol gets its own
    deterministic price series so the published partitions differ from each
    another.
    """
    failures = dict(errors or {})

    def factory(symbol: str) -> SessionFakeProvider:
        if symbol in failures:
            return SessionFakeProvider(error=failures[symbol])
        return SessionFakeProvider(
            start=start,
            end=end,
            price_scale=price_scale,
            seed=seed + sum(ord(character) for character in symbol),
        )

    return factory


def write_universe_file(
    path: Path,
    symbols: Sequence[str],
    *,
    inactive: Sequence[str] = (),
    exchange: str = "NASDAQ",
) -> Path:
    """Write a universe CSV that satisfies the real contract (>= 10 rows)."""
    rows = [
        {
            "symbol": symbol,
            "name": f"{symbol} Inc.",
            "exchange": exchange,
            "security_type": "common_stock",
            "is_active": "true",
        }
        for symbol in symbols
    ]
    rows.extend(
        {
            "symbol": symbol,
            "name": f"{symbol} Inc.",
            "exchange": exchange,
            "security_type": "common_stock",
            "is_active": "false",
        }
        for symbol in inactive
    )

    present = {row["symbol"] for row in rows}
    for index in range(MIN_UNIVERSE_SIZE):
        if len(rows) >= MIN_UNIVERSE_SIZE:
            break
        filler = f"F{index:02d}"
        if filler in present:
            continue
        present.add(filler)
        rows.append(
            {
                "symbol": filler,
                "name": f"{filler} Inc.",
                "exchange": exchange,
                "security_type": "common_stock",
                "is_active": "false",
            }
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=list(REQUIRED_COLUMNS)).to_csv(path, index=False)
    return path


__all__ = [
    "NEVER_NULL_COLUMNS",
    "RAW_COLUMNS",
    "FakeProvider",
    "SessionFakeProvider",
    "make_provider_factory",
    "make_raw_frame",
    "make_session_prices",
    "make_standardized_frame",
    "session_dates",
    "write_universe_file",
]
