"""Yahoo Finance provider (``yfinance`` based).

Design notes
------------
* ``auto_adjust=False`` is mandatory: it returns *unadjusted* OHLC **and** a
  separate ``Adj Close`` column.  The default (``auto_adjust=True``) would bake
  the adjustment into OHLC and drop ``Adj Close``, which conflicts with the
  requirement to keep raw OHLC plus an independent adjusted close.
* ``multi_level_index=False`` yields flat columns (``['Adj Close', 'Close', ...]``)
  for a single ticker instead of ``('Close', 'AAPL')`` tuples.
* ``yfinance``'s ``end`` is *exclusive*; :func:`fetch_bars` adds one day so the
  requested range is inclusive at both ends.
* Transient failures (network hiccups) are retried with ``tenacity``.  The
  retry predicate is intentionally broad because ``yfinance`` raises a
  heterogeneous set of exception types for what are the same transient issues.
"""

from __future__ import annotations

import logging
import re
from datetime import date, timedelta
from typing import Any

import pandas as pd
import yfinance as yf
from tenacity import (
    Retrying,
    before_sleep_log,
    retry_if_exception_type,
    stop_after_attempt,
    wait_fixed,
)

from data_sys.errors import DownloadError
from data_sys.providers.base import RawBars
from data_sys.schema import RAW_REQUIRED_COLUMNS, SYMBOL_PATTERN
from data_sys.utils import utc_now_iso

logger = logging.getLogger(__name__)


def _download_once(symbol: str, start: date, end_exclusive: date) -> pd.DataFrame:
    """Single, un-retried ``yfinance`` call (kept module level for clarity)."""
    return yf.download(
        symbol,
        start=start.isoformat(),
        end=end_exclusive.isoformat(),
        auto_adjust=False,       # keep unadjusted OHLC + independent Adj Close
        progress=False,
        multi_level_index=False,  # flat columns for a single ticker
        actions=False,
    )


class YahooFinanceProvider:
    """``DataProvider`` implementation backed by Yahoo Finance."""

    name = "yahoo_finance"

    def __init__(self, max_attempts: int = 3, wait_seconds: float = 1.0) -> None:
        self.max_attempts = max(1, int(max_attempts))
        self.wait_seconds = float(wait_seconds)
        self._retryer = Retrying(
            stop=stop_after_attempt(self.max_attempts),
            wait=wait_fixed(self.wait_seconds),
            retry=retry_if_exception_type(Exception),
            reraise=True,
            before_sleep=before_sleep_log(logger, logging.WARNING),
        )

    # -- public API ----------------------------------------------------------
    def fetch_bars(self, symbol: str, start: date, end: date) -> RawBars:
        """Download raw daily bars for ``symbol`` over ``[start, end]``."""
        symbol = str(symbol).strip().upper()
        if not symbol or not re.match(SYMBOL_PATTERN, symbol):
            raise DownloadError(f"invalid symbol: {symbol!r}")
        if end < start:
            raise DownloadError(f"end ({end}) must not be before start ({start})")

        end_exclusive = end + timedelta(days=1)
        try:
            frame = self._retryer(_download_once, symbol, start, end_exclusive)
        except DownloadError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalize vendor errors
            raise DownloadError(
                f"Yahoo Finance download failed for {symbol} "
                f"({start.isoformat()}..{end.isoformat()}): {exc}"
            ) from exc

        frame = self._flatten_columns(frame)
        self._assert_usable(frame, symbol, start, end)

        meta: dict[str, Any] = {
            "provider": self.name,
            "symbol": symbol,
            "requested_start": start.isoformat(),
            "requested_end": end.isoformat(),
            "fetched_at_utc": utc_now_iso(),
            "raw_rows": int(len(frame)),
            "raw_columns": [str(c) for c in frame.columns],
        }
        return RawBars(frame=frame, meta=meta)

    # -- helpers -------------------------------------------------------------
    @staticmethod
    def _flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
        """Collapse a ``(field, ticker)`` MultiIndex into plain field names."""
        if isinstance(frame.columns, pd.MultiIndex):
            frame = frame.copy()
            frame.columns = [str(col[0]) for col in frame.columns]
        return frame

    @staticmethod
    def _assert_usable(frame: pd.DataFrame, symbol: str, start: date, end: date) -> None:
        if frame is None or frame.empty:
            raise DownloadError(
                f"Yahoo Finance returned no rows for {symbol} "
                f"({start.isoformat()}..{end.isoformat()})"
            )
        missing = [c for c in RAW_REQUIRED_COLUMNS if c not in frame.columns]
        if missing:
            raise DownloadError(
                f"Yahoo Finance response for {symbol} is missing columns: {missing}; "
                f"got {list(frame.columns)}"
            )


__all__ = ["YahooFinanceProvider"]
