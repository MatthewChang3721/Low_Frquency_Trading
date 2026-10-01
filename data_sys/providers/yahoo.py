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
* *Vendor ticker spelling*: Yahoo writes dot-separated share classes with a
  dash (``BRK-B``), while this project's canonical symbols keep the exchange
  spelling (``BRK.B``, see ``config/universe_seed.csv``).  :meth:`fetch_bars`
  therefore requests the canonical spelling first and falls back to the dashed
  variant only when the vendor returns nothing usable, so a guess can never
  shadow a valid ticker (``RY.TO``-style exchange suffixes are fetched as-is).
  The substitution is *request-only*: raw files, standardized frames and
  published partitions keep ``BRK.B``, and the sidecar's ``vendor_symbol``
  records the spelling that actually produced the data.
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


def _vendor_ticker_candidates(symbol: str) -> tuple[str, ...]:
    """Vendor tickers to ask for, in order, starting with the canonical symbol.

    The canonical spelling always comes first: the dashed variant is a
    *fallback* for share classes the vendor spells differently (``BRK.B`` ->
    ``BRK-B``), never an override, so a wrong guess cannot shadow a ticker that
    the vendor does answer to (``RY.TO``, ``VOD.L``, ...).
    """
    candidates = [symbol]
    if "." in symbol:
        dashed = symbol.replace(".", "-")
        if dashed != symbol and re.match(SYMBOL_PATTERN, dashed):
            candidates.append(dashed)
    return tuple(candidates)


def _frame_problem(frame: pd.DataFrame | None) -> str | None:
    """Describe why a vendor frame cannot be used, or return ``None`` if it can."""
    if frame is None or frame.empty:
        return "no rows"
    missing = [column for column in RAW_REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        return f"missing columns {missing}; got {list(frame.columns)}"
    return None


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
        """Download raw daily bars for the canonical ``symbol`` over ``[start, end]``.

        ``symbol`` stays canonical everywhere outside the vendor request (see the
        module docstring); the returned ``meta`` records both the symbol and the
        ``vendor_symbol`` that actually answered.
        """
        symbol = str(symbol).strip().upper()
        if not symbol or not re.match(SYMBOL_PATTERN, symbol):
            raise DownloadError(f"invalid symbol: {symbol!r}")
        if end < start:
            raise DownloadError(f"end ({end}) must not be before start ({start})")

        frame, vendor_symbol = self._fetch_first_usable(symbol, start, end)

        meta: dict[str, Any] = {
            "provider": self.name,
            "symbol": symbol,
            "vendor_symbol": vendor_symbol,
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

    def _fetch_first_usable(
        self, symbol: str, start: date, end: date
    ) -> tuple[pd.DataFrame, str]:
        """Return ``(flattened frame, vendor symbol)`` of the first spelling that works.

        The canonical ``symbol`` is requested first; the dashed share-class
        spelling is only tried when that returns nothing usable (see the module
        docstring).  Every rejection is logged, and reported in the error when
        no spelling works, so a failure never hides which tickers were asked for.

        Raises
        ------
        DownloadError
            If no spelling returned a frame with the required raw columns.
        """
        candidates = _vendor_ticker_candidates(symbol)
        end_exclusive = end + timedelta(days=1)
        attempts: list[str] = []
        cause: Exception | None = None

        for index, ticker in enumerate(candidates):
            try:
                frame = self._retryer(_download_once, ticker, start, end_exclusive)
            except Exception as exc:  # noqa: BLE001 - normalize vendor errors
                cause = exc
                attempts.append(f"{ticker}: {exc}")
                continue

            frame = self._flatten_columns(frame)
            problem = _frame_problem(frame)
            if problem is None:
                if ticker != symbol:
                    logger.warning(
                        "yahoo_finance: %s resolved to the vendor ticker %s", symbol, ticker
                    )
                return frame, ticker

            attempts.append(f"{ticker}: {problem}")
            if index + 1 < len(candidates):
                logger.warning(
                    "yahoo_finance: %s returned nothing usable (%s); trying %s",
                    ticker,
                    problem,
                    candidates[index + 1],
                )

        raise DownloadError(
            f"Yahoo Finance returned no usable bars for {symbol} "
            f"({start.isoformat()}..{end.isoformat()}); tried "
            f"{', '.join(candidates)}: {'; '.join(attempts)}"
        ) from cause


__all__ = ["YahooFinanceProvider"]
