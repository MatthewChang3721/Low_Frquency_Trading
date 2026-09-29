"""Map raw provider output onto the standardized daily-bar contract.

Rules enforced here (see :mod:`data_sys.schema`):

* column names are renamed to snake_case,
* ``symbol`` is upper-cased and validated,
* ``date`` is a timezone-naive daily date (exchange trading day),
* prices/``volume`` are cast to ``float64`` / ``int64``,
* ``adjusted_close`` is Yahoo's ``Adj Close`` (kept as one field, not derived),
* ``daily_return`` (first row null) and ``dollar_volume`` are derived.

Null / type problems are *not* silently dropped here: values that cannot be
coerced are left as ``NaN`` so the quality stage can report them precisely.
"""

from __future__ import annotations

import logging
import re

import pandas as pd

from data_sys.errors import StandardizationError
from data_sys.schema import (
    RAW_REQUIRED_COLUMNS,
    STANDARD_COLUMNS,
    SYMBOL_PATTERN,
    YAHOO_COLUMN_RENAME,
)

logger = logging.getLogger(__name__)

PRICE_COLUMNS: list[str] = ["open", "high", "low", "close", "adjusted_close"]
REQUIRED_STANDARD_COLUMNS: list[str] = ["date", *PRICE_COLUMNS, "volume"]


def _ensure_date_column(raw: pd.DataFrame) -> pd.DataFrame:
    """Return ``raw`` with the trading date available as a ``date`` column."""
    frame = raw.copy()
    if isinstance(frame.index, pd.DatetimeIndex) and "Date" not in frame.columns:
        if frame.index.name is None:
            frame.index.name = "Date"
        frame = frame.reset_index()
    return frame


def standardize_bars(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Transform a raw Yahoo-style frame into the standardized daily-bar frame."""
    if raw is None or len(raw) == 0:
        raise StandardizationError(f"cannot standardize an empty raw frame for {symbol!r}")

    sym = str(symbol).strip().upper()
    if not sym or not re.match(SYMBOL_PATTERN, sym):
        raise StandardizationError(f"invalid symbol: {symbol!r}")

    missing_raw = [c for c in RAW_REQUIRED_COLUMNS if c not in raw.columns]
    if missing_raw:
        raise StandardizationError(
            f"raw frame is missing required columns {missing_raw}; got {list(raw.columns)}"
        )

    frame = _ensure_date_column(raw).rename(columns=YAHOO_COLUMN_RENAME)

    missing = [c for c in REQUIRED_STANDARD_COLUMNS if c not in frame.columns]
    if missing:
        raise StandardizationError(
            f"raw frame is missing required columns {missing}; got {list(frame.columns)}"
        )

    out = frame[REQUIRED_STANDARD_COLUMNS].copy()

    # -- date ---------------------------------------------------------------
    parsed = pd.to_datetime(out["date"], errors="coerce")
    if parsed.isna().any():
        bad = int(parsed.isna().sum())
        raise StandardizationError(f"{bad} row(s) have unparseable dates")
    out["date"] = parsed.dt.normalize().astype("datetime64[s]")

    # -- symbol -------------------------------------------------------------
    out["symbol"] = sym

    # -- numeric types ------------------------------------------------------
    for column in PRICE_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="coerce").astype("float64")

    volume = pd.to_numeric(out["volume"], errors="coerce")
    if volume.isna().any():
        # keep NaN as float so the quality stage reports "volume must not be null"
        out["volume"] = volume.astype("float64")
    else:
        out["volume"] = volume.astype("int64")

    # -- canonical ordering -------------------------------------------------
    out = out.sort_values("date", kind="stable").reset_index(drop=True)

    # -- derived features ---------------------------------------------------
    out["daily_return"] = out["close"].pct_change()
    out["dollar_volume"] = out["close"] * out["volume"]

    return out[STANDARD_COLUMNS]


__all__ = ["standardize_bars", "PRICE_COLUMNS", "REQUIRED_STANDARD_COLUMNS"]
