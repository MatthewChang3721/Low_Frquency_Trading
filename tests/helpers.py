"""Deterministic test-data builders shared by the data-system test-suite.

The generated frames deliberately satisfy every quality rule (positive OHLC,
``high >= max(open, close, low)``, ``low <= min(open, close, high)``, unique
``(symbol, date)``) so that a failing test always points at the code under test
rather than at the fixture.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from data_sys.providers.base import RawBars
from data_sys.standardize import standardize_bars
from data_sys.utils import utc_now_iso

RAW_COLUMNS: list[str] = ["Adj Close", "Close", "High", "Low", "Open", "Volume"]


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


__all__ = ["make_raw_frame", "make_standardized_frame", "FakeProvider", "RAW_COLUMNS"]
