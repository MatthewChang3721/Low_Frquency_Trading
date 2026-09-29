"""Provider abstraction: the only boundary the rest of the pipeline depends on.

Following the architecture document ("stable objects + replaceable
implementations"), the standardization/storage/quality layers depend on the
:class:`DataProvider` protocol and the :class:`RawBars` container, never on
``yfinance`` directly.  Swapping in a paid vendor later means adding a new
provider, not touching the rest of the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Protocol, runtime_checkable

import pandas as pd


@dataclass
class RawBars:
    """Raw, untouched provider output plus fetch metadata.

    Attributes
    ----------
    frame:
        Exactly what the provider returned (raw column names preserved).  The
        pipeline writes this to the ``raw/`` layer before any transformation.
    meta:
        JSON-serializable fetch metadata (provider, symbol, range, timestamp).
    """

    frame: pd.DataFrame
    meta: dict[str, Any]


@runtime_checkable
class DataProvider(Protocol):
    """Protocol every market-data provider must implement."""

    name: str

    def fetch_bars(self, symbol: str, start: date, end: date) -> RawBars:
        """Return raw daily bars for ``symbol`` over ``[start, end]`` inclusive.

        Raises
        ------
        data_sys.errors.DownloadError
            If the provider cannot return usable data.
        """
        ...  # pragma: no cover - protocol definition


__all__ = ["RawBars", "DataProvider"]
