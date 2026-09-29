"""The standardized daily-bar contract.

This module is the single source of truth for:

* the ordered standardized column set,
* the mapping from raw Yahoo Finance column names to standardized names,
* the :mod:`pandera` schema encoding every value and cross-field quality rule.

Downstream modules (factors, strategies, ...) may rely on
:data:`STANDARD_COLUMNS` and :data:`StandardizedBarSchema` being stable.
"""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa

# ---------------------------------------------------------------------------
# Column contracts
# ---------------------------------------------------------------------------
#: Ordered columns of a standardized bar frame (excluding Hive partition keys).
STANDARD_COLUMNS: list[str] = [
    "date",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "adjusted_close",
    "volume",
    "daily_return",
    "dollar_volume",
]

#: Raw columns that must be present after a successful Yahoo download.
RAW_REQUIRED_COLUMNS: list[str] = [
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
]

#: Yahoo Finance column name -> standardized column name.
YAHOO_COLUMN_RENAME: dict[str, str] = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adjusted_close",
    "Volume": "volume",
    "Date": "date",
}

#: Symbol must be upper-case exchange-style ticker characters.
SYMBOL_PATTERN: str = r"^[A-Z0-9.\-]+$"

# ---------------------------------------------------------------------------
# Cross-field check helpers
# ---------------------------------------------------------------------------
def _strictly_increasing_dates(df: pd.DataFrame) -> pd.Series:
    """``True`` where ``date`` is strictly greater than the previous row's date.

    The first row has no predecessor and is therefore never a violation.  NaT
    comparisons would not help here: ``NaT > Timedelta(0)`` evaluates to
    ``False`` (not NaN), so the first row is forced to ``True`` explicitly.  A
    missing or unparseable date is already reported by the column-level
    ``DateTime`` check.
    """
    try:
        differences = pd.to_datetime(df["date"], errors="coerce").diff()
        increasing = differences.gt(pd.Timedelta(0))
    except (TypeError, ValueError):
        return pd.Series(False, index=df.index, dtype=bool)

    if len(increasing) > 0:
        increasing = increasing.copy()
        increasing.iloc[0] = True
    return increasing


# ---------------------------------------------------------------------------
# Pandera schema
# ---------------------------------------------------------------------------
# Cross-field checks return a boolean Series (one entry per row) so that
# ``lazy=True`` validation reports the offending row indices, not just a single
# aggregate failure.
StandardizedBarSchema: pa.DataFrameSchema = pa.DataFrameSchema(
    columns={
        "date": pa.Column(pa.DateTime, nullable=False),
        "symbol": pa.Column(str, pa.Check.str_matches(SYMBOL_PATTERN), nullable=False),
        "open": pa.Column(float, pa.Check.gt(0), nullable=False),
        "high": pa.Column(float, pa.Check.gt(0), nullable=False),
        "low": pa.Column(float, pa.Check.gt(0), nullable=False),
        "close": pa.Column(float, pa.Check.gt(0), nullable=False),
        "adjusted_close": pa.Column(float, pa.Check.gt(0), nullable=False),
        "volume": pa.Column(int, pa.Check.ge(0), nullable=False),
        "daily_return": pa.Column(float, nullable=True),
        "dollar_volume": pa.Column(float, pa.Check.ge(0), nullable=False),
    },
    checks=[
        pa.Check(
            lambda df: ~df.duplicated(subset=["symbol", "date"]),
            error="duplicate (symbol, date) row",
        ),
        pa.Check(
            _strictly_increasing_dates,
            error="date is not strictly increasing",
        ),
        pa.Check(
            lambda df: df["high"] >= df[["open", "close", "low"]].max(axis=1),
            error="high < max(open, close, low)",
        ),
        pa.Check(
            lambda df: df["low"] <= df[["open", "close", "high"]].min(axis=1),
            error="low > min(open, close, high)",
        ),
        pa.Check(
            lambda df: bool(len(df) == 0 or pd.isna(df["daily_return"].iloc[0])),
            error="first daily_return must be null",
        ),
        pa.Check(
            lambda df: (
                len(df) == 0
                or bool(df["daily_return"].iloc[1:].notna().all())
            ),
            error="daily_return must not be null after the first row",
        ),
    ],
    strict=False,
    coerce=False,
)


__all__ = [
    "STANDARD_COLUMNS",
    "RAW_REQUIRED_COLUMNS",
    "YAHOO_COLUMN_RENAME",
    "SYMBOL_PATTERN",
    "StandardizedBarSchema",
]
