"""Market data system for the US equity low-frequency trading project.

The data system is responsible for fetching, standardizing, validating,
storing and reading market data. This package currently implements the
"market daily bars MVP": a single-symbol (AAPL) daily OHLCV closed loop
built on Yahoo Finance -> standardized Parquet -> DuckDB verification.

Public entry point:
    ``python -m data_sys.data_query``
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
