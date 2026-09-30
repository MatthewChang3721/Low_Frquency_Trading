"""Market data system for the US equity low-frequency trading project.

The data system is responsible for fetching, standardizing, validating,
storing and reading market data. This package currently implements the
"market daily bars MVP": an *incremental* daily OHLCV closed loop for a single
symbol or a whole validated universe, built on Yahoo Finance -> standardized
Parquet -> staging-only DuckDB verification.

Public entry points:
    ``python -m data_sys.data_query``    # one symbol
    ``python -m data_sys.batch_query``   # every active symbol of a universe
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
