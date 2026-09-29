# Required Packages

## Data System

### Runtime dependencies

1. **Storage and query**: `duckdb`, `pyarrow`
   - Query, write, and validate Parquet datasets.
2. **Tabular computing**: `pandas`, `numpy`
   - Clean daily bars, calculate returns, and provide a common tabular layer.
3. **Market-data providers**: `yfinance`, `httpx`
   - `yfinance` is the first Yahoo Finance adapter.
   - `httpx` retrieves SEC EDGAR data and downloads Google Drive data packages.
4. **Data contracts and quality**: `pydantic`, `pandera`
   - `pydantic` validates configuration and domain objects.
   - `pandera` validates DataFrame schemas and financial-data quality rules.
5. **Trading calendar**: `exchange-calendars`
   - Provides NYSE sessions, holidays, and rebalance-date adjustments.
6. **Reliability and command line**: `tenacity`, `typer`
   - `tenacity` retries transient download failures.
   - `typer` supports commands such as `python scripts/download_data.py sync`.

```powershell
pip install duckdb pyarrow pandas numpy yfinance httpx pydantic pandera exchange-calendars tenacity typer
```

### Development dependencies

1. `pytest` — unit and integration tests.
2. `pytest-cov` — test coverage reporting.
3. `ruff` — linting and formatting checks.

```powershell
pip install pytest pytest-cov ruff
```

### Python standard library (no installation required)

`json`, `logging`, `pathlib`, `tempfile`, `zipfile`, and `shutil` support release-manifest parsing, structured logging, Google Drive package extraction, and recoverable local data replacement.

## Deferred packages

Do not install these until their corresponding systems begin implementation: `backtrader`, `vectorbt`, `cvxpy`, `scikit-learn`, and `matplotlib`.
