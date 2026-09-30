"""Pipeline configuration and derived filesystem layout.

The layout follows the architecture document: raw downloads, standardized
Parquet and run metadata are stored under a single ``data/`` root that is
excluded from Git.  ``DataProvider`` implementations and the storage layer
never hard-code paths; they receive the paths derived here.

Layout::

    data/
      raw/market_bars/                     # untouched provider output
      standardized/market_bars/            # Hive partitioned by symbol & year
      .staging/<run_id>/                   # atomic publish workspace
      .trash/<run_id>/                     # previous version backup during swap
      metadata/run_<run_id>.json           # per-symbol reproducibility record
      metadata/batch_<batch_run_id>.json   # per-batch universe summary
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

# Project root = parent directory of the ``data_sys`` package.  Using an
# absolute default keeps the pipeline CWD independent.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

DEFAULT_SYMBOL: str = "AAPL"
DEFAULT_START: date = date(2020, 1, 1)

#: NYSE sessions re-downloaded on every incremental update so that Yahoo's
#: retroactive corrections to recent prices / adjustment factors are picked up.
DEFAULT_REFRESH_OVERLAP_SESSIONS: int = 5


@dataclass(frozen=True)
class PipelineConfig:
    """Immutable configuration for one end-to-end daily-bar run."""

    symbol: str = DEFAULT_SYMBOL
    start: date = DEFAULT_START
    end: date = field(default_factory=date.today)
    data_root: Path = PROJECT_ROOT / "data"

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError(
                f"end ({self.end.isoformat()}) must not be before "
                f"start ({self.start.isoformat()})"
            )

    # -- raw -----------------------------------------------------------------
    @property
    def raw_dir(self) -> Path:
        """Directory holding untouched provider downloads."""
        return self.data_root / "raw" / "market_bars"

    @property
    def raw_symbol_dir(self) -> Path:
        """Directory holding this symbol's untouched provider downloads."""
        return self.raw_dir / self.symbol

    # -- standardized --------------------------------------------------------
    @property
    def standardized_dir(self) -> Path:
        """Root of the published Hive-partitioned dataset."""
        return self.data_root / "standardized" / "market_bars"

    @property
    def live_symbol_dir(self) -> Path:
        """Published partition directory for this symbol."""
        return self.standardized_dir / f"symbol={self.symbol}"

    # -- staging / trash -----------------------------------------------------
    @property
    def staging_dir(self) -> Path:
        """Root of the staging area used for atomic publishes."""
        return self.data_root / ".staging"

    @property
    def trash_dir(self) -> Path:
        """Root of the backup area used while swapping in new data."""
        return self.data_root / ".trash"

    # -- metadata ------------------------------------------------------------
    @property
    def metadata_dir(self) -> Path:
        """Directory holding per-run metadata JSON files."""
        return self.data_root / "metadata"

    def metadata_path(self, run_id: str) -> Path:
        """Path of the metadata file for ``run_id``."""
        return self.metadata_dir / f"run_{run_id}.json"

    def batch_metadata_path(self, batch_run_id: str) -> Path:
        """Path of the batch summary metadata file for ``batch_run_id``."""
        return self.metadata_dir / f"batch_{batch_run_id}.json"


__all__ = [
    "PROJECT_ROOT",
    "DEFAULT_SYMBOL",
    "DEFAULT_START",
    "DEFAULT_REFRESH_OVERLAP_SESSIONS",
    "PipelineConfig",
]
