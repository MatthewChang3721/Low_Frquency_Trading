"""Logging setup and per-run metadata records.

Every pipeline run writes one JSON file describing *what ran, on what, with
which library versions and how it ended*.  Together with the untouched raw
download in ``data/raw/`` this makes any run reproducible.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from data_sys.utils import jsonable, utc_now_iso

#: Libraries whose versions are recorded for reproducibility.
TRACKED_PACKAGES: tuple[str, ...] = (
    "pandas",
    "numpy",
    "pyarrow",
    "duckdb",
    "pandera",
    "yfinance",
    "pydantic",
    "tenacity",
    "typer",
    "exchange-calendars",
)

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logging once; safe to call repeatedly."""
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=level, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    root.setLevel(level)


def collect_package_versions() -> dict[str, str | None]:
    """Return the installed version of every tracked dependency."""
    versions: dict[str, str | None] = {}
    for package in TRACKED_PACKAGES:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:  # pragma: no cover - dependency missing
            versions[package] = None
    return versions


@dataclass
class RunMetadata:
    """Reproducibility record for one pipeline run."""

    run_id: str
    data_source: str
    symbol: str
    requested_start: str
    requested_end: str
    output_path: str
    run_at_utc: str = field(default_factory=utc_now_iso)
    status: str = "running"
    row_count: int = 0
    min_date: str | None = None
    max_date: str | None = None
    raw_path: str | None = None
    years: list[tuple[int, int]] = field(default_factory=list)
    failure_reasons: list[str] = field(default_factory=list)
    package_versions: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return jsonable(asdict(self))


def write_metadata(meta: RunMetadata, out_dir: Path | str) -> Path:
    """Write ``meta`` as ``run_<run_id>.json`` under ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    path = out_dir / f"run_{meta.run_id}.json"
    path.write_text(
        json.dumps(meta.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


__all__ = [
    "RunMetadata",
    "configure_logging",
    "collect_package_versions",
    "write_metadata",
    "TRACKED_PACKAGES",
]
