"""Exception hierarchy for the data pipeline.

All data-pipeline specific errors derive from :class:`DataPipelineError` so the
orchestrator can catch a single base class, record the failure reason and exit
with a non-zero status without leaking an unexpected traceback.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence


class DataPipelineError(Exception):
    """Base class for every error raised by the data pipeline."""


class DownloadError(DataPipelineError):
    """Raised when a market-data provider cannot deliver usable raw data."""


class StandardizationError(DataPipelineError):
    """Raised when raw data cannot be mapped to the standardized schema."""


class QualityCheckError(DataPipelineError):
    """Raised when standardized data violates one or more quality rules.

    Attributes
    ----------
    failures:
        Human readable failure descriptions. Usually the simplified strings of
        :class:`data_sys.quality.Failure` objects, kept as plain ``str`` here so
        this module has no dependency on the quality module.
    """

    def __init__(self, message: str, failures: Sequence[str] | None = None) -> None:
        self.failures: list[str] = list(failures) if failures else []
        if self.failures:
            detail = "\n  - " + "\n  - ".join(self.failures)
            message = f"{message}{detail}"
        super().__init__(message)


class PublishError(DataPipelineError):
    """Raised when the atomic publish step cannot replace the live dataset."""


class DuckDBVerificationError(DataPipelineError):
    """Raised when the DuckDB verification layer cannot confirm the dataset."""


__all__ = [
    "DataPipelineError",
    "DownloadError",
    "StandardizationError",
    "QualityCheckError",
    "PublishError",
    "DuckDBVerificationError",
]
