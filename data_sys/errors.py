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


class UniverseError(DataPipelineError):
    """Raised when the security master / universe configuration is invalid.

    Attributes
    ----------
    problems:
        Human readable descriptions of every violation found.  The whole file is
        validated before raising, so one run reports *all* mistakes instead of
        only the first one.
    """

    def __init__(self, message: str, problems: Sequence[str] | None = None) -> None:
        self.problems: list[str] = list(problems) if problems else []
        if self.problems:
            detail = "\n  - " + "\n  - ".join(self.problems)
            message = f"{message}{detail}"
        super().__init__(message)


class CalendarError(DataPipelineError):
    """Raised when the exchange trading calendar cannot be loaded or used."""


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


class DatasetSummaryError(DataPipelineError):
    """Raised when a summary of the published dataset cannot be produced."""


__all__ = [
    "DataPipelineError",
    "DownloadError",
    "StandardizationError",
    "UniverseError",
    "CalendarError",
    "QualityCheckError",
    "PublishError",
    "DuckDBVerificationError",
    "DatasetSummaryError",
]
