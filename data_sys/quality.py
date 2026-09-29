"""Data-quality validation built on the pandera schema.

Every failure is collected (``lazy=True``) and translated into a
:class:`Failure` describing *which* rule failed, *which* column/row, and the
offending value.  Nothing is dropped silently: a non-empty ``failures`` list
makes the pipeline abort the publish and keep the previous live dataset.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import pandera.pandas as pa

from data_sys.schema import StandardizedBarSchema

logger = logging.getLogger(__name__)

#: Columns allowed to contain a null value without being a defect.
NULLABLE_COLUMNS: frozenset[str] = frozenset({"daily_return"})


@dataclass(frozen=True)
class Failure:
    """A single quality-rule violation."""

    rule: str
    column: str | None = None
    index: int | None = None
    value: Any = None

    def describe(self) -> str:
        where = f"row={self.index}" if self.index is not None else "dataset"
        scope = f"column={self.column}" if self.column else "dataset"
        return f"[{scope} {where}] {self.rule} (value={self.value!r})"


@dataclass
class QualityReport:
    """Aggregate result of validating one frame."""

    ok: bool
    failures: list[Failure] = field(default_factory=list)
    row_count: int = 0
    stats: dict[str, Any] = field(default_factory=dict)

    def reasons(self) -> list[str]:
        """Human-readable descriptions of every failure."""
        return [failure.describe() for failure in self.failures]


def _basic_stats(frame: pd.DataFrame | None) -> dict[str, Any]:
    stats: dict[str, Any] = {"row_count": 0 if frame is None else int(len(frame))}
    if frame is None or len(frame) == 0:
        return stats
    if "date" in frame.columns:
        dates = pd.to_datetime(frame["date"])
        stats["min_date"] = dates.min().isoformat()
        stats["max_date"] = dates.max().isoformat()
    if "symbol" in frame.columns:
        stats["symbol"] = str(frame["symbol"].iloc[0])
    return stats


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _to_failure(row: pd.Series) -> Failure:
    column = row.get("column")
    column = None if _is_missing(column) else str(column)

    index = row.get("index")
    index = None if _is_missing(index) else int(index)

    value = row.get("failure_case")

    return Failure(rule=str(row.get("check")), column=column, index=index, value=value)


def validate_bars(frame: pd.DataFrame | None) -> QualityReport:
    """Validate ``frame`` against :data:`StandardizedBarSchema`."""
    stats = _basic_stats(frame)
    row_count = stats["row_count"]

    if frame is None or row_count == 0:
        return QualityReport(
            ok=False,
            failures=[Failure(rule="dataset must not be empty")],
            row_count=row_count,
            stats=stats,
        )

    try:
        StandardizedBarSchema.validate(frame, lazy=True)
    except pa.errors.SchemaErrors as exc:
        failures = [_to_failure(row) for _, row in exc.failure_cases.iterrows()]
        logger.warning("quality validation failed with %d failure(s)", len(failures))
        return QualityReport(ok=False, failures=failures, row_count=row_count, stats=stats)
    except pa.errors.SchemaError as exc:  # non-lazy fallback
        return QualityReport(
            ok=False,
            failures=[Failure(rule=str(exc))],
            row_count=row_count,
            stats=stats,
        )

    logger.info("quality validation passed for %d row(s)", row_count)
    return QualityReport(ok=True, failures=[], row_count=row_count, stats=stats)


__all__ = ["Failure", "QualityReport", "validate_bars", "NULLABLE_COLUMNS"]
