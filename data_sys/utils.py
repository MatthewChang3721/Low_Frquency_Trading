"""Small, dependency-free helpers shared across the data pipeline."""

from __future__ import annotations

import re
from dataclasses import asdict, is_dataclass
from datetime import UTC, date, datetime
from typing import Any
from uuid import uuid4

_RUN_ID_SAFE = re.compile(r"[^0-9A-Za-z]+")


def utc_now() -> datetime:
    """Current time as a timezone-aware UTC datetime."""
    return datetime.now(UTC)


def utc_now_iso() -> str:
    """Current UTC time as an ISO-8601 string with seconds precision."""
    return utc_now().isoformat(timespec="seconds")


def parse_date(value: date | datetime | str | None) -> date | None:
    """Normalize a date-like value to :class:`datetime.date` (``None`` in -> out)."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text:
        return None
    return date.fromisoformat(text[:10])


def new_run_id(now: datetime | None = None) -> str:
    """A sortable, filesystem-safe run identifier (UTC timestamp + short uuid)."""
    stamp = (now or utc_now()).astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{uuid4().hex[:6]}"


def jsonable(value: Any) -> Any:
    """Best-effort conversion of common values into JSON-serializable objects."""
    if is_dataclass(value) and not isinstance(value, type):
        return {k: jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


__all__ = [
    "utc_now",
    "utc_now_iso",
    "parse_date",
    "new_run_id",
    "jsonable",
]
