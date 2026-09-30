"""Merging newly downloaded bars into the published history.

The published dataset is the *authoritative* history: an incremental update never
throws it away, it re-derives it.  For every symbol:

1. the untouched published frame and all freshly downloaded frames are
   concatenated,
2. ``(symbol, date)`` is de-duplicated **keeping the newly downloaded row**, so
   a refreshed session replaces the stale one,
3. rows are sorted by ``(symbol, date)``,
4. ``daily_return`` and ``dollar_volume`` are recomputed over the *whole* merged
   history.

Step 4 is not optional.  Keeping the previously published ``daily_return`` would
leave a stale value on the boundary row of every refreshed window, because that
row's return depends on the previous session's ``adjusted_close`` -- which the
download may have just revised.  Recomputing from the merged series is the only
way the column stays internally consistent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

import pandas as pd

from data_sys.errors import DataPipelineError
from data_sys.schema import STANDARD_COLUMNS
from data_sys.standardize import derive_features

logger = logging.getLogger(__name__)

#: Primary key of the standardized dataset.
KEY_COLUMNS: list[str] = ["symbol", "date"]

#: Columns a frame must expose before it can take part in a merge.  The derived
#: columns are recomputed, so they are not required as input.
MERGE_REQUIRED_COLUMNS: list[str] = [
    "date",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "adjusted_close",
    "volume",
]


@dataclass(frozen=True)
class MergeReport:
    """What the merge did, in numbers a metadata file can record."""

    existing_row_count: int = 0
    downloaded_row_count: int = 0
    merged_row_count: int = 0
    new_dates_count: int = 0
    replaced_dates_count: int = 0
    retained_row_count: int = 0
    min_date: date | None = None
    max_date: date | None = None


def date_set(frame: pd.DataFrame | None) -> set[date]:
    """The distinct trading dates of ``frame`` as plain :class:`datetime.date`."""
    if frame is None or len(frame) == 0 or "date" not in frame.columns:
        return set()
    values = pd.to_datetime(frame["date"], errors="coerce")
    return {stamp.date() for stamp in values.dropna()}


def empty_bars_frame() -> pd.DataFrame:
    """An empty frame carrying the standardized column order."""
    return pd.DataFrame(columns=STANDARD_COLUMNS)


def _prepare(frame: pd.DataFrame | None, label: str) -> pd.DataFrame:
    """Coerce a frame into mergeable shape or explain precisely why not."""
    if frame is None or len(frame) == 0:
        return empty_bars_frame()

    missing = [column for column in MERGE_REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise DataPipelineError(
            f"the {label} frame is missing required column(s) {missing}; "
            f"found {list(frame.columns)}"
        )

    out = frame.loc[:, MERGE_REQUIRED_COLUMNS].copy()
    parsed = pd.to_datetime(out["date"], errors="coerce")
    if parsed.isna().any():
        bad = int(parsed.isna().sum())
        raise DataPipelineError(f"the {label} frame has {bad} unparseable date(s)")
    out["date"] = parsed.dt.normalize().astype("datetime64[s]")
    out["symbol"] = out["symbol"].astype(str)
    return out


def merge_bars(
    existing: pd.DataFrame | None, downloaded: pd.DataFrame | None
) -> tuple[pd.DataFrame, MergeReport]:
    """Merge ``downloaded`` into ``existing`` and recompute derived columns.

    Parameters
    ----------
    existing:
        Currently published history for the symbol (empty/``None`` when the
        symbol is new).
    downloaded:
        Standardized frames downloaded in this run; these values win on any
        overlapping date.

    Returns
    -------
    tuple[pandas.DataFrame, MergeReport]
        The merged frame in :data:`~data_sys.schema.STANDARD_COLUMNS` order, and a
        quantitative summary of the merge.
    """
    published = _prepare(existing, "published")
    fresh = _prepare(downloaded, "downloaded")

    if published.empty and fresh.empty:
        raise DataPipelineError(
            "nothing to merge: the published and downloaded frames are both empty"
        )

    existing_dates = date_set(published)
    downloaded_dates = date_set(fresh)
    replaced = len(downloaded_dates & existing_dates)
    new_dates = len(downloaded_dates - existing_dates)

    # Empty frames are dropped before concatenating: an empty, all-``object``
    # frame would otherwise upcast every real column to ``object`` (pandas does
    # not ignore the dtypes of an empty operand), which the quality schema
    # rejects.  `fresh` stays last so `keep="last"` lets this run's values win.
    parts = [frame for frame in (published, fresh) if not frame.empty]
    combined = parts[0].copy() if len(parts) == 1 else pd.concat(parts, ignore_index=True)
    before = len(combined)
    combined = combined.drop_duplicates(subset=KEY_COLUMNS, keep="last")
    duplicates = before - len(combined)

    combined = combined.sort_values(KEY_COLUMNS, kind="stable").reset_index(drop=True)
    combined = derive_features(combined)[STANDARD_COLUMNS]

    report = MergeReport(
        existing_row_count=len(published),
        downloaded_row_count=len(fresh),
        merged_row_count=len(combined),
        new_dates_count=new_dates,
        replaced_dates_count=replaced,
        retained_row_count=max(len(published) - replaced, 0),
        min_date=None if combined.empty else combined["date"].min().date(),
        max_date=None if combined.empty else combined["date"].max().date(),
    )
    logger.info(
        "merged %d published + %d downloaded row(s) -> %d row(s) "
        "(new dates=%d, replaced dates=%d, duplicates collapsed=%d)",
        report.existing_row_count,
        report.downloaded_row_count,
        report.merged_row_count,
        report.new_dates_count,
        report.replaced_dates_count,
        duplicates,
    )
    return combined, report


__all__ = [
    "KEY_COLUMNS",
    "MERGE_REQUIRED_COLUMNS",
    "MergeReport",
    "date_set",
    "empty_bars_frame",
    "merge_bars",
]
