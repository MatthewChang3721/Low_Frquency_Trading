"""Tests for ``scripts/run_factor_batch.py`` -- the multi-date factor entry point.

Two kinds of test, deliberately separated:

* **orchestration tests** replace the single-day entry point with a controlled
  stand-in (``monkeypatch`` on the module attribute).  They check what the batch
  is responsible for -- which dates it visits, in which order, with which
  arguments, how it counts and records them, and that one failing date never
  stops the others -- without paying for a single momentum computation;
* **end-to-end tests** publish real Hive Parquet bars in ``tmp_path`` and run the
  real registry, universe, runner, store and analyzer through the real
  :func:`scripts.run_factor.run_factor`, so the cache behaviour (``miss`` ->
  ``hit`` -> ``forced``), the stored files and the CLI output are what a user
  would really get.

Nothing here reads the project's ``data/``, touches the network, or uses the
user's panels.  One test runs the documented command as a real process, because
"works from the project root" is a claim that a function call cannot prove.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, NamedTuple

import pandas as pd
import pytest

import scripts.run_factor_batch as batch
from data_sys.standardize import standardize_bars
from data_sys.storage import write_partitioned
from factor_sys import (
    FactorPanel,
    FactorQualityIssue,
    FactorQualityReport,
    FactorRegistryError,
    FactorScheduleError,
    QuickAnalyzer,
)
from factor_sys.errors import FactorStoreError
from scripts.run_factor import FactorRunOutcome
from scripts.run_factor_batch import (
    BatchRunError,
    DateFailure,
    FactorBatchOutcome,
    build_parser,
    format_summary,
    main,
    manifest_payload,
    run_factor_batch,
    write_manifest,
)
from tests.helpers import make_session_prices, session_dates, write_universe_file

#: The command under test, as a file (for the boundary and subprocess tests).
SCRIPT: Path = Path(batch.__file__).resolve()

#: The project root the documented command is run from.
PROJECT_ROOT: Path = SCRIPT.parents[1]

#: A small pool; ``write_universe_file`` pads it to the contract's minimum size
#: with inactive symbols, which the static provider leaves out.
SYMBOLS: tuple[str, ...] = ("AAPL", "MSFT", "NVDA", "AMZN")

FACTOR_NAME: str = "momentum_12_1"
FACTOR_VERSION: str = "1.0.0"

#: Enough history for a 253-session momentum window by January 2024.
HISTORY_START: str = "2022-06-01"
WINDOW_START: str = "2024-01-01"
WINDOW_END: str = "2024-01-31"

#: The Thursdays of January 2024: four decision days, four stored panels.
THURSDAYS: tuple[date, ...] = (
    date(2024, 1, 4),
    date(2024, 1, 11),
    date(2024, 1, 18),
    date(2024, 1, 25),
)

#: The summary block this command always prints, in this order.
SUMMARY_FIELDS: tuple[str, ...] = (
    "factor_name",
    "factor_version",
    "start_date",
    "end_date",
    "frequency",
    "anchor_weekday",
    "biweekly_anchor",
    "calendar_name",
    "selected_dates",
    "computed_count",
    "cached_count",
    "forced_count",
    "failed_count",
    "manifest_path",
)

#: The flag destinations the command accepts -- and nothing else.
EXPECTED_DESTS: frozenset[str] = frozenset(
    {
        "factor",
        "start_date",
        "end_date",
        "frequency",
        "anchor_weekday",
        "data_root",
        "universe_path",
        "calendar",
        "force",
        "help",
    }
)


# ---------------------------------------------------------------------------
# Fixtures and builders: real bars on disk, or a controlled single-day stand-in
# ---------------------------------------------------------------------------
class Market(NamedTuple):
    """One published dataset plus the universe file and bars that go with it."""

    root: Path
    universe: Path
    frame: pd.DataFrame


def publish(
    root: Path,
    symbols: tuple[str, ...] = SYMBOLS,
    *,
    start: str = HISTORY_START,
    end: str = WINDOW_END,
    seed: int = 20200102,
) -> pd.DataFrame:
    """Publish contract-valid bars for ``symbols`` under ``root`` and return them."""
    sessions = session_dates(start, end)
    frames = [
        standardize_bars(make_session_prices(sessions, seed=seed + position), symbol)
        for position, symbol in enumerate(symbols)
    ]
    frame = pd.concat(frames, ignore_index=True)
    write_partitioned(frame, root / "standardized" / "market_bars")
    return frame


@pytest.fixture
def market(tmp_path: Path) -> Market:
    """A complete history for :data:`SYMBOLS` plus a matching universe file."""
    root = tmp_path / "data"
    frame = publish(root)
    universe = write_universe_file(tmp_path / "universe.csv", SYMBOLS)
    return Market(root=root, universe=universe, frame=frame)


def fake_panel(day: date) -> FactorPanel:
    """A tiny but fully valid panel for ``day`` (no file, no factor, no bars)."""
    universe = ["AAPL", "MSFT"]
    frame = pd.DataFrame(
        {
            "as_of_date": [pd.Timestamp(day)] * len(universe),
            "symbol": universe,
            "factor_name": [FACTOR_NAME] * len(universe),
            "factor_version": [FACTOR_VERSION] * len(universe),
            "raw_value": [0.10, 0.20],
            "is_eligible": [True, True],
        }
    )
    return FactorPanel(frame, universe=universe, as_of_date=day)


def reported_quality_report(panel: FactorPanel) -> FactorQualityReport:
    """A report about ``panel`` with one soft issue: a finding, not a failure."""
    return FactorQualityReport(
        as_of_date=panel.as_of_date,
        factor_name=panel.factor_name,
        factor_version=panel.factor_version,
        universe_size=len(panel.universe),
        eligible_count=len(panel.universe),
        eligible_coverage=1.0,
        issues=(
            FactorQualityIssue(
                code="low_coverage",
                severity="warning",
                message="only one symbol carries a usable value",
                affected_symbols=panel.symbols[:1],
            ),
        ),
    )


def fake_outcome(
    day: date,
    path: Path,
    *,
    status: str = "miss",
    reported: bool = False,
) -> FactorRunOutcome:
    """The outcome a single-day run would return for ``day``, built by hand."""
    panel = fake_panel(day)
    return FactorRunOutcome(
        panel=panel,
        statistics=QuickAnalyzer().summarize(panel),
        quality_report=reported_quality_report(panel) if reported else None,
        stored_panel_path=path,
        cache_status=status,
    )


def install_fake_run_factor(
    monkeypatch: pytest.MonkeyPatch,
    behaviour: Any,
) -> list[dict[str, Any]]:
    """Replace the single-day entry point with a recorder driven by ``behaviour``.

    ``behaviour(day, force)`` returns the outcome to hand back (normally
    :func:`fake_outcome`) or raises the domain error the date should fail with.
    Every call is appended to the returned list, exactly as the batch made it.
    """
    calls: list[dict[str, Any]] = []

    def fake(
        factor_name: str,
        as_of_date: Any,
        *,
        data_root: Any = None,
        universe_path: Any = None,
        calendar_name: Any = None,
        force: bool = False,
    ) -> FactorRunOutcome:
        day = as_of_date if isinstance(as_of_date, date) else date.fromisoformat(str(as_of_date))
        calls.append(
            {
                "factor_name": factor_name,
                "as_of_date": day,
                "data_root": data_root,
                "universe_path": universe_path,
                "calendar_name": calendar_name,
                "force": force,
            }
        )
        return behaviour(day, force)

    monkeypatch.setattr(batch, "run_factor", fake)
    return calls


def panel_files(root: Path) -> list[Path]:
    """Every stored factor panel file under ``root``, sorted."""
    panels = root / "factors" / "panels"
    return sorted(panels.rglob("*.parquet")) if panels.exists() else []


def manifest_files(root: Path) -> list[Path]:
    """Every batch manifest under ``root``, sorted."""
    manifests = root.joinpath(*batch.MANIFEST_DIRECTORY)
    return sorted(manifests.glob("*.json")) if manifests.exists() else []


def temp_files(root: Path) -> list[Path]:
    """Every leftover temporary file of either atomic write, under ``root``."""
    return sorted(root.rglob("*.tmp")) if root.exists() else []


def read_manifest(path: Path) -> dict[str, Any]:
    """The manifest at ``path``, parsed."""
    return json.loads(path.read_text(encoding="utf-8"))


def summary_fields(text: str) -> dict[str, str]:
    """Parse the ``key: value`` summary block out of the command's output."""
    fields: dict[str, str] = {}
    for line in text.splitlines():
        key, separator, value = line.partition(": ")
        if separator and key in SUMMARY_FIELDS:
            fields[key] = value
    return fields


def status_lines(text: str) -> list[list[str]]:
    """The per-date status lines of the command's output, split into fields."""
    lines = []
    for line in text.splitlines():
        parts = line.split()
        if parts and parts[0][:2].isdigit() and len(parts) == 3:
            lines.append(parts)
    return lines


def run_cli(args: list[str], capsys: pytest.CaptureFixture) -> tuple[int, str, str]:
    """Drive :func:`main` like a shell would and return ``(code, stdout, stderr)``."""
    code = main(args)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def arguments(root: Path, universe: Path, *, force: bool = False, **overrides: str) -> list[str]:
    """The full command line for one batch, with any flag replaced."""
    flags = {
        "--factor": FACTOR_NAME,
        "--start": WINDOW_START,
        "--end": WINDOW_END,
        "--data-root": str(root),
        "--universe-path": str(universe),
    }
    flags.update(overrides)
    args: list[str] = []
    for flag, value in flags.items():
        args.extend([flag, value])
    if force:
        args.append("--force")
    return args


def file_state(path: Path) -> tuple[int, bytes]:
    """Modification time and exact bytes of one file (a write is visible here)."""
    return path.stat().st_mtime_ns, path.read_bytes()


# ---------------------------------------------------------------------------
# 2. Orchestration: a controlled single-day run, one call per selected date
# ---------------------------------------------------------------------------
def test_every_selected_date_is_run_exactly_once_in_schedule_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The batch visits Thursday after Thursday, and nothing else."""
    root = tmp_path / "data"
    calls = install_fake_run_factor(
        monkeypatch, lambda day, force: fake_outcome(day, root / f"{day.isoformat()}.parquet")
    )

    outcome = run_factor_batch(
        FACTOR_NAME,
        WINDOW_START,
        WINDOW_END,
        data_root=root,
        universe_path=tmp_path / "universe.csv",
    )

    assert [call["as_of_date"] for call in calls] == list(THURSDAYS)
    assert outcome.selected_dates == THURSDAYS
    assert [item.statistics.as_of_date for item in outcome.outcomes] == list(THURSDAYS)
    assert outcome.failures == ()
    assert outcome.failed_count == 0


def test_the_shared_flags_reach_the_single_day_entry_point(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Factor, root, pool, calendar and ``--force`` are passed through unchanged."""
    root = tmp_path / "data"
    universe = tmp_path / "universe.csv"
    calls = install_fake_run_factor(
        monkeypatch,
        lambda day, force: fake_outcome(day, root / "panel.parquet", status="forced"),
    )

    run_factor_batch(
        FACTOR_NAME,
        WINDOW_START,
        WINDOW_END,
        data_root=root,
        universe_path=universe,
        calendar_name="XNYS",
        force=True,
    )

    assert len(calls) == len(THURSDAYS)
    for call in calls:
        assert call["factor_name"] == FACTOR_NAME
        assert call["data_root"] == root
        assert call["universe_path"] == universe
        assert call["calendar_name"] == "XNYS"
        assert call["force"] is True


def test_the_counts_follow_each_dates_cache_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One miss, one hit and one forced date are counted once each."""
    root = tmp_path / "data"
    statuses = iter(["miss", "hit", "forced", "hit"])

    def behaviour(day: date, force: bool) -> FactorRunOutcome:
        return fake_outcome(day, root / f"{day.isoformat()}.parquet", status=next(statuses))

    install_fake_run_factor(monkeypatch, behaviour)

    outcome = run_factor_batch(FACTOR_NAME, WINDOW_START, WINDOW_END, data_root=root)

    assert [item.cache_status for item in outcome.outcomes] == ["miss", "hit", "forced", "hit"]
    assert (outcome.computed_count, outcome.cached_count, outcome.forced_count) == (1, 2, 1)
    assert (outcome.succeeded_count, outcome.failed_count, outcome.selection_size) == (4, 0, 4)


def test_a_failing_date_does_not_stop_the_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The second date fails; the third and fourth are still run and recorded."""
    root = tmp_path / "data"

    def behaviour(day: date, force: bool) -> FactorRunOutcome:
        if day == date(2024, 1, 11):
            raise FactorStoreError(f"the stored panel for {day.isoformat()} cannot be read")
        return fake_outcome(day, root / f"{day.isoformat()}.parquet", status="hit")

    calls = install_fake_run_factor(monkeypatch, behaviour)

    outcome = run_factor_batch(FACTOR_NAME, WINDOW_START, WINDOW_END, data_root=root)

    assert [call["as_of_date"] for call in calls] == list(THURSDAYS)
    assert [item.statistics.as_of_date for item in outcome.outcomes] == [
        date(2024, 1, 4),
        date(2024, 1, 18),
        date(2024, 1, 25),
    ]
    assert outcome.failures == (
        DateFailure(
            as_of_date=date(2024, 1, 11),
            error_type="FactorStoreError",
            message="the stored panel for 2024-01-11 cannot be read",
        ),
    )
    assert outcome.failed_count == 1
    assert outcome.cached_count == 3


def test_a_quality_report_is_not_a_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A soft finding is reported, counted as a success and printed as ``reported``."""
    root = tmp_path / "data"
    install_fake_run_factor(
        monkeypatch,
        lambda day, force: fake_outcome(
            day,
            root / f"{day.isoformat()}.parquet",
            reported=True,
        ),
    )

    outcome = run_factor_batch(FACTOR_NAME, WINDOW_START, WINDOW_END, data_root=root)

    assert outcome.failed_count == 0
    assert [item.quality_status for item in outcome.outcomes] == ["reported"] * 4
    assert "2024-01-04  miss    reported" in format_summary(outcome)


def test_an_unknown_factor_fails_before_any_date_is_run(tmp_path: Path) -> None:
    """A typo in the factor name is one error, not one error per Thursday."""
    root = tmp_path / "data"

    with pytest.raises(FactorRegistryError, match="not registered"):
        run_factor_batch("value_quality", WINDOW_START, WINDOW_END, data_root=root)

    assert not (root / "factors").exists()
    assert manifest_files(root) == []


def test_an_unknown_calendar_fails_before_any_date_is_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The calendar is resolved while selecting dates, so no date is attempted."""
    root = tmp_path / "data"
    calls = install_fake_run_factor(monkeypatch, lambda day, force: fake_outcome(day, root))

    with pytest.raises(FactorScheduleError, match="could not load the trading calendar"):
        run_factor_batch(
            FACTOR_NAME, WINDOW_START, WINDOW_END, data_root=root, calendar_name="NOPE"
        )

    assert calls == []
    assert manifest_files(root) == []


def test_a_blank_data_root_is_refused_before_any_date_is_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A blank root would mean the current directory: refuse it up front."""
    calls = install_fake_run_factor(monkeypatch, lambda day, force: fake_outcome(day, tmp_path))

    with pytest.raises(BatchRunError, match="--data-root must not be blank"):
        run_factor_batch(FACTOR_NAME, WINDOW_START, WINDOW_END, data_root="   ")

    assert calls == []
    assert manifest_files(tmp_path) == []


def test_an_inverted_window_is_refused(tmp_path: Path) -> None:
    """The window is closed and ordered; a reversed one is a typo."""
    with pytest.raises(FactorScheduleError, match="must not be after end_date"):
        run_factor_batch(FACTOR_NAME, "2024-01-31", "2024-01-01", data_root=tmp_path / "data")


def test_the_outcome_is_immutable_and_in_schedule_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The result object cannot be edited, and it keeps the order it ran in."""
    root = tmp_path / "data"
    install_fake_run_factor(
        monkeypatch, lambda day, force: fake_outcome(day, root / f"{day.isoformat()}.parquet")
    )

    outcome = run_factor_batch(FACTOR_NAME, WINDOW_START, WINDOW_END, data_root=root)

    with pytest.raises(FrozenInstanceError):
        outcome.frequency = "monthly"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        outcome.outcomes = ()  # type: ignore[misc]

    assert list(outcome.selected_dates) == sorted(outcome.selected_dates)
    assert list(outcome.outcomes) == sorted(
        outcome.outcomes, key=lambda item: item.statistics.as_of_date
    )
    assert repr(outcome) == (
        "FactorBatchOutcome(factor_name='momentum_12_1', factor_version='1.0.0', "
        "frequency='weekly', anchor_weekday='THU', dates=4, computed=4, cached=0, "
        "forced=0, failed=0)"
    )


def batch_fields(tmp_path: Path, **overrides: Any) -> dict[str, Any]:
    """The fields of one valid four-date batch, with any of them replaced."""
    root = tmp_path / "data"
    fields: dict[str, Any] = {
        "factor_name": FACTOR_NAME,
        "factor_version": FACTOR_VERSION,
        "start_date": date(2024, 1, 1),
        "end_date": date(2024, 1, 31),
        "frequency": "weekly",
        "anchor_weekday": "THU",
        "calendar_name": "XNYS",
        "biweekly_anchor": None,
        "selected_dates": THURSDAYS,
        "outcomes": tuple(
            fake_outcome(day, root / f"{day.isoformat()}.parquet") for day in THURSDAYS
        ),
        "failures": (),
        "manifest_path": root / "factors" / "batch_runs" / "run.json",
    }
    fields.update(overrides)
    return fields


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"end_date": date(2023, 12, 31)}, "must not be after end_date"),
        ({"factor_version": "  "}, "factor_version must be a non-empty string"),
        ({"frequency": "daily"}, "frequency must be one of"),
        ({"anchor_weekday": None}, "anchor_weekday must be one of"),
        ({"biweekly_anchor": batch.BIWEEKLY_ANCHOR}, "biweekly_anchor must be None"),
        ({"frequency": "monthly", "anchor_weekday": "THU"}, "carries no anchor"),
        ({"selected_dates": THURSDAYS[::-1]}, "strictly ascending"),
        ({"selected_dates": (date(2023, 12, 28),)}, "must lie inside"),
        ({"selected_dates": [date(2024, 1, 4)]}, "selected_dates must be a tuple"),
        ({"outcomes": []}, "outcomes must be a tuple"),
        ({"failures": [DateFailure(date(2024, 1, 4), "FactorError", "x")]}, "must be a tuple"),
        (
            {"failures": (DateFailure(date(2024, 1, 4), "FactorError", "boom"),)},
            "both as an outcome and as a failure",
        ),
        (
            {"failures": (DateFailure(date(2023, 12, 28), "FactorError", "boom"),)},
            "is not one of the selected dates",
        ),
        (
            {
                "failures": (
                    DateFailure(date(2024, 1, 11), "FactorError", "later"),
                    DateFailure(date(2024, 1, 4), "FactorError", "earlier"),
                )
            },
            "strictly ascending",
        ),
        ({"manifest_path": "relative.json"}, "manifest_path must be a pathlib.Path"),
        ({"manifest_error": 5}, "manifest_error must be a string or None"),
        (
            {"outcomes": (), "failures": ()},
            "every selected date needs exactly one outcome or one failure",
        ),
    ],
)
def test_an_inconsistent_batch_outcome_is_refused(
    tmp_path: Path, overrides: dict[str, Any], message: str
) -> None:
    """A result object that contradicts itself must not describe a run."""
    with pytest.raises(BatchRunError, match=message):
        FactorBatchOutcome(**batch_fields(tmp_path, **overrides))


def test_an_outcome_of_another_factor_version_is_refused(tmp_path: Path) -> None:
    """Panels of another factor version cannot be this batch's outcomes."""
    with pytest.raises(BatchRunError, match="describes 'momentum_12_1' v1.0.0"):
        FactorBatchOutcome(**batch_fields(tmp_path, factor_version="2.0.0"))


def test_a_date_failure_checks_its_own_fields() -> None:
    """A failure record needs a plain date and a non-empty error type."""
    with pytest.raises(BatchRunError, match="plain date"):
        DateFailure(as_of_date=datetime(2024, 1, 4, 16), error_type="E", message="m")
    with pytest.raises(BatchRunError, match="non-empty error_type"):
        DateFailure(as_of_date=date(2024, 1, 4), error_type=" ", message="m")
    with pytest.raises(BatchRunError, match="must be a string"):
        DateFailure(as_of_date=date(2024, 1, 4), error_type="E", message=5)  # type: ignore[arg-type]

    assert repr(DateFailure(date(2024, 1, 4), "FactorError", "m")) == (
        "DateFailure(as_of_date='2024-01-04', error_type='FactorError')"
    )


def test_the_valid_batch_fields_build_an_outcome(tmp_path: Path) -> None:
    """The same fields, unmodified, are a valid record."""
    outcome = FactorBatchOutcome(**batch_fields(tmp_path))

    assert outcome.selection_size == 4
    assert outcome.computed_count == 4
    assert outcome.manifest_written is True


def test_a_selected_date_without_any_record_is_refused(tmp_path: Path) -> None:
    """A gap in the record is refused at construction, not left for the summary."""
    recorded = (fake_outcome(THURSDAYS[0], tmp_path / "first.parquet"),)

    with pytest.raises(BatchRunError) as excinfo:
        FactorBatchOutcome(**batch_fields(tmp_path, outcomes=recorded))

    message = str(excinfo.value)
    assert "every selected date needs exactly one outcome or one failure" in message
    assert "nothing was recorded for 2024-01-11, 2024-01-18, 2024-01-25" in message
    assert "2024-01-04" not in message.split("nothing was recorded for")[1]


def test_outcomes_and_failures_together_cover_every_selected_date(tmp_path: Path) -> None:
    """Two successes and two failures tile the four dates: a valid, printable record."""
    outcome = FactorBatchOutcome(
        **batch_fields(
            tmp_path,
            outcomes=tuple(
                fake_outcome(day, tmp_path / f"{day.isoformat()}.parquet") for day in THURSDAYS[:2]
            ),
            failures=(
                DateFailure(THURSDAYS[2], "FactorStoreError", "damaged"),
                DateFailure(THURSDAYS[3], "FactorStoreError", "damaged"),
            ),
        )
    )

    assert (outcome.succeeded_count, outcome.failed_count) == (2, 2)

    text = format_summary(outcome)
    assert "2024-01-04  miss    clean" in text
    assert "2024-01-18  failed  FactorStoreError" in text
    assert "failed_count: 2" in text


# ---------------------------------------------------------------------------
# 3. The manifest: one JSON record per run
# ---------------------------------------------------------------------------
def test_the_manifest_records_the_whole_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Window, cadence, anchor, calendar, every date and every result are there."""
    root = tmp_path / "data"

    def behaviour(day: date, force: bool) -> FactorRunOutcome:
        panel_path = root / "factors" / "panels" / f"as_of_date={day.isoformat()}.parquet"
        return fake_outcome(day, panel_path)

    install_fake_run_factor(monkeypatch, behaviour)

    outcome = run_factor_batch(FACTOR_NAME, WINDOW_START, WINDOW_END, data_root=root)

    assert outcome.manifest_written is True
    assert outcome.manifest_error is None
    assert outcome.manifest_path.parent == root.joinpath(*batch.MANIFEST_DIRECTORY)
    assert outcome.manifest_path.suffix == ".json"
    assert outcome.manifest_path.is_file()
    assert manifest_files(root) == [outcome.manifest_path]

    manifest = read_manifest(outcome.manifest_path)
    assert manifest["format_version"] == batch.MANIFEST_FORMAT_VERSION == 1
    assert manifest["batch_run_id"] == outcome.manifest_path.stem
    created = datetime.fromisoformat(manifest["created_at_utc"])
    assert created.utcoffset() == timedelta(0)
    assert manifest["factor_name"] == FACTOR_NAME
    assert manifest["factor_version"] == FACTOR_VERSION
    assert manifest["requested_start_date"] == WINDOW_START
    assert manifest["requested_end_date"] == WINDOW_END
    assert manifest["frequency"] == "weekly"
    assert manifest["anchor_weekday"] == "THU"
    assert manifest["biweekly_anchor"] is None
    assert manifest["calendar_name"] == "XNYS"
    assert manifest["selected_dates"] == [day.isoformat() for day in THURSDAYS]

    assert [result["as_of_date"] for result in manifest["results"]] == [
        day.isoformat() for day in THURSDAYS
    ]
    assert [result["status"] for result in manifest["results"]] == ["miss"] * 4
    assert [result["quality_status"] for result in manifest["results"]] == ["clean"] * 4
    assert manifest["failures"] == []
    for result in manifest["results"]:
        path = result["panel_path"]
        assert path == f"factors/panels/as_of_date={result['as_of_date']}.parquet"
        assert not Path(path).is_absolute()
        assert "\\" not in path  # portable separators


def test_a_panel_outside_the_data_root_is_recorded_as_an_absolute_path(
    tmp_path: Path,
) -> None:
    """A path the root cannot explain is quoted as-is instead of invented."""
    root = tmp_path / "data"
    outside = tmp_path / "elsewhere" / "panel.parquet"
    outcome = FactorBatchOutcome(
        **batch_fields(
            tmp_path,
            manifest_path=root / "m.json",
            outcomes=(fake_outcome(THURSDAYS[0], outside),),
            selected_dates=(THURSDAYS[0],),
        )
    )

    payload = manifest_payload(
        outcome, batch_run_id="run", created_at_utc="2024-01-01T00:00:00+00:00", data_root=root
    )

    assert payload["results"][0]["panel_path"] == outside.as_posix()
    assert payload["selected_dates"] == [THURSDAYS[0].isoformat()]


def test_a_failed_date_is_recorded_in_the_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The manifest is complete evidence: successes and failures, in order."""
    root = tmp_path / "data"

    def behaviour(day: date, force: bool) -> FactorRunOutcome:
        if day == date(2024, 1, 18):
            raise FactorStoreError("the stored panel is damaged")
        return fake_outcome(day, root / f"{day.isoformat()}.parquet", status="hit")

    install_fake_run_factor(monkeypatch, behaviour)

    outcome = run_factor_batch(FACTOR_NAME, WINDOW_START, WINDOW_END, data_root=root)
    manifest = read_manifest(outcome.manifest_path)

    assert manifest["selected_dates"] == [day.isoformat() for day in THURSDAYS]
    assert [result["as_of_date"] for result in manifest["results"]] == [
        "2024-01-04",
        "2024-01-11",
        "2024-01-25",
    ]
    assert manifest["failures"] == [
        {
            "as_of_date": "2024-01-18",
            "error_type": "FactorStoreError",
            "message": "the stored panel is damaged",
        }
    ]
    assert manifest["results"][0]["status"] == "hit"


def test_two_runs_write_two_independent_manifests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A manifest is a record of one run, never a cache key."""
    root = tmp_path / "data"
    install_fake_run_factor(
        monkeypatch,
        lambda day, force: fake_outcome(
            day, root / f"{day.isoformat()}.parquet", status="forced" if force else "miss"
        ),
    )

    first = run_factor_batch(FACTOR_NAME, WINDOW_START, WINDOW_END, data_root=root)
    first_state = file_state(first.manifest_path)
    second = run_factor_batch(FACTOR_NAME, WINDOW_START, WINDOW_END, data_root=root, force=True)

    assert first.manifest_path != second.manifest_path
    assert len(manifest_files(root)) == 2
    assert file_state(first.manifest_path) == first_state
    assert (
        read_manifest(first.manifest_path)["batch_run_id"]
        != read_manifest(second.manifest_path)["batch_run_id"]
    )
    assert [item["status"] for item in read_manifest(second.manifest_path)["results"]] == [
        "forced"
    ] * 4


def test_an_empty_selection_still_writes_an_empty_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A window that holds no decision day is honoured, and recorded as empty."""
    root = tmp_path / "data"
    install_fake_run_factor(monkeypatch, lambda day, force: fake_outcome(day, root / "x.parquet"))

    outcome = run_factor_batch(FACTOR_NAME, "2024-01-01", "2024-01-03", data_root=root)

    assert outcome.selected_dates == ()
    assert (outcome.outcomes, outcome.failures) == ((), ())
    assert outcome.selection_size == 0
    manifest = read_manifest(outcome.manifest_path)
    assert manifest["selected_dates"] == []
    assert (manifest["results"], manifest["failures"]) == ([], [])
    assert summary_fields(format_summary(outcome))["selected_dates"] == "0"


def test_a_failed_manifest_write_is_reported_and_keeps_the_old_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """A record that cannot be written fails the command, never the panels."""
    root = tmp_path / "data"
    universe = tmp_path / "universe.csv"
    install_fake_run_factor(
        monkeypatch, lambda day, force: fake_outcome(day, root / f"{day.isoformat()}.parquet")
    )
    first = run_factor_batch(FACTOR_NAME, WINDOW_START, WINDOW_END, data_root=root)
    before = file_state(first.manifest_path)

    def forbidden(source: object, target: object) -> None:
        raise OSError("the disk is on fire")

    monkeypatch.setattr(os, "replace", forbidden)

    outcome = run_factor_batch(FACTOR_NAME, WINDOW_START, WINDOW_END, data_root=root)

    assert outcome.manifest_written is False
    assert outcome.manifest_error is not None
    assert "could not write the batch manifest" in outcome.manifest_error
    assert "the disk is on fire" in outcome.manifest_error
    assert [item.statistics.as_of_date for item in outcome.outcomes] == list(THURSDAYS)
    assert file_state(first.manifest_path) == before
    assert temp_files(root) == []

    code, out, err = run_cli(arguments(root, universe), capsys)

    assert code == 1
    assert summary_fields(out)["failed_count"] == "0"
    assert err.startswith("error: ")
    assert "could not write the batch manifest" in err
    assert "Traceback" not in err


def test_write_manifest_writes_a_complete_file_atomically(tmp_path: Path) -> None:
    """A fresh manifest is created, replaced in place, and leaves no temp file."""
    path = tmp_path / "nested" / "dir" / "run.json"

    returned = write_manifest({"format_version": 1, "runs": ["a"]}, path)

    assert returned == path
    assert read_manifest(path) == {"format_version": 1, "runs": ["a"]}
    assert path.read_text(encoding="utf-8").endswith("\n")

    write_manifest({"format_version": 1, "runs": ["b"]}, path)

    assert read_manifest(path) == {"format_version": 1, "runs": ["b"]}
    assert sorted(path.parent.iterdir()) == [path]


def test_a_manifest_write_that_fails_leaves_the_previous_file_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The temporary file is removed and the live manifest is never half-written."""
    path = tmp_path / "run.json"
    write_manifest({"format_version": 1, "runs": ["old"]}, path)
    before = file_state(path)

    def forbidden(source: object, target: object) -> None:
        raise OSError("no rename today")

    monkeypatch.setattr(os, "replace", forbidden)

    with pytest.raises(BatchRunError, match="could not write the batch manifest"):
        write_manifest({"format_version": 1, "runs": ["new"]}, path)

    assert file_state(path) == before
    assert sorted(path.parent.iterdir()) == [path]


def test_a_manifest_directory_that_cannot_be_created_is_a_batch_error(
    tmp_path: Path,
) -> None:
    """A file where the directory should be is reported, not raised as an OSError."""
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("I am a file", encoding="utf-8")

    with pytest.raises(BatchRunError, match="cannot create the batch-run directory"):
        write_manifest({"format_version": 1}, blocker / "run.json")


# ---------------------------------------------------------------------------
# 4. End to end: real bars in, real panels and a real manifest out
# ---------------------------------------------------------------------------
def test_a_batch_computes_the_missing_panels_and_reads_them_back(market: Market) -> None:
    """Four Thursdays are computed once, stored once, and read back unchanged."""
    first = run_factor_batch(
        FACTOR_NAME,
        WINDOW_START,
        WINDOW_END,
        data_root=market.root,
        universe_path=market.universe,
    )

    assert [item.cache_status for item in first.outcomes] == ["miss"] * 4
    assert [item.statistics.as_of_date for item in first.outcomes] == list(THURSDAYS)
    assert [item.quality_status for item in first.outcomes] == ["clean"] * 4
    assert [item.statistics.valid_count for item in first.outcomes] == [len(SYMBOLS)] * 4
    files = panel_files(market.root)
    assert [item.stored_panel_path for item in first.outcomes] == files
    assert (first.computed_count, first.cached_count, first.forced_count) == (4, 0, 0)

    states = {path: file_state(path) for path in files}
    second = run_factor_batch(
        FACTOR_NAME,
        WINDOW_START,
        WINDOW_END,
        data_root=market.root,
        universe_path=market.universe,
    )

    assert [item.cache_status for item in second.outcomes] == ["hit"] * 4
    assert (second.computed_count, second.cached_count, second.forced_count) == (0, 4, 0)
    assert [item.statistics.mean for item in second.outcomes] == [
        item.statistics.mean for item in first.outcomes
    ]
    assert {path: file_state(path) for path in panel_files(market.root)} == states
    assert temp_files(market.root) == []
    assert len(manifest_files(market.root)) == 2


def test_a_forced_batch_recomputes_every_date(
    market: Market, capsys: pytest.CaptureFixture
) -> None:
    """``--force`` recomputes all four dates and still leaves exactly four files."""
    run_factor_batch(
        FACTOR_NAME, WINDOW_START, WINDOW_END, data_root=market.root, universe_path=market.universe
    )
    before = panel_files(market.root)

    code, out, err = run_cli(arguments(market.root, market.universe, force=True), capsys)

    assert (code, err) == (0, "")
    fields = summary_fields(out)
    assert fields["forced_count"] == "4"
    assert fields["computed_count"] == "0"
    assert fields["cached_count"] == "0"
    assert [row[1] for row in status_lines(out)] == ["forced"] * 4
    assert panel_files(market.root) == before
    assert temp_files(market.root) == []


def test_a_damaged_panel_fails_only_that_date(
    market: Market, capsys: pytest.CaptureFixture
) -> None:
    """One unreadable panel is reported for its date; the others still run."""
    first = run_factor_batch(
        FACTOR_NAME, WINDOW_START, WINDOW_END, data_root=market.root, universe_path=market.universe
    )
    damaged = first.outcomes[2].stored_panel_path
    damaged.write_bytes(b"not a parquet file")
    state = file_state(damaged)

    outcome = run_factor_batch(
        FACTOR_NAME, WINDOW_START, WINDOW_END, data_root=market.root, universe_path=market.universe
    )

    assert [item.statistics.as_of_date for item in outcome.outcomes] == [
        THURSDAYS[0],
        THURSDAYS[1],
        THURSDAYS[3],
    ]
    assert [failure.as_of_date for failure in outcome.failures] == [THURSDAYS[2]]
    assert outcome.failures[0].error_type == "FactorStoreError"
    assert "cannot be read as parquet" in outcome.failures[0].message
    assert outcome.failed_count == 1
    assert outcome.cached_count == 3
    assert file_state(damaged) == state
    assert len(panel_files(market.root)) == 4

    code, out, err = run_cli(arguments(market.root, market.universe), capsys)

    assert code == 1
    assert err == ""
    assert "2024-01-18  failed  FactorStoreError" in out
    assert summary_fields(out)["failed_count"] == "1"
    assert summary_fields(out)["cached_count"] == "3"
    assert out.splitlines()[-1].startswith("manifest_path: ")


# ---------------------------------------------------------------------------
# 5. The command line: output, exit codes and argparse
# ---------------------------------------------------------------------------
def test_the_cli_reports_each_date_then_the_summary(
    market: Market, capsys: pytest.CaptureFixture
) -> None:
    """One status line per Thursday, then the fourteen summary fields in order."""
    code, out, err = run_cli(arguments(market.root, market.universe), capsys)

    assert (code, err) == (0, "")
    assert status_lines(out) == [[day.isoformat(), "miss", "clean"] for day in THURSDAYS]

    fields = summary_fields(out)
    assert tuple(fields) == SUMMARY_FIELDS
    assert fields["factor_name"] == FACTOR_NAME
    assert fields["factor_version"] == FACTOR_VERSION
    assert fields["start_date"] == WINDOW_START
    assert fields["end_date"] == WINDOW_END
    assert fields["frequency"] == "weekly"
    assert fields["anchor_weekday"] == "THU"
    assert fields["biweekly_anchor"] == "n/a"
    assert fields["calendar_name"] == "XNYS"
    assert fields["selected_dates"] == "4"
    assert fields["computed_count"] == "4"
    assert fields["failed_count"] == "0"
    assert Path(fields["manifest_path"]) == manifest_files(market.root)[0]
    assert out.splitlines()[-1] == f"manifest_path: {fields['manifest_path']}"


def test_a_monthly_run_reports_no_anchor(market: Market, capsys: pytest.CaptureFixture) -> None:
    """A monthly batch has no anchor and no phase, and says so with ``n/a``."""
    code, out, err = run_cli(
        arguments(
            market.root,
            market.universe,
            **{"--start": "2023-12-01", "--end": "2024-01-31", "--frequency": "monthly"},
        ),
        capsys,
    )

    assert (code, err) == (0, "")
    assert status_lines(out) == [
        ["2023-12-29", "miss", "clean"],
        ["2024-01-31", "miss", "clean"],
    ]
    fields = summary_fields(out)
    assert fields["frequency"] == "monthly"
    assert fields["anchor_weekday"] == "n/a"
    assert fields["biweekly_anchor"] == "n/a"
    assert fields["selected_dates"] == "2"


def test_an_unknown_factor_exits_one_with_one_line(
    market: Market, capsys: pytest.CaptureFixture
) -> None:
    """A domain error is one ``error:`` line on stderr, never a traceback."""
    code, out, err = run_cli(
        arguments(market.root, market.universe, **{"--factor": "value_quality"}), capsys
    )

    assert (code, out) == (1, "")
    assert err.startswith("error: ")
    assert "value_quality" in err
    assert "Traceback" not in err
    assert panel_files(market.root) == []
    assert manifest_files(market.root) == []


def test_an_unusable_window_exits_one_with_one_line(
    market: Market, capsys: pytest.CaptureFixture
) -> None:
    """An inverted window is a domain error, not a crash."""
    code, out, err = run_cli(
        arguments(market.root, market.universe, **{"--start": "2024-01-31", "--end": "2024-01-01"}),
        capsys,
    )

    assert (code, out) == (1, "")
    assert "must not be after end_date" in err
    assert "Traceback" not in err


@pytest.mark.parametrize(
    "args",
    [
        [],
        ["--factor", FACTOR_NAME],
        ["--factor", FACTOR_NAME, "--start", WINDOW_START],
        ["--factor", FACTOR_NAME, "--start", WINDOW_START, "--end", WINDOW_END, "--as-of", "x"],
        ["--factor", FACTOR_NAME, "--start", WINDOW_START, "--end", WINDOW_END, "--date", "x"],
        [
            "--factor",
            FACTOR_NAME,
            "--start",
            WINDOW_START,
            "--end",
            WINDOW_END,
            "--frequency",
            "daily",
        ],
        [
            "--factor",
            FACTOR_NAME,
            "--start",
            WINDOW_START,
            "--end",
            WINDOW_END,
            "--anchor-weekday",
            "THU2",
        ],
        ["--force"],
    ],
)
def test_argparse_keeps_its_own_exit_code(args: list[str]) -> None:
    """A missing or unknown flag is argparse's exit code 2, not ours."""
    with pytest.raises(SystemExit) as exit_info:
        main(args)
    assert exit_info.value.code == 2


def test_an_anchor_with_a_monthly_cadence_is_refused_by_argparse(
    capsys: pytest.CaptureFixture,
) -> None:
    """A flag that cannot apply is refused up front instead of silently ignored."""
    with pytest.raises(SystemExit) as exit_info:
        main(
            [
                "--factor",
                FACTOR_NAME,
                "--start",
                WINDOW_START,
                "--end",
                WINDOW_END,
                "--frequency",
                "monthly",
                "--anchor-weekday",
                "THU",
            ]
        )

    assert exit_info.value.code == 2
    assert "--anchor-weekday does not apply to --frequency monthly" in capsys.readouterr().err


def test_the_parser_expects_exactly_the_documented_flags() -> None:
    """No hidden flag, and no alternative date range behind another name."""
    actions = {action.dest for action in build_parser()._actions}
    assert actions == EXPECTED_DESTS

    parsed = build_parser().parse_args(
        ["--factor", FACTOR_NAME, "--start", WINDOW_START, "--end", WINDOW_END]
    )
    assert (parsed.frequency, parsed.anchor_weekday) == ("weekly", None)
    assert (parsed.data_root, parsed.universe_path, parsed.calendar) == (None, None, None)
    assert parsed.force is False


def test_the_documented_command_runs_in_a_real_process(market: Market) -> None:
    """``python scripts/run_factor_batch.py ...`` works from the project root."""
    command = [
        sys.executable,
        str(SCRIPT),
        "--factor",
        FACTOR_NAME,
        "--start",
        WINDOW_START,
        "--end",
        WINDOW_END,
        "--frequency",
        "weekly",
        "--anchor-weekday",
        "THU",
        "--data-root",
        str(market.root),
        "--universe-path",
        str(market.universe),
    ]

    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    assert "Traceback" not in completed.stdout
    assert status_lines(completed.stdout) == [
        [day.isoformat(), "miss", "clean"] for day in THURSDAYS
    ]
    assert summary_fields(completed.stdout)["selected_dates"] == "4"
    assert len(manifest_files(market.root)) == 1


# ---------------------------------------------------------------------------
# 6. The module boundary: orchestration only
# ---------------------------------------------------------------------------
def script_tree() -> ast.Module:
    """The parsed source of the command under test."""
    return ast.parse(SCRIPT.read_text(encoding="utf-8"))


def imported_modules(tree: ast.Module) -> set[str]:
    """Every module the script imports, by dotted name."""
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
    return modules


def referenced_names(tree: ast.Module) -> set[str]:
    """Every bare name and attribute the script's code mentions."""
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    names |= {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    return names


def string_constants(tree: ast.Module) -> set[str]:
    """Every string literal in the script (docstrings included)."""
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }


def test_the_script_imports_no_formula_no_network_and_no_subprocess() -> None:
    """Composition only: no download, no updater, no HTTP, no second process."""
    modules = imported_modules(script_tree())
    forbidden = (
        "factors",
        "yfinance",
        "httpx",
        "requests",
        "urllib",
        "http",
        "socket",
        "subprocess",
        "data_sys.providers",
        "data_sys.update",
        "data_sys.data_query",
        "data_sys.batch_query",
        "data_sys.planner",
        "data_sys.merge",
    )
    for module in sorted(modules):
        assert not any(
            module == banned or module.startswith(f"{banned}.") for banned in forbidden
        ), f"the batch entry point must not import {module!r}"

    assert {
        "factor_sys",
        "data_sys.config",
        "data_sys.market_calendar",
        "data_sys.utils",
        "scripts.run_factor",
    } <= modules


def test_the_script_delegates_the_whole_date_and_panel_work() -> None:
    """It calls the single-day entry point and the schedule rule instead of copying."""
    names = referenced_names(script_tree())

    assert {
        "run_factor",
        "decision_schedule",
        "default_factor_registry",
        "manifest_payload",
        "write_manifest",
        "utc_now_iso",
        "new_run_id",
    } <= names
    for forbidden in (
        "FactorPanel",
        "FactorContext",
        "ParquetFactorStore",
        "FactorRunner",
        "StaticUniverseProvider",
        "ParquetMarketDataReader",
        "QuickAnalyzer",
        "DefaultFactorQualityChecker",
        "read_partitioned",
        "write_partitioned",
        "exists",
        "read",
    ):
        assert forbidden not in names, f"the batch must not do the work itself: {forbidden!r}"


def test_the_script_never_reproduces_the_store_layout() -> None:
    """The panel path comes from the outcome; only the manifest path is built here."""
    tree = script_tree()
    constants = string_constants(tree)

    assert not any(".parquet" in value for value in constants)
    assert not any("year=" in value for value in constants)

    # The single path this module builds is the manifest's, in one place.
    names = referenced_names(tree)
    assert {"joinpath", "MANIFEST_DIRECTORY", "stored_panel_path"} <= names


def test_the_manifest_names_its_directory_in_one_place() -> None:
    """The batch writes to ``<data_root>/factors/batch_runs`` and nowhere else."""
    assert batch.MANIFEST_DIRECTORY == ("factors", "batch_runs")
    assert Path(*batch.MANIFEST_DIRECTORY, "run.json").as_posix() == "factors/batch_runs/run.json"
    assert batch.MANIFEST_FORMAT_VERSION == 1
