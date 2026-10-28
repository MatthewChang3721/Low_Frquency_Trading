"""Tests for ``scripts/analyze_factor.py`` -- the multi-date research entry point.

Two kinds of test, deliberately separated:

* **orchestration tests** replace the batch entry point and the forward-return
  calculator with controlled stand-ins (``monkeypatch`` on the module
  attributes).  They check what this module is responsible for -- that it asks
  the existing batch for its dates, that it labels each panel once with the
  factor's own calendar, that the three analyzers see the same objects, and how
  it refuses an incomplete run -- without paying for a single momentum
  computation or touching a bar;
* **end-to-end tests** publish real Hive Parquet bars in ``tmp_path`` and run the
  real registry, universe, runner, store, schedule, label calculator and
  analyzers through the real :func:`scripts.run_factor_batch.run_factor_batch`,
  so the panels, the labels, the report and the CLI output are what a user would
  really get.

Nothing here reads the project's ``data/``, touches the network, or uses the
user's panels.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import FrozenInstanceError, replace
from datetime import date, timedelta
from pathlib import Path
from typing import Any, NamedTuple

import pandas as pd
import pytest

import scripts.analyze_factor as analyze
from data_sys.standardize import standardize_bars
from data_sys.storage import write_partitioned
from factor_sys import (
    FORWARD_RETURN_COLUMNS,
    FactorPanel,
    ForwardReturnPanel,
    ICAnalysisError,
    ICAnalyzer,
    ParquetMarketDataReader,
    QuantileAnalyzer,
    QuickAnalyzer,
)
from factor_sys.errors import FactorResearchError
from factor_sys.ic_analysis import IC_OBSERVATION_COLUMNS
from factor_sys.quantile_analysis import QUANTILE_OBSERVATION_COLUMNS, QUANTILE_SPREAD_COLUMNS
from factor_sys.quick_analysis import STATISTICS_COLUMNS
from scripts.analyze_factor import (
    METADATA_FILE,
    TABLE_FILES,
    FactorResearchOutcome,
    analyze_factor_history,
    build_parser,
    format_summary,
    main,
    report_metadata,
)
from scripts.run_factor import FactorRunOutcome
from scripts.run_factor_batch import DateFailure, FactorBatchOutcome
from tests.helpers import make_session_prices, session_dates, write_universe_file

#: The command under test, as a file (for the boundary tests).
SCRIPT: Path = Path(analyze.__file__).resolve()

#: The project root the documented command is run from.
PROJECT_ROOT: Path = SCRIPT.parents[1]

FACTOR_NAME: str = "momentum_12_1"
FACTOR_VERSION: str = "1.0.0"

#: A pool big enough for a real cross-section; ``write_universe_file`` pads it to
#: the contract's minimum size with inactive symbols, which the provider leaves out.
SYMBOLS: tuple[str, ...] = ("AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META")

#: Enough history for a 253-session momentum window by January 2024.
HISTORY_START: str = "2022-06-01"

#: Bars published well past the window, so a 20-session label is available for
#: every January decision date.
PRICE_END: str = "2024-03-29"

WINDOW_START: str = "2024-01-01"
WINDOW_END: str = "2024-01-31"

#: The Thursdays of January 2024 and the Wednesdays of the same window: two
#: different samplings of the same month.
THURSDAYS: tuple[date, ...] = (
    date(2024, 1, 4),
    date(2024, 1, 11),
    date(2024, 1, 18),
    date(2024, 1, 25),
)
WEDNESDAYS: tuple[date, ...] = (
    date(2024, 1, 3),
    date(2024, 1, 10),
    date(2024, 1, 17),
    date(2024, 1, 24),
    date(2024, 1, 31),
)

#: The summary block this command always prints, in this order.
SUMMARY_FIELDS: tuple[str, ...] = (
    "factor_name",
    "factor_version",
    "start_date",
    "end_date",
    "frequency",
    "anchor_weekday",
    "calendar_name",
    "horizon_sessions",
    "quantiles",
    "min_pairs",
    "selected_panel_count",
    "computed_count",
    "cached_count",
    "forced_count",
    "valid_ic_count",
    "mean_ic",
    "ic_std",
    "icir",
    "report_path",
    "batch_manifest_path",
)

#: The flag destinations the command accepts -- and nothing else.
EXPECTED_DESTS: frozenset[str] = frozenset(
    {
        "factor",
        "start_date",
        "end_date",
        "frequency",
        "anchor_weekday",
        "horizon_sessions",
        "quantiles",
        "min_pairs",
        "data_root",
        "universe_path",
        "calendar",
        "force",
        "report_root",
        "help",
    }
)


# ---------------------------------------------------------------------------
# Fixtures and builders: real bars on disk, or controlled stand-ins
# ---------------------------------------------------------------------------
class Market(NamedTuple):
    """One published dataset plus the universe file that goes with it."""

    root: Path
    universe: Path


def publish(
    root: Path,
    symbols: tuple[str, ...] = SYMBOLS,
    *,
    start: str = HISTORY_START,
    end: str = PRICE_END,
    seed: int = 20200102,
) -> None:
    """Publish contract-valid bars for ``symbols`` under ``root``."""
    sessions = session_dates(start, end)
    frames = [
        standardize_bars(make_session_prices(sessions, seed=seed + position), symbol)
        for position, symbol in enumerate(symbols)
    ]
    write_partitioned(pd.concat(frames, ignore_index=True), root / "standardized" / "market_bars")


@pytest.fixture
def market(tmp_path: Path) -> Market:
    """A complete history for :data:`SYMBOLS` plus a matching universe file."""
    root = tmp_path / "data"
    publish(root)
    universe = write_universe_file(tmp_path / "universe.csv", SYMBOLS)
    return Market(root=root, universe=universe)


#: The four symbols of a hand-built panel and its deliberately ordered values.
FAKE_SYMBOLS: tuple[str, ...] = ("AAPL", "MSFT", "NVDA", "AMZN")
FAKE_VALUES: tuple[float, float, float, float] = (0.40, 0.30, 0.20, 0.10)


def label_returns(count: int) -> tuple[float, ...]:
    """Forward returns that rank exactly against :data:`FAKE_VALUES`: IC ``-1.0``."""
    return tuple(-0.02 + 0.01 * position for position in range(count))


def fake_panel(day: date, *, version: str = FACTOR_VERSION) -> FactorPanel:
    """A tiny but fully valid panel for ``day`` (no file, no factor, no bars)."""
    universe = list(FAKE_SYMBOLS)
    frame = pd.DataFrame(
        {
            "as_of_date": [pd.Timestamp(day)] * len(universe),
            "symbol": universe,
            "factor_name": [FACTOR_NAME] * len(universe),
            "factor_version": [version] * len(universe),
            "raw_value": list(FAKE_VALUES),
            "is_eligible": [True] * len(universe),
        }
    )
    return FactorPanel(frame, universe=universe, as_of_date=day)


def fake_labels(panel: FactorPanel, horizon_sessions: int) -> ForwardReturnPanel:
    """Labels for ``panel`` whose ranks are perfectly inverted to its values."""
    universe = list(panel.universe)
    end_date = panel.as_of_date + timedelta(days=horizon_sessions)
    frame = pd.DataFrame(
        {
            "as_of_date": [pd.Timestamp(panel.as_of_date)] * len(universe),
            "end_date": [pd.Timestamp(end_date)] * len(universe),
            "symbol": universe,
            "factor_name": [panel.factor_name] * len(universe),
            "factor_version": [panel.factor_version] * len(universe),
            "horizon_sessions": [horizon_sessions] * len(universe),
            "forward_return": list(label_returns(len(universe))),
            "is_available": [True] * len(universe),
        }
    )
    return ForwardReturnPanel(
        frame,
        universe=universe,
        as_of_date=panel.as_of_date,
        end_date=end_date,
        factor_name=panel.factor_name,
        factor_version=panel.factor_version,
        horizon_sessions=horizon_sessions,
    )


class FakeCalculator:
    """A stand-in for ``ForwardReturnCalculator`` that records every call.

    One instance is built per analysis, exactly like the real class, and every
    ``compute`` call is appended to :attr:`calls`.  ``behaviour`` lets a test
    replace the answer (or raise) without touching the module under test.
    """

    instances: list[FakeCalculator] = []
    behaviour: Callable[[FakeCalculator, Any, int], ForwardReturnPanel] | None = None

    def __init__(self, market_data_reader: Any, *, calendar_name: str) -> None:
        self.market_data_reader = market_data_reader
        self.calendar_name = calendar_name
        self.calls: list[tuple[Any, int]] = []
        FakeCalculator.instances.append(self)

    def compute(self, panel: Any, horizon_sessions: int) -> ForwardReturnPanel:
        """Record the call, then answer with ``behaviour`` or the default labels."""
        self.calls.append((panel, horizon_sessions))
        if FakeCalculator.behaviour is not None:
            return FakeCalculator.behaviour(self, panel, horizon_sessions)
        return fake_labels(panel, horizon_sessions)


def install_calculator(
    monkeypatch: pytest.MonkeyPatch,
    behaviour: Callable[[FakeCalculator, Any, int], ForwardReturnPanel] | None = None,
) -> list[FakeCalculator]:
    """Replace the label calculator with :class:`FakeCalculator`, resetting state."""
    FakeCalculator.instances = []
    FakeCalculator.behaviour = behaviour
    monkeypatch.setattr(analyze, "ForwardReturnCalculator", FakeCalculator)
    return FakeCalculator.instances


def fake_run_outcome(
    day: date, path: Path, *, status: str = "miss", version: str = FACTOR_VERSION
) -> FactorRunOutcome:
    """The outcome the single-day entry point would return for ``day``."""
    panel = fake_panel(day, version=version)
    return FactorRunOutcome(
        panel=panel,
        statistics=QuickAnalyzer().summarize(panel),
        quality_report=None,
        stored_panel_path=path,
        cache_status=status,
    )


def fake_batch(
    dates: Sequence[date],
    *,
    failed: Sequence[date] = (),
    manifest: Path,
    frequency: str = "weekly",
    anchor_weekday: str | None = "THU",
    calendar_name: str = "XNYS",
    factor_name: str = FACTOR_NAME,
    factor_version: str = FACTOR_VERSION,
    start_date: date | None = None,
    end_date: date | None = None,
) -> FactorBatchOutcome:
    """A hand-built batch outcome over ``dates``, with ``failed`` recorded as holes."""
    selected = tuple(dates)
    broken = set(failed)
    outcomes = tuple(
        fake_run_outcome(
            day,
            manifest.parent / "panels" / f"{day.isoformat()}.parquet",
            version=factor_version,
        )
        for day in selected
        if day not in broken
    )
    failures = tuple(
        DateFailure(day, "FactorStoreError", "the stored panel cannot be read as parquet")
        for day in selected
        if day in broken
    )
    return FactorBatchOutcome(
        factor_name=factor_name,
        factor_version=factor_version,
        start_date=start_date if start_date is not None else selected[0],
        end_date=end_date if end_date is not None else selected[-1],
        frequency=frequency,
        anchor_weekday=anchor_weekday,
        calendar_name=calendar_name,
        biweekly_anchor=None,
        selected_dates=selected,
        outcomes=outcomes,
        failures=failures,
        manifest_path=manifest,
    )


def install_batch(
    monkeypatch: pytest.MonkeyPatch, outcome: FactorBatchOutcome
) -> list[dict[str, Any]]:
    """Replace the batch entry point with a recorder that answers ``outcome``."""
    calls: list[dict[str, Any]] = []

    def fake(factor_name: str, start_date: Any, end_date: Any, **kwargs: Any) -> Any:
        calls.append(
            {"factor_name": factor_name, "start_date": start_date, "end_date": end_date, **kwargs}
        )
        return outcome

    monkeypatch.setattr(analyze, "run_factor_batch", fake)
    return calls


def install_analyzers(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Record what the three analyzers were handed, and delegate the real work."""
    record: dict[str, Any] = {}

    class RecordingQuick(QuickAnalyzer):
        def summarize_history(self, panels: Any) -> Any:
            record["quick"] = tuple(panels)
            return super().summarize_history(panels)

    class RecordingIC(ICAnalyzer):
        def analyze(self, panels: Any, labels: Any) -> Any:
            record["ic"] = (tuple(panels), tuple(labels))
            return super().analyze(panels, labels)

    class RecordingQuantile(QuantileAnalyzer):
        def analyze(self, panels: Any, labels: Any) -> Any:
            record["quantile"] = (tuple(panels), tuple(labels))
            return super().analyze(panels, labels)

    monkeypatch.setattr(analyze, "QuickAnalyzer", RecordingQuick)
    monkeypatch.setattr(analyze, "ICAnalyzer", RecordingIC)
    monkeypatch.setattr(analyze, "QuantileAnalyzer", RecordingQuantile)
    return record


# -- reading what a run left on disk ----------------------------------------
def panel_files(root: Path) -> list[Path]:
    """Every stored factor panel under ``root``, sorted."""
    panels = root / "factors" / "panels"
    return sorted(panels.rglob("*.parquet")) if panels.exists() else []


def manifest_files(root: Path) -> list[Path]:
    """Every batch manifest under ``root``, sorted."""
    manifests = root / "factors" / "batch_runs"
    return sorted(manifests.glob("*.json")) if manifests.exists() else []


def report_root(root: Path) -> Path:
    """Where an analysis reports under ``root`` by default."""
    return root.joinpath(*analyze.RESEARCH_DIRECTORY)


def report_dirs(root: Path) -> list[Path]:
    """Every published research report under ``root``, sorted."""
    directory = report_root(root)
    if not directory.exists():
        return []
    return sorted(path for path in directory.iterdir() if path.is_dir())


def same_objects(left: Sequence[Any], right: Sequence[Any]) -> bool:
    """``True`` when the two sequences hold the very same objects, in the same order."""
    return len(left) == len(right) and all(a is b for a, b in zip(left, right, strict=True))


def staging_dirs(root: Path) -> list[Path]:
    """Every leftover staging directory of a report publish, under ``root``."""
    directory = report_root(root)
    return sorted(directory.glob(".*.tmp")) if directory.exists() else []


def report_files(path: Path) -> list[str]:
    """The file names of one report directory, sorted."""
    return sorted(item.name for item in path.iterdir())


def read_metadata(path: Path) -> dict[str, Any]:
    """The metadata of one report, parsed strictly (``NaN`` is not valid JSON)."""

    def refuse(token: str) -> None:
        pytest.fail(f"metadata must not contain {token}: it is not valid JSON")

    text = (path / METADATA_FILE).read_text(encoding="utf-8")
    assert "NaN" not in text
    assert "Infinity" not in text
    return json.loads(text, parse_constant=refuse)


def read_table(path: Path) -> pd.DataFrame:
    """Read one report table, parsing whichever of its two date columns exist."""
    header = pd.read_csv(path, nrows=0)
    dates = [name for name in ("as_of_date", "end_date") if name in header.columns]
    return pd.read_csv(path, parse_dates=dates)


def summary_fields(text: str) -> dict[str, str]:
    """Parse the ``key: value`` summary block out of the command's output."""
    fields: dict[str, str] = {}
    for line in text.splitlines():
        key, separator, value = line.partition(": ")
        if separator and key in SUMMARY_FIELDS:
            fields[key] = value
    return fields


def run_cli(args: list[str], capsys: pytest.CaptureFixture) -> tuple[int, str, str]:
    """Drive :func:`main` like a shell would and return ``(code, stdout, stderr)``."""
    code = main(args)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def arguments(
    root: Path,
    universe: Path,
    *,
    horizon: str = "20",
    force: bool = False,
    **overrides: str,
) -> list[str]:
    """The full command line for one analysis, with any flag replaced.

    ``--flag None`` removes the flag, so a test can exercise the defaults.
    """
    flags: dict[str, str] = {
        "--factor": FACTOR_NAME,
        "--start": WINDOW_START,
        "--end": WINDOW_END,
        "--horizon-sessions": horizon,
        "--data-root": str(root),
        "--universe-path": str(universe),
    }
    for flag, value in overrides.items():
        if value is None:
            flags.pop(flag, None)
        else:
            flags[flag] = value
    args: list[str] = []
    for flag, value in flags.items():
        args.extend([flag, value])
    if force:
        args.append("--force")
    return args


def file_state(path: Path) -> tuple[int, bytes]:
    """Modification time and exact bytes of one file (a write is visible here)."""
    return path.stat().st_mtime_ns, path.read_bytes()


def analyze_market(market: Market, **kwargs: Any) -> FactorResearchOutcome:
    """Analyze :data:`FACTOR_NAME` over the January window of ``market``."""
    options: dict[str, Any] = {
        "horizon_sessions": 20,
        "data_root": market.root,
        "universe_path": market.universe,
    }
    options.update(kwargs)
    return analyze_factor_history(FACTOR_NAME, WINDOW_START, WINDOW_END, **options)


def session_after(day: date, count: int) -> date:
    """The ``count``-th NYSE session after ``day``, computed independently."""
    from data_sys.market_calendar import get_calendar, sessions_in_range

    window = sessions_in_range(get_calendar("XNYS"), day, day + timedelta(days=count * 3 + 10))
    return window[count]


# ---------------------------------------------------------------------------
# 1. The request: cadence, horizon and the two analysis knobs
# ---------------------------------------------------------------------------
def test_the_horizon_is_independent_of_the_cadence(market: Market) -> None:
    """``weekly + THU + 20`` labels every Thursday twenty sessions ahead."""
    outcome = analyze_market(market, horizon_sessions=20, frequency="weekly", anchor_weekday="THU")

    metadata = read_metadata(outcome.report_path)
    assert metadata["frequency"] == "weekly"
    assert metadata["anchor_weekday"] == "THU"
    assert metadata["horizon_sessions"] == 20
    assert [day.isoformat() for day in outcome.batch_outcome.selected_dates] == [
        day.isoformat() for day in THURSDAYS
    ]

    labels = read_table(outcome.report_path / TABLE_FILES["forward_returns"])
    assert labels["horizon_sessions"].unique().tolist() == [20]
    # Each label ends on the twentieth session after its own decision date: the
    # horizon is counted in sessions, and the cadence has nothing to do with it.
    ends = labels.groupby("as_of_date")["end_date"].first()
    for day in THURSDAYS:
        assert ends[pd.Timestamp(day)].date() == session_after(day, 20)


def test_a_wednesday_and_a_thursday_anchor_measure_different_dates(market: Market) -> None:
    """The anchor decides *which* cross-sections exist, not how far they look."""
    wednesday = analyze_market(market, anchor_weekday="WED")
    thursday = analyze_market(market, anchor_weekday="THU")

    wednesday_dates = read_metadata(wednesday.report_path)["selected_dates"]
    thursday_dates = read_metadata(thursday.report_path)["selected_dates"]
    assert wednesday_dates == [day.isoformat() for day in WEDNESDAYS]
    assert thursday_dates == [day.isoformat() for day in THURSDAYS]
    assert set(wednesday_dates).isdisjoint(thursday_dates)
    assert wednesday.report_path != thursday.report_path


def test_a_biweekly_schedule_keeps_every_second_candidate(market: Market) -> None:
    """``biweekly`` follows the batch rule: the first candidate, then every other."""
    outcome = analyze_market(market, frequency="biweekly", anchor_weekday="THU")

    metadata = read_metadata(outcome.report_path)
    assert metadata["frequency"] == "biweekly"
    assert metadata["biweekly_anchor"] == "first_selected_week_in_requested_range"
    assert metadata["selected_dates"] == ["2024-01-04", "2024-01-18"]


def test_a_monthly_schedule_takes_the_last_session_of_each_month(tmp_path: Path) -> None:
    """``monthly`` has no anchor and follows the month-end rule of the batch."""
    root = tmp_path / "data"
    publish(root)
    universe = write_universe_file(tmp_path / "universe.csv", SYMBOLS)

    outcome = analyze_factor_history(
        FACTOR_NAME,
        "2024-01-01",
        "2024-03-31",
        horizon_sessions=5,
        frequency="monthly",
        data_root=root,
        universe_path=universe,
    )

    metadata = read_metadata(outcome.report_path)
    assert metadata["frequency"] == "monthly"
    assert metadata["anchor_weekday"] is None
    assert metadata["selected_dates"] == ["2024-01-31", "2024-02-29", "2024-03-28"]


def test_a_monthly_schedule_with_an_anchor_exits_two(
    market: Market, capsys: pytest.CaptureFixture
) -> None:
    """``monthly`` refuses ``--anchor-weekday`` instead of ignoring it."""
    args = arguments(
        market.root, market.universe, **{"--frequency": "monthly", "--anchor-weekday": "THU"}
    )

    with pytest.raises(SystemExit) as raised:
        main(args)

    assert raised.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "--anchor-weekday does not apply to --frequency monthly" in captured.err
    assert report_dirs(market.root) == []


@pytest.mark.parametrize(
    "flag, value",
    [
        ("--horizon-sessions", "0"),
        ("--horizon-sessions", "-5"),
        ("--horizon-sessions", "5.5"),
        ("--quantiles", "1"),
        ("--quantiles", "two"),
        ("--min-pairs", "1"),
    ],
)
def test_a_bad_number_is_refused_by_argparse(
    flag: str, value: str, market: Market, capsys: pytest.CaptureFixture
) -> None:
    """A horizon, a group count or a pair threshold the analysis cannot use is ``2``."""
    args = arguments(market.root, market.universe)
    if flag == "--horizon-sessions":
        args[args.index("--horizon-sessions") + 1] = value
    else:
        args.extend([flag, value])

    with pytest.raises(SystemExit) as raised:
        main(args)

    assert raised.value.code == 2
    captured = capsys.readouterr()
    assert flag in captured.err
    assert panel_files(market.root) == []
    assert report_dirs(market.root) == []


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"horizon_sessions": 0}, "horizon_sessions must be at least 1"),
        ({"horizon_sessions": -1}, "horizon_sessions must be at least 1"),
        ({"horizon_sessions": 5.0}, "horizon_sessions must be a whole number"),
        ({"horizon_sessions": True}, "horizon_sessions must be a whole number"),
        ({"quantiles": 1}, "quantiles must be at least 2"),
        ({"quantiles": "5"}, "quantiles must be a whole number"),
        ({"min_pairs": 1}, "min_pairs must be at least 2"),
        ({"min_pairs": 2.5}, "min_pairs must be a whole number"),
    ],
)
def test_a_bad_number_is_refused_by_the_function(
    kwargs: dict[str, Any], expected: str, market: Market
) -> None:
    """The importable function refuses the same values, with one clear message."""
    with pytest.raises(FactorResearchError) as raised:
        analyze_market(market, **kwargs)

    assert expected in str(raised.value)
    assert panel_files(market.root) == []
    assert report_dirs(market.root) == []


@pytest.mark.parametrize("value", ["", "   ", 17, object()])
def test_an_unusable_data_root_is_refused(value: Any, market: Market) -> None:
    """A blank or foreign ``data_root`` fails before the batch reads anything."""
    with pytest.raises(FactorResearchError) as raised:
        analyze_factor_history(
            FACTOR_NAME, WINDOW_START, WINDOW_END, horizon_sessions=20, data_root=value
        )

    assert "data_root" in str(raised.value)
    assert panel_files(market.root) == []


@pytest.mark.parametrize("value", ["", "   ", 17])
def test_an_unusable_report_root_is_refused(value: Any, market: Market) -> None:
    """A blank or foreign ``report_root`` is refused, not turned into a directory."""
    with pytest.raises(FactorResearchError) as raised:
        analyze_market(market, report_root=value)

    assert "report_root" in str(raised.value)
    assert report_dirs(market.root) == []


def test_an_empty_schedule_is_refused_before_anything_is_measured(market: Market) -> None:
    """A window with no Thursday has nothing to analyze: a clear refusal, no report."""
    with pytest.raises(FactorResearchError) as raised:
        analyze_factor_history(
            FACTOR_NAME,
            "2024-01-01",
            "2024-01-02",
            horizon_sessions=20,
            data_root=market.root,
            universe_path=market.universe,
        )

    message = str(raised.value)
    assert "selected no decision date" in message
    assert "2024-01-01..2024-01-02" in message
    # The batch ran and wrote its own (empty) manifest; the analysis wrote nothing.
    assert report_dirs(market.root) == []
    assert len(manifest_files(market.root)) == 1
    assert panel_files(market.root) == []


# ---------------------------------------------------------------------------
# 2. The batch hand-off: one call, its own panels, no second schedule
# ---------------------------------------------------------------------------
def test_the_analysis_asks_the_batch_entry_point_for_its_dates(
    market: Market, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One call to ``run_factor_batch`` carries the request through unchanged."""
    manifest = market.root / "factors" / "batch_runs" / "fake.json"
    batch = fake_batch((date(2024, 1, 4), date(2024, 1, 11)), manifest=manifest)
    calls = install_batch(monkeypatch, batch)
    install_calculator(monkeypatch)

    analyze_market(market)

    assert len(calls) == 1
    call = calls[0]
    assert (call["factor_name"], call["start_date"], call["end_date"]) == (
        FACTOR_NAME,
        WINDOW_START,
        WINDOW_END,
    )
    assert call["frequency"] == "weekly"
    assert call["anchor_weekday"] is None
    assert call["data_root"] == market.root
    assert call["universe_path"] == market.universe
    assert call["calendar_name"] is None
    assert call["force"] is False


def test_the_shared_flags_reach_the_batch_unchanged(
    market: Market, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``--force``, ``--calendar``, ``--universe-path`` and ``--data-root`` are passed on."""
    manifest = market.root / "factors" / "batch_runs" / "fake.json"
    batch = fake_batch(
        (date(2024, 1, 4),),
        manifest=manifest,
        calendar_name="XNYS",
    )
    calls = install_batch(monkeypatch, batch)
    instances = install_calculator(monkeypatch)
    pool = write_universe_file(tmp_path / "other.csv", SYMBOLS)

    analyze_market(
        market,
        force=True,
        calendar_name="XNYS",
        universe_path=pool,
        frequency="biweekly",
        anchor_weekday="WED",
    )

    call = calls[0]
    assert call["force"] is True
    assert call["calendar_name"] == "XNYS"
    assert call["universe_path"] == pool
    assert call["frequency"] == "biweekly"
    assert call["anchor_weekday"] == "WED"
    assert [item.calendar_name for item in instances] == ["XNYS"]


def test_only_the_panels_of_this_batch_are_analyzed(
    market: Market, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stored panel outside this batch is never measured, and none is read."""
    stray = market.root / "factors" / "panels" / "factor_name=x" / "as_of_date=2023-06-01.parquet"
    stray.parent.mkdir(parents=True, exist_ok=True)
    stray.write_bytes(b"not read by this analysis")

    manifest = market.root / "factors" / "batch_runs" / "fake.json"
    batch = fake_batch((date(2024, 1, 4), date(2024, 1, 11)), manifest=manifest)
    install_batch(monkeypatch, batch)
    install_calculator(monkeypatch)

    outcome = analyze_market(market)

    assert [panel.as_of_date for panel in outcome.factor_panels] == [
        date(2024, 1, 4),
        date(2024, 1, 11),
    ]
    assert [label.as_of_date for label in outcome.forward_return_panels] == [
        date(2024, 1, 4),
        date(2024, 1, 11),
    ]
    assert read_metadata(outcome.report_path)["selected_dates"] == ["2024-01-04", "2024-01-11"]
    assert stray.read_bytes() == b"not read by this analysis"


def test_the_panels_keep_the_schedule_order(
    market: Market, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The analysed panels are the batch's successes, in the batch's order."""
    manifest = market.root / "factors" / "batch_runs" / "fake.json"
    batch = fake_batch(THURSDAYS, manifest=manifest)
    install_batch(monkeypatch, batch)
    install_calculator(monkeypatch)

    outcome = analyze_market(market)

    assert [panel.as_of_date for panel in outcome.factor_panels] == list(THURSDAYS)
    assert [entry.as_of_date for entry in outcome.panel_statistics] == list(THURSDAYS)
    assert [item.as_of_date for item in outcome.ic_report.observations] == list(THURSDAYS)
    assert outcome.selected_panel_count == len(THURSDAYS) == len(outcome)


def test_a_failed_date_stops_the_analysis(market: Market, monkeypatch: pytest.MonkeyPatch) -> None:
    """One failed date refuses the whole analysis, naming the manifest, and reports nothing."""
    manifest = market.root / "factors" / "batch_runs" / "fake.json"
    batch = fake_batch(THURSDAYS, failed=(THURSDAYS[2],), manifest=manifest)
    install_batch(monkeypatch, batch)
    calculator = install_calculator(monkeypatch)
    record = install_analyzers(monkeypatch)

    with pytest.raises(FactorResearchError) as raised:
        analyze_market(market)

    message = str(raised.value)
    assert "1 of 4 selected dates failed" in message
    assert "2024-01-18" in message
    assert str(manifest) in message
    assert calculator == []
    assert record == {}
    assert report_dirs(market.root) == []


def test_a_batch_whose_manifest_is_missing_stops_the_analysis(
    market: Market, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A report must point at a real manifest, so an unwritten one is refused."""
    from dataclasses import replace

    manifest = market.root / "factors" / "batch_runs" / "fake.json"
    batch = replace(
        fake_batch(THURSDAYS, manifest=manifest), manifest_error="disk full (write refused)"
    )
    install_batch(monkeypatch, batch)
    install_calculator(monkeypatch)

    with pytest.raises(FactorResearchError) as raised:
        analyze_market(market)

    assert "manifest" in str(raised.value)
    assert report_dirs(market.root) == []


# ---------------------------------------------------------------------------
# 3. End to end: real bars, the real cache, the real report
# ---------------------------------------------------------------------------
def test_a_missing_panel_is_computed_and_an_existing_one_is_read_back(market: Market) -> None:
    """The first analysis computes the four panels; the second reads them back."""
    first = analyze_market(market)

    assert [item.cache_status for item in first.batch_outcome.outcomes] == ["miss"] * 4
    assert (first.batch_outcome.computed_count, first.batch_outcome.cached_count) == (4, 0)
    files = panel_files(market.root)
    assert [item.stored_panel_path for item in first.batch_outcome.outcomes] == files
    states = {path: file_state(path) for path in files}
    assert read_metadata(first.report_path)["summary"]["panel_count"] == 4

    second = analyze_market(market)

    assert [item.cache_status for item in second.batch_outcome.outcomes] == ["hit"] * 4
    assert (second.batch_outcome.computed_count, second.batch_outcome.cached_count) == (0, 4)
    assert {path: file_state(path) for path in panel_files(market.root)} == states
    assert second.ic_report.mean_ic == first.ic_report.mean_ic
    assert second.report_path != first.report_path
    assert len(report_dirs(market.root)) == 2


def test_force_recomputes_every_selected_date(market: Market) -> None:
    """``--force`` reaches the batch, so every date is ``forced`` and none is ``hit``."""
    analyze_market(market)
    before = panel_files(market.root)

    forced = analyze_market(market, force=True)

    assert [item.cache_status for item in forced.batch_outcome.outcomes] == ["forced"] * 4
    assert forced.batch_outcome.forced_count == 4
    assert panel_files(market.root) == before


def test_a_damaged_panel_stops_the_whole_analysis(market: Market) -> None:
    """One unreadable panel is a hole in the sample: refuse, and publish no report."""
    first = analyze_market(market)
    damaged = first.batch_outcome.outcomes[2].stored_panel_path
    damaged.write_bytes(b"not a parquet file")

    with pytest.raises(FactorResearchError) as raised:
        analyze_market(market)

    message = str(raised.value)
    assert "1 of 4 selected dates failed" in message
    assert "2024-01-18" in message
    assert "batch_runs" in message
    assert report_dirs(market.root) == [first.report_path]
    assert damaged.read_bytes() == b"not a parquet file"


# ---------------------------------------------------------------------------
# 4. Labels and the three analyses
# ---------------------------------------------------------------------------
def test_each_panel_is_labelled_exactly_once(
    market: Market, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One calculator, one ``compute`` call per panel, always with the same horizon."""
    manifest = market.root / "factors" / "batch_runs" / "fake.json"
    install_batch(monkeypatch, fake_batch(THURSDAYS, manifest=manifest))
    instances = install_calculator(monkeypatch)

    outcome = analyze_market(market, horizon_sessions=17)

    assert len(instances) == 1
    calls = instances[0].calls
    assert [horizon for _, horizon in calls] == [17] * 4
    assert [panel.as_of_date for panel, _ in calls] == list(THURSDAYS)
    labelled = [panel for panel, _ in calls]
    assert same_objects(labelled, outcome.factor_panels)


def test_the_labels_read_the_standardized_bars_and_the_batch_calendar(
    market: Market, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The calculator gets the reader the data root implies and the batch's calendar."""
    manifest = market.root / "factors" / "batch_runs" / "fake.json"
    install_batch(monkeypatch, fake_batch(THURSDAYS, manifest=manifest, calendar_name="XNYS"))
    instances = install_calculator(monkeypatch)

    analyze_market(market)

    reader = instances[0].market_data_reader
    assert isinstance(reader, ParquetMarketDataReader)
    assert reader.root == market.root / "standardized" / "market_bars"
    assert instances[0].calendar_name == "XNYS"


def test_every_label_matches_its_panel(market: Market) -> None:
    """Labels are one-to-one with panels: date, identity, universe order and horizon."""
    outcome = analyze_market(market)

    assert len(outcome.forward_return_panels) == len(outcome.factor_panels)
    for panel, label in zip(outcome.factor_panels, outcome.forward_return_panels, strict=True):
        assert label.as_of_date == panel.as_of_date
        assert (label.factor_name, label.factor_version) == (
            panel.factor_name,
            panel.factor_version,
        )
        assert label.universe == panel.universe
        assert label.symbols == panel.symbols
        assert label.horizon_sessions == 20
        assert label.end_date == session_after(panel.as_of_date, 20)


def test_an_unfinished_label_window_stays_in_the_report(tmp_path: Path) -> None:
    """A date whose future prices are unpublished keeps its rows, with no IC and no zero."""
    root = tmp_path / "data"
    publish(root, end=WINDOW_END)  # bars stop at 2024-01-31
    universe = write_universe_file(tmp_path / "universe.csv", SYMBOLS)

    outcome = analyze_factor_history(
        FACTOR_NAME,
        WINDOW_START,
        WINDOW_END,
        horizon_sessions=5,
        data_root=root,
        universe_path=universe,
    )

    ic = read_table(outcome.report_path / TABLE_FILES["ic_observations"])
    assert len(ic) == 4
    tail = ic.iloc[-1]
    assert tail["as_of_date"].date() == date(2024, 1, 25)
    assert tail["pair_coverage"] == 0.0
    assert pd.isna(tail["ic"])
    assert outcome.valid_ic_count == 3
    assert pd.isna(outcome.ic_report.observations[-1].ic)

    labels = read_table(outcome.report_path / TABLE_FILES["forward_returns"])
    assert len(labels) == 4 * len(SYMBOLS)
    tail_rows = labels[labels["as_of_date"] == pd.Timestamp("2024-01-25")]
    assert len(tail_rows) == len(SYMBOLS)
    assert not tail_rows["is_available"].any()
    assert tail_rows["forward_return"].isna().all()
    # The earlier dates stay usable: a missing tail is not an error and not zero.
    head_rows = labels[labels["as_of_date"] == pd.Timestamp("2024-01-04")]
    assert head_rows["is_available"].all()
    assert head_rows["forward_return"].notna().all()


def test_the_three_analyses_receive_the_same_panels_and_labels(
    market: Market, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Quick, IC and quantile are handed one sample, not three views of it."""
    record = install_analyzers(monkeypatch)

    outcome = analyze_market(market)

    for key in ("quick", "ic", "quantile"):
        assert key in record
    assert same_objects(record["quick"], outcome.factor_panels)
    ic_panels, ic_labels = record["ic"]
    quantile_panels, quantile_labels = record["quantile"]
    assert same_objects(ic_panels, outcome.factor_panels)
    assert same_objects(ic_labels, outcome.forward_return_panels)
    assert same_objects(quantile_panels, ic_panels)
    assert same_objects(quantile_labels, ic_labels)


def test_a_longer_horizon_moves_the_label_end_date(market: Market) -> None:
    """The horizon, not the cadence, decides where a label ends -- and it is recorded."""
    short = analyze_market(market, horizon_sessions=5)
    long = analyze_market(market, horizon_sessions=20)

    short_labels = read_table(short.report_path / TABLE_FILES["forward_returns"])
    long_labels = read_table(long.report_path / TABLE_FILES["forward_returns"])
    first_day = pd.Timestamp(THURSDAYS[0])
    assert short_labels.loc[short_labels["as_of_date"] == first_day, "end_date"].iloc[0].date() == (
        session_after(THURSDAYS[0], 5)
    )
    assert long_labels.loc[long_labels["as_of_date"] == first_day, "end_date"].iloc[0].date() == (
        session_after(THURSDAYS[0], 20)
    )
    assert read_metadata(short.report_path)["horizon_sessions"] == 5
    assert read_metadata(long.report_path)["horizon_sessions"] == 20
    # Same schedule, different labels: the cadence did not change with the horizon.
    assert (
        read_metadata(short.report_path)["selected_dates"]
        == read_metadata(long.report_path)["selected_dates"]
    )


def test_overlapping_label_windows_are_allowed(market: Market) -> None:
    """``weekly + 20 sessions`` overlaps on purpose; every date is still measured."""
    outcome = analyze_market(market, frequency="weekly", anchor_weekday="THU", horizon_sessions=20)

    labels = outcome.forward_return_panels
    assert len(labels) == 4
    for current, following in zip(labels, labels[1:], strict=False):
        # The next decision date is inside the previous label window, so the ICs
        # are not independent observations -- the README says so explicitly.
        assert current.end_date > following.as_of_date
    assert outcome.valid_ic_count == 4


def test_a_hard_refusal_of_the_ic_layer_propagates(
    market: Market, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Labels of another date are refused by the analyzer, not silently re-paired."""
    manifest = market.root / "factors" / "batch_runs" / "fake.json"
    install_batch(monkeypatch, fake_batch(THURSDAYS, manifest=manifest))

    def mislabel(calculator: FakeCalculator, panel: Any, horizon: int) -> ForwardReturnPanel:
        return fake_labels(fake_panel(date(2023, 6, 1)), horizon)

    install_calculator(monkeypatch, behaviour=mislabel)

    with pytest.raises(ICAnalysisError) as raised:
        analyze_market(market)

    assert "2023-06-01" in str(raised.value)
    assert report_dirs(market.root) == []


# ---------------------------------------------------------------------------
# 5. The report: one directory, the five tables, a JSON-safe record
# ---------------------------------------------------------------------------
def test_the_report_holds_the_metadata_and_every_table(market: Market) -> None:
    """A run publishes exactly the documented files, in a fresh directory."""
    outcome = analyze_market(market)

    assert report_files(outcome.report_path) == sorted([METADATA_FILE, *TABLE_FILES.values()])
    assert outcome.report_path.parent == report_root(market.root)
    assert outcome.report_path.name == read_metadata(outcome.report_path)["research_run_id"]
    assert staging_dirs(market.root) == []
    assert [item.name for item in report_dirs(market.root)] == [outcome.report_path.name]


def test_every_table_keeps_the_columns_of_its_object(market: Market) -> None:
    """The CSVs are renderings of the objects, with the objects' own columns."""
    outcome = analyze_market(market)
    report = outcome.report_path

    statistics = read_table(report / TABLE_FILES["panel_statistics"])
    assert list(statistics.columns) == list(STATISTICS_COLUMNS)
    assert len(statistics) == 4

    labels = read_table(report / TABLE_FILES["forward_returns"])
    assert list(labels.columns) == list(FORWARD_RETURN_COLUMNS)
    assert len(labels) == 4 * len(SYMBOLS)

    ic = read_table(report / TABLE_FILES["ic_observations"])
    assert list(ic.columns) == list(IC_OBSERVATION_COLUMNS)
    assert len(ic) == len(outcome.ic_report.observations)

    observations = read_table(report / TABLE_FILES["quantile_observations"])
    assert list(observations.columns) == list(QUANTILE_OBSERVATION_COLUMNS)
    assert len(observations) == len(outcome.quantile_report.observations)

    spreads = read_table(report / TABLE_FILES["quantile_spreads"])
    assert list(spreads.columns) == list(QUANTILE_SPREAD_COLUMNS)
    assert len(spreads) == len(outcome.quantile_report.spreads) == 4


def test_the_metadata_records_the_whole_request(market: Market) -> None:
    """Schedule, horizon, knobs, the batch manifest and the IC summary are all there."""
    outcome = analyze_market(market, horizon_sessions=20, quantiles=3, min_pairs=4)

    metadata = read_metadata(outcome.report_path)
    assert metadata["format_version"] == analyze.REPORT_FORMAT_VERSION == 1
    assert metadata["factor_name"] == FACTOR_NAME
    assert metadata["factor_version"] == outcome.batch_outcome.factor_version
    assert metadata["requested_start_date"] == WINDOW_START
    assert metadata["requested_end_date"] == WINDOW_END
    assert metadata["frequency"] == "weekly"
    assert metadata["anchor_weekday"] == "THU"
    assert metadata["biweekly_anchor"] is None
    assert metadata["calendar_name"] == "XNYS"
    assert metadata["horizon_sessions"] == 20
    assert metadata["quantiles"] == 3
    assert metadata["min_pairs"] == 4
    assert metadata["selected_dates"] == [day.isoformat() for day in THURSDAYS]
    assert metadata["created_at_utc"].endswith("+00:00")

    manifest = metadata["batch_manifest_path"]
    assert "/" in manifest and "\\" not in manifest
    assert not Path(manifest).is_absolute()
    assert (market.root / manifest).is_file()

    summary = metadata["summary"]
    assert summary["panel_count"] == 4
    assert summary["valid_ic_count"] == outcome.valid_ic_count
    assert summary["mean_ic"] == pytest.approx(outcome.mean_ic)
    assert summary["ic_std"] == pytest.approx(outcome.ic_std)
    assert summary["icir"] == pytest.approx(outcome.icir)


def test_a_missing_number_reaches_the_metadata_as_null(
    market: Market, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One date cannot have a deviation or an ICIR: the JSON says ``null``, never ``NaN``."""
    manifest = market.root / "factors" / "batch_runs" / "fake.json"
    install_batch(monkeypatch, fake_batch((date(2024, 1, 4),), manifest=manifest))
    install_calculator(monkeypatch)

    outcome = analyze_market(market)

    metadata = read_metadata(outcome.report_path)
    summary = metadata["summary"]
    assert summary["panel_count"] == 1
    assert summary["valid_ic_count"] == 1
    assert summary["mean_ic"] == pytest.approx(-1.0)
    assert summary["ic_std"] is None
    assert summary["icir"] is None

    raw = (outcome.report_path / METADATA_FILE).read_text(encoding="utf-8")
    assert '"ic_std": null' in raw
    assert '"icir": null' in raw
    assert "NaN" not in raw and "Infinity" not in raw


def test_two_runs_write_two_independent_reports(market: Market) -> None:
    """The same request twice publishes two complete, distinct directories."""
    first = analyze_market(market)
    second = analyze_market(market)

    assert first.report_path != second.report_path
    for outcome in (first, second):
        assert report_files(outcome.report_path) == sorted([METADATA_FILE, *TABLE_FILES.values()])
        assert read_metadata(outcome.report_path)["research_run_id"] == outcome.report_path.name
    assert (
        read_metadata(first.report_path)["batch_manifest_path"]
        != read_metadata(second.report_path)["batch_manifest_path"]
    )


def test_a_report_that_cannot_be_published_leaves_nothing_behind(
    market: Market, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed rename removes the staging directory and keeps panels and manifest."""
    manifest = market.root / "factors" / "batch_runs" / "fake.json"
    install_batch(monkeypatch, fake_batch(THURSDAYS, manifest=manifest))
    install_calculator(monkeypatch)
    panels_before = panel_files(market.root)
    manifest_before = manifest_files(market.root)

    def refuse(staging: Path, final: Path) -> None:
        raise OSError("the rename was refused")

    monkeypatch.setattr(analyze, "_rename_report", refuse)

    with pytest.raises(FactorResearchError) as raised:
        analyze_market(market)

    assert "could not publish the research report" in str(raised.value)
    assert report_dirs(market.root) == []
    assert staging_dirs(market.root) == []
    # The earlier stage is untouched: panels and manifests are not rolled back.
    assert panel_files(market.root) == panels_before
    assert manifest_files(market.root) == manifest_before


def test_a_report_root_that_is_a_file_is_refused(market: Market) -> None:
    """The analysis refuses to publish into something that is not a directory."""
    blocked = report_root(market.root)
    blocked.parent.mkdir(parents=True, exist_ok=True)
    blocked.write_text("not a directory", encoding="utf-8")

    with pytest.raises(FactorResearchError) as raised:
        analyze_market(market)

    assert "cannot create the research-report directory" in str(raised.value)
    assert blocked.read_text(encoding="utf-8") == "not a directory"
    # The panels were computed and stored by the batch before the publish failed.
    assert len(panel_files(market.root)) == 4
    assert len(manifest_files(market.root)) == 1


def test_write_report_refuses_an_incomplete_table_set(tmp_path: Path) -> None:
    """A caller that forgets a table is told which one, before anything is written."""
    root = tmp_path / "reports"
    tables = {name: pd.DataFrame({"as_of_date": []}) for name in TABLE_FILES}
    del tables["quantile_spreads"]

    with pytest.raises(FactorResearchError) as raised:
        analyze.write_report(root, "run", {"format_version": 1}, tables)

    assert "quantile_spreads" in str(raised.value)
    assert not root.exists()


def test_write_report_refuses_to_overwrite_a_directory(tmp_path: Path) -> None:
    """A run id that already exists is refused rather than merged into."""
    root = tmp_path / "reports"
    existing = root / "run"
    existing.mkdir(parents=True)
    tables = {name: pd.DataFrame({"as_of_date": []}) for name in TABLE_FILES}

    with pytest.raises(FactorResearchError) as raised:
        analyze.write_report(root, "run", {"format_version": 1}, tables)

    assert "already exists" in str(raised.value)
    assert list(existing.iterdir()) == []


# ---------------------------------------------------------------------------
# 6. The command line: summary, exit codes, error style
# ---------------------------------------------------------------------------
def test_the_cli_prints_the_summary_and_exits_zero(
    market: Market, capsys: pytest.CaptureFixture
) -> None:
    """A complete analysis prints the documented block and nothing else."""
    code, out, err = run_cli(arguments(market.root, market.universe), capsys)

    assert (code, err) == (0, "")
    fields = summary_fields(out)
    assert list(fields) == list(SUMMARY_FIELDS)
    assert fields["factor_name"] == FACTOR_NAME
    assert fields["frequency"] == "weekly"
    assert fields["anchor_weekday"] == "THU"
    assert fields["horizon_sessions"] == "20"
    assert fields["quantiles"] == "5"
    assert fields["min_pairs"] == "2"
    assert fields["selected_panel_count"] == "4"
    assert fields["computed_count"] == "4"
    assert fields["cached_count"] == "0"
    assert fields["forced_count"] == "0"
    assert fields["valid_ic_count"] == "4"
    assert fields["mean_ic"] != analyze.MISSING_TEXT
    assert out.splitlines()[-1].startswith("batch_manifest_path: ")

    report = Path(fields["report_path"])
    assert report.is_dir()
    assert report_files(report) == sorted([METADATA_FILE, *TABLE_FILES.values()])
    assert Path(fields["batch_manifest_path"]).is_file()
    # The terminal gets the summary, never a table or a per-symbol label.
    assert "as_of_date" not in out
    assert "forward_return" not in out


def test_the_cli_prints_an_undefined_number_as_na(
    market: Market, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """A single date has no deviation and no ICIR: the summary says ``n/a``."""
    manifest = market.root / "factors" / "batch_runs" / "fake.json"
    install_batch(monkeypatch, fake_batch((date(2024, 1, 4),), manifest=manifest))
    install_calculator(monkeypatch)

    code, out, err = run_cli(arguments(market.root, market.universe), capsys)

    assert (code, err) == (0, "")
    fields = summary_fields(out)
    assert fields["selected_panel_count"] == "1"
    assert fields["valid_ic_count"] == "1"
    assert fields["ic_std"] == analyze.MISSING_TEXT
    assert fields["icir"] == analyze.MISSING_TEXT


def test_the_cli_reports_a_failed_date_as_one_line(
    market: Market, capsys: pytest.CaptureFixture
) -> None:
    """A hole in the sample is one ``error:`` line on stderr, with exit code ``1``."""
    first = analyze_market(market)
    first.batch_outcome.outcomes[2].stored_panel_path.write_bytes(b"not a parquet file")

    code, out, err = run_cli(arguments(market.root, market.universe), capsys)

    assert code == 1
    assert out == ""
    assert err.count("\n") == 1
    assert err.startswith("error: ")
    assert "1 of 4 selected dates failed" in err
    assert "Traceback" not in err
    assert report_dirs(market.root) == [first.report_path]


def test_the_cli_reports_an_unknown_factor_as_one_line(
    market: Market, capsys: pytest.CaptureFixture
) -> None:
    """An unknown factor fails before any panel, with one clear line."""
    args = arguments(market.root, market.universe, **{"--factor": "no_such_factor"})

    code, out, err = run_cli(args, capsys)

    assert code == 1
    assert out == ""
    assert err.startswith("error: ")
    assert "no_such_factor" in err
    assert "Traceback" not in err
    assert panel_files(market.root) == []
    assert report_dirs(market.root) == []


def test_the_parser_expects_exactly_the_documented_flags() -> None:
    """The command accepts the flags the README documents -- and nothing else."""
    parser = build_parser()

    assert {action.dest for action in parser._actions} == EXPECTED_DESTS
    args = parser.parse_args(
        [
            "--factor",
            FACTOR_NAME,
            "--start",
            WINDOW_START,
            "--end",
            WINDOW_END,
            "--horizon-sessions",
            "20",
        ]
    )
    assert args.frequency == analyze.DEFAULT_FREQUENCY == "weekly"
    assert args.anchor_weekday is None
    assert args.quantiles == analyze.DEFAULT_QUANTILES
    assert args.min_pairs is None
    assert (args.data_root, args.universe_path, args.calendar, args.report_root) == (
        None,
        None,
        None,
        None,
    )
    assert args.force is False


def test_the_documented_command_runs_in_a_real_process(market: Market) -> None:
    """``python scripts/analyze_factor.py`` works from the project root as documented."""
    args = [
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
        "--horizon-sessions",
        "20",
        "--data-root",
        str(market.root),
        "--universe-path",
        str(market.universe),
    ]

    completed = subprocess.run(
        args, cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=600
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    fields = summary_fields(completed.stdout)
    assert fields["selected_panel_count"] == "4"
    assert fields["horizon_sessions"] == "20"
    assert Path(fields["report_path"]).is_dir()


# ---------------------------------------------------------------------------
# 7. The module boundary: composition only
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


def test_the_script_implements_no_formula_no_store_and_no_network() -> None:
    """It composes the layers it may use, and imports nothing that owns their work."""
    modules = imported_modules(script_tree())
    allowed = {
        "__future__",
        "argparse",
        "json",
        "math",
        "os",
        "shutil",
        "sys",
        "tempfile",
        "collections.abc",
        "dataclasses",
        "datetime",
        "pathlib",
        "typing",
        "pandas",
        "data_sys.config",
        "data_sys.utils",
        "factor_sys",
        "factor_sys.errors",
        "scripts.run_factor_batch",
    }
    assert modules <= allowed, f"the analysis must not import {sorted(modules - allowed)}"
    assert {
        "pandas",
        "factor_sys",
        "data_sys.config",
        "data_sys.utils",
        "scripts.run_factor_batch",
    } <= modules

    for forbidden in (
        "factors",
        "yfinance",
        "requests",
        "urllib",
        "http",
        "socket",
        "subprocess",
        "duckdb",
        "pyarrow",
        "data_sys.providers",
        "data_sys.update",
        "data_sys.planner",
        "data_sys.storage",
        "data_sys.data_query",
        "data_sys.batch_query",
        "scripts.run_factor",
    ):
        assert not any(
            module == forbidden or module.startswith(f"{forbidden}.") for module in modules
        ), f"the analysis entry point must not import {forbidden!r}"


def test_the_script_delegates_the_whole_schedule_and_panel_work() -> None:
    """The dates, the panels and the labels come from the layers that own them."""
    names = referenced_names(script_tree())

    assert {
        "run_factor_batch",
        "ForwardReturnCalculator",
        "ParquetMarketDataReader",
        "QuickAnalyzer",
        "ICAnalyzer",
        "QuantileAnalyzer",
        "new_run_id",
        "utc_now_iso",
        "replace",
    } <= names
    for forbidden in (
        "decision_schedule",
        "select_decision_dates",
        "run_factor",
        "default_factor_registry",
        "FactorRunner",
        "FactorContext",
        "StaticUniverseProvider",
        "ParquetFactorStore",
        "read_range",
        "read_partitioned",
        "write_partitioned",
    ):
        assert forbidden not in names, f"the analysis must not do the work itself: {forbidden!r}"


def test_the_script_recomputes_no_statistic() -> None:
    """No Spearman ranking, no quantile grouping, no moment is computed here."""
    names = referenced_names(script_tree())

    for forbidden in (
        "spearman",
        "_spearman",
        "pearson",
        "_pearson",
        "rankdata",
        "rank",
        "corr",
        "cov",
        "ddof",
        "quantile_assignments",
        "groupby",
        "describe",
    ):
        assert forbidden not in names, f"the analysis must not recompute {forbidden!r}"


def test_the_script_builds_only_the_report_paths() -> None:
    """The only paths this module names are its own report's, never a panel's."""
    tree = script_tree()
    constants = string_constants(tree)

    assert not any(".parquet" in value for value in constants)
    assert not any("year=" in value or "factor_name=" in value for value in constants)
    assert {
        "metadata.json",
        "panel_statistics.csv",
        "forward_returns.csv",
        "ic_observations.csv",
        "quantile_observations.csv",
        "quantile_spreads.csv",
    } <= constants


def test_the_report_location_is_declared_in_one_place() -> None:
    """The layout is a constant, derived from the data root instead of hard-coded."""
    assert analyze.RESEARCH_DIRECTORY == ("factors", "research_runs")
    assert analyze.REPORT_FORMAT_VERSION == 1
    assert analyze.METADATA_FILE == "metadata.json"
    assert Path(*analyze.RESEARCH_DIRECTORY, "run", METADATA_FILE).as_posix() == (
        "factors/research_runs/run/metadata.json"
    )


def test_the_report_root_defaults_to_the_data_root(market: Market, tmp_path: Path) -> None:
    """Without ``--report-root`` the report follows ``--data-root``; with it, it does not."""
    default = analyze_market(market)
    assert default.report_path.parent == market.root.joinpath(*analyze.RESEARCH_DIRECTORY)

    elsewhere = tmp_path / "elsewhere"
    custom = analyze_market(market, report_root=elsewhere)
    assert custom.report_path.parent == elsewhere
    assert report_dirs(market.root) == [default.report_path]


# ---------------------------------------------------------------------------
# 8. The result object: frozen, self-checking, self-describing
# ---------------------------------------------------------------------------
def test_the_outcome_is_immutable_and_reports_its_own_counts(
    market: Market, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every number of the outcome is derived from the panels it really holds."""
    manifest = market.root / "factors" / "batch_runs" / "fake.json"
    install_batch(monkeypatch, fake_batch(THURSDAYS, manifest=manifest))
    install_calculator(monkeypatch)

    outcome = analyze_market(market, horizon_sessions=7)

    with pytest.raises(FrozenInstanceError):
        outcome.report_path = Path("somewhere-else")  # type: ignore[misc]

    assert outcome.selected_panel_count == len(outcome) == 4
    assert outcome.factor_name == FACTOR_NAME
    assert outcome.factor_version == outcome.batch_outcome.factor_version
    assert outcome.horizon_sessions == 7
    assert outcome.valid_ic_count == 4
    assert outcome.mean_ic == outcome.ic_report.mean_ic
    assert outcome.ic_std == outcome.ic_report.ic_std
    assert outcome.icir == outcome.ic_report.icir
    assert outcome.mean_ic == pytest.approx(-1.0)

    text = repr(outcome)
    assert FACTOR_NAME in text
    assert "dates=4" in text
    assert "horizon_sessions=7" in text


def test_the_outcome_holds_the_panels_the_batch_recorded(
    market: Market, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The analysed sample *is* the batch's sample: same dates, same objects."""
    manifest = market.root / "factors" / "batch_runs" / "fake.json"
    batch = fake_batch(THURSDAYS, manifest=manifest)
    install_batch(monkeypatch, batch)
    install_calculator(monkeypatch)

    outcome = analyze_market(market)

    recorded = tuple(item.panel for item in batch.outcomes)
    assert same_objects(recorded, outcome.factor_panels)
    assert tuple(panel.as_of_date for panel in recorded) == batch.selected_dates
    assert tuple(panel.as_of_date for panel in outcome.factor_panels) == batch.selected_dates
    assert outcome.batch_outcome is batch
    assert batch.failures == ()
    assert batch.succeeded_count == batch.selection_size == outcome.selected_panel_count


def test_an_outcome_whose_panel_is_a_look_alike_is_refused(
    market: Market, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rebuilt panel with the same date, factor and version is another sample."""
    manifest = market.root / "factors" / "batch_runs" / "fake.json"
    install_batch(monkeypatch, fake_batch(THURSDAYS, manifest=manifest))
    install_calculator(monkeypatch)
    outcome = analyze_market(market)

    look_alike = fake_panel(THURSDAYS[1])
    assert look_alike.as_of_date == outcome.factor_panels[1].as_of_date
    assert look_alike is not outcome.factor_panels[1]

    panels = (outcome.factor_panels[0], look_alike, *outcome.factor_panels[2:])
    with pytest.raises(FactorResearchError) as raised:
        replace(outcome, factor_panels=panels)

    message = str(raised.value)
    assert "factor_panels[1] is not the panel the batch recorded for 2024-01-11" in message
    assert "is a different sample" in message
    assert "factor_panels[0]" not in message


def test_an_outcome_of_another_batch_is_refused(
    market: Market, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Panels of one run cannot be published under the record of another run."""
    manifest = market.root / "factors" / "batch_runs" / "fake.json"
    install_batch(monkeypatch, fake_batch(THURSDAYS, manifest=manifest))
    install_calculator(monkeypatch)
    outcome = analyze_market(market)

    other = fake_batch(WEDNESDAYS, manifest=manifest)
    with pytest.raises(FactorResearchError) as raised:
        replace(outcome, batch_outcome=other)

    message = str(raised.value)
    assert "factor_panels holds 4 panel(s) for 5 successful date(s)" in message
    assert "is not the panel the batch recorded for 2024-01-03" in message


def test_an_outcome_of_a_batch_with_a_failed_date_is_refused(
    market: Market, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A hole in the sample is not a smaller sample, even after the fact."""
    manifest = market.root / "factors" / "batch_runs" / "fake.json"
    install_batch(monkeypatch, fake_batch(THURSDAYS, manifest=manifest))
    install_calculator(monkeypatch)
    outcome = analyze_market(market)

    holed = fake_batch(THURSDAYS, failed=(THURSDAYS[1],), manifest=manifest)
    with pytest.raises(FactorResearchError) as raised:
        replace(outcome, batch_outcome=holed)

    message = str(raised.value)
    assert "records 1 failed date(s) -- 2024-01-11" in message
    assert "a report covers every selected date or none" in message


def test_an_outcome_that_skips_a_selected_date_is_refused(
    market: Market, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dropping one date from the panels would silently shorten the study."""
    manifest = market.root / "factors" / "batch_runs" / "fake.json"
    install_batch(monkeypatch, fake_batch(THURSDAYS, manifest=manifest))
    install_calculator(monkeypatch)
    outcome = analyze_market(market)

    dropped = (*outcome.factor_panels[:2], outcome.factor_panels[3])
    with pytest.raises(FactorResearchError) as raised:
        replace(outcome, factor_panels=dropped)

    message = str(raised.value)
    assert "factor_panels holds 3 panel(s) for 4 successful date(s)" in message
    assert "is not the panel the batch recorded for 2024-01-18" in message


def test_an_outcome_of_a_batch_about_another_factor_version_is_refused(
    market: Market, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The report takes its factor identity from the batch, so it must match."""
    manifest = market.root / "factors" / "batch_runs" / "fake.json"
    install_batch(monkeypatch, fake_batch(THURSDAYS, manifest=manifest))
    install_calculator(monkeypatch)
    outcome = analyze_market(market)

    relabelled = fake_batch(THURSDAYS, manifest=manifest, factor_version="9.9.9")
    with pytest.raises(FactorResearchError) as raised:
        replace(outcome, batch_outcome=relabelled)

    message = str(raised.value)
    assert (
        "the batch measured momentum_12_1/9.9.9, but the panels are momentum_12_1/1.0.0" in message
    )
    assert "the report takes its factor name from the batch record" in message


def test_an_outcome_of_a_batch_without_a_manifest_is_refused(
    market: Market, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The report names the manifest, so an unwritten record cannot be published."""
    manifest = market.root / "factors" / "batch_runs" / "fake.json"
    install_batch(monkeypatch, fake_batch(THURSDAYS, manifest=manifest))
    install_calculator(monkeypatch)
    outcome = analyze_market(market)

    unwritten = replace(outcome.batch_outcome, manifest_error="the disk is full")
    with pytest.raises(FactorResearchError) as raised:
        replace(outcome, batch_outcome=unwritten)

    message = str(raised.value)
    assert "its manifest could not be written (the disk is full)" in message
    assert "would leave the run untraceable" in message


def test_an_outcome_whose_label_belongs_to_another_date_is_refused(
    market: Market, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A label of another session cannot be smuggled into an analysis."""
    manifest = market.root / "factors" / "batch_runs" / "fake.json"
    install_batch(monkeypatch, fake_batch(THURSDAYS, manifest=manifest))
    install_calculator(monkeypatch)
    outcome = analyze_market(market)

    shifted = tuple(fake_labels(fake_panel(day + timedelta(days=1)), 20) for day in THURSDAYS)
    with pytest.raises(FactorResearchError) as raised:
        replace(outcome, forward_return_panels=shifted)

    assert "forward_return_panels[0]" in str(raised.value)


def test_an_outcome_with_a_foreign_object_is_refused(
    market: Market, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tuple is checked item by item: a symbol name is not a panel."""
    manifest = market.root / "factors" / "batch_runs" / "fake.json"
    install_batch(monkeypatch, fake_batch(THURSDAYS, manifest=manifest))
    install_calculator(monkeypatch)
    outcome = analyze_market(market)

    with pytest.raises(FactorResearchError) as raised:
        replace(outcome, factor_panels=("AAPL", "MSFT"))

    assert "factor_panels[0] is a str, not a FactorPanel" in str(raised.value)


def test_an_outcome_with_an_impossible_pair_threshold_is_refused(
    market: Market, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``min_pairs`` is a policy of the run, and a correlation cannot use one pair."""
    manifest = market.root / "factors" / "batch_runs" / "fake.json"
    install_batch(monkeypatch, fake_batch(THURSDAYS, manifest=manifest))
    install_calculator(monkeypatch)
    outcome = analyze_market(market)

    with pytest.raises(FactorResearchError) as raised:
        replace(outcome, min_pairs=1)

    assert "min_pairs must be at least 2" in str(raised.value)


def test_an_outcome_whose_ic_report_measures_another_horizon_is_refused(
    market: Market, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One analysis measures one horizon, so a foreign report cannot be grafted on."""
    manifest = market.root / "factors" / "batch_runs" / "fake.json"
    install_batch(monkeypatch, fake_batch(THURSDAYS, manifest=manifest))
    install_calculator(monkeypatch)
    outcome = analyze_market(market)

    short_labels = tuple(fake_labels(panel, 5) for panel in outcome.factor_panels)
    short_report = ICAnalyzer().analyze(outcome.factor_panels, short_labels)
    with pytest.raises(FactorResearchError) as raised:
        replace(outcome, ic_report=short_report)

    message = str(raised.value)
    assert "ic_report measures 5 session(s) ahead" in message
    assert "ic_report holds 4 observation(s) for 4 panel(s)" not in message


def test_the_outcome_rejects_an_empty_analysis() -> None:
    """An outcome of no date describes nothing, and is refused at construction."""
    day = THURSDAYS[0]
    panel = fake_panel(day)
    labels = (fake_labels(panel, 5),)

    with pytest.raises(FactorResearchError) as raised:
        FactorResearchOutcome(
            batch_outcome=fake_batch((day,), manifest=Path("manifest.json")),
            factor_panels=(),
            forward_return_panels=(),
            panel_statistics=(),
            ic_report=ICAnalyzer().analyze((panel,), labels),
            quantile_report=QuantileAnalyzer().analyze((panel,), labels),
            min_pairs=2,
            report_path=Path("reports/run"),
        )

    assert "at least one panel" in str(raised.value)


# ---------------------------------------------------------------------------
# 9. The two rendering helpers: the terminal block and the metadata payload
# ---------------------------------------------------------------------------
def test_the_summary_names_every_field_in_a_stable_order(
    market: Market, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``format_summary`` is the only rendering, and its order never moves."""
    manifest = market.root / "factors" / "batch_runs" / "fake.json"
    install_batch(monkeypatch, fake_batch((THURSDAYS[0],), manifest=manifest))
    install_calculator(monkeypatch)

    text = format_summary(analyze_market(market, horizon_sessions=9))

    fields = summary_fields(text)
    assert list(fields) == list(SUMMARY_FIELDS)
    assert fields["horizon_sessions"] == "9"
    assert fields["selected_panel_count"] == "1"
    assert fields["min_pairs"] == "2"
    assert fields["quantiles"] == "5"
    assert fields["ic_std"] == analyze.MISSING_TEXT
    assert fields["icir"] == analyze.MISSING_TEXT
    assert text.splitlines()[-1].startswith("batch_manifest_path: ")


def test_the_metadata_payload_is_json_safe_and_portable(market: Market, tmp_path: Path) -> None:
    """A path inside the data root is stored relatively; a missing number is ``null``."""
    day = THURSDAYS[0]
    batch = fake_batch((day,), manifest=market.root / "factors" / "batch_runs" / "fake.json")
    panel = fake_panel(day)
    labels = (fake_labels(panel, 20),)
    ic_report = ICAnalyzer().analyze((panel,), labels)

    payload = report_metadata(
        batch,
        research_run_id="20240102T030405Z_9f8e7d",
        created_at_utc="2024-01-02T03:04:05+00:00",
        data_root=market.root,
        horizon_sessions=20,
        quantiles=5,
        min_pairs=2,
        ic_report=ic_report,
    )

    assert payload["batch_manifest_path"] == "factors/batch_runs/fake.json"
    assert payload["anchor_weekday"] == "THU"
    assert payload["selected_dates"] == ["2024-01-04"]
    assert payload["summary"] == {
        "panel_count": 1,
        "valid_ic_count": 1,
        "mean_ic": -1.0,
        "ic_std": None,
        "icir": None,
    }
    # ``allow_nan=False`` would raise here if a NaN had crept in.
    assert json.loads(json.dumps(payload, allow_nan=False)) == payload

    outside = report_metadata(
        fake_batch((day,), manifest=tmp_path / "elsewhere" / "manifest.json"),
        research_run_id="run",
        created_at_utc="2024-01-02T03:04:05+00:00",
        data_root=market.root,
        horizon_sessions=20,
        quantiles=5,
        min_pairs=2,
        ic_report=ic_report,
    )
    # A manifest outside the data root keeps its absolute path, as the batch does.
    assert Path(outside["batch_manifest_path"]).is_absolute()
