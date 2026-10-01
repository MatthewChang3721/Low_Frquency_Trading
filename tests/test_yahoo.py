"""Tests for :mod:`data_sys.providers.yahoo`.

``yfinance.download`` is always monkeypatched: the test-suite never touches the
network.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from data_sys.errors import DownloadError
from data_sys.providers import yahoo
from data_sys.providers.yahoo import YahooFinanceProvider
from tests.helpers import make_raw_frame

START = date(2020, 1, 2)
END = date(2020, 1, 10)


def test_download_arguments_and_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = make_raw_frame(periods=5)
    captured: dict[str, object] = {}

    def fake_download(tickers, **kwargs):
        captured["tickers"] = tickers
        captured.update(kwargs)
        return frame

    monkeypatch.setattr(yahoo.yf, "download", fake_download)

    result = YahooFinanceProvider(max_attempts=1, wait_seconds=0.0).fetch_bars("aapl", START, END)

    assert captured["tickers"] == "AAPL"
    assert captured["auto_adjust"] is False
    assert captured["multi_level_index"] is False
    assert captured["progress"] is False
    assert captured["start"] == "2020-01-02"
    assert captured["end"] == "2020-01-11"  # yfinance end is exclusive

    assert result.meta["provider"] == "yahoo_finance"
    assert result.meta["symbol"] == "AAPL"
    assert result.meta["vendor_symbol"] == "AAPL"  # nothing was rewritten
    assert result.meta["raw_rows"] == 5
    assert list(result.frame.columns) == [
        "Adj Close",
        "Close",
        "High",
        "Low",
        "Open",
        "Volume",
    ]


def test_multiindex_columns_are_flattened(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = make_raw_frame(periods=4)
    frame.columns = pd.MultiIndex.from_product([list(frame.columns), ["AAPL"]])

    monkeypatch.setattr(yahoo.yf, "download", lambda *a, **k: frame)

    result = YahooFinanceProvider(wait_seconds=0.0).fetch_bars("AAPL", START, END)

    assert not isinstance(result.frame.columns, pd.MultiIndex)
    assert list(result.frame.columns) == [
        "Adj Close",
        "Close",
        "High",
        "Low",
        "Open",
        "Volume",
    ]


def test_empty_response_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(yahoo.yf, "download", lambda *a, **k: pd.DataFrame())

    with pytest.raises(DownloadError):
        YahooFinanceProvider(wait_seconds=0.0).fetch_bars("AAPL", START, END)


def test_missing_columns_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = make_raw_frame(periods=3).drop(columns=["Adj Close"])
    monkeypatch.setattr(yahoo.yf, "download", lambda *a, **k: frame)

    with pytest.raises(DownloadError):
        YahooFinanceProvider(wait_seconds=0.0).fetch_bars("AAPL", START, END)


def test_a_dotted_symbol_falls_back_to_the_vendor_spelling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``BRK.B`` is canonical here; Yahoo only answers to ``BRK-B``."""
    frame = make_raw_frame(periods=3)
    seen: list[str] = []

    def fake_download(tickers, **kwargs):
        seen.append(tickers)
        return pd.DataFrame() if tickers == "BRK.B" else frame

    monkeypatch.setattr(yahoo.yf, "download", fake_download)

    result = YahooFinanceProvider(max_attempts=1, wait_seconds=0.0).fetch_bars(
        "brk.b", START, END
    )

    assert seen == ["BRK.B", "BRK-B"]
    assert result.meta["symbol"] == "BRK.B"         # the canonical spelling survives
    assert result.meta["vendor_symbol"] == "BRK-B"  # ...this is what answered
    assert len(result.frame) == 3


def test_the_canonical_spelling_is_never_overridden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An exchange suffix (``RY.TO``) is fetched as-is: no guessing, no shadowing."""
    frame = make_raw_frame(periods=3)
    seen: list[str] = []

    def fake_download(tickers, **kwargs):
        seen.append(tickers)
        return frame

    monkeypatch.setattr(yahoo.yf, "download", fake_download)

    result = YahooFinanceProvider(max_attempts=1, wait_seconds=0.0).fetch_bars(
        "RY.TO", START, END
    )

    assert seen == ["RY.TO"]
    assert result.meta["vendor_symbol"] == "RY.TO"


def test_a_response_without_the_required_columns_also_triggers_the_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    incomplete = make_raw_frame(periods=3).drop(columns=["Adj Close"])
    complete = make_raw_frame(periods=3)

    def fake_download(tickers, **kwargs):
        return incomplete if tickers == "BRK.B" else complete

    monkeypatch.setattr(yahoo.yf, "download", fake_download)

    result = YahooFinanceProvider(max_attempts=1, wait_seconds=0.0).fetch_bars(
        "BRK.B", START, END
    )

    assert result.meta["vendor_symbol"] == "BRK-B"
    assert "Adj Close" in result.frame.columns


def test_every_spelling_tried_is_reported_when_none_works(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[str] = []

    def empty_download(tickers, **kwargs):
        seen.append(tickers)
        return pd.DataFrame()

    monkeypatch.setattr(yahoo.yf, "download", empty_download)

    with pytest.raises(DownloadError) as excinfo:
        YahooFinanceProvider(max_attempts=1, wait_seconds=0.0).fetch_bars(
            "BRK.B", START, END
        )

    assert seen == ["BRK.B", "BRK-B"]
    message = str(excinfo.value)
    assert "BRK.B" in message
    assert "BRK-B" in message
    assert "no rows" in message


def test_transient_failure_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = make_raw_frame(periods=3)
    calls = {"count": 0}

    def flaky_download(tickers, **kwargs):
        calls["count"] += 1
        if calls["count"] < 3:
            raise ConnectionError("temporary network problem")
        return frame

    monkeypatch.setattr(yahoo.yf, "download", flaky_download)

    result = YahooFinanceProvider(max_attempts=3, wait_seconds=0.0).fetch_bars(
        "AAPL", START, END
    )

    assert calls["count"] == 3
    assert len(result.frame) == 3


def test_persistent_failure_raises_download_error(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"count": 0}

    def always_failing(tickers, **kwargs):
        calls["count"] += 1
        raise ConnectionError("network is down")

    monkeypatch.setattr(yahoo.yf, "download", always_failing)

    with pytest.raises(DownloadError):
        YahooFinanceProvider(max_attempts=2, wait_seconds=0.0).fetch_bars("AAPL", START, END)

    assert calls["count"] == 2


def test_invalid_symbol_raises_without_calling_the_vendor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected(tickers, **kwargs):  # pragma: no cover - must not be called
        raise AssertionError("yf.download should not be called for an invalid symbol")

    monkeypatch.setattr(yahoo.yf, "download", unexpected)

    with pytest.raises(DownloadError):
        YahooFinanceProvider(wait_seconds=0.0).fetch_bars("bad symbol!", START, END)


def test_end_before_start_raises() -> None:
    with pytest.raises(DownloadError):
        YahooFinanceProvider(wait_seconds=0.0).fetch_bars("AAPL", END, START)
