"""Tests de las funciones de fetch de src.data con red mockeada."""

from unittest.mock import Mock, patch

import pandas as pd
import pytest
import requests

from src.data import (
    fetch_available_tickers,
    fetch_history,
    fetch_history_collector,
    fetch_history_yahoo,
)
from src.errors import DataUnavailableError


def _mock_response(payload: dict, status_code: int = 200) -> Mock:
    resp = Mock()
    resp.status_code = status_code
    resp.raise_for_status = Mock()
    resp.json.return_value = payload
    return resp


def _yahoo_payload(n: int = 5) -> dict:
    timestamps = [1_700_000_000 + i * 86_400 for i in range(n)]
    closes = [100.0 + i for i in range(n)]
    return {
        "chart": {
            "result": [
                {
                    "timestamp": timestamps,
                    "indicators": {
                        "quote": [
                            {
                                "open": closes,
                                "high": [c + 1 for c in closes],
                                "low": [c - 1 for c in closes],
                                "close": closes,
                                "volume": [1000.0 + i for i in range(n)],
                            }
                        ]
                    },
                }
            ]
        }
    }


def test_fetch_history_yahoo_parses_candles() -> None:
    with patch("src.data.requests.get", return_value=_mock_response(_yahoo_payload())):
        df = fetch_history_yahoo("GGAL", days=100)

    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert len(df) == 5
    assert df.index.is_monotonic_increasing


def test_fetch_history_yahoo_raises_on_request_exception() -> None:
    with patch("src.data.requests.get", side_effect=requests.RequestException("timeout")):
        with pytest.raises(DataUnavailableError):
            fetch_history_yahoo("GGAL")


def test_fetch_history_yahoo_raises_on_bad_json() -> None:
    bad = Mock()
    bad.raise_for_status = Mock()
    bad.json.side_effect = ValueError("bad json")
    with patch("src.data.requests.get", return_value=bad):
        with pytest.raises(DataUnavailableError):
            fetch_history_yahoo("GGAL")


def test_fetch_history_yahoo_raises_on_empty_result() -> None:
    with patch("src.data.requests.get", return_value=_mock_response({"chart": {"result": None}})):
        with pytest.raises(DataUnavailableError):
            fetch_history_yahoo("GGAL")


def test_fetch_history_yahoo_raises_when_no_candles() -> None:
    payload = {"chart": {"result": [{"timestamp": None, "indicators": {"quote": [{}]}}]}}
    with patch("src.data.requests.get", return_value=_mock_response(payload)):
        with pytest.raises(DataUnavailableError):
            fetch_history_yahoo("GGAL")


def _collector_payload(n: int = 5) -> dict:
    return {
        "data": [
            {
                "open_amount": 100.0 + i,
                "high_amount": 101.0 + i,
                "low_amount": 99.0 + i,
                "close_amount": 100.0 + i,
                "volume": 1000 + i,
                "ts": 1_700_000_000_000 + i * 86_400_000,
            }
            for i in range(n)
        ]
    }


def test_fetch_history_collector_parses_rows() -> None:
    with patch("src.config.settings.data_collector_url", "https://collector.test"):
        with patch("src.data.requests.post", return_value=_mock_response(_collector_payload())):
            df = fetch_history_collector("GGAL", days=100)
    assert len(df) == 5
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]


def test_fetch_history_collector_requires_base_url() -> None:
    with patch("src.config.settings.data_collector_url", None):
        with pytest.raises(DataUnavailableError):
            fetch_history_collector("GGAL")


def test_fetch_history_collector_raises_on_empty_rows() -> None:
    with patch("src.config.settings.data_collector_url", "https://collector.test"):
        with patch("src.data.requests.post", return_value=_mock_response({"data": []})):
            with pytest.raises(DataUnavailableError):
                fetch_history_collector("GGAL")


def test_fetch_history_collector_raises_on_request_exception() -> None:
    with patch("src.config.settings.data_collector_url", "https://collector.test"):
        with patch("src.data.requests.post", side_effect=requests.RequestException("down")):
            with pytest.raises(DataUnavailableError):
                fetch_history_collector("GGAL")


def test_fetch_history_dispatches_to_collector() -> None:
    with patch("src.config.settings.data_source", "collector"):
        with patch("src.data.fetch_history_collector", return_value=pd.DataFrame()) as mocked:
            fetch_history("GGAL", 10)
    mocked.assert_called_once_with("GGAL", 10)


def test_fetch_history_dispatches_to_yahoo_by_default() -> None:
    with patch("src.config.settings.data_source", "yahoo"):
        with patch("src.data.fetch_history_yahoo", return_value=pd.DataFrame()) as mocked:
            fetch_history("GGAL", 10)
    mocked.assert_called_once_with("GGAL", 10)


def test_fetch_available_tickers_requires_base_url() -> None:
    with patch("src.config.settings.data_collector_url", None):
        with pytest.raises(DataUnavailableError):
            fetch_available_tickers()


def test_fetch_available_tickers_returns_list() -> None:
    payload = {"message": {"tickers": ["GGAL", "YPFD"]}}
    with patch("src.config.settings.data_collector_url", "https://collector.test"):
        with patch("src.data.requests.post", return_value=_mock_response(payload)):
            tickers = fetch_available_tickers()
    assert tickers == ["GGAL", "YPFD"]


def test_fetch_available_tickers_raises_on_empty_list() -> None:
    payload = {"message": {"tickers": []}}
    with patch("src.config.settings.data_collector_url", "https://collector.test"):
        with patch("src.data.requests.post", return_value=_mock_response(payload)):
            with pytest.raises(DataUnavailableError):
                fetch_available_tickers()


def test_fetch_available_tickers_retries_then_raises(monkeypatch) -> None:
    monkeypatch.setattr("src.data.time.sleep", lambda *_: None)
    with patch("src.config.settings.data_collector_url", "https://collector.test"):
        with patch(
            "src.data.requests.post",
            side_effect=requests.RequestException("down"),
        ) as mocked:
            with pytest.raises(DataUnavailableError):
                fetch_available_tickers()
    assert mocked.call_count == 3


def test_fetch_available_tickers_recovers_after_transient_failure(monkeypatch) -> None:
    monkeypatch.setattr("src.data.time.sleep", lambda *_: None)
    ok_payload = _mock_response({"message": {"tickers": ["GGAL"]}})
    with patch("src.config.settings.data_collector_url", "https://collector.test"):
        with patch(
            "src.data.requests.post",
            side_effect=[requests.RequestException("down"), ok_payload],
        ):
            tickers = fetch_available_tickers()
    assert tickers == ["GGAL"]
