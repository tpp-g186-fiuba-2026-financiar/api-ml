"""Tests del acceso a las series macro del data-colector (sin red)."""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import requests

from src import macro_data
from src.errors import DataUnavailableError
from src.macro_data import (
    REQUIRED,
    SERIES,
    build_macro_frame,
    fetch_argentina_macro,
    fetch_collector_series,
    todays_macro_frame,
)

DAY_MS = 86_400_000


def response(payload: dict, status_code: int = 200) -> Mock:
    resp = Mock()
    resp.status_code = status_code
    resp.raise_for_status = Mock()
    resp.json.return_value = payload
    return resp


def payload(values: list[float], start_ms: int = 1_767_225_600_000) -> dict:
    rows = [
        {"source": "ARGDATOS", "series_id": "CCL", "ts": start_ms + i * DAY_MS, "value": str(v)}
        for i, v in enumerate(values)
    ]
    return {"status": 200, "source": "ARGDATOS", "series": "CCL", "data": rows, "cached": True}


def series(values: list[float], start: str = "2026-01-01") -> pd.Series:
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="D"))


def test_fetch_collector_series_parses_points_sorted_and_deduplicated() -> None:
    body = payload([1600.0, 1610.5, 1620.0])
    body["data"].append(dict(body["data"][0]))  # duplicado: se queda con uno
    with patch("src.macro_data.requests.post", return_value=response(body)) as post:
        result = fetch_collector_series("macro/argdatos", "CCL")

    assert post.call_args.args[0].endswith("/macro/argdatos/CCL")
    assert list(result.values) == [1600.0, 1610.5, 1620.0]
    assert result.index.is_monotonic_increasing and result.index.is_unique


def test_fetch_collector_series_reports_embedded_errors_and_empty_data() -> None:
    error = {"status": 500, "message": {"error": "Failed to fetch interest rate series"}}
    with patch("src.macro_data.requests.post", return_value=response(error)):
        with pytest.raises(DataUnavailableError, match="Failed to fetch"):
            fetch_collector_series("macro/argdatos", "CCL")
    with patch("src.macro_data.requests.post", return_value=response({"status": 200, "data": []})):
        with pytest.raises(DataUnavailableError, match="no devolvio datos"):
            fetch_collector_series("macro/argdatos", "CCL")


def test_fetch_collector_series_wraps_network_failures() -> None:
    with patch("src.macro_data.requests.post", side_effect=requests.ConnectionError("caido")):
        with pytest.raises(DataUnavailableError, match="no se pudo obtener"):
            fetch_collector_series("interest-rate/ar", "BADLAR")


def test_base_url_prefers_the_macro_specific_collector(monkeypatch) -> None:
    monkeypatch.setattr(macro_data.settings, "macro_collector_url", "http://localhost:3055/")
    assert macro_data._base_url() == "http://localhost:3055"
    monkeypatch.setattr(macro_data.settings, "macro_collector_url", None)
    monkeypatch.setattr(macro_data.settings, "data_collector_url", "https://dc.example/")
    assert macro_data._base_url() == "https://dc.example"
    monkeypatch.setattr(macro_data.settings, "data_collector_url", None)
    with pytest.raises(DataUnavailableError):
        macro_data._base_url()


def test_build_macro_frame_is_daily_forward_filled_and_lagged() -> None:
    raw = {
        "ccl": series([100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0]),
        "reservas": series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
    }
    frame = build_macro_frame(raw)
    # CCL se retrasa 2 dias y las reservas del BCRA 5: nunca se ve un dato de "hoy".
    assert SERIES["ccl"][2] == 2 and SERIES["reservas"][2] == 5
    assert frame.loc["2026-01-05", "ccl"] == 120.0  # el 5/1 solo se conoce lo del 3/1
    assert frame.loc["2026-01-08", "reservas"] == 3.0
    assert pd.isna(frame.loc["2026-01-02", "ccl"])
    assert frame.index.is_monotonic_increasing
    assert (frame.index[1:] - frame.index[:-1] == pd.Timedelta(days=1)).all()


def test_build_macro_frame_extends_to_the_requested_end_with_the_last_value() -> None:
    frame = build_macro_frame(
        {"ccl": series([100.0, 110.0, 120.0])}, end=pd.Timestamp("2026-01-10")
    )
    assert frame.index[-1] == pd.Timestamp("2026-01-10")
    assert frame.loc["2026-01-10", "ccl"] == 120.0
    with pytest.raises(DataUnavailableError):
        build_macro_frame({})


def test_fetch_argentina_macro_requires_the_main_series_but_tolerates_optional_ones() -> None:
    def fake(route: str, name: str) -> pd.Series:
        if name == "MEP":
            raise DataUnavailableError("MEP caido")  # opcional: se sigue sin ella
        return series(list(np.linspace(100, 200, 30)))

    with patch("src.macro_data.fetch_collector_series", side_effect=fake):
        frame = fetch_argentina_macro()
    assert "mep" not in frame.columns
    assert set(REQUIRED) <= set(frame.columns)

    def fake_without_risk(route: str, name: str) -> pd.Series:
        if name == "RIESGO_PAIS":
            raise DataUnavailableError("riesgo pais caido")
        return series(list(np.linspace(100, 200, 30)))

    with patch("src.macro_data.fetch_collector_series", side_effect=fake_without_risk):
        with pytest.raises(DataUnavailableError, match="imprescindibles"):
            fetch_argentina_macro()


def test_todays_macro_frame_downloads_once_per_day() -> None:
    macro_data._CACHE.clear()
    frame = pd.DataFrame({"ccl": [1.0]})
    with patch("src.macro_data.fetch_argentina_macro", return_value=frame) as fetch:
        assert todays_macro_frame() is frame
        assert todays_macro_frame() is frame
    assert fetch.call_count == 1
    macro_data._CACHE.clear()
