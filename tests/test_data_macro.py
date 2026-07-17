"""Tests de la serie macro (tasas de interes) y su alineacion a un OHLCV (sin red)."""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from src.data import attach_macro_feature, fetch_interest_rate, fetch_macro_series
from src.errors import DataUnavailableError


def _mock_response(payload: dict, status_code: int = 200) -> Mock:
    resp = Mock()
    resp.status_code = status_code
    resp.raise_for_status = Mock()
    resp.json.return_value = payload
    return resp


def test_fetch_interest_rate_parses_series() -> None:
    payload = {
        "status": 200,
        "source": "US",
        "series": "TNX",
        "cached": True,
        "data": [
            {"series_id": "TNX", "source": "US", "ts": 1468844400000, "value": "1.587"},
            {"series_id": "TNX", "source": "US", "ts": 1468930800000, "value": "1.558"},
        ],
    }
    with patch("src.data.requests.post", return_value=_mock_response(payload)):
        series = fetch_interest_rate("us", "TNX", days=750)

    assert len(series) == 2
    assert series.iloc[0] == pytest.approx(1.587)
    assert series.index.is_monotonic_increasing


def test_fetch_interest_rate_raises_on_embedded_error_status() -> None:
    """data-colector devuelve HTTP 200 aunque el body diga status=500 (ver BCRA caido)."""
    payload = {"status": 500, "message": {"error": "Failed to fetch interest rate series"}}
    with patch("src.data.requests.post", return_value=_mock_response(payload)):
        with pytest.raises(DataUnavailableError):
            fetch_interest_rate("ar", "TPM")


def test_fetch_interest_rate_raises_on_empty_data() -> None:
    payload = {"status": 200, "data": []}
    with patch("src.data.requests.post", return_value=_mock_response(payload)):
        with pytest.raises(DataUnavailableError):
            fetch_interest_rate("us", "TNX")


def test_fetch_macro_series_swallows_failures() -> None:
    """No debe propagar: si la fuente macro esta caida, el entrenamiento sigue sin ella."""
    with patch("src.data.fetch_interest_rate", side_effect=DataUnavailableError("caido")):
        assert fetch_macro_series(750) is None


def test_attach_macro_feature_none_fills_nan() -> None:
    df = pd.DataFrame(
        {"close": [1.0, 2.0, 3.0]}, index=pd.date_range("2024-01-01", periods=3, freq="B")
    )
    result = attach_macro_feature(df, "macro_rate", None)
    assert result["macro_rate"].isna().all()
    assert "macro_rate" not in df.columns  # no muta el original


def test_attach_macro_feature_forward_fills_without_lookahead() -> None:
    df = pd.DataFrame(
        {"close": [1.0, 2.0, 3.0, 4.0]},
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-04", "2024-01-08"]),
    )
    # La tasa no cotiza todos los dias (feriados distintos, etc.): 01-03 falta.
    macro = pd.Series([10.0, 11.0], index=pd.to_datetime(["2024-01-01", "2024-01-03"]))

    result = attach_macro_feature(df, "macro_rate", macro)

    assert result.loc["2024-01-01", "macro_rate"] == 10.0
    assert result.loc["2024-01-02", "macro_rate"] == 10.0  # ffill del ultimo valor conocido
    assert result.loc["2024-01-04", "macro_rate"] == 11.0
    assert result.loc["2024-01-08", "macro_rate"] == 11.0  # nunca ve un valor "futuro"


def test_build_features_uses_macro_rate_when_present() -> None:
    from src.lstm import FEATURE_NAMES, build_features

    n = 60
    index = pd.date_range("2024-01-01", periods=n, freq="B")
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    df = pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": rng.integers(1000, 5000, n).astype(float),
        },
        index=index,
    )
    # Tasa que sube de forma constante: el cambio a 5 ruedas deberia ser > 0
    # en todas las filas con suficiente historia previa.
    df["macro_rate"] = np.linspace(30.0, 40.0, n)

    feats, _ = build_features(df, horizon=5)
    macro_col = FEATURE_NAMES.index("macro_rate_chg5")
    assert (feats[10:, macro_col] > 0).all()
