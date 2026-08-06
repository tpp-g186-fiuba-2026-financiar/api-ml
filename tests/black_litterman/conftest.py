from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.schemas import PerfilRiesgo, TrendResponse, Usuario


def fake_ohlcv(
    seed: int,
    n: int = 100,
    base: float = 100.0,
    vol: float = 0.02,
    start: str = "2024-01-01",
    hour: int = 14,
) -> pd.DataFrame:
    """DataFrame OHLCV con la misma forma que fetch_history real: columnas
    open/high/low/close/volume, indice de fechas ascendente con hora fija
    (simulando el timestamp de cierre de sesion de una fuente puntual).
    """
    rng = np.random.default_rng(seed)
    closes = [base]
    for _ in range(n - 1):
        closes.append(closes[-1] * (1 + rng.normal(0, vol)))
    idx = pd.date_range(start, periods=n, freq="D") + pd.Timedelta(hours=hour)
    return pd.DataFrame(
        {"open": closes, "high": closes, "low": closes, "close": closes, "volume": [1000] * n},
        index=idx,
    )


@pytest.fixture
def historiales_3_tickers() -> dict[str, pd.DataFrame]:
    return {
        "GGAL": fake_ohlcv(seed=1, base=100.0),
        "YPFD": fake_ohlcv(seed=2, base=400.0),
        "PAMP": fake_ohlcv(seed=3, base=50.0),
    }


@pytest.fixture
def precios_3_tickers(historiales_3_tickers) -> dict[str, float]:
    return {t: float(df["close"].iloc[-1]) for t, df in historiales_3_tickers.items()}


@pytest.fixture
def usuario_con_cartera() -> Usuario:
    return Usuario(
        perfil_riesgo=PerfilRiesgo.MODERADO,
        tenencias=[
            {"ticker": "GGAL", "cantidad": 150},
            {"ticker": "YPFD", "cantidad": 40},
            {"ticker": "PAMP", "cantidad": 0},
        ],
    )


@pytest.fixture
def usuario_sin_cartera() -> Usuario:
    return Usuario(perfil_riesgo=PerfilRiesgo.CONSERVADOR, tenencias=[])


def make_predict_function(retornos: dict[str, float]):
    """Arma una funcion (ticker -> TrendResponse) para inyectar en entry(),
    a partir de un dict {ticker: retorno_esperado} (ej. 0.05 = +5%).

    Los campos que no afectan a Q (signal/horizon_days/condition/as_of/
    model_version) se completan con valores dummy validos, ya que
    Prediccion solo usa predicted_close y last_close.
    """

    def _predict(ticker: str) -> TrendResponse:
        r = retornos.get(ticker, 0.0)
        last_close = 100.0
        predicted_close = last_close * (1 + r)
        return TrendResponse(
            symbol=ticker,
            signal="alza" if r > 0 else ("baja" if r < 0 else "neutral"),
            horizon_days=5,
            last_close=last_close,
            predicted_close=predicted_close,
            condition="neutral",
            as_of="2025-01-01",
            model_version="test",
        )

    return _predict
