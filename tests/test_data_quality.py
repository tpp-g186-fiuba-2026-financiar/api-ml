"""Tests del control de series estancadas y del RSI sin movimiento de precio (sin red)."""

import numpy as np
import pandas as pd
import pytest

from src import registry
from src.data_quality import drop_stale, is_stale
from src.errors import NotEnoughDataError
from src.indicators import rsi_series
from src.lstm import fetch_histories


def make_history(rows: int = 300, seed: int = 0, flat: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-02", periods=rows)
    close = (
        np.full(rows, 100.0) if flat else 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.02, rows)))
    )
    return pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": rng.integers(1_000, 50_000, rows).astype(float),
        },
        index=idx,
    )


def test_is_stale_and_drop_stale() -> None:
    assert is_stale(make_history(flat=True)) is True
    assert is_stale(make_history()) is False
    assert is_stale(make_history(rows=1)) is True
    assert list(drop_stale({"A3": make_history(flat=True), "GGAL": make_history()})) == ["GGAL"]


def test_rsi_of_a_flat_series_is_neutral_not_overbought() -> None:
    assert rsi_series(np.full(40, 100.0))[-1] == 50.0
    assert rsi_series(np.arange(1.0, 41.0))[-1] == 100.0  # solo ganancias: sobrecompra real


def test_fetch_histories_skips_stale_series() -> None:
    data = {"GGAL": make_history(seed=1), "A3": make_history(flat=True)}
    histories = fetch_histories(["GGAL", "A3"], 300, lambda symbol, days: data[symbol])
    assert list(histories) == ["GGAL"]


def test_registry_refuses_to_predict_stale_series() -> None:
    calls = {"n": 0}

    class Model:
        version = "m-1"

        def predict_df(self, df):
            calls["n"] += 1
            return {"signal": "neutral"}

    service = registry.TrendService("m", "models/none.pkl", lambda path: Model(), 300)
    service._model = Model()
    with pytest.raises(NotEnoughDataError, match="estancados"):
        service.predict_on(make_history(flat=True), "A3")
    assert calls["n"] == 0
    assert service.predict_on(make_history(), "ggal")["symbol"] == "GGAL"

    on_demand = registry.OnDemandTrendService("m", Model(), 300)
    with pytest.raises(NotEnoughDataError):
        on_demand.predict_on(make_history(flat=True), "A3")
