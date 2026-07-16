"""Tests del backtesting walk-forward con datos sinteticos (sin red)."""

import numpy as np
import pandas as pd

from src.backtest import select_model_names, summarize, walk_forward
from src.lstm import TrainConfig, TrendModel
from src.registry import RemoteTrendService, TrendRegistry, TrendService


def _synthetic_ohlcv(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0005, 0.02, n)
    close = 100 * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(rng.normal(0, 0.01, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.01, n)))
    volume = rng.integers(1_000, 10_000, n).astype(float)
    index = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )


def _fake_lstm_service() -> TrendService:
    df = _synthetic_ohlcv(300)
    config = TrainConfig(window=20, horizon=5, epochs=5, patience=10)
    model = TrendModel(config=config)
    model.fit({"SYN": df})
    service = TrendService("lstm", "unused.pt", TrendModel.load, history_days=750)
    service._model = model
    return service


def test_walk_forward_produces_comparable_records() -> None:
    df = _synthetic_ohlcv(300)
    service = _fake_lstm_service()

    records = walk_forward(service, df, "SYN", step=10, min_history=50)

    assert len(records) > 0
    for r in records:
        assert r["signal"] in {"alza", "baja", "neutral"}
        assert isinstance(r["predicted_log_return"], float)
        assert isinstance(r["realized_log_return"], float)


def test_walk_forward_stops_without_enough_future_data() -> None:
    df = _synthetic_ohlcv(300)
    service = _fake_lstm_service()

    records = walk_forward(service, df, "SYN", step=10, min_history=50)
    # ningun registro deberia pedir datos mas alla del propio historico
    assert all(r["as_of"] <= df.index[-6].strftime("%Y-%m-%d") for r in records)


def test_summarize_empty_records() -> None:
    assert summarize([]) == {"n_predictions": 0}


def test_summarize_metrics_in_range() -> None:
    df = _synthetic_ohlcv(300)
    service = _fake_lstm_service()
    records = walk_forward(service, df, "SYN", step=10, min_history=50)

    summary = summarize(records)

    assert summary["n_predictions"] == len(records)
    assert 0.0 <= summary["directional_accuracy"] <= 1.0
    assert 0.0 <= summary["neutral_rate"] <= 1.0
    if summary["signal_hit_rate"] is not None:
        assert 0.0 <= summary["signal_hit_rate"] <= 1.0


def test_select_model_names_excludes_remote_by_default() -> None:
    registry = TrendRegistry(history_days=750)
    registry.register("lstm", "models/lstm.pt", TrendModel.load, default=True)
    registry.register_remote("lstm-modal", "https://example.invalid")

    assert select_model_names(registry, None) == ["lstm"]


def test_select_model_names_skips_explicit_remote_request() -> None:
    registry = TrendRegistry(history_days=750)
    registry.register("lstm", "models/lstm.pt", TrendModel.load, default=True)
    registry.register_remote("lstm-modal", "https://example.invalid")

    assert isinstance(registry.resolve("lstm-modal"), RemoteTrendService)
    assert select_model_names(registry, ["lstm", "lstm-modal"]) == ["lstm"]
