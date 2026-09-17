"""Tests del backtesting walk-forward con datos sinteticos (sin red)."""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from src.backtest import (
    _fetch_histories,
    _print_comparison,
    main,
    run_backtest,
    select_model_names,
    summarize,
    walk_forward,
)
from src.errors import DataUnavailableError
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


def test_fetch_histories_skips_unavailable_and_short_history() -> None:
    def fake_fetch_history(symbol, days):
        if symbol == "BAD":
            raise DataUnavailableError("no data")
        return _synthetic_ohlcv(300 if symbol == "OK" else 5)

    with patch("src.backtest.fetch_macro_series", return_value=None):
        with patch("src.backtest.fetch_history", side_effect=fake_fetch_history):
            histories = _fetch_histories(["BAD", "SHORT", "OK"], days=300, min_rows=100)

    assert list(histories) == ["OK"]
    assert "macro_rate" in histories["OK"].columns


def test_run_backtest_skips_unloaded_models_and_aggregates() -> None:
    registry = TrendRegistry(history_days=750)
    service = _fake_lstm_service()
    registry.register("lstm", "models/lstm.pt", TrendModel.load, default=True)
    registry._services["lstm"] = service
    registry.register("xgb", "models/xgb.pkl", Mock(), default=False)

    histories = {"SYN": _synthetic_ohlcv(300)}
    results = run_backtest(registry, histories, ["lstm", "xgb"], step=10, min_history=50)

    assert "xgb" not in results
    assert results["lstm"]["overall"]["n_predictions"] > 0
    assert "SYN" in results["lstm"]["per_ticker"]


def test_print_comparison_handles_empty_and_nonempty(capsys) -> None:
    _print_comparison({})
    assert "Sin resultados" in capsys.readouterr().out

    _print_comparison(
        {"lstm": {"version": "lstm-1", "overall": {"n_predictions": 3, "mae_logret": 0.01}}}
    )
    out = capsys.readouterr().out
    assert "lstm" in out


def test_main_end_to_end(monkeypatch, tmp_path) -> None:
    registry = TrendRegistry(history_days=750)
    service = _fake_lstm_service()
    registry.register("lstm", "models/lstm.pt", TrendModel.load, default=True)
    registry._services["lstm"] = service

    monkeypatch.setattr("src.backtest.fetch_available_tickers", lambda: ["SYN"])
    monkeypatch.setattr("src.backtest.fetch_macro_series", lambda days: None)
    monkeypatch.setattr("src.backtest.fetch_history", lambda symbol, days: _synthetic_ohlcv(300))
    monkeypatch.setattr("src.backtest.build_registry", lambda days: registry)
    monkeypatch.setattr(registry, "load_all", lambda: None)

    out_path = tmp_path / "backtest_results.json"
    monkeypatch.setattr(
        "sys.argv",
        ["backtest", "--step", "10", "--min-history", "50", "--out", str(out_path)],
    )
    main()
    assert out_path.exists()


def test_main_aborts_when_no_histories(monkeypatch) -> None:
    monkeypatch.setattr("src.backtest.fetch_available_tickers", lambda: ["SYN"])
    monkeypatch.setattr("src.backtest.fetch_macro_series", lambda days: None)
    monkeypatch.setattr(
        "src.backtest.fetch_history",
        lambda symbol, days: (_ for _ in ()).throw(DataUnavailableError("nope")),
    )
    monkeypatch.setattr("sys.argv", ["backtest"])
    with pytest.raises(SystemExit):
        main()


def test_main_aborts_when_no_valid_models(monkeypatch) -> None:
    registry = TrendRegistry(history_days=750)
    registry.register_remote("lstm-modal", "https://example.invalid")

    monkeypatch.setattr("src.backtest.fetch_available_tickers", lambda: ["SYN"])
    monkeypatch.setattr("src.backtest.fetch_macro_series", lambda days: None)
    monkeypatch.setattr("src.backtest.fetch_history", lambda symbol, days: _synthetic_ohlcv(300))
    monkeypatch.setattr("src.backtest.build_registry", lambda days: registry)
    monkeypatch.setattr(registry, "load_all", lambda: None)
    monkeypatch.setattr("sys.argv", ["backtest", "--models", "lstm-modal"])

    with pytest.raises(SystemExit):
        main()
