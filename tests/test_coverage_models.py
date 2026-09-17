from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import requests
import torch

from src import (
    arima_trend,
    garch_volatility,
    modal_client,
    retrain_guard,
    svm_direction,
    train,
    train_transformer,
    train_xgb,
    transformer,
    trend_common,
    xgb_trend,
)
from src.errors import DataUnavailableError, NotEnoughDataError


def history(rows: int = 90) -> pd.DataFrame:
    close = np.linspace(100.0, 120.0, rows)
    return pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.linspace(1_000, 2_000, rows),
        },
        index=pd.date_range("2026-01-01", periods=rows),
    )


def test_arima_prediction_and_modal_translation(monkeypatch):
    frame = history()
    fitted = SimpleNamespace(forecast=lambda steps: pd.Series([125.0] * steps))
    arima = Mock(return_value=SimpleNamespace(fit=lambda: fitted))
    monkeypatch.setattr(arima_trend, "ARIMA", arima)
    monkeypatch.setattr(
        arima_trend,
        "derive_trend_output",
        lambda df, value, horizon, band: {
            "value": value,
            "horizon": horizon,
            "band": band,
        },
    )

    model = arima_trend.ArimaTrendModel(horizon=2)
    assert model.version == "arima-1-1-1"
    assert model.predict_df(frame)["horizon"] == 2
    with pytest.raises(NotEnoughDataError):
        model.predict_df(frame.iloc[:10])

    base = {"valor_actual": 100, "cant_predicciones": 2}
    assert (
        arima_trend.translate_modal_arima_response({**base, "prediction": [100, 110]})["signal"]
        == "alza"
    )
    assert (
        arima_trend.translate_modal_arima_response({**base, "prediction": [100, 90]})["signal"]
        == "baja"
    )
    assert (
        arima_trend.translate_modal_arima_response({**base, "prediction": [100, 100.5]})["signal"]
        == "neutral"
    )
    with pytest.raises(DataUnavailableError):
        arima_trend.translate_modal_arima_response({"prediction": []})


def test_garch_prediction_and_variance(monkeypatch):
    frame = history()
    variance = SimpleNamespace(to_numpy=lambda: np.array([[1.0, 4.0, 9.0]]))
    result = SimpleNamespace(forecast=lambda horizon: SimpleNamespace(variance=variance))
    fitted = SimpleNamespace(fit=lambda **kwargs: result)
    factory = Mock(return_value=fitted)
    monkeypatch.setattr(garch_volatility, "arch_model", factory)

    model = garch_volatility.GarchVolatilityModel(horizon=3)
    assert model.version == "garch-1-1"
    output = model.predict_df(frame)
    assert output["daily_volatility_pct"] == [1.0, 2.0, 3.0]
    assert output["cumulative_volatility_pct"] == pytest.approx(3.7417)
    assert model.forecast_daily_variance(frame) == pytest.approx(0.0001)
    with pytest.raises(NotEnoughDataError):
        model._forecast_variance_pct2(np.arange(10.0))


@pytest.mark.parametrize(("predicted", "signal"), [(0, "alza"), (1, "baja")])
def test_svm_direction_prediction(monkeypatch, predicted, signal):
    class FakeSVC:
        def fit(self, x, y):
            assert len(x) == len(y)

        def predict(self, x):
            return np.array([predicted])

        def decision_function(self, x):
            return np.array([2.5])

    monkeypatch.setattr(svm_direction.svm, "SVC", FakeSVC)
    model = svm_direction.SvmDirectionModel()
    assert model.version == "svm-rbf"
    result = model.predict_df(history())
    assert result["signal"] == signal
    assert result["confidence"] == 1.0
    with pytest.raises(NotEnoughDataError):
        model.predict_df(history(10))


@pytest.mark.parametrize(
    ("log_return", "rsi_value", "signal", "condition"),
    [
        (0.1, None, "alza", "indeterminado"),
        (-0.1, 75.0, "baja", "sobrecompra"),
        (0.0, 25.0, "neutral", "sobreventa"),
        (0.0, 50.0, "neutral", "neutral"),
    ],
)
def test_derive_trend_output_branches(monkeypatch, log_return, rsi_value, signal, condition):
    monkeypatch.setattr(trend_common, "rsi", lambda values: rsi_value)
    result = trend_common.derive_trend_output(history(), log_return, horizon=5)
    assert result["signal"] == signal
    assert result["condition"] == condition


def test_walk_forward_backtest_handles_short_errors_and_missing_predictions():
    assert trend_common.backtest_predict_df(Mock(), history(10), 5, 10) is None

    class Predictor:
        calls = 0

        def predict_df(self, frame):
            self.calls += 1
            if self.calls == 1:
                raise DataUnavailableError("skip")
            if self.calls == 2:
                return {}
            return {"predicted_close": float(frame["close"].iloc[-1]) * 1.01}

    result = trend_common.backtest_predict_df(Predictor(), history(), 2, 15, step=2)
    assert result is not None
    assert result["observations"] > 0
    assert result["mae"] >= 0


def test_modal_client_success_and_errors(monkeypatch):
    response = Mock()
    response.json.return_value = {"signal": "alza"}
    monkeypatch.setattr(modal_client.requests, "get", Mock(return_value=response))
    assert modal_client.call_modal("https://model", "GGAL", horizon=2) == {"signal": "alza"}
    response.raise_for_status.assert_called_once()

    monkeypatch.setattr(
        modal_client.requests,
        "get",
        Mock(side_effect=requests.RequestException("offline")),
    )
    with pytest.raises(DataUnavailableError):
        modal_client.call_modal("https://model", "GGAL")

    bad_json = Mock()
    bad_json.json.side_effect = ValueError("bad json")
    monkeypatch.setattr(modal_client.requests, "get", Mock(return_value=bad_json))
    with pytest.raises(DataUnavailableError):
        modal_client.call_modal("https://model", "GGAL")


def test_retrain_guard_scores_and_promotes(monkeypatch, tmp_path):
    frames = {"A": history(), "B": history()}
    results = iter(
        [
            None,
            {"observations": 4, "directional_accuracy": 0.75},
        ]
    )
    monkeypatch.setattr(retrain_guard, "backtest_predict_df", lambda *a, **k: next(results))
    score = retrain_guard.pooled_directional_accuracy(object(), frames, 5)
    assert score == {"directional_accuracy": 0.75, "observations": 4}

    monkeypatch.setattr(retrain_guard, "backtest_predict_df", lambda *a, **k: None)
    assert retrain_guard.pooled_directional_accuracy(object(), frames, 5)["observations"] == 0

    model_path = str(tmp_path / "model.bin")
    assert retrain_guard.incumbent_score(model_path, Mock(), frames, 5) is None

    score_path = retrain_guard._score_path(model_path)
    score_path.write_text(json.dumps({"directional_accuracy": 0.6}))
    assert retrain_guard.incumbent_score(model_path, Mock(), frames, 5) == 0.6
    score_path.write_text("not-json")
    (tmp_path / "model.bin").write_text("artifact")
    monkeypatch.setattr(
        retrain_guard,
        "pooled_directional_accuracy",
        lambda *a, **k: {"directional_accuracy": 0.7, "observations": 3},
    )
    assert retrain_guard.incumbent_score(model_path, lambda path: object(), frames, 5) == 0.7

    model = Mock()
    monkeypatch.setattr(retrain_guard, "incumbent_score", lambda *a, **k: 0.5)
    promoted, candidate = retrain_guard.promote_if_better(model, model_path, Mock(), frames, 5)
    assert promoted is True
    assert candidate["incumbent_directional_accuracy"] == 0.5
    model.save.assert_called_once_with(model_path)

    monkeypatch.setattr(
        retrain_guard,
        "pooled_directional_accuracy",
        lambda *a, **k: {"directional_accuracy": 0.4, "observations": 3},
    )
    assert retrain_guard.promote_if_better(Mock(), model_path, Mock(), frames, 5)[0] is False

    forced = Mock()
    retrain_guard.force_promote(forced, model_path, frames, 5)
    forced.save.assert_called_once_with(model_path)


class FakeBooster:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, x, y, **kwargs):
        self.fitted = (x, y, kwargs)

    def predict(self, x):
        return np.full(len(x), 0.02)


def test_xgb_model_fit_predict_save_and_load(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "xgboost", SimpleNamespace(XGBRegressor=FakeBooster))
    feats = np.ones((80, 3))
    target = np.linspace(-0.1, 0.1, 80)
    monkeypatch.setattr(xgb_trend, "build_features", lambda df, horizon: (feats, target))
    monkeypatch.setattr(
        xgb_trend,
        "make_windows",
        lambda f, t, window: (np.ones((50, window, 3)), np.linspace(-0.1, 0.1, 50)),
    )
    monkeypatch.setattr(
        xgb_trend,
        "derive_trend_output",
        lambda df, value, horizon, band: {"value": value, "horizon": horizon},
    )

    model = xgb_trend.XGBTrendModel(xgb_trend.XGBConfig(window=2, n_estimators=1))
    assert model.version == "untrained"
    metrics = model.fit({"GGAL": history()})
    assert metrics["val_samples"] > 0
    assert model.version.startswith("xgb-")
    assert model.predict_df(history())["horizon"] == 5
    assert model._evaluate(np.empty((0, 6)), np.empty((0,))) == {}

    path = tmp_path / "xgb.pkl"
    model.save(path)
    loaded = xgb_trend.XGBTrendModel.load(path)
    assert loaded.predict_df(history())["value"] == pytest.approx(0.02)

    empty = xgb_trend.XGBTrendModel(xgb_trend.XGBConfig(window=2))
    with pytest.raises(NotEnoughDataError):
        empty.predict_df(history())
    with pytest.raises(NotEnoughDataError):
        empty.save(path)
    monkeypatch.setattr(xgb_trend, "make_windows", lambda *a: (np.empty((0,)), np.empty((0,))))
    with pytest.raises(NotEnoughDataError):
        empty.fit({"GGAL": history()})


def test_transformer_building_blocks_and_guards(tmp_path):
    positional = transformer._LearnedPositionalEncoding(max_len=4, d_model=4)
    assert positional(torch.zeros(2, 3, 4)).shape == (2, 3, 4)

    network = transformer._TransformerRegressor(
        input_size=3,
        window=2,
        d_model=4,
        nhead=2,
        num_layers=1,
        dim_feedforward=8,
        dropout=0,
    )
    assert network(torch.ones(2, 2, 3)).shape == (2,)
    with pytest.raises(ValueError):
        transformer.TransformerTrainConfig(d_model=5, nhead=2)

    model = transformer.TransformerTrendModel()
    assert model.version == "untrained"
    assert model._evaluate(np.empty((0, 1, 1)), np.empty((0,))) == {}
    assert model._to_tensor(np.ones((1, 2))).dtype == torch.float32
    with pytest.raises(NotEnoughDataError):
        model.predict_df(history())
    with pytest.raises(NotEnoughDataError):
        model.save(tmp_path / "transformer.pt")


@pytest.mark.parametrize(
    ("module", "model_name", "config_name"),
    [
        (train, "TrendModel", "TrainConfig"),
        (train_xgb, "XGBTrendModel", "XGBConfig"),
        (train_transformer, "TransformerTrendModel", "TransformerTrainConfig"),
    ],
)
def test_training_entrypoints_force_and_reject(monkeypatch, module, model_name, config_name):
    config = SimpleNamespace(horizon=5)
    config_class = Mock(return_value=config)
    for field in (
        "epochs",
        "window",
        "hidden_size",
        "num_layers",
        "batch_size",
        "learning_rate",
        "horizon",
        "n_estimators",
        "max_depth",
        "d_model",
        "nhead",
        "dim_feedforward",
        "dropout",
        "warmup_epochs",
    ):
        setattr(config_class, field, 1)
    monkeypatch.setattr(module, config_name, config_class)

    model = Mock(version="test-version")
    model.fit.return_value = {"accuracy": 0.8}
    model_class = Mock(return_value=model)
    model_class.load = Mock()
    monkeypatch.setattr(module, model_name, model_class)
    monkeypatch.setattr(module, "fetch_histories", lambda *a: {"GGAL": history()})
    monkeypatch.setattr(module, "fetch_available_tickers", lambda: ["GGAL"])
    monkeypatch.setattr(module, "force_promote", Mock(return_value={"score": 0.8}))
    monkeypatch.setattr(module, "promote_if_better", Mock(return_value=(False, {"score": 0.7})))

    monkeypatch.setattr(sys, "argv", ["train", "--tickers", "GGAL", "--force"])
    module.main()
    module.force_promote.assert_called_once()

    monkeypatch.setattr(sys, "argv", ["train", "--tickers", "GGAL"])
    module.main()
    module.promote_if_better.assert_called_once()

    monkeypatch.setattr(module, "fetch_histories", lambda *a: {})
    with pytest.raises(SystemExit):
        module.main()
