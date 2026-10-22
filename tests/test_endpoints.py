from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

import src.main as main
from src.black_litterman.black_litterman import ResultadoBL
from src.errors import (
    DataUnavailableError,
    ModelNotLoadedError,
    NotEnoughDataError,
    UnknownModelError,
)
from src.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def make_trend_result(**overrides) -> dict:
    result = {
        "symbol": "GGAL",
        "signal": "alza",
        "horizon_days": 5,
        "last_close": 100.0,
        "predicted_close": 105.0,
        "rsi": 55.0,
        "condition": "neutral",
        "as_of": "2026-09-03",
        "model": "lstm",
        "model_version": "v1",
    }
    result.update(overrides)
    return result


def make_direction_result(**overrides) -> dict:
    # Tal como lo devuelve svm_direction_model.predict_df: main.py completa
    # despues symbol/model/model_version, asi que no van aca.
    result = {
        "signal": "alza",
        "horizon_days": 1,
        "confidence": 0.72,
        "last_close": 100.0,
        "as_of": "2026-09-03",
    }
    result.update(overrides)
    return result


def make_volatility_result(**overrides) -> dict:
    result = {
        "horizon_days": 5,
        "daily_volatility_pct": [1.1, 1.2, 1.3, 1.25, 1.15],
        "cumulative_volatility_pct": 2.8,
        "last_close": 100.0,
        "as_of": "2026-09-03",
    }
    result.update(overrides)
    return result


# --- Endpoints originales ---


def test_root() -> None:
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()


def test_health_reports_model_loaded() -> None:
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True


def test_predict_returns_prediction() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"symbol": "GGAL", "features": {"opening": 1.0, "offered": 3.0}},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["symbol"] == "GGAL"
        assert body["prediction"] == 2.0
        assert "model_version" in body


def test_predict_rejects_empty_symbol() -> None:
    with TestClient(app) as client:
        response = client.post("/predict", json={"symbol": "", "features": {}})
        assert response.status_code == 422


# --- /health (usando trend_registry.statuses) ---


def test_health_reports_trend_models_status(client: TestClient, monkeypatch) -> None:
    fake_registry = MagicMock()
    fake_registry.statuses.return_value = {
        "lstm": {"loaded": True, "version": "v1", "is_default": True},
        "xgboost": {"loaded": False, "version": "unloaded", "is_default": False},
    }
    monkeypatch.setattr(main, "trend_registry", fake_registry)

    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["lstm_loaded"] is True
    assert body["trend_models"]["xgboost"]["loaded"] is False


# --- /models ---


def test_models_lists_default_and_statuses(client: TestClient, monkeypatch) -> None:
    fake_registry = MagicMock()
    fake_registry.default = "lstm"
    fake_registry.statuses.return_value = {
        "lstm": {"loaded": True, "version": "v1", "is_default": True},
    }
    monkeypatch.setattr(main, "trend_registry", fake_registry)

    response = client.get("/models")

    assert response.status_code == 200
    body = response.json()
    assert body["default"] == "lstm"
    assert body["models"]["lstm"]["loaded"] is True


# --- /predict/trend (POST) ---


def test_predict_trend_post_success(client: TestClient, monkeypatch) -> None:
    fake_service = MagicMock()
    fake_service.predict.return_value = make_trend_result()
    fake_registry = MagicMock()
    fake_registry.resolve.return_value = fake_service
    monkeypatch.setattr(main, "trend_registry", fake_registry)

    response = client.post("/predict/trend", json={"symbol": "GGAL", "model": "lstm"})

    assert response.status_code == 200
    body = response.json()
    assert body["symbol"] == "GGAL"
    assert body["signal"] == "alza"
    fake_registry.resolve.assert_called_once_with("lstm")
    fake_service.predict.assert_called_once_with("GGAL")


def test_predict_trend_post_unknown_model(client: TestClient, monkeypatch) -> None:
    fake_registry = MagicMock()
    fake_registry.resolve.side_effect = UnknownModelError("modelo 'foo' desconocido")
    monkeypatch.setattr(main, "trend_registry", fake_registry)

    response = client.post("/predict/trend", json={"symbol": "GGAL", "model": "foo"})

    assert response.status_code == 404


# --- /predict/trend/{symbol} (GET) ---


def test_predict_trend_get_success(client: TestClient, monkeypatch) -> None:
    fake_service = MagicMock()
    fake_service.predict.return_value = make_trend_result(symbol="YPFD", model="xgboost")
    fake_registry = MagicMock()
    fake_registry.resolve.return_value = fake_service
    monkeypatch.setattr(main, "trend_registry", fake_registry)

    response = client.get("/predict/trend/YPFD", params={"model": "xgboost"})

    assert response.status_code == 200
    assert response.json()["model"] == "xgboost"
    fake_registry.resolve.assert_called_once_with("xgboost")


@pytest.mark.parametrize(
    ("exc", "status"),
    [
        (UnknownModelError("modelo desconocido"), 404),
        (ModelNotLoadedError("modelo no entrenado"), 503),
        (DataUnavailableError("no se pudo descargar el historico"), 502),
        (NotEnoughDataError("historico insuficiente"), 422),
    ],
)
def test_predict_trend_get_maps_errors(client: TestClient, monkeypatch, exc, status) -> None:
    fake_registry = MagicMock()
    fake_registry.resolve.side_effect = exc
    monkeypatch.setattr(main, "trend_registry", fake_registry)

    response = client.get("/predict/trend/GGAL")

    assert response.status_code == status


def test_predict_trend_get_data_unavailable_on_predict(client: TestClient, monkeypatch) -> None:
    fake_service = MagicMock()
    fake_service.predict.side_effect = DataUnavailableError("Yahoo caido")
    fake_registry = MagicMock()
    fake_registry.resolve.return_value = fake_service
    monkeypatch.setattr(main, "trend_registry", fake_registry)

    response = client.get("/predict/trend/GGAL")

    assert response.status_code == 502


# --- /predict/trend/compare/{symbol} ---


def test_predict_trend_compare_success(client: TestClient, monkeypatch) -> None:
    fake_registry = MagicMock()
    fake_registry.compare.return_value = {
        "symbol": "GGAL",
        "as_of": "2026-09-03",
        "default_model": "lstm",
        "predictions": {
            "lstm": make_trend_result(),
            "xgboost": {"available": False, "reason": "modelo no entrenado"},
        },
    }
    monkeypatch.setattr(main, "trend_registry", fake_registry)

    response = client.get("/predict/trend/compare/GGAL")

    assert response.status_code == 200
    body = response.json()
    assert body["default_model"] == "lstm"
    assert body["predictions"]["xgboost"]["available"] is False
    fake_registry.compare.assert_called_once_with("GGAL")


def test_predict_trend_compare_data_unavailable(client: TestClient, monkeypatch) -> None:
    fake_registry = MagicMock()
    fake_registry.compare.side_effect = DataUnavailableError("no se pudo descargar el historico")
    monkeypatch.setattr(main, "trend_registry", fake_registry)

    response = client.get("/predict/trend/compare/GGAL")

    assert response.status_code == 502


# --- /predict/direction/{symbol} (SVM) ---


def test_predict_direction_success(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(main, "fetch_history", MagicMock(return_value=MagicMock()))
    fake_svm = MagicMock()
    fake_svm.predict_df.return_value = make_direction_result()
    fake_svm.version = "svm-v2"
    monkeypatch.setattr(main, "svm_direction_model", fake_svm)

    response = client.get("/predict/direction/ggal")

    assert response.status_code == 200
    body = response.json()
    assert body["symbol"] == "GGAL"  # normalizado a upper/strip
    assert body["model"] == "svm"
    assert body["model_version"] == "svm-v2"


@pytest.mark.parametrize(
    ("exc", "status"),
    [
        (DataUnavailableError("no se pudo descargar el historico"), 502),
        (NotEnoughDataError("historico insuficiente"), 422),
    ],
)
def test_predict_direction_maps_errors(client: TestClient, monkeypatch, exc, status) -> None:
    fake_fetch = MagicMock(side_effect=exc)
    monkeypatch.setattr(main, "fetch_history", fake_fetch)

    response = client.get("/predict/direction/GGAL")

    assert response.status_code == status


# --- /predict/volatility/{symbol} (GARCH) ---


def test_predict_volatility_success(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(main, "fetch_history", MagicMock(return_value=MagicMock()))
    fake_garch = MagicMock()
    fake_garch.predict_df.return_value = make_volatility_result()
    fake_garch.version = "garch-v3"
    monkeypatch.setattr(main, "garch_volatility_model", fake_garch)

    response = client.get("/predict/volatility/ggal")

    assert response.status_code == 200
    body = response.json()
    assert body["symbol"] == "GGAL"
    assert body["model"] == "garch"
    assert body["model_version"] == "garch-v3"
    assert len(body["daily_volatility_pct"]) == 5


@pytest.mark.parametrize(
    ("exc", "status"),
    [
        (DataUnavailableError("no se pudo descargar el historico"), 502),
        (NotEnoughDataError("historico insuficiente"), 422),
    ],
)
def test_predict_volatility_maps_errors(client: TestClient, monkeypatch, exc, status) -> None:
    fake_fetch = MagicMock(side_effect=exc)
    monkeypatch.setattr(main, "fetch_history", fake_fetch)

    response = client.get("/predict/volatility/GGAL")

    assert response.status_code == status


# --- /predict/direction/modal/{symbol} ---


def test_predict_direction_modal_not_configured(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(main.settings, "modal_svm_url", None)

    response = client.get("/predict/direction/modal/GGAL")

    assert response.status_code == 404


def test_predict_direction_modal_success(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(main.settings, "modal_svm_url", "https://modal.example/svm")
    fake_call_modal = MagicMock(return_value={"prediction": "Buy"})
    monkeypatch.setattr(main, "call_modal", fake_call_modal)

    response = client.get("/predict/direction/modal/ggal")

    assert response.status_code == 200
    body = response.json()
    assert body["symbol"] == "GGAL"
    assert body["model"] == "svm-modal"
    assert body["prediction"] == "Buy"
    fake_call_modal.assert_called_once_with("https://modal.example/svm", "ggal")


def test_predict_direction_modal_data_unavailable(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(main.settings, "modal_svm_url", "https://modal.example/svm")
    monkeypatch.setattr(
        main, "call_modal", MagicMock(side_effect=DataUnavailableError("modal caido"))
    )

    response = client.get("/predict/direction/modal/GGAL")

    assert response.status_code == 502


# --- /predict/volatility/modal/{symbol} ---


def test_predict_volatility_modal_not_configured(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(main.settings, "modal_garch_url", None)

    response = client.get("/predict/volatility/modal/GGAL")

    assert response.status_code == 404


def test_predict_volatility_modal_success(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(main.settings, "modal_garch_url", "https://modal.example/garch")
    fake_call_modal = MagicMock(return_value={"cumulative_volatility_pct": 3.1})
    monkeypatch.setattr(main, "call_modal", fake_call_modal)

    response = client.get("/predict/volatility/modal/ggal")

    assert response.status_code == 200
    body = response.json()
    assert body["symbol"] == "GGAL"
    assert body["model"] == "garch-modal"
    assert body["cumulative_volatility_pct"] == 3.1


def test_predict_volatility_modal_data_unavailable(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(main.settings, "modal_garch_url", "https://modal.example/garch")
    monkeypatch.setattr(
        main, "call_modal", MagicMock(side_effect=DataUnavailableError("modal caido"))
    )

    response = client.get("/predict/volatility/modal/GGAL")

    assert response.status_code == 502


# --- /portfolio/recomendacion (Black-Litterman) ---


@pytest.fixture(autouse=True)
def _clear_recomendacion_cache():
    main._recomendacion_cache.clear()
    yield
    main._recomendacion_cache.clear()


def usuario_payload() -> dict:
    return {
        "perfil_riesgo": "moderate",
        "tenencias": [
            {"ticker": "GGAL", "cantidad": 10.0},
            {"ticker": "YPFD", "cantidad": 5.0},
        ],
    }


def fake_resultado(tickers=("GGAL", "YPFD"), pesos=(0.6, 0.4), **overrides) -> ResultadoBL:
    n = len(tickers)
    campos = {
        "tickers": list(tickers),
        "pesos": np.array(pesos),
        "peso_actual": np.full(n, 1.0 / n),
        "pi": np.full(n, 1e-4),
        "q": np.full(n, 2e-3),
        "omega": np.full(n, 2e-5),
        "mu_bl": np.full(n, 1.5e-3),
        "horizon_days": [5] * n,
        "tickers_excluidos": {},
        "cartera_ancla": "propia",
    }
    campos.update(overrides)
    return ResultadoBL(**campos)


def test_portfolio_recomendacion_success(client: TestClient, monkeypatch) -> None:
    fake_entry = MagicMock(
        return_value=fake_resultado(
            peso_actual=np.array([0.8, 0.2]),
            tickers_excluidos={"ECOG": "17 ruedas, minimo 60"},
        )
    )
    monkeypatch.setattr(main, "entry", fake_entry)

    response = client.post("/portfolio/recomendacion", json=usuario_payload())

    assert response.status_code == 200
    body = response.json()
    assert body["pesos_recomendados"] == {"GGAL": 0.6, "YPFD": 0.4}
    assert body["tickers_excluidos"] == {"ECOG": "17 ruedas, minimo 60"}
    assert body["cartera_ancla"] == "propia"
    assert body["detalle"][0] == {
        "ticker": "GGAL",
        "peso_actual": 0.8,
        "peso_recomendado": 0.6,
        "pi": 1e-4,
        "q": 2e-3,
        "omega": 2e-5,
        "mu_bl": 1.5e-3,
        "horizon_days": 5,
    }


def test_portfolio_recomendacion_with_model_and_cartera_ancla(
    client: TestClient, monkeypatch
) -> None:
    fake_entry = MagicMock(return_value=fake_resultado(["GGAL"], [1.0]))
    monkeypatch.setattr(main, "entry", fake_entry)

    response = client.post(
        "/portfolio/recomendacion",
        params={"model": "xgboost", "cartera_ancla": "equal_weight"},
        json=usuario_payload(),
    )

    assert response.status_code == 200
    fake_entry.assert_called_once()


@pytest.mark.parametrize(
    ("exc", "status"),
    [
        (UnknownModelError("modelo desconocido"), 404),
        (ModelNotLoadedError("modelo no entrenado"), 503),
        (DataUnavailableError("no se pudo descargar el historico"), 502),
        (NotEnoughDataError("historico insuficiente"), 422),
        (NotImplementedError("cartera_ancla 'mercado' no implementada"), 501),
        (ValueError("pesos no suman 1"), 422),
    ],
)
def test_portfolio_recomendacion_maps_errors(client: TestClient, monkeypatch, exc, status) -> None:
    monkeypatch.setattr(main, "entry", MagicMock(side_effect=exc))

    response = client.post("/portfolio/recomendacion", json=usuario_payload())

    assert response.status_code == status


def test_portfolio_recomendacion_reuses_a_recent_result(client: TestClient, monkeypatch) -> None:
    fake_entry = MagicMock(return_value=fake_resultado(["GGAL"], [1.0]))
    monkeypatch.setattr(main, "entry", fake_entry)

    first = client.post("/portfolio/recomendacion", json=usuario_payload())
    second = client.post("/portfolio/recomendacion", json=usuario_payload())

    assert first.json() == second.json()
    fake_entry.assert_called_once()


def test_portfolio_recomendacion_cache_is_per_portfolio_and_expires(
    client: TestClient, monkeypatch
) -> None:
    fake_entry = MagicMock(return_value=fake_resultado(["GGAL"], [1.0]))
    monkeypatch.setattr(main, "entry", fake_entry)
    other = {"perfil_riesgo": "moderate", "tenencias": [{"ticker": "BMA", "cantidad": 1.0}]}

    client.post("/portfolio/recomendacion", json=usuario_payload())
    client.post("/portfolio/recomendacion", json=other)
    assert fake_entry.call_count == 2  # otra cartera, otra corrida

    monkeypatch.setattr(main, "RECOMENDACION_TTL_SEGUNDOS", -1)
    client.post("/portfolio/recomendacion", json=usuario_payload())
    assert fake_entry.call_count == 3  # vencido, se recalcula


def test_portfolio_recomendacion_errors_are_not_cached(client: TestClient, monkeypatch) -> None:
    fake_entry = MagicMock(side_effect=ValueError("sin datos"))
    monkeypatch.setattr(main, "entry", fake_entry)

    assert client.post("/portfolio/recomendacion", json=usuario_payload()).status_code == 422
    assert client.post("/portfolio/recomendacion", json=usuario_payload()).status_code == 422
    assert fake_entry.call_count == 2


# --- /predict/trend/consensus/{symbol} (componentes) ---


def test_consensus_includes_components(client: TestClient, monkeypatch) -> None:
    fake_registry = MagicMock()
    fake_registry.compare.return_value = {
        "symbol": "GGAL",
        "as_of": "2026-09-03",
        "default_model": "lstm",
        "predictions": {
            "lstm": make_trend_result(backtest={"directional_accuracy": 0.6}),
            "xgboost": {"available": False, "reason": "modelo no entrenado"},
        },
    }
    monkeypatch.setattr(main, "trend_registry", fake_registry)
    monkeypatch.setattr(main, "load_track_record", lambda profile, symbol: None)

    response = client.get("/predict/trend/consensus/GGAL")

    assert response.status_code == 200
    components = response.json()["components"]
    assert len(components) == 1
    assert components[0]["model"] == "lstm"
    assert components[0]["expected_return"] == pytest.approx(np.log(1.05), abs=1e-5)
    assert components[0]["quality"] == 0.6
    assert components[0]["agreement"] == 1.0
    assert components[0]["signal"] == "alza"


# --- /predict/trend/xgboost/explain/{symbol} ---


def test_explain_xgboost_success(client: TestClient, monkeypatch) -> None:
    fake_model = MagicMock()
    fake_model.version = "xgb-1"
    fake_model.explain.return_value = {
        "as_of": "2026-09-03",
        "probability_up": 0.61,
        "base_value": 0.1,
        "contributions": [
            {"feature": "a", "label": "A", "value": 1.0, "contribution": 0.2},
            {"feature": "b", "label": "B", "value": None, "contribution": -0.05},
        ],
    }
    fake_registry = MagicMock()
    fake_registry.resolve.return_value = MagicMock(model=fake_model)
    monkeypatch.setattr(main, "trend_registry", fake_registry)
    monkeypatch.setattr(main, "fetch_history_with_macro", MagicMock(return_value="df"))

    response = client.get("/predict/trend/xgboost/explain/ggal")

    assert response.status_code == 200
    body = response.json()
    assert body["symbol"] == "GGAL"
    assert body["probability_up"] == 0.61
    assert body["contributions"][1]["value"] is None
    fake_registry.resolve.assert_called_once_with("xgboost")
    fake_model.explain.assert_called_once_with("df")


def test_explain_xgboost_not_loaded(client: TestClient, monkeypatch) -> None:
    fake_registry = MagicMock()
    fake_registry.resolve.return_value = MagicMock(model=None)
    monkeypatch.setattr(main, "trend_registry", fake_registry)

    assert client.get("/predict/trend/xgboost/explain/GGAL").status_code == 503


@pytest.mark.parametrize(
    ("exc", "status"),
    [
        (DataUnavailableError("sin historico"), 502),
        (NotEnoughDataError("pocas ruedas"), 422),
    ],
)
def test_explain_xgboost_maps_errors(client: TestClient, monkeypatch, exc, status) -> None:
    fake_registry = MagicMock()
    fake_registry.resolve.return_value = MagicMock(model=MagicMock())
    monkeypatch.setattr(main, "trend_registry", fake_registry)
    monkeypatch.setattr(main, "fetch_history_with_macro", MagicMock(side_effect=exc))

    assert client.get("/predict/trend/xgboost/explain/GGAL").status_code == status


# --- /signals/history ---


def test_signals_history_reads_the_ledger(client: TestClient, monkeypatch, tmp_path) -> None:
    import json

    ledger = tmp_path / "ledger.json"
    base = {"horizon_days": 5, "last_close": 100.0, "predicted_close": 104.0}
    ledger.write_text(
        json.dumps(
            {
                "pending": [
                    {**base, "symbol": "YPFD", "model": "lstm", "as_of": "2026-10-10",
                     "signal": "alza"},
                ],
                "resolved": [
                    {**base, "symbol": "GGAL", "model": "xgboost", "as_of": "2026-10-01",
                     "signal": "alza", "realized_close": 106.0, "resolved_at": "2026-10-08"},
                ],
            }
        )
    )  # fmt: skip
    monkeypatch.setattr(main, "LEDGER_PATH", ledger)

    body = client.get("/signals/history").json()
    assert [r["symbol"] for r in body["rows"]] == ["YPFD", "GGAL"]
    assert body["rows"][1]["hit"] is True
    summary = body["summary"]
    assert {k: summary[k] for k in ("evaluated", "hits", "hit_rate")} == {
        "evaluated": 1,
        "hits": 1,
        "hit_rate": 1.0,
    }
    assert summary["last_signal"] == max(r["as_of"] for r in body["rows"])

    body = client.get("/signals/history", params={"symbols": "GGAL", "limit": 1}).json()
    assert [r["symbol"] for r in body["rows"]] == ["GGAL"]


def test_signals_history_without_ledger_is_empty(client: TestClient, monkeypatch, tmp_path):
    monkeypatch.setattr(main, "LEDGER_PATH", tmp_path / "missing.json")

    body = client.get("/signals/history").json()
    assert body["rows"] == []
    assert body["summary"]["evaluated"] == 0
    assert body["summary"]["hit_rate"] is None
    assert body["summary"]["last_signal"] is None


# --- /models/diagnostics ---


def test_models_diagnostics(client: TestClient, monkeypatch, tmp_path) -> None:
    results = tmp_path / "backtest.json"
    results.write_text('{"run_at": "2026-09-03T03:19:34Z", "results": {}}')
    monkeypatch.setattr(main, "BACKTEST_RESULTS_PATH", results)
    fake_model = MagicMock(version="xgb-1", trained_at="20261019T000000Z")
    fake_model.metrics = {"oos_auc": 0.55, "oos_baja_hit_rate": float("nan")}
    fake_registry = MagicMock()
    fake_registry.resolve.return_value = MagicMock(model=fake_model)
    monkeypatch.setattr(main, "trend_registry", fake_registry)

    body = client.get("/models/diagnostics").json()

    assert body["backtest"]["run_at"] == "2026-09-03T03:19:34Z"
    assert body["xgboost"] == {
        "version": "xgb-1",
        "trained_at": "20261019T000000Z",
        "metrics": {"oos_auc": 0.55, "oos_baja_hit_rate": None},
    }


def test_models_diagnostics_without_artifacts(client: TestClient, monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(main, "BACKTEST_RESULTS_PATH", tmp_path / "missing.json")
    fake_registry = MagicMock()
    fake_registry.resolve.side_effect = UnknownModelError("no registrado")
    monkeypatch.setattr(main, "trend_registry", fake_registry)

    assert client.get("/models/diagnostics").json() == {"backtest": None, "xgboost": None}
