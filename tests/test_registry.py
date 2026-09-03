"""Tests del TrendRegistry / TrendService (sin red)."""

import numpy as np
import pandas as pd
import pytest

import src.registry as registry_module
from src.errors import (
    DataUnavailableError,
    ModelNotLoadedError,
    StaleArtifactError,
    UnknownModelError,
)
from src.registry import EnsembleTrendService, RemoteTrendService, TrendRegistry, TrendService


def test_load_tolerates_incompatible_artifact(tmp_path) -> None:
    """Un artefacto que no matchea el feature set actual no debe tirar abajo el resto."""
    path = tmp_path / "fake.pt"
    path.write_bytes(b"no importa el contenido, solo que el archivo exista")

    def _raising_loader(_path):
        raise StaleArtifactError("artefacto viejo, hay que reentrenar")

    service = TrendService("lstm", str(path), _raising_loader, history_days=750)
    service.load()  # no debe propagar la excepcion

    assert service.is_loaded is False
    assert service.version == "unloaded"


def test_load_skips_missing_artifact(tmp_path) -> None:
    path = tmp_path / "missing.pt"
    service = TrendService("lstm", str(path), lambda p: None, history_days=750)
    service.load()
    assert service.is_loaded is False


def test_remote_predict_raises_when_modal_has_no_model_for_ticker(monkeypatch) -> None:
    """Modal responde 200 con {"error": "..."} cuando no hay modelo entrenado
    para el ticker (ver `call_modal`, no levanta por status). Antes de este
    fix ese payload sin `as_of`/`signal`/etc. pasaba intacto y explotaba mas
    adelante (paper_trading, /predict/trend) con un KeyError/ValidationError
    crudo en vez de ser tratado como un fallo esperable.
    """

    def _fake_call_modal(base_url, ticker, **extra_params):
        return {"error": "todavia no hay un modelo entrenado para 'OIL' con horizonte 5"}

    monkeypatch.setattr(registry_module, "call_modal", _fake_call_modal)
    service = RemoteTrendService("lstm-modal", "https://example.invalid", history_days=750)

    with pytest.raises(DataUnavailableError):
        service.predict("OIL")


def test_remote_predict_raises_when_required_field_missing(monkeypatch) -> None:
    """Respuesta 200 "exitosa" pero incompleta (ej: cambio de contrato en
    Modal) tambien debe fallar de forma controlada en vez de propagar un
    KeyError mas adelante.
    """

    def _fake_call_modal(base_url, ticker, **extra_params):
        return {
            "signal": "alza",
            "horizon_days": 5,
            "last_close": 100.0,
            "predicted_close": 105.0,
            # falta "as_of" a proposito
        }

    monkeypatch.setattr(registry_module, "call_modal", _fake_call_modal)
    service = RemoteTrendService("xgboost-modal", "https://example.invalid", history_days=750)

    with pytest.raises(DataUnavailableError):
        service.predict("GGAL")


def test_remote_predict_returns_result_when_payload_is_complete(monkeypatch) -> None:
    def _fake_call_modal(base_url, ticker, **extra_params):
        return {
            "signal": "alza",
            "horizon_days": 5,
            "last_close": 100.0,
            "predicted_close": 105.0,
            "as_of": "2026-08-30",
            "condition": "neutral",
        }

    monkeypatch.setattr(registry_module, "call_modal", _fake_call_modal)
    service = RemoteTrendService("lstm-modal", "https://example.invalid", history_days=750)

    result = service.predict("GGAL")

    assert result["symbol"] == "GGAL"
    assert result["as_of"] == "2026-08-30"
    assert result["model"] == "lstm-modal"
    assert result["model_version"] == "modal:lstm-modal"


def _fake_df() -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=30, freq="B")
    return pd.DataFrame({"close": np.linspace(90.0, 100.0, len(dates))}, index=dates)


def _loaded_service(name: str, predicted_close: float, last_close: float) -> TrendService:
    """TrendService con un modelo fake ya 'cargado' (sin pasar por `load()`/disco)."""

    class _FakeModel:
        version = f"{name}-fake"

        def predict_df(self, df: pd.DataFrame) -> dict:
            return {
                "signal": "alza",
                "horizon_days": 5,
                "predicted_close": predicted_close,
                "last_close": last_close,
                "rsi": None,
                "condition": "neutral",
                "as_of": df.index[-1].strftime("%Y-%m-%d"),
            }

    service = TrendService(name, "unused", lambda p: _FakeModel(), history_days=750)
    service._model = _FakeModel()
    return service


def test_ensemble_predict_on_is_weighted_average_of_log_returns() -> None:
    lstm = _loaded_service("lstm", predicted_close=105.0, last_close=100.0)
    xgboost = _loaded_service("xgboost", predicted_close=110.0, last_close=100.0)
    ensemble = EnsembleTrendService(
        "ensemble", {"lstm": (lstm, 1.0), "xgboost": (xgboost, 3.0)}, history_days=750
    )

    result = ensemble.predict_on(_fake_df(), "GGAL")

    expected_log_return = np.average(
        [np.log(105.0 / 100.0), np.log(110.0 / 100.0)], weights=[1.0, 3.0]
    )
    # derive_trend_output redondea a 4 decimales.
    assert result["predicted_close"] == pytest.approx(100.0 * np.exp(expected_log_return), abs=1e-4)
    assert result["symbol"] == "GGAL"
    assert result["model"] == "ensemble"
    assert "lstm-fake" in result["model_version"]
    assert "xgboost-fake" in result["model_version"]


def test_ensemble_is_loaded_requires_all_members() -> None:
    lstm = _loaded_service("lstm", predicted_close=105.0, last_close=100.0)
    xgboost = TrendService("xgboost", "missing/path.pkl", lambda p: None, history_days=750)
    ensemble = EnsembleTrendService(
        "ensemble", {"lstm": (lstm, 1.0), "xgboost": (xgboost, 1.0)}, history_days=750
    )

    assert ensemble.is_loaded is False
    with pytest.raises(ModelNotLoadedError):
        ensemble.predict_on(_fake_df(), "GGAL")


def test_register_ensemble_rejects_unregistered_member() -> None:
    reg = TrendRegistry(history_days=750)
    reg.register("lstm", "unused", lambda p: None)

    with pytest.raises(UnknownModelError):
        reg.register_ensemble("ensemble", {"lstm": 1.0, "xgboost": 1.0})


def test_register_ensemble_resolves_and_predicts() -> None:
    reg = TrendRegistry(history_days=750)
    reg._services["lstm"] = _loaded_service("lstm", predicted_close=105.0, last_close=100.0)
    reg._services["xgboost"] = _loaded_service("xgboost", predicted_close=95.0, last_close=100.0)

    reg.register_ensemble("ensemble", {"lstm": 1.0, "xgboost": 1.0})
    ensemble = reg.resolve("ensemble")

    assert isinstance(ensemble, EnsembleTrendService)
    assert ensemble.is_loaded is True
    result = ensemble.predict_on(_fake_df(), "GGAL")
    expected_log_return = np.average([np.log(1.05), np.log(0.95)], weights=[1.0, 1.0])
    assert result["predicted_close"] == pytest.approx(100.0 * np.exp(expected_log_return), abs=1e-4)
    assert result["signal"] == "neutral"
