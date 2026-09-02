"""Tests del TrendRegistry / TrendService (sin red)."""

import pytest

import src.registry as registry_module
from src.errors import DataUnavailableError, StaleArtifactError
from src.registry import RemoteTrendService, TrendService


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
