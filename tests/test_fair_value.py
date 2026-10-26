"""Precio estimado por regresion lineal."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import src.main as main
from src.errors import NotEnoughDataError
from src.fair_value import ajustar, precio_estimado, tramo_de


def _serie(n=252, crecimiento=0.001, ruido=0.01, salto_final=0.0, seed=0):
    rng = np.random.default_rng(seed)
    log_p = np.log(100) + crecimiento * np.arange(n) + rng.normal(0, ruido, n)
    log_p[-1] += salto_final
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame({"close": np.exp(log_p)}, index=idx)


def test_ajuste_recupera_la_pendiente_de_una_tendencia_conocida() -> None:
    log_p = np.log(100) + 0.002 * np.arange(300)
    ajuste = ajustar(log_p)
    assert ajuste.pendiente == pytest.approx(0.002)
    assert ajuste.r2 == pytest.approx(1.0)


def test_precio_en_linea_con_su_tendencia() -> None:
    out = precio_estimado(_serie())
    assert out["posicion"] == "en_linea"
    assert abs(out["diferencia"]) < 0.05
    assert out["tendencia_anual"] == pytest.approx(np.exp(0.001 * 252) - 1, abs=0.05)
    assert len(out["canal"]) == 252
    canal = out["canal"][-1]
    assert canal["lo2"] < canal["lo1"] < canal["estimado"] < canal["hi1"] < canal["hi2"]


def test_detecta_precio_muy_por_encima_y_muy_por_debajo() -> None:
    arriba = precio_estimado(_serie(salto_final=0.15))
    abajo = precio_estimado(_serie(salto_final=-0.15))
    assert arriba["posicion"] == "por_encima" and arriba["desvios"] > 2
    assert abajo["posicion"] == "por_debajo" and abajo["desvios"] < -2
    assert arriba["historico"]["desde"] == 2.0
    assert abajo["historico"]["hasta"] == -2.0


def test_pocas_ruedas_no_alcanzan() -> None:
    with pytest.raises(NotEnoughDataError):
        precio_estimado(_serie(n=30))


def test_tramo_de_cubre_todo_el_rango() -> None:
    assert tramo_de(-5)["hasta"] == -2.0
    assert tramo_de(0)["desde"] == -1.0
    assert tramo_de(1.0)["desde"] == 1.0
    assert tramo_de(9)["desde"] == 2.0


def test_endpoint_fair_value(monkeypatch) -> None:
    monkeypatch.setattr(main, "fetch_history", lambda symbol, days: _serie())
    with TestClient(main.app) as client:
        response = client.get("/predict/fair-value/ggal")
    assert response.status_code == 200
    body = response.json()
    assert body["symbol"] == "GGAL"
    assert body["posicion"] in {"por_encima", "por_debajo", "en_linea"}
    assert body["historico"]["horizonte"] == 20


def test_endpoint_fair_value_con_pocas_ruedas_es_422(monkeypatch) -> None:
    monkeypatch.setattr(main, "fetch_history", lambda symbol, days: _serie(n=20))
    with TestClient(main.app) as client:
        response = client.get("/predict/fair-value/GGAL")
    assert response.status_code == 422


def test_trend_features_coincide_con_el_precio_estimado_en_la_ultima_rueda() -> None:
    from src.fair_value import TREND_FEATURES, trend_features

    df = _serie(n=400, salto_final=0.1)
    feats = trend_features(df["close"])
    assert list(feats.columns) == TREND_FEATURES
    ultimo = precio_estimado(df, ventana=252)
    assert feats["tz252"].iloc[-1] == pytest.approx(ultimo["desvios"], rel=1e-6)
    assert feats["tr2_252"].iloc[-1] == pytest.approx(ultimo["r2"], rel=1e-6)
    # antes de completar la ventana no hay valor (no se inventa)
    assert feats["tz252"].iloc[:251].isna().all()
