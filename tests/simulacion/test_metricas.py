import numpy as np
import pandas as pd
import pytest

from src.simulacion.metricas import (
    calidad_modelos,
    metricas_cartera,
    promociones_por_modelo,
    resolver_predicciones,
    rotacion,
)


def test_metricas_cartera() -> None:
    valores = pd.Series([100.0, 110.0, 99.0, 121.0])
    m = metricas_cartera(valores)
    assert m["ruedas"] == 4
    assert m["retorno_total"] == pytest.approx(0.21)
    assert m["max_drawdown"] == pytest.approx(-0.1)
    assert m["sharpe"] is not None
    assert metricas_cartera(pd.Series([100.0])) == {"ruedas": 1}
    assert metricas_cartera(pd.Series([100.0, 100.0, 100.0]))["sharpe"] is None


def test_resolver_predicciones_y_calidad(snapshot) -> None:
    dias = snapshot.trading_days()
    close = snapshot.prices["GGAL"]["close"]
    preds = pd.DataFrame(
        [
            # Resuelta: a 5 ruedas.
            {"fecha": dias[-20], "model": "lstm", "symbol": "GGAL", "signal": "alza",
             "horizon_days": 5, "last_close": close.iloc[-20],
             "predicted_close": close.iloc[-20] * 1.02},
            # Sin resolver todavia.
            {"fecha": dias[-2], "model": "lstm", "symbol": "GGAL", "signal": "alza",
             "horizon_days": 5, "last_close": close.iloc[-2],
             "predicted_close": close.iloc[-2] * 1.02},
            # Fecha que no es rueda.
            {"fecha": pd.Timestamp("2026-06-06"), "model": "lstm", "symbol": "GGAL",
             "signal": "alza", "horizon_days": 5, "last_close": 1.0, "predicted_close": 1.0},
        ]
    )  # fmt: skip
    resueltas = resolver_predicciones(preds, snapshot, str(dias[-1].date()))
    assert len(resueltas) == 1
    assert resueltas["realized_log_return"].iloc[0] == pytest.approx(
        np.log(close.iloc[-15] / close.iloc[-20])
    )

    calidad = calidad_modelos(resueltas)
    assert calidad["lstm"]["overall"]["n_predictions"] == 1
    assert len(calidad["lstm"]["por_mes"]) == 1
    assert calidad_modelos(resueltas, universo=["OTRO"]) == {}
    assert calidad_modelos(pd.DataFrame()) == {}


def test_promociones_por_modelo() -> None:
    promociones = pd.DataFrame(
        [
            {"fecha": "2026-06-01", "model": "lstm", "promoted": True, "bootstrap": True},
            {"fecha": "2026-06-02", "model": "lstm", "promoted": True},
            {"fecha": "2026-06-03", "model": "lstm", "promoted": False},
        ]
    )
    resultado = promociones_por_modelo(promociones)
    assert resultado["lstm"] == {
        "reentrenamientos": 2,
        "promociones": 1,
        "fechas_promocion": ["2026-06-02"],
    }
    sin_bootstrap = promociones.drop(columns="bootstrap")
    assert promociones_por_modelo(sin_bootstrap)["lstm"]["reentrenamientos"] == 3
    assert promociones_por_modelo(pd.DataFrame()) == {}


def test_rotacion() -> None:
    operaciones = pd.DataFrame(
        [
            {"cartera": "a", "monto": 500.0, "comision": 2.5},
            {"cartera": "a", "monto": 500.0, "comision": 2.5},
        ]
    )
    assert rotacion(operaciones, 1000.0)["a"] == {
        "operaciones": 2,
        "monto_operado_sobre_capital": 1.0,
        "comisiones_sobre_capital": 0.005,
    }
    assert rotacion(pd.DataFrame(), 1000.0) == {}
