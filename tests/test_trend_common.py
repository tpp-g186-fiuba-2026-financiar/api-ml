"""Post-procesamiento compartido: senal, centrado, senal por cuantil e IC por fecha."""

import numpy as np
import pytest

from src.trend_common import (
    acierto_balanceado,
    centrar_predicciones,
    ic_por_fecha,
    senal,
    senal_por_cuantil,
)


def _pred(last: float, predicted: float, signal: str = "alza") -> dict:
    return {"last_close": last, "predicted_close": predicted, "signal": signal}


def test_senal() -> None:
    assert senal(0.02, 0.01) == "alza"
    assert senal(-0.02, 0.01) == "baja"
    assert senal(0.005, 0.01) == "neutral"


def test_centrar_predicciones_resta_el_promedio_del_universo() -> None:
    preds = {
        "A": _pred(100, 100 * np.exp(0.03)),
        "B": _pred(10, 10 * np.exp(0.02)),
        "C": _pred(50, 50 * np.exp(0.01)),
        "FUERA": _pred(20, 20 * np.exp(0.10)),
        "SIN": {"last_close": 5.0, "predicted_close": None, "signal": "neutral"},
    }
    centradas = centrar_predicciones(preds, 0.005, universo=["A", "B", "C"])
    # Promedio del universo: 2%. A queda +1%, B en 0, C -1%.
    assert [centradas[t]["signal"] for t in "ABC"] == ["alza", "neutral", "baja"]
    assert centradas["A"]["predicted_close"] == pytest.approx(100 * np.exp(0.01), abs=1e-3)
    assert centradas["FUERA"]["signal"] == "alza"
    assert centradas["SIN"] is preds["SIN"]
    # No modifica la entrada.
    assert preds["B"]["signal"] == "alza"


def test_senal_por_cuantil() -> None:
    preds = {f"T{i}": _pred(100, 100 * np.exp(0.001 * i), "neutral") for i in range(20)}
    con_senal = senal_por_cuantil(preds, 0.15)
    senales = [con_senal[f"T{i}"]["signal"] for i in range(20)]
    assert senales == ["baja"] * 3 + ["neutral"] * 14 + ["alza"] * 3
    assert con_senal["T0"]["predicted_close"] == preds["T0"]["predicted_close"]
    assert preds["T0"]["signal"] == "neutral"  # no modifica la entrada
    assert [senal_por_cuantil(preds, 0.3)[f"T{i}"]["signal"] for i in (5, 6, 14)] == [
        "baja", "neutral", "alza"
    ]  # fmt: skip
    pocos = senal_por_cuantil({t: preds[t] for t in ("T0", "T19")}, 0.15)
    assert {r["signal"] for r in pocos.values()} == {"neutral"}
    with pytest.raises(ValueError, match="cuantil"):
        senal_por_cuantil(preds, 0.5)


def test_centrar_predicciones_sin_suficientes_tickers_no_cambia() -> None:
    preds = {"A": _pred(100, 103), "B": _pred(10, 10.2)}
    assert centrar_predicciones(preds, 0.01) is preds


def test_ic_por_fecha() -> None:
    fechas = np.array(["d1"] * 4 + ["d2"] * 4 + ["d3"] * 2)
    real = np.array([1, 2, 3, 4, 1, 2, 3, 4, 1, 2], dtype=float)
    pred = np.array([1, 2, 3, 4, 4, 3, 2, 1, 5, 6], dtype=float)
    # d1 ordena perfecto (+1), d2 al reves (-1), d3 tiene menos de 3 tickers.
    assert ic_por_fecha(pred, real, fechas) == pytest.approx(0.0)
    assert ic_por_fecha(pred[:4], real[:4], fechas[:4]) == pytest.approx(1.0)
    assert ic_por_fecha(pred[8:], real[8:], fechas[8:]) == 0.0


def test_acierto_balanceado() -> None:
    real = np.array([0.02, 0.01, 0.03, -0.01])
    # Decir siempre "sube" da 0,5 aunque acierte 3 de 4.
    assert acierto_balanceado(np.ones(4), real) == pytest.approx(0.5)
    assert acierto_balanceado(np.sign(real), real) == pytest.approx(1.0)
    # Acierta 2 de 3 subas y la baja: (2/3 + 1) / 2.
    assert acierto_balanceado(np.array([1, 1, -1, -1]), real) == pytest.approx(5 / 6)
    # Sin bajas cuenta solo el acierto en subas; sin casos, 0,5.
    assert acierto_balanceado(np.array([1, -1]), np.array([0.1, 0.2])) == pytest.approx(0.5)
    assert acierto_balanceado(np.array([]), np.array([])) == 0.5
