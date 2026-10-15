import json

import numpy as np
import pandas as pd
import pytest

import src.simulacion.experimento as experimento
from src.errors import ApiMlError
from src.simulacion.experimento import (
    ReversionModel,
    ReversionPanelModel,
    correr,
    metricas,
    predecir_periodo,
    resumen,
)
from tests.simulacion.conftest import TICKERS, FakeTrend

PERIODO = ("2026-05-04", "2026-06-12")


def test_reversion_predice_lo_contrario_de_la_ultima_semana() -> None:
    index = pd.bdate_range("2026-01-01", periods=10)
    close = np.linspace(100, 110, 10)
    df = pd.DataFrame({"close": close, "high": close, "low": close}, index=index)
    modelo = ReversionModel()
    assert modelo.fit({}) == {}
    assert modelo.predict_df(df)["predicted_close"] < close[-1]
    with pytest.raises(ApiMlError):
        modelo.predict_df(df.iloc[:3])


def test_reversion_panel_predice_igual_y_saltea_los_cortos() -> None:
    index = pd.bdate_range("2026-01-01", periods=10)
    close = np.linspace(100, 110, 10)
    df = pd.DataFrame({"close": close, "high": close, "low": close}, index=index)
    modelo = ReversionPanelModel()
    resultados = modelo.predict_panel({"A": df, "CORTO": df.iloc[:3]})
    assert set(resultados) == {"A"}
    assert resultados["A"] == ReversionModel().predict_df(df)


def test_reversion_panel_mide_igual_que_la_de_a_un_ticker(snapshot) -> None:
    de_a_uno = predecir_periodo(snapshot, ReversionModel, *PERIODO, TICKERS)
    panel = predecir_periodo(snapshot, ReversionPanelModel, *PERIODO, TICKERS)
    pd.testing.assert_frame_equal(de_a_uno, panel)
    assert metricas(panel) == metricas(de_a_uno)


def test_predecir_periodo_no_mira_el_futuro(snapshot) -> None:
    df = predecir_periodo(snapshot, FakeTrend, *PERIODO, [*TICKERS, "NOEXISTE"])
    assert set(df["symbol"]) == set(TICKERS)
    assert df["fecha"].min() >= PERIODO[0] and df["fecha"].max() <= PERIODO[1]
    fila = df.iloc[0]
    close = snapshot.prices[fila["symbol"]]["close"]
    pos = close.index.get_loc(pd.Timestamp(fila["fecha"]))
    assert fila["real"] == pytest.approx(np.log(close.iloc[pos + 5] / close.iloc[pos]))


def test_predecir_periodo_saltea_errores_y_el_final_sin_futuro(snapshot) -> None:
    class Falla(FakeTrend):
        def predict_df(self, df):
            raise ApiMlError("sin datos")

    assert predecir_periodo(snapshot, Falla, *PERIODO, TICKERS).empty
    ultimo = snapshot.trading_days()[-1].strftime("%Y-%m-%d")
    assert predecir_periodo(snapshot, FakeTrend, ultimo, ultimo, TICKERS).empty


def test_metricas_con_prediccion_perfecta() -> None:
    rng = np.random.default_rng(0)
    filas = []
    for fecha in pd.bdate_range("2026-01-01", periods=20).strftime("%Y-%m-%d"):
        real = rng.normal(0, 0.05, 8)
        for i, r in enumerate(real):
            filas.append(
                {
                    "fecha": fecha,
                    "symbol": f"T{i}",
                    "pred": r,
                    "real": r,
                    "signal": "alza" if r > 0.01 else "baja" if r < -0.01 else "neutral",
                }
            )
    m = metricas(pd.DataFrame(filas))
    assert m["ic"] == pytest.approx(1.0)
    assert m["hit_rel"] == pytest.approx(1.0)
    assert m["ls"] > 0
    assert m["precision_alza"] == 1.0
    assert m["precision_baja"] == 1.0
    assert m["ret_alza"] > m["ret_medio"] > m["ret_baja"]


def test_metricas_sin_ruedas_validas() -> None:
    df = pd.DataFrame(
        {
            "fecha": ["d1"] * 2,
            "symbol": ["A", "B"],
            "pred": [0.0, 0.0],
            "real": [0.1, -0.1],
            "signal": ["neutral"] * 2,
        }
    )
    m = metricas(df)
    assert m["ic"] == 0.0 and m["ic_t"] == 0.0 and m["ls"] == 0.0
    assert m["precision_alza"] is None and m["precision_baja"] is None
    assert m["ret_alza"] is None and m["ret_baja"] is None and m["ret_medio"] == 0.0


def test_correr_y_resumen(snapshot, tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setitem(experimento.VARIANTES, "fake", FakeTrend)
    filas = correr(snapshot, "prueba", ["fake"], [PERIODO], TICKERS, base=tmp_path)
    assert filas[0]["variante"] == "fake" and filas[0]["n"] > 0
    guardadas = (tmp_path / "prueba" / "resultados.jsonl").read_text().splitlines()
    assert json.loads(guardadas[0])["desde"] == PERIODO[0]
    assert (tmp_path / "prueba" / "predicciones" / f"fake_{PERIODO[0]}.csv").exists()
    tabla = resumen("prueba", base=tmp_path)
    assert "fake" in tabla and "ic medio" in tabla


def test_main(snapshot_dir, tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setitem(experimento.VARIANTES, "fake", FakeTrend)
    monkeypatch.setattr(experimento, "DIR_EXPERIMENTOS", tmp_path)
    monkeypatch.setattr(experimento, "PERIODOS", [PERIODO])
    monkeypatch.setattr(
        "sys.argv",
        ["experimento", "--nombre", "x", "--variantes", "fake", "--snapshot", str(snapshot_dir),
         "--limpiar"],
    )  # fmt: skip
    experimento.main()
    assert "fake" in capsys.readouterr().out
