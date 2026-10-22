import numpy as np
import pandas as pd
import pytest

from src.schemas import PerfilRiesgo
from src.simulacion.estrategias import (
    GarchCache,
    accuracy_ajustada,
    backtest_sin_neutrales,
    consenso_por_ticker,
    equal_weight,
    mejor_modelo_por_ticker,
    normalizar,
    pesos_black_litterman,
    pesos_consenso,
    pesos_cuantil,
    pesos_histeresis,
    pesos_inclinacion,
    pesos_senal,
    predicciones_del_mejor,
    ranking_del_dia,
)
from tests.simulacion.conftest import TICKERS, FakeMacro, FakeTrend


def test_equal_weight_y_senal() -> None:
    assert equal_weight([]) == {}
    preds = {"A": {"signal": "alza"}, "B": {"signal": "baja"}, "C": {"signal": "alza"}}
    assert pesos_senal(preds, ["A", "B", "C", "D"]) == {"A": 0.5, "C": 0.5}


def test_normalizar_descarta_ruido_y_reescala() -> None:
    assert normalizar({"A": 0.6, "B": 0.4 + 1e-9, "C": 1e-9}) == pytest.approx(
        {"A": 0.6, "B": 0.4}, rel=1e-6
    )
    assert normalizar({"A": 0.3}) == {"A": 0.3}


def test_accuracy_ajustada_acerca_al_azar_con_pocos_casos() -> None:
    assert accuracy_ajustada({"directional_accuracy": 0.9, "observations": 50}) == pytest.approx(
        0.7
    )
    assert accuracy_ajustada({"directional_accuracy": 0.9, "observations": 0}) == 0.9


def test_mejor_modelo_por_ticker() -> None:
    backtests = {
        # 90% con 2 casos pierde contra 60% con 12: con el ajuste, 0.515 vs 0.519.
        "macro": {"A": {"directional_accuracy": 0.9, "observations": 2, "mae": 0.01}},
        "lstm": {
            "A": {"directional_accuracy": 0.6, "observations": 12, "mae": 0.05},
            "B": {"directional_accuracy": 0.5, "observations": 12, "mae": 0.05},
        },
        "xgboost": {"B": {"directional_accuracy": 0.5, "observations": 12, "mae": 0.02}},
    }
    elegidos = mejor_modelo_por_ticker(backtests, ("lstm", "xgboost", "macro"), ["A", "B", "C"])
    assert elegidos["A"]["model"] == "lstm"
    assert elegidos["B"]["model"] == "xgboost"  # empate de accuracy: gana menor mae
    assert "C" not in elegidos
    # Si el ganador no esta entre los candidatos, no se lo elige.
    assert mejor_modelo_por_ticker(backtests, ("macro",), ["A"])["A"]["model"] == "macro"

    predicciones = {"lstm": {"A": {"signal": "alza"}}, "xgboost": {}}
    assert predicciones_del_mejor(predicciones, elegidos) == {"A": {"signal": "alza"}}


def test_backtest_sin_neutrales_descarta_las_neutrales(snapshot) -> None:
    df = snapshot.history("GGAL", snapshot.trading_days()[-1])
    bt = backtest_sin_neutrales(FakeMacro(), df, 20)
    # 12 cortes, la mitad con vista.
    assert bt["observations"] == 6
    assert 0 <= bt["directional_accuracy"] <= 1

    class SiempreNeutral(FakeMacro):
        def predict_df(self, df):
            return super().predict_df(df) | {"signal": "neutral"}

    assert backtest_sin_neutrales(SiempreNeutral(), df, 20) is None
    assert backtest_sin_neutrales(FakeMacro(), df.iloc[:50], 20) is None


def test_backtest_sin_neutrales_saltea_errores(snapshot) -> None:
    from src.errors import DataUnavailableError

    class Falla(FakeTrend):
        def predict_df(self, df):
            raise DataUnavailableError("sin datos")

    df = snapshot.history("GGAL", snapshot.trading_days()[-1])
    assert backtest_sin_neutrales(Falla(), df, 5) is None


def test_consenso() -> None:
    pred = {"signal": "alza", "last_close": 100.0, "predicted_close": 110.0}
    por_modelo = {"lstm": {"A": pred}, "xgboost": {"A": pred}}
    consenso = consenso_por_ticker(por_modelo, ["A"], PerfilRiesgo.ARRIESGADO)
    assert set(consenso) == {"A"}
    assert pesos_consenso(
        {"A": {"classification": "sobrecompra"}, "B": {"classification": "x"}}
    ) == {"A": 1.0}


def test_garch_cache_reusa_el_forecast() -> None:
    cache = GarchCache()
    llamadas = []
    cache._model.forecast_variance_horizonte = lambda df, h: llamadas.append(h) or 0.0004 * h
    df = pd.DataFrame(
        {"close": np.linspace(1, 2, 10)}, index=pd.bdate_range("2026-01-01", periods=10)
    )
    assert cache.forecast_variance_horizonte(df, 5) == cache.forecast_variance_horizonte(df, 5)
    assert cache.forecast_variance_horizonte(df, 1) == pytest.approx(0.0004)
    assert llamadas == [5, 1]  # una vez por horizonte


def test_pesos_black_litterman(snapshot) -> None:
    dia = snapshot.trading_days()[-1]
    preds = {}
    for t in TICKERS:
        df = snapshot.history(t, dia)
        preds[t] = FakeTrend().predict_df(df) | {"symbol": t, "model_version": "v"}

    def fetch_hist(ticker, days):
        return snapshot.history(ticker, dia, days)

    pesos = pesos_black_litterman(
        preds, PerfilRiesgo.MODERADO, fetch_hist, GarchCache(), universo=TICKERS
    )
    assert set(pesos) <= set(TICKERS)
    assert all(w >= 0 for w in pesos.values())
    assert sum(pesos.values()) == pytest.approx(1.0, abs=1e-6)
    assert pesos_black_litterman({}, PerfilRiesgo.MODERADO, fetch_hist, GarchCache(), TICKERS) == {}


def test_reglas_de_cuantil_histeresis_e_inclinacion() -> None:
    preds = {
        t: {"predicted_close": 100.0 + i, "last_close": 100.0, "signal": s}
        for i, (t, s) in enumerate(
            zip(TICKERS, ["baja", "neutral", "neutral", "neutral", "neutral", "alza", "alza"],
                strict=True)
        )
    }  # fmt: skip
    rango = ranking_del_dia({**preds, "OTRO": preds["GGAL"]}, TICKERS[:-1])
    assert list(rango.index) == TICKERS[:-1] and rango["COME"] == 1.0  # solo el universo
    assert pesos_cuantil(preds, TICKERS, 0.30) == pytest.approx(
        {t: 1 / 3 for t in ("CEPU", "COME", "GGAL")}
    )
    assert pesos_histeresis(preds, TICKERS, set(), 0.15, 0.30) == pytest.approx(
        {"COME": 0.5, "GGAL": 0.5}
    )
    con_tenencia = pesos_histeresis(preds, TICKERS, {"CEPU", "ALUA"}, 0.15, 0.40)
    assert set(con_tenencia) == {"CEPU", "COME", "GGAL"}
    inclinado = pesos_inclinacion(preds, [*TICKERS, "SIN_PRED"], 0.5)
    assert sum(inclinado.values()) == pytest.approx(1.0)
    assert inclinado["GGAL"] == pytest.approx(3 * inclinado["ALUA"])
    assert inclinado["SIN_PRED"] == pytest.approx(inclinado["BMA"])
    assert pesos_inclinacion(preds, [], 0.5) == {}
    with pytest.raises(ValueError):
        pesos_inclinacion(preds, TICKERS, 1.5)


def test_pesos_mantener_conserva_las_neutrales() -> None:
    from src.simulacion.estrategias import pesos_mantener

    preds = {"A": {"signal": "alza"}, "B": {"signal": "neutral"}, "C": {"signal": "baja"},
             "D": {"signal": "neutral"}}  # fmt: skip
    universo = ["A", "B", "C", "D", "E"]
    # B y C ya estaban: B (neutral) se queda, C (baja) se vende; D (neutral, no estaba) no
    # se compra; E (sin prediccion ese dia) se mantiene.
    assert pesos_mantener(preds, universo, {"B", "C", "E"}) == {"A": 1 / 3, "B": 1 / 3, "E": 1 / 3}
    assert pesos_mantener(preds, universo, set()) == {"A": 1.0}
    assert pesos_mantener({}, universo, set()) == {}
