import numpy as np
import pandas as pd
import pytest

from src.errors import ApiMlError
from src.panel.ensamble import EnsambleConfig, EnsamblePanelModel, reversion
from src.panel.gbm import PanelGBMConfig
from tests.panel.test_gbm import RUEDAS, TICKERS, _histories, _ohlcv, _proveedor


def _modelo(peso: float) -> EnsamblePanelModel:
    config = EnsambleConfig(peso_reversion=peso, ridge=PanelGBMConfig(tipo="ridge"))
    modelo = EnsamblePanelModel(config, proveedor=_proveedor)
    modelo.fit({t: df.iloc[:-50] for t, df in _histories().items()})
    return modelo


def _log(resultados: dict) -> pd.Series:
    return pd.Series(
        {s: np.log(r["predicted_close"] / r["last_close"]) for s, r in resultados.items()}
    )


def test_reversion() -> None:
    df = pd.DataFrame({"close": np.linspace(100, 110, 10)})
    assert reversion(df, 5) == pytest.approx(-np.log(110 / df["close"].iloc[-6]))
    assert reversion(df.iloc[:5], 5) is None
    assert reversion(df.assign(close=0.0), 5) is None


def test_config_invalida() -> None:
    with pytest.raises(ValueError, match="peso_reversion"):
        EnsambleConfig(peso_reversion=1.5)
    with pytest.raises(ValueError, match="cuantil_senal"):
        EnsambleConfig(cuantil_senal=0.6)


def test_senal_por_cuantil_del_score() -> None:
    config = EnsambleConfig(ridge=PanelGBMConfig(tipo="ridge"), cuantil_senal=0.25)
    modelo = EnsamblePanelModel(config, proveedor=_proveedor)
    modelo.fit({t: df.iloc[:-50] for t, df in _histories().items()})
    resultados = modelo.predict_panel(_histories())
    senales = pd.Series({s: r["signal"] for s, r in resultados.items()})
    assert (senales == "alza").sum() == 2 and (senales == "baja").sum() == 2
    log = _log(resultados)
    assert log[senales == "alza"].min() > log[senales == "neutral"].max()
    assert log[senales == "baja"].max() < log[senales == "neutral"].min()
    assert modelo.version == "panel-ensamble-0.5-q0.25"


def test_pesos_extremos_reproducen_cada_componente() -> None:
    dfs = _histories()
    solo_reversion = _log(_modelo(1.0).predict_panel(dfs))
    esperado = pd.Series({t: reversion(dfs[t], 5) for t in solo_reversion.index})
    assert solo_reversion.rank().equals(esperado.rank())

    modelo = _modelo(0.0)
    solo_ridge = _log(modelo.predict_panel(dfs))
    ridge = _log(modelo.ridge.predict_panel(dfs))
    assert solo_ridge.rank().equals(ridge[solo_ridge.index].rank())
    # Misma escala que el Ridge ese dia.
    assert solo_ridge.std() == pytest.approx(ridge.std(), rel=1e-3)


def test_combina_y_casos_vacios() -> None:
    modelo = _modelo(0.5)
    dfs = _histories()
    resultados = modelo.predict_panel(dfs)
    assert set(resultados) == set(TICKERS)
    assert modelo.version == "panel-ensamble-0.5"
    assert modelo.resumen()["config"]["peso_reversion"] == 0.5
    assert modelo.predict_panel({"NOESTA": _ohlcv(0)}) == {}
    with pytest.raises(ApiMlError, match="predict_panel"):
        modelo.predict_df(dfs["GGAL"])


def test_sin_reversion_valida_o_sin_dispersion(monkeypatch) -> None:
    modelo = _modelo(0.5)
    dfs = _histories()
    monkeypatch.setattr("src.panel.ensamble.reversion", lambda df, h: None)
    assert modelo.predict_panel(dfs) == {}
    monkeypatch.setattr("src.panel.ensamble.reversion", lambda df, h: 0.0)
    monkeypatch.setattr(modelo.ridge, "predict_panel", lambda d: {t: _fijo(d[t]) for t in TICKERS})
    resultados = modelo.predict_panel(dfs)
    assert all(r["predicted_close"] == pytest.approx(r["last_close"]) for r in resultados.values())


def _fijo(df: pd.DataFrame) -> dict:
    c = float(df["close"].iloc[-1])
    return {"predicted_close": c, "last_close": c}


def test_ruedas_es_el_calendario_de_los_tests() -> None:
    assert len(RUEDAS) == 600
