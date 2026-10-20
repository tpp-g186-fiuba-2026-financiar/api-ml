import numpy as np
import pandas as pd
import pytest

from src.errors import ApiMlError, NotEnoughDataError
from src.panel.mercado import FEATURES_MERCADO, MercadoConfig, MercadoModel, dias_mercado
from tests.panel._panel_sintetico import (
    RUEDAS,
    TICKERS,
    histories,
    mercado_que_revierte,
    ohlcv,
    proveedor,
)


def test_config_invalida() -> None:
    with pytest.raises(ValueError, match="tipo"):
        MercadoConfig(tipo="gbm")
    with pytest.raises(ValueError, match="universo"):
        MercadoConfig(universo="todos")
    with pytest.raises(ValueError, match="zona_neutral"):
        MercadoConfig(zona_neutral=-1)
    with pytest.raises(ValueError, match="horizonte"):
        MercadoConfig(tipo="regla", horizonte=3)


def test_dias_mercado_no_mira_el_futuro_salvo_el_target() -> None:
    h = histories()
    ccl = proveedor(RUEDAS[-1])[0]
    completo = dias_mercado(h, ccl)
    assert list(completo.features.columns) == FEATURES_MERCADO
    corte = RUEDAS[400]
    cortado = dias_mercado({t: df[df.index <= corte] for t, df in h.items()}, ccl)
    pd.testing.assert_frame_equal(cortado.features, completo.features.loc[:corte])
    # El target de las ultimas `horizonte` fechas necesita futuro.
    assert cortado.target.iloc[-5:].isna().all()
    assert cortado.target.loc[: RUEDAS[390]].notna().sum() > 300
    assert completo.target.abs().max() <= 3.0
    assert completo.features["amplitud"].dropna().between(0, 1).all()


def test_target_es_el_retorno_del_mercado_a_horizonte() -> None:
    h = histories()
    ccl = proveedor(RUEDAS[-1])[0]
    d = dias_mercado(h, ccl, horizonte=5)
    usd = pd.DataFrame({t: np.log(df["close"] / ccl.reindex(df.index)) for t, df in h.items()})
    mercado = usd.diff().mean(axis=1)
    t = 300
    esperado = mercado.iloc[t + 1 : t + 6].sum() / (d.vol.iloc[t] * np.sqrt(5))
    assert d.target.iloc[t] == pytest.approx(np.clip(esperado, -3, 3))


@pytest.mark.parametrize(
    "config",
    [
        MercadoConfig(tipo="regla"),
        MercadoConfig(),
        MercadoConfig(tipo="logit", macro=True, zona_neutral=0.3, vida_media_anios=None),
    ],
)
def test_fit_y_predict_panel_da_lo_mismo_a_todas(config) -> None:
    modelo = MercadoModel(config, proveedor=proveedor)
    info = modelo.fit(histories())
    assert (info == {}) == (config.tipo == "regla")
    dfs = histories()
    dfs["CORTO"] = ohlcv(50, RUEDAS[:-1])  # sin dato el ultimo dia: afuera
    resultados = modelo.predict_panel(dfs)
    assert set(resultados) == set(TICKERS)
    retornos = {
        round(np.log(r["predicted_close"] / r["last_close"]), 3) for r in resultados.values()
    }
    assert len(retornos) == 1
    assert modelo.resumen()["version"] == modelo.version == f"panel-mercado-{config.tipo}"


@pytest.mark.parametrize("tipo", ["ridge", "logit"])
def test_aprende_la_reversion_del_mercado(tipo) -> None:
    m = mercado_que_revierte()
    h = histories(mercado=m)
    corte = RUEDAS[450]
    modelo = MercadoModel(MercadoConfig(tipo=tipo, alpha=1.0, c=1.0), proveedor=proveedor)
    modelo.fit({t: df[df.index <= corte] for t, df in h.items()})
    aciertos = []
    # Al cierre del ultimo dia de cada bloque: las proximas 5 ruedas son el bloque siguiente.
    for pos in range(464, 590, 5):
        dia = RUEDAS[pos]
        score, _ = modelo.score_dia({t: df[df.index <= dia] for t, df in h.items()})
        aciertos.append(np.sign(score) == np.sign(m[pos + 1 : pos + 6].sum()))
    assert np.mean(aciertos) > 0.7


def test_errores() -> None:
    with pytest.raises(ApiMlError, match="proveedor"):
        MercadoModel().fit(histories())
    sin_ccl = MercadoModel(proveedor=lambda hasta: (None, None))
    with pytest.raises(ApiMlError, match="CCL"):
        sin_ccl.fit(histories())
    sin_macro = MercadoModel(MercadoConfig(macro=True), proveedor=lambda h: (proveedor(h)[0], None))
    with pytest.raises(ApiMlError, match="macro"):
        sin_macro.fit(histories())
    corto = {t: df.iloc[:120] for t, df in histories().items()}
    with pytest.raises(NotEnoughDataError):
        MercadoModel(proveedor=proveedor).fit(corto)
    sin_entrenar = MercadoModel(proveedor=proveedor)
    with pytest.raises(NotEnoughDataError):
        sin_entrenar.predict_panel(histories())
    with pytest.raises(ApiMlError):
        sin_entrenar.predict_df(ohlcv(0))
    assert sin_entrenar.predict_panel({}) == {}
    assert sin_entrenar.predict_panel({"NOESTA": ohlcv(1)}) == {}


def test_dia_sin_mercado_valido_no_predice() -> None:
    modelo = MercadoModel(MercadoConfig(tipo="regla"), proveedor=proveedor)
    pocos = {t: df for t, df in list(histories().items())[:3]}  # < 5 acciones validas
    assert modelo.predict_panel(pocos) == {}
