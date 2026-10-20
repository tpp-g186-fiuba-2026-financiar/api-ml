import numpy as np
import pytest

from src.errors import ApiMlError, NotEnoughDataError
from src.panel.meta import ESCALA, MetaReversionConfig, MetaReversionModel
from tests.panel._panel_sintetico import RUEDAS, TICKERS, histories, ohlcv, proveedor


def test_config_invalida() -> None:
    with pytest.raises(ValueError, match="tipo"):
        MetaReversionConfig(tipo="rf")
    with pytest.raises(ValueError, match="universo"):
        MetaReversionConfig(universo="todos")


@pytest.mark.parametrize(
    "config",
    [
        MetaReversionConfig(),
        MetaReversionConfig(tipo="gbm", max_iter=20, macro=True, zona_neutral=0.3),
    ],
)
def test_el_lado_es_siempre_el_de_la_reversion(config) -> None:
    modelo = MetaReversionModel(config, proveedor=proveedor)
    info = modelo.fit(histories())
    assert info["filas"] > 2000 and 0 < info["tasa_acierto"] < 1
    dfs = histories()
    dfs["NOESTA"] = ohlcv(99)
    resultados = modelo.predict_panel(dfs)
    assert set(resultados) <= set(TICKERS) and len(resultados) >= 6
    ccl = proveedor(RUEDAS[-1])[0]
    for s, r in resultados.items():
        pred = np.log(r["predicted_close"] / r["last_close"])
        usd = np.log(dfs[s]["close"] / ccl.reindex(dfs[s].index))
        pasado = usd.iloc[-1] - usd.iloc[-6]
        assert np.sign(pred) == -np.sign(pasado)
        assert 0 < abs(pred) <= ESCALA + 1e-3
    assert modelo.resumen()["version"] == modelo.version == f"panel-meta-reversion-{config.tipo}"


def test_errores() -> None:
    sin_entrenar = MetaReversionModel(proveedor=proveedor)
    with pytest.raises(NotEnoughDataError):
        sin_entrenar.predict_panel(histories())
    with pytest.raises(ApiMlError):
        sin_entrenar.predict_df(ohlcv(0))
    corto = {t: df.iloc[:60] for t, df in histories().items()}
    with pytest.raises(NotEnoughDataError):
        sin_entrenar.fit(corto)
    modelo = MetaReversionModel(proveedor=proveedor)
    modelo.fit(histories())
    assert modelo.predict_panel({}) == {}
    assert modelo.predict_panel({"NOESTA": ohlcv(1)}) == {}
    # Sin dato valido el ultimo dia (precio quieto => fuera de la mascara de liquidez).
    quietos = {t: df.assign(close=100.0, open=100.0, high=100.0, low=100.0) for t, df in
               histories().items()}  # fmt: skip
    assert modelo.predict_panel(quietos) == {}
