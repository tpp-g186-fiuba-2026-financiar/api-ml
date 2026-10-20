import numpy as np
import pandas as pd
import pytest

from src.errors import ApiMlError, NotEnoughDataError
from src.panel.gbm import PanelGBMConfig, PanelGBMModel

RUEDAS = pd.bdate_range("2023-01-02", periods=600)
TICKERS = ["GGAL", "BMA", "YPFD", "PAMP", "ALUA", "TXAR", "CEPU", "EDN"]


def _ohlcv(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.001, 0.02, len(RUEDAS))))
    return pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99, "close": close,
         "volume": rng.integers(500, 2000, len(RUEDAS)).astype(float)},
        index=RUEDAS,
    )  # fmt: skip


def _histories() -> dict[str, pd.DataFrame]:
    return {t: _ohlcv(i) for i, t in enumerate(TICKERS)}


def _macro() -> pd.DataFrame:
    dias = pd.date_range(RUEDAS[0] - pd.Timedelta(days=400), RUEDAS[-1], freq="D")
    return pd.DataFrame(
        {"ccl": np.linspace(300, 1500, len(dias)), "oficial": np.linspace(200, 1400, len(dias)),
         "riesgo_pais": np.linspace(2000, 600, len(dias)), "badlar": 40.0},
        index=dias,
    )  # fmt: skip


def _proveedor(hasta):
    macro = _macro()
    macro = macro[macro.index <= hasta]
    return macro["ccl"], macro


def test_config_invalida() -> None:
    with pytest.raises(ValueError, match="tipo"):
        PanelGBMConfig(tipo="rf")
    with pytest.raises(ValueError, match="universo"):
        PanelGBMConfig(universo="todos")
    with pytest.raises(ValueError, match="cuantil_senal"):
        PanelGBMConfig(cuantil_senal=0.0)


@pytest.mark.parametrize(
    "config",
    [
        PanelGBMConfig(max_iter=20),
        PanelGBMConfig(tipo="ridge", macro=True, target="residual"),
        PanelGBMConfig(moneda="ars", rankear=False, vida_media_anios=None, max_iter=20),
        PanelGBMConfig(tipo="ridge", cuantil_senal=0.25),
    ],
)
def test_fit_y_predict_panel(config) -> None:
    modelo = PanelGBMModel(config, proveedor=_proveedor)
    info = modelo.fit(_histories())
    assert info["filas"] > 2000
    assert ("a_brecha" in modelo.columnas) == config.macro
    dia = RUEDAS[-1]
    dfs = {t: df[df.index <= dia] for t, df in _histories().items()}
    dfs["NOESTA"] = _ohlcv(99)  # fuera del universo: se ignora
    resultados = modelo.predict_panel(dfs)
    assert set(resultados) == set(TICKERS)
    for t, r in resultados.items():
        assert r["last_close"] == pytest.approx(dfs[t]["close"].iloc[-1], rel=1e-3)
        assert r["signal"] in ("alza", "baja", "neutral")
    if config.cuantil_senal:
        senales = [r["signal"] for r in resultados.values()]
        assert senales.count("alza") == senales.count("baja") == 2
    assert modelo.resumen()["version"] == modelo.version


def test_ticker_sin_dato_el_ultimo_dia_queda_afuera() -> None:
    modelo = PanelGBMModel(PanelGBMConfig(max_iter=10), proveedor=_proveedor)
    modelo.fit(_histories())
    dfs = _histories()
    dfs["GGAL"] = dfs["GGAL"].iloc[:-1]
    assert "GGAL" not in modelo.predict_panel(dfs) and len(modelo.predict_panel(dfs)) == 7


def test_casos_vacios_y_errores() -> None:
    modelo = PanelGBMModel(PanelGBMConfig(max_iter=10), proveedor=_proveedor)
    with pytest.raises(NotEnoughDataError, match="entrenado"):
        modelo.predict_panel(_histories())
    with pytest.raises(NotEnoughDataError, match="entrenado"):
        modelo._predecir(np.zeros((1, 3)))
    with pytest.raises(ApiMlError, match="predict_panel"):
        modelo.predict_df(_ohlcv(0))
    modelo.fit(_histories())
    assert modelo.predict_panel({}) == {}
    assert modelo.predict_panel({"NOESTA": _ohlcv(0)}) == {}
    quietos = {t: df.assign(volume=0.0) for t, df in _histories().items()}
    assert modelo.predict_panel(quietos) == {}


def test_datos_externos_faltantes() -> None:
    with pytest.raises(ApiMlError, match="proveedor"):
        PanelGBMModel(PanelGBMConfig()).fit(_histories())
    sin_macro = PanelGBMModel(
        PanelGBMConfig(macro=True), proveedor=lambda h: (_proveedor(h)[0], None)
    )
    with pytest.raises(ApiMlError, match="macro"):
        sin_macro.fit(_histories())
    pocas = {t: df.iloc[:150] for t, df in _histories().items()}
    with pytest.raises(NotEnoughDataError, match="pocas filas"):
        PanelGBMModel(PanelGBMConfig(moneda="ars")).fit(pocas)


def test_aprende_una_senal_de_reversion() -> None:
    """Con retornos que revierten la semana anterior, el modelo ordena bien."""
    rng = np.random.default_rng(7)
    n, k = len(RUEDAS), 12
    tickers = [*TICKERS, "SUPV", "BBAR", "TGSU2", "LOMA"]
    r = rng.normal(0, 0.02, (n, k))
    for t in range(5, n):
        r[t] -= 0.15 * r[t - 5 : t].sum(axis=0)
    close = 100 * np.exp(np.cumsum(r, axis=0))
    histories = {
        tk: pd.DataFrame({"open": close[:, j], "high": close[:, j] * 1.01,
                          "low": close[:, j] * 0.99, "close": close[:, j], "volume": 1000.0},
                         index=RUEDAS)
        for j, tk in enumerate(tickers)
    }  # fmt: skip
    corte = 450
    modelo = PanelGBMModel(PanelGBMConfig(tipo="ridge", moneda="ars"))
    modelo.fit({tk: df.iloc[:corte] for tk, df in histories.items()})
    ics = []
    for t in range(corte, n - 5, 5):
        pred = modelo.predict_panel({tk: df.iloc[: t + 1] for tk, df in histories.items()})
        p = pd.Series(
            {tk: np.log(v["predicted_close"] / v["last_close"]) for tk, v in pred.items()}
        )
        real = pd.Series(
            {tk: np.log(close[t + 5, j] / close[t, j]) for j, tk in enumerate(tickers)}
        )
        ics.append(p.rank().corr(real[p.index].rank()))
    assert np.mean(ics) > 0.1
