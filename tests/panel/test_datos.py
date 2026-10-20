import numpy as np
import pandas as pd
import pytest

from src.panel.datos import (
    SECTORES,
    UNIVERSOS,
    armar_panel,
    ccl_en,
    en_dolares,
    ruedas_activas,
    universo,
)

RUEDAS = pd.bdate_range("2025-01-01", periods=120)


def _ohlcv(seed: int, index: pd.DatetimeIndex = RUEDAS) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, len(index))))
    return pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99, "close": close,
         "volume": 1000.0},
        index=index,
    )  # fmt: skip


def _ccl() -> pd.Series:
    dias = pd.date_range(RUEDAS[0] - pd.Timedelta(days=10), RUEDAS[-1], freq="D")
    return pd.Series(np.linspace(1000, 1500, len(dias)), index=dias)


def test_universos_tienen_sector_y_sin_cedears() -> None:
    assert universo("panel") == UNIVERSOS["panel"]
    assert set(universo("panel")) < set(universo("amplio"))
    for nombre in UNIVERSOS:
        for t in universo(nombre):
            assert t in SECTORES
            assert not t.endswith("D") or t == "YPFD"
    with pytest.raises(ValueError, match="universo"):
        universo("todos")


def test_ccl_en_solo_mira_hacia_atras() -> None:
    ccl = pd.Series([10.0, 20.0, 30.0, 31.0], index=pd.to_datetime(
        ["2025-01-02", "2025-01-06", "2025-01-08", "2025-01-08"]))  # fmt: skip
    fechas = pd.to_datetime(["2025-01-01", "2025-01-03", "2025-01-06", "2025-01-07", "2025-01-09"])
    alineado = ccl_en(ccl, fechas)
    assert np.isnan(alineado.iloc[0])
    assert alineado.iloc[1:].tolist() == [10.0, 20.0, 20.0, 31.0]


def test_en_dolares_divide_precios_y_no_volumen() -> None:
    df = _ohlcv(0)
    usd = en_dolares(df, _ccl())
    tipo_cambio = ccl_en(_ccl(), df.index)
    pd.testing.assert_series_equal(usd["close"], df["close"] / tipo_cambio, check_names=False)
    pd.testing.assert_series_equal(usd["volume"], df["volume"])


def test_ruedas_activas_detecta_precio_quieto_y_volumen_cero() -> None:
    df = _ohlcv(1)
    df.iloc[60:, df.columns.get_loc("close")] = df["close"].iloc[59]
    activa = ruedas_activas(df, ventana=20, umbral=0.3)
    assert activa.iloc[:60].all()
    assert not activa.iloc[80:].any()
    sin_volumen = _ohlcv(2)
    sin_volumen["volume"] = 0.0
    assert not ruedas_activas(sin_volumen).any()
    sin_columna = _ohlcv(3).drop(columns="volume")
    assert ruedas_activas(sin_columna).all()


def test_armar_panel_alinea_sin_rellenar() -> None:
    corto = _ohlcv(4).iloc[50:]
    corto = corto.drop(corto.index[10])  # un hueco
    histories = {"GGAL": _ohlcv(5), "BMA": corto, "OTRO": _ohlcv(6), "VACIO": _ohlcv(7).iloc[:0]}
    panel = armar_panel(histories, ["GGAL", "BMA", "VACIO", "NOESTA"], _ccl())
    assert panel.tickers == ["GGAL", "BMA"] and panel.moneda == "usd"
    assert panel.fechas.equals(RUEDAS)
    assert panel["close"]["BMA"].iloc[:50].isna().all()
    hueco = corto.index[9] + pd.offsets.BDay(1)
    assert np.isnan(panel["close"].loc[hueco, "BMA"]) and not panel.mascara.loc[hueco, "BMA"]
    assert panel.mascara["GGAL"].all()
    assert panel.sectores() == ["financiero", "financiero"]
    cobertura = panel.cobertura()
    assert cobertura.loc["BMA", "desde"] == RUEDAS[50]
    assert cobertura.loc["GGAL", "validas"] == 1.0


def test_armar_panel_en_pesos_y_mascara_por_ccl_y_precios() -> None:
    df = _ohlcv(8)
    df.iloc[5, df.columns.get_loc("close")] = 0.0
    ars = armar_panel({"GGAL": df}, ["GGAL"], moneda="ars")
    assert ars["close"]["GGAL"].iloc[10] == df["close"].iloc[10]
    assert not ars.mascara["GGAL"].iloc[5]
    ccl_tardio = _ccl()[_ccl().index >= RUEDAS[30]]
    usd = armar_panel({"GGAL": df}, ["GGAL"], ccl_tardio)
    assert not usd.mascara["GGAL"].iloc[:30].any() and usd.mascara["GGAL"].iloc[31:].all()
    sector_desconocido = armar_panel({"XYZ": df}, ["XYZ"], moneda="ars")
    assert sector_desconocido.sectores() == ["otro"]
    vacio = armar_panel({"XYZ": df.assign(volume=0.0)}, ["XYZ"], moneda="ars")
    assert vacio.cobertura()["desde"].isna().all()


def test_armar_panel_errores() -> None:
    with pytest.raises(ValueError, match="moneda"):
        armar_panel({"GGAL": _ohlcv(0)}, ["GGAL"], moneda="eur")
    with pytest.raises(ValueError, match="CCL"):
        armar_panel({"GGAL": _ohlcv(0)}, ["GGAL"])
    with pytest.raises(ValueError, match="historia"):
        armar_panel({"GGAL": _ohlcv(0)}, ["BMA"], _ccl())


@pytest.mark.parametrize("moneda", ["usd", "ars"])
def test_armar_panel_no_mira_el_futuro(moneda) -> None:
    """Ninguna celda de la fecha t cambia si se borran los datos posteriores a t."""
    quieto = _ohlcv(9)
    quieto.iloc[70:90, quieto.columns.get_loc("close")] = quieto["close"].iloc[69]
    histories = {"GGAL": _ohlcv(10), "BMA": _ohlcv(11).iloc[20:], "PAMP": quieto}
    tickers = list(histories)
    completo = armar_panel(histories, tickers, _ccl(), moneda=moneda, ventana_liquidez=20)
    for corte in RUEDAS[[25, 60, 80, 100]]:
        hasta = {t: df[df.index <= corte] for t, df in histories.items()}
        ccl = _ccl()[_ccl().index <= corte]
        parcial = armar_panel(hasta, tickers, ccl, moneda=moneda, ventana_liquidez=20)
        n = len(parcial.fechas)
        assert parcial.fechas.equals(completo.fechas[:n])
        pd.testing.assert_frame_equal(parcial.mascara, completo.mascara.iloc[:n])
        for campo in ("close", "volume", "high"):
            pd.testing.assert_frame_equal(parcial[campo], completo[campo].iloc[:n])
