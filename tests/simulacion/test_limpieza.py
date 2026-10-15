import numpy as np
import pandas as pd
import pytest

import src.simulacion.limpieza as limpieza
from src.simulacion.limpieza import (
    ajustar_salto,
    detectar_saltos,
    limpiar,
    limpiar_precios,
    reporte,
    retorno_mercado,
)
from tests.simulacion.conftest import RUEDAS, TICKERS, _ohlcv

SPLIT = RUEDAS[100]
CAIDA = RUEDAS[200]


def _precios() -> dict[str, pd.DataFrame]:
    prices = {t: _ohlcv(i, RUEDAS) for i, t in enumerate(TICKERS)}
    # Split no ajustado en ALUA: de ahi en adelante el precio vale un decimo.
    alua = prices["ALUA"].copy()
    alua.loc[alua.index >= SPLIT, ["open", "high", "low", "close"]] /= 10
    prices["ALUA"] = alua
    # Derrumbe general (todo el panel cae a la mitad): es real, no se toca.
    for t in TICKERS:
        df = prices[t].copy()
        df.loc[df.index >= CAIDA, ["open", "high", "low", "close"]] *= 0.5
        prices[t] = df
    # Una rueda con precio negativo.
    bbar = prices["BBAR"].copy()
    bbar.loc[RUEDAS[50], "close"] = -1.0
    prices["BBAR"] = bbar
    return prices


def test_detecta_el_split_y_no_el_derrumbe_general() -> None:
    prices = _precios()
    mercado = retorno_mercado({t: limpieza.sin_precios_no_positivos(p) for t, p in prices.items()})
    saltos = detectar_saltos(prices["ALUA"], mercado)
    assert [f for f, _ in saltos] == [SPLIT]
    assert saltos[0][1] == pytest.approx(0.1, rel=0.2)
    assert detectar_saltos(prices["GGAL"], mercado) == []


def test_ajustar_salto_deja_el_retorno_en_cero_y_ajusta_el_volumen() -> None:
    df = _ohlcv(0, RUEDAS)
    df.loc[df.index >= SPLIT, "close"] /= 10
    factor = float(df["close"].loc[SPLIT] / df["close"].shift().loc[SPLIT])
    ajustado = ajustar_salto(df, SPLIT, factor)
    assert np.log(ajustado["close"]).diff().loc[SPLIT] == pytest.approx(0.0)
    assert ajustado["volume"].iloc[0] == pytest.approx(1000.0 / factor)
    assert ajustado["volume"].loc[SPLIT] == 1000.0
    assert df["close"].iloc[0] != ajustado["close"].iloc[0]  # no muta la entrada


def test_limpiar_precios() -> None:
    limpios, correcciones = limpiar_precios(_precios())
    tipos = {(c["symbol"], c["tipo"]) for c in correcciones}
    assert tipos == {("ALUA", "salto"), ("BBAR", "precio<=0")}
    assert RUEDAS[50] not in limpios["BBAR"].index
    retornos = np.log(limpios["ALUA"]["close"]).diff()
    assert retornos.loc[SPLIT] == pytest.approx(0.0)
    assert retornos.loc[CAIDA] == pytest.approx(np.log(0.5), abs=0.1)
    texto = reporte(correcciones)
    assert "ALUA" in texto and "1 ruedas" in texto
    assert reporte([]) == "sin correcciones"


def test_limpiar_snapshot_y_main(snapshot, snapshot_dir, monkeypatch, capsys) -> None:
    limpio, correcciones = limpiar(snapshot)
    assert correcciones == [] and limpio is not snapshot
    assert set(limpio.prices) == set(snapshot.prices)
    monkeypatch.setattr("sys.argv", ["limpieza", "--snapshot", str(snapshot_dir)])
    limpieza.main()
    assert "sin correcciones" in capsys.readouterr().out
