import json

import numpy as np
import pandas as pd
import pytest

import src.simulacion.reglas as rg
from tests.simulacion.conftest import TICKERS

PERIODO = [("2026-03-02", "2026-05-29")]


@pytest.fixture
def p(snapshot) -> pd.DataFrame:
    return rg.precios_usd(snapshot, TICKERS)


def test_precios_usd_divide_por_el_ccl(snapshot, p) -> None:
    dia = pd.Timestamp("2026-06-10")
    ccl = snapshot.ccl_hasta(dia)
    esperado = np.log(snapshot.prices["GGAL"].at[dia, "close"] / ccl.loc[:dia].iloc[-1])
    assert p.at[dia, "GGAL"] == pytest.approx(esperado)
    assert list(p.columns) == TICKERS


def test_reglas_solo_miran_hacia_atras(p) -> None:
    corte = pd.Timestamp("2026-04-15")
    completas = rg.reglas(p)
    cortadas = rg.reglas(p.loc[:corte])
    for nombre, pred in completas.items():
        pd.testing.assert_frame_equal(pred.loc[:corte], cortadas[nombre], check_freq=False)
    dia = pd.Timestamp("2026-06-10")
    i = p.index.get_loc(dia)
    assert completas["rev_5"].at[dia, "BMA"] == pytest.approx(
        -(p["BMA"].iloc[i] - p["BMA"].iloc[i - 5])
    )
    assert completas["mom_6_1"].at[dia, "BMA"] == pytest.approx(
        p["BMA"].iloc[i - 20] - p["BMA"].iloc[i - 125]
    )
    mercado = completas["mercado_mom_60"].loc[dia]
    assert mercado.nunique() == 1


def test_filas_formato_de_significancia(p) -> None:
    df = rg.filas(rg.reglas(p)["rev_5"], p, 5, PERIODO)
    assert set(df.columns) == {"desde", "fecha", "symbol", "pred", "futuro", "pasado"}
    assert set(df["desde"]) == {"2026-03-02"}
    assert df["fecha"].min() >= "2026-03-02" and df["fecha"].max() <= "2026-05-29"
    fila = df[(df["fecha"] == "2026-04-01") & (df["symbol"] == "ALUA")].iloc[0]
    i = p.index.get_loc(pd.Timestamp("2026-04-01"))
    assert fila["futuro"] == pytest.approx(p["ALUA"].iloc[i + 5] - p["ALUA"].iloc[i])
    assert fila["pred"] == pytest.approx(-fila["pasado"])


def test_umbrales_usan_solo_fechas_previas(p) -> None:
    pred = pd.DataFrame(1.0, index=p.index[:100], columns=["A", "B"])
    pred.iloc[70:] = 100.0  # el salto no cambia el umbral del mismo dia
    umbral = rg.umbrales(pred, 0.5, ventana=10, minimo=5)
    assert umbral.iloc[:5].isna().all()
    assert umbral.iloc[70] == 1.0
    assert umbral.iloc[80] == 100.0


def test_apuestas_solo_compran_lo_fuerte(p) -> None:
    pred = pd.DataFrame(0.0, index=p.index, columns=p.columns)
    pred["GGAL"], pred["BMA"] = 1.0, -1.0  # BMA es fuerte pero negativa: no se compra
    df = rg.apuestas(pred, p, 5, PERIODO, cobertura=0.2)
    assert set(df["symbol"]) == {"GGAL"}
    fila = df.iloc[0]
    i = p.index.get_loc(fila["fecha"])
    assert fila["ret"] == pytest.approx(np.expm1(p["GGAL"].iloc[i + 5] - p["GGAL"].iloc[i]))
    ret = np.expm1(p.shift(-5) - p).iloc[i]
    assert fila["panel"] == pytest.approx(ret.mean())


def test_resumen_apuestas() -> None:
    assert rg.resumen_apuestas(pd.DataFrame()) == {"apuestas": 0}
    fechas = pd.bdate_range("2026-01-01", periods=60)
    df = pd.DataFrame({"fecha": fechas, "symbol": "X", "ret": 0.03, "panel": 0.01})
    r = rg.resumen_apuestas(df, costo=0.01, bloque=5)
    assert (r["apuestas"], r["fechas"], r["acierto"]) == (60, 60, 1.0)
    assert r["neto"] == pytest.approx(0.02) and r["exceso"] == pytest.approx(0.01)
    assert r["exceso_lo"] == pytest.approx(0.01) and r["exceso_hi"] == pytest.approx(0.01)


def test_main_con_corrida(monkeypatch, tmp_path, snapshot_dir, p, capsys) -> None:
    monkeypatch.setattr(rg, "DIR_EXPERIMENTOS", tmp_path)
    monkeypatch.setitem(rg.PERIODOS, "prueba", PERIODO)
    corrida = tmp_path / "modelos"
    (corrida / "predicciones").mkdir(parents=True)
    (corrida / "configs.json").write_text(json.dumps({"m20": {"horizonte": 20}, "m5": {}}))
    mom = rg.reglas(p)["mom_60"].loc["2026-03-02":"2026-05-29"].stack().rename("pred")
    filas = mom.rename_axis(["fecha", "symbol"]).reset_index()
    filas["fecha"] = filas["fecha"].dt.strftime("%Y-%m-%d")
    filas.assign(real=0.0).to_csv(corrida / "predicciones" / "m20__0__2026-03-02.csv", index=False)
    filas.assign(real=0.0).to_csv(corrida / "predicciones" / "m5__0__2026-03-02.csv", index=False)

    rg.main(["--nombre", "r3", "--snapshot", str(snapshot_dir), "--periodos", "prueba",
             "--corrida", "modelos"])  # fmt: skip
    salida = tmp_path / "r3"
    h20 = pd.read_csv(salida / "ic_prueba_h20.csv", index_col=0)
    # El modelo de la corrida mide lo mismo que la regla de la que sale.
    assert h20.loc["m20", "ic"] == pytest.approx(h20.loc["mom_60", "ic"])
    assert "m20" not in pd.read_csv(salida / "ic_prueba_h5.csv", index_col=0).index
    selectiva = pd.read_csv(salida / "selectiva.csv")
    assert set(selectiva["cobertura"]) == set(rg.COBERTURAS)
    assert {"m5", "m20", "rev_20"} <= set(selectiva["variante"])
    assert "Senal selectiva" in capsys.readouterr().out
