import sys

import numpy as np
import pandas as pd
import pytest

import src.simulacion.direccion as direccion
import src.simulacion.significancia as significancia
from src.simulacion.direccion import retornos_usd
from src.simulacion.significancia import (
    conteos_por_fecha,
    legible,
    metricas_desde_sumas,
    pesos_bootstrap,
)
from src.trend_common import acierto_balanceado, ic_por_fecha

PERIODOS = [("2026-05-04", "2026-05-29"), ("2026-06-01", "2026-06-19")]


def _panel(seed: int = 0, n_fechas: int = 40, n_tickers: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    fechas = pd.bdate_range("2026-01-05", periods=n_fechas).strftime("%Y-%m-%d")
    df = pd.DataFrame(
        [(f, f"T{i}") for f in fechas for i in range(n_tickers)], columns=["fecha", "symbol"]
    )
    df["desde"] = np.where(df["fecha"] < fechas[n_fechas // 2], "p1", "p2")
    mercado = pd.Series(rng.normal(0, 0.02, n_fechas), index=fechas)
    df["futuro"] = df["fecha"].map(mercado) + rng.normal(0, 0.01, len(df))
    df["pred"] = rng.normal(0, 1, len(df))
    return df


def test_metricas_desde_sumas_coinciden_con_las_directas() -> None:
    df = _panel()
    m = metricas_desde_sumas(conteos_por_fecha(df).sum())
    assert m["bal_usd"] == pytest.approx(acierto_balanceado(df["pred"], df["futuro"]))
    assert m["ic"] == pytest.approx(ic_por_fecha(df["pred"], df["futuro"], df["fecha"]))
    dia = df.groupby("fecha")[["pred", "futuro"]].mean()
    assert m["bal_mercado"] == pytest.approx(acierto_balanceado(dia["pred"], dia["futuro"]))
    rel = df[["pred", "futuro"]] - df.groupby("fecha")[["pred", "futuro"]].transform("mean")
    assert m["bal_relativo"] == pytest.approx(acierto_balanceado(rel["pred"], rel["futuro"]))


def test_descomposicion_mercado_y_relativo() -> None:
    df = _panel()
    mercado = df.groupby("fecha")["futuro"].transform("mean")
    solo_mercado = metricas_desde_sumas(conteos_por_fecha(df.assign(pred=mercado)).sum())
    assert solo_mercado["bal_mercado"] == 1.0 and solo_mercado["bal_relativo"] == 0.5
    solo_relativo = metricas_desde_sumas(
        conteos_por_fecha(df.assign(pred=df["futuro"] - mercado)).sum()
    )
    assert solo_relativo["bal_relativo"] == 1.0 and solo_relativo["ic"] == pytest.approx(1.0)


def test_bal_top_usa_el_umbral_de_cada_periodo() -> None:
    df = _panel()
    # En p2 las predicciones son 100 veces mas grandes: con un umbral comun, el top
    # seria todo de p2.
    df.loc[df["desde"] == "p2", "pred"] *= 100
    c = conteos_por_fecha(df)
    en_top = c[["top_sube_n", "top_baja_n"]].sum(axis=1)
    p1 = c.index < df.loc[df["desde"] == "p2", "fecha"].min()
    assert en_top[p1].sum() > 0 and en_top[~p1].sum() > 0


def test_sin_casos_de_un_lado() -> None:
    m = metricas_desde_sumas(
        {f"{p}_{k}": 0.0 for p in ("usd", "top", "merc", "rel") for k in
         ("sube_ok", "sube_n", "baja_ok", "baja_n")} | {"ic_suma": 0.0, "ic_n": 0.0}
    )  # fmt: skip
    assert all(v == 0.5 for k, v in m.items() if k != "ic") and m["ic"] == 0.0


def test_pesos_bootstrap() -> None:
    pesos = pesos_bootstrap(23, n_boot=50, bloque=5, seed=1)
    assert pesos.shape == (50, 23)
    np.testing.assert_array_equal(pesos.sum(1), 23)
    # Bloque mas largo que la serie: un solo bloque con todo.
    np.testing.assert_array_equal(pesos_bootstrap(3, n_boot=2, bloque=10), np.ones((2, 3)))


def test_significancia_separa_senal_de_ruido() -> None:
    # Con la semilla 0 el "ruido" da 0,54 por azar (pasa 1 de cada ~100).
    df = _panel(seed=3, n_fechas=120)
    variantes = {
        "buena": df.assign(pred=df["futuro"] + np.random.default_rng(1).normal(0, 0.01, len(df))),
        "ruido": df,
        "alza": df.assign(pred=1.0),
    }
    tabla = significancia.significancia(variantes, contra="alza", n_boot=300)
    assert tabla.index[0] == "buena"
    assert tabla.loc["buena", "bal_usd_lo"] > 0.5
    assert tabla.loc["ruido", "bal_usd_lo"] < 0.5 < tabla.loc["ruido", "bal_usd_hi"]
    assert tabla.loc["buena", "p_mejor_bal_usd"] == 1.0
    assert tabla.loc["alza", "dif_bal_usd"] == 0.0
    texto = legible(tabla, "alza")
    assert "vs_alza_bal_usd" in texto and texto.loc["alza", "bal_usd"].startswith("0.500")
    # Sin referencia no hay diferencias.
    assert "dif_bal_usd" not in significancia.significancia(variantes, contra=None, n_boot=10)


def test_main(snapshot_dir, snapshot, tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setitem(direccion.PERIODOS, "eleccion", PERIODOS)
    monkeypatch.setattr(significancia, "DIR_EXPERIMENTOS", tmp_path)
    salida = tmp_path / "corrida" / "predicciones"
    salida.mkdir(parents=True)
    retornos = retornos_usd(snapshot).dropna(subset=["futuro"])
    for desde, hasta in PERIODOS:
        filas = retornos[(retornos["fecha"] >= desde) & (retornos["fecha"] <= hasta)]
        filas.assign(pred=filas["futuro"], real=0.0)[["fecha", "symbol", "pred", "real"]].to_csv(
            salida / f"buena__0__{desde}.csv", index=False
        )
    argv = [
        "significancia",
        "--nombre",
        "corrida",
        "--snapshot",
        str(snapshot_dir),
        "--n-boot",
        "50",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    significancia.main()
    salida_texto = capsys.readouterr().out
    assert "buena" in salida_texto and "oraculo_mercado" in salida_texto
    tabla = pd.read_csv(tmp_path / "corrida" / "significancia_eleccion.csv", index_col=0)
    assert tabla.loc["buena", "bal_usd"] == 1.0
