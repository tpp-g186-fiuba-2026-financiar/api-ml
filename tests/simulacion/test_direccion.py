import sys

import numpy as np
import pandas as pd
import pytest

import src.simulacion.direccion as direccion
from src.simulacion.direccion import (
    horizontes,
    metricas,
    por_periodo,
    referencias,
    resumen,
    retornos_usd,
)

PERIODOS = [("2026-05-04", "2026-05-29"), ("2026-06-01", "2026-06-19")]


def test_metricas_siempre_alza_da_medio_y_la_inflacion_no_cuenta() -> None:
    real_usd = pd.Series([0.01, -0.02, 0.03, -0.01])
    real_ars = real_usd + 0.05  # en pesos todo sube
    m = metricas(pd.Series([1.0] * 4), real_usd, real_ars)
    assert m["bal_usd"] == 0.5 and m["acierto_ars"] == 1.0 and m["pct_alza"] == 1.0
    assert m["prec_alza"] == 0.0 and np.isnan(m["prec_baja"])
    perfecta = metricas(real_usd, real_usd, real_ars)
    assert perfecta["bal_usd"] == 1.0 and perfecta["prec_alza"] == 0.5


def test_acierto_top_mira_solo_las_predicciones_fuertes() -> None:
    real = pd.Series([0.01, -0.01] * 5 + [0.05, -0.05])
    # Las 10 chicas aciertan al reves; las 2 grandes (top 20%) aciertan.
    pred = pd.Series([-0.001, 0.001] * 5 + [0.5, -0.5])
    m = metricas(pred, real, real)
    assert m["bal_top"] == 1.0 and m["bal_usd"] < 0.5


def test_retornos_usd(snapshot) -> None:
    r = retornos_usd(snapshot).set_index(["fecha", "symbol"])
    df = snapshot.prices["GGAL"]
    ccl = snapshot.ccl_hasta(df.index[-1])
    usd = df["close"] / ccl.reindex(df.index)
    fecha = df.index[20]
    esperado_futuro = np.log(usd.iloc[25] / usd.iloc[20])
    esperado_pasado = np.log(usd.iloc[20] / usd.iloc[15])
    fila = r.loc[(fecha.strftime("%Y-%m-%d"), "GGAL")]
    assert fila["futuro"] == pytest.approx(esperado_futuro)
    assert fila["pasado"] == pytest.approx(esperado_pasado)
    assert np.isnan(r.loc[(df.index[-1].strftime("%Y-%m-%d"), "GGAL"), "futuro"])


def _escribir_predicciones(salida, snapshot, retornos) -> None:
    (salida / "predicciones").mkdir(parents=True)
    # "lejos" se mide a 20 ruedas (configs.json); el resto, a 5 (default).
    (salida / "configs.json").write_text('{"lejos": {"horizonte": 20}, "buena": {}, "x": null}')
    lejos = retornos_usd(snapshot, 20)
    rng = np.random.default_rng(0)
    for desde, hasta in PERIODOS:
        filas = retornos[(retornos["fecha"] >= desde) & (retornos["fecha"] <= hasta)]
        filas = filas.dropna(subset=["futuro"])
        for semilla in range(2):
            df = filas[["fecha", "symbol"]].copy()
            # "buena" predice el signo en dolares; "ruido" es azar.
            df["pred"] = filas["futuro"].to_numpy() + rng.normal(0, 1e-6, len(df))
            df["real"] = filas["futuro"].to_numpy() + 0.03
            df.to_csv(salida / "predicciones" / f"buena__{semilla}__{desde}.csv", index=False)
            df["pred"] = rng.normal(0, 1, len(df))
            df.to_csv(salida / "predicciones" / f"ruido__{semilla}__{desde}.csv", index=False)
            # Predice perfecto el retorno a 20 ruedas.
            f20 = df[["fecha", "symbol"]].merge(lejos, on=["fecha", "symbol"])
            f20 = f20.dropna(subset=["futuro"]).assign(pred=lambda d: d["futuro"], real=0.0)
            f20[["fecha", "symbol", "pred", "real"]].to_csv(
                salida / "predicciones" / f"lejos__{semilla}__{desde}.csv", index=False
            )


def test_por_periodo_y_resumen(snapshot, tmp_path, monkeypatch) -> None:
    monkeypatch.setitem(direccion.PERIODOS, "eleccion", PERIODOS)
    retornos = retornos_usd(snapshot)
    _escribir_predicciones(tmp_path, snapshot, retornos)
    assert horizontes(tmp_path) == {"lejos": 20, "buena": 5, "x": 5}
    tabla = por_periodo(tmp_path, {5: retornos, 20: retornos_usd(snapshot, 20)}, "eleccion")
    variantes = {"buena", "ruido", "lejos", *referencias(retornos.head(1))}
    assert set(tabla["variante"]) == variantes
    assert len(tabla) == len(variantes) * len(PERIODOS)
    tabla_resumen = resumen(tabla)
    assert tabla_resumen.index[0] == "buena" and tabla_resumen.loc["buena", "bal_usd"] == 1.0
    assert tabla_resumen.loc[direccion.REFERENCIA_ALZA, "bal_usd"] == 0.5
    assert tabla_resumen.loc[direccion.REFERENCIA_ORACULO, "bal_usd"] > 0.5
    assert tabla_resumen.loc["buena", "periodos_>0,5"] == len(PERIODOS)
    # Medida contra su horizonte (20), la prediccion perfecta a 20 ruedas acierta todo.
    assert tabla_resumen.loc["lejos", "bal_usd"] == 1.0
    assert tabla_resumen.loc["lejos", "bal_top"] == 1.0


def test_main(snapshot_dir, snapshot, tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setitem(direccion.PERIODOS, "eleccion", PERIODOS)
    monkeypatch.setattr(direccion, "DIR_EXPERIMENTOS", tmp_path)
    _escribir_predicciones(tmp_path / "corrida", snapshot, retornos_usd(snapshot))
    argv = ["direccion", "--nombre", "corrida", "--snapshot", str(snapshot_dir), "--limpiar"]
    monkeypatch.setattr(sys, "argv", argv)
    direccion.main()
    assert "buena" in capsys.readouterr().out
    assert (tmp_path / "corrida" / "direccion_eleccion.csv").exists()


def test_referencias_de_mercado() -> None:
    base = pd.DataFrame(
        {
            "fecha": ["d1", "d1", "d2", "d2"],
            "futuro": [0.03, 0.01, -0.02, 0.04],
            "pasado": [0.02, -0.04, np.nan, 0.06],
        }
    )
    refs = referencias(base)
    assert refs[direccion.REFERENCIA_ALZA]["pred"].eq(1.0).all()
    # Sin pasado no hay reversion ni timing de mercado.
    assert len(refs[direccion.REFERENCIA_REVERSION]) == 3
    np.testing.assert_allclose(
        refs[direccion.REFERENCIA_MERCADO_SIGUE]["pred"], [-0.01, -0.01, 0.06]
    )
    np.testing.assert_allclose(
        refs[direccion.REFERENCIA_MERCADO_REVIERTE]["pred"], [0.01, 0.01, -0.06]
    )
    np.testing.assert_allclose(refs[direccion.REFERENCIA_ORACULO]["pred"], [0.02, 0.02, 0.01, 0.01])
    np.testing.assert_allclose(
        refs[direccion.REFERENCIA_ORACULO_RELATIVO]["pred"], [0.01, -0.01, -0.03, 0.03]
    )
