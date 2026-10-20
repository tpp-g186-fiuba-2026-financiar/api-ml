import json

import pandas as pd
import pytest

import src.simulacion.un_modelo as um
from src.simulacion.entrenamiento import reentreno_due
from src.simulacion.mesa import BENCHMARK_MERVAL, Mesa, nombres_estrategias
from src.simulacion.rearmar import rearmar

DESDE = "2026-06-01"


@pytest.fixture
def resultados(monkeypatch, tmp_path):
    monkeypatch.setattr(um, "RESULTADOS_DIR", str(tmp_path))
    return tmp_path


def _correr(snapshot_dir, *extra: str) -> None:
    um.main(["--nombre", "rapida", "--desde", DESDE, "--modelo", "reversion_panel",
             "--historia", "200", "--reentreno", "semanal", "--snapshot", str(snapshot_dir),
             *extra])  # fmt: skip


def test_reentreno_mensual_y_nunca() -> None:
    # Mensual: el dia de los crons del 1 (el 1 de julio de 2026 es rueda).
    fin_de_mes, primero = pd.Timestamp("2026-06-30"), pd.Timestamp("2026-07-01")
    assert not reentreno_due("mensual", fin_de_mes, primero)
    assert reentreno_due("mensual", primero, pd.Timestamp("2026-07-02"))
    assert not reentreno_due("mensual", primero, None)  # ultima rueda: no hay a quien servirle
    assert not reentreno_due("nunca", primero, pd.Timestamp("2026-07-02"))
    with pytest.raises(ValueError, match="desconocida"):
        reentreno_due("anual", fin_de_mes, primero)


def test_config_toma_modelo_e_historia_del_json() -> None:
    cfg = um.ConfigUnModelo(
        "x", DESDE, "2026-06-24", config={"modelo": "ensamble_panel", "historia": 1500, "a": 1}
    )
    assert (cfg.modelo, cfg.historia, cfg.config) == ("ensamble_panel", 1500, {"a": 1})
    with pytest.raises(ValueError, match="modelo"):
        um.ConfigUnModelo("x", DESDE, "2026-06-24", modelo="lstm")
    with pytest.raises(ValueError, match="reentreno"):
        um.ConfigUnModelo("x", DESDE, "2026-06-24", reentreno="anual")


def test_mesa_simple_sin_black_litterman(snapshot) -> None:
    nombres = nombres_estrategias(("cualquiera",), completa=False)
    assert "senal_cualquiera" in nombres and "histeresis_cualquiera" in nombres
    assert not [n for n in nombres if n.startswith(("bl_", "consenso_", "senal_mejor"))]
    mesa = Mesa(snapshot, 1000.0, 0.005, 5, ("cualquiera",), completa=False)
    assert set(mesa.carteras) == set(nombres)


def test_corrida_completa(resultados, snapshot, snapshot_dir, capsys) -> None:
    _correr(snapshot_dir)
    out = resultados / "rapida"
    config = json.loads((out / "config.json").read_text())
    assert config["solo_modelo"] == "reversion_panel"
    assert config["hasta"] == "2026-06-30"  # la ultima rueda completa del snapshot

    entrenamientos = [json.loads(x) for x in (out / "entrenamientos.jsonl").read_text().split("\n")
                      if x]  # fmt: skip
    assert [e["fecha"] for e in entrenamientos] == [
        "2026-05-29", "2026-06-05", "2026-06-12", "2026-06-19", "2026-06-26"
    ]  # fmt: skip
    predicciones = pd.read_json(out / "predicciones.jsonl", lines=True)
    assert set(predicciones["model"]) == {"reversion_panel"}

    direccion = pd.read_csv(out / "direccion.csv", index_col=0)
    assert {"reversion_panel", "reversion_usd", "oraculo_mercado"} <= set(direccion.index)
    # La regla revierte el retorno en pesos y la referencia en dolares: el CCL
    # es el mismo para todas las acciones del dia, asi que ordenan igual.
    fila, ref = direccion.loc["reversion_panel"], direccion.loc["reversion_usd"]
    assert fila["ic"] == pytest.approx(ref["ic"])
    assert fila["dif_ic"] == pytest.approx(0.0)

    carteras = pd.read_csv(out / "carteras.csv")
    esperadas = set(nombres_estrategias(("reversion_panel",), completa=False))
    assert set(carteras["cartera"]) == esperadas | {BENCHMARK_MERVAL}
    valores = pd.read_json(out / "valores.jsonl", lines=True)
    assert len(valores) == len(pd.bdate_range(DESDE, "2026-06-30"))
    assert valores["senal_reversion_panel"].nunique() > 1

    por_senal = pd.read_csv(out / "compra_por_senal.csv").set_index("grupo")
    assert por_senal.loc["todas", "casos"] == por_senal.drop(index="todas")["casos"].sum()
    confianza = pd.read_csv(out / "compra_confianza.csv")
    assert list(confianza["cobertura"]) == list(um.COBERTURAS_COMPRA)
    assert confianza["casos"].is_monotonic_decreasing

    reporte = out / "reporte"
    tex = (reporte / "reporte.tex").read_text()
    graficos = {p.name for p in (reporte / "graficos").iterdir()}
    assert {"balance_pesos.png", "balance_usd.png", "drawdown_usd.png", "comisiones.png",
            "ic.png", "confianza.png"} <= graficos  # fmt: skip
    assert "graficos/balance_usd.png" in tex and r"\end{document}" in tex
    assert "equal weight" in tex

    salida = capsys.readouterr().out
    assert "## Direccion en dolares" in salida and "senal_reversion_panel" in salida
    assert "Reporte:" in salida
    assert (out / "resumen.md").read_text() in salida

    with pytest.raises(SystemExit, match="ya existe"):
        _correr(snapshot_dir)

    # Otra comision sin entrenar: mismas predicciones, otras carteras.
    um.main(["--nombre", "rapida", "--solo-analisis", "--comision", "0.02"])
    assert "Comision 2.00%" in (out / "resumen.md").read_text()
    assert pd.read_json(out / "predicciones.jsonl", lines=True).equals(predicciones)
    otros = pd.read_json(out / "valores.jsonl", lines=True)
    assert len(otros) == len(valores)
    assert otros["senal_reversion_panel"].iloc[-1] < valores["senal_reversion_panel"].iloc[-1]

    # ``rearmar`` reconoce la corrida de un solo modelo.
    cfg = rearmar(out, out / "rearmados" / "igual", snapshot)
    assert cfg["solo_modelo"] == "reversion_panel"
    rearmado = pd.read_json(out / "rearmados" / "igual" / "valores.jsonl", lines=True)
    pd.testing.assert_frame_equal(rearmado, valores)


def test_config_invalida_no_crea_la_corrida(resultados, snapshot_dir) -> None:
    with pytest.raises(TypeError):
        um.main(["--nombre", "mala", "--desde", DESDE, "--modelo", "panel_gbm",
                 "--config", '{"no_existe": 1}', "--snapshot", str(snapshot_dir)])  # fmt: skip
    assert not (resultados / "mala").exists()


def test_sin_desde_o_sin_corrida(resultados, snapshot_dir) -> None:
    with pytest.raises(SystemExit):
        um.main(["--nombre", "x", "--snapshot", str(snapshot_dir)])
    with pytest.raises(SystemExit, match="no existe"):
        um.main(["--nombre", "x", "--solo-analisis"])
    (resultados / "vacia").mkdir()
    cfg = um.ConfigUnModelo("vacia", DESDE, "2026-06-24", snapshot_dir=str(snapshot_dir))
    with pytest.raises(SystemExit, match="no tiene predicciones"):
        um.analizar(cfg, None, None, resultados / "vacia")


def test_tabla_direccion_sin_horizonte_cumplido(snapshot) -> None:
    assert um.tabla_direccion(pd.DataFrame(), snapshot, "x") is None
    ultima = {"fecha": "2026-06-30", "symbol": "GGAL", "predicted_close": 101.0,
              "last_close": 100.0, "horizon_days": 5}  # fmt: skip
    assert um.tabla_direccion(pd.DataFrame([ultima]), snapshot, "x") is None


def test_metricas_compra_vacias(snapshot) -> None:
    vacio = um.metricas_compra(pd.DataFrame(), snapshot)
    assert all(t.empty for t in vacio.values())
    ultima = {"fecha": "2026-06-30", "symbol": "GGAL", "signal": "alza",
              "predicted_close": 101.0, "last_close": 100.0, "horizon_days": 5}  # fmt: skip
    assert all(t.empty for t in um.metricas_compra(pd.DataFrame([ultima]), snapshot).values())


def test_reporte_sin_pdflatex_o_si_falla(monkeypatch, tmp_path) -> None:
    import src.simulacion.reporte_un_modelo as rum

    monkeypatch.setattr(rum.shutil, "which", lambda _: None)
    assert rum.compilar(tmp_path) is None
    monkeypatch.setattr(rum.shutil, "which", lambda _: "/usr/bin/pdflatex")
    monkeypatch.setattr(rum.subprocess, "run", lambda *a, **k: type("P", (), {"returncode": 1}))
    assert rum.compilar(tmp_path) is None
    monkeypatch.setattr(rum.subprocess, "run", lambda *a, **k: type("P", (), {"returncode": 0}))
    assert rum.compilar(tmp_path) == tmp_path / "reporte.pdf"


def test_reporte_main_de_corrida_vacia(monkeypatch, tmp_path, capsys) -> None:
    import src.simulacion.reporte_un_modelo as rum

    monkeypatch.setattr(rum, "RESULTADOS_DIR", str(tmp_path))
    monkeypatch.setattr(rum, "compilar", lambda destino: None)
    (tmp_path / "x").mkdir()
    config = {"nombre": "x", "solo_modelo": "m", "desde": "2026-01-02", "hasta": "2026-02-02",
              "reentreno": "mensual", "historia": 750, "config": {}, "cuantil": None,
              "capital": 1000.0, "comision": 0.005, "rebalanceo": 5}  # fmt: skip
    (tmp_path / "x" / "config.json").write_text(json.dumps(config))
    rum.main(["--nombre", "x"])
    assert "reporte.tex" in capsys.readouterr().out
    assert r"\end{document}" in (tmp_path / "x" / "reporte" / "reporte.tex").read_text()
