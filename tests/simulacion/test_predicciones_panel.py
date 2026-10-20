import dataclasses
import json

import pandas as pd
import pytest

import src.simulacion.predicciones_panel as pp
import src.simulacion.reporte as reporte
import src.simulacion.simulador as simulador
from src.simulacion.experimento import ReversionPanelModel
from src.simulacion.mesa import BENCHMARK_MERVAL, MODELOS_PANEL, Mesa, nombres_estrategias
from src.simulacion.rearmar import rearmar
from src.simulacion.simulador import Config, Simulacion
from tests.simulacion.conftest import TICKERS
from tests.simulacion.test_simulador import FakeArima, FakeModeloMacro, _modelos

DESDE, HASTA = "2026-06-01", "2026-06-24"


class ContadorReversion(ReversionPanelModel):
    """Reversion que cuenta los entrenamientos y recuerda hasta que dia entreno."""

    version = "contador"
    entrenamientos: list[pd.Timestamp] = []

    def fit(self, histories):
        ContadorReversion.entrenamientos.append(max(df.index[-1] for df in histories.values()))
        return {}


def _escalado(snapshot, factor: float):
    """Copia del snapshot con precios multiplicados (como un ajuste por split)."""
    precios = {s: df * factor for s, df in snapshot.prices.items()}
    return dataclasses.replace(snapshot, prices=precios)


def test_predecir_dia_usa_el_cierre_crudo(snapshot) -> None:
    limpio = _escalado(snapshot, 0.1)
    dia = pd.Timestamp("2026-06-10")
    preds = pp.predecir_dia(ReversionPanelModel(), snapshot, limpio, dia, 200)
    assert set(preds) == set(TICKERS)  # NUEVA no es del panel lider
    for symbol, p in preds.items():
        cierre = snapshot.prices[symbol].at[dia, "close"]
        assert p["last_close"] == pytest.approx(cierre, abs=1e-4)
        # La misma prediccion relativa que con los precios crudos.
        directa = ReversionPanelModel().predict_df(snapshot.history(symbol, dia, 200))
        assert p["predicted_close"] == pytest.approx(directa["predicted_close"], rel=1e-4)
    senales = [p["signal"] for p in preds.values()]
    # Cuantil 0,15 sobre 7 tickers: rango > 0,85 (6/7 y 7/7) y rango <= 0,15 (1/7).
    assert senales.count("alza") == 2 and senales.count("baja") == 1


def test_predecir_dia_sin_cotizacion_o_sin_cuantil(snapshot) -> None:
    sabado = pd.Timestamp("2026-06-13")
    assert pp.predecir_dia(ReversionPanelModel(), snapshot, snapshot, sabado, 200) == {}
    dia = pd.Timestamp("2026-06-10")
    preds = pp.predecir_dia(ReversionPanelModel(), snapshot, snapshot, dia, 200, cuantil=None)
    # Banda fija de +-1%: la senal sale de derive_trend_output.
    directa = ReversionPanelModel().predict_df(snapshot.history("GGAL", dia, 200))
    assert preds["GGAL"]["signal"] == directa["signal"]


def test_predecir_dia_descarta_precios_invalidos(snapshot) -> None:
    dia = pd.Timestamp("2026-06-10")
    crudo = dataclasses.replace(
        snapshot,
        prices={
            **snapshot.prices,
            "GGAL": snapshot.prices["GGAL"].assign(close=0.0),
            "BMA": snapshot.prices["BMA"].drop(index=dia),
        },
    )
    preds = pp.predecir_dia(ReversionPanelModel(), crudo, snapshot, dia, 200)
    assert "GGAL" not in preds and "BMA" not in preds and "ALUA" in preds


def test_generar_reentrena_cada_semana(snapshot, tmp_path) -> None:
    ContadorReversion.entrenamientos = []
    destino = tmp_path / "panel" / "x.jsonl"
    destino.parent.mkdir()
    destino.write_text("viejo\n")
    filas = pp.generar(
        "reversion", ContadorReversion, snapshot, snapshot, DESDE, HASTA, days=200, destino=destino
    )
    # Primer modelo con datos hasta la rueda anterior; despues, el ultimo dia de cada semana
    # (salvo el ultimo del periodo, que no predice nada despues).
    assert [str(d.date()) for d in ContadorReversion.entrenamientos] == [
        "2026-05-29", "2026-06-05", "2026-06-12", "2026-06-19"
    ]  # fmt: skip
    leidas = [json.loads(linea) for linea in destino.read_text().splitlines()]
    assert leidas == json.loads(json.dumps(filas, default=str))
    assert {f["fecha"] for f in filas} == {str(d.date()) for d in pd.bdate_range(DESDE, HASTA)}
    primera = filas[0]
    assert primera["model"] == "reversion" and primera["model_version"] == "contador-2026-05-29"
    assert {"signal", "predicted_close", "last_close", "symbol", "as_of"} <= set(primera)
    assert filas[-1]["model_version"] == "contador-2026-06-19"


def test_generar_sin_ruedas(snapshot) -> None:
    with pytest.raises(ValueError, match="sin ruedas"):
        pp.generar("x", ReversionPanelModel, snapshot, snapshot, "2030-01-01", "2030-02-01")


def test_fabricas(snapshot) -> None:
    fabricas = pp.fabricas(snapshot)
    assert set(fabricas) == set(MODELOS_PANEL)
    ensamble = fabricas["ensamble"]()
    assert ensamble.config.cuantil_senal == pp.CUANTIL_SENAL
    transformer = fabricas["panel_transformer"]()
    assert transformer.config.ventana == 1 and transformer.config.n_semillas == 5
    assert isinstance(fabricas["reversion"](), ReversionPanelModel)


@pytest.fixture
def corrida(monkeypatch, tmp_path, snapshot, snapshot_dir):
    """Corrida del simulador (modelos falsos), como las de ``simulacion/resultados``."""
    monkeypatch.setattr(simulador, "RESULTADOS_DIR", str(tmp_path))
    monkeypatch.setattr(simulador, "build_modelos_diarios", _modelos)
    monkeypatch.setattr(simulador, "ModeloMacro", FakeModeloMacro)
    monkeypatch.setattr(simulador, "ArimaTrendModel", FakeArima)
    monkeypatch.setattr(pp, "RESULTADOS_DIR", str(tmp_path))
    config = Config(nombre="orig", desde=DESDE, hasta=HASTA, reentreno="semanal",
                    history_days=200, snapshot_dir=str(snapshot_dir))  # fmt: skip
    Simulacion(config, snapshot).correr()
    return tmp_path / "orig"


def test_copia_unir_y_rearmar(corrida, snapshot, monkeypatch) -> None:
    monkeypatch.setattr(pp, "fabricas", lambda snap, disp="cpu": {"reversion": ReversionPanelModel})
    original = (corrida / "predicciones.jsonl").read_text()
    pp.main(["--corrida", "orig", "--copia", "copia", "--modelos", "reversion"])
    copia = corrida.parent / "copia"

    assert (corrida / "predicciones.jsonl").read_text() == original  # la original no cambia
    config = json.loads((copia / "config.json").read_text())
    assert config["nombre"] == "copia" and config["origen"] == "orig"
    texto = (copia / "predicciones.jsonl").read_text()
    assert texto.startswith(original) and '"model": "reversion"' in texto[len(original) :]
    # Unir de nuevo no duplica.
    assert pp.unir(copia) == len(texto[len(original) :].splitlines())
    assert (copia / "predicciones.jsonl").read_text() == texto
    with pytest.raises(SystemExit):
        pp.main(["--corrida", "orig", "--copia", "orig"])

    # Con los parametros de la corrida, las carteras viejas quedan igual y se suman las de panel.
    cfg = rearmar(copia, copia / "rearmados" / "igual", snapshot)
    assert cfg["modelos_panel"] == ["reversion"]
    viejo = pd.read_json(corrida / "valores.jsonl", lines=True)
    nuevo = pd.read_json(copia / "rearmados" / "igual" / "valores.jsonl", lines=True)
    assert set(nuevo.columns) == set(viejo.columns) | set(
        nombres_estrategias(("reversion",))[len(nombres_estrategias()) :]
    )
    pd.testing.assert_frame_equal(nuevo[viejo.columns], viejo)
    assert nuevo["senal_reversion"].iloc[-1] != nuevo["senal_reversion"].iloc[0]

    monkeypatch.setattr(reporte, "RESULTADOS_DIR", str(corrida.parent))
    reporte.main(["--nombre", "copia", "--rearmado", "igual"])
    destino = copia / "rearmados" / "igual" / "reporte"
    resumen = json.loads((destino / "resumen.json").read_text())
    assert "reversion" in resumen["calidad_modelos"]["todos_los_tickers"]
    assert "histeresis_reversion" in resumen["rotacion"]
    graficos = {p.name for p in (destino / "graficos").iterdir()}
    assert "carteras_reglas_reversion.png" in graficos


def test_rearmar_sin_modelos_de_panel_no_los_agrega(corrida, snapshot) -> None:
    cfg = rearmar(corrida, corrida / "rearmados" / "igual", snapshot)
    assert "modelos_panel" not in cfg
    valores = pd.read_json(corrida / "rearmados" / "igual" / "valores.jsonl", lines=True)
    assert set(valores.columns) == set(nombres_estrategias()) | {BENCHMARK_MERVAL, "fecha", "ccl"}


def test_mesa_rechaza_modelos_desconocidos(snapshot) -> None:
    with pytest.raises(ValueError, match="desconocidos"):
        Mesa(snapshot, 1000.0, 0.005, 5, ("lstm",))


def test_mesa_reglas_de_panel(snapshot) -> None:
    mesa = Mesa(snapshot, 1000.0, 0.005, 5, ("ensamble",))
    dia = pd.Timestamp("2026-06-10")
    retornos = dict(zip(TICKERS, [0.05, 0.04, 0.0, -0.01, -0.02, -0.03, -0.06], strict=True))
    preds = {
        "ensamble": {
            t: {"predicted_close": 100 * (1 + r), "last_close": 100.0, "horizon_days": 5,
                "rsi": 50.0, "condition": "neutral", "as_of": "2026-06-10", "symbol": t,
                "model_version": "v",
                "signal": "alza" if r > 0.03 else "baja" if r < -0.05 else "neutral"}
            for t, r in retornos.items()
        }
    }  # fmt: skip
    mesa.carteras["histeresis_ensamble"].tenencias = {"BMA": 1.0, "GGAL": 1.0}
    objetivos, errores, _ = mesa.objetivos(dia, preds, {}, TICKERS)
    assert not [e for e in errores if "ensamble" in e["cartera"]]
    assert objetivos["senal_ensamble"] == {"ALUA": 0.5, "BBAR": 0.5}
    # Compra el 15% superior (ALUA, BBAR); mantiene BMA (tercero, dentro del 30%) pero no GGAL.
    tercio = 1 / 3
    assert objetivos["histeresis_ensamble"] == pytest.approx(
        {"ALUA": tercio, "BBAR": tercio, "BMA": tercio}
    )
    assert objetivos["cuantil30_ensamble"] == pytest.approx(
        {"ALUA": tercio, "BBAR": tercio, "BMA": tercio}
    )
    assert "bl_ensamble_moderate" in objetivos
    inclinado = objetivos["inclinado_ensamble"]
    assert sum(inclinado.values()) == pytest.approx(1.0)
    assert inclinado["ALUA"] == pytest.approx(3 * inclinado["GGAL"])
