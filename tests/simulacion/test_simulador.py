import json

import pandas as pd
import pytest

import src.simulacion.rearmar as rearmar_module
import src.simulacion.simulador as simulador
from src.errors import DataUnavailableError
from src.simulacion.entrenamiento import ModeloDiario
from src.simulacion.mesa import BENCHMARK_MERVAL, nombres_estrategias
from src.simulacion.rearmar import rearmar
from src.simulacion.simulador import Config, Simulacion, ultima_rueda_completa
from src.trend_common import derive_trend_output
from tests.simulacion.conftest import FakeMacro, FakeTrend

DESDE, HASTA = "2026-06-15", "2026-06-24"


class FakeModeloMacro:
    """Reemplazo de ``ModeloMacro``: promueve siempre, sin entrenar nada."""

    NOMBRE = "macro"

    def __init__(self, snapshot, model_path):
        self.model_path = model_path
        self.vigente = None

    def set_today(self, hasta):
        pass

    def retrain(self, hasta, force=False):
        self.vigente = FakeMacro()
        self.vigente.save(self.model_path)
        return {"model": self.NOMBRE, "promoted": True, "bootstrap": force}

    def resume(self):
        self.vigente = FakeMacro.load(self.model_path)


class FakeModeloContexto(FakeModeloMacro):
    """Reemplazo de ``ModeloContexto`` (mismo esquema mensual que el macro)."""

    NOMBRE = "contexto"


class FakeArima(FakeTrend):
    """Ademas falla en un ticker, como ARIMA cuando no converge."""

    def predict_df(self, df):
        if df.index[-1].day % 5 == 0:
            raise DataUnavailableError("no converge")
        return super().predict_df(df)


def _modelos(artefactos, transformer=None):
    artefactos.mkdir(parents=True, exist_ok=True)
    return [
        ModeloDiario("lstm", FakeTrend, FakeTrend.load, artefactos / "lstm.pkl"),
        ModeloDiario(
            "xgboost", lambda: FakeTrend(sesgo=0.01), FakeTrend.load, artefactos / "x.pkl"
        ),
    ]


@pytest.fixture
def entorno(monkeypatch, tmp_path, snapshot_dir):
    monkeypatch.setattr(simulador, "RESULTADOS_DIR", str(tmp_path))
    monkeypatch.setattr(simulador, "build_modelos_diarios", _modelos)
    monkeypatch.setattr(simulador, "ModeloMacro", FakeModeloMacro)
    monkeypatch.setattr(simulador, "ModeloContexto", FakeModeloContexto)
    monkeypatch.setattr(simulador, "ArimaTrendModel", FakeArima)
    return tmp_path


def _config(**kwargs) -> Config:
    base = {"nombre": "t", "desde": DESDE, "hasta": HASTA, "reentreno": "semanal",
            "history_days": 200}  # fmt: skip
    return Config(**(base | kwargs))


def _jsonl(path):
    return pd.read_json(path, lines=True)


def test_corrida_completa_retomar_y_rearmar(entorno, snapshot) -> None:
    Simulacion(_config(hasta="2026-06-19"), snapshot).correr()
    # Retoma desde donde quedo y la extiende.
    Simulacion(_config(), snapshot).correr()
    out = entorno / "t"

    valores = _jsonl(out / "valores.jsonl")
    ruedas = [d for d in snapshot.trading_days() if pd.Timestamp(DESDE) <= d <= pd.Timestamp(HASTA)]
    assert list(pd.to_datetime(valores["fecha"])) == ruedas
    assert set(nombres_estrategias()) | {BENCHMARK_MERVAL, "fecha", "ccl"} == set(valores.columns)
    assert valores.iloc[0][BENCHMARK_MERVAL] == pytest.approx(1_000_000.0)

    # Reentreno semanal: solo los viernes (mas el bootstrap y el macro del 1).
    promociones = _jsonl(out / "promociones.jsonl")
    reentrenos = promociones[promociones["model"] == "lstm"]["fecha"].astype(str).tolist()
    assert reentrenos == ["2026-06-12", "2026-06-19"]
    assert (out / "artefactos" / "versiones" / "lstm" / "2026-06-12.pkl").exists()

    # Rebalanceo cada 5 ruedas, con backtest de todos los modelos ese dia.
    pesos = _jsonl(out / "pesos.jsonl")
    assert sorted(pesos["fecha"].astype(str).unique()) == ["2026-06-15", "2026-06-22"]
    backtests = _jsonl(out / "backtests.jsonl")
    assert set(backtests["model"]) == {"lstm", "xgboost", "arima", "macro", "contexto"}
    bootstrap = promociones[promociones["bootstrap"].fillna(False).astype(bool)]
    assert {"macro", "contexto"} <= set(bootstrap["model"])
    assert (out / "artefactos" / "versiones" / "contexto").exists()
    elecciones = _jsonl(out / "elecciones.jsonl")
    assert set(elecciones["seleccion"]) == {"prod", "todos"}
    assert set(elecciones[elecciones["seleccion"] == "prod"]["model"]) <= {"lstm", "xgboost"}

    predicciones = _jsonl(out / "predicciones.jsonl")
    assert "backtest" not in predicciones.columns

    # Las carteras se rehacen igual a partir de lo guardado.
    rearmar(out, out / "rearmados" / "igual", snapshot)
    rearmado = _jsonl(out / "rearmados" / "igual" / "valores.jsonl")
    pd.testing.assert_frame_equal(rearmado, valores)

    rearmar(out, out / "rearmados" / "caro", snapshot, comision=0.02, rebalanceo=10)
    caro = _jsonl(out / "rearmados" / "caro" / "valores.jsonl")
    assert (
        caro["benchmark_equal_weight_rebalanceo"].iloc[-1]
        < valores["benchmark_equal_weight_rebalanceo"].iloc[-1]
    )
    assert not json.loads((out / "rearmados" / "caro" / "config.json").read_text())[
        "backtest_arrastrado"
    ]

    # Con un rebalanceo que no es multiplo del original se arrastra el ultimo backtest.
    rearmar(out, out / "rearmados" / "diario", snapshot, rebalanceo=1)
    config = json.loads((out / "rearmados" / "diario" / "config.json").read_text())
    assert config["backtest_arrastrado"]
    elecciones = _jsonl(out / "rearmados" / "diario" / "elecciones.jsonl")
    assert elecciones["fecha"].nunique() == len(valores)
    with pytest.raises(SystemExit):
        rearmar(out, out / "rearmados" / "x", snapshot, rebalanceo=0)


def test_al_terminar_registra_movimientos_con_y_sin_comision(entorno, snapshot) -> None:
    Simulacion(_config(), snapshot).correr()
    out = entorno / "t"
    con = pd.read_csv(out / "movimientos.csv")
    sin = pd.read_csv(out / "rearmados" / simulador.SIN_COMISION / "movimientos.csv")
    operaciones = _jsonl(out / "operaciones.jsonl")
    assert len(con) == len(operaciones) and (con["comision"] > 0).all()
    assert (sin["comision"] == 0).all()
    assert (
        json.loads((out / "rearmados" / "sin-comision" / "config.json").read_text())["comision"]
        == 0.0
    )

    # Cada operacion de una cartera de senal lleva la prediccion de su modelo ese dia.
    senal = con[con["cartera"] == "senal_lstm"]
    assert not senal.empty and (senal["modelo_usado"] == "lstm").all()
    preds = _jsonl(out / "predicciones.jsonl")
    preds = preds[preds["model"] == "lstm"].assign(fecha=lambda d: d["fecha"].astype(str))
    fila = senal.iloc[0]
    pred = preds[(preds["fecha"] == fila["fecha"]) & (preds["symbol"] == fila["ticker"])].iloc[0]
    assert fila["senal_modelo"] == pred["signal"] == fila["senal_lstm"]
    # En las de "mejor modelo", el modelo usado es el elegido para ese ticker.
    mejor = con[con["cartera"] == "senal_mejor_prod"]
    assert set(mejor["modelo_usado"].dropna()) <= {"lstm", "xgboost"}
    assert {"senal_macro", "senal_contexto", "retorno_real_5", "retorno_real_20"} <= set(con)
    compras = con[con["lado"] == "compra"]
    assert (compras.groupby(["fecha", "cartera"])["peso_objetivo"].sum() <= 1 + 1e-9).all()

    # Una corrida sin comision no se rearma (seria la misma).
    Simulacion(_config(nombre="gratis", comision=0.0), snapshot).correr()
    assert not (entorno / "gratis" / "rearmados").exists()


def test_retomar_con_otra_config_falla(entorno, snapshot) -> None:
    Simulacion(_config(hasta=DESDE), snapshot).correr()
    with pytest.raises(SystemExit):
        Simulacion(_config(comision=0.01), snapshot).correr()


def test_rango_sin_ruedas(entorno, snapshot) -> None:
    with pytest.raises(SystemExit):
        Simulacion(_config(desde="2030-01-01", hasta="2030-02-01"), snapshot).correr()


def test_rearmar_sin_predicciones(tmp_path, snapshot) -> None:
    config = {"nombre": "v", "capital": 1.0, "comision": 0.0, "rebalanceo": 5}
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(SystemExit):
        rearmar(tmp_path, tmp_path / "out", snapshot)


def test_ultima_rueda_completa(snapshot) -> None:
    assert ultima_rueda_completa(snapshot) == "2026-06-30"


def test_main_y_rearmar_main(entorno, snapshot_dir, monkeypatch) -> None:
    args = ["--nombre", "cli", "--desde", HASTA, "--snapshot", str(snapshot_dir),
            "--reentreno", "semanal", "--transformer", '{"target": "relativo"}']  # fmt: skip
    simulador.main(args)
    config = json.loads((entorno / "cli" / "config.json").read_text())
    assert config["hasta"] == "2026-06-30"
    assert config["reentreno"] == "semanal"
    assert config["transformer"] == {"target": "relativo"}

    monkeypatch.setattr(rearmar_module, "RESULTADOS_DIR", str(entorno))
    rearmar_module.main(["--nombre", "cli", "--salida", "otra", "--comision", "0.01"])
    assert (entorno / "cli" / "rearmados" / "otra" / "valores.jsonl").exists()


def test_reporte(entorno, snapshot, snapshot_dir, monkeypatch) -> None:
    import src.simulacion.reporte as reporte

    config = _config(desde="2026-06-01", snapshot_dir=str(snapshot_dir))
    Simulacion(config, snapshot).correr()
    out = entorno / "t"
    destino = reporte.generar(out, snapshot)

    resumen = json.loads((destino / "resumen.json").read_text())
    assert resumen["periodo"]["desde"] == "2026-06-01"
    carteras = {c["cartera"] for c in resumen["carteras"]}
    assert BENCHMARK_MERVAL in carteras and "senal_mejor_todos" in carteras
    assert "retorno_total_usd" in resumen["carteras"][0]
    assert set(resumen["elecciones"]) == {"prod", "todos"}
    assert resumen["calidad_modelos"]["todos_los_tickers"]["lstm"]["overall"]["n_predictions"] > 0

    # Al terminar, la corrida deja las carteras sin comision: el reporte las compara.
    sin = {c["cartera"]: c for c in resumen["sin_comision"]}
    for c in resumen["carteras"]:
        if c["cartera"] != BENCHMARK_MERVAL:
            assert sin[c["cartera"]]["retorno_total"] >= c["retorno_total"] - 1e-12

    texto = (destino / "reporte.md").read_text()
    assert "## Carteras" in texto and "Reentreno semanal" in texto
    assert "### Con y sin comision" in texto and "macro y contexto el 1" in texto
    # Las curvas de las carteras van junto a su tabla, antes de la calidad de los modelos.
    assert texto.index("carteras_senal.png") < texto.index("## Calidad de las predicciones")
    tex = (destino / "reporte.tex").read_text()
    assert r"\subsection*{Con y sin comision}" in tex
    assert tex.startswith(r"\documentclass") and tex.rstrip().endswith(r"\end{document}")
    assert r"\texttt{senal\_mejor\_todos}" in tex and r"graficos/carteras_senal.png" in tex
    graficos = {p.name for p in (destino / "graficos").iterdir()}
    assert {"carteras_senal.png", "carteras_consenso.png", "accuracy_mensual.png",
            "promociones.png", "elecciones.png"} <= graficos  # fmt: skip

    rearmar(out, out / "rearmados" / "otra", snapshot, comision=0.01)
    monkeypatch.setattr(reporte, "RESULTADOS_DIR", str(entorno))
    reporte.main(["--nombre", "t", "--rearmado", "otra"])
    rearmado = json.loads((out / "rearmados" / "otra" / "reporte" / "resumen.json").read_text())
    assert rearmado["config_carteras"]["comision"] == 0.01


def test_reporte_formatos() -> None:
    from src.simulacion.reporte import _num, _pct, _tex, latex, markdown

    assert _pct(None) == "-" and _pct(float("nan")) == "-" and _pct(0.0) == "0.0%"
    assert _pct(0.1234) == "+12.3%"
    assert _num(None) == "-" and _num(1.234) == "1.23"
    resumen = {
        "config": {"nombre": "x"},
        "config_carteras": {"capital": 1.0, "comision": 0.0, "rebalanceo": 5},
        "periodo": {"desde": "a", "hasta": "b", "ruedas": 1},
        "errores": 0,
        "rotacion": {},
        "carteras": [],
        "calidad_modelos": {},
        "promociones": {},
        "elecciones": {},
    }
    texto = markdown(resumen, [])
    assert "Reentreno de LSTM / XGBoost / Transformer: diario" in texto
    assert "## Graficos" not in texto
    assert "includegraphics" not in latex(resumen, [])
    resumen["config_carteras"]["backtest_arrastrado"] = True
    assert "ultimo backtest disponible" in markdown(resumen, [])
    assert _tex("a_b `c%d` $1 & #2") == r"a\_b \texttt{c\%d} \$1 \& \#2"
    assert _tex("{x}~^\\") == r"\{x\}\textasciitilde{}\textasciicircum{}\textbackslash{}"


class _Relativo(FakeTrend):
    """Modelo con target relativo: predice +5% en todos los tickers."""

    class config:  # noqa: N801
        target = "relativo"
        neutral_band = 0.001

    def predict_df(self, df):
        return derive_trend_output(df, 0.05, 5)


def test_predecir_centra_los_modelos_relativos(entorno, snapshot) -> None:
    sim = Simulacion(_config(), snapshot)
    sim.modelos[0].vigente = _Relativo()
    sim.modelos[1].vigente = FakeTrend()
    predicciones, _ = sim.predecir(pd.Timestamp(HASTA), set())
    relativas = predicciones["lstm"]
    assert relativas and all(r["signal"] == "neutral" for r in relativas.values())
    assert all(
        r["predicted_close"] == pytest.approx(r["last_close"], rel=1e-3) for r in relativas.values()
    )
