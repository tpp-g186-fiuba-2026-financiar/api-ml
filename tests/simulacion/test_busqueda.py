import json

import numpy as np
import pandas as pd
import pytest

import src.simulacion.busqueda as busqueda
from src.simulacion.busqueda import (
    PLAN_ABLACION,
    PLAN_V2,
    PLAN_V2_RAPIDO,
    REFERENCIAS,
    correr,
    ejecutar,
    muestrear_hiper,
    resumen,
    trabajos,
)
from src.simulacion.experimento import ReversionPanelModel
from src.transformer import TransformerTrainConfig, TransformerTrendModel
from tests.simulacion.conftest import FakeTrend

PERIODOS = [("2026-05-04", "2026-05-29"), ("2026-06-01", "2026-06-19")]


def test_plan_ablacion_y_muestreo_dan_configs_validas() -> None:
    for config in PLAN_ABLACION.values():
        TransformerTrainConfig(**config)
    planes = muestrear_hiper(10, {"target": "relativo"}, seed=3)
    assert len(planes) == 11 and planes["hiper_base"] == {"target": "relativo"}
    for config in planes.values():
        assert TransformerTrainConfig(**config).target == "relativo"
    assert muestrear_hiper(10, {"target": "relativo"}, seed=3) == planes


def test_plan_v2_da_configs_validas() -> None:
    from src.transformer_v2 import TransformerV2Config

    for nombre, config in (PLAN_V2 | PLAN_V2_RAPIDO).items():
        config = dict(config)
        config.pop("historia", None)
        if config.pop("modelo") == "transformer_v2":
            assert TransformerV2Config(**config).n_semillas == 1, nombre
    assert PLAN_V2["original"] == {"modelo": "transformer"}


def test_historia_cambia_solo_las_ruedas_de_entrenamiento(snapshot, monkeypatch) -> None:
    llamadas = []

    def predecir(snapshot, fabrica, desde, hasta, tickers, days_fit=None):
        llamadas.append(days_fit)
        fabrica()  # la config del modelo no recibe "historia"
        return pd.DataFrame(
            {"fecha": ["2026-05-04"] * 5, "symbol": list("ABCDE"),
             "pred": np.arange(5.0), "signal": "alza", "real": np.arange(5.0)}
        )  # fmt: skip

    monkeypatch.setattr(busqueda, "predecir_periodo", predecir)
    trabajo = {"variante": "r", "semilla": 0, "desde": "2026-05-04", "hasta": "2026-05-29"}
    for config in ({"modelo": "reversion_panel", "historia": 1500}, {"modelo": "reversion_panel"}):
        ejecutar({**trabajo, "config": config}, snapshot)
    assert llamadas == [1500, None]


def test_trabajos() -> None:
    lista = trabajos({"a": {}, "b": {}}, 3, PERIODOS, baselines=["reversion"])
    assert len(lista) == 2 * 3 * 2 + 2
    assert {t["semilla"] for t in lista if t["variante"] == "reversion"} == {0}


def test_fabrica() -> None:
    modelo = busqueda._fabrica({"variante": "x", "config": {"window": 10}, "semilla": 2})()
    assert isinstance(modelo, TransformerTrendModel)
    assert modelo.config.seed == busqueda.SEMILLA_BASE + 2 and modelo.config.window == 10
    assert (
        busqueda._fabrica({"variante": "reversion", "config": None, "semilla": 0})
        is (busqueda.VARIANTES["reversion"])
    )


def test_fabrica_elige_el_modelo_sin_tocar_la_config() -> None:
    config = {"modelo": "reversion_panel"}
    modelo = busqueda._fabrica({"variante": "r", "config": config, "semilla": 0})()
    assert isinstance(modelo, ReversionPanelModel)
    assert config == {"modelo": "reversion_panel"}
    explicito = {"modelo": "transformer", "window": 10}
    modelo = busqueda._fabrica({"variante": "t", "config": explicito, "semilla": 0})()
    assert isinstance(modelo, TransformerTrendModel) and modelo.config.window == 10


def test_fabrica_panel_gbm_con_proveedor_del_snapshot(snapshot) -> None:
    from src.panel_gbm import PanelGBMModel

    trabajo = {"variante": "g", "config": {"modelo": "panel_gbm", "max_iter": 5}, "semilla": 1}
    modelo = busqueda._fabrica(trabajo, snapshot)()
    assert isinstance(modelo, PanelGBMModel) and modelo.config.seed == busqueda.SEMILLA_BASE + 1
    dia = snapshot.trading_days()[100]
    ccl, macro = modelo.proveedor(dia)
    assert ccl.index[-1] == dia and macro.index[-1] == dia
    sin_macro = busqueda.proveedor_snapshot(
        type(snapshot)(prices=snapshot.prices, tickers=[], rate=None, macro_raw={}, meta={})
    )
    assert sin_macro(dia) == (None, None)


def test_fabrica_ensamble_panel(snapshot) -> None:
    from src.panel_ensamble import EnsamblePanelModel

    config = {"modelo": "ensamble_panel", "peso_reversion": 0.3, "alpha": 5.0, "cuantil_senal": 0.2}
    modelo = busqueda._fabrica({"variante": "e", "config": config, "semilla": 0}, snapshot)()
    assert isinstance(modelo, EnsamblePanelModel)
    assert modelo.config.peso_reversion == 0.3 and modelo.config.cuantil_senal == 0.2
    assert modelo.config.ridge.cuantil_senal is None
    assert modelo.config.ridge.tipo == "ridge" and modelo.config.ridge.alpha == 5.0
    assert modelo.ridge.proveedor is not None


def test_fabrica_panel_transformer(snapshot) -> None:
    from src.panel_transformer import PanelTransformerModel

    config = {"modelo": "panel_transformer", "ventana": 20, "d_model": 16, "nhead": 2}
    modelo = busqueda._fabrica({"variante": "p", "config": config, "semilla": 3}, snapshot)()
    assert isinstance(modelo, PanelTransformerModel)
    assert modelo.config.seed == busqueda.SEMILLA_BASE + 3 and modelo.config.ventana == 20
    assert modelo.proveedor is not None and modelo.dispositivo == "cpu"


def test_trabajos_rechaza_modelo_desconocido() -> None:
    with pytest.raises(ValueError, match="nada"):
        trabajos({"x": {"modelo": "nada"}}, 1, PERIODOS)


def test_reversion_panel_por_la_busqueda_da_el_mismo_ic(snapshot) -> None:
    desde, hasta = PERIODOS[0]
    comunes = {"semilla": 0, "desde": desde, "hasta": hasta}
    baseline = ejecutar({"variante": "reversion", "config": None, **comunes}, snapshot)
    panel = ejecutar(
        {"variante": "rp", "config": {"modelo": "reversion_panel"}, **comunes}, snapshot
    )
    assert panel["n"] > 0 and panel["ic"] == baseline["ic"]


def _fake_fabrica(trabajo, snapshot=None):
    return lambda: FakeTrend(sesgo=0.001 * trabajo["semilla"])


def test_ejecutar(snapshot, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(busqueda, "_fabrica", _fake_fabrica)
    (tmp_path / "predicciones").mkdir()
    trabajo = {"variante": "f", "config": {}, "semilla": 1, "desde": PERIODOS[0][0],
               "hasta": PERIODOS[0][1]}  # fmt: skip
    fila = ejecutar(trabajo, snapshot, tmp_path)
    assert fila["variante"] == "f" and fila["n"] > 0
    assert (tmp_path / "predicciones" / f"f__1__{PERIODOS[0][0]}.csv").exists()


def test_correr_retoma_y_resumen(snapshot_dir, tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(busqueda, "_fabrica", _fake_fabrica)
    monkeypatch.setitem(busqueda.PERIODOS, "eleccion", PERIODOS)
    lista = trabajos({"f": {}, "g": {}}, 2, PERIODOS)
    assert correr(str(snapshot_dir), "b", lista, 2, base=tmp_path) == len(lista)
    # Retomar: no repite nada.
    assert correr(str(snapshot_dir), "b", lista, 2, base=tmp_path) == 0
    lineas = (tmp_path / "b" / "resultados.jsonl").read_text().splitlines()
    assert len(lineas) == len(lista)
    assert json.loads((tmp_path / "b" / "configs.json").read_text()) == {"f": {}, "g": {}}

    tabla = resumen("b", base=tmp_path)
    assert set(tabla.index) == {"f", "g"}
    assert (tabla["semillas"] == 2).all() and (tabla["periodos"] == 2).all()
    assert tabla["ic_ens"].notna().all()


def test_resumen_con_referencias(snapshot_dir, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(busqueda, "_fabrica", _fake_fabrica)
    monkeypatch.setitem(busqueda.PERIODOS, "eleccion", PERIODOS)
    correr(str(snapshot_dir), "ref", trabajos({"f": {}, "g": {}}, 2, PERIODOS), 1, base=tmp_path)
    correr(str(snapshot_dir), "nueva", trabajos({"h": {}}, 1, PERIODOS), 1, base=tmp_path)
    tabla = resumen("nueva", base=tmp_path, referencias=["ref:f"])
    assert set(tabla.index) == {"h", "ref:f"}
    assert tabla.loc["ref:f", "semillas"] == 2 and not np.isnan(tabla.loc["ref:f", "ic_ens"])
    assert {"ventaja_alza", "ventaja_baja", "pct_senal", "exceso_alza"} <= set(tabla.columns)
    with pytest.raises(SystemExit, match="no tiene resultados"):
        resumen("nueva", base=tmp_path, referencias=["ref:x"])
    correr(str(snapshot_dir), "limpia", trabajos({"h": {}}, 1, PERIODOS[:1]), 1, base=tmp_path,
           limpiar=True)  # fmt: skip
    with pytest.raises(SystemExit, match="misma limpieza"):
        resumen("limpia", base=tmp_path, referencias=["ref:f"])


def test_ventaja_con_corridas_viejas_y_senales_que_no_se_dieron() -> None:
    g = pd.DataFrame({"precision_alza": [0.6, None], "base_up": [0.5, 0.4]})
    assert busqueda._ventaja(g, "precision_alza", g["base_up"]) == pytest.approx(0.1)
    assert np.isnan(busqueda._ventaja(g, "precision_baja", 1 - g["base_up"]))
    assert np.isnan(busqueda._ventaja(g, "ret_alza", g.get("ret_medio")))


def test_correr_no_mezcla_opciones(snapshot_dir, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(busqueda, "_fabrica", _fake_fabrica)
    lista = trabajos({"f": {}}, 1, PERIODOS[:1])
    assert correr(str(snapshot_dir), "o", lista, 1, base=tmp_path, limpiar=True) == 1
    with pytest.raises(SystemExit, match="limpiar"):
        correr(str(snapshot_dir), "o", lista, 1, base=tmp_path)


def test_init_worker_con_limpieza_y_dispositivo(snapshot_dir, monkeypatch) -> None:
    monkeypatch.setattr(busqueda, "_SNAPSHOT", None)
    monkeypatch.setattr(busqueda, "_DISPOSITIVO", "cpu")
    busqueda._init_worker(str(snapshot_dir), True, "cuda")
    assert busqueda._SNAPSHOT is not None
    modelo = busqueda._fabrica({"variante": "x", "config": {}, "semilla": 0})()
    assert modelo.dispositivo == "cuda"


def test_resumen_sin_ensamble(snapshot_dir, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(busqueda, "_fabrica", _fake_fabrica)
    monkeypatch.setitem(busqueda.PERIODOS, "eleccion", PERIODOS[:1])
    correr(str(snapshot_dir), "c", trabajos({"f": {}}, 1, PERIODOS[:1]), 1, base=tmp_path)
    tabla = resumen("c", base=tmp_path)
    assert tabla["ic_ens"].isna().all()
    assert tabla.loc["f", "ic_medio"] == pytest.approx(tabla.loc["f", "min_periodo"])


@pytest.mark.parametrize(
    ("plan", "extra"),
    [
        ("ablacion", []),
        ("hiper", ["--n", "1", "--base", '{"target": "relativo"}']),
        ("config", ["--configs", '{"x": {}}', "--baselines"]),
    ],
)
def test_main(snapshot_dir, tmp_path, monkeypatch, capsys, plan, extra) -> None:
    corridas = []
    monkeypatch.setattr(busqueda, "DIR_EXPERIMENTOS", tmp_path)
    monkeypatch.setattr(busqueda, "correr", lambda *a, **k: corridas.append(a[2]))
    monkeypatch.setattr(busqueda, "resumen", lambda *a, **k: "tabla")
    monkeypatch.setattr(
        "sys.argv",
        ["busqueda", "--nombre", "m", "--plan", plan, "--semillas", "1",
         "--snapshot", str(snapshot_dir), *extra],
    )  # fmt: skip
    busqueda.main()
    assert corridas and corridas[0]
    assert "tabla" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("extra", "esperadas"),
    [([], None), (["--referencias"], REFERENCIAS), (["--referencias", "a:b"], ["a:b"])],
)
def test_main_referencias(tmp_path, monkeypatch, capsys, extra, esperadas) -> None:
    pedidas = []
    monkeypatch.setattr(busqueda, "DIR_EXPERIMENTOS", tmp_path)
    monkeypatch.setattr(
        busqueda, "resumen", lambda *a, **k: pedidas.append(k["referencias"]) or "tabla"
    )
    monkeypatch.setattr("sys.argv", ["busqueda", "--nombre", "m", "--resumen", *extra])
    busqueda.main()
    assert pedidas == [esperadas]
