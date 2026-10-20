import json

import numpy as np
import optuna
import pandas as pd
import pytest

import src.simulacion.optuna_v2 as ov
import src.simulacion.reporte_optuna as ro
from src.simulacion.experimento import ReversionPanelModel
from src.transformer_v2 import TransformerV2Config

FOLDS = [("2026-03-02", "2026-03-20"), ("2026-04-01", "2026-04-17")]


def test_espacio_da_configs_validas() -> None:
    s = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    for _ in range(20):
        config = ov.espacio(s.ask())
        TransformerV2Config(**config)  # no levanta: d_model divisible por nhead, etc.
        assert config["target"] == "sin_media" and config["horizonte"] in ov.HORIZONTES


def test_ic_contra_reversion() -> None:
    fechas = ["2026-01-02"] * 4 + ["2026-01-05"] * 4
    futuro = [0.01, 0.02, 0.03, 0.04] * 2
    df = pd.DataFrame({"fecha": fechas, "symbol": list("ABCD") * 2, "futuro": futuro,
                       "pred": futuro, "pasado": futuro})  # fmt: skip
    r = ov.ic_contra_reversion(df)
    # pred ordena perfecto (1) y la reversion (-pasado) al reves (-1).
    assert (r["ic"], r["ic_rev"], r["dif"], r["fechas"]) == (1.0, -1.0, 2.0, 2)


def test_con_retornos_alinea_fechas() -> None:
    pred = pd.DataFrame({"fecha": ["2026-01-02", "2026-01-05"], "symbol": ["A", "A"],
                         "pred": [0.1, 0.2]})  # fmt: skip
    ret = pd.DataFrame({"fecha": ["2026-01-02", "2026-01-05"], "symbol": ["A", "A"],
                        "futuro": [0.01, np.nan], "pasado": [0.0, 0.0]})  # fmt: skip
    assert list(ov.con_retornos(pred, ret)["fecha"]) == ["2026-01-02"]


def test_pbo() -> None:
    rng = np.random.default_rng(0)
    # Una configuracion es mejor en todos los folds: nunca queda abajo fuera de muestra.
    consistente = rng.normal(0, 0.01, (6, 4))
    consistente[2] += 1.0
    assert ov.pbo(consistente) == 0.0
    # Puro ruido: la mejor dentro de muestra cae abajo seguido.
    assert 0.2 < ov.pbo(rng.normal(0, 1, (30, 8))) < 0.8
    assert ov.pbo(np.zeros((1, 4))) is None and ov.pbo(np.zeros((3, 3))) is None


class EvaluadorFalso(ov.Evaluador):
    """Usa la regla de reversion en vez del V2 (sin entrenar redes)."""

    def v2(self, config, seed=0):
        return ReversionPanelModel


@pytest.fixture
def evaluador(snapshot, monkeypatch):
    # El snapshot sintetico es corto para el Ridge del ensamble: la regla lo reemplaza.
    monkeypatch.setattr(ov, "EnsamblePanelModel", lambda *a, **k: ReversionPanelModel())
    return EvaluadorFalso(snapshot, historia=200)


def test_busqueda_cierre_e_informe(monkeypatch, tmp_path, evaluador, capsys) -> None:
    monkeypatch.setattr(ro, "compilar", lambda destino: None)
    s = ov.buscar(tmp_path, evaluador, horas=None, trials=3, folds=FOLDS)
    assert len(ov.completos(s)) == 3
    assert all(len(t.user_attrs["difs"]) == len(FOLDS) for t in s.trials)
    assert "trial 2 complete" in capsys.readouterr().out

    resumen = ov.cerrar(tmp_path, evaluador, FOLDS, FOLDS[:1], k=2, semillas=(1,))
    assert resumen["completos"] == 3 and resumen["pbo"] is not None
    top = pd.read_csv(tmp_path / "top.csv")
    assert len(top) == 2 and set(top["semillas"]) == {2}
    evaluacion = pd.read_csv(tmp_path / "evaluacion.csv", index_col=0)
    assert {"v2_optuna", "ensamble", "reversion_usd"} <= set(evaluacion.index)
    mejor = json.loads((tmp_path / "mejor.json").read_text())
    assert mejor["historia"] == ov.HISTORIA and "n_semillas" not in mejor
    tex = (tmp_path / "reporte" / "reporte.tex").read_text()
    assert "graficos/historia.png" in tex and "PBO" in tex
    # Retomar el estudio suma trials en vez de empezar de cero.
    assert len(ov.buscar(tmp_path, evaluador, horas=None, trials=1, folds=FOLDS).trials) == 4


def test_cerrar_sin_trials(tmp_path, evaluador) -> None:
    with pytest.raises(SystemExit, match="no tiene trials"):
        ov.cerrar(tmp_path, evaluador, FOLDS)


def test_paso_no_corta_la_cadena(tmp_path, capsys) -> None:
    registro = tmp_path / "noche.txt"
    assert ov._paso("bien", lambda: 1, registro) == 1
    assert ov._paso("mal", lambda: 1 / 0, registro) is None
    assert registro.read_text().count("\n") == 2 and "mal: fallo" in registro.read_text()


def test_noche_en_miniatura(monkeypatch, tmp_path, evaluador, snapshot_dir) -> None:
    import src.simulacion.un_modelo as um

    monkeypatch.setattr(ro, "compilar", lambda destino: None)
    monkeypatch.setattr(um, "RESULTADOS_DIR", str(tmp_path / "resultados"))
    monkeypatch.setattr(ov, "PRUEBA", ov.PRUEBA | {
        "folds": FOLDS, "evaluacion": FOLDS[:1],
        "simulaciones": (("prueba", "2026-06-01", "2026-06-12"),)})  # fmt: skip
    # La simulacion entrena el modelo de verdad: con reversion_panel es instantaneo.
    monkeypatch.setattr(um.ConfigUnModelo, "__post_init__", lambda self: None)
    original = um.correr

    def correr(cfg, dispositivo="cpu"):
        cfg.modelo, cfg.historia, cfg.config = "reversion_panel", 200, {}
        return original(cfg, dispositivo)

    monkeypatch.setattr(um, "correr", correr)
    salida = tmp_path / "estudio"
    ov.noche(salida, evaluador, None, str(snapshot_dir), "cpu", trials=2, prueba=True)
    registro = (salida / "noche.txt").read_text()
    assert "busqueda: ok" in registro and "cierre: ok" in registro
    assert "simulacion prueba: ok" in registro


def test_main_valida_argumentos(monkeypatch, snapshot_dir) -> None:
    with pytest.raises(SystemExit):
        ov.main(["noche", "--nombre", "x", "--snapshot", str(snapshot_dir)])
