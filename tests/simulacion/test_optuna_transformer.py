"""Busqueda de hiperparametros del Transformer con Optuna."""

import pandas as pd
import pytest

import src.simulacion.optuna_transformer as ot

CHICA = {"window": 10, "d_model": 8, "nhead": 2, "dim_feedforward": 16, "epochs": 2}


def _evaluar_falso(config, por_ticker, semilla, reporte=None):
    """Acierto que crece con el learning rate (y un poco con la semilla)."""
    valor = 0.5 + 10 * config.get("learning_rate", 1e-3) + semilla * 1e-4
    for k in range(3):
        if reporte is not None:
            reporte(k, {"val_acierto_balanceado": valor})
    return {
        "val_acierto_balanceado": valor,
        "val_ic": 0.01,
        "val_directional_accuracy": valor,
        "epocas": 2,
    }


@pytest.fixture
def datos_falsos(monkeypatch):
    monkeypatch.setattr(ot, "evaluar", _evaluar_falso)
    monkeypatch.setattr(
        ot, "_DATOS", {(c, v): {} for c in ("fin", "2025-12-30") for v in ot.VENTANAS}
    )


def test_preparar_y_evaluar_con_reporte_por_bloque(snapshot) -> None:
    por_ticker = ot.preparar(snapshot, snapshot.trading_days()[-1], 10)
    assert por_ticker and all(X.shape[1] == 10 for X, _, _ in por_ticker.values())
    reportes = []
    metricas = ot.evaluar(CHICA, por_ticker, 42, lambda k, m: reportes.append(k))
    assert reportes == [0, 1, 2]
    assert 0 <= metricas["val_acierto_balanceado"] <= 1 and metricas["folds"] == 3


def test_buscar_confirmar_y_resumen(tmp_path, datos_falsos) -> None:
    estudio = ot.buscar(tmp_path, "fin", trials=6, procesos=2, hilos=1)
    assert ot._terminados(estudio) == 6
    # Retomable: con los trials ya hechos no corre nada nuevo.
    assert ot._terminados(ot.buscar(tmp_path, "fin", trials=6, procesos=2, hilos=1)) == 6

    configs = ot.candidatos(estudio, top=2)
    assert list(configs)[0] == "actual" and len(configs) == 3
    nombre = list(configs)[1]
    assert configs[nombre]["dim_feedforward"] == 2 * configs[nombre]["d_model"]

    filas = ot.confirmar(configs, ["fin", "2025-12-30"], semillas=2, procesos=2, hilos=1)
    assert len(filas) == 3 * 2 * 2
    tabla = ot.resumen(filas)
    assert tabla.loc["actual", "vs_actual"] == 0 and tabla.loc["actual", "gana_en"] == 0
    assert {"bal_fin", "bal_2025-12-30", "desvio", "ic"} <= set(tabla.columns)

    texto = ot.escribir_resumen(tmp_path, tabla, configs, estudio)
    assert "Mejor" in texto and (tmp_path / "resumen.md").exists()


def test_main(tmp_path, snapshot_dir, monkeypatch, datos_falsos, capsys) -> None:
    monkeypatch.setattr(ot, "DIR_OPTUNA", tmp_path)
    monkeypatch.setattr(ot, "preparar", lambda snapshot, corte, ventana: {})
    argumentos = ["x", "--nombre", "p", "--snapshot", str(snapshot_dir), "--trials", "2",
                  "--top", "1", "--semillas", "1", "--procesos", "1", "--hilos", "1"]  # fmt: skip
    monkeypatch.setattr("sys.argv", argumentos)
    ot.main()
    assert (tmp_path / "p" / "confirmacion.csv").exists()
    assert "Mejor" in capsys.readouterr().out

    monkeypatch.setattr("sys.argv", [*argumentos, "--solo-confirmar", "--cortes", "fin"])
    ot.main()
    filas = pd.read_csv(tmp_path / "p" / "confirmacion.csv")
    assert set(filas["corte"]) == {"fin"}
