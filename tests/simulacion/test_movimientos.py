import pytest

import src.simulacion.movimientos as movimientos
from src.simulacion.movimientos import describir_cartera


@pytest.mark.parametrize(
    ("cartera", "esperado"),
    [
        ("senal_lstm", ("senal", "lstm", None)),
        ("bl_xgboost_moderate", ("bl", "xgboost", "moderate")),
        ("senal_mejor_prod", ("senal", "mejor_prod", None)),
        ("bl_mejor_todos_aggressive", ("bl", "mejor_todos", "aggressive")),
        ("consenso_conservative", ("consenso", None, "conservative")),
        ("benchmark_equal_weight_rebalanceo", ("benchmark", None, None)),
        ("histeresis_panel_transformer", ("histeresis", "panel_transformer", None)),
        ("otra", ("otra", None, None)),
    ],
)
def test_describir_cartera(cartera, esperado) -> None:
    d = describir_cartera(cartera)
    assert (d["estrategia"], d["modelo"], d["perfil"]) == esperado


def test_corrida_sin_operaciones(tmp_path, snapshot, monkeypatch, capsys) -> None:
    corrida = tmp_path / "vacia"
    corrida.mkdir()
    (corrida / "config.json").write_text('{"snapshot_dir": "x"}')
    assert movimientos.movimientos(corrida, snapshot).empty
    assert movimientos.escribir(corrida, snapshot) is None
    monkeypatch.setattr(movimientos, "RESULTADOS_DIR", str(tmp_path))
    monkeypatch.setattr(movimientos.Snapshot, "load", lambda path: snapshot)
    movimientos.main(["--nombre", "vacia"])
    assert "no tiene operaciones" in capsys.readouterr().out
