import pandas as pd
import pytest

import src.simulacion.entrenamiento as entrenamiento
from src.data import MACRO_COLUMN
from src.errors import DataUnavailableError
from src.simulacion.entrenamiento import (
    ModeloDiario,
    ModeloMacro,
    build_modelos_diarios,
    histories_at,
    macro_retrain_due,
    reentreno_due,
)
from tests.simulacion.conftest import TICKER_CORTO, TICKERS, FakeMacro, FakeTrend


def test_histories_at_deja_afuera_las_series_cortas(snapshot) -> None:
    corte = snapshot.trading_days()[-1]
    histories = histories_at(snapshot, corte, days=200)
    assert set(histories) == set(TICKERS)
    assert TICKER_CORTO not in histories
    assert all(len(df) == 200 and df.index[-1] == corte for df in histories.values())
    assert all(MACRO_COLUMN in df for df in histories.values())


def test_modelo_diario_bootstrap_retrain_resume(snapshot, tmp_path) -> None:
    histories = histories_at(snapshot, snapshot.trading_days()[-1], days=200)
    modelo = ModeloDiario("lstm", FakeTrend, FakeTrend.load, tmp_path / "lstm.pkl")

    fila = modelo.bootstrap(histories)
    assert fila["promoted"] and fila["bootstrap"]
    assert modelo.vigente.entrenado_con == len(TICKERS)
    assert (tmp_path / "lstm.pkl.score.json").exists()

    # Mismo modelo: no supera al vigente, no se promueve.
    vigente = modelo.vigente
    fila = modelo.retrain(histories)
    assert fila["promoted"] is False
    assert modelo.vigente is vigente

    otro = ModeloDiario("lstm", FakeTrend, FakeTrend.load, tmp_path / "lstm.pkl")
    otro.resume()
    assert isinstance(otro.vigente, FakeTrend)
    ModeloDiario("x", FakeTrend, FakeTrend.load, tmp_path / "no.pkl").resume()


def test_build_modelos_diarios(tmp_path) -> None:
    modelos = build_modelos_diarios(tmp_path / "artefactos")
    assert [m.name for m in modelos] == ["lstm", "xgboost", "transformer"]
    assert (tmp_path / "artefactos").is_dir()


class _MacroOk(FakeMacro):
    def __init__(self, config=None):
        super().__init__()

    def fit(self, histories, macro):
        self.entrenado_con = len(histories)
        return {"oos_auc": 0.6}

    @classmethod
    def load(cls, path):
        return FakeMacro.load(path)


class _MacroMalo(_MacroOk):
    def fit(self, histories, macro):
        return {"oos_auc": 0.4}


class _MacroFalla(_MacroOk):
    def fit(self, histories, macro):
        raise DataUnavailableError("sin macro")


def test_modelo_macro(monkeypatch, snapshot, tmp_path) -> None:
    corte = snapshot.trading_days()[-1]
    path = tmp_path / "macro.pkl"

    monkeypatch.setattr(entrenamiento, "MacroTrendModel", _MacroMalo)
    macro = ModeloMacro(snapshot, path)
    assert macro.retrain(corte)["promoted"] is False
    assert macro.vigente is None
    assert macro.retrain(corte, force=True)["promoted"] is True

    monkeypatch.setattr(entrenamiento, "MacroTrendModel", _MacroOk)
    macro = ModeloMacro(snapshot, path)
    fila = macro.retrain(corte)
    assert fila["promoted"] and fila["oos_auc"] == 0.6
    assert macro.vigente.macro_provider().index[-1] == corte

    monkeypatch.setattr(entrenamiento, "MacroTrendModel", _MacroFalla)
    assert "sin macro" in ModeloMacro(snapshot, path).retrain(corte)["error"]

    monkeypatch.setattr(entrenamiento, "MacroTrendModel", _MacroOk)
    retomado = ModeloMacro(snapshot, path)
    retomado.resume()
    assert retomado.vigente is not None
    ModeloMacro(snapshot, tmp_path / "no.pkl").resume()

    sin_macro = ModeloMacro(
        type(snapshot)(snapshot.prices, snapshot.tickers, None, {}, {}), tmp_path / "m.pkl"
    )
    assert "error" in sin_macro.retrain(corte)


@pytest.mark.parametrize(
    ("dia", "siguiente", "esperado"),
    [
        ("2026-06-01", "2026-06-02", True),  # el 1 es rueda
        ("2026-05-29", "2026-06-02", True),  # el 1 cae entre ruedas
        ("2026-06-02", "2026-06-03", False),
        ("2026-06-02", None, False),
    ],
)
def test_macro_retrain_due(dia, siguiente, esperado) -> None:
    sig = pd.Timestamp(siguiente) if siguiente else None
    assert macro_retrain_due(pd.Timestamp(dia), sig) is esperado


def test_reentreno_due() -> None:
    jueves, viernes, lunes = (pd.Timestamp(d) for d in ("2026-06-04", "2026-06-05", "2026-06-08"))
    assert reentreno_due("diario", jueves, viernes)
    assert not reentreno_due("semanal", jueves, viernes)
    assert reentreno_due("semanal", viernes, lunes)
    # Jueves con feriado el viernes: es la ultima rueda de la semana.
    assert reentreno_due("semanal", jueves, lunes)
    assert reentreno_due("semanal", jueves, None)
    with pytest.raises(ValueError):
        reentreno_due("mensual", jueves, viernes)
