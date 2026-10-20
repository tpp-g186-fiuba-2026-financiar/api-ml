"""Gate de promocion por IC."""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.errors import ApiMlError
from src.retrain_guard import force_promote, pooled_ic, promote_if_better
from src.trend_common import derive_trend_output


class _Oraculo:
    """Predice el retorno real futuro (IC 1) o su opuesto (IC -1)."""

    def __init__(self, futuro: dict[str, pd.Series], signo: float = 1.0, tickers=None):
        self.futuro = futuro
        self.signo = signo
        self.tickers = tickers or []

    def predict_df(self, df: pd.DataFrame) -> dict:
        if df.index[-1].day == 1:
            raise ApiMlError("sin dato")
        symbol = df.attrs["symbol"]
        lr = self.signo * float(self.futuro[symbol].loc[df.index[-1]])
        return derive_trend_output(df, lr, 5)

    def save(self, path) -> None:
        Path(path).write_bytes(pickle.dumps(self))

    @staticmethod
    def load(path):
        return pickle.loads(Path(path).read_bytes())


def _histories(n: int = 90) -> dict[str, pd.DataFrame]:
    index = pd.bdate_range("2026-01-01", periods=n)
    out = {}
    for i, symbol in enumerate(["A", "B", "C", "D"]):
        rng = np.random.default_rng(i)
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
        df = pd.DataFrame({"close": close, "high": close, "low": close}, index=index)
        df.attrs["symbol"] = symbol
        out[symbol] = df
    return out


def _futuro(histories) -> dict[str, pd.Series]:
    return {
        s: (np.log(df["close"]).shift(-5) - np.log(df["close"])).fillna(0.0)
        for s, df in histories.items()
    }


def test_pooled_ic(tmp_path) -> None:
    histories = _histories()
    futuro = _futuro(histories)
    bueno = pooled_ic(_Oraculo(futuro), histories, 5)
    assert bueno["ic"] == pytest.approx(1.0)
    assert bueno["directional_accuracy"] == pytest.approx(1.0)
    assert pooled_ic(_Oraculo(futuro, -1.0), histories, 5)["ic"] == pytest.approx(-1.0)
    # Solo evalua el universo del modelo.
    solo_a = pooled_ic(_Oraculo(futuro, tickers=["A", "ZZZ"]), histories, 5)
    assert solo_a["observations"] < bueno["observations"]
    cortas = {s: df.iloc[:3] for s, df in histories.items()}
    assert pooled_ic(_Oraculo(futuro), cortas, 5) == {
        "ic": None,
        "directional_accuracy": None,
        "observations": 0,
    }


def test_promote_if_better_por_ic_reevalua_al_vigente(tmp_path) -> None:
    histories = _histories()
    futuro = _futuro(histories)
    path = str(tmp_path / "m.pkl")

    malo = _Oraculo(futuro, -1.0)
    score = force_promote(malo, path, histories, 5, metrica="ic")
    assert score["ic"] == pytest.approx(-1.0)
    assert json.loads(Path(path + ".score.json").read_text())["ic"] == pytest.approx(-1.0)

    promovido, score = promote_if_better(
        _Oraculo(futuro), path, _Oraculo.load, histories, 5, metrica="ic"
    )
    assert promovido and score["incumbent_ic"] == pytest.approx(-1.0)

    # El vigente ahora es el bueno: uno igual no lo supera.
    promovido, score = promote_if_better(
        _Oraculo(futuro), path, _Oraculo.load, histories, 5, metrica="ic"
    )
    assert not promovido
    assert score["incumbent_directional_accuracy"] == pytest.approx(1.0)


def test_promote_if_better_por_ic_sin_vigente(tmp_path) -> None:
    histories = _histories()
    path = str(tmp_path / "nuevo.pkl")
    promovido, score = promote_if_better(
        _Oraculo(_futuro(histories)), path, _Oraculo.load, histories, 5, metrica="ic"
    )
    assert promovido and score["incumbent_ic"] is None
    assert Path(path).exists()
