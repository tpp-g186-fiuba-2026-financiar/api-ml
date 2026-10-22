"""Snapshot sintetico y modelos falsos para testear la simulacion sin red ni entrenamiento."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.simulacion.snapshot import Snapshot
from src.trend_common import derive_trend_output

# Siete del panel lider: con el techo de 20% por ticker de Black-Litterman,
# menos de cinco no alcanzarian a sumar 100%.
TICKERS = ["ALUA", "BBAR", "BMA", "BYMA", "CEPU", "COME", "GGAL"]
# Fuera del panel lider y con poca historia: queda afuera del entrenamiento.
TICKER_CORTO = "NUEVA"
RUEDAS = pd.bdate_range("2025-06-02", "2026-06-30")


def _ohlcv(seed: int, index: pd.DatetimeIndex) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, len(index))))
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 1000.0,
        },
        index=index,
    )


def escribir_snapshot(root: Path) -> Path:
    (root / "precios").mkdir(parents=True)
    (root / "macro").mkdir()
    for i, ticker in enumerate(TICKERS):
        _ohlcv(i, RUEDAS).to_csv(root / "precios" / f"{ticker}.csv", index_label="date")
    _ohlcv(99, RUEDAS[-30:]).to_csv(root / "precios" / f"{TICKER_CORTO}.csv", index_label="date")
    _ohlcv(50, RUEDAS).to_csv(root / "merval.csv", index_label="date")
    tasa = pd.Series(4.0 + np.linspace(0, 0.5, len(RUEDAS)), index=RUEDAS, name="value")
    tasa.to_csv(root / "tnx.csv", index_label="date")
    dias = pd.date_range(RUEDAS[0], RUEDAS[-1], freq="D")
    ccl = pd.Series(np.linspace(1000, 1500, len(dias)), index=dias, name="value")
    ccl.to_csv(root / "macro" / "ccl.csv", index_label="date")
    meta = {
        "downloaded_at": "2026-07-01T12:00:00Z",
        "tickers": [*TICKERS, TICKER_CORTO],
        "macro_series": ["ccl"],
    }
    (root / "meta.json").write_text(json.dumps(meta))
    return root


@pytest.fixture(scope="session")
def snapshot_dir(tmp_path_factory) -> Path:
    return escribir_snapshot(tmp_path_factory.mktemp("snapshot"))


@pytest.fixture(scope="session")
def snapshot(snapshot_dir) -> Snapshot:
    return Snapshot.load(snapshot_dir)


class FakeTrend:
    """Modelo de tendencia trivial (momentum de 5 ruedas) con la interfaz de los reales."""

    version = "fake-v1"

    def __init__(self, sesgo: float = 0.0, horizon: int = 5):
        self.sesgo = sesgo
        self.horizon = horizon
        self.entrenado_con = 0

    def fit(self, histories: dict) -> None:
        self.entrenado_con = len(histories)

    def predict_df(self, df: pd.DataFrame) -> dict:
        close = df["close"]
        retorno = float(np.log(close.iloc[-1] / close.iloc[-6])) * 0.5 + self.sesgo
        return derive_trend_output(df, retorno, self.horizon)

    def save(self, path) -> None:
        Path(path).write_bytes(pickle.dumps(self))

    @classmethod
    def load(cls, path) -> FakeTrend:
        return pickle.loads(Path(path).read_bytes())


class FakeMacro(FakeTrend):
    """Como el modelo macro: horizonte 20 y "neutral" (retorno 0) en las ruedas pares."""

    version = "fake-macro"

    def __init__(self):
        super().__init__(horizon=20)

    def predict_df(self, df: pd.DataFrame) -> dict:
        retorno = 0.0 if len(df) % 2 == 0 else 0.05
        return derive_trend_output(df, retorno, self.horizon, neutral_band=0.0)
