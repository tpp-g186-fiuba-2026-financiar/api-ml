"""Regimen y backtest de GarchVolatilityModel con el ajuste real de `arch`."""

import numpy as np
import pandas as pd
import pytest

from src.errors import NotEnoughDataError
from src.garch_volatility import BACKTEST_WINDOWS, GarchVolatilityModel


def _serie(n: int, seed: int = 0, choque: int | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sigma = np.full(n, 0.015)
    if choque is not None:
        sigma[choque:] = 0.05  # tramo final mucho mas agitado
    close = 100 * np.exp(np.cumsum(rng.normal(0, sigma)))
    return pd.DataFrame({"close": close}, index=pd.bdate_range("2022-01-03", periods=n))


def test_backtest_mide_ventanas_sin_solapar_y_cobertura():
    out = GarchVolatilityModel().contexto(_serie(700), horizon=10)
    bt = out["backtest"]
    assert len(bt["windows"]) == BACKTEST_WINDOWS
    fechas = [w["as_of"] for w in bt["windows"]]
    assert fechas == sorted(fechas)
    # con retornos normales, GARCH queda cerca de la cobertura teorica
    assert 45 <= bt["inside_68_pct"] <= 95
    assert bt["inside_95_pct"] >= bt["inside_68_pct"]


def test_regimen_marca_alta_cuando_el_final_esta_agitado():
    tranquila = GarchVolatilityModel().contexto(_serie(700), horizon=5)["regime"]
    agitada = GarchVolatilityModel().contexto(_serie(700, choque=650), horizon=5)["regime"]
    assert agitada["percentile"] > tranquila["percentile"]
    assert agitada["level"] == "alta"
    assert 0 < agitada["persistence"] < 1.0001


def test_contexto_pide_historia_minima():
    with pytest.raises(NotEnoughDataError):
        GarchVolatilityModel().contexto(_serie(70), horizon=20)
