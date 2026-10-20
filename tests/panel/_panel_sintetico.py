"""Historicos y proveedor de CCL / macro sinteticos para los tests de modelos de panel."""

import numpy as np
import pandas as pd

RUEDAS = pd.bdate_range("2023-01-02", periods=600)
TICKERS = ["GGAL", "BMA", "YPFD", "PAMP", "ALUA", "TXAR", "CEPU", "EDN"]


def ohlcv(seed: int, index: pd.DatetimeIndex = RUEDAS, mercado: np.ndarray | None = None):
    rng = np.random.default_rng(seed)
    r = rng.normal(0.001, 0.02, len(index))
    if mercado is not None:
        r = r + mercado
    close = 100 * np.exp(np.cumsum(r))
    return pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99, "close": close,
         "volume": rng.integers(500, 2000, len(index)).astype(float)},
        index=index,
    )  # fmt: skip


def histories(mercado: np.ndarray | None = None) -> dict[str, pd.DataFrame]:
    return {t: ohlcv(i, mercado=mercado) for i, t in enumerate(TICKERS)}


def macro() -> pd.DataFrame:
    dias = pd.date_range(RUEDAS[0] - pd.Timedelta(days=400), RUEDAS[-1], freq="D")
    return pd.DataFrame(
        {"ccl": np.linspace(300, 1500, len(dias)), "oficial": np.linspace(200, 1400, len(dias)),
         "riesgo_pais": np.linspace(2000, 600, len(dias)), "badlar": 40.0},
        index=dias,
    )  # fmt: skip


def proveedor(hasta):
    tabla = macro()
    tabla = tabla[tabla.index <= hasta]
    return tabla["ccl"], tabla


def mercado_que_revierte(seed: int = 7, fuerza: float = 0.8) -> np.ndarray:
    """Retorno diario comun a todas las acciones, por bloques de 5 ruedas: cada
    bloque revierte ``fuerza`` veces lo que hizo el anterior (mas ruido). Una
    senal de mercado que la cabeza de mercado puede aprender, mirando desde el
    inicio de cada bloque."""
    rng = np.random.default_rng(seed)
    n_bloques = -(-len(RUEDAS) // 5)
    bloques = np.zeros(n_bloques)
    for k in range(n_bloques):
        bloques[k] = rng.normal(0, 0.04) - fuerza * (bloques[k - 1] if k else 0.0)
    diario = np.repeat(bloques / 5, 5)[: len(RUEDAS)]
    return diario + rng.normal(0, 0.002, len(RUEDAS))
