"""Dos cabezas: direccion de cada accion = la del mercado + la de la accion contra el resto.

La direccion en dolares de una accion se descompone en el retorno del mercado
(el promedio del panel) mas el de la accion contra ese promedio. En la
simulacion las dos partes se comportan distinto: el mercado decide la mayor
parte de la direccion (acertarlo siempre daria 0,80) y nadie lo predice; la
parte relativa tiene senal chica pero real (el ensamble Ridge + reversion).
Por eso se modelan por separado:

- **Cabeza de mercado**: cualquier modelo de panel que devuelva el mismo
  retorno para todas las acciones del dia (``src.panel.mercado``).
- **Cabeza relativa**: cualquier modelo de panel entrenado con un target sin la
  media del dia (``ensamble_panel``, ``panel_gbm``, ``panel_transformer`` con
  ``target="sin_media"``); se le resta igual su promedio del dia, para que no
  meta una opinion sobre el mercado.

Las dos predicen retornos log esperados a ``horizonte`` ruedas, asi que se
suman: E[r_i] = E[mercado] + E[r_i - mercado]. ``peso_mercado`` escala la
cabeza de mercado (0 = solo la relativa, que con el target sin media dice
"alza" la mitad de las veces; 1 = la suma de esperanzas).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.errors import ApiMlError
from src.trend_common import derive_trend_output


def _log_returns(resultados: dict[str, dict]) -> pd.Series:
    return pd.Series(
        {s: float(np.log(r["predicted_close"] / r["last_close"])) for s, r in resultados.items()},
        dtype=float,
    )


class DosCabezasModel:
    def __init__(self, mercado, relativo, peso_mercado: float = 1.0, horizonte: int = 5):
        if peso_mercado < 0:
            raise ValueError("peso_mercado no puede ser negativo")
        self.mercado = mercado
        self.relativo = relativo
        self.peso_mercado = peso_mercado
        self.horizonte = horizonte

    @property
    def version(self) -> str:
        return f"dos-cabezas-{self.peso_mercado:g}"

    def fit(self, histories: dict[str, pd.DataFrame]) -> dict:
        return {
            "mercado": self.mercado.fit(histories),
            "relativo": self.relativo.fit(histories),
        }

    def predict_panel(self, dfs: dict[str, pd.DataFrame]) -> dict[str, dict]:
        relativo = _log_returns(self.relativo.predict_panel(dfs))
        if relativo.empty:
            return {}
        relativo = relativo - relativo.mean()
        mercado = _log_returns(self.mercado.predict_panel(dfs)) if self.peso_mercado else None
        if mercado is not None and mercado.empty:
            return {}
        nivel = float(mercado.mean()) if mercado is not None else 0.0
        total = relativo + self.peso_mercado * nivel
        return {s: derive_trend_output(dfs[s], float(v), self.horizonte) for s, v in total.items()}

    def predict_df(self, df: pd.DataFrame) -> dict:
        raise ApiMlError("el modelo de dos cabezas predice con todos los tickers del dia")
