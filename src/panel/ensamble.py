"""Ensamble de panel: promedio de rankings del Ridge de panel y la regla de reversion.

En la fase 3 del plan del Transformer, el Ridge de panel (``src.panel.gbm``) y
la regla de reversion a 5 ruedas ordenaron parecido en promedio (IC ~0,05)
pero con rankings poco correlacionados (0,26-0,47) y buenos en periodos
distintos. Promediar sus rankings dio IC 0,063, positivo en los 7 periodos.

Cada dia, cada componente ordena los tickers; el score es el promedio
ponderado de los rankings centrados (``peso_reversion`` para la reversion, el
resto para el Ridge). Para devolver un retorno, el score se lleva a la escala
de las predicciones del Ridge ese dia (mismo desvio): el orden es lo que
importa, la escala solo sirve para las senales.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from src.errors import ApiMlError
from src.panel.gbm import PanelGBMConfig, PanelGBMModel, Proveedor
from src.trend_common import derive_trend_output, senal_por_cuantil


@dataclass(frozen=True)
class EnsambleConfig:
    peso_reversion: float = 0.5
    ridge: PanelGBMConfig = field(default_factory=lambda: PanelGBMConfig(tipo="ridge"))
    # Senal por posicion en el dia (``senal_por_cuantil``): "alza" el cuantil
    # superior del score, "baja" el inferior. None = banda fija de +-1%, que con
    # la escala del Ridge casi nunca se pasa (< 5% de senales).
    cuantil_senal: float | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.peso_reversion <= 1.0:
            raise ValueError("peso_reversion tiene que estar entre 0 y 1")
        if self.cuantil_senal is not None and not 0 < self.cuantil_senal < 0.5:
            raise ValueError("cuantil_senal tiene que estar entre 0 y 0,5")


def reversion(df: pd.DataFrame, horizonte: int) -> float | None:
    """Retorno de las ultimas ``horizonte`` ruedas, con el signo cambiado."""
    close = df["close"].to_numpy(dtype=np.float64)
    if len(close) <= horizonte or close[-1] <= 0 or close[-1 - horizonte] <= 0:
        return None
    return -float(np.log(close[-1] / close[-1 - horizonte]))


def _rank_centrado(valores: pd.Series) -> pd.Series:
    return valores.rank(pct=True) - 0.5


class EnsamblePanelModel:
    def __init__(self, config: EnsambleConfig | None = None, proveedor: Proveedor | None = None):
        self.config = config or EnsambleConfig()
        self.ridge = PanelGBMModel(self.config.ridge, proveedor=proveedor)

    @property
    def version(self) -> str:
        cuantil = self.config.cuantil_senal
        return f"panel-ensamble-{self.config.peso_reversion:g}" + (
            f"-q{cuantil:g}" if cuantil else ""
        )

    def fit(self, histories: dict[str, pd.DataFrame]) -> dict:
        return self.ridge.fit(histories)

    def predict_panel(self, dfs: dict[str, pd.DataFrame]) -> dict[str, dict]:
        horizonte = self.config.ridge.horizonte
        del_ridge = self.ridge.predict_panel(dfs)
        log_ridge = pd.Series(
            {s: float(np.log(r["predicted_close"] / r["last_close"])) for s, r in del_ridge.items()}
        )
        rev = pd.Series({s: reversion(dfs[s], horizonte) for s in log_ridge.index}, dtype=float)
        validos = rev.dropna().index
        if len(validos) == 0:
            return {}
        log_ridge, rev = log_ridge[validos], rev[validos]
        peso = self.config.peso_reversion
        score = peso * _rank_centrado(rev) + (1 - peso) * _rank_centrado(log_ridge)
        desvio_score = score.std()
        escala = log_ridge.std() / desvio_score if desvio_score > 0 else 0.0
        escala = escala if np.isfinite(escala) else 0.0
        resultados = {
            s: derive_trend_output(dfs[s], float(score[s] * escala), horizonte) for s in validos
        }
        if self.config.cuantil_senal is not None:
            resultados = senal_por_cuantil(resultados, self.config.cuantil_senal)
        return resultados

    def predict_df(self, df: pd.DataFrame) -> dict:
        raise ApiMlError(
            "el ensamble de panel predice con todos los tickers del dia (predict_panel)"
        )

    def resumen(self) -> dict:
        return {"version": self.version, "config": asdict(self.config)}
