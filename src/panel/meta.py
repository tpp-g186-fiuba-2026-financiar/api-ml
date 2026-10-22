"""Meta-labeling sobre la reversion: la regla elige el lado, el modelo decide si confiar.

La regla de reversion en dolares (lo contrario del retorno de las ultimas
``horizonte`` ruedas) fue la mejor senal simple de direccion de la simulacion,
pero acierta poco mas que la mitad de las veces. En vez de que un modelo
prediga la direccion de cero, aca se entrena un clasificador que predice **si
la reversion va a acertar** en cada caso (meta-labeling, Lopez de Prado):

- **Lado**: el de la reversion (signo contrario a ``ret_5_nivel``, el retorno
  en dolares de las ultimas 5 ruedas en unidades de vol, sin rankear). El
  modelo nunca lo da vuelta.
- **Label**: 1 si el retorno real a ``horizonte`` ruedas en dolares tuvo el
  signo del lado. Con ``zona_neutral`` > 0 no se entrena con los movimientos
  chicos.
- **Features**: las del panel (``src.panel.features``) mas la macro opcional.
- **Clasificador**: ``"logit"`` (regresion logistica sobre las features
  estandarizadas) o ``"gbm"`` (``HistGradientBoostingClassifier`` chico).

La prediccion es ``lado * p * ESCALA``, con ``p`` la probabilidad de que la
reversion acierte: el signo es siempre el de la regla (``bal_usd`` igual al de
la reversion) y la magnitud ordena por confianza, que es lo que mide
``bal_top``. Con ``ESCALA = 0,02`` y la banda neutral de +-1% de la web, hay
senal solo cuando ``p > 0,5``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from src.errors import ApiMlError, NotEnoughDataError
from src.panel.datos import universo
from src.panel.features import NOMBRES, Normalizador, PreprocesoConfig, pesos_temporales
from src.panel.gbm import MIN_FILAS, RUEDAS_PREDICCION, Proveedor, features_y_contexto
from src.trend_common import derive_trend_output

TIPOS = ("logit", "gbm")
ESCALA = 0.02
_RET_5 = NOMBRES.index("ret_5_nivel")


@dataclass(frozen=True)
class MetaReversionConfig:
    tipo: str = "logit"
    universo: str = "panel"
    moneda: str = "usd"
    macro: bool = False
    vida_media_anios: float | None = 2.0
    horizonte: int = 5
    zona_neutral: float = 0.0
    c: float = 0.1  # logit
    max_iter: int = 200  # gbm
    learning_rate: float = 0.05
    max_depth: int = 3
    min_samples_leaf: int = 200
    seed: int = 0

    def __post_init__(self) -> None:
        if self.tipo not in TIPOS:
            raise ValueError(f"tipo {self.tipo!r} desconocido ({TIPOS})")
        universo(self.universo)

    def preproceso(self) -> PreprocesoConfig:
        # Target absoluto: el label mira el signo del retorno real, no el relativo.
        return PreprocesoConfig(
            horizonte=self.horizonte, target="absoluto", zona_neutral=self.zona_neutral
        )


class MetaReversionModel:
    def __init__(
        self, config: MetaReversionConfig | None = None, proveedor: Proveedor | None = None
    ):
        self.config = config or MetaReversionConfig()
        self.proveedor = proveedor
        self._modelo = None
        self._normalizador: Normalizador | None = None

    @property
    def version(self) -> str:
        return f"panel-meta-reversion-{self.config.tipo}"

    @staticmethod
    def _filas(fp, ctx: pd.DataFrame, mascara: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Matriz de features (panel + macro) y lado de la reversion de las celdas."""
        t, n = np.nonzero(mascara)
        x = fp.valores[t, n]
        if ctx.shape[1]:
            x = np.hstack([x, ctx.to_numpy(dtype=float)[t]])
        return x, -np.sign(fp.valores[t, n, _RET_5])

    def fit(self, histories: dict[str, pd.DataFrame]) -> dict:
        cfg = self.config
        fp, ctx = features_y_contexto(histories, cfg, self.proveedor)
        mascara = fp.mascara_target & (fp.valores[..., _RET_5] != 0)
        if mascara.sum() < MIN_FILAS:
            raise NotEnoughDataError(f"muy pocas filas para entrenar ({int(mascara.sum())})")
        x, lado = self._filas(fp, ctx, mascara)
        acierta = (lado * fp.retorno[mascara] > 0).astype(int)
        t, _ = np.nonzero(mascara)
        pesos = pesos_temporales(fp.fechas, cfg.vida_media_anios)[t]
        if cfg.tipo == "logit":
            self._normalizador = Normalizador().fit(x)
            x = np.nan_to_num(self._normalizador.transform(x))
            self._modelo = LogisticRegression(C=cfg.c, max_iter=1000)
        else:
            self._modelo = HistGradientBoostingClassifier(
                max_iter=cfg.max_iter,
                learning_rate=cfg.learning_rate,
                max_depth=cfg.max_depth,
                min_samples_leaf=cfg.min_samples_leaf,
                early_stopping=False,
                random_state=cfg.seed,
            )
        self._modelo.fit(x, acierta, sample_weight=pesos)
        return {"filas": int(len(acierta)), "tasa_acierto": float(acierta.mean())}

    def predict_panel(self, dfs: dict[str, pd.DataFrame]) -> dict[str, dict]:
        if self._modelo is None:
            raise NotEnoughDataError("el meta-modelo no esta entrenado")
        recortes = {s: df.iloc[-RUEDAS_PREDICCION:] for s, df in dfs.items() if not df.empty}
        if not recortes:
            return {}
        try:
            fp, ctx = features_y_contexto(recortes, self.config, self.proveedor)
        except ValueError:  # ningun ticker del universo
            return {}
        ultimo = len(fp.fechas) - 1
        mascara = np.zeros_like(fp.mascara)
        mascara[ultimo] = fp.mascara[ultimo] & (fp.valores[ultimo, :, _RET_5] != 0)
        if not mascara.any():
            return {}
        x, lado = self._filas(fp, ctx, mascara)
        if self._normalizador is not None:
            x = np.nan_to_num(self._normalizador.transform(x))
        p = self._modelo.predict_proba(x)[:, 1]
        resultados = {}
        for (_, n), lado_n, p_n in zip(np.argwhere(mascara), lado, p, strict=True):
            symbol = fp.tickers[n]
            resultados[symbol] = derive_trend_output(
                recortes[symbol], float(lado_n * p_n * ESCALA), self.config.horizonte
            )
        return resultados

    def predict_df(self, df: pd.DataFrame) -> dict:
        raise ApiMlError("el meta-modelo predice con todos los tickers del dia")

    def resumen(self) -> dict:
        return {"version": self.version, "config": asdict(self.config)}
