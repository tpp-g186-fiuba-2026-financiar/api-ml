"""Modelo de panel con arboles (o lineal): control de la fase 3 del plan del Transformer.

Usa las features de ``src.panel.features`` (todas las acciones del dia,
normalizadas por vol y rankeadas) para predecir el target del panel. Sirve para
responder si los datos y el preprocesamiento nuevos ya ordenan mejor que la
regla de reversion, antes de cambiar la arquitectura del Transformer.

- ``tipo="gbm"``: ``HistGradientBoostingRegressor`` chico y regularizado.
  ``max_features < 1`` sortea features por nodo, asi que la semilla importa y
  tiene sentido promediar varias (ensamble).
- ``tipo="ridge"``: regresion lineal con penalizacion L2 sobre las features
  estandarizadas (control lineal: si empata con el GBM, no hay interacciones
  que aprender).

Como predice mirando todos los tickers del dia, expone ``predict_panel(dfs)``;
``predict_df`` (un solo ticker) no tiene sentido y falla.

El CCL (para pasar a dolares) y la tabla macro no vienen en los historicos: los
da ``proveedor(hasta) -> (ccl, macro)``, con los datos conocidos a esa fecha
(en la simulacion, ``Snapshot.ccl_hasta`` y ``Snapshot.macro_frame``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from src.errors import ApiMlError, NotEnoughDataError
from src.panel.datos import armar_panel, universo
from src.panel.features import (
    FeaturesPanel,
    Normalizador,
    PreprocesoConfig,
    construir,
    contexto_macro,
    pesos_temporales,
)
from src.trend_common import derive_trend_output, senal_por_cuantil

Proveedor = Callable[[pd.Timestamp], tuple[pd.Series | None, pd.DataFrame | None]]

# Ruedas de historia que se usan para predecir (alcanza para dd_252 y la vol EWMA).
RUEDAS_PREDICCION = 320
MIN_FILAS = 2000


def features_y_contexto(
    histories: dict[str, pd.DataFrame], config, proveedor: Proveedor | None
) -> tuple[FeaturesPanel, pd.DataFrame]:
    """Features del panel y, si ``config.macro``, el contexto macro por fecha.

    ``config`` es cualquier config de modelo de panel con ``universo``,
    ``moneda``, ``macro`` y ``preproceso()``. El CCL y la macro se piden al
    ``proveedor`` con la ultima fecha de los historicos.
    """
    hasta = max(df.index[-1] for df in histories.values() if not df.empty)
    ccl = macro = None
    if config.moneda == "usd" or config.macro:
        if proveedor is None:
            raise ApiMlError("el modelo de panel necesita un proveedor de CCL / macro")
        ccl, macro = proveedor(hasta)
    panel = armar_panel(histories, universo(config.universo), ccl, moneda=config.moneda)
    fp = construir(panel, config.preproceso())
    if config.macro:
        if macro is None:
            raise ApiMlError("el proveedor no devolvio la tabla macro")
        ctx = contexto_macro(macro, fp.fechas)
    else:
        ctx = pd.DataFrame(index=fp.fechas)
    return fp, ctx


@dataclass(frozen=True)
class PanelGBMConfig:
    tipo: str = "gbm"  # "gbm" o "ridge"
    universo: str = "panel"
    moneda: str = "usd"
    normalizar_vol: bool = True
    rankear: bool = True
    target: str = "sin_media"
    macro: bool = False  # sumar el contexto macro (comun a todas las acciones del dia)
    vida_media_anios: float | None = 2.0
    horizonte: int = 5
    zona_neutral: float = 0.0  # ver PreprocesoConfig
    # GBM
    max_iter: int = 200
    learning_rate: float = 0.05
    max_depth: int = 3
    min_samples_leaf: int = 200
    l2_regularization: float = 1.0
    max_features: float = 0.5
    # Ridge
    alpha: float = 10.0
    # Senal por posicion en el dia (``senal_por_cuantil``) en vez de la banda de +-1%.
    cuantil_senal: float | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        if self.tipo not in ("gbm", "ridge"):
            raise ValueError(f"tipo {self.tipo!r} desconocido (gbm o ridge)")
        universo(self.universo)
        if self.cuantil_senal is not None and not 0 < self.cuantil_senal < 0.5:
            raise ValueError("cuantil_senal tiene que estar entre 0 y 0,5")

    def preproceso(self) -> PreprocesoConfig:
        return PreprocesoConfig(
            horizonte=self.horizonte,
            normalizar_vol=self.normalizar_vol,
            rankear=self.rankear,
            target=self.target,
            zona_neutral=self.zona_neutral,
        )


class PanelGBMModel:
    def __init__(self, config: PanelGBMConfig | None = None, proveedor: Proveedor | None = None):
        self.config = config or PanelGBMConfig()
        self.proveedor = proveedor
        self.columnas: list[str] = []
        self._modelo = None
        self._normalizador: Normalizador | None = None

    @property
    def version(self) -> str:
        return f"panel-{self.config.tipo}-{self.config.moneda}"

    # ------------------------------------------------------------------ datos
    def _features(self, histories: dict[str, pd.DataFrame]) -> tuple[FeaturesPanel, pd.DataFrame]:
        return features_y_contexto(histories, self.config, self.proveedor)

    def _filas(self, fp: FeaturesPanel, ctx: pd.DataFrame, mascara: np.ndarray) -> np.ndarray:
        """Matriz (filas = celdas de ``mascara``) con features por accion + macro del dia."""
        t, n = np.nonzero(mascara)
        x = fp.valores[t, n]
        if ctx.shape[1]:
            x = np.hstack([x, ctx.to_numpy(dtype=float)[t]])
        return x

    # ---------------------------------------------------------- entrenamiento
    def fit(self, histories: dict[str, pd.DataFrame]) -> dict:
        fp, ctx = self._features(histories)
        mascara = fp.mascara_target
        if mascara.sum() < MIN_FILAS:
            raise NotEnoughDataError(f"muy pocas filas para entrenar ({int(mascara.sum())})")
        x = self._filas(fp, ctx, mascara)
        y = fp.target[mascara]
        t, _ = np.nonzero(mascara)
        pesos = pesos_temporales(fp.fechas, self.config.vida_media_anios)[t]
        self.columnas = fp.nombres + list(ctx.columns)
        cfg = self.config
        if cfg.tipo == "gbm":
            self._modelo = HistGradientBoostingRegressor(
                max_iter=cfg.max_iter,
                learning_rate=cfg.learning_rate,
                max_depth=cfg.max_depth,
                min_samples_leaf=cfg.min_samples_leaf,
                l2_regularization=cfg.l2_regularization,
                max_features=cfg.max_features,
                early_stopping=False,
                random_state=cfg.seed,
            )
            self._modelo.fit(x, y, sample_weight=pesos)
        else:
            self._normalizador = Normalizador().fit(x)
            x = np.nan_to_num(self._normalizador.transform(x))
            self._modelo = Ridge(alpha=cfg.alpha).fit(x, y, sample_weight=pesos)
        return {"filas": int(len(y)), "fechas": int(mascara.any(axis=1).sum())}

    # -------------------------------------------------------------- prediccion
    def _predecir(self, x: np.ndarray) -> np.ndarray:
        if self._modelo is None:
            raise NotEnoughDataError("el modelo de panel no esta entrenado")
        if self._normalizador is not None:
            x = np.nan_to_num(self._normalizador.transform(x))
        return self._modelo.predict(x)

    def predict_panel(self, dfs: dict[str, pd.DataFrame]) -> dict[str, dict]:
        """Prediccion de la ultima rueda para cada ticker de ``dfs`` (todos al
        mismo dia). Los que no tienen dato valido ese dia quedan afuera."""
        if self._modelo is None:
            raise NotEnoughDataError("el modelo de panel no esta entrenado")
        recortes = {s: df.iloc[-RUEDAS_PREDICCION:] for s, df in dfs.items() if not df.empty}
        if not recortes:
            return {}
        try:
            fp, ctx = self._features(recortes)
        except ValueError:  # ningun ticker del universo
            return {}
        ultimo = len(fp.fechas) - 1
        mascara = np.zeros_like(fp.mascara)
        mascara[ultimo] = fp.mascara[ultimo]
        if not mascara.any():
            return {}
        scores = self._predecir(self._filas(fp, ctx, mascara))
        vol = fp.vol[ultimo]
        resultados = {}
        for (_, n), score in zip(np.argwhere(mascara), scores, strict=True):
            symbol = fp.tickers[n]
            # El score esta en unidades de vol a `horizonte` ruedas: se lleva a retorno log.
            log_return = float(score) * float(vol[n]) * np.sqrt(self.config.horizonte)
            resultados[symbol] = derive_trend_output(
                recortes[symbol], log_return, self.config.horizonte
            )
        if self.config.cuantil_senal is not None:
            resultados = senal_por_cuantil(resultados, self.config.cuantil_senal)
        return resultados

    def predict_df(self, df: pd.DataFrame) -> dict:
        raise ApiMlError("el modelo de panel predice con todos los tickers del dia (predict_panel)")

    def resumen(self) -> dict:
        return {"version": self.version, "config": asdict(self.config), "columnas": self.columnas}
