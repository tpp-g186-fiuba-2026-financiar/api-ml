"""Cabeza de mercado: hacia donde va el panel entero en dolares en las proximas ruedas.

La direccion en dolares de una accion es la del mercado mas la de la accion
contra el resto. En la simulacion, acertar siempre el mercado daria un acierto
balanceado de 0,80 y ningun modelo de tendencia lo predice (``bal_mercado`` de
``src.simulacion.significancia`` ~0,50). Este modelo ataca solo esa parte, con
una fila por dia (no por accion):

- **Mercado**: promedio de los retornos log diarios en dolares de las acciones
  validas del dia (el mismo que usa ``src.panel_features``, no el Merval).
- **Features del dia** (todas miran hacia atras): retornos del mercado a 1, 5,
  20 y 60 ruedas en unidades de su vol EWMA de 20 ruedas, vol de 20 ruedas
  (nivel y relativa a la de 60), drawdown de 60 ruedas, dispersion entre
  acciones y amplitud (fraccion de acciones que subieron en 5 ruedas), retorno
  del CCL a 5 y 20 ruedas y, con ``macro=True``, el contexto macro
  (``contexto_macro``, con retrasos de publicacion).
- **Target**: retorno del mercado a ``horizonte`` ruedas en unidades de vol
  (recortado a +-3). Con ``zona_neutral`` > 0 no se entrena con los dias en que
  el mercado se movio menos que eso.
- **Tipos**: ``"regla"`` (sin entrenamiento: el mercado revierte su retorno de
  las ultimas ``horizonte`` ruedas; la referencia ``mercado_revierte``),
  ``"ridge"`` (regresion sobre el target) o ``"logit"`` (clasificacion sube /
  baja; el retorno esperado es ``(2p - 1)`` por el desvio medio del target).

``predict_panel`` devuelve **el mismo retorno para todas las acciones** del
dia: sirve sola (para medir ``bal_mercado``) o como cabeza de mercado de
``src.panel_dos_cabezas``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge

from src.errors import ApiMlError, NotEnoughDataError
from src.panel_datos import armar_panel, universo
from src.panel_features import Normalizador, contexto_macro, pesos_temporales, vol_ewma
from src.panel_gbm import RUEDAS_PREDICCION, Proveedor
from src.trend_common import derive_trend_output

TIPOS = ("regla", "ridge", "logit")
FEATURES_MERCADO = [
    "m_ret_1", "m_ret_5", "m_ret_20", "m_ret_60", "m_vol_20", "m_vol_rel",
    "m_dd_60", "dispersion", "amplitud", "ccl_ret_5", "ccl_ret_20",
]  # fmt: skip
MIN_DIAS = 150
CLIP_TARGET = 3.0
# E|Z| de una normal estandar: desvio medio del target si fuera N(0, 1).
_DESVIO_MEDIO_NORMAL = float(np.sqrt(2 / np.pi))


@dataclass(frozen=True)
class MercadoConfig:
    tipo: str = "ridge"
    universo: str = "panel"
    macro: bool = False
    vida_media_anios: float | None = 2.0
    horizonte: int = 5
    zona_neutral: float = 0.0
    alpha: float = 10.0  # ridge
    c: float = 0.1  # logit (inversa de la regularizacion)
    seed: int = 0

    def __post_init__(self) -> None:
        if self.tipo not in TIPOS:
            raise ValueError(f"tipo {self.tipo!r} desconocido ({TIPOS})")
        universo(self.universo)
        if self.zona_neutral < 0:
            raise ValueError("zona_neutral no puede ser negativa")
        if self.tipo == "regla" and f"m_ret_{self.horizonte}" not in FEATURES_MERCADO:
            raise ValueError("la regla necesita un horizonte de 1, 5, 20 o 60 ruedas")


@dataclass
class DiasMercado:
    """Features ``[fecha, feature]``, target y vol del mercado por fecha."""

    features: pd.DataFrame
    target: pd.Series
    vol: pd.Series
    validas: pd.Series  # fechas con al menos 5 acciones validas


def dias_mercado(
    histories: dict[str, pd.DataFrame],
    ccl: pd.Series,
    nombre_universo: str = "panel",
    horizonte: int = 5,
) -> DiasMercado:
    """Features y target del mercado por fecha (ver docstring del modulo)."""
    panel = armar_panel(histories, universo(nombre_universo), ccl, moneda="usd")
    close = panel["close"].where(panel.mascara)
    r = np.log(close).diff()
    mercado = r.mean(axis=1)
    validas = r.notna().sum(axis=1) >= 5
    mercado = mercado.where(validas)
    indice = mercado.fillna(0.0).cumsum()
    vol = vol_ewma(mercado.to_frame(), 20).iloc[:, 0]
    f = pd.DataFrame(index=panel.fechas)
    for k in (1, 5, 20, 60):
        f[f"m_ret_{k}"] = (indice - indice.shift(k)) / (vol * np.sqrt(k))
    vol_20 = mercado.rolling(20, min_periods=15).std()
    f["m_vol_20"] = np.log(vol_20)
    f["m_vol_rel"] = np.log(vol_20 / mercado.rolling(60, min_periods=40).std())
    f["m_dd_60"] = indice - indice.rolling(60, min_periods=20).max()
    ret_5 = np.log(close).diff(5)
    f["dispersion"] = ret_5.std(axis=1) / (vol * np.sqrt(5))
    f["amplitud"] = (ret_5 > 0).sum(axis=1) / ret_5.notna().sum(axis=1).replace(0, np.nan)
    log_ccl = np.log(ccl.reindex(ccl.index.union(panel.fechas)).ffill().reindex(panel.fechas))
    f["ccl_ret_5"] = log_ccl.diff(5)
    f["ccl_ret_20"] = log_ccl.diff(20)
    f = f[FEATURES_MERCADO].replace([np.inf, -np.inf], np.nan)
    futuro = indice.shift(-horizonte) - indice
    # Hace falta que el mercado tenga dato en todas las ruedas del horizonte.
    v = validas.astype(float)[::-1]
    completo = v.rolling(horizonte, min_periods=horizonte).min()[::-1].shift(-1)
    futuro = futuro.where(completo.fillna(0.0) > 0)
    target = (futuro / (vol * np.sqrt(horizonte))).clip(-CLIP_TARGET, CLIP_TARGET)
    return DiasMercado(features=f, target=target, vol=vol, validas=validas)


class MercadoModel:
    def __init__(self, config: MercadoConfig | None = None, proveedor: Proveedor | None = None):
        self.config = config or MercadoConfig()
        self.proveedor = proveedor
        self.columnas: list[str] = []
        self._modelo = None
        self._normalizador: Normalizador | None = None
        self._escala_logit = _DESVIO_MEDIO_NORMAL

    @property
    def version(self) -> str:
        return f"panel-mercado-{self.config.tipo}"

    def _datos(self, histories: dict[str, pd.DataFrame]) -> tuple[DiasMercado, pd.DataFrame]:
        """Dias del mercado y matriz de features (con la macro si ``macro``)."""
        if self.proveedor is None:
            raise ApiMlError("la cabeza de mercado necesita un proveedor de CCL / macro")
        hasta = max(df.index[-1] for df in histories.values() if not df.empty)
        ccl, macro = self.proveedor(hasta)
        if ccl is None:
            raise ApiMlError("el proveedor no devolvio el CCL")
        dias = dias_mercado(histories, ccl, self.config.universo, self.config.horizonte)
        x = dias.features
        if self.config.macro:
            if macro is None:
                raise ApiMlError("el proveedor no devolvio la tabla macro")
            x = x.join(contexto_macro(macro, x.index))
        return dias, x

    # ---------------------------------------------------------- entrenamiento
    def fit(self, histories: dict[str, pd.DataFrame]) -> dict:
        cfg = self.config
        if cfg.tipo == "regla":
            return {}
        dias, x = self._datos(histories)
        filas = dias.validas & dias.target.notna() & x["m_ret_60"].notna()
        if cfg.zona_neutral > 0:
            filas &= dias.target.abs() >= cfg.zona_neutral
        if filas.sum() < MIN_DIAS:
            raise NotEnoughDataError(f"muy pocos dias para la cabeza de mercado ({filas.sum()})")
        self.columnas = list(x.columns)
        xs = x[filas].to_numpy(dtype=float)
        self._normalizador = Normalizador().fit(xs)
        xs = np.nan_to_num(self._normalizador.transform(xs))
        y = dias.target[filas].to_numpy()
        pesos = pesos_temporales(x.index, cfg.vida_media_anios)[filas.to_numpy()]
        if cfg.tipo == "ridge":
            self._modelo = Ridge(alpha=cfg.alpha).fit(xs, y, sample_weight=pesos)
        else:
            self._modelo = LogisticRegression(C=cfg.c, max_iter=1000).fit(
                xs, (y > 0).astype(int), sample_weight=pesos
            )
            self._escala_logit = float(np.mean(np.abs(y)))
        return {"dias": int(filas.sum())}

    # -------------------------------------------------------------- prediccion
    def score_dia(self, histories: dict[str, pd.DataFrame]) -> tuple[float, float] | None:
        """Target predicho (unidades de vol) y vol del mercado de la ultima
        fecha; None si ese dia no hay mercado valido."""
        cfg = self.config
        dias, x = self._datos(histories)
        ultimo = x.index[-1]
        vol = float(dias.vol.iloc[-1])
        if not dias.validas.iloc[-1] or not np.isfinite(vol):
            return None
        if cfg.tipo == "regla":
            score = -float(x.loc[ultimo, f"m_ret_{cfg.horizonte}"])
            return (score, vol) if np.isfinite(score) else None
        if self._modelo is None or self._normalizador is None:
            raise NotEnoughDataError("la cabeza de mercado no esta entrenada")
        fila = np.nan_to_num(self._normalizador.transform(x.iloc[[-1]].to_numpy(dtype=float)))
        if cfg.tipo == "ridge":
            return float(self._modelo.predict(fila)[0]), vol
        p = float(self._modelo.predict_proba(fila)[0, 1])
        return (2 * p - 1) * self._escala_logit, vol

    def predict_panel(self, dfs: dict[str, pd.DataFrame]) -> dict[str, dict]:
        """El retorno esperado del mercado, igual para cada ticker de ``dfs`` con
        dato en la ultima rueda."""
        recortes = {s: df.iloc[-RUEDAS_PREDICCION:] for s, df in dfs.items() if not df.empty}
        if not recortes:
            return {}
        try:
            resultado = self.score_dia(recortes)
        except ValueError:  # ningun ticker del universo
            return {}
        if resultado is None:
            return {}
        score, vol = resultado
        log_return = score * vol * float(np.sqrt(self.config.horizonte))
        ultima = max(df.index[-1] for df in recortes.values())
        return {
            s: derive_trend_output(df, log_return, self.config.horizonte)
            for s, df in recortes.items()
            if df.index[-1] == ultima
        }

    def predict_df(self, df: pd.DataFrame) -> dict:
        raise ApiMlError("la cabeza de mercado predice con todos los tickers del dia")

    def resumen(self) -> dict:
        return {"version": self.version, "config": asdict(self.config), "columnas": self.columnas}
