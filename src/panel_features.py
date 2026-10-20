"""Features y target de panel: todas las acciones del dia, comparables entre si.

Parte de un ``Panel`` (``src.panel_datos``) y devuelve tensores
``[fecha, ticker, feature]`` listos para un modelo de panel:

1. **Features por accion** sobre los retornos log diarios (en la moneda del
   panel; las celdas fuera de la mascara no cuentan): retornos de 1 a 60
   ruedas, volatilidades, rango, gap, volumen anormal, distancia a medias,
   drawdowns, rachas, beta y correlacion con el mercado, y las de
   **reversion**: retorno de 5 ruedas relativo al panel y a su sector.
2. **Normalizacion por volatilidad**: los retornos se dividen por la vol EWMA
   de 20 ruedas del propio ticker (escalada por la raiz del plazo) y se
   recortan a +-``clip_features``. Sin esto, las acciones volatiles dominan.
3. **Rank gaussiano por dia**: cada feature se reemplaza por su rank entre los
   tickers validos ese dia, llevado a cuantiles de una normal. El modelo ve
   "mas que el resto" en vez de niveles que cambian con el regimen. Dos
   features quedan ademas **sin rankear** (``NIVEL``) para que vea el nivel.
4. **Target**: retorno log a ``horizonte`` ruedas, dividido por la vol del
   ticker, recortado a +-``clip_target`` y con la media del dia restada
   (``"sin_media"``), o residuo despues de sacar ``beta * mercado``
   (``"residual"``), o igual que ``"sin_media"`` pero sin restar la media
   (``"absoluto"``: conserva si sube o baja, para predecir la direccion), o el
   retorno crudo (``"crudo"``). Se guarda tambien el retorno crudo para evaluar.
   Con ``zona_neutral`` > 0, los targets chicos (``|target| < zona_neutral``)
   quedan fuera de la mascara de entrenamiento.

El "mercado" es el promedio de los retornos de los tickers validos de cada
dia (no el Merval): sale del mismo panel y no depende de otra serie.

Todo mira hacia atras salvo el target: las features de la fecha ``t`` no
cambian si se borran los datos posteriores a ``t`` (lo verifican los tests).
Las estadisticas de normalizacion que dependen de la muestra (``Normalizador``)
se ajustan solo con las fechas de entrenamiento.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.macro_trend import event_features, macro_features
from src.panel_datos import Panel

FEATURES_ACCION = [
    "ret_1", "ret_5", "ret_10", "ret_20", "ret_60",
    "rel_panel_5", "rel_sector_5",
    "vol_5", "vol_20", "vol_60", "rango", "gap", "volumen_z",
    "sma_20", "sma_50", "dd_60", "dd_252", "racha", "beta_60", "corr_60",
]  # fmt: skip
# Retornos que se dividen por la vol del ticker (y su plazo en ruedas).
RETORNOS = {
    "ret_1": 1, "ret_5": 5, "ret_10": 10, "ret_20": 20, "ret_60": 60,
    "rel_panel_5": 5, "rel_sector_5": 5, "gap": 1,
}  # fmt: skip
# Copias sin rankear (nivel de la accion, no su posicion en el dia).
NIVEL = ["ret_5_nivel", "vol_20_nivel"]
FLAGS = ["historia_corta"]
# Todas las features, en el orden de la ultima dimension de ``FeaturesPanel.valores``.
NOMBRES = FEATURES_ACCION + NIVEL + FLAGS

TARGETS = ("sin_media", "residual", "absoluto", "crudo")


@dataclass(frozen=True)
class PreprocesoConfig:
    horizonte: int = 5
    normalizar_vol: bool = True
    rankear: bool = True
    target: str = "sin_media"
    clip_features: float = 5.0
    clip_target: float = 3.0
    span_vol: int = 20  # span de la vol EWMA para normalizar retornos y target
    # Celdas con |target| menor que esto (en unidades del target: vol a
    # ``horizonte`` ruedas) quedan fuera de ``mascara_target``: no se entrena con
    # movimientos chicos, que son sobre todo ruido. 0 = todas.
    zona_neutral: float = 0.0

    def __post_init__(self) -> None:
        if self.target not in TARGETS:
            raise ValueError(f"target {self.target!r} desconocido ({TARGETS})")
        if self.zona_neutral < 0:
            raise ValueError("zona_neutral no puede ser negativa")


@dataclass
class FeaturesPanel:
    """Features ``[fecha, ticker, feature]`` y target ``[fecha, ticker]``.

    ``mascara`` marca las celdas donde las features son usables (dato valido y
    vol conocida); ``mascara_target`` ademas exige el retorno a ``horizonte``.
    Los faltantes de ``valores`` ya estan imputados con 0. ``vol`` es la vol
    EWMA diaria de cada celda (para pasar un score en unidades de vol a retorno).
    """

    valores: np.ndarray
    nombres: list[str]
    fechas: pd.DatetimeIndex
    tickers: list[str]
    sectores: list[str]
    mascara: np.ndarray
    target: np.ndarray
    retorno: np.ndarray
    mascara_target: np.ndarray
    vol: np.ndarray | None = None

    def tabla(self) -> pd.DataFrame:
        """Formato largo (una fila por fecha y ticker validos), para inspeccionar o un GBM."""
        t, n = np.nonzero(self.mascara)
        df = pd.DataFrame(self.valores[t, n], columns=self.nombres)
        df.insert(0, "ticker", np.asarray(self.tickers)[n])
        df.insert(0, "fecha", self.fechas[t])
        df["target"] = self.target[t, n]
        df["retorno"] = self.retorno[t, n]
        return df


def rank_gaussiano(df: pd.DataFrame) -> pd.DataFrame:
    """Rank de cada fila (entre las celdas no nulas) llevado a cuantiles de N(0, 1)."""
    rango = df.rank(axis=1, method="average")
    n = df.notna().sum(axis=1)
    cuantil = rango.sub(0.5).div(n.replace(0, np.nan), axis=0)
    return pd.DataFrame(norm.ppf(cuantil), index=df.index, columns=df.columns)


def _racha(r: pd.Series) -> pd.Series:
    """Ruedas seguidas subiendo (positivo) o bajando (negativo)."""
    signo = np.sign(r).fillna(0)
    grupo = (signo != signo.shift()).cumsum()
    return signo.groupby(grupo).cumsum()


def _por_columna(df: pd.DataFrame, f) -> pd.DataFrame:
    return pd.DataFrame({c: f(df[c]) for c in df.columns}, index=df.index)


def vol_ewma(r: pd.DataFrame, span: int) -> pd.DataFrame:
    """Desvio EWMA de los retornos diarios (mira hacia atras)."""
    return r.ewm(span=span, min_periods=span // 2).std()


def features_accion(panel: Panel, config: PreprocesoConfig) -> dict[str, pd.DataFrame]:
    """Features crudas por accion (sin rankear), cada una ``[fecha x ticker]``."""
    m = panel.mascara
    close = panel["close"].where(m)
    log_c = np.log(close)
    r = log_c.diff()
    mercado = r.mean(axis=1)
    f: dict[str, pd.DataFrame] = {}
    for k in (1, 5, 10, 20, 60):
        f[f"ret_{k}"] = log_c.diff(k)
    sectores = pd.Series(panel.sectores(), index=panel.tickers)
    ret5 = f["ret_5"]
    f["rel_panel_5"] = ret5.sub(ret5.mean(axis=1), axis=0)
    media_sector = ret5.T.groupby(sectores).transform("mean").T
    f["rel_sector_5"] = ret5 - media_sector
    for k in (5, 20, 60):
        f[f"vol_{k}"] = r.rolling(k, min_periods=max(3, k * 3 // 4)).std()
    f["rango"] = np.log(panel["high"].where(m) / panel["low"].where(m))
    f["gap"] = np.log(panel["open"].where(m) / close.shift())
    log_vol = np.log1p(panel["volume"].where(m))
    media_vol = log_vol.rolling(20, min_periods=10)
    f["volumen_z"] = (log_vol - media_vol.mean()) / media_vol.std().replace(0, np.nan)
    for k in (20, 50):
        f[f"sma_{k}"] = close / close.rolling(k, min_periods=k * 3 // 4).mean() - 1
    f["dd_60"] = close / close.rolling(60, min_periods=20).max() - 1
    f["dd_252"] = close / close.rolling(252, min_periods=60).max() - 1
    f["racha"] = _por_columna(r, _racha).where(m)
    var_m = mercado.rolling(60, min_periods=40).var()
    f["beta_60"] = _por_columna(r, lambda s: s.rolling(60, min_periods=40).cov(mercado) / var_m)
    f["corr_60"] = _por_columna(r, lambda s: s.rolling(60, min_periods=40).corr(mercado))
    vol = vol_ewma(r, config.span_vol)
    if config.normalizar_vol:
        for nombre, plazo in RETORNOS.items():
            escala = vol * np.sqrt(plazo)
            f[nombre] = (f[nombre] / escala).clip(-config.clip_features, config.clip_features)
    f["ret_5_nivel"] = f["ret_5"].copy()
    f["vol_20_nivel"] = np.log(f["vol_20"])
    f["historia_corta"] = f["ret_60"].isna().astype(float).where(m)
    f["_vol"] = vol
    f["_mercado"] = pd.DataFrame({c: mercado for c in panel.tickers})
    return {k: v.replace([np.inf, -np.inf], np.nan) for k, v in f.items()}


def contexto_macro(macro: pd.DataFrame, fechas: pd.DatetimeIndex) -> pd.DataFrame:
    """Variables macro y de eventos por fecha (comunes a todas las acciones).

    ``macro`` tiene que traer los retrasos de publicacion (en la simulacion,
    ``Snapshot.macro_frame(hasta)``). Reusa las features del modelo macro.
    """
    return macro_features(macro, fechas).join(event_features(fechas))


def construir(panel: Panel, config: PreprocesoConfig | None = None) -> FeaturesPanel:
    """Features, mascaras y target del panel (ver docstring del modulo)."""
    config = config or PreprocesoConfig()
    crudas = features_accion(panel, config)
    vol, mercado = crudas.pop("_vol"), crudas.pop("_mercado")
    mascara = panel.mascara & vol.notna()

    capas = []
    for nombre in FEATURES_ACCION:
        df = crudas[nombre].where(mascara)
        capas.append(rank_gaussiano(df) if config.rankear else df)
    for nombre in NIVEL + FLAGS:
        capas.append(crudas[nombre].where(mascara))
    valores = np.stack([c.to_numpy(dtype=float) for c in capas], axis=-1)
    valores = np.nan_to_num(valores, nan=0.0)

    retorno, target, mascara_target = _target(panel, mascara, vol, mercado, crudas, config)
    return FeaturesPanel(
        valores=valores,
        nombres=list(NOMBRES),
        fechas=panel.fechas,
        tickers=panel.tickers,
        sectores=panel.sectores(),
        mascara=mascara.to_numpy(),
        target=target,
        retorno=retorno,
        mascara_target=mascara_target,
        vol=vol.where(mascara).to_numpy(dtype=float),
    )


def _target(
    panel: Panel,
    mascara: pd.DataFrame,
    vol: pd.DataFrame,
    mercado: pd.DataFrame,
    crudas: dict[str, pd.DataFrame],
    config: PreprocesoConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h = config.horizonte
    log_c = np.log(panel["close"].where(panel.mascara))
    retorno = log_c.shift(-h) - log_c
    valido = mascara & panel.mascara.shift(-h, fill_value=False) & retorno.notna()
    retorno = retorno.where(valido)
    if config.target == "crudo":
        target = retorno
    else:
        if config.target == "residual":
            mercado_fwd = mercado.rolling(h).sum().shift(-h)
            retorno_target = retorno - crudas["beta_60"].fillna(1.0) * mercado_fwd
        else:
            retorno_target = retorno
        escala = vol * np.sqrt(h)
        target = (retorno_target / escala).clip(-config.clip_target, config.clip_target)
        if config.target != "absoluto":
            target = target.sub(target.mean(axis=1), axis=0)
    valido &= target.notna()
    if config.zona_neutral > 0:
        valido &= target.abs() >= config.zona_neutral
    return (
        retorno.where(valido).to_numpy(dtype=float),
        target.where(valido).to_numpy(dtype=float),
        valido.to_numpy(),
    )


@dataclass
class Normalizador:
    """Z-score por feature con media y desvio de las celdas de entrenamiento."""

    media: np.ndarray | None = None
    desvio: np.ndarray | None = None

    def fit(self, valores: np.ndarray, mascara: np.ndarray | None = None) -> Normalizador:
        """``valores`` ``[..., feature]``; ``mascara`` (misma forma sin la ultima
        dimension) elige las celdas que cuentan."""
        x = valores if mascara is None else valores[mascara]
        x = x.reshape(-1, valores.shape[-1])
        self.media = np.nanmean(x, axis=0)
        desvio = np.nanstd(x, axis=0)
        self.desvio = np.where(desvio > 1e-12, desvio, 1.0)
        return self

    def transform(self, valores: np.ndarray) -> np.ndarray:
        if self.media is None:
            raise ValueError("el normalizador no esta ajustado")
        return (valores - self.media) / self.desvio


def particion(
    fechas: pd.DatetimeIndex, val_fraction: float = 0.15, gap: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """Indices de entrenamiento y validacion por fecha, con ``gap`` fechas
    descartadas entre ambos (con target a ``gap`` ruedas, las ultimas de train
    comparten futuro con las primeras de validacion)."""
    n = len(fechas)
    n_val = max(1, int(round(n * val_fraction)))
    fin_train = n - n_val - gap
    if fin_train <= 0:
        raise ValueError(f"muy pocas fechas ({n}) para validar {n_val} con gap {gap}")
    return np.arange(fin_train), np.arange(n - n_val, n)


def pesos_temporales(fechas: pd.DatetimeIndex, vida_media_anios: float | None = 2.0) -> np.ndarray:
    """Peso de cada fecha: 1 la mas reciente, la mitad cada ``vida_media_anios``
    hacia atras (``None`` = todas pesan igual). Normalizados a media 1."""
    if vida_media_anios is None:
        return np.ones(len(fechas))
    edad = (fechas.max() - fechas).days.to_numpy() / 365.25
    pesos = 0.5 ** (edad / vida_media_anios)
    return pesos / pesos.mean()
