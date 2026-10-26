"""Precio estimado por regresion lineal (canal de tendencia).

Ajusta ``log(precio) = a + b * t`` con minimos cuadrados sobre las ultimas
``ventana`` ruedas. El valor de la recta hoy es el "precio estimado por la
tendencia"; la distancia del precio real a esa recta, medida en desvios
estandar de los residuos, dice si la accion esta por encima o por debajo de
su tendencia. Es el mismo calculo que el "canal de regresion lineal" de las
plataformas de graficos, en escala logaritmica para que un 10% pese igual a
cualquier precio.

No es un modelo de prediccion: describe donde esta el precio respecto de su
propia tendencia. Lo que paso historicamente despues esta en ``EVIDENCIA``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.errors import NotEnoughDataError

VENTANA_DEFAULT = 252
MIN_RUEDAS = 60
# A partir de cuantos desvios se considera que el precio se aparto de la tendencia.
UMBRAL_DESVIOS = 1.0

# Que paso en las 20 ruedas siguientes segun la distancia a la tendencia.
# Panel lider (20 acciones), sep 2021 - sep 2026, cada 3 ruedas, ajustando la
# recta solo con el año anterior a cada fecha (7.842 casos). "relativo" es el
# retorno log menos el promedio del panel ese dia. Correlacion de rango entre
# acciones cada dia: -0,05 (t = -2,6, inflado por ventanas solapadas): una
# reversion a la tendencia debil pero consistente.
EVIDENCIA = {
    "horizonte": 20,
    "casos": 7842,
    "desde": "2021-09",
    "hasta": "2026-09",
    "tramos": [
        {"desde": None, "hasta": -2.0, "sube": 0.725, "relativo": 0.0083, "casos": 397},
        {"desde": -2.0, "hasta": -1.0, "sube": 0.668, "relativo": 0.0014, "casos": 1737},
        {"desde": -1.0, "hasta": 1.0, "sube": 0.595, "relativo": 0.0017, "casos": 3905},
        {"desde": 1.0, "hasta": 2.0, "sube": 0.594, "relativo": -0.0056, "casos": 1123},
        {"desde": 2.0, "hasta": None, "sube": 0.575, "relativo": -0.0090, "casos": 680},
    ],
}


def tramo_de(desvios: float) -> dict:
    """Tramo de ``EVIDENCIA`` en el que cae una distancia a la tendencia."""
    for tramo in EVIDENCIA["tramos"]:
        if (tramo["desde"] is None or desvios >= tramo["desde"]) and (
            tramo["hasta"] is None or desvios < tramo["hasta"]
        ):
            return tramo
    return EVIDENCIA["tramos"][-1]


@dataclass(frozen=True)
class AjusteLineal:
    pendiente: float  # por rueda, en log
    ordenada: float
    desvio: float  # desvio estandar de los residuos, en log
    r2: float


def ajustar(log_precios: np.ndarray) -> AjusteLineal:
    """Minimos cuadrados de log(precio) contra 0..n-1."""
    n = len(log_precios)
    t = np.arange(n, dtype=np.float64)
    pendiente, ordenada = np.polyfit(t, log_precios, 1)
    residuos = log_precios - (ordenada + pendiente * t)
    total = float(np.sum((log_precios - log_precios.mean()) ** 2))
    r2 = 1.0 - float(np.sum(residuos**2)) / total if total > 0 else 0.0
    return AjusteLineal(
        pendiente=float(pendiente),
        ordenada=float(ordenada),
        desvio=float(np.std(residuos, ddof=2)) if n > 2 else 0.0,
        r2=r2,
    )


def precio_estimado(df: pd.DataFrame, ventana: int = VENTANA_DEFAULT) -> dict:
    """Precio estimado por la tendencia y donde esta el precio real respecto
    de ella, con la recta y el canal de +-1 y +-2 desvios para graficar."""
    close = df["close"].dropna()
    close = close[close > 0].iloc[-ventana:]
    if len(close) < MIN_RUEDAS:
        raise NotEnoughDataError(
            f"hacen falta al menos {MIN_RUEDAS} ruedas para la regresion; hay {len(close)}"
        )
    log_p = np.log(close.to_numpy(dtype=np.float64))
    ajuste = ajustar(log_p)
    t = np.arange(len(log_p), dtype=np.float64)
    recta = ajuste.ordenada + ajuste.pendiente * t
    estimado = float(np.exp(recta[-1]))
    actual = float(close.iloc[-1])
    desvios = (log_p[-1] - recta[-1]) / ajuste.desvio if ajuste.desvio > 0 else 0.0
    if desvios > UMBRAL_DESVIOS:
        posicion = "por_encima"
    elif desvios < -UMBRAL_DESVIOS:
        posicion = "por_debajo"
    else:
        posicion = "en_linea"
    s = ajuste.desvio
    return {
        "as_of": close.index[-1].strftime("%Y-%m-%d"),
        "ventana": len(close),
        "precio_actual": actual,
        "precio_estimado": estimado,
        "diferencia": actual / estimado - 1.0,
        "desvios": float(desvios),
        "posicion": posicion,
        "tendencia_anual": float(np.exp(ajuste.pendiente * 252) - 1.0),
        "r2": ajuste.r2,
        "desvio_log": s,
        "historico": {**tramo_de(float(desvios)), "horizonte": EVIDENCIA["horizonte"]},
        "evidencia": EVIDENCIA,
        "canal": [
            {
                "date": fecha.strftime("%Y-%m-%d"),
                "estimado": float(np.exp(r)),
                "lo1": float(np.exp(r - s)),
                "hi1": float(np.exp(r + s)),
                "lo2": float(np.exp(r - 2 * s)),
                "hi2": float(np.exp(r + 2 * s)),
            }
            for fecha, r in zip(close.index, recta, strict=True)
        ],
    }


TREND_WINDOWS = (126, 252)
TREND_FEATURES = [f"{name}{w}" for w in TREND_WINDOWS for name in ("tz", "tslope", "tr2_")]


def trend_features(close: pd.Series, windows: tuple[int, ...] = TREND_WINDOWS) -> pd.DataFrame:
    """La misma recta de ``precio_estimado`` en ventana movil, para cada rueda
    (sin mirar el futuro): distancia del precio a la recta en desvios
    (``tz``), pendiente anual (``tslope``) y R2 (``tr2_``). Las usa XGBoost:
    sumarlas mejoro el AUC por mes de 0,580 a 0,588 (IC95 de la diferencia
    pareada por mes [+0,001; +0,014]) en el walk-forward 2016-2026."""
    y = np.log(close.where(close > 0).to_numpy(dtype=np.float64))
    n = len(y)
    t = np.arange(n, dtype=np.float64)

    def acum(a: np.ndarray) -> np.ndarray:
        return np.concatenate([[0.0], np.nancumsum(a)])

    sy, sty, syy, st, stt = acum(y), acum(t * y), acum(y * y), acum(t), acum(t * t)
    out = {}
    for w in windows:
        z = np.full(n, np.nan)
        slope = np.full(n, np.nan)
        r2 = np.full(n, np.nan)
        if n >= w:
            fin = np.arange(w, n + 1)
            ini = fin - w
            s_y, s_ty, s_yy = sy[fin] - sy[ini], sty[fin] - sty[ini], syy[fin] - syy[ini]
            s_t, s_tt = st[fin] - st[ini], stt[fin] - stt[ini]
            b = (w * s_ty - s_t * s_y) / (w * s_tt - s_t * s_t)
            a = (s_y - b * s_t) / w
            sse = s_yy - 2 * a * s_y - 2 * b * s_ty + w * a * a + 2 * a * b * s_t + b * b * s_tt
            sst = s_yy - s_y * s_y / w
            sd = np.sqrt(np.maximum(sse, 0.0) / (w - 2))
            ult = fin - 1
            with np.errstate(divide="ignore", invalid="ignore"):
                z[ult] = np.where(sd > 0, (y[ult] - (a + b * t[ult])) / sd, 0.0)
                r2[ult] = np.where(sst > 0, 1 - sse / sst, 0.0)
            slope[ult] = b * 252
        out[f"tz{w}"], out[f"tslope{w}"], out[f"tr2_{w}"] = z, slope, r2
    return pd.DataFrame(out, index=close.index)[
        [f"{name}{w}" for w in windows for name in ("tz", "tslope", "tr2_")]
    ]
