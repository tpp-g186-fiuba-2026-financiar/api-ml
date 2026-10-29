"""Limpieza de los precios del snapshot: saltos que no son movimientos reales.

Los historicos traen eventos corporativos sin ajustar (splits, canjes,
cambios de ratio de CEDEARs) que aparecen como un salto de precio de un dia:
ECOG se divide por 10 el 2025-08-19, A3 se multiplica por 10,8 el
2024-05-27 (canje de MATba Rofex), AEM y ADGO se dividen por 2 el
2024-01-24. Ademas el futuro de petroleo (OIL) cerro con precio negativo el
2020-04-20. Cada salto contamina el target de las ventanas que lo cruzan
(+-200% en 5 ruedas) y la feature ``log_return`` de las ruedas siguientes.

Criterio: un salto es un retorno diario de mas de ``UMBRAL_SALTO`` (en
valor absoluto, log) que el mercado no acompano: la diferencia contra la
mediana del panel lider ese dia tambien supera el umbral. Asi un derrumbe
general (las PASO de 2019, -45% en un dia) no se toca. Cada salto se ajusta
como un split: las ruedas anteriores se multiplican por el factor del salto
(y el volumen se divide), con lo que el retorno de ese dia queda en 0.

Ajustar toda la serie no mira el futuro: las features son invariantes a la
escala (retornos, ratios, RSI), asi que una rueda anterior al salto produce
las mismas features con o sin ajuste.

Uso (reporte de los saltos detectados):
    python -m src.simulacion.limpieza
"""

from __future__ import annotations

import argparse
import dataclasses
import math

import numpy as np
import pandas as pd

from src.black_litterman.black_litterman import PANEL_LIDER_TICKERS
from src.simulacion.snapshot import DEFAULT_DIR, Snapshot

# Retorno log diario a partir del cual se sospecha un salto (x1,8 o /1,8).
UMBRAL_SALTO = math.log(1.8)
_PRECIOS = ["open", "high", "low", "close"]


def retorno_mercado(prices: dict[str, pd.DataFrame], tickers=PANEL_LIDER_TICKERS) -> pd.Series:
    """Mediana diaria del retorno log de ``tickers`` (robusta a un salto aislado)."""
    retornos = {s: np.log(prices[s]["close"]).diff() for s in tickers if s in prices}
    return pd.DataFrame(retornos).median(axis=1)


def sin_precios_no_positivos(df: pd.DataFrame) -> pd.DataFrame:
    """Descarta las ruedas con algun precio <= 0 (no tienen retorno log)."""
    columnas = [c for c in _PRECIOS if c in df.columns]
    return df[(df[columnas] > 0).all(axis=1)]


def detectar_saltos(
    df: pd.DataFrame, mercado: pd.Series, umbral: float = UMBRAL_SALTO
) -> list[tuple[pd.Timestamp, float]]:
    """Fechas y factores (cierre / cierre anterior) de los saltos de ``df``."""
    retorno = np.log(df["close"]).diff()
    relativo = retorno - mercado.reindex(retorno.index).fillna(0.0)
    es_salto = (retorno.abs() > umbral) & (relativo.abs() > umbral)
    return [(fecha, float(np.exp(retorno[fecha]))) for fecha in retorno.index[es_salto]]


def ajustar_salto(df: pd.DataFrame, fecha: pd.Timestamp, factor: float) -> pd.DataFrame:
    """Ajusta las ruedas anteriores a ``fecha`` como un split de ``factor`` (copia)."""
    df = df.copy()
    antes = df.index < fecha
    columnas = [c for c in _PRECIOS if c in df.columns]
    df.loc[antes, columnas] = df.loc[antes, columnas] * factor
    if "volume" in df.columns:
        df["volume"] = df["volume"].astype(float)
        df.loc[antes, "volume"] = df.loc[antes, "volume"] / factor
    return df


def limpiar_precios(
    prices: dict[str, pd.DataFrame], umbral: float = UMBRAL_SALTO
) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    """Precios sin ruedas con precio <= 0 y con los saltos ajustados, y el
    detalle de cada correccion (``symbol``, ``fecha``, ``tipo``, ``factor``)."""
    limpios = {s: sin_precios_no_positivos(df) for s, df in prices.items()}
    correcciones = [
        {"symbol": s, "fecha": None, "tipo": "precio<=0", "factor": None, "ruedas": n}
        for s in prices
        if (n := len(prices[s]) - len(limpios[s]))
    ]
    mercado = retorno_mercado(limpios)
    for symbol, df in limpios.items():
        for fecha, factor in detectar_saltos(df, mercado, umbral):
            df = ajustar_salto(df, fecha, factor)
            correcciones.append(
                {"symbol": symbol, "fecha": fecha, "tipo": "salto", "factor": factor, "ruedas": 1}
            )
        limpios[symbol] = df
    return limpios, correcciones


def limpiar(snapshot: Snapshot, umbral: float = UMBRAL_SALTO) -> tuple[Snapshot, list[dict]]:
    """Copia del snapshot con los precios limpios (ver ``limpiar_precios``)."""
    prices, correcciones = limpiar_precios(snapshot.prices, umbral)
    return dataclasses.replace(snapshot, prices=prices), correcciones


def reporte(correcciones: list[dict]) -> str:
    if not correcciones:
        return "sin correcciones"
    filas = []
    for c in correcciones:
        fecha = c["fecha"].date() if c["fecha"] is not None else "-"
        factor = f"x{c['factor']:.3f}" if c["factor"] is not None else f"{c['ruedas']} ruedas"
        filas.append(f"  {c['symbol']:<6} {fecha}  {c['tipo']:<9} {factor}")
    return "\n".join(filas)


def main() -> None:
    parser = argparse.ArgumentParser(description="Saltos detectados en el snapshot.")
    parser.add_argument("--snapshot", default=DEFAULT_DIR)
    parser.add_argument("--umbral", type=float, default=UMBRAL_SALTO, help="Retorno log diario.")
    args = parser.parse_args()
    _, correcciones = limpiar(Snapshot.load(args.snapshot), args.umbral)
    print(reporte(correcciones))


if __name__ == "__main__":
    main()
