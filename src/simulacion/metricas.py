"""Metricas de una corrida: rendimiento de cada cartera y calidad de cada modelo."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest import summarize
from src.simulacion.snapshot import Snapshot

RUEDAS_POR_ANIO = 252


def metricas_cartera(valores: pd.Series) -> dict:
    """Rendimiento de una serie diaria de valor de cartera (sin tasa libre de riesgo)."""
    valores = valores.dropna()
    if len(valores) < 2:
        return {"ruedas": len(valores)}
    retornos = valores.pct_change().dropna()
    total = float(valores.iloc[-1] / valores.iloc[0] - 1)
    anual = float((1 + total) ** (RUEDAS_POR_ANIO / len(retornos)) - 1)
    volatilidad = float(retornos.std(ddof=1) * np.sqrt(RUEDAS_POR_ANIO))
    pico = valores.cummax()
    drawdown = float((valores / pico - 1).min())
    sharpe = (
        float(retornos.mean() / retornos.std(ddof=1) * np.sqrt(RUEDAS_POR_ANIO))
        if retornos.std(ddof=1) > 0
        else None
    )
    return {
        "ruedas": len(valores),
        "retorno_total": round(total, 4),
        "retorno_anualizado": round(anual, 4),
        "volatilidad_anualizada": round(volatilidad, 4),
        "sharpe": round(sharpe, 3) if sharpe is not None else None,
        "max_drawdown": round(drawdown, 4),
    }


def resolver_predicciones(
    predicciones: pd.DataFrame, snapshot: Snapshot, hasta: str
) -> pd.DataFrame:
    """Agrega el retorno realizado de cada prediccion cuyo horizonte ya se cumplio.

    El horizonte se cuenta en ruedas del propio ticker, igual que el paper
    trading. Las que todavia no se cumplieron a ``hasta`` quedan afuera.
    """
    limite = pd.Timestamp(hasta)
    filas = []
    for row in predicciones.itertuples(index=False):
        closes = snapshot.prices[row.symbol]["close"]
        closes = closes.loc[closes.index <= limite]
        fecha = pd.Timestamp(row.fecha)
        pos = closes.index.searchsorted(fecha)
        if pos >= len(closes) or closes.index[pos] != fecha:
            continue
        futuro = pos + int(row.horizon_days)
        if futuro >= len(closes):
            continue
        filas.append(
            {
                "fecha": fecha,
                "model": row.model,
                "symbol": row.symbol,
                "signal": row.signal,
                "predicted_log_return": float(np.log(row.predicted_close / row.last_close)),
                "realized_log_return": float(np.log(closes.iloc[futuro] / row.last_close)),
            }
        )
    return pd.DataFrame(filas)


def _summary(df: pd.DataFrame) -> dict:
    return summarize(df.to_dict("records"))


def calidad_modelos(resueltas: pd.DataFrame, universo: list[str] | None = None) -> dict:
    """Metricas de ``src.backtest.summarize`` por modelo: total y por mes."""
    if resueltas.empty:
        return {}
    if universo is not None:
        resueltas = resueltas[resueltas["symbol"].isin(universo)]
    resultado = {}
    for modelo, grupo in resueltas.groupby("model"):
        por_mes = {
            str(mes): _summary(g) for mes, g in grupo.groupby(grupo["fecha"].dt.to_period("M"))
        }
        resultado[modelo] = {"overall": _summary(grupo), "por_mes": por_mes}
    return resultado


def promociones_por_modelo(promociones: pd.DataFrame) -> dict:
    """Cuantas veces se reentreno y cuantas se promovio cada modelo."""
    if promociones.empty:
        return {}
    if "bootstrap" in promociones:
        bootstrap = promociones["bootstrap"].eq(True)
    else:
        bootstrap = pd.Series(False, index=promociones.index)
    corridas = promociones[~bootstrap]
    resultado = {}
    for modelo, grupo in corridas.groupby("model"):
        promovidas = grupo[grupo["promoted"].astype(bool)]
        resultado[modelo] = {
            "reentrenamientos": len(grupo),
            "promociones": len(promovidas),
            "fechas_promocion": [str(f) for f in promovidas["fecha"]],
        }
    return resultado


def rotacion(operaciones: pd.DataFrame, capital: float) -> dict:
    """Monto operado y comisiones por cartera, relativos al capital inicial."""
    if operaciones.empty:
        return {}
    resultado = {}
    for cartera, grupo in operaciones.groupby("cartera"):
        resultado[cartera] = {
            "operaciones": len(grupo),
            "monto_operado_sobre_capital": round(float(grupo["monto"].sum()) / capital, 3),
            "comisiones_sobre_capital": round(float(grupo["comision"].sum()) / capital, 4),
        }
    return resultado
