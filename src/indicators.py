"""Indicadores tecnicos vectorizados sobre series de precios.

Se usan tanto como features de entrada de los modelos de tendencia (LSTM,
XGBoost, Transformer via ``src.lstm.build_features``) como para el
post-procesamiento de la respuesta (RSI/condicion en ``src.trend_common``).

Todas las funciones devuelven un array alineado 1:1 con la serie de entrada
(mismo largo), con ``NaN`` en las posiciones sin suficiente historia previa
para calcularse.
"""

from __future__ import annotations

import numpy as np


def _rsi_from_averages(avg_gain: float, avg_loss: float) -> float:
    if avg_loss == 0:
        return 50.0 if avg_gain == 0 else 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def rsi_series(close: np.ndarray, period: int = 14) -> np.ndarray:
    """RSI de Wilder, alineado a ``close`` (NaN en las primeras ``period`` posiciones)."""
    n = len(close)
    out = np.full(n, np.nan)
    if n < period + 1:
        return out

    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    avg_gain = gain[:period].mean()
    avg_loss = loss[:period].mean()
    out[period] = _rsi_from_averages(avg_gain, avg_loss)
    for i in range(period, len(delta)):
        avg_gain = (avg_gain * (period - 1) + gain[i]) / period
        avg_loss = (avg_loss * (period - 1) + loss[i]) / period
        out[i + 1] = _rsi_from_averages(avg_gain, avg_loss)
    return out


def sma(values: np.ndarray, window: int) -> np.ndarray:
    """Media movil simple, alineada (NaN en las primeras ``window - 1`` posiciones)."""
    n = len(values)
    out = np.full(n, np.nan)
    if n < window:
        return out
    cumsum = np.cumsum(np.insert(values, 0, 0.0))
    out[window - 1 :] = (cumsum[window:] - cumsum[:-window]) / window
    return out


def ema(values: np.ndarray, span: int) -> np.ndarray:
    """Media movil exponencial, alineada (arranca en el primer valor, sin NaN)."""
    alpha = 2.0 / (span + 1.0)
    out = np.empty(len(values), dtype=np.float64)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


def macd_histogram(
    close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9
) -> np.ndarray:
    """Histograma MACD (linea MACD menos su EMA de senal), alineado a ``close``.

    No tiene NaN (las EMA arrancan calentando desde el primer valor), pero los
    primeros ``slow`` + ``signal`` valores son ruidosos por el calentamiento.
    """
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    return macd_line - signal_line
