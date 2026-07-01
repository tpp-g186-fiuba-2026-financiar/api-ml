"""Post-procesamiento compartido por todos los modelos de tendencia.

Cada modelo (LSTM, XGBoost, y los que vengan) hace una unica cosa distinta:
predecir el *retorno logaritmico acumulado* de los proximos ``horizon`` dias.
A partir de ese numero, la derivacion de la senal, el RSI, el precio proyectado
y la confianza es identica para todos. Centralizarla aca garantiza que las
predicciones de modelos distintos sean directamente comparables (misma escala,
mismos umbrales, mismo formato de respuesta).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.lstm import rsi

# Retorno esperado (en valor absoluto) que satura la confianza en 1.0.
_CONFIDENCE_SCALE = 0.03


def derive_trend_output(
    df: pd.DataFrame,
    log_return: float,
    horizon: int,
    neutral_band: float = 0.01,
) -> dict:
    """Convierte un retorno log predicho en el dict de respuesta de tendencia.

    ``log_return`` es la salida cruda del modelo (retorno log acumulado a
    ``horizon`` dias). El resto se deriva igual para cualquier modelo.
    """
    last_close = float(df["close"].iloc[-1])
    expected_return = float(np.expm1(log_return))
    predicted_close = float(last_close * np.exp(log_return))

    if expected_return > neutral_band:
        signal = "alza"
    elif expected_return < -neutral_band:
        signal = "baja"
    else:
        signal = "neutral"

    rsi_value = rsi(df["close"].to_numpy(dtype=np.float64))
    if rsi_value is None:
        condition = "indeterminado"
    elif rsi_value >= 70:
        condition = "sobrecompra"
    elif rsi_value <= 30:
        condition = "sobreventa"
    else:
        condition = "neutral"

    confidence = float(min(1.0, abs(expected_return) / _CONFIDENCE_SCALE))

    return {
        "signal": signal,
        "horizon_days": horizon,
        "expected_return": round(expected_return, 6),
        "predicted_close": round(predicted_close, 4),
        "last_close": round(last_close, 4),
        "rsi": round(rsi_value, 2) if rsi_value is not None else None,
        "condition": condition,
        "confidence": round(confidence, 4),
        "as_of": df.index[-1].strftime("%Y-%m-%d"),
    }
