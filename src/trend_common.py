"""Post-procesamiento compartido por todos los modelos de tendencia.

Cada modelo (LSTM, XGBoost, y los que vengan) hace una unica cosa distinta:
predecir el *retorno logaritmico acumulado* de los proximos ``horizon`` dias.
A partir de ese numero, la derivacion de la senal, el RSI, el precio proyectado
y la confianza es identica para todos. Centralizarla aca garantiza que las
predicciones de modelos distintos sean directamente comparables (misma escala,
mismos umbrales, mismo formato de respuesta).
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd

from src.errors import ApiMlError
from src.lstm import rsi

BACKTEST_DAYS = 60


class _DfPredictor(Protocol):
    def predict_df(self, df: pd.DataFrame) -> dict: ...


def derive_trend_output(
    df: pd.DataFrame,
    log_return: float,
    horizon: int,
    neutral_band: float = 0.01,
) -> dict:
    """Convierte un retorno log predicho en el dict de respuesta de tendencia.

    ``log_return`` es la salida cruda del modelo (retorno log acumulado a
    ``horizon`` dias), usada internamente para derivar ``signal`` y
    ``predicted_close``. No se expone en la respuesta (ver `expected_return`
    y `confidence` sacados del schema por decision del equipo: el retorno
    esperado se resume en `predicted_close`, y `confidence` saturaba en 1.0
    demasiado facil como para ser util).
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

    return {
        "signal": signal,
        "horizon_days": horizon,
        "predicted_close": round(predicted_close, 4),
        "last_close": round(last_close, 4),
        "rsi": round(rsi_value, 2) if rsi_value is not None else None,
        "condition": condition,
        "as_of": df.index[-1].strftime("%Y-%m-%d"),
    }


def backtest_predict_df(
    predictor: _DfPredictor,
    df: pd.DataFrame,
    horizon: int,
    backtest_days: int = BACKTEST_DAYS,
) -> dict | None:
    """Backtest walk-forward generico para cualquier modelo con `predict_df`.

    Reusa el modelo ya entrenado (no reentrena en cada paso, igual que
    `backtest_lstm`/`backtest_xgboost` en el repo `models`): para cada dia de
    los ultimos `backtest_days`, predice con el historico disponible hasta
    ese dia y compara contra lo que paso `horizon` dias despues. Devuelve
    `None` si no hay historia suficiente en vez de tirar, para que el
    comparador siga mostrando la prediccion aunque no pueda medir accuracy.
    """
    split = len(df) - backtest_days - horizon
    if split < 1:
        return None
    predicted_returns: list[float] = []
    actual_returns: list[float] = []
    series: list[dict] = []
    for cutoff in range(split, len(df) - horizon):
        try:
            result = predictor.predict_df(df.iloc[: cutoff + 1])
        except ApiMlError:
            continue
        predicted_close = result.get("predicted_close")
        if predicted_close is None:
            continue
        last_close = float(df["close"].iloc[cutoff])
        future_close = float(df["close"].iloc[cutoff + horizon])
        predicted_returns.append(float(np.log(predicted_close / last_close)))
        actual_returns.append(float(np.log(future_close / last_close)))
        series.append(
            {
                "date": df.index[cutoff + horizon].strftime("%Y-%m-%d"),
                "predicted": predicted_close,
                "actual": future_close,
            }
        )
    if not actual_returns:
        return None
    predicted_arr = np.asarray(predicted_returns)
    actual_arr = np.asarray(actual_returns)
    return {
        "directional_accuracy": float(np.mean(np.sign(predicted_arr) == np.sign(actual_arr))),
        "mae": float(np.mean(np.abs(predicted_arr - actual_arr))),
        "observations": len(actual_returns),
        "series": series[-30:],
    }
