"""Modelo ARIMA de tendencia, ajustado al vuelo por ticker en cada request.

Adaptado del ARIMA que vive en el repo `models` (Modal). A diferencia de
LSTM/XGBoost, ARIMA es un modelo univariado: no tiene sentido "poolear"
series de tickers distintos en un unico modelo, cada ticker necesita su
propio ajuste. Por eso este modelo no se entrena offline ni persiste un
artefacto -- se ajusta con el historico de precios del ticker pedido en
cada request. Es mucho mas liviano que la LSTM (ajuste en fracciones de
segundo a un par de segundos), asi que el costo de reajustar en cada
llamada es aceptable.

Se pronostica el precio a `horizon` ruedas y se deriva la salida con
`derive_trend_output` (igual que LSTM/XGBoost), para que quede en el mismo
formato de respuesta y sea comparable.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from src.errors import NotEnoughDataError
from src.trend_common import derive_trend_output

HORIZON = 5
# El arima_model.py de Modal usa media_movil=20 por default, pero un orden
# de MA tan alto genera inestabilidad numerica en el ajuste (overflow en el
# filtro de Kalman) sin mejorar el AIC de forma significativa (probado con
# datos reales de GGAL: AIC practicamente igual entre q=1 y q=20, pero
# q=1 es el unico que ajusta sin warnings). Se usa un ARIMA(1,1,1) simple.
MOVING_AVERAGE_ORDER = 1
NEUTRAL_BAND = 0.01
MIN_ROWS = 60


@dataclass
class ArimaTrendModel:
    horizon: int = HORIZON
    order_ma: int = MOVING_AVERAGE_ORDER
    neutral_band: float = NEUTRAL_BAND

    @property
    def version(self) -> str:
        return f"arima-1-1-{self.order_ma}"

    def predict_df(self, df: pd.DataFrame) -> dict:
        close = df["close"]
        if len(close) < MIN_ROWS:
            raise NotEnoughDataError(f"se necesitan al menos {MIN_ROWS} ruedas, hay {len(close)}")

        model = ARIMA(close.reset_index(drop=True), order=(1, 1, self.order_ma))
        result = model.fit()
        forecast = result.forecast(steps=self.horizon)

        last_close = float(close.iloc[-1])
        forecast_close = float(forecast.iloc[-1])
        log_return = float(np.log(forecast_close / last_close))

        return derive_trend_output(df, log_return, self.horizon, self.neutral_band)


def translate_modal_arima_response(payload: dict) -> dict:
    """Traduce la respuesta del ``arima_model.py`` de Modal (repo `models`,
    version original del equipo, no la nuestra) al formato de TrendResponse.

    Esa version de Modal usa otros parametros (``predictions``,
    ``media_movil`` en vez de ``horizon``/``order_ma``) y no manda el
    historico de cierres -- por eso no se puede calcular RSI/condition ahi
    (quedan en None/"indeterminado"), y el ``as_of`` es una aproximacion
    (fecha de hoy), no la fecha real de la ultima rueda que uso Modal.
    """
    forecast = payload.get("prediction") or []
    if not forecast:
        raise ValueError("Modal (arima) no devolvio pronostico")

    last_close = float(payload["valor_actual"])
    forecast_close = float(forecast[-1])
    log_return = float(np.log(forecast_close / last_close))
    expected_return = float(np.expm1(log_return))
    predicted_close = float(last_close * np.exp(log_return))
    horizon = int(payload.get("cant_predicciones") or len(forecast))

    if expected_return > NEUTRAL_BAND:
        signal = "alza"
    elif expected_return < -NEUTRAL_BAND:
        signal = "baja"
    else:
        signal = "neutral"

    return {
        "signal": signal,
        "horizon_days": horizon,
        "predicted_close": round(predicted_close, 4),
        "last_close": round(last_close, 4),
        "rsi": None,
        "condition": "indeterminado",
        "as_of": dt.date.today().isoformat(),
    }
