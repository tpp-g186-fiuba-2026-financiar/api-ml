"""Modelo GARCH de volatilidad, ajustado al vuelo por ticker en cada request.

Adaptado del GARCH que vive en el repo `models` (Modal), con una correccion
deliberada: GARCH modela la volatilidad de RETORNOS, no de precios en
nivel -- la version de Modal lo ajusta directo sobre precios crudos, que no
son estacionarios y pueden hacer que el ajuste no converja bien o de
resultados sin sentido. Aca se ajusta sobre retornos logaritmicos diarios
(escalados x100, practica estandar de la libreria `arch` para evitar
problemas numericos de escala).

No predice direccion (alza/baja) -- predice cuanto se espera que se mueva
el precio, para cualquier lado. Por eso no comparte el formato de
`TrendResponse`, tiene su propio endpoint y forma de respuesta.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from arch import arch_model

from src.errors import NotEnoughDataError

HORIZON = 5
MIN_ROWS = 60


@dataclass
class GarchVolatilityModel:
    horizon: int = HORIZON

    @property
    def version(self) -> str:
        return "garch-1-1"

    def _forecast_variance_pct2(self, close: np.ndarray) -> np.ndarray:
        if len(close) < MIN_ROWS:
            raise NotEnoughDataError(f"se necesitan al menos {MIN_ROWS} ruedas, hay {len(close)}")

        returns_pct = np.diff(np.log(close)) * 100.0

        model = arch_model(returns_pct, vol="Garch", p=1, q=1)
        result = model.fit(update_freq=0, disp="off")

        forecast = result.forecast(horizon=self.horizon)
        return forecast.variance.to_numpy()[-1]

    def predict_df(self, df: pd.DataFrame) -> dict:
        close = df["close"].to_numpy(dtype=np.float64)
        variance_pct2 = self._forecast_variance_pct2(close)
        daily_volatility_pct = np.sqrt(variance_pct2)
        # Volatilidad acumulada asumiendo retornos diarios independientes.
        cumulative_volatility_pct = float(np.sqrt(variance_pct2.sum()))

        return {
            "horizon_days": self.horizon,
            "daily_volatility_pct": [round(float(v), 4) for v in daily_volatility_pct],
            "cumulative_volatility_pct": round(cumulative_volatility_pct, 4),
            "last_close": round(float(close[-1]), 4),
            "as_of": df.index[-1].strftime("%Y-%m-%d"),
        }

    def forecast_daily_variance(self, df: pd.DataFrame) -> float:
        close = df["close"].to_numpy(dtype=np.float64)
        variance_pct2 = self._forecast_variance_pct2(close)
        primer_dia_pct2 = float(variance_pct2[0])
        return primer_dia_pct2 / (100.0**2)
