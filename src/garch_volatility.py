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
# Ventanas del backtest por accion (no se solapan: cada una dura `horizon`
# ruedas) y ruedas con las que se compara el regimen de hoy.
BACKTEST_WINDOWS = 24
REGIME_DAYS = 504


@dataclass
class GarchVolatilityModel:
    horizon: int = HORIZON

    @property
    def version(self) -> str:
        return "garch-1-1"

    def _forecast_variance_pct2(self, close: np.ndarray, horizon: int | None = None) -> np.ndarray:
        if len(close) < MIN_ROWS:
            raise NotEnoughDataError(f"se necesitan al menos {MIN_ROWS} ruedas, hay {len(close)}")

        returns_pct = np.diff(np.log(close)) * 100.0

        model = arch_model(returns_pct, vol="Garch", p=1, q=1)
        result = model.fit(update_freq=0, disp="off")

        forecast = result.forecast(horizon=horizon or self.horizon)
        return forecast.variance.to_numpy()[-1]

    def predict_df(self, df: pd.DataFrame, horizon: int | None = None) -> dict:
        close = df["close"].to_numpy(dtype=np.float64)
        horizon = horizon or self.horizon
        variance_pct2 = self._forecast_variance_pct2(close, horizon)
        daily_volatility_pct = np.sqrt(variance_pct2)
        # Volatilidad acumulada asumiendo retornos diarios independientes.
        cumulative_volatility_pct = float(np.sqrt(variance_pct2.sum()))

        return {
            "horizon_days": horizon,
            "daily_volatility_pct": [round(float(v), 4) for v in daily_volatility_pct],
            "cumulative_volatility_pct": round(cumulative_volatility_pct, 4),
            "last_close": round(float(close[-1]), 4),
            "as_of": df.index[-1].strftime("%Y-%m-%d"),
        }

    def forecast_variance_horizonte(self, df: pd.DataFrame, horizonte: int) -> float:
        """Varianza acumulada de los proximos ``horizonte`` dias (suma de las
        varianzas diarias del forecast), en unidades de retorno (no %^2).
        """
        close = df["close"].to_numpy(dtype=np.float64)
        variance_pct2 = self._forecast_variance_pct2(close, horizonte)
        return float(variance_pct2.sum()) / (100.0**2)

    def contexto(self, df: pd.DataFrame, horizon: int | None = None) -> dict:
        """Regimen de hoy y backtest de GARCH sobre esta accion.

        - ``regime``: la volatilidad diaria pronosticada para manana, ubicada
          como percentil entre las volatilidades realizadas a 20 ruedas de los
          ultimos ``REGIME_DAYS`` dias; mas la de largo plazo del modelo
          (``sqrt(omega / (1 - alpha - beta))``).
        - ``backtest``: ``BACKTEST_WINDOWS`` ventanas de ``horizon`` ruedas
          que no se solapan. En cada una se ajusta GARCH solo con los datos
          anteriores y se compara su pronostico con lo que paso: si el retorno
          quedo dentro de +-1 y +-1,96 desvios, y la correlacion con la
          volatilidad realizada, contra el pronostico ingenuo (volatilidad de
          las 20 ruedas previas).
        """
        horizon = horizon or self.horizon
        close = df["close"].to_numpy(dtype=np.float64)
        if len(close) < MIN_ROWS + horizon * 2:
            raise NotEnoughDataError(f"pocas ruedas para el contexto de volatilidad ({len(close)})")
        returns = np.diff(np.log(close))
        return {
            "regime": self._regime(returns),
            "backtest": self._backtest(df, returns, horizon),
        }

    def _regime(self, returns: np.ndarray) -> dict:
        fit = arch_model(returns * 100.0, vol="Garch", p=1, q=1).fit(update_freq=0, disp="off")
        today = float(np.sqrt(fit.forecast(horizon=1).variance.to_numpy()[-1][0])) / 100.0
        omega, alpha, beta = (float(fit.params[k]) for k in ("omega", "alpha[1]", "beta[1]"))
        persistence = alpha + beta
        long_run = float(np.sqrt(omega / (1 - persistence))) / 100.0 if persistence < 1 else None
        realized = pd.Series(returns[-REGIME_DAYS:]).rolling(20).std().dropna().to_numpy()
        percentile = float((realized < today).mean() * 100) if len(realized) else None
        if percentile is None:
            level = None
        elif percentile >= 75:
            level = "alta"
        elif percentile <= 25:
            level = "baja"
        else:
            level = "normal"
        return {
            "daily_pct": round(today * 100, 4),
            "long_run_daily_pct": round(long_run * 100, 4) if long_run is not None else None,
            "percentile": round(percentile, 1) if percentile is not None else None,
            "level": level,
            "persistence": round(persistence, 4),
        }

    def _backtest(self, df: pd.DataFrame, returns: np.ndarray, horizon: int) -> dict | None:
        n = len(returns)
        windows = []
        for k in range(BACKTEST_WINDOWS, 0, -1):
            cut = n - k * horizon  # se ajusta con returns[:cut], se mide returns[cut:cut+h]
            if cut < MIN_ROWS + 20:
                continue
            past, future = returns[:cut], returns[cut : cut + horizon]
            fit = arch_model(past * 100.0, vol="Garch", p=1, q=1).fit(update_freq=0, disp="off")
            variance = fit.forecast(horizon=horizon).variance.to_numpy()[-1]
            sigma = float(np.sqrt(variance.sum())) / 100
            naive = float(past[-20:].std() * np.sqrt(horizon))
            windows.append(
                {
                    "as_of": df.index[cut].strftime("%Y-%m-%d"),
                    "forecast_pct": round(sigma * 100, 3),
                    "naive_pct": round(naive * 100, 3),
                    "realized_pct": round(float(future.std() * np.sqrt(horizon)) * 100, 3),
                    "return_pct": round(float(future.sum()) * 100, 3),
                }
            )
        if len(windows) < 6:
            return None
        w = pd.DataFrame(windows)
        inside = (w.return_pct.abs() <= w.forecast_pct).mean()
        inside95 = (w.return_pct.abs() <= 1.96 * w.forecast_pct).mean()

        def corr(col: str) -> float | None:
            c = w[col].corr(w.realized_pct)
            return round(float(c), 3) if pd.notna(c) else None

        def mae(col: str) -> float:
            return round(float((w[col] - w.realized_pct).abs().mean()), 3)

        return {
            "windows": windows,
            "horizon_days": horizon,
            "inside_68_pct": round(float(inside) * 100, 1),
            "inside_95_pct": round(float(inside95) * 100, 1),
            "corr_garch": corr("forecast_pct"),
            "corr_naive": corr("naive_pct"),
            "mae_garch_pct": mae("forecast_pct"),
            "mae_naive_pct": mae("naive_pct"),
        }
