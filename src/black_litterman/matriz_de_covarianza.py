import numpy as np
import pandas as pd


class MatrizDeCovarianza:
    def __init__(self, tickers: list[str], historiales: dict[str, pd.DataFrame]):
        self.tickers = tickers
        self.matriz = self._construir_matriz(tickers, historiales)

    def _construir_retornos(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        df = df.sort_index()
        close = df["close"].to_numpy(dtype=np.float64)
        returns = np.diff(close) / close[:-1]
        dates = df.index.normalize().values[1:]
        return dates, returns

    def _construir_matriz(
        self, tickers: list[str], historiales: dict[str, pd.DataFrame]
    ) -> np.ndarray:
        retornos_por_ticker = {
            ticker: self._construir_retornos(historiales[ticker]) for ticker in tickers
        }

        fechas_comunes = None
        for dates, _ in retornos_por_ticker.values():
            fechas_set = set(dates.tolist())
            fechas_comunes = fechas_set if fechas_comunes is None else (fechas_comunes & fechas_set)

        if not fechas_comunes:
            raise ValueError("no hay fechas en comun entre los tickers disponibles")

        fechas_comunes = np.array(sorted(fechas_comunes))

        retornos_alineados = np.empty((len(tickers), len(fechas_comunes)), dtype=np.float64)
        for i, ticker in enumerate(tickers):
            dates, returns = retornos_por_ticker[ticker]
            idx = np.searchsorted(dates, fechas_comunes)
            retornos_alineados[i] = returns[idx]

        return np.cov(retornos_alineados, rowvar=True)

    def matriz_garch(self, historiales: dict[str, pd.DataFrame], garch_model) -> np.ndarray:
        desvios_muestrales = np.sqrt(np.diag(self.matriz))
        correlacion = self.matriz / np.outer(desvios_muestrales, desvios_muestrales)

        varianzas_garch = np.array(
            [garch_model.forecast_daily_variance(historiales[ticker]) for ticker in self.tickers]
        )
        desvios_garch = np.sqrt(varianzas_garch)

        return correlacion * np.outer(desvios_garch, desvios_garch)

    def __repr__(self):
        return f"MatrizDeCovarianza(tickers={self.tickers}, shape={self.matriz.shape})"
