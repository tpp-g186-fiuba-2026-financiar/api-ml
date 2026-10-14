import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


class MatrizDeCovarianza:
    """Covarianza de retornos simples al horizonte de las vistas.

    Se estima sobre retornos diarios con shrinkage de Ledoit-Wolf (la
    muestral cruda queda mal condicionada con ~20 acciones muy
    correlacionadas entre si, y Black-Litterman la invierte) y se escala por
    ``horizonte`` (retornos diarios ~independientes => la varianza a h
    ruedas es h veces la diaria). Asi Sigma queda en la misma escala que Q.
    """

    def __init__(
        self,
        tickers: list[str],
        historiales: dict[str, pd.DataFrame],
        horizonte: int = 1,
    ):
        self.tickers = tickers
        self.horizonte = horizonte
        self.matriz = self._construir_matriz(tickers, historiales) * horizonte

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

        if not fechas_comunes or len(fechas_comunes) < 2:
            raise ValueError("no hay fechas en comun entre los tickers disponibles")

        fechas_comunes = np.array(sorted(fechas_comunes))

        retornos_alineados = np.empty((len(fechas_comunes), len(tickers)), dtype=np.float64)
        for i, ticker in enumerate(tickers):
            dates, returns = retornos_por_ticker[ticker]
            idx = np.searchsorted(dates, fechas_comunes)
            retornos_alineados[:, i] = returns[idx]

        return LedoitWolf().fit(retornos_alineados).covariance_

    def matriz_garch(self, historiales: dict[str, pd.DataFrame], garch_model) -> np.ndarray:
        """Mantiene la correlacion estimada y reemplaza las varianzas por el
        forecast GARCH acumulado al mismo horizonte de la matriz.
        """
        desvios = np.sqrt(np.diag(self.matriz))
        correlacion = self.matriz / np.outer(desvios, desvios)

        varianzas_garch = np.array(
            [
                garch_model.forecast_variance_horizonte(historiales[ticker], self.horizonte)
                for ticker in self.tickers
            ]
        )
        desvios_garch = np.sqrt(varianzas_garch)

        return correlacion * np.outer(desvios_garch, desvios_garch)

    def __repr__(self):
        return (
            f"MatrizDeCovarianza(tickers={self.tickers}, horizonte={self.horizonte}, "
            f"shape={self.matriz.shape})"
        )
