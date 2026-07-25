import numpy as np
import pandas as pd


class MatrizDeCovarianza:
    """Matriz de covarianza muestral tradicional entre varios tickers.

    Recibe los historicos YA DESCARGADOS como DataFrames OHLCV (la forma real
    que devuelve fetch_history: columnas open/high/low/close/volume, indice
    de fechas ascendente) -- no hace fetch ella misma. Alinea por
    INTERSECCION de fechas: si un dia no esta en TODOS los tickers, se
    descarta para todos (no se rellena).
    """

    def __init__(self, tickers: list[str], historiales: dict[str, pd.DataFrame]):
        self.tickers = tickers
        self.matriz = self._construir_matriz(tickers, historiales)

    def _construir_retornos(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Retornos porcentuales dia a dia de un ticker, a partir de su
        DataFrame OHLCV.

        Devuelve ``(dates, returns)``: ``dates[i]`` (datetime64) es la fecha
        a la que corresponde ``returns[i]`` (close[i] vs close[i-1]).
        """
        df = df.sort_index()
        close = df["close"].to_numpy(dtype=np.float64)
        returns = np.diff(close) / close[:-1]
        dates = df.index.values[1:]
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
            fechas_comunes = (
                fechas_set if fechas_comunes is None else (fechas_comunes & fechas_set)
            )

        if not fechas_comunes:
            raise ValueError("no hay fechas en comun entre los tickers disponibles")

        fechas_comunes = np.array(sorted(fechas_comunes))

        retornos_alineados = np.empty((len(tickers), len(fechas_comunes)), dtype=np.float64)
        for i, ticker in enumerate(tickers):
            dates, returns = retornos_por_ticker[ticker]
            idx = np.searchsorted(dates, fechas_comunes)
            retornos_alineados[i] = returns[idx]

        return np.cov(retornos_alineados, rowvar=True)

    def __repr__(self):
        return f"MatrizDeCovarianza(tickers={self.tickers}, shape={self.matriz.shape})"