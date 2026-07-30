import numpy as np

from src.schemas import Usuario


class CarteraAncla:
    def __init__(self, usuario: Usuario, tickers: list[str], precios: dict[str, float]):
        self.tickers = tickers
        self.vector = self._construir_vector(usuario, tickers, precios)

    def _construir_vector(
        self, usuario: Usuario, tickers: list[str], precios: dict[str, float]
    ) -> np.ndarray:
        cantidades = usuario.cantidades_por_ticker()

        valores = np.array(
            [cantidades.get(ticker, 0.0) * precios[ticker] for ticker in tickers],
            dtype=np.float64,
        )

        total = valores.sum()
        return valores / total

    def __repr__(self):
        return f"CarteraAncla(tickers={self.tickers}, vector={self.vector})"
