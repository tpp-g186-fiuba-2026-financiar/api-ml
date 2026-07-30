from __future__ import annotations

from enum import StrEnum

import numpy as np

from src.schemas import Usuario


class TipoCarteraAncla(StrEnum):
    """Estrategia para construir la cartera ancla (w en Pi = delta * Sigma * w)."""

    PROPIA = "propia"
    EQUAL_WEIGHT = "equal_weight"
    MERCADO = "mercado"  # TODO: pesos por capitalizacion bursatil / free float del Merval


class CarteraAncla:
    def __init__(
        self,
        usuario: Usuario,
        tickers: list[str],
        precios: dict[str, float],
        tipo: TipoCarteraAncla | None = None,
    ):
        self.tickers = tickers
        self.tipo = tipo or self._elegir_tipo_por_defecto(usuario, tickers, precios)
        self.vector = self._construir_vector(usuario, tickers, precios)

    def _elegir_tipo_por_defecto(
        self, usuario: Usuario, tickers: list[str], precios: dict[str, float]
    ) -> TipoCarteraAncla:
        """Si no se pide un tipo explicito, usa la cartera propia del
        usuario cuando tiene valor positivo, y cae a equal-weight cuando
        no (sin cartera, o cartera con valor total <= 0). No cae a MERCADO
        automaticamente porque todavia no esta implementada.
        """
        cantidades = usuario.cantidades_por_ticker()
        valor_total = sum(cantidades.get(ticker, 0.0) * precios[ticker] for ticker in tickers)
        return TipoCarteraAncla.PROPIA if valor_total > 0 else TipoCarteraAncla.EQUAL_WEIGHT

    def _construir_vector(
        self, usuario: Usuario, tickers: list[str], precios: dict[str, float]
    ) -> np.ndarray:
        match self.tipo:
            case TipoCarteraAncla.PROPIA:
                return self._vector_propia(usuario, tickers, precios)
            case TipoCarteraAncla.EQUAL_WEIGHT:
                return self._vector_equal_weight(tickers)
            case TipoCarteraAncla.MERCADO:
                return self._vector_mercado(tickers)
            case _:
                raise ValueError(f"tipo de cartera ancla desconocido: {self.tipo}")

    def _vector_propia(
        self, usuario: Usuario, tickers: list[str], precios: dict[str, float]
    ) -> np.ndarray:
        cantidades = usuario.cantidades_por_ticker()
        valores = np.array(
            [cantidades.get(ticker, 0.0) * precios[ticker] for ticker in tickers],
            dtype=np.float64,
        )
        total = valores.sum()
        if total <= 0:
            raise ValueError(
                "la cartera propia del usuario no tiene valor positivo; "
                "usar TipoCarteraAncla.EQUAL_WEIGHT o MERCADO en su lugar"
            )
        return valores / total

    def _vector_equal_weight(self, tickers: list[str]) -> np.ndarray:
        n = len(tickers)
        return np.full(n, 1.0 / n, dtype=np.float64)

    def _vector_mercado(self, tickers: list[str]) -> np.ndarray:
        """Pendiente: pesos por capitalizacion bursatil / free float de
        cada ticker del Merval. Requiere integrar una fuente de datos de
        cap. bursatil que hoy no esta disponible en el servicio.
        """
        raise NotImplementedError(
            "TipoCarteraAncla.MERCADO todavia no esta implementada "
            "(falta fuente de datos de capitalizacion bursatil / free float)"
        )

    def __repr__(self):
        return f"CarteraAncla(tipo={self.tipo}, tickers={self.tickers}, vector={self.vector})"
