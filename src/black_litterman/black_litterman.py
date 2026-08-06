from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.linalg import inv

from src.black_litterman.cartera_ancla import CarteraAncla, TipoCarteraAncla
from src.black_litterman.matriz_de_covarianza import MatrizDeCovarianza
from src.black_litterman.optimizador import optimizar_pesos
from src.config import settings
from src.data import fetch_available_tickers, fetch_history
from src.schemas import PerfilRiesgo, TrendResponse, Usuario

TAU = 0.05
ALFA_CONSERVADOR = 0.75
ALFA_MODERADO = 0.5
ALFA_ARRIESGADO = 0.25
TICKERS_NO_INVERTIBLES: frozenset[str] = frozenset({"GOLD", "OIL"})


def entry(
    usuario: Usuario,
    predict_function,
    garch_model=None,
    tipo_cartera_ancla: TipoCarteraAncla | None = None,
    fetch_tickers=fetch_available_tickers,
    fetch_hist=fetch_history,
):
    tickers, precios_historicos, precios_actuales = get_data(fetch_tickers, fetch_hist)
    cartera_ancla = CarteraAncla(usuario, tickers, precios_actuales, tipo=tipo_cartera_ancla)
    matriz_de_covarianza = MatrizDeCovarianza(tickers, precios_historicos)
    if garch_model is not None:
        # Reemplaza la diagonal (varianzas) por el forecast GARCH a un dia,
        # manteniendo la correlacion muestral. Afecta por igual a Pi,
        # Omega y bl_sigma porque los tres se calculan a partir de esta
        # misma matriz.
        matriz_de_covarianza.matriz = matriz_de_covarianza.matriz_garch(
            precios_historicos, garch_model
        )
    q = get_predicciones(tickers, predict_function)
    omega = construir_omega_tradicional(matriz_de_covarianza.matriz, TAU)
    bl = BlackLittermanPrediction(usuario, matriz_de_covarianza, cartera_ancla, q, omega)
    return tickers, bl.predecir()


class BlackLittermanPrediction:
    def __init__(self, usuario, matriz_de_covarianza, cartera_ancla, q, omega):
        self.usuario: Usuario = usuario
        self.perfil_de_riesgo: float = self.obtener_alfa()
        self.matriz_de_covarianza: MatrizDeCovarianza = matriz_de_covarianza
        self.cartera_ancla: CarteraAncla = cartera_ancla
        self.q: list[float] = q
        self.omega: list[list[float]] = omega
        self.pi: np.ndarray = self.get_pi()
        self.tau = TAU
        self.inv_tau_sigma = None

    def get_pi(self):
        x = np.dot(self.matriz_de_covarianza.matriz, self.cartera_ancla.vector)
        return np.dot(self.perfil_de_riesgo, x)

    def obtener_alfa(self):
        match self.usuario.perfil_riesgo:
            case PerfilRiesgo.CONSERVADOR:
                return ALFA_CONSERVADOR
            case PerfilRiesgo.MODERADO:
                return ALFA_MODERADO
            case PerfilRiesgo.ARRIESGADO:
                return ALFA_ARRIESGADO

    def bl_mu(self):
        left_term = inv(self._inv_tau_sigma() + inv(self.omega))
        right_term = np.dot(self._inv_tau_sigma(), self.pi) + np.dot(inv(self.omega), self.q)
        return np.dot(left_term, right_term)

    def _inv_tau_sigma(self):
        if self.inv_tau_sigma is None:
            self.inv_tau_sigma = inv(np.dot(self.tau, self.matriz_de_covarianza.matriz))
        return self.inv_tau_sigma

    def bl_sigma(self):
        return self.matriz_de_covarianza.matriz + inv(self._inv_tau_sigma() + inv(self.omega))

    def predecir(self):
        return optimizar_pesos(self.bl_mu(), self.bl_sigma(), self.perfil_de_riesgo)


def get_data(
    fetch_tickers=fetch_available_tickers,
    fetch_hist=fetch_history,
    excluir: frozenset[str] = TICKERS_NO_INVERTIBLES,
):
    """Descarga tickers + historicos y arma los precios actuales.

    fetch_tickers/fetch_hist son inyectables (default = las funciones reales
    de src.data) para poder testear sin mockear el modulo -- se les pasa un
    fake directo.
    """
    tickers = [t for t in fetch_tickers() if t not in excluir]
    historiales = {ticker: fetch_hist(ticker, settings.history_days) for ticker in tickers}
    precios_actuales = obtener_precios_actuales(historiales)
    return tickers, historiales, precios_actuales


def obtener_precios_actuales(historiales: dict[str, pd.DataFrame]) -> dict[str, float]:
    """A partir de {ticker: DataFrame OHLCV}, devuelve el precio de cierre
    mas reciente (columna 'close', ultima fila por fecha) de cada ticker.
    """
    precios_actuales = {}
    for ticker, df in historiales.items():
        if df.empty:
            raise ValueError(f"{ticker} no tiene ningun dato")
        precios_actuales[ticker] = float(df.sort_index()["close"].iloc[-1])
    return precios_actuales


def get_predicciones(tickers, predict_functions):
    r = {}
    for ticker in tickers:
        prediction: TrendResponse = predict_functions(ticker)
        r[ticker] = Prediccion(prediction)
    q = [r[ticker].prediccion for ticker in tickers]
    return q


def construir_omega_tradicional(sigma: np.ndarray, tau: float) -> np.ndarray:
    """Omega clasico de Black-Litterman para vistas absolutas (P = identidad):

        Omega_ii = tau * Sigma_ii

    La incertidumbre de cada vista es proporcional a la propia varianza de
    ese activo en Sigma, escalada por el mismo tau que ya se usa en Pi y en
    la mezcla bayesiana -- mismo orden de magnitud por construccion, a
    diferencia de un placeholder generico como la identidad. Todavia NO usa
    confidence del modelo; es el default estandar de la literatura antes de
    calibrar con eso.
    """
    varianzas = np.diag(sigma)  # Sigma_ii de cada ticker, en el mismo orden
    return np.diag(tau * varianzas)


class Prediccion:
    def __init__(self, t: TrendResponse):
        self.prediccion = (t.predicted_close / t.last_close) - 1.0
        self.ticker = t.symbol
        self.confianza = 1