import numpy as np
import pandas as pd
from numpy.linalg import inv

from src.black_litterman.cartera_ancla import CarteraAncla
from src.black_litterman.matriz_de_covarianza import MatrizDeCovarianza
from src.config import settings
from src.data import fetch_available_tickers, fetch_history
from src.schemas import PerfilRiesgo, TrendResponse, Usuario


TAU = 0.05
ALFA_CONSERVADOR = 0.75
ALFA_MODERADO = 0.5
ALFA_ARRIESGADO = 0.25


def entry(usuario, predict_function):
    tickers, precios_historicos, precios_actuales = get_data()
    cartera_ancla = CarteraAncla(usuario, tickers, precios_actuales)
    matriz_de_covarianza = MatrizDeCovarianza(tickers, precios_historicos)
    q, omega = get_predicciones(tickers, predict_function)
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
        return np.dot(inv(np.dot(self.perfil_de_riesgo, self.bl_sigma())), self.bl_mu())


def get_data():
    tickers = fetch_available_tickers()
    historiales = {ticker: fetch_history(ticker, settings.history_days) for ticker in tickers}
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
    q = []
    omega = []
    for ticker in tickers:
        q.append(r[ticker].prediccion)

    for i in range(len(tickers)):
        row = [0] * len(tickers)
        row[i] = 1
        omega.append(row)
    return q, omega


class Prediccion:
    def __init__(self, t: TrendResponse):
        self.prediccion = (t.predicted_close / t.last_close) - 1.0
        self.ticker = t.symbol
        self.confianza = 1