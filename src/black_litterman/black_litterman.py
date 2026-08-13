from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.linalg import inv

from src.black_litterman.cartera_ancla import CarteraAncla, TipoCarteraAncla
from src.black_litterman.matriz_de_covarianza import MatrizDeCovarianza
from src.black_litterman.optimizador import optimizar_pesos
from src.config import settings
from src.data import fetch_history
from src.schemas import PerfilRiesgo, TrendResponse, Usuario

TAU = 0.05
ALFA_CONSERVADOR = 0.75
ALFA_MODERADO = 0.5
ALFA_ARRIESGADO = 0.25

# ALFA_* (0.25-0.75) funciona bien como delta de mercado en get_pi(), donde
# Pi = delta * Sigma @ w_ancla -- ahi Pi queda en la misma escala que mu
# porque los dos lados de esa cuenta usan Sigma. Pero optimizar_pesos()
# resuelve mu.w - (delta/2) * w'.Sigma.w, y ahi mu (retornos esperados,
# ~1e-2) y Sigma (varianzas de retornos diarios, ~1e-4) NO estan en la
# misma escala relativa: con delta=0.25-0.75 el termino de riesgo queda
# ordenes de magnitud mas chico que el de retorno, y el optimizador termina
# concentrando todo en el ticker de mayor mu (solucion de esquina), no
# porque el solver falle sino porque el objetivo, tal como queda planteado,
# casi no penaliza el riesgo.
#
# ESCALA_DELTA_OPTIMIZADOR reescala ALFA_* para que el termino de riesgo
# vuelva a competir. OJO: el valor correcto depende de la correlacion entre
# tickers, no solo de mu/Sigma_ii -- con tickers muy correlacionados (varios
# bancos del panel lider) probado empiricamente con datos sinteticos
# representativos, hicieron falta valores de delta del orden de 1e3-1e4
# para dejar de dar soluciones de esquina, no ~200 como una cuenta ingenua
# de mu_tipico/Sigma_ii_tipico sugeriria. 1000 es un punto de partida
# razonable, pero HAY QUE recalibrar corriendo optimizar_pesos con mu/Sigma
# reales de una corrida (ver script de calibracion) antes de confiar en
# este numero solo -- por eso PESO_MAXIMO_POR_TICKER es la salvaguarda que
# de verdad garantiza diversificacion, independientemente de que este bien
# calibrado o no.
ESCALA_DELTA_OPTIMIZADOR = 1000.0

# Techo de concentracion por ticker en la cartera final: ningun activo
# puede pesar mas que esto, mas alla de que delta este bien calibrado.
# Esta es la salvaguarda que efectivamente evita la solucion de esquina
# aunque ESCALA_DELTA_OPTIMIZADOR este mal calibrada -- ver comentario de
# arriba.
PESO_MAXIMO_POR_TICKER = 0.20

# Minimo de ruedas de historial para que un ticker entre a la corrida.
# Mismo valor que MIN_ROWS en garch_volatility.py: un ticker con menos
# ruedas que esto no llega a ajustar GARCH (NotEnoughDataError) y ademas
# distorsiona las fechas comunes de MatrizDeCovarianza para TODOS los
# demas tickers. Se filtra aca, antes de que nada lo vea, en vez de dejar
# que explote mas abajo.
MIN_RUEDAS = 60

# Universo de Black-Litterman: panel lider del Merval, nada mas. Reemplaza
# al blocklist TICKERS_NO_INVERTIBLES (que solo tapaba GOLD/OIL) por un
# allowlist fijo, porque el problema no es solo GOLD/OIL: fetch_available_tickers()
# trae TODO lo que cachea data-colector (lider panel + CEDEARs + commodities
# + ETFs, sin distinguir mercado -- ver docstring de esa funcion en
# src/data.py), y cualquier CEDEAR recien listado con poca historia (ej.
# YPFDD, 17 ruedas) tira abajo el forecast GARCH de TODA la corrida
# (NotEnoughDataError), no solo la de ese ticker. Ir agregando nombres a un
# blocklist a mano cada vez que aparece uno asi no escala; con un allowlist
# fijo el universo queda controlado de una. Si el panel lider cambia,
# actualizar esta lista a mano.
PANEL_LIDER_TICKERS: list[str] = [
    "ALUA",
    "BBAR",
    "BMA",
    "BYMA",
    "CEPU",
    "COME",
    "CRES",
    "ECOG",
    "EDN",
    "GGAL",
    "LOMA",
    "METR",
    "PAMP",
    "SUPV",
    "TGNO4",
    "TGSU2",
    "TRAN",
    "TXAR",
    "VALO",
    "YPFD",
]


def entry(
    usuario: Usuario,
    predict_function,
    garch_model=None,
    tipo_cartera_ancla: TipoCarteraAncla | None = None,
    tickers: list[str] = PANEL_LIDER_TICKERS,
    fetch_hist=fetch_history,
):
    tickers, precios_historicos, precios_actuales = get_data(tickers, fetch_hist)
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
    print(f"Predicciones: {q}")
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
        delta_optimizador = self.perfil_de_riesgo * ESCALA_DELTA_OPTIMIZADOR
        n = len(self.q)
        # Si PESO_MAXIMO_POR_TICKER * n < 1 (universo chico, ej. tras
        # filtrar tickers con pocas ruedas en get_data), el cap fijo es
        # inviable. OJO: relajar a exactamente 1/n (como hacia antes) NO
        # sirve -- con cap*n == 1 la unica solucion factible es w = [1/n]*n,
        # es decir fuerza pesos iguales a la fuerza sin importar mu/Sigma,
        # dejando al optimizador sin ningun grado de libertad (regresion
        # detectada por test_con_garch_model_no_explota_y_cambia_el_resultado,
        # que esperaba que GARCH pudiera diferenciar pesos incluso con
        # pocos tickers). Se usa 2/n en cambio: deja cap*n == 2, con margen
        # real para que el optimizador siga pudiendo concentrar mas en el
        # activo mas favorecido, solo que sin llegar a una esquina absoluta.
        peso_maximo_efectivo = max(PESO_MAXIMO_POR_TICKER, 2.0 / n)
        return optimizar_pesos(
            self.bl_mu(), self.bl_sigma(), delta_optimizador, peso_maximo=peso_maximo_efectivo
        )


def get_data(
    tickers: list[str] = PANEL_LIDER_TICKERS,
    fetch_hist=fetch_history,
    min_ruedas: int = MIN_RUEDAS,
):
    """Descarga los historicos del panel lider Merval (lista curada, ver
    PANEL_LIDER_TICKERS mas arriba) y arma los precios actuales.

    Descarta (sin tirar excepcion) cualquier ticker con menos de
    min_ruedas de historial: no llega a ajustar GARCH y ademas achica
    fechas_comunes en MatrizDeCovarianza para el resto de los tickers.
    Se corta aca, antes de que nada mas lo vea.

    tickers/fetch_hist son inyectables (default = la lista real / la
    funcion real de src.data) para poder testear sin mockear el modulo --
    se les pasa un fake directo.

    Devuelve (tickers_validos, historiales, precios_actuales, tickers_excluidos),
    donde tickers_excluidos es un dict {ticker: motivo} para poder
    reportarlo en el endpoint.
    """
    historiales = {}
    tickers_excluidos: dict[str, str] = {}
    for ticker in tickers:
        df = fetch_hist(ticker, settings.history_days)
        # Log de cada historial tal como llega en ESTA corrida real (no un
        # diagnostico aparte): si algo explota mas abajo por pocas ruedas,
        # esto deja la foto exacta -- ticker, cantidad y rango de fechas --
        # en el mismo log del pedido que fallo.
        rango = f"{df.index[0].date()} -> {df.index[-1].date()}" if not df.empty else "vacio"
        print(f"  [historial] {ticker}: {len(df)} ruedas ({rango})")
        if len(df) < min_ruedas:
            motivo = f"{len(df)} ruedas, minimo {min_ruedas}"
            print(f"  [historial] {ticker}: excluido ({motivo})")
            tickers_excluidos[ticker] = motivo
            continue
        historiales[ticker] = df

    if not historiales:
        raise ValueError("ningun ticker llego al minimo de ruedas requerido")

    tickers_validos = [t for t in tickers if t not in tickers_excluidos]
    precios_actuales = obtener_precios_actuales(historiales)
    return tickers_validos, historiales, precios_actuales


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
