from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

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

# Unidades: TODO en Black-Litterman esta en retornos DIARIOS. Sigma sale de
# retornos diarios (y GARCH da la varianza a un dia), Pi = delta * Sigma @ w
# hereda esa escala y Omega = tau * Sigma_ii tambien. Por eso Q tiene que ser
# diario: cada modelo predice el retorno a su propio horizonte (5 ruedas LSTM,
# 20 XGBoost/Contexto), y se pasa a diario como log(pc/lc) / horizon_days
# (ver Prediccion). Antes Q era pc/lc - 1 al horizonte, 5-20 veces mas grande
# que Pi y Omega: las vistas pesaban de mas y el optimizador caia en esquinas.
#
# ALFA_* (0.25-0.75) es el delta de mercado en get_pi(). En optimizar_pesos()
# (max mu.w - (delta/2) * w'.Sigma.w) un delta de ese orden igual deja el
# termino de riesgo muy chico frente al de retorno: las vistas de los modelos
# (+-3-8% en 5-20 ruedas, o sea ~0.1-1.5% diario) son mucho mas fuertes que
# cualquier prima de equilibrio, y con delta < 10 la cartera se va a 1-2
# tickers. ESCALA_DELTA_OPTIMIZADOR reescala ALFA_* solo para el optimizador.
#
# Calibracion (2026-10-20, ya con Q diario): universo sintetico tipo panel
# lider (20 tickers, vol diaria 2-3.5%, correlacion 0.45 y 0.75 entre bancos),
# 30 escenarios de vistas (60% neutrales, resto +-3-8% a 5 o 20 ruedas), sin
# techo por ticker:
#
#   escala | riesgo/retorno en el optimo     | corr(pesos, vistas)
#          | arriesgado     conservador      | arriesgado  conservador
#   10     | 0.23           0.63             | 0.60        0.62
#   50     | 0.99           3.6              | 0.58        0.39
#   200    | 6.2            15.6             | 0.32        0.14
#   1000   | 26             84               | 0.10        0.05
#
# Con 50 los dos terminos del objetivo quedan del mismo orden (1x-4x segun
# perfil): los pesos siguen a las vistas pero el riesgo diversifica, y los
# perfiles se diferencian. El 1000 anterior compensaba el Q inflado por el
# horizonte (equivalia a ~50-200 en escala diaria); con Q diario ahogaria las
# vistas. PESO_MAXIMO_POR_TICKER sigue siendo la salvaguarda contra esquinas.
ESCALA_DELTA_OPTIMIZADOR = 50.0

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

# Hilos para descargar historicos y predecir en paralelo (I/O dominante).
MAX_WORKERS_DESCARGA = 6

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


@dataclass
class ResultadoBL:
    """Resultado de una corrida de Black-Litterman, con los intermedios que se
    muestran en el endpoint. pi/q/omega/mu_bl estan en retornos diarios (log);
    omega es la diagonal (las vistas son independientes)."""

    tickers: list[str]
    pesos: np.ndarray
    peso_actual: np.ndarray  # participacion real del usuario por valor (0 si no tiene)
    pi: np.ndarray
    q: np.ndarray
    omega: np.ndarray
    mu_bl: np.ndarray
    horizon_days: list[int]
    tickers_excluidos: dict[str, str] = field(default_factory=dict)
    cartera_ancla: str = TipoCarteraAncla.EQUAL_WEIGHT.value

    def detalle(self) -> list[dict]:
        return [
            {
                "ticker": ticker,
                "peso_actual": float(self.peso_actual[i]),
                "peso_recomendado": float(self.pesos[i]),
                "pi": float(self.pi[i]),
                "q": float(self.q[i]),
                "omega": float(self.omega[i]),
                "mu_bl": float(self.mu_bl[i]),
                "horizon_days": int(self.horizon_days[i]),
            }
            for i, ticker in enumerate(self.tickers)
        ]


def entry(
    usuario: Usuario,
    predict_function,
    garch_model=None,
    tipo_cartera_ancla: TipoCarteraAncla | None = None,
    tickers: list[str] = PANEL_LIDER_TICKERS,
    fetch_hist=fetch_history,
) -> ResultadoBL:
    tickers, precios_historicos, precios_actuales, tickers_excluidos = get_data(tickers, fetch_hist)
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
    predicciones = get_predicciones(tickers, predict_function)
    q = [p.prediccion for p in predicciones]
    print(f"Predicciones (retorno diario): {q}")
    omega = construir_omega_tradicional(matriz_de_covarianza.matriz, TAU)
    bl = BlackLittermanPrediction(usuario, matriz_de_covarianza, cartera_ancla, q, omega)
    pesos = bl.predecir()
    return ResultadoBL(
        tickers=tickers,
        pesos=pesos,
        peso_actual=pesos_actuales_por_valor(usuario, tickers, precios_actuales),
        pi=np.asarray(bl.pi, dtype=np.float64),
        q=np.asarray(q, dtype=np.float64),
        omega=np.diag(omega).astype(np.float64),
        mu_bl=bl.bl_mu(),
        horizon_days=[p.horizon_days for p in predicciones],
        tickers_excluidos=tickers_excluidos,
        cartera_ancla=cartera_ancla.tipo.value,
    )


def pesos_actuales_por_valor(
    usuario: Usuario, tickers: list[str], precios: dict[str, float]
) -> np.ndarray:
    """Participacion real de cada ticker en la cartera del usuario, por valor
    (cantidad * ultimo cierre), sobre el universo de la corrida. Es lo que el
    usuario tiene hoy, independientemente de la cartera ancla elegida (con
    equal_weight el ancla es 1/n pero esto sigue siendo su cartera real).
    Todo cero si no tiene tenencias con valor en el universo.
    """
    cantidades = usuario.cantidades_por_ticker()
    valores = np.array(
        [max(cantidades.get(t, 0.0), 0.0) * precios[t] for t in tickers], dtype=np.float64
    )
    total = valores.sum()
    return valores / total if total > 0 else np.zeros(len(tickers))


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
        self._bl_mu: np.ndarray | None = None

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
        # Cacheado: predecir() lo usa y entry() lo vuelve a pedir para el detalle.
        if self._bl_mu is None:
            left_term = inv(self._inv_tau_sigma() + inv(self.omega))
            right_term = np.dot(self._inv_tau_sigma(), self.pi) + np.dot(inv(self.omega), self.q)
            self._bl_mu = np.dot(left_term, right_term)
        return self._bl_mu

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
    # Las descargas son I/O puro e independientes entre si: en paralelo la
    # corrida pasa de ~20 pedidos en fila a unos pocos segundos. ``map``
    # conserva el orden de ``tickers``.
    with ThreadPoolExecutor(max_workers=MAX_WORKERS_DESCARGA) as pool:
        descargas = list(pool.map(lambda t: fetch_hist(t, settings.history_days), tickers))
    for ticker, df in zip(tickers, descargas, strict=True):
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
    return tickers_validos, historiales, precios_actuales, tickers_excluidos


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


def get_predicciones(tickers, predict_functions) -> list[Prediccion]:
    """Una Prediccion por ticker, en el mismo orden que ``tickers``."""
    # Cada prediccion baja su propio historico: se hacen en paralelo (``map``
    # conserva el orden de ``tickers``).
    with ThreadPoolExecutor(max_workers=MAX_WORKERS_DESCARGA) as pool:
        predictions = list(pool.map(predict_functions, tickers))
    return [Prediccion(prediction) for prediction in predictions]


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
    """Vista de un modelo para un ticker, como retorno log DIARIO: el modelo
    predice el cierre a ``horizon_days`` ruedas y se reparte parejo en el
    horizonte, para quedar en la misma escala que Sigma/Pi/Omega (ver el
    comentario de ESCALA_DELTA_OPTIMIZADOR).
    """

    def __init__(self, t: TrendResponse):
        self.horizon_days = max(int(t.horizon_days), 1)
        self.prediccion = float(np.log(t.predicted_close / t.last_close)) / self.horizon_days
        self.ticker = t.symbol
        self.confianza = 1
