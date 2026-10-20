from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from src.black_litterman.cartera_ancla import CarteraAncla, TipoCarteraAncla
from src.black_litterman.matriz_de_covarianza import MatrizDeCovarianza
from src.black_litterman.optimizador import optimizar_pesos
from src.config import settings
from src.data import fetch_history
from src.schemas import PerfilRiesgo, TrendResponse, Usuario

# Black-Litterman con vistas absolutas (P = identidad): una vista por ticker,
# el retorno que predice el modelo de tendencia a `horizon_days` ruedas.
# Todo (Q, Sigma, Pi, Omega, tasa libre de riesgo) se lleva a ese mismo
# horizonte para que las cuentas sean comparables.

# Incertidumbre del prior: Cov(Pi) = tau * Sigma. Valor habitual en la
# literatura (He-Litterman).
TAU = 0.05

# Aversion al riesgo del MERCADO, para el prior Pi = delta * Sigma * w_ancla
# (equilibrio implicito). No depende del usuario: es la que hace que la
# cartera ancla sea optima, y es adimensional (no cambia con el horizonte).
# 2.5 es el valor de referencia de la literatura (prima de riesgo /
# varianza del mercado).
DELTA_MERCADO = 2.5

# Aversion al riesgo del USUARIO, solo para el optimizador final. Con un
# valor cercano a DELTA_MERCADO y vistas sin confianza la recomendacion
# vuelve a la cartera ancla; mas alto lleva hacia minima varianza, mas
# bajo deja que las vistas concentren mas.
AVERSION_AL_RIESGO: dict[PerfilRiesgo, float] = {
    PerfilRiesgo.CONSERVADOR: 6.0,
    PerfilRiesgo.MODERADO: 3.0,
    PerfilRiesgo.ARRIESGADO: 1.5,
}

# Techo de concentracion por ticker en la cartera final, segun perfil.
PESO_MAXIMO_POR_PERFIL: dict[PerfilRiesgo, float] = {
    PerfilRiesgo.CONSERVADOR: 0.15,
    PerfilRiesgo.MODERADO: 0.20,
    PerfilRiesgo.ARRIESGADO: 0.30,
}

# Confianza en las vistas: fraccion (aprox.) del camino entre el prior y la
# vista que recorre el retorno posterior de cada ticker (ver
# construir_omega). Los modelos de tendencia casi no tienen poder
# predictivo sobre el retorno a 5 ruedas: accuracy direccional ~0.50-0.53
# en paper trading y correlacion prediccion/realizado ~0-0.05 en el panel
# lider. Con eso una confianza alta solo amplifica ruido. Por ticker no hay
# historia suficiente (~40 predicciones resueltas) para calibrarla aparte.
CONFIANZA_VISTAS = 0.05

RUEDAS_POR_ANIO = 252

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


def entry(
    usuario: Usuario,
    predict_function,
    garch_model=None,
    tipo_cartera_ancla: TipoCarteraAncla | None = None,
    tickers: list[str] = PANEL_LIDER_TICKERS,
    fetch_hist=fetch_history,
    tasa_libre_riesgo_anual: float = 0.0,
    confianza_vistas: float = CONFIANZA_VISTAS,
):
    """Arma la recomendacion de cartera para ``usuario``.

    ``tasa_libre_riesgo_anual`` es una TNA en decimal (0.35 = 35%): se
    descuenta de las predicciones para que Q sea retorno en exceso, igual
    que Pi.
    """
    tickers, precios_historicos, precios_actuales = get_data(tickers, fetch_hist)
    predicciones = get_predicciones(tickers, predict_function)
    horizonte = horizonte_comun(predicciones)
    tasa_horizonte = tasa_al_horizonte(tasa_libre_riesgo_anual, horizonte)
    q = [p.prediccion - tasa_horizonte for p in predicciones]
    print(f"Vistas en exceso a {horizonte} ruedas (tasa {tasa_horizonte:.4f}): {q}")

    cartera_ancla = CarteraAncla(usuario, tickers, precios_actuales, tipo=tipo_cartera_ancla)
    matriz_de_covarianza = MatrizDeCovarianza(tickers, precios_historicos, horizonte=horizonte)
    if garch_model is not None:
        # Reemplaza las varianzas por el forecast GARCH al mismo horizonte,
        # manteniendo la correlacion. Afecta por igual a Pi, Omega y
        # bl_sigma porque los tres se calculan a partir de esta matriz.
        matriz_de_covarianza.matriz = matriz_de_covarianza.matriz_garch(
            precios_historicos, garch_model
        )
    omega = construir_omega(matriz_de_covarianza.matriz, TAU, confianza_vistas)
    bl = BlackLittermanPrediction(usuario, matriz_de_covarianza, cartera_ancla, q, omega)
    return tickers, bl.predecir()


class BlackLittermanPrediction:
    def __init__(self, usuario, matriz_de_covarianza, cartera_ancla, q, omega, tau=TAU):
        self.usuario: Usuario = usuario
        self.aversion_al_riesgo: float = obtener_aversion_al_riesgo(usuario.perfil_riesgo)
        self.peso_maximo: float = PESO_MAXIMO_POR_PERFIL[usuario.perfil_riesgo]
        self.matriz_de_covarianza: MatrizDeCovarianza = matriz_de_covarianza
        self.cartera_ancla: CarteraAncla = cartera_ancla
        self.q: np.ndarray = np.asarray(q, dtype=np.float64)
        self.omega: np.ndarray = np.asarray(omega, dtype=np.float64)
        self.tau: float = tau
        self.pi: np.ndarray = self.get_pi()

    @property
    def sigma(self) -> np.ndarray:
        return self.matriz_de_covarianza.matriz

    def get_pi(self) -> np.ndarray:
        """Retornos en exceso de equilibrio: los que hacen optima a la
        cartera ancla para un inversor con aversion DELTA_MERCADO.
        """
        return DELTA_MERCADO * self.sigma @ self.cartera_ancla.vector

    def _incertidumbre_posterior(self) -> np.ndarray:
        """M = [(tau*Sigma)^-1 + Omega^-1]^-1, covarianza de la media posterior."""
        precision = np.linalg.inv(self.tau * self.sigma) + np.linalg.inv(self.omega)
        return np.linalg.inv(precision)

    def bl_mu(self) -> np.ndarray:
        """mu_BL = M [(tau*Sigma)^-1 Pi + Omega^-1 Q]."""
        derecha = np.linalg.solve(self.tau * self.sigma, self.pi) + np.linalg.solve(
            self.omega, self.q
        )
        return self._incertidumbre_posterior() @ derecha

    def bl_sigma(self) -> np.ndarray:
        """Covarianza predictiva de los retornos: Sigma + M (He-Litterman)."""
        return self.sigma + self._incertidumbre_posterior()

    def predecir(self) -> np.ndarray:
        n = len(self.q)
        # Con universo chico (tickers filtrados en get_data) el techo puede
        # ser inviable (techo * n < 1). Se relaja a 2/n y no a 1/n: con
        # techo * n == 1 la unica solucion factible es pesos iguales.
        peso_maximo = max(self.peso_maximo, 2.0 / n)
        return optimizar_pesos(
            self.bl_mu(), self.bl_sigma(), self.aversion_al_riesgo, peso_maximo=peso_maximo
        )


def obtener_aversion_al_riesgo(perfil: PerfilRiesgo) -> float:
    try:
        return AVERSION_AL_RIESGO[perfil]
    except KeyError as exc:
        raise ValueError(f"perfil de riesgo desconocido: {perfil}") from exc


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

    Devuelve (tickers_validos, historiales, precios_actuales).
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


def get_predicciones(tickers, predict_function) -> list[Prediccion]:
    # Cada prediccion baja su propio historico: se hacen en paralelo (mismo
    # orden de resultados que ``tickers``).
    with ThreadPoolExecutor(max_workers=MAX_WORKERS_DESCARGA) as pool:
        return [Prediccion(t) for t in pool.map(predict_function, tickers)]


def horizonte_comun(predicciones: list[Prediccion]) -> int:
    """Horizonte (en ruedas) de las vistas. Todas tienen que compartirlo:
    Q, Sigma y la tasa se escalan a ese horizonte.
    """
    horizontes = {p.horizonte for p in predicciones}
    if len(horizontes) != 1:
        raise ValueError(f"las predicciones tienen horizontes distintos: {sorted(horizontes)}")
    return horizontes.pop()


def tasa_al_horizonte(tasa_anual: float, horizonte: int) -> float:
    """TNA (decimal) -> tasa simple para ``horizonte`` ruedas."""
    return tasa_anual * horizonte / RUEDAS_POR_ANIO


def construir_omega(sigma: np.ndarray, tau: float, confianza: float | np.ndarray) -> np.ndarray:
    """Omega diagonal a partir de la confianza en cada vista (estilo Idzorek):

        Omega_ii = tau * Sigma_ii * (1 - c_i) / c_i

    Para una vista aislada, el retorno posterior queda en
    Pi_i + c_i * (Q_i - Pi_i): c = 0.5 equivale al Omega clasico
    tau * Sigma_ii (mezcla mitad y mitad), c -> 0 ignora la vista y c -> 1
    la toma tal cual. Con vistas sobre activos correlacionados la mezcla
    real se aparta un poco de c_i, pero el orden de magnitud se mantiene.
    ``confianza`` puede ser un valor unico o uno por ticker.
    """
    varianzas = np.diag(sigma)
    c = np.broadcast_to(np.asarray(confianza, dtype=np.float64), varianzas.shape)
    if np.any(c <= 0) or np.any(c >= 1):
        raise ValueError(f"la confianza en las vistas tiene que estar en (0, 1): {confianza}")
    return np.diag(tau * varianzas * (1 - c) / c)


class Prediccion:
    def __init__(self, t: TrendResponse):
        self.prediccion = (t.predicted_close / t.last_close) - 1.0
        self.ticker = t.symbol
        self.horizonte = t.horizon_days
