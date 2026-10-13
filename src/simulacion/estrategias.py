"""Como se traduce cada recomendacion del sistema en pesos de cartera.

Todas las estrategias invierten solo en el panel lider (``PANEL_LIDER_TICKERS``,
el mismo universo de Black-Litterman), sin cortos ni apalancamiento; lo que no
se asigna queda en efectivo (sin interes).
"""

from __future__ import annotations

import contextlib
import io
from collections.abc import Callable

import numpy as np
import pandas as pd

from src.black_litterman.black_litterman import PANEL_LIDER_TICKERS, entry
from src.black_litterman.cartera_ancla import TipoCarteraAncla
from src.consensus import ACCURACY_PRIOR_CASES, compute_consensus
from src.errors import ApiMlError
from src.garch_volatility import GarchVolatilityModel
from src.schemas import PerfilRiesgo, TrendResponse, Usuario
from src.trend_common import BACKTEST_DAYS, BACKTEST_STEP

PERFILES = (PerfilRiesgo.CONSERVADOR, PerfilRiesgo.MODERADO, PerfilRiesgo.ARRIESGADO)
# Clasificacion del consenso que se lee como "comprar" (ver src.consensus).
CONSENSO_COMPRA = "sobrecompra"


def equal_weight(tickers: list[str]) -> dict[str, float]:
    return {t: 1.0 / len(tickers) for t in tickers} if tickers else {}


def pesos_senal(predicciones: dict[str, dict], universo: list[str]) -> dict[str, float]:
    """Partes iguales entre los tickers con senal ``alza``; el resto en efectivo."""
    elegidos = [t for t in universo if (predicciones.get(t) or {}).get("signal") == "alza"]
    return equal_weight(elegidos)


def consenso_por_ticker(
    predicciones_por_modelo: dict[str, dict[str, dict]],
    universo: list[str],
    perfil: PerfilRiesgo,
) -> dict[str, dict]:
    """``compute_consensus`` por ticker, con las predicciones locales del dia."""
    resultado = {}
    for ticker in universo:
        por_modelo = {
            modelo: preds[ticker]
            for modelo, preds in predicciones_por_modelo.items()
            if ticker in preds
        }
        resultado[ticker] = compute_consensus(por_modelo, perfil)
    return resultado


def pesos_consenso(consenso: dict[str, dict]) -> dict[str, float]:
    """Partes iguales entre los tickers cuyo consenso da lectura de compra."""
    return equal_weight([t for t, c in consenso.items() if c["classification"] == CONSENSO_COMPRA])


def backtest_sin_neutrales(
    model,
    df: pd.DataFrame,
    horizon: int,
    backtest_days: int = BACKTEST_DAYS,
    step: int = BACKTEST_STEP,
) -> dict | None:
    """Backtest en vivo (mismos cortes que ``backtest_predict_df``) contando solo
    las predicciones con vista.

    Para el modelo macro: cuando dice "neutral" proyecta el ultimo cierre
    (retorno 0, o casi: ``predicted_close`` viene redondeado) y
    ``backtest_predict_df`` lo cuenta como acierto o error de direccion segun
    ese signo, aunque en realidad no opino. Aca esas predicciones se descartan
    por su ``signal``, como el ``signal_hit_rate`` del paper trading. ``None``
    si no hubo ninguna con vista.
    """
    split = len(df) - backtest_days - horizon
    if split < 1:
        return None
    aciertos: list[bool] = []
    errores: list[float] = []
    for cutoff in range(split, len(df) - horizon, step):
        try:
            result = model.predict_df(df.iloc[: cutoff + 1])
        except ApiMlError:
            continue
        if result.get("signal") == "neutral":
            continue
        last_close = float(df["close"].iloc[cutoff])
        predicho = float(np.log(result["predicted_close"] / last_close))
        real = float(np.log(df["close"].iloc[cutoff + horizon] / last_close))
        aciertos.append(np.sign(predicho) == np.sign(real))
        errores.append(abs(predicho - real))
    if not aciertos:
        return None
    return {
        "directional_accuracy": float(np.mean(aciertos)),
        "mae": float(np.mean(errores)),
        "observations": len(aciertos),
    }


def accuracy_ajustada(backtest: dict) -> float:
    """Accuracy acercada al 50% segun cantidad de casos: el mismo criterio que
    ``pick_best_model`` de backend-website y el consenso (``ACCURACY_PRIOR_CASES``).
    """
    n = backtest.get("observations") or 0
    accuracy = backtest["directional_accuracy"]
    return 0.5 + (accuracy - 0.5) * n / (n + ACCURACY_PRIOR_CASES) if n > 0 else accuracy


def mejor_modelo_por_ticker(
    backtests: dict[str, dict[str, dict]], candidatos: tuple[str, ...], universo: list[str]
) -> dict[str, dict]:
    """Para cada ticker, el candidato con mejor accuracy ajustada (desempata menor mae).

    ``backtests`` es modelo -> ticker -> metricas. Los tickers sin ningun
    candidato con backtest quedan afuera (en produccion se caeria al default).
    """
    elegidos = {}
    for ticker in universo:
        mejor: tuple[float, float] | None = None
        for modelo in candidatos:
            bt = backtests.get(modelo, {}).get(ticker)
            if not bt or bt.get("directional_accuracy") is None:
                continue
            score = (accuracy_ajustada(bt), -bt.get("mae", float("inf")))
            if mejor is None or score > mejor:
                mejor = score
                elegidos[ticker] = {"model": modelo, "accuracy_ajustada": score[0], **bt}
    return elegidos


def predicciones_del_mejor(
    predicciones: dict[str, dict[str, dict]], elegidos: dict[str, dict]
) -> dict[str, dict]:
    """ticker -> prediccion del modelo elegido para ese ticker."""
    return {
        t: predicciones[e["model"]][t]
        for t, e in elegidos.items()
        if t in predicciones.get(e["model"], {})
    }


class GarchCache:
    """``GarchVolatilityModel`` con cache por serie: BL lo pide una vez por
    ticker en cada corrida y hay una corrida por modelo y perfil el mismo dia.
    """

    def __init__(self) -> None:
        self._model = GarchVolatilityModel()
        self._cache: dict[tuple, float] = {}

    def forecast_daily_variance(self, df: pd.DataFrame) -> float:
        key = (len(df), df.index[-1], float(df["close"].iloc[-1]))
        if key not in self._cache:
            self._cache[key] = self._model.forecast_daily_variance(df)
        return self._cache[key]


def pesos_black_litterman(
    predicciones: dict[str, dict],
    perfil: PerfilRiesgo,
    fetch_hist: Callable[[str, int], pd.DataFrame],
    garch: GarchCache,
    universo: list[str] = PANEL_LIDER_TICKERS,
) -> dict[str, float]:
    """Pesos de ``/portfolio/recomendacion`` con las predicciones del dia.

    Cartera ancla equal-weight: la cartera "propia" depende de cada usuario.
    Los tickers sin prediccion ese dia quedan fuera de la corrida (en
    produccion haria fallar el pedido entero).
    """
    tickers = [t for t in universo if t in predicciones]
    if not tickers:
        return {}

    def predict(ticker: str) -> TrendResponse:
        return TrendResponse(**predicciones[ticker])

    usuario = Usuario(perfil_riesgo=perfil, tenencias=[])
    with contextlib.redirect_stdout(io.StringIO()):
        validos, pesos = entry(
            usuario,
            predict,
            garch,
            TipoCarteraAncla.EQUAL_WEIGHT,
            tickers=tickers,
            fetch_hist=fetch_hist,
        )
    return {t: max(float(w), 0.0) for t, w in zip(validos, pesos, strict=True)}


def normalizar(pesos: dict[str, float]) -> dict[str, float]:
    """Corrige el ruido numerico del optimizador (suma 1 + 1e-12, pesos -1e-15)."""
    pesos = {t: w for t, w in pesos.items() if w > 1e-6}
    total = sum(pesos.values())
    if total > 1:
        pesos = {t: w / total for t, w in pesos.items()}
    return pesos
