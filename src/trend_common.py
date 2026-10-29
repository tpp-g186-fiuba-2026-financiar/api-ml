"""Post-procesamiento compartido por todos los modelos de tendencia.

Cada modelo (LSTM, XGBoost, y los que vengan) hace una unica cosa distinta:
predecir el *retorno logaritmico acumulado* de los proximos ``horizon`` dias.
A partir de ese numero, la derivacion de la senal, el RSI, el precio proyectado
y la confianza es identica para todos. Centralizarla aca garantiza que las
predicciones de modelos distintos sean directamente comparables (misma escala,
mismos umbrales, mismo formato de respuesta).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

import numpy as np
import pandas as pd

from src.errors import ApiMlError
from src.lstm import rsi

BACKTEST_DAYS = 60
# Cada cuantos dias mover el cutoff en `backtest_predict_df`. En vivo (dentro
# de un request HTTP, a diferencia del `make backtest` offline) no podemos
# pagar 60 forward-pass por modelo: en un dyno con CPU compartida (Render
# free tier) eso paso de ~0.6s por modelo en local a timeout (>40s) en
# produccion. step=5 deja 12 observaciones en vez de 60 -- mismo backtest_days
# de cobertura, menos resolucion.
BACKTEST_STEP = 5


class _DfPredictor(Protocol):
    def predict_df(self, df: pd.DataFrame) -> dict: ...


def senal(expected_return: float, neutral_band: float) -> str:
    """alza / baja / neutral segun el retorno esperado y la banda neutral."""
    if expected_return > neutral_band:
        return "alza"
    if expected_return < -neutral_band:
        return "baja"
    return "neutral"


def centrar_predicciones(
    resultados: dict[str, dict],
    neutral_band: float,
    universo: Iterable[str] | None = None,
    minimo: int = 3,
) -> dict[str, dict]:
    """Resta a cada prediccion el promedio del dia y recalcula la senal.

    Para modelos con target relativo (exceso sobre el promedio del mercado,
    ver ``TransformerTrainConfig.target``): el modelo predice cada ticker por
    separado y a veces predice que casi todos le ganan al mercado el mismo
    dia, lo que no puede pasar. Centrando el retorno log predicho (el
    promedio se toma sobre ``universo``, default todos) la senal queda como
    "mejor / peor que el resto" de verdad. ``resultados`` es
    ``{ticker: respuesta de predict_df}``; si hay menos de ``minimo`` tickers
    del universo no hay promedio representativo y se devuelve sin cambios.
    """
    universo = set(universo) if universo is not None else None
    retornos = {
        t: float(np.log(r["predicted_close"] / r["last_close"]))
        for t, r in resultados.items()
        if r.get("predicted_close") and r.get("last_close")
    }
    base = [v for t, v in retornos.items() if universo is None or t in universo]
    if len(base) < minimo:
        return resultados
    media = float(np.mean(base))
    centrados = {}
    for t, r in resultados.items():
        if t not in retornos:
            centrados[t] = r
            continue
        lr = retornos[t] - media
        centrados[t] = {
            **r,
            "predicted_close": round(float(r["last_close"] * np.exp(lr)), 4),
            "signal": senal(float(np.expm1(lr)), neutral_band),
        }
    return centrados


def senal_por_cuantil(
    resultados: dict[str, dict], cuantil: float, minimo: int = 5
) -> dict[str, dict]:
    """Recalcula la senal por posicion en el dia en vez de por banda fija.

    Para modelos de panel, cuyo retorno predicho es relativo (sin la media del
    dia) y chico frente a la banda de +-1%: "alza" para el ``cuantil``
    superior de los retornos predichos del dia, "baja" para el inferior y
    "neutral" el resto (con 20 tickers y 0,15: 3 y 3). Con menos de ``minimo``
    tickers no hay ranking representativo y todo queda "neutral". No cambia
    ``predicted_close``.
    """
    if not 0 < cuantil < 0.5:
        raise ValueError("cuantil tiene que estar entre 0 y 0,5")
    retornos = pd.Series(
        {t: float(np.log(r["predicted_close"] / r["last_close"])) for t, r in resultados.items()},
        dtype=float,
    )
    rango = retornos.rank(pct=True)
    salida = {}
    for t, r in resultados.items():
        if len(retornos) < minimo:
            signal = "neutral"
        elif rango[t] > 1 - cuantil:
            signal = "alza"
        elif rango[t] <= cuantil:
            signal = "baja"
        else:
            signal = "neutral"
        salida[t] = {**r, "signal": signal}
    return salida


def acierto_balanceado(pred: np.ndarray, real: np.ndarray) -> float:
    """Promedio entre el acierto en las subas (de los casos con ``real > 0``,
    cuantos tienen ``pred > 0``) y en las bajas (``real <= 0`` y ``pred <= 0``).

    A diferencia del acierto comun, no se infla con el sesgo alcista: decir
    siempre "sube" (o siempre "baja") da 0,5. Si no hay casos de uno de los
    dos lados cuenta solo el otro; sin casos devuelve 0,5.
    """
    pred, real = np.asarray(pred), np.asarray(real)
    sube = real > 0
    partes = []
    if sube.any():
        partes.append(np.mean(pred[sube] > 0))
    if (~sube).any():
        partes.append(np.mean(pred[~sube] <= 0))
    return float(np.mean(partes)) if partes else 0.5


def ic_por_fecha(pred: np.ndarray, real: np.ndarray, fechas: np.ndarray) -> float:
    """Promedio, entre fechas, de la correlacion de rangos (Spearman) entre
    prediccion y retorno real de los tickers de esa fecha.

    Mide si el modelo ordena bien los tickers, independientemente del nivel
    general del mercado. Fechas con menos de 3 tickers o prediccion constante
    no cuentan; sin ninguna fecha valida devuelve 0.
    """
    df = pd.DataFrame({"fecha": fechas, "pred": pred, "real": real})
    ics = []
    for _, g in df.groupby("fecha"):
        if len(g) < 3 or g["pred"].nunique() < 2 or g["real"].nunique() < 2:
            continue
        ics.append(g["pred"].rank().corr(g["real"].rank()))
    return float(np.mean(ics)) if ics else 0.0


def derive_trend_output(
    df: pd.DataFrame,
    log_return: float,
    horizon: int,
    neutral_band: float = 0.01,
) -> dict:
    """Convierte un retorno log predicho en el dict de respuesta de tendencia.

    ``log_return`` es la salida cruda del modelo (retorno log acumulado a
    ``horizon`` dias), usada internamente para derivar ``signal`` y
    ``predicted_close``. No se expone en la respuesta (ver `expected_return`
    y `confidence` sacados del schema por decision del equipo: el retorno
    esperado se resume en `predicted_close`, y `confidence` saturaba en 1.0
    demasiado facil como para ser util).
    """
    last_close = float(df["close"].iloc[-1])
    expected_return = float(np.expm1(log_return))
    predicted_close = float(last_close * np.exp(log_return))

    signal = senal(expected_return, neutral_band)

    rsi_value = rsi(df["close"].to_numpy(dtype=np.float64))
    if rsi_value is None:
        condition = "indeterminado"
    elif rsi_value >= 70:
        condition = "sobrecompra"
    elif rsi_value <= 30:
        condition = "sobreventa"
    else:
        condition = "neutral"

    return {
        "signal": signal,
        "horizon_days": horizon,
        "predicted_close": round(predicted_close, 4),
        "last_close": round(last_close, 4),
        "rsi": round(rsi_value, 2) if rsi_value is not None else None,
        "condition": condition,
        "as_of": df.index[-1].strftime("%Y-%m-%d"),
    }


def backtest_predict_df(
    predictor: _DfPredictor,
    df: pd.DataFrame,
    horizon: int,
    backtest_days: int = BACKTEST_DAYS,
    step: int = BACKTEST_STEP,
    since: pd.Timestamp | None = None,
) -> dict | None:
    """Backtest walk-forward generico para cualquier modelo con `predict_df`.

    Reusa el modelo ya entrenado (no reentrena en cada paso, igual que
    `backtest_lstm`/`backtest_xgboost` en el repo `models`): cada `step` dias
    de los ultimos `backtest_days`, predice con el historico disponible hasta
    ese dia y compara contra lo que paso `horizon` dias despues. Devuelve
    `None` si no hay historia suficiente en vez de tirar, para que el
    comparador siga mostrando la prediccion aunque no pueda medir accuracy.
    """
    split = len(df) - backtest_days - horizon
    if split < 1:
        return None
    predicted_returns: list[float] = []
    actual_returns: list[float] = []
    series: list[dict] = []
    for cutoff in range(split, len(df) - horizon, step):
        # Fechas que el modelo ya vio al entrenarse no son una prueba: con
        # ``since`` (fecha de entrenamiento) solo se evaluan las posteriores.
        if since is not None and df.index[cutoff] <= since:
            continue
        try:
            result = predictor.predict_df(df.iloc[: cutoff + 1])
        except ApiMlError:
            continue
        predicted_close = result.get("predicted_close")
        if predicted_close is None:
            continue
        last_close = float(df["close"].iloc[cutoff])
        future_close = float(df["close"].iloc[cutoff + horizon])
        predicted_returns.append(float(np.log(predicted_close / last_close)))
        actual_returns.append(float(np.log(future_close / last_close)))
        series.append(
            {
                "date": df.index[cutoff + horizon].strftime("%Y-%m-%d"),
                "predicted": predicted_close,
                "actual": future_close,
            }
        )
    if not actual_returns:
        return None
    predicted_arr = np.asarray(predicted_returns)
    actual_arr = np.asarray(actual_returns)
    # Los modelos que se abstienen (neutral = precio proyectado igual al de
    # hoy) no apostaron a ningun lado: esos casos no cuentan como error. El
    # acierto se mide solo donde el modelo opino, y se informa en cuantos.
    opined = predicted_arr != 0
    # Retorno de "seguir la senal al pie de la letra" (long si predijo alza,
    # short si predijo baja, afuera en neutral) vs. comprar-y-mantener lisa y
    # llanamente -- misma cuenta que `summarize()` en `src/backtest.py`, pero
    # sobre el mismo backtest en vivo que ya corre `compare()`, para que se
    # vea junto al resto de las metricas sin pagar un segundo backtest.
    position = np.sign(predicted_arr)
    strategy_logrets = position * actual_arr
    hits = np.sign(predicted_arr[opined]) == np.sign(actual_arr[opined])
    return {
        "directional_accuracy": float(np.mean(hits)) if opined.any() else None,
        "mae": float(np.mean(np.abs(predicted_arr - actual_arr))),
        "observations": int(opined.sum()),
        "cases": len(actual_returns),
        "series": series[-30:],
        "avg_strategy_return_pct": float(np.expm1(np.mean(strategy_logrets)) * 100),
        "avg_buy_hold_return_pct": float(np.expm1(np.mean(actual_arr)) * 100),
    }


# Un lado de la senal (alza o baja) solo se emite si, fuera de muestra, le
# gano a su tasa base por al menos este margen: +3 puntos sobre 20.000 casos
# solapados no alcanzan para distinguirlo del azar.
MIN_SIGNAL_EDGE = 0.05


def weak_signal_sides(metrics: dict) -> tuple[bool, bool]:
    """(alza_debil, baja_debil) segun las metricas fuera de muestra del artefacto."""
    up = metrics.get("oos_up_rate")
    if up is None and metrics.get("oos_down_rate") is not None:
        up = 1.0 - float(metrics["oos_down_rate"])
    if up is None:
        return False, False
    alza, baja = metrics.get("oos_alza_hit_rate"), metrics.get("oos_baja_hit_rate")
    alza_weak = alza is not None and float(alza) - up < MIN_SIGNAL_EDGE
    baja_weak = baja is not None and float(baja) - (1.0 - up) < MIN_SIGNAL_EDGE
    return alza_weak, baja_weak


def oos_signal_backtest(metrics: dict, horizon: int) -> dict | None:
    """Acierto de las senales medido fuera de muestra al entrenar (todas las
    acciones juntas). Las observaciones diarias a ``horizon`` ruedas se
    solapan: se informa una cantidad efectiva de casos (n / horizon)."""
    alza_weak, baja_weak = weak_signal_sides(metrics)
    hits = 0.0
    n = 0.0
    for side, weak in (("alza", alza_weak), ("baja", baja_weak)):
        rate = metrics.get(f"oos_{side}_hit_rate")
        count = metrics.get(f"oos_{side}_samples")
        if weak or rate is None or not count or metrics.get(f"{side}_disabled"):
            continue
        hits += float(rate) * float(count)
        n += float(count)
    if n == 0:
        return None
    effective = max(1, int(round(n / max(1, horizon))))
    return {
        "directional_accuracy": hits / n,
        "observations": effective,
        "scope": "oos_global",
    }


def fit_probability_calibration(p_up: np.ndarray, up: np.ndarray) -> dict:
    """Curva que lleva la P(sube) del modelo a "que tan seguido subio de verdad"
    con esa P(sube), medida fuera de muestra (regresion isotonica).

    El promedio de arboles exagera: medido en 2016-2026, cuando el modelo
    decia 30% subio el 58% de las veces y cuando decia 85%, el 62%. La curva
    se guarda como puntos (x, y) para interpolar sin sklearn al predecir.
    """
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99)
    iso.fit(np.asarray(p_up, dtype=float), np.asarray(up, dtype=float))
    return {
        "x": [round(float(v), 6) for v in iso.X_thresholds_],
        "y": [round(float(v), 6) for v in iso.y_thresholds_],
        "base_up": round(float(np.mean(up)), 4),
    }


def calibrated_probability(calibration: dict | None, p_up: float) -> float | None:
    """P(sube) calibrada con la curva de ``fit_probability_calibration``; None
    si el artefacto es anterior a la calibracion."""
    if not calibration or not calibration.get("x"):
        return None
    return round(float(np.interp(p_up, calibration["x"], calibration["y"])), 4)
