"""Gate de promocion para el reentrenamiento automatico (cron de GitHub Actions).

A diferencia de correr `make train` a mano y mirar los numeros antes de
commitear, un cron no tiene a nadie mirando: sin este chequeo, un mal dia de
entrenamiento (datos raros, mala suerte de inicializacion) pisaria el
artefacto bueno en produccion sin que nadie se entere. Mismo criterio que ya
usa el cron de Modal (`models/modelos/*_model.py::retrain_one`): solo se
promueve un candidato si supera al vigente en accuracy direccional.

La medicion usa `backtest_predict_df` (walk-forward, ultimos `BACKTEST_DAYS`
por ticker -- la misma logica que ya se muestra en la web por ticker,
ver `src.trend_common`) pooleada sobre todo el universo de entrenamiento, en
vez del `val_directional_accuracy` de un unico split que guarda cada modelo:
esa metrica ya se documento como ruidosa y no confiable para comparar
corridas (ver notas de `src.backtest`).

El ultimo score promovido se guarda en un sidecar json junto al artefacto
(`<model_path>.score.json`) para no tener que re-backtestear el modelo
vigente en cada corrida.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from src.errors import ApiMlError
from src.trend_common import BACKTEST_DAYS, backtest_predict_df, ic_por_fecha

# Metricas del gate: "accuracy" (LSTM), "ic", para modelos con target relativo
# (ver `pooled_ic`), o "walk_forward" (Transformer, ver `_promote_by_walk_forward`).
METRICAS = ("accuracy", "ic", "walk_forward")
# Piso del gate "walk_forward": acierto balanceado fuera de muestra (decir
# siempre lo mismo da 0.5) e IC por encima de 0.
MIN_ACIERTO_WALK_FORWARD = 0.5


def _score_path(model_path: str) -> Path:
    return Path(model_path).with_suffix(Path(model_path).suffix + ".score.json")


def pooled_directional_accuracy(
    model,
    histories: dict[str, pd.DataFrame],
    horizon: int,
    backtest_days: int = BACKTEST_DAYS,
) -> dict:
    """Accuracy direccional pooleada sobre todos los tickers de `histories`.

    A diferencia del backtest que se muestra en vivo en el comparador (que
    usa un `step` grande para no tardar dentro de un request HTTP), este gate
    corre offline en el cron de reentrenamiento -- sin apuro por tiempo, usa
    `step=1` (todos los dias) para una decision de promocion mas precisa.
    """
    total_observations = 0
    total_correct = 0.0
    for df in histories.values():
        result = backtest_predict_df(model, df, horizon, backtest_days, step=1)
        if result is None:
            continue
        total_observations += result["observations"]
        total_correct += (result["directional_accuracy"] or 0.0) * result["observations"]
    if total_observations == 0:
        return {"directional_accuracy": None, "observations": 0}
    return {
        "directional_accuracy": total_correct / total_observations,
        "observations": total_observations,
    }


def pooled_ic(
    model,
    histories: dict[str, pd.DataFrame],
    horizon: int,
    backtest_days: int = BACKTEST_DAYS,
) -> dict:
    """IC (correlacion de rangos por dia entre retorno predicho y real) sobre
    las ultimas `backtest_days` ruedas, todos los dias.

    Es la metrica que corresponde a un modelo con target relativo: predice si
    un ticker le gana al resto, no si sube, asi que comparar su signo con el
    del retorno absoluto (`pooled_directional_accuracy`) no mide lo que
    predice. Se evalua sobre los tickers con los que se entreno el modelo
    (`model.tickers`, si lo tiene), que son el universo del que es relativo.
    La accuracy direccional se calcula igual y se devuelve como referencia.
    """
    universo = [t for t in (getattr(model, "tickers", None) or histories) if t in histories]
    fechas, pred, real = [], [], []
    for symbol in universo:
        df = histories[symbol]
        close = df["close"].to_numpy(dtype=np.float64)
        for cutoff in range(max(len(df) - backtest_days - horizon, 1), len(df) - horizon):
            try:
                result = model.predict_df(df.iloc[: cutoff + 1])
            except ApiMlError:
                continue
            fechas.append(df.index[cutoff])
            pred.append(np.log(result["predicted_close"] / close[cutoff]))
            real.append(np.log(close[cutoff + horizon] / close[cutoff]))
    if not pred:
        return {"ic": None, "directional_accuracy": None, "observations": 0}
    pred_arr, real_arr = np.asarray(pred), np.asarray(real)
    return {
        "ic": ic_por_fecha(pred_arr, real_arr, np.asarray(fechas)),
        "directional_accuracy": float(np.mean(np.sign(pred_arr) == np.sign(real_arr))),
        "observations": len(pred),
    }


def _save_score(model_path: str, score: dict) -> None:
    _score_path(model_path).write_text(json.dumps(score, indent=2))


def incumbent_score(
    model_path: str,
    loader: Callable[[Path], object],
    histories: dict[str, pd.DataFrame],
    horizon: int,
) -> float | None:
    """Score del modelo vigente, para comparar contra un candidato nuevo.

    Si ya hay un sidecar de una corrida anterior, lo reusa (evita
    re-backtestear el vigente en cada corrida). Si no hay sidecar pero si
    hay un artefacto en disco (primera vez que corre este gate sobre un
    modelo entrenado a mano), lo backtestea una vez para arrancar con una
    base real en vez de promover a ciegas.
    """
    path = _score_path(model_path)
    if path.exists():
        try:
            return json.loads(path.read_text())["directional_accuracy"]
        except (json.JSONDecodeError, KeyError, OSError):
            pass
    model_file = Path(model_path)
    if not model_file.exists():
        return None
    incumbent = loader(model_file)
    score = pooled_directional_accuracy(incumbent, histories, horizon)
    if score["directional_accuracy"] is not None:
        _save_score(model_path, score)
    return score["directional_accuracy"]


def _promote_by_ic(
    model,
    model_path: str,
    loader: Callable[[Path], object],
    histories: dict[str, pd.DataFrame],
    horizon: int,
) -> tuple[bool, dict]:
    """Gate por IC. A diferencia del de accuracy, el vigente se re-evalua sobre
    la misma ventana que el candidato en vez de reusar el score de cuando se
    promovio: un score viejo y alto no se compara contra datos nuevos."""
    candidate = pooled_ic(model, histories, horizon)
    incumbent = None
    if Path(model_path).exists():
        incumbent = pooled_ic(loader(Path(model_path)), histories, horizon)
        candidate["incumbent_directional_accuracy"] = incumbent["directional_accuracy"]
        incumbent = incumbent["ic"]
    candidate["incumbent_ic"] = incumbent
    ic = candidate["ic"]
    should_promote = ic is not None and (incumbent is None or ic > incumbent)
    if should_promote:
        model.save(model_path)
        _save_score(model_path, candidate)
    return should_promote, candidate


def _score_walk_forward(model) -> dict:
    """Metricas fuera de muestra del walk-forward que ``fit`` deja en ``model.metrics``."""
    metrics = getattr(model, "metrics", None) or {}
    return {
        "metrica": "walk_forward",
        "val_acierto_balanceado": metrics.get("val_acierto_balanceado"),
        "val_ic": metrics.get("val_ic"),
        "observations": int(metrics.get("val_samples", 0)),
    }


def _promote_by_walk_forward(model, model_path: str) -> tuple[bool, dict]:
    """Gate por las metricas del walk-forward del propio entrenamiento.

    El backtest de las ultimas ruedas no sirve para un modelo que al final se
    reentrena con todas las fechas: esas ruedas estan dentro del entrenamiento.
    Tampoco se compara contra el vigente: su walk-forward es de otra ventana
    (un dia antes, casi la misma) y la diferencia entre dos reentrenos seguidos
    es ruido. Se promueve si el candidato pasa el piso: acierto balanceado mayor
    a ``MIN_ACIERTO_WALK_FORWARD`` e IC positivo.
    """
    score = _score_walk_forward(model)
    acierto, ic = score["val_acierto_balanceado"], score["val_ic"]
    should_promote = (
        acierto is not None and acierto > MIN_ACIERTO_WALK_FORWARD and ic is not None and ic > 0
    )
    if should_promote:
        model.save(model_path)
        _save_score(model_path, score)
    return should_promote, score


def promote_if_better(
    model,
    model_path: str,
    loader: Callable[[Path], object],
    histories: dict[str, pd.DataFrame],
    horizon: int,
    metrica: str = "accuracy",
) -> tuple[bool, dict]:
    """Backtestea `model` y lo guarda en `model_path` solo si supera al vigente.

    Devuelve `(promovido, candidate_score)`. No tirar si no se puede medir:
    un candidato sin observaciones suficientes no se promueve (mas vale
    quedarse con el vigente que arriesgar a ciegas). `metrica="ic"` usa
    `pooled_ic` (modelos con target relativo).
    """
    if metrica == "ic":
        return _promote_by_ic(model, model_path, loader, histories, horizon)
    if metrica == "walk_forward":
        return _promote_by_walk_forward(model, model_path)
    candidate = pooled_directional_accuracy(model, histories, horizon)
    incumbent = incumbent_score(model_path, loader, histories, horizon)
    accuracy = candidate["directional_accuracy"]
    should_promote = accuracy is not None and (incumbent is None or accuracy > incumbent)
    candidate["incumbent_directional_accuracy"] = incumbent
    if should_promote:
        model.save(model_path)
        _save_score(model_path, candidate)
    return should_promote, candidate


def force_promote(
    model,
    model_path: str,
    histories: dict[str, pd.DataFrame],
    horizon: int,
    metrica: str = "accuracy",
) -> dict:
    """Guarda `model` sin comparar contra el vigente (uso manual, `--force`)."""
    if metrica == "walk_forward":
        model.save(model_path)
        score = _score_walk_forward(model)
        _save_score(model_path, score)
        return score
    if metrica == "ic":
        score = pooled_ic(model, histories, horizon)
    else:
        score = pooled_directional_accuracy(model, histories, horizon)
    model.save(model_path)
    if score["directional_accuracy"] is not None:
        _save_score(model_path, score)
    return score
