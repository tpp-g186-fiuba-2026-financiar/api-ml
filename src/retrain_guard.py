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
from pathlib import Path
from typing import Callable

import pandas as pd

from src.trend_common import BACKTEST_DAYS, backtest_predict_df


def _score_path(model_path: str) -> Path:
    return Path(model_path).with_suffix(Path(model_path).suffix + ".score.json")


def pooled_directional_accuracy(
    model,
    histories: dict[str, pd.DataFrame],
    horizon: int,
    backtest_days: int = BACKTEST_DAYS,
) -> dict:
    """Accuracy direccional pooleada sobre todos los tickers de `histories`."""
    total_observations = 0
    total_correct = 0.0
    for df in histories.values():
        result = backtest_predict_df(model, df, horizon, backtest_days)
        if result is None:
            continue
        total_observations += result["observations"]
        total_correct += result["directional_accuracy"] * result["observations"]
    if total_observations == 0:
        return {"directional_accuracy": None, "observations": 0}
    return {
        "directional_accuracy": total_correct / total_observations,
        "observations": total_observations,
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


def promote_if_better(
    model,
    model_path: str,
    loader: Callable[[Path], object],
    histories: dict[str, pd.DataFrame],
    horizon: int,
) -> tuple[bool, dict]:
    """Backtestea `model` y lo guarda en `model_path` solo si supera al vigente.

    Devuelve `(promovido, candidate_score)`. No tirar si no se puede medir:
    un candidato sin observaciones suficientes no se promueve (mas vale
    quedarse con el vigente que arriesgar a ciegas).
    """
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
    model, model_path: str, histories: dict[str, pd.DataFrame], horizon: int
) -> dict:
    """Guarda `model` sin comparar contra el vigente (uso manual, `--force`)."""
    score = pooled_directional_accuracy(model, histories, horizon)
    model.save(model_path)
    if score["directional_accuracy"] is not None:
        _save_score(model_path, score)
    return score
