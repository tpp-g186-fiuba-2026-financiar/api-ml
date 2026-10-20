"""Gate de promocion del Transformer V2 (cron mensual ``retrain-transformer-v2.yml``).

El gate de los otros modelos (``src.retrain_guard``) compara el candidato con
el vigente en las ultimas ruedas, que el candidato ya vio al entrenar: con un
modelo que se reentrena con todos los datos eso lo favorece siempre. Aca el
IC se mide fuera de muestra:

1. Se entrena un candidato de evaluacion con la misma config pero solo con
   los datos anteriores a la ventana de evaluacion (las ultimas
   ``BACKTEST_DAYS`` ruedas con retorno realizado).
2. Se mide su IC (correlacion de rangos por dia entre retorno predicho y
   real, ``ic_por_fecha``) en esa ventana, prediciendo cada dia con los datos
   hasta ese dia.
3. Si el IC es mayor que ``IC_MINIMO`` (ordena mejor que el azar), se entrena
   el modelo final con todos los datos y se guarda; si no, queda el vigente.

El IC del vigente en la misma ventana se calcula y se guarda como referencia,
pero no decide: el vigente se entreno hace un mes con parte de esa ventana, asi
que la comparacion lo favoreceria (y con la misma config, la diferencia entre
los dos es sobre todo ruido de semillas).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from src.errors import ApiMlError, NotEnoughDataError
from src.retrain_guard import _score_path
from src.trend_common import BACKTEST_DAYS, ic_por_fecha

IC_MINIMO = 0.0


def ic_panel(
    modelo, historias: dict[str, pd.DataFrame], horizonte: int, dias: int = BACKTEST_DAYS
) -> dict:
    """IC de ``modelo`` (con ``predict_panel``) en las ultimas ``dias`` ruedas
    con retorno a ``horizonte`` realizado."""
    fechas = sorted(set().union(*(df.index for df in historias.values())))
    ventana = fechas[-(dias + horizonte) : -horizonte]
    pred, real, dia_obs = [], [], []
    for dia in ventana:
        resultados = modelo.predict_panel({s: df[df.index <= dia] for s, df in historias.items()})
        for symbol, r in resultados.items():
            close = historias[symbol]["close"]
            i = close.index.get_indexer([dia])[0]
            if i < 0 or i + horizonte >= len(close):
                continue
            pred.append(np.log(r["predicted_close"] / r["last_close"]))
            real.append(np.log(close.iloc[i + horizonte] / close.iloc[i]))
            dia_obs.append(dia)
    if not pred:
        return {"ic": None, "observations": 0}
    return {
        "ic": round(ic_por_fecha(np.array(pred), np.array(real), np.array(dia_obs)), 4),
        "observations": len(pred),
        "desde": ventana[0].strftime("%Y-%m-%d"),
        "hasta": ventana[-1].strftime("%Y-%m-%d"),
    }


def entrenar_con_gate(
    fabrica: Callable[[], object],
    loader: Callable[[Path], object],
    historias: dict[str, pd.DataFrame],
    model_path: str,
    horizonte: int,
    dias: int = BACKTEST_DAYS,
) -> tuple[bool, dict]:
    """Entrena con el gate de arriba. ``fabrica()`` devuelve un modelo sin
    entrenar (con su config y proveedor). Devuelve ``(promovido, score)``."""
    fechas = sorted(set().union(*(df.index for df in historias.values())))
    if len(fechas) <= dias + horizonte:
        raise NotEnoughDataError(f"muy pocas ruedas para el gate ({len(fechas)})")
    inicio = fechas[-(dias + horizonte)]
    previas = {s: df[df.index < inicio] for s, df in historias.items()}

    evaluador = fabrica()
    evaluador.fit(previas)
    score = ic_panel(evaluador, historias, horizonte, dias)
    score["ic_vigente"] = None
    if Path(model_path).exists():
        try:
            vigente = loader(Path(model_path))
            vigente.proveedor = evaluador.proveedor
            score["ic_vigente"] = ic_panel(vigente, historias, horizonte, dias)["ic"]
        except ApiMlError as exc:  # p. ej. artefacto viejo (StaleArtifactError)
            print(f"  [warn] no se pudo evaluar el vigente: {exc}")

    promovido = score["ic"] is not None and score["ic"] > IC_MINIMO
    if promovido:
        final = fabrica()
        final.fit(historias)
        final.save(model_path)
        _score_path(model_path).write_text(json.dumps(score, indent=2))
    return promovido, score
