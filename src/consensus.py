"""Algoritmo de consenso: combina las predicciones de todos los modelos
(ver ``TrendRegistry.compare()``) en una sola lectura (sobrecompra /
sobreventa / neutral), ajustada por el perfil de riesgo del inversor.

Adaptado de ``decision_maker_playground`` (issue #154, decidir que hacer con
ese prototipo). De las 6 formulas que probaba ese playground se porto la
ecuacion 1 ("confianza-ponderado + acuerdo"): es la base de la que tambien
partian las ecuaciones 5 y 6 (ajuste por RSI / por varianza), y a diferencia
de ``simple_average`` (ignora que tan bueno es cada modelo) o
``winner_takes_all`` (descarta todo menos uno) no tira informacion que ya
tenemos. Las variantes con ajuste extra (RSI, varianza entre modelos) quedan
para una segunda vuelta si esta primera version muestra que hace falta --
no hay datos de produccion todavia para justificar esa complejidad extra.

Dos adaptaciones respecto del playground:

1. El playground pedia ``confidence`` por prediccion (un input manual, 0-1).
   Ese campo ya no existe en el contrato real de TrendResponse -- se saco
   porque saturaba en 1.0 con cualquier retorno >3% (ver
   ``registry._DROPPED_TREND_FIELDS``). Aca se usa en su lugar el
   ``directional_accuracy`` del backtest de cada modelo: un numero medido,
   no una autoevaluacion del modelo. Ver ``_model_quality``.
2. El playground pedia ``expected_return`` directo. Aca se deriva de
   ``predicted_close``/``last_close`` (mismo criterio que el resto del
   sistema, ver ``derive_trend_output``). Ver ``_expected_return``.

Los modelos sin precio (``svm-modal``: solo da Buy/Sell, sin magnitud) o sin
direccion (``garch-modal``: solo volatilidad) quedan afuera del score por
diseño -- no se les inventa un retorno o una direccion.

Todos los modelos considerados pesan igual (``weight=1``): a diferencia del
playground, que dejaba un peso manual por modelo, ac'a no hay una razon
objetiva para privilegiar a mano un modelo sobre otro mas alla de lo que ya
capta ``_model_quality`` -- agregar un peso fijo adicional seria un numero
inventado.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.schemas import PerfilRiesgo

# Umbrales de partida (ver decision_maker_playground/index.html). c0 baja de
# 0.60 (el default del playground) a 0.50: alla "confidence" era una
# autoevaluacion del modelo tipica ~0.6-0.9, aca es directional_accuracy de
# backtest real, que para estos modelos y horizontes cortos ronda 0.45-0.60
# (apenas mejor que azar) -- con el 0.60 original el consenso caeria en
# NEUTRAL casi siempre. Son valores de arranque, no calibrados todavia con
# datos de produccion: revisar cuando haya suficientes lecturas reales.
THETA0_DEFAULT = 0.01
ALPHA_DEFAULT = 0.005
C0_DEFAULT = 0.50
BETA_DEFAULT = 0.05

# Mismo perfil de riesgo que usa `black_litterman` (ver `PerfilRiesgo` en
# src.schemas, alineado con `users.risk_profile` en backend-website), pero
# mapeado al p de -1..1 del playground: a mayor p, mas facil sobrecompra y
# mas dificil sobreventa (umbrales asimetricos).
_PROFILE_SCORE: dict[PerfilRiesgo, int] = {
    PerfilRiesgo.CONSERVADOR: -1,
    PerfilRiesgo.MODERADO: 0,
    PerfilRiesgo.ARRIESGADO: 1,
}


@dataclass(frozen=True)
class ConsensusParams:
    theta0: float = THETA0_DEFAULT
    alpha: float = ALPHA_DEFAULT
    c0: float = C0_DEFAULT
    beta: float = BETA_DEFAULT


def _expected_return(prediction: dict) -> float | None:
    """Retorno log implicito en la prediccion. None si el modelo no da
    precio (ej: svm-modal), o si el precio actual no es valido.
    """
    predicted = prediction.get("predicted_close")
    last = prediction.get("last_close")
    if predicted is None or last is None or last <= 0:
        return None
    return float(np.log(predicted / last))


# Casos "virtuales" al 50% que se suman a cada accuracy: con pocas
# observaciones el numero se acerca al azar. Mismo criterio que
# ``pick_best_model`` en backend-website, para que ambos ordenen igual.
ACCURACY_PRIOR_CASES = 50


def _model_quality(prediction: dict) -> float:
    """Reemplazo de "confidence": accuracy direccional del backtest de ese
    modelo para este ticker puntual, ajustada por cantidad de casos: un 90%
    sobre 20 observaciones pesa mucho menos que un 60% sobre 500, porque con
    pocos casos el numero es casi suerte. 0.5 (ni mejor ni peor que azar) para
    modelos sin backtest todavia -- no los excluye, pero tampoco les da de
    arranque mas peso que a una moneda al aire.
    """
    backtest = prediction.get("backtest") or {}
    accuracy = backtest.get("directional_accuracy")
    if accuracy is None:
        return 0.5
    observations = backtest.get("observations")
    if not observations or observations <= 0:
        return float(accuracy)
    weight = observations / (observations + ACCURACY_PRIOR_CASES)
    return float(0.5 + (accuracy - 0.5) * weight)


def _agreement(signal: str | None, expected_return: float) -> float:
    """Penaliza (x0.5) cuando el signal categorico no coincide con el signo
    del retorno implicito -- pasa cerca de la banda neutral (`neutral_band`
    en `derive_trend_output`): ahi el modelo dice "neutral" pero su retorno
    crudo todavia tiene signo. Igual que la ecuacion 1 del playground.
    """
    signal_num = {"alza": 1, "baja": -1}.get(signal or "", 0)
    if expected_return > 0:
        sign = 1
    elif expected_return < 0:
        sign = -1
    else:
        sign = 0
    return 1.0 if (signal_num == sign or expected_return == 0) else 0.5


def _build_explanation(
    classification: str,
    considered: int,
    alza_count: int,
    baja_count: int,
    confidence: float,
    confidence_min: float,
) -> str:
    """Traduce la lectura numerica a una frase en espanol llano, para que el
    usuario entienda el "por que" sin tener que leer score/thresholds.
    """
    if classification == "sin_datos":
        return "Todavia no hay modelos disponibles para este ticker."

    if classification == "neutral":
        if confidence < confidence_min:
            return (
                f"{considered} modelo(s) opinaron, pero su acierto historico "
                f"promedio ({confidence:.0%}) esta por debajo del minimo exigido "
                f"({confidence_min:.0%}): no hay confianza suficiente para una senal."
            )
        return (
            f"Los modelos estan divididos: {alza_count} en alza y {baja_count} "
            f"en baja (de {considered} considerados) -- no hay una tendencia clara."
        )

    if classification == "sobrecompra":
        return (
            f"{alza_count} de {considered} modelos coinciden en una lectura "
            f"alcista, con un acierto historico promedio del {confidence:.0%}."
        )

    return (
        f"{baja_count} de {considered} modelos coinciden en una lectura "
        f"bajista, con un acierto historico promedio del {confidence:.0%}."
    )


def compute_consensus(
    predictions: dict[str, dict],
    profile: PerfilRiesgo,
    params: ConsensusParams | None = None,
) -> dict:
    """Combina las predicciones de ``TrendRegistry.compare()`` en una unica
    lectura. ``profile`` es el perfil de riesgo del inversor (mismo enum que
    ``black_litterman``); a mayor perfil, mas facil sobrecompra y mas
    dificil sobreventa (umbrales asimetricos, igual que el playground).
    Modelos no disponibles o sin precio (ver ``_expected_return``) se
    ignoran.
    """
    params = params or ConsensusParams()
    profile_score = _PROFILE_SCORE[profile]

    num_score = 0.0
    den_score = 0.0
    num_confidence = 0.0
    den_confidence = 0.0
    considered = 0
    alza_count = 0
    baja_count = 0

    for prediction in predictions.values():
        if prediction.get("available") is False:
            continue
        expected_return = _expected_return(prediction)
        if expected_return is None:
            continue

        quality = _model_quality(prediction)
        agreement = _agreement(prediction.get("signal"), expected_return)

        num_score += quality * expected_return * agreement
        den_score += quality
        num_confidence += quality
        den_confidence += 1.0
        considered += 1
        if expected_return > 0:
            alza_count += 1
        elif expected_return < 0:
            baja_count += 1

    score = num_score / den_score if den_score else 0.0
    confidence = num_confidence / den_confidence if den_confidence else 0.0

    theta_buy = params.theta0 - params.alpha * profile_score
    theta_sell = -params.theta0 + params.alpha * profile_score
    confidence_min = params.c0 - params.beta * profile_score

    if considered == 0:
        classification = "sin_datos"
    elif confidence < confidence_min:
        classification = "neutral"
    elif score > theta_buy:
        classification = "sobrecompra"
    elif score < theta_sell:
        classification = "sobreventa"
    else:
        classification = "neutral"

    explanation = _build_explanation(
        classification, considered, alza_count, baja_count, confidence, confidence_min
    )

    return {
        "classification": classification,
        "composite_score": round(score, 4),
        "aggregate_confidence": round(confidence, 3),
        "threshold_buy": round(theta_buy, 4),
        "threshold_sell": round(theta_sell, 4),
        "confidence_min": round(confidence_min, 3),
        "models_considered": considered,
        "explanation": explanation,
    }
