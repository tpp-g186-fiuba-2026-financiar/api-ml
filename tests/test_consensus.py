"""Tests del algoritmo de consenso (ver src/consensus.py, issue #154)."""

import pytest

from src.consensus import (
    ConsensusParams,
    _agreement,
    _expected_return,
    _model_quality,
    compute_consensus,
)
from src.schemas import PerfilRiesgo


def _prediction(
    last_close: float = 100.0,
    predicted_close: float = 103.0,
    signal: str = "alza",
    directional_accuracy: float | None = 0.6,
    available: bool = True,
) -> dict:
    prediction = {
        "last_close": last_close,
        "predicted_close": predicted_close,
        "signal": signal,
        "available": available,
    }
    if directional_accuracy is not None:
        prediction["backtest"] = {"directional_accuracy": directional_accuracy}
    return prediction


def test_expected_return_is_log_of_price_ratio() -> None:
    result = _expected_return({"last_close": 100.0, "predicted_close": 110.0})
    assert result == pytest.approx(0.09531, abs=1e-4)


def test_expected_return_is_none_without_prices() -> None:
    assert _expected_return({"last_close": None, "predicted_close": 110.0}) is None
    assert _expected_return({"last_close": 100.0, "predicted_close": None}) is None
    assert _expected_return({"last_close": 0.0, "predicted_close": 110.0}) is None


def test_model_quality_uses_backtest_accuracy() -> None:
    assert _model_quality({"backtest": {"directional_accuracy": 0.72}}) == 0.72


def test_model_quality_falls_back_to_coin_flip_without_backtest() -> None:
    assert _model_quality({}) == 0.5
    assert _model_quality({"backtest": {}}) == 0.5


def test_agreement_is_full_when_signal_matches_return_sign() -> None:
    assert _agreement("alza", 0.02) == 1.0
    assert _agreement("baja", -0.02) == 1.0


def test_agreement_is_penalized_near_the_neutral_band() -> None:
    # El retorno crudo es positivo pero chico (menor al neutral_band de
    # derive_trend_output): el modelo ya clasifico "neutral", disagreement.
    assert _agreement("neutral", 0.003) == 0.5


def test_compute_consensus_weights_by_backtest_accuracy() -> None:
    predictions = {
        "lstm": _prediction(predicted_close=105.0, signal="alza", directional_accuracy=0.4),
        "xgboost": _prediction(predicted_close=110.0, signal="alza", directional_accuracy=0.8),
    }
    result = compute_consensus(predictions, PerfilRiesgo.MODERADO)
    # xgboost pesa el doble (0.8 vs 0.4): el score queda mas cerca de su
    # retorno (~0.0953) que del de lstm (~0.0488).
    assert result["composite_score"] == pytest.approx(0.0798, abs=1e-3)
    assert result["models_considered"] == 2


def test_compute_consensus_ignores_unavailable_models() -> None:
    predictions = {
        "lstm": _prediction(available=False),
        "xgboost": _prediction(predicted_close=105.0, directional_accuracy=0.6),
    }
    result = compute_consensus(predictions, PerfilRiesgo.MODERADO)
    assert result["models_considered"] == 1


def test_compute_consensus_ignores_models_without_price() -> None:
    """svm-modal: solo Buy/Sell, sin last_close/predicted_close."""
    predictions = {
        "svm-modal": {
            "available": True,
            "signal": "alza",
            "last_close": None,
            "predicted_close": None,
        },
        "xgboost": _prediction(predicted_close=105.0, directional_accuracy=0.6),
    }
    result = compute_consensus(predictions, PerfilRiesgo.MODERADO)
    assert result["models_considered"] == 1


def test_compute_consensus_returns_sin_datos_when_nothing_is_usable() -> None:
    predictions = {"garch-modal": {"available": False}}
    result = compute_consensus(predictions, PerfilRiesgo.MODERADO)
    assert result["classification"] == "sin_datos"
    assert result["models_considered"] == 0


def test_compute_consensus_returns_neutral_below_confidence_floor() -> None:
    predictions = {
        "lstm": _prediction(predicted_close=110.0, directional_accuracy=0.2),
    }
    result = compute_consensus(predictions, PerfilRiesgo.MODERADO)
    assert result["aggregate_confidence"] == pytest.approx(0.2)
    assert result["classification"] == "neutral"


def test_compute_consensus_classifies_sobrecompra_above_threshold() -> None:
    predictions = {
        "lstm": _prediction(predicted_close=110.0, directional_accuracy=0.7),
        "xgboost": _prediction(predicted_close=112.0, directional_accuracy=0.7),
    }
    result = compute_consensus(predictions, PerfilRiesgo.MODERADO)
    assert result["classification"] == "sobrecompra"


def test_compute_consensus_classifies_sobreventa_below_threshold() -> None:
    predictions = {
        "lstm": _prediction(predicted_close=90.0, signal="baja", directional_accuracy=0.7),
        "xgboost": _prediction(predicted_close=88.0, signal="baja", directional_accuracy=0.7),
    }
    result = compute_consensus(predictions, PerfilRiesgo.MODERADO)
    assert result["classification"] == "sobreventa"


def test_aggressive_profile_makes_sobrecompra_easier_than_conservative() -> None:
    predictions = {
        "lstm": _prediction(predicted_close=101.2, directional_accuracy=0.65),
    }
    conservative = compute_consensus(predictions, PerfilRiesgo.CONSERVADOR)
    aggressive = compute_consensus(predictions, PerfilRiesgo.ARRIESGADO)
    assert conservative["threshold_buy"] > aggressive["threshold_buy"]
    assert conservative["classification"] != "sobrecompra"
    assert aggressive["classification"] == "sobrecompra"


def test_custom_params_are_respected() -> None:
    predictions = {"lstm": _prediction(predicted_close=101.0, directional_accuracy=0.9)}
    strict = compute_consensus(predictions, PerfilRiesgo.MODERADO, ConsensusParams(theta0=0.5))
    assert strict["classification"] == "neutral"
