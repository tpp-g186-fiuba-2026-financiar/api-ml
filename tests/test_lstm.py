"""Tests del pipeline LSTM con datos sinteticos (sin red)."""

import numpy as np
import pandas as pd
import pytest

from src.errors import NotEnoughDataError, StaleArtifactError
from src.lstm import (
    FEATURE_NAMES,
    TrainConfig,
    TrendModel,
    build_features,
    check_feature_compatibility,
    make_windows,
    rsi,
)


def _synthetic_ohlcv(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Camino con una leve tendencia + ruido para que haya algo que aprender.
    returns = rng.normal(0.0005, 0.02, n)
    close = 100 * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(rng.normal(0, 0.01, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.01, n)))
    volume = rng.integers(1_000, 10_000, n).astype(float)
    index = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )


def test_build_features_shapes() -> None:
    df = _synthetic_ohlcv(100)
    feats, target = build_features(df, horizon=5)
    assert feats.shape[1] == len(FEATURE_NAMES)
    assert len(feats) == len(target) == len(df) - 1
    # Las ultimas filas no tienen target futuro completo.
    assert np.isnan(target[-1])
    # Ninguna feature queda en NaN/inf (se limpian con 0 durante el warmup).
    assert np.isfinite(feats).all()


def test_make_windows_drops_nan_targets() -> None:
    df = _synthetic_ohlcv(100)
    feats, target = build_features(df, horizon=5)
    X, y = make_windows(feats, target, window=10)
    assert X.shape[1] == 10
    assert X.shape[2] == len(FEATURE_NAMES)
    assert len(X) == len(y)
    assert not np.isnan(y).any()


def test_check_feature_compatibility_accepts_current_names() -> None:
    check_feature_compatibility(FEATURE_NAMES)
    check_feature_compatibility(None)  # artefactos viejos sin el campo: no rompe


def test_check_feature_compatibility_rejects_stale_artifact() -> None:
    with pytest.raises(StaleArtifactError):
        check_feature_compatibility(["log_return", "log_volume_change", "range_pct"])


def test_rsi_bounds() -> None:
    df = _synthetic_ohlcv(100)
    value = rsi(df["close"].to_numpy())
    assert value is not None
    assert 0.0 <= value <= 100.0


def test_train_predict_save_load(tmp_path) -> None:
    histories = {f"SYN{i}": _synthetic_ohlcv(seed=i) for i in range(3)}
    config = TrainConfig(window=20, horizon=5, epochs=5, patience=10)
    model = TrendModel(config=config)
    metrics = model.fit(histories)
    assert "val_directional_accuracy" in metrics

    result = model.predict_df(histories["SYN0"])
    assert result["signal"] in {"alza", "baja", "neutral"}
    assert result["condition"] in {"sobrecompra", "sobreventa", "neutral", "indeterminado"}
    assert result["horizon_days"] == 5
    assert "predicted_close" in result

    path = tmp_path / "lstm.pt"
    model.save(path)
    reloaded = TrendModel.load(path)
    reloaded_result = reloaded.predict_df(histories["SYN0"])
    assert reloaded_result["predicted_close"] == pytest.approx(result["predicted_close"], rel=1e-4)


def test_predict_requires_enough_data() -> None:
    histories = {f"SYN{i}": _synthetic_ohlcv(seed=i) for i in range(3)}
    model = TrendModel(config=TrainConfig(window=20, epochs=3))
    model.fit(histories)
    with pytest.raises(NotEnoughDataError):
        model.predict_df(_synthetic_ohlcv(n=10))


def test_load_rejects_artifact_with_stale_feature_set(tmp_path) -> None:
    """Un .pt entrenado antes de cambiar `build_features` no debe cargar en silencio."""
    import torch

    histories = {f"SYN{i}": _synthetic_ohlcv(seed=i) for i in range(3)}
    model = TrendModel(config=TrainConfig(window=20, horizon=5, epochs=3, patience=10))
    model.fit(histories)
    path = tmp_path / "lstm.pt"
    model.save(path)

    blob = torch.load(path, map_location="cpu", weights_only=False)
    blob["feature_names"] = ["log_return", "log_volume_change", "range_pct"]
    torch.save(blob, path)

    with pytest.raises(StaleArtifactError):
        TrendModel.load(path)
