"""Tests del pipeline Transformer con datos sinteticos (sin red)."""

import numpy as np
import pandas as pd
import pytest

from src.errors import NotEnoughDataError
from src.transformer import TransformerTrainConfig, TransformerTrendModel


def _synthetic_ohlcv(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
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


def _small_config(**overrides) -> TransformerTrainConfig:
    base = dict(
        window=10,
        horizon=5,
        d_model=4,
        nhead=2,
        num_layers=1,
        dim_feedforward=8,
        dropout=0.0,
        epochs=3,
        batch_size=16,
        patience=1,
        warmup_epochs=1,
    )
    base.update(overrides)
    return TransformerTrainConfig(**base)


def test_train_predict_save_load(tmp_path) -> None:
    histories = {f"SYN{i}": _synthetic_ohlcv(seed=i) for i in range(3)}
    model = TransformerTrendModel(config=_small_config())
    assert model.version == "untrained"

    metrics = model.fit(histories)
    assert "val_directional_accuracy" in metrics
    assert model.version.startswith("transformer-")

    result = model.predict_df(histories["SYN0"])
    assert result["signal"] in {"alza", "baja", "neutral"}
    assert result["condition"] in {"sobrecompra", "sobreventa", "neutral", "indeterminado"}
    assert "predicted_close" in result

    path = tmp_path / "transformer.pt"
    model.save(path)
    reloaded = TransformerTrendModel.load(path)
    reloaded_result = reloaded.predict_df(histories["SYN0"])
    assert reloaded_result["predicted_close"] == pytest.approx(result["predicted_close"], rel=1e-4)


def test_fit_without_validation_split() -> None:
    """val_fraction=0 deja Xva vacio: cubre la rama sin early stopping."""
    histories = {f"SYN{i}": _synthetic_ohlcv(seed=i) for i in range(2)}
    config = _small_config(val_fraction=0.0, epochs=2)
    model = TransformerTrendModel(config=config)
    metrics = model.fit(histories)
    assert metrics == {}


def test_fit_raises_without_enough_data() -> None:
    histories = {"SYN0": _synthetic_ohlcv(n=5)}
    model = TransformerTrendModel(config=_small_config())
    with pytest.raises(NotEnoughDataError):
        model.fit(histories)


def test_predict_requires_enough_data() -> None:
    histories = {f"SYN{i}": _synthetic_ohlcv(seed=i) for i in range(2)}
    model = TransformerTrendModel(config=_small_config())
    model.fit(histories)
    with pytest.raises(NotEnoughDataError):
        model.predict_df(_synthetic_ohlcv(n=5))


def test_config_rejects_incompatible_d_model_and_nhead() -> None:
    with pytest.raises(ValueError):
        TransformerTrainConfig(d_model=5, nhead=2)
