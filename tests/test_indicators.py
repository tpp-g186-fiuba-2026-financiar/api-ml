"""Tests de los indicadores tecnicos vectorizados (sin red)."""

import numpy as np

from src.indicators import ema, macd_histogram, rsi_series, sma


def _synthetic_close(n: int = 100, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0005, 0.02, n)
    return 100 * np.exp(np.cumsum(returns))


def test_rsi_series_bounds_and_warmup() -> None:
    close = _synthetic_close(100)
    series = rsi_series(close, period=14)
    assert len(series) == len(close)
    assert np.isnan(series[:14]).all()
    valid = series[14:]
    assert np.isfinite(valid).all()
    assert ((valid >= 0.0) & (valid <= 100.0)).all()


def test_rsi_series_matches_last_value_of_scalar_rsi() -> None:
    from src.lstm import rsi

    close = _synthetic_close(100)
    assert rsi(close) == rsi_series(close)[-1]


def test_sma_warmup_and_values() -> None:
    values = np.arange(1, 11, dtype=np.float64)  # 1..10
    result = sma(values, window=3)
    assert np.isnan(result[:2]).all()
    assert result[2] == 2.0  # media de 1,2,3
    assert result[-1] == 9.0  # media de 8,9,10


def test_ema_no_nan_and_starts_at_first_value() -> None:
    close = _synthetic_close(50)
    result = ema(close, span=12)
    assert len(result) == len(close)
    assert not np.isnan(result).any()
    assert result[0] == close[0]


def test_macd_histogram_no_nan() -> None:
    close = _synthetic_close(100)
    result = macd_histogram(close)
    assert len(result) == len(close)
    assert np.isfinite(result).all()
