"""Tests del job de paper trading con datos sinteticos (sin red)."""

import numpy as np
import pandas as pd

from src.paper_trading import (
    build_summary,
    known_tickers,
    record_new_predictions,
    resolve_pending,
    select_model_names,
)
from src.registry import TrendRegistry


def _synthetic_ohlcv(n: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0005, 0.02, n)
    close = 100 * np.exp(np.cumsum(returns))
    index = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame({"close": close}, index=index)


class _FakeService:
    """Doble de TrendService/OnDemandTrendService: solo lo que usa paper_trading."""

    def __init__(self, prediction: dict, *, loaded: bool = True):
        self.is_loaded = loaded
        self._prediction = prediction

    def predict(self, symbol: str) -> dict:
        return {**self._prediction, "symbol": symbol}


def _registry_with(name: str, service: _FakeService) -> TrendRegistry:
    registry = TrendRegistry(history_days=750)
    registry._services[name] = service
    registry._default = name
    return registry


_PREDICTION = {
    "model_version": "v1",
    "as_of": "2024-01-10",
    "signal": "alza",
    "horizon_days": 5,
    "last_close": 100.0,
    "predicted_close": 105.0,
}


def test_select_model_names_defaults_to_loaded() -> None:
    registry = _registry_with("lstm", _FakeService(_PREDICTION))
    registry._services["xgboost"] = _FakeService(_PREDICTION, loaded=False)

    assert select_model_names(registry, None) == ["lstm"]


def test_select_model_names_skips_unloaded_when_requested() -> None:
    registry = _registry_with("lstm", _FakeService(_PREDICTION))
    registry._services["xgboost"] = _FakeService(_PREDICTION, loaded=False)

    assert select_model_names(registry, ["lstm", "xgboost"]) == ["lstm"]


def test_record_new_predictions_builds_entries() -> None:
    registry = _registry_with("lstm", _FakeService(_PREDICTION))

    entries = record_new_predictions(registry, ["lstm"], ["GGAL"], existing_keys=set())

    assert len(entries) == 1
    entry = entries[0]
    assert entry["symbol"] == "GGAL"
    assert entry["model"] == "lstm"
    assert entry["as_of"] == "2024-01-10"
    assert entry["predicted_log_return"] == np.log(105.0 / 100.0)


def test_record_new_predictions_skips_existing_keys() -> None:
    registry = _registry_with("lstm", _FakeService(_PREDICTION))
    existing_keys = {("GGAL", "lstm", "2024-01-10")}

    entries = record_new_predictions(registry, ["lstm"], ["GGAL"], existing_keys)

    assert entries == []


def test_resolve_pending_moves_matured_predictions(monkeypatch) -> None:
    df = _synthetic_ohlcv(40)
    monkeypatch.setattr("src.paper_trading.fetch_history", lambda symbol, days: df)

    as_of = df.index[10].strftime("%Y-%m-%d")
    last_close = float(df["close"].iloc[10])
    pending_entry = {
        "symbol": "GGAL",
        "model": "lstm",
        "model_version": "v1",
        "as_of": as_of,
        "signal": "alza",
        "horizon_days": 5,
        "last_close": last_close,
        "predicted_close": last_close * 1.02,
        "predicted_log_return": float(np.log(1.02)),
    }

    still_pending, resolved = resolve_pending([pending_entry], days=750)

    assert still_pending == []
    assert len(resolved) == 1
    expected_close = float(df["close"].iloc[15])
    assert resolved[0]["realized_close"] == expected_close
    assert resolved[0]["realized_log_return"] == np.log(expected_close / last_close)


def test_resolve_pending_keeps_immature_predictions(monkeypatch) -> None:
    df = _synthetic_ohlcv(40)
    monkeypatch.setattr("src.paper_trading.fetch_history", lambda symbol, days: df)

    as_of = df.index[35].strftime("%Y-%m-%d")
    pending_entry = {
        "symbol": "GGAL",
        "model": "lstm",
        "model_version": "v1",
        "as_of": as_of,
        "signal": "alza",
        "horizon_days": 10,
        "last_close": 100.0,
        "predicted_close": 102.0,
        "predicted_log_return": float(np.log(1.02)),
    }

    still_pending, resolved = resolve_pending([pending_entry], days=750)

    assert resolved == []
    assert still_pending == [pending_entry]


def test_resolve_pending_pads_approximate_as_of(monkeypatch) -> None:
    """arima-modal manda un as_of aproximado (hoy), no siempre una rueda real."""
    df = _synthetic_ohlcv(40)
    monkeypatch.setattr("src.paper_trading.fetch_history", lambda symbol, days: df)

    last_trading_idx = 9  # df.index[9] es un viernes (2024-01-12)
    not_a_trading_day = df.index[last_trading_idx] + pd.Timedelta(days=2)  # domingo
    last_close = float(df["close"].iloc[last_trading_idx])
    pending_entry = {
        "symbol": "GGAL",
        "model": "arima-modal",
        "model_version": "modal:arima-modal",
        "as_of": not_a_trading_day.strftime("%Y-%m-%d"),
        "signal": "alza",
        "horizon_days": 5,
        "last_close": last_close,
        "predicted_close": last_close * 1.02,
        "predicted_log_return": float(np.log(1.02)),
    }

    still_pending, resolved = resolve_pending([pending_entry], days=750)

    assert still_pending == []
    assert len(resolved) == 1
    assert resolved[0]["realized_close"] == float(df["close"].iloc[last_trading_idx + 5])


def test_build_summary_groups_by_model() -> None:
    resolved = [
        {
            "model": "lstm",
            "symbol": "GGAL",
            "signal": "alza",
            "predicted_log_return": 0.01,
            "realized_log_return": 0.02,
        },
        {
            "model": "lstm",
            "symbol": "YPFD",
            "signal": "baja",
            "predicted_log_return": -0.01,
            "realized_log_return": -0.02,
        },
    ]

    summary = build_summary(resolved)

    assert set(summary) == {"lstm"}
    assert summary["lstm"]["overall"]["n_predictions"] == 2
    assert set(summary["lstm"]["per_ticker"]) == {"GGAL", "YPFD"}
    assert summary["lstm"]["per_ticker"]["GGAL"]["n_predictions"] == 1


def test_known_tickers_uses_pending_and_resolved_without_duplicates() -> None:
    ledger = {
        "pending": [{"symbol": "YPFD"}, {"symbol": "GGAL"}],
        "resolved": [{"symbol": "GGAL"}, {"symbol": "ALUA"}],
    }

    assert known_tickers(ledger) == ["ALUA", "GGAL", "YPFD"]


def _entry(model: str, signal: str, predicted: float, symbol: str = "GGAL") -> dict:
    return {
        "symbol": symbol,
        "model": model,
        "model_version": "v1",
        "as_of": "2024-01-10",
        "signal": signal,
        "horizon_days": 5,
        "last_close": 100.0,
        "predicted_close": predicted,
        "predicted_log_return": float(np.log(predicted / 100.0)),
    }


def test_record_consensus_entries_one_per_profile() -> None:
    from src.paper_trading import record_consensus_entries

    new = [_entry("lstm", "alza", 105.0), _entry("xgboost", "alza", 104.0)]
    entries = record_consensus_entries([], new, [], set())

    assert {e["model"] for e in entries} == {
        "consensus-conservative",
        "consensus-moderate",
        "consensus-aggressive",
    }
    moderate = next(e for e in entries if e["model"] == "consensus-moderate")
    assert moderate["signal"] == "alza"
    assert moderate["symbol"] == "GGAL"
    assert moderate["predicted_log_return"] > 0


def test_record_consensus_entries_skips_existing_and_its_own_inputs() -> None:
    from src.paper_trading import record_consensus_entries

    new = [_entry("lstm", "alza", 105.0), _entry("consensus-moderate", "baja", 90.0)]
    existing = {("GGAL", "consensus-moderate", "2024-01-10")}
    entries = record_consensus_entries([], new, [], existing)

    assert "consensus-moderate" not in {e["model"] for e in entries}
    # Sin el input "baja" colado, el consenso solo ve el lstm (alza o neutral
    # segun el perfil, nunca baja).
    assert all(e["signal"] in ("alza", "neutral") for e in entries)


def test_consensus_entries_are_measured_like_any_model() -> None:
    from src.paper_trading import build_summary

    resolved = [
        {**_entry("consensus-moderate", "alza", 105.0), "realized_log_return": 0.02},
        {**_entry("consensus-moderate", "baja", 95.0), "realized_log_return": 0.01},
        {**_entry("consensus-moderate", "neutral", 100.5), "realized_log_return": 0.01},
    ]
    overall = build_summary(resolved)["consensus-moderate"]["overall"]

    assert overall["n_predictions"] == 3
    assert overall["signal_hit_rate"] == 0.5
