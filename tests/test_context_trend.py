"""Tests del modelo de contexto de mercado."""

import joblib
import numpy as np
import pandas as pd
import pytest

from src import context_trend, train_context, xgb_trend
from src.context_trend import (
    CONTEXT_FEATURES,
    FEATURES,
    LEADERS,
    MIN_ROWS,
    ContextConfig,
    ContextTrendModel,
    build_features,
    context_features,
    market_frame,
)
from src.errors import NotEnoughDataError, StaleArtifactError

ROWS = 700
TICKERS = list(LEADERS) + ["BBAR", "CRES"]


def make_history(rows: int = ROWS, seed: int = 0, flat: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-04", periods=rows)
    close = (
        np.full(rows, 100.0) if flat else 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, rows)))
    )
    return pd.DataFrame(
        {
            "open": close * (1 + rng.normal(0, 0.002, rows)),
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": rng.integers(1_000, 50_000, rows).astype(float),
        },
        index=idx,
    )


@pytest.fixture(scope="module")
def histories() -> dict[str, pd.DataFrame]:
    return {t: make_history(seed=i) for i, t in enumerate(TICKERS)}


@pytest.fixture(scope="module")
def market(histories) -> pd.DataFrame:
    return market_frame(histories)


@pytest.fixture(scope="module")
def universe(histories):
    return xgb_trend.build_universe(histories)


@pytest.fixture(scope="module")
def trained(histories, universe) -> ContextTrendModel:
    model = ContextTrendModel(config=ContextConfig(oos_folds=2))
    model.fit(histories)
    model.universe_provider = lambda: universe
    return model


def test_market_frame_needs_enough_leaders() -> None:
    with pytest.raises(NotEnoughDataError):
        market_frame({"GGAL": make_history(), "YPFD": make_history(seed=1)})


def test_market_frame_ignores_non_leaders(histories) -> None:
    only_leaders = market_frame({k: v for k, v in histories.items() if k in LEADERS})
    pd.testing.assert_frame_equal(market_frame(histories), only_leaders)


def test_build_features_follow_feature_order(histories, market) -> None:
    f = build_features(histories["BBAR"], market)
    assert list(f.columns) == FEATURES
    assert set(CONTEXT_FEATURES) <= set(f.columns)
    assert f.index.equals(histories["BBAR"].index)


def test_context_features_do_not_look_ahead(histories, market) -> None:
    """El contexto en una fecha no cambia si se corta la serie ahi (sin mirar el futuro)."""
    df = histories["BBAR"]
    cut = df.index[400]
    full = context_features(df, market)
    partial = context_features(df.loc[:cut], market.loc[:cut])
    pd.testing.assert_series_equal(full.loc[cut], partial.loc[cut], check_names=False)


def test_relative_strength_is_stock_minus_market(histories, market) -> None:
    df = histories["BBAR"]
    f = context_features(df, market)
    expected = np.log(df["close"]).diff(20) - market["m_ret20"].reindex(df.index, method="ffill")
    pd.testing.assert_series_equal(f["m_rel20"], expected, check_names=False)


def test_fit_records_out_of_sample_metrics(trained) -> None:
    m = trained.metrics
    assert 0.0 <= m["oos_auc"] <= 1.0
    assert m["oos_samples"] > 0
    assert trained.alza_threshold <= trained.baja_threshold
    assert {"baja_disabled", "alza_disabled"} <= set(m)
    assert trained.version.startswith("contexto-")
    assert trained.tickers


def test_predict_df_returns_a_coherent_trend_response(trained, histories) -> None:
    out = trained.predict_df(histories["BBAR"])
    assert out["signal"] in {"alza", "baja", "neutral"}
    assert out["horizon_days"] == trained.config.horizon
    assert out["probability_up"] == pytest.approx(1 - out["probability_down"], abs=1e-3)
    if out["signal"] == "neutral":
        assert out["predicted_close"] == pytest.approx(out["last_close"])


def test_signal_follows_calibrated_thresholds(trained, histories) -> None:
    df = histories["BBAR"]
    p = trained.probability_down(df)
    expected = (
        "baja"
        if p >= trained.baja_threshold
        else "alza"
        if p <= trained.alza_threshold
        else "neutral"
    )
    assert trained.predict_df(df)["signal"] == expected


def test_prediction_uses_market_as_of_the_last_date(trained, histories) -> None:
    """Predecir con el historico cortado no usa datos de mercado posteriores al corte."""
    df = histories["BBAR"].iloc[:500]
    seen = {}
    original = xgb_trend.build_features

    def spy(frame, mkt):
        seen["last"] = mkt.index[-1]
        return original(frame, mkt)

    xgb_trend.build_features, backup = spy, xgb_trend.build_features
    try:
        trained.probability_down(df)
    finally:
        xgb_trend.build_features = backup
    assert seen["last"] <= df.index[-1]


def test_short_or_stale_history_is_refused(trained) -> None:
    with pytest.raises(NotEnoughDataError):
        trained.probability_down(make_history(rows=MIN_ROWS - 10))
    with pytest.raises(NotEnoughDataError):
        trained.probability_down(make_history(flat=True))


def test_untrained_model_cannot_predict_or_save(tmp_path) -> None:
    model = ContextTrendModel()
    with pytest.raises(NotEnoughDataError):
        model.probability_down(make_history())
    with pytest.raises(NotEnoughDataError):
        model.save(tmp_path / "x.pkl")
    assert model.version == "untrained"


def test_fit_rejects_too_little_data() -> None:
    few = {t: make_history(rows=MIN_ROWS + 5, seed=i) for i, t in enumerate(LEADERS)}
    with pytest.raises(NotEnoughDataError):
        ContextTrendModel().fit(few)


def test_save_and_load_roundtrip(trained, histories, universe, tmp_path) -> None:
    path = tmp_path / "contexto.pkl"
    trained.save(path)
    loaded = ContextTrendModel.load(path)
    loaded.universe_provider = lambda: universe
    assert loaded.version == trained.version
    df = histories["CRES"]
    assert loaded.probability_down(df) == pytest.approx(trained.probability_down(df))


def test_load_refuses_artifacts_with_other_features(trained, tmp_path) -> None:
    path = tmp_path / "old.pkl"
    trained.save(path)
    blob = joblib.load(path)
    blob["feature_names"] = xgb_trend.ALL_FEATURES[:-1]
    joblib.dump(blob, path)
    with pytest.raises(StaleArtifactError):
        ContextTrendModel.load(path)


def test_load_refuses_artifacts_with_old_format(trained, tmp_path) -> None:
    path = tmp_path / "old.pkl"
    trained.save(path)
    blob = joblib.load(path)
    del blob["baja_threshold"]
    joblib.dump(blob, path)
    with pytest.raises(StaleArtifactError):
        ContextTrendModel.load(path)


def test_todays_market_frame_is_cached_per_day(monkeypatch, histories) -> None:
    calls = []

    def fake_fetch(symbol, days):
        calls.append(symbol)
        return histories[symbol]

    monkeypatch.setattr(context_trend, "fetch_history", fake_fetch)
    context_trend._market_cache.clear()
    first = context_trend.todays_market_frame()
    n = len(calls)
    assert n == len(LEADERS)
    assert context_trend.todays_market_frame() is first
    assert len(calls) == n


def test_todays_market_frame_survives_a_missing_leader(monkeypatch, histories) -> None:
    from src.errors import DataUnavailableError

    def fake_fetch(symbol, days):
        if symbol == "GGAL":
            raise DataUnavailableError("sin datos")
        return histories[symbol]

    monkeypatch.setattr(context_trend, "fetch_history", fake_fetch)
    context_trend._market_cache.clear()
    assert "m_ret20" in context_trend.todays_market_frame().columns


def test_train_script_saves_only_when_auc_is_good(monkeypatch, histories, tmp_path, capsys) -> None:
    monkeypatch.setattr(train_context, "fetch_history", lambda s, d: histories[s])
    monkeypatch.setattr(train_context, "UNIVERSE", tuple(histories))
    out = tmp_path / "m.pkl"
    monkeypatch.setattr("sys.argv", ["train", "--out", str(out), "--min-auc", "0.99"])
    train_context.main()
    assert not out.exists()
    monkeypatch.setattr("sys.argv", ["train", "--out", str(out), "--min-auc", "0.0"])
    train_context.main()
    assert out.exists()
    assert "Modelo guardado" in capsys.readouterr().out


def test_a_side_without_edge_out_of_sample_is_never_emitted(histories) -> None:
    """Si 'baja' (o 'alza') no le gana a su tasa base fuera de muestra, queda apagada."""
    model = ContextTrendModel(config=ContextConfig(oos_folds=2))
    panel = model._panel(histories, market_frame(histories))
    original = model._oos_scores

    def rigged(p):
        oos = original(p)
        # puntajes al reves: los extremos "baja" aciertan menos que la tasa base
        oos["p"] = np.where(oos["down"] == 1, 0.1, 0.9) + np.random.default_rng(0).normal(
            0, 0.01, len(oos)
        )
        return oos

    model._oos_scores = rigged
    metrics = model.fit_panel(panel)
    assert metrics["baja_disabled"] == 1.0
    assert model.baja_threshold == float("inf")


def test_uses_the_same_variables_as_xgboost_including_trend_and_ranks(trained) -> None:
    """Contexto entrena con las variables de XGBoost: tendencia y puesto vs pares."""
    assert trained._models[0].n_features_in_ == len(xgb_trend.ALL_FEATURES)
    assert "r_tz252" in xgb_trend.ALL_FEATURES
