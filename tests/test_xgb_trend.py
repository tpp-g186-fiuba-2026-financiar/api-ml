"""Tests del modelo XGBoost v2 (clasificador regularizado + rangos cross-seccionales)."""

import joblib
import numpy as np
import pandas as pd
import pytest

from src import train_xgb, xgb_trend
from src.context_trend import LEADERS, MIN_ROWS
from src.errors import DataUnavailableError, NotEnoughDataError, StaleArtifactError
from src.fair_value import TREND_FEATURES
from src.xgb_trend import (
    ALL_FEATURES,
    FEATURES,
    RANK_FEATURES,
    XGBConfig,
    XGBTrendModel,
    build_features,
    build_universe,
    rank_against_peers,
)

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
def universe(histories):
    return build_universe(histories)


@pytest.fixture(scope="module")
def trained(histories, universe) -> XGBTrendModel:
    model = XGBTrendModel(config=XGBConfig(oos_folds=2, n_seeds=1))
    model.fit(histories)
    model.universe_provider = lambda: universe
    return model


def test_feature_layout() -> None:
    assert ALL_FEATURES == FEATURES + RANK_FEATURES
    assert len(set(ALL_FEATURES)) == len(ALL_FEATURES)
    # la distancia a la tendencia (src.fair_value) entra con su rango vs pares
    assert set(TREND_FEATURES) <= set(FEATURES)
    assert {f"r_{t}" for t in TREND_FEATURES} <= set(RANK_FEATURES)


def test_panel_has_one_row_per_stock_and_day_with_ranks_inside_each_day(histories) -> None:
    model = XGBTrendModel()
    panel = model._panel(histories, build_universe(histories).market)
    assert len(panel) == len(histories) * (ROWS - model.config.horizon)
    day = panel.loc[panel.index.unique()[300]]
    assert day["r_t_ret20"].between(0, 1).all()
    assert day.sort_values("t_ret20")["r_t_ret20"].is_monotonic_increasing


def test_rank_against_peers_is_as_of_and_matches_training_ranks(histories, universe) -> None:
    """El rango al predecir es el mismo que el del panel de entrenamiento para esa fecha."""
    model = XGBTrendModel()
    panel = model._panel(histories, universe.market)
    date = panel.index.unique()[300]
    row = panel[(panel.index == date) & (panel["ticker"] == "BBAR")]
    own = build_features(histories["BBAR"].loc[:date], universe.market.loc[:date]).iloc[[-1]]
    ranked = rank_against_peers(own, universe)
    # el universo incluye a BBAR, que aparece dos veces (propio + par): el rango difiere
    # como mucho en un puesto de ~19 acciones
    assert np.allclose(ranked.iloc[0].to_numpy(), row[RANK_FEATURES].iloc[0].to_numpy(), atol=0.1)


def test_rank_against_peers_ignores_the_future(histories, universe) -> None:
    date = histories["BBAR"].index[400]
    own = build_features(histories["BBAR"].loc[:date], universe.market.loc[:date]).iloc[[-1]]
    full = rank_against_peers(own, universe)
    truncated = xgb_trend.Universe(
        universe.market.loc[:date], {s: f.loc[:date] for s, f in universe.features.items()}
    )
    pd.testing.assert_frame_equal(full, rank_against_peers(own, truncated))


def test_rank_needs_enough_peers(histories) -> None:
    few = build_universe({t: histories[t] for t in LEADERS[:3]} | {"X": histories["BBAR"]})
    own = build_features(histories["CRES"], few.market).iloc[[-1]]
    with pytest.raises(NotEnoughDataError):
        rank_against_peers(own, few)


def test_fit_records_out_of_sample_metrics(trained) -> None:
    m = trained.metrics
    assert 0.0 <= m["oos_auc"] <= 1.0
    assert m["oos_samples"] > 0
    assert trained.baja_threshold <= trained.alza_threshold
    assert {"alza_disabled", "baja_disabled"} <= set(m)
    assert trained.version.startswith("xgb-")
    assert trained.tickers


def test_predict_df_returns_a_coherent_trend_response(trained, histories) -> None:
    out = trained.predict_df(histories["BBAR"])
    assert out["signal"] in {"alza", "baja", "neutral"}
    assert out["horizon_days"] == trained.config.horizon
    assert out["probability_up"] == pytest.approx(1 - out["probability_down"], abs=1e-3)
    if out["signal"] == "neutral":
        assert out["predicted_close"] == pytest.approx(out["last_close"])


def test_signal_follows_the_daily_ranking(trained, histories) -> None:
    """Alza/baja segun el puesto de la accion entre las del universo ese dia."""
    df = histories["BBAR"]
    p = trained.probability_up(df)
    hi, lo = trained.day_cuts(df.index[-1])
    expected = "alza" if p >= hi else "baja" if p <= lo else "neutral"
    assert trained.predict_df(df)["signal"] == expected


def test_ranking_gives_signals_on_both_sides_every_day(trained) -> None:
    filas = trained.rank_universe()
    signals = [f["signal"] for f in filas]
    assert "alza" in signals and "baja" in signals
    # El de mas arriba del ranking es alza y el de mas abajo, baja.
    assert filas[0]["signal"] == "alza" and filas[-1]["signal"] == "baja"


def test_ranking_cuts_need_enough_stocks() -> None:
    from src.xgb_trend import ranking_cuts

    assert ranking_cuts([0.1, 0.2]) is None
    hi, lo = ranking_cuts(np.linspace(0, 1, 11))
    assert hi == pytest.approx(0.8) and lo == pytest.approx(0.2)


def test_short_or_stale_history_is_refused(trained) -> None:
    with pytest.raises(NotEnoughDataError):
        trained.probability_up(make_history(rows=MIN_ROWS - 10))
    with pytest.raises(NotEnoughDataError):
        trained.probability_up(make_history(flat=True))


def test_untrained_model_cannot_predict_or_save(tmp_path) -> None:
    model = XGBTrendModel()
    with pytest.raises(NotEnoughDataError):
        model.probability_up(make_history())
    with pytest.raises(NotEnoughDataError):
        model.save(tmp_path / "x.pkl")
    assert model.version == "untrained"


def test_fit_rejects_too_little_data() -> None:
    few = {t: make_history(rows=MIN_ROWS + 5, seed=i) for i, t in enumerate(LEADERS)}
    with pytest.raises(NotEnoughDataError):
        XGBTrendModel().fit(few)


def test_save_and_load_roundtrip(trained, histories, universe, tmp_path) -> None:
    path = tmp_path / "xgb.pkl"
    trained.save(path)
    loaded = XGBTrendModel.load(path)
    loaded.universe_provider = lambda: universe
    assert loaded.version == trained.version
    df = histories["CRES"]
    assert loaded.probability_up(df) == pytest.approx(trained.probability_up(df))
    assert loaded.calibration == trained.calibration


def test_prediction_reports_calibrated_probability_and_base_rate(trained, histories) -> None:
    out = trained.predict_df(histories["CRES"])
    assert 0 < out["probability_up_calibrated"] < 1
    assert out["base_up_rate"] == trained.calibration["base_up"]
    fila = trained.rank_universe()[0]
    assert "probability_up_calibrated" in fila and "base_up_rate" in fila


def test_artifacts_without_calibration_still_predict(trained, histories) -> None:
    trained_cal = trained.calibration
    trained.calibration = None
    try:
        out = trained.predict_df(histories["CRES"])
    finally:
        trained.calibration = trained_cal
    assert out["probability_up_calibrated"] is None
    assert out["base_up_rate"] == trained.metrics["oos_up_rate"]


def test_load_refuses_v1_artifacts(trained, tmp_path) -> None:
    path = tmp_path / "old.pkl"
    trained.save(path)
    blob = joblib.load(path)
    blob["feature_names"] = ["ret", "vol"]
    joblib.dump(blob, path)
    with pytest.raises(StaleArtifactError):
        XGBTrendModel.load(path)


def test_load_refuses_artifacts_with_old_format(trained, tmp_path) -> None:
    path = tmp_path / "old.pkl"
    trained.save(path)
    blob = joblib.load(path)
    del blob["alza_threshold"]
    joblib.dump(blob, path)
    with pytest.raises(StaleArtifactError):
        XGBTrendModel.load(path)


def test_todays_universe_is_cached_per_day_and_survives_missing_tickers(
    monkeypatch, histories
) -> None:
    calls = []

    def fake_fetch(symbol, days):
        calls.append(symbol)
        if symbol == "GGAL":
            raise DataUnavailableError("sin datos")
        if symbol not in histories:
            raise DataUnavailableError("fuera del universo de test")
        return histories[symbol]

    monkeypatch.setattr(xgb_trend, "fetch_history", fake_fetch)
    xgb_trend._universe_cache.clear()
    first = xgb_trend.todays_universe()
    n = len(calls)
    assert "GGAL" not in first.features
    assert xgb_trend.todays_universe() is first
    assert len(calls) == n
    xgb_trend._universe_cache.clear()


def test_train_script_saves_only_when_auc_is_good(monkeypatch, histories, tmp_path, capsys) -> None:
    monkeypatch.setattr(train_xgb, "fetch_history", lambda s, d: histories[s])
    monkeypatch.setattr(train_xgb, "UNIVERSE", tuple(histories))
    out = tmp_path / "m.pkl"
    monkeypatch.setattr("sys.argv", ["train", "--out", str(out), "--min-auc", "0.99"])
    train_xgb.main()
    assert not out.exists()
    monkeypatch.setattr("sys.argv", ["train", "--out", str(out), "--min-auc", "0.0"])
    train_xgb.main()
    assert out.exists()
    assert "Modelo guardado" in capsys.readouterr().out


def test_a_side_without_edge_out_of_sample_is_never_emitted(histories) -> None:
    """Si 'alza' (o 'baja') no le gana a su tasa base fuera de muestra, queda apagada."""
    model = XGBTrendModel(config=XGBConfig(oos_folds=2, n_seeds=1))
    panel = model._panel(histories, build_universe(histories).market)
    original = model._oos_scores

    def rigged(p):
        oos = original(p)
        # puntajes al reves: los extremos aciertan menos que la tasa base
        noise = np.random.default_rng(0).normal(0, 0.01, len(oos))
        oos["p"] = np.where(oos["up"] == 1, 0.1, 0.9) + noise
        return oos

    model._oos_scores = rigged
    metrics = model.fit_panel(panel)
    assert metrics["alza_disabled"] == 1.0
    assert model.alza_threshold == float("inf")


def test_explain_contributions_add_up_to_the_tree_probability(trained, histories) -> None:
    """Con una semilla, sigmoid(base_value + sum(contribuciones)) == P(sube) de XGBoost."""
    df = histories["BBAR"]
    out = trained.explain(df)
    assert out["as_of"] == df.index[-1].strftime("%Y-%m-%d")
    assert out["probability_up"] == pytest.approx(trained.probability_up(df), abs=1e-4)
    trees = [m for m in trained._models if xgb_trend._is_tree(m)]
    p_trees = float(xgb_trend._score(trees, trained._latest_row(df).to_numpy())[0])
    contributions = out["contributions"]
    assert [c["feature"] for c in contributions] != ALL_FEATURES  # ordenadas por |aporte|
    assert sorted(c["feature"] for c in contributions) == sorted(ALL_FEATURES)
    magnitudes = [abs(c["contribution"]) for c in contributions]
    assert magnitudes == sorted(magnitudes, reverse=True)
    margin = out["base_value"] + sum(c["contribution"] for c in contributions)
    assert 1 / (1 + np.exp(-margin)) == pytest.approx(p_trees, abs=1e-3)


def test_explain_reports_the_same_signal_as_the_prediction(trained, histories) -> None:
    for symbol in ("BBAR", "CRES", "GGAL"):
        out = trained.explain(histories[symbol])
        assert out["signal"] == trained.predict_df(histories[symbol])["signal"]
        assert out["base_up_rate"] == trained.metrics["oos_up_rate"]
    model = XGBTrendModel(alza_threshold=float("inf"), baja_threshold=0.3)
    assert model.signal_for(0.99) == "neutral"
    assert model.signal_for(0.3) == "baja"
    assert xgb_trend._finite_or_none(float("inf")) is None


def test_ensemble_blends_trees_with_a_logistic_half(trained, histories) -> None:
    trees = [m for m in trained._models if xgb_trend._is_tree(m)]
    linear = [m for m in trained._models if not xgb_trend._is_tree(m)]
    assert len(trees) == trained.config.n_seeds and len(linear) == 1
    row = trained._latest_row(histories["BBAR"]).to_numpy()
    expected = (trees[0].predict_proba(row)[0, 1] + linear[0].predict_proba(row)[0, 1]) / 2
    assert trained.probability_up(histories["BBAR"]) == pytest.approx(expected, abs=1e-9)


def test_logistic_half_tolerates_missing_and_infinite_values(trained) -> None:
    row = np.full((1, len(ALL_FEATURES)), np.nan)
    row[0, 0] = np.inf
    p = xgb_trend._score(trained._models, row)[0]
    assert 0.0 < p < 1.0


def test_artifacts_without_logistic_score_with_trees_only(trained, histories) -> None:
    trees = [m for m in trained._models if xgb_trend._is_tree(m)]
    row = trained._latest_row(histories["BBAR"]).to_numpy()
    assert xgb_trend._score(trees, row)[0] == pytest.approx(trees[0].predict_proba(row)[0, 1])


def test_explain_labels_are_human_readable(trained, histories) -> None:
    by_feature = {c["feature"]: c for c in trained.explain(histories["CRES"])["contributions"]}
    assert by_feature["t_ret20"]["label"] == "Variación del último mes"
    rank_label = "Rendimiento vs. el mercado (mes) (vs. otras acciones)"
    assert by_feature["r_m_rel20"]["label"] == rank_label
    assert all(c["label"] != c["feature"] for c in by_feature.values())


def test_explain_refuses_short_history(trained) -> None:
    with pytest.raises(NotEnoughDataError):
        trained.explain(make_history(rows=MIN_ROWS - 10))


def test_every_feature_has_a_plain_explanation() -> None:
    from src.feature_labels import feature_help, feature_unit

    for name in xgb_trend.ALL_FEATURES:
        assert feature_help(name), name
    assert feature_unit("t_ret20") == "pct"
    assert feature_unit("r_t_ret20") == "rank"
    assert feature_unit("t_death_cross") == "flag"
    assert feature_unit("m_beta60") == "num"


def test_rank_universe_orders_every_stock_with_a_score(trained, universe) -> None:
    filas = trained.rank_universe()
    assert {f["symbol"] for f in filas} <= set(universe.features)
    probs = [f["probability_up"] for f in filas]
    assert probs == sorted(probs, reverse=True)
    assert [f["puesto"] for f in filas] == list(range(1, len(filas) + 1))
    assert filas[0]["puntaje"] == 10 and filas[-1]["puntaje"] >= 1
    assert all(f["signal"] in {"alza", "baja", "neutral"} for f in filas)
