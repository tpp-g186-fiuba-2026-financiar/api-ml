"""Tests del modelo macro."""

import sys

import numpy as np
import pandas as pd
import pytest

from src import macro_trend, train_macro
from src.errors import NotEnoughDataError, StaleArtifactError
from src.macro_data import SERIES
from src.macro_trend import (
    FEATURES,
    MIN_ROWS,
    MacroConfig,
    MacroTrendModel,
    build_features,
    event_features,
    macro_features,
    technical_features,
)

ROWS = 700
TICKERS = ["GGAL", "YPFD", "BMA", "PAMP", "TECO2", "ALUA", "BBAR", "CEPU"]


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


def make_macro(start: str = "2020-01-01", days: int = 1300, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=days, freq="D")
    walk = lambda base, vol: base * np.exp(np.cumsum(rng.normal(0, vol, days)))  # noqa: E731
    ccl = walk(1000.0, 0.004)
    return pd.DataFrame(
        {
            "ccl": ccl,
            "mep": ccl * 0.99,
            "oficial": ccl * 0.9,
            "mayorista": ccl * 0.88,
            "blue": ccl * 1.01,
            "riesgo_pais": walk(800.0, 0.01),
            "reservas": walk(40_000.0, 0.003),
            "badlar": 30 + np.cumsum(rng.normal(0, 0.2, days)),
            "base_monetaria": walk(1e7, 0.002),
        },
        index=idx,
    )


@pytest.fixture(scope="module")
def histories() -> dict[str, pd.DataFrame]:
    return {t: make_history(seed=i) for i, t in enumerate(TICKERS)}


@pytest.fixture(scope="module")
def macro() -> pd.DataFrame:
    return make_macro()


@pytest.fixture(scope="module")
def trained(histories, macro) -> MacroTrendModel:
    model = MacroTrendModel(config=MacroConfig(oos_folds=2))
    model.fit(histories, macro)
    model.macro_provider = lambda: macro
    return model


def test_feature_groups_and_order() -> None:
    assert len(FEATURES) == 65 and len(set(FEATURES)) == 65
    assert (
        FEATURES
        == macro_trend.TECH_FEATURES + macro_trend.MACRO_FEATURES + macro_trend.EVENT_FEATURES
    )
    assert "e_month" not in FEATURES and "e_dow" not in FEATURES  # estacionalidad = sobreajuste


def test_technical_features_do_not_look_ahead() -> None:
    df = make_history()
    base = technical_features(df)
    changed = df.copy()
    changed.iloc[-30:, changed.columns.get_loc("close")] *= 3.0
    pd.testing.assert_frame_equal(base.iloc[:-31], technical_features(changed).iloc[:-31])


def test_macro_features_do_not_look_ahead(macro) -> None:
    index = make_history().index
    base = macro_features(macro, index)
    changed = macro.copy()
    changed.iloc[-60:] = changed.iloc[-60:] * 3.0
    cutoff = macro.index[-61]
    pd.testing.assert_frame_equal(base.loc[:cutoff], macro_features(changed, index).loc[:cutoff])


def test_macro_features_tolerate_missing_optional_series(macro) -> None:
    reduced = macro.drop(columns=["mep", "blue", "mayorista", "reservas", "base_monetaria"])
    feats = macro_features(reduced, make_history().index)
    assert feats["a_mep_ccl"].isna().all() and feats["a_res20"].isna().all()
    assert feats["a_rp"].notna().any() and feats["a_badlar"].notna().any()


def test_event_features_follow_election_dates() -> None:
    index = pd.DatetimeIndex(["2023-10-18", "2023-10-22", "2023-11-30", "2026-02-02"])
    e = event_features(index)
    assert e.loc["2023-10-18", "e_days_to_el"] == 4
    assert e.loc["2023-10-22", "e_days_to_el"] == 0 and e.loc["2023-10-22", "e_el_window"] == 1
    assert e.loc["2023-11-30", "e_days_since_el"] == 11  # la ultima fue el 19/11
    assert e.loc["2026-02-02", "e_days_to_el"] == 90  # tope
    assert e.loc["2023-11-30", "e_aguinaldo"] == 0 and e.loc["2023-10-18", "e_aguinaldo"] == 0


def test_build_features_has_the_documented_columns(histories, macro) -> None:
    feats = build_features(histories["GGAL"], macro)
    assert list(feats.columns) == FEATURES and len(feats) == ROWS


def test_fit_reports_out_of_sample_metrics_and_thresholds(trained) -> None:
    m = trained.metrics
    assert m["oos_samples"] > 1000
    assert 0.0 <= m["oos_auc"] <= 1.0
    assert trained.alza_threshold < trained.baja_threshold
    assert trained.baja_move <= 0 <= trained.alza_move
    assert trained.version.startswith("macro-") and trained.tickers == sorted(TICKERS)


def test_predict_returns_a_trend_response_compatible_dict(trained, histories) -> None:
    out = trained.predict_df(histories["GGAL"])
    assert out["signal"] in {"alza", "baja", "neutral"}
    assert out["horizon_days"] == 20
    assert out["probability_down"] + out["probability_up"] == pytest.approx(1.0, abs=1e-3)
    assert {"last_close", "predicted_close", "rsi", "condition", "as_of"} <= set(out)


def test_signals_follow_the_thresholds_and_the_projected_price(trained, histories) -> None:
    df = histories["YPFD"]
    p = trained.probability_down(df)
    last = float(df["close"].iloc[-1])
    saved = (trained.baja_threshold, trained.alza_threshold)
    try:
        trained.baja_threshold, trained.alza_threshold = p - 1e-6, p - 2e-6  # P >= baja
        baja = trained.predict_df(df)
        trained.baja_threshold, trained.alza_threshold = p + 2e-6, p + 1e-6  # P <= alza
        alza = trained.predict_df(df)
        trained.baja_threshold, trained.alza_threshold = p + 1e-6, p - 1e-6  # en el medio
        neutral = trained.predict_df(df)
    finally:
        trained.baja_threshold, trained.alza_threshold = saved
    assert (baja["signal"], alza["signal"], neutral["signal"]) == ("baja", "alza", "neutral")
    assert baja["predicted_close"] <= last <= alza["predicted_close"]
    assert neutral["predicted_close"] == pytest.approx(last, rel=1e-4)  # sin vista


def test_predict_refuses_stale_short_or_untrained(trained) -> None:
    with pytest.raises(NotEnoughDataError, match="movimiento"):
        trained.predict_df(make_history(flat=True))
    with pytest.raises(NotEnoughDataError, match=str(MIN_ROWS)):
        trained.predict_df(make_history(rows=100))
    with pytest.raises(NotEnoughDataError):
        MacroTrendModel().probability_down(make_history())
    with pytest.raises(NotEnoughDataError):
        MacroTrendModel().save("/tmp/never.pkl")


def test_fit_rejects_empty_or_tiny_datasets(macro) -> None:
    with pytest.raises(NotEnoughDataError):
        MacroTrendModel().fit({"A3": make_history(flat=True)}, macro)
    with pytest.raises(NotEnoughDataError):
        MacroTrendModel().fit({"GGAL": make_history(rows=MIN_ROWS + 30)}, macro)


def test_save_and_load_roundtrip(trained, histories, tmp_path) -> None:
    path = tmp_path / "macro.pkl"
    trained.save(path)
    loaded = MacroTrendModel.load(path)
    loaded.macro_provider = trained.macro_provider
    df = histories["BMA"]
    assert loaded.version == trained.version
    assert loaded.probability_down(df) == pytest.approx(trained.probability_down(df))
    assert (loaded.baja_threshold, loaded.alza_threshold) == (
        trained.baja_threshold,
        trained.alza_threshold,
    )


def test_load_rejects_artifacts_trained_with_other_features(trained, tmp_path) -> None:
    import joblib

    path = tmp_path / "old.pkl"
    trained.save(path)
    blob = joblib.load(path)
    blob["feature_names"] = FEATURES[:-1]
    joblib.dump(blob, path)
    with pytest.raises(StaleArtifactError):
        MacroTrendModel.load(path)


def test_load_reports_an_outdated_config_as_stale_instead_of_crashing(trained, tmp_path) -> None:
    import joblib

    path = tmp_path / "old_config.pkl"
    trained.save(path)
    blob = joblib.load(path)
    blob["config"]["val_fraction"] = 0.15  # campo de una version anterior
    joblib.dump(blob, path)
    with pytest.raises(StaleArtifactError, match="desactualizado"):
        MacroTrendModel.load(path)
    del blob["config"]["val_fraction"]
    del blob["baja_move"]
    joblib.dump(blob, path)
    with pytest.raises(StaleArtifactError, match="desactualizado"):
        MacroTrendModel.load(path)


def test_registry_includes_the_macro_model_and_reports_data_failures(
    trained, histories, monkeypatch, tmp_path
) -> None:
    from src.config import settings
    from src.errors import DataUnavailableError
    from src.registry import build_registry

    path = tmp_path / "macro.pkl"
    trained.save(path)
    monkeypatch.setattr(settings, "macro_model_path", str(path))
    reg = build_registry(750)
    assert "macro" in reg.names()
    service = reg.resolve("macro")
    service.load()
    assert service.is_loaded

    def macro_down():
        raise DataUnavailableError("data-colector caido")

    service._model.macro_provider = macro_down
    with pytest.raises(DataUnavailableError, match="caido"):
        service.predict_on(histories["GGAL"], "GGAL")


def test_series_catalog_matches_what_the_data_collector_serves() -> None:
    routes = {route for route, _, _ in SERIES.values()}
    assert routes == {"macro/argdatos", "interest-rate/ar"}
    assert all(lag in (2, 5) for _, _, lag in SERIES.values())


def test_train_macro_entrypoint_trains_saves_and_respects_the_auc_gate(
    monkeypatch, histories, macro, tmp_path, capsys
) -> None:
    out = tmp_path / "macro.pkl"
    monkeypatch.setattr(train_macro, "fetch_history", lambda s, d: histories[s])
    monkeypatch.setattr(train_macro, "fetch_argentina_macro", lambda: macro)
    monkeypatch.setattr(
        train_macro,
        "MacroConfig",
        lambda horizon: MacroConfig(horizon=horizon, oos_folds=2),
    )
    argv = ["train_macro", "--tickers", *TICKERS, "--out", str(out)]

    monkeypatch.setattr(sys, "argv", argv + ["--min-auc", "0.99"])
    train_macro.main()
    assert not out.exists()  # no supera el gate: no se guarda
    assert "no se guarda" in capsys.readouterr().out

    monkeypatch.setattr(sys, "argv", argv + ["--min-auc", "0.0"])
    train_macro.main()
    assert out.exists()
    assert MacroTrendModel.load(out).tickers == sorted(TICKERS)


def test_train_macro_entrypoint_aborts_without_data(monkeypatch, tmp_path) -> None:
    from src.errors import DataUnavailableError

    def no_data(symbol: str, days: int):
        raise DataUnavailableError("sin datos")

    monkeypatch.setattr(train_macro, "fetch_history", no_data)
    monkeypatch.setattr(sys, "argv", ["train_macro", "--out", str(tmp_path / "x.pkl")])
    with pytest.raises(SystemExit):
        train_macro.main()
