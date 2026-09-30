import datetime as dt
import json

import pytest

from src import quality_report as qr


def _rec(model, symbol, as_of, signal, pred, real, version="v1"):
    return {
        "symbol": symbol,
        "model": model,
        "model_version": version,
        "as_of": as_of,
        "signal": signal,
        "horizon_days": 5,
        "predicted_log_return": pred,
        "realized_log_return": real,
    }


def _hits_and_misses(model, hits, misses, as_of="2026-08-03", version="v1", symbol="GGAL"):
    """``hits`` predicciones que aciertan la direccion y ``misses`` que no."""
    ok = [_rec(model, symbol, as_of, "alza", 0.02, 0.03, version) for _ in range(hits)]
    bad = [_rec(model, symbol, as_of, "alza", 0.02, -0.03, version) for _ in range(misses)]
    return ok + bad


def _backtest(accuracy, tickers=("GGAL",), model="xgboost"):
    overall = {"n_predictions": 100, "directional_accuracy": accuracy, "signal_hit_rate": 0.9}
    return {
        "run_at": "2026-09-03T00:00:00Z",
        "params": {"tickers": list(tickers)},
        "results": {model: {"overall": overall}},
    }


def test_wilson_interval_narrows_with_more_data() -> None:
    small = qr.wilson_interval(6, 10)
    large = qr.wilson_interval(600, 1000)
    assert (small[1] - small[0]) > (large[1] - large[0])
    assert 0.0 <= small[0] <= 0.6 <= small[1] <= 1.0


def test_wilson_interval_handles_empty_sample() -> None:
    assert qr.wilson_interval(0, 0) == (0.0, 1.0)


def test_iso_week_groups_monday_to_sunday() -> None:
    assert qr.iso_week("2026-08-03") == qr.iso_week("2026-08-09")
    assert qr.iso_week("2026-08-09") != qr.iso_week("2026-08-10")


def test_filter_since_drops_older_records() -> None:
    resolved = [
        _rec("m", "A", "2026-07-30", "alza", 0.1, 0.1),
        _rec("m", "A", "2026-08-05", "alza", 0.1, 0.1),
    ]
    assert len(qr.filter_since(resolved, "2026-08-01")) == 1
    assert len(qr.filter_since(resolved, None)) == 2


def test_weekly_evolution_reports_cumulative_accuracy() -> None:
    resolved = _hits_and_misses("m", 10, 0, as_of="2026-08-03") + _hits_and_misses(
        "m", 0, 10, as_of="2026-08-10"
    )
    rows = qr.weekly_evolution(resolved, "m")
    assert [r["week"] for r in rows] == ["2026-W32", "2026-W33"]
    assert rows[0]["week_stats"]["directional_accuracy"] == 1.0
    assert rows[1]["week_stats"]["directional_accuracy"] == 0.0
    assert rows[1]["cumulative_accuracy"] == 0.5


def test_gap_detects_backtest_overestimation() -> None:
    # Paper: 50/100 = 50%. Backtest: 76% -> brecha 26 pp, por encima del umbral.
    resolved = _hits_and_misses("xgboost", 50, 50)
    (row,) = qr.gap_table(_backtest(0.76), resolved, min_n=30, gap_threshold=0.10)
    assert row["gap_accuracy"] == pytest.approx(0.26)
    assert row["gap_alert"] is True
    assert row["paper_vs_chance"] == "indistinguible del azar"


def test_gap_does_not_alert_when_paper_confirms_backtest() -> None:
    resolved = _hits_and_misses("xgboost", 76, 24)
    (row,) = qr.gap_table(_backtest(0.76), resolved, min_n=30, gap_threshold=0.10)
    assert row["gap_alert"] is False
    assert row["paper_vs_chance"] == "mejor que el azar"


def test_gap_does_not_alert_with_too_little_paper_data() -> None:
    resolved = _hits_and_misses("xgboost", 2, 8)  # 20% pero solo 10 muestras
    (row,) = qr.gap_table(_backtest(0.90), resolved, min_n=30, gap_threshold=0.10)
    assert row["gap_alert"] is False
    assert row["paper_vs_chance"] == "pocos datos"


def test_gap_pairs_local_and_modal_variants_but_ranks_only_local() -> None:
    resolved = _hits_and_misses("xgboost", 50, 50) + _hits_and_misses("xgboost-modal", 60, 40)
    rows = qr.gap_table(_backtest(0.76), resolved, min_n=30, gap_threshold=0.10)
    by_name = {r["paper_model"]: r for r in rows}
    assert set(by_name) == {"xgboost", "xgboost-modal"}
    assert by_name["xgboost"]["rank_paper"] == 1
    assert by_name["xgboost-modal"]["rank_paper"] is None


def test_gap_ranking_shows_backtest_winner_losing_in_paper() -> None:
    backtest = _backtest(0.76)
    backtest["results"]["lstm"] = {
        "overall": {"directional_accuracy": 0.51, "signal_hit_rate": 0.5}
    }
    resolved = _hits_and_misses("xgboost", 40, 60) + _hits_and_misses("lstm", 55, 45)
    rows = {r["paper_model"]: r for r in qr.gap_table(backtest, resolved, 30, 0.10)}
    assert (rows["xgboost"]["rank_backtest"], rows["xgboost"]["rank_paper"]) == (1, 2)
    assert (rows["lstm"]["rank_backtest"], rows["lstm"]["rank_paper"]) == (2, 1)


def test_version_comparison_flags_significant_change_only() -> None:
    resolved = (
        _hits_and_misses("m", 30, 70, as_of="2026-08-03", version="v1")
        + _hits_and_misses("m", 70, 30, as_of="2026-08-17", version="v2")  # clara mejora
        + _hits_and_misses("m", 72, 28, as_of="2026-08-24", version="v3")  # ruido vs v2
    )
    rows = qr.version_comparison(resolved, "m", min_n=30)
    assert [r["version"] for r in rows] == ["v1", "v2", "v3"]
    assert [r["verdict"] for r in rows] == ["primera", "mejoro", "sin diferencia significativa"]


def test_version_comparison_detects_regression_and_skips_small_versions() -> None:
    resolved = (
        _hits_and_misses("m", 70, 30, as_of="2026-08-03", version="v1")
        + _hits_and_misses("m", 3, 2, as_of="2026-08-10", version="v-chica")
        + _hits_and_misses("m", 30, 70, as_of="2026-08-17", version="v2")
    )
    rows = qr.version_comparison(resolved, "m", min_n=30)
    assert [r["version"] for r in rows] == ["v1", "v2"]
    assert rows[1]["verdict"] == "empeoro"


def test_update_history_replaces_same_day_and_keeps_order(tmp_path) -> None:
    path = tmp_path / "history.json"
    ledger = {"resolved": _hits_and_misses("xgboost", 50, 50), "pending": []}

    def snap(day):
        return qr.build_snapshot(ledger, _backtest(0.76), dt.date(2026, 9, day), None, 30, 0.10)

    path.write_text(json.dumps(qr.update_history(path, snap(2))))
    path.write_text(json.dumps(qr.update_history(path, snap(1))))
    history = qr.update_history(path, snap(2))
    assert [h["date"] for h in history] == ["2026-09-01", "2026-09-02"]
    assert history[0]["paper_accuracy"]["xgboost"] == 0.5


def test_render_markdown_contains_all_sections_and_alert() -> None:
    ledger = {"resolved": _hits_and_misses("xgboost", 50, 50), "pending": []}
    snapshot = qr.build_snapshot(ledger, _backtest(0.76), dt.date(2026, 9, 29), None, 30, 0.10)
    text = qr.render_markdown(snapshot, ledger["resolved"], 30, 0.10)
    assert "## 1. Brecha backtest vs paper trading" in text
    assert "## 2. Evolucion semanal" in text
    assert "## 3. Que se probo y si mejoro" in text
    assert "xgboost: el backtest sobreestima" in text


def test_main_writes_report_and_history_and_strict_fails_on_alert(tmp_path, monkeypatch) -> None:
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps({"resolved": _hits_and_misses("xgboost", 50, 50), "pending": []}))
    backtest = tmp_path / "backtest.json"
    backtest.write_text(json.dumps(_backtest(0.76)))
    argv = [
        "quality_report",
        "--ledger",
        str(ledger),
        "--backtest",
        str(backtest),
        "--report",
        str(tmp_path / "report.md"),
        "--history",
        str(tmp_path / "history.json"),
    ]
    monkeypatch.setattr("sys.argv", argv)
    qr.main()
    assert (tmp_path / "report.md").exists()
    assert json.loads((tmp_path / "history.json").read_text())[0]["gap_accuracy"]["xgboost"] > 0.1

    monkeypatch.setattr("sys.argv", argv + ["--strict"])
    with pytest.raises(SystemExit) as exc:
        qr.main()
    assert exc.value.code == 1
