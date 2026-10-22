"""Snapshot diario del comparador (sin red)."""

import datetime as dt

import pytest

from src.compare_snapshot import is_fresh, last_scheduled_run, load_snapshot, save_snapshot
from src.registry import TrendRegistry

UTC = dt.UTC


def _at(day: int, hour: int, minute: int = 0) -> dt.datetime:
    # Octubre 2026: el 19 es lunes, el 23 viernes.
    return dt.datetime(2026, 10, day, hour, minute, tzinfo=UTC)


@pytest.mark.parametrize(
    ("now", "expected"),
    [
        (_at(21, 22), _at(21, 21)),  # miercoles despues de la corrida
        (_at(21, 10), _at(20, 21)),  # miercoles antes de la corrida
        (_at(24, 12), _at(23, 21)),  # sabado -> viernes
        (_at(26, 10), _at(23, 21)),  # lunes a la manana -> viernes
    ],
)
def test_last_scheduled_run(now, expected) -> None:
    assert last_scheduled_run(now) == expected


def test_snapshot_stays_fresh_until_next_run_plus_grace() -> None:
    generated = _at(20, 21, 30)  # martes, corrida normal
    assert is_fresh(generated, _at(21, 15))  # miercoles de dia
    assert is_fresh(generated, _at(21, 23))  # la corrida del miercoles todavia puede venir
    assert not is_fresh(generated, _at(22, 1))  # paso la corrida del miercoles + margen


def test_friday_snapshot_covers_the_weekend() -> None:
    generated = _at(23, 21, 15)
    assert is_fresh(generated, _at(26, 18))  # lunes antes de la corrida


def test_save_and_load_roundtrip(tmp_path) -> None:
    path = tmp_path / "snap.json"
    save_snapshot(path, {"GGAL": {"symbol": "GGAL"}}, now=_at(21, 21, 5))

    loaded = load_snapshot(path)

    assert loaded["generated_at"] == _at(21, 21, 5)
    assert loaded["symbols"] == {"GGAL": {"symbol": "GGAL"}}


def test_load_snapshot_tolerates_missing_or_broken_file(tmp_path) -> None:
    assert load_snapshot(tmp_path / "no-existe.json") is None
    broken = tmp_path / "roto.json"
    broken.write_text("{no es json")
    assert load_snapshot(broken) is None


def test_compare_serves_fresh_snapshot_without_fetching(tmp_path, monkeypatch) -> None:
    path = tmp_path / "snap.json"
    entry = {"symbol": "GGAL", "as_of": "2026-10-21", "predictions": {"lstm": {"signal": "alza"}}}
    save_snapshot(path, {"GGAL": entry})

    def _no_fetch(*_args, **_kwargs):
        raise AssertionError("no deberia descargar historico")

    monkeypatch.setattr("src.registry._fetch_history_with_macro", _no_fetch)
    registry = TrendRegistry(history_days=750, snapshot_path=path)

    result = registry.compare("ggal")

    assert result["source"] == "snapshot"
    assert result["predictions"] == entry["predictions"]


def test_compare_goes_live_when_snapshot_is_stale_or_skipped(tmp_path, monkeypatch) -> None:
    path = tmp_path / "snap.json"
    save_snapshot(path, {"GGAL": {"symbol": "GGAL", "predictions": {}}}, now=_at(1, 21))
    fetched = []

    def _fetch(symbol, _days):
        fetched.append(symbol)
        raise RuntimeError("descarga en vivo")

    monkeypatch.setattr("src.registry._fetch_history_with_macro", _fetch)
    registry = TrendRegistry(history_days=750, snapshot_path=path)

    with pytest.raises(RuntimeError):
        registry.compare("GGAL")
    save_snapshot(path, {"GGAL": {"symbol": "GGAL", "predictions": {}}})
    with pytest.raises(RuntimeError):
        registry.compare("GGAL", use_snapshot=False)

    assert fetched == ["GGAL", "GGAL"]
