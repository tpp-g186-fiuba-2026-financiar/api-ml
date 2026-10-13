import json

import pandas as pd
import pytest

import src.simulacion.snapshot as snapshot_module
from src.errors import DataUnavailableError
from src.simulacion.snapshot import Snapshot, download, fetch_merval
from tests.simulacion.conftest import TICKER_CORTO, TICKERS


def test_load_y_cortes_por_fecha(snapshot) -> None:
    assert snapshot.tickers == [*TICKERS, TICKER_CORTO]
    dias = snapshot.trading_days()
    corte = dias[100]

    historia = snapshot.history("GGAL", corte)
    assert historia.index[-1] == corte
    assert len(snapshot.history("GGAL", corte, days=20)) == 20
    assert snapshot.rate_until(corte).index[-1] <= corte
    assert snapshot.close_on("GGAL", corte) == historia["close"].iloc[-1]
    assert snapshot.close_on("NOEXISTE", corte) is None
    assert snapshot.close_on(TICKER_CORTO, dias[0]) is None
    assert snapshot.merval_on(corte) is not None
    assert snapshot.ccl_on(corte) is not None

    macro = snapshot.macro_frame(corte)
    # La serie entra con su retraso de publicacion: nunca un dato posterior al corte.
    assert macro.index[-1] == corte
    assert macro["ccl"].iloc[-1] <= snapshot.macro_raw["ccl"].loc[:corte].iloc[-1]


def test_snapshot_sin_tasa_ni_merval(snapshot) -> None:
    vacio = Snapshot(prices=snapshot.prices, tickers=[], rate=None, macro_raw={}, meta={})
    dia = snapshot.trading_days()[-1]
    assert vacio.rate_until(dia) is None
    assert vacio.merval_on(dia) is None
    assert vacio.ccl_on(dia) is None


def test_load_sin_snapshot(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        Snapshot.load(tmp_path)


def test_download(monkeypatch, tmp_path, snapshot) -> None:
    def fetch_history(symbol, days):
        if symbol == "FALLA":
            raise DataUnavailableError("sin datos")
        # Yahoo marca cada rueda con la hora de apertura: se normaliza a la fecha.
        df = snapshot.prices[symbol].copy()
        df.index = df.index + pd.Timedelta(hours=13)
        return df

    def fetch_rate(source, series, days):
        raise DataUnavailableError("sin tasa")

    def fetch_series(route, series):
        if series == "CCL":
            return snapshot.macro_raw["ccl"]
        raise DataUnavailableError("sin serie")

    monkeypatch.setattr(snapshot_module, "fetch_available_tickers", lambda: ["GGAL", "FALLA"])
    monkeypatch.setattr(snapshot_module, "fetch_history", fetch_history)
    monkeypatch.setattr(snapshot_module, "fetch_interest_rate", fetch_rate)
    monkeypatch.setattr(snapshot_module, "fetch_merval", lambda days: fetch_history("BMA", days))
    monkeypatch.setattr(snapshot_module, "fetch_collector_series", fetch_series)

    meta = download(tmp_path, days=100)
    assert meta["tickers"] == ["GGAL"]
    assert "FALLA" in meta["failed"]
    assert meta["macro_rate"] is False
    assert meta["macro_series"] == ["ccl"]

    cargado = Snapshot.load(tmp_path)
    assert cargado.rate is None
    assert cargado.prices["GGAL"].index[0] == snapshot.prices["GGAL"].index[0]
    assert json.loads((tmp_path / "meta.json").read_text())["days"] == 100


def test_download_sin_merval(monkeypatch, tmp_path) -> None:
    def falla(*args):
        raise DataUnavailableError("x")

    monkeypatch.setattr(snapshot_module, "fetch_available_tickers", lambda: [])
    monkeypatch.setattr(snapshot_module, "fetch_interest_rate", falla)
    monkeypatch.setattr(snapshot_module, "fetch_merval", falla)
    monkeypatch.setattr(snapshot_module, "fetch_collector_series", falla)
    assert download(tmp_path)["merval"] is False


def test_fetch_merval_sin_sufijo(monkeypatch) -> None:
    sufijos = []
    monkeypatch.setattr(
        snapshot_module,
        "fetch_history_yahoo",
        lambda symbol, days: sufijos.append(snapshot_module.settings.market_suffix) or symbol,
    )
    original = snapshot_module.settings.market_suffix
    assert fetch_merval(10) == "^MERV"
    assert sufijos == [""]
    assert snapshot_module.settings.market_suffix == original


def test_main(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setattr(snapshot_module, "download", lambda directory, days: {"tickers": ["A"]})
    monkeypatch.setattr("sys.argv", ["snapshot", "--dir", str(tmp_path)])
    snapshot_module.main()
    assert "1 tickers" in capsys.readouterr().out
