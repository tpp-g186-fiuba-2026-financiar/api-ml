import json

import pandas as pd
import pytest

import src.simulacion.un_modelo as um
from src.arima_trend import ArimaTrendModel
from src.context_trend import ContextTrendModel
from src.errors import NotEnoughDataError
from src.lstm import TrendModel
from src.macro_trend import MacroTrendModel
from src.simulacion import por_ticker
from src.simulacion.busqueda import MODELOS
from src.simulacion.entrenamiento import historias_contexto, historias_macro, universo_al
from src.xgb_trend import XGBTrendModel
from tests.simulacion.conftest import TICKER_CORTO, TICKERS, FakeTrend

CORTE = pd.Timestamp("2026-05-29")
DIA = pd.Timestamp("2026-06-10")


def _dfs(snapshot, dia: pd.Timestamp, tickers=TICKERS) -> dict[str, pd.DataFrame]:
    return {s: snapshot.history(s, dia, 200) for s in tickers}


class FakeConProveedor(FakeTrend):
    """Como macro / contexto: ``fit`` con un argumento extra opcional y lo del dia
    por un proveedor, que tiene que estar seteado antes de predecir."""

    def __init__(self):
        super().__init__()
        self.macro_provider = None
        self.universe_provider = None
        self.fit_args: tuple = ()
        self.vistos: list = []

    def fit(self, histories: dict, macro=None) -> dict:
        self.fit_args = (histories, macro)
        return {"oos_auc": 0.6}

    def predict_df(self, df: pd.DataFrame) -> dict:
        proveedor = self.macro_provider or self.universe_provider
        self.vistos.append(proveedor())
        return super().predict_df(df)


class SinFit:
    version = "sin-fit"

    def predict_df(self, df: pd.DataFrame) -> dict:
        if len(df) < 100:
            raise NotEnoughDataError("pocas ruedas")
        if df["close"].iloc[-1] > 1e9:
            raise ValueError("no converge")
        return FakeTrend().predict_df(df)


def test_por_ticker_entrena_y_predice_cada_ticker(snapshot) -> None:
    modelo = por_ticker.PorTicker(FakeTrend())
    assert modelo.version == "fake-v1"
    modelo.fit(_dfs(snapshot, CORTE))
    assert modelo.modelo.entrenado_con == len(TICKERS)
    dfs = _dfs(snapshot, DIA)
    resultados = modelo.predict_panel(dfs)
    assert set(resultados) == set(TICKERS)
    assert resultados["GGAL"] == FakeTrend().predict_df(dfs["GGAL"])
    assert modelo.predict_panel({}) == {}


def test_por_ticker_sin_fit_y_tickers_que_no_puede_predecir(snapshot) -> None:
    modelo = por_ticker.PorTicker(SinFit())
    assert modelo.fit(_dfs(snapshot, CORTE)) == {}
    dfs = _dfs(snapshot, DIA)
    dfs[TICKER_CORTO] = snapshot.history(TICKER_CORTO, DIA, 200)  # 30 ruedas: no alcanza
    dfs["RARO"] = dfs["GGAL"].assign(close=1e10)
    assert set(modelo.predict_panel(dfs)) == set(TICKERS)


def test_macro_entrena_con_el_universo_y_predice_con_la_macro_del_dia(snapshot) -> None:
    fake = FakeConProveedor()
    modelo = por_ticker.MacroPorTicker(fake, snapshot)
    modelo.fit(_dfs(snapshot, CORTE, TICKERS[:2]))
    historias, macro = fake.fit_args
    esperadas = historias_macro(snapshot, CORTE)
    # NUEVA no cotizaba todavia al corte: queda afuera.
    assert set(historias) == set(esperadas) == set(TICKERS)
    assert historias["GGAL"].index[-1] == CORTE and len(historias["GGAL"]) > 200
    pd.testing.assert_frame_equal(macro, snapshot.macro_frame(CORTE))

    modelo.predict_panel(_dfs(snapshot, DIA))
    assert len(fake.vistos) == len(TICKERS)
    pd.testing.assert_frame_equal(fake.vistos[0], snapshot.macro_frame(DIA))
    assert fake.vistos[0].index.max() <= DIA  # nunca datos posteriores al dia


def test_contexto_entrena_con_el_universo_y_predice_con_el_mercado_del_dia(snapshot) -> None:
    fake = FakeConProveedor()
    modelo = por_ticker.ContextoPorTicker(fake, snapshot)
    modelo.fit(_dfs(snapshot, CORTE))
    historias, extra = fake.fit_args
    assert extra is None
    assert set(historias) == set(historias_contexto(snapshot, CORTE))
    assert max(df.index[-1] for df in historias.values()) == CORTE

    modelo.predict_panel(_dfs(snapshot, DIA))
    pd.testing.assert_frame_equal(fake.vistos[0].market, universo_al(snapshot, DIA).market)
    assert fake.vistos[0].market.index.max() <= DIA


def test_constructores_de_busqueda(snapshot) -> None:
    tipos = {"lstm": TrendModel, "xgboost": XGBTrendModel, "arima": ArimaTrendModel,
             "macro": MacroTrendModel, "contexto": ContextTrendModel}  # fmt: skip
    for nombre, tipo in tipos.items():
        modelo = MODELOS[nombre]({}, "cpu", snapshot)
        assert isinstance(modelo, por_ticker.PorTicker) and isinstance(modelo.modelo, tipo)
    assert MODELOS["lstm"]({"epochs": 3}, "cpu", None).modelo.config.epochs == 3
    assert MODELOS["xgboost"]({"seed": 7}, "cpu", None).modelo.config.seed == 7
    assert MODELOS["arima"]({"seed": 7, "horizon": 10}, "cpu", None).modelo.horizon == 10
    assert MODELOS["macro"]({"seed": 3}, "cpu", snapshot).modelo.config.seed == 3
    for nombre in ("macro", "contexto"):
        with pytest.raises(ValueError, match="snapshot"):
            MODELOS[nombre]({}, "cpu", None)
    with pytest.raises(TypeError):
        MODELOS["lstm"]({"no_existe": 1}, "cpu", None)


def test_un_modelo_con_arima(monkeypatch, tmp_path, snapshot_dir, capsys) -> None:
    monkeypatch.setattr(um, "RESULTADOS_DIR", str(tmp_path))
    monkeypatch.setattr(um, "generar_reporte", lambda out: out / "reporte" / "reporte.tex")
    um.main(["--nombre", "arima", "--modelo", "arima", "--desde", "2026-06-15", "--hasta",
             "2026-06-19", "--historia", "200", "--snapshot", str(snapshot_dir)])  # fmt: skip
    out = tmp_path / "arima"
    assert json.loads((out / "config.json").read_text())["solo_modelo"] == "arima"
    predicciones = pd.read_json(out / "predicciones.jsonl", lines=True)
    assert set(predicciones["model"]) == {"arima"}
    assert predicciones["fecha"].nunique() == 5
    assert set(predicciones["symbol"]) <= set(TICKERS)
    assert "senal_arima" in set(pd.read_csv(out / "carteras.csv")["cartera"])
    salida = capsys.readouterr().out
    assert "Reporte LaTeX:" in salida and "sin PDF" in salida
