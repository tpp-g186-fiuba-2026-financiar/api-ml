import threading
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.errors import DataUnavailableError, NotEnoughDataError, StaleArtifactError
from src.panel import servicio as sv
from src.panel.servicio import PanelTrendService, tendencia_panel
from tests.panel._panel_sintetico import RUEDAS, TICKERS, histories


class _PanelFalso:
    """Predice un retorno relativo fijo por ticker (el orden de ``TICKERS``)."""

    version = "transformer-v2-usd"

    def __init__(self):
        self.universo = list(TICKERS)
        self.config = SimpleNamespace(horizonte=5)
        self.proveedor = None
        self.pedidos = []

    def predict_panel(self, dfs):
        self.pedidos.append(max(df.index[-1] for df in dfs.values()))
        relativos = np.linspace(-0.02, 0.02, len(self.universo))
        salida = {}
        for symbol, rel in zip(self.universo, relativos, strict=True):
            if symbol in dfs:
                ultimo = float(dfs[symbol]["close"].iloc[-1])
                salida[symbol] = {"last_close": ultimo, "predicted_close": ultimo * np.exp(rel)}
        return salida


class _Respaldo:
    is_loaded = True
    version = "transformer-v1-abc"

    def predict(self, symbol):
        return {"symbol": symbol, "model": "transformer-v1", "model_version": self.version}

    def predict_on(self, df, symbol):
        return self.predict(symbol)


def _servicio(tmp_path, historias=None, cargar=True, **kwargs):
    historias = historias or histories()
    ruta = tmp_path / "v2.pt"
    ruta.write_text("x")
    descargas = []

    def fetch(symbol, days):
        descargas.append((symbol, days))
        return historias[symbol]

    servicio = PanelTrendService(
        "transformer", str(ruta), lambda p: _PanelFalso(), _Respaldo(), 750, fetch_hist=fetch,
        **kwargs,
    )  # fmt: skip
    if cargar:
        servicio.load()
    return servicio, descargas


def test_tendencia_panel() -> None:
    idx = pd.bdate_range("2024-01-01", periods=300)
    sube = pd.DataFrame({"close": np.exp(np.arange(300) * 0.001)}, index=idx)
    plano = pd.DataFrame({"close": np.ones(300)}, index=idx)
    assert tendencia_panel({"A": sube, "B": plano}, 5) == pytest.approx(0.0025)
    assert tendencia_panel({"A": sube.iloc[:100]}, 5) == 0.0  # sin un anio de historia


def test_predict_suma_la_tendencia_y_ordena_la_senal(tmp_path) -> None:
    proveedores = []
    servicio, descargas = _servicio(
        tmp_path, proveedor_factory=lambda: proveedores.append(1) or "p"
    )
    hist = histories()
    tendencia = tendencia_panel(hist, 5)

    mejor = servicio.predict(TICKERS[-1].lower())
    peor = servicio.predict(TICKERS[0])
    medio = servicio.predict(TICKERS[3])
    assert (mejor["signal"], peor["signal"], medio["signal"]) == ("alza", "baja", "neutral")
    retorno = np.log(mejor["predicted_close"] / mejor["last_close"])
    assert retorno == pytest.approx(0.02 + tendencia, abs=1e-4)
    assert mejor["symbol"] == TICKERS[-1] and mejor["model"] == "transformer"
    assert mejor["model_version"] == "transformer-v2-usd"
    assert mejor["as_of"] == RUEDAS[-1].strftime("%Y-%m-%d")
    # Datos y prediccion del panel una sola vez por dia.
    assert len(descargas) == len(TICKERS) and descargas[0][1] == 750 + sv.RUEDAS_PREDICCION
    assert servicio._model.pedidos == [RUEDAS[-1]] and servicio._model.proveedor == "p"
    assert proveedores == [1]


def test_fuera_del_panel_responde_el_respaldo(tmp_path) -> None:
    servicio, _ = _servicio(tmp_path)
    resultado = servicio.predict("AAPL")
    assert resultado["model"] == "transformer"
    assert resultado["model_version"] == "respaldo:transformer-v1-abc"
    df = histories()["GGAL"]
    assert servicio.predict_on(df, "AAPL")["model_version"].startswith("respaldo:")


def test_sin_artefacto_todo_va_al_respaldo(tmp_path) -> None:
    servicio, _ = _servicio(tmp_path, cargar=False)
    assert servicio.is_loaded and servicio.version == "respaldo:transformer-v1-abc"
    assert servicio.predict("GGAL")["model_version"] == "respaldo:transformer-v1-abc"
    servicio.model_path = tmp_path / "no.pt"
    servicio.load()
    assert servicio._model is None


def test_artefacto_viejo_no_corta_la_carga(tmp_path, capsys) -> None:
    servicio, _ = _servicio(tmp_path, cargar=False)

    def viejo(path):
        raise StaleArtifactError("otro universo")

    servicio._loader = viejo
    servicio.load()
    assert servicio._model is None and "no se pudo cargar" in capsys.readouterr().out


def test_predict_on_predice_con_los_datos_hasta_la_fecha_del_df(tmp_path) -> None:
    servicio, _ = _servicio(tmp_path)
    df = histories()["GGAL"].iloc[:-10]
    resultado = servicio.predict_on(df, "GGAL")
    assert resultado["as_of"] == RUEDAS[-11].strftime("%Y-%m-%d")
    assert resultado["last_close"] == pytest.approx(df["close"].iloc[-1], rel=1e-4)
    servicio.predict_on(df, "BMA")
    assert servicio._model.pedidos == [RUEDAS[-11]]  # cache por fecha
    assert servicio.backtest_on(df, 5) is None


def test_ticker_sin_dato_ese_dia_va_al_respaldo(tmp_path) -> None:
    hist = histories()
    hist["GGAL"] = hist["GGAL"].iloc[:-1]
    servicio, _ = _servicio(tmp_path, historias=hist)
    servicio._model.predict_panel = lambda dfs: {
        s: {"last_close": 1.0, "predicted_close": 1.0} for s in dfs if s != "GGAL"
    }
    assert servicio.predict("GGAL")["model_version"].startswith("respaldo:")
    assert servicio.predict_on(hist["BMA"], "GGAL")["model_version"].startswith("respaldo:")


def test_series_estancadas_y_errores_de_datos(tmp_path) -> None:
    hist = histories()
    hist["GGAL"] = hist["GGAL"].assign(close=100.0)
    servicio, _ = _servicio(tmp_path, historias=hist)
    with pytest.raises(NotEnoughDataError, match="estancados"):
        servicio.predict("GGAL")
    with pytest.raises(NotEnoughDataError, match="estancados"):
        servicio.predict_on(hist["GGAL"], "GGAL")
    anterior = pd.DataFrame(
        {"close": [1.0, 2.0, 1.5]},
        index=pd.bdate_range(RUEDAS[0] - pd.Timedelta(days=30), periods=3),
    )
    with pytest.raises(NotEnoughDataError, match="sin datos del panel"):
        servicio.predict_on(anterior, "BMA")


def test_panel_vacio_o_sin_historicos(tmp_path) -> None:
    servicio, _ = _servicio(tmp_path)
    servicio._model.predict_panel = lambda dfs: {}
    with pytest.raises(NotEnoughDataError, match="datos suficientes"):
        servicio.predict("GGAL")

    def falla(symbol, days):
        raise DataUnavailableError("collector caido")

    # Sin datos del panel (data-colector caido) responde el respaldo
    otro, _ = _servicio(tmp_path)
    otro._fetch_hist = falla
    assert otro.predict("GGAL")["model_version"] == "respaldo:transformer-v1-abc"


def test_pedidos_en_paralelo_predicen_el_panel_una_vez(tmp_path) -> None:
    servicio, descargas = _servicio(tmp_path)
    hilos = [threading.Thread(target=servicio.predict, args=(t,)) for t in TICKERS]
    for h in hilos:
        h.start()
    for h in hilos:
        h.join()
    assert len(descargas) == len(TICKERS) and len(servicio._model.pedidos) == 1
