import sys

import pandas as pd
import pytest

import src.panel.train_transformer_v2 as train
from src.config import settings
from src.errors import DataUnavailableError
from src.panel.transformer_v2 import TransformerV2Model
from tests.panel.test_transformer import TICKERS, _histories, _macro


def test_proveedor_collector_corta_el_ccl_a_la_fecha(monkeypatch) -> None:
    ccl = _macro()["ccl"]
    pedidos = []
    monkeypatch.setattr(
        train, "fetch_collector_series", lambda ruta, serie: pedidos.append(serie) or ccl
    )
    proveedor = train.proveedor_collector()
    hasta = pd.Timestamp("2023-06-01")
    serie, macro = proveedor(hasta)
    assert macro is None and serie.index.max() <= hasta and pedidos == ["CCL"]
    proveedor(hasta)
    assert pedidos == ["CCL"]  # se pide una sola vez


def test_proveedor_collector_con_macro_aplica_retrasos(monkeypatch) -> None:
    macro = _macro()
    pedidas = []

    def fetch(ruta, serie):
        pedidas.append(serie)
        nombre = {"CCL": "ccl", "OFICIAL": "oficial", "RIESGO_PAIS": "riesgo_pais",
                  "BADLAR": "badlar"}.get(serie)  # fmt: skip
        if nombre is None:
            raise DataUnavailableError(f"sin {serie}")
        return macro[nombre]

    monkeypatch.setattr(train, "fetch_collector_series", fetch)
    proveedor = train.proveedor_collector(macro=True)
    hasta = pd.Timestamp("2023-06-01")
    ccl, tabla = proveedor(hasta)
    assert set(tabla.columns) == {"ccl", "oficial", "riesgo_pais", "badlar"}
    # El CCL para pasar a dolares no tiene retraso; el de la tabla macro, 2 dias.
    assert ccl.index.max() == hasta and tabla.loc[hasta, "ccl"] == macro["ccl"].loc[:hasta].iloc[-3]
    assert len(pedidas) == len(train.SERIES)

    def sin_badlar(ruta, serie):
        if serie == "BADLAR":
            raise DataUnavailableError("sin BADLAR")
        return fetch(ruta, serie)

    monkeypatch.setattr(train, "fetch_collector_series", sin_badlar)
    with pytest.raises(DataUnavailableError, match="BADLAR"):
        train.proveedor_collector(macro=True)


def test_main_entrena_y_guarda(tmp_path, monkeypatch, capsys) -> None:
    historias = _histories()
    monkeypatch.setattr(train, "PANEL_LIDER_TICKERS", TICKERS)
    monkeypatch.setattr(train, "fetch_history_collector", lambda symbol, days: historias[symbol])
    monkeypatch.setattr(train, "fetch_collector_series", lambda ruta, serie: _macro()["ccl"])
    monkeypatch.setattr(settings, "data_collector_url", settings.data_collector_url)
    ruta = tmp_path / "v2.pt"
    argv = ["train", "--collector-url", "http://collector:3000", "--epocas", "2",
            "--n-semillas", "1", "--out", str(ruta), "--force"]  # fmt: skip
    monkeypatch.setattr(sys, "argv", argv)
    train.main()
    assert settings.data_collector_url == "http://collector:3000"
    modelo = TransformerV2Model.load(ruta)
    assert modelo.config.epocas == 2 and len(modelo._redes) == 1
    assert "Modelo guardado" in capsys.readouterr().out


def test_main_sin_datos_aborta(monkeypatch) -> None:
    def falla(symbol, days):
        raise RuntimeError("sin datos")

    monkeypatch.setattr(train, "fetch_history_collector", falla)
    monkeypatch.setattr(sys, "argv", ["train"])
    try:
        train.main()
    except SystemExit as exc:
        assert "No se pudo descargar" in str(exc)
    else:
        raise AssertionError("tendria que abortar sin datos")


@pytest.mark.parametrize("promovido", [True, False])
def test_main_pasa_por_el_gate(tmp_path, monkeypatch, capsys, promovido) -> None:
    historias = _histories()
    monkeypatch.setattr(train, "PANEL_LIDER_TICKERS", TICKERS)
    monkeypatch.setattr(train, "fetch_history_collector", lambda symbol, days: historias[symbol])
    monkeypatch.setattr(train, "fetch_collector_series", lambda ruta, serie: _macro()["ccl"])
    monkeypatch.setattr(settings, "data_collector_url", settings.data_collector_url)
    llamadas = []

    def gate(fabrica, loader, histories, out, horizonte):
        llamadas.append((fabrica().config.n_semillas, sorted(histories), out, horizonte))
        return promovido, {"ic": 0.05 if promovido else -0.01}

    monkeypatch.setattr(train, "entrenar_con_gate", gate)
    ruta = str(tmp_path / "v2.pt")
    monkeypatch.setattr(sys, "argv", ["train", "--n-semillas", "2", "--out", ruta])
    train.main()
    assert llamadas == [(2, sorted(TICKERS), ruta, 5)]
    salida = capsys.readouterr().out
    assert ("Modelo guardado" in salida) is promovido
    assert ("no se toca el artefacto" in salida) is not promovido
