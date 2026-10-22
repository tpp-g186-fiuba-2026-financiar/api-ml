import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.errors import NotEnoughDataError, StaleArtifactError
from src.panel.gate import entrenar_con_gate, ic_panel
from tests.panel._panel_sintetico import RUEDAS, histories

HORIZONTE = 5


class _Adivino:
    """Modelo de panel falso que "predice" el retorno real (o su opuesto) a
    ``HORIZONTE`` ruedas, mirando las historias completas."""

    def __init__(self, completas, signo=1.0):
        self.completas = completas
        self.signo = signo
        self.config = SimpleNamespace(horizonte=HORIZONTE)
        self.proveedor = "proveedor"
        self.visto_hasta = None

    def fit(self, historias):
        self.visto_hasta = max(df.index[-1] for df in historias.values())
        return {}

    def predict_panel(self, dfs):
        salida = {}
        for symbol, df in dfs.items():
            close = self.completas[symbol]["close"]
            i = close.index.get_loc(df.index[-1])
            real = np.log(close.iloc[min(i + HORIZONTE, len(close) - 1)] / close.iloc[i])
            ultimo = float(df["close"].iloc[-1])
            salida[symbol] = {
                "last_close": ultimo,
                "predicted_close": ultimo * np.exp(self.signo * real),
            }
        return salida

    def save(self, path):
        Path(path).write_text(json.dumps({"signo": self.signo}))


def test_ic_panel_mide_en_la_ventana_con_retorno_realizado() -> None:
    hist = histories()
    score = ic_panel(_Adivino(hist), hist, HORIZONTE, dias=20)
    assert score["ic"] == pytest.approx(1.0)
    assert score["observations"] == 20 * len(hist)
    assert score["hasta"] == RUEDAS[-1 - HORIZONTE].strftime("%Y-%m-%d")
    assert ic_panel(_Adivino(hist, -1.0), hist, HORIZONTE, dias=20)["ic"] == pytest.approx(-1.0)


def test_ic_panel_sin_predicciones() -> None:
    class _Mudo(_Adivino):
        def predict_panel(self, dfs):
            return {}

    hist = histories()
    assert ic_panel(_Mudo(hist), hist, HORIZONTE, dias=10) == {"ic": None, "observations": 0}


def test_promueve_si_el_ic_fuera_de_muestra_es_positivo(tmp_path) -> None:
    hist = histories()
    ruta = tmp_path / "v2.pt"
    ruta.write_text("{}")  # vigente
    creados = []

    def fabrica():
        creados.append(_Adivino(hist))
        return creados[-1]

    promovido, score = entrenar_con_gate(
        fabrica, lambda p: _Adivino(hist, -1.0), hist, str(ruta), HORIZONTE, dias=20
    )
    assert promovido and score["ic"] == pytest.approx(1.0)
    assert score["ic_vigente"] == pytest.approx(-1.0)  # referencia, no decide
    # El de evaluacion no vio la ventana; el final, todo.
    assert creados[0].visto_hasta < RUEDAS[-(20 + HORIZONTE)]
    assert creados[1].visto_hasta == RUEDAS[-1]
    assert json.loads(ruta.read_text()) == {"signo": 1.0}
    assert json.loads((tmp_path / "v2.pt.score.json").read_text())["ic"] == pytest.approx(1.0)


def test_no_promueve_si_no_ordena_mejor_que_el_azar(tmp_path) -> None:
    hist = histories()
    ruta = tmp_path / "v2.pt"

    def vigente_viejo(path):
        raise StaleArtifactError("otro universo")

    promovido, score = entrenar_con_gate(
        lambda: _Adivino(hist, -1.0), vigente_viejo, hist, str(ruta), HORIZONTE, dias=20
    )
    assert not promovido and score["ic"] == pytest.approx(-1.0)
    assert score["ic_vigente"] is None and not ruta.exists()


def test_gate_con_pocas_ruedas() -> None:
    corto = {t: df.iloc[:20] for t, df in histories().items()}
    with pytest.raises(NotEnoughDataError):
        entrenar_con_gate(lambda: None, lambda p: None, corto, "x.pt", HORIZONTE, dias=20)


def test_ic_panel_saltea_dias_sin_dato_del_ticker() -> None:
    hist = histories()
    hist["GGAL"] = hist["GGAL"].drop(RUEDAS[-10])
    score = ic_panel(_Adivino(histories()), hist, HORIZONTE, dias=20)
    assert score["observations"] == 20 * len(hist) - 1
    assert isinstance(pd.Timestamp(score["desde"]), pd.Timestamp)
