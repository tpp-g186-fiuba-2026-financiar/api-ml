import numpy as np
import pandas as pd
import pytest

from src.errors import ApiMlError
from src.panel.dos_cabezas import DosCabezasModel
from src.trend_common import derive_trend_output
from tests.panel._panel_sintetico import histories


class Fija:
    """Cabeza falsa: devuelve retornos fijos por ticker (o el mismo para todos)."""

    def __init__(self, retornos: dict[str, float] | float):
        self.retornos = retornos
        self.entrenada = False

    def fit(self, histories):
        self.entrenada = True
        return {"ok": True}

    def predict_panel(self, dfs):
        return {
            s: derive_trend_output(
                df, self.retornos if isinstance(self.retornos, float) else self.retornos[s], 5
            )
            for s, df in dfs.items()
            if isinstance(self.retornos, float) or s in self.retornos
        }


def _log(resultados: dict) -> pd.Series:
    return pd.Series(
        {s: np.log(r["predicted_close"] / r["last_close"]) for s, r in resultados.items()}
    )


def test_suma_el_mercado_y_la_parte_relativa_sin_su_media() -> None:
    dfs = {s: df for s, df in list(histories().items())[:3]}
    a, b, c = dfs
    relativo = Fija({a: 0.03, b: 0.01, c: -0.01})  # media 0,01: se le resta
    modelo = DosCabezasModel(Fija(0.02), relativo, peso_mercado=1.0)
    assert modelo.fit(dfs) == {"mercado": {"ok": True}, "relativo": {"ok": True}}
    r = _log(modelo.predict_panel(dfs))
    np.testing.assert_allclose(r[[a, b, c]], [0.04, 0.02, 0.0], atol=2e-4)
    medio = _log(DosCabezasModel(Fija(0.02), relativo, peso_mercado=0.5).predict_panel(dfs))
    np.testing.assert_allclose(medio[[a, b, c]], [0.03, 0.01, -0.01], atol=2e-4)
    solo_relativo = _log(DosCabezasModel(Fija(0.02), relativo, 0.0).predict_panel(dfs))
    assert solo_relativo.mean() == pytest.approx(0, abs=2e-4)
    assert modelo.version == "dos-cabezas-1"


def test_casos_vacios_y_errores() -> None:
    dfs = {s: df for s, df in list(histories().items())[:2]}
    assert DosCabezasModel(Fija(0.01), Fija({})).predict_panel(dfs) == {}
    sin_mercado = DosCabezasModel(Fija({}), Fija({s: 0.01 for s in dfs}))
    assert sin_mercado.predict_panel(dfs) == {}
    with pytest.raises(ValueError, match="peso_mercado"):
        DosCabezasModel(Fija(0.0), Fija(0.0), peso_mercado=-1)
    with pytest.raises(ApiMlError):
        DosCabezasModel(Fija(0.0), Fija(0.0)).predict_df(next(iter(dfs.values())))
