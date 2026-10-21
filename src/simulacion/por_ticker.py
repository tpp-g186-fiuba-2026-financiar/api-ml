"""Los modelos de la API que predicen ticker por ticker, con la interfaz de los de panel.

``un_modelo`` (y ``predicciones_panel``) entrenan con ``fit(histories)`` y
predicen todos los tickers del dia juntos con ``predict_panel(dfs)``. Los
modelos de produccion (LSTM, XGBoost, ARIMA, macro, contexto) predicen con
``predict_df`` de a un ticker; ``PorTicker`` los adapta:

- ``predict_panel`` llama a ``predict_df`` por ticker y deja afuera los que el
  modelo no puede predecir (pocos datos, serie quieta), como la API.
- ARIMA no se entrena: se ajusta en cada prediccion.
- Macro y contexto se entrenan como en el simulador completo
  (``entrenamiento.historias_macro`` / ``historias_contexto``, con los datos del
  snapshot hasta el corte) y predicen con la macro / el mercado del dia simulado,
  nunca posteriores. A diferencia de produccion no hay piso de AUC: cada
  reentreno reemplaza al anterior, como el resto de ``un_modelo``.
"""

from __future__ import annotations

import pandas as pd

from src.arima_trend import ArimaTrendModel
from src.context_trend import ContextConfig, ContextTrendModel
from src.errors import ApiMlError
from src.lstm import TrainConfig, TrendModel
from src.macro_trend import MacroConfig, MacroTrendModel
from src.simulacion.entrenamiento import historias_contexto, historias_macro, mercado_al
from src.simulacion.snapshot import Snapshot
from src.xgb_trend import XGBConfig, XGBTrendModel


def _corte(dfs: dict[str, pd.DataFrame]) -> pd.Timestamp:
    """Ultima rueda de los datos: la fecha de corte de un ``fit`` o el dia de una prediccion."""
    return max(df.index[-1] for df in dfs.values() if not df.empty)


class PorTicker:
    """Modelo con ``predict_df`` (y ``fit(histories)``, si se entrena) como modelo de panel."""

    def __init__(self, modelo) -> None:
        self.modelo = modelo

    @property
    def version(self) -> str:
        return self.modelo.version

    def fit(self, histories: dict[str, pd.DataFrame]) -> dict:
        if not hasattr(self.modelo, "fit"):
            return {}
        return self.modelo.fit(histories)

    def preparar_dia(self, dia: pd.Timestamp) -> None:
        """Lo que el modelo necesita del dia ademas del historico del ticker."""

    def predict_panel(self, dfs: dict[str, pd.DataFrame]) -> dict[str, dict]:
        if not dfs:
            return {}
        self.preparar_dia(_corte(dfs))
        resultados = {}
        for symbol, df in dfs.items():
            try:
                resultados[symbol] = self.modelo.predict_df(df)
            except (ApiMlError, ValueError, ArithmeticError):  # como el simulador
                continue
        return resultados


class MacroPorTicker(PorTicker):
    """``macro``: entrena con el universo y la macro hasta el corte; predice con la del dia."""

    def __init__(self, modelo: MacroTrendModel, snapshot: Snapshot) -> None:
        super().__init__(modelo)
        self.snapshot = snapshot

    def fit(self, histories: dict[str, pd.DataFrame]) -> dict:
        hasta = _corte(histories)
        historias = historias_macro(self.snapshot, hasta)
        return self.modelo.fit(historias, self.snapshot.macro_frame(hasta))

    def preparar_dia(self, dia: pd.Timestamp) -> None:
        macro = self.snapshot.macro_frame(dia)
        self.modelo.macro_provider = lambda: macro


class ContextoPorTicker(PorTicker):
    """``contexto``: entrena con el universo hasta el corte; predice con el mercado del dia."""

    def __init__(self, modelo: ContextTrendModel, snapshot: Snapshot) -> None:
        super().__init__(modelo)
        self.snapshot = snapshot

    def fit(self, histories: dict[str, pd.DataFrame]) -> dict:
        return self.modelo.fit(historias_contexto(self.snapshot, _corte(histories)))

    def preparar_dia(self, dia: pd.Timestamp) -> None:
        mercado = mercado_al(self.snapshot, dia)
        self.modelo.market_provider = lambda: mercado


def _snapshot(snapshot: Snapshot | None, modelo: str) -> Snapshot:
    if snapshot is None:
        raise ValueError(f"el modelo {modelo} necesita el snapshot (macro / mercado)")
    return snapshot


# Constructores con la firma de ``busqueda.MODELOS``: (config, dispositivo, snapshot).
# ``config`` cambia campos de la config del modelo; el dispositivo no se usa.
def lstm(config: dict, dispositivo: str, snapshot: Snapshot | None) -> PorTicker:
    return PorTicker(TrendModel(config=TrainConfig(**config)))


def xgboost(config: dict, dispositivo: str, snapshot: Snapshot | None) -> PorTicker:
    return PorTicker(XGBTrendModel(config=XGBConfig(**config)))


def arima(config: dict, dispositivo: str, snapshot: Snapshot | None) -> PorTicker:
    # Deterministico: ignora la ``seed`` que agrega la busqueda.
    return PorTicker(ArimaTrendModel(**{k: v for k, v in config.items() if k != "seed"}))


def macro(config: dict, dispositivo: str, snapshot: Snapshot | None) -> MacroPorTicker:
    snapshot = _snapshot(snapshot, "macro")
    return MacroPorTicker(MacroTrendModel(config=MacroConfig(**config)), snapshot)


def contexto(config: dict, dispositivo: str, snapshot: Snapshot | None) -> ContextoPorTicker:
    snapshot = _snapshot(snapshot, "contexto")
    return ContextoPorTicker(ContextTrendModel(config=ContextConfig(**config)), snapshot)
