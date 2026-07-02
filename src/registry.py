"""Registro de modelos de tendencia.

Todos los modelos (LSTM, XGBoost, y los que vengan) exponen la misma interfaz
``TrendPredictor`` (``version`` + ``predict_df``). El registro los mantiene a
todos cargados en paralelo y permite elegir cual usar en cada request, sin que
un modelo nuevo afecte a los anteriores.

Para sumar un modelo nuevo alcanza con:

1. Escribir su clase (con ``predict_df``, ``version``, ``save``, ``load``).
2. Registrarlo en ``build_registry`` (una linea).
3. Agregar su ``*_model_path`` en ``src.config``.

Nada mas cambia: endpoints, health y comparacion lo toman automaticamente.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol, runtime_checkable

import pandas as pd

from src.data import fetch_history
from src.errors import ApiMlError, ModelNotLoadedError, UnknownModelError


@runtime_checkable
class TrendPredictor(Protocol):
    @property
    def version(self) -> str: ...

    def predict_df(self, df: pd.DataFrame) -> dict: ...


LoaderFn = Callable[[Path], TrendPredictor]


class TrendService:
    """Envuelve un artefacto de modelo: lo carga y predice para un ticker."""

    def __init__(self, name: str, model_path: str, loader: LoaderFn, history_days: int):
        self.name = name
        self.model_path = Path(model_path)
        self._loader = loader
        self.history_days = history_days
        self._model: TrendPredictor | None = None

    def load(self) -> None:
        if self.model_path.exists():
            self._model = self._loader(self.model_path)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def version(self) -> str:
        return self._model.version if self._model else "unloaded"

    def predict_on(self, df: pd.DataFrame, symbol: str) -> dict:
        """Predice sobre un historico ya descargado (para reusar entre modelos)."""
        if self._model is None:
            raise ModelNotLoadedError(
                f"el modelo '{self.name}' no esta entrenado; correr su script de training"
            )
        result = self._model.predict_df(df)
        result["symbol"] = symbol.strip().upper()
        result["model"] = self.name
        result["model_version"] = self._model.version
        return result

    def predict(self, symbol: str) -> dict:
        df = fetch_history(symbol, self.history_days)
        return self.predict_on(df, symbol)


class TrendRegistry:
    """Coleccion de modelos de tendencia, con uno marcado como default."""

    def __init__(self, history_days: int):
        self._history_days = history_days
        self._services: dict[str, TrendService] = {}
        self._default: str | None = None

    def register(
        self, name: str, model_path: str, loader: LoaderFn, *, default: bool = False
    ) -> TrendRegistry:
        key = name.strip().lower()
        self._services[key] = TrendService(key, model_path, loader, self._history_days)
        if default or self._default is None:
            self._default = key
        return self

    def load_all(self) -> None:
        for service in self._services.values():
            service.load()

    def resolve(self, name: str | None) -> TrendService:
        key = (name or self._default or "").strip().lower()
        if key not in self._services:
            available = ", ".join(self._services) or "(ninguno)"
            raise UnknownModelError(f"modelo '{name}' desconocido; disponibles: {available}")
        return self._services[key]

    @property
    def default(self) -> str | None:
        return self._default

    def names(self) -> list[str]:
        return list(self._services)

    def compare(self, symbol: str) -> dict:
        """Corre todos los modelos entrenados sobre el mismo historico.

        Descarga la serie una sola vez y la pasa a cada modelo, asi la
        comparacion es justa (todos ven exactamente los mismos datos). Los
        modelos no entrenados se reportan como no disponibles, sin cortar.
        """
        df = fetch_history(symbol, self._history_days)
        predictions: dict[str, dict] = {}
        for name, service in self._services.items():
            if not service.is_loaded:
                predictions[name] = {"available": False, "reason": "modelo no entrenado"}
                continue
            try:
                predictions[name] = service.predict_on(df, symbol)
            except ApiMlError as exc:
                predictions[name] = {"available": False, "reason": str(exc)}
        return {
            "symbol": symbol.strip().upper(),
            "as_of": df.index[-1].strftime("%Y-%m-%d"),
            "default_model": self._default,
            "predictions": predictions,
        }

    def statuses(self) -> dict[str, dict]:
        """Estado de cada modelo, para el endpoint de health / listado."""
        return {
            name: {
                "loaded": svc.is_loaded,
                "version": svc.version,
                "is_default": name == self._default,
            }
            for name, svc in self._services.items()
        }


def build_registry(history_days: int) -> TrendRegistry:
    """Arma el registro con todos los modelos disponibles.

    >>> Para sumar un modelo nuevo, agregar una linea `.register(...)` aca. <<<
    """
    from src.config import settings
    from src.lstm import TrendModel
    from src.transformer import TransformerTrendModel
    from src.xgb_trend import XGBTrendModel

    registry = TrendRegistry(history_days)
    registry.register("lstm", settings.lstm_model_path, TrendModel.load, default=True)
    registry.register("xgboost", settings.xgb_model_path, XGBTrendModel.load)
    registry.register("transformer", settings.transformer_model_path, TransformerTrendModel.load)
    # Proximos modelos: registry.register("randomforest", settings.rf_model_path, RFTrendModel.load)
    return registry
