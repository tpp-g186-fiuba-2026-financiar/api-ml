from pathlib import Path

import joblib

from src.data import fetch_history
from src.errors import ModelNotLoadedError
from src.lstm import TrendModel


class PredictionModel:
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self._model = None
        self._version = "unloaded"

    def load(self) -> None:
        if not self.model_path.exists():
            self._model = _PlaceholderModel()
            self._version = "placeholder"
            return
        self._model = joblib.load(self.model_path)
        self._version = self.model_path.stem

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def version(self) -> str:
        return self._version

    def predict(self, features: dict[str, float]) -> float:
        if self._model is None:
            raise ModelNotLoadedError("model has not been loaded yet")
        return float(self._model.predict_one(features))


class _PlaceholderModel:
    def predict_one(self, features: dict[str, float]) -> float:
        if not features:
            return 0.0
        return sum(features.values()) / len(features)


class TrendService:
    """Servicio del modelo LSTM de tendencia.

    Carga el artefacto entrenado (si existe) y, dado un ticker, descarga su
    historico reciente y devuelve la prediccion de tendencia del proximo dia.
    """

    def __init__(self, model_path: str, history_days: int):
        self.model_path = Path(model_path)
        self.history_days = history_days
        self._model: TrendModel | None = None

    def load(self) -> None:
        if self.model_path.exists():
            self._model = TrendModel.load(self.model_path)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def version(self) -> str:
        return self._model.version if self._model else "unloaded"

    def predict(self, symbol: str) -> dict:
        if self._model is None:
            raise ModelNotLoadedError(
                "el modelo LSTM no esta entrenado; correr `python -m src.train`"
            )
        df = fetch_history(symbol, self.history_days)
        result = self._model.predict_df(df)
        result["symbol"] = symbol.strip().upper()
        result["model_version"] = self._model.version
        return result
