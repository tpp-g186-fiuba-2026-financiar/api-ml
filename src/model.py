from pathlib import Path

import joblib

from src.errors import ModelNotLoadedError


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
