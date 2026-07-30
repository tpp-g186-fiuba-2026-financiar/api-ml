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

from src.data import MACRO_COLUMN, attach_macro_feature, fetch_history, fetch_macro_series
from src.errors import ApiMlError, ModelNotLoadedError, UnknownModelError
from src.modal_client import call_modal


def _fetch_history_with_macro(symbol: str, days: int) -> pd.DataFrame:
    """Historico OHLCV de un ticker con la feature macro ya alineada (ver ``src.lstm``)."""
    df = fetch_history(symbol, days)
    return attach_macro_feature(df, MACRO_COLUMN, fetch_macro_series(days))


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
        if not self.model_path.exists():
            return
        try:
            self._model = self._loader(self.model_path)
        except ApiMlError as exc:
            # No corta el arranque del resto de los modelos por uno solo con
            # problemas (ej: StaleArtifactError si cambio el feature set y
            # este artefacto todavia no se reentreno) -- se reporta como no
            # cargado, igual que un modelo sin entrenar.
            print(f"  [warn] no se pudo cargar el modelo '{self.name}': {exc}")

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
        df = _fetch_history_with_macro(symbol, self.history_days)
        return self.predict_on(df, symbol)


class OnDemandTrendService:
    """Modelo sin artefacto persistido: se ajusta al vuelo en cada request.

    Para modelos por-ticker que no se pueden "poolear" en un unico artefacto
    pre-entrenado (ej: ARIMA). Siempre esta `is_loaded`, no hay nada que
    cargar de disco.
    """

    def __init__(self, name: str, predictor: TrendPredictor, history_days: int):
        self.name = name
        self._predictor = predictor
        self.history_days = history_days

    def load(self) -> None:
        pass

    @property
    def is_loaded(self) -> bool:
        return True

    @property
    def version(self) -> str:
        return self._predictor.version

    def predict_on(self, df: pd.DataFrame, symbol: str) -> dict:
        result = self._predictor.predict_df(df)
        result["symbol"] = symbol.strip().upper()
        result["model"] = self.name
        result["model_version"] = self._predictor.version
        return result

    def predict(self, symbol: str) -> dict:
        df = fetch_history(symbol, self.history_days)
        return self.predict_on(df, symbol)


TranslatorFn = Callable[[dict], dict]


def _identity_translator(payload: dict) -> dict:
    """Traductor por defecto: solo renombra 'ticker' a 'symbol' si vino asi."""
    if "ticker" in payload and "symbol" not in payload:
        payload["symbol"] = payload.pop("ticker")
    return payload


# Campos que el equipo decidio sacar del contrato de TrendResponse
# (expected_return era redundante con predicted_close; confidence saturaba
# en 1.0 con cualquier retorno > 3%). Los modelos locales ya no los generan,
# pero las copias en Modal (repo `models`) todavia si -- se filtran aca para
# que /predict/trend/compare (que no valida contra el schema) sea
# consistente entre todos los modelos.
_DROPPED_TREND_FIELDS = ("expected_return", "confidence")


class RemoteTrendService:
    """Modelo que corre en otro servicio (Modal) via HTTP, no en este proceso.

    A diferencia de TrendService/OnDemandTrendService, no reusa el ``df`` ya
    descargado en ``compare()``: Modal baja su propio historico en el
    momento de la request, asi que el ``as_of`` puede diferir levemente del
    resto de los modelos en una comparacion.
    """

    def __init__(
        self,
        name: str,
        base_url: str,
        history_days: int,
        *,
        translator: TranslatorFn = _identity_translator,
        extra_params: dict | None = None,
    ):
        self.name = name
        self.base_url = base_url
        self.history_days = history_days
        self._translator = translator
        self._extra_params = extra_params or {}

    def load(self) -> None:
        pass

    @property
    def is_loaded(self) -> bool:
        return True

    @property
    def version(self) -> str:
        return f"modal:{self.name}"

    def predict_on(self, df: pd.DataFrame, symbol: str) -> dict:
        return self.predict(symbol)

    def predict(self, symbol: str) -> dict:
        payload = call_modal(self.base_url, symbol, **self._extra_params)
        result = self._translator(payload)
        for field in _DROPPED_TREND_FIELDS:
            result.pop(field, None)
        result["symbol"] = symbol.strip().upper()
        result["model"] = self.name
        result.setdefault("model_version", self.version)
        return result


class TrendRegistry:
    """Coleccion de modelos de tendencia, con uno marcado como default."""

    def __init__(self, history_days: int):
        self._history_days = history_days
        self._services: dict[str, TrendService | OnDemandTrendService | RemoteTrendService] = {}
        self._default: str | None = None

    def register(
        self, name: str, model_path: str, loader: LoaderFn, *, default: bool = False
    ) -> TrendRegistry:
        key = name.strip().lower()
        self._services[key] = TrendService(key, model_path, loader, self._history_days)
        if default or self._default is None:
            self._default = key
        return self

    def register_on_demand(
        self, name: str, predictor: TrendPredictor, *, default: bool = False
    ) -> TrendRegistry:
        """Registra un modelo sin artefacto persistido (se ajusta al vuelo por ticker)."""
        key = name.strip().lower()
        self._services[key] = OnDemandTrendService(key, predictor, self._history_days)
        if default or self._default is None:
            self._default = key
        return self

    def register_remote(
        self,
        name: str,
        base_url: str,
        *,
        translator: TranslatorFn = _identity_translator,
        extra_params: dict | None = None,
        default: bool = False,
    ) -> TrendRegistry:
        """Registra un modelo que corre en Modal via HTTP."""
        key = name.strip().lower()
        self._services[key] = RemoteTrendService(
            key, base_url, self._history_days, translator=translator, extra_params=extra_params
        )
        if default:
            self._default = key
        return self

    def load_all(self) -> None:
        for service in self._services.values():
            service.load()

    def resolve(self, name: str | None) -> TrendService | OnDemandTrendService | RemoteTrendService:
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
        df = _fetch_history_with_macro(symbol, self._history_days)
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
    from src.arima_trend import HORIZON as ARIMA_HORIZON
    from src.arima_trend import ArimaTrendModel, translate_modal_arima_response
    from src.config import settings
    from src.lstm import TrendModel
    from src.transformer import TransformerTrendModel
    from src.xgb_trend import XGBTrendModel

    registry = TrendRegistry(history_days)
    registry.register("lstm", settings.lstm_model_path, TrendModel.load, default=True)
    registry.register("xgboost", settings.xgb_model_path, XGBTrendModel.load)
    registry.register("transformer", settings.transformer_model_path, TransformerTrendModel.load)
    # ARIMA es por-ticker (no se puede poolear): se ajusta al vuelo, sin artefacto persistido.
    registry.register_on_demand("arima", ArimaTrendModel())
    # Proximos modelos: registry.register("randomforest", settings.rf_model_path, RFTrendModel.load)

    # Alternativas que corren en Modal (repo `models`), independientes de
    # este proceso. Solo se registran si su URL esta configurada. lstm-modal
    # es default a proposito: se reentrena solo en cada request (no depende
    # de que alguien corra `make train` y suba el artefacto a mano), asi que
    # mientras Modal responda, la prediccion "principal" nunca queda vieja.
    # Si Modal no esta configurado, este `if` ni se ejecuta y el default
    # sigue siendo el `lstm` local de arriba -- sin riesgo de romper nada.
    if settings.modal_lstm_url:
        registry.register_remote("lstm-modal", settings.modal_lstm_url, default=True)
    if settings.modal_xgboost_url:
        registry.register_remote("xgboost-modal", settings.modal_xgboost_url)
    if settings.modal_arima_url:
        # El arima_model.py de Modal (version del equipo) pide "predictions"
        # y "media_movil" en vez de horizon/order_ma.
        registry.register_remote(
            "arima-modal",
            settings.modal_arima_url,
            translator=translate_modal_arima_response,
            extra_params={"predictions": ARIMA_HORIZON, "media_movil": 1},
        )

    return registry
