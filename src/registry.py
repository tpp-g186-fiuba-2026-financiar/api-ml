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
from datetime import date
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from src.data import MACRO_COLUMN, attach_macro_feature, fetch_history, fetch_macro_series
from src.errors import ApiMlError, DataUnavailableError, ModelNotLoadedError, UnknownModelError
from src.modal_client import call_modal
from src.trend_common import backtest_predict_df, derive_trend_output


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

    def backtest_on(self, df: pd.DataFrame, horizon: int) -> dict | None:
        """Backtest walk-forward sobre el artefacto ya cargado (ver `backtest_predict_df`)."""
        if self._model is None:
            return None
        return backtest_predict_df(self._model, df, horizon)


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


def translate_modal_svm_response(payload: dict) -> dict:
    """Traduce la respuesta de ``svm_model.py`` de Modal (repo `models`) al
    formato de TrendResponse.

    El SVM es un clasificador binario (Buy/Sell) sobre el retorno del dia
    siguiente: no predice un precio, asi que ``last_close``/``predicted_close``
    quedan en None (mismo criterio que ``rsi``/``condition`` en ARIMA-modal,
    que tampoco los puede calcular -- ver ``translate_modal_arima_response``).
    El ``as_of`` es una aproximacion (fecha de hoy): Modal no manda la fecha
    real de la ultima rueda que uso. Se preserva el ``backtest`` original
    (``directional_accuracy``) para que compita de igual a igual en
    ``_pick_best_model``.
    """
    prediction = payload.get("prediction")
    if prediction not in ("Buy", "Sell"):
        raise DataUnavailableError("Modal (svm) no devolvio una prediccion valida")
    return {
        "signal": "alza" if prediction == "Buy" else "baja",
        "horizon_days": 1,
        "predicted_close": None,
        "last_close": None,
        "rsi": None,
        "condition": "indeterminado",
        "as_of": date.today().isoformat(),
        "backtest": payload.get("backtest"),
        "model_version": payload.get("model_version"),
    }


# Campos que el equipo decidio sacar del contrato de TrendResponse
# (expected_return era redundante con predicted_close; confidence saturaba
# en 1.0 con cualquier retorno > 3%). Los modelos locales ya no los generan,
# pero las copias en Modal (repo `models`) todavia si -- se filtran aca para
# que /predict/trend/compare (que no valida contra el schema) sea
# consistente entre todos los modelos.
_DROPPED_TREND_FIELDS = ("expected_return", "confidence")

# Campos minimos de TrendResponse (ver src.schemas) que RemoteTrendService
# necesita para armar una entrada valida. Cuando Modal no tiene un modelo
# entrenado para el ticker devuelve HTTP 200 con {"error": "..."} en vez de
# un 4xx/5xx, asi que call_modal no lo detecta: hay que validarlo aca.
_REQUIRED_TREND_FIELDS = ("as_of", "signal", "horizon_days", "last_close", "predicted_close")


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
        missing = [field for field in _REQUIRED_TREND_FIELDS if field not in result]
        if "error" in result or missing:
            reason = result.get("error") or f"faltan campos {missing} en la respuesta de Modal"
            raise DataUnavailableError(f"{self.name}/{symbol}: {reason}")
        for field in _DROPPED_TREND_FIELDS:
            result.pop(field, None)
        result["symbol"] = symbol.strip().upper()
        result["model"] = self.name
        result.setdefault("model_version", self.version)
        return result


class EnsembleTrendService:
    """Promedio pesado del retorno log de otros modelos ya registrados.

    No tiene artefacto propio ni entrena nada: en cada prediccion le pide a
    cada miembro su ``predicted_close``/``last_close`` (via ``predict_on``,
    reusando el mismo ``df`` que ya bajo el resto de los modelos), los
    convierte al retorno log implicito, promedia con los pesos fijos, y
    deriva la salida final con ``derive_trend_output`` -- asi la senal del
    ensamble usa el mismo umbral/RSI/formato que un modelo individual. Los
    pesos se resuelven contra los miembros pasados al construir el registro
    (ver ``TrendRegistry.register_ensemble``), no contra un registro global.
    """

    def __init__(
        self,
        name: str,
        members: dict[str, tuple[TrendService, float]],
        history_days: int,
    ):
        self.name = name
        self._members = members
        self.history_days = history_days

    def load(self) -> None:
        pass  # cada miembro se carga solo via su propio TrendService.load()

    @property
    def is_loaded(self) -> bool:
        # Requiere los 3 miembros cargados: un promedio parcial silencioso
        # (ej: si transformer no cargo) daria una senal distinta a la que se
        # esta probando en el issue sin que se note en el nombre del modelo.
        return all(svc.is_loaded for svc, _weight in self._members.values())

    @property
    def version(self) -> str:
        parts = ",".join(f"{n}:{svc.version}:{w}" for n, (svc, w) in self._members.items())
        return f"ensemble({parts})"

    def predict_on(self, df: pd.DataFrame, symbol: str) -> dict:
        if not self.is_loaded:
            raise ModelNotLoadedError(
                f"el ensamble '{self.name}' necesita todos sus modelos miembro cargados"
            )
        log_returns = []
        weights = []
        horizon = None
        for svc, weight in self._members.values():
            result = svc.predict_on(df, symbol)
            horizon = horizon or int(result["horizon_days"])
            log_returns.append(np.log(result["predicted_close"] / result["last_close"]))
            weights.append(weight)
        avg_log_return = float(np.average(log_returns, weights=weights))

        output = derive_trend_output(df, avg_log_return, horizon)
        output["symbol"] = symbol.strip().upper()
        output["model"] = self.name
        output["model_version"] = self.version
        return output

    def predict(self, symbol: str) -> dict:
        df = _fetch_history_with_macro(symbol, self.history_days)
        return self.predict_on(df, symbol)


_LIVE_BACKTEST_MODELS = {"lstm", "xgboost"}


def _pick_best_model(predictions: dict[str, dict], fallback: str | None) -> str | None:
    """Elige, entre los modelos con backtest, el de mejor accuracy direccional.

    El "default" ya no es un modelo fijo: cada ticker puede tener un ganador
    distinto segun como le fue prediciendolo. Solo lstm/xgboost calculan
    backtest en vivo en `compare()` (ver `_LIVE_BACKTEST_MODELS`), asi que la
    eleccion queda entre esos dos; en empate de accuracy gana el de menor
    error (mae). Si ninguno tiene metricas todavia (poca historia, backtest
    fallo), se cae al default global de siempre.
    """
    best_name: str | None = None
    best_score: tuple[float, float] | None = None
    for name, result in predictions.items():
        if result.get("available") is False:
            continue
        backtest = result.get("backtest")
        if not backtest or backtest.get("directional_accuracy") is None:
            continue
        score = (backtest["directional_accuracy"], -backtest.get("mae", float("inf")))
        if best_score is None or score > best_score:
            best_score = score
            best_name = name
    return best_name or fallback


class TrendRegistry:
    """Coleccion de modelos de tendencia, con uno marcado como default."""

    def __init__(self, history_days: int):
        self._history_days = history_days
        self._services: dict[
            str, TrendService | OnDemandTrendService | RemoteTrendService | EnsembleTrendService
        ] = {}
        self._default: str | None = None
        # (symbol, dia) -> resultado de compare() ya calculado. Los datos son
        # velas diarias, asi que recalcular todo (incluido el backtest en
        # vivo) mas de una vez por dia por ticker es trabajo tirado -- y en el
        # dyno free de Render (CPU compartida) es justo lo que hace que el
        # endpoint tarde o se caiga. Solo el primer pedido del dia paga el
        # costo completo.
        self._compare_cache: dict[tuple[str, str], dict] = {}

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

    def register_ensemble(
        self, name: str, weights: dict[str, float], *, default: bool = False
    ) -> TrendRegistry:
        """Registra un promedio pesado de modelos ya registrados en *este* registry.

        ``weights`` mapea el nombre de cada miembro (ya registrado con
        ``register``, ej: ``{"lstm": 0.25, "xgboost": 0.5, "transformer": 0.25}``)
        a su peso. Los pesos no necesitan sumar 1: se normalizan en cada
        prediccion (``np.average``). Los miembros deben ser instancias de
        ``TrendService`` (con artefacto propio, no on-demand/remoto) porque
        el ensamble reusa su ``predict_on`` sobre el mismo ``df``.
        """
        key = name.strip().lower()
        resolved: dict[str, tuple[TrendService, float]] = {}
        for member_name, weight in weights.items():
            member_key = member_name.strip().lower()
            service = self._services.get(member_key)
            if not isinstance(service, TrendService):
                raise UnknownModelError(
                    f"'{member_name}' no esta registrado como TrendService; "
                    "registralo antes de armar el ensamble"
                )
            resolved[member_key] = (service, weight)
        self._services[key] = EnsembleTrendService(key, resolved, self._history_days)
        if default:
            self._default = key
        return self

    def load_all(self) -> None:
        for service in self._services.values():
            service.load()

    def resolve(
        self, name: str | None
    ) -> TrendService | OnDemandTrendService | RemoteTrendService | EnsembleTrendService:
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
        modelos no entrenados, o que fallan al predecir/backtestear, se
        reportan como no disponibles en vez de cortar la comparacion entera
        -- un modelo roto (o sin recursos en el dyno de Render) no debe
        tumbar a los demas.
        """
        symbol_key = symbol.strip().upper()
        today = date.today().isoformat()
        cached = self._compare_cache.get((symbol_key, today))
        if cached is not None:
            return cached

        df = _fetch_history_with_macro(symbol, self._history_days)
        predictions: dict[str, dict] = {}
        for name, service in self._services.items():
            if not service.is_loaded:
                predictions[name] = {"available": False, "reason": "modelo no entrenado"}
                continue
            try:
                result = service.predict_on(df, symbol)
            except Exception as exc:  # noqa: BLE001 - un modelo no debe tumbar la comparacion
                predictions[name] = {"available": False, "reason": str(exc)}
                continue
            # Backtest en vivo solo para lstm/xgboost: son los que importa
            # comparar (los mas fuertes) y ya así el comparador se puso al
            # limite de timeout en el dyno de Render (CPU compartida) con
            # los 3 -- transformer es el mas caro (self-attention) y el
            # menos diferencial, se deja afuera del calculo en vivo.
            # ARIMA local reajusta por-ticker en cada predict_df, asi que
            # walk-forward serian fits reales -- tambien afuera a proposito.
            if name in _LIVE_BACKTEST_MODELS and isinstance(service, TrendService):
                try:
                    result["backtest"] = service.backtest_on(df, result.get("horizon_days", 5))
                except Exception:  # noqa: BLE001 - backtest es best-effort
                    result["backtest"] = None
            predictions[name] = result
        response = {
            "symbol": symbol_key,
            "as_of": df.index[-1].strftime("%Y-%m-%d"),
            "default_model": _pick_best_model(predictions, self._default),
            "predictions": predictions,
        }
        self._compare_cache[(symbol_key, today)] = response
        return response

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

    # Ensamble numerico (issue #163, "probar" un promedio pesado entre LSTM/
    # XGBoost/Transformer): se probo con `register_ensemble` (ver clase
    # `EnsembleTrendService` mas abajo) en varios esquemas de pesos -- ningun
    # promedio le gano a XGBoost solo de forma significativa (mejor caso:
    # +0.1pp de accuracy direccional, dentro del ruido con ~2700 predicciones
    # pooled; MAE y retorno de estrategia siempre peores que XGBoost solo).
    # No se registra en produccion a proposito: cada entrada de ensamble
    # implica repetir la inferencia de los 3 modelos sin beneficio medido, y
    # /predict/trend/compare ya esta al limite de timeout en Render (ver nota
    # mas abajo). Queda `register_ensemble` disponible para volver a probar
    # si se suma un modelo realmente distinto (no correlacionado con estos 3).

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
    if settings.modal_transformer_url:
        registry.register_remote("transformer-modal", settings.modal_transformer_url)
    if settings.modal_arima_url:
        # El arima_model.py de Modal (version del equipo) pide "predictions"
        # y "media_movil" en vez de horizon/order_ma.
        registry.register_remote(
            "arima-modal",
            settings.modal_arima_url,
            translator=translate_modal_arima_response,
            extra_params={"predictions": ARIMA_HORIZON, "media_movil": 1},
        )
    if settings.modal_svm_url:
        # svm_model.py (repo `models`, issue #159) es un clasificador binario
        # Buy/Sell: si tiene signal, entra al mismo comparador/backtest que
        # el resto (ver translate_modal_svm_response).
        registry.register_remote(
            "svm-modal", settings.modal_svm_url, translator=translate_modal_svm_response
        )
    # garch_model.py (repo `models`, issue #159) NO se registra aca a
    # proposito: pronostica volatilidad (varianza a 5 dias), no una
    # direccion -- no tiene "signal" y forzarlo en este registry rompe el
    # contrato de TrendPredictor que asumen paper_trading.py y TrendResponse
    # (alza/baja/neutral). Se suma igual "al comparador" pero un nivel mas
    # arriba, en backend-website (compare_trend_logic.rs), que solo relaya
    # JSON y no necesita ese contrato -- ahi se lo marca explicitamente como
    # no apto para el ranking de tendencia (ver `fetch_garch`).

    return registry
