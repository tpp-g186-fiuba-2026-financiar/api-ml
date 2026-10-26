import json
import math
import time
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query

from src.black_litterman.black_litterman import MAX_ACCIONES_NUEVAS, recomendacion_detallada
from src.black_litterman.cartera_ancla import TipoCarteraAncla
from src.config import settings
from src.consensus import LEDGER_PATH, compute_consensus, load_track_record
from src.data import fetch_history, fetch_interest_rate
from src.errors import (
    ApiMlError,
    DataUnavailableError,
    ModelNotLoadedError,
    NotEnoughDataError,
    UnknownModelError,
)
from src.fair_value import VENTANA_DEFAULT, precio_estimado
from src.garch_volatility import GarchVolatilityModel
from src.modal_client import call_modal
from src.model import PredictionModel
from src.paper_trading import load_ledger, signal_history
from src.registry import _fetch_history_with_macro as fetch_history_with_macro
from src.registry import build_registry
from src.schemas import (
    ConsensusResponse,
    DirectionResponse,
    ExplainResponse,
    HealthResponse,
    PerfilRiesgo,
    PredictionRequest,
    PredictionResponse,
    SignalHistoryResponse,
    TrendRequest,
    TrendResponse,
    Usuario,
    VolatilityResponse,
)
from src.svm_direction import SvmDirectionModel

BACKTEST_RESULTS_PATH = Path(__file__).resolve().parent.parent / "models" / "backtest_results.json"
RECOMENDACION_TTL_SEGUNDOS = 600
_RECOMENDACION_CACHE_MAX = 64
_recomendacion_cache: dict[tuple, tuple[float, dict]] = {}

prediction_model = PredictionModel(settings.model_path)
trend_registry = build_registry(settings.history_days)
svm_direction_model = SvmDirectionModel()
garch_volatility_model = GarchVolatilityModel()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    prediction_model.load()
    trend_registry.load_all()
    yield


app = FastAPI(
    title="api-ml",
    version="0.1.0",
    description=(
        "Servicio de modelos de ML para predicciones de tendencia sobre acciones "
        "del Merval. Soporta multiples modelos (LSTM, XGBoost, ...) registrados en "
        "paralelo: se puede elegir cual usar por request y compararlos lado a lado."
    ),
    openapi_tags=[
        {"name": "General", "description": "Root y health check."},
        {"name": "Modelos", "description": "Listado y estado de los modelos registrados."},
        {
            "name": "Predicciones",
            "description": "Prediccion de tendencia y comparacion de modelos.",
        },
    ],
    lifespan=lifespan,
)


@app.get("/", response_model=dict, tags=["General"], summary="Root")
async def root() -> dict[str, str]:
    return {"message": "Welcome to the api-ml service."}


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["General"],
    summary="Health check",
    description="Estado del servicio y de cada modelo de tendencia registrado (cargado + version).",
)
async def health() -> HealthResponse:
    statuses = trend_registry.statuses()
    return HealthResponse(
        status="ok",
        model_loaded=prediction_model.is_loaded,
        lstm_loaded=statuses.get("lstm", {}).get("loaded", False),
        trend_models=statuses,
    )


@app.get(
    "/models",
    response_model=dict,
    tags=["Modelos"],
    summary="Listar modelos",
    description="Modelo default + estado (cargado y version) de cada modelo registrado.",
)
async def models() -> dict:
    """Lista los modelos de tendencia registrados y su estado."""
    return {"default": trend_registry.default, "models": trend_registry.statuses()}


@app.post(
    "/predict", response_model=PredictionResponse, tags=["Predicciones"], summary="Predict generico"
)
async def predict(payload: PredictionRequest) -> PredictionResponse:
    try:
        value = prediction_model.predict(payload.features)
    except ApiMlError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return PredictionResponse(
        symbol=payload.symbol,
        prediction=value,
        model_version=prediction_model.version,
    )


def _trend(symbol: str, model: str | None) -> TrendResponse:
    try:
        service = trend_registry.resolve(model)
        result = service.predict(symbol)
    except UnknownModelError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except DataUnavailableError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except NotEnoughDataError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return TrendResponse(**result)


@app.post(
    "/predict/trend",
    response_model=TrendResponse,
    tags=["Predicciones"],
    summary="Prediccion de tendencia (body)",
    description=(
        "Predice la tendencia de un ticker. `model` elige el modelo (ver "
        "/models para el default vigente)."
    ),
)
async def predict_trend(payload: TrendRequest) -> TrendResponse:
    return _trend(payload.symbol, payload.model)


@app.get(
    "/predict/trend/compare/{symbol}",
    response_model=dict,
    tags=["Predicciones"],
    summary="Comparar todos los modelos",
    description=(
        "Corre todos los modelos entrenados sobre el mismo historico (una sola "
        "descarga) y devuelve las predicciones lado a lado. Los modelos no "
        "entrenados se reportan como no disponibles."
    ),
)
def predict_trend_compare(symbol: str) -> dict:
    """Corre todos los modelos entrenados y devuelve las predicciones lado a lado."""
    try:
        return trend_registry.compare(symbol)
    except DataUnavailableError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get(
    "/predict/trend/consensus/{symbol}",
    response_model=ConsensusResponse,
    tags=["Predicciones"],
    summary="Consenso entre modelos (sobrecompra/sobreventa)",
    description=(
        "Combina las predicciones de todos los modelos en una sola lectura "
        "(sobrecompra/sobreventa/neutral), ajustada por el perfil de riesgo "
        "del inversor. Adaptado de decision_maker_playground (issue #154) -- "
        "ver src.consensus para el detalle de la formula y por que cambia "
        "respecto del prototipo original."
    ),
)
def predict_trend_consensus(
    symbol: str,
    profile: Annotated[
        PerfilRiesgo,
        Query(description="Perfil de riesgo del inversor (ver users.risk_profile)"),
    ] = PerfilRiesgo.MODERADO,
) -> ConsensusResponse:
    try:
        compared = trend_registry.compare(symbol)
    except DataUnavailableError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    consensus = compute_consensus(compared["predictions"], profile)
    return ConsensusResponse(
        symbol=compared["symbol"],
        investor_profile=profile,
        track_record=load_track_record(profile, compared["symbol"]),
        **consensus,
    )


@app.get(
    "/predict/fair-value/{symbol}",
    response_model=dict,
    tags=["Predicciones"],
    summary="Precio estimado por regresion lineal",
    description=(
        "Ajusta log(precio) contra el tiempo con minimos cuadrados sobre las "
        "ultimas `ventana` ruedas y devuelve el precio que marca la tendencia "
        "hoy, cuanto se aparta el precio real (en % y en desvios estandar), "
        "si esta por encima o por debajo, la recta con el canal de +-1 y +-2 "
        "desvios, y que paso historicamente en las 20 ruedas siguientes en el "
        "panel lider desde una distancia parecida."
    ),
)
def predict_fair_value(
    symbol: str,
    ventana: Annotated[int, Query(ge=60, le=1260)] = VENTANA_DEFAULT,
) -> dict:
    try:
        df = fetch_history(symbol, max(ventana + 10, settings.history_days))
        return {"symbol": symbol.strip().upper(), **precio_estimado(df, ventana)}
    except DataUnavailableError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except NotEnoughDataError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get(
    "/predict/trend/xgboost/explain/{symbol}",
    response_model=ExplainResponse,
    tags=["Predicciones"],
    summary="Por que XGBoost dice lo que dice (contribuciones SHAP)",
    description=(
        "Contribucion de cada variable a la P(sube) de la ultima rueda, en "
        "log-odds (TreeSHAP de XGBoost, promediado entre las semillas del "
        "ensamble). `base_value + sum(contribution)` es el margen del modelo; "
        "positivo empuja a 'sube'. Ordenadas por |contribution| descendente; "
        "`label` es un nombre en castellano para mostrar."
    ),
)
async def explain_xgboost(symbol: str) -> ExplainResponse:
    try:
        service = trend_registry.resolve("xgboost")
        model = getattr(service, "model", None)
        if model is None:
            raise ModelNotLoadedError("el modelo 'xgboost' no esta entrenado")
        # El mismo historico que usa /compare, para que la probabilidad coincida.
        df = fetch_history_with_macro(symbol, settings.history_days)
        result = model.explain(df)
    except UnknownModelError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except DataUnavailableError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except NotEnoughDataError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return ExplainResponse(symbol=symbol.strip().upper(), model_version=model.version, **result)


@app.get(
    "/predict/trend/{symbol}",
    response_model=TrendResponse,
    tags=["Predicciones"],
    summary="Prediccion de tendencia (path)",
    description=(
        "Predice la tendencia de un ticker. Query `model` elige el modelo. "
        "Default: `lstm-modal` si esta configurado (se reentrena en cada "
        "request, nunca queda desactualizado), sino `lstm` local."
    ),
)
async def predict_trend_get(
    symbol: str,
    model: str | None = Query(
        default=None,
        description=(
            "lstm | xgboost | transformer (V2) | transformer-v1 | arima | lstm-modal | "
            "xgboost-modal | arima-modal | ... (ver /models para el default vigente)"
        ),
    ),
) -> TrendResponse:
    return _trend(symbol, model)


@app.get(
    "/predict/direction/{symbol}",
    response_model=DirectionResponse,
    tags=["Predicciones"],
    summary="Direccion binaria (SVM)",
    description=(
        "Clasificador SVM: sube o baja el cierre de la rueda SIGUIENTE (no un "
        "retorno a horizonte de varios dias como /predict/trend). No predice "
        "magnitud, por eso no comparte el formato de TrendResponse. Se ajusta "
        "al vuelo con el historico del ticker pedido."
    ),
)
async def predict_direction(symbol: str) -> DirectionResponse:
    try:
        df = fetch_history(symbol, settings.history_days)
        result = svm_direction_model.predict_df(df)
    except DataUnavailableError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except NotEnoughDataError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    result["symbol"] = symbol.strip().upper()
    result["model"] = "svm"
    result["model_version"] = svm_direction_model.version
    return DirectionResponse(**result)


@app.get(
    "/predict/volatility/{symbol}",
    response_model=VolatilityResponse,
    tags=["Predicciones"],
    summary="Volatilidad (GARCH)",
    description=(
        "Pronostico GARCH(1,1) de volatilidad a `horizon_days` ruedas. No "
        "predice direccion (alza/baja), solo la magnitud del movimiento "
        "esperado. Se ajusta al vuelo con el historico del ticker pedido."
    ),
)
async def predict_volatility(symbol: str) -> VolatilityResponse:
    try:
        df = fetch_history(symbol, settings.history_days)
        result = garch_volatility_model.predict_df(df)
    except DataUnavailableError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except NotEnoughDataError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    result["symbol"] = symbol.strip().upper()
    result["model"] = "garch"
    result["model_version"] = garch_volatility_model.version
    return VolatilityResponse(**result)


@app.get(
    "/predict/direction/modal/{symbol}",
    response_model=dict,
    tags=["Predicciones"],
    summary="Direccion binaria (SVM en Modal)",
    description=(
        "Corre el svm_model.py deployado en Modal (repo `models`) y devuelve "
        "su respuesta cruda ({'prediction': 'Buy'|'Sell'}), sin traducir: "
        "esa version no expone confidence/last_close/as_of, por eso no "
        "comparte el formato de DirectionResponse."
    ),
)
async def predict_direction_modal(symbol: str) -> dict:
    if not settings.modal_svm_url:
        raise HTTPException(status_code=404, detail="modal_svm_url no esta configurada")
    try:
        payload = call_modal(settings.modal_svm_url, symbol)
    except DataUnavailableError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"symbol": symbol.strip().upper(), "model": "svm-modal", **payload}


@app.get(
    "/predict/volatility/modal/{symbol}",
    response_model=dict,
    tags=["Predicciones"],
    summary="Volatilidad (GARCH en Modal)",
    description=(
        "Corre el garch_model.py deployado en Modal (repo `models`) y "
        "devuelve su respuesta cruda, sin traducir: tiene una forma "
        "distinta a VolatilityResponse, por eso no se fuerza a ese schema."
    ),
)
async def predict_volatility_modal(symbol: str) -> dict:
    if not settings.modal_garch_url:
        raise HTTPException(status_code=404, detail="modal_garch_url no esta configurada")
    try:
        payload = call_modal(settings.modal_garch_url, symbol)
    except DataUnavailableError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"symbol": symbol.strip().upper(), "model": "garch-modal", **payload}


def _predict_function(model: str | None):
    """Arma la funcion (ticker -> TrendResponse) que necesita Black-Litterman,
    usando el trend_registry que main.py ya tiene. Asi black_litterman.py
    nunca necesita importar nada de main.py.
    """

    def _predict(ticker: str) -> TrendResponse:
        service = trend_registry.resolve(model)
        resultado = service.predict(ticker)
        return TrendResponse(**resultado)

    return _predict


def _tasa_libre_riesgo_anual() -> float:
    """Ultima TNA (decimal) de la serie configurada en
    ``settings.tasa_libre_riesgo_series`` (BADLAR por default), para que las
    vistas de Black-Litterman sean retorno en exceso. Si data-colector o el
    BCRA fallan se sigue con 0 (vistas nominales) en vez de cortar el pedido.
    """
    try:
        serie = fetch_interest_rate("ar", settings.tasa_libre_riesgo_series, days=30)
        return float(serie.iloc[-1]) / 100.0
    except ApiMlError as exc:
        print(f"  [warn] sin tasa libre de riesgo, se usa 0: {exc}")
        return 0.0


@app.post(
    "/portfolio/recomendacion",
    response_model=dict,
    tags=["Predicciones"],
    summary="Recomendacion de cartera (Black-Litterman)",
    description=(
        "Combina el equilibrio implicito de la cartera ancla (Pi) con las "
        "predicciones de tendencia de cada ticker del panel lider (Q, en exceso "
        "sobre la tasa BADLAR, con confianza baja y fija), y devuelve los pesos "
        "optimos segun el perfil de riesgo del usuario. `detalle` trae, por "
        "ticker, peso actual (participacion real por valor), peso recomendado "
        "y los intermedios pi/q/omega/mu_bl (retornos log diarios: multiplicar "
        "por horizon_days para verlos al horizonte); `tickers_excluidos` los "
        "descartados con su motivo; `cartera_ancla` la estrategia usada."
    ),
)
async def portfolio_recomendacion(
    usuario: Usuario,
    model: str | None = None,
    cartera_ancla: Annotated[
        TipoCarteraAncla | None,
        Query(
            description=(
                "Estrategia de cartera ancla: propia | equal_weight | mercado. "
                "Default (sin especificar): propia si el usuario tiene holdings "
                "con valor positivo, equal_weight si no."
            )
        ),
    ] = None,
    max_nuevas: Annotated[
        int | None,
        Query(
            ge=0,
            le=20,
            description=(
                "Cuantas acciones que el usuario no tiene puede sumar la "
                "recomendacion (default 2). 0 = solo redistribuir entre las suyas; "
                "sin valor y con todas=true, todo el panel."
            ),
        ),
    ] = MAX_ACCIONES_NUEVAS,
    todas: Annotated[
        bool, Query(description="Si es true, puede usar todo el panel (ignora max_nuevas).")
    ] = False,
) -> dict:
    limite_nuevas = None if todas else max_nuevas
    # La recomendacion tarda segundos (historicos + predicciones + GARCH) y solo
    # cambia con la cartera, el modelo o el dia: se reutiliza unos minutos.
    cache_key = (
        usuario.model_dump_json(),
        model,
        cartera_ancla.value if cartera_ancla else None,
        limite_nuevas,
        date.today().isoformat(),
    )
    cached = _recomendacion_cache.get(cache_key)
    if cached and time.monotonic() - cached[0] < RECOMENDACION_TTL_SEGUNDOS:
        return cached[1]
    try:
        resultado = recomendacion_detallada(
            usuario,
            _predict_function(model),
            garch_volatility_model,
            cartera_ancla,
            tasa_libre_riesgo_anual=_tasa_libre_riesgo_anual(),
            max_nuevas=limite_nuevas,
        )
    except UnknownModelError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except DataUnavailableError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except NotEnoughDataError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    result = {
        "pesos_recomendados": dict(zip(resultado.tickers, resultado.pesos.tolist(), strict=True)),
        "detalle": resultado.detalle(),
        "tickers_excluidos": resultado.tickers_excluidos,
        "cartera_ancla": resultado.cartera_ancla,
        "acciones_nuevas": resultado.acciones_nuevas,
        "rotacion": round(resultado.rotacion, 4),
        "costo_estimado": round(resultado.costo_estimado, 5),
        "costo_operacion": resultado.costo_operacion,
        "max_nuevas": limite_nuevas,
    }
    if len(_recomendacion_cache) >= _RECOMENDACION_CACHE_MAX:
        _recomendacion_cache.pop(next(iter(_recomendacion_cache)))
    _recomendacion_cache[cache_key] = (time.monotonic(), result)
    return result


@app.get(
    "/signals/history",
    response_model=SignalHistoryResponse,
    tags=["Modelos"],
    summary="Historial de senales (paper trading)",
    description=(
        "Senales registradas por el paper trading diario (pendientes y ya "
        "resueltas), la mas nueva primero. `hit` es true/false si ya se cumplio "
        "el horizonte y la senal fue alza/baja; null si esta pendiente o fue "
        "neutral. `summary` resume las filas devueltas con resultado."
    ),
)
async def signals_history(
    symbols: Annotated[
        str | None, Query(description="Tickers separados por coma (ej: GGAL,YPFD). Vacio = todos.")
    ] = None,
    limit: Annotated[int, Query(ge=1, le=5000)] = 200,
) -> SignalHistoryResponse:
    wanted = [s for s in (symbols or "").split(",") if s.strip()] or None
    try:
        ledger = load_ledger(LEDGER_PATH)
    except (OSError, ValueError):
        ledger = {"pending": [], "resolved": []}
    return SignalHistoryResponse(**signal_history(ledger, wanted, limit))


def _json_safe(value):
    """NaN/inf no son JSON valido: se devuelven como null."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


@app.get(
    "/models/diagnostics",
    response_model=dict,
    tags=["Modelos"],
    summary="Diagnostico de modelos",
    description=(
        "`backtest`: ultimo backtest walk-forward (models/backtest_results.json, "
        "null si no hay). `xgboost`: version y metricas fuera de muestra del "
        "artefacto XGBoost cargado (null si no esta cargado)."
    ),
)
async def models_diagnostics() -> dict:
    try:
        backtest = json.loads(BACKTEST_RESULTS_PATH.read_text())
    except (OSError, ValueError):
        backtest = None

    xgboost = None
    try:
        model = getattr(trend_registry.resolve("xgboost"), "model", None)
    except UnknownModelError:
        model = None
    if model is not None:
        xgboost = {
            "version": model.version,
            "trained_at": getattr(model, "trained_at", None),
            "metrics": dict(getattr(model, "metrics", {}) or {}),
        }
    return _json_safe({"backtest": backtest, "xgboost": xgboost})
