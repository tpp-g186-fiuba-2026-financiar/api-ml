from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query

from src.black_litterman.black_litterman import entry
from src.config import settings
from src.data import fetch_history
from src.errors import (
    ApiMlError,
    DataUnavailableError,
    ModelNotLoadedError,
    NotEnoughDataError,
    UnknownModelError,
)
from src.garch_volatility import GarchVolatilityModel
from src.modal_client import call_modal
from src.model import PredictionModel
from src.registry import build_registry
from src.schemas import (
    DirectionResponse,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
    TrendRequest,
    TrendResponse,
    Usuario,
    VolatilityResponse,
)
from src.svm_direction import SvmDirectionModel

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
async def predict_trend_compare(symbol: str) -> dict:
    """Corre todos los modelos entrenados y devuelve las predicciones lado a lado."""
    try:
        return trend_registry.compare(symbol)
    except DataUnavailableError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get(
    "/predict/trend/{symbol}",
    response_model=TrendResponse,
    tags=["Predicciones"],
    summary="Prediccion de tendencia (path)",
    description=(
        "Predice la tendencia de un ticker. Query `model` elige el modelo. "
        "Default: `lstm-modal` si esta configurado (temporal, mientras el "
        "LSTM local no se reentrena), sino `lstm`."
    ),
)
async def predict_trend_get(
    symbol: str,
    model: str | None = Query(
        default=None,
        description=(
            "lstm | xgboost | transformer | arima | lstm-modal | xgboost-modal | "
            "arima-modal | ... (ver /models para el default vigente)"
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
    """Arma la funcion (ticker -> TrendResponse) que necesita entry(),
    usando el trend_registry que main.py ya tiene. Asi black_litterman.py
    nunca necesita importar nada de main.py.
    """

    def _predict(ticker: str) -> TrendResponse:
        service = trend_registry.resolve(model)
        resultado = service.predict(ticker)
        return TrendResponse(**resultado)

    return _predict


def _predict_function(model: str | None):
    """Arma la funcion (ticker -> TrendResponse) que necesita entry(),
    usando el trend_registry que main.py ya tiene. Asi black_litterman.py
    nunca necesita importar nada de main.py.
    """

    def _predict(ticker: str) -> TrendResponse:
        service = trend_registry.resolve(model)
        resultado = service.predict(ticker)
        return TrendResponse(**resultado)

    return _predict


@app.post(
    "/portfolio/recomendacion",
    response_model=dict,
    tags=["Predicciones"],
    summary="Recomendacion de cartera (Black-Litterman)",
    description=(
        "Arma Q y Omega con las predicciones de tendencia de cada ticker "
        "(placeholder: confidence=1 para todos), Pi a partir de la cartera "
        "actual del usuario, y devuelve los pesos optimos recomendados."
    ),
)
async def portfolio_recomendacion(usuario: Usuario, model: str | None = None) -> dict:
    try:
        tickers, pesos = entry(usuario, _predict_function(model), garch_volatility_model)
    except UnknownModelError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except DataUnavailableError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except NotEnoughDataError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {"pesos_recomendados": dict(zip(tickers, pesos.tolist(), strict=True))}
