from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query

from src.config import settings
from src.errors import (
    ApiMlError,
    DataUnavailableError,
    ModelNotLoadedError,
    NotEnoughDataError,
    UnknownModelError,
)
from src.model import PredictionModel
from src.registry import build_registry
from src.schemas import (
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
    TrendRequest,
    TrendResponse,
)

prediction_model = PredictionModel(settings.model_path)
trend_registry = build_registry(settings.history_days)


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
    description="Predice la tendencia de un ticker. `model` elige el modelo (default: lstm).",
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
    description="Predice la tendencia de un ticker. Query `model` elige el modelo (default: lstm).",
)
async def predict_trend_get(
    symbol: str,
    model: str | None = Query(default=None, description="lstm | xgboost | ... (default: lstm)"),
) -> TrendResponse:
    return _trend(symbol, model)
