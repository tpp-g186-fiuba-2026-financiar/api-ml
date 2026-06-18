from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from src.config import settings
from src.errors import ApiMlError, DataUnavailableError, ModelNotLoadedError, NotEnoughDataError
from src.model import PredictionModel, TrendService
from src.schemas import (
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
    TrendRequest,
    TrendResponse,
)

prediction_model = PredictionModel(settings.model_path)
trend_service = TrendService(settings.lstm_model_path, settings.history_days)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    prediction_model.load()
    trend_service.load()
    yield


app = FastAPI(title="api-ml", version="0.1.0", lifespan=lifespan)


@app.get("/", response_model=dict)
async def root() -> dict[str, str]:
    return {"message": "Welcome to the api-ml service."}


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=prediction_model.is_loaded,
        lstm_loaded=trend_service.is_loaded,
    )


@app.post("/predict", response_model=PredictionResponse)
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


def _trend(symbol: str) -> TrendResponse:
    try:
        result = trend_service.predict(symbol)
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except DataUnavailableError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except NotEnoughDataError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return TrendResponse(**result)


@app.post("/predict/trend", response_model=TrendResponse)
async def predict_trend(payload: TrendRequest) -> TrendResponse:
    return _trend(payload.symbol)


@app.get("/predict/trend/{symbol}", response_model=TrendResponse)
async def predict_trend_get(symbol: str) -> TrendResponse:
    return _trend(symbol)
