from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from src.config import settings
from src.errors import ApiMlError
from src.model import PredictionModel
from src.schemas import HealthResponse, PredictionRequest, PredictionResponse

prediction_model = PredictionModel(settings.model_path)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    prediction_model.load()
    yield


app = FastAPI(title="api-ml", version="0.1.0", lifespan=lifespan)


@app.get("/", response_model=dict)
async def root() -> dict[str, str]:
    return {"message": "Welcome to the api-ml service."}


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=prediction_model.is_loaded)


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
