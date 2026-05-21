from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=32)
    features: dict[str, float] = Field(default_factory=dict)


class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    symbol: str
    prediction: float
    model_version: str


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool
