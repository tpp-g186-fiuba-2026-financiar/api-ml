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
    lstm_loaded: bool = False


class TrendRequest(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=32)


class TrendResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    symbol: str
    signal: str  # alza | baja | neutral
    horizon_days: int
    expected_return: float
    last_close: float
    predicted_close: float
    rsi: float | None = None
    condition: str  # sobrecompra | sobreventa | neutral | indeterminado
    confidence: float
    as_of: str
    model_version: str
