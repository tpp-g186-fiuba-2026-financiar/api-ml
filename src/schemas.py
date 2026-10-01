import enum

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
    # Estado de cada modelo de tendencia registrado: nombre -> {loaded, version, is_default}.
    trend_models: dict[str, dict] = Field(default_factory=dict)


class TrendRequest(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=32)
    # Modelo a usar (ej: "lstm", "xgboost"). None => el default del registro.
    model: str | None = Field(default=None, max_length=32)


class TrendResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    symbol: str
    signal: str  # alza | baja | neutral
    horizon_days: int
    last_close: float
    predicted_close: float
    rsi: float | None = None
    condition: str  # sobrecompra | sobreventa | neutral | indeterminado
    as_of: str
    model: str = "lstm"  # que modelo genero la prediccion
    model_version: str


class DirectionResponse(BaseModel):
    """Direccion binaria (SVM) para la rueda siguiente. No predice magnitud."""

    model_config = ConfigDict(protected_namespaces=())

    symbol: str
    signal: str  # alza | baja
    horizon_days: int
    confidence: float
    last_close: float
    as_of: str
    model: str = "svm"
    model_version: str


class VolatilityResponse(BaseModel):
    """Pronostico de volatilidad (GARCH). No predice direccion, solo magnitud."""

    model_config = ConfigDict(protected_namespaces=())

    symbol: str
    horizon_days: int
    daily_volatility_pct: list[float]
    cumulative_volatility_pct: float
    last_close: float
    as_of: str
    model: str = "garch"
    model_version: str


class PerfilRiesgo(enum.StrEnum):
    CONSERVADOR = "conservative"
    MODERADO = "moderate"
    ARRIESGADO = "aggressive"


class ConsensusResponse(BaseModel):
    """Lectura unica (sobrecompra/sobreventa/neutral) combinando todos los
    modelos de tendencia, ajustada por perfil de riesgo. Ver src.consensus.
    """

    symbol: str
    investor_profile: PerfilRiesgo
    classification: str  # sobrecompra | sobreventa | neutral | sin_datos
    composite_score: float
    aggregate_confidence: float
    threshold_buy: float
    threshold_sell: float
    confidence_min: float
    models_considered: int
    explanation: str


class Tenencia(BaseModel):
    ticker: str
    cantidad: float


class Usuario(BaseModel):
    """Toda la info del usuario necesaria para armar la recomendacion:
    su perfil de riesgo y su cartera actual (una tenencia por ticker).
    """

    perfil_riesgo: PerfilRiesgo
    tenencias: list[Tenencia]

    def cantidades_por_ticker(self) -> dict[str, float]:
        """Devuelve un dict {ticker: cantidad}, util para cruzar con precios
        y calcular pesos de cartera (w_actual) mas adelante.
        """
        return {t.ticker: t.cantidad for t in self.tenencias}
