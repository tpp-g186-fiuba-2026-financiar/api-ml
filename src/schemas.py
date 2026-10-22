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


class ConsensusTrackRecord(BaseModel):
    """Cuantas veces acerto este consenso en paper trading (ya resuelto, sin
    mirar hacia adelante). Ver ``src.paper_trading``."""

    n_resolved: int
    signal_hit_rate: float | None  # aciertos / veces que dio alza o baja
    neutral_rate: float | None  # % de veces que no se animo a dar senal
    scope: str  # "ticker" | "global"


class ConsensusComponent(BaseModel):
    """Aporte de un modelo al consenso (ver ``src.consensus.compute_consensus``)."""

    model_config = ConfigDict(protected_namespaces=())

    model: str
    expected_return: float  # retorno log implicito al horizonte del modelo
    quality: float  # acierto direccional de backtest ajustado por casos (0.5 sin datos)
    agreement: float  # 1.0, o 0.5 si signal y signo del retorno no coinciden
    signal: str | None = None


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
    components: list[ConsensusComponent] = Field(default_factory=list)
    track_record: ConsensusTrackRecord | None = None


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


class FeatureContribution(BaseModel):
    feature: str  # nombre tecnico (ej: t_ret20, r_m_rel20)
    label: str  # nombre en castellano para mostrar
    value: float | None  # valor de la variable en la ultima rueda (None si falta)
    contribution: float  # aporte en log-odds a P(sube); positivo empuja a "sube"
    help: str | None = None  # que mide, en palabras simples
    unit: str = "num"  # como mostrar value: pct, flag, days, index, rank o num


class ExplainResponse(BaseModel):
    """Contribuciones SHAP de XGBoost a la P(sube) de la ultima rueda."""

    model_config = ConfigDict(protected_namespaces=())

    symbol: str
    as_of: str
    probability_up: float
    base_value: float  # log-odds de partida (antes de mirar las variables)
    contributions: list[FeatureContribution]
    model_version: str | None = None
    # Senal que emite el modelo con esa probabilidad (la misma que /predict/trend).
    signal: str | None = None
    # Desde que P(sube) dice "alza" / hasta cual dice "baja"; None = lado apagado.
    alza_threshold: float | None = None
    baja_threshold: float | None = None
    # Que tan seguido subieron las acciones fuera de muestra ("lo normal" real).
    base_up_rate: float | None = None


class SignalRow(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    symbol: str
    model: str
    model_version: str | None = None
    as_of: str
    signal: str
    horizon_days: int
    last_close: float
    predicted_close: float
    realized_close: float | None = None
    hit: bool | None = None  # None: pendiente o senal neutral
    resolved_at: str | None = None


class SignalHistorySummary(BaseModel):
    evaluated: int  # filas con resultado (resueltas y no neutrales)
    hits: int
    hit_rate: float | None
    last_signal: str | None = None  # fecha de la ultima senal registrada
    updated_at: str | None = None  # ultimo guardado del ledger


class SignalHistoryResponse(BaseModel):
    rows: list[SignalRow]
    summary: SignalHistorySummary
