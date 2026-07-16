from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    api_port: int = 8000
    model_path: str = "models/model.pkl"
    database_url: str | None = None

    # --- Modelos de tendencia / datos ---
    # Artefacto del modelo LSTM de tendencia.
    lstm_model_path: str = "models/lstm.pt"
    # Artefacto del modelo XGBoost de tendencia.
    xgb_model_path: str = "models/xgb.pkl"
    transformer_model_path: str = "models/transformer.pt"
    # Fuente de datos histtoricos: "yahoo" (directo) o "collector" (data-colector).
    data_source: str = "yahoo"
    # URL base del servicio data-colector. Se usa para el historico si
    # data_source == "collector", y siempre para listar tickers disponibles
    # (el catalogo de tickers solo vive en data-colector, no en Yahoo).
    data_collector_url: str | None = "https://data-colector.onrender.com"
    # Sufijo de mercado para Yahoo Finance (BYMA = ".BA").
    market_suffix: str = ".BA"
    # Cantidad de ruedas de historia a pedir para entrenar / predecir.
    history_days: int = 750
    # Serie macro (tasa de interes) usada como feature exogena en los modelos
    # de tendencia (ver src.data.fetch_macro_series). Default: rendimiento a
    # 10 anios del Tesoro de EEUU (afecta el apetito por riesgo emergente,
    # Merval incluido) via data-colector -> Yahoo. La alternativa local
    # ("ar", "TPM" = tasa de politica monetaria del BCRA) es mas relevante
    # para acciones argentinas pero depende de la API del BCRA, que al momento
    # de agregar esta feature estaba caida del lado de data-colector.
    macro_rate_source: str = "us"
    macro_rate_series: str = "TNX"

    # --- Modelos alternativos deployados en Modal (repo `models`) ---
    # Ninguno configurado por default: hay que pegar la URL de `modal deploy`
    # (no la de `modal serve`, que es efimera). Si no estan seteadas, esas
    # alternativas simplemente no aparecen en el registro.
    modal_lstm_url: str | None = None
    modal_xgboost_url: str | None = None
    modal_arima_url: str | None = None
    modal_svm_url: str | None = None
    modal_garch_url: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )


settings = Settings()
