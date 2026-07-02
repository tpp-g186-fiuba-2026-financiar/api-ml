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
    # URL base del servicio data-colector (si data_source == "collector").
    data_collector_url: str | None = None
    # Sufijo de mercado para Yahoo Finance (BYMA = ".BA").
    market_suffix: str = ".BA"
    # Cantidad de ruedas de historia a pedir para entrenar / predecir.
    history_days: int = 750

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )


settings = Settings()
