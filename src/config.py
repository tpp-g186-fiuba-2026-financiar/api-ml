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
    # XGBoost de panel a 5 ruedas: reemplaza a los modelos de 5 ruedas que solo
    # miran el precio (LSTM: 47,9% de acierto de direccion en vivo).
    xgb5_model_path: str = "models/xgb5.pkl"
    transformer_model_path: str = "models/transformer.pt"
    # Artefacto del Transformer V2 (src/panel/transformer_v2.py), que la API sirve como
    # "transformer" (el original, transformer_model_path, queda como "transformer-v1").
    transformer_v2_model_path: str = "models/transformer_v2.pt"
    macro_model_path: str = "models/macro.pkl"
    context_model_path: str = "models/contexto.pkl"
    # Comparacion de modelos de cada accion que deja el paper trading diario
    # (src/compare_snapshot.py). Vacio = siempre calcular en vivo.
    compare_snapshot_path: str | None = "models/compare_snapshot.json"
    # Copias en GitHub de lo que deja el job diario (rama main). El dyno de
    # Render puede no redeployarse con los commits del bot ([skip ci]): si la
    # copia local esta vencida, se lee de aca. Vacio = solo la copia local.
    compare_snapshot_url: str | None = (
        "https://raw.githubusercontent.com/tpp-g186-fiuba-2026-financiar/api-ml/main/"
        "models/compare_snapshot.json"
    )
    ledger_url: str | None = (
        "https://raw.githubusercontent.com/tpp-g186-fiuba-2026-financiar/api-ml/main/"
        "models/paper_trading_ledger.json"
    )
    # Fuente de datos histtoricos: "yahoo" (directo) o "collector" (data-colector).
    data_source: str = "yahoo"
    # URL base del servicio data-colector. Se usa para el historico si
    # data_source == "collector", y siempre para listar tickers disponibles
    # (el catalogo de tickers solo vive en data-colector, no en Yahoo).
    # Default: el data-colector local (red docker financiar_shared_network).
    # El deployado solo se usa si se pasa DATA_COLLECTOR_URL explicitamente
    # (crons de GitHub Actions, deploy de Render).
    data_collector_url: str | None = "http://service_data_collector:3000"
    # URL del data-colector que sirve las series macro (default: data_collector_url).
    macro_collector_url: str | None = None
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
    # Serie del BCRA (via data-colector, /interest-rate/ar/{serie}) usada
    # como tasa libre de riesgo en Black-Litterman (TNA en %).
    tasa_libre_riesgo_series: str = "BADLAR"

    # --- Modelos alternativos deployados en Modal (repo `models`) ---
    # Ninguno configurado por default: hay que pegar la URL de `modal deploy`
    # (no la de `modal serve`, que es efimera). Si no estan seteadas, esas
    # alternativas simplemente no aparecen en el registro.
    modal_lstm_url: str | None = None
    modal_xgboost_url: str | None = None
    modal_transformer_url: str | None = None
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
