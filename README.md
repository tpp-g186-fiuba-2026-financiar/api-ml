# api-ml

Servicio HTTP que expone modelos de ML para predicciones sobre acciones del
Merval.

## Endpoints

- `GET /` — root
- `GET /health` — estado del servicio + de cada modelo registrado (`trend_models`)
- `GET /models` — lista los modelos de tendencia y su estado (cargado + versión)
- `POST /predict` — modelo genérico: recibe `{"symbol": "GGAL", "features": {...}}` y devuelve `{"symbol", "prediction", "model_version"}`
- `POST /predict/trend` — tendencia; el body acepta `{"symbol": "GGAL", "model": "xgboost"}` (`model` opcional, default `lstm`)
- `GET /predict/trend/{symbol}` — tendencia por path param; query `?model=` elige el modelo (default `lstm`)
- `GET /predict/trend/compare/{symbol}` — corre **todos** los modelos entrenados y devuelve las predicciones lado a lado

La documentación interactiva (Swagger UI) queda en `http://localhost:8000/docs`.

### Selección de modelo

El servicio mantiene varios modelos cargados en paralelo (ver [Registro de modelos](#registro-de-modelos)).
Se elige cuál usar en cada request:

```bash
GET /predict/trend/GGAL                 # usa el default (lstm)
GET /predict/trend/GGAL?model=xgboost   # fuerza XGBoost
```

- Modelo desconocido → `404`.
- Modelo registrado pero sin entrenar → `503` (el resto sigue funcionando).

Ejemplo de respuesta de `/predict/trend`:

```json
{
  "symbol": "GGAL",
  "signal": "alza",
  "horizon_days": 5,
  "expected_return": 0.0176,
  "last_close": 8365.0,
  "predicted_close": 8511.82,
  "rsi": 73.23,
  "condition": "sobrecompra",
  "confidence": 0.585,
  "as_of": "2026-06-17",
  "model": "lstm",
  "model_version": "lstm-20260618T003942Z"
}
```

- `signal`: tendencia estimada para los próximos `horizon_days` (`alza` / `baja` / `neutral`).
- `expected_return` / `predicted_close`: retorno y precio de cierre esperados al cabo del horizonte.
- `rsi` + `condition`: indicador RSI clásico (`sobrecompra` si RSI ≥ 70, `sobreventa` si ≤ 30).
- `model` / `model_version`: qué modelo generó la predicción.

### Comparar modelos

`GET /predict/trend/compare/{symbol}` descarga el histórico **una sola vez** y corre
todos los modelos entrenados sobre los mismos datos (comparación justa):

```json
{
  "symbol": "GGAL",
  "as_of": "2026-06-30",
  "default_model": "lstm",
  "predictions": {
    "lstm":    { "signal": "alza",    "expected_return": 0.0264, "confidence": 0.88, "...": "..." },
    "xgboost": { "signal": "neutral", "expected_return": 0.0041, "confidence": 0.13, "...": "..." }
  }
}
```

## Registro de modelos

Los modelos de tendencia se manejan con un **registro** (`src/registry.py`): todos
comparten la interfaz `predict_df(df) -> dict` y el mismo post-procesamiento
(`src/trend_common.py`), de modo que sus salidas son directamente comparables.
Para **sumar un modelo nuevo**:

1. Escribir su clase con `predict_df`, `version`, `save`, `load` (ver `src/xgb_trend.py` como ejemplo).
2. Registrarlo con una línea en `build_registry` (`src/registry.py`).
3. Agregar su `*_model_path` en `src/config.py`.

Endpoints, `/health`, `/models` y `/compare` lo toman automáticamente.

## Modelo LSTM de tendencia

Red LSTM (PyTorch) que, a partir de una ventana de los últimos días de OHLCV,
predice el **retorno logarítmico acumulado de los próximos `horizon` días** y de
ahí deriva la señal de tendencia. Detalles de diseño:

- **Features estacionarias** (retorno log de precio y volumen, rango intradiario
  relativo) → un único modelo generaliza a todos los tickers sin importar la
  escala de precios.
- **Un solo modelo global** entrenado pooleando las ventanas de todo el panel
  líder del Merval (ver `default_merval_tickers`).
- **Normalización** calculada solo con el set de entrenamiento, **split temporal**
  train/val, **weight decay** y **early stopping** sobre la pérdida de validación.

Código:

- `src/data.py` — descarga del histórico OHLCV (Yahoo Finance directo o `data-colector`).
- `src/lstm.py` — feature engineering, red, entrenamiento, inferencia y persistencia.
- `src/train.py` — script CLI de entrenamiento del LSTM.
- `src/trend_common.py` — post-procesamiento compartido (señal, RSI, confianza).
- `src/registry.py` — registro que orquesta todos los modelos + fetch de datos.

### Entrenar el LSTM

```bash
make train
# o con parámetros:
python -m src.train --tickers GGAL YPFD PAMP --epochs 60 --window 30
```

Esto descarga el histórico, entrena y guarda el artefacto en `LSTM_MODEL_PATH`
(default `models/lstm.pt`), imprimiendo métricas de validación (MAE del retorno y
accuracy direccional).

> El `.pt` no se commitea (está en `.gitignore`). Hay que entrenar al menos una
> vez para que `/predict/trend` responda; si no existe, el endpoint devuelve 503.

## Modelo XGBoost de tendencia

Modelo tabular (gradient boosting) que resuelve la **misma tarea** que el LSTM
—predecir el retorno log acumulado a `horizon` días— para poder compararlos:

- **Mismos features base y mismo target** que el LSTM (`src/lstm.py`).
- Como XGBoost no es secuencial, la ventana de los últimos `window` días se
  **aplana** a un único vector de features.
- **Misma salida** (`src/trend_common.py`), así las predicciones son comparables.

Código: `src/xgb_trend.py` (modelo) y `src/train_xgb.py` (entrenamiento).

### Entrenar el XGBoost

```bash
python -m src.train_xgb
# o con parámetros:
python -m src.train_xgb --n-estimators 500 --max-depth 5 --window 30
```

Guarda el artefacto en `XGB_MODEL_PATH` (default `models/xgb.pkl`). Requiere la
dependencia `xgboost` (ya está en `requirements.txt`). Igual que el LSTM: si el
`.pkl` no existe, `?model=xgboost` devuelve `503` hasta que se entrene.

### Fuente de datos

Por defecto `DATA_SOURCE=yahoo` (consulta directa a Yahoo Finance, los tickers
BYMA se mapean agregando `.BA`). Para usar el `data-colector` interno, setear
`DATA_SOURCE=collector` y `DATA_COLLECTOR_URL`.

## Modelo genérico (placeholder)

El servicio también busca un artefacto en `MODEL_PATH` (default `models/model.pkl`).
Si no existe, levanta con un placeholder (devuelve el promedio de las features)
para que `/predict` arranque igual. Para reemplazarlo, dejar el `.pkl` (joblib)
en `models/` con la API `predict_one(features: dict) -> float`.

## Desarrollo local

### Con docker

Asegurarse de tener la red creada:

```
docker network create financiar_shared_network
# o
make network
```

Levantar:

```
cp .env.example .env
make dev
```

El servicio queda en `http://localhost:8000`.

### Sin docker

```
python -m venv .venv
source .venv/bin/activate
make install
make run
make test
```

## Deploy en Render

El repo incluye `render.yaml` (Blueprint de Render). El flujo es:

1. En Render: **New > Blueprint**, conectar el repo `api-ml`. Render detecta el `render.yaml` y crea el servicio automáticamente.
2. En el tab **Environment** del servicio, cargar las variables:
   - `MODEL_PATH` (ej: `models/model.pkl`)
   - `DATABASE_URL`
3. Cada push a `main` dispara un build + deploy automático (autoDeploy on).

Detalles:

- **Build**: Render usa el `Dockerfile` (stage `production`).
- **Puerto**: Render asigna `$PORT` en runtime, uvicorn lo respeta.
- **Healthcheck**: Render pollea `/health` para zero-downtime deploys.

### Artefacto del modelo

El `.pkl` no se commitea (está en `.gitignore`). Tres opciones para producción:

1. **Commitearlo** (si pesa poco, < 10 MB): sacarlo del `.gitignore` y se va en la imagen.
2. **Render Disk**: mountar un disco persistente y subirlo manualmente.
3. **Descargar al startup** desde S3/GCS/etc. (requiere agregar la lógica en `src/model.py`).

Mientras no haya artefacto, el servicio levanta con el placeholder y `/health` devuelve `model_loaded: true` con `model_version: "placeholder"`.
