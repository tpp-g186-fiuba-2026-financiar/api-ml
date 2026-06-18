# api-ml

Servicio HTTP que expone modelos de ML para predicciones sobre acciones del
Merval.

## Endpoints

- `GET /` — root
- `GET /health` — estado + flags de modelos cargados (`model_loaded`, `lstm_loaded`)
- `POST /predict` — modelo genérico: recibe `{"symbol": "GGAL", "features": {...}}` y devuelve `{"symbol", "prediction", "model_version"}`
- `POST /predict/trend` — modelo LSTM de tendencia: recibe `{"symbol": "GGAL"}`
- `GET /predict/trend/{symbol}` — igual que el anterior pero por path param

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
  "model_version": "lstm-20260618T003942Z"
}
```

- `signal`: tendencia estimada para los próximos `horizon_days` (`alza` / `baja` / `neutral`).
- `expected_return` / `predicted_close`: retorno y precio de cierre esperados al cabo del horizonte.
- `rsi` + `condition`: indicador RSI clásico (`sobrecompra` si RSI ≥ 70, `sobreventa` si ≤ 30).

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
- `src/train.py` — script CLI de entrenamiento.
- `src/model.py` — `TrendService`, orquesta fetch + predicción para la API.

### Entrenar el modelo

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
