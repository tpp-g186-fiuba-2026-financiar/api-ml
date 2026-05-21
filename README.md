# api-ml

Servicio HTTP que expone un modelo de ML para predicciones sobre tickers.

## Endpoints

- `GET /` — root
- `GET /health` — estado + flag de modelo cargado
- `POST /predict` — recibe `{"symbol": "GGAL", "features": {...}}` y devuelve `{"symbol", "prediction", "model_version"}`

## Modelo

El servicio busca el artefacto en la ruta de `MODEL_PATH` (default `models/model.pkl`). Si no existe, levanta con un modelo placeholder (devuelve el promedio de las features) para que la API arranque igual.

Para reemplazarlo, dejar el `.pkl` (joblib) en `models/` con la API `predict_one(features: dict) -> float`.

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
