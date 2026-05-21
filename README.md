# api-ml

Servicio HTTP que expone un modelo de ML para predicciones sobre tickers.

## Antes de iniciar

Asegurarse de tener la red de docker creada:

```
docker network create financiar_shared_network
# o
make network
```

## Levantar el servicio

```
cp .env.example .env
make dev
```

El servicio queda en `http://localhost:8000`.

## Endpoints

- `GET /` — root
- `GET /health` — estado + flag de modelo cargado
- `POST /predict` — recibe `{"symbol": "GGAL", "features": {...}}` y devuelve `{"symbol", "prediction", "model_version"}`

## Modelo

El servicio busca el artefacto en la ruta de `MODEL_PATH` (default `models/model.pkl`). Si no existe, levanta con un modelo placeholder (devuelve el promedio de las features) para que la API arranque igual.

Para reemplazarlo, dejar el `.pkl` (joblib) en `models/` con la API `predict_one(features: dict) -> float`.

## Desarrollo local (sin docker)

```
python -m venv .venv
source .venv/bin/activate
make install
make run
make test
```
