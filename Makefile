dev: ## Levanta api_ml en docker (con build, en primer plano)
	docker compose up --build
dev-d: ## Igual que dev, en segundo plano
	docker compose up --build -d
stop: ## Baja los contenedores de docker
	docker compose down
network: ## Crea la red docker compartida financiar_shared_network (una vez)
	docker network create financiar_shared_network
install: ## Instala las dependencias de desarrollo (requirements-dev.txt)
	pip install -r requirements-dev.txt
run: ## Levanta la API con uvicorn en el host (:8000, con reload)
	uvicorn src.main:app --reload --port 8000
train: ## Entrena el LSTM de tendencia (models/lstm.pt, con gate de promocion)
	python -m src.train
train-xgb: ## Entrena el XGBoost de tendencia (models/xgb.pkl, con gate de promocion)
	python -m src.train_xgb
train-transformer: ## Entrena el Transformer original (models/transformer.pt, con gate de promocion)
	python -m src.train_transformer
train-transformer-v2: ## Entrena el Transformer V2 con datos del collector (models/transformer_v2.pt, sin gate)
	python -m src.train_transformer_v2 $(ARGS)
# Transformer V2: snapshot con datos del collector local, busqueda de variantes
# (retomable: lo ya corrido no se repite) y acierto de la direccion en dolares.
V2_SNAPSHOT ?= simulacion/snapshot-collector
V2_BUSQUEDA ?= v2-direccion-2
# Ruedas por ticker del snapshot: todo lo que tiene el collector (~10 anios),
# para poder entrenar con mas historia (variantes "historia" de la busqueda).
V2_DIAS ?= 2600
snapshot-collector: ## Baja el snapshot de la simulacion desde el collector local (V2_SNAPSHOT)
	DATA_SOURCE=collector DATA_COLLECTOR_URL=$${DATA_COLLECTOR_URL:-http://localhost:3000} \
		python -m src.simulacion.snapshot --dir $(V2_SNAPSHOT) --days $(V2_DIAS)
buscar-transformer-v2: ## Compara 11 variantes del V2 con el original en 7 periodos 2022-2025 (GPU, ~4-5 h, retomable)
	python -m src.simulacion.busqueda --nombre $(V2_BUSQUEDA) --plan v2 --semillas 5 \
		--baselines --snapshot $(V2_SNAPSHOT) --limpiar --dispositivo cuda --procesos 3 $(ARGS)
direccion-transformer-v2: ## Tabla de acierto de la direccion en dolares de la busqueda del V2
	python -m src.simulacion.direccion --nombre $(V2_BUSQUEDA) --snapshot $(V2_SNAPSHOT) \
		--limpiar $(ARGS)
V2_RAPIDA ?= v2-vs-original
probar-transformer-v2: ## Prueba rapida de ajustes e hiperparametros del V2 (1 semilla, ~15 min) y tabla de direccion
	python -m src.simulacion.busqueda --nombre $(V2_RAPIDA) --plan rapido --semillas 1 \
		--snapshot $(V2_SNAPSHOT) --limpiar --dispositivo cuda --procesos 3 $(ARGS)
	python -m src.simulacion.direccion --nombre $(V2_RAPIDA) --snapshot $(V2_SNAPSHOT) --limpiar
backtest: ## Backtest walk-forward de los modelos ya entrenados (deja una foto en JSON)
	python -m src.backtest
paper-trade: ## Paper trading: registra predicciones y resuelve las vencidas
	python -m src.paper_trading
test: ## Corre los tests con cobertura (minimo 80%)
	pytest --cov=src --cov-report=term-missing --cov-fail-under=80
lint: ## Revisa el codigo con ruff
	ruff check src tests
fmt: ## Formatea el codigo con ruff
	ruff format src tests
help: ## Lista los comandos disponibles (varios aceptan ARGS="..." con opciones extra)
	@grep -E '^[a-zA-Z0-9_-]+:.*## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*## "}; {printf "  %-26s %s\n", $$1, $$2}'
