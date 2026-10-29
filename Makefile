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
train-transformer: ## Entrena el Transformer (models/transformer.pt, en dolares, gate por walk-forward; ARGS="--force" lo salta)
	python -m src.train_transformer $(ARGS)
# Snapshot de la simulacion con datos del collector local: todo lo que tiene
# (~10 anios por ticker), para entrenar con mas historia.
SNAPSHOT_COLLECTOR ?= simulacion/snapshot-collector
DIAS_COLLECTOR ?= 2600
snapshot-collector: ## Baja el snapshot de la simulacion desde el collector local (SNAPSHOT_COLLECTOR)
	DATA_SOURCE=collector DATA_COLLECTOR_URL=$${DATA_COLLECTOR_URL:-http://localhost:3000} \
		python -m src.simulacion.snapshot --dir $(SNAPSHOT_COLLECTOR) --days $(DIAS_COLLECTOR)
# Busqueda de hiperparametros del Transformer: Optuna en paralelo (~1 h) y confirmacion de los
# mejores con varias semillas y cortes. ARGS='--trials 4 --top 2 --semillas 2' para probar.
OPTUNA_NOMBRE ?= transformer
optuna-transformer: ## Busqueda de hiperparametros del Transformer con Optuna (SNAPSHOT_COLLECTOR, retomable)
	python -m src.simulacion.optuna_transformer --nombre $(OPTUNA_NOMBRE) \
		--snapshot $(SNAPSHOT_COLLECTOR) $(ARGS)
# Ronda 2 del plan de mejora: filtro con modelos baratos (CPU) e intervalos de cada grupo.
R2_FILTRO ?= r2-filtro
filtro-r2: ## Ronda 2: cabezas relativa / mercado, meta-labeling y dos cabezas con modelos baratos (CPU, ~10 min)
	python -m src.simulacion.busqueda --nombre $(R2_FILTRO) --plan r2 --semillas 1 \
		--snapshot $(SNAPSHOT_COLLECTOR) --limpiar $(ARGS)
	python -m src.simulacion.significancia --nombre $(R2_FILTRO) --snapshot $(SNAPSHOT_COLLECTOR) \
		--limpiar --contra ens
# Simulacion con chequeo previo de requisitos (scripts/check_simulacion.sh) y el
# reporte en LaTeX / PDF al final (imprime las rutas). Con el snapshot SNAPSHOT_COLLECTOR:
#   make simular MODELO=lstm NOMBRE=lstm-2026 DESDE=2026-01-02
#   make simulacion NOMBRE=completa-2026 DESDE=2026-06-01 REENTRENO=semanal
PYTHON ?= python
MODELO ?= ensamble_panel
NOMBRE ?=
DESDE ?=
HASTA ?=
# Vacio: el default de cada simulador (un_modelo: mensual; simulador: diario).
REENTRENO ?=
# JSON con campos de la config del modelo (p. ej. CONFIG='{"epochs": 20}').
CONFIG ?= {}
SNAPSHOT ?= $(SNAPSHOT_COLLECTOR)
DISPOSITIVO ?= $(shell $(PYTHON) -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')" 2>/dev/null || echo cpu)
CHECK = PYTHON=$(PYTHON) scripts/check_simulacion.sh
USO = MODELO=<modelo> NOMBRE=<corrida> DESDE=YYYY-MM-DD [HASTA=YYYY-MM-DD] [REENTRENO=...]
define chequear
	@test -n "$(NOMBRE)" -a -n "$(DESDE)" || { echo "Uso: make $@ $(USO)"; exit 1; }
	@$(CHECK) python && $(CHECK) snapshot $(SNAPSHOT)
	@$(CHECK) latex || { echo "AVISO: se va a generar reporte.tex pero no el PDF."; sleep 5; }
endef
# PHONY: hay una carpeta simulacion/ y make daria el target por hecho.
.PHONY: simular simulacion simulacion-reporte simulacion-check
simular: ## Simulacion de un solo modelo, cualquiera (MODELO=lstm|xgboost|arima|macro|contexto|ensamble_panel|...): direccion y carteras sin Black-Litterman, con reporte en PDF
	$(chequear)
	$(PYTHON) -m src.simulacion.un_modelo --nombre $(NOMBRE) --modelo $(MODELO) --desde $(DESDE) \
		$(if $(HASTA),--hasta $(HASTA)) $(if $(REENTRENO),--reentreno $(REENTRENO)) \
		--config '$(CONFIG)' --snapshot $(SNAPSHOT) --dispositivo $(DISPOSITIVO) $(ARGS)
simulacion: ## Simulacion completa (todos los modelos, Black-Litterman y consenso), con reporte en PDF; tarda horas
	$(chequear)
	$(PYTHON) -m src.simulacion.simulador --nombre $(NOMBRE) --desde $(DESDE) \
		$(if $(HASTA),--hasta $(HASTA)) $(if $(REENTRENO),--reentreno $(REENTRENO)) \
		--snapshot $(SNAPSHOT) $(ARGS)
	$(PYTHON) -m src.simulacion.reporte --nombre $(NOMBRE)
simulacion-reporte: ## Rehace el reporte (y el PDF) de una corrida de simulacion completa ya hecha (NOMBRE=...)
	@test -n "$(NOMBRE)" || { echo "Uso: make $@ NOMBRE=<corrida>"; exit 1; }
	$(PYTHON) -m src.simulacion.reporte --nombre $(NOMBRE)
simulacion-check: ## Chequea los requisitos de simular / simulacion (Python, snapshot, LaTeX) sin correr nada
	@ok=0; $(CHECK) python || ok=1; $(CHECK) snapshot $(SNAPSHOT) || ok=1; \
	$(CHECK) latex || ok=1; \
	if [ $$ok = 0 ]; then echo "OK: estan todos los requisitos de la simulacion."; else exit 1; fi
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
train-context: ## Entrena el modelo contexto (models/contexto.pkl, con piso de AUC)
	python -m src.train_context
help: ## Lista los comandos disponibles (varios aceptan ARGS="..." con opciones extra)
	@grep -E '^[a-zA-Z0-9_-]+:.*## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*## "}; {printf "  %-26s %s\n", $$1, $$2}'
