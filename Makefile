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
train-transformer-v2: ## Entrena el Transformer V2, el "transformer" de la API (collector, gate por IC; ARGS="--force" lo salta)
	python -m src.panel.train_transformer_v2 $(ARGS)
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
# Ronda 2 del plan de mejora: filtro con modelos baratos (CPU) e intervalos de cada grupo.
R2_FILTRO ?= r2-filtro
filtro-r2: ## Ronda 2: cabezas relativa / mercado, meta-labeling y dos cabezas con modelos baratos (CPU, ~10 min)
	python -m src.simulacion.busqueda --nombre $(R2_FILTRO) --plan r2 --semillas 1 \
		--snapshot $(V2_SNAPSHOT) --limpiar $(ARGS)
	python -m src.simulacion.significancia --nombre $(R2_FILTRO) --snapshot $(V2_SNAPSHOT) \
		--limpiar --contra ens
R2_TRANSFORMER ?= r2-transformer
transformer-r2: ## Ronda 2: V2 como cabeza relativa vs ensamble, 3 semillas (GPU, ~30-40 min, retomable)
	python -m src.simulacion.busqueda --nombre $(R2_TRANSFORMER) --plan r2-transformer --semillas 3 \
		--snapshot $(V2_SNAPSHOT) --limpiar --dispositivo cuda --procesos 3 $(ARGS)
	python -m src.simulacion.significancia --nombre $(R2_TRANSFORMER) --snapshot $(V2_SNAPSHOT) \
		--limpiar --contra ens_h1500
# Simulacion rapida de un solo modelo (por default el V2): entrena con datos
# anteriores a SIM_DESDE, predice cada rueda, mide la direccion en dolares con
# intervalos y arma carteras simples, sin Black-Litterman ni consenso.
SIM_NOMBRE ?= v2-rapida
SIM_DESDE ?= 2025-01-02
# JSON con campos de la config del modelo (p. ej. SIM_CONFIG='{"n_semillas": 1}').
SIM_CONFIG ?= {}
simular-v2: ## Simulacion rapida de un solo modelo: metricas de direccion y carteras sin Black-Litterman (GPU)
	python -m src.simulacion.un_modelo --nombre $(SIM_NOMBRE) --desde $(SIM_DESDE) \
		--config '$(SIM_CONFIG)' --snapshot $(V2_SNAPSHOT) --dispositivo cuda $(ARGS)
# Simulacion con chequeo previo de requisitos (scripts/check_simulacion.sh) y el
# reporte en LaTeX / PDF al final (imprime las rutas). Con el snapshot V2_SNAPSHOT:
#   make simular MODELO=lstm NOMBRE=lstm-2026 DESDE=2026-01-02
#   make simulacion NOMBRE=completa-2026 DESDE=2026-06-01 REENTRENO=semanal
PYTHON ?= python
MODELO ?= transformer_v2
NOMBRE ?=
DESDE ?=
HASTA ?=
# Vacio: el default de cada simulador (un_modelo: mensual; simulador: diario).
REENTRENO ?=
# JSON con campos de la config del modelo (p. ej. CONFIG='{"epochs": 20}').
CONFIG ?= {}
SNAPSHOT ?= $(V2_SNAPSHOT)
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
simular: ## Simulacion de un solo modelo, cualquiera (MODELO=lstm|xgboost|arima|macro|contexto|transformer_v2|...): direccion y carteras sin Black-Litterman, con reporte en PDF
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
