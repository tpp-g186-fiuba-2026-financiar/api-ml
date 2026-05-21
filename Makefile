dev:
	docker compose up --build

dev-d:
	docker compose up --build -d

stop:
	docker compose down

network:
	docker network create financiar_shared_network

install:
	pip install -r requirements-dev.txt

run:
	uvicorn src.main:app --reload --port 8000

test:
	pytest

lint:
	ruff check src tests

fmt:
	ruff format src tests
