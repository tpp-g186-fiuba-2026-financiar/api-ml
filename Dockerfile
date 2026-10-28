# torch y xgboost de PyPI traen CUDA (~3.5 GB de librerias nvidia, triton y
# nccl) que la API no usa: solo predice, en CPU. En la imagen se instalan sus
# builds CPU con las mismas versiones de requirements.txt: torch X+cpu desde el
# indice de PyTorch y xgboost-cpu (mismo modulo `xgboost`, sin nccl).
# Entrenar con GPU se hace en el host.
ARG TORCH_CPU_INDEX=https://download.pytorch.org/whl/cpu

FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt ./
RUN sed -e 's/^torch==\(.*\)/torch==\1+cpu/' -e 's/^xgboost==/xgboost-cpu==/' \
        requirements.txt > requirements-cpu.txt

# El contenedor de desarrollo solo sirve la API (con reload): tests, lint,
# notebooks y entrenamiento corren en el host con requirements-dev.txt.
FROM base AS development
ARG TORCH_CPU_INDEX
RUN pip install --extra-index-url ${TORCH_CPU_INDEX} -r requirements-cpu.txt
COPY . .
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

FROM base AS builder
ARG TORCH_CPU_INDEX
RUN pip install --prefix=/install --extra-index-url ${TORCH_CPU_INDEX} -r requirements-cpu.txt

FROM python:3.12-slim AS production
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /install /usr/local
COPY src ./src
COPY models ./models
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT}/health" || exit 1
CMD ["sh", "-c", "uvicorn src.main:app --host 0.0.0.0 --port ${PORT}"]
