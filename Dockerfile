# MedGuard API — container image for the FastAPI serving layer (app/main.py)
# Base: slim Python 3.11 (matches requires-python >=3.11). uv for fast, locked installs.
FROM python:3.11-slim AS base

# uv binary (fast dependency install, uses your uv.lock)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# System libs some ML wheels expect at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- dependency layer (cached unless pyproject/uv.lock change) ---
# Install core deps + the `nli` group (torch/transformers) so the NLI verifier works.
# For a much lighter image, drop `--group nli` and run with VERIFIER=llm_judge instead.
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --group nli --no-install-project

# --- application layer ---
COPY app/ ./app/
COPY data/ ./data/

# Put the venv on PATH so `uvicorn` resolves
ENV PATH="/app/.venv/bin:$PATH"

# HF/torch caches: writable, so runtime model downloads are cached across requests
ENV HF_HOME=/app/.cache/huggingface
RUN mkdir -p /app/.cache/huggingface

EXPOSE 8000
# Container Apps/K8s send SIGTERM; uvicorn handles graceful shutdown.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
