FROM python:3.11-slim
# above installs python image

# uv for fast, reproducible installs
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# switch to app dir
WORKDIR /app

# Install deps first (better layer caching) - copies pyproject and uv into app dir
COPY pyproject.toml uv.lock ./
# installs exactly what is in pyproject with the frozen flag without dev deps
RUN uv sync --frozen --no-dev

# copies app code after the above 2 commands because app code changes often and not deps - rebuilds are faster with cached deps
COPY app ./app

# Non-root user (HF Spaces requirement) - not a root (0-999) user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app

# once all commands requiring root permissions are executed - switch to app user 1000 id - protects the app from attackers
USER appuser

# HF requires the app running on this port
EXPOSE 7860
# start the server where the container runs
CMD ["uv", "run", "--no-sync", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
