# ── Credit Risk Intelligence Platform — Production Dockerfile ─────────────────
#
# Build:  docker build -t credit-risk-api .
# Run:    docker run -p 8000:8000 credit-risk-api
#
# On first boot the entrypoint script downloads the UCI dataset, trains the
# XGBoost champion model (~30 s), then starts the FastAPI server.
# Subsequent runs reuse the cached model if you mount a volume:
#   docker run -p 8000:8000 -v $(pwd)/models:/app/models credit-risk-api
#
# Environment variables:
#   PORT   — port the server listens on (default: 8000)

FROM python:3.11-slim

# Metadata
LABEL maintainer="Sriram Tummala"
LABEL description="Credit Risk Scoring Service — XGBoost default-probability API"
LABEL version="1.0"

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

# ── Install dependencies ───────────────────────────────────────────────────────
# Copy only requirements first so Docker can cache this layer independently
# from the source code. Rebuilds after code changes won't reinstall packages.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ── Copy application source ────────────────────────────────────────────────────
COPY api/        ./api/
COPY src/        ./src/
COPY main.py     .
COPY docker/entrypoint.sh .

# Create directories that gitignore would have excluded from the build context
RUN mkdir -p data/raw data/processed models reports/figures

# Make the entrypoint executable
RUN chmod +x entrypoint.sh

# ── Expose & run ───────────────────────────────────────────────────────────────
EXPOSE ${PORT}

# Health check — Docker will restart the container if /health stops responding
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health')" \
    || exit 1

ENTRYPOINT ["./entrypoint.sh"]
