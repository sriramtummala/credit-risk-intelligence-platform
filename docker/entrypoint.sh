#!/bin/sh
# entrypoint.sh — run inside the container at startup
#
# 1. If the model artifact is missing (first boot, or ephemeral container),
#    download the UCI dataset and train the XGBoost champion model.
# 2. Start the FastAPI server.
#
# The model lives at models/champion_model.joblib. Because that path is
# gitignored it will not be present in a freshly built image. Training takes
# ~10 s on a single CPU; subsequent restarts skip the training step.

set -e

MODEL_PATH="models/champion_model.joblib"

if [ ! -f "$MODEL_PATH" ]; then
    echo "[entrypoint] Model artifact not found at $MODEL_PATH"
    echo "[entrypoint] Running training pipeline (this takes ~30 s on first boot)..."
    python main.py --download
    echo "[entrypoint] Training complete."
else
    echo "[entrypoint] Model artifact found — skipping training."
fi

echo "[entrypoint] Starting API on port ${PORT:-8000}..."
exec uvicorn api.main:app --host 0.0.0.0 --port "${PORT:-8000}"
