# syntax=docker/dockerfile:1
ARG PY_IMAGE=python:3.10
FROM ${PY_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /app

# Install system build tools for heavy Python wheels if using slim variants
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy source code
COPY . /app

# Create non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Default env paths aligned to repo structure
ENV DATASET_PATH="data/dataset/dataset_training.csv" \
    LSTM_MODEL_PATH="data/lstm_models/chatbot_model.h5" \
    LSTM_TOKENIZER_PATH="data/lstm_models/tokenizer.pkl" \
    LSTM_LABEL_ENCODER_PATH="data/lstm_models/label_encoder.pkl" \
    BERT_MODEL_PATH="data/bert_model" \
    API_HOST="0.0.0.0" \
    API_PORT="8000" \
    DEBUG="False"

EXPOSE 8000

# Run using uvicorn (production-friendly, without reload)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
