# ============================================================
# Clinical Document Intelligence Platform
# Multi-stage Docker build for production deployment
# Module 16: Cloud-Native Deployment
# ============================================================

FROM python:3.11-slim AS base

WORKDIR /app

# System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# === Production stage ===
FROM base AS production

WORKDIR /app

# Copy application code
COPY config/ config/
COPY src/ src/

# Create data directories
RUN mkdir -p data/sample_labels data/eval vector_store

# Non-root user for security
RUN useradd -m -r appuser && chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

# Run with uvicorn
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

# === Development stage ===
FROM base AS development

WORKDIR /app

# Additional dev dependencies
RUN pip install --no-cache-dir pytest pytest-asyncio

COPY . .

RUN mkdir -p data/sample_labels data/eval vector_store

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
