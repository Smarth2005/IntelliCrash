# ─────────────────────────────────────────────────────────────
# IntelliCrash — Dockerfile
#
# Containerizes the FastAPI backend for portable deployment.
#
# Build:   docker build -t intellicrash-api .
# Run:     docker run -p 8000:8000 intellicrash-api
# Compose: docker-compose up (if docker-compose.yml exists)
# ─────────────────────────────────────────────────────────────

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# System dependencies (needed for some pip packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directory for SQLite
RUN mkdir -p /app/data

# Expose the API port
EXPOSE 8000

# Environment defaults
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run the FastAPI server
CMD ["uvicorn", "src.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
