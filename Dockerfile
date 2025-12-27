# NeuroScan AI - Production Dockerfile
# Multi-stage build for optimized image size

FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FLASK_ENV=production \
    PORT=5000

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder (system-wide installation)
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application files
COPY . .

# Create necessary directories with proper permissions
RUN mkdir -p instance/uploads instance/results models && \
    chmod -R 755 instance models && \
    chown -R nobody:nogroup instance

# Switch to non-root user for security
USER nobody

# Expose port
EXPOSE 5000

# Health check (using python -m to avoid PATH issues)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -m gunicorn --check-config || curl -f http://localhost:5000/api/health || exit 1

# Run application with Gunicorn using python -m to ensure it's found
CMD ["python", "-m", "gunicorn", \
     "--bind", "0.0.0.0:5000", \
     "--workers", "4", \
     "--worker-class", "sync", \
     "--timeout", "120", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "--log-level", "info", \
     "app:app"]
