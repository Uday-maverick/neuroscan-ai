# NeuroScan AI - Production Dockerfile
# Multi-stage build for optimized image size

FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies system-wide
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir --no-warn-script-location -r requirements.txt && \
    pip uninstall -y opencv-python && \
    pip install --no-cache-dir opencv-python-headless==4.11.0.86

# Production stage
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FLASK_ENV=production \
    MPLCONFIGDIR=/tmp/.matplotlib

# Install runtime dependencies for OpenCV headless and Flask
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    libglib2.0-0 \
    libfontconfig1 \
    libice6 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages and scripts from builder (system-wide installation)
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application files
COPY . .

# Create necessary directories with proper permissions
RUN mkdir -p instance/uploads instance/results models /tmp/.matplotlib && \
    chmod -R 755 instance models /tmp/.matplotlib && \
    chown -R nobody:nogroup instance /tmp/.matplotlib

# Switch to non-root user for security
USER nobody

# Expose port (Render will override with $PORT)
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT:-5000}/api/health || exit 1

# Start application with Gunicorn
# Using 1 worker for free tier (512MB RAM), increase for paid tiers
CMD gunicorn \
    --bind 0.0.0.0:${PORT:-5000} \
    --workers 1 \
    --threads 4 \
    --worker-class gthread \
    --timeout 120 \
    --access-logfile - \
    --error-logfile - \
    --log-level info \
    api:app

