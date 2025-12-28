#!/bin/bash
# Model Download Script for Render Deployment
# This script downloads the ML model from cloud storage if MODEL_DOWNLOAD_URL is set

set -e  # Exit on error

MODEL_PATH="${MODEL_PATH:-models/best_model.pth}"
MODEL_DIR=$(dirname "$MODEL_PATH")

echo "🔍 Checking for model file..."

# Create models directory if it doesn't exist
mkdir -p "$MODEL_DIR"

# Check if model already exists
if [ -f "$MODEL_PATH" ]; then
    echo "✅ Model file already exists at $MODEL_PATH"
    exit 0
fi

# Check if download URL is provided
if [ -z "$MODEL_DOWNLOAD_URL" ]; then
    echo "⚠️  Warning: MODEL_DOWNLOAD_URL not set and model file not found"
    echo "⚠️  The application will fail if the model is required"
    echo ""
    echo "Options to fix this:"
    echo "1. Set MODEL_DOWNLOAD_URL environment variable in Render dashboard"
    echo "2. Upload model file manually to persistent disk"
    echo "3. Use Git LFS to include model in repository"
    exit 0
fi

# Download model from URL
echo "📥 Downloading model from $MODEL_DOWNLOAD_URL..."

# Use curl with retry and progress
curl -L \
    --retry 3 \
    --retry-delay 5 \
    --max-time 300 \
    --progress-bar \
    -o "$MODEL_PATH" \
    "$MODEL_DOWNLOAD_URL"

if [ -f "$MODEL_PATH" ]; then
    MODEL_SIZE=$(du -h "$MODEL_PATH" | cut -f1)
    echo "✅ Model downloaded successfully ($MODEL_SIZE)"
else
    echo "❌ Failed to download model"
    exit 1
fi
