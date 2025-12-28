#!/bin/bash
# Fix OpenCV conflict - ensure only headless version is installed
# This script removes opencv-python if it was installed by dependencies

set -e

echo "🔍 Checking for opencv-python conflicts..."

# Check if opencv-python (non-headless) is installed
if pip show opencv-python &>/dev/null; then
    echo "⚠️  Found opencv-python (non-headless version)"
    echo "📦 Uninstalling opencv-python..."
    pip uninstall -y opencv-python
    echo "✅ Removed opencv-python"
fi

# Verify opencv-python-headless is installed
if pip show opencv-python-headless &>/dev/null; then
    echo "✅ opencv-python-headless is installed"
else
    echo "❌ opencv-python-headless is NOT installed!"
    exit 1
fi

echo "✨ OpenCV configuration verified!"
